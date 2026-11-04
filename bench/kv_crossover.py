#!/usr/bin/env python3
"""Measure when swapping a KV cache beats recomputing it.

For each context length this records:
  * swap-out: copy every K and V tensor from the device to pinned CPU memory
  * swap-in:  copy them back
  * recompute: chunked prefill from token ids, which is what a dropped cache costs
    (through the engine by default; --backend hf times transformers instead, whose
    cache concatenates on every chunk and so overstates the cost)

The output is a CSV, a plot, and cost_model.json, the fit that the agent cost
model in layover.agent.cost_model loads. Only a run on a CUDA device gives numbers
that describe the target hardware; every other device is marked as such in all
three outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layover.config import ModelConfig  # noqa: E402

DEFAULT_TOKENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
CPU_SMOKE_TOKENS = [64, 128, 256, 512, 1024]
BIT_VIEWS = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Row:
    tokens: int
    kv_bytes: int
    swap_out_s: float
    swap_in_s: float
    swap_s: float
    swap_min_s: float
    swap_max_s: float
    recompute_s: float
    recompute_min_s: float
    recompute_max_s: float
    cheaper: str


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start


def load_hf_model(name: str, device: torch.device, dtype: torch.dtype, seed: int):
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    if name == "micro":
        cfg = ModelConfig.micro()
        torch.manual_seed(seed)
        hf_cfg = Qwen3Config(**{k: v for k, v in cfg.to_hf_dict().items() if k != "model_type"})
        model = Qwen3ForCausalLM(hf_cfg).to(dtype=dtype, device=device)
    else:
        cfg = ModelConfig.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").to(device)
    return cfg, model.eval()


@torch.inference_mode()
def hf_prefill(model, ids: torch.Tensor, chunk: int) -> list[torch.Tensor]:
    """Chunked prefill through the base model.

    Calling model.model skips lm_head: 32,768 tokens of full-vocabulary logits
    would take about 10 GB in BF16. Chunking bounds the attention matrix when the
    CUDA SDPA path materializes it (heads x chunk x context).
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    for start in range(0, ids.shape[1], chunk):
        model.model(ids[:, start : start + chunk], past_key_values=cache, use_cache=True)
    return [t for layer in cache.layers for t in (layer.keys, layer.values)]


def load_layover_model(name: str, device: torch.device, dtype: torch.dtype, seed: int):
    from layover.model import Qwen3
    from layover.weights import load_model

    if name == "micro":
        cfg = ModelConfig.micro()
        torch.manual_seed(seed)
        return cfg, Qwen3(cfg).to(dtype=dtype, device=device).eval()
    model = load_model(name, device=device, dtype=dtype)
    return model.cfg, model


def layover_prefill(model, ids: torch.Tensor, chunk: int, dtype: torch.dtype) -> list[torch.Tensor]:
    """Chunked prefill through the engine, which is what a dropped cache really costs."""
    from layover.kv import BlockTable, PagedKVCache

    tokens = ids.shape[1]
    if tokens % 16:
        raise ValueError("the layover backend measures whole 16-token blocks")
    cache = PagedKVCache(model.cfg, tokens // 16, ids.device, dtype)
    table = BlockTable()
    for start in range(0, tokens, chunk):
        model.prefill(cache, table, ids[0, start : start + chunk], start)
    return [cache.pool]


def make_ids(cfg: ModelConfig, tokens: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + tokens)
    return torch.randint(0, cfg.vocab_size, (1, tokens), generator=gen).to(device)


def checksum(tensors: list[torch.Tensor]) -> int:
    crc = 0
    for t in tensors:
        raw = t.detach().to("cpu").contiguous().view(BIT_VIEWS[t.element_size()])
        crc = zlib.crc32(raw.numpy().tobytes(), crc)
    return crc


def measure_swap(kv: list[torch.Tensor], device: torch.device) -> tuple[float, float]:
    """Round-trip the tensors once through CPU memory and verify the bytes."""
    pin = device.type == "cuda"
    staging = [torch.empty_like(t, device="cpu", pin_memory=pin) for t in kv]
    back = [torch.empty_like(t, device=device) for t in kv]

    def out():
        for s, t in zip(staging, kv):
            s.copy_(t, non_blocking=pin)

    def into():
        for b, s in zip(back, staging):
            b.copy_(s, non_blocking=pin)

    out_s = timed(device, out)
    in_s = timed(device, into)
    if checksum(kv) != checksum(back):
        raise RuntimeError("KV bytes changed across the swap round trip")
    return out_s, in_s


def measure(model, cfg, tokens, args, device) -> Row:
    ids = make_ids(cfg, tokens, device, args.seed)
    out_samples, in_samples, recompute_samples = [], [], []
    for i in range(args.warmup + args.repeats):
        holder: list[list[torch.Tensor]] = []
        if args.backend == "layover":
            prefill = lambda: holder.append(layover_prefill(model, ids, args.chunk, DTYPES[args.dtype]))  # noqa: E731
        else:
            prefill = lambda: holder.append(hf_prefill(model, ids, args.chunk))  # noqa: E731
        recompute_s = timed(device, prefill)
        kv = holder[0]
        if sum(t.numel() for t in kv) * kv[0].element_size() != tokens * cfg.kv_bytes_per_token(args.dtype):
            raise RuntimeError("measured KV size disagrees with the config-derived size")
        out_s, in_s = measure_swap(kv, device)
        del kv, holder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if i >= args.warmup:
            out_samples.append(out_s)
            in_samples.append(in_s)
            recompute_samples.append(recompute_s)
    swap_samples = [a + b for a, b in zip(out_samples, in_samples)]
    swap = statistics.median(swap_samples)
    recompute = statistics.median(recompute_samples)
    return Row(
        tokens=tokens,
        kv_bytes=tokens * cfg.kv_bytes_per_token(args.dtype),
        swap_out_s=statistics.median(out_samples),
        swap_in_s=statistics.median(in_samples),
        swap_s=swap,
        swap_min_s=min(swap_samples),
        swap_max_s=max(swap_samples),
        recompute_s=recompute,
        recompute_min_s=min(recompute_samples),
        recompute_max_s=max(recompute_samples),
        cheaper="swap" if swap < recompute else "recompute",
    )


def fit_nonnegative(x: np.ndarray, y: np.ndarray, degree: int) -> list[float]:
    """Least-squares polynomial with non-negative coefficients, low order first.

    The cost model extrapolates from these fits, so a negative term must not turn
    a large context into a cheap one. Terms with a negative coefficient are
    removed and the rest refitted.
    """
    active = list(range(degree + 1))
    while True:
        design = np.stack([x.astype(float) ** p for p in active], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        if (coef >= 0).all() or len(active) == 1:
            break
        del active[int(np.argmin(coef))]
    result = [0.0] * (degree + 1)
    for power, value in zip(active, coef):
        result[power] = max(float(value), 0.0)
    return result


def find_crossover(rows: list[Row]) -> dict:
    """Locate where swapping starts to beat recomputing among measured points.

    Returns the interpolated token count at a sign change, or says which side won
    the whole measured range, in which case there is no crossover to report.
    """
    rows = sorted(rows, key=lambda r: r.tokens)
    diffs = [r.swap_s - r.recompute_s for r in rows]
    if all(d < 0 for d in diffs):
        return {"crossover_tokens": None, "outcome": "swap_cheaper_at_every_length"}
    if all(d >= 0 for d in diffs):
        return {"crossover_tokens": None, "outcome": "recompute_cheaper_at_every_length"}
    # Report the last sign change: the length past which swapping stays cheaper.
    for a, b in zip(reversed(rows[:-1]), reversed(rows[1:])):
        da, db = a.swap_s - a.recompute_s, b.swap_s - b.recompute_s
        if (da < 0) != (db < 0):
            t = da / (da - db)
            tokens = float(np.exp(np.log(a.tokens) + t * (np.log(b.tokens) - np.log(a.tokens))))
            return {
                "crossover_tokens": tokens,
                "outcome": "swap_cheaper_above" if db < 0 else "recompute_cheaper_above",
            }
    raise AssertionError("sign change not found")


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def write_plot(rows: list[Row], meta: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    swap_color, recompute_color = "#2a78d6", "#eb6834"
    tokens = [r.tokens for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    series = [
        ("swap round trip", "swap", swap_color),
        ("recompute (chunked prefill)", "recompute", recompute_color),
    ]
    for label, key, color in series:
        y = [getattr(r, f"{key}_s") for r in rows]
        low = [getattr(r, f"{key}_min_s") for r in rows]
        high = [getattr(r, f"{key}_max_s") for r in rows]
        ax.fill_between(tokens, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(tokens, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(
            label.split(" (")[0], (tokens[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8, color="#52514e",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Time (s, median with min-max band)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xticks(tokens)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.14)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    title = f"KV swap vs recompute: {meta['model']}, {meta['dtype']}, {meta['device_name']}"
    if not meta["measured_on_gpu"]:
        title += "\nNOT a GPU measurement: pipeline check only"
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{platform.machine()} {device.type}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="micro", help="'micro', a hub id, or a local checkpoint")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--backend", default="layover", choices=["layover", "hf"], help="whose prefill to time")
    p.add_argument("--tokens", type=int, nargs="+", help="context lengths to measure")
    p.add_argument("--chunk", type=int, default=1024, help="prefill chunk size")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="leading repeats to discard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/kv_crossover"))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available")
    tokens = args.tokens or (DEFAULT_TOKENS if device.type == "cuda" else CPU_SMOKE_TOKENS)
    dtype = DTYPES[args.dtype]

    loader = load_layover_model if args.backend == "layover" else load_hf_model
    cfg, model = loader(args.model, device, dtype, args.seed)
    too_long = [t for t in tokens if t > cfg.max_position_embeddings]
    if too_long:
        raise SystemExit(f"{too_long} exceed max_position_embeddings={cfg.max_position_embeddings}")

    rows = []
    for n in tokens:
        rows.append(measure(model, cfg, n, args, device))
        r = rows[-1]
        print(
            f"{r.tokens:>6} tokens  {r.kv_bytes / 2**30:7.3f} GiB  swap {r.swap_s:9.4f}s"
            f"  recompute {r.recompute_s:9.4f}s  cheaper: {r.cheaper}",
            flush=True,
        )

    x = np.array([r.tokens for r in rows])
    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "backend": args.backend,
        "device": args.device,
        "device_name": device_name(device),
        "measured_on_gpu": device.type == "cuda",
        "kv_bytes_per_token": cfg.kv_bytes_per_token(args.dtype),
        "chunk": args.chunk,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "swap_out_s_coef": fit_nonnegative(x, np.array([r.swap_out_s for r in rows]), 1),
        "swap_in_s_coef": fit_nonnegative(x, np.array([r.swap_in_s for r in rows]), 1),
        "recompute_s_coef": fit_nonnegative(x, np.array([r.recompute_s for r in rows]), 2),
        **find_crossover(rows),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "kv_crossover.csv")
    (args.out / "cost_model.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_plot(rows, meta, args.out / "kv_crossover.png")
    print(f"KV bytes per token ({args.dtype}): {meta['kv_bytes_per_token']:,}")
    print(f"Outcome: {meta['outcome']}, crossover: {meta['crossover_tokens']}")
    if not meta["measured_on_gpu"]:
        print("WARNING: not a GPU measurement; do not use these numbers as results.")
    return meta


if __name__ == "__main__":
    main()

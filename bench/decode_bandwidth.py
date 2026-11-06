#!/usr/bin/env python3
"""Measure the decode kernel's achieved memory bandwidth as a percentage of peak.

Decode attention is memory-bound, so the score that says how good the kernel is, is how
close it gets to the bandwidth the GPU's memory can supply. This runs the kernel on
Qwen3-4B-shaped random KV, for several batch sizes and context lengths, with block tables
either scattered across the pool or laid out in order, and times the PyTorch reference on
the same inputs for comparison.

What the bandwidth counts: the bytes of K and V the kernel loads, summed over sequences,
query heads, and whole blocks. Query heads sharing a KV head load the same K and V, and
those GQA re-reads are counted because the kernel does issue them, but a re-read can be
served by a cache and so need not reach DRAM. unique_gbps counts each KV head once, which
is the least the memory has to deliver. The real figure for DRAM lies between the two.
q, the block table, and the output are not counted; they are tiny next to K and V.

Each sample launches the kernel once per layer, as a decode step does, so one layer's KV
pushes the previous layer's out of the L2 and the launches are part of the time. Repeating
one layer would measure the cache once the context fits in it.

Only a run with --device cuda describes a GPU. Everything else is a pipeline check, and
every output says so.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layover.kernels.decode_attention import paged_decode_attention_torch  # noqa: E402

NOT_GPU_LABEL = "NOT a GPU measurement: pipeline check only"
GPU_LABEL = "GPU measurement"
ACCOUNTING_NOTE = (
    "achieved_gbps counts K and V bytes loaded over sequences, query heads, and whole blocks, "
    "including GQA re-reads of the same KV head, which caches may serve. unique_gbps counts each "
    "KV head once. q, the block table, and the output are not counted. GB is 10^9 bytes."
)
# Peak memory bandwidth in GB/s, keyed by the device name without "NVIDIA " and "GeForce ".
# Exact names, because a variant with the same marketing name can have a narrower bus:
# the RTX 4090 Laptop GPU has a 256-bit bus. L4: NVIDIA's datasheet. RTX 4090: GDDR6X,
# 384-bit at 21 Gbps per pin, 384 / 8 * 21.
PEAK_GBPS = {"L4": 300.0, "RTX 4090": 1008.0}
PLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_COLOR = "#52514e"
LAYOUTS = ["fragmented", "contiguous"]
DEFAULT_BATCHES = [1, 4, 16, 32]
DEFAULT_TOKENS = [1024, 4096, 16384, 32768]
CPU_BATCHES = [1, 2]
CPU_TOKENS = [32, 64]
# A 24 GB card with about 8 GB of weights, leaving room for the reference's copies.
DEFAULT_POOL_GIB = {"cuda": 14.0, "cpu": 0.25}


@dataclass(frozen=True)
class KVShape:
    layers: int
    kv_heads: int
    kv_groups: int
    head_dim: int
    slots: int

    @property
    def query_heads(self) -> int:
        return self.kv_heads * self.kv_groups

    def pool_shape(self, blocks: int) -> tuple[int, ...]:
        return (blocks, 2, self.layers, self.kv_heads, self.slots, self.head_dim)

    def block_bytes(self, element_size: int) -> int:
        return 2 * self.layers * self.kv_heads * self.slots * self.head_dim * element_size


SHAPES = {
    "qwen3-4b": KVShape(layers=36, kv_heads=8, kv_groups=4, head_dim=128, slots=16),
    # Same head relationships at a size the interpreter gets through in seconds.
    "micro": KVShape(layers=2, kv_heads=2, kv_groups=2, head_dim=24, slots=16),
}


@dataclass
class Row:
    measurement: str
    kernel: str
    layout: str
    batch: int
    tokens: int
    layers_per_sample: int
    bytes_per_sample: int
    unique_bytes_per_sample: int
    kernel_s: float
    kernel_min_s: float
    kernel_max_s: float
    achieved_gbps: float
    achieved_mean_gbps: float
    achieved_min_gbps: float
    achieved_max_gbps: float
    achieved_stdev_gbps: float
    unique_gbps: float
    achieved_pct: float | None
    achieved_mean_pct: float | None
    achieved_min_pct: float | None
    achieved_max_pct: float | None
    unique_pct: float | None
    # None when the reference's gathered copies did not fit next to the pool.
    reference_s: float | None
    reference_over_kernel: float | None
    max_abs_diff_from_reference: float | None
    samples_s: list[float] = field(default_factory=list)


def sequence_blocks(lengths: list[int], slots: int) -> int:
    return sum(-(-n // slots) for n in lengths)


def bytes_loaded_per_layer(lengths: list[int], shape: KVShape, element_size: int) -> int:
    """K and V bytes one launch loads: whole blocks, once per query head, GQA re-reads included.

    The kernel loads the last block whole even when it is partly empty, so this rounds up.
    """
    per_block_per_kind = shape.slots * shape.head_dim * element_size
    return sequence_blocks(lengths, shape.slots) * shape.query_heads * 2 * per_block_per_kind


def unique_bytes_per_layer(lengths: list[int], shape: KVShape, element_size: int) -> int:
    return bytes_loaded_per_layer(lengths, shape, element_size) // shape.kv_groups


def resolve_peak(name: str, override: float | None) -> tuple[float | None, str]:
    """Peak GB/s and where it came from; None for a device whose peak is not known."""
    if override is not None:
        return override, "--peak-gbps"
    short = name.removeprefix("NVIDIA ").removeprefix("GeForce ").strip()
    if short in PEAK_GBPS:
        return PEAK_GBPS[short], f"table: {short}"
    return None, "unknown"


def percent(gbps: float, peak: float | None) -> float | None:
    return None if peak is None else 100.0 * gbps / peak


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{platform.machine()} {device.type}"


def fill_normal(tensor: torch.Tensor, seed_source: torch.Generator) -> None:
    """Normals through FP32, in pieces, so a pool near device memory is not held twice."""
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=seed_source))
    gen = torch.Generator(tensor.device).manual_seed(seed)
    for start in range(0, tensor.shape[0], 256):
        piece = tensor[start : start + 256]
        piece.copy_(torch.randn(piece.shape, generator=gen, device=tensor.device))


def block_table(layout: str, batch: int, blocks_each: int, pool_blocks: int, device, gen: torch.Generator):
    """Scattered ids across the whole pool, or each sequence's blocks in order.

    In-order is not contiguous for one head: the same layer and head of the next block is
    a block away, 2.36 MB on Qwen3-4B. It differs from scattered in having a regular stride.
    """
    needed = batch * blocks_each
    if layout == "fragmented":
        ids = torch.randperm(pool_blocks, generator=gen)[:needed]
    else:
        ids = torch.arange(needed)
    return ids.reshape(batch, blocks_each).to(dtype=torch.int32, device=device)


def load_triton_backend(device: torch.device):
    """The kernel's launch and check functions, refusing a mode that would mislabel a run."""
    # @triton.jit reads this when triton is first imported, so nothing earlier may import it.
    if device.type != "cuda":
        os.environ.setdefault("TRITON_INTERPRET", "1")
    import triton

    from layover.kernels import triton_decode

    interpreting = bool(triton.knobs.runtime.interpret)
    if device.type == "cuda" and interpreting:
        raise SystemExit("TRITON_INTERPRET is set, which would time the interpreter and not the GPU")
    if device.type != "cuda" and not interpreting:
        raise SystemExit("TRITON_INTERPRET=0 needs --device cuda")
    return triton_decode


def select_kernel(requested: str, device: torch.device):
    """Returns (name, launch, check). launch takes the reference's arguments plus the output."""
    if requested == "auto":
        requested = "triton" if device.type == "cuda" or importlib.util.find_spec("triton") else "reference"
        if requested == "reference":
            print("triton is not installed: timing only the PyTorch reference", flush=True)
    if requested == "reference":

        def launch(q, pool, layer, table, lengths, groups, out):
            out.copy_(paged_decode_attention_torch(q, pool, layer, table, lengths, groups))

        return requested, launch, None
    backend = load_triton_backend(device)
    return requested, backend.run_kernel, backend.validate


def measure(args, shape, pool, device, kernel_name, launch, check, layout, batch, tokens, peak, label, gen) -> Row:
    element_size = pool.element_size()
    lengths = [tokens] * batch
    blocks_each = -(-tokens // shape.slots)
    table = block_table(layout, batch, blocks_each, pool.shape[0], device, gen)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    q = torch.empty(batch, shape.query_heads, shape.head_dim, dtype=pool.dtype, device=device)
    fill_normal(q, gen)
    out = torch.empty_like(q)
    layers = range(args.layers_per_sample)
    if check is not None:
        # Validated once and outside the timing, since the launch itself never changes.
        for layer in layers:
            check(q, pool, layer, table, lengths_tensor, shape.kv_groups)

    def run_kernel():
        for layer in layers:
            launch(q, pool, layer, table, lengths_tensor, shape.kv_groups, out)

    def run_reference():
        for layer in layers:
            paged_decode_attention_torch(q, pool, layer, table, lengths_tensor, shape.kv_groups)

    samples = [timed(device, run_kernel) for _ in range(args.warmup + args.repeats)][args.warmup:]

    # The last layer's output against what attention can produce, a weighted mean of V. A
    # wrong kernel should not get to report a bandwidth.
    v_max = float(pool[:, 1, layers[-1]].float().abs().max())
    if not out.isfinite().all() or float(out.float().abs().max()) > v_max * 1.01:
        raise RuntimeError(f"kernel output is not a weighted mean of V at batch {batch}, {tokens} tokens")

    # The reference gathers every K and V and repeats them per query head, up to several
    # times the bytes the kernel touches. It is only the comparison, so running out of
    # memory for it must not cost the measurement that was just taken.
    reference_s, diff = None, None
    try:
        reference_samples = [timed(device, run_reference) for _ in range(args.warmup + args.repeats)][args.warmup:]
        reference_s = statistics.median(reference_samples)
        expected = paged_decode_attention_torch(q, pool, layers[-1], table, lengths_tensor, shape.kv_groups)
        diff = float((out.float() - expected.float()).abs().max())
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {batch}, {tokens} tokens: the reference does not fit in memory, kernel not compared", flush=True)
    # Far above any rounding, and still far below what wrong blocks or heads give on short
    # contexts. tests/test_triton_decode.py is the accurate check.
    if diff is not None and diff > 0.25 * v_max:
        raise RuntimeError(f"kernel differs from the reference by {diff:.3g} at batch {batch}, {tokens} tokens")

    loaded = bytes_loaded_per_layer(lengths, shape, element_size) * len(layers)
    unique = unique_bytes_per_layer(lengths, shape, element_size) * len(layers)
    rates = [loaded / s / 1e9 for s in samples]
    median_s = statistics.median(samples)
    achieved = statistics.median(rates)
    unique_gbps = unique / median_s / 1e9
    return Row(
        measurement=label,
        kernel=kernel_name,
        layout=layout,
        batch=batch,
        tokens=tokens,
        layers_per_sample=len(layers),
        bytes_per_sample=loaded,
        unique_bytes_per_sample=unique,
        kernel_s=median_s,
        kernel_min_s=min(samples),
        kernel_max_s=max(samples),
        achieved_gbps=achieved,
        achieved_mean_gbps=statistics.fmean(rates),
        achieved_min_gbps=min(rates),
        achieved_max_gbps=max(rates),
        achieved_stdev_gbps=statistics.stdev(rates) if len(rates) > 1 else 0.0,
        unique_gbps=unique_gbps,
        achieved_pct=percent(achieved, peak),
        achieved_mean_pct=percent(statistics.fmean(rates), peak),
        achieved_min_pct=percent(min(rates), peak),
        achieved_max_pct=percent(max(rates), peak),
        unique_pct=percent(unique_gbps, peak),
        reference_s=reference_s,
        reference_over_kernel=None if reference_s is None else reference_s / median_s,
        max_abs_diff_from_reference=diff,
        samples_s=samples,
    )


def summarize(rows: list[Row]) -> dict:
    """Across cases, per layout: the headline with its spread."""
    summary = {}
    for layout in LAYOUTS:
        chosen = [r for r in rows if r.layout == layout]
        if not chosen:
            continue
        gbps = [r.achieved_gbps for r in chosen]
        summary[layout] = {
            "cases": len(chosen),
            "achieved_gbps_mean": statistics.fmean(gbps),
            "achieved_gbps_min": min(gbps),
            "achieved_gbps_max": max(gbps),
        }
        if chosen[0].achieved_pct is not None:
            pct = [r.achieved_pct for r in chosen]
            summary[layout] |= {
                "achieved_pct_mean": statistics.fmean(pct),
                "achieved_pct_min": min(pct),
                "achieved_pct_max": max(pct),
                "unique_pct_mean": statistics.fmean(r.unique_pct for r in chosen),
            }
    return summary


def write_csv(rows: list[Row], path: Path) -> None:
    fields = [k for k in asdict(rows[0]) if k != "samples_s"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow({k: "" if values[k] is None else values[k] for k in fields})


def write_plot(rows: list[Row], meta: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    use_pct = meta["peak_gbps"] is not None
    # Nothing here reads a DRAM counter, so the figure is only known to lie in this range.
    low_key, high_key = ("unique_pct", "achieved_pct") if use_pct else ("unique_gbps", "achieved_gbps")
    layouts = [layout for layout in LAYOUTS if any(r.layout == layout for r in rows)]
    batches = sorted({r.batch for r in rows})
    fig, axes = plt.subplots(
        1, len(layouts), figsize=(4.2 * len(layouts) + 1, 4.4), dpi=160, sharey=True, squeeze=False
    )
    for ax, layout in zip(axes[0], layouts):
        for color, batch in zip(PLOT_COLORS, batches):
            series = sorted((r for r in rows if r.layout == layout and r.batch == batch), key=lambda r: r.tokens)
            if not series:
                continue
            x = [r.tokens for r in series]
            ax.fill_between(
                x, [getattr(r, low_key) for r in series], [getattr(r, high_key) for r in series],
                color=color, alpha=0.18, linewidth=0,
            )
            ax.plot(
                x, [getattr(r, high_key) for r in series], color=color, linewidth=1.5, marker="o", markersize=4,
                label=f"batch {batch}, with GQA re-reads",
            )
            ax.plot(
                x, [getattr(r, low_key) for r in series], color=color, linewidth=1.2, linestyle="--", marker="o", markersize=3,
                label=f"batch {batch}, each KV head once",
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xticks(sorted({r.tokens for r in rows}))
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=8, colors=TEXT_COLOR)
        ax.set_xlabel("Context length (tokens)", fontsize=9, color=TEXT_COLOR)
        ax.set_title(f"{layout} block table", fontsize=9, loc="left", color=TEXT_COLOR)
        ax.set_ylim(bottom=0)
        ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c8c8c8")
    ylabel = f"K/V load rate, % of {meta['peak_gbps']:g} GB/s peak" if use_pct else "K/V load rate (GB/s)"
    axes[0][0].set_ylabel(f"{ylabel}\nband: each KV head once to with GQA re-reads", fontsize=9, color=TEXT_COLOR)
    legend = axes[0][-1].legend(frameon=False, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    title = f"Decode kernel loads ({meta['kernel']}): {meta['shape']}, {meta['dtype']}, {meta['device_name']}"
    title += "\nDRAM's share lies in the band; no DRAM counter was read" + ("" if meta["measured_on_gpu"] else f". {meta['label']}")
    fig.suptitle(title, fontsize=9, x=0.01, ha="left", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--kernel", default="auto", choices=["auto", "triton", "reference"], help="what to report on")
    p.add_argument("--shape", default="qwen3-4b", choices=sorted(SHAPES))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch-sizes", type=int, nargs="+", help="sequences per launch")
    p.add_argument("--tokens", type=int, nargs="+", help="context lengths")
    p.add_argument("--layouts", nargs="+", default=LAYOUTS, choices=LAYOUTS)
    p.add_argument("--pool-gib", type=float, help="size of the KV pool the tables scatter across")
    p.add_argument("--layers-per-sample", type=int, help="launches per timed sample; every layer on CUDA, one on CPU")
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--warmup", type=int, default=2, help="leading repeats to discard")
    p.add_argument("--peak-gbps", type=float, help="peak memory bandwidth, for a GPU not in the table")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/decode_bandwidth"))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"
    if on_gpu and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmup < 1:
        raise SystemExit("--warmup must be at least 1: the first launch compiles the kernel")
    if args.peak_gbps is not None and args.peak_gbps <= 0:
        raise SystemExit("--peak-gbps must be positive")
    if args.layers_per_sample is not None and args.layers_per_sample < 1:
        raise SystemExit("--layers-per-sample must be at least 1")
    label = GPU_LABEL if on_gpu else NOT_GPU_LABEL
    shape = SHAPES[args.shape]
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    batches = args.batch_sizes or (DEFAULT_BATCHES if on_gpu else CPU_BATCHES)
    tokens = args.tokens or (DEFAULT_TOKENS if on_gpu else CPU_TOKENS)
    if len(set(batches)) > len(PLOT_COLORS):
        raise SystemExit(f"at most {len(PLOT_COLORS)} batch sizes: the plot has that many colors")
    if args.layers_per_sample is None:
        args.layers_per_sample = shape.layers if on_gpu else 1
    args.layers_per_sample = min(args.layers_per_sample, shape.layers)

    element_size = torch.empty(0, dtype=dtype).element_size()
    pool_gib = DEFAULT_POOL_GIB[device.type] if args.pool_gib is None else args.pool_gib
    pool_blocks = int(pool_gib * 2**30 // shape.block_bytes(element_size))
    if pool_blocks < 1:
        raise SystemExit(f"--pool-gib {pool_gib} holds no blocks of {shape.block_bytes(element_size):,} bytes")
    if on_gpu:
        free, _ = torch.cuda.mem_get_info(device)
        if pool_blocks * shape.block_bytes(element_size) > free:
            raise SystemExit(f"a {pool_gib} GiB pool does not fit in {free / 2**30:.1f} GiB of free device memory")
    name = device_name(device)
    peak, peak_source = resolve_peak(name, args.peak_gbps)
    kernel_name, launch, check = select_kernel(args.kernel, device)

    gen = torch.Generator().manual_seed(args.seed)
    pool = torch.empty(shape.pool_shape(pool_blocks), dtype=dtype, device=device)
    fill_normal(pool, gen)

    def build_meta(rows_so_far: list[Row], complete: bool) -> dict:
        return {
            "label": label,
            "complete": complete,
            "measured_on_gpu": on_gpu,
            "device": args.device,
            "device_name": name,
            "kernel": kernel_name,
            "shape": args.shape,
            "kv_shape": asdict(shape),
            "dtype": args.dtype,
            "pool_blocks": pool_blocks,
            "block_bytes": shape.block_bytes(element_size),
            "peak_gbps": peak,
            "peak_source": peak_source,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "layers_per_sample": args.layers_per_sample,
            "seed": args.seed,
            "torch": torch.__version__,
            "accounting": ACCOUNTING_NOTE,
            "summary": summarize(rows_so_far),
            "skipped": skipped,
            "rows": [asdict(r) for r in rows_so_far],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    print(f"{label}. Device {name}, pool {pool_blocks:,} blocks, kernel {kernel_name}.", flush=True)
    print(ACCOUNTING_NOTE, flush=True)
    if peak is None:
        print(f"No known peak for '{name}': refusing to print a percentage. Pass --peak-gbps.", flush=True)
    for layout in args.layouts:
        for batch in batches:
            for count in tokens:
                needed = batch * -(-count // shape.slots)
                if needed > pool_blocks:
                    skipped.append({"layout": layout, "batch": batch, "tokens": count, "blocks_needed": needed})
                    print(f"{layout:>10} batch {batch:>3} {count:>6} tokens: skipped, needs {needed:,} blocks", flush=True)
                    continue
                row = measure(args, shape, pool, device, kernel_name, launch, check, layout, batch, count, peak, label, gen)
                rows.append(row)
                pct = "" if row.achieved_pct is None else f" {row.unique_pct:6.1f}% to {row.achieved_pct:6.1f}% of peak"
                versus = "" if row.reference_over_kernel is None else f"; reference {row.reference_over_kernel:.3g}x the kernel's time"
                print(
                    f"{layout:>10} batch {row.batch:>3} {row.tokens:>6} tokens: {row.achieved_gbps:9.3g} GB/s{pct}"
                    f" (min {row.achieved_min_gbps:.3g}, max {row.achieved_max_gbps:.3g});"
                    f" unique {row.unique_gbps:.3g} GB/s{versus}",
                    flush=True,
                )
                if row.achieved_pct is not None and row.achieved_pct > 100:
                    print("  above peak: part of these loads is being served from cache, not memory", flush=True)
                # Saved as it goes, so a crash late in a rented session keeps what it measured.
                meta = build_meta(rows, complete=False)
                write_csv(rows, args.out / "decode_bandwidth.csv")
                (args.out / "decode_bandwidth.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not rows:
        raise SystemExit("every requested case needs more blocks than the pool holds")

    meta = build_meta(rows, complete=True)
    write_csv(rows, args.out / "decode_bandwidth.csv")
    (args.out / "decode_bandwidth.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_plot(rows, meta, args.out / "decode_bandwidth.png")
    for layout, stats in meta["summary"].items():
        if "achieved_pct_mean" in stats:
            print(
                f"{layout}: {stats['unique_pct_mean']:.1f}% to {stats['achieved_pct_mean']:.1f}% of {peak:g} GB/s peak on"
                f" average over {stats['cases']} cases, from each KV head once to every GQA re-read; DRAM's share lies"
                f" between them. With re-reads, cases range {stats['achieved_pct_min']:.1f}% to {stats['achieved_pct_max']:.1f}%"
            )
        else:
            print(
                f"{layout}: achieved {stats['achieved_gbps_mean']:.3g} GB/s on average over {stats['cases']} cases,"
                f" range {stats['achieved_gbps_min']:.3g} to {stats['achieved_gbps_max']:.3g}"
            )
    if not on_gpu:
        print(f"WARNING: {NOT_GPU_LABEL}. Do not use these numbers as results.")
    return meta


if __name__ == "__main__":
    main()

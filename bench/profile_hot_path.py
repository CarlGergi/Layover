#!/usr/bin/env python3
"""Prefill and decode rates of Layover's engine, and of vLLM on the same shapes, for profiling.

    python bench/profile_hot_path.py --out results/profile/layover
    LAYOVER_NVTX=1 nsys profile --trace=cuda,nvtx,osrt -o results/profile/layover \\
        python bench/profile_hot_path.py --out results/profile/layover --quick
    python bench/profile_hot_path.py --out results/profile/layover --torch-trace
    .venvs/vllm/bin/python bench/profile_hot_path.py --system vllm --out results/profile/vllm

Prefill is measured as a chunk arriving after a context, the matrix's case: each agent turn adds
tokens to a long conversation. Decode is one step at a batch of forks of one context, as the
smoke measures it. Both run the model in bfloat16 with its matched prefill chunk, and every
number is a median of timed repeats after a warm-up. vLLM runs with prefix caching off, so the
context it prefills first is not reused, and gets the same chunk and lengths.

--device cpu runs the micro model through the same steps to test this script; its numbers
describe nothing and say so.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PIPELINE_LABEL = "PIPELINE CHECK on CPU: micro model. Measures nothing about a GPU."


def shapes(on_gpu: bool, quick: bool) -> dict:
    """Contexts and batches; quick keeps an nsys capture short enough to read."""
    if not on_gpu:
        return {"chunk": 32, "contexts": [0, 64], "decode_context": 64, "batches": [1, 4], "repeats": 2}
    if quick:
        return {"chunk": 2048, "contexts": [0, 8192], "decode_context": 8192, "batches": [1, 32], "repeats": 3}
    return {"chunk": 2048, "contexts": [0, 8192, 16384, 32768], "decode_context": 8192, "batches": [1, 8, 32], "repeats": 5}


def median_after_warmup(samples: list[float]) -> float:
    return statistics.median(samples[1:] if len(samples) > 1 else samples)


def layover_rates(model_name: str, device_name: str, plan: dict, torch_trace: Path | None) -> dict:
    import torch

    from bench.kv_crossover import load_layover_model, make_ids, timed
    from layover.kernels import triton_decode
    from layover.kv import BlockTable, PagedKVCache

    device = torch.device(device_name)
    cfg, model = load_layover_model(model_name, device, torch.bfloat16, 0)
    if device.type == "cuda":
        # The kernel the server runs. A tree from before the trusted path, profiled for a
        # before-and-after, has only the checked one.
        model.decode_attention = getattr(triton_decode, "paged_decode_attention_triton_trusted", triton_decode.paged_decode_attention_triton)
    chunk = plan["chunk"]
    longest = max(plan["contexts"] + [plan["decode_context"]]) + chunk
    cache = PagedKVCache(cfg, longest // 16 + max(plan["batches"]) + 16, device, torch.bfloat16)
    ids = make_ids(cfg, longest, device, 0)
    prefill, decode = {}, {}

    def fill(table, end):
        for start in range(0, end, chunk):
            model.prefill(cache, table, ids[0, start : min(start + chunk, end)], start)

    with torch.inference_mode():
        for context in plan["contexts"]:
            table = BlockTable()
            fill(table, context)
            samples = []
            for _ in range(plan["repeats"] + 1):
                samples.append(timed(device, lambda: model.prefill(cache, table, ids[0, context : context + chunk], context)))
                cache.truncate(table, context)
            prefill[str(context)] = chunk / median_after_warmup(samples)
            cache.release(table)
        base = BlockTable()
        fill(base, plan["decode_context"])
        gen = torch.Generator().manual_seed(0)
        for batch in plan["batches"]:
            tables = [cache.fork(base) for _ in range(batch)]
            tokens = torch.randint(0, cfg.vocab_size, (batch,), generator=gen).to(device)
            samples = [timed(device, lambda: model.decode(cache, tables, tokens)) for _ in range(plan["repeats"] + 1)]
            decode[str(batch)] = median_after_warmup(samples)
            for table in tables:
                cache.release(table)
        if torch_trace is not None:
            record_torch_trace(torch_trace, device, lambda: model.prefill(cache, base, ids[0, plan["decode_context"] : plan["decode_context"] + chunk], plan["decode_context"]),
                               lambda: model.decode(cache, [cache.fork(base) for _ in range(max(plan["batches"]))], torch.zeros(max(plan["batches"]), dtype=torch.long, device=device)))
    return {"prefill_tokens_per_s": prefill, "decode_step_s": decode}


def record_torch_trace(path: Path, device, prefill_step, decode_step) -> None:
    """A Chrome trace and a table of the most expensive operators, for one prefill chunk and one decode step."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    path.mkdir(parents=True, exist_ok=True)
    for name, step in (("prefill", prefill_step), ("decode", decode_step)):
        step()
        with profile(activities=activities, record_shapes=True) as prof:
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(path / f"torch_{name}.json"))
        sort = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        (path / f"torch_{name}_top.txt").write_text(prof.key_averages().table(sort_by=sort, row_limit=30))


def vllm_rates(model_name: str, plan: dict) -> dict:
    """The same shapes through vLLM's offline engine: prefill as the time to the first token of
    a prompt of context + chunk less that of the context alone, decode as a step at a batch."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", enable_prefix_caching=False, max_num_batched_tokens=plan["chunk"],
              enable_chunked_prefill=True, max_model_len=max(plan["contexts"]) + plan["chunk"] + 64, gpu_memory_utilization=0.85, seed=0)
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def run(prompt_ids, params):
        start = time.perf_counter()
        llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
        return time.perf_counter() - start

    vocab = 1000
    prefill = {}
    for context in plan["contexts"]:
        whole = [[(i * 7919) % vocab + 10 for i in range(context + plan["chunk"])]]
        head = [whole[0][:context]] if context else None
        samples = []
        for _ in range(plan["repeats"] + 1):
            samples.append(run(whole, one) - (run(head, one) if head else 0.0))
        prefill[str(context)] = plan["chunk"] / max(median_after_warmup(samples), 1e-9)
    decode = {}
    steps = 64
    for batch in plan["batches"]:
        prompts = [[(i * 7919 + b) % vocab + 10 for i in range(plan["decode_context"])] for b in range(batch)]
        many = SamplingParams(max_tokens=steps + 1, temperature=0.0, ignore_eos=True)
        samples = [(run(prompts, many) - run(prompts, one)) / steps for _ in range(plan["repeats"] + 1)]
        decode[str(batch)] = median_after_warmup(samples)
    return {"prefill_tokens_per_s": prefill, "decode_step_s": decode}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["layover", "vllm"], default="layover")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", action="store_true", help="fewer shapes, for an nsys capture")
    p.add_argument("--torch-trace", action="store_true", help="also record torch.profiler traces of one prefill chunk and one decode step")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    on_gpu = args.device == "cuda"
    if args.system == "vllm" and not on_gpu:
        print("Refused: vLLM is measured on a GPU only.", file=sys.stderr)
        return 2
    plan = shapes(on_gpu, args.quick)
    model = args.model if on_gpu else "micro"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.system == "layover":
        rates = layover_rates(model, args.device, plan, args.out if args.torch_trace else None)
    else:
        rates = vllm_rates(model, plan)
    result = {"system": args.system, "model": model, "device": args.device, "shapes": plan, **rates}
    if not on_gpu:
        result["label"] = PIPELINE_LABEL
    (args.out / "rates.json").write_text(json.dumps(result, indent=2) + "\n")
    for context, rate in result["prefill_tokens_per_s"].items():
        print(f"prefill: {plan['chunk']} tokens after {context}: {rate:,.0f} tokens/s")
    for batch, step in result["decode_step_s"].items():
        print(f"decode: batch {batch} at {plan['decode_context']} tokens: {step * 1000:.1f} ms per step, {int(batch) / step:,.0f} tokens/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

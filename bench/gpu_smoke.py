#!/usr/bin/env python3
"""Stop a broken GPU setup in minutes, before the expensive stages.

    python bench/gpu_smoke.py                on the GPU machine, right after setup
    python bench/gpu_smoke.py --device cpu   a pipeline check of this runner, measures nothing

Each check drives the code a benchmark would drive, because testing a re-implementation proves
nothing: the real weights, the Triton kernel, the native library, bench/kv_crossover.py, and the
server started with the matrix's own flags. Every check runs in a process of its own, with its own
time limit, so a hang or a crash costs one check and the GPU comes back between checks. A check
whose dependency failed is recorded as blocked, never skipped in silence.

results/smoke/smoke.json is rewritten after every check. On success measured_rates.json goes
through bench/run_matrix.py --measured-rates, and refined_hours.txt holds the refined plan.
Exits 0 if nothing failed, 1 if something did, and 2 if it refuses to run.

--device cpu runs what needs no GPU, on the micro model and the host build of the native library,
and marks the rest not applicable. It writes to results/smoke_pipeline_check, labelled throughout,
so that its numbers cannot be taken for a GPU's.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.systems import BLOCK_TOKENS, _descendants, _kill_quietly  # noqa: E402

DEFAULT_CONFIG = REPO / "bench" / "configs" / "matrix.json"
PIPELINE_LABEL = "PIPELINE CHECK on CPU: micro model, host build. Measures nothing about a GPU."
DEFAULT_PLAN = "under-20h"

# A normal run is about 10 minutes (an assumption until the first GPU session). This limit is on
# the broken case: a check that starts past it is recorded as blocked.
TOTAL_LIMIT_S = 1200
# How long a timed-out check gets to stop its server, whose own stop can take 30 s.
TERM_GRACE_S = 40
GPU_TESTS_LIMIT_S = 180

PROMPTS = ("The capital of France is", "def fibonacci(n):")
GREEDY_STEPS = 4
# Largest step between neighbouring BF16 values, relative, and half of it as rounding.
BF16_EPS = 2.0**-7
BF16_ROUNDOFF = 2.0**-8

# The GPU tests most likely to catch a broken setup, each named for what it would catch: the
# library and torch disagreeing about the context, the FP32 attention path that needs 8 GiB, a
# kernel that will not launch at the matrix's shapes, and the scheduler's real swap path.
GPU_TESTS = (
    "tests/test_gpu.py::test_native_library_and_torch_share_a_context",
    "tests/test_gpu.py::test_long_context_prefill_takes_a_fused_attention_kernel_on_cuda",
    "tests/test_gpu.py::test_sanitize_shapes_launch_on_cuda",
    "tests/test_gpu.py::test_scheduler_swaps_through_the_native_path_on_real_weights[fp8]",
    "tests/test_native.py::test_gpu_flavor_is_the_cuda_build",
    "tests/test_transfer.py::test_gpu_fp8_round_trip_within_the_bound_and_matches_the_reference",
    "tests/test_triton_decode.py::test_kernel_matches_reference_on_cuda_at_qwen3_4b_shape",
    "tests/test_fp8.py::test_block_kernels_on_cuda_match_the_reference",
)


class SmokeFailure(RuntimeError):
    """A check found the machine broken. The message is the row in the table."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


@dataclass
class Context:
    """What a check's process is told. Measurements of earlier checks travel in measured."""

    device: str
    model: str
    config: str
    out: str
    tmp: str
    plan: str = DEFAULT_PLAN
    measured: dict = field(default_factory=dict)

    @property
    def on_gpu(self) -> bool:
        return self.device == "cuda"

    @property
    def cost_model(self) -> Path:
        return Path(self.tmp) / "crossover" / "cost_model.json"

    def matrix(self) -> dict:
        """The matrix config at the settings of the plan being priced, which may change them."""
        from bench.run_matrix import effective_config

        return effective_config(json.loads(Path(self.config).read_text()), self.plan)


def bf16():
    import torch

    return torch.bfloat16


def load_real_or_micro(ctx: Context):
    """The model through kv_crossover's loader, which knows both the micro model and checkpoints."""
    import torch

    from bench.kv_crossover import load_layover_model

    return load_layover_model(ctx.model, torch.device(ctx.device), bf16(), 0)


def prefill_table(model, cache, ids, chunk: int):
    from layover.kv import BlockTable

    table = BlockTable()
    for start in range(0, ids.shape[1], chunk):
        model.prefill(cache, table, ids[0, start : start + chunk], start)
    return table


# ---- the checks ---------------------------------------------------------------------------------


def check_environment(ctx: Context) -> dict:
    require("TRITON_INTERPRET" not in os.environ, "TRITON_INTERPRET is set: kernels would run as Python on the CPU")
    import torch

    from bench.gpu_sampler import GIB, NvmlBackend
    from bench.run_matrix import checkpoint_weight_bytes

    require(torch.cuda.is_available(), "torch cannot see a CUDA GPU")
    require(torch.version.cuda is not None, f"this torch ({torch.__version__}) is a CPU-only build")
    gpu = NvmlBackend(ctx.matrix().get("gpu_index", 0))
    try:
        info = gpu.device_info()
        idle = gpu.read().memory_used_bytes
    finally:
        gpu.close()
    # A wrong library would still answer, so the product has to be right and not only arrive.
    gen = torch.Generator().manual_seed(0)
    a, b = torch.randn(1024, 1024, generator=gen), torch.randn(1024, 1024, generator=gen)
    product = (a.to("cuda") @ b.to("cuda")).cpu()
    expected = a @ b
    error = float((product - expected).abs().max() / expected.abs().max())
    require(error < 1e-3, f"a cuBLAS matmul disagrees with the CPU by {error:.2e} of its largest entry")
    matched = ctx.matrix()["matched"]
    free, total = torch.cuda.mem_get_info()
    weights = checkpoint_weight_bytes(ctx.model, matched["dtype"])
    needed = weights + matched["kv_gib"] * GIB + GIB
    require(
        free >= needed,
        f"{free / GIB:.1f} GiB of GPU memory is free and the matrix needs {needed / GIB:.1f} GiB "
        f"(weights, {matched['kv_gib']} GiB of KV, 1 GiB of room); is something else on the GPU?",
    )
    return {
        "summary": f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"{free / GIB:.1f} of {total / GIB:.1f} GiB free",
        "nvml": info, "nvml_memory_used_bytes": idle, "matmul_relative_error": error,
        "gpu_free_bytes": free, "gpu_needed_bytes": needed,
    }


def logits_gap(got, expected) -> float:
    """The largest logit difference as a fraction of test_engine's BF16 tolerance. 1 is the limit."""
    got, expected = got.float(), expected.float()
    limit = 4 * BF16_EPS * expected.abs().max().item()
    return float((got - expected).abs().max().item() / limit)


def check_reference_parity(ctx: Context) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from layover.kv import BlockTable, PagedKVCache
    from layover.weights import load_model, resolve_checkpoint

    root = str(resolve_checkpoint(ctx.model))
    model = load_model(root, device=ctx.device, dtype=bf16())
    hf = AutoModelForCausalLM.from_pretrained(root, dtype=bf16(), attn_implementation="sdpa").to(ctx.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(root)
    worst = 0.0
    with torch.inference_mode():
        for text in PROMPTS:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(ctx.device)
            cache = PagedKVCache(model.cfg, 8, device=ctx.device, dtype=bf16())
            table = BlockTable()
            logits = model.prefill(cache, table, ids[0], 0, all_logits=True)
            gap = logits_gap(logits, hf(ids).logits[0])
            require(gap <= 1, f"prefill logits on {text!r} are {gap:.2f}x the BF16 tolerance off transformers")
            worst = max(worst, gap)
            sequence = ids[0].tolist()
            token = int(logits[-1].argmax())
            for step in range(GREEDY_STEPS):
                sequence.append(token)
                row = model.decode(cache, [table], torch.tensor([token], device=ctx.device))[0]
                expected = hf(torch.tensor([sequence], device=ctx.device)).logits[0, -1]
                gap = logits_gap(row, expected)
                require(gap <= 1, f"decode step {step} on {text!r} is {gap:.2f}x the BF16 tolerance off transformers")
                worst = max(worst, gap)
                token = int(row.argmax())
    return {"summary": f"{len(PROMPTS)} prompts, worst logit gap {worst:.3f}x the BF16 tolerance", "worst_gap": worst}


def attention_gap(got, reference, values) -> float:
    """The largest difference as a fraction of the tolerance test_triton_decode reasons out for
    BF16 on CUDA, which rounds partial sums and so does not round like the kernel's FP32."""
    spread = 2 * 6 * BF16_ROUNDOFF * math.sqrt(got.shape[-1]) / 2.45 * values.pow(2).mean().sqrt()
    limit = spread + 2 * BF16_ROUNDOFF * values.abs().max()
    return float((got.float() - reference.float()).abs().max().item() / limit.item())


def triton_gaps(cfg, model, device: str, tokens: int, chunk: int) -> tuple[float, float]:
    """(worst attention gap, logit gap) of the Triton kernel against the reference, on real KV."""
    import torch

    from bench.kv_crossover import make_ids
    from layover.kernels.decode_attention import paged_decode_attention_torch as reference
    from layover.kernels.triton_decode import paged_decode_attention_triton as kernel
    from layover.kv import PagedKVCache

    cache = PagedKVCache(cfg, tokens // BLOCK_TOKENS + 32, device, bf16())
    gaps = []

    def both(q, pool, layer, block_table, lengths, kv_groups):
        expected = reference(q, pool, layer, block_table, lengths, kv_groups)
        got = kernel(q, pool, layer, block_table, lengths, kv_groups)
        gaps.append(attention_gap(got, expected, pool[block_table.long().unique(), 1, layer].float()))
        return expected

    with torch.inference_mode():
        table = prefill_table(model, cache, make_ids(cfg, tokens, torch.device(device), 0), chunk)
        # Unequal lengths, the padded case a real batch has.
        lengths = (tokens, tokens - 24, 17, 16)
        next_tokens = torch.tensor([1, 2, 3, 4], device=device)
        model.decode_attention = both
        expected = model.decode(cache, [cache.fork(table, n) for n in lengths], next_tokens)
        model.decode_attention = kernel
        got = model.decode(cache, [cache.fork(table, n) for n in lengths], next_tokens)
    return max(gaps), logits_gap(got, expected)


def check_triton_decode(ctx: Context) -> dict:
    import triton

    require(not triton.knobs.runtime.interpret, "Triton is in its interpreter")
    cfg, model = load_real_or_micro(ctx)
    attention, logits = triton_gaps(cfg, model, ctx.device, 1024, ctx.matrix()["matched"]["prefill_chunk"])
    require(attention <= 1, f"Triton's attention on real KV is {attention:.2f}x the tolerance off the reference")
    require(logits <= 1, f"logits through Triton are {logits:.2f}x the BF16 tolerance off the reference")
    return {
        "summary": f"{cfg.num_hidden_layers} layers on real KV, worst attention gap {attention:.3f}x, logit gap {logits:.3f}x",
        "worst_attention_gap": attention, "logit_gap": logits,
    }


def check_decode_rates(ctx: Context) -> dict:
    """Decode step times at the kernel the matrix uses, for the refined plan."""
    import torch

    from bench.kv_crossover import make_ids, timed
    from layover.kv import PagedKVCache

    cfg, model = load_real_or_micro(ctx)
    device = torch.device(ctx.device)
    matched = ctx.matrix()["matched"]
    context, batch_sizes, chunk = (8192, (1, matched["max_running"]), matched["prefill_chunk"]) if ctx.on_gpu else (64, (1, 4), 64)
    if ctx.on_gpu:
        from layover.kernels.triton_decode import paged_decode_attention_triton_trusted

        # As the server runs it: the model checks each step's tables on the host.
        model.decode_attention = paged_decode_attention_triton_trusted
    cache = PagedKVCache(cfg, context // 16 + max(batch_sizes) + 8, ctx.device, bf16())
    steps, gen = {}, torch.Generator().manual_seed(0)
    with torch.inference_mode():
        table = prefill_table(model, cache, make_ids(cfg, context, device, 0), chunk)
        for batch in batch_sizes:
            # Forks share the context, so a whole batch of it needs only a block each to write in.
            tables = [cache.fork(table) for _ in range(batch)]
            tokens = torch.randint(0, cfg.vocab_size, (batch,), generator=gen).to(device)
            samples = [timed(device, lambda: model.decode(cache, tables, tokens)) for _ in range(9)]
            steps[str(batch)] = statistics.median(samples[1:])
            for forked in tables:
                cache.release(forked)
    return {
        "summary": f"at {context} tokens of context: " + ", ".join(f"batch {b} {s * 1e3:.1f} ms" for b, s in steps.items()),
        "measured": {"decode_step_s": steps, "decode_context_tokens": context},
    }


def torch_shares_a_context(backend, cache) -> None:
    """A pinned buffer of the library's must look pinned to torch, and torch's stream must work in it."""
    import zlib

    import torch

    device = cache.device
    before = zlib.crc32(cache.pool[[5, 2]].cpu().view(torch.uint16).numpy().tobytes())
    out = backend.start_out(cache, [5, 2], False)
    out.wait()
    target = torch.zeros(cache.block_bytes * 2, dtype=torch.uint8, device=device)
    target.copy_(out.payload.memory[: cache.block_bytes * 2], non_blocking=True)
    torch.cuda.current_stream().synchronize()
    require(zlib.crc32(target.cpu().numpy().tobytes()) == before, "torch read other bytes from the library's pinned buffer")
    backend.free(out.payload)
    stream = torch.cuda.Stream()
    backend.engine.ext_wait(stream.cuda_stream, 0, 0)
    backend.engine.ext_record(stream.cuda_stream, 0, 0)
    finished = backend.engine.ext_finish(stream.cuda_stream)
    backend.engine.wait(finished)
    backend.engine.release(finished)


def check_native_transfer(ctx: Context) -> dict:
    import torch

    from bench.kv_crossover import layover_prefill_cache, make_ids, measure_transfer
    from layover.kv.native import ABI_VERSION, build_host, load_library
    from layover.kv.transfer import NativeBackend

    device = torch.device(ctx.device)
    if not ctx.on_gpu:
        build_host()
    # load_library refuses a stale build by itself, and says so.
    abi = load_library("cuda" if ctx.on_gpu else "host").lt_abi_version()
    backend = NativeBackend(device)
    try:
        cfg, model = load_real_or_micro(ctx)
        tokens, chunk = (1024, ctx.matrix()["matched"]["prefill_chunk"]) if ctx.on_gpu else (64, 64)
        with torch.inference_mode():
            cache = layover_prefill_cache(model, make_ids(cfg, tokens, device, 0), chunk, bf16())
        # Both raise on a wrong byte, or a value outside FP8's bound, over a random scatter.
        bf16_s = measure_transfer(backend, cache, False, device, 0)
        fp8_s = measure_transfer(backend, cache, True, device, 1)
        if ctx.on_gpu:
            torch_shares_a_context(backend, cache)
    finally:
        backend.close()
    return {
        "summary": f"ABI {abi}; {tokens} tokens of real KV: BF16 byte-exact, FP8 within its bound",
        "abi": abi, "abi_expected": ABI_VERSION, "bf16_out_in_s": bf16_s, "fp8_out_in_s": fp8_s,
    }


def crossover_argv(ctx: Context) -> list[str]:
    tokens, chunk, repeats, warmup = (["512", "2048"], ctx.matrix()["matched"]["prefill_chunk"], 2, 1) if ctx.on_gpu else (["64", "128"], 64, 1, 0)
    return [
        "--device", ctx.device, "--model", ctx.model, "--transfer", "native", "--offload-dtype", "fp8",
        "--tokens", *tokens, "--chunk", str(chunk), "--repeats", str(repeats), "--warmup", str(warmup),
        "--out", str(ctx.cost_model.parent),
    ]


def check_crossover(ctx: Context) -> dict:
    """The one tiny benchmark. It goes to the temporary directory, so no result can come from it."""
    from bench.kv_crossover import main as kv_crossover

    meta = kv_crossover(crossover_argv(ctx))
    require(meta["complete"] and meta["transfer"] == "native" and meta["offload_dtype"] == "fp8", f"kv_crossover ran the wrong path: {meta}")
    require(meta["measured_on_gpu"] == ctx.on_gpu, f"kv_crossover says measured_on_gpu={meta['measured_on_gpu']}")
    require(all(sum(meta[f"fp8_swap_{way}_s_coef"]) > 0 for way in ("out", "in")), "FP8 swap times came out as zero")
    with (ctx.cost_model.parent / "kv_crossover.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        require(float(row["swap_s"]) > 0 and float(row["recompute_s"]) > 0, f"no real times at {row['tokens']} tokens: {row}")
    longest = rows[-1]
    return {
        "summary": "; ".join(f"{r['tokens']} tokens: swap {float(r['swap_s']) * 1e3:.1f} ms, recompute {float(r['recompute_s']) * 1e3:.1f} ms" for r in rows),
        "rows": rows,
        "measured": {"crossover_prefill_tokens_per_s": int(longest["tokens"]) / float(longest["recompute_s"])},
    }


def session_problems(after_keep: dict, after_swap: dict, after_resume: dict, record_bytes: int, restored_at_least: int) -> list[str]:
    """What /metrics says went wrong with the session, or nothing.

    FP8 shows in the bytes: a tier's copies count their records, so the bytes moved per block say
    which format crossed. The plain path reports no host pins, which only the native library has.
    """
    problems = []
    if after_keep["kept"] != 1 or after_keep["swapped"] != 0:
        problems.append(f"the pause was not kept: kept={after_keep['kept']}, swapped={after_keep['swapped']}")
    if after_swap["swapped"] != 1 or after_swap["ttl_expired"] != 1:
        problems.append(f"the time-to-live did not swap it: swapped={after_swap['swapped']}, ttl_expired={after_swap['ttl_expired']}")
    if after_resume["restored_tokens"] < restored_at_least:
        problems.append(
            f"restored {after_resume['restored_tokens']} tokens, expected at least {restored_at_least}"
            f" (restores_avoided={after_resume['restores_avoided']}, dropped={after_resume['dropped']})"
        )
    if after_resume["bytes_restored"] <= 0:
        problems.append("no bytes were copied back")
    if after_resume["host_pins"] <= 0:
        problems.append("no host pins, so this was not the native path")
    expected = after_resume["cpu_tier_peak_blocks"] * record_bytes
    if after_resume["bytes_offloaded"] != expected or expected == 0:
        problems.append(
            f"{after_resume['bytes_offloaded']} bytes offloaded for {after_resume['cpu_tier_peak_blocks']} blocks, "
            f"and FP8 records would make that {expected}"
        )
    return problems


def run_session(server, system, limits: dict, vocab_size: int, session_tokens: int, filler_tokens: int) -> dict:
    """One session through keep, swap and resume, against a running server.

    Under a small hint with nobody waiting the pause is kept, with the longest time-to-live. Waiting
    it out demotes the session whatever the cost model says of swap against recompute. The tree still
    holds those blocks, and a resume would rightly take them from it and copy nothing, so one-off
    prompts first push them out: eviction is least recently used, and these fill the whole pool.
    """
    rng = random.Random(0)

    def prompt(length: int) -> list[int]:
        return [rng.randrange(vocab_size) for _ in range(length)]

    def complete(tokens: list[int], max_tokens: int, **extra) -> dict:
        response = server.client.post(
            "/v1/completions", timeout=limits["request_timeout_s"],
            json={"model": system.settings.model, "prompt": tokens, "max_tokens": max_tokens, "ignore_eos": True, **extra},
        )
        require(response.status_code == 200, f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def metrics() -> dict:
        return system.server_metrics(server.client)

    first_prompt = prompt(session_tokens)
    hints = {"tool_name": "smoke_tool", "expected_tool_s": 1.0}
    first = complete(first_prompt, 4, session_id="smoke", layover=hints)
    after_keep = metrics()
    deadline = time.monotonic() + system.ttl_cap_s + 60
    after_swap = metrics()
    while after_swap["bytes_offloaded"] == 0 and time.monotonic() < deadline:
        time.sleep(0.25)
        after_swap = metrics()

    # Asking for a whole pool overflows it by the session's own blocks, which are the oldest.
    pool_tokens = after_swap["blocks_total"] * BLOCK_TOKENS
    sent, fill_started = 0, time.monotonic()
    while sent < pool_tokens:
        complete(prompt(filler_tokens), 1)
        sent += filler_tokens
    fill_s = time.monotonic() - fill_started

    resume_prompt = first_prompt + first["choices"][0]["token_ids"] + prompt(8)
    resumed = complete(resume_prompt, 4, session_id="smoke", layover={"end_session": True})
    require(len(resumed["choices"][0]["token_ids"]) == 4, f"the resumed turn gave {resumed['choices'][0]['token_ids']}")
    return {
        "after_keep": after_keep, "after_swap": after_swap, "after_resume": metrics(),
        "cached_tokens": resumed["usage"]["prompt_tokens_details"]["cached_tokens"],
        "filled_tokens": sent, "fill_s": fill_s,
    }


def serve_one_session(system, limits: dict, tolerances: dict, log_path: Path, vocab_size: int, session_tokens: int, filler_tokens: int, record_bytes: int, gpu=None, gpu_expected=None) -> dict:
    """Start a server, run the session, check what /metrics says, and always stop it."""
    from bench.run_matrix import free_port
    from bench.systems import ManagedServer, check_cpu_tier, check_gpu_memory, check_kv_capacity, wait_gpu_released

    idle = gpu.read().memory_used_bytes if gpu is not None else None
    server = ManagedServer(system, log_path, free_port(), stop_timeout_s=limits["stop_timeout_s"])
    started = time.monotonic()
    try:
        server.start()
        server.wait_ready(limits["startup_timeout_s"])
        startup_s = time.monotonic() - started
        translation = system.translation
        budgets = check_kv_capacity(system.name, system.reported_kv_tokens(server.client, ""), translation.kv_tokens_expected, tolerances["kv_capacity_frac"])
        budgets |= check_cpu_tier(system.name, system.reported_cpu_tier_gib(server.client, ""), translation.cpu_gib_effective, tolerances["kv_capacity_frac"])
        if gpu is not None:
            budgets |= check_gpu_memory(system.name, gpu.read().memory_used_bytes, idle, *gpu_expected, tolerances["gpu_memory_gib"])
        session = run_session(server, system, limits, vocab_size, session_tokens, filler_tokens)
    except Exception as exc:
        raise SmokeFailure(f"{exc}; log {log_path}") from exc
    finally:
        server.stop()
    if gpu is not None:
        wait_gpu_released(gpu, idle + tolerances["gpu_idle_gib"] * 2**30, limits["gpu_release_timeout_s"])
    # A block's worth of the end of the prompt is cheaper to prefill than to copy.
    problems = session_problems(session["after_keep"], session["after_swap"], session["after_resume"], record_bytes, session_tokens - BLOCK_TOKENS)
    if session["cached_tokens"] < session_tokens - BLOCK_TOKENS:
        problems.append(f"the resumed turn reused only {session['cached_tokens']} tokens")
    require(not problems, "; ".join(problems) + f"; log {log_path}")
    return {**session, "startup_s": startup_s, "budgets": budgets}


def matrix_system(ctx: Context):
    """The matrix's reference system, exactly as configured, on the smoke's cost model."""
    from bench.run_matrix import system_specs
    from bench.systems import MatchedSettings, build_system
    from layover.config import ModelConfig

    config = ctx.matrix()
    settings = MatchedSettings.from_config(config["matched"])
    cfg = ModelConfig.from_pretrained(settings.model)
    name = config["reference_system"]
    spec = {**next(s for s in system_specs(config) if s["name"] == name), "cost_model": str(ctx.cost_model)}
    require(spec.get("transfer") == "native" and spec.get("offload_dtype") == "fp8", f"{name} is not on the native path with FP8: {spec}")
    return build_system(spec, settings, cfg.kv_bytes_per_token(settings.dtype), cfg), cfg


def check_server_session(ctx: Context) -> dict:
    from bench.gpu_sampler import GIB, NvmlBackend
    from bench.run_matrix import checkpoint_weight_bytes
    from bench.systems import offload_record_bytes

    config = ctx.matrix()
    system, cfg = matrix_system(ctx)
    settings = system.settings
    gpu = NvmlBackend(config.get("gpu_index", 0))
    try:
        session = serve_one_session(
            system, config["limits"], config["tolerances"], Path(ctx.out) / "logs" / "server.log", cfg.vocab_size, 2048,
            settings.max_context_tokens - 8, offload_record_bytes(cfg, "fp8"), gpu,
            (checkpoint_weight_bytes(settings.model, settings.dtype), int(settings.kv_gib * GIB)),
        )
    finally:
        gpu.close()
    resume = session["after_resume"]
    return {
        "summary": f"kept, swapped {resume['bytes_offloaded'] / 2**20:.1f} MiB of FP8, restored {resume['restored_tokens']} tokens, "
        f"started in {session['startup_s']:.0f} s",
        "session": session,
        "measured": {"server_startup_s": session["startup_s"], "server_prefill_tokens_per_s": session["filled_tokens"] / session["fill_s"]},
    }


def check_gpu_tests(ctx: Context) -> dict:
    # In a process of its own, because conftest reads sys.argv to decide on Triton's interpreter.
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--gpu", "--checkpoint", ctx.model, *GPU_TESTS]
    try:
        result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=GPU_TESTS_LIMIT_S)
    except subprocess.TimeoutExpired as exc:
        raise SmokeFailure(f"the quick GPU tests passed their limit of {GPU_TESTS_LIMIT_S} s") from exc
    print(result.stdout + result.stderr, flush=True)
    last = (result.stdout.strip().splitlines() or [""])[-1]
    require(result.returncode == 0, f"pytest exited {result.returncode}: {last}")
    passed = re.search(r"(\d+) passed", last)
    # A skip passes too, and a test that skips here proved nothing about this machine.
    require(passed is not None and int(passed[1]) == len(GPU_TESTS), f"expected {len(GPU_TESTS)} to pass: {last}")
    return {"summary": f"{passed[1]} tests within {GPU_TESTS_LIMIT_S} s: {last}"}


def measured_rates(measured: dict, on_gpu: bool) -> dict:
    """The file bench/run_matrix.py --measured-rates reads, from what the checks measured."""
    steps = measured["decode_step_s"]
    largest = max(steps, key=int)
    prefill = measured.get("server_prefill_tokens_per_s") or measured["crossover_prefill_tokens_per_s"]
    rates = {
        "prefill_tokens_per_s": prefill,
        "decode_tokens_per_s_per_stream": 1 / steps["1"],
        "decode_tokens_per_s_total": int(largest) / steps[largest],
    }
    if "server_startup_s" in measured:
        rates["startup_s"] = measured["server_startup_s"]
    context = measured["decode_context_tokens"]
    source = (
        f"bench/gpu_smoke.py: decode steps at {context} tokens of context, batch 1 and {largest}, "
        + ("prefill through the server up to the matrix's longest context, startup of the matched server" if "server_prefill_tokens_per_s" in measured else "prefill from the crossover")
    )
    return {"source": source if on_gpu else f"{PIPELINE_LABEL} {source}", "estimate_assumptions": rates}


def check_measured_rates(ctx: Context) -> dict:
    from bench.run_matrix import apply_measured_rates

    path = Path(ctx.out) / "measured_rates.json"
    rates = measured_rates(ctx.measured, ctx.on_gpu)
    path.write_text(json.dumps(rates, indent=2) + "\n")
    # The reader's own validation, so that a file it would refuse fails here and not at planning.
    apply_measured_rates(ctx.matrix(), path)
    return {"summary": ", ".join(f"{key} {value:.4g}" for key, value in rates["estimate_assumptions"].items()), "rates": rates["estimate_assumptions"]}


# What of a dry run's output the plan's reader needs: the estimate, its limits, and the budget.
REFINED_LINES = ("token rates", "estimated total", "NOT IN THE ESTIMATE", "Not included", "Budget:", "OVER BUDGET")


def parse_refined_hours(text: str) -> tuple[float, float]:
    found = re.search(r"estimated total: ([\d.]+) hours, up to ([\d.]+) hours", text)
    require(found is not None, f"no estimate in bench/run_matrix.py's output: {text[-300:]}")
    return float(found[1]), float(found[2])


def check_refine_plan(ctx: Context) -> dict:
    out = Path(ctx.out)
    argv = [
        sys.executable, str(REPO / "bench" / "run_matrix.py"), "--config", ctx.config, "--dry-run", "--plan", ctx.plan,
        "--measured-rates", str(out / "measured_rates.json"), "--out", "results/matrix",
    ]
    result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=150)
    print(result.stdout + result.stderr, flush=True)
    require(result.returncode == 0, f"run_matrix --dry-run exited {result.returncode}: {result.stderr.strip()[-300:]}")
    low, high = parse_refined_hours(result.stdout)
    lines = [line for line in result.stdout.splitlines() if any(mark in line for mark in REFINED_LINES)]
    # The first line is for the runner's shell, which has no JSON parser it can count on.
    (out / "refined_hours.txt").write_text(f"{low} {high}\n" + "\n".join(lines) + "\n")
    return {"summary": f"plan {ctx.plan}: {low:.1f} to {high:.1f} hours", "plan": ctx.plan, "low_hours": low, "high_hours": high, "lines": lines}


# ---- running them -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[Context], dict]
    timeout_s: float
    needs: tuple[str, ...] = ()
    on_cpu: bool = False


CHECKS = (
    Check("environment", check_environment, 120),
    Check("reference_parity", check_reference_parity, 300, ("environment",)),
    Check("triton_decode", check_triton_decode, 240, ("environment",)),
    Check("decode_rates", check_decode_rates, 180, ("triton_decode",), on_cpu=True),
    Check("native_transfer", check_native_transfer, 240, ("environment",), on_cpu=True),
    Check("crossover", check_crossover, 240, ("native_transfer",), on_cpu=True),
    Check("server_session", check_server_session, 480, ("crossover",)),
    Check("gpu_tests", check_gpu_tests, GPU_TESTS_LIMIT_S + 30, ("environment",)),
    Check("measured_rates", check_measured_rates, 60, ("decode_rates", "crossover", "server_session"), on_cpu=True),
    Check("refine_plan", check_refine_plan, 180, ("measured_rates",)),
)
FAILING = ("failed", "blocked")


def write_json(path: Path, data: dict) -> None:
    """Atomic, so that a fetch at any moment reads a whole file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(temporary, path)


def worker_argv(check: Check, context_path: Path, result_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", check.name, "--context", str(context_path), "--result", str(result_path)]


def stop_group(process: subprocess.Popen, grace_s: float) -> None:
    """Ask, wait, then kill the group and anything under it that left it: a server has a session of its own."""
    strays = _descendants(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in strays:
        _kill_quietly(pid, signal.SIGKILL)
    process.wait(timeout=10)


def run_in_process(check: Check, ctx: Context, argv_for, timeout_s: float, grace_s: float) -> tuple[str, str, dict]:
    """(status, summary, details) of one check, run where a hang cannot take the runner with it."""
    tmp, logs = Path(ctx.tmp), Path(ctx.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    context_path, result_path = tmp / "context.json", tmp / f"{check.name}.result.json"
    context_path.write_text(json.dumps(asdict(ctx)))
    log_path = logs / f"{check.name}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "SMOKE_RESULT": str(result_path)}
    with log_path.open("w") as log:
        process = subprocess.Popen(argv_for(check, context_path, result_path), stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env, start_new_session=True)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_group(process, grace_s)
            return "failed", f"timed out after {timeout_s:.0f} s; log {log_path}", {}
        except BaseException:
            stop_group(process, grace_s)
            raise
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if process.returncode == 0 and result is not None and result.get("ok"):
        details = result.get("details", {})
        return "passed", details.pop("summary", ""), details
    if result is not None and not result.get("ok", True):
        return "failed", result.get("error", "failed"), {}
    tail = " | ".join(log_path.read_text(errors="replace").strip().splitlines()[-3:])
    return "failed", f"exited {process.returncode} with no result: {tail}; log {log_path}", {}


def blocker(check: Check, statuses: dict[str, str]) -> list[str]:
    return [name for name in check.needs if statuses.get(name) in FAILING]


def run_checks(checks, ctx: Context, report_path: Path, *, argv_for=worker_argv, total_limit_s: float = TOTAL_LIMIT_S, grace_s: float = TERM_GRACE_S, announce=print) -> dict:
    """Every check in order. The report on disk is whole after each one, so an interruption keeps it."""
    started = time.time()
    report = {
        "label": None if ctx.on_gpu else PIPELINE_LABEL,
        "device": ctx.device, "model": ctx.model, "complete": False, "passed": False,
        "started_unix": started, "total_limit_s": total_limit_s,
        "checks": [
            {"name": c.name, "status": "pending", "needs": list(c.needs), "timeout_s": c.timeout_s, "seconds": None, "summary": "", "details": {}}
            for c in checks
        ],
    }
    write_json(report_path, report)
    statuses: dict[str, str] = {}
    for check, row in zip(checks, report["checks"]):
        blocked_by = blocker(check, statuses)
        left = total_limit_s - (time.time() - started)
        if not ctx.on_gpu and not check.on_cpu:
            row["status"], row["summary"] = "not_applicable", "needs a GPU"
        elif blocked_by:
            row["status"], row["summary"], row["blocked_by"] = "blocked", f"blocked by {', '.join(blocked_by)}", blocked_by
        elif left <= 0:
            row["status"], row["summary"] = "blocked", f"the whole smoke passed its limit of {total_limit_s:.0f} s"
        else:
            announce(f"--- {check.name}, limit {min(check.timeout_s, left):.0f} s", flush=True)
            began = time.time()
            status, summary, details = run_in_process(check, ctx, argv_for, min(check.timeout_s, left), grace_s)
            row.update(status=status, summary=summary, details=details, seconds=round(time.time() - began, 2))
            ctx.measured.update(details.pop("measured", {}))
        statuses[check.name] = row["status"]
        write_json(report_path, report)
        announce(f"    {row['status']}: {row['summary']}", flush=True)
    report["complete"] = True
    report["passed"] = not any(row["status"] in FAILING for row in report["checks"])
    report["finished_unix"] = time.time()
    report["measured"] = ctx.measured
    write_json(report_path, report)
    return report


def table(report: dict) -> str:
    lines = []
    if report["label"]:
        lines.append(report["label"])
    lines.append(f"{'check':<18} {'status':<15} {'seconds':>8}  summary")
    for row in report["checks"]:
        seconds = "-" if row["seconds"] is None else f"{row['seconds']:.1f}"
        lines.append(f"{row['name']:<18} {row['status']:<15} {seconds:>8}  {row['summary']}")
    return "\n".join(lines)


def worker_main(name: str, context_path: str, result_path: str, checks=CHECKS) -> int:
    # A stop during a request reaches the finally that stops the server.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    check = next(c for c in checks if c.name == name)
    ctx = Context(**json.loads(Path(context_path).read_text()))
    try:
        details = check.run(ctx)
        Path(result_path).write_text(json.dumps({"ok": True, "details": details}, default=str))
        return 0
    except (Exception, SystemExit) as exc:
        traceback.print_exc()
        Path(result_path).write_text(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}" if not isinstance(exc, SmokeFailure) else str(exc)}))
        return 1


def torch_sees_a_gpu() -> bool:
    # Asked of another process, so that this one holds no GPU memory for the checks to fight over.
    probe = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=120).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model", help="default: the matrix config's model, or the micro model with --device cpu")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--plan", default=os.environ.get("SMOKE_PLAN") or DEFAULT_PLAN, help="the plan to refine the hours of; default SMOKE_PLAN, or " + DEFAULT_PLAN)
    p.add_argument("--out", type=Path, help="default: results/smoke, or results/smoke_pipeline_check with --device cpu")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--context", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None, checks=None, argv_for=worker_argv) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args.worker, args.context, args.result)
    checks = CHECKS if checks is None else checks
    on_gpu = args.device == "cuda"
    if on_gpu and argv_for is worker_argv and not torch_sees_a_gpu():
        print("Refused: torch cannot see a CUDA GPU. Run scripts/setup_gpu.sh, or pass --device cpu for a pipeline check.", file=sys.stderr)
        return 2
    out = args.out or Path("results/smoke" if on_gpu else "results/smoke_pipeline_check")
    out = out if out.is_absolute() else Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    # A failed smoke must not leave an earlier one's hours looking current.
    for stale in ("measured_rates.json", "refined_hours.txt"):
        (out / stale).unlink(missing_ok=True)
    model = args.model or (json.loads(args.config.read_text())["matched"]["model"] if on_gpu else "micro")
    tmp = tempfile.mkdtemp(prefix="layover-smoke-")

    def on_terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_terminate)
    try:
        ctx = Context(args.device, model, str(args.config.resolve()), str(out), tmp, args.plan)
        report = run_checks(checks, ctx, out / "smoke.json", argv_for=argv_for)
    except KeyboardInterrupt:
        print("Interrupted; the checks so far are in smoke.json.", file=sys.stderr)
        return 130
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + table(report))
    refined = next((row for row in report["checks"] if row["name"] == "refine_plan" and row["status"] == "passed"), None)
    if refined:
        print(f"\nRefined {refined['details']['plan']}: {refined['details']['low_hours']:.1f} to {refined['details']['high_hours']:.1f} hours (an estimate from measured rates).")
    print(f"\nReport: {out / 'smoke.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

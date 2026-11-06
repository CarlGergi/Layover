#!/usr/bin/env python3
"""Compare gather plus one bulk DMA against one copy per block, on the real transfer library.

Sweeps block counts, fragmentation, and BF16 versus FP8, and reports the median and spread
of each direction with warm-up repeats excluded. A separate phase runs a matmul loop on
torch's stream and reports how much it slows while each path is copying, which is what the
gather's dedicated streams exist to protect.

Only --device cuda describes the hardware. On CPU the library's host build runs the same
pack logic on threads, so the output says it is a pipeline check and nothing else: there is
no PCIe and no DMA, and the two paths' relative cost means nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layover.config import ModelConfig  # noqa: E402
from layover.kernels.fp8 import Fp8Layout, round_trip_within_bound  # noqa: E402
from layover.kv import PagedKVCache  # noqa: E402
from layover.kv.transfer import NativeBackend  # noqa: E402

GPU_COUNTS = [16, 64, 256, 1024]
CPU_COUNTS = [4, 16]
FRAGMENTATIONS = ["contiguous", "strided", "shuffled"]
PATHS = ["gather", "per_block"]
FORMATS = ["bf16", "fp8"]
PIPELINE_NOTE = "NOT a GPU measurement: pipeline check only"


@dataclass
class Row:
    path: str
    format: str
    fragmentation: str
    blocks: int
    bytes_crossing: int
    out_s: float
    out_min_s: float
    out_max_s: float
    in_s: float
    in_min_s: float
    in_max_s: float
    out_gib_per_s: float
    in_gib_per_s: float
    # The library's own launches. FP8's kernels are Triton's, so they count as none here.
    native_kernels_per_round_trip: int
    copies_per_round_trip: int


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "host threads"


def layout_of_pool(count: int, fragmentation: str, half: int, seed: int) -> tuple[list[int], list[int]]:
    """Source ids in the lower half of the pool, targets in the upper, so sources stay intact."""
    if count > half:
        raise ValueError(f"{count} blocks do not fit in half a pool, {half}")
    generator = torch.Generator().manual_seed(seed)
    if fragmentation == "contiguous":
        source = list(range(count))
    elif fragmentation == "strided":
        stride = half // count
        source = [i * stride for i in range(count)]
    elif fragmentation == "shuffled":
        source = torch.randperm(half, generator=generator)[:count].tolist()
    else:
        raise ValueError(fragmentation)
    return source, [half + block for block in source]


def fill(cache: PagedKVCache, seed: int) -> None:
    """Random KV in slices, because one whole-pool randn would need twice the pool's memory."""
    torch.manual_seed(seed)
    step = 64
    for start in range(0, cache.pool.shape[0], step):
        part = cache.pool[start : start + step]
        part.copy_(torch.randn(part.shape, device=cache.device, dtype=torch.float32).to(part.dtype))
    sync(cache.device)


def crc(cache: PagedKVCache, ids: list[int]) -> int:
    index = torch.tensor(ids, device=cache.device)
    return zlib.crc32(cache.pool[index].cpu().view(torch.uint16).numpy().tobytes())


def check_restored(cache: PagedKVCache, source: list[int], target: list[int], fp8: bool) -> None:
    if not fp8:
        if crc(cache, source) != crc(cache, target):
            raise RuntimeError("bytes changed across a round trip")
        return
    layout = Fp8Layout.from_cache(cache)
    before = cache.pool[torch.tensor(source, device=cache.device)].cpu()
    after = cache.pool[torch.tensor(target, device=cache.device)].cpu()
    if not round_trip_within_bound(before, after, layout.group_elems):
        raise RuntimeError("an FP8 round trip left its error bound")


def round_trip(backend: NativeBackend, cache: PagedKVCache, path: str, source: list[int], target: list[int], fp8: bool) -> tuple[float, float]:
    """One timed round trip. The clock stops when the ticket says the data has landed."""
    sync(cache.device)
    start = time.perf_counter()
    if path == "gather":
        out = backend.start_out(cache, source, fp8)
    else:
        out = backend.start_out_per_block(cache, source, fp8)
    out.wait()
    middle = time.perf_counter()
    if path == "gather":
        into = backend.start_in(cache, target, out.payload)
    else:
        into = backend.start_in_per_block(cache, target, out.payload)
    into.wait()
    end = time.perf_counter()
    backend.free(out.payload)
    return middle - start, end - middle


def measure(backend, cache, args, path, fmt, fragmentation, count) -> Row:
    fp8 = fmt == "fp8"
    half = cache.allocator.num_blocks // 2
    source, target = layout_of_pool(count, fragmentation, half, args.seed)
    outs, ins = [], []
    stats_before = None
    for repeat in range(args.warmup + args.repeats):
        if repeat == args.warmup:
            stats_before = backend.engine.stats()
        out_s, in_s = round_trip(backend, cache, path, source, target, fp8)
        if repeat >= args.warmup:
            outs.append(out_s)
            ins.append(in_s)
    stats_after = backend.engine.stats()
    # Every measured repeat did the same thing, so one check covers what they copied.
    check_restored(cache, source, target, fp8)
    record = Fp8Layout.from_cache(cache).record_bytes if fp8 else cache.block_bytes
    crossing = count * record
    out_s, in_s = statistics.median(outs), statistics.median(ins)
    return Row(
        path,
        fmt,
        fragmentation,
        count,
        crossing,
        out_s,
        min(outs),
        max(outs),
        in_s,
        min(ins),
        max(ins),
        crossing / out_s / 2**30,
        crossing / in_s / 2**30,
        (stats_after["kernels"] - stats_before["kernels"]) // args.repeats,
        (stats_after["memcpys"] - stats_before["memcpys"]) // args.repeats,
    )


def matmul_iterations(device: torch.device, size: int, stop: threading.Event, record: list[float]) -> None:
    """Time a matmul loop until told to stop.

    Reading a result synchronizes only this stream. torch.cuda.synchronize would wait for
    every stream on the device and so measure the transfers instead of their interference.
    """
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    while not stop.is_set():
        start = time.perf_counter()
        (a @ b)[0, 0].item()
        record.append(time.perf_counter() - start)


def run_matmul(device: torch.device, size: int, during) -> list[float]:
    stop = threading.Event()
    record: list[float] = []
    worker = threading.Thread(target=matmul_iterations, args=(device, size, stop, record), daemon=True)
    worker.start()
    try:
        during()
    finally:
        stop.set()
        worker.join()
    return record


def measure_interference(backend, cache, args) -> list[dict]:
    """How much a matmul loop slows during each path, against an idle baseline each time."""
    half = cache.allocator.num_blocks // 2
    count = min(args.interference_blocks, half)
    source, target = layout_of_pool(count, "shuffled", half, args.seed)
    results = []
    for path in PATHS:
        for fmt in FORMATS:
            fp8 = fmt == "fp8"
            # The first iterations include allocator and kernel start-up on either side.
            baseline = run_matmul(cache.device, args.matmul_size, lambda: time.sleep(args.interference_s))[args.warmup_iterations :]

            def transfer_until_elapsed():
                deadline = time.perf_counter() + args.interference_s
                rounds = 0
                while time.perf_counter() < deadline:
                    round_trip_without_device_sync(backend, cache, path, source, target, fp8)
                    rounds += 1
                loaded["rounds"] = rounds

            loaded: dict = {}
            during = run_matmul(cache.device, args.matmul_size, transfer_until_elapsed)[args.warmup_iterations :]
            if not baseline or not during:
                raise RuntimeError("the matmul loop made too few iterations: raise --interference-s")
            idle_s, busy_s = statistics.median(baseline), statistics.median(during)
            results.append(
                {
                    "path": path,
                    "format": fmt,
                    "blocks": count,
                    "rounds": loaded["rounds"],
                    "idle_iteration_s": idle_s,
                    "busy_iteration_s": busy_s,
                    "idle_iterations": len(baseline),
                    "busy_iterations": len(during),
                    "busy_iteration_min_s": min(during),
                    "busy_iteration_max_s": max(during),
                    "slowdown_percent": 100.0 * (busy_s / idle_s - 1.0),
                }
            )
    return results


def round_trip_without_device_sync(backend, cache, path, source, target, fp8) -> None:
    """A device-wide sync would stop the matmul loop this runs against."""
    out = backend.start_out(cache, source, fp8) if path == "gather" else backend.start_out_per_block(cache, source, fp8)
    out.wait()
    if path == "gather":
        into = backend.start_in(cache, target, out.payload)
    else:
        into = backend.start_in_per_block(cache, target, out.payload)
    into.wait()
    backend.free(out.payload)


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="micro", help="'micro', a hub id, or a local checkpoint; only its config is read")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--blocks", type=int, nargs="+", help="block counts per transfer")
    p.add_argument("--fragmentation", nargs="+", default=FRAGMENTATIONS, choices=FRAGMENTATIONS)
    p.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    p.add_argument("--staging-mib", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="leading repeats to discard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interference-blocks", type=int, default=256)
    p.add_argument("--interference-s", type=float, default=5.0, help="length of each phase of the matmul comparison")
    p.add_argument("--matmul-size", type=int, default=2048)
    p.add_argument("--warmup-iterations", type=int, default=3, help="leading matmul iterations to discard")
    p.add_argument("--out", type=Path, default=Path("results/transfer_gather"))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available")
    counts = args.blocks or (GPU_COUNTS if device.type == "cuda" else CPU_COUNTS)
    # Reads only config.json, and 'micro' reads nothing, so a sweep never needs the weights.
    cfg = ModelConfig.micro() if args.model == "micro" else ModelConfig.from_pretrained(args.model)

    # Sources and targets get a half each, and FP8 restores never touch the sources.
    cache = PagedKVCache(cfg, 2 * max(counts), device, torch.bfloat16)
    backend = NativeBackend(device, staging_bytes=args.staging_mib << 20)
    rows: list[Row] = []
    on_gpu = device.type == "cuda"

    def save(interference: list[dict] | None) -> dict:
        result = {
            "model": args.model,
            "device": args.device,
            "device_name": device_name(device),
            "measured_on_gpu": on_gpu,
            "note": None if on_gpu else PIPELINE_NOTE,
            "block_bytes": cache.block_bytes,
            "staging_mib": args.staging_mib,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "complete": interference is not None,
            "sweep": [asdict(r) for r in rows],
            "interference": interference or [],
        }
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.out / "transfer_gather.csv")
        (args.out / "transfer_gather.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    try:
        fill(cache, args.seed)
        if "fp8" in args.formats:
            backend.prepare(PagedKVCache(cfg, 1, device, torch.bfloat16), fp8=True)
        backend.prepare(cache)
        for count in counts:
            for fragmentation in args.fragmentation:
                for fmt in args.formats:
                    for path in PATHS:
                        row = measure(backend, cache, args, path, fmt, fragmentation, count)
                        rows.append(row)
                        # After every row, so a late crash keeps what was measured.
                        save(None)
                        print(
                            f"{path:>9} {fmt} {fragmentation:>10} {count:>5} blocks"
                            f"  out {row.out_s:9.5f}s ({row.out_min_s:.5f}-{row.out_max_s:.5f})"
                            f"  in {row.in_s:9.5f}s ({row.in_min_s:.5f}-{row.in_max_s:.5f})"
                            f"  {row.copies_per_round_trip} copies, {row.native_kernels_per_round_trip} native kernels",
                            flush=True,
                        )
        interference = measure_interference(backend, cache, args)
        for entry in interference:
            print(
                f"{entry['path']:>9} {entry['format']}: matmul iteration {entry['idle_iteration_s'] * 1e3:.2f} ms idle,"
                f" {entry['busy_iteration_s'] * 1e3:.2f} ms during transfers, {entry['slowdown_percent']:+.1f}%",
                flush=True,
            )
    finally:
        backend.close()

    result = save(interference)
    if not on_gpu:
        print(f"WARNING: {PIPELINE_NOTE}. Do not use these numbers as results.")
    return result


if __name__ == "__main__":
    main()

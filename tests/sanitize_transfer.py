#!/usr/bin/env python3
"""Exercise every custom kernel and stream path once, for compute-sanitizer.

    make -C native cuda-debug
    export LAYOVER_NATIVE_LIB=native/build/cuda-debug-$(uname -s)-$(uname -m)/liblayover_transfer.so
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    compute-sanitizer --tool memcheck --error-exitcode 1 python tests/sanitize_transfer.py
    compute-sanitizer --tool racecheck --error-exitcode 1 python tests/sanitize_transfer.py

The kernels: gather, scatter, FP8's quantize and dequantize, and the Triton decode kernel.
Racecheck reports shared-memory hazards inside a kernel, so it says nothing about the ordering
between streams; only the data checks here speak to that.

Torch's caching allocator hands out pieces of larger allocations, where an overrun past the
end of a tensor lands in memory the sanitizer sees as valid. So a CUDA run refuses to start
unless PYTORCH_NO_CUDA_MEMORY_CACHING=1 gives every tensor its own allocation.

Exits non-zero on any data mismatch, so a run passes only when the sanitizer stays quiet and
the data is right. --device cpu runs the same sequence on the host build, as a check of this
script, with decode only where Triton's interpreter is available. --corrupt flips one byte
first, to show that a mismatch does fail.
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layover.config import ModelConfig  # noqa: E402
from layover.kernels.decode_attention import paged_decode_attention_torch  # noqa: E402
from layover.kernels.fp8 import Fp8Layout, round_trip_bound  # noqa: E402
from layover.kernels.triton_decode import paged_decode_attention_triton, triton  # noqa: E402
from layover.kv import PagedKVCache  # noqa: E402
from layover.kv.transfer import NativeBackend  # noqa: E402

CFG = ModelConfig.micro()
POOL_BLOCKS = 48
# Decode shapes as [layers, kv_heads, kv_groups, head_dim, slots]. The first is Qwen3-4B's, whose
# powers of two use the kernel's unmasked tile loads. The second needs its masked ones.
DECODE_SHAPES = {"qwen3_4b": (36, 8, 4, 128, 16), "padded": (2, 2, 2, 24, 14)}
DECODE_POOL_BLOCKS = 8
BF16_ROUNDOFF = 2.0**-8


def fail(message: str) -> None:
    print(f"mismatch: {message}", file=sys.stderr)
    raise SystemExit(1)


def crc(cache: PagedKVCache, ids: list[int]) -> int:
    index = torch.tensor(ids, device=cache.device)
    return zlib.crc32(cache.pool[index].cpu().view(torch.uint16).numpy().tobytes())


def shuffled(count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(POOL_BLOCKS, generator=generator)[:count].tolist()


def bf16_round_trip(backend, cache, ids, target, label, corrupt=False):
    before = crc(cache, ids)
    out = backend.start_out(cache, ids, False)
    out.wait()
    if zlib.crc32(out.payload.memory[: out.payload.nbytes].numpy().tobytes()) != before:
        fail(f"{label}: host bytes after gather")
    if corrupt:
        out.payload.memory[0] ^= 1
    into = backend.start_in(cache, target, out.payload)
    into.wait()
    if crc(cache, target) != before:
        fail(f"{label}: pool bytes after scatter")
    backend.free(out.payload)


def fp8_round_trip(backend, cache, ids, target, label):
    layout = Fp8Layout.from_cache(cache)
    before = cache.pool[torch.tensor(ids, device=cache.device)].cpu().clone()
    out = backend.start_out(cache, ids, True)
    out.wait()
    into = backend.start_in(cache, target, out.payload)
    into.wait()
    after = cache.pool[torch.tensor(target, device=cache.device)].cpu()
    bits = before.view(torch.uint16).reshape(-1, layout.group_elems)
    gap = (after.to(torch.float64) - before.to(torch.float64)).abs().reshape(-1, layout.group_elems)
    if not (gap <= round_trip_bound(bits)).all():
        fail(f"{label}: FP8 round trip outside its error bound")
    backend.free(out.payload)


def require_uncached_allocator(device: str, environ=os.environ) -> None:
    if device == "cuda" and environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        print(
            "PYTORCH_NO_CUDA_MEMORY_CACHING=1 is required: under torch's caching allocator memcheck cannot see an overrun",
            file=sys.stderr,
        )
        raise SystemExit(2)


def decode_case(shape: tuple[int, int, int, int, int], device: str):
    """A batch whose every row is awkward, in a pool whose last byte gets read.

    Row 0 ends at the last slot of the pool's last block, through a scattered table. Row 1 is
    shorter: its table row is padded with entries past its length, and it ends partway through a
    block. Reading the last layer makes the highest address the kernel can compute a real one.
    """
    layers, kv_heads, kv_groups, head_dim, slots = shape
    last = DECODE_POOL_BLOCKS - 1
    table = torch.tensor([[2, 5, 0, last], [6, 1, 0, 0]], dtype=torch.int32, device=device)
    lengths = torch.tensor([4 * slots, slots + 3], dtype=torch.int32, device=device)
    pool = torch.randn(DECODE_POOL_BLOCKS, 2, layers, kv_heads, slots, head_dim).to(torch.bfloat16).to(device)
    q = torch.randn(2, kv_heads * kv_groups, head_dim).to(torch.bfloat16).to(device)
    return q, pool, layers - 1, table, lengths, kv_groups


def decode_launches(device: str, shapes=DECODE_SHAPES) -> int:
    """Launch the decode kernel at each shape and compare with the torch reference."""
    for name, shape in shapes.items():
        q, pool, layer, table, lengths, groups = decode_case(shape, device)
        # .double() reads the result, which waits for the launch: the kernel returns before it
        # finishes on CUDA, and has to finish under the sanitizer.
        got = paged_decode_attention_triton(q, pool, layer, table, lengths, groups).double()
        # On CPU, so that a finding from torch's own attention kernels cannot fail this run.
        expected = paged_decode_attention_torch(q.cpu(), pool.cpu(), layer, table.cpu(), lengths.cpu(), groups).double()
        # The reference accumulates in BF16 when lengths differ, so both are good to a few
        # roundings of max|V|. This is a check that the launch read the right memory.
        allowed = 8 * BF16_ROUNDOFF * pool[:, 1, layer].double().abs().max().cpu()
        if not (got.cpu() - expected).abs().max() <= allowed:
            fail(f"decode at the {name} shape disagrees with the reference")
    return len(shapes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--corrupt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require_uncached_allocator(args.device)

    torch.manual_seed(0)
    cache = PagedKVCache(CFG, POOL_BLOCKS, args.device, torch.bfloat16)
    cache.pool.copy_(torch.randn(cache.pool.shape).to(torch.bfloat16).to(args.device))
    # Three blocks per staging buffer and a small id ring, so long transfers go through every
    # segment and staging slot many times and the ring wraps.
    backend = NativeBackend(args.device, staging_bytes=3 * cache.block_bytes, ids_capacity=64)
    try:
        backend.prepare(cache)
        bf16_round_trip(backend, cache, shuffled(2, 1), shuffled(2, 2), "within one segment", args.corrupt)
        bf16_round_trip(backend, cache, shuffled(40, 3), shuffled(40, 4), "many segments")
        for seed in range(8):
            bf16_round_trip(backend, cache, shuffled(30, 10 + seed), shuffled(30, 20 + seed), f"staging wraparound {seed}")
        # Overlap: a second transfer prepared while the first is in flight.
        first = backend.start_out(cache, shuffled(20, 5), False)
        second = backend.start_out(cache, shuffled(20, 6), False)
        second.wait()
        first.wait()
        backend.free(first.payload)
        backend.free(second.payload)
        # The rejected path has no kernels, but its copies go through the same pool.
        out = backend.start_out_per_block(cache, shuffled(10, 7))
        out.wait()
        into = backend.start_in_per_block(cache, shuffled(10, 8), out.payload)
        into.wait()
        backend.free(out.payload)
        # Lower magnitudes first so FP8's scale is exercised over a spread of groups.
        cache.pool[:24] = (cache.pool[:24].float() * 1e-4).to(torch.bfloat16)
        fp8_round_trip(backend, cache, shuffled(2, 9), shuffled(2, 30), "FP8 within one segment")
        fp8_round_trip(backend, cache, shuffled(30, 31), shuffled(30, 32), "FP8 many segments")
        # Destroying with work in flight must wait for it and leave nothing behind.
        backend.start_out(cache, shuffled(30, 33), False)
        backend.start_out(cache, shuffled(30, 34), True)
    finally:
        backend.close()
    # A CPU check of this script may have no Triton to run. A CUDA run never skips this.
    if args.device == "cuda" or (triton is not None and triton.knobs.runtime.interpret):
        print(f"decode launches: {decode_launches(args.device)}")
    else:
        print("decode launches: skipped, no Triton interpreter")
    print("sanitize_transfer: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

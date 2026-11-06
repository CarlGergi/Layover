"""FP8 E4M3 quantize and dequantize of KV blocks, as Triton kernels and in torch.

Everything works on the uint16 bit view of BF16 data with integer operations, so it has
no dependency on hardware float8 and runs under the Triton interpreter. The torch
functions are the implementation on CPU and on machines without Triton, and the
reference the kernels are tested against.

One group is K or V of one block, layer, and head. A group gets a power-of-two scale
stored as an int16 shift: every value is divided by 2**shift. The largest finite nonzero
exponent in the group maps to 7, one below FP8's maximum of 8, so rounding that carries
into the next binade cannot overflow.

Values that have no FP8 representation:
- BF16 subnormals become zero, keeping their sign. They are below 2**-126, smaller than
  anything KV holds in practice, and excluding them keeps the scale meaningful.
- NaN and infinity become FP8 NaN, keeping the sign, and come back as NaN. FP8 E4M3 has
  no infinity. A non-finite value must stay visibly non-finite: torch's own conversion
  saturates infinity to 448, which would turn a numerical failure into a plausible value.
  They do not enter the group's scale, which would push every other value to zero.
- A magnitude out of range for an explicit shift also becomes NaN. The computed shift
  never causes this; only callers passing their own shift can.
- On the way back, a result below BF16's smallest normal becomes zero, and a rounding carry
  past BF16's largest finite value saturates to it.

Error after a round trip, with shift s: at most 2**-4 relative when the scaled value is in
FP8's normal range, and at most 2**(s-10) absolute below it (half the unit of FP8's
subnormals, scaled back), or 2**-126 where BF16 itself flushes to zero.

A stored block is one record: its FP8 bytes, then one int16 shift per group. Records are
self-contained, so any prefix of a payload is a smaller valid payload.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton has no macOS wheels; CPU paths use the torch functions.
    triton = None

BF16_EXP_MASK = 0xFF
BF16_MAX_FINITE = 0x7F7F
BF16_NAN = 0x7FC0
FP8_MAX_FINITE = 0x7E
FP8_NAN = 0x7F
BF16_BIAS = 127
FP8_BIAS = 7
# The scale maps a group's largest exponent here; FP8's own maximum is one higher.
FP8_TARGET_EXP = 7
# Bits lost from BF16's 8-bit significand when it becomes FP8's 4-bit one.
SIGNIFICAND_DROP = 4


@dataclass(frozen=True)
class Fp8Layout:
    """Sizes of one stored block, as data then shifts."""

    group_elems: int
    groups_per_block: int

    @classmethod
    def from_cache(cls, cache) -> "Fp8Layout":
        cfg = cache.cfg
        return cls(cache.block_tokens * cfg.head_dim, 2 * cfg.num_hidden_layers * cfg.num_key_value_heads)

    @property
    def block_elems(self) -> int:
        return self.group_elems * self.groups_per_block

    @property
    def header_bytes(self) -> int:
        return 2 * self.groups_per_block

    @property
    def record_bytes(self) -> int:
        return self.block_elems + self.header_bytes


# Torch implementation: the reference, and what runs without a GPU.


def group_shifts(bits: torch.Tensor) -> torch.Tensor:
    """Per-group int32 shifts for uint16 bits shaped [..., group_elems]."""
    raw_exp = (bits.to(torch.int32) >> 7) & BF16_EXP_MASK
    finite_nonzero = (raw_exp > 0) & (raw_exp < BF16_EXP_MASK)
    largest = torch.where(finite_nonzero, raw_exp, torch.zeros_like(raw_exp)).amax(dim=-1)
    # A group with nothing to scale keeps shift 0, so all-zero groups stay zero.
    return torch.where(largest > 0, largest - BF16_BIAS - FP8_TARGET_EXP, torch.zeros_like(largest))


def encode_bits(bits: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    """BF16 bits to FP8 bytes after dividing by 2**shift, rounding to nearest, ties to even."""
    wide = bits.to(torch.int32)
    shift = torch.as_tensor(shift, dtype=torch.int32, device=wide.device)
    sign = (wide >> 15) & 1
    raw_exp = (wide >> 7) & BF16_EXP_MASK
    biased = raw_exp - BF16_BIAS - shift + FP8_BIAS
    significand = (wide & 0x7F) | 0x80
    # Subnormals shift further right, one bit per binade below FP8's smallest normal. The
    # clamp keeps the shift legal; past 9 bits of shift the result is zero anyway.
    right = (SIGNIFICAND_DROP + (1 - biased).clamp(min=0)).clamp(max=15)
    kept = significand >> right
    remainder = significand & ((1 << right) - 1)
    half = 1 << (right - 1)
    kept = kept + ((remainder > half) | ((remainder == half) & ((kept & 1) == 1))).to(torch.int32)
    # Adding the significand to the exponent field lets a rounding carry ripple into it, and
    # a subnormal that rounds up to 8 lands on the smallest normal the same way.
    magnitude = torch.where(biased > 0, ((biased - 1) << 3) + kept, kept)
    magnitude = torch.where(magnitude > FP8_MAX_FINITE, FP8_NAN, magnitude)
    magnitude = torch.where(raw_exp == 0, 0, magnitude)
    magnitude = torch.where(raw_exp == BF16_EXP_MASK, FP8_NAN, magnitude)
    return ((sign << 7) | magnitude).to(torch.uint8)


def decode_bits(fp8: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    """FP8 bytes back to BF16 bits, multiplying by 2**shift."""
    wide = fp8.to(torch.int32)
    shift = torch.as_tensor(shift, dtype=torch.int32, device=wide.device)
    sign = (wide >> 7) & 1
    magnitude = wide & 0x7F
    raw_exp = magnitude >> 3
    fraction = magnitude & 7
    # A subnormal is fraction * 2**-9: normalize it around its top bit.
    top = torch.where(fraction >= 4, 2, torch.where(fraction >= 2, 1, 0))
    exponent = torch.where(raw_exp > 0, raw_exp - FP8_BIAS, top - 9)
    fraction = torch.where(raw_exp > 0, fraction, (fraction << (3 - top)) & 7)
    biased = exponent + shift + BF16_BIAS
    out = (biased << 7) | (fraction << SIGNIFICAND_DROP)
    out = torch.where(biased >= BF16_EXP_MASK, BF16_MAX_FINITE, out)
    out = torch.where(biased <= 0, 0, out)
    out = torch.where(magnitude == 0, 0, out)
    out = (sign << 15) | out
    return torch.where(magnitude == FP8_NAN, BF16_NAN, out).to(torch.uint16)


def round_trip_bound(bits: torch.Tensor, shifts: torch.Tensor | None = None) -> torch.Tensor:
    """Largest allowed absolute error per value, as float64, for bits shaped [..., group_elems]."""
    wide = bits.to(torch.int32)
    shifts = group_shifts(bits) if shifts is None else shifts
    raw_exp = (wide >> 7) & BF16_EXP_MASK
    significand = (wide & 0x7F) | 0x80
    exponent = raw_exp - BF16_BIAS
    value = significand.to(torch.float64) * torch.pow(2.0, (exponent - 7).to(torch.float64))
    scaled_exp = exponent - shifts[..., None]
    relative = value / 16
    absolute = torch.pow(2.0, (shifts[..., None] - 10).to(torch.float64))
    bound = torch.where(scaled_exp >= -6, relative, absolute)
    return torch.maximum(bound, torch.full_like(bound, 2.0**-126))


def round_trip_within_bound(
    before: torch.Tensor,
    after: torch.Tensor,
    group_elems: int,
    ignore_nonfinite: bool = False,
    slice_blocks: int = 16,
) -> bool:
    """Whether BF16 blocks [block, ...] came back from FP8 within round_trip_bound.

    Widening takes about 36 times the memory of what it checks. A pool of 32,768 tokens at
    Qwen3-4B's shape would need about 160 GiB of host RAM at once, so this goes through a few
    blocks at a time. A group never spans blocks, so slicing leaves its shift unchanged.
    ignore_nonfinite skips values that were never finite to begin with.
    """
    for start in range(0, before.shape[0], slice_blocks):
        bits = before[start : start + slice_blocks].view(torch.uint16).reshape(-1, group_elems)
        values = bits.view(torch.bfloat16).to(torch.float64)
        gap = (after[start : start + slice_blocks].to(torch.float64).reshape(-1, group_elems) - values).abs()
        within = gap <= round_trip_bound(bits)
        if not (within[values.isfinite()] if ignore_nonfinite else within).all():
            return False
    return True


def _quantize_torch(pool_bits: torch.Tensor, ids: torch.Tensor, stage: torch.Tensor, layout: Fp8Layout) -> None:
    count = ids.shape[0]
    bits = pool_bits[ids.long()].reshape(count * layout.groups_per_block, layout.group_elems)
    shifts = group_shifts(bits)
    encoded = encode_bits(bits, shifts[:, None]).reshape(count, layout.block_elems)
    shifted = shifts.to(torch.int16).reshape(count, layout.groups_per_block)
    records = stage[: count * layout.record_bytes].reshape(count, layout.record_bytes)
    records[:, : layout.block_elems] = encoded
    records[:, layout.block_elems :] = shifted.view(torch.uint8)


def _dequantize_torch(stage: torch.Tensor, ids: torch.Tensor, pool_bits: torch.Tensor, layout: Fp8Layout) -> None:
    count = ids.shape[0]
    records = stage[: count * layout.record_bytes].reshape(count, layout.record_bytes)
    encoded = records[:, : layout.block_elems].reshape(count * layout.groups_per_block, layout.group_elems)
    # A copy, because a slice of bytes need not be aligned for int16.
    shifts = records[:, layout.block_elems :].clone().view(torch.int16).reshape(-1, 1).to(torch.int32)
    decoded = decode_bits(encoded, shifts).reshape(count, *pool_bits.shape[1:])
    # Indexed assignment does not exist for uint16.
    pool_bits.view(torch.int16)[ids.long()] = decoded.view(torch.int16)


# Triton kernels. These repeat the arithmetic above in Triton's language, and test_fp8.py
# proves the two agree on every bit pattern.

if triton is not None:

    @triton.jit
    def _encode(wide, shift):
        sign = (wide >> 15) & 1
        raw_exp = (wide >> 7) & 0xFF
        biased = raw_exp - 127 - shift + 7
        significand = (wide & 0x7F) | 0x80
        right = tl.minimum(4 + tl.maximum(1 - biased, 0), 15)
        kept = significand >> right
        remainder = significand & ((1 << right) - 1)
        half = 1 << (right - 1)
        round_up = (remainder > half) | ((remainder == half) & ((kept & 1) == 1))
        kept = kept + tl.where(round_up, 1, 0)
        magnitude = tl.where(biased > 0, ((biased - 1) << 3) + kept, kept)
        magnitude = tl.where(magnitude > 0x7E, 0x7F, magnitude)
        magnitude = tl.where(raw_exp == 0, 0, magnitude)
        magnitude = tl.where(raw_exp == 0xFF, 0x7F, magnitude)
        return (sign << 7) | magnitude

    @triton.jit
    def _decode(wide, shift):
        sign = (wide >> 7) & 1
        magnitude = wide & 0x7F
        raw_exp = magnitude >> 3
        fraction = magnitude & 7
        top = tl.where(fraction >= 4, 2, tl.where(fraction >= 2, 1, 0))
        exponent = tl.where(raw_exp > 0, raw_exp - 7, top - 9)
        fraction = tl.where(raw_exp > 0, fraction, (fraction << (3 - top)) & 7)
        biased = exponent + shift + 127
        out = (biased << 7) | (fraction << 4)
        out = tl.where(biased >= 0xFF, 0x7F7F, out)
        out = tl.where(biased <= 0, 0, out)
        out = tl.where(magnitude == 0, 0, out)
        out = (sign << 15) | out
        return tl.where(magnitude == 0x7F, 0x7FC0, out)

    @triton.jit
    def _quantize_kernel(
        pool_ptr,
        ids_ptr,
        data_ptr,
        shift_ptr,
        block_elems,
        groups_per_block,
        record_bytes,
        GROUP_ELEMS: tl.constexpr,
        TILE: tl.constexpr,
    ):
        program = tl.program_id(0)
        index = (program // groups_per_block).to(tl.int64)
        group = (program % groups_per_block).to(tl.int64)
        block = tl.load(ids_ptr + index).to(tl.int64)
        source = pool_ptr + block * block_elems + group * GROUP_ELEMS
        target = data_ptr + index * record_bytes + group * GROUP_ELEMS
        # Tile sizes must be powers of two. Qwen3-4B's 2048-element groups need no mask, but
        # masked loads with a constant true mask compile away, so one path serves every model.
        offsets = tl.arange(0, TILE)
        in_range = offsets < GROUP_ELEMS
        wide = tl.load(source + offsets, mask=in_range, other=0).to(tl.int32)
        raw_exp = (wide >> 7) & 0xFF
        usable = tl.where((raw_exp > 0) & (raw_exp < 0xFF), raw_exp, 0)
        largest = tl.max(usable, axis=0)
        shift = tl.where(largest > 0, largest - 127 - 7, 0)
        tl.store(target + offsets, _encode(wide, shift).to(tl.uint8), mask=in_range)
        # The shifts follow the block's data; their pointer counts int16s.
        tl.store(shift_ptr + (index * record_bytes + block_elems) // 2 + group, shift.to(tl.int16))

    @triton.jit
    def _dequantize_kernel(
        data_ptr,
        shift_ptr,
        ids_ptr,
        pool_ptr,
        block_elems,
        groups_per_block,
        record_bytes,
        GROUP_ELEMS: tl.constexpr,
        TILE: tl.constexpr,
    ):
        program = tl.program_id(0)
        index = (program // groups_per_block).to(tl.int64)
        group = (program % groups_per_block).to(tl.int64)
        block = tl.load(ids_ptr + index).to(tl.int64)
        source = data_ptr + index * record_bytes + group * GROUP_ELEMS
        target = pool_ptr + block * block_elems + group * GROUP_ELEMS
        offsets = tl.arange(0, TILE)
        in_range = offsets < GROUP_ELEMS
        shift = tl.load(shift_ptr + (index * record_bytes + block_elems) // 2 + group).to(tl.int32)
        wide = tl.load(source + offsets, mask=in_range, other=0).to(tl.int32)
        tl.store(target + offsets, _decode(wide, shift).to(tl.uint16), mask=in_range)

    @triton.jit
    def _encode_kernel(in_ptr, out_ptr, shift, TILE: tl.constexpr):
        offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
        tl.store(out_ptr + offsets, _encode(tl.load(in_ptr + offsets).to(tl.int32), shift).to(tl.uint8))

    @triton.jit
    def _decode_kernel(in_ptr, out_ptr, shift, TILE: tl.constexpr):
        offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
        tl.store(out_ptr + offsets, _decode(tl.load(in_ptr + offsets).to(tl.int32), shift).to(tl.uint16))


def _tile(group_elems: int) -> int:
    return 1 << (group_elems - 1).bit_length()


def _use_triton(device: torch.device, impl: str | None) -> bool:
    if impl is None:
        impl = "triton" if device.type == "cuda" else "torch"
    if impl == "triton" and triton is None:
        raise RuntimeError("FP8 kernels need Triton, which is not installed")
    return impl == "triton"


def encode_bits_triton(bits: torch.Tensor, shift: int = 0) -> torch.Tensor:
    """Run the kernel arithmetic over a flat tensor whose length is a multiple of 256."""
    out = torch.empty(bits.shape[0], dtype=torch.uint8, device=bits.device)
    _encode_kernel[(bits.shape[0] // 256,)](bits, out, shift, TILE=256)
    return out


def decode_bits_triton(fp8: torch.Tensor, shift: int = 0) -> torch.Tensor:
    out = torch.empty(fp8.shape[0], dtype=torch.uint16, device=fp8.device)
    _decode_kernel[(fp8.shape[0] // 256,)](fp8, out, shift, TILE=256)
    return out


def quantize_launch(pool_bits: torch.Tensor, ids: torch.Tensor, stage: torch.Tensor, layout: Fp8Layout):
    """Grid, arguments, and constexprs of the quantize kernel, one program per group."""
    used = stage[: ids.shape[0] * layout.record_bytes]
    args = (pool_bits, ids, used, used.view(torch.int16), layout.block_elems, layout.groups_per_block, layout.record_bytes)
    return (ids.shape[0] * layout.groups_per_block,), args, _kernel_options(layout)


def dequantize_launch(stage: torch.Tensor, ids: torch.Tensor, pool_bits: torch.Tensor, layout: Fp8Layout):
    used = stage[: ids.shape[0] * layout.record_bytes]
    args = (used, used.view(torch.int16), ids, pool_bits, layout.block_elems, layout.groups_per_block, layout.record_bytes)
    return (ids.shape[0] * layout.groups_per_block,), args, _kernel_options(layout)


def _kernel_options(layout: Fp8Layout) -> dict:
    return {"GROUP_ELEMS": layout.group_elems, "TILE": _tile(layout.group_elems)}


def quantize_blocks(
    pool: torch.Tensor, ids: torch.Tensor, stage: torch.Tensor, layout: Fp8Layout, impl: str | None = None
) -> None:
    """Write one record per block of a BF16 pool to the front of `stage`.

    ids is int32 on the pool's device; stage is uint8 on the pool's device. The kernel is
    only enqueued: on CUDA it runs on the current stream.
    """
    if pool.dtype != torch.bfloat16:
        raise ValueError(f"FP8 offload needs a BF16 pool, not {pool.dtype}")
    pool_bits = pool.view(torch.uint16)
    count = ids.shape[0]
    if count == 0:
        return
    if not _use_triton(pool.device, impl):
        _quantize_torch(pool_bits, ids, stage, layout)
        return
    grid, args, options = quantize_launch(pool_bits, ids, stage, layout)
    _quantize_kernel[grid](*args, **options)


def dequantize_blocks(
    stage: torch.Tensor, ids: torch.Tensor, pool: torch.Tensor, layout: Fp8Layout, impl: str | None = None
) -> None:
    """Restore the first len(ids) records of `stage` into the pool's blocks `ids`."""
    if pool.dtype != torch.bfloat16:
        raise ValueError(f"FP8 offload needs a BF16 pool, not {pool.dtype}")
    pool_bits = pool.view(torch.uint16)
    count = ids.shape[0]
    if count == 0:
        return
    if not _use_triton(pool.device, impl):
        _dequantize_torch(stage, ids, pool_bits, layout)
        return
    grid, args, options = dequantize_launch(stage, ids, pool_bits, layout)
    _dequantize_kernel[grid](*args, **options)

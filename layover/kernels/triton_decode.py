"""Paged decode attention as a Triton kernel.

The kernel reads K and V straight out of the pool through the block table, without
gathering them first, and keeps a running softmax in the manner of FlashAttention's
decoding (Dao, "FlashAttention-2", 2023): it holds the running maximum m, the running
weight sum l, and the running weighted sum of V, rescaling by exp(m_old - m_new) when
a new block raises the maximum. Nothing the size of the context is ever materialized.

Triton has no macOS wheels, so like fp8 this module still imports without it and only
refuses when the kernel is called.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

KV_KINDS = 2
KV_DTYPES = (torch.bfloat16, torch.float32)

if triton is not None:

    @triton.jit
    def _load_tile(pointer, in_range, PADDED: tl.constexpr):
        if PADDED:
            return tl.load(pointer, mask=in_range, other=0.0)
        return tl.load(pointer)


    # The table width changes as sequences grow. Specializing on it would recompile the kernel
    # in the middle of serving, the first time the width happens to be a multiple of 16.
    @triton.jit(do_not_specialize=["table_width"])
    def _paged_decode_kernel(
        q_ptr,
        pool_ptr,
        table_ptr,
        lengths_ptr,
        out_ptr,
        table_width,
        query_heads,
        block_stride,
        kind_stride,
        layer_offset,
        head_stride,
        HEAD_DIM: tl.constexpr,
        SLOTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        KV_GROUPS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        sequence = tl.program_id(0)
        query_head = tl.program_id(1)

        # Tile shapes must be powers of two. Qwen3-4B needs no padding, so its kernel has no
        # masked loads, which cost a branch on every tile.
        PADDED: tl.constexpr = HEAD_DIM != BLOCK_D or SLOTS != BLOCK_S
        offs_s = tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        dim_in_range = offs_d < HEAD_DIM
        tile_in_range = (offs_s[:, None] < SLOTS) & dim_in_range[None, :]
        # A block's K for one layer and head is one contiguous run, slot after slot.
        tile_offsets = offs_s[:, None] * HEAD_DIM + offs_d[None, :]

        out_row = (sequence * query_heads + query_head) * HEAD_DIM
        q = _load_tile(q_ptr + out_row + offs_d, dim_in_range, PADDED).to(tl.float32)

        length = tl.load(lengths_ptr + sequence)
        num_blocks = (length + SLOTS - 1) // SLOTS
        head_offset = layer_offset + (query_head // KV_GROUPS) * head_stride

        m = tl.full([], -float("inf"), tl.float32)
        l = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([BLOCK_D], tl.float32)

        # One block at a time, K then V: each load is one contiguous run, which memory serves
        # better than scattered accesses, and each is read once per head because the softmax
        # state replaces any need to come back to it. Query heads sharing a KV head are
        # adjacent in the grid, so they repeat these addresses close together in time, which
        # gives the caches their best chance. Gathering first, as the reference does, costs a
        # read and a write per byte before any attention happens.
        for index in range(num_blocks):
            block = tl.load(table_ptr + sequence * table_width + index).to(tl.int64)
            # block * block_stride exceeds 32 bits once the pool passes 4 GiB of BF16, which a
            # 24 GB card reaches, so the block id is widened before the multiply.
            k_base = pool_ptr + block * block_stride + head_offset

            k = _load_tile(k_base + tile_offsets, tile_in_range, PADDED).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * SCALE
            # Whole blocks are loaded even when the last is partly empty: those slots are
            # allocated, and a masked load would add a branch to every block for the few
            # bytes saved. Their scores are excluded here instead.
            visible = offs_s < tl.minimum(length - index * SLOTS, SLOTS)
            scores = tl.where(visible, scores, -float("inf"))

            m_new = tl.maximum(m, tl.max(scores, axis=0))
            rescale = tl.exp(m - m_new)
            weights = tl.exp(scores - m_new)
            v = _load_tile(k_base + kind_stride + tile_offsets, tile_in_range, PADDED).to(tl.float32)
            # A zero weight is not enough: slots past the end hold whatever was there before,
            # stale or uninitialized, and 0 * NaN or 0 * inf would poison the sum.
            v = tl.where(visible[:, None], v, 0.0)

            acc = acc * rescale + tl.sum(weights[:, None] * v, axis=0)
            l = l * rescale + tl.sum(weights, axis=0)
            m = m_new

        result = (acc / l).to(out_ptr.dtype.element_ty)
        if PADDED:
            tl.store(out_ptr + out_row + offs_d, result, mask=dim_in_range)
        else:
            tl.store(out_ptr + out_row + offs_d, result)


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def default_num_warps(block_s: int, block_d: int) -> int:
    """One warp per 512 tile elements. Qwen3-4B's 16x128 tile gets 4, or 16 elements per thread."""
    return min(8, next_power_of_two(max(1, block_s * block_d // 512)))


def kernel_launch(
    q: torch.Tensor,
    pool: torch.Tensor,
    layer: int,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    kv_groups: int,
    out: torch.Tensor,
) -> tuple[tuple[int, int], tuple, dict]:
    """Grid, positional arguments, and keyword arguments for one launch."""
    batch, query_heads, head_dim = q.shape
    _, _, layers, kv_heads, slots, _ = pool.shape
    # Strides in elements for [block, kind, layer, kv_head, slot, dim], from the shape.
    head_stride = slots * head_dim
    layer_stride = kv_heads * head_stride
    kind_stride = layers * layer_stride
    block_stride = KV_KINDS * kind_stride
    block_s, block_d = next_power_of_two(slots), next_power_of_two(head_dim)
    args = (
        q,
        pool,
        block_table,
        lengths,
        out,
        block_table.shape[1],
        query_heads,
        block_stride,
        kind_stride,
        layer * layer_stride,
        head_stride,
    )
    options = {
        "HEAD_DIM": head_dim,
        "SLOTS": slots,
        "BLOCK_D": block_d,
        "BLOCK_S": block_s,
        "KV_GROUPS": kv_groups,
        "SCALE": 1.0 / math.sqrt(head_dim),
        "num_warps": default_num_warps(block_s, block_d),
        # Stated because the default is changing between Triton releases.
        "enable_fp_fusion": True,
    }
    return (batch, query_heads), args, options


def run_kernel(q, pool, layer, block_table, lengths, kv_groups, out) -> None:
    """Launch without validating. On CUDA this returns before the kernel finishes."""
    grid, args, options = kernel_launch(q, pool, layer, block_table, lengths, kv_groups, out)
    _paged_decode_kernel[grid](*args, **options)


def validate_shapes(q, pool, layer, block_table, lengths, kv_groups) -> None:
    """Everything validate checks that the host already knows: ranks, dtypes, devices, layer."""
    if triton is None:
        raise RuntimeError("the Triton decode kernel needs triton, which has Linux wheels only")
    if q.dim() != 3 or pool.dim() != 6 or block_table.dim() != 2 or lengths.dim() != 1:
        raise ValueError(f"bad ranks: q {q.shape}, pool {pool.shape}, table {block_table.shape}, lengths {lengths.shape}")
    if not pool.is_contiguous():
        raise ValueError("the pool must be contiguous: its strides are derived from its shape")
    if q.dtype not in KV_DTYPES or pool.dtype not in KV_DTYPES:
        raise ValueError(f"q is {q.dtype} and the pool {pool.dtype}; the kernel takes {KV_DTYPES}")
    batch, query_heads, head_dim = q.shape
    blocks, kinds, layers, kv_heads, slots, pool_dim = pool.shape
    if kinds != KV_KINDS:
        raise ValueError(f"pool has {kinds} kinds, expected K and V")
    if head_dim != pool_dim:
        raise ValueError(f"q head_dim {head_dim} differs from the pool's {pool_dim}")
    if kv_groups < 1 or query_heads != kv_heads * kv_groups:
        raise ValueError(f"{query_heads} query heads cannot share {kv_heads} KV heads in groups of {kv_groups}")
    if not 0 <= layer < layers:
        raise ValueError(f"layer {layer} out of range for {layers} layers")
    if block_table.shape[0] != batch or lengths.shape[0] != batch:
        raise ValueError(f"batch {batch} disagrees with table {block_table.shape} and lengths {lengths.shape}")
    devices = {q.device, pool.device, block_table.device, lengths.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {devices}")
    # The kernel runs on the current CUDA device and would dereference anything it is
    # given as a GPU address. Only the interpreter can run on other tensors.
    if not triton.knobs.runtime.interpret:
        if q.device.type != "cuda":
            raise ValueError(f"tensors are on {q.device} and TRITON_INTERPRET is not set")
        if q.device.index != torch.cuda.current_device():
            raise ValueError(f"tensors are on {q.device}, the current device is {torch.cuda.current_device()}")
    if block_table.is_floating_point() or lengths.is_floating_point():
        raise TypeError("the block table and lengths must be integers")


def validate(q, pool, layer, block_table, lengths, kv_groups) -> None:
    """Refuse what would otherwise be an out-of-bounds read on the GPU.

    A bad block id in a kernel does not raise: it returns garbage or takes the device
    down, so it is cheaper to spend two small launches and one read per call checking here.
    """
    validate_shapes(q, pool, layer, block_table, lengths, kv_groups)
    slots, blocks = pool.shape[4], pool.shape[0]
    # One read for all four, which waits for the GPU: the decode loop uses the trusted path instead.
    bounds = torch.stack([*torch.aminmax(lengths.long()), *torch.aminmax(block_table.long())])
    shortest, longest, lowest, highest = bounds.tolist()
    if shortest < 1 or longest > block_table.shape[1] * slots:
        raise ValueError(f"lengths in [{shortest}, {longest}] do not fit a table of {block_table.shape[1]} blocks")
    if lowest < 0 or highest >= blocks:
        raise ValueError(f"block ids in [{lowest}, {highest}] are outside a pool of {blocks} blocks")


def paged_decode_attention_triton(
    q: torch.Tensor,
    pool: torch.Tensor,
    layer: int,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    kv_groups: int,
) -> torch.Tensor:
    """Drop-in for paged_decode_attention_torch; the result has q's dtype and device."""
    validate(q, pool, layer, block_table, lengths, kv_groups)
    return _launch(q, pool, layer, block_table, lengths, kv_groups)


def paged_decode_attention_triton_trusted(
    q: torch.Tensor,
    pool: torch.Tensor,
    layer: int,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    kv_groups: int,
) -> torch.Tensor:
    """paged_decode_attention_triton for a table whose lengths and block ids the caller checked
    on the host, as PagedKVCache.decode_batch does once per step. It reads nothing back from the
    GPU, so a step's layers queue without waiting on one another; validate's read did that on
    every layer of every step."""
    validate_shapes(q, pool, layer, block_table, lengths, kv_groups)
    return _launch(q, pool, layer, block_table, lengths, kv_groups)


def _launch(q, pool, layer, block_table, lengths, kv_groups) -> torch.Tensor:
    q = q.contiguous()
    block_table, lengths = block_table.contiguous(), lengths.contiguous()
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    run_kernel(q, pool, layer, block_table, lengths, kv_groups, out)
    return out

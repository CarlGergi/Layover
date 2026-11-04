"""Paged decode attention: one query token per sequence against its block table."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def paged_decode_attention_torch(
    q: torch.Tensor,
    pool: torch.Tensor,
    layer: int,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    kv_groups: int,
) -> torch.Tensor:
    """Reference implementation that gathers each table into a padded tensor.

    q: [batch, query_heads, head_dim]; pool: [block, kind, layer, kv_head, slot, dim];
    block_table: [batch, max_blocks]; lengths: [batch]. Returns the attended values,
    shaped like q. BLAS-backed through torch's matmul on CUDA this is cuBLAS.
    """
    table = block_table.long()
    keys = pool[table, 0, layer]  # [batch, blocks, kv_head, slot, dim]
    values = pool[table, 1, layer]
    batch, blocks, heads, slots, dim = keys.shape
    # Trimming to the longest sequence means a batch of equal lengths is unpadded, and
    # gives the same rounding as an unbatched reference.
    longest = int(lengths.max())
    keys = keys.permute(0, 2, 1, 3, 4).reshape(batch, heads, blocks * slots, dim)[:, :, :longest]
    values = values.permute(0, 2, 1, 3, 4).reshape(batch, heads, blocks * slots, dim)[:, :, :longest]
    keys = keys.repeat_interleave(kv_groups, dim=1)
    values = values.repeat_interleave(kv_groups, dim=1)
    mask = None
    if int(lengths.min()) != longest:
        mask = (torch.arange(longest, device=q.device)[None] < lengths[:, None])[:, None, None, :]
    return F.scaled_dot_product_attention(q.unsqueeze(2), keys, values, attn_mask=mask).squeeze(2)

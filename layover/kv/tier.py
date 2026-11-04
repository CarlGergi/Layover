"""CPU memory tier for KV blocks of paused agents.

This is the plain copy path: it moves one tensor per offload through torch and works
on any device. The GPU transfer path (gather/scatter kernel, pinned staging buffer,
dedicated copy stream, FP8) replaces the two copy calls here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from layover.kv.paged_cache import BlockTable, PagedKVCache


@dataclass
class CpuHandle:
    data: torch.Tensor
    num_tokens: int

    @property
    def num_blocks(self) -> int:
        return self.data.shape[0]


class CpuTier:
    def __init__(self, capacity_blocks: int):
        self.capacity_blocks = capacity_blocks
        self.used_blocks = 0
        self.bytes_moved = 0

    def has_room(self, blocks: int) -> bool:
        return self.used_blocks + blocks <= self.capacity_blocks

    def offload(self, cache: PagedKVCache, table: BlockTable) -> CpuHandle:
        """Copy a table's blocks out and release them from the pool."""
        if not self.has_room(len(table.blocks)):
            raise MemoryError("CPU tier is full")
        ids = torch.tensor(table.blocks, device=cache.device, dtype=torch.long)
        data = cache.pool[ids].to("cpu", copy=True)
        handle = CpuHandle(data, table.num_tokens)
        self.used_blocks += handle.num_blocks
        self.bytes_moved += data.numel() * data.element_size()
        cache.release(table)
        return handle

    def restore(self, cache: PagedKVCache, handle: CpuHandle) -> BlockTable:
        """Copy a handle back into freshly allocated blocks and free its CPU memory."""
        blocks = cache.allocator.alloc(handle.num_blocks)
        ids = torch.tensor(blocks, device=cache.device, dtype=torch.long)
        cache.pool[ids] = handle.data.to(cache.device)
        self.bytes_moved += handle.data.numel() * handle.data.element_size()
        self.drop(handle)
        return BlockTable(blocks, handle.num_tokens)

    def drop(self, handle: CpuHandle) -> None:
        self.used_blocks -= handle.num_blocks

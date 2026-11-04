"""Paged KV storage: one pool, one block table per sequence.

Pool layout is [block, kind, layer, kv_head, slot, dim], where kind 0 is K and 1 is
V. Keeping a block contiguous across layers means the CPU tier can move a block as
one range and copy-on-write can clone one with a single assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from layover.config import ModelConfig
from layover.kv.allocator import BlockAllocator

KEY, VALUE = 0, 1


@dataclass
class BlockTable:
    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0


class PagedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        block_tokens: int = 16,
    ):
        self.cfg = cfg
        self.block_tokens = block_tokens
        self.device = torch.device(device)
        self.dtype = dtype
        self.allocator = BlockAllocator(num_blocks)
        self.pool = torch.zeros(
            num_blocks,
            2,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            block_tokens,
            cfg.head_dim,
            dtype=dtype,
            device=self.device,
        )

    @property
    def block_bytes(self) -> int:
        return self.pool[0].numel() * self.pool.element_size()

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_tokens)

    def ensure_capacity(self, table: BlockTable, tokens: int) -> None:
        missing = self.blocks_for(tokens) - len(table.blocks)
        if missing > 0:
            table.blocks.extend(self.allocator.alloc(missing))

    def make_writable(self, table: BlockTable, index: int) -> None:
        """Give the table a private copy of one block if the block is shared."""
        block = table.blocks[index]
        if self.allocator.refcount(block) > 1:
            fresh = self.allocator.alloc(1)[0]
            self.pool[fresh] = self.pool[block]
            self.allocator.decref(block)
            table.blocks[index] = fresh

    def prepare_write(self, table: BlockTable, start: int, end: int) -> None:
        """Make [start, end) writable, growing the table and breaking sharing."""
        self.ensure_capacity(table, end)
        for index in range(start // self.block_tokens, self.blocks_for(end)):
            self.make_writable(table, index)
        table.num_tokens = max(table.num_tokens, end)

    def fork(self, table: BlockTable, tokens: int | None = None) -> BlockTable:
        """Share the blocks that hold the first `tokens` tokens with a new table."""
        tokens = table.num_tokens if tokens is None else tokens
        if tokens > table.num_tokens:
            raise ValueError("cannot fork past the end of a table")
        blocks = table.blocks[: self.blocks_for(tokens)]
        for block in blocks:
            self.allocator.incref(block)
        return BlockTable(list(blocks), tokens)

    def truncate(self, table: BlockTable, tokens: int) -> None:
        """Forget everything from token `tokens` on, freeing blocks no longer needed."""
        keep = self.blocks_for(tokens)
        for block in table.blocks[keep:]:
            self.allocator.decref(block)
        del table.blocks[keep:]
        table.num_tokens = min(table.num_tokens, tokens)

    def release(self, table: BlockTable) -> None:
        for block in table.blocks:
            self.allocator.decref(block)
        table.blocks = []
        table.num_tokens = 0

    def _locate(self, table: BlockTable, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(start, start + count, device=self.device)
        ids = torch.tensor(table.blocks, device=self.device, dtype=torch.long)
        return ids[positions // self.block_tokens], positions % self.block_tokens

    def write_range(self, table: BlockTable, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k and v of shape [kv_heads, count, head_dim] at token `start`.

        The table must have been prepared with prepare_write.
        """
        blocks, slots = self._locate(table, start, k.shape[1])
        # Advanced indices around a slice put the index dimension first.
        self.pool[blocks, KEY, layer, :, slots] = k.transpose(0, 1).to(self.dtype)
        self.pool[blocks, VALUE, layer, :, slots] = v.transpose(0, 1).to(self.dtype)

    def read_range(self, table: BlockTable, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K and V of the first `end` tokens as [kv_heads, end, head_dim]."""
        ids = torch.tensor(table.blocks[: self.blocks_for(end)], device=self.device, dtype=torch.long)
        k = self.pool[ids, KEY, layer].permute(1, 0, 2, 3).flatten(1, 2)[:, :end]
        v = self.pool[ids, VALUE, layer].permute(1, 0, 2, 3).flatten(1, 2)[:, :end]
        return k, v

    def block_table_tensor(self, tables: list[BlockTable]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad block tables into [batch, max_blocks] and lengths into [batch]."""
        width = max(len(t.blocks) for t in tables)
        table = torch.zeros(len(tables), width, dtype=torch.int32)
        for row, t in enumerate(tables):
            table[row, : len(t.blocks)] = torch.tensor(t.blocks, dtype=torch.int32)
        lengths = torch.tensor([t.num_tokens for t in tables], dtype=torch.int32)
        return table.to(self.device), lengths.to(self.device)

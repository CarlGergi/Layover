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

    def _to_device(self, host: torch.Tensor) -> torch.Tensor:
        """One copy to the device. From pinned memory it is queued behind the work already
        launched instead of waiting for it, which a copy from pageable memory does."""
        if self.device.type != "cuda":
            return host.to(self.device)
        return host.pin_memory().to(self.device, non_blocking=True)

    def chunk_locations(self, table: BlockTable, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For tokens [start, start + count): the ids of every block they and the tokens before
        them occupy, then each token's block and slot. One copy to the device, which every
        layer of a prefill chunk shares."""
        ids = self._to_device(torch.tensor(table.blocks[: self.blocks_for(start + count)], dtype=torch.long))
        positions = torch.arange(start, start + count, device=self.device)
        return ids, ids[positions // self.block_tokens], positions % self.block_tokens

    def write_at(self, blocks: torch.Tensor, slots: torch.Tensor, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k and v of shape [kv_heads, count, head_dim] at the given blocks and slots."""
        # Advanced indices around a slice put the index dimension first.
        self.pool[blocks, KEY, layer, :, slots] = k.transpose(0, 1).to(self.dtype)
        self.pool[blocks, VALUE, layer, :, slots] = v.transpose(0, 1).to(self.dtype)

    def read_blocks(self, ids: torch.Tensor, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V of the first `end` tokens held in blocks `ids`, as [kv_heads, end, head_dim]."""
        k = self.pool[ids, KEY, layer].permute(1, 0, 2, 3).flatten(1, 2)[:, :end]
        v = self.pool[ids, VALUE, layer].permute(1, 0, 2, 3).flatten(1, 2)[:, :end]
        return k, v

    def write_range(self, table: BlockTable, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k and v of shape [kv_heads, count, head_dim] at token `start`.

        The table must have been prepared with prepare_write.
        """
        _, blocks, slots = self.chunk_locations(table, start, k.shape[1])
        self.write_at(blocks, slots, layer, k, v)

    def read_range(self, table: BlockTable, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K and V of the first `end` tokens as [kv_heads, end, head_dim]."""
        ids = self._to_device(torch.tensor(table.blocks[: self.blocks_for(end)], dtype=torch.long))
        return self.read_blocks(ids, layer, end)

    def decode_batch(self, tables: list[BlockTable]) -> "DecodeBatch":
        """Everything one decode step's layers need about its tables, checked on the host and
        copied to the device once.

        Each table must already hold the token being written, as prepare_write leaves it. The
        checks are the ones a kernel cannot make for itself, since a bad block id there reads
        another sequence's memory or takes the device down; here they cost no GPU read.
        """
        width = max(len(t.blocks) for t in tables)
        rows = []
        for t in tables:
            if not 1 <= t.num_tokens <= len(t.blocks) * self.block_tokens:
                raise ValueError(f"a table of {len(t.blocks)} blocks cannot hold {t.num_tokens} tokens")
            if t.blocks and not 0 <= min(t.blocks) <= max(t.blocks) < self.allocator.num_blocks:
                raise ValueError(f"block ids {t.blocks} are outside a pool of {self.allocator.num_blocks} blocks")
            last = t.num_tokens - 1
            # Padding repeats block 0, a real block, and the kernel never reads past a length.
            rows.append([t.blocks[last // self.block_tokens], last % self.block_tokens, t.num_tokens, *t.blocks, *[0] * (width - len(t.blocks))])
        packed = self._to_device(torch.tensor(rows, dtype=torch.int32))
        return DecodeBatch(packed[:, 0].long(), packed[:, 1].long(), packed[:, 3:], packed[:, 2])

    def block_table_tensor(self, tables: list[BlockTable]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad block tables into [batch, max_blocks] and lengths into [batch]."""
        batch = self.decode_batch(tables)
        return batch.table, batch.lengths


@dataclass(frozen=True)
class DecodeBatch:
    """One decode step's tables on the device: where each new token's K and V go, the padded
    block table [batch, max_blocks] and the lengths [batch] that include the new token."""

    write_blocks: torch.Tensor
    write_slots: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor

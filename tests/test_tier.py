"""CPU tier round trips on any device, verified by checksum."""

import zlib

import pytest
import torch

from layover.config import ModelConfig
from layover.kv import BlockTable, PagedKVCache
from layover.kv.tier import CpuTier

CFG = ModelConfig.micro()


def crc(cache, table):
    ids = torch.tensor(table.blocks)
    return zlib.crc32(cache.pool[ids].cpu().view(torch.uint16).numpy().tobytes())


def filled(cache, tokens):
    table = BlockTable()
    cache.prepare_write(table, 0, tokens)
    for layer in range(CFG.num_hidden_layers):
        cache.write_range(
            table, layer, 0,
            torch.randn(CFG.num_key_value_heads, tokens, CFG.head_dim),
            torch.randn(CFG.num_key_value_heads, tokens, CFG.head_dim),
        )
    return table


def test_round_trip_is_byte_exact_across_scattered_blocks():
    torch.manual_seed(0)
    cache = PagedKVCache(CFG, 32, dtype=torch.bfloat16)
    # Interleave two tables so the blocks are not contiguous in the pool.
    a, b = BlockTable(), filled(cache, 16)
    for start in range(0, 64, 16):
        cache.prepare_write(a, start, start + 16)
        cache.prepare_write(b, b.num_tokens, b.num_tokens + 16)
    for layer in range(CFG.num_hidden_layers):
        heads, dim = CFG.num_key_value_heads, CFG.head_dim
        cache.write_range(a, layer, 0, torch.randn(heads, 64, dim), torch.randn(heads, 64, dim))
    before = crc(cache, a)
    tier = CpuTier(16)
    handle = tier.offload(cache, a)
    assert tier.used_blocks == 4 and not a.blocks
    # Scribble over the freed blocks to prove the restore does not rely on them.
    cache.pool[cache.pool.shape[0] - cache.allocator.num_free :] = 0
    restored = tier.restore(cache, handle)
    assert crc(cache, restored) == before
    assert restored.num_tokens == 64 and tier.used_blocks == 0


def test_offload_refuses_when_full_and_leaves_table_intact():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    tier = CpuTier(1)
    with pytest.raises(MemoryError):
        tier.offload(cache, table)
    assert len(table.blocks) == 2 and cache.allocator.num_used == 2


def test_offload_of_shared_block_keeps_it_for_the_sibling():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 16)
    sibling = cache.fork(table)
    tier = CpuTier(4)
    handle = tier.offload(cache, table)
    assert cache.allocator.refcount(sibling.blocks[0]) == 1
    tier.drop(handle)
    cache.release(sibling)
    assert cache.allocator.num_used == 0

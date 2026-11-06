"""CPU tier round trips on any device and backend, verified by checksum."""

import zlib

import pytest
import torch

from layover.config import ModelConfig
from layover.kv import BlockTable, PagedKVCache
from layover.kv.tier import CpuTier
from layover.kv.transfer import DelayedBackend, NativeBackend, PlainCopyBackend

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


@pytest.fixture(params=["plain", "delayed", "native"])
def backend(request):
    if request.param == "plain":
        created = PlainCopyBackend()
    elif request.param == "delayed":
        created = DelayedBackend(polls=2)
    else:
        created = NativeBackend("cpu", staging_bytes=1 << 16, ids_capacity=256)
    yield created
    created.close()


def test_round_trip_is_byte_exact_across_scattered_blocks(backend):
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
    tier = CpuTier(16, backend)
    handle = tier.offload(cache, a)
    assert tier.used_blocks == 4 and not a.blocks
    # Scribble over the freed blocks to prove the restore does not rely on them.
    cache.pool[cache.pool.shape[0] - cache.allocator.num_free :] = 0
    restored = tier.restore(cache, handle)
    assert crc(cache, restored) == before
    assert restored.num_tokens == 64 and tier.used_blocks == 0
    assert tier.bytes_out == tier.bytes_in == 4 * cache.block_bytes
    assert tier.bytes_moved == 8 * cache.block_bytes


def test_offload_refuses_when_full_and_leaves_table_intact():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    tier = CpuTier(1)
    with pytest.raises(MemoryError):
        tier.offload(cache, table)
    assert len(table.blocks) == 2 and cache.allocator.num_used == 2


def test_offload_of_shared_block_keeps_it_for_the_sibling(backend):
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 16)
    sibling = cache.fork(table)
    tier = CpuTier(4, backend)
    handle = tier.offload(cache, table)
    assert cache.allocator.refcount(sibling.blocks[0]) == 1
    tier.drop(handle)
    tier.poll()
    cache.release(sibling)
    assert cache.allocator.num_used == 0
    assert tier.used_blocks == 0


def test_blocks_stay_allocated_until_the_offload_completes():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    tier = CpuTier(8, DelayedBackend(polls=2))
    handle = tier.start_offload(cache, table)
    assert handle.in_flight and not table.blocks
    assert cache.allocator.num_used == 2 and tier.blocks_freeing() == 2
    assert tier.poll() is False
    assert cache.allocator.num_used == 2
    assert tier.poll() is True
    assert cache.allocator.num_used == 0 and tier.blocks_freeing() == 0
    assert not handle.in_flight and tier.in_flight == 0


def test_blocks_shared_with_a_sibling_are_not_counted_as_freeing():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    sibling = cache.fork(table, 16)
    tier = CpuTier(8, DelayedBackend(polls=5))
    tier.start_offload(cache, table)
    assert tier.blocks_freeing() == 1
    tier.flush()
    assert cache.allocator.refcount(sibling.blocks[0]) == 1


def test_an_offload_reads_the_block_as_of_completion_with_its_reference_held():
    # A sibling writing to a shared block must copy it first, so the offload's bytes survive.
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 16)
    sibling = cache.fork(table)
    before = crc(cache, table)
    tier = CpuTier(8, DelayedBackend(polls=4))
    handle = tier.start_offload(cache, table)
    cache.prepare_write(sibling, 8, 16)
    assert cache.allocator.refcount(sibling.blocks[0]) == 1
    cache.pool[sibling.blocks[0]] = 3.0
    restored = tier.restore(cache, handle)
    assert crc(cache, restored) == before


def test_restore_gives_blocks_at_once_and_returns_memory_on_completion():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    before = crc(cache, table)
    tier = CpuTier(8, DelayedBackend(polls=3))
    handle = tier.offload(cache, table)
    assert tier.used_blocks == 2
    restored, ticket = tier.start_restore(cache, handle)
    assert len(restored.blocks) == 2 and cache.allocator.num_used == 2
    assert not ticket.done()
    assert tier.used_blocks == 2 and tier.in_flight == 1
    while not ticket.done():
        pass
    tier.poll()
    assert crc(cache, restored) == before
    assert tier.used_blocks == 0 and tier.in_flight == 0


def test_partial_restore_takes_a_prefix_and_frees_the_whole_handle(backend):
    cache = PagedKVCache(CFG, 16, dtype=torch.bfloat16)
    table = filled(cache, 64)
    prefix = BlockTable(table.blocks[:2], 32)
    before = crc(cache, prefix)
    tier = CpuTier(16, backend)
    handle = tier.offload(cache, table)
    restored, ticket = tier.start_restore(cache, handle, 2)
    ticket.wait()
    tier.poll()
    assert len(restored.blocks) == 2 and restored.num_tokens == 32
    assert crc(cache, restored) == before
    assert tier.used_blocks == 0
    assert tier.bytes_in == 2 * cache.block_bytes


def test_empty_restore_needs_no_blocks(backend):
    cache = PagedKVCache(CFG, 4, dtype=torch.bfloat16)
    table = filled(cache, 16)
    tier = CpuTier(4, backend)
    handle = tier.offload(cache, table)
    restored, ticket = tier.start_restore(cache, handle, 0)
    assert ticket.done() and not restored.blocks
    tier.poll()
    assert tier.used_blocks == 0


def test_restore_of_an_offload_in_flight_is_refused_until_it_is_done():
    cache = PagedKVCache(CFG, 4, dtype=torch.bfloat16)
    tier = CpuTier(4, DelayedBackend(polls=3))
    handle = tier.start_offload(cache, filled(cache, 16))
    with pytest.raises(RuntimeError, match="in flight"):
        tier.start_restore(cache, handle)
    # The blocking form waits instead.
    assert tier.restore(cache, handle).blocks


def test_dropping_an_offload_in_flight_finishes_it_then_frees_everything():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    tier = CpuTier(8, DelayedBackend(polls=3))
    handle = tier.start_offload(cache, filled(cache, 32))
    tier.drop(handle)
    assert cache.allocator.num_used == 2 and tier.used_blocks == 2
    tier.flush()
    assert cache.allocator.num_used == 0 and tier.used_blocks == 0


def test_flush_waits_for_both_directions():
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    tier = CpuTier(8, DelayedBackend(polls=100))
    first = tier.start_offload(cache, filled(cache, 16))
    tier.flush()
    second = tier.start_offload(cache, filled(cache, 16))
    tier.start_restore(cache, first)
    assert tier.in_flight == 2
    tier.flush()
    assert tier.in_flight == 0 and not second.in_flight


def test_fp8_tier_stores_half_the_bytes_and_stays_within_error(backend):
    torch.manual_seed(1)
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = filled(cache, 32)
    original = cache.pool[torch.tensor(table.blocks)].clone().float()
    tier = CpuTier(8, backend, fp8=True)
    handle = tier.offload(cache, table)
    assert handle.payload.fp8
    assert tier.bytes_out < 0.51 * 2 * cache.block_bytes
    restored = tier.restore(cache, handle)
    after = cache.pool[torch.tensor(restored.blocks)].float()
    assert (after - original).abs().max() <= 2.0**-4 * original.abs().max()
    assert ((after - original).abs() <= 2.0**-4 * original.abs() + 1e-1).all()


class _Held:
    """A ticket that never finishes, so blocks stay held while a test looks at them."""

    payload = None
    ready_at = None

    def done(self):
        return False

    def wait(self):
        raise AssertionError("nothing here may block")


class _HeldBackend(PlainCopyBackend):
    def start_out(self, cache, blocks, fp8=False):
        return _Held()


def test_blocks_freeing_counts_blocks_only_copies_and_an_evictable_owner_hold():
    cache = PagedKVCache(CFG, 16)
    tier = CpuTier(100, _HeldBackend())
    # 0 only the copy, 1 the copy and the tree, 2 also a live table, 3 two copies at once.
    table = BlockTable(cache.allocator.alloc(4), 64)
    tree = {1, 2}
    for block in tree:
        cache.allocator.incref(block)
    cache.allocator.incref(2)
    cache.allocator.incref(3)
    other = BlockTable([3], 16)
    tier.start_offload(cache, table)
    tier.start_offload(cache, other)
    assert tier.blocks_freeing() == 2  # 0 and 3: the tree's references count against 1 and 2.
    assert tier.blocks_freeing(tree) == 3  # 1 joins; 2 has an owner that will not let go.


def test_peak_usage_outlives_the_session_that_set_it():
    cache = PagedKVCache(CFG, 16)
    tier = CpuTier(100)
    handle = tier.offload(cache, filled(cache, 64))
    tier.drop(handle)
    assert tier.used_blocks == 0 and tier.peak_used_blocks == 4

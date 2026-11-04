"""Property tests for the block allocator, reference counts, and copy-on-write."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, invariant, precondition, rule

from layover.config import ModelConfig
from layover.kv import BlockAllocator, BlockTable, OutOfBlocks, PagedKVCache


class AllocatorMachine(RuleBasedStateMachine):
    """Compare the allocator against a dictionary of expected reference counts."""

    NUM_BLOCKS = 32
    held = Bundle("held")

    def __init__(self):
        super().__init__()
        self.allocator = BlockAllocator(self.NUM_BLOCKS)
        self.expected: dict[int, int] = {}

    @rule(target=held, count=st.integers(1, 8))
    def alloc(self, count):
        if count > self.allocator.num_free:
            before = self.allocator.num_free
            with pytest.raises(OutOfBlocks):
                self.allocator.alloc(count)
            assert self.allocator.num_free == before
            return multiple_empty()
        blocks = self.allocator.alloc(count)
        assert len(set(blocks)) == count
        for block in blocks:
            assert block not in self.expected, "allocated a live block"
            self.expected[block] = 1
        return multiple_of(blocks)

    @rule(block=held)
    def incref(self, block):
        if block in self.expected:
            self.allocator.incref(block)
            self.expected[block] += 1

    @rule(block=held)
    def decref(self, block):
        if block in self.expected:
            freed = self.allocator.decref(block)
            self.expected[block] -= 1
            assert freed == (self.expected[block] == 0)
            if freed:
                del self.expected[block]
        else:
            with pytest.raises(ValueError):
                self.allocator.decref(block)

    @invariant()
    def counts_agree(self):
        assert self.allocator.num_used == len(self.expected)
        assert self.allocator.num_free + self.allocator.num_used == self.NUM_BLOCKS
        for block in range(self.NUM_BLOCKS):
            assert self.allocator.refcount(block) == self.expected.get(block, 0)


def multiple_of(values):
    from hypothesis.stateful import multiple

    return multiple(*values)


def multiple_empty():
    from hypothesis.stateful import multiple

    return multiple()


TestAllocator = AllocatorMachine.TestCase


CFG = ModelConfig.micro(num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1, head_dim=2)
BLOCK = 4


class CacheMachine(RuleBasedStateMachine):
    """Sequences hold a shadow copy of their tokens; sharing must never leak writes."""

    NUM_BLOCKS = 16

    def __init__(self):
        super().__init__()
        self.cache = PagedKVCache(CFG, self.NUM_BLOCKS, dtype=torch.float32, block_tokens=BLOCK)
        self.seqs: list[tuple[BlockTable, list[float]]] = [(BlockTable(), [])]
        self.counter = 1.0

    def _write(self, table, shadow, start, count):
        self.cache.prepare_write(table, start, start + count)
        values = [self.counter + i for i in range(count)]
        self.counter += count
        tensor = torch.tensor(values).view(1, count, 1).expand(1, count, 2)
        self.cache.write_range(table, 0, start, tensor, -tensor)
        shadow[start:start + count] = values

    @rule(index=st.integers(0, 15), count=st.integers(1, 9))
    def append(self, index, count):
        table, shadow = self.seqs[index % len(self.seqs)]
        need = self.cache.blocks_for(len(shadow) + count) - len(table.blocks)
        index = len(shadow) // BLOCK
        if index < len(table.blocks) and self.cache.allocator.refcount(table.blocks[index]) > 1:
            need += 1
        if need > self.cache.allocator.num_free:
            return
        self._write(table, shadow, len(shadow), count)

    @rule(index=st.integers(0, 15), tokens=st.integers(0, 64))
    def fork(self, index, tokens):
        table, shadow = self.seqs[index % len(self.seqs)]
        tokens = min(tokens, len(shadow))
        child = self.cache.fork(table, tokens)
        self.seqs.append((child, shadow[:tokens]))

    @rule(index=st.integers(0, 15), tokens=st.integers(0, 64))
    def truncate(self, index, tokens):
        table, shadow = self.seqs[index % len(self.seqs)]
        tokens = min(tokens, len(shadow))
        self.cache.truncate(table, tokens)
        del shadow[tokens:]

    @precondition(lambda self: len(self.seqs) > 1)
    @rule(index=st.integers(0, 15))
    def release(self, index):
        table, _ = self.seqs.pop(index % len(self.seqs))
        self.cache.release(table)

    def teardown(self):
        for table, _ in self.seqs:
            self.cache.release(table)
        assert self.cache.allocator.num_used == 0, "a block outlived every table"

    @invariant()
    def contents_and_refcounts(self):
        references: dict[int, int] = {}
        for table, shadow in self.seqs:
            assert table.num_tokens == len(shadow)
            assert len(table.blocks) == self.cache.blocks_for(len(shadow))
            for block in table.blocks:
                references[block] = references.get(block, 0) + 1
            if shadow:
                k, v = self.cache.read_range(table, 0, len(shadow))
                assert k[0, :, 0].tolist() == shadow
                assert v[0, :, 0].tolist() == [-x for x in shadow]
        for block in range(self.NUM_BLOCKS):
            assert self.cache.allocator.refcount(block) == references.get(block, 0)


TestCache = CacheMachine.TestCase
TestCache.settings = settings(max_examples=60, stateful_step_count=40, deadline=None)


def test_alloc_is_all_or_nothing():
    allocator = BlockAllocator(2)
    with pytest.raises(OutOfBlocks):
        allocator.alloc(3)
    assert allocator.num_free == 2


def test_double_free_is_rejected():
    allocator = BlockAllocator(1)
    block = allocator.alloc()[0]
    assert allocator.decref(block)
    with pytest.raises(ValueError):
        allocator.decref(block)


@given(st.integers(1, 100), st.integers(1, 64))
def test_blocks_for_covers_tokens_tightly(tokens, block_tokens):
    cache = PagedKVCache(CFG, 1, dtype=torch.float32, block_tokens=block_tokens)
    blocks = cache.blocks_for(tokens)
    assert (blocks - 1) * block_tokens < tokens <= blocks * block_tokens


def test_write_to_shared_partial_block_does_not_reach_the_sibling():
    cache = PagedKVCache(CFG, 8, dtype=torch.float32, block_tokens=BLOCK)
    parent = BlockTable()
    cache.prepare_write(parent, 0, 2)
    ones = torch.ones(1, 2, 2)
    cache.write_range(parent, 0, 0, ones, ones)
    child = cache.fork(parent)
    assert parent.blocks == child.blocks
    cache.prepare_write(child, 2, 3)
    cache.write_range(child, 0, 2, 5 * torch.ones(1, 1, 2), 5 * torch.ones(1, 1, 2))
    assert parent.blocks != child.blocks
    assert cache.pool[parent.blocks[0], 0, 0, 0, 2].sum() == 0
    assert cache.read_range(child, 0, 2)[0].equal(ones)

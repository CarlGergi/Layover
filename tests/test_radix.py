"""Radix tree against a dictionary model, and copy-on-write through real blocks."""

import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from layover.config import ModelConfig
from layover.kv import BlockAllocator, BlockTable, PagedKVCache
from layover.kv.radix import RadixTree

SIZE = 2
# A small alphabet makes shared prefixes and edge splits common.
token_lists = st.lists(st.integers(0, 1), min_size=0, max_size=14)


class RadixMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.allocator = BlockAllocator(256)
        self.tree = RadixTree(self.allocator, SIZE)
        # Every stored block-aligned prefix mapped to the block for its last block.
        self.model: dict[tuple[int, ...], int] = {}
        self.held: list[list[int]] = []

    def _model_match(self, tokens):
        out, count = [], 0
        while (count + 1) * SIZE <= len(tokens):
            block = self.model.get(tuple(tokens[: (count + 1) * SIZE]))
            if block is None:
                break
            out.append(block)
            count += 1
        return out

    @rule(tokens=token_lists)
    def insert(self, tokens):
        count = len(tokens) // SIZE
        if count > self.allocator.num_free:
            return
        blocks = self.allocator.alloc(count)
        self.tree.insert(tokens, blocks)
        for i in range(count):
            self.model.setdefault(tuple(tokens[: (i + 1) * SIZE]), blocks[i])
        # The caller's references end here, as when a turn finishes.
        for block in blocks:
            self.allocator.decref(block)

    @rule(tokens=token_lists)
    def match(self, tokens):
        assert self.tree.match(tokens) == self._model_match(tokens)

    @rule(tokens=token_lists)
    def hold(self, tokens):
        blocks = self.tree.match(tokens)
        for block in blocks:
            self.allocator.incref(block)
        self.held.append(blocks)

    @rule()
    def unhold(self):
        if self.held:
            for block in self.held.pop():
                self.allocator.decref(block)

    @rule(count=st.integers(1, 10))
    def evict(self, count):
        before = self.allocator.num_free
        freed = self.tree.evict(count)
        assert freed <= count
        assert self.allocator.num_free == before + freed
        alive = self._reachable()
        self.model = {k: v for k, v in self.model.items() if v in alive}

    def _reachable(self):
        out, stack = set(), [self.tree.root]
        while stack:
            node = stack.pop()
            out.update(node.blocks)
            stack.extend(node.children.values())
        return out

    @invariant()
    def references_add_up(self):
        counts: dict[int, int] = {}
        for block in self._reachable():
            counts[block] = counts.get(block, 0) + 1
        for blocks in self.held:
            for block in blocks:
                counts[block] = counts.get(block, 0) + 1
        for block in range(self.allocator.num_blocks):
            assert self.allocator.refcount(block) == counts.get(block, 0)

    def teardown(self):
        for blocks in self.held:
            for block in blocks:
                self.allocator.decref(block)
        self.tree.clear()
        assert self.allocator.num_used == 0


TestRadix = RadixMachine.TestCase
TestRadix.settings = settings(max_examples=120, stateful_step_count=40, deadline=None)


def test_edge_split_keeps_both_branches():
    allocator = BlockAllocator(8)
    tree = RadixTree(allocator, 2)
    a = allocator.alloc(3)
    b = allocator.alloc(3)
    tree.insert([1, 1, 2, 2, 3, 3], a)
    tree.insert([1, 1, 2, 2, 9, 9], b)
    assert tree.match([1, 1, 2, 2, 3, 3]) == a
    assert tree.match([1, 1, 2, 2, 9, 9]) == [a[0], a[1], b[2]]
    assert tree.match([1, 1, 7, 7]) == [a[0]]


def test_partial_tail_tokens_are_not_stored():
    allocator = BlockAllocator(4)
    tree = RadixTree(allocator, 2)
    assert tree.insert([1, 2, 3], allocator.alloc(2)) == 1
    assert tree.num_blocks == 1


def test_evict_spares_blocks_in_use():
    allocator = BlockAllocator(4)
    tree = RadixTree(allocator, 2)
    blocks = allocator.alloc(2)
    tree.insert([1, 1, 2, 2], blocks)
    # The caller still holds both, so nothing can be freed.
    assert tree.evict(2) == 0
    allocator.decref(blocks[1])
    assert tree.evict(2) == 1
    assert tree.match([1, 1, 2, 2]) == [blocks[0]]


def test_evict_prefers_least_recently_used():
    allocator = BlockAllocator(4)
    tree = RadixTree(allocator, 2)
    old, fresh = allocator.alloc(1), allocator.alloc(1)
    tree.insert([1, 1], old)
    tree.insert([2, 2], fresh)
    allocator.decref(old[0])
    allocator.decref(fresh[0])
    tree.match([1, 1])
    assert tree.evict(1) == 1
    assert tree.match([1, 1]) == old and tree.match([2, 2]) == []


def test_fork_from_tree_writes_copy_on_write():
    cfg = ModelConfig.micro(num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1, head_dim=2)
    cache = PagedKVCache(cfg, 8, dtype=torch.float32, block_tokens=2)
    tree = RadixTree(cache.allocator, 2)
    parent = BlockTable()
    cache.prepare_write(parent, 0, 4)
    data = torch.arange(1.0, 9.0).view(1, 4, 2)
    cache.write_range(parent, 0, 0, data, data)
    tree.insert([5, 6, 7, 8], parent.blocks)
    before = cache.pool.clone()

    shared = tree.match([5, 6, 7, 8])
    for block in shared:
        cache.allocator.incref(block)
    child = BlockTable(list(shared), 3)
    cache.prepare_write(child, 3, 4)
    cache.write_range(child, 0, 3, torch.full((1, 1, 2), 99.0), torch.full((1, 1, 2), 99.0))
    assert child.blocks[0] == parent.blocks[0] and child.blocks[1] != parent.blocks[1]
    assert cache.pool[parent.blocks[1]].equal(before[parent.blocks[1]])
    assert cache.read_range(child, 0, 3)[0].equal(data[:, :3])

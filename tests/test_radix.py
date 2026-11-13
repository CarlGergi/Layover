"""Radix tree against a dictionary model and an LRU free queue, and copy-on-write through real blocks."""

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


def test_evict_takes_a_path_from_its_tail():
    allocator = BlockAllocator(4)
    tree = RadixTree(allocator, 2)
    blocks = allocator.alloc(3)
    tree.insert([1, 1, 2, 2, 3, 3], blocks)
    for block in blocks:
        allocator.decref(block)
    assert tree.evict(1) == 1
    assert allocator.refcount(blocks[2]) == 0 and allocator.refcount(blocks[1]) == 1
    assert tree.evict(1) == 1
    assert allocator.refcount(blocks[1]) == 0 and allocator.refcount(blocks[0]) == 1


def stored(allocator, tree, *paths):
    """Insert each path on fresh blocks, leaving only the tree's references."""
    out = []
    for path in paths:
        blocks = allocator.alloc(len(path) // 2)
        tree.insert(path, blocks)
        for block in blocks:
            allocator.decref(block)
        out.append(blocks)
    return out


def test_a_partial_hit_refreshes_only_the_blocks_it_covered():
    allocator = BlockAllocator(8)
    tree = RadixTree(allocator, 2)
    long, other = stored(allocator, tree, [1, 1, 2, 2, 3, 3], [7, 7])
    assert tree.match([1, 1, 9, 9]) == [long[0]]
    # The two blocks past the hit are older than the other path, so they go before it.
    assert tree.evict(2) == 2
    assert allocator.refcount(long[2]) == allocator.refcount(long[1]) == 0
    assert allocator.refcount(other[0]) == 1
    assert tree.evict(1) == 1
    assert allocator.refcount(other[0]) == 0 and allocator.refcount(long[0]) == 1


def test_losing_a_sibling_does_not_make_the_other_look_recent():
    allocator = BlockAllocator(8)
    tree = RadixTree(allocator, 2)
    first, second, newer = stored(allocator, tree, [1, 1, 2, 2], [1, 1, 3, 3], [9, 9])
    # Only the shared block is now newer than [9, 9].
    assert tree.match([1, 1]) == [first[0]]
    assert tree.evict(1) == 1
    assert allocator.refcount(first[1]) == 0
    # Its edge now has no sibling, but [1, 1, 3, 3] is still older than [9, 9].
    assert tree.evict(1) == 1
    assert allocator.refcount(second[1]) == 0 and allocator.refcount(newer[0]) == 1


class FreeQueueMachine(RuleBasedStateMachine):
    """Eviction order against a block-level LRU free queue, the order vLLM keeps.

    As in vllm/v1/core/block_pool.py: a hit takes blocks out of the queue while a turn holds
    them, a finished turn puts them at the end tail first, and eviction takes from the front.
    The queue is written from that description. The tree keeps one copy of a block a concurrent
    turn computed too, so a finish counts for the whole path it reached.
    """

    def __init__(self):
        super().__init__()
        self.allocator = BlockAllocator(256)
        self.tree = RadixTree(self.allocator, SIZE)
        self.queue: list[int] = []
        self.holders: dict[int, int] = {}
        self.turns: list[tuple[list[int], list[int], list[int]]] = []

    @rule(tokens=token_lists)
    def start(self, tokens):
        if len(tokens) // SIZE > self.allocator.num_free:
            return
        hit = self.tree.match(tokens)
        for block in hit:
            self.allocator.incref(block)
            self.holders[block] = self.holders.get(block, 0) + 1
            if block in self.queue:
                self.queue.remove(block)
        fresh = self.allocator.alloc(len(tokens) // SIZE - len(hit))
        self.turns.append((tokens, hit, fresh))

    @rule(pick=st.integers(0, 100))
    def finish(self, pick):
        if not self.turns:
            return
        tokens, hit, fresh = self.turns.pop(pick % len(self.turns))
        self.tree.insert(tokens, hit + fresh)
        for block in hit:
            self.holders[block] -= 1
        for block in hit + fresh:
            self.allocator.decref(block)
        for block in reversed(self.tree.match(tokens)):
            if not self.holders.get(block):
                if block in self.queue:
                    self.queue.remove(block)
                self.queue.append(block)

    @rule(count=st.integers(1, 3))
    def evict(self, count):
        before = self.tree.block_ids()
        expected, self.queue = self.queue[:count], self.queue[count:]
        assert self.tree.evict(count) == len(expected)
        assert before - self.tree.block_ids() == set(expected)

    @invariant()
    def queue_is_what_only_the_tree_holds(self):
        held = {block for _, hit, _ in self.turns for block in hit}
        assert self.tree.block_ids() == set(self.queue) | held
        assert all(self.allocator.refcount(block) == 1 for block in self.queue)

    @invariant()
    def an_ancestor_is_never_older_than_its_descendants(self):
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                assert child.last_used <= node.last_used
                stack.append(child)

    def teardown(self):
        for _, hit, fresh in self.turns:
            for block in hit + fresh:
                self.allocator.decref(block)
        self.tree.clear()
        assert self.allocator.num_used == 0


TestFreeQueue = FreeQueueMachine.TestCase
TestFreeQueue.settings = settings(max_examples=300, stateful_step_count=40, deadline=None)

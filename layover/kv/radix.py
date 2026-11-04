"""Token-prefix radix tree over full KV blocks.

Each edge holds a run of whole blocks and the tokens they encode, so a lookup returns
blocks that are safe to share: a KV entry depends only on the tokens up to it. The
tree owns one reference on every block it stores. SGLang's RadixAttention (Zheng et
al., NeurIPS 2024) is the reference design; here the tree also backs copy-on-write
forks through PagedKVCache, so a shared block is cloned rather than overwritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from layover.kv.allocator import BlockAllocator


class _Node:
    __slots__ = ("key", "tokens", "blocks", "children", "parent", "last_used")

    def __init__(self, key: tuple[int, ...], tokens: tuple[int, ...], blocks: list[int], parent: "_Node | None"):
        # The first block of the edge, as filed in the parent. Eviction shortens
        # tokens, so the key has to be remembered rather than recomputed.
        self.key = key
        self.tokens = tokens
        self.blocks = blocks
        self.children: dict[tuple[int, ...], _Node] = {}
        self.parent = parent
        self.last_used = 0


class RadixTree:
    def __init__(self, allocator: BlockAllocator, block_tokens: int = 16):
        self.allocator = allocator
        self.block_tokens = block_tokens
        self.root = _Node((), (), [], None)
        self._tick = 0

    def _key(self, tokens: Sequence[int], start: int = 0) -> tuple[int, ...]:
        return tuple(tokens[start : start + self.block_tokens])

    def _shared_blocks(self, edge: tuple[int, ...], tokens: Sequence[int], offset: int) -> int:
        """How many leading blocks of `edge` equal `tokens` from `offset` on."""
        size = self.block_tokens
        count = 0
        while (count + 1) * size <= len(edge) and offset + (count + 1) * size <= len(tokens):
            if edge[count * size : (count + 1) * size] != tuple(tokens[offset + count * size : offset + (count + 1) * size]):
                break
            count += 1
        return count

    def match(self, tokens: Sequence[int]) -> list[int]:
        """Blocks of the longest stored prefix of `tokens`, whole blocks only."""
        self._tick += 1
        node, offset, blocks = self.root, 0, []
        while True:
            node.last_used = self._tick
            child = node.children.get(self._key(tokens, offset))
            if child is None:
                return blocks
            shared = self._shared_blocks(child.tokens, tokens, offset)
            blocks.extend(child.blocks[:shared])
            if shared < len(child.blocks):
                child.last_used = self._tick
                return blocks
            node, offset = child, offset + shared * self.block_tokens

    def _split(self, node: _Node, at_blocks: int) -> _Node:
        """Cut an edge after `at_blocks` blocks and return the new upper node."""
        cut = at_blocks * self.block_tokens
        parent = node.parent
        upper = _Node(node.key, node.tokens[:cut], node.blocks[:at_blocks], parent)
        upper.last_used = node.last_used
        parent.children[upper.key] = upper
        node.tokens, node.blocks = node.tokens[cut:], node.blocks[at_blocks:]
        node.key = self._key(node.tokens)
        node.parent = upper
        upper.children[node.key] = node
        return upper

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> int:
        """Store a prefix, taking a reference on each block not already present.

        Tokens past the last whole block are ignored. Where the tree already holds a
        prefix it keeps its own blocks. Returns how many blocks were added.
        """
        size = self.block_tokens
        count = min(len(tokens) // size, len(blocks))
        tokens = tuple(tokens[: count * size])
        self._tick += 1
        node, done = self.root, 0
        while done < count:
            node.last_used = self._tick
            child = node.children.get(self._key(tokens, done * size))
            if child is None:
                new_blocks = list(blocks[done:count])
                for block in new_blocks:
                    self.allocator.incref(block)
                key = self._key(tokens, done * size)
                leaf = _Node(key, tokens[done * size :], new_blocks, node)
                leaf.last_used = self._tick
                node.children[key] = leaf
                return len(new_blocks)
            shared = self._shared_blocks(child.tokens, tokens, done * size)
            if shared < len(child.blocks):
                child = self._split(child, shared)
            node, done = child, done + shared
        node.last_used = self._tick
        return 0

    def _leaves(self) -> list[_Node]:
        leaves, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node is not self.root:
                leaves.append(node)
        return leaves

    def _remove_if_empty(self, node: _Node) -> None:
        if node.blocks or node.children or node.parent is None:
            return
        parent = node.parent
        del parent.children[node.key]
        if parent.parent is not None and len(parent.children) == 1:
            self._merge_with_only_child(parent)

    def _merge_with_only_child(self, node: _Node) -> None:
        child = next(iter(node.children.values()))
        node.tokens += child.tokens
        node.blocks += child.blocks
        node.children = child.children
        for grandchild in node.children.values():
            grandchild.parent = node
        node.last_used = max(node.last_used, child.last_used)

    def evict(self, count: int) -> int:
        """Free up to `count` blocks, least recently used leaves first.

        Blocks another table still references are left alone, since dropping the tree's
        reference would free nothing while losing the entry. Returns the blocks freed.
        """
        freed = 0
        while freed < count:
            candidates = sorted(self._leaves(), key=lambda n: n.last_used)
            progress = False
            for leaf in candidates:
                while leaf.blocks and freed < count and self.allocator.refcount(leaf.blocks[-1]) == 1:
                    self.allocator.decref(leaf.blocks.pop())
                    leaf.tokens = leaf.tokens[: len(leaf.blocks) * self.block_tokens]
                    freed += 1
                    progress = True
                if not leaf.blocks:
                    self._remove_if_empty(leaf)
                    break
                if freed >= count:
                    break
            if not progress:
                break
        return freed

    def clear(self) -> None:
        """Drop every reference the tree holds."""
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            for block in node.blocks:
                self.allocator.decref(block)
        self.root.children = {}

    @property
    def num_blocks(self) -> int:
        total, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            total += len(node.blocks)
            stack.extend(node.children.values())
        return total

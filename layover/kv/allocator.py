"""Reference-counted allocator for fixed-size KV blocks.

The paged layout follows the idea in Kwon et al., "Efficient Memory Management for
Large Language Model Serving with PagedAttention" (SOSP 2023). The implementation
here is independent of theirs.
"""

from __future__ import annotations


class OutOfBlocks(RuntimeError):
    """The pool cannot satisfy an allocation. No state changed."""


class BlockAllocator:
    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        # Popping from the end reuses recently freed blocks first, which keeps the
        # active part of the pool small and its pages warm in the TLB.
        self._free = list(range(num_blocks - 1, -1, -1))
        self._refcounts = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def alloc(self, count: int = 1) -> list[int]:
        """Allocate blocks with a reference count of one, all or nothing."""
        if count < 0:
            raise ValueError("count must not be negative")
        if count > len(self._free):
            raise OutOfBlocks(f"need {count} blocks, {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(count)]
        for block in blocks:
            self._refcounts[block] = 1
        return blocks

    def refcount(self, block: int) -> int:
        return self._refcounts[block]

    def incref(self, block: int) -> None:
        if self._refcounts[block] <= 0:
            raise ValueError(f"block {block} is not allocated")
        self._refcounts[block] += 1

    def decref(self, block: int) -> bool:
        """Drop one reference. Returns True if the block went back to the free list."""
        if self._refcounts[block] <= 0:
            raise ValueError(f"block {block} is not allocated")
        self._refcounts[block] -= 1
        if self._refcounts[block] == 0:
            self._free.append(block)
            return True
        return False

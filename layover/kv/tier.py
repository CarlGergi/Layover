"""CPU memory tier for KV blocks of paused agents, asynchronous on top of a TransferBackend.

Blocks being offloaded stay allocated, with their references held here, until the copy
completes; only then can the pool reuse them. Blocks being restored belong to the caller at
once, who must not touch them until the ticket is done. Whether a tier stores BF16 or FP8 is
fixed when it is built, so one cost model describes it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

from layover.kv.paged_cache import BlockTable, PagedKVCache
from layover.kv.transfer import Payload, PlainCopyBackend, ReadyTicket, Ticket, TransferBackend


@dataclass(eq=False)
class CpuHandle:
    num_tokens: int
    num_blocks: int
    payload: Payload
    ticket: Ticket | None = None
    # References to pool blocks kept until the offload completes.
    held: list[int] = field(default_factory=list)
    cache: PagedKVCache | None = None
    dropped: bool = False

    @property
    def in_flight(self) -> bool:
        return self.ticket is not None


@dataclass(eq=False)
class _Restore:
    ticket: Ticket
    handle: CpuHandle
    bytes_moved: int


class CpuTier:
    def __init__(self, capacity_blocks: int, backend: TransferBackend | None = None, fp8: bool = False):
        self.capacity_blocks = capacity_blocks
        self.backend = backend or PlainCopyBackend()
        self.fp8 = fp8
        self.used_blocks = 0
        self.peak_used_blocks = 0
        self.bytes_out = 0
        self.bytes_in = 0
        self._offloading: list[CpuHandle] = []
        self._restoring: list[_Restore] = []

    @property
    def bytes_moved(self) -> int:
        return self.bytes_out + self.bytes_in

    @property
    def in_flight(self) -> int:
        return len(self._offloading) + len(self._restoring)

    def has_room(self, blocks: int) -> bool:
        return self.used_blocks + blocks <= self.capacity_blocks

    def prepare(self, cache: PagedKVCache) -> None:
        self.backend.prepare(cache, self.fp8)

    def start_offload(self, cache: PagedKVCache, table: BlockTable) -> CpuHandle:
        """Begin copying a table's blocks out. The table is empty afterwards."""
        if not self.has_room(len(table.blocks)):
            raise MemoryError("CPU tier is full")
        blocks = table.blocks
        ticket = self.backend.start_out(cache, blocks, self.fp8)
        handle = CpuHandle(table.num_tokens, len(blocks), ticket.payload, ticket, blocks, cache)
        self.used_blocks += handle.num_blocks
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        table.blocks = []
        table.num_tokens = 0
        self._offloading.append(handle)
        self.poll()
        return handle

    def start_restore(self, cache: PagedKVCache, handle: CpuHandle, blocks: int | None = None) -> tuple[BlockTable, Ticket]:
        """Begin copying a handle's first `blocks` blocks into newly allocated ones.

        The handle's memory goes back once the ticket is done. A session that only reuses
        part of its context restores a prefix: copying the rest and freeing it at once would
        hand blocks to other turns while the copy still writes to them.
        """
        if handle.in_flight:
            raise RuntimeError("the handle's offload is still in flight")
        count = handle.num_blocks if blocks is None else blocks
        if count > handle.num_blocks:
            raise ValueError(f"handle holds {handle.num_blocks} blocks, asked for {count}")
        ids = cache.allocator.alloc(count)
        ticket = self.backend.start_in(cache, ids, handle.payload) if count else ReadyTicket()
        table = BlockTable(ids, min(handle.num_tokens, count * cache.block_tokens))
        self._restoring.append(_Restore(ticket, handle, count * handle.payload.record_bytes))
        self.poll()
        return table, ticket

    def _complete_offload(self, handle: CpuHandle) -> None:
        for block in handle.held:
            handle.cache.allocator.decref(block)
        handle.held = []
        handle.ticket = None
        self.bytes_out += handle.payload.nbytes
        if handle.dropped:
            self._release(handle)

    def _release(self, handle: CpuHandle) -> None:
        self.backend.free(handle.payload)
        self.used_blocks -= handle.num_blocks

    def poll(self) -> bool:
        """Finish whatever has completed, without blocking. Returns whether anything did."""
        finished = [handle for handle in self._offloading if handle.ticket.done()]
        for handle in finished:
            self._complete_offload(handle)
        self._offloading = [handle for handle in self._offloading if handle.in_flight]
        restored = [restore for restore in self._restoring if restore.ticket.done()]
        for restore in restored:
            self.bytes_in += restore.bytes_moved
            self._release(restore.handle)
        self._restoring = [restore for restore in self._restoring if restore not in restored]
        return bool(finished or restored)

    def blocks_freeing(self, also_evictable: Collection[int] = ()) -> int:
        """Pool blocks that can return to the free list once offloads in flight complete.

        A block counts when offloads hold every reference to it, apart from at most one owner
        that gives it up under pressure, which is what also_evictable names. Offloads can share
        a block, since a fork's table starts with its parent's.
        """
        holders: dict[tuple[int, int], int] = {}
        allocators = {}
        for handle in self._offloading:
            allocators[id(handle.cache)] = handle.cache.allocator
            for block in handle.held:
                key = (id(handle.cache), block)
                holders[key] = holders.get(key, 0) + 1
        freeing = 0
        for (cache, block), held in holders.items():
            others = allocators[cache].refcount(block) - held
            if others == 0 or (others == 1 and block in also_evictable):
                freeing += 1
        return freeing

    def next_ready_at(self) -> float | None:
        """When the next modelled copy finishes, for drivers on a virtual clock."""
        tickets = [handle.ticket for handle in self._offloading] + [restore.ticket for restore in self._restoring]
        times = [ticket.ready_at for ticket in tickets if ticket.ready_at is not None]
        return min(times) if times else None

    def flush(self) -> None:
        """Block until no copy is in flight."""
        for handle in self._offloading:
            handle.ticket.wait()
        for restore in self._restoring:
            restore.ticket.wait()
        self.poll()

    def wait_for(self, handle: CpuHandle) -> None:
        if handle.in_flight:
            handle.ticket.wait()
            self.poll()

    def offload(self, cache: PagedKVCache, table: BlockTable) -> CpuHandle:
        """Copy a table's blocks out and release them from the pool, before returning."""
        handle = self.start_offload(cache, table)
        self.wait_for(handle)
        return handle

    def restore(self, cache: PagedKVCache, handle: CpuHandle) -> BlockTable:
        """Copy a handle back into new blocks and free its CPU memory, before returning."""
        self.wait_for(handle)
        table, ticket = self.start_restore(cache, handle)
        ticket.wait()
        self.poll()
        return table

    def drop(self, handle: CpuHandle) -> None:
        """Forget a handle. One whose offload is in flight frees itself when that completes."""
        if handle.in_flight:
            handle.dropped = True
        else:
            self._release(handle)

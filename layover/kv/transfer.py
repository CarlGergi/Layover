"""Moving KV blocks between the pool and CPU memory, behind tickets.

A backend starts a copy and returns a Ticket that completes later; nothing above this file
knows how the bytes move. PlainCopyBackend copies through torch and finishes before it
returns. NativeBackend uses the gather/scatter library in native/. DelayedBackend exists for
tests: it finishes after a number of polls, so asynchronous paths run on CPU. ModelledBackend
finishes when a virtual clock says so, for the simulator.

Offloaded blocks live in CPU memory as an array of equal records, so any prefix of a payload
is valid to restore. A BF16 record is the block's raw bytes; an FP8 record is described in
layover/kernels/fp8.py.
"""

from __future__ import annotations

import ctypes
import random
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from layover.kernels.fp8 import Fp8Layout, dequantize_blocks, quantize_blocks
from layover.kv.native import SLOT_FILLED, SLOT_RELEASED, Engine, HostMemoryError, NativeError, NativeUnavailable, build_host
from layover.kv.paged_cache import PagedKVCache

__all__ = [
    "DelayedBackend",
    "HostMemoryError",
    "ModelledBackend",
    "NativeBackend",
    "NativeError",
    "NativeUnavailable",
    "Payload",
    "PlainCopyBackend",
    "Ticket",
    "TransferBackend",
    "make_backend",
]


PIN_STATS = ("host_pins", "host_pin_bytes", "host_pin_ns", "host_pin_max_ns", "host_pin_failures")


class Payload:
    """Offloaded blocks in CPU memory. Contents are valid once the offload's ticket is done."""

    def __init__(self, memory: torch.Tensor, num_blocks: int, record_bytes: int, fp8: bool, pointer: int | None = None):
        self.memory: torch.Tensor | None = memory  # uint8, flat.
        self.num_blocks = num_blocks
        self.record_bytes = record_bytes
        self.fp8 = fp8
        self.pointer = pointer

    @property
    def nbytes(self) -> int:
        return self.num_blocks * self.record_bytes


class Ticket(ABC):
    """A copy in progress. done() never blocks; wait() does."""

    payload: Payload | None = None
    # When it finishes, for a modelled copy on a virtual clock. A real copy cannot know.
    ready_at: float | None = None

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def wait(self) -> None: ...


class ReadyTicket(Ticket):
    def __init__(self, payload: Payload | None = None):
        self.payload = payload

    def done(self) -> bool:
        return True

    def wait(self) -> None:
        pass


class TransferBackend(ABC):
    name = ""

    @abstractmethod
    def start_out(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        """Begin copying blocks to CPU memory; ticket.payload holds the destination.

        The caller keeps the blocks allocated and unwritten until the ticket is done.
        """

    @abstractmethod
    def start_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        """Begin copying the first len(blocks) records into `blocks`.

        The caller keeps the payload alive and the blocks unused until the ticket is done.
        """

    def free(self, payload: Payload) -> None:
        """Give back a payload's memory. Its offload and restores must be done."""

    def prepare(self, cache: PagedKVCache, fp8: bool = False) -> None:
        """Do first-use work now, such as allocating staging and compiling kernels."""

    def pin_stats(self) -> dict[str, int]:
        """Counters of host pins that stalled a step. Empty for a backend that pins nothing."""
        return {}

    def close(self) -> None:
        pass


def _record_bytes(cache: PagedKVCache, fp8: bool) -> int:
    return Fp8Layout.from_cache(cache).record_bytes if fp8 else cache.block_bytes


class PlainCopyBackend(TransferBackend):
    """One tensor per offload through torch, finished before it returns."""

    name = "plain"

    def start_out(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        ids = torch.tensor(blocks, device=cache.device, dtype=torch.long)
        if fp8:
            layout = Fp8Layout.from_cache(cache)
            stage = torch.empty(len(blocks) * layout.record_bytes, dtype=torch.uint8, device=cache.device)
            quantize_blocks(cache.pool, ids.int(), stage, layout)
            memory = stage.to("cpu")
        else:
            memory = cache.pool[ids].to("cpu", copy=True).reshape(-1).view(torch.uint8)
        return ReadyTicket(Payload(memory, len(blocks), _record_bytes(cache, fp8), fp8))

    def start_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        ids = torch.tensor(blocks, device=cache.device, dtype=torch.long)
        used = payload.memory[: len(blocks) * payload.record_bytes]
        if payload.fp8:
            dequantize_blocks(used.to(cache.device), ids.int(), cache.pool, Fp8Layout.from_cache(cache))
        else:
            shape = (len(blocks), *cache.pool.shape[1:])
            cache.pool[ids] = used.view(cache.pool.dtype).reshape(shape).to(cache.device)
        return ReadyTicket()


class _DelayedTicket(Ticket):
    def __init__(self, polls: int, action: Callable[[], None], payload: Payload | None = None):
        self.payload = payload
        self._remaining = polls
        self._action = action

    def _finish(self) -> None:
        action, self._action = self._action, None
        action()

    def done(self) -> bool:
        if self._action is None:
            return True
        if self._remaining > 0:
            self._remaining -= 1
            return False
        self._finish()
        return True

    def wait(self) -> None:
        if self._action is not None:
            self._finish()


class DelayedBackend(TransferBackend):
    """Finishes each copy after some polls, doing the work as late as it legally can.

    Reading at the end of an offload exposes blocks freed or reused early. Poisoning a
    restore's blocks at the start and writing them at the end exposes a turn that runs
    before its ticket is done: its output goes wrong instead of staying luckily correct.
    """

    name = "delayed"

    def __init__(self, polls: int | Callable[[], int] = 2, inner: TransferBackend | None = None, seed: int | None = None):
        self.inner = inner or PlainCopyBackend()
        self._rng = random.Random(seed)
        self._polls = polls

    def _next_polls(self) -> int:
        return self._polls() if callable(self._polls) else self._polls

    def start_out(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        record = _record_bytes(cache, fp8)
        memory = torch.empty(len(blocks) * record, dtype=torch.uint8)
        payload = Payload(memory, len(blocks), record, fp8)

        def finish() -> None:
            payload.memory = self.inner.start_out(cache, blocks, fp8).payload.memory

        return _DelayedTicket(self._next_polls(), finish, payload)

    def start_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        ids = torch.tensor(blocks, device=cache.device, dtype=torch.long)
        # Huge rather than NaN, because a turn with a shorter table pads with block 0, and a
        # masked NaN still poisons torch's attention. Anything that really reads this goes wrong.
        cache.pool[ids] = torch.finfo(cache.pool.dtype).max ** 0.5
        return _DelayedTicket(self._next_polls(), lambda: self.inner.start_in(cache, blocks, payload))


class _ClockTicket(Ticket):
    def __init__(self, clock: Callable[[], float], ready_at: float, payload: Payload | None = None):
        self.payload = payload
        self.ready_at = ready_at
        self._clock = clock
        self._forced = False

    def done(self) -> bool:
        return self._forced or self._clock() >= self.ready_at

    def wait(self) -> None:
        # Nothing here can sleep on a virtual clock, and drivers wait only to shut down.
        self._forced = True


class ModelledBackend(TransferBackend):
    """Copies that take modelled time on one serial copy stream, for a virtual clock.

    The bytes move at once through `inner`, but a ticket is done only when the clock reaches
    its copy's end. So blocks stay held and restoring turns stay gated exactly as with a real
    backend, and simulations exercise the same paths.
    """

    name = "modelled"

    def __init__(
        self,
        clock: Callable[[], float],
        out_s: Callable[[int], float],
        in_s: Callable[[int], float],
        inner: TransferBackend | None = None,
    ):
        self.inner = inner or PlainCopyBackend()
        self.busy_until = 0.0
        self._clock = clock
        self._out_s = out_s
        self._in_s = in_s

    def _occupy(self, seconds: float) -> float:
        self.busy_until = max(self.busy_until, self._clock()) + seconds
        return self.busy_until

    def start_out(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        payload = self.inner.start_out(cache, blocks, fp8).payload
        return _ClockTicket(self._clock, self._occupy(self._out_s(len(blocks) * cache.block_tokens)), payload)

    def start_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        self.inner.start_in(cache, blocks, payload)
        return _ClockTicket(self._clock, self._occupy(self._in_s(len(blocks) * cache.block_tokens)))

    def free(self, payload: Payload) -> None:
        self.inner.free(payload)


class _NativeTicket(Ticket):
    def __init__(self, backend: "NativeBackend", transfer: int, payload: Payload | None, keepalive: list):
        self.payload = payload
        self._backend = backend
        self._transfer = transfer
        self._keepalive = keepalive

    def _settle(self) -> None:
        self._backend.engine.release(self._transfer)
        self._transfer = None
        self._keepalive = None

    def done(self) -> bool:
        # Closing the engine waited for everything.
        if self._transfer is None or self._backend.engine.closed:
            return True
        if not self._backend.engine.poll(self._transfer):
            return False
        self._settle()
        return True

    def wait(self) -> None:
        if self._transfer is None or self._backend.engine.closed:
            return
        self._backend.engine.wait(self._transfer)
        self._settle()


class NativeBackend(TransferBackend):
    """Gather into one staging buffer, one DMA on a dedicated copy stream, events in between.

    Runs on the CUDA build for a CUDA pool and on the host build for a CPU pool. BF16 uses the
    library's gather and scatter kernels. FP8's kernels are Triton's, on a torch stream of
    their own, with the library ordering them against the copy stream through its events.
    """

    name = "native"

    def __init__(
        self,
        device: torch.device | str = "cpu",
        staging_bytes: int = 128 << 20,
        num_staging: int = 2,
        ids_capacity: int = 1 << 20,
        host_cache_bytes: int = 8 << 30,
        fp8_impl: str | None = None,
    ):
        self.device = torch.device(device)
        self.flavor = "cuda" if self.device.type == "cuda" else "host"
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"no native transfer path for device {device}")
        if self.flavor == "host":
            build_host()
        index = (self.device.index or 0) if self.flavor == "cuda" else 0
        self.engine = Engine(self.flavor, index, staging_bytes, num_staging, ids_capacity, host_cache_bytes)
        self.fp8_impl = fp8_impl
        self._next_stage = 0
        self._stages: list[torch.Tensor] = []
        self._stream = None
        self._live: list[Payload] = []

    def close(self) -> None:
        self.engine.destroy()
        for payload in self._live:
            payload.memory = None
        self._live = []
        self._stages = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def pin_stats(self) -> dict[str, int]:
        if self.engine.closed:
            return {}
        stats = self.engine.stats()
        return {name: stats[name] for name in PIN_STATS}

    # Memory

    def _host_payload(self, count: int, record: int, fp8: bool) -> Payload:
        nbytes = count * record
        pointer = self.engine.host_alloc(nbytes)
        buffer = (ctypes.c_uint8 * nbytes).from_address(pointer)
        payload = Payload(torch.frombuffer(buffer, dtype=torch.uint8), count, record, fp8, pointer)
        self._live.append(payload)
        return payload

    def free(self, payload: Payload) -> None:
        if payload.pointer is None or self.engine.closed:
            return
        self._live.remove(payload)
        self.engine.host_free(payload.pointer)
        payload.memory = None
        payload.pointer = None

    def _check_cache(self, cache: PagedKVCache, blocks: list[int] | None = None) -> None:
        if cache.device.type != self.device.type:
            raise ValueError(f"pool is on {cache.device}, this backend drives {self.device}")
        # Kernels index the pool with these unchecked, and a wild id faults the whole context.
        if blocks and not (0 <= min(blocks) and max(blocks) < cache.allocator.num_blocks):
            raise ValueError(f"block ids {min(blocks)}..{max(blocks)} are outside a pool of {cache.allocator.num_blocks}")

    def prepare(self, cache: PagedKVCache, fp8: bool = False) -> None:
        self._check_cache(cache)
        if not fp8:
            self.engine.prepare_staging()
            return
        layout = Fp8Layout.from_cache(cache)
        self._ensure_stages(layout)
        # The first launch of a Triton kernel compiles it, which must not land on a live
        # transfer. Warming up writes to a pool block, so it needs a free one, and on an
        # untouched pool of zeros the round trip leaves the block as it was.
        if cache.allocator.num_used == 0:
            ids = torch.zeros(1, dtype=torch.int32, device=cache.device)
            quantize_blocks(cache.pool, ids, self._stages[0], layout, self.fp8_impl)
            dequantize_blocks(self._stages[0], ids, cache.pool, layout, self.fp8_impl)
            self._sync_torch()

    def _sync_torch(self) -> None:
        if self.flavor == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    # BF16

    def start_out(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        self._check_cache(cache, blocks)
        # The offload's kernels run on other streams, so nothing orders them after the writes
        # torch has queued. Decode and prefill write KV eagerly on torch's stream, and the
        # scheduler can demote a session while later work is queued there, so this wait is
        # what makes the gather see those writes. It is short: a turn's own writes were
        # already flushed when its logits were read, which leaves at most one queued prefill
        # chunk. This is deliberately the current stream only: torch.cuda.synchronize() calls
        # cudaDeviceSynchronize, which would also wait for every copy in flight on our streams
        # and destroy the overlap. A restore needs no such wait: its blocks were free, and a
        # turn only frees them after its logits were read. The other stall on this path is
        # _host_payload's pin when the engine's cache has no buffer of this size, counted in
        # pin_stats.
        self._sync_torch()
        if not blocks:
            return ReadyTicket(Payload(torch.empty(0, dtype=torch.uint8), 0, _record_bytes(cache, fp8), fp8))
        if fp8:
            return self._fp8_out(cache, blocks)
        payload = self._host_payload(len(blocks), cache.block_bytes, False)
        transfer = self.engine.out_gather(cache.pool.data_ptr(), cache.block_bytes, blocks, payload.pointer)
        return _NativeTicket(self, transfer, payload, [])

    def start_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        self._check_cache(cache, blocks)
        if not blocks:
            return ReadyTicket()
        if payload.fp8:
            return self._fp8_in(cache, blocks, payload)
        transfer = self.engine.in_scatter(cache.pool.data_ptr(), cache.block_bytes, blocks, payload.pointer)
        return _NativeTicket(self, transfer, None, [])

    # The rejected design, measured by bench/transfer_gather.py. Never used by the tier.

    # With FP8 the kernels still pack into one contiguous stage, then each record gets its
    # own copy, so the comparison isolates the copies. They run on the gather path's stream for
    # the same reason: on torch's stream they would queue competing work (the interference
    # benchmark's matmul) behind them, and charge that to this path.

    def _run_kernels(self, fn: Callable[[], None]) -> None:
        """Run kernels on our own stream and wait for that stream alone."""
        self._ensure_stream()
        keepalive: list = []
        self._launch(fn, keepalive)
        finished = self.engine.ext_finish(self._stream_handle())
        self.engine.wait(finished)
        self.engine.release(finished)

    def start_out_per_block(self, cache: PagedKVCache, blocks: list[int], fp8: bool = False) -> Ticket:
        self._check_cache(cache, blocks)
        self._sync_torch()
        if fp8:
            layout = Fp8Layout.from_cache(cache)
            stage = torch.empty(len(blocks) * layout.record_bytes, dtype=torch.uint8, device=self.device)

            def kernel():
                ids = torch.tensor(blocks, dtype=torch.int32, device=self.device)
                quantize_blocks(cache.pool, ids, stage, layout, self.fp8_impl)

            self._run_kernels(kernel)
            payload = self._host_payload(len(blocks), layout.record_bytes, True)
            records = list(range(len(blocks)))
            transfer = self.engine.out_per_block(stage.data_ptr(), layout.record_bytes, records, payload.pointer)
            return _NativeTicket(self, transfer, payload, [stage])
        payload = self._host_payload(len(blocks), cache.block_bytes, False)
        transfer = self.engine.out_per_block(cache.pool.data_ptr(), cache.block_bytes, blocks, payload.pointer)
        return _NativeTicket(self, transfer, payload, [])

    def start_in_per_block(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        self._check_cache(cache, blocks)
        if payload.fp8:
            layout = Fp8Layout.from_cache(cache)
            stage = torch.empty(len(blocks) * layout.record_bytes, dtype=torch.uint8, device=self.device)
            records = list(range(len(blocks)))
            transfer = self.engine.in_per_block(stage.data_ptr(), layout.record_bytes, records, payload.pointer)
            # Only the benchmark runs this, timing the whole call, so blocking here costs nothing.
            self.engine.wait(transfer)
            self.engine.release(transfer)

            def kernel():
                ids = torch.tensor(blocks, dtype=torch.int32, device=self.device)
                dequantize_blocks(stage, ids, cache.pool, layout, self.fp8_impl)

            self._run_kernels(kernel)
            return ReadyTicket()
        transfer = self.engine.in_per_block(cache.pool.data_ptr(), cache.block_bytes, blocks, payload.pointer)
        return _NativeTicket(self, transfer, None, [])

    # FP8: the same pipeline with Triton kernels and stages owned here, because a Triton kernel
    # needs a torch tensor to write to. Everything is enqueued up front, as for BF16.

    def _ensure_stages(self, layout: Fp8Layout) -> None:
        if self.engine.staging_bytes < layout.record_bytes:
            raise ValueError(f"staging of {self.engine.staging_bytes} bytes cannot hold one {layout.record_bytes} byte record")
        if not self._stages:
            self._stages = [
                torch.empty(self.engine.staging_bytes, dtype=torch.uint8, device=self.device)
                for _ in range(self.engine.num_staging)
            ]
        self._ensure_stream()

    def _ensure_stream(self) -> None:
        if self.flavor == "cuda" and self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)

    def _stream_handle(self) -> int:
        return self._stream.cuda_stream if self.flavor == "cuda" else self.engine.host_stream()

    def _launch(self, fn: Callable[[], None], keepalive: list) -> None:
        if self.flavor == "cuda":
            with torch.cuda.stream(self._stream):
                fn()
        else:
            keepalive.append(self.engine.host_call(self.engine.host_stream(), fn))

    def _segment_ids(self, blocks: list[int], keepalive: list) -> torch.Tensor:
        """Block ids on the device, uploaded on the current stream without stalling it.

        Called inside a launched kernel, so the upload runs in stream order. Pinned memory
        keeps cudaMemcpyAsync from synchronizing, as for the ids in the native library.
        """
        ids = torch.tensor(blocks, dtype=torch.int32)
        if self.flavor != "cuda":
            return ids
        pinned = ids.pin_memory()
        uploaded = torch.empty_like(ids, device=self.device)
        uploaded.copy_(pinned, non_blocking=True)
        keepalive.extend((pinned, uploaded))
        return uploaded

    def _take_stage(self) -> int:
        stage = self._next_stage
        self._next_stage = (stage + 1) % len(self._stages)
        return stage

    def _fp8_out(self, cache: PagedKVCache, blocks: list[int]) -> Ticket:
        layout = Fp8Layout.from_cache(cache)
        self._ensure_stages(layout)
        payload = self._host_payload(len(blocks), layout.record_bytes, True)
        per_segment = self.engine.staging_bytes // layout.record_bytes
        stream = self._stream_handle()
        keepalive: list = [cache.pool]
        for start in range(0, len(blocks), per_segment):
            segment = blocks[start : start + per_segment]
            slot = self._take_stage()
            stage = self._stages[slot]

            def kernel(segment=segment, stage=stage):
                ids = self._segment_ids(segment, keepalive)
                quantize_blocks(cache.pool, ids, stage, layout, self.fp8_impl)

            self.engine.ext_wait(stream, slot, SLOT_RELEASED)
            self._launch(kernel, keepalive)
            self.engine.ext_record(stream, slot, SLOT_FILLED)
            target = payload.pointer + start * layout.record_bytes
            self.engine.ext_copy_out(stage.data_ptr(), target, len(segment) * layout.record_bytes, slot)
        return _NativeTicket(self, self.engine.ext_finish(None), payload, keepalive)

    def _fp8_in(self, cache: PagedKVCache, blocks: list[int], payload: Payload) -> Ticket:
        layout = Fp8Layout.from_cache(cache)
        self._ensure_stages(layout)
        per_segment = self.engine.staging_bytes // layout.record_bytes
        stream = self._stream_handle()
        keepalive: list = [cache.pool]
        for start in range(0, len(blocks), per_segment):
            segment = blocks[start : start + per_segment]
            slot = self._take_stage()
            stage = self._stages[slot]

            def kernel(segment=segment, stage=stage):
                ids = self._segment_ids(segment, keepalive)
                dequantize_blocks(stage, ids, cache.pool, layout, self.fp8_impl)

            source = payload.pointer + start * layout.record_bytes
            self.engine.ext_copy_in(stage.data_ptr(), source, len(segment) * layout.record_bytes, slot)
            self.engine.ext_wait(stream, slot, SLOT_FILLED)
            self._launch(kernel, keepalive)
            self.engine.ext_record(stream, slot, SLOT_RELEASED)
        return _NativeTicket(self, self.engine.ext_finish(stream), None, keepalive)


def make_backend(name: str, device: torch.device | str = "cpu", **options) -> TransferBackend:
    """Build a backend from a command-line name."""
    if name == "plain":
        return PlainCopyBackend()
    if name == "native":
        return NativeBackend(device, **options)
    raise ValueError(f"unknown transfer backend {name!r}")

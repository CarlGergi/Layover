"""Continuous batching with chunked prefill and greedy decoding.

Each step advances prefilling requests by at most prefill_budget tokens, then runs
one decode batch. Requests join and leave between steps, as in Orca (Yu et al., OSDI
2022), and long prompts are split so they do not stall running decodes, as in
Sarathi-Serve (Agrawal et al., OSDI 2024).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from layover.kv.paged_cache import BlockTable, PagedKVCache
from layover.model import Qwen3


class State(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    stop_tokens: frozenset[int] = frozenset()
    state: State = State.WAITING
    table: BlockTable = field(default_factory=BlockTable)
    generated: list[int] = field(default_factory=list)
    prefilled: int = 0
    last_logits: torch.Tensor | None = None
    on_token: Callable[["Request", int], None] | None = None
    on_finish: Callable[["Request"], None] | None = None
    # Earliest time the request may run, for example while its KV is being copied in.
    not_before: float = 0.0
    submitted_at: float = 0.0
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    @property
    def max_tokens(self) -> int:
        return len(self.prompt) + self.max_new_tokens

    @property
    def done(self) -> bool:
        if len(self.generated) >= self.max_new_tokens:
            return True
        return bool(self.generated) and self.generated[-1] in self.stop_tokens


@dataclass(frozen=True)
class EngineConfig:
    max_batch: int = 32
    prefill_chunk: int = 512
    prefill_budget: int = 512


class RequestTooLarge(ValueError):
    """The request could never fit in the KV pool."""


class Engine:
    def __init__(
        self,
        model: Qwen3,
        cache: PagedKVCache,
        config: EngineConfig = EngineConfig(),
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.cache = cache
        self.config = config
        self.clock = clock
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add_request(self, request: Request) -> None:
        if not request.prompt:
            raise ValueError("prompt must not be empty")
        if request.max_tokens > self.model.cfg.max_position_embeddings:
            raise RequestTooLarge(f"{request.request_id}: exceeds max_position_embeddings")
        if self.cache.blocks_for(request.max_tokens) > self.cache.allocator.num_blocks:
            raise RequestTooLarge(f"{request.request_id}: larger than the KV pool")
        request.submitted_at = self.clock()
        self.waiting.append(request)

    @property
    def idle(self) -> bool:
        return not self.waiting and not self.running

    def blocks_still_needed(self, request: Request) -> int:
        """Worst-case blocks the request may still allocate before it finishes.

        A table can share at most the block it writes into next, because tables only
        append; everything before it is read-only and everything after it is absent.
        """
        table = request.table
        need = max(0, self.cache.blocks_for(request.max_tokens) - len(table.blocks))
        index = table.num_tokens // self.cache.block_tokens
        if index < len(table.blocks) and self.cache.allocator.refcount(table.blocks[index]) > 1:
            need += 1
        return need

    def available_blocks(self) -> int:
        """Free blocks not spoken for by requests already running.

        Reserving worst-case growth means a running request can never run out of
        blocks, so no preemption path is needed and memory cannot deadlock.
        """
        return self.cache.allocator.num_free - sum(self.blocks_still_needed(r) for r in self.running)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.config.max_batch:
            request = self.waiting[0]
            if self.blocks_still_needed(request) > self.available_blocks():
                return
            self.waiting.popleft()
            self._start(request)

    def _start(self, request: Request) -> None:
        request.state = State.PREFILL
        request.admitted_at = self.clock()
        self.running.append(request)

    def _emit(self, request: Request, logits: torch.Tensor) -> None:
        token = int(logits.argmax(-1))
        request.last_logits = logits
        request.generated.append(token)
        if request.first_token_at is None:
            request.first_token_at = self.clock()
        if request.on_token is not None:
            request.on_token(request, token)

    def _prefill(self) -> bool:
        budget = self.config.prefill_budget
        device = self.cache.device
        worked = False
        now = self.clock()
        for request in self.running:
            if request.state is not State.PREFILL or request.not_before > now:
                continue
            while budget > 0 and request.prefilled < len(request.prompt):
                count = min(self.config.prefill_chunk, budget, len(request.prompt) - request.prefilled)
                chunk = request.prompt[request.prefilled : request.prefilled + count]
                logits = self.model.prefill(
                    self.cache, request.table, torch.tensor(chunk, device=device), request.prefilled
                )
                request.prefilled += count
                budget -= count
                worked = True
                if request.prefilled == len(request.prompt):
                    self._emit(request, logits)
                    request.state = State.DECODE
            if budget <= 0:
                break
        return worked

    def _decode(self) -> bool:
        batch = [r for r in self.running if r.state is State.DECODE and not r.done]
        if not batch:
            return False
        batch = batch[: self.config.max_batch]
        tokens = torch.tensor([r.generated[-1] for r in batch], device=self.cache.device)
        logits = self.model.decode(self.cache, [r.table for r in batch], tokens)
        for request, row in zip(batch, logits):
            self._emit(request, row)
        return True

    def _retire(self) -> bool:
        still_running = []
        retired = False
        for request in self.running:
            if request.state is State.DECODE and request.done:
                request.state = State.FINISHED
                request.finished_at = self.clock()
                self._finish(request)
                self.finished.append(request)
                if request.on_finish is not None:
                    request.on_finish(request)
                retired = True
            else:
                still_running.append(request)
        self.running = still_running
        return retired

    def _finish(self, request: Request) -> None:
        self.cache.release(request.table)

    def step(self) -> bool:
        """Advance every runnable request once. Returns whether anything happened."""
        self._admit()
        prefilled = self._prefill()
        decoded = self._decode()
        retired = self._retire()
        return prefilled or decoded or retired

    def run_until_idle(self, max_steps: int = 1_000_000) -> list[Request]:
        for _ in range(max_steps):
            if self.idle:
                return self.finished
            self.step()
        raise RuntimeError("engine did not become idle")

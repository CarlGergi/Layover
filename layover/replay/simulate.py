"""Run the real scheduler on a virtual clock, with modelled compute and transfers.

Only the model is replaced. Allocation, the radix tree, placement, and ordering are the
production code, which is what makes this useful for fuzzing. Results from here are
never benchmark numbers: compute and copy times come straight from the cost model.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import torch

from layover.agent.cost_model import CostModel
from layover.agent.scheduler import AgentConfig, AgentScheduler, AgentTurn
from layover.config import ModelConfig
from layover.engine import EngineConfig
from layover.kv.paged_cache import BlockTable, PagedKVCache
from layover.replay.results import ReplayResult, TurnRecord, summarize
from layover.replay.trace import ContextBuilder, TraceJob

SIM_VOCAB = 1024
# Keeps the pool's real memory negligible: its contents do not matter in simulation.
SIM_CONFIG = ModelConfig.micro(
    num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1, head_dim=2, vocab_size=SIM_VOCAB
)


class Deadlock(RuntimeError):
    """Work remains, nothing can run, and no future event can change that."""


class VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SimModel:
    """Stands in for Qwen3: advances the clock and produces deterministic tokens."""

    def __init__(self, cfg: ModelConfig, clock: VirtualClock, costs: CostModel, decode_step_s: float, decode_per_seq_s: float):
        self.cfg = cfg
        self.clock = clock
        self.costs = costs
        self.decode_step_s = decode_step_s
        self.decode_per_seq_s = decode_per_seq_s
        self.calls = 0

    def _token_logits(self) -> torch.Tensor:
        self.calls += 1
        logits = torch.zeros(SIM_VOCAB)
        logits[self.calls % SIM_VOCAB] = 1.0
        return logits

    def prefill(self, cache: PagedKVCache, table: BlockTable, tokens: torch.Tensor, start: int, **_) -> torch.Tensor:
        count = tokens.shape[0]
        cache.prepare_write(table, start, start + count)
        self.clock.now += max(self.costs.recompute_s(start + count) - self.costs.recompute_s(start), 0.0)
        return self._token_logits()

    def decode(self, cache: PagedKVCache, tables: list[BlockTable], tokens: torch.Tensor) -> torch.Tensor:
        for table in tables:
            cache.prepare_write(table, table.num_tokens, table.num_tokens + 1)
        self.clock.now += self.decode_step_s + self.decode_per_seq_s * len(tables)
        return torch.stack([self._token_logits() for _ in tables])


@dataclass
class _Live:
    job: TraceJob
    session_index: int
    turn: int
    context: list[int]
    session_id: str


@dataclass
class SimRun:
    result: ReplayResult
    scheduler: AgentScheduler
    waits_s: list[float] = field(default_factory=list)
    steps: int = 0


def simulate(
    jobs: list[TraceJob],
    costs: CostModel,
    pool_blocks: int,
    agent: AgentConfig = AgentConfig(),
    engine: EngineConfig = EngineConfig(max_batch=16, prefill_chunk=512, prefill_budget=512),
    decode_step_s: float = 0.01,
    decode_per_seq_s: float = 0.001,
    warmup_jobs: int = 0,
    seed: int = 0,
    max_steps: int = 2_000_000,
    system_tokens: int = 256,
) -> SimRun:
    clock = VirtualClock()
    cache = PagedKVCache(SIM_CONFIG, pool_blocks, dtype=torch.bfloat16)
    model = SimModel(SIM_CONFIG, clock, costs, decode_step_s, decode_per_seq_s)
    scheduler = AgentScheduler(model, cache, costs, engine, agent, clock=clock, charge_transfers=True)
    builder = ContextBuilder(SIM_VOCAB, seed, system_tokens)

    events: list[tuple[float, int, object]] = []
    sequence = itertools.count()

    def schedule(when: float, payload) -> None:
        heapq.heappush(events, (when, next(sequence), payload))

    arrivals = {job.job_id: job.arrival_s for job in jobs}
    finishes: dict[str, float] = {}
    open_sessions = {job.job_id: len(job.sessions) for job in jobs}
    records: list[TurnRecord] = []
    waits: list[float] = []
    in_flight: dict[str, tuple[_Live, AgentTurn]] = {}

    def submit(live: _Live) -> None:
        session = _session(live)
        spec = session.turns[live.turn]
        live.context = live.context + builder.new_tokens(session.session_id, live.turn, spec.new_tokens)
        if not live.context:
            live.context = builder.system_prompt()[:1]
        is_last = live.turn == len(session.turns) - 1
        turn = AgentTurn(
            request_id=f"{session.session_id}#{live.turn}",
            prompt=list(live.context),
            max_new_tokens=spec.output_tokens,
            session_id=session.session_id,
            program_id=live.job.job_id,
            tool_name=None if is_last else spec.tool_name,
            end_session=is_last,
        )
        scheduler.submit(turn)
        in_flight[turn.request_id] = (live, turn)

    def _session(live: _Live):
        return live.job.sessions[live.session_index]

    for job in jobs:
        main_index = job.sessions.index(job.main)
        schedule(job.arrival_s, _Live(job, main_index, 0, list(builder.system_prompt()), job.main.session_id))

    def on_finished(live: _Live, turn: AgentTurn) -> None:
        session = _session(live)
        spec = session.turns[live.turn]
        records.append(
            TurnRecord(live.job.job_id, session.session_id, live.turn, turn.submitted_at, turn.first_token_at, turn.finished_at)
        )
        waits.append(turn.admitted_at - turn.submitted_at)
        context = turn.prompt + turn.generated
        for child in live.job.children_of(session.session_id, live.turn):
            index = live.job.sessions.index(child)
            if scheduler.sessions.get(session.session_id) is not None:
                scheduler.fork_session(session.session_id, child.session_id)
            schedule(clock.now, _Live(live.job, index, 0, list(context), child.session_id))
        if live.turn == len(session.turns) - 1:
            open_sessions[live.job.job_id] -= 1
            if open_sessions[live.job.job_id] == 0:
                finishes[live.job.job_id] = clock.now
            return
        schedule(clock.now + spec.tool_s, _Live(live.job, live.session_index, live.turn + 1, list(context), session.session_id))

    steps = 0
    seen_finished = 0
    while len(finishes) < len(jobs):
        steps += 1
        if steps > max_steps:
            raise Deadlock(f"no completion after {max_steps} steps")
        while events and events[0][0] <= clock.now:
            submit(heapq.heappop(events)[2])
        worked = scheduler.step()
        for turn in scheduler.finished[seen_finished:]:
            live, _ = in_flight.pop(turn.request_id)
            on_finished(live, turn)
        seen_finished = len(scheduler.finished)
        if worked or (events and events[0][0] <= clock.now):
            continue
        upcoming = [e for e in (events[0][0] if events else None, scheduler.next_event_time()) if e is not None and e > clock.now]
        if not upcoming:
            raise Deadlock(
                f"t={clock.now}: {len(scheduler.waiting)} waiting, {len(scheduler.running)} running, "
                f"{cache.allocator.num_free} free blocks"
            )
        clock.now = min(upcoming)

    source = jobs[0].source if jobs else "empty"
    result = summarize(source, "layover-sim", arrivals, finishes, records, warmup_jobs)
    return SimRun(result, scheduler, waits, steps)

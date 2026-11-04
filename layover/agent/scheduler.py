"""Agent-aware scheduling on top of the continuous-batching engine.

Sessions pause at tool calls and their KV moves between GPU memory, the CPU tier, and
nowhere, under the placement policy. Turns are ordered by program arrival with aging,
following Autellix (Luo et al., 2025) and Continuum's program-level first-come
first-served, so a program mid-flight is not interleaved with newcomers.

Why this cannot deadlock on memory: a running turn reserves its worst-case growth, and
everything else holding blocks (radix entries and paused sessions) can be evicted. Once
running turns finish, any turn that fits in the pool at all can be admitted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from layover.agent.cost_model import CostModel, Placement, PlacementPolicy, Policy
from layover.agent.estimator import ToolDurationEstimator
from layover.engine import Engine, EngineConfig, Request, State
from layover.kv.paged_cache import BlockTable
from layover.kv.radix import RadixTree
from layover.kv.tier import CpuHandle, CpuTier


class Where(Enum):
    NONE = "none"
    ACTIVE = "active"
    KEPT = "kept"
    CPU = "cpu"


class SessionBusy(RuntimeError):
    """A session already has a turn in flight."""


@dataclass(frozen=True)
class AgentConfig:
    policy: Policy = Policy.COMBINED
    ttl_cap_s: float = 30.0
    # Seconds of priority a turn gains for each second it waits.
    aging_rate: float = 1.0
    # A program past this many served tokens loses demotion_s of priority.
    service_cap_tokens: int = 1_000_000
    demotion_s: float = 60.0
    # A blocked turn older than this stops lower-priority turns from overtaking it.
    no_backfill_age_s: float = 5.0
    cpu_capacity_blocks: int = 1 << 20
    prefix_cache: bool = True


@dataclass
class Program:
    program_id: str
    arrival: float
    attained_tokens: int = 0


@dataclass
class Session:
    session_id: str
    program_id: str
    where: Where = Where.NONE
    table: BlockTable = field(default_factory=BlockTable)
    kv_tokens: list[int] = field(default_factory=list)
    handle: CpuHandle | None = None
    expires_at: float = 0.0
    paused_at: float | None = None
    tool_name: str | None = None
    expected_resume_at: float = 0.0
    busy: bool = False


@dataclass
class AgentTurn(Request):
    session_id: str = ""
    program_id: str = ""
    tool_name: str | None = None
    expected_tool_s: float | None = None
    end_session: bool = False
    # Reads the tool name out of generated tokens when the client gave no hint.
    detect_tool: Callable[[list[int]], str | None] | None = None
    reused_tokens: int = 0


@dataclass
class SchedulerStats:
    kept: int = 0
    swapped: int = 0
    dropped: int = 0
    ttl_expired: int = 0
    early_demotions: int = 0
    prefix_hit_tokens: int = 0
    restored_tokens: int = 0
    kept_hit_tokens: int = 0
    prefilled_tokens: int = 0


class AgentScheduler(Engine):
    def __init__(
        self,
        model,
        cache,
        costs: CostModel,
        config: EngineConfig = EngineConfig(),
        agent: AgentConfig = AgentConfig(),
        clock=None,
        estimator: ToolDurationEstimator | None = None,
        charge_transfers: bool = False,
    ):
        super().__init__(model, cache, config, **({"clock": clock} if clock else {}))
        self.agent = agent
        self.costs = costs
        self.placement = PlacementPolicy(costs, agent.policy, agent.ttl_cap_s)
        self.estimator = estimator or ToolDurationEstimator()
        self.tree = RadixTree(cache.allocator, cache.block_tokens)
        self.tier = CpuTier(agent.cpu_capacity_blocks)
        self.sessions: dict[str, Session] = {}
        self.programs: dict[str, Program] = {}
        self.stats = SchedulerStats()
        # With a virtual clock, copies take modelled time on one serial copy stream.
        self.charge_transfers = charge_transfers
        self.copy_busy_until = 0.0

    # Client-facing operations

    def submit(self, turn: AgentTurn) -> None:
        now = self.clock()
        session = self.sessions.get(turn.session_id)
        if session is None:
            session = self.sessions[turn.session_id] = Session(turn.session_id, turn.program_id or turn.session_id)
        if session.busy:
            raise SessionBusy(turn.session_id)
        turn.program_id = session.program_id
        self.programs.setdefault(session.program_id, Program(session.program_id, now))
        self.add_request(turn)
        session.busy = True
        if session.paused_at is not None:
            self.estimator.observe(session.tool_name, now - session.paused_at)
            session.paused_at = None

    def fork_session(self, parent_id: str, child_id: str) -> Session:
        """Start a sub-agent sharing the parent's KV copy-on-write.

        The child joins the parent's program. A parent whose KV is not in GPU memory
        gives the child no blocks; its first turn finds what it can in the radix tree.
        """
        parent = self.sessions[parent_id]
        child = Session(child_id, parent.program_id)
        if parent.where is Where.KEPT:
            child.table = self.cache.fork(parent.table)
            child.kv_tokens = list(parent.kv_tokens)
            child.where = Where.KEPT
            child.expires_at = parent.expires_at
            child.expected_resume_at = self.clock()
        self.sessions[child_id] = child
        return child

    def close_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        if session.busy:
            raise SessionBusy(session_id)
        del self.sessions[session_id]
        self._vacate(session)

    def drain_caches(self) -> None:
        """Release all paused KV and radix entries, leaving the pool empty when idle."""
        for session in list(self.sessions.values()):
            if not session.busy:
                self._vacate(session)
        self.tree.clear()

    def _vacate(self, session: Session) -> None:
        if session.where is Where.KEPT:
            self.cache.release(session.table)
        elif session.where is Where.CPU:
            self.tier.drop(session.handle)
            session.handle = None
        session.table = BlockTable()
        session.kv_tokens = []
        session.where = Where.NONE

    # Ordering

    def effective_priority(self, turn: AgentTurn, now: float) -> float:
        """Smaller runs first: program arrival, minus aging, plus any service demotion."""
        program = self.programs[turn.program_id]
        priority = program.arrival - self.agent.aging_rate * (now - turn.submitted_at)
        if program.attained_tokens > self.agent.service_cap_tokens:
            priority += self.agent.demotion_s
        return priority

    # Memory

    def _charge_copy(self, seconds: float) -> float:
        """Occupy the copy stream; returns when this copy will be complete."""
        now = self.clock()
        if not self.charge_transfers:
            return now
        self.copy_busy_until = max(self.copy_busy_until, now) + seconds
        return self.copy_busy_until

    def _demote(self, session: Session) -> None:
        """Move a kept session to the CPU tier, or drop it when the tier is full."""
        tokens = session.table.num_tokens
        if self.tier.has_room(len(session.table.blocks)):
            session.handle = self.tier.offload(self.cache, session.table)
            session.where = Where.CPU
            session.table = BlockTable()
            self._charge_copy(self.costs.swap_out_s(tokens))
            self.stats.swapped += 1
        else:
            self.cache.release(session.table)
            session.table = BlockTable()
            session.kv_tokens = []
            session.where = Where.NONE
            self.stats.dropped += 1

    def _expire(self) -> None:
        now = self.clock()
        for session in self.sessions.values():
            if session.where is Where.KEPT and session.expires_at <= now:
                self._demote(session)
                self.stats.ttl_expired += 1

    def _make_room(self, need: int, own: Session, own_table: BlockTable) -> bool:
        """Evict until `need` blocks are free for a turn being admitted.

        Returns False, evicting nothing, if running turns alone rule it out for now.
        Order: radix entries nobody uses, then kept sessions expected back last.
        """
        pinned = sum(len(r.table.blocks) + self.blocks_still_needed(r) for r in self.running)
        if need + pinned + len(own_table.blocks) > self.cache.allocator.num_blocks:
            return False
        short = need - self.available_blocks()
        if short <= 0:
            return True
        short -= self.tree.evict(short)
        kept = [s for s in self.sessions.values() if s.where is Where.KEPT and s is not own]
        kept.sort(key=lambda s: -s.expected_resume_at)
        for session in kept:
            if short <= 0:
                break
            before = self.cache.allocator.num_free
            self._demote(session)
            self.stats.early_demotions += 1
            short -= self.cache.allocator.num_free - before
            short -= self.tree.evict(max(short, 0))
        return short <= 0

    # Admission

    def _plan_table(self, turn: AgentTurn, session: Session) -> tuple[BlockTable, int]:
        """Find reusable KV for a turn: returns (table, tokens already cached).

        At least the last prompt token is always run, because its logits are needed.
        """
        limit = len(turn.prompt) - 1
        if session.where is Where.KEPT:
            common = _common_prefix(session.kv_tokens, turn.prompt, limit)
            return session.table, common
        if not self.agent.prefix_cache:
            return BlockTable(), 0
        matched = self.tree.match(turn.prompt[:limit])
        table = BlockTable(list(matched), len(matched) * self.cache.block_tokens)
        for block in matched:
            self.cache.allocator.incref(block)
        return table, table.num_tokens

    def _try_admit(self, turn: AgentTurn) -> bool:
        session = self.sessions[turn.session_id]
        blocks_total = self.cache.blocks_for(turn.max_tokens)
        now = self.clock()

        if session.where is Where.CPU:
            restore_blocks = max(blocks_total, session.handle.num_blocks)
            if not self._make_room(restore_blocks, session, BlockTable()):
                return False
            tokens = session.handle.num_tokens
            table = self.tier.restore(self.cache, session.handle)
            session.handle = None
            common = _common_prefix(session.kv_tokens, turn.prompt, len(turn.prompt) - 1)
            self.cache.truncate(table, common)
            turn.not_before = self._charge_copy(self.costs.swap_in_s(tokens))
            self.stats.restored_tokens += common
        else:
            table, common = self._plan_table(turn, session)
            kept = session.where is Where.KEPT
            if kept:
                self.cache.truncate(table, common)
                # Admission may still fail below, so the session must stay consistent.
                session.kv_tokens = session.kv_tokens[:common]
            turn.table = table
            need = self.blocks_still_needed(turn)
            if not self._make_room(need, session, table):
                if not kept:
                    self.cache.release(table)
                turn.table = BlockTable()
                return False
            if kept:
                self.stats.kept_hit_tokens += common
            else:
                self.stats.prefix_hit_tokens += common

        turn.table = table
        table.num_tokens = common
        turn.prefilled = common
        turn.reused_tokens = common
        self.stats.prefilled_tokens += len(turn.prompt) - common
        session.table = BlockTable()
        session.where = Where.ACTIVE
        session.kv_tokens = []
        turn.admitted_at = now
        self._start(turn)
        return True

    def _admit(self) -> None:
        self._expire()
        if not self.waiting:
            return
        now = self.clock()
        order = sorted(self.waiting, key=lambda t: (self.effective_priority(t, now), t.submitted_at, t.request_id))
        for turn in order:
            if len(self.running) >= self.config.max_batch:
                return
            if self._try_admit(turn):
                self.waiting.remove(turn)
            elif now - turn.submitted_at >= self.agent.no_backfill_age_s:
                return

    # Turn end

    def _finish(self, turn: AgentTurn) -> None:
        session = self.sessions[turn.session_id]
        program = self.programs[turn.program_id]
        now = self.clock()
        # The last generated token has not been fed through the model, so it has no KV.
        kv_tokens = (turn.prompt + turn.generated)[: turn.table.num_tokens]
        program.attained_tokens += len(turn.prompt) - turn.reused_tokens + len(turn.generated)
        session.busy = False
        if self.agent.prefix_cache:
            self.tree.insert(kv_tokens, turn.table.blocks)

        tool = turn.tool_name
        if tool is None and turn.detect_tool is not None:
            tool = turn.detect_tool(turn.generated)
        if turn.end_session:
            self.cache.release(turn.table)
            self.sessions.pop(turn.session_id, None)
            return

        estimate = self.estimator.estimate(tool, turn.expected_tool_s)
        blocks = len(turn.table.blocks)
        decision = self.placement.decide(
            len(kv_tokens),
            estimate,
            blocks / self.cache.allocator.num_blocks,
            len(self.waiting),
            self.tier.has_room(blocks),
        )
        session.tool_name = tool
        session.paused_at = now
        session.expected_resume_at = now + estimate.mean_s
        session.kv_tokens = kv_tokens
        session.table = turn.table
        turn.table = BlockTable()
        if decision.placement is Placement.KEEP:
            session.where = Where.KEPT
            session.expires_at = now + decision.ttl_s
            self.stats.kept += 1
        elif decision.placement is Placement.SWAP:
            self._demote(session)
        else:
            self.cache.release(session.table)
            session.table = BlockTable()
            session.kv_tokens = []
            session.where = Where.NONE
            self.stats.dropped += 1

    # Used by drivers to sleep until something can happen.

    def next_event_time(self) -> float | None:
        times = [s.expires_at for s in self.sessions.values() if s.where is Where.KEPT]
        times += [r.not_before for r in self.running if r.not_before > self.clock()]
        return min(times) if times else None

    def can_progress(self) -> bool:
        return any(r.state is not State.FINISHED for r in self.running) or bool(self.waiting)


def _common_prefix(a: list[int], b: list[int], limit: int) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y or count >= limit:
            break
        count += 1
    return count

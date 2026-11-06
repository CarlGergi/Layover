"""Agent-aware scheduling on top of the continuous-batching engine.

Sessions pause at tool calls and their KV moves between GPU memory, the CPU tier, and
nowhere, under the placement policy. Turns are ordered by program arrival with aging,
following Autellix (Luo et al., 2025) and Continuum's program-level first-come
first-served, so a program mid-flight is not interleaved with newcomers.

Why this cannot deadlock on memory: a running turn reserves its worst-case growth, and
everything else holding blocks can be released without the turn being admitted: radix
entries and paused sessions are evicted, and an offload in flight finishes by itself
because a copy depends only on work already enqueued. Once running turns finish, any turn
that fits in the pool at all can be admitted.

Copies are asynchronous: a step never waits for one. Starting an offload can still stall the
step, when the transfer library has no pinned buffer of that size cached and must pin one;
SchedulerStats counts those stalls, and a pin that fails drops the session's KV instead. A turn
being restored holds its slot and its blocks but does not run until its ticket is done.
Waiting stays bounded because copies share one FIFO copy stream and every session has at most
one offload and one restore outstanding, so the backlog ahead of any copy is at most one round
trip per session. With charge_transfers on a virtual clock, ModelledBackend gives copies that
same backlog in modelled time, so the simulator runs these paths with blocks really held
during copies.

Discard with the prefix cache on gives up the session's references at the tool call. Every whole
block stays in the radix tree holding only the tree's reference: reusable if the session returns
first, evicted when memory is needed, least recently used path first and tail first. vLLM 0.33.0
(commit g4ba787ec9) treats freed blocks the same way, in vllm/v1/core/block_pool.py, kv_cache_utils.py
and single_type_kv_cache_manager.py, so this ablation isolates placement, not a stricter cache.
Discard with the prefix cache off keeps nothing and is the true recompute lower bound.
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
from layover.kv.transfer import HostMemoryError, ModelledBackend, TransferBackend


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
    # Bytes of finished copies, as they crossed PCIe: FP8 counts its smaller records.
    bytes_offloaded: int = 0
    bytes_restored: int = 0
    # Times a turn could not be admitted only because an offload's blocks were still copying.
    held_by_copies: int = 0
    # Resumes of swapped sessions the radix tree served, skipping a copy back from CPU.
    restores_avoided: int = 0
    # Sessions dropped, and counted in `dropped` too, because host memory could not be pinned.
    pin_failure_drops: int = 0
    # Stalls of a step while the transfer library pinned host memory it had not cached.
    host_pins: int = 0
    host_pin_bytes: int = 0
    host_pin_ns: int = 0
    host_pin_max_ns: int = 0


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
        transfer: TransferBackend | None = None,
        offload_fp8: bool = False,
    ):
        super().__init__(model, cache, config, **({"clock": clock} if clock else {}))
        self.agent = agent
        # Placement prices the swap the tier will really do.
        self.costs = costs.for_offload(offload_fp8)
        self.placement = PlacementPolicy(self.costs, agent.policy, agent.ttl_cap_s)
        self.estimator = estimator or ToolDurationEstimator()
        self.tree = RadixTree(cache.allocator, cache.block_tokens)
        if charge_transfers:
            if transfer is not None:
                raise ValueError("charge_transfers models the copies, so it takes no transfer backend")
            transfer = ModelledBackend(self.clock, self.costs.swap_out_s, self.costs.swap_in_s)
        self.tier = CpuTier(agent.cpu_capacity_blocks, transfer, offload_fp8)
        self.tier.prepare(cache)
        self.sessions: dict[str, Session] = {}
        self.programs: dict[str, Program] = {}
        self._stats = SchedulerStats()

    @property
    def stats(self) -> SchedulerStats:
        self._stats.bytes_offloaded = self.tier.bytes_out
        self._stats.bytes_restored = self.tier.bytes_in
        pins = self.tier.backend.pin_stats()
        if pins:
            self._stats.host_pins = pins["host_pins"]
            self._stats.host_pin_bytes = pins["host_pin_bytes"]
            self._stats.host_pin_ns = pins["host_pin_ns"]
            self._stats.host_pin_max_ns = pins["host_pin_max_ns"]
        return self._stats

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
        self.tier.flush()

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

    def _poll_transfers(self) -> bool:
        return self.tier.poll()

    def _demote(self, session: Session) -> None:
        """Move a kept session to the CPU tier, or drop it when that cannot hold it."""
        if self.tier.has_room(len(session.table.blocks)):
            try:
                session.handle = self.tier.start_offload(self.cache, session.table)
            except HostMemoryError:
                # Nothing was enqueued and the table is intact. This costs one session a
                # recompute; the engine carries on, and /metrics shows why the numbers moved.
                self.stats.pin_failure_drops += 1
                self._drop(session)
                return
            session.where = Where.CPU
            session.table = BlockTable()
            self.stats.swapped += 1
        else:
            self._drop(session)

    def _drop(self, session: Session) -> None:
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

        An offload does not free blocks until its copy completes, so demoting can leave this
        short: it then returns False and the turn retries on later steps, without blocking.
        Blocks already on their way back count against the shortfall, or every retry during a
        long copy would demote one more session for nothing. That includes blocks the radix
        tree also holds, which the retry evicts once the copy has let go of them. Counting too
        many cannot wedge a turn: with no copy in flight nothing counts, and demotion resumes.
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
        # Evicting cannot change this set for blocks an offload holds: their refcount is above 1.
        in_tree = self.tree.block_ids()
        for session in kept:
            if short <= self.tier.blocks_freeing(in_tree):
                break
            before = self.cache.allocator.num_free
            self._demote(session)
            self.stats.early_demotions += 1
            short -= self.cache.allocator.num_free - before
            short -= self.tree.evict(max(short, 0))
        if short > 0 and self.tier.blocks_freeing(in_tree) > 0:
            self.stats.held_by_copies += 1
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

    def _tree_covers_restore(self, turn: AgentTurn, session: Session) -> tuple[BlockTable, int] | None:
        """The radix tree's table for a swapped session, if it makes restoring pointless.

        A swap leaves the tree's references alone, so with no memory pressure the tree still has
        the exact bytes in the pool, while a restore would copy a second, possibly lossy, version
        across PCIe. It covers the session when it holds every whole block the turn can reuse;
        the few tokens of a partial last block are cheaper to prefill than to copy.

        The returned table holds references, so eviction cannot take its blocks.
        """
        if not self.agent.prefix_cache:
            return None
        table, tokens = self._plan_table(turn, session)
        size = self.cache.block_tokens
        if tokens // size >= _common_prefix(session.kv_tokens, turn.prompt, len(turn.prompt) - 1) // size:
            return table, tokens
        self.cache.release(table)
        return None

    def _try_admit(self, turn: AgentTurn) -> bool:
        session = self.sessions[turn.session_id]
        blocks_total = self.cache.blocks_for(turn.max_tokens)
        now = self.clock()

        swapped = session.where is Where.CPU
        # Its own offload is still copying: the ticket finishes it, and a later step admits the turn.
        if swapped and session.handle.in_flight:
            return False
        from_tree = self._tree_covers_restore(turn, session) if swapped else None
        if swapped and from_tree is None:
            if not self._make_room(blocks_total, session, BlockTable()):
                return False
            common = _common_prefix(session.kv_tokens, turn.prompt, len(turn.prompt) - 1)
            table, turn.transfer = self.tier.start_restore(self.cache, session.handle, self.cache.blocks_for(common))
            session.handle = None
            self.stats.restored_tokens += common
        else:
            table, common = from_tree or self._plan_table(turn, session)
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
            if swapped:
                self.tier.drop(session.handle)
                session.handle = None
                self.stats.restores_avoided += 1

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
            self._drop(session)

    # Used by drivers to sleep until something can happen.

    def next_event_time(self) -> float | None:
        times = [s.expires_at for s in self.sessions.values() if s.where is Where.KEPT]
        ready = self.tier.next_ready_at()
        if ready is not None:
            times.append(ready)
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

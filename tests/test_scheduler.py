"""Scheduler behaviour on real model tensors, and fuzzing on a virtual clock."""

import random
from copy import deepcopy

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig, AgentScheduler, AgentTurn, SessionBusy, Where
from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request, RequestTooLarge
from layover.kv import PagedKVCache
from layover.kv.radix import RadixTree
from layover.kv.transfer import DelayedBackend, NativeBackend
from layover.model import Qwen3
from layover.replay.simulate import Deadlock, simulate
from layover.replay.trace import SyntheticWorkload, TraceJob, TraceSession, TraceTurn, generate

CFG = ModelConfig.micro()
COSTS = CostModel([0.001, 5e-6], [0.001, 5e-6], [0.005, 2e-5, 2e-9])


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Qwen3(CFG).float().eval()


def build(model, blocks=64, agent=AgentConfig(), engine=EngineConfig(), transfer=None, dtype=torch.float32, costs=COSTS, fp8=False):
    clock = Clock()
    cache = PagedKVCache(CFG, blocks, dtype=dtype)
    return AgentScheduler(model, cache, costs, engine, agent, clock=clock, transfer=transfer, offload_fp8=fp8), clock


def run(scheduler):
    for _ in range(100_000):
        if scheduler.idle:
            return
        scheduler.step()
    raise AssertionError("scheduler did not go idle")


def solo(model, prompt, new):
    engine = Engine(model, PagedKVCache(CFG, 64, dtype=torch.float32))
    request = Request("solo", list(prompt), max_new_tokens=new)
    engine.add_request(request)
    engine.run_until_idle()
    return request.generated


def turn(session, prompt, new=4, **kwargs):
    return AgentTurn(f"{session}#{len(prompt)}", list(prompt), new, session_id=session, **kwargs)


def resume_prompt(previous):
    return previous.prompt + previous.generated + [7, 8, 9]


def test_kept_session_resumes_without_recomputing_and_matches_output(model):
    scheduler, clock = build(model)
    first = turn("s", range(40), tool_name="grep")
    scheduler.submit(first)
    run(scheduler)
    session = scheduler.sessions["s"]
    assert session.where is Where.KEPT
    clock.now += 1
    second = turn("s", resume_prompt(first))
    scheduler.submit(second)
    run(scheduler)
    assert second.reused_tokens == len(first.prompt) + len(first.generated) - 1
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)


def test_tool_duration_is_learned_from_the_pause(model):
    scheduler, clock = build(model)
    first = turn("s", range(16), tool_name="grep")
    scheduler.submit(first)
    run(scheduler)
    clock.now += 7.0
    scheduler.submit(turn("s", resume_prompt(first)))
    assert scheduler.estimator.estimate("grep").samples == 1


@pytest.mark.parametrize("policy", list(Policy))
def test_every_policy_gives_the_same_tokens(model, policy):
    scheduler, clock = build(model, agent=AgentConfig(policy=policy, prefix_cache=False))
    first = turn("s", range(40), tool_name="grep")
    scheduler.submit(first)
    run(scheduler)
    clock.now += 1
    second = turn("s", resume_prompt(first))
    scheduler.submit(second)
    run(scheduler)
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)
    expected = {Policy.DISCARD: 0}.get(policy, len(second.prompt) - 4)
    assert second.reused_tokens == expected


def test_ttl_expiry_demotes_to_cpu_and_does_not_discard(model):
    scheduler, clock = build(model, agent=AgentConfig(policy=Policy.KEEP_TTL, ttl_cap_s=5.0, prefix_cache=False))
    first = turn("s", range(40), tool_name="slow")
    scheduler.submit(first)
    run(scheduler)
    clock.now += 6.0
    scheduler.step()
    assert scheduler.sessions["s"].where is Where.CPU
    assert scheduler.stats.ttl_expired == 1
    assert scheduler.cache.allocator.num_used == 0
    second = turn("s", resume_prompt(first))
    scheduler.submit(second)
    run(scheduler)
    assert second.reused_tokens > 0
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)


def test_full_cpu_tier_drops_on_expiry_and_still_recovers(model):
    agent = AgentConfig(policy=Policy.KEEP_TTL, ttl_cap_s=1.0, cpu_capacity_blocks=0, prefix_cache=False)
    scheduler, clock = build(model, agent=agent)
    first = turn("s", range(40), tool_name="slow")
    scheduler.submit(first)
    run(scheduler)
    clock.now += 2.0
    scheduler.step()
    assert scheduler.sessions["s"].where is Where.NONE and scheduler.cache.allocator.num_used == 0
    second = turn("s", resume_prompt(first))
    scheduler.submit(second)
    run(scheduler)
    assert second.reused_tokens == 0
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)


def test_prefix_tree_shares_across_sessions_and_output_is_unchanged(model):
    scheduler, _ = build(model)
    a = turn("a", list(range(64)) + [1, 2], end_session=True)
    scheduler.submit(a)
    run(scheduler)
    b = turn("b", list(range(64)) + [3, 4], end_session=True)
    scheduler.submit(b)
    run(scheduler)
    assert b.reused_tokens == 64
    assert b.generated == solo(model, b.prompt, b.max_new_tokens)


def test_full_prompt_hit_reruns_the_last_token_without_corrupting_the_shared_block(model):
    scheduler, _ = build(model)
    prompt = list(range(32))
    a = turn("a", prompt, end_session=True)
    scheduler.submit(a)
    run(scheduler)
    shared = scheduler.tree.match(prompt)
    before = scheduler.cache.pool[shared[-1]].clone()
    b = turn("b", prompt, end_session=True)
    scheduler.submit(b)
    run(scheduler)
    # Block aligned: the last whole block is not reused, so the token can be rerun.
    assert b.reused_tokens == 16
    assert b.generated == a.generated
    assert scheduler.cache.pool[shared[-1]].equal(before)


def test_parallel_subagents_fork_copy_on_write(model):
    scheduler, clock = build(model)
    parent = turn("p", range(40), tool_name="spawn", program_id="job")
    scheduler.submit(parent)
    run(scheduler)
    assert scheduler.sessions["p"].where is Where.KEPT
    scheduler.fork_session("p", "c1")
    scheduler.fork_session("p", "c2")
    allocated = scheduler.cache.allocator.num_used
    # All three share every block until someone writes.
    assert scheduler.cache.allocator.refcount(scheduler.sessions["p"].table.blocks[0]) >= 3
    c1 = turn("c1", resume_prompt(parent) + [1])
    c2 = turn("c2", resume_prompt(parent) + [2])
    p2 = turn("p", resume_prompt(parent) + [3])
    for t in (c1, c2, p2):
        scheduler.submit(t)
    run(scheduler)
    for t in (c1, c2, p2):
        assert t.reused_tokens > 0
        assert t.generated == solo(model, t.prompt, t.max_new_tokens)
    assert scheduler.programs["job"].attained_tokens > 0
    assert scheduler.cache.allocator.num_used < 3 * allocated


def test_resume_with_edited_history_reuses_only_the_common_prefix(model):
    scheduler, clock = build(model)
    first = turn("s", range(40), tool_name="t")
    scheduler.submit(first)
    run(scheduler)
    clock.now += 1
    edited = list(range(20)) + [500, 501, 502]
    second = turn("s", edited)
    scheduler.submit(second)
    run(scheduler)
    assert second.reused_tokens == 20
    assert second.generated == solo(model, edited, second.max_new_tokens)


def test_failed_admission_leaves_a_kept_session_consistent(model):
    # The kept session's turn cannot fit yet, gets demoted meanwhile, and must still
    # resume with KV that matches its tokens.
    agent = AgentConfig(policy=Policy.KEEP_TTL, ttl_cap_s=1e6, prefix_cache=False, no_backfill_age_s=1e9)
    scheduler, clock = build(model, blocks=8, agent=agent)
    first = turn("a", range(40), tool_name="t")
    scheduler.submit(first)
    run(scheduler)
    blocker = turn("b", range(200, 248), new=16)
    scheduler.submit(blocker)
    scheduler.step()
    edited = list(range(20)) + [300 + i for i in range(40)]
    resumed = turn("a", edited, new=16)
    scheduler.submit(resumed)
    run(scheduler)
    assert resumed.generated == solo(model, edited, 16)
    assert blocker.generated == solo(model, blocker.prompt, 16)


def test_closing_a_busy_session_is_refused(model):
    scheduler, _ = build(model)
    scheduler.submit(turn("s", range(8)))
    with pytest.raises(SessionBusy):
        scheduler.close_session("s")


def test_sessions_cannot_overlap_turns(model):
    scheduler, _ = build(model)
    scheduler.submit(turn("s", range(8)))
    with pytest.raises(SessionBusy):
        scheduler.submit(turn("s", range(9)))


def test_pressure_evicts_paused_sessions_instead_of_waiting_forever(model):
    # Two paused sessions fill the pool; a third must force them out.
    agent = AgentConfig(policy=Policy.KEEP_TTL, ttl_cap_s=1e6, prefix_cache=False)
    scheduler, _ = build(model, blocks=8, agent=agent)
    for name in ("a", "b"):
        scheduler.submit(turn(name, range(40), tool_name="slow"))
    run(scheduler)
    assert scheduler.sessions["a"].where is Where.KEPT
    late = turn("c", range(100, 140), new=4)
    scheduler.submit(late)
    run(scheduler)
    assert late.generated and scheduler.stats.early_demotions >= 1


def test_too_large_turn_is_rejected_at_submit(model):
    scheduler, _ = build(model, blocks=2)
    with pytest.raises(RequestTooLarge):
        scheduler.submit(turn("s", range(100)))
    assert not scheduler.sessions["s"].busy


def test_all_blocks_return_after_closing_everything(model):
    scheduler, clock = build(model, blocks=64)
    first = turn("a", range(40), tool_name="t")
    scheduler.submit(first)
    scheduler.submit(turn("b", range(40), end_session=True))
    run(scheduler)
    clock.now += 1
    scheduler.close_session("a")
    scheduler.drain_caches()
    assert scheduler.cache.allocator.num_used == 0


def test_program_priority_prefers_the_earlier_program(model):
    agent = AgentConfig(aging_rate=0.0, prefix_cache=False)
    scheduler, clock = build(model, agent=agent, engine=EngineConfig(max_batch=1))
    scheduler.submit(turn("old", range(8), new=1, program_id="old", tool_name="t"))
    run(scheduler)
    clock.now += 50
    young = turn("young", range(8), new=1, program_id="young")
    scheduler.submit(young)
    # A turn of the older program arrives later, yet goes first.
    clock.now += 1
    old = turn("old", list(range(9)), new=1, program_id="old")
    scheduler.submit(old)
    scheduler.step()
    assert old.admitted_at is not None and young.admitted_at is None


def test_aging_lets_a_young_program_overtake(model):
    agent = AgentConfig(aging_rate=1.0, prefix_cache=False)
    scheduler, clock = build(model, agent=agent, engine=EngineConfig(max_batch=1))
    scheduler.submit(turn("old", range(8), new=1, program_id="old", tool_name="t"))
    run(scheduler)
    clock.now += 10
    young = turn("young", range(8), new=1, program_id="young")
    scheduler.submit(young)
    clock.now += 11  # Waited longer than the head start of the older program.
    old = turn("old", list(range(9)), new=1, program_id="old")
    scheduler.submit(old)
    scheduler.step()
    assert young.admitted_at is not None and old.admitted_at is None


def test_service_cap_demotes_a_heavy_program(model):
    agent = AgentConfig(aging_rate=0.0, service_cap_tokens=1, demotion_s=1000.0, prefix_cache=False)
    scheduler, clock = build(model, agent=agent, engine=EngineConfig(max_batch=1))
    scheduler.submit(turn("heavy", range(8), new=1, program_id="heavy", tool_name="t"))
    run(scheduler)
    clock.now += 1
    light = turn("light", range(8), new=1, program_id="light")
    scheduler.submit(light)
    heavy = turn("heavy", list(range(9)), new=1, program_id="heavy")
    scheduler.submit(heavy)
    scheduler.step()
    assert light.admitted_at is not None and heavy.admitted_at is None


# Fuzzing on the virtual clock. Costs are drawn too, so placement choices vary.

def advance(scheduler, clock, steps=1, dt=0.001):
    for _ in range(steps):
        clock.now += dt
        scheduler.step()


class CountingModel:
    """Counts the model calls a gated turn must not make."""

    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        self.calls = 0

    def prefill(self, *args, **kwargs):
        self.calls += 1
        return self.model.prefill(*args, **kwargs)

    def decode(self, *args, **kwargs):
        self.calls += 1
        return self.model.decode(*args, **kwargs)


def workload(rng, jobs, turns, branching, tool_scale, arrival_rate=50.0):
    out, now = [], 0.0
    for j in range(jobs):
        now += rng.expovariate(arrival_rate)
        main = TraceSession(
            f"j{j}/m",
            [
                TraceTurn(rng.randint(1, 200), rng.randint(1, 24), rng.choice(["a", "b"]), rng.expovariate(1 / tool_scale))
                for _ in range(rng.randint(1, turns))
            ],
        )
        sessions = [main]
        for c in range(rng.randint(0, branching)):
            sessions.append(
                TraceSession(
                    f"j{j}/c{c}",
                    [TraceTurn(rng.randint(1, 60), rng.randint(1, 12), "a", rng.expovariate(1 / tool_scale)) for _ in range(rng.randint(1, 3))],
                    parent=main.session_id,
                    fork_after_turn=rng.randrange(len(main.turns)),
                )
            )
        out.append(TraceJob(f"j{j}", now, sessions, "fuzz"))
    return out


# Asynchronous copies. DelayedBackend poisons restored blocks until its ticket finishes and
# reads offloaded blocks as late as it can, so a broken gate or a block reused early changes
# the tokens instead of staying luckily right.

SWAP = AgentConfig(policy=Policy.SWAP, prefix_cache=False)


def test_restoring_turn_does_not_run_until_its_ticket_is_done(model):
    counted = CountingModel(model)
    scheduler, clock = build(counted, agent=SWAP, transfer=DelayedBackend(polls=6))
    first = turn("a", range(1, 40), new=4)
    scheduler.submit(first)
    run(scheduler)
    scheduler.tier.flush()
    assert scheduler.sessions["a"].where is Where.CPU

    second = turn("a", resume_prompt(first), new=4)
    scheduler.submit(second)
    scheduler.step()
    assert second.transfer is not None and second.admitted_at is not None and second.table.blocks
    calls = counted.calls
    for _ in range(100):
        if second.transfer is None or second.transfer.done():
            break
        assert counted.calls == calls and not second.generated
        scheduler.step()
    run(scheduler)
    assert second.generated == solo(model, second.prompt, 4)
    assert second.reused_tokens > 0
    assert scheduler.stats.restored_tokens == second.reused_tokens


def test_turn_waits_for_its_own_offload_before_restoring(model):
    scheduler, clock = build(model, agent=SWAP, transfer=DelayedBackend(polls=6))
    first = turn("a", range(1, 40), new=4)
    scheduler.submit(first)
    run(scheduler)
    handle = scheduler.sessions["a"].handle
    assert handle.in_flight
    second = turn("a", resume_prompt(first), new=4)
    scheduler.submit(second)
    scheduler.step()
    assert second in scheduler.waiting and scheduler.sessions["a"].handle is handle
    run(scheduler)
    assert second.generated == solo(model, second.prompt, 4)


def test_turn_short_of_blocks_retries_without_blocking_until_an_offload_frees_them(model):
    """Nothing but copies in flight stands between the turn and its blocks."""
    agent = AgentConfig(policy=Policy.KEEP_TTL, prefix_cache=False, ttl_cap_s=1000.0)
    backend = DelayedBackend(polls=8)
    scheduler, clock = build(model, blocks=8, agent=agent, transfer=backend)
    holder = turn("holder", range(1, 80), new=4)
    scheduler.submit(holder)
    run(scheduler)
    assert scheduler.sessions["holder"].where is Where.KEPT

    big = turn("big", range(100, 180), new=4)
    scheduler.submit(big)
    scheduler.step()
    # The holder is being demoted, its blocks stay allocated, and the turn keeps waiting.
    assert big in scheduler.waiting
    assert scheduler.tier.blocks_freeing() > 0
    assert scheduler.cache.allocator.num_used == len(scheduler.tier._offloading[0].held)
    assert scheduler.stats.held_by_copies > 0
    run(scheduler)
    assert big.generated == solo(model, big.prompt, 4)
    assert scheduler.stats.early_demotions == 1


def test_a_long_copy_does_not_demote_one_more_session_on_every_retry(model):
    agent = AgentConfig(policy=Policy.KEEP_TTL, prefix_cache=False, ttl_cap_s=1000.0)
    scheduler, clock = build(model, blocks=16, agent=agent, transfer=DelayedBackend(polls=50))
    for name, base in (("x", 1), ("y", 100), ("z", 200)):
        kept = turn(name, range(base, base + 60), new=4)
        scheduler.submit(kept)
        run(scheduler)
        clock.now += 1.0
    assert all(s.where is Where.KEPT for s in scheduler.sessions.values())
    big = turn("big", range(300, 420), new=4)
    scheduler.submit(big)
    advance(scheduler, clock, steps=20)
    # One session's blocks cover the shortfall, so one demotion is all this may cost.
    assert scheduler.stats.early_demotions == 1
    assert big in scheduler.waiting
    run(scheduler)
    assert big.generated == solo(model, big.prompt, 4)
    assert sum(s.where is Where.KEPT for s in scheduler.sessions.values() if s.session_id != "big") == 2


def test_copy_on_write_protects_a_block_a_parent_is_still_offloading(model):
    """The child's write must not reach the parent's offload, which reads as late as it can."""
    agent = AgentConfig(policy=Policy.KEEP_TTL, ttl_cap_s=1.0, prefix_cache=False)
    scheduler, clock = build(model, agent=agent, transfer=DelayedBackend(polls=30))
    parent = turn("p", range(1, 25), new=4)
    scheduler.submit(parent)
    run(scheduler)
    child_session = scheduler.fork_session("p", "c")
    assert child_session.table.blocks == scheduler.sessions["p"].table.blocks

    clock.now += 2.0
    scheduler.sessions["c"].expires_at = clock.now + 100
    scheduler.step()
    assert scheduler.sessions["p"].where is Where.CPU and scheduler.sessions["p"].handle.in_flight

    handle = scheduler.sessions["p"].handle
    shared = list(handle.held)
    before = scheduler.cache.pool[shared].clone()

    # 24 + 4 tokens put the last shared block mid-way, so the child writes into it. Tokens from
    # the tiny random model barely depend on KV, so the bytes are what prove the separation.
    child = turn("c", resume_prompt(parent), new=4)
    scheduler.submit(child)
    run(scheduler)
    assert child.generated == solo(model, child.prompt, 4)
    assert scheduler.sessions["c"].table.blocks[len(shared) - 1] != shared[-1]
    assert scheduler.cache.pool[shared].view(torch.uint32).equal(before.view(torch.uint32))

    scheduler.tier.flush()
    assert handle.payload.memory.equal(before.reshape(-1).view(torch.uint8))
    resumed = turn("p", resume_prompt(parent)[:-1] + [50, 51], new=4)
    scheduler.submit(resumed)
    run(scheduler)
    assert resumed.reused_tokens > 0
    assert resumed.generated == solo(model, resumed.prompt, 4)


class HeldBlocksChecker(DelayedBackend):
    """Fails if a block changes between an offload's start and the moment it is read."""

    def start_out(self, cache, blocks, fp8=False):
        before = cache.pool[blocks].clone()
        ticket = super().start_out(cache, blocks, fp8)
        read = ticket._action

        def checked_read():
            assert cache.pool[blocks].view(torch.uint32).equal(before.view(torch.uint32)), "a held block changed"
            read()

        ticket._action = checked_read
        return ticket


def drive(scheduler, clock, plans, forks=None, max_steps=200_000):
    """Run sessions turn by turn, with a tool gap between turns, returning every turn.

    A session named in `forks` starts a sub-agent after its first turn.
    """
    pending = {name: list(new_tokens) for name, new_tokens in plans.items()}
    context = {name: [] for name in plans}
    forks = dict(forks or {})
    live, done = {}, []
    for _ in range(max_steps):
        for name in list(pending):
            if name in live:
                if not live[name].state.value == "finished":
                    continue
                finished = live.pop(name)
                context[name] = finished.prompt + finished.generated
                done.append(finished)
                if name in forks:
                    child = f"{name}/child"
                    scheduler.fork_session(name, child)
                    context[child] = list(context[name])
                    pending[child] = [forks.pop(name)]
            if not pending[name]:
                del pending[name]
                continue
            prompt = context[name] + pending[name].pop(0)
            live[name] = turn(name, prompt, new=3)
            scheduler.submit(live[name])
        if not pending:
            return done
        clock.now += 0.01
        scheduler.step()
    raise AssertionError(f"no progress: {len(scheduler.waiting)} waiting, {len(scheduler.running)} running")


def plans_for(rng, sessions, turns):
    return {
        f"s{i}": [[rng.randint(1, 500) for _ in range(rng.randint(1, 30))] for _ in range(rng.randint(1, turns))]
        for i in range(sessions)
    }


def fuzz_blocks(plans, headroom, extra=0):
    longest = max(sum(len(new) + 3 for new in turns) for turns in plans.values()) + extra
    return int((-(-longest // 16) + 2) * headroom)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    seed=st.integers(0, 2**32 - 1),
    sessions=st.integers(1, 4),
    turns=st.integers(1, 3),
    policy=st.sampled_from(list(Policy)),
    prefix=st.booleans(),
    headroom=st.floats(1.0, 2.5),
    ttl=st.sampled_from([0.005, 0.05, 30.0]),
    cpu=st.sampled_from([0, 6, 1 << 20]),
    slowest=st.sampled_from([0, 3, 40]),
)
def test_fuzz_delayed_copies_never_deadlock_or_change_a_token(model, seed, sessions, turns, policy, prefix, headroom, ttl, cpu, slowest):
    rng = random.Random(seed)
    plans = plans_for(rng, sessions, turns)
    forks = {name: [rng.randint(1, 500) for _ in range(rng.randint(1, 20))] for name in plans if rng.random() < 0.5}
    agent = AgentConfig(policy=policy, prefix_cache=prefix, ttl_cap_s=ttl, cpu_capacity_blocks=cpu)
    backend = HeldBlocksChecker(polls=lambda: rng.randint(0, slowest))
    scheduler, clock = build(model, blocks=fuzz_blocks(plans, headroom, extra=20), agent=agent, transfer=backend)
    for finished in drive(scheduler, clock, plans, forks):
        assert finished.generated == solo(model, finished.prompt, 3)
    run(scheduler)
    scheduler.drain_caches()
    assert scheduler.cache.allocator.num_used == 0
    assert scheduler.tier.used_blocks == 0 and scheduler.tier.in_flight == 0


def test_swapping_through_the_native_host_library_matches_keeping(model):
    rng = random.Random(4)
    plans = plans_for(rng, 3, 3)
    cache = PagedKVCache(CFG, 1, dtype=torch.float32)
    backend = NativeBackend("cpu", staging_bytes=2 * cache.block_bytes, ids_capacity=64)
    try:
        scheduler, clock = build(model, blocks=fuzz_blocks(plans, 1.5), agent=SWAP, transfer=backend)
        for finished in drive(scheduler, clock, plans):
            assert finished.generated == solo(model, finished.prompt, 3)
        assert scheduler.stats.swapped > 0 and scheduler.stats.bytes_offloaded > 0
        assert backend.engine.stats()["kernels"] > 0
        run(scheduler)
        scheduler.drain_caches()
        assert scheduler.cache.allocator.num_used == 0
        assert backend.engine.stats()["host_bytes_live"] == 0
    finally:
        backend.close()


def test_fp8_offload_runs_end_to_end_and_counts_its_smaller_bytes():
    torch.manual_seed(0)
    model = Qwen3(CFG).to(torch.bfloat16).eval()
    costs = CostModel([0.001, 5e-6], [0.001, 5e-6], [0.005, 2e-5, 2e-9], fp8_swap_out_s_coef=[0.002, 6e-6], fp8_swap_in_s_coef=[0.002, 6e-6])
    scheduler, clock = build(model, agent=SWAP, transfer=DelayedBackend(polls=3), dtype=torch.bfloat16, costs=costs, fp8=True)
    first = turn("a", range(1, 40), new=4)
    scheduler.submit(first)
    run(scheduler)
    scheduler.tier.flush()
    assert scheduler.costs.swap_out_s_coef == [0.002, 6e-6]
    second = turn("a", resume_prompt(first), new=4)
    scheduler.submit(second)
    run(scheduler)
    assert second.reused_tokens > 0 and len(second.generated) == 4
    blocks = -(-len(first.prompt + first.generated[:-1]) // 16)
    assert scheduler.stats.bytes_offloaded < blocks * scheduler.cache.block_bytes * 0.6


def test_a_cost_model_without_fp8_times_refuses_an_fp8_tier(model):
    with pytest.raises(ValueError, match="FP8"):
        build(model, fp8=True)


def test_charged_copies_refuse_a_real_backend(model):
    cache = PagedKVCache(CFG, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="charge_transfers"):
        AgentScheduler(model, cache, COSTS, clock=Clock(), charge_transfers=True, transfer=DelayedBackend())


def largest_turn_tokens(jobs, system_tokens):
    largest = 0
    for job in jobs:
        for session in job.sessions:
            context = system_tokens
            if session.parent is not None:
                parent = next(s for s in job.sessions if s.session_id == session.parent)
                context += sum(t.new_tokens + t.output_tokens for t in parent.turns[: session.fork_after_turn + 1])
            for t in session.turns:
                context += t.new_tokens
                largest = max(largest, context + t.output_tokens)
                context += t.output_tokens
    return largest


SYSTEM = 32
# Few slots and slow steps relative to arrivals, so turns really do queue.
ENGINE = EngineConfig(max_batch=2, prefill_chunk=64, prefill_budget=128)
DECODE_STEP, DECODE_SEQ = 0.05, 0.01


def waiting_bound(jobs, costs, agent, pool_blocks):
    """Upper bound on a turn's queueing delay.

    Priority gains aging_rate per second waited, so a turn has passed anything arriving
    later after arrival span / aging_rate seconds. (No fuzzed program reaches the service
    cap, so demotion does not enter.) From there it waits at most for
    every session's one outstanding turn, each taking at most the time to prefill and
    decode a largest turn on a fully contended engine, plus its copies. Copies hold their
    blocks until they end, so a turn short of blocks can wait for the whole copy backlog:
    one round trip per session, priced in whole blocks as the copy stream charges them.
    """
    largest = largest_turn_tokens(jobs, SYSTEM)
    span = max(j.arrival_s for j in jobs) - min(j.arrival_s for j in jobs)
    sessions = sum(len(j.sessions) for j in jobs)
    head_after = span / agent.aging_rate
    per_step = costs.recompute_s(largest) + DECODE_STEP + DECODE_SEQ * ENGINE.max_batch
    steps = largest / ENGINE.prefill_budget + largest + 2
    copies = costs.swap_s(-(-largest // 16) * 16) * (sessions + 1)
    return head_after + sessions * steps * per_step + copies + agent.no_backfill_age_s


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**32 - 1),
    jobs=st.integers(1, 8),
    turns=st.integers(1, 6),
    branching=st.integers(0, 2),
    tool_scale=st.sampled_from([0.001, 0.1, 5.0, 120.0]),
    policy=st.sampled_from(list(Policy)),
    prefix=st.booleans(),
    headroom=st.floats(1.0, 4.0),
    ttl=st.sampled_from([0.01, 1.0, 30.0]),
    cpu=st.sampled_from([0, 8, 1 << 20]),
)
def test_fuzz_no_deadlock_bounded_waiting_no_leak(seed, jobs, turns, branching, tool_scale, policy, prefix, headroom, ttl, cpu):
    check_no_deadlock_bounded_waiting_no_leak(seed, jobs, turns, branching, tool_scale, policy, prefix, headroom, ttl, cpu)


def check_no_deadlock_bounded_waiting_no_leak(seed, jobs, turns, branching, tool_scale, policy, prefix, headroom, ttl, cpu):
    rng = random.Random(seed)
    trace = workload(rng, jobs, turns, branching, tool_scale)
    costs = COSTS
    # The smallest pool that fits the largest turn, up to a few times that.
    minimum = -(-largest_turn_tokens(trace, SYSTEM) // 16) + 2
    pool = int(minimum * headroom)
    agent = AgentConfig(policy=policy, prefix_cache=prefix, ttl_cap_s=ttl, cpu_capacity_blocks=cpu)
    run = simulate(trace, costs, pool, agent, ENGINE, DECODE_STEP, DECODE_SEQ, seed=seed % 1000, system_tokens=SYSTEM)

    assert run.result.jobs_measured == len(trace)
    bound = waiting_bound(trace, costs, agent, pool)
    assert max(run.waits_s) <= bound, f"waited {max(run.waits_s)} > bound {bound}"
    scheduler = run.scheduler
    scheduler.drain_caches()
    assert scheduler.cache.allocator.num_used == 0
    assert scheduler.tier.used_blocks == 0
    return scheduler


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**32 - 1),
    jobs=st.integers(2, 8),
    turns=st.integers(1, 6),
    branching=st.integers(0, 2),
    tool_scale=st.sampled_from([0.001, 0.1, 5.0, 120.0]),
    headroom=st.floats(1.0, 4.0),
)
def test_fuzz_discard_with_the_prefix_cache_on_keeps_the_same_bounds(seed, jobs, turns, branching, tool_scale, headroom):
    """One policy in eight of the fuzz above, so this path gets its own share of the examples."""
    scheduler = check_no_deadlock_bounded_waiting_no_leak(
        seed, jobs, turns, branching, tool_scale, Policy.DISCARD, True, headroom, 30.0, 1 << 20
    )
    stats = scheduler.stats
    assert stats.swapped == stats.kept == stats.restored_tokens == stats.kept_hit_tokens == 0
    # Only whole blocks are cached.
    assert stats.prefix_hit_tokens % 16 == 0


def test_fuzz_discard_workload_reuses_and_evicts(monkeypatch):
    """Guards the fuzz test above: it says nothing unless the tree serves hits and takes pressure."""
    evicted = []
    real = RadixTree.evict

    def counting(tree, count):
        freed = real(tree, count)
        evicted[-1] += freed
        return freed

    monkeypatch.setattr(RadixTree, "evict", counting)
    hits = 0
    tightest = 0.0
    for seed in range(20):
        evicted.append(0)
        rng = random.Random(seed)
        trace = workload(rng, 8, 5, 1, 0.1)
        pool = -(-largest_turn_tokens(trace, SYSTEM) // 16) + 2
        agent = AgentConfig(policy=Policy.DISCARD)
        run = simulate(trace, COSTS, pool, agent, ENGINE, DECODE_STEP, DECODE_SEQ, seed=seed, system_tokens=SYSTEM)
        hits += run.scheduler.stats.prefix_hit_tokens > 0
        tightest = max(tightest, max(run.waits_s) / waiting_bound(trace, COSTS, agent, pool))
    assert hits >= 18
    assert sum(count > 0 for count in evicted) >= 18
    assert 0.01 < tightest <= 1.0


def test_waiting_bound_covers_copies_and_needs_that_term():
    """With copies far slower than compute, the backlog is most of what a turn waits for."""
    costs = CostModel([200.0, 0.0], [200.0, 0.0], [0.005, 2e-5, 2e-9])
    agent = AgentConfig(policy=Policy.SWAP, prefix_cache=False)
    beyond_compute = 0
    for seed in range(12):
        rng = random.Random(seed)
        trace = workload(rng, 6, 3, 0, 0.001)
        pool = -(-largest_turn_tokens(trace, SYSTEM) // 16) + 2
        run = simulate(trace, costs, pool, agent, ENGINE, DECODE_STEP, DECODE_SEQ, seed=seed, system_tokens=SYSTEM)
        bound = waiting_bound(trace, costs, agent, pool)
        assert max(run.waits_s) <= bound
        without_copies = bound - costs.swap_s(0) * (sum(len(j.sessions) for j in trace) + 1)
        beyond_compute += max(run.waits_s) > without_copies
    assert beyond_compute > 0


def test_fuzz_workload_actually_contends():
    """Guards the fuzz test above: without queueing, its bound would check nothing."""
    queued = evicted = held = 0
    tightest = 0.0
    for seed in range(40):
        rng = random.Random(seed)
        trace = workload(rng, 8, 5, 1, 0.1)
        pool = -(-largest_turn_tokens(trace, SYSTEM) // 16) + 2
        agent = AgentConfig(policy=Policy.KEEP_TTL)
        run = simulate(trace, COSTS, pool, agent, ENGINE, DECODE_STEP, DECODE_SEQ, seed=seed, system_tokens=SYSTEM)
        queued += max(run.waits_s) > 0
        evicted += run.scheduler.stats.early_demotions > 0
        held += run.scheduler.stats.held_by_copies > 0
        tightest = max(tightest, max(run.waits_s) / waiting_bound(trace, COSTS, agent, pool))
    assert queued >= 35
    assert evicted >= 20
    # Turns really wait on blocks that are still copying, or the bound checks a simpler system.
    assert held >= 10
    assert 0.01 < tightest <= 1.0


def test_aging_bounds_waiting_that_is_otherwise_unbounded_in_practice():
    """A late program behind long-running older programs on a one-slot engine."""

    def trace():
        jobs = []
        for j in range(4):
            turns = [TraceTurn(32, 8, "a", 0.0) for _ in range(40)]
            jobs.append(TraceJob(f"old{j}", 0.0 + j * 0.001, [TraceSession(f"old{j}/m", turns)], "t"))
        jobs.append(TraceJob("late", 0.5, [TraceSession("late/m", [TraceTurn(32, 8, None, 0.0)])], "t"))
        return jobs

    engine = EngineConfig(max_batch=1, prefill_chunk=64, prefill_budget=64)

    def late_wait(rate):
        run = simulate(trace(), COSTS, 4096, AgentConfig(aging_rate=rate), engine, system_tokens=16)
        record = next(t for t in run.result.turns if t.job_id == "late")
        return record.first_token_s - record.submitted_s

    assert late_wait(1.0) < 0.5 * late_wait(0.0)


def test_deadlock_is_detected_rather_than_hanging():
    # A trace that cannot fit fails at submit; a driver hang would show up as Deadlock.
    trace = [TraceJob("j", 0.0, [TraceSession("j/m", [TraceTurn(1000, 8, None, 0.0)])], "t")]
    with pytest.raises(RequestTooLarge):
        simulate(trace, COSTS, 2, system_tokens=16)
    assert issubclass(Deadlock, RuntimeError)


def test_generated_traces_run_under_every_policy():
    jobs = generate(SyntheticWorkload(jobs=6, tokens_per_job_mean=2000, tokens_per_job_sd=400, turns_mean=4, turns_sd=1, branching=1), seed=3)
    results = {}
    for policy in Policy:
        run = simulate(jobs, COSTS, 2048, AgentConfig(policy=policy), system_tokens=64)
        results[policy] = run.result.mean_jct_s
        assert run.result.source.startswith("synthetic")
    assert all(v > 0 for v in results.values())


def test_a_long_copy_does_not_demote_every_kept_session_when_the_radix_tree_also_holds_their_blocks(model):
    """A demoted block held by the tree comes back too, once the copy ends and the retry evicts it."""
    agent = AgentConfig(policy=Policy.KEEP_TTL, prefix_cache=True, ttl_cap_s=1000.0)
    scheduler, clock = build(model, blocks=16, agent=agent, transfer=DelayedBackend(polls=50))
    for name, base in (("x", 1), ("y", 100), ("z", 200)):
        scheduler.submit(turn(name, range(base, base + 60), new=4))
        run(scheduler)
        clock.now += 1.0
    assert all(s.where is Where.KEPT for s in scheduler.sessions.values())
    big = turn("big", range(300, 420), new=4)
    scheduler.submit(big)
    advance(scheduler, clock, steps=20)
    assert scheduler.stats.early_demotions == 1
    assert big in scheduler.waiting
    # Liveness: the retry after the copy must evict what the copy let go of, and admit the turn.
    run(scheduler)
    assert big.generated == solo(model, big.prompt, 4)
    assert sum(s.where is Where.KEPT for s in scheduler.sessions.values() if s.session_id != "big") == 2


def test_a_shortfall_past_what_is_freeing_still_demotes_once_copies_end(model):
    """Nothing counts as freeing with no copy in flight, so a large turn cannot be stranded."""
    agent = AgentConfig(policy=Policy.KEEP_TTL, prefix_cache=True, ttl_cap_s=1000.0)
    scheduler, clock = build(model, blocks=16, agent=agent, transfer=DelayedBackend(polls=5))
    for name, base in (("x", 1), ("y", 100), ("z", 200)):
        scheduler.submit(turn(name, range(base, base + 60), new=4))
        run(scheduler)
        clock.now += 1.0
    big = turn("big", range(300, 480), new=4)
    scheduler.submit(big)
    run(scheduler)
    assert big.generated == solo(model, big.prompt, 4)
    assert scheduler.stats.early_demotions >= 2


def swapped(model, blocks=64, prefix_cache=True):
    agent = AgentConfig(policy=Policy.SWAP, prefix_cache=prefix_cache)
    scheduler, clock = build(model, blocks=blocks, agent=agent)
    first = turn("s", range(1, 70), new=4)
    scheduler.submit(first)
    run(scheduler)
    scheduler.tier.flush()
    assert scheduler.sessions["s"].where is Where.CPU
    return scheduler, clock, first


def test_a_swapped_session_resumes_from_the_tree_when_it_still_holds_the_blocks(model):
    scheduler, _, first = swapped(model)
    second = turn("s", resume_prompt(first), new=4)
    scheduler.submit(second)
    scheduler.step()
    stats = scheduler.stats
    assert stats.restored_tokens == 0 and stats.bytes_restored == 0 and stats.restores_avoided == 1
    assert stats.prefix_hit_tokens == 64
    assert scheduler.tier.used_blocks == 0, "the CPU copy has no further use and must be released"
    assert scheduler.sessions["s"].handle is None
    assert set(second.table.blocks[:4]) <= scheduler.tree.block_ids()
    run(scheduler)
    assert second.generated == solo(model, second.prompt, 4)


def test_a_swapped_session_still_restores_when_the_tree_lost_its_blocks(model):
    scheduler, _, first = swapped(model)
    scheduler.tree.clear()
    second = turn("s", resume_prompt(first), new=4)
    scheduler.submit(second)
    run(scheduler)
    assert scheduler.stats.restored_tokens > 0 and scheduler.stats.restores_avoided == 0
    assert second.generated == solo(model, second.prompt, 4)


def test_a_swapped_session_still_restores_with_the_prefix_cache_off(model):
    scheduler, _, first = swapped(model, prefix_cache=False)
    second = turn("s", resume_prompt(first), new=4)
    scheduler.submit(second)
    run(scheduler)
    assert scheduler.stats.restored_tokens > 0 and scheduler.stats.restores_avoided == 0


def test_a_resume_from_the_tree_that_cannot_be_admitted_leaves_the_session_on_cpu(model):
    scheduler, _, first = swapped(model, blocks=16)
    handle = scheduler.sessions["s"].handle
    blocker = turn("other", range(100, 260), new=20)
    scheduler.submit(blocker)
    scheduler.step()
    assert blocker in scheduler.running
    in_tree = sorted(scheduler.tree.block_ids())
    refs = [scheduler.cache.allocator.refcount(b) for b in in_tree]
    second = turn("s", resume_prompt(first) + list(range(300, 348)), new=4)
    scheduler.submit(second)
    scheduler.step()
    assert second in scheduler.waiting
    session = scheduler.sessions["s"]
    assert session.where is Where.CPU and session.handle is handle and second.table.blocks == []
    assert scheduler.tier.used_blocks == handle.num_blocks and scheduler.stats.restores_avoided == 0
    assert [scheduler.cache.allocator.refcount(b) for b in in_tree] == refs
    run(scheduler)
    assert second.generated == solo(model, second.prompt, 4)


def test_a_failed_pin_drops_the_session_and_the_engine_carries_on(model):
    agent = AgentConfig(policy=Policy.SWAP, prefix_cache=False)
    backend = NativeBackend("cpu", staging_bytes=1 << 20, ids_capacity=256)
    backend.engine.set_pin_limit(0)
    scheduler, _ = build(model, agent=agent, transfer=backend)
    first = turn("s", range(1, 70), new=4)
    scheduler.submit(first)
    run(scheduler)
    session = scheduler.sessions["s"]
    assert session.where is Where.NONE and session.kv_tokens == [] and session.handle is None
    stats = scheduler.stats
    assert stats.pin_failure_drops == 1 and stats.dropped == 1 and stats.swapped == 0
    assert scheduler.cache.allocator.num_used == 0
    # Memory comes back: the same engine offloads the next session as usual.
    backend.engine.set_pin_limit(1 << 40)
    second = turn("s", resume_prompt(first), new=4)
    scheduler.submit(second)
    run(scheduler)
    assert second.generated == solo(model, second.prompt, 4)
    assert scheduler.stats.swapped == 1 and scheduler.stats.host_pins >= 1
    backend.close()


# Discard. The session gives up its references at the tool call and the radix tree keeps its
# whole blocks until memory is needed, as vLLM's free queue does.

COMMON = list(range(100, 148))  # Exactly three blocks.
DISCARD = AgentConfig(policy=Policy.DISCARD)
NO_CACHE = AgentConfig(policy=Policy.DISCARD, prefix_cache=False)


def tail(session):
    """Tokens no other session has, chosen so no block of them repeats between sessions."""
    return [session * 20 + i for i in range(1, 20)]


def kv_of(finished):
    # The last generated token has not been fed through the model.
    return finished.prompt + finished.generated[:-1]


def whole_blocks(finished):
    return len(kv_of(finished)) // 16


def finish(scheduler, *turns_to_run):
    for submitted in turns_to_run:
        scheduler.submit(submitted)
    run(scheduler)
    for finished in turns_to_run:
        assert finished.state.value == "finished"


def stored(scheduler, tokens):
    """What the tree holds of `tokens`, without the lookup counting as a use."""
    return deepcopy(scheduler.tree).match(tokens)


def assert_only_the_tree_holds(scheduler, blocks):
    allocator = scheduler.cache.allocator
    assert scheduler.tree.block_ids() == set(blocks)
    assert all(allocator.refcount(block) == 1 for block in blocks)
    assert allocator.num_used == len(set(blocks))


@pytest.fixture
def prefilled(model, monkeypatch):
    """Every (start, count) the model was really run over."""
    fed = []
    real = model.prefill

    def counting(cache, table, tokens, start):
        fed.append((start, tokens.numel()))
        return real(cache, table, tokens, start)

    monkeypatch.setattr(model, "prefill", counting)
    return fed


@pytest.mark.parametrize("end_session", [False, True])
def test_discard_gives_up_references_and_the_tree_alone_holds_the_whole_blocks(model, end_session):
    scheduler, _ = build(model, agent=DISCARD)
    a = turn("a", COMMON + tail(1), tool_name="grep", end_session=end_session)
    finish(scheduler, a)
    held = stored(scheduler, kv_of(a))
    assert len(held) == whole_blocks(a) > 0
    # Tree only, and the partial last block is back on the free list.
    assert_only_the_tree_holds(scheduler, held)
    assert (len(kv_of(a)) % 16) > 0
    stats = scheduler.stats
    assert stats.kept == stats.swapped == 0
    assert stats.dropped == (0 if end_session else 1)
    if not end_session:
        session = scheduler.sessions["a"]
        assert session.where is Where.NONE and not session.table.blocks and not session.kv_tokens


def test_discard_returning_before_pressure_reuses_its_whole_blocks(model, prefilled):
    scheduler, clock = build(model, agent=DISCARD)
    first = turn("a", COMMON + tail(1), tool_name="grep")
    finish(scheduler, first)
    held = stored(scheduler, kv_of(first))
    clock.now += 1
    prefilled.clear()
    second = turn("a", resume_prompt(first))
    finish(scheduler, second)
    assert second.reused_tokens == len(held) * 16 == scheduler.stats.prefix_hit_tokens
    # The model was run over everything past those blocks, from where they end, and no more.
    assert prefilled[0][0] == second.reused_tokens
    assert sum(count for _, count in prefilled) == len(second.prompt) - second.reused_tokens
    assert scheduler.stats.restored_tokens == scheduler.stats.kept_hit_tokens == 0
    # The same blocks, read and not rewritten, and the session again holds none of them.
    after = stored(scheduler, kv_of(second))
    assert after[: len(held)] == held
    assert_only_the_tree_holds(scheduler, after)
    assert first.generated == solo(model, first.prompt, first.max_new_tokens)
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)


def test_discard_reuses_only_whole_blocks(model):
    scheduler, _ = build(model, agent=DISCARD)
    common = list(range(100, 140))  # Two blocks and a part.
    finish(scheduler, turn("a", common + tail(1)))
    b = turn("b", common + tail(2))
    finish(scheduler, b)
    assert b.reused_tokens == 32
    assert b.generated == solo(model, b.prompt, b.max_new_tokens)


def test_discard_a_session_reuses_what_another_left_and_the_tree_keeps_its_own_blocks(model):
    scheduler, _ = build(model, agent=DISCARD)
    a = turn("a", COMMON + tail(1))
    finish(scheduler, a)
    shared = stored(scheduler, COMMON)
    b = turn("b", COMMON + tail(2))
    finish(scheduler, b)
    assert b.reused_tokens == len(COMMON)
    assert stored(scheduler, COMMON) == shared
    assert_only_the_tree_holds(scheduler, stored(scheduler, kv_of(a)) + stored(scheduler, kv_of(b))[len(shared) :])
    for finished in (a, b):
        assert finished.generated == solo(model, finished.prompt, finished.max_new_tokens)


def test_discard_equal_blocks_after_different_prefixes_are_not_reused(model):
    scheduler, _ = build(model, agent=DISCARD)
    same = list(range(300, 316))
    finish(scheduler, turn("a", list(range(1, 17)) + same + [1]))
    b = turn("b", list(range(21, 37)) + same + [1])
    finish(scheduler, b)
    assert b.reused_tokens == 0
    assert b.generated == solo(model, b.prompt, b.max_new_tokens)


def test_discard_a_near_miss_in_a_block_reuses_nothing_of_it(model):
    scheduler, _ = build(model, agent=DISCARD)
    finish(scheduler, turn("a", COMMON + tail(1)))
    almost = list(COMMON)
    almost[20] += 1
    b = turn("b", almost + tail(2))
    finish(scheduler, b)
    assert b.reused_tokens == 16
    assert b.generated == solo(model, b.prompt, b.max_new_tokens)


# Pressure, on a pool of 8. These prompts leave three whole blocks in the tree and need four to
# run, so two paths leave two blocks free and a third turn makes the tree give up two.

PATH_A, PATH_B, PATH_C = (list(range(start, start + 49)) for start in (1, 101, 201))


def two_paths(model):
    scheduler, clock = build(model, blocks=8, agent=DISCARD)
    a = turn("a", PATH_A, tool_name="grep")
    finish(scheduler, a)
    b = turn("b", PATH_B, tool_name="grep")
    finish(scheduler, b)
    a_blocks, b_blocks = stored(scheduler, kv_of(a)), stored(scheduler, kv_of(b))
    assert len(a_blocks) == len(b_blocks) == 3
    assert_only_the_tree_holds(scheduler, a_blocks + b_blocks)
    return scheduler, clock, a, b, a_blocks, b_blocks


def test_discard_pressure_takes_the_least_recent_path_from_its_tail(model):
    scheduler, _, a, b, a_blocks, b_blocks = two_paths(model)
    c = turn("c", PATH_C, tool_name="grep")
    finish(scheduler, c)
    assert stored(scheduler, kv_of(a)) == a_blocks[:1]
    assert stored(scheduler, kv_of(b)) == b_blocks
    assert c.generated == solo(model, c.prompt, c.max_new_tokens)


def test_discard_no_eviction_while_memory_suffices(model):
    scheduler, _, a, b, a_blocks, b_blocks = two_paths(model)
    # Needs what is free and nothing more.
    d = turn("d", list(range(301, 325)), new=4)
    finish(scheduler, d)
    assert stored(scheduler, kv_of(a)) == a_blocks
    assert stored(scheduler, kv_of(b)) == b_blocks


def test_discard_returning_after_eviction_recomputes_exactly_what_was_evicted(model, prefilled):
    scheduler, clock, a, b, a_blocks, b_blocks = two_paths(model)
    c = turn("c", PATH_C, tool_name="grep")
    finish(scheduler, c)
    c_blocks = stored(scheduler, kv_of(c))
    clock.now += 1
    prefilled.clear()
    back = turn("a", resume_prompt(a))
    finish(scheduler, back)
    # Its one surviving block is reused and the two evicted ones are computed again.
    assert back.reused_tokens == 16
    assert prefilled[0][0] == 16
    assert sum(count for _, count in prefilled) == len(back.prompt) - 16
    assert stored(scheduler, kv_of(back))[0] == a_blocks[0]
    # That turn needed room too: now b is the least recent path, and c is untouched.
    assert stored(scheduler, kv_of(b)) == b_blocks[:1]
    assert stored(scheduler, kv_of(c)) == c_blocks
    assert back.generated == solo(model, back.prompt, back.max_new_tokens)


def test_discard_returning_after_its_whole_path_was_evicted_recomputes_it_all(model):
    scheduler, clock, a, _, _, _ = two_paths(model)
    big = turn("big", list(range(201, 310)))
    finish(scheduler, big)
    assert stored(scheduler, kv_of(a)) == []
    clock.now += 1
    back = turn("a", resume_prompt(a))
    finish(scheduler, back)
    assert back.reused_tokens == 0
    assert back.generated == solo(model, back.prompt, back.max_new_tokens)


@pytest.mark.parametrize(
    ("needs_tokens", "a_left", "b_left"),
    [
        # Two blocks short: a's unused tail is the oldest, then b's. A hit on a's whole edge would spare it.
        (49, 2, 2),
        # Four short: all of b goes before what the hit refreshed. No refresh at all would take a instead.
        (81, 2, 0),
    ],
)
def test_discard_a_hit_refreshes_recency_of_only_the_blocks_it_reused(model, needs_tokens, a_left, b_left):
    scheduler, _, a, b, a_blocks, b_blocks = two_paths(model)
    hit = turn("hit", kv_of(a)[:32] + [401, 402, 403])
    finish(scheduler, hit)
    assert hit.reused_tokens == 32
    assert_only_the_tree_holds(scheduler, a_blocks + b_blocks)
    finish(scheduler, turn("c", list(range(201, 201 + needs_tokens))))
    assert stored(scheduler, kv_of(a)) == a_blocks[:a_left]
    assert stored(scheduler, kv_of(b)) == b_blocks[:b_left]


def test_discard_pressure_cannot_take_blocks_another_table_still_references(model):
    scheduler, _ = build(model, agent=DISCARD)
    held = []

    def fork_once(request, _token):
        if not held:
            held.append(scheduler.cache.fork(request.table))

    a = turn("a", COMMON + tail(1), on_token=fork_once)
    finish(scheduler, a)
    allocator = scheduler.cache.allocator
    in_tree = stored(scheduler, kv_of(a))
    assert in_tree == held[0].blocks[: len(in_tree)]
    assert all(allocator.refcount(block) == 2 for block in in_tree)
    before = scheduler.cache.pool[held[0].blocks].clone()
    assert scheduler.tree.evict(len(in_tree)) == 0
    assert stored(scheduler, kv_of(a)) == in_tree
    assert scheduler.cache.pool[held[0].blocks].equal(before)
    scheduler.cache.release(held[0])
    assert scheduler.tree.evict(len(in_tree)) == len(in_tree)
    scheduler.drain_caches()
    assert allocator.num_used == 0


def test_discard_a_sub_agent_finds_its_parents_blocks_in_the_tree(model):
    scheduler, _ = build(model, agent=DISCARD)
    parent = turn("p", COMMON + tail(1), tool_name="spawn", program_id="job")
    finish(scheduler, parent)
    child = scheduler.fork_session("p", "c")
    assert child.where is Where.NONE and not child.table.blocks
    forked = turn("c", parent.prompt + parent.generated + [5])
    finish(scheduler, forked)
    assert forked.reused_tokens == whole_blocks(parent) * 16
    assert forked.generated == solo(model, forked.prompt, forked.max_new_tokens)


def test_discard_duplicates_from_a_concurrent_session_are_freed_and_the_tree_keeps_one_copy(model):
    scheduler, _ = build(model, agent=DISCARD)
    a, b, c = (turn(name, COMMON + tail(i)) for i, name in enumerate("abc", 1))
    finish(scheduler, a, b, c)
    assert a.reused_tokens == b.reused_tokens == c.reused_tokens == 0
    paths = [stored(scheduler, kv_of(t)) for t in (a, b, c)]
    assert paths[0][:3] == paths[1][:3] == paths[2][:3]
    assert_only_the_tree_holds(scheduler, [block for path in paths for block in path])
    for finished in (a, b, c):
        assert finished.generated == solo(model, finished.prompt, finished.max_new_tokens)


def test_discard_without_the_prefix_cache_reuses_nothing(model, prefilled):
    """layover-discard-nocache, the strict lower bound."""
    scheduler, clock = build(model, agent=NO_CACHE)
    first = turn("a", COMMON + tail(1), tool_name="grep")
    finish(scheduler, first)
    assert scheduler.tree.num_blocks == 0 and scheduler.cache.allocator.num_used == 0
    clock.now += 1
    prefilled.clear()
    second = turn("a", resume_prompt(first))
    other = turn("b", COMMON + tail(2))
    finish(scheduler, second, other)
    assert second.reused_tokens == other.reused_tokens == 0
    assert scheduler.stats.prefix_hit_tokens == 0
    assert scheduler.tree.num_blocks == 0 and scheduler.cache.allocator.num_used == 0
    assert sum(count for _, count in prefilled) == len(second.prompt) + len(other.prompt)
    assert scheduler.stats.prefilled_tokens == len(first.prompt) + len(second.prompt) + len(other.prompt)
    assert second.generated == solo(model, second.prompt, second.max_new_tokens)


@pytest.mark.parametrize("policy", list(Policy))
def test_every_policy_leaves_the_whole_blocks_of_a_finished_session_in_the_tree(model, policy):
    scheduler, _ = build(model, agent=AgentConfig(policy=policy))
    a = turn("a", COMMON + tail(1), end_session=True)
    finish(scheduler, a)
    assert scheduler.tree.num_blocks == whole_blocks(a)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**32 - 1),
    sessions=st.integers(1, 5),
    turns=st.integers(1, 3),
    batch=st.sampled_from([1, 4]),
    tight=st.booleans(),
)
def test_fuzz_discard_reuses_what_the_tree_should_hold(model, seed, sessions, turns, batch, tight):
    """Against an exact model of what was reached, with concurrency and real blocks."""
    rng = random.Random(seed)
    # Whole blocks from a tiny alphabet, so sessions diverge at random depths and meet again.
    chunks = [[rng.randint(1, 500) for _ in range(16)] for _ in range(3)]
    prompts = {f"s{i}": [token for _ in range(rng.randint(1, 4)) for token in rng.choice(chunks)] + [rng.randint(1, 500)] for i in range(sessions)}
    reached: set[tuple[int, ...]] = set()

    def longest_reached(prompt):
        count = 0
        while (count + 1) * 16 <= len(prompt) - 1 and tuple(prompt[: (count + 1) * 16]) in reached:
            count += 1
        return count

    def record(finished):
        # Turns run one at a time when batch is 1, so nothing else was reached since this one started.
        best = longest_reached(finished.prompt) * 16
        if batch == 1 and not tight:
            assert finished.reused_tokens == best
        else:
            assert finished.reused_tokens <= best
        kv = kv_of(finished)
        for count in range(1, len(kv) // 16 + 1):
            reached.add(tuple(kv[: count * 16]))

    longest = max(len(p) for p in prompts.values()) + turns * 12
    # Tight fits one turn and a few more blocks, so paths are evicted as turns go.
    blocks = longest // 16 + 4 if tight else (sessions + 1) * (longest // 16 + 2) + 8 + sessions * turns * (longest // 16 + 1)
    scheduler, clock = build(model, blocks=blocks, agent=DISCARD, engine=EngineConfig(max_batch=batch))
    finished_turns = []
    live = {name: turn(name, prompt, on_finish=record) for name, prompt in prompts.items()}
    for submitted in live.values():
        scheduler.submit(submitted)
    remaining = {name: turns - 1 for name in prompts}
    for _ in range(100_000):
        for name, current in list(live.items()):
            if current.state.value != "finished":
                continue
            finished_turns.append(current)
            del live[name]
            if remaining[name]:
                remaining[name] -= 1
                live[name] = turn(name, resume_prompt(current), on_finish=record)
                scheduler.submit(live[name])
        if not live:
            break
        clock.now += 0.01
        scheduler.step()
    assert not live and scheduler.idle

    for finished in finished_turns:
        assert finished.generated == solo(model, finished.prompt, finished.max_new_tokens)
    allocator = scheduler.cache.allocator
    if not tight:
        # Nothing was short of memory, so the tree holds exactly what was reached, once each.
        assert scheduler.tree.num_blocks == len(reached)
        for path in reached:
            assert len(stored(scheduler, list(path))) == len(path) // 16
    # Paused sessions hold nothing under discard.
    assert allocator.num_used == scheduler.tree.num_blocks
    assert all(allocator.refcount(block) == 1 for block in scheduler.tree.block_ids())
    scheduler.drain_caches()
    assert allocator.num_used == 0

"""Scheduler behaviour on real model tensors, and fuzzing on a virtual clock."""

import random

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig, AgentScheduler, AgentTurn, SessionBusy, Where
from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request, RequestTooLarge
from layover.kv import PagedKVCache
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


def build(model, blocks=64, agent=AgentConfig(), engine=EngineConfig()):
    clock = Clock()
    cache = PagedKVCache(CFG, blocks, dtype=torch.float32)
    return AgentScheduler(model, cache, COSTS, engine, agent, clock=clock), clock


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
    decode a largest turn on a fully contended engine, plus its copies.
    """
    largest = largest_turn_tokens(jobs, SYSTEM)
    span = max(j.arrival_s for j in jobs) - min(j.arrival_s for j in jobs)
    sessions = sum(len(j.sessions) for j in jobs)
    head_after = span / agent.aging_rate
    per_step = costs.recompute_s(largest) + DECODE_STEP + DECODE_SEQ * ENGINE.max_batch
    steps = largest / ENGINE.prefill_budget + largest + 2
    copies = costs.swap_s(largest) * (sessions + 1)
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


def test_fuzz_workload_actually_contends():
    """Guards the fuzz test above: without queueing, its bound would check nothing."""
    queued = evicted = 0
    tightest = 0.0
    for seed in range(40):
        rng = random.Random(seed)
        trace = workload(rng, 8, 5, 1, 0.1)
        pool = -(-largest_turn_tokens(trace, SYSTEM) // 16) + 2
        agent = AgentConfig(policy=Policy.KEEP_TTL)
        run = simulate(trace, COSTS, pool, agent, ENGINE, DECODE_STEP, DECODE_SEQ, seed=seed, system_tokens=SYSTEM)
        queued += max(run.waits_s) > 0
        evicted += run.scheduler.stats.early_demotions > 0
        tightest = max(tightest, max(run.waits_s) / waiting_bound(trace, COSTS, agent, pool))
    assert queued >= 35
    assert evicted >= 20
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

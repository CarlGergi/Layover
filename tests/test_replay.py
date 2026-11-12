"""Trace formats, the synthetic generator, metrics, and replay over HTTP."""

import asyncio
import json
import statistics

import pytest
import torch

from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig
from layover.config import ModelConfig
from layover.model import Qwen3
from layover.replay.__main__ import main as cli
from layover.replay.http_replay import replay
from layover.replay.results import TurnRecord, percentile, summarize
from layover.replay.simulate import simulate
from layover.replay.trace import (
    BFCL_SHAPE, ContextBuilder, SWE_BENCH_SHAPE, SyntheticWorkload, TraceJob, TraceSession, TraceTurn,
    from_mini_swe_agent, generate, job_from_dict, load_jsonl, save_jsonl,
)

from test_server import ByteTokenizer, Served

COSTS = CostModel([0.001, 5e-6], [0.001, 5e-6], [0.005, 2e-5, 2e-9])
CFG = ModelConfig.micro()


def job(job_id="j", arrival=0.0):
    return TraceJob(job_id, arrival, [
        TraceSession(f"{job_id}/m", [TraceTurn(20, 3, "t", 0.01), TraceTurn(10, 3, "t", 0.0)]),
        TraceSession(f"{job_id}/c", [TraceTurn(5, 2, "t", 0.0)], parent=f"{job_id}/m", fork_after_turn=0),
    ], "unit")


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    save_jsonl([job("b", 2.0), job("a", 1.0)], path)
    loaded = load_jsonl(path)
    assert [j.job_id for j in loaded] == ["a", "b"]
    assert loaded[0].source == "unit" and loaded[0].sessions[1].parent == "a/m"
    assert loaded[0].sessions[0].turns[0].tool_name == "t"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["sessions"][1].update(parent="nope"),
        lambda d: d["sessions"][1].update(fork_after_turn=9),
        lambda d: d["sessions"][0]["turns"].clear(),
        lambda d: d["sessions"][0]["turns"][0].update(output_tokens=0),
        lambda d: d["sessions"][1].update(parent=None),
    ],
)
def test_invalid_traces_are_rejected(change):
    from dataclasses import asdict

    raw = asdict(job())
    change(raw)
    with pytest.raises(ValueError):
        job_from_dict(raw)


def trajectory(**kwargs):
    return {
        "trajectory_format": "mini-swe-agent-1",
        "messages": [
            {"role": "system", "content": "aaaa", "timestamp": 1.0},
            {"role": "user", "content": "bb", "timestamp": 1.0},
            {"role": "assistant", "content": "ccc", "timestamp": 5.0},
            {"role": "user", "content": "dddddd", "timestamp": 7.5},
            {"role": "assistant", "content": "ee", "timestamp": 9.0},
        ],
        **kwargs,
    }


def test_mini_swe_agent_conversion_uses_timestamps():
    converted = from_mini_swe_agent(trajectory(), len, "traj")
    turns = converted.main.turns
    assert [t.new_tokens for t in turns] == [6, 6]
    assert [t.output_tokens for t in turns] == [3, 2]
    assert turns[0].tool_s == pytest.approx(2.5)
    assert converted.source == "mini-swe-agent-1"


def test_mini_swe_agent_without_timestamps_says_so_in_its_source():
    raw = trajectory()
    for message in raw["messages"]:
        del message["timestamp"]
    converted = from_mini_swe_agent(raw, len, "traj", default_tool_s=3.0)
    assert converted.main.turns[0].tool_s == 3.0
    assert "assumed" in converted.source


def test_unknown_trajectory_format_is_rejected():
    with pytest.raises(ValueError):
        from_mini_swe_agent({"trajectory_format": "other", "messages": []}, len, "x")


def test_generator_is_deterministic_and_labelled_synthetic():
    workload = SyntheticWorkload(jobs=5)
    a, b = generate(workload, 1), generate(workload, 1)
    assert a == b and a != generate(workload, 2)
    assert all(j.source.startswith("synthetic") for j in a)


@pytest.mark.parametrize("shape", [SWE_BENCH_SHAPE, BFCL_SHAPE])
def test_generator_follows_the_published_shape(shape):
    jobs = generate(SyntheticWorkload(**{**shape.__dict__, "jobs": 800}), 0)
    turns = [len(j.main.turns) for j in jobs]
    assert statistics.fmean(turns) == pytest.approx(shape.turns_mean, rel=0.1)
    tokens = [sum(t.new_tokens + t.output_tokens for t in j.main.turns) for j in jobs]
    assert statistics.fmean(tokens) == pytest.approx(shape.tokens_per_job_mean, rel=0.25)


def test_branching_sweep_adds_subagents():
    jobs = generate(SyntheticWorkload(jobs=3, branching=2), 0)
    assert all(len(j.sessions) == 3 for j in jobs)


def test_context_builder_shares_system_prompt_only():
    builder = ContextBuilder(100, seed=1, system_tokens=8)
    assert builder.system_prompt() == ContextBuilder(100, seed=1, system_tokens=8).system_prompt()
    assert builder.new_tokens("a", 0, 8) == builder.new_tokens("a", 0, 8)
    assert builder.new_tokens("a", 0, 8) != builder.new_tokens("b", 0, 8)


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 0.95) == 95
    assert percentile(values, 1.0) == 100
    assert percentile([7], 0.95) == 7


def test_summary_excludes_warmup_jobs():
    arrivals = {"a": 0.0, "b": 10.0, "c": 20.0}
    finishes = {"a": 100.0, "b": 12.0, "c": 26.0}
    turns = [TurnRecord("a", "a/m", 0, 0, 50, 100), TurnRecord("b", "b/m", 0, 10, 11, 12), TurnRecord("c", "c/m", 0, 20, 21, 26)]
    result = summarize("src", "sys", arrivals, finishes, turns, warmup_jobs=1)
    assert result.jobs_measured == 2 and "a" not in result.jct_s
    assert result.mean_jct_s == pytest.approx(4.0)
    assert result.programs_per_hour == pytest.approx(2 / 16 * 3600)
    assert all(t.job_id != "a" for t in result.turns)
    assert result.source == "src"
    with pytest.raises(ValueError):
        summarize("s", "s", {"a": 0.0}, {"a": 1.0}, [], warmup_jobs=1)


def test_simulated_policies_rank_sensibly_when_recompute_is_expensive():
    """Sanity of the simulator, not a result: reuse has to beat recomputing everything."""
    jobs = generate(
        SyntheticWorkload(jobs=8, arrival_rate_per_s=0.1, tokens_per_job_mean=8000, tokens_per_job_sd=1000, turns_mean=6, turns_sd=1, tool_s_median=0.2),
        seed=5,
    )
    def jct(policy):
        agent = AgentConfig(policy=policy, prefix_cache=False)
        return simulate(jobs, COSTS, 4096, agent, system_tokens=64).result.mean_jct_s
    assert jct(Policy.DISCARD) > jct(Policy.COMBINED)
    assert jct(Policy.DISCARD) > jct(Policy.SWAP)


@pytest.fixture(scope="module")
def micro_model():
    torch.manual_seed(0)
    return Qwen3(CFG).float().eval()


def test_http_replay_end_to_end_with_forks(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro", blocks=512)
    try:
        jobs = [job("a", 0.0), job("b", 0.01), job("warm", 0.0)]
        jobs[2].arrival_s = -0.0
        result = asyncio.run(replay(jobs, served.url, "micro", CFG.vocab_size, time_scale=0.1, system_tokens=32, warmup_jobs=1))
        assert result.jobs_total == 3 and result.jobs_measured == 2
        assert result.mean_jct_s > 0 and result.programs_per_hour > 0
        assert {t.session_id.split("/")[1] for t in result.turns} == {"m", "c"}
        metrics = served.client.get("/metrics").json()
        assert metrics["kept_hit_tokens"] + metrics["prefix_hit_tokens"] > 0
        assert metrics["running"] == 0 and metrics["waiting"] == 0
        served.loop.call(served.scheduler.drain_caches).result()
        assert served.scheduler.cache.allocator.num_used == 0
    finally:
        served.close()


def test_http_replay_without_hints_works_like_a_plain_openai_client(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro", blocks=512)
    try:
        result = asyncio.run(replay([job("a")], served.url, "micro", CFG.vocab_size, time_scale=0.1, system_tokens=32, hints=False))
        assert result.jobs_measured == 1
    finally:
        served.close()


def test_cli_generate_and_convert(tmp_path):
    out = tmp_path / "t.jsonl"
    cli(["generate", "--shape", "bfcl", "--jobs", "3", "--branching", "1", "--tool-median-s", "2", "--out", str(out)])
    jobs = load_jsonl(out)
    assert len(jobs) == 3 and len(jobs[0].sessions) == 2 and jobs[0].source == "synthetic-bfcl-shape"

    path = tmp_path / "x.traj.json"
    path.write_text(json.dumps(trajectory()))
    converted = tmp_path / "c.jsonl"
    cli(["convert", str(path), "--out", str(converted)])
    assert load_jsonl(converted)[0].source == "mini-swe-agent-1"


def test_replay_pauses_for_the_trace_s_stored_tool_durations_and_runs_nothing(micro_model, monkeypatch):
    """Every system must see identical tool time: the frozen durations, never a live command."""
    import subprocess
    import sys

    import layover.replay.http_replay as http_replay

    def refuse(*args, **kwargs):
        raise AssertionError(f"the replay tried to run a command: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    slept = []
    real_sleep = asyncio.sleep

    async def record(seconds):
        # The server's own event loop sleeps too; only the replay's pauses count here.
        if sys._getframe(1).f_globals.get("__name__") == http_replay.__name__:
            slept.append(seconds)
            seconds = 0
        await real_sleep(seconds)

    monkeypatch.setattr(http_replay.asyncio, "sleep", record)
    trace = TraceJob("j", 0.0, [TraceSession("j/m", [
        TraceTurn(20, 3, "pip", 41.25), TraceTurn(10, 3, "pytest", 7.5), TraceTurn(5, 2, None, 99.0),
    ])], "recorded:test")
    served = Served(micro_model, ByteTokenizer(), "micro", blocks=512)
    try:
        asyncio.run(replay([trace], served.url, "micro", CFG.vocab_size, time_scale=2.0, system_tokens=32))
    finally:
        served.close()
    # The arrival, then each pause scaled; the last turn's tool time is never waited on.
    assert slept == [0.0, 41.25 * 2.0, 7.5 * 2.0]

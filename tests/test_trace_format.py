"""The canonical trace format, its published moments, and saved trajectories."""

import json
import math
import statistics
from dataclasses import replace

import pytest

from layover.replay.__main__ import main as cli
from layover.replay.trace import (
    BFCL_SHAPE, BFCL_PUBLISHED, ContextBuilder, SWE_BENCH_PUBLISHED, SWE_BENCH_SHAPE, TOOL_VOCABULARY,
    SyntheticWorkload, TraceJob, TraceSession, TraceTurn, calibrated_workload, from_mini_swe_agent,
    generate, job_from_dict, job_to_dict, load_jsonl, lognormal_from_moments, save_jsonl, tool_category,
)


def test_lognormal_from_moments_matches_the_closed_form():
    median, sigma = lognormal_from_moments(0.925, 3.55)
    assert sigma == pytest.approx(math.sqrt(math.log(1 + (3.55 / 0.925) ** 2)))
    assert sigma == pytest.approx(1.66, abs=0.005)
    assert median == pytest.approx(0.234, abs=0.001)
    # Back to the moments the paper gives.
    assert median * math.exp(sigma**2 / 2) == pytest.approx(0.925)
    variance = (math.exp(sigma**2) - 1) * median**2 * math.exp(sigma**2)
    assert math.sqrt(variance) == pytest.approx(3.55)
    with pytest.raises(ValueError):
        lognormal_from_moments(0.0, 1.0)


def test_swe_bench_tool_shape_is_the_papers_moments_as_a_lognormal():
    assert SWE_BENCH_PUBLISHED["tool_s"] == (0.925, 3.550)
    assert BFCL_PUBLISHED["tool_s"] == (1.923, 2.133)
    assert SWE_BENCH_SHAPE.tool_s_sigma == pytest.approx(1.66, abs=0.005)
    assert SWE_BENCH_SHAPE.tool_s_median == pytest.approx(0.234, abs=0.001)


@pytest.mark.parametrize("shape, published", [(SWE_BENCH_SHAPE, SWE_BENCH_PUBLISHED), (BFCL_SHAPE, BFCL_PUBLISHED)])
def test_shapes_use_their_published_tool_moments(shape, published):
    median, sigma = lognormal_from_moments(*published["tool_s"])
    assert (shape.tool_s_median, shape.tool_s_sigma) == (median, sigma)
    jobs = generate(replace(shape, jobs=1500), 0)
    tools = [t.tool_s for j in jobs for t in j.main.turns]
    assert statistics.median(tools) == pytest.approx(median, rel=0.08)
    assert statistics.pstdev(math.log(t) for t in tools) == pytest.approx(sigma, rel=0.03)


def test_default_workload_is_the_swe_bench_table():
    assert SyntheticWorkload().tool_s_median == SWE_BENCH_SHAPE.tool_s_median
    assert SyntheticWorkload().tool_s_sigma == SWE_BENCH_SHAPE.tool_s_sigma


@pytest.mark.parametrize("shape, published", [(SWE_BENCH_SHAPE, SWE_BENCH_PUBLISHED), (BFCL_SHAPE, BFCL_PUBLISHED)])
def test_generated_tokens_and_turns_match_published_moments(shape, published):
    jobs = generate(replace(shape, jobs=4000), 1)
    tokens = [sum(t.new_tokens + t.output_tokens for t in j.main.turns) for j in jobs]
    turns = [len(j.main.turns) for j in jobs]
    assert statistics.fmean(tokens) == pytest.approx(published["tokens_per_job"][0], rel=0.05)
    assert statistics.stdev(tokens) == pytest.approx(published["tokens_per_job"][1], rel=0.1)
    assert statistics.fmean(turns) == pytest.approx(published["turns"][0], rel=0.03)
    assert statistics.stdev(turns) == pytest.approx(published["turns"][1], rel=0.1)


def test_cumulative_basis_adds_up_turn_prompts_rather_than_unique_tokens():
    jobs = generate(replace(SWE_BENCH_SHAPE, jobs=2000, tokens_basis="cumulative"), 2)
    billed, unique = [], []
    for job in jobs:
        context = total = 0
        for turn in job.main.turns:
            context += turn.new_tokens
            total += context + turn.output_tokens
            context += turn.output_tokens
        billed.append(total)
        unique.append(context)
    assert statistics.fmean(billed) == pytest.approx(SWE_BENCH_PUBLISHED["tokens_per_job"][0], rel=0.1)
    assert statistics.fmean(unique) < SWE_BENCH_PUBLISHED["tokens_per_job"][0] / 4
    with pytest.raises(ValueError):
        generate(replace(SWE_BENCH_SHAPE, jobs=1, tokens_basis="neither"), 0)


def test_job_scale_sigma_spreads_output_length_between_jobs():
    narrow = generate(replace(SWE_BENCH_SHAPE, jobs=800, turns_mean=30, turns_sd=0.1), 3)
    wide = generate(replace(SWE_BENCH_SHAPE, jobs=800, turns_mean=30, turns_sd=0.1, job_scale_sigma=1.0), 3)

    def spread(jobs):
        return statistics.pstdev(math.log(statistics.fmean(t.output_tokens for t in j.main.turns)) for j in jobs)

    assert spread(narrow) < 0.3
    assert spread(wide) == pytest.approx(math.sqrt(1.0 + spread(narrow) ** 2), rel=0.12)


def old_style_line() -> str:
    return json.dumps(
        {
            "job_id": "old", "arrival_s": 1.5, "source": "mini-swe-agent-1",
            "sessions": [{"session_id": "old/main", "parent": None, "fork_after_turn": None,
                          "turns": [{"new_tokens": 9, "output_tokens": 2, "tool_name": "bash", "tool_s": 0.5}]}],
        }
    )


def test_older_jsonl_still_loads_and_writes_without_the_new_fields(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(old_style_line() + "\n")
    job = load_jsonl(path)[0]
    assert job.shared_prefix_tokens is None and job.meta == {}
    assert job.main.turns[0].provider_new_tokens is None
    assert json.loads(old_style_line()) == job_to_dict(job)


def test_new_fields_round_trip(tmp_path):
    job = TraceJob(
        "j", 0.0, [TraceSession("j/main", [TraceTurn(5, 1, "read", 0.25, 100, 7)])], "recorded:x",
        shared_prefix_tokens=0, meta={"exit_status": "Submitted"},
    )
    path = tmp_path / "t.jsonl"
    save_jsonl([job], path)
    loaded = load_jsonl(path)[0]
    assert loaded == job
    assert loaded.shared_prefix_tokens == 0


def test_negative_prefix_is_rejected():
    raw = job_to_dict(TraceJob("j", 0.0, [TraceSession("j/main", [TraceTurn(5, 1)])]))
    raw["shared_prefix_tokens"] = -1
    with pytest.raises(ValueError):
        job_from_dict(raw)


def test_context_builder_uses_a_job_prefix_and_keeps_old_behaviour():
    old = ContextBuilder(100, seed=1, system_tokens=8)
    system = old.system_prompt()
    assert len(system) == 8 and system == ContextBuilder(100, seed=1, system_tokens=8).system_prompt()
    plain = TraceJob("a", 0.0, [TraceSession("a/main", [TraceTurn(1, 1)])])
    assert ContextBuilder(100, seed=1, system_tokens=8).prefix_for(plain) == system
    long, short = replace(plain, shared_prefix_tokens=20), replace(plain, shared_prefix_tokens=3)
    builder = ContextBuilder(100, seed=1, system_tokens=8)
    # The real prefix, whatever order jobs ask in, is one shared stream.
    assert builder.prefix_for(short) == system[:3]
    assert builder.prefix_for(long)[:8] == system
    assert len(builder.prefix_for(long)) == 20
    assert builder.prefix_for(replace(plain, shared_prefix_tokens=0)) == []


@pytest.mark.parametrize(
    "command, category",
    [
        ("cat foo.py", "read"),
        ("cat > foo.py <<'EOF'\nx = 1\nEOF", "edit"),
        ("sed -n '1,20p' foo.py", "read"),
        ("sed -i 's/a/b/' foo.py", "edit"),
        ("grep -rn foo .", "search"),
        ("find . -name '*.py'", "search"),
        ("cd /testbed && grep foo bar", "search"),
        ("FOO=1 BAR=2 python x.py", "python"),
        ("python -m pytest tests/", "test"),
        ("python tests/runtests.py auth", "test"),
        ("pytest -q", "test"),
        ("git diff", "vcs"),
        ("pip install foo", "build"),
        ("echo hello", "other"),
        ("echo hi > file", "edit"),
        ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt", "submit"),
        ("/usr/bin/python3 x.py", "python"),
        ("frobnicate --all", "other"),
        ("unbalanced 'quote", "other"),
        ("", "none"),
    ],
)
def test_tool_category(command, category):
    assert tool_category(command) == category
    assert category in TOOL_VOCABULARY


def v2_trajectory():
    return {
        "trajectory_format": "mini-swe-agent-1.1",
        "messages": [
            {"role": "system", "content": "aaaa"},
            {"role": "user", "content": "bb"},
            {"role": "assistant", "content": "ccc", "tool_calls": [
                {"function": {"name": "bash", "arguments": "{}"}}],
             "extra": {"timestamp": 5.0, "actions": [{"command": "grep x"}, {"command": "ls"}]}},
            {"role": "tool", "content": "dd", "extra": {"timestamp": 6.0}},
            {"role": "tool", "content": "eee", "extra": {"timestamp": 8.5}},
            {"role": "assistant", "content": "f", "extra": {"timestamp": 9.0, "actions": [{"command": "git diff"}]}},
            {"role": "exit", "content": "patch", "extra": {"exit_status": "Submitted"}},
        ],
    }


def test_saved_v2_trajectories_convert_with_parallel_tool_messages():
    job = from_mini_swe_agent(v2_trajectory(), len, "traj")
    first, second = job.main.turns
    assert first.new_tokens == 6 and second.new_tokens == 5
    assert first.tool_s == pytest.approx(3.5) and first.tool_name == "search"
    assert second.tool_name == "vcs"
    assert job.source == "mini-swe-agent-1.1"


def calibration_file(path, sigma=0.5, basis="cumulative"):
    path.write_text(json.dumps({"schema": "layover-calibration-1", "generator_overrides": {"job_scale_sigma": sigma, "tokens_basis": basis}}))
    return path


def test_calibrated_generation_is_labelled_synthetic_and_names_the_file(tmp_path):
    import hashlib

    path = calibration_file(tmp_path / "calibration.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    workload = calibrated_workload(SWE_BENCH_SHAPE, path)
    assert workload.job_scale_sigma == 0.5 and workload.tokens_basis == "cumulative"
    out = tmp_path / "t.jsonl"
    cli(["generate", "--jobs", "4", "--calibration", str(path), "--out", str(out)])
    jobs = load_jsonl(out)
    assert all(j.source.startswith("synthetic") and digest in j.source for j in jobs)
    with pytest.raises(SystemExit):
        cli(["generate", "--shape", "bfcl", "--calibration", str(path), "--out", str(out)])
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}")
    with pytest.raises(ValueError, match="layover-calibration-1"):
        calibrated_workload(SWE_BENCH_SHAPE, wrong)


def test_generate_without_calibration_keeps_its_plain_label(tmp_path):
    out = tmp_path / "t.jsonl"
    cli(["generate", "--jobs", "2", "--out", str(out)])
    assert all(j.source == "synthetic-swebench-shape" for j in load_jsonl(out))

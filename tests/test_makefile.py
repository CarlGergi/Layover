"""The real Makefile's guards, because a re-measured cost model strands a matrix's saved runs."""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEASURED = {"measured_on_gpu": True, "complete": True}


@pytest.fixture
def tree(tmp_path):
    shutil.copy(ROOT / "Makefile", tmp_path / "Makefile")
    log = tmp_path / "calls.log"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\necho "$@" >> "{log}"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def make(tree, target, **variables):
    args = ["make", "-s", target, f"PYTHON={tree / 'python-stub'}", *[f"{k}={v}" for k, v in variables.items()]]
    subprocess.run(args, cwd=tree, check=True, env={**os.environ, "MAKEFLAGS": ""}, capture_output=True)
    log = tree / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def put(tree, path, content):
    target = tree / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content if isinstance(content, str) else json.dumps(content))


CROSSOVERS = ["kv_crossover_plain", "kv_crossover_native_bf16", "kv_crossover"]


def test_crossover_measures_all_three_paths_when_nothing_exists(tree):
    calls = make(tree, "kv-crossover")
    assert len(calls) == 3
    assert "--out results/kv_crossover" in calls[2] and "--offload-dtype fp8" in calls[2]


def test_crossover_keeps_finished_cost_models_so_a_resume_is_not_refused(tree):
    for name in CROSSOVERS:
        put(tree, f"results/{name}/cost_model.json", MEASURED)
    assert make(tree, "kv-crossover") == []


def test_only_the_missing_crossover_runs(tree):
    for name in CROSSOVERS[:2]:
        put(tree, f"results/{name}/cost_model.json", MEASURED)
    calls = make(tree, "kv-crossover")
    assert len(calls) == 1 and "--offload-dtype fp8" in calls[0]


def test_a_cpu_cost_model_does_not_count_as_measured(tree):
    for name in CROSSOVERS:
        put(tree, f"results/{name}/cost_model.json", {"measured_on_gpu": False})
    assert len(make(tree, "kv-crossover")) == 3


def test_force_re_measures(tree):
    for name in CROSSOVERS:
        put(tree, f"results/{name}/cost_model.json", MEASURED)
    assert len(make(tree, "kv-crossover", FORCE=1)) == 3


@pytest.mark.parametrize("target,path", [
    ("decode-bandwidth", "results/decode_bandwidth/decode_bandwidth.json"),
    ("transfer-gather", "results/transfer_gather/transfer_gather.json"),
])
def test_benchmarks_skip_only_complete_gpu_output(tree, target, path):
    put(tree, path, {"measured_on_gpu": True, "complete": False})
    assert len(make(tree, target)) == 1
    (tree / "calls.log").unlink()
    put(tree, path, MEASURED)
    assert make(tree, target) == []


def test_a_recorded_trace_is_kept_because_rerunning_changes_its_hash(tree):
    put(tree, "traces/swebench_30.jsonl", json.dumps({"source": "recorded:x"}) + "\n")
    assert make(tree, "traces") == []


def test_a_pipeline_check_trace_is_not_kept(tree):
    put(tree, "traces/swebench_30.jsonl", json.dumps({"source": "pipeline-check:x"}) + "\n")
    calls = make(tree, "traces", TRACES_ARGS="--allow-iam-role")
    assert "--allow-iam-role" in calls[0] and "convert" in calls[1]


def test_reproduce_does_not_recompute_the_sweeps_calibration():
    text = (ROOT / "Makefile").read_text()
    line = next(l for l in text.splitlines() if l.startswith("reproduce:"))
    assert "calibration" not in line and "matrix" in line


def test_matrix_passes_the_chosen_plan(tree):
    calls = make(tree, "matrix", MATRIX_ARGS="--plan under-20h --max-hours 16")
    assert "--plan under-20h --max-hours 16" in calls[0]

"""Facts a re-executed trace carries must reach the benchmark report and agree with its config."""

import json
from pathlib import Path

from bench.run_matrix import trace_meta_summary
from test_kv_crossover import kv
from test_reexecute import (  # noqa: F401  shim is a fixture
    converted,
    image_of,
    log_of,
    shim,
)


def test_a_replaced_task_and_its_reason_reach_the_report(tmp_path, shim):
    shim.write_config({"exec_override": {image_of("a__a-1"): {"rc": 7}}})
    jobs = converted(tmp_path, shim, tasks=["a__a-1", "b__b-2"], reserves=["r__r-1", "r__r-2"])
    assert [j.job_id for j in jobs] == ["b__b-2", "r__r-1"]
    summary = trace_meta_summary(jobs)
    assert len(summary["replacements"]) == 1  # One string shared by every job, so the report keeps it.
    text = summary["replacements"][0]
    assert "a__a-1 (low_fidelity: return-code match" in text and "Reserves brought in: r__r-1" in text
    assert "r__r-2" not in text
    assert "Recorded tool time 10 s for the 2 tasks selected, 10 s for the set measured" in text


def test_the_exclusion_rate_and_bias_note_reach_the_report(tmp_path, shim):
    jobs = converted(tmp_path, shim, tasks=["a__a-1", "b__b-2"])
    assert trace_meta_summary(jobs)["exclusions"] == [
        '20% of 50 walked tasks excluded before selection ({"context_over_limit": 10}) at max_context 32768. The longest sessions are gone'
    ]


def test_the_install_share_is_measured_and_reaches_the_report(tmp_path, shim):
    install = json.dumps({"rc": 0, "sleep": 0.2, "cmd": "pip3 install numpy 2>&1 | tail -5"})
    grep = json.dumps({"rc": 0, "cmd": "grep -n pip setup.py"})
    jobs = converted(tmp_path, shim, tasks=["a__a-1"], commands={"a__a-1": [[(install, 0, "")], [(grep, 0, "")]]})
    records = [r for r in log_of(tmp_path, "a__a-1") if r["type"] == "command"]
    first, total = records[0]["duration_s"], sum(r["duration_s"] for r in records)
    assert trace_meta_summary(jobs)["install_share"] == [
        f"{first / total:.1%} of measured tool time ({first:.0f} of {total:.0f} s) is install or download commands"
    ]
    assert first / total > 0.5


def test_fidelity_summary_and_settings_reach_the_report(tmp_path, shim):
    jobs = converted(tmp_path, shim, tasks=["a__a-1", "b__b-2"])
    summary = trace_meta_summary(jobs)
    assert summary["run_settings"] == ["1 worker(s) at once; network on, unpinned installs resolved at run time"]
    assert "return-code match min 1.000, median 1.000 (floor 0.9)" in summary["fidelity_summary"][0]
    assert "output similarity min 1.000, median 1.000 (floor 0.5)" in summary["fidelity_summary"][0]
    assert "final patch matched in 0 of 0 checked" in summary["fidelity_summary"][0]
    assert "network on, unpinned installs resolved at run time" in summary["tool_s_basis"][0]
    assert all(j.meta["workers"] == 1 for j in jobs)


def test_recompute_is_priced_at_the_chunk_the_matrix_serves_with():
    matrix = json.loads((Path(__file__).resolve().parents[1] / "bench" / "configs" / "matrix.json").read_text())
    assert kv.parse_args([]).chunk == matrix["matched"]["prefill_chunk"]

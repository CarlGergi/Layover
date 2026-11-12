"""The profiling script on CPU: its steps run, its traces are written, and it labels itself."""

import json

from bench import profile_hot_path as profile


def test_the_cpu_run_measures_every_shape_and_says_it_measures_nothing(tmp_path, capsys):
    assert profile.main(["--device", "cpu", "--out", str(tmp_path)]) == 0
    rates = json.loads((tmp_path / "rates.json").read_text())
    plan = profile.shapes(False, False)
    assert set(rates["prefill_tokens_per_s"]) == {str(c) for c in plan["contexts"]}
    assert set(rates["decode_step_s"]) == {str(b) for b in plan["batches"]}
    assert all(v > 0 for v in [*rates["prefill_tokens_per_s"].values(), *rates["decode_step_s"].values()])
    assert rates["label"] == profile.PIPELINE_LABEL and rates["model"] == "micro"
    assert "prefill: 32 tokens after 64" in capsys.readouterr().out


def test_the_torch_trace_covers_one_prefill_chunk_and_one_decode_step(tmp_path):
    assert profile.main(["--device", "cpu", "--out", str(tmp_path), "--torch-trace"]) == 0
    for name in ("prefill", "decode"):
        assert json.loads((tmp_path / f"torch_{name}.json").read_text())["traceEvents"]
        assert "Self CPU" in (tmp_path / f"torch_{name}_top.txt").read_text()


def test_vllm_is_measured_on_a_gpu_only(tmp_path):
    assert profile.main(["--system", "vllm", "--device", "cpu", "--out", str(tmp_path)]) == 2


def test_the_gpu_shapes_match_the_matrix_chunk_and_reach_its_long_contexts():
    for quick in (False, True):
        plan = profile.shapes(True, quick)
        assert plan["chunk"] == 2048 and 0 in plan["contexts"] and 8192 in plan["contexts"]
        assert 32 in plan["batches"] and plan["repeats"] >= 3
    assert max(profile.shapes(True, False)["contexts"]) == 32768


def test_decode_can_run_through_graphs(tmp_path):
    assert profile.main(["--device", "cpu", "--out", str(tmp_path), "--cuda-graphs"]) == 0
    rates = json.loads((tmp_path / "rates.json").read_text())
    assert rates["cuda_graphs"] is True and all(v > 0 for v in rates["decode_step_s"].values())

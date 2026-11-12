"""The GPU smoke's runner on CPU: what it does when checks pass, fail, hang, or crash.

Stub processes stand in for the checks, since the runner is the point here. The checks that need
no GPU also run for real through --device cpu, and the server session runs for real on Qwen3-0.6B.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from bench import gpu_smoke as smoke
from bench.gpu_smoke import Check, Context, SmokeFailure

REPO = Path(__file__).resolve().parent.parent

OK = """
import json, sys
json.dump({"ok": True, "details": {"summary": "fine", "measured": {sys.argv[2]: 1.5}}}, open(sys.argv[1], "w"))
"""
BAD = """
import json, sys
json.dump({"ok": False, "error": "it broke"}, open(sys.argv[1], "w"))
sys.exit(1)
"""
CRASH = "import os; print('core dumped, more or less'); os._exit(3)"
SILENT = "pass"
# Copies the report as it stood when this check began.
SNAPSHOT = OK + "\nimport shutil; shutil.copy(sys.argv[3], sys.argv[3] + '.' + sys.argv[2])\n"


def noop(ctx):
    return {}


def checks_of(*names, **options):
    return tuple(Check(name, noop, options.get("timeout_s", 60), options.get("needs", {}).get(name, ()), on_cpu=True) for name in names)


class Harness:
    def __init__(self, tmp_path, device="cuda"):
        self.out = tmp_path / "out"
        self.tmp = tmp_path / "work"
        self.out.mkdir()
        self.tmp.mkdir()
        self.report_path = self.out / "smoke.json"
        self.ctx = Context(device, "m", str(smoke.DEFAULT_CONFIG), str(self.out), str(self.tmp))
        self.behaviour = {}
        self.ran = []

    def argv_for(self, check, _context, result):
        self.ran.append(check.name)
        return [sys.executable, "-c", self.behaviour.get(check.name, OK), str(result), check.name, str(self.report_path)]

    def run(self, checks, **options):
        return smoke.run_checks(checks, self.ctx, self.report_path, argv_for=self.argv_for, announce=lambda *a, **k: None, **options)


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def statuses(report):
    return {row["name"]: row["status"] for row in report["checks"]}


def test_checks_run_in_registry_order_and_measurements_are_kept(harness):
    report = harness.run(checks_of("a", "b", "c"))
    assert harness.ran == ["a", "b", "c"]
    assert report["passed"] and report["complete"]
    assert set(statuses(report).values()) == {"passed"}
    assert report["checks"][0]["seconds"] is not None and report["checks"][0]["summary"] == "fine"
    assert report["measured"] == {"a": 1.5, "b": 1.5, "c": 1.5}


def test_a_failure_does_not_stop_a_check_that_does_not_depend_on_it(harness):
    harness.behaviour["a"] = BAD
    report = harness.run(checks_of("a", "b"))
    assert harness.ran == ["a", "b"]
    assert statuses(report) == {"a": "failed", "b": "passed"}
    assert report["checks"][0]["summary"] == "it broke"
    assert not report["passed"]


def test_a_dependent_is_blocked_and_says_by_what_all_the_way_down(harness):
    harness.behaviour["a"] = BAD
    needs = {"b": ("a",), "c": ("b",), "e": ("d",)}
    report = harness.run(checks_of("a", "b", "c", "d", "e", needs=needs))
    assert harness.ran == ["a", "d", "e"]
    assert statuses(report) == {"a": "failed", "b": "blocked", "c": "blocked", "d": "passed", "e": "passed"}
    rows = {row["name"]: row for row in report["checks"]}
    assert rows["b"]["blocked_by"] == ["a"] and "blocked by a" in rows["b"]["summary"]
    assert rows["c"]["blocked_by"] == ["b"]


def test_the_report_on_disk_is_whole_after_every_check(harness):
    for name in "abc":
        harness.behaviour[name] = SNAPSHOT
    harness.run(checks_of("a", "b", "c"))
    # While a check runs, those before it are final and it and those after are pending.
    at_b = statuses(json.loads(Path(f"{harness.report_path}.b").read_text()))
    assert at_b == {"a": "passed", "b": "pending", "c": "pending"}
    at_c = statuses(json.loads(Path(f"{harness.report_path}.c").read_text()))
    assert at_c == {"a": "passed", "b": "passed", "c": "pending"}


def test_a_hanging_check_times_out_with_what_it_spawned_and_the_rest_still_run(harness, tmp_path):
    pid_file = tmp_path / "server.pid"
    # A server has a session of its own, so a group signal alone would leave it holding the GPU.
    harness.behaviour["hang"] = f"""
import subprocess, sys, time
server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
open({str(pid_file)!r}, "w").write(str(server.pid))
time.sleep(60)
"""
    checks = (Check("hang", noop, 2), Check("after", noop, 60))
    started = time.monotonic()
    report = harness.run(checks, grace_s=1)
    assert time.monotonic() - started < 30
    assert statuses(report) == {"hang": "failed", "after": "passed"}
    assert "timed out after 2 s" in report["checks"][0]["summary"]
    deadline = time.monotonic() + 10
    pid = int(pid_file.read_text())
    while not process_gone(pid):
        assert time.monotonic() < deadline, "the check's server outlived its timeout"
        time.sleep(0.05)


def process_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # A killed child of a dead parent can linger as a zombie, which holds nothing.
    return smoke_is_zombie(pid)


def smoke_is_zombie(pid):
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return state.startswith("Z") or not state


@pytest.mark.parametrize("script,needle", [(CRASH, "exited 3 with no result"), (SILENT, "exited 0 with no result")])
def test_a_check_that_leaves_no_result_failed_and_its_log_is_quoted(harness, script, needle):
    harness.behaviour["a"] = script
    report = harness.run(checks_of("a"))
    assert statuses(report) == {"a": "failed"}
    assert needle in report["checks"][0]["summary"]
    if script is CRASH:
        assert "core dumped" in report["checks"][0]["summary"]
        assert "core dumped" in (harness.out / "logs" / "a.log").read_text()


def test_what_is_left_after_the_total_limit_is_blocked_and_says_why(harness):
    report = harness.run(checks_of("a", "b"), total_limit_s=0)
    assert harness.ran == []
    assert statuses(report) == {"a": "blocked", "b": "blocked"}
    assert "limit" in report["checks"][0]["summary"]
    assert not report["passed"]


def test_a_check_gets_only_what_is_left_of_the_total_limit(harness):
    harness.behaviour["slow"] = "import time; time.sleep(30)"
    started = time.monotonic()
    report = harness.run((Check("slow", noop, 60),), total_limit_s=1, grace_s=1)
    assert time.monotonic() - started < 15
    assert "timed out after 1 s" in report["checks"][0]["summary"]


def test_on_cpu_what_needs_a_gpu_is_not_applicable_and_blocks_nothing(tmp_path):
    harness = Harness(tmp_path, device="cpu")
    report = harness.run((Check("gpu_only", noop, 60), Check("host", noop, 60, ("gpu_only",), on_cpu=True)))
    assert harness.ran == ["host"]
    assert statuses(report) == {"gpu_only": "not_applicable", "host": "passed"}
    assert report["passed"] and report["label"] == smoke.PIPELINE_LABEL


def test_a_gpu_run_carries_no_pipeline_label(harness):
    assert harness.run(checks_of("a"))["label"] is None


def test_every_real_dependency_names_an_earlier_check():
    names = [check.name for check in smoke.CHECKS]
    assert len(set(names)) == len(names)
    for index, check in enumerate(smoke.CHECKS):
        assert set(check.needs) <= set(names[:index]), check.name
    assert sum(check.timeout_s for check in smoke.CHECKS) >= smoke.TOTAL_LIMIT_S


def test_the_quick_gpu_tests_have_a_stated_limit_and_exist():
    assert smoke.GPU_TESTS_LIMIT_S <= 240
    for node in smoke.GPU_TESTS:
        path, name = node.split("::")
        function = name.split("[")[0]
        text = (REPO / path).read_text()
        assert re.search(rf"^(@pytest[^\n]*\n)*def {function}\(", text, re.M), node
        if "[" in name:
            assert name.split("[")[1].rstrip("]") in text, node


def fake_pytest(monkeypatch, stdout, returncode=0):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    monkeypatch.setattr(smoke.subprocess, "run", run)
    return calls


def context(tmp_path, device="cuda"):
    return Context(device, "Qwen/Qwen3-4B", str(smoke.DEFAULT_CONFIG), str(tmp_path), str(tmp_path))


def test_quick_gpu_tests_run_as_the_gpu_suite_on_the_real_model(monkeypatch, tmp_path):
    calls = fake_pytest(monkeypatch, f"{len(smoke.GPU_TESTS)} passed in 10.0s\n")
    assert "passed" in smoke.check_gpu_tests(context(tmp_path))["summary"]
    argv = calls[0]
    assert "--gpu" in argv and argv[argv.index("--checkpoint") + 1] == "Qwen/Qwen3-4B"
    assert set(smoke.GPU_TESTS) <= set(argv)


@pytest.mark.parametrize(
    "stdout,code",
    [("7 passed, 1 skipped in 3s\n", 0), ("1 failed, 7 passed in 3s\n", 1), ("no tests ran\n", 4), ("", 0)],
)
def test_a_skip_or_a_missing_test_is_not_a_pass(monkeypatch, tmp_path, stdout, code):
    fake_pytest(monkeypatch, stdout, code)
    with pytest.raises(SmokeFailure):
        smoke.check_gpu_tests(context(tmp_path))


def test_quick_gpu_tests_that_overrun_say_their_limit(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(smoke.subprocess, "run", run)
    with pytest.raises(SmokeFailure, match=f"limit of {smoke.GPU_TESTS_LIMIT_S} s"):
        smoke.check_gpu_tests(context(tmp_path))


def test_environment_refuses_the_interpreter_before_anything_else(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    with pytest.raises(SmokeFailure, match="TRITON_INTERPRET"):
        smoke.check_environment(context(tmp_path))


def test_environment_needs_a_gpu_and_a_gpu_build_of_torch(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SmokeFailure, match="cannot see a CUDA GPU"):
        smoke.check_environment(context(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(SmokeFailure, match="CPU-only build"):
        smoke.check_environment(context(tmp_path))


def test_parity_on_real_weights_and_that_it_can_fail(monkeypatch, tmp_path, checkpoint):
    """The check itself, on the small checkpoint and the CPU."""
    from layover.model import Qwen3

    ctx = Context("cpu", checkpoint, str(smoke.DEFAULT_CONFIG), str(tmp_path), str(tmp_path))
    assert smoke.check_reference_parity(ctx)["worst_gap"] <= 1
    real = Qwen3.decode
    monkeypatch.setattr(Qwen3, "decode", lambda self, *args: real(self, *args) + 5.0)
    with pytest.raises(SmokeFailure, match="decode step 0"):
        smoke.check_reference_parity(ctx)


def test_triton_on_real_kv_and_that_the_comparison_can_fail(monkeypatch, checkpoint):
    """The comparison itself, in Triton's interpreter on the small checkpoint, where Triton installs."""
    pytest.importorskip("triton")
    from layover.kernels import triton_decode
    from layover.weights import load_model, resolve_checkpoint

    model = load_model(str(resolve_checkpoint(checkpoint)), device="cpu", dtype=__import__("torch").bfloat16)
    attention, logits = smoke.triton_gaps(model.cfg, model, "cpu", 32, 32)
    assert attention <= 1 and logits <= 1
    real = triton_decode.paged_decode_attention_triton
    monkeypatch.setattr(triton_decode, "paged_decode_attention_triton", lambda *args: real(*args) * 1.25)
    attention, logits = smoke.triton_gaps(model.cfg, model, "cpu", 32, 32)
    assert attention > 1 and logits > 1


def test_the_server_gets_the_matrix_flags_and_the_smokes_cost_model(tmp_path):
    system, _ = smoke.matrix_system(context(tmp_path))
    command = system.command("127.0.0.1", 9)
    flags = dict(zip(command, command[1:]))
    matched = real_config()["matched"]
    assert flags["--policy"] == "combined" and flags["--transfer"] == "native" and flags["--offload-dtype"] == "fp8"
    assert flags["--decode-backend"] == "triton" and flags["--model"] == matched["model"]
    assert float(flags["--pool-gib"]) == matched["kv_gib"] and flags["--cost-model"] == str(tmp_path / "crossover" / "cost_model.json")
    assert flags["--prefill-chunk"] == str(matched["prefill_chunk"]) and "--no-prefix-cache" not in command


def test_the_server_runs_at_the_settings_of_the_plan_being_priced(tmp_path):
    def cpu_tier(plan):
        ctx = Context("cuda", "Qwen/Qwen3-4B", str(smoke.DEFAULT_CONFIG), str(tmp_path), str(tmp_path), plan)
        command = smoke.matrix_system(ctx)[0].command("127.0.0.1", 9)
        return float(command[command.index("--cpu-tier-gib") + 1])

    assert cpu_tier("under-20h") == real_config()["matched"]["cpu_offload_gib"] == 32.0
    assert cpu_tier("runpod-l4") == 24.0


def test_a_matrix_off_the_native_path_is_refused(tmp_path):
    config = real_config()
    config["layover"]["transfer"] = "plain"
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SmokeFailure, match="native path"):
        smoke.matrix_system(Context("cuda", "m", str(path), str(tmp_path), str(tmp_path)))


# ---- the numbers it hands the plan ------------------------------------------------------------------

MEASURED = {
    "decode_step_s": {"1": 0.02, "32": 0.08},
    "decode_context_tokens": 8192,
    "crossover_prefill_tokens_per_s": 9000.0,
    "server_prefill_tokens_per_s": 14000.0,
    "server_startup_s": 70.0,
}


def real_config():
    return json.loads(smoke.DEFAULT_CONFIG.read_text())


def test_measured_rates_are_what_run_matrix_accepts_and_come_from_the_steps(tmp_path):
    from bench.run_matrix import apply_measured_rates

    rates = smoke.measured_rates(MEASURED, on_gpu=True)
    assert rates["estimate_assumptions"] == pytest.approx(
        {"prefill_tokens_per_s": 14000.0, "decode_tokens_per_s_per_stream": 50.0, "decode_tokens_per_s_total": 400.0, "startup_s": 70.0}
    )
    assert "PIPELINE" not in rates["source"] and "8192" in rates["source"]
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(rates))
    config = real_config()
    source = apply_measured_rates(config, path)
    assert "bench/gpu_smoke.py" in source
    assert config["estimate_assumptions"]["prefill_tokens_per_s"] == pytest.approx(14000.0)
    assert config["estimate_assumptions"]["decode_tokens_per_s_total"] == pytest.approx(400.0)


def test_without_the_server_prefill_comes_from_the_crossover():
    measured = {k: v for k, v in MEASURED.items() if not k.startswith("server_")}
    rates = smoke.measured_rates(measured, on_gpu=False)
    assert rates["estimate_assumptions"]["prefill_tokens_per_s"] == 9000.0
    assert "startup_s" not in rates["estimate_assumptions"]
    assert rates["source"].startswith(smoke.PIPELINE_LABEL)


def test_measured_rates_check_refuses_a_file_the_reader_would(tmp_path):
    from bench.run_matrix import Refused

    measured = {**MEASURED, "decode_step_s": {"1": float("inf"), "32": float("inf")}}
    ctx = replace(context(tmp_path), measured=measured)
    with pytest.raises(Refused):
        smoke.check_measured_rates(ctx)


def test_refined_hours_are_parsed_or_fail_loudly():
    text = "  estimated total: 12.5 hours, up to 19.25 hours if every turn recomputes its context\n"
    assert smoke.parse_refined_hours(text) == (12.5, 19.25)
    with pytest.raises(SmokeFailure, match="no estimate"):
        smoke.parse_refined_hours("nothing of the kind")


def test_refine_plan_asks_run_matrix_with_the_rates_and_leaves_a_line_for_the_shell(monkeypatch, tmp_path):
    output = "  token rates MEASURED, x\n  estimated total: 12.5 hours, up to 19.25 hours\n  Not included: retries\nchatter\n"
    calls = fake_pytest(monkeypatch, output)
    details = smoke.check_refine_plan(context(tmp_path))
    argv = calls[0]
    assert argv[argv.index("--plan") + 1] == "final" and "--dry-run" in argv
    assert argv[argv.index("--measured-rates") + 1] == str(tmp_path / "measured_rates.json")
    assert details["low_hours"] == 12.5 and details["high_hours"] == 19.25
    written = (tmp_path / "refined_hours.txt").read_text().splitlines()
    assert written[0] == "12.5 19.25"
    assert "token rates MEASURED, x" in written[1] and "chatter" not in "\n".join(written)


def test_the_plan_to_price_comes_from_the_argument_then_the_environment_then_the_default(monkeypatch):
    monkeypatch.delenv("SMOKE_PLAN", raising=False)
    assert smoke.parse_args([]).plan == "final"
    monkeypatch.setenv("SMOKE_PLAN", "")
    assert smoke.parse_args([]).plan == "final"
    monkeypatch.setenv("SMOKE_PLAN", "first-pass")
    assert smoke.parse_args([]).plan == "first-pass"
    assert smoke.parse_args(["--plan", "three-loads-lean"]).plan == "three-loads-lean"


def test_refine_plan_prices_the_plan_it_was_given(monkeypatch, tmp_path):
    calls = fake_pytest(monkeypatch, "  estimated total: 1.0 hours, up to 2.0 hours\n")
    ctx = context(tmp_path)
    ctx.plan = "under-20h"
    assert smoke.check_refine_plan(ctx)["plan"] == "under-20h"
    assert calls[0][calls[0].index("--plan") + 1] == "under-20h"


def test_refine_plan_fails_when_the_dry_run_does(monkeypatch, tmp_path):
    fake_pytest(monkeypatch, "", 2)
    with pytest.raises(SmokeFailure, match="exited 2"):
        smoke.check_refine_plan(context(tmp_path))


def test_refined_hours_through_the_real_dry_run_of_the_real_plan(tmp_path):
    (tmp_path / "measured_rates.json").write_text(json.dumps(smoke.measured_rates(MEASURED, True)))
    details = smoke.check_refine_plan(context(tmp_path))
    assert 0 < details["low_hours"] <= details["high_hours"]
    assert any("MEASURED" in line for line in details["lines"])
    assert any("Budget:" in line for line in details["lines"])


def test_the_tiny_benchmark_stays_short_and_in_the_temporary_directory(tmp_path):
    argv = smoke.crossover_argv(context(tmp_path))
    assert argv[argv.index("--out") + 1] == str(tmp_path / "crossover")
    tokens = argv[argv.index("--tokens") + 1 : argv.index("--chunk")]
    assert len(tokens) == 2 and max(map(int, tokens)) <= 2048
    assert argv[argv.index("--transfer") + 1] == "native" and argv[argv.index("--offload-dtype") + 1] == "fp8"
    # The matrix refuses a cost model measured at another chunk, so the server here must not differ.
    assert argv[argv.index("--chunk") + 1] == str(real_config()["matched"]["prefill_chunk"])


# ---- what the session must show in /metrics ---------------------------------------------------------

GOOD_KEEP = {"kept": 1, "swapped": 0}
GOOD_SWAP = {"swapped": 1, "ttl_expired": 1}
GOOD_RESUME = {
    "restored_tokens": 2051, "restores_avoided": 0, "dropped": 0, "bytes_restored": 1000, "host_pins": 1,
    "cpu_tier_peak_blocks": 10, "bytes_offloaded": 5000,
}


def problems(keep=None, swap=None, resume=None):
    return smoke.session_problems({**GOOD_KEEP, **(keep or {})}, {**GOOD_SWAP, **(swap or {})}, {**GOOD_RESUME, **(resume or {})}, 500, 2032)


def test_a_good_session_has_no_problems():
    assert problems() == []


@pytest.mark.parametrize(
    "changes,needle",
    [
        (dict(keep={"kept": 0}), "not kept"),
        (dict(keep={"swapped": 1}), "not kept"),
        (dict(swap={"swapped": 0}), "time-to-live"),
        (dict(swap={"ttl_expired": 0}), "time-to-live"),
        # The radix tree served the resume: nothing proves the copy back works.
        (dict(resume={"restored_tokens": 0, "restores_avoided": 1}), "restores_avoided=1"),
        (dict(resume={"bytes_restored": 0}), "no bytes"),
        (dict(resume={"host_pins": 0}), "not the native path"),
        # BF16 moved 9,184 bytes a block where FP8 records hold about half.
        (dict(resume={"bytes_offloaded": 91840}), "FP8"),
        (dict(resume={"bytes_offloaded": 0, "cpu_tier_peak_blocks": 0}), "FP8"),
    ],
)
def test_each_way_a_session_can_go_wrong_is_named(changes, needle):
    found = problems(**changes)
    assert found and any(needle in problem for problem in found), found


def test_the_session_on_a_real_server_goes_through_keep_swap_and_restore(tmp_path, checkpoint):
    """Real weights, the real server through its real flags, and real /metrics, on the host build."""
    from bench.systems import MatchedSettings, build_system, offload_record_bytes
    from layover.config import ModelConfig

    cost_model = tmp_path / "cost_model.json"
    cost_model.write_text(json.dumps({
        "swap_out_s_coef": [0, 1e-6], "swap_in_s_coef": [0, 1e-6], "recompute_s_coef": [0, 1e-3, 0],
        "fp8_swap_out_s_coef": [0, 5e-7], "fp8_swap_in_s_coef": [0, 5e-7], "measured_on_gpu": False, "transfer": "native",
    }))
    settings = MatchedSettings(checkpoint, "cpu", "bf16", 512, 0.05, 0.05, 4, 64)
    cfg = ModelConfig.from_pretrained(checkpoint)
    spec = {"name": "t", "kind": "layover", "policy": "combined", "cost_model": str(cost_model), "ttl_cap_s": 1.0, "transfer": "native", "offload_dtype": "fp8"}
    system = build_system(spec, settings, cfg.kv_bytes_per_token("bf16"), cfg)
    limits = {"startup_timeout_s": 180, "stop_timeout_s": 15, "request_timeout_s": 180, "gpu_release_timeout_s": 10}
    tolerances = {"kv_capacity_frac": 0.02, "gpu_memory_gib": 3.0, "gpu_idle_gib": 1.0}
    session = smoke.serve_one_session(system, limits, tolerances, tmp_path / "server.log", cfg.vocab_size, 64, 256, offload_record_bytes(cfg, "fp8"))
    resume = session["after_resume"]
    assert resume["kept"] == 1 and resume["swapped"] == 1 and resume["ttl_expired"] == 1
    assert resume["restored_tokens"] >= 48 and resume["restores_avoided"] == 0
    assert session["cached_tokens"] >= 48
    assert session["startup_s"] > 0 and session["filled_tokens"] >= resume["blocks_total"] * 16
    # With the wrong record size it is the same session, and it is reported.
    wrong = smoke.session_problems(session["after_keep"], session["after_swap"], resume, offload_record_bytes(cfg, None), 48)
    assert any("FP8" in problem for problem in wrong)


# ---- the command line --------------------------------------------------------------------------------


def run_cli(*args, env=None, timeout=240):
    return subprocess.run(
        [sys.executable, str(REPO / "bench" / "gpu_smoke.py"), *args], capture_output=True, text=True, timeout=timeout, cwd=REPO,
        env={**os.environ, **(env or {})},
    )


def main_with(tmp_path, monkeypatch, behaviour, checks):
    harness = Harness.__new__(Harness)
    harness.behaviour, harness.ran = behaviour, []
    harness.report_path = tmp_path / "smoke.json"
    monkeypatch.setattr(smoke.signal, "signal", lambda *_: None)
    return smoke.main(["--device", "cuda", "--model", "m", "--out", str(tmp_path)], checks=checks, argv_for=harness.argv_for)


def test_main_exits_zero_when_all_pass_and_prints_a_table(tmp_path, monkeypatch, capsys):
    assert main_with(tmp_path, monkeypatch, {}, checks_of("a", "b")) == 0
    printed = capsys.readouterr().out
    assert re.search(r"^a\s+passed", printed, re.M) and re.search(r"^b\s+passed", printed, re.M)


def test_main_exits_one_on_a_failure_or_a_block(tmp_path, monkeypatch, capsys):
    code = main_with(tmp_path, monkeypatch, {"a": BAD}, checks_of("a", "b", needs={"b": ("a",)}))
    assert code == 1
    printed = capsys.readouterr().out
    assert re.search(r"^a\s+failed", printed, re.M) and re.search(r"^b\s+blocked", printed, re.M)


def test_main_removes_its_temporary_data_even_when_a_check_fails(tmp_path, monkeypatch):
    work = tmp_path / "tmpdir"
    work.mkdir()
    monkeypatch.setattr(smoke.tempfile, "tempdir", str(work))
    out = tmp_path / "out"
    out.mkdir()
    assert main_with(out, monkeypatch, {"a": BAD}, checks_of("a")) == 1
    assert list(work.iterdir()) == []


def test_main_does_not_leave_an_earlier_runs_hours_looking_current(tmp_path, monkeypatch):
    for stale in ("measured_rates.json", "refined_hours.txt"):
        (tmp_path / stale).write_text("old")
    main_with(tmp_path, monkeypatch, {"a": BAD}, checks_of("a"))
    assert not (tmp_path / "measured_rates.json").exists() and not (tmp_path / "refined_hours.txt").exists()


def test_a_worker_reports_each_way_a_check_can_end(tmp_path):
    def fails(ctx):
        raise SmokeFailure("named the problem")

    def surprises(ctx):
        raise KeyError("oops")

    def refuses(ctx):
        raise SystemExit("--device cuda requested but no CUDA GPU is available")

    checks = (Check("fails", fails, 1), Check("surprises", surprises, 1), Check("refuses", refuses, 1), Check("fine", lambda ctx: {"summary": "s"}, 1))
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps({"device": "cpu", "model": "m", "config": "c", "out": "o", "tmp": "t"}))
    result = tmp_path / "result.json"
    expected = {"fails": "named the problem", "surprises": "KeyError", "refuses": "no CUDA GPU"}
    for name, needle in expected.items():
        assert smoke.worker_main(name, str(context_path), str(result), checks) == 1
        answer = json.loads(result.read_text())
        assert answer["ok"] is False and needle in answer["error"]
    assert smoke.worker_main("fine", str(context_path), str(result), checks) == 0
    assert json.loads(result.read_text()) == {"ok": True, "details": {"summary": "s"}}


@pytest.mark.skipif(__import__("torch").cuda.is_available(), reason="this machine has a GPU")
def test_it_refuses_without_a_gpu_and_touches_nothing(tmp_path):
    result = run_cli("--out", str(tmp_path / "out"))
    assert result.returncode == 2
    assert "--device cpu" in result.stderr
    assert not (tmp_path / "out").exists()


def test_the_pipeline_check_runs_the_real_checks_and_says_what_it_is(tmp_path):
    work = tmp_path / "tmpdir"
    work.mkdir()
    out = tmp_path / "out"
    result = run_cli("--device", "cpu", "--out", str(out), env={"TMPDIR": str(work)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert smoke.PIPELINE_LABEL in result.stdout
    report = json.loads((out / "smoke.json").read_text())
    assert report["label"] == smoke.PIPELINE_LABEL and report["model"] == "micro" and report["passed"]
    found = statuses(report)
    assert found == {
        "environment": "not_applicable", "reference_parity": "not_applicable", "triton_decode": "not_applicable",
        "decode_rates": "passed", "native_transfer": "passed", "crossover": "passed",
        "server_session": "not_applicable", "gpu_tests": "not_applicable",
        "measured_rates": "passed", "refine_plan": "not_applicable",
    }
    rates = json.loads((out / "measured_rates.json").read_text())
    assert rates["source"].startswith(smoke.PIPELINE_LABEL)
    assert not (out / "refined_hours.txt").exists()
    assert list(work.iterdir()) == []
    for row in report["checks"]:
        if row["status"] == "passed":
            assert (out / "logs" / f"{row['name']}.log").exists()

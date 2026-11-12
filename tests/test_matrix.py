"""The benchmark matrix on CPU: commands, budgets, resume, retries, statistics, and a real pipeline.

Run as a script, this file is a stand-in baseline server for the pipeline check. Its
response format follows what vLLM 0.33.0 and SGLang 0.5.23 document, not Layover's: a
prompt_token_ids field on the first chunk, a usage chunk with no choices, and token ids only
when asked for through return_token_ids.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def run_stub(port: int, model: str, text_only: bool, fail_turns: int = 0, cached_tokens: int | None = None, short_by: int = 0) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def reply(self, body: bytes, kind: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(b"")
            elif self.path == "/v1/models":
                self.reply(json.dumps({"data": [{"id": model}]}).encode())
            elif self.path == "/metrics":
                text = 'stub_requests_total{engine="0"} 3\nstub_requests_total{engine="1"} 4\n# HELP x\n'
                self.reply(text.encode(), "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(event) -> None:
                self.wfile.write(f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            wants_ids = request.get("return_token_ids") and not text_only
            for index in range(request["max_tokens"] - short_by):
                choice = {"index": 0, "text": " stub", "finish_reason": None}
                if wants_ids:
                    choice["token_ids"] = [(index * 7 + len(request["prompt"])) % 100]
                    if index == 0:
                        choice["prompt_token_ids"] = request["prompt"]
                send({"id": "x", "choices": [choice]})
            send({"id": "x", "choices": [{"index": 0, "text": "", "finish_reason": "length"}]})
            if (request.get("stream_options") or {}).get("include_usage"):
                usage = {"prompt_tokens": len(request["prompt"]), "completion_tokens": request["max_tokens"]}
                if cached_tokens is not None:
                    usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
                send({"id": "x", "choices": [], "usage": usage})
            send("[DONE]")
            self.close_connection = True

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub-port", type=int, required=True)
    parser.add_argument("--stub-model", required=True)
    parser.add_argument("--stub-text-only", action="store_true")
    parser.add_argument("--stub-cached-tokens", type=int)
    parser.add_argument("--stub-short-by", type=int, default=0)
    options = parser.parse_args()
    run_stub(options.stub_port, options.stub_model, options.stub_text_only, cached_tokens=options.stub_cached_tokens, short_by=options.stub_short_by)
    sys.exit(0)

import asyncio  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from bench import report, run_matrix  # noqa: E402
from bench.gpu_sampler import GIB, PCIE_NOTE, GpuReading, GpuSampler, StubBackend  # noqa: E402
from bench.run_matrix import Cell, MatrixContext, Refused, Workload  # noqa: E402
from bench.systems import (  # noqa: E402
    SGLANG_VERSION, VLLM_VERSION, BudgetMismatch, CommandSystem, GpuNotReleased, LayoverSystem, ManagedServer,
    MatchedSettings, ServerFailed, SglangSystem, VllmSystem, _descendants, build_system, check_gpu_memory,
    check_kv_capacity, parse_prometheus, process_running, running_commands, translate, wait_gpu_released,
)
from layover.config import ModelConfig  # noqa: E402
from layover.replay.http_replay import job_tool_s, replay  # noqa: E402
from layover.replay.results import ReplayResult, percentile, summarize  # noqa: E402
from layover.replay.trace import SyntheticWorkload, TraceJob, TraceSession, TraceTurn, generate, save_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SETTINGS = MatchedSettings("Qwen/Qwen3-4B", "cuda", "bf16", 40960, 12.0, 32.0, 32, 2048, 0)
# 36 layers, 8 KV heads, head_dim 128, BF16.
TOKEN_BYTES = 147456


def flags(command: list[str]) -> dict[str, str | None]:
    """Parse a command's flags, a flag with no value mapping to None."""
    parsed: dict[str, str | None] = {}
    index = 0
    while index < len(command):
        if command[index].startswith("--"):
            has_value = index + 1 < len(command) and not command[index + 1].startswith("--")
            parsed[command[index]] = command[index + 1] if has_value else None
            index += 2 if has_value else 1
        else:
            index += 1
    return parsed


def test_vllm_command_at_the_checked_version():
    command = VllmSystem("vllm", SETTINGS, TOKEN_BYTES, ".venvs/vllm").command("127.0.0.1", 9000)
    assert command[0].endswith(".venvs/vllm/bin/vllm") and command[1:3] == ["serve", "Qwen/Qwen3-4B"]
    assert "--model" not in command
    assert flags(command) == {
        "--served-model-name": "Qwen/Qwen3-4B", "--host": "127.0.0.1", "--port": "9000",
        "--dtype": "bfloat16", "--kv-cache-dtype": "bfloat16", "--max-model-len": "40960",
        "--kv-cache-memory-bytes": str(12 * GIB), "--enable-prefix-caching": None,
        "--kv-offloading-backend": "native", "--kv-offloading-size": "32", "--scheduling-policy": "fcfs",
        "--max-num-seqs": "32", "--enable-chunked-prefill": None, "--max-num-batched-tokens": "2080",
        "--stream-interval": "1", "--generation-config": "vllm", "--enable-prompt-tokens-details": None,
        "--disable-uvicorn-access-log": None, "--seed": "0",
    }


def test_sglang_command_at_the_checked_version():
    command = SglangSystem("sglang", SETTINGS, TOKEN_BYTES, ".venvs/sglang").command("127.0.0.1", 9000)
    assert command[0].endswith(".venvs/sglang/bin/sglang") and command[1] == "serve"
    assert flags(command) == {
        "--model-path": "Qwen/Qwen3-4B", "--served-model-name": "Qwen/Qwen3-4B", "--host": "127.0.0.1", "--port": "9000",
        "--dtype": "bfloat16", "--kv-cache-dtype": "bf16", "--context-length": "40960",
        "--max-total-tokens": str(12 * GIB // TOKEN_BYTES), "--mem-fraction-static": "0.9",
        "--enable-hierarchical-cache": None, "--hicache-ratio": repr(32.0 / 12.0), "--schedule-policy": "fcfs",
        "--max-running-requests": "32", "--chunked-prefill-size": "2048", "--stream-interval": "1",
        "--sampling-defaults": "openai", "--enable-metrics": None, "--enable-cache-report": None, "--random-seed": "0",
    }


def test_layover_command_wires_backends_only_when_configured():
    plain = LayoverSystem("l", SETTINGS, TOKEN_BYTES, "swap", "/c.json").command("127.0.0.1", 9000)
    assert plain[:3] == [sys.executable, "-m", "layover.server"]
    parsed = flags(plain)
    assert parsed["--pool-gib"] == "12" and parsed["--cpu-tier-gib"] == "32" and parsed["--max-batch"] == "32"
    assert parsed["--policy"] == "swap" and parsed["--cost-model"] == "/c.json" and parsed["--prefill-chunk"] == "2048"
    assert parsed["--device"] == "cuda" and parsed["--dtype"] == "bf16"
    assert not {"--decode-backend", "--transfer", "--offload-dtype"} & set(parsed)
    full = LayoverSystem("l", SETTINGS, TOKEN_BYTES, "swap", "/c.json", 30.0, "triton", "native", "fp8").command("h", 1)
    parsed = flags(full)
    assert (parsed["--decode-backend"], parsed["--transfer"], parsed["--offload-dtype"]) == ("triton", "native", "fp8")


def test_only_layover_gets_hints():
    assert LayoverSystem("l", SETTINGS, TOKEN_BYTES, "swap", "/c").hints
    assert not VllmSystem("v", SETTINGS, TOKEN_BYTES).hints and not SglangSystem("s", SETTINGS, TOKEN_BYTES).hints
    assert not CommandSystem("c", SETTINGS, TOKEN_BYTES, ["x"]).hints


def test_every_system_gets_the_same_kv_budget():
    translations = {kind: translate(kind, SETTINGS, TOKEN_BYTES) for kind in ("vllm", "sglang", "layover")}
    budget_tokens = 12 * GIB / TOKEN_BYTES
    # A block is 16 tokens, so nobody can be more than a block under budget.
    assert all(budget_tokens - 16 <= t.kv_tokens_expected <= budget_tokens for t in translations.values())
    assert int(translations["vllm"].kv_value) == 12 * GIB
    assert int(translations["sglang"].kv_value) * TOKEN_BYTES <= 12 * GIB
    assert float(translations["layover"].kv_value) == 12.0


def test_every_offloading_system_gets_the_same_cpu_tier():
    translation = translate("sglang", SETTINGS, TOKEN_BYTES)
    assert translation.cpu_flag == "--hicache-ratio" and translation.cpu_gib_effective == 32
    # What SGLang 0.5.23's pool_host/base.py does with these flags: ratio x device tokens, one page more.
    page_size = 64
    device_tokens = int(translation.kv_value)
    pages = int(device_tokens * float(translation.cpu_value)) // page_size + 1
    assert pages * page_size * TOKEN_BYTES / GIB == pytest.approx(32, abs=0.01)
    assert translate("vllm", SETTINGS, TOKEN_BYTES).cpu_value == "32"
    assert translate("layover", SETTINGS, TOKEN_BYTES).cpu_gib_effective == 32
    # The published config does not bend this per system: there is one budget.
    assert run_matrix.load_config(REPO / "bench/configs/matrix.json")["matched"]["cpu_offload_gib"] == 32.0


def test_sglang_tier_is_read_back_from_the_bytes_it_logged():
    system = SglangSystem("s", SETTINGS, TOKEN_BYTES)
    line = "INFO Allocating  hierarchical KV host pool: 233024 tokens, 34.36 GB host memory."
    assert system.reported_cpu_tier_gib(None, line) == pytest.approx(34.36e9 / GIB)
    # The old whole-gigabyte flag would be caught by the same check: 31.66 GiB is off by over 1%.
    with pytest.raises(BudgetMismatch, match="31.66"):
        run_matrix.check_cpu_tier("s", system.reported_cpu_tier_gib(None, line.replace("34.36", "34.00")), 32.0, 0.01)
    assert system.reported_cpu_tier_gib(None, "nothing here") is None


def test_the_simulated_tier_holds_what_the_server_would_size(qwen3_4b_config):
    """Sized in the same bytes: 32 GiB of FP8 records is about twice as many blocks as of BF16."""
    from layover.server import cpu_tier_capacity_blocks

    class Cache:
        cfg = qwen3_4b_config
        block_tokens = 16
        block_bytes = 16 * qwen3_4b_config.kv_bytes_per_token("bf16")

    for dtype, fp8 in (("fp8", True), ("bf16", False)):
        simulated = int(32.0 * GIB / run_matrix.offload_record_bytes(qwen3_4b_config, dtype))
        assert simulated == cpu_tier_capacity_blocks(Cache(), 32.0, fp8)
    assert int(32.0 * GIB / run_matrix.offload_record_bytes(qwen3_4b_config, "fp8")) == 29098
    assert int(32.0 * GIB / run_matrix.offload_record_bytes(qwen3_4b_config, "bf16")) == 14563
    captured = {}

    def fake_simulate(jobs, costs, pool, agent, *_, **__):
        captured["blocks"] = agent.cpu_capacity_blocks
        raise StopIteration

    import layover.replay.simulate as simulate_module

    original = simulate_module.simulate
    simulate_module.simulate = fake_simulate
    try:
        config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
        with pytest.raises(StopIteration):
            run_matrix.simulate_point(config, SETTINGS, qwen3_4b_config, [job()], 0, 1.0, 0, ("combined",))
    finally:
        simulate_module.simulate = original
    assert captured["blocks"] == 29098


def test_vllm_batch_budget_makes_room_for_the_decode_batch():
    assert translate("vllm", SETTINGS, TOKEN_BYTES).prefill_value == SETTINGS.prefill_chunk + SETTINGS.max_running
    assert translate("sglang", SETTINGS, TOKEN_BYTES).prefill_value == SETTINGS.prefill_chunk


def test_unknown_system_kind_is_rejected():
    with pytest.raises(ValueError):
        translate("tensorrt", SETTINGS, TOKEN_BYTES)
    with pytest.raises(ValueError):
        build_system({"name": "x", "kind": "tensorrt"}, SETTINGS, TOKEN_BYTES)


def test_kv_capacity_check():
    assert check_kv_capacity("s", 1000, 1000, 0.02)["reported_kv_tokens"] == 1000
    check_kv_capacity("s", 1019, 1000, 0.02)
    with pytest.raises(BudgetMismatch, match="did not report"):
        check_kv_capacity("s", None, 1000, 0.02)
    with pytest.raises(BudgetMismatch, match="1,100"):
        check_kv_capacity("s", 1100, 1000, 0.02)
    with pytest.raises(BudgetMismatch):
        check_kv_capacity("s", 900, 1000, 0.02)


def test_gpu_memory_check_fails_in_both_directions():
    kv, weights, idle = 12 * GIB, 8 * GIB, GIB // 2
    within = check_gpu_memory("s", idle + kv + weights + GIB, idle, weights, kv, 3.0)
    assert within["gpu_deviation_gib"] == pytest.approx(1.0) and within["gpu_expected_gib"] == pytest.approx(20.0)
    with pytest.raises(BudgetMismatch, match="off by \\+4.00"):
        check_gpu_memory("s", idle + kv + weights + 4 * GIB, idle, weights, kv, 3.0)
    # A pool promised and not allocated is as wrong as one too large.
    with pytest.raises(BudgetMismatch, match="off by -4.00"):
        check_gpu_memory("s", idle + kv + weights - 4 * GIB, idle, weights, kv, 3.0)


def reading(used_gib: float, **kwargs) -> GpuReading:
    return GpuReading(int(used_gib * GIB), 24 * GIB, **kwargs)


def test_wait_gpu_released():
    backend = StubBackend([reading(9), reading(5), reading(0.5)])
    assert wait_gpu_released(backend, GIB, timeout_s=5, poll_s=0) == int(0.5 * GIB)
    with pytest.raises(GpuNotReleased, match="still holds 9.00"):
        wait_gpu_released(StubBackend([reading(9)]), GIB, timeout_s=0, poll_s=0)


def test_sampler_integrates_pcie_by_hand():
    clock = iter([0.0, 1.0, 3.0])
    backend = StubBackend(
        [
            reading(6, gpu_util_pct=10, memory_busy_pct=20, pcie_tx_kib_s=1000.0, pcie_rx_kib_s=0.0),
            reading(12, gpu_util_pct=20, memory_busy_pct=40, pcie_tx_kib_s=3000.0, pcie_rx_kib_s=2000.0),
            reading(18, gpu_util_pct=30, memory_busy_pct=60, pcie_tx_kib_s=1000.0, pcie_rx_kib_s=0.0),
        ]
    )
    sampler = GpuSampler(backend, clock=lambda: next(clock))
    for _ in range(3):
        sampler.sample_once()
    summary = sampler.summary()
    # TX: (1000+3000)/2 * 1 + (3000+1000)/2 * 2 = 6000 KiB. RX: 1000 + 2000 = 3000 KiB.
    assert summary["pcie_tx_gb"] == pytest.approx(6000 * 1024 / 1e9)
    assert summary["pcie_rx_gb"] == pytest.approx(3000 * 1024 / 1e9)
    assert summary["pcie_gb"] == pytest.approx(9000 * 1024 / 1e9)
    assert summary["gpu_mem_used_gib_mean"] == pytest.approx(12)
    assert summary["gpu_mem_used_gib_peak"] == pytest.approx(18)
    assert summary["gpu_mem_frac_mean"] == pytest.approx(0.5)
    assert summary["gpu_util_pct_mean"] == pytest.approx(20)
    assert summary["gpu_mem_busy_pct_mean"] == pytest.approx(40)
    assert "estimate" in summary["pcie_note"]


def test_sampler_copes_with_counters_a_gpu_refuses():
    sampler = GpuSampler(StubBackend([reading(1), reading(2)]), clock=iter([0.0, 1.0]).__next__)
    sampler.sample_once()
    sampler.sample_once()
    summary = sampler.summary()
    assert summary["pcie_gb"] is None and summary["gpu_util_pct_mean"] is None
    assert summary["gpu_mem_used_gib_mean"] == pytest.approx(1.5)


def test_sampler_counts_lost_samples_and_runs_as_a_thread(tmp_path):
    class Flaky(StubBackend):
        def read(self):
            if self.next_index == 1:
                self.next_index += 1
                raise RuntimeError("nvml hiccup")
            return super().read()

    sampler = GpuSampler(Flaky([reading(1)]), period_s=0.001)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    assert sampler.read_errors == 1 and len(sampler.samples) > 2
    sampler.write_csv(tmp_path / "s.csv")
    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert len(rows) == len(sampler.samples) and rows[0]["t_s"] == "0.000"


def test_parse_prometheus_sums_label_sets():
    text = '# HELP a b\nvllm:x_total{model="m",engine="0"} 2.0\nvllm:x_total{model="m",engine="1"} 3.0\nplain 7\nbad line here\n'
    assert parse_prometheus(text) == {"vllm:x_total": 5.0, "plain": 7.0}


def test_spread_against_hand_computed_values():
    five = report.spread([100.0, 1.0, 3.0, 2.0, 4.0])
    assert (five.n, five.median, five.minimum, five.maximum) == (5, 3.0, 1.0, 100.0)
    # Inclusive quartiles of 1 2 3 4 100 fall exactly on 2 and 4.
    assert (five.q1, five.q3, five.iqr) == (2.0, 4.0, 2.0)
    four = report.spread([1.0, 2.0, 3.0, 4.0])
    assert four.median == 2.5
    # Positions 0.75 and 2.25 of the sorted values: 1.75 and 3.25.
    assert (four.q1, four.q3, four.iqr) == (1.75, 3.25, 1.5)
    one = report.spread([7.0])
    assert (one.median, one.iqr, one.minimum, one.maximum) == (7.0, 0.0, 7.0, 7.0)
    with pytest.raises(ValueError):
        report.spread([])


def job(job_id: str = "j", source: str = "recorded:x", arrival: float = 0.0, new: int = 20, output: int = 3, tool: float = 0.0) -> TraceJob:
    return TraceJob(job_id, arrival, [TraceSession(f"{job_id}/m", [TraceTurn(new, output, "t", tool), TraceTurn(5, output, "t", 0.0)])], source)


def matrix_config(trace: Path | None = None, role: str = "headline", **top) -> dict:
    config = {
        "matched": asdict(SETTINGS),
        "systems": [{"name": "layover-combined", "kind": "layover", "policy": "combined", "cost_model": "/c.json"}, {"name": "vllm", "kind": "vllm"}],
        "tolerances": {"kv_capacity_frac": 0.02, "gpu_memory_gib": 3.0, "gpu_idle_gib": 1.0},
        "limits": {"startup_timeout_s": 5, "stop_timeout_s": 5, "gpu_release_timeout_s": 1, "request_timeout_s": 30, "run_timeout_s": 30},
        "runs": 2, "warmup_jobs": 0, "system_tokens": 8, "seed_base": 100, "reference_system": "layover-combined",
        "estimate_assumptions": {"startup_s": 10, "prefill_tokens_per_s": 100, "decode_tokens_per_s_per_stream": 10, "decode_tokens_per_s_total": 100},
    }
    if trace is not None:
        config["workloads"] = [{"name": "w", "role": role, "trace": str(trace), "arrival_rate_per_s": 1.0}]
    return {**config, **top}


@pytest.mark.parametrize("source", ["synthetic-swebench-shape", "mini-swe-agent-1", "mini-swe-agent-1 (tool times assumed)", "unknown", "RECORDED:x"])
def test_headline_refuses_anything_not_recorded(tmp_path, source):
    trace = tmp_path / "t.jsonl"
    save_jsonl([job(source=source)], trace)
    config = matrix_config(trace)
    with pytest.raises(Refused, match="does not start with 'recorded:'"):
        run_matrix.load_workloads(config, SETTINGS, pipeline_check=False)
    assert run_matrix.load_workloads(config, SETTINGS, pipeline_check=True)[0].source == source


def test_headline_accepts_recorded_and_refuses_a_mixture(tmp_path):
    trace = tmp_path / "t.jsonl"
    save_jsonl([job("a", "recorded:mini-swe-agent-2.1:m:h")], trace)
    assert run_matrix.load_workloads(matrix_config(trace), SETTINGS, False)[0].source.startswith("recorded:")
    save_jsonl([job("a", "recorded:x"), job("b", "synthetic")], trace)
    with pytest.raises(Refused, match="does not start with"):
        run_matrix.load_workloads(matrix_config(trace), SETTINGS, False)
    # Two recorded sources are both named; pipeline-check still may not blend in synthetic jobs.
    save_jsonl([job("a", "recorded:x"), job("b", "recorded:y")], trace)
    assert run_matrix.load_workloads(matrix_config(trace), SETTINGS, False)[0].source == "recorded:x; recorded:y"
    save_jsonl([job("a", "recorded:x"), job("b", "synthetic")], trace)
    with pytest.raises(Refused, match="mixes trace sources"):
        run_matrix.load_workloads(matrix_config(trace), SETTINGS, True)


def test_sweep_role_takes_synthetic_and_labels_it(tmp_path):
    trace = tmp_path / "t.jsonl"
    save_jsonl([job(source="synthetic-x")], trace)
    assert run_matrix.load_workloads(matrix_config(trace, role="sweep"), SETTINGS, False)[0].source == "synthetic-x"


def test_missing_trace_is_refused_except_in_a_dry_run(tmp_path):
    config = matrix_config(tmp_path / "absent.jsonl")
    with pytest.raises(Refused, match="does not exist"):
        run_matrix.load_workloads(config, SETTINGS, False)
    assert "does not exist" in run_matrix.load_workloads(config, SETTINGS, False, dry_run=True)[0].unavailable


def sweep_config(**base) -> dict:
    return matrix_config(
        sweeps={
            "systems": ["vllm"], "calibration_group": "g", "warmup_jobs": 1,
            "base": {
                "jobs": 3, "seed": 1, "tool_median_s": 0.5,
                "overrides": {"turns_mean": 2, "turns_sd": 0, "tokens_per_job_mean": 200, "tokens_per_job_sd": 0, "output_tokens_mean": 4}, **base,
            },
            "tool_median_s": [0.1, 0.5], "branching": [0, 2],
        }
    )


def test_sweep_expands_with_shared_base_point():
    workloads = run_matrix.load_workloads(sweep_config(), SETTINGS, False)
    assert [w.name for w in workloads] == ["sweep-tool0.1-branch0", "sweep-tool0.5-branch0", "sweep-tool0.5-branch2"]
    shared = workloads[1]
    assert {(m["sweep"], m["x"]) for m in shared.sweeps} == {("tool_median_s", 0.5), ("branching", 0)}
    assert shared.calibration_reference and not workloads[0].calibration_reference
    assert all(w.source.startswith("synthetic") for w in workloads)
    assert len(workloads[2].jobs[0].sessions) == 3 and len(workloads[0].jobs[0].sessions) == 1


def test_missing_synthetic_calibration_is_refused(tmp_path):
    config = sweep_config(calibration_file=str(tmp_path / "absent.json"))
    with pytest.raises(Refused, match="calibration file"):
        run_matrix.load_workloads(config, SETTINGS, False)
    assert run_matrix.load_workloads(config, SETTINGS, False, dry_run=True)


def test_peak_context_follows_forks():
    forked = TraceJob("j", 0.0, [
        TraceSession("m", [TraceTurn(100, 10), TraceTurn(20, 5)]),
        TraceSession("c", [TraceTurn(50, 7)], parent="m", fork_after_turn=0),
    ])
    # The parent's context after turn 0 is 8+110; the child adds 57. The parent alone reaches 143.
    assert run_matrix.peak_context(forked, 8) == 175
    forked.shared_prefix_tokens = 1000
    assert run_matrix.peak_context(forked, 8) == 1167


def test_jobs_over_the_context_limit_are_refused_or_dropped_openly():
    jobs = [job("small", new=10), job("big", new=1000)]
    with pytest.raises(Refused, match="1 of 2 jobs exceed"):
        run_matrix.enforce_context("w", jobs, 100, 8, drop=False)
    kept, dropped = run_matrix.enforce_context("w", jobs, 100, 8, drop=True)
    assert [j.job_id for j in kept] == ["small"] and dropped == ["big"]


def test_warmup_must_leave_jobs_to_measure(tmp_path):
    trace = tmp_path / "t.jsonl"
    save_jsonl([job("a"), job("b")], trace)
    config = matrix_config(trace, warmup_jobs=2)
    with pytest.raises(Refused, match="warm-up"):
        run_matrix.load_workloads(config, SETTINGS, False)


def test_retime_is_deterministic_keeps_order_and_hits_the_rate():
    jobs = [job(str(i)) for i in range(2000)]
    timed = run_matrix.retime(jobs, 2.0, 5)
    assert timed == run_matrix.retime(jobs, 2.0, 5)
    assert timed != run_matrix.retime(jobs, 2.0, 6)
    assert [j.job_id for j in timed] == [j.job_id for j in jobs]
    assert timed[0].arrival_s == 0.0
    assert timed[-1].arrival_s / 1999 == pytest.approx(0.5, rel=0.1)
    assert jobs[5].arrival_s == 0.0


def test_cells_go_round_robin_and_only_filters():
    workloads = [Workload("a", "headline", ["x", "y"], 0), Workload("b", "sweep", ["x"], 0)]
    cells = run_matrix.build_cells(workloads, 2)
    assert [c.label for c in cells] == ["a/x/load100/0", "a/y/load100/0", "a/x/load100/1", "a/y/load100/1", "b/x/load100/0", "b/x/load100/1"]
    assert [c.label for c in run_matrix.select_cells(cells, ["a/y/*", "*/*/*/1"])] == ["a/y/load100/0", "a/x/load100/1", "a/y/load100/1", "b/x/load100/1"]
    assert run_matrix.select_cells(cells, None) == cells


def test_load_labels_are_percent():
    assert [run_matrix.load_label(f) for f in (0.5, 1.0, 1.25)] == ["load50", "load100", "load125"]
    assert [run_matrix.load_fraction(run_matrix.load_label(f)) for f in (0.5, 1.0, 1.25)] == [0.5, 1.0, 1.25]


HEADLINE = ["layover-combined", "vllm", "sglang"]
ABLATIONS = {"layover-discard", "layover-discard-nocache", "layover-combined-bf16", "layover-keep_ttl", "layover-swap"}


def published_plan(qwen3_4b_config, name="first-pass"):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    workloads = run_matrix.load_workloads(config, SETTINGS, False, dry_run=True)
    return config, run_matrix.plan_cells(config, workloads, name)


def test_the_first_pass_plan_is_what_was_asked_for_in_the_order_asked(qwen3_4b_config):
    config, cells = published_plan(qwen3_4b_config)
    assert len(cells) == 60 and len(set(cells)) == 60
    assert {c.workload for c in cells} == {"headline-swebench"}
    # Headline load100, 3 systems x 5 runs, first, each run round before the next.
    first = cells[:15]
    assert {c.load for c in first} == {"load100"} and {c.system for c in first} == set(HEADLINE)
    assert [c.run for c in first] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    # Then the other headline loads.
    second = cells[15:45]
    assert {c.load for c in second} == {"load50", "load125"} and {c.system for c in second} == set(HEADLINE)
    assert {(c.system, c.load, c.run) for c in second} == {(s, l, r) for s in HEADLINE for l in ("load50", "load125") for r in range(5)}
    # Then each ablation at 100%, 3 runs.
    third = cells[45:]
    assert {c.system for c in third} == ABLATIONS and {c.load for c in third} == {"load100"} and {c.run for c in third} == {0, 1, 2}
    assert not any(c.workload.startswith("sweep") for c in cells)


def test_the_full_plan_extends_first_pass_with_the_sweeps(qwen3_4b_config):
    _, first = published_plan(qwen3_4b_config)
    _, full = published_plan(qwen3_4b_config, "full")
    assert full[:60] == first
    rest = full[60:]
    assert rest and all(c.workload.startswith("sweep-") and c.load == "load100" for c in rest)
    assert {c.system for c in rest} == set(HEADLINE) and {c.run for c in rest} == set(range(5))


def test_the_runpod_plan_puts_layover_against_vllm_first_then_sglang_then_ablations(qwen3_4b_config):
    _, cells = published_plan(qwen3_4b_config, "runpod-l4")
    assert len(cells) == 12 + 6 + 10 and len(set(cells)) == len(cells)
    pairs = {"layover-combined", "vllm"}
    assert {(c.system, c.load, c.run) for c in cells[:12]} == {(s, l, r) for s in pairs for l in ("load100", "load125") for r in range(3)}
    # Each run round at both loads before the next, so a budget stop leaves both loads evenly short.
    assert [(c.load, c.run) for c in cells[:4]] == [("load100", 0)] * 2 + [("load125", 0)] * 2
    assert {(c.system, c.load, c.run) for c in cells[12:18]} == {("sglang", l, r) for l in ("load100", "load125") for r in range(3)}
    assert {(c.system, c.load, c.run) for c in cells[18:]} == {(s, "load100", r) for s in ABLATIONS for r in range(2)}


def test_the_final_plan_is_the_default_and_puts_layover_against_vllm_first(qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    assert config["default_plan"] == "final" and smoke_default_plan() == "final"
    final = run_matrix.effective_config(config, "final")
    workloads = run_matrix.load_workloads(final, SETTINGS, False, dry_run=True)
    cells = run_matrix.plan_cells(final, workloads, "final")
    pairs = {"layover-combined", "vllm"}
    assert [(c.system in pairs, c.load, c.run) for c in cells[:4]] == [(True, "load100", 0)] * 2 + [(True, "load100", 1)] * 2
    assert [(c.system, c.run) for c in cells[4:6]] == [("layover-discard", 0), ("layover-discard", 1)]
    assert {c.system for c in cells[6:]} == {"layover-keep_ttl", "layover-swap"} and len(cells) == 8
    assert "sglang" not in {c.system for c in cells} and "SGLang is left out" in config["plans"]["final"]["description"]
    assert final["matched"]["cpu_offload_gib"] == 24.0
    assert final["workloads"][0]["max_jobs"] == 15 and "max_jobs" not in config["workloads"][0]


def smoke_default_plan():
    from bench import gpu_smoke

    return gpu_smoke.DEFAULT_PLAN


def test_a_plan_runs_fewer_jobs_of_a_workload_and_only_what_it_may_change():
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    config["plans"]["small"] = {"description": "x", "workloads": {"headline-swebench": {"max_jobs": 4, "warmup_jobs": 1}}, "groups": []}
    small = run_matrix.effective_config(config, "small")
    assert small["workloads"][0] | {"trace": "t"} == config["workloads"][0] | {"trace": "t", "max_jobs": 4, "warmup_jobs": 1}
    config["plans"]["child"] = {"extends": "small", "description": "x", "workloads": {"headline-swebench": {"max_jobs": 6}}, "groups": []}
    assert run_matrix.effective_config(config, "child")["workloads"][0]["max_jobs"] == 6
    assert run_matrix.effective_config(config, "child")["workloads"][0]["warmup_jobs"] == 1
    config["plans"]["ghost"] = {"description": "x", "workloads": {"nope": {"max_jobs": 1}}, "groups": []}
    with pytest.raises(Refused, match="workload 'nope'"):
        run_matrix.effective_config(config, "ghost")
    config["plans"]["trace"] = {"description": "x", "workloads": {"headline-swebench": {"trace": "other.jsonl"}}, "groups": []}
    with pytest.raises(Refused, match="may change only"):
        run_matrix.effective_config(config, "trace")


def test_a_plan_changes_the_matched_settings_of_every_system_and_nothing_else():
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    runpod = run_matrix.effective_config(config, "runpod-l4")
    assert runpod["matched"] == {**config["matched"], "cpu_offload_gib": 24.0}
    assert {key: value for key, value in runpod.items() if key != "matched"} == {k: v for k, v in config.items() if k != "matched"}
    assert config["matched"]["cpu_offload_gib"] == 32.0, "the loaded config must not change"
    for plan in ("under-20h", "first-pass", None, "not-a-plan"):
        assert run_matrix.effective_config(config, plan) is config
    assert run_matrix.effective_config(config, "runpod-l4")["workloads"] == config["workloads"]
    # What a plan extends comes first, and the plan's own changes win.
    config["plans"]["child"] = {"extends": "runpod-l4", "description": "x", "matched": {"kv_gib": 10.0}, "groups": []}
    assert run_matrix.effective_config(config, "child")["matched"] == {**config["matched"], "cpu_offload_gib": 24.0, "kv_gib": 10.0}
    config["plans"]["typo"] = {"description": "x", "matched": {"cpu_offload_gb": 24.0}, "groups": []}
    with pytest.raises(Refused, match="cpu_offload_gb"):
        run_matrix.effective_config(config, "typo")


def test_main_runs_a_plan_at_its_own_settings(monkeypatch, qwen3_4b_config):
    seen = {}

    def record(config, settings, *args, **kwargs):
        seen["config"], seen["settings"] = config, settings
        return 0.0

    monkeypatch.setattr(run_matrix, "dry_run", record)
    assert run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--dry-run", "--plan", "runpod-l4"]) == 0
    assert seen["settings"].cpu_offload_gib == 24.0 and seen["config"]["matched"]["cpu_offload_gib"] == 24.0
    assert run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--dry-run", "--plan", "under-20h"]) == 0
    assert seen["settings"].cpu_offload_gib == 32.0


def test_a_plan_with_an_unknown_name_or_system_is_refused(qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    with pytest.raises(Refused, match="no plan 'nope'"):
        run_matrix.plan_groups(config, "nope")
    config["plans"]["bad"] = {"description": "x", "groups": [{"role": "headline", "systems": ["ghost"], "loads": [1.0], "runs": 1}]}
    with pytest.raises(Refused, match="ghost"):
        run_matrix.plan_cells(config, [Workload("w", "headline", [], 0)], "bad")
    with pytest.raises(Refused, match="no plan"):
        run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--dry-run", "--plan", "nope"])


def test_only_a_plans_roles_are_loaded(tmp_path):
    """first-pass must not need the synthetic calibration file, which the sweeps are built from."""
    config = sweep_config()
    config["sweeps"]["base"]["calibration_file"] = str(tmp_path / "absent.json")
    assert run_matrix.load_workloads(config, SETTINGS, False, roles={"headline"}) == []
    with pytest.raises(Refused, match="synthetic calibration file"):
        run_matrix.load_workloads(config, SETTINGS, False, roles={"sweep"})


def test_estimate_by_hand():
    assumed = {"startup_s": 10, "prefill_tokens_per_s": 100, "decode_tokens_per_s_per_stream": 10, "decode_tokens_per_s_total": 100}
    jobs = [job("a", new=100, output=5, tool=2.0), job("b", new=100, output=5, tool=2.0)]
    # Each job: tool 2 + prefill (100+5)/100 + decode 10/10 = 4.05 s. b arrives 1 s later at rate 1.
    # The serial bound is 2 * (1.05 + 0.1) = 2.3 s, below that. Plus 10 s startup.
    assert run_matrix.estimate_run_s(jobs, 1.0, assumed) == pytest.approx(15.05)
    long_job = TraceJob("c", 0.0, [TraceSession("c/main", [TraceTurn(100, 10, "bash", 1.0), TraceTurn(100, 10)])], "recorded:x")
    # New only: 200 prefilled. Recompute: 100, then 100 + 10 + 100.
    assert run_matrix._prefilled_tokens(long_job, False) == 200 and run_matrix._prefilled_tokens(long_job, True) == 310
    # Tool 1 + prefill 3.1 + decode 20/10 = 6.1 s, the serial bound 3.1 + 0.2 is below that, plus startup.
    assert run_matrix.estimate_run_s([long_job], 1.0, assumed, recompute=True) == pytest.approx(16.1)


def ladder_points(*pairs):
    return [{"rate_per_s": rate, "slowdown": slowdown} for rate, slowdown in pairs]


def test_saturation_is_log_interpolated_between_the_rungs_that_bracket_it():
    found = run_matrix.saturation_rate(ladder_points((0.1, 1.1), (0.2, 1.3), (0.4, 1.7)), 1.5)
    # Halfway in slowdown between 0.2 and 0.4 is their geometric mean, by hand.
    assert found["bracketed"] and found["saturation_rate_per_s"] == pytest.approx((0.2 * 0.4) ** 0.5)
    assert [p["rate_per_s"] for p in found["bracket"]] == [0.2, 0.4]
    # A rung exactly on the target is a crossing.
    assert run_matrix.saturation_rate(ladder_points((0.1, 1.2), (0.2, 1.5)), 1.5)["saturation_rate_per_s"] == pytest.approx(0.2)


def test_saturation_says_which_edge_when_nothing_brackets_it():
    never = run_matrix.saturation_rate(ladder_points((0.1, 1.1), (0.2, 1.2)), 1.5)
    assert not never["bracketed"] and never["edge"] == "above_ladder" and never["saturation_rate_per_s"] == 0.2
    past = run_matrix.saturation_rate(ladder_points((0.1, 2.0), (0.2, 3.0)), 1.5)
    assert not past["bracketed"] and past["edge"] == "below_ladder" and past["saturation_rate_per_s"] == 0.1


def test_the_walk_starts_at_its_guess_and_stops_as_soon_as_it_is_bracketed():
    ladder = [0.1, 0.2, 0.4, 0.8]
    assert run_matrix.next_rung(ladder, [], 1.5, 0.2) == 0.2
    assert run_matrix.next_rung(ladder, ladder_points((0.2, 1.1)), 1.5, 0.2) == 0.4
    assert run_matrix.next_rung(ladder, ladder_points((0.2, 1.9)), 1.5, 0.2) == 0.1
    assert run_matrix.next_rung(ladder, ladder_points((0.2, 1.1), (0.4, 1.9)), 1.5, 0.2) is None
    assert run_matrix.next_rung(ladder, ladder_points((0.2, 1.1), (0.4, 1.2), (0.8, 1.3)), 1.5, 0.2) is None
    assert run_matrix.next_rung(ladder, ladder_points((0.2, 1.9), (0.1, 1.8)), 1.5, 0.2) is None


def make_ctx(tmp_path: Path, pipeline_check: bool = False, gpu=None, runs: int = 2, trace_source: str = "recorded:x") -> tuple[MatrixContext, list[Workload]]:
    config = matrix_config(runs=runs)
    workloads = [Workload("w", "headline", ["layover-combined", "vllm"], 0, [job(source=trace_source)], trace_source, "h", arrival_rate_per_s=1.0)]
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in config["systems"]}
    ctx = MatrixContext(config, SETTINGS, systems, {w.name: w for w in workloads}, tmp_path, pipeline_check, gpu, 8 * GIB, 100, None)
    manifest = run_matrix.manifest_for(config, workloads, run_matrix.build_cells(workloads, runs), pipeline_check)
    run_matrix.write_json(tmp_path / "manifest.json", manifest)
    return ctx, workloads


def fake_result(mean: float) -> dict:
    return {
        "metrics": {
            "mean_jct_s": mean, "p95_jct_s": mean * 2, "programs_per_hour": 3600 / mean, "mean_first_token_s": mean / 10,
            "p95_first_token_s": mean / 5, "gpu_mem_frac_mean": None, "kv_pool_used_frac_mean": 0.5, "pcie_gb": None,
        },
        "source": "recorded:x", "token_sources": {"server": 2}, "output_count_mismatches": 0, "prompt_tokens_total": 100,
        "output_tokens_total": 6, "server_metrics": {"vllm:external_prefix_cache_hits_total": 5}, "gpu": {}, "turns": [{"session": "s", "turn": 0, "crc": 1, "source": "server", "first_token_s": 0.1}],
    }


def test_resume_never_repeats_a_finished_run(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    calls = []

    def measure(_, cell, *__):
        calls.append(cell.label)
        return fake_result(10.0)

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    monkeypatch.setattr(run_matrix, "preflight", lambda *_: None)
    args = argparse.Namespace(skip_preflight=True, preflight_only=False)
    cells = run_matrix.build_cells(workloads, 2)
    assert run_matrix.run_all(ctx, args, workloads, cells[:2], None) == 0
    assert calls == ["w/layover-combined/load100/0", "w/vllm/load100/0"]
    assert run_matrix.run_all(ctx, args, workloads, cells, None) == 0
    assert calls == ["w/layover-combined/load100/0", "w/vllm/load100/0", "w/layover-combined/load100/1", "w/vllm/load100/1"]
    assert run_matrix.run_all(ctx, args, workloads, cells, None) == 0
    assert len(calls) == 4
    saved = json.loads(ctx.run_path(Cell("w", "vllm", 1)).read_text())
    assert saved["status"] == "ok" and saved["seed"] == 101 and len(saved["attempts"]) == 1
    assert saved["trace_source"] == "recorded:x" and saved["arrival"]["arrival_rate_per_s"] == 1.0


def test_failed_attempt_is_retried_once_and_recorded(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path)
    outcomes = [ServerFailed("flag rejected"), fake_result(5.0)]

    def measure(*_):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    record = run_matrix.run_cell(ctx, Cell("w", "vllm", 0), {"arrival_rate_per_s": 1.0, "chosen_by": "t"})
    assert record["status"] == "ok" and not outcomes
    assert [a["status"] for a in record["attempts"]] == ["failed", "ok"]
    assert "flag rejected" in record["attempts"][0]["error"]
    assert json.loads(ctx.run_path(Cell("w", "vllm", 0)).read_text())["status"] == "ok"


def test_a_run_failing_twice_is_kept_retried_on_resume_and_reported_missing(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path, runs=1)
    calls = []

    def measure(_, cell, *__):
        calls.append(cell.label)
        if cell.system == "vllm":
            raise ServerFailed("out of memory")
        return fake_result(10.0)

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    args = argparse.Namespace(skip_preflight=True, preflight_only=False)
    cells = run_matrix.build_cells(workloads, 1)
    assert run_matrix.run_all(ctx, args, workloads, cells, None) == 1
    assert calls.count("w/vllm/load100/0") == 2 and calls.count("w/layover-combined/load100/0") == 1
    saved = json.loads(ctx.run_path(Cell("w", "vllm", 0)).read_text())
    assert saved["status"] == "failed" and len(saved["attempts"]) == 2
    assert "out of memory" in saved["attempts"][1]["error"]

    text = report.write_report(tmp_path).read_text()
    assert "INCOMPLETE: 1 of 2 runs missing" in text
    assert "`w/vllm/load100/0`: failed after 2 attempts: ServerFailed: out of memory" in text
    assert "0/1 **missing runs**" in text

    run_matrix.run_all(ctx, args, workloads, cells, None)
    assert calls.count("w/vllm/load100/0") == 4 and calls.count("w/layover-combined/load100/0") == 1
    assert len(json.loads(ctx.run_path(Cell("w", "vllm", 0)).read_text())["attempts"]) == 4


def test_gpu_that_will_not_release_ends_the_matrix_with_the_error_saved(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path, runs=1)

    def measure(*_):
        raise GpuNotReleased("still 9 GiB")

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    args = argparse.Namespace(skip_preflight=True, preflight_only=False)
    with pytest.raises(GpuNotReleased):
        run_matrix.run_all(ctx, args, workloads, run_matrix.build_cells(workloads, 1), None)
    assert "GpuNotReleased" in json.loads(ctx.run_path(Cell("w", "layover-combined", 0)).read_text())["attempts"][0]["error"]
    assert not ctx.run_path(Cell("w", "vllm", 0)).exists()


def test_interrupt_is_saved_and_the_run_resumes(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path, runs=1)

    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_matrix, "measure_run", interrupted)
    args = argparse.Namespace(skip_preflight=True, preflight_only=False)
    with pytest.raises(KeyboardInterrupt):
        run_matrix.run_all(ctx, args, workloads, run_matrix.build_cells(workloads, 1), None)
    path = ctx.run_path(Cell("w", "layover-combined", 0))
    assert json.loads(path.read_text())["attempts"][0]["error"] == "interrupted"
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    assert run_matrix.run_all(ctx, args, workloads, run_matrix.build_cells(workloads, 1), None) == 0
    assert json.loads(path.read_text())["status"] == "ok"


def test_max_hours_stops_after_the_current_run_and_reports_the_rest(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    calls = []
    deadline = time.monotonic() + 0.05

    def measure(_, cell, *__):
        calls.append(cell.label)
        time.sleep(0.1)
        return fake_result(10.0)

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    args = argparse.Namespace(skip_preflight=True, preflight_only=False)
    code = run_matrix.run_all(ctx, args, workloads, run_matrix.build_cells(workloads, 2), deadline)
    assert code == 1 and calls == ["w/layover-combined/load100/0"]
    assert json.loads(ctx.run_path(Cell("w", "layover-combined", 0)).read_text())["status"] == "ok"
    skipped = json.loads((tmp_path / "session.json").read_text())["budget_skipped"]
    assert skipped == ["w/vllm/load100/0", "w/layover-combined/load100/1", "w/vllm/load100/1"]
    text = report.write_report(tmp_path).read_text()
    assert "`w/vllm/load100/1`: skipped by the time budget" in text
    assert "3 of 4 expected runs have no result" in text


def test_resume_refuses_another_config_or_trace(tmp_path):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    same = run_matrix.manifest_for(ctx.config, workloads, cells, False)
    run_matrix.check_resume(tmp_path, same)
    with pytest.raises(Refused, match="config_hash"):
        run_matrix.check_resume(tmp_path, run_matrix.manifest_for({**ctx.config, "runs": 9}, workloads, cells, False))
    with pytest.raises(Refused, match="pipeline_check"):
        run_matrix.check_resume(tmp_path, run_matrix.manifest_for(ctx.config, workloads, cells, True))
    workloads[0].trace_hash = "other"
    with pytest.raises(Refused, match="trace for w changed"):
        run_matrix.check_resume(tmp_path, run_matrix.manifest_for(ctx.config, workloads, cells, False))


@pytest.fixture()
def qwen3_4b_config(monkeypatch):
    # The published Qwen/Qwen3-4B config, so these tests need no download.
    config = ModelConfig(151936, 2560, 10240, 36, 32, 8, 128, 262144)
    monkeypatch.setattr(run_matrix.ModelConfig, "from_pretrained", classmethod(lambda cls, name: config))
    return config


def config_without_its_trace(tmp_path) -> Path:
    """The published config as it is before the GPU machine builds the trace, whether or not this
    checkout already holds one."""
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    config["workloads"][0]["trace"] = "traces/never-built-for-this-test.jsonl"
    assert not (REPO / config["workloads"][0]["trace"]).exists()
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config))
    return path


def test_dry_run_prints_every_command_and_an_estimate_without_starting_anything(capsys, monkeypatch, qwen3_4b_config, tmp_path):
    def explode(*_, **__):
        raise AssertionError("a dry run must start nothing")

    monkeypatch.setattr(run_matrix, "ManagedServer", explode)
    monkeypatch.setattr(run_matrix, "NvmlBackend", explode)
    assert run_matrix.main(["--config", str(config_without_its_trace(tmp_path)), "--dry-run", "--plan", "first-pass", "--out", "/nonexistent"]) == 0
    printed = capsys.readouterr().out
    for name in ("layover-combined", "layover-keep_ttl", "layover-swap", "layover-discard", "layover-discard-nocache", "layover-combined-bf16", "vllm", "sglang"):
        assert f"# {name}\n" in printed
    assert "--decode-backend triton --transfer native --offload-dtype fp8" in printed
    assert "--kv-cache-memory-bytes 12884901888" in printed
    assert re.search(r"estimated total: [\d.]+ hours, up to [\d.]+ hours", printed)
    assert "not a measurement" in printed
    # The headline trace is built later on the GPU machine, and the plan says so.
    assert "headline-swebench: NO TRACE YET: 30 jobs" in printed
    assert re.search(r"load125: [\d.]+ to [\d.]+ jobs/s, 15 runs\n +layover-combined: 5 runs x [\d.]+ to [\d.]+ min", printed)
    assert re.search(r"first-pass: 60 runs, [\d.]+ to [\d.]+ matrix hours", printed) and re.search(r"full: 150 runs, [\d.]+ to [\d.]+ matrix hours", printed)
    assert re.search(r"calibration about \d+ to \d+ min \([\d.]+ to [\d.]+ hours\)", printed)
    assert "Budget: 20 hours less about 3.5 hours of other stages leaves 16.5 hours" in printed
    assert not Path("/nonexistent").exists()


def test_published_config_refuses_cpu_and_stand_ins_without_pipeline_check(tmp_path, monkeypatch):
    with pytest.raises(Refused, match="device 'cpu'"):
        run_matrix.main(["--config", str(REPO / "bench/configs/pipeline_check.json"), "--out", str(tmp_path / "o")])
    config = json.loads((REPO / "bench/configs/pipeline_check.json").read_text())
    config["matched"]["device"] = "cuda"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config))
    with pytest.raises(Refused, match="stand-in"):
        run_matrix.main(["--config", str(path), "--out", str(tmp_path / "o")])
    with pytest.raises(Refused, match="--no-gpu"):
        run_matrix.main(["--config", str(path), "--out", str(tmp_path / "o"), "--no-gpu"])


def test_only_that_matches_nothing_is_refused(qwen3_4b_config):
    with pytest.raises(Refused, match="matches no run"):
        run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--dry-run", "--only", "nothing/*"])


def test_preflight_runs_before_the_trace_exists_and_leaves_nothing_a_run_must_agree_with(tmp_path, monkeypatch):
    config = json.loads((REPO / "bench/configs/pipeline_check.json").read_text())
    config["workloads"] = [{"name": "headline", "role": "headline", "trace": str(tmp_path / "not-yet.jsonl"), "calibration_group": "g"}]
    config["layover"]["cost_model"] = str(tmp_path / "cost.json")
    (tmp_path / "cost.json").write_text("{}")
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config))
    called = []
    monkeypatch.setattr(run_matrix, "preflight", lambda ctx, names, out: called.append(names))
    out = tmp_path / "out"
    arguments = ["--config", str(path), "--out", str(out), "--pipeline-check", "--no-gpu"]
    assert run_matrix.main([*arguments, "--preflight-only"]) == 0
    assert called and "layover-combined" in called[0]
    assert not (out / "manifest.json").exists()
    with pytest.raises(Refused, match="does not exist"):
        run_matrix.main(arguments)


def test_the_baselines_can_be_preflighted_with_no_cost_model_or_trace(tmp_path, monkeypatch, qwen3_4b_config):
    called = []
    monkeypatch.setattr(run_matrix, "preflight", lambda ctx, names, out: called.append(names))
    monkeypatch.setattr(run_matrix, "check_versions", lambda systems, names: called.append(sorted(names)) or {})
    monkeypatch.setattr(run_matrix, "NvmlBackend", lambda index: StubBackend([GpuReading(0, 24 * GIB)]))
    monkeypatch.setattr(run_matrix, "checkpoint_weight_bytes", lambda *_: 8 * GIB)
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    config["layover"]["cost_model"] = str(tmp_path / "absent.json")
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "out"
    arguments = ["--config", str(path), "--out", str(out), "--preflight-only"]
    # Without kinds it wants the Layover systems too, so it wants their cost model.
    with pytest.raises(Refused, match="cost model"):
        run_matrix.main(arguments)
    assert run_matrix.main([*arguments, "--preflight-kinds", "vllm", "sglang"]) == 0
    assert called == [["sglang", "vllm"], ["vllm", "sglang"]]
    assert not (out / "manifest.json").exists()
    with pytest.raises(Refused, match="no system of that kind"):
        run_matrix.main([*arguments, "--preflight-kinds", "tgi"])
    with pytest.raises(Refused, match="only with --preflight-only"):
        run_matrix.main(["--config", str(path), "--out", str(out), "--preflight-kinds", "vllm"])


class StubBaseline(CommandSystem):
    """A stand-in that gets a baseline's stricter preflight."""

    kind = "vllm"


def baseline_stub(tmp_path, *options) -> MatrixContext:
    ctx, _ = make_ctx(tmp_path)
    ctx.limits = {**ctx.limits, "startup_timeout_s": 60, "stop_timeout_s": 5}
    command = ["{python}", str(REPO / "tests/test_matrix.py"), "--stub-port", "{port}", "--stub-model", "{model}", *options]
    ctx.systems = {"base": StubBaseline("base", replace(SETTINGS, model="m"), TOKEN_BYTES, command)}
    ctx.settings = replace(SETTINGS, model="m")
    ctx.config["preflight_output_tokens"] = 8
    return ctx


def test_a_preflight_checks_a_baseline_against_a_real_server(tmp_path):
    good = baseline_stub(tmp_path / "good", "--stub-cached-tokens", "240")
    run_matrix.preflight(good, ["base"], tmp_path / "good")
    saved = json.loads((tmp_path / "good" / "preflight" / "base.json").read_text())
    assert saved["output_tokens"] == 8 and saved["token_source"] == "server" and saved["repeat_cached_tokens"] == 240
    assert (tmp_path / "good" / "logs" / "preflight" / "base.metrics.txt").exists()
    with pytest.raises(Refused, match="cached_tokens=None"):
        run_matrix.preflight(baseline_stub(tmp_path / "none"), ["base"], tmp_path / "none")
    with pytest.raises(Refused, match="cached_tokens=16"):
        run_matrix.preflight(baseline_stub(tmp_path / "little", "--stub-cached-tokens", "16"), ["base"], tmp_path / "little")
    with pytest.raises(Refused, match="ignore_eos did not hold"):
        run_matrix.preflight(baseline_stub(tmp_path / "short", "--stub-cached-tokens", "240", "--stub-short-by", "2"), ["base"], tmp_path / "short")
    with pytest.raises(Refused, match="no token ids"):
        ctx = baseline_stub(tmp_path / "text", "--stub-cached-tokens", "240", "--stub-text-only")
        ctx.encode = lambda text: [0] * 8
        run_matrix.preflight(ctx, ["base"], tmp_path / "text")


def test_a_baseline_at_the_wrong_version_or_without_a_prefix_cache_is_refused(monkeypatch):
    system = VllmSystem("vllm", SETTINGS, TOKEN_BYTES)
    monkeypatch.setattr(system, "installed_version", lambda: "0.0.1")
    with pytest.raises(Refused, match="installed version 0.0.1, pinned"):
        run_matrix.check_versions({"vllm": system}, ["vllm"])
    monkeypatch.setattr(system, "installed_version", lambda: VLLM_VERSION)
    assert run_matrix.check_versions({"vllm": system}, ["vllm"]) == {"vllm": VLLM_VERSION}
    run_matrix.check_prefix_cache("vllm", 240)
    for missing in (0, None):
        with pytest.raises(Refused, match="prefix cache"):
            run_matrix.check_prefix_cache("vllm", missing)


def test_a_retry_does_not_start_after_the_time_budget(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path)
    ctx.deadline = time.monotonic() - 1

    def fails(*_):
        raise ServerFailed("boom")

    monkeypatch.setattr(run_matrix, "measure_run", fails)
    record = run_matrix.run_cell(ctx, Cell("w", "vllm", 0), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    assert record["status"] == "failed" and len(record["attempts"]) == 1


def test_config_that_cannot_hold_its_own_context_is_refused(tmp_path, qwen3_4b_config):
    config = json.loads((REPO / "bench/configs/matrix.json").read_text())
    config["matched"]["kv_gib"] = 1.0
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config))
    with pytest.raises(Refused, match="does not fit"):
        run_matrix.main(["--config", str(path), "--dry-run"])


def test_pins_and_flag_reasons_agree_with_the_code():
    script = (REPO / "scripts/setup_baselines.sh").read_text()
    assert f"VLLM_VERSION={VLLM_VERSION}\n" in script and f"SGLANG_VERSION={SGLANG_VERSION}\n" in script
    table = (REPO / "bench/configs/baseline_flags.md").read_text()
    assert f"vLLM {VLLM_VERSION}" in table and f"SGLang {SGLANG_VERSION}" in table
    used = set(flags(VllmSystem("v", SETTINGS, TOKEN_BYTES).command("h", 1))) | set(flags(SglangSystem("s", SETTINGS, TOKEN_BYTES).command("h", 1)))
    assert not {flag for flag in used if f"`{flag}" not in table}


def test_published_configs_expand_cleanly(qwen3_4b_config, tmp_path):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    settings = MatchedSettings.from_config(config["matched"])
    assert settings == SETTINGS
    assert config["runs"] == 5 and config["matched"]["dtype"] == "bf16"
    assert config["workloads"][0]["trace"] == "traces/swebench_30.jsonl"
    # Before the trace is built, the headline workload is there and says it cannot run yet.
    unbuilt = run_matrix.load_config(config_without_its_trace(tmp_path))
    workloads = run_matrix.load_workloads(unbuilt, settings, False, dry_run=True)
    assert workloads[0].role == "headline" and workloads[0].unavailable
    # Set on a baseline, so a load that suits Layover cannot be the reason it wins.
    assert config["calibration"]["system"] == "vllm"
    specs = {spec["name"]: spec for spec in run_matrix.system_specs(config)}
    assert {s["policy"] for s in specs.values() if s["kind"] == "layover"} == {"combined", "keep_ttl", "swap", "discard"}
    assert all(s["decode_backend"] == "triton" and s["transfer"] == "native" for s in specs.values() if s["kind"] == "layover")
    # The definition is written down, not implied by the code.
    definition = config["calibration"]["definition"]
    for words in ("vLLM's mean JCT", "target_slowdown", "full headline workload", "warm-up jobs excluded", "log-interpolated"):
        assert words in definition
    assert config["calibration"]["target_slowdown"] == 1.5
    assert specs["layover-combined-bf16"]["offload_dtype"] == "bf16" and specs["layover-combined"]["offload_dtype"] == "fp8"
    assert specs["layover-combined-bf16"]["cost_model"] == specs["layover-combined"]["cost_model"]
    assert run_matrix.system_specs(config)[1]["name"] == "vllm" and config["reference_system"] == "layover-combined"


def write_stub_process(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def alive(pid: int) -> bool:
    return process_running(pid)


def test_process_table_sees_a_child_until_it_dies():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert process_running(child.pid)
        assert child.pid in _descendants(os.getpid())
        assert [command for command in running_commands() if "time.sleep(60)" in command]
        child.kill()
        # Not reaped yet, so a zombie: it still has a pid and no memory.
        deadline = time.monotonic() + 10
        while process_running(child.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_running(child.pid)
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(Path("/proc/self/stat").exists(), reason="Linux reads /proc and needs no ps")
def test_a_missing_ps_is_an_error_and_not_an_empty_table(monkeypatch):
    def no_ps(*_, **__):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(subprocess, "run", no_ps)
    with pytest.raises(FileNotFoundError):
        _descendants(os.getpid())


def test_stop_leaves_no_process_behind(tmp_path):
    """Including a worker that left the server's session, which a group signal cannot reach."""
    script = tmp_path / "serve.py"
    write_stub_process(
        script,
        f"""
        import subprocess, sys, time
        from pathlib import Path
        same = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        away = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
        Path({str(tmp_path / 'pids')!r}).write_text(f"{{same.pid}} {{away.pid}}")
        time.sleep(600)
        """,
    )
    system = CommandSystem("c", SETTINGS, TOKEN_BYTES, ["{python}", str(script)])
    server = ManagedServer(system, tmp_path / "log", free_port(), stop_timeout_s=2)
    server.start()
    deadline = time.monotonic() + 20
    while not (tmp_path / "pids").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pids = [int(p) for p in (tmp_path / "pids").read_text().split()]
    main_pid = server.process.pid
    assert all(alive(p) for p in [main_pid, *pids])
    server.stop()
    for _ in range(50):
        if not any(alive(p) for p in [main_pid, *pids]):
            break
        time.sleep(0.1)
    assert not any(alive(p) for p in [main_pid, *pids])
    server.stop()


def test_stop_escalates_when_a_server_ignores_sigterm(tmp_path):
    script = tmp_path / "serve.py"
    write_stub_process(
        script,
        f"""
        import signal, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path({str(tmp_path / 'ready')!r}).write_text("x")
        time.sleep(600)
        """,
    )
    server = ManagedServer(CommandSystem("c", SETTINGS, TOKEN_BYTES, ["{python}", str(script)]), tmp_path / "log", free_port(), stop_timeout_s=0.2)
    server.start()
    deadline = time.monotonic() + 20
    while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = server.process.pid
    server.stop()
    assert not alive(pid)


def test_early_exit_and_timeout_fail_with_the_log(tmp_path):
    system = CommandSystem("c", SETTINGS, TOKEN_BYTES, ["{python}", "-c", "import sys; print('bad flag: --nope'); sys.exit(2)"])
    server = ManagedServer(system, tmp_path / "log", free_port())
    server.start()
    with pytest.raises(ServerFailed, match="(?s)exited with code 2.*bad flag: --nope"):
        server.wait_ready(30, poll_s=0.05)
    server.stop()

    system = CommandSystem("c", SETTINGS, TOKEN_BYTES, ["{python}", "-c", "import time; print('stuck'); time.sleep(60)"])
    server = ManagedServer(system, tmp_path / "log2", free_port())
    server.start()
    with pytest.raises(ServerFailed, match="not healthy after"):
        server.wait_ready(0.2, poll_s=0.05)
    server.stop()


@pytest.fixture()
def stub_url():
    port = free_port()
    server = subprocess.Popen([sys.executable, __file__, "--stub-port", str(port), "--stub-model", "m"])
    yield from _ready(server, port)


@pytest.fixture()
def text_stub_url():
    port = free_port()
    server = subprocess.Popen([sys.executable, __file__, "--stub-port", str(port), "--stub-model", "m", "--stub-text-only"])
    yield from _ready(server, port)


def _ready(server: subprocess.Popen, port: int):
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                httpx.get(url + "/health", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.05)
        yield url
    finally:
        server.kill()
        server.wait()


def test_replay_reads_the_baseline_token_id_format(stub_url):
    """prompt_token_ids must not leak into the context, and an empty final chunk must not crash."""
    result = asyncio.run(replay([job()], stub_url, "m", 100, "stub", hints=False, system_tokens=8))
    assert result.token_sources == {"server": 2} and result.output_count_mismatches == 0
    assert [t.prompt_tokens for t in result.turns] == [28, 36]
    assert [t.output_tokens for t in result.turns] == [3, 3]
    assert result.output_tokens_total == 6 and result.prompt_tokens_total == 64
    assert all(t.first_token_s <= t.finished_s for t in result.turns)


def test_replay_falls_back_to_the_tokenizer_and_says_so(text_stub_url):
    encode = lambda text: [1] * text.count("stub")  # noqa: E731
    result = asyncio.run(replay([job()], text_stub_url, "m", 100, "stub", hints=False, encode=encode, system_tokens=8))
    assert result.token_sources == {"tokenizer": 2}
    # Each turn's three pieces become one token per piece.
    assert [t.output_tokens for t in result.turns] == [3, 3]

    mismatched = lambda text: [1]  # noqa: E731
    result = asyncio.run(replay([job()], text_stub_url, "m", 100, "stub", hints=False, encode=mismatched, system_tokens=8))
    assert result.output_count_mismatches == 2

    with pytest.raises(RuntimeError, match="no token ids"):
        asyncio.run(replay([job()], text_stub_url, "m", 100, "stub", hints=False, system_tokens=8))


def test_replay_sends_baselines_no_session_fields_and_asks_for_ids():
    received = []

    class Recording(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {"choices": [{"index": 0, "text": "x", "token_ids": [1, 2, 3], "finish_reason": "length"}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        asyncio.run(replay([job()], url, "m", 100, "plain", hints=False, system_tokens=8, seed=3))
        first_run = list(received)
        received.clear()
        asyncio.run(replay([job()], url, "m", 100, "plain", hints=False, system_tokens=8, seed=3))
        assert [r["prompt"] for r in received] == [r["prompt"] for r in first_run]
    finally:
        server.shutdown()
    request = first_run[0]
    assert "session_id" not in request and "layover" not in request
    assert request["return_token_ids"] is True and request["ignore_eos"] is True and request["stream"] is True
    assert request["stream_options"] == {"include_usage": True} and request["temperature"] == 0
    assert all(isinstance(t, int) for t in request["prompt"])
    # The second turn's prompt holds the ids the server returned.
    assert first_run[1]["prompt"][len(request["prompt"]) : len(request["prompt"]) + 3] == [1, 2, 3]


def test_every_system_gets_the_same_ids_whatever_it_is_told_about_sessions():
    """Layover's hints sit beside the prompt and change nothing in it."""
    received: list[dict] = []

    class Recording(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {"choices": [{"index": 0, "text": "x", "token_ids": [1, 2, 3], "finish_reason": "length"}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        arrivals = run_matrix.retime([job("a"), job("b")], 50.0, 11)
        asyncio.run(replay(arrivals, url, "m", 100, "plain", hints=False, system_tokens=8, seed=11))
        plain = sorted(received, key=lambda r: (r["prompt"], r["max_tokens"]))
        received.clear()
        asyncio.run(replay(arrivals, url, "m", 100, "hinted", hints=True, system_tokens=8, seed=11))
        hinted = sorted(received, key=lambda r: (r["prompt"], r["max_tokens"]))
    finally:
        server.shutdown()
    assert len(plain) == len(hinted) == 4
    assert [(r["prompt"], r["max_tokens"]) for r in plain] == [(r["prompt"], r["max_tokens"]) for r in hinted]
    assert all("session_id" in r for r in hinted) and not any("session_id" in r for r in plain)
    # A hint names the tool and never says how long it will take.
    assert all("expected_tool_s" not in r["layover"] for r in hinted)


def test_warmup_is_the_same_jobs_on_every_system_even_when_arrivals_swap():
    arrivals = {"a": 0.2, "b": 0.1, "c": 0.3}
    finishes = {"a": 1.2, "b": 9.1, "c": 2.3}
    result = summarize("recorded:x", "s", arrivals, finishes, [], 1, ["a", "b", "c"], clock_origin_s=100.0)
    assert result.jct_s == pytest.approx({"b": 9.0, "c": 2.0})
    assert (result.window_start_s, result.window_end_s, result.clock_origin_s) == (0.1, 9.1, 100.0)
    unplanned = summarize("recorded:x", "s", arrivals, finishes, [], 1)
    assert set(unplanned.jct_s) == {"a", "c"}


def test_not_comparable_catches_short_outputs_and_reencoded_text():
    base = dict(source="recorded:x", system="s", jobs_total=1, jobs_measured=1, mean_jct_s=1.0, p95_jct_s=1.0, programs_per_hour=1.0, mean_first_token_s=1.0)
    assert run_matrix.not_comparable(ReplayResult(**base, token_sources={"server": 3})) is None
    assert "ignore_eos" in run_matrix.not_comparable(ReplayResult(**base, token_sources={"server": 3}, output_count_mismatches=1))
    assert "re-encoded" in run_matrix.not_comparable(ReplayResult(**base, token_sources={"tokenizer": 2}))


def test_measure_run_refuses_an_incomparable_run_except_in_a_pipeline_check(tmp_path, monkeypatch):
    class Fake:
        client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.1)
        url = "http://127.0.0.1:1"
        log_text = staticmethod(str)

    @contextmanager
    def fake_serving(*_):
        yield Fake(), {}

    bad = ReplayResult("recorded:x", "s", 1, 1, 1.0, 1.0, 1.0, 1.0, output_count_mismatches=2)
    monkeypatch.setattr(run_matrix, "serving", fake_serving)
    monkeypatch.setattr(run_matrix, "_replay_jobs", lambda *_: bad)
    ctx, _ = make_ctx(tmp_path)
    monkeypatch.setattr(ctx.systems["vllm"], "server_metrics", lambda client: {})
    rate = {"arrival_rate_per_s": 1.0}
    with pytest.raises(ServerFailed, match="not comparable"):
        run_matrix.measure_run(ctx, Cell("w", "vllm", 0), tmp_path / "log", rate)
    ctx.pipeline_check = True
    assert run_matrix.measure_run(ctx, Cell("w", "vllm", 0), tmp_path / "log", rate)["output_count_mismatches"] == 2


def test_every_system_replays_the_same_arrivals_and_seed(tmp_path, monkeypatch):
    class Fake:
        client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.1)
        url = "http://127.0.0.1:1"
        log_text = staticmethod(str)

    @contextmanager
    def fake_serving(*_):
        yield Fake(), {}

    seen = {}

    def fake_replay(ctx, system, server, jobs, warmup, seed, timeout_s):
        seen[system.name] = ([job.arrival_s for job in jobs], warmup, seed)
        return ReplayResult("recorded:x", system.name, 1, 1, 1.0, 1.0, 1.0, 1.0, token_sources={"server": 1})

    monkeypatch.setattr(run_matrix, "serving", fake_serving)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay)
    ctx, _ = make_ctx(tmp_path)
    for system in ctx.systems.values():
        monkeypatch.setattr(system, "server_metrics", lambda client: {})
    for name in ("vllm", "layover-combined"):
        run_matrix.measure_run(ctx, Cell("w", name, 3), tmp_path / "log", {"arrival_rate_per_s": 1.0})
    assert seen["vllm"] == seen["layover-combined"]


def test_the_sampler_window_reaches_the_gpu_figures(tmp_path, monkeypatch):
    class Fake:
        client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.1)
        url = "http://127.0.0.1:1"
        log_text = staticmethod(str)

    @contextmanager
    def fake_serving(*_):
        yield Fake(), {}

    # A window nobody sampled, so every sample has been left out.
    later = ReplayResult("recorded:x", "s", 1, 1, 1.0, 1.0, 1.0, 1.0, token_sources={"server": 1}, window_start_s=1e12, window_end_s=2e12)
    monkeypatch.setattr(run_matrix, "serving", fake_serving)
    monkeypatch.setattr(run_matrix, "_replay_jobs", lambda *_: later)
    ctx, _ = make_ctx(tmp_path, gpu=StubBackend([GpuReading(GIB, 24 * GIB)]))
    monkeypatch.setattr(ctx.systems["vllm"], "server_metrics", lambda client: {})
    with pytest.raises(ServerFailed, match="measured part"):
        run_matrix.measure_run(ctx, Cell("w", "vllm", 0), tmp_path / "log", {"arrival_rate_per_s": 1.0})


def test_trace_facts_reach_the_table():
    machine = {"machine": "x86_64", "cpu_model": "EPYC"}
    jobs = [replace(job("a"), meta={"submission": "s1", "machine": machine, "instance": "a"}), replace(job("b"), meta={"submission": "s1", "machine": machine, "instance": "b"}),
            replace(job("c"), meta={"submission": "s1", "machine": machine, "instance": "c"}), replace(job("d"), meta={"submission": "s1", "machine": machine, "instance": "d"})]
    meta = run_matrix.trace_meta_summary(jobs)
    assert meta["submission"] == ["s1"] and "EPYC" in meta["machine"][0] and "instance" not in meta
    manifest = {"pipeline_check": False, "runs": 1}
    workload = {"name": "w", "role": "headline", "source": "recorded:x", "systems": [], "dropped_jobs": [], "jobs": 4, "warmup_jobs": 1, "trace_meta": meta}
    text = "\n".join(report.workload_table(manifest, {}, workload, []))
    assert "Trace submission: s1." in text and "EPYC" in text


def test_gpu_and_kv_figures_leave_out_warmup():
    readings = [GpuReading(GIB, 10 * GIB, pcie_tx_kib_s=0.0, pcie_rx_kib_s=0.0), GpuReading(9 * GIB, 10 * GIB, pcie_tx_kib_s=1000.0, pcie_rx_kib_s=0.0)]
    times = iter([0.0, 1.0, 2.0, 3.0])
    sampler = GpuSampler(StubBackend([readings[0], readings[0], readings[1], readings[1]]), clock=lambda: next(times))
    for _ in range(4):
        sampler.sample_once()
    window = sampler.summary(2.0, 3.0)
    assert window["samples"] == 2 and window["gpu_mem_used_gib_mean"] == pytest.approx(9.0)
    assert window["pcie_tx_gb"] == pytest.approx(1000 * 1024 / 1e9)
    assert sampler.summary()["samples"] == 4 and sampler.summary()["gpu_mem_used_gib_mean"] == pytest.approx(5.0)
    assert sampler.summary(10.0, 11.0)["samples"] == 0
    system = CommandSystem("c", SETTINGS, TOKEN_BYTES, ["x"])
    poller = run_matrix.KvUsagePoller(system, "http://127.0.0.1:1", 60.0)
    poller.values = [(0.0, 0.9, 0.1), (2.0, 0.1, 0.4), (3.0, 0.3, 0.6)]
    usage = poller.summary(2.0, 3.0)
    assert usage["kv_pool_used_frac_mean"] == pytest.approx(0.2) and usage["kv_pool_used_frac_peak"] == pytest.approx(0.3)
    assert usage["kv_pool_tree_only_frac_mean"] == pytest.approx(0.5) and usage["kv_pool_occupied_frac_peak"] == pytest.approx(0.9)


def test_replay_fails_a_turn_the_server_aborted():
    class Aborting(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            event = {"choices": [{"index": 0, "text": "", "finish_reason": "abort"}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Aborting)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="finish_reason=abort"):
            asyncio.run(replay([job()], f"http://127.0.0.1:{server.server_port}", "m", 100, "x", hints=False, system_tokens=8))
    finally:
        server.shutdown()


def test_replay_gives_recorded_jobs_their_shared_prefix():
    received = []

    class Recording(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            event = {"choices": [{"index": 0, "text": "x", "token_ids": [1], "finish_reason": "length"}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
            self.close_connection = True

    long_prefix = replace(job("a"), shared_prefix_tokens=40)
    short_prefix = replace(job("b"), shared_prefix_tokens=30)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asyncio.run(replay([long_prefix, short_prefix], f"http://127.0.0.1:{server.server_port}", "m", 100, "x", hints=False, system_tokens=8))
    finally:
        server.shutdown()
    by_length = {len(r["prompt"]): r["prompt"] for r in received}
    assert by_length[60][:30] == by_length[50][:30]


def test_report_tables_and_csv_against_hand_values(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path, runs=5)
    for run, mean in enumerate([10.0, 20.0, 30.0, 40.0, 100.0]):
        monkeypatch.setattr(run_matrix, "measure_run", lambda *_, m=mean: fake_result(m))
        run_matrix.run_cell(ctx, Cell("w", "layover-combined", run), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    for run, mean in enumerate([60.0, 60.0, 60.0, 60.0]):
        monkeypatch.setattr(run_matrix, "measure_run", lambda *_, m=mean: fake_result(m))
        run_matrix.run_cell(ctx, Cell("w", "vllm", run), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    text = report.write_report(tmp_path).read_text()
    assert "### w, load100 (1 jobs/s): headline, trace source `recorded:x`" in text
    assert "PIPELINE" not in text
    assert "30.00 (10.00 to 100.00, IQR 20.00)" in text
    assert "INCOMPLETE: 1 of 10 runs missing" in text
    assert "`w/vllm/load100/4`: not run" in text
    assert "4/5 **missing runs**" in text and "5/5 |" in text
    # Runs 0 to 3 pair up: 60 against 10, 20, 30, 40 is 6, 3, 2, 1.5, so the median is 2.50, not 60 over the median of 30.
    assert re.search(r"\| vllm \| 4/5 \| 100, 101, 102, 103 \| 2\.50 \| 2\.50 \| 0\.42 \| 2\.50 \|", text)
    assert "not comparable" in text and "not pure queueing delay" in text
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    row = next(r for r in rows if r["system"] == "layover-combined" and r["metric"] == "mean_jct_s")
    assert float(row["median"]) == 30.0 and float(row["iqr"]) == 20.0 and row["runs"] == "5" and row["source"] == "recorded:x"
    assert next(r for r in rows if r["system"] == "vllm" and r["metric"] == "mean_jct_s")["missing_runs"] == "1"
    empty = next(r for r in rows if r["metric"] == "pcie_gb")
    assert empty["median"] == "" and empty["runs"] == "0"
    assert (tmp_path / "plots" / "w-load100-mean_jct_s.png").stat().st_size > 1000


def test_report_refuses_an_unrecorded_headline_and_labels_a_pipeline_check(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    make_ctx(bad, trace_source="synthetic-x")
    with pytest.raises(SystemExit, match="refusing to report"):
        report.write_report(bad)
    good = tmp_path / "good"
    ctx, _ = make_ctx(good, pipeline_check=True, trace_source="synthetic-x")
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    run_matrix.run_cell(ctx, Cell("w", "vllm", 0), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    text = report.write_report(good).read_text()
    assert text.startswith(f"# {report.PIPELINE_BANNER}")
    assert "PIPELINE CHECK, NOT A RESULT: w, load100 (1 jobs/s): headline, trace source `synthetic-x`" in text
    assert "True" in (good / "results.csv").read_text().splitlines()[1]


def test_colors_follow_the_system_and_never_cycle():
    assert report.PALETTE == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    colors = report.system_colors(["a", "b", "c"])
    assert colors == {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a"}
    with pytest.raises(ValueError):
        report.system_colors([str(i) for i in range(9)])


def test_plots_have_one_y_axis(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path, runs=2)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(3.0))
    for system in ("layover-combined", "vllm"):
        for run in range(2):
            run_matrix.run_cell(ctx, Cell("w", system, run), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    axes = []
    import matplotlib.figure as figure

    original = figure.Figure.savefig
    monkeypatch.setattr(figure.Figure, "savefig", lambda self, *a, **k: (axes.append(len(self.axes)), original(self, *a, **k))[1])
    manifest, runs = report.load_results(tmp_path)
    assert report.bar_plot(tmp_path / "x.png", manifest, runs, manifest["workloads"][0], "mean_jct_s", "s", report.system_colors(manifest["systems"]))
    assert axes == [1]


def pipeline_config(tmp_path: Path, **changes) -> Path:
    config = json.loads((REPO / "bench/configs/pipeline_check.json").read_text())
    cost_model = tmp_path / "cost_model.json"
    cost_model.write_text(json.dumps({"swap_out_s_coef": [0.001, 5e-6], "swap_in_s_coef": [0.001, 5e-6], "recompute_s_coef": [0.005, 2e-5, 2e-9], "measured_on_gpu": False}))
    config["layover"]["cost_model"] = str(cost_model)
    config.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_pipeline_check_end_to_end_with_a_real_server(tmp_path):
    """The real layover.server and two stand-in baselines, through calibration, runs, resume, and report."""
    out = tmp_path / "out"
    config = pipeline_config(tmp_path)
    arguments = ["--config", str(config), "--out", str(out), "--pipeline-check", "--no-gpu"]
    assert run_matrix.main(arguments) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["pipeline_check"] is True
    assert len(manifest["cells"]) == 2 * 3 * 2
    calibration = json.loads((out / "calibration" / "pipeline.json").read_text())
    assert calibration["system"] == "layover-combined" and calibration["points"]
    assert 0.5 <= calibration["saturation_rate_per_s"] <= 2.0 and "held fixed" in calibration["chosen_by"]

    runs = sorted((out / "runs").rglob("run-*.json"))
    assert len(runs) == 12
    by_system: dict[str, list[dict]] = {}
    for path in runs:
        record = json.loads(path.read_text())
        assert record["status"] == "ok", record["attempts"]
        assert record["pipeline_check"] is True and record["trace_source"].startswith("synthetic")
        by_system.setdefault(record["system"], []).append(record)

    for record in by_system["layover-combined"]:
        result = record["result"]
        assert set(result["token_sources"]) == {"server"} and result["output_count_mismatches"] == 0
        assert result["budget_checks"]["reported_kv_tokens"] == result["budget_checks"]["expected_kv_tokens"]
        assert result["budget_checks"]["translation"]["kv_flag"] == "--pool-gib"
        assert result["server_metrics"]["blocks_total"] > 0
        # The session hints reached a server that understands them.
        assert result["server_metrics"]["kept"] + result["server_metrics"]["swapped"] + result["server_metrics"]["dropped"] > 0
        assert result["kv_usage"]["kv_usage_errors"] == 0
    for record in by_system["stub-token-ids"]:
        assert set(record["result"]["token_sources"]) == {"server"}
        assert record["result"]["output_count_mismatches"] == 0
        # Metrics stay per system, and pre-warm-up jobs stay out of the counts.
        assert record["result"]["server_metrics"] == {"stub_requests_total": 7.0}
    for record in by_system["stub-text-only"]:
        assert set(record["result"]["token_sources"]) == {"tokenizer"}

    # Same seed, same prompt tokens before anything is generated: the count fields line up.
    same_run = [r for r in sum(by_system.values(), []) if r["run"] == 0 and r["workload"] == runs[0].parts[-4]]
    assert len({r["seed"] for r in same_run}) == 1

    text = (out / "report.md").read_text()
    assert text.startswith(f"# {report.PIPELINE_BANNER}")
    assert "PIPELINE CHECK, NOT A RESULT" in text and "synthetic-swebench-shape" in text
    assert "## Missing runs\n\nNone." in text
    assert run_matrix.PIPELINE_BANNER == report.PIPELINE_BANNER
    assert re.search(r"\| layover-combined \| 2336 \| 2336 \| n/a \| n/a \| 0\.25 \| 0\.25 \| [^|]+ \| cpu_tier_bytes_moved \| [\d.e+]+ \|", text)
    assert "layover-combined | not exposed" in text
    assert "stub-text-only" in text
    assert "True" in (out / "results.csv").read_text().splitlines()[1]
    assert (out / "plots" / "sweep-tool_median_s-mean_jct_s.png").exists()
    assert (out / "plots" / "sweep-branching-mean_jct_s.png").exists()
    logs = list((out / "logs").rglob("*.log"))
    assert logs and "layover.server" in (out / "logs" / "preflight" / "layover-combined.log").read_text()

    before = {path: path.stat().st_mtime_ns for path in runs}
    assert run_matrix.main(arguments) == 0
    assert {path: path.stat().st_mtime_ns for path in runs} == before

    # A real server is not left behind on the port-holding machine: the runs above stopped it.
    assert not [command for command in running_commands() if str(config.parent) in command]


class FakeServer:
    def __init__(self, log: str):
        self.client = None
        self.log = log

    def log_text(self) -> str:
        return self.log


def test_budgets_are_read_from_the_server_and_checked_against_the_gpu(tmp_path):
    expected_tokens = translate("vllm", SETTINGS, TOKEN_BYTES).kv_tokens_expected
    ready = f"INFO GPU KV cache size: {expected_tokens:,} tokens, Maximum concurrency for 65,536 tokens per request: 1.33x"
    held = int(0.5 * GIB) + 8 * GIB + 12 * GIB
    ctx, _ = make_ctx(tmp_path, gpu=StubBackend([GpuReading(held, 24 * GIB)]))
    system = ctx.systems["vllm"]
    checks = run_matrix.verify_budgets(ctx, system, FakeServer(ready), int(0.5 * GIB))
    assert checks["reported_kv_tokens"] == expected_tokens and checks["gpu_deviation_gib"] == pytest.approx(0.0)
    with pytest.raises(BudgetMismatch, match="did not report"):
        run_matrix.verify_budgets(ctx, system, FakeServer("the log line changed"), int(0.5 * GIB))
    with pytest.raises(BudgetMismatch, match="against a budget"):
        run_matrix.verify_budgets(ctx, system, FakeServer(ready.replace(f"{expected_tokens:,}", f"{expected_tokens // 2:,}")), int(0.5 * GIB))
    ctx.gpu = StubBackend([GpuReading(int(0.5 * GIB) + 8 * GIB + 6 * GIB, 24 * GIB)])
    with pytest.raises(BudgetMismatch, match="off by -6.00"):
        run_matrix.verify_budgets(ctx, system, FakeServer(ready), int(0.5 * GIB))


def test_every_run_gets_a_new_server_and_an_empty_gpu_or_the_matrix_ends(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    ctx.limits = {**ctx.limits, "gpu_release_timeout_s": 0.2, "startup_timeout_s": 60, "stop_timeout_s": 5}
    ctx.tolerances = {**ctx.tolerances, "gpu_idle_gib": 1.0}
    stub = ["{python}", str(REPO / "tests/test_matrix.py"), "--stub-port", "{port}", "--stub-model", "{model}"]
    system = CommandSystem("stub", replace(SETTINGS, model="m"), TOKEN_BYTES, stub)
    empty, full = GpuReading(0, 24 * GIB), GpuReading(5 * GIB, 24 * GIB)
    ctx.gpu = StubBackend([empty])
    pids = []
    for name in ("a", "b"):
        with run_matrix.serving(ctx, system, tmp_path / f"{name}.log") as (server, _):
            pids.append(server.process.pid)
        assert not process_running(pids[-1])
    assert pids[0] != pids[1]
    # Memory left over from before: the next server does not start.
    ctx.gpu = StubBackend([full])
    with pytest.raises(GpuNotReleased, match="still holds 5.00 GiB"):
        with run_matrix.serving(ctx, system, tmp_path / "c.log"):
            raise AssertionError("started on a dirty GPU")
    # Memory left behind by this server, after a run that went well.
    ctx.gpu = StubBackend([empty, full])
    with pytest.raises(GpuNotReleased):
        with run_matrix.serving(ctx, system, tmp_path / "d.log"):
            pass


def test_the_layover_command_parses_with_the_real_server():
    system = LayoverSystem("l", SETTINGS, TOKEN_BYTES, "swap", "/c.json")
    done = subprocess.run([sys.executable, "-m", "layover.server", "--help"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert done.returncode == 0
    assert not [flag for flag in flags(system.command("h", 1)) if flag not in done.stdout]
    for policy in ("combined", "keep_ttl", "swap", "discard"):
        assert policy in done.stdout


def test_a_server_that_breaks_its_budget_fails_the_run_and_is_reported(tmp_path):
    """A stand-in claiming the wrong KV capacity must not produce a number."""
    out = tmp_path / "out"
    config = pipeline_config(tmp_path, runs=1)
    raw = json.loads(config.read_text())
    raw["systems"] = [{"name": "layover-combined", "kind": "layover", "policy": "combined"}]
    raw["tolerances"]["kv_capacity_frac"] = -1
    raw["sweeps"].update(tool_median_s=[], branching=[0], arrival_rate_per_s=1.0)
    config.write_text(json.dumps(raw))
    code = run_matrix.main(["--config", str(config), "--out", str(out), "--pipeline-check", "--no-gpu", "--skip-preflight"])
    assert code == 1
    record = json.loads(next((out / "runs").rglob("run-0.json")).read_text())
    assert record["status"] == "failed" and "BudgetMismatch" in record["attempts"][0]["error"]
    assert "failed after 2 attempts: BudgetMismatch" in (out / "report.md").read_text()


def published_system(name: str) -> LayoverSystem:
    specs = {spec["name"]: spec for spec in run_matrix.system_specs(run_matrix.load_config(REPO / "bench/configs/matrix.json"))}
    return build_system(specs[name], SETTINGS, TOKEN_BYTES)


def test_discard_has_an_evictable_pool_and_only_nocache_truly_recomputes():
    command = published_system("layover-discard-nocache").command("h", 1)
    assert "--no-prefix-cache" in command and flags(command)["--policy"] == "discard"
    done = subprocess.run([sys.executable, "-m", "layover.server", "--help"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert "--no-prefix-cache" in done.stdout
    for name in ("layover-combined", "layover-keep_ttl", "layover-swap", "layover-discard", "layover-combined-bf16"):
        assert "--no-prefix-cache" not in published_system(name).command("h", 1)
    assert flags(published_system("layover-discard").command("h", 1))["--policy"] == "discard"
    # The estimate plans on the same thing the flags say.
    assert published_system("layover-discard-nocache").recomputes_context
    assert not published_system("layover-discard").recomputes_context


def test_the_bf16_ablation_changes_only_what_the_tier_stores(qwen3_4b_config):
    specs = {spec["name"]: spec for spec in run_matrix.system_specs(run_matrix.load_config(REPO / "bench/configs/matrix.json"))}
    plain = flags(build_system(specs["layover-combined"], SETTINGS, TOKEN_BYTES).command("h", 1))
    bf16 = build_system(specs["layover-combined-bf16"], SETTINGS, TOKEN_BYTES, qwen3_4b_config)
    changed = flags(bf16.command("h", 1))
    assert changed["--offload-dtype"] == "bf16" and plain["--offload-dtype"] == "fp8"
    assert {key: value for key, value in changed.items() if key != "--offload-dtype"} == {key: value for key, value in plain.items() if key != "--offload-dtype"}
    # Same budget in bytes, so its tier reports 32 GiB through its own record size.
    assert bf16.record_bytes == 16 * TOKEN_BYTES
    # One cost model file serves both: the base coefficients are the BF16 copies, by kv_crossover.
    from layover.agent.cost_model import CostModel

    costs = CostModel([1.0], [2.0], [3.0], True, [4.0], [5.0], "native")
    assert costs.for_offload(False).swap_out_s_coef == [1.0] and costs.for_offload(True).swap_out_s_coef == [4.0]


def test_the_lower_bound_is_labelled_wherever_it_appears(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path, runs=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["config"]["systems"][1]["label"] = "lower bound"
    run_matrix.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    run_matrix.run_cell(ctx, Cell("w", "vllm", 0), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    run_matrix.run_cell(ctx, Cell("w", "layover-combined", 0), {"arrival_rate_per_s": 1.0, "chosen_by": "fixed"})
    text = report.write_report(tmp_path).read_text()
    for table_row in ("| vllm (lower bound) | 1/1 |", "| w | load100 | vllm (lower bound) |", "| vllm (lower bound) | n/a | n/a |"):
        assert table_row in text
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert {r["system_label"] for r in rows if r["system"] == "vllm"} == {"vllm (lower bound)"}
    published = {s["name"]: s for s in run_matrix.load_config(REPO / "bench/configs/matrix.json")["systems"]}
    assert published["layover-discard-nocache"]["label"] == "lower bound" and "label" not in published["layover-discard"]


def test_discard_with_the_cache_on_reuses_a_returning_agents_kv_until_evicted():
    """What the column measures, at the settings the matrix runs: an evictable pool."""
    from layover.agent.cost_model import CostModel, Policy
    from layover.agent.scheduler import AgentConfig
    from layover.replay.simulate import simulate

    costs = CostModel([0.001, 1e-5], [0.001, 1e-5], [0.0, 1 / 15000])
    trace = [TraceJob("j", 0.0, [TraceSession("j/m", [TraceTurn(64, 8, "t", 0.1), TraceTurn(64, 8, "t", 0.0)])], "s")]
    # Roomy: nothing displaces the freed blocks. The whole blocks of the first turn's 16 + 64 tokens
    # come back, so the second prefills only what is after them: 8 generated and 64 new.
    roomy = simulate(trace, costs, 64, AgentConfig(policy=Policy("discard")), system_tokens=16).scheduler.stats
    assert roomy.prefix_hit_tokens == 80 and roomy.prefilled_tokens == 80 + 72
    # Tight: another job's turn runs in the gap and displaces them, so the agent pays again.
    other = TraceJob("o", 0.01, [TraceSession("o/m", [TraceTurn(180, 8, "t", 0.0)])], "s")
    tight = simulate([*trace, other], costs, 16, AgentConfig(policy=Policy("discard")), system_tokens=16).scheduler.stats
    assert tight.prefilled_tokens > roomy.prefilled_tokens + 180


def test_the_flag_really_removes_reuse_under_discard():
    """What the column measures: with the tree on, a dropped turn's KV is reused anyway."""
    from layover.agent.cost_model import CostModel, Policy
    from layover.agent.scheduler import AgentConfig
    from layover.replay.simulate import simulate

    costs = CostModel([0.001, 1e-5], [0.001, 1e-5], [0.0, 1 / 15000])
    tokens = {}
    for prefix_cache in (True, False):
        trace = [TraceJob("j", 0.0, [TraceSession("j/m", [TraceTurn(64, 8, "t", 0.1), TraceTurn(64, 8, "t", 0.0)])], "s")]
        agent = AgentConfig(policy=Policy("discard"), prefix_cache=prefix_cache)
        tokens[prefix_cache] = simulate(trace, costs, 64, agent, system_tokens=16).scheduler.stats.prefilled_tokens
    # Reused: 80 for the first turn plus 64 new. Recomputed: the second turn's whole 152 again.
    assert tokens[True] < tokens[False] and tokens[False] == 80 + 152


def calibration_ctx(tmp_path, jct, warmup=2, jobs=6, ladder=(0.1, 0.2, 0.4)):
    config = matrix_config(calibration={"system": "vllm", "unloaded_jobs": 2, "rates_per_s": [9.0], "rates_by_group": {"g": list(ladder)}, "target_slowdown": 1.5})
    trace = [job(f"j{i}") for i in range(jobs)]
    workload = Workload("w", "sweep", ["vllm"], warmup, trace, "s", "h", calibration_group="g")
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in config["systems"]}
    ctx = MatrixContext(config, SETTINGS, systems, {"w": workload}, tmp_path, False, None, 8 * GIB, 100, None)
    return ctx, workload


def fake_replay(calls, jct_by_rung, fail_at=None):
    """Unloaded jobs take 10 s. Each loaded replay takes the next rung's time, and measures job by job."""

    def run(ctx, system, server, jobs, warmup, seed, timeout):
        if len(jobs) == 1:
            calls.append(("alone", jobs[0].job_id))
            return ReplayResult("s", "v", 1, 1, 10.0, 10.0, 1.0, 1.0, jct_s={jobs[0].job_id: 10.0})
        rung = sum(call[0] == "loaded" for call in calls)
        if rung == fail_at:
            raise ServerFailed("rung died")
        calls.append(("loaded", len(jobs), warmup))
        measured = {j.job_id: jct_by_rung[rung] for j in jobs[warmup:]}
        return ReplayResult("s", "v", len(jobs), len(measured), jct_by_rung[rung] + 100, 1.0, 1.0, 1.0, jct_s=measured)

    return run


@contextmanager
def no_server(*_):
    yield None, {}


def test_the_ladder_replays_the_whole_workload_without_warmup(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(calls, [12.0, 16.0, 40.0]))
    record = run_matrix.calibrate(ctx, "g", workload)
    # Two unloaded jobs from the measured ones, then rungs of all 6 jobs with warm-up left out, walking
    # from the middle rung up and stopping as soon as two rungs bracket the target.
    assert calls == [("alone", "j2"), ("alone", "j4"), ("loaded", 6, 2), ("loaded", 6, 2)]
    assert record["bracketed"] and record["edge"] is None
    assert record["saturation_rate_per_s"] == pytest.approx(0.2 * 2 ** ((1.5 - 1.2) / 0.4))
    # The same jobs are compared, not the measured mean of 116.
    assert record["points"][1]["slowdown"] == pytest.approx(1.6) and record["points"][1]["measured_mean_jct_s"] == 116.0
    for text in ("0.1, 0.2, 0.4", "all 6 jobs", "2 as warm-up (4 measured)", "between 0.2 jobs/s (1.20x) and 0.4 jobs/s (1.60x)", "log-interpolated", "held fixed"):
        assert text in record["chosen_by"]
    assert run_matrix.calibrate(ctx, "g", workload) == record and len(calls) == 4


def test_the_ladder_says_when_it_did_not_bracket_the_knee(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    monkeypatch.setattr(run_matrix, "serving", no_server)
    # Past the target at the middle rung and at the one below: it starts past saturation.
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay([], [30.0, 30.0]))
    record = run_matrix.calibrate(ctx, "g", workload)
    assert not record["bracketed"] and record["edge"] == "below_ladder" and "NOT BRACKETED" in record["chosen_by"]
    with pytest.raises(Refused, match="SATURATION NOT FOUND") as refusal:
        run_matrix.refuse_unbracketed(record, False)
    assert "Nothing else has run" in str(refusal.value)
    run_matrix.refuse_unbracketed(record, True)
    other, _ = calibration_ctx(tmp_path / "top", None)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay([], [11.0, 13.0]))
    record = run_matrix.calibrate(other, "g", workload)
    assert not record["bracketed"] and record["edge"] == "above_ladder" and record["saturation_rate_per_s"] == 0.4
    assert "never reached it" in record["chosen_by"]
    with pytest.raises(Refused, match="SATURATION NOT FOUND"):
        run_matrix.refuse_unbracketed(record, False)


def test_a_failed_rung_keeps_what_the_ladder_already_measured(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(calls, [12.0, 40.0], fail_at=1))
    with pytest.raises(ServerFailed):
        run_matrix.calibrate(ctx, "g", workload)
    assert not (tmp_path / "calibration" / "g.json").exists()
    resumed = []
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(resumed, [40.0]))
    record = run_matrix.calibrate(ctx, "g", workload)
    # Neither the unloaded jobs nor the rung already measured run again.
    assert [c[0] for c in resumed] == ["loaded"]
    assert [p["rate_per_s"] for p in record["points"]] == [0.2, 0.4] and record["bracketed"]


def test_the_report_flags_a_rate_that_is_only_a_ladders_edge():
    workload = {"name": "sweep-x", "role": "sweep", "sweeps": []}
    arrival = {"arrival_rate_per_s": 1.6, "chosen_by": "ladder", "bracketed": False, "edge": "above_ladder", "workload": "sweep-y", "system": "vllm"}
    runs = {("sweep-x", "vllm", "load100", 0): {"arrival": arrival, "status": "ok"}}
    manifest = {"workloads": [workload], "config": {"calibration": {"definition": "the stated rule"}}}
    text = "\n".join(report.arrival_section(manifest, runs, Path("/nonexistent")))
    assert "SATURATION NOT FOUND on sweep-y: vllm never reached it" in text
    assert "Saturation: the stated rule" in text and "same absolute rates" in text
    assert "SATURATION NOT FOUND" in "\n".join(report.banners(manifest, runs))
    fine = {("sweep-x", "vllm", "load100", 0): {"arrival": {**arrival, "bracketed": True}, "status": "ok"}}
    assert "SATURATION NOT FOUND" not in "\n".join(report.arrival_section(manifest, fine, Path("/nonexistent")))
    assert report.banners(manifest, fine) == []


def tier_manifest(workloads=("w",), systems=("layover-combined", "vllm", "sglang")) -> dict:
    kinds = {"layover-combined": "layover", "vllm": "vllm", "sglang": "sglang"}
    return {
        "pipeline_check": False, "runs": 1, "systems": list(systems),
        "config": {"systems": [{"name": name, "kind": kinds[name]} for name in systems]},
        "workloads": [
            {"name": name, "role": "headline", "source": "recorded:x", "systems": list(systems), "dropped_jobs": [], "jobs": 4, "warmup_jobs": 1, "sweeps": []}
            for name in workloads
        ],
    }


def tier_run(mean=10.0, peak=None, **counters) -> dict:
    result = fake_result(mean)
    result["metrics"]["kv_pool_used_frac_peak"] = peak
    result["server_metrics"] = {key.replace("__", ":"): value for key, value in counters.items()}
    return {"status": "ok", "result": result}


def occupied_run(occupied: float, **counters) -> dict:
    record = tier_run(10.0, 0.1, **counters)
    record["result"]["metrics"]["kv_pool_occupied_frac_peak"] = occupied
    return record


VLLM_READS = "vllm__external_prefix_cache_hits_total"
SGLANG_READS = "sglang__load_back_tokens_total"


def test_a_baseline_with_no_reads_is_flagged_in_its_table_cells():
    manifest = tier_manifest()
    runs = {
        ("w", "layover-combined", "load100", 0): tier_run(10.0, 0.5, early_demotions=3),
        ("w", "vllm", "load100", 0): tier_run(20.0, 1.0, **{VLLM_READS: 0}),
        # A host gauge is not a read: write-through raises it with no reload.
        ("w", "sglang", "load100", 0): tier_run(30.0, 1.0, sglang__hicache_host_used_tokens=9000),
    }
    workload = manifest["workloads"][0]
    table = "\n".join(report.workload_table(manifest, runs, workload, []))
    assert "| vllm **CPU tier unverified** | 1/1 |" in table and "| sglang **CPU tier unverified** | 1/1 |" in table
    ablation = "\n".join(report.ablation_table(manifest, runs, workload, "layover-combined"))
    assert "| vllm **CPU tier unverified** | 1/1 | run 0 | 2.00* |" in ablation and "| layover-combined | 1/1 | run 0 | 1.00 |" in ablation
    evidence = "\n".join(report.tier_section(manifest, runs))
    assert "| w | load100 | vllm | vllm:external_prefix_cache_hits_total | 1/1 | 0 | 1.000 | CPU tier unverified |" in evidence
    assert "| w | load100 | sglang | sglang:load_back_tokens_total | 1/1 | n/a | 1.000 | CPU tier unverified |" in evidence


def test_a_baseline_that_read_from_its_tier_is_not_flagged():
    manifest = tier_manifest()
    runs = {
        ("w", "vllm", "load100", 0): tier_run(20.0, 1.0, **{VLLM_READS: 5}),
        ("w", "sglang", "load100", 0): tier_run(30.0, 1.0, **{SGLANG_READS: 7}),
    }
    table = "\n".join(report.workload_table(manifest, runs, manifest["workloads"][0], []))
    assert "**" not in table.replace("**missing runs**", "") and "| vllm | 1/1 |" in table and "| sglang | 1/1 |" in table


def test_no_reads_and_no_pressure_says_unused_and_unknown_pressure_stays_unverified():
    manifest = tier_manifest()
    workload = manifest["workloads"][0]
    calm = {("w", "vllm", "load100", 0): tier_run(20.0, 0.6, **{VLLM_READS: 0}), ("w", "layover-combined", "load100", 0): occupied_run(0.7, early_demotions=0)}
    assert report.tier_flag(manifest, calm, workload, "vllm") == "CPU tier unused, KV never filled on Layover"
    pressed_elsewhere = {**calm, ("w", "layover-combined", "load100", 0): occupied_run(0.7, early_demotions=2)}
    assert report.tier_flag(manifest, pressed_elsewhere, workload, "vllm") == "CPU tier unverified"
    unknown = {("w", "vllm", "load100", 0): tier_run(20.0, None, **{VLLM_READS: 0})}
    assert report.tier_flag(manifest, unknown, workload, "vllm") == "CPU tier unverified"
    assert report.tier_flag(manifest, calm, workload, "layover-combined") is None


def test_one_run_without_reads_flags_the_cell_and_a_sweep_cannot_hide_the_headline():
    manifest = tier_manifest(workloads=("headline", "sweep"), systems=("vllm",))
    runs = {
        ("headline", "vllm", "load100", 0): tier_run(20.0, 1.0, **{VLLM_READS: 0}),
        ("headline", "vllm", "load100", 1): tier_run(20.0, 1.0, **{VLLM_READS: 9}),
        ("sweep", "vllm", "load100", 0): tier_run(20.0, 1.0, **{VLLM_READS: 5000}),
    }
    headline, sweep = manifest["workloads"]
    assert report.tier_flag(manifest, runs, headline, "vllm") == "CPU tier unverified"
    assert report.tier_flag(manifest, runs, sweep, "vllm") is None


def test_the_tier_flag_is_per_load():
    """A light load may never fill the pool. That must not hide, or borrow, the heavy load's verdict."""
    manifest = tier_manifest(systems=("layover-combined", "vllm"))
    runs = {
        ("w", "vllm", "load50", 0): tier_run(20.0, 0.5, **{VLLM_READS: 0}), ("w", "vllm", "load100", 0): tier_run(20.0, 1.0, **{VLLM_READS: 0}),
        ("w", "layover-combined", "load50", 0): occupied_run(0.5), ("w", "layover-combined", "load100", 0): occupied_run(1.0),
    }
    workload = manifest["workloads"][0]
    assert report.tier_flag(manifest, runs, workload, "vllm", "load50") == "CPU tier unused, KV never filled on Layover"
    assert report.tier_flag(manifest, runs, workload, "vllm", "load100") == "CPU tier unverified"


def test_sglang_reload_counter_is_kept_from_its_metrics():
    class Client:
        def get(self, _):
            return httpx.Response(200, text='sglang:load_back_tokens_total{pool="kv"} 12\nsglang:hicache_host_used_tokens 3\n', request=httpx.Request("GET", "http://x"))

    kept = SglangSystem("s", SETTINGS, TOKEN_BYTES).server_metrics(Client())
    assert kept["sglang:load_back_tokens_total"] == 12.0
    assert report.CPU_TIER_EVIDENCE["sglang"] == "sglang:load_back_tokens_total"


def test_a_preflight_saves_each_baselines_metric_names(tmp_path):
    class Server:
        system = VllmSystem("v", SETTINGS, TOKEN_BYTES)
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="vllm:x 1\n")), base_url="http://x")

    run_matrix.save_metric_names(Server(), tmp_path / "logs" / "v.metrics.txt")
    assert (tmp_path / "logs" / "v.metrics.txt").read_text() == "vllm:x 1\n"


def published_sweeps(qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    settings = MatchedSettings.from_config(config["matched"])
    return config, settings, [w for w in run_matrix.load_workloads(config, settings, False, dry_run=True) if w.role == "sweep"]


@pytest.mark.skipif(not (REPO / "results/calibration/calibration.json").exists(), reason="the sweeps are generated from this file")
def test_every_published_sweep_point_separates_in_simulation(qwen3_4b_config):
    config, settings, workloads = published_sweeps(qwen3_4b_config)
    selection = config["sweeps"]["selection"]
    assert {w.name for w in workloads} == set(selection["simulated"])
    for rate in selection["expected_rates_per_s"]:
        assert rate in config["calibration"]["rates_by_group"]["synthetic"]
    for workload in workloads:
        found = run_matrix.simulate_point(config, settings, qwen3_4b_config, workload.jobs, workload.warmup_jobs, selection["expected_rates_per_s"][0], config["seed_base"], ("combined", "discard"))
        assert run_matrix.point_separates([found], selection["min_gap"]), (workload.name, found)
    assert all(w.source.startswith("synthetic") for w in workloads)


@pytest.mark.skipif(not (REPO / "results/calibration/calibration.json").exists(), reason="the sweeps are generated from this file")
def test_the_separation_check_rejects_a_point_without_pressure(qwen3_4b_config):
    config, settings, _ = published_sweeps(qwen3_4b_config)
    config["sweeps"]["base"]["jobs"] = 12
    jobs = run_matrix._synthetic_jobs(config["sweeps"], {"tool_s_median": 0.5, "branching": 0})
    found = run_matrix.simulate_point(config, settings, qwen3_4b_config, jobs, 2, 1.6, 1000, ("combined", "discard"))
    assert not run_matrix.point_separates([found], config["sweeps"]["selection"]["min_gap"])


def test_point_separates_needs_a_gap_and_early_demotions():
    def row(gap, early):
        return {"gap": gap, "combined_early_demotions": early}

    assert run_matrix.point_separates([row(1.2, 1)], 1.15)
    assert not run_matrix.point_separates([row(1.1, 9)], 1.15)
    assert not run_matrix.point_separates([row(2.0, 0)], 1.15)
    # A gap must hold in every run; early demotions in the median run, as a table cell reports.
    assert run_matrix.point_separates([row(2.0, 0), row(2.0, 1), row(2.0, 4)], 1.15)
    assert not run_matrix.point_separates([row(2.0, 0), row(2.0, 0), row(2.0, 4)], 1.15)
    assert not run_matrix.point_separates([row(1.1, 5), row(2.0, 5), row(2.0, 5)], 1.15)


def cost_model_ctx(tmp_path):
    path = tmp_path / "cost_model.json"
    path.write_text('{"chunk": 2048}')
    config = matrix_config()
    config["systems"][0]["cost_model"] = str(path)
    return config, path


def test_the_cost_model_hash_reaches_the_manifest_and_every_run_record(tmp_path, monkeypatch):
    config, path = cost_model_ctx(tmp_path)
    workloads = [Workload("w", "headline", ["layover-combined", "vllm"], 0, [job()], "recorded:x", "h", arrival_rate_per_s=1.0)]
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in config["systems"]}
    ctx = MatrixContext(config, SETTINGS, systems, {"w": workloads[0]}, tmp_path, False, None, 8 * GIB, 100, None)
    manifest = run_matrix.manifest_for(config, workloads, [], False)
    assert manifest["cost_model_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    for system in ("layover-combined", "vllm"):
        record = run_matrix.run_cell(ctx, Cell("w", system, 0), {"arrival_rate_per_s": 1.0})
        assert record["cost_model_hash"] == manifest["cost_model_hash"]


def test_resume_refuses_a_re_measured_cost_model(tmp_path):
    config, path = cost_model_ctx(tmp_path)
    workloads = [Workload("w", "headline", ["layover-combined", "vllm"], 0, [job()], "recorded:x", "h", arrival_rate_per_s=1.0)]
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.manifest_for(config, workloads, [], False))
    run_matrix.check_resume(tmp_path, run_matrix.manifest_for(config, workloads, [], False))
    path.write_text('{"chunk": 2048, "recompute_s_coef": [1]}')
    # The config still holds only the path, so its hash cannot see this.
    assert run_matrix.manifest_for(config, workloads, [], False)["config_hash"] == json.loads((tmp_path / "manifest.json").read_text())["config_hash"]
    with pytest.raises(Refused, match="different cost model"):
        run_matrix.check_resume(tmp_path, run_matrix.manifest_for(config, workloads, [], False))


def test_a_cost_model_measured_at_another_chunk_is_refused(tmp_path):
    config, path = cost_model_ctx(tmp_path)
    run_matrix.check_cost_model_chunk(config, SETTINGS, {"layover-combined"})
    path.write_text('{"chunk": 1024}')
    with pytest.raises(Refused, match="--chunk 2048"):
        run_matrix.check_cost_model_chunk(config, SETTINGS, {"layover-combined"})
    run_matrix.check_cost_model_chunk(config, SETTINGS, {"vllm"})


def test_the_report_says_exactly_which_columns_leave_out_warmup():
    manifest = {"pipeline_check": False, "runs": 1}
    workload = {"name": "w", "role": "headline", "source": "recorded:x", "systems": [], "dropped_jobs": [], "jobs": 4, "warmup_jobs": 1}
    text = "\n".join(report.workload_table(manifest, {}, workload, []))
    assert "from every column" not in text and "job-time and first-token columns only" in text and "warm-up jobs still running" in text
    assert "warm-up jobs still running" in PCIE_NOTE
    flags_doc = (REPO / "bench/configs/baseline_flags.md").read_text()
    assert "GPU and PCIe figures included" not in flags_doc and "warm-up jobs still running" in flags_doc


@pytest.mark.parametrize(
    "measured, text",
    [(25, "the second-longest job"), (10, "the longest job, the maximum"), (35, "the second-longest job"), (40, "the third-longest job"), (100, "job 6 from the longest")],
)
def test_the_p95_note_names_the_job_it_is(measured, text):
    assert f"over {measured} measured jobs, so it is {text}," in report.p95_note(measured)
    # The note follows the percentile the tables use.
    from_top = measured - percentile(list(range(measured)), 0.95)
    assert from_top == {"the longest job, the maximum": 1, "the second-longest job": 2, "the third-longest job": 3}.get(text, 6)


def test_tool_time_is_the_longest_chain_of_pauses():
    assert job_tool_s(job(tool=2.5)) == 2.5
    forked = TraceJob(
        "j", 0.0,
        [
            TraceSession("m", [TraceTurn(1, 1, "t", 3.0), TraceTurn(1, 1, "t", 4.0), TraceTurn(1, 1, "t", 99.0)]),
            # Forked after turn 1, which starts 3 s in, so this chain is 3 + 10 + 5.
            TraceSession("c", [TraceTurn(1, 1, "t", 10.0), TraceTurn(1, 1, "t", 5.0), TraceTurn(1, 1, "t", 77.0)], parent="m", fork_after_turn=1),
        ],
        "s",
    )
    assert job_tool_s(forked) == 18.0


def test_summary_leaves_tool_time_out_only_where_it_says():
    arrivals = {"a": 0.0, "b": 0.0}
    finishes = {"a": 100.0, "b": 50.0}
    result = summarize("s", "x", arrivals, finishes, [], tool_s={"a": 90.0, "b": 10.0})
    assert result.mean_jct_minus_tool_s == pytest.approx(25.0) and result.p95_jct_minus_tool_s == pytest.approx(40.0)
    assert result.p95_jct_s == 100.0
    assert summarize("s", "x", arrivals, finishes, []).p95_jct_minus_tool_s is None


def test_replay_takes_each_jobs_scaled_tool_time_off(stub_url):
    result = asyncio.run(replay([job(tool=0.2)], stub_url, "m", 100, "stub", hints=False, system_tokens=8, time_scale=0.5))
    assert result.p95_jct_minus_tool_s == pytest.approx(result.p95_jct_s - 0.1)


def test_a_run_record_keeps_the_tool_free_tail_and_each_jobs_time(tmp_path, monkeypatch):
    class Fake:
        client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.1)
        url = "http://127.0.0.1:1"
        log_text = staticmethod(str)

    @contextmanager
    def fake_serving(*_):
        yield Fake(), {}

    result = ReplayResult(
        "recorded:x", "s", 1, 1, 9.0, 9.0, 1.0, 1.0, token_sources={"server": 1}, jct_s={"j": 9.0},
        mean_jct_minus_tool_s=2.0, p95_jct_minus_tool_s=3.0,
    )
    monkeypatch.setattr(run_matrix, "serving", fake_serving)
    monkeypatch.setattr(run_matrix, "_replay_jobs", lambda *_: result)
    ctx, _ = make_ctx(tmp_path)
    monkeypatch.setattr(ctx.systems["vllm"], "server_metrics", lambda client: {})
    saved = run_matrix.measure_run(ctx, Cell("w", "vllm", 0), tmp_path / "log", {"arrival_rate_per_s": 1.0})
    assert saved["metrics"]["p95_jct_minus_tool_s"] == 3.0 and saved["metrics"]["mean_jct_minus_tool_s"] == 2.0
    assert saved["jct_s"] == {"j": 9.0}
    assert "p95_jct_minus_tool_s" in {key for key, _, _ in report.METRICS}


def test_host_facts_use_what_the_sync_recorded_when_there_is_no_git(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "REPO", tmp_path)
    state = tmp_path / ".layover-run"
    state.mkdir()
    (state / "synced_commit").write_text("f415951-dirty-2554875262\n")
    facts = run_matrix.host_facts(None)
    assert facts["commit"] == "f415951-dirty-2554875262" and facts["dirty"] is True
    (state / "synced_commit").write_text("f415951\n")
    assert run_matrix.host_facts(None)["dirty"] is False
    # A real checkout wins over a state file left behind.
    (tmp_path / ".git").mkdir()
    assert run_matrix.host_facts(None)["commit"] != "f415951"
    (tmp_path / ".git").rmdir()
    (state / "synced_commit").unlink()
    assert run_matrix.host_facts(None)["commit"] is None


def test_the_refusal_of_an_unrecorded_headline_names_the_real_step():
    with pytest.raises(Refused, match="make traces") as refusal:
        run_matrix.refuse_unrecorded("w", [job(source="pipeline-check:x")], False)
    assert "Record real" not in str(refusal.value)


def test_the_matrix_refuses_a_cost_model_from_another_chunk_before_starting_anything(tmp_path, qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    config["layover"]["cost_model"] = str(tmp_path / "cost_model.json")
    (tmp_path / "cost_model.json").write_text('{"chunk": 1024}')
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config))
    with pytest.raises(Refused, match="--chunk 2048"):
        run_matrix.main(["--config", str(path), "--preflight-only", "--out", str(tmp_path / "o")])


def test_the_separation_table_is_labelled_a_simulation(tmp_path, capsys, qwen3_4b_config):
    config = sweep_config()
    config["sweeps"]["selection"] = {"min_gap": 1.15, "swap_s_per_token": 7.4e-6}
    config["calibration"] = {"system": "vllm", "unloaded_jobs": 1, "rates_per_s": [1.0], "target_slowdown": 1.5}
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config))
    assert run_matrix.main(["--config", str(path), "--sweep-separation"]) == 0
    printed = capsys.readouterr().out
    assert "SIMULATED" in printed and "not a measurement" in printed
    assert re.search(r"sweep-tool0.5-branch2 at 1/s: gap [\d.]+ to [\d.]+, .*does not separate", printed)


def test_estimate_limits_decode_to_what_fits_in_the_pool():
    assumed = {"startup_s": 0, "prefill_tokens_per_s": 1e9, "decode_tokens_per_s_per_stream": 10, "decode_tokens_per_s_total": 1000}
    jobs = [job(str(i), new=100, output=50, tool=0.0) for i in range(4)]
    # Peak context 100 + 50 + 5 + 50 = 205. Unlimited, the serial bound is 400 / 1000 and one job's own decode 10 s wins.
    assert run_matrix.estimate_run_s(jobs, 1e9, assumed) == pytest.approx(10.0, abs=0.01)
    # A pool of one and a half contexts decodes 15 tokens/s, so 400 tokens take 26.7 s.
    assert run_matrix.estimate_run_s(jobs, 1e9, assumed, pool_tokens=int(205 * 1.5)) == pytest.approx(400 / 15, abs=0.1)
    # Never below one stream, whatever the pool.
    assert run_matrix.estimate_run_s(jobs, 1e9, assumed, pool_tokens=1) == pytest.approx(40.0, abs=0.01)


def test_dry_run_counts_a_recomputing_system_at_the_slow_end(capsys, qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    settings = MatchedSettings.from_config(config["matched"])
    workloads = run_matrix.load_workloads(config, settings, False, dry_run=True)
    cells = run_matrix.build_cells(workloads, config["runs"])

    def low_total(prefix_cache: bool) -> float:
        raw = json.loads(json.dumps(config))
        next(s for s in raw["systems"] if s["name"] == "layover-discard")["prefix_cache"] = prefix_cache
        systems = {spec["name"]: build_system(spec, settings, TOKEN_BYTES) for spec in run_matrix.system_specs(raw)}
        return run_matrix.dry_run(raw, settings, systems, workloads, cells, pool_tokens=87381)

    assert low_total(False) > low_total(True)
    assert "layover-discard has no prefix cache" in capsys.readouterr().out


def test_dry_run_plans_the_ladder_on_the_whole_workload(capsys):
    config = sweep_config()
    # A slow rung, so that a whole workload's arrivals span far longer than one job's run.
    config["calibration"] = {"system": "vllm", "unloaded_jobs": 1, "rates_per_s": [0.001], "target_slowdown": 1.5}
    workloads = run_matrix.load_workloads(config, SETTINGS, False)
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in config["systems"]}
    cells = run_matrix.build_cells(workloads, 1)
    run_matrix.dry_run(config, SETTINGS, systems, workloads, cells)
    assumed = config["estimate_assumptions"]
    reference = next(w for w in workloads if w.calibration_reference)
    sample = reference.jobs[1:][:1]
    expected = run_matrix.estimate_run_s(sample, 1.0, assumed) + run_matrix.estimate_run_s(reference.jobs, 0.001, assumed)
    assert expected > 1800
    assert f"calibration about {expected / 60:.0f} to" in capsys.readouterr().out


def test_a_plan_without_the_trace_uses_the_prepared_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "REPO", tmp_path)
    prepared = tmp_path / "traces" / "prepared"
    prepared.mkdir(parents=True)
    workload = Workload("w", "headline", [], 0, estimate_jobs=2)
    jobs, label = run_matrix.planning_jobs(workload)
    assert "synthetic" in label and len(jobs) == 2
    from layover.replay import trajectories

    (prepared / "manifest.json").write_text("{}")
    tasks = ["a", "b", "c"]
    monkeypatch.setattr(trajectories, "load_manifest", lambda directory: {"tasks": tasks})
    document = {"turns": [{"new_tokens": 10, "output_tokens": 5, "recorded_gap_s": 2.0}, {"new_tokens": 3, "output_tokens": 1, "recorded_gap_s": None}]}
    monkeypatch.setattr(trajectories, "load_prepared", lambda directory, task: document)
    jobs, label = run_matrix.planning_jobs(workload)
    assert "traces/prepared" in label and [j.job_id for j in jobs] == ["a", "b"]
    assert jobs[0].main.turns[0].tool_s == 2.0 and jobs[0].main.turns[1].tool_s == 0.0


def run_until_killed(body) -> None:
    """Run body in a child that dies by SIGKILL, as on a spot reclaim: no finally, no flush, nothing saved."""
    pid = os.fork()
    if pid == 0:
        signal.alarm(60)  # A fork in a threaded process could hang; that should fail, not wedge the suite.
        try:
            body()
        finally:
            os._exit(99)  # Only if nothing killed it, which the assert below turns into a failure.
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL, "the child was meant to be killed"


def die(*_, **__) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def resume_args() -> argparse.Namespace:
    return argparse.Namespace(skip_preflight=True, preflight_only=False)


# Each kills a session while it handles its second run (index 1), and says how many runs a resume should find finished.
def kill_during_the_run(_):
    pass


def kill_after_rename(monkeypatch):
    real = os.replace

    def replace_then_die(source, target):
        real(source, target)
        if Path(target).name == "run-0.json" and "vllm" in str(target):
            die()

    monkeypatch.setattr(run_matrix.os, "replace", replace_then_die)


def kill_before_rename(monkeypatch):
    real = os.replace

    def die_first(source, target):
        if Path(target).name == "run-0.json" and "vllm" in str(target):
            die()
        real(source, target)

    monkeypatch.setattr(run_matrix.os, "replace", die_first)


def kill_mid_write(monkeypatch):
    real = os.fsync
    calls = 0

    def torn(descriptor):
        # A record fsyncs its file then its directory, so the third call is the second run's file.
        # Half of it is on disk under its temporary name when the machine goes.
        nonlocal calls
        calls += 1
        if calls == 3:
            os.ftruncate(descriptor, os.fstat(descriptor).st_size // 2)
            die()
        return real(descriptor)

    monkeypatch.setattr(run_matrix.os, "fsync", torn)


KILL_POINTS = [
    # The second run dies inside its measurement: one finished before it.
    ("during-the-run", kill_during_the_run, 1),
    # Half a temporary file: not finished, and the leftover must not count.
    ("mid-write", kill_mid_write, 1),
    # The whole temporary file, not yet renamed: not finished either.
    ("before-the-rename", kill_before_rename, 1),
    # Renamed: this run is safe, so a resume must not repeat it.
    ("after-the-rename", kill_after_rename, 2),
]


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
@pytest.mark.parametrize("name, arm, finished", KILL_POINTS, ids=[point[0] for point in KILL_POINTS])
def test_a_kill_at_any_point_leaves_exactly_the_finished_runs(tmp_path, monkeypatch, name, arm, finished):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))

    def session():
        calls = 0

        def measure(*_):
            nonlocal calls
            calls += 1
            # On its second call, which is the run the kill point is about.
            if name == "during-the-run" and calls == 2:
                die()
            return fake_result(1.0)

        run_matrix.measure_run = measure
        arm(pytest.MonkeyPatch())
        run_matrix.run_all(ctx, resume_args(), workloads, cells, None)

    run_until_killed(session)
    done = [c for c in cells if run_matrix.run_finished(run_matrix.load_run(ctx.run_path(c)))]
    assert [c.label for c in done] == [c.label for c in cells[:finished]]
    before = {c: ctx.run_path(c).stat().st_mtime_ns for c in done}
    if name in ("mid-write", "before-the-rename"):
        assert list((tmp_path / "runs").rglob("*.tmp")), "the kill should have left a temporary file"

    repeated = []

    def measure(_, cell, *__):
        repeated.append(cell.label)
        return fake_result(1.0)

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    # Exactly the rest, in order, and the finished ones untouched.
    assert repeated == [c.label for c in cells[finished:]]
    assert {c: ctx.run_path(c).stat().st_mtime_ns for c in done} == before
    assert all(run_matrix.run_finished(run_matrix.load_run(ctx.run_path(c))) for c in cells)
    # A leftover temporary file never reaches a table.
    text = report.write_report(tmp_path).read_text()
    assert "## Missing runs\n\nNone." in text
    assert len(report.load_results(tmp_path)[1]) == len(cells)


def test_a_torn_record_counts_as_not_done_and_is_rerun(tmp_path, monkeypatch, capsys):
    """rsync cut off mid-file, say, on the way back from the machine that was reclaimed."""
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    path = ctx.run_path(cells[1])
    path.write_text(path.read_text()[:100])
    # Not a parse error in the report either: it is a missing run, and it says so.
    missing = report.write_report(tmp_path).read_text()
    assert f"`{cells[1].label}`: not run" in missing
    repeated = []
    monkeypatch.setattr(run_matrix, "measure_run", lambda _, cell, *__: repeated.append(cell.label) or fake_result(1.0))
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    assert repeated == [cells[1].label] and "is not valid JSON" in capsys.readouterr().out
    assert run_matrix.run_finished(run_matrix.load_run(path))


def test_a_run_only_counts_with_a_status_of_ok_and_a_result():
    assert run_matrix.run_finished({"status": "ok", "result": {"metrics": {}}})
    for record in (None, {}, {"status": "failed", "result": {"metrics": {}}}, {"status": "ok"}, {"status": "ok", "result": None}, {"status": "ok", "result": {}}):
        assert not run_matrix.run_finished(record)


def test_a_rerun_that_fails_leaves_no_old_result_to_be_mistaken_for_it(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path)
    cell = Cell("w", "vllm", 0)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    assert run_matrix.run_cell(ctx, cell, {"arrival_rate_per_s": 1.0})["status"] == "ok"

    def fails(*_):
        raise ServerFailed("boom")

    monkeypatch.setattr(run_matrix, "measure_run", fails)
    record = run_matrix.run_cell(ctx, cell, {"arrival_rate_per_s": 1.0})
    assert record["status"] == "failed" and "result" not in record


def test_a_record_is_flushed_to_disk_before_it_gets_its_name(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(run_matrix.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(run_matrix.os, "replace", lambda a, b: (order.append("rename"), real_replace(a, b))[1])
    run_matrix.write_json(tmp_path / "x" / "r.json", {"a": 1})
    # The file's data, then its name, then the directory holding that name.
    assert order == ["fsync", "rename", "fsync"]
    assert json.loads((tmp_path / "x" / "r.json").read_text()) == {"a": 1}
    assert not list((tmp_path / "x").glob("*.tmp"))


def test_a_leftover_temporary_file_does_not_block_or_count(tmp_path):
    stale = tmp_path / f"r.json.{os.getpid()}.tmp"
    stale.write_text('{"status": "ok", "torn')
    (tmp_path / "s.json.99999.tmp").write_text("{")
    assert run_matrix.load_run(tmp_path / "r.json") is None
    run_matrix.write_json(tmp_path / "r.json", {"a": 2})
    assert run_matrix.load_run(tmp_path / "r.json") == {"a": 2}


CALIBRATION_KILLS = [
    # On the second unloaded job: the first stays paid for.
    ("second-unloaded-job", 1, [("alone", "j4"), ("loaded", 6, 2), ("loaded", 6, 2)]),
    # On the first rung: all unloaded jobs stay paid for.
    ("first-rung", 2, [("loaded", 6, 2), ("loaded", 6, 2)]),
    # On the second rung: the first rung does too.
    ("second-rung", 3, [("loaded", 6, 2)]),
]


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
@pytest.mark.parametrize("name, dies_on_call, then", CALIBRATION_KILLS, ids=[point[0] for point in CALIBRATION_KILLS])
def test_a_kill_during_calibration_repeats_only_what_was_in_progress(tmp_path, monkeypatch, name, dies_on_call, then):
    ctx, workload = calibration_ctx(tmp_path, None)
    monkeypatch.setattr(run_matrix, "serving", no_server)

    def session():
        calls = []
        inner = fake_replay(calls, [12.0, 16.0])

        def run(*arguments):
            if len(calls) == dies_on_call:
                die()
            return inner(*arguments)

        run_matrix._replay_jobs = run
        run_matrix.calibrate(ctx, "g", workload)

    run_until_killed(session)
    assert not (tmp_path / "calibration" / "g.json").exists()
    resumed = []
    remaining = {"second-unloaded-job": [12.0, 16.0], "first-rung": [12.0, 16.0], "second-rung": [16.0]}[name]
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(resumed, remaining))
    record = run_matrix.calibrate(ctx, "g", workload)
    assert resumed == then
    assert record["bracketed"] and [p["rate_per_s"] for p in record["points"]] == [0.2, 0.4]
    assert record["unloaded_mean_jct_s"] == pytest.approx(10.0)
    # And a finished calibration costs nothing again.
    assert run_matrix.calibrate(ctx, "g", workload) == record and resumed == then


def test_max_hours_stops_a_calibration_cleanly_and_keeps_what_it_measured(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    monkeypatch.setattr(run_matrix, "serving", no_server)
    inner = fake_replay(calls, [12.0, 16.0])

    def run(*arguments):
        result = inner(*arguments)
        # The budget ends during the first rung.
        if calls[-1][0] == "loaded":
            ctx.deadline = time.monotonic() - 1
        return result

    monkeypatch.setattr(run_matrix, "_replay_jobs", run)
    cell = Cell("w", "vllm", 0)
    code = run_matrix.run_all(ctx, resume_args(), [workload], [cell], None)
    assert code == 1
    assert json.loads((tmp_path / "session.json").read_text())["budget_skipped"] == [cell.label]
    assert not (tmp_path / "calibration" / "g.json").exists() and not ctx.run_path(cell).exists()
    assert len(json.loads((tmp_path / "calibration" / "partial" / "g.json").read_text())["points"]) == 1


def host(instance: str = "i-1", gpu: str = "L4", driver: str = "550") -> dict:
    return {"instance_id": instance, "gpu_name": gpu, "gpu_driver": driver, "commit": "abc", "dirty": False}


def test_a_new_machine_resumes_the_same_runs_and_hardware_must_match(tmp_path):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    first = run_matrix.manifest_for(ctx.config, workloads, cells, False)
    first["hardware"] = {"name": "L4", "driver": "550", "memory_total_bytes": 24}
    first["host"] = {"instance_id": "i-1"}
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.merge_manifest(first, None, cells))
    # A different instance id is fine: it is not even compared.
    second = {**first, "host": {"instance_id": "i-2"}}
    run_matrix.check_resume(tmp_path, second)
    for changed, field in (({"name": "A10G"}, "A10G"), ({"driver": "560"}, "560"), ({"memory_total_bytes": 48}, "48")):
        other = {**first, "hardware": {**first["hardware"], **changed}}
        with pytest.raises(Refused, match=f"{field}.*--accept-mixed-hardware"):
            run_matrix.check_resume(tmp_path, other)
        run_matrix.check_resume(tmp_path, other, accept_mixed_hardware=True)
    # The first machine's stays the reference, and both instances are remembered.
    merged = run_matrix.merge_manifest({**first, "hardware": {"name": "A10G"}, "host": {"instance_id": "i-2"}}, run_matrix.load_run(tmp_path / "manifest.json"), [])
    assert merged["hardware"]["name"] == "L4" and [h["instance_id"] for h in merged["hosts"]] == ["i-1", "i-2"]


def test_the_cost_model_hash_survives_a_move_when_the_file_is_restored(tmp_path):
    config, path = cost_model_ctx(tmp_path)
    workloads = [Workload("w", "headline", ["layover-combined", "vllm"], 0, [job()], "recorded:x", "h", arrival_rate_per_s=1.0)]
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.manifest_for(config, workloads, [], False))
    # As fetched and restored: other times, other inode, same bytes.
    contents = path.read_bytes()
    path.unlink()
    path.write_bytes(contents)
    os.utime(path, (1, 1))
    run_matrix.check_resume(tmp_path, run_matrix.manifest_for(config, workloads, [], False))
    # Re-measured on the new machine instead: refused, the same as on the old one.
    path.write_text('{"chunk": 2048, "swap_in_s_coef": [9]}')
    with pytest.raises(Refused, match="different cost model"):
        run_matrix.check_resume(tmp_path, run_matrix.manifest_for(config, workloads, [], False))


def test_every_finished_record_must_match_this_session_even_without_a_manifest(tmp_path, monkeypatch):
    """A restored directory can lose its manifest. Each record still says what it came from."""
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 1)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    (tmp_path / "manifest.json").unlink()
    run_matrix.refuse_mixed_results(ctx, cells, {})
    for key, value in (("config_hash", "other"), ("cost_model_hash", "other"), ("trace_hash", "other")):
        record = json.loads(ctx.run_path(cells[0]).read_text())
        saved = record[key]
        record[key] = value
        run_matrix.write_json(ctx.run_path(cells[0]), record)
        with pytest.raises(Refused, match=key):
            run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
        record[key] = saved
        run_matrix.write_json(ctx.run_path(cells[0]), record)


def test_each_record_names_its_instance_and_the_report_counts_them(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    ctx.instance = host("i-1")
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells[:2], None) == 0
    record = json.loads(ctx.run_path(cells[0]).read_text())
    assert record["instance"]["instance_id"] == "i-1" and record["instance"]["gpu_name"] == "L4"
    assert record["attempts"][0]["instance_id"] == "i-1"
    text = report.write_report(tmp_path).read_text()
    assert "Runs here came from 1 instance." in text and "MIXED" not in text and "2 finished runs on 1 instance." in text

    ctx.instance = host("i-2")
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    text = report.write_report(tmp_path).read_text()
    assert "Runs here came from 2 instances." in text and "MIXED" not in text and "4 finished runs on 2 instances." in text
    assert "| i-1 | L4 | 550 | abc | 2 |" in text and "| i-2 | L4 | 550 | abc | 2 |" in text


def test_mixed_hardware_or_code_is_flagged_at_the_top_and_on_the_table(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(1.0))
    ctx.instance = host("i-1")
    run_matrix.run_all(ctx, resume_args(), workloads, cells[:2], None)
    ctx.instance = {**host("i-2", driver="560"), "commit": "def"}
    run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
    text = report.write_report(tmp_path).read_text()
    head = text.split("## Results")[0]
    assert "MIXED HARDWARE" in head and "MIXED CODE" in head
    table = text.split("## Results")[1].split("## Ablation")[0]
    assert "MIXED HARDWARE: L4, driver 550; L4, driver 560." in table and "MIXED CODE: abc, def." in table


def test_instance_id_comes_from_the_environment_or_a_local_file(monkeypatch):
    monkeypatch.setenv("LAYOVER_INSTANCE_ID", "i-env")
    assert run_matrix.instance_id() == "i-env"
    monkeypatch.delenv("LAYOVER_INSTANCE_ID")
    assert run_matrix.instance_id()


def test_a_killed_attempts_log_does_not_answer_for_the_next_server(tmp_path):
    log = tmp_path / "log"
    log.write_text("KV cache size: 999 tokens\n")
    system = CommandSystem("c", SETTINGS, TOKEN_BYTES, [sys.executable, "-c", "print('fresh', flush=True)"])
    server = ManagedServer(system, log, free_port())
    server.start()
    server.process.wait(timeout=30)
    assert "999" not in server.log_text() and "fresh" in server.log_text()
    server.stop()


def multi_load_run(ctx, cell: Cell, mean: float, rate: float, monkeypatch) -> None:
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(mean))
    run_matrix.run_cell(ctx, cell, {"arrival_rate_per_s": rate, "chosen_by": "fixed"})


def test_a_table_and_a_line_of_jct_against_load(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    workloads[0].load_fractions = [0.5, 1.0, 1.25]
    cells = run_matrix.build_cells(workloads, 2)
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.manifest_for(ctx.config, workloads, cells, False))
    rates = {"load50": 0.5, "load100": 1.0, "load125": 1.25}
    for cell in cells:
        base = {"vllm": 40.0, "layover-combined": 20.0}[cell.system]
        multi_load_run(ctx, cell, base * rates[cell.load], rates[cell.load], monkeypatch)
    text = report.write_report(tmp_path).read_text()
    assert "### Mean JCT (s) against load, w" in text and "| System | load50 (0.5 jobs/s) | load100 (1 jobs/s) | load125 (1.25 jobs/s) |" in text
    # Median over 2 runs at each load, by hand.
    assert "| layover-combined | 10.00 (10.00 to 10.00, IQR 0.00) | 20.00 (20.00 to 20.00, IQR 0.00) | 25.00 (25.00 to 25.00, IQR 0.00) |" in text
    # A table per load, and every load's ratio is against the same load's reference.
    assert "### w, load50 (0.5 jobs/s): headline" in text and "### w, load125 (1.25 jobs/s): headline" in text
    assert text.count("relative to layover-combined, trace source") == 3
    assert (tmp_path / "plots" / "w-jct-vs-load.png").stat().st_size > 1000
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    row = next(r for r in rows if (r["system"], r["load"], r["metric"]) == ("vllm", "load50", "mean_jct_s"))
    assert float(row["median"]) == 20.0 and float(row["arrival_rate_per_s"]) == 0.5


def test_the_load_plot_draws_the_same_rates_for_every_system_and_one_axis(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    workloads[0].load_fractions = [0.5, 1.0]
    cells = run_matrix.build_cells(workloads, 1)
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.manifest_for(ctx.config, workloads, cells, False))
    for cell in cells:
        multi_load_run(ctx, cell, 5.0, {"load50": 0.5, "load100": 1.0}[cell.load], monkeypatch)
    drawn = {}
    import matplotlib.axes as axes

    original = axes.Axes.plot
    monkeypatch.setattr(axes.Axes, "plot", lambda self, xs, ys, **k: (drawn.__setitem__(k["label"], list(xs)), original(self, xs, ys, **k))[1])
    manifest, runs = report.load_results(tmp_path)
    assert report.load_plot(tmp_path / "x.png", manifest, runs, manifest["workloads"][0], "mean_jct_s", "s", report.system_colors(manifest["systems"]))
    assert drawn == {"layover-combined": [0.5, 1.0], "vllm": [0.5, 1.0]}


def test_a_single_load_workload_has_no_load_plot_or_table(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path, runs=1)
    multi_load_run(ctx, Cell("w", "vllm", 0), 5.0, 1.0, monkeypatch)
    manifest, runs = report.load_results(tmp_path)
    workload = manifest["workloads"][0]
    assert not report.load_plot(tmp_path / "x.png", manifest, runs, workload, "mean_jct_s", "s", report.system_colors(manifest["systems"]))
    assert report.load_table(manifest, runs, workload, "mean_jct_s", "s", "{:.2f}") == []


def test_a_narrower_session_reports_only_what_was_asked_of_this_directory(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path)
    cells = run_matrix.build_cells(workloads, 2)
    manifest = run_matrix.manifest_for(ctx.config, workloads, cells[:1], False)
    run_matrix.write_json(tmp_path / "manifest.json", run_matrix.merge_manifest(manifest, None, cells[:1]))
    multi_load_run(ctx, cells[0], 1.0, 1.0, monkeypatch)
    assert "## Missing runs\n\nNone." in report.write_report(tmp_path).read_text()
    # A wider one later: the union is expected, and nothing is forgotten.
    merged = run_matrix.merge_manifest(run_matrix.manifest_for(ctx.config, workloads, cells[1:2], False), run_matrix.load_run(tmp_path / "manifest.json"), cells[1:2])
    run_matrix.write_json(tmp_path / "manifest.json", merged)
    assert len(merged["cells"]) == 2
    assert f"`{cells[1].label}`: not run" in report.write_report(tmp_path).read_text()


def test_measured_rates_replace_the_assumptions_and_a_typo_is_refused(tmp_path, capsys, qwen3_4b_config):
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({
        "source": "smoke run",
        "estimate_assumptions": {"prefill_tokens_per_s": 30000, "startup_s": 45},
        "by_system": {"vllm": {"prefill_tokens_per_s": 20000}},
        "expected_saturation_per_s": {"recorded": [0.05, 0.06]},
    }))
    arguments = ["--config", str(REPO / "bench/configs/matrix.json"), "--dry-run", "--measured-rates", str(rates), "--out", "/nonexistent"]
    assert run_matrix.main(arguments) == 0
    printed = capsys.readouterr().out
    assert "token rates MEASURED, smoke run" in printed and "'prefill_tokens_per_s': 30000" in printed
    assert "100% load is 0.05 to 0.06 jobs/s" in printed
    rates.write_text(json.dumps({"estimate_assumptions": {"prefil_tokens_per_s": 1}}))
    with pytest.raises(Refused, match="unknown keys in estimate_assumptions"):
        run_matrix.main(arguments)
    with pytest.raises(Refused, match="only changes a --dry-run"):
        run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--measured-rates", str(rates)])


def test_the_dry_run_says_plainly_when_a_plan_is_over_budget(capsys, qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    workloads = run_matrix.load_workloads(config, SETTINGS, False, dry_run=True)
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in run_matrix.system_specs(config)}
    cells = run_matrix.plan_cells(config, workloads, "first-pass")
    config["plans"]["first-pass"]["budget_hours"] = 1000
    run_matrix.dry_run(config, SETTINGS, systems, workloads, cells, pool_tokens=87381, plan_name="first-pass")
    assert "OVER BUDGET" not in capsys.readouterr().out
    config["plans"]["first-pass"]["budget_hours"] = 4
    run_matrix.dry_run(config, SETTINGS, systems, workloads, cells, pool_tokens=87381, plan_name="first-pass")
    printed = capsys.readouterr().out
    assert re.search(r"OVER BUDGET AT THE SLOW END: [\d.]+ hours is [\d.]+ hours over", printed)
    assert re.search(r"OVER BUDGET EVEN AT THE LOW END: [\d.]+ hours is [\d.]+ hours over", printed)


def test_the_fairness_notes_cite_what_was_read_and_state_the_budgets():
    notes = (REPO / "bench/configs/baseline_flags.md").read_text()
    for cited in (
        "vLLM 0.33.0", "g4ba787ec9", "vllm/v1/core/block_pool.py", "BlockPool.free_blocks", "get_new_blocks", "_maybe_evict_cached_block",
        "vllm/v1/core/kv_cache_utils.py", "FreeKVCacheBlockQueue", "vllm/v1/core/single_type_kv_cache_manager.py", "published sdist",
        "layover-discard-nocache", "lower bound", "32 GiB", "--hicache-ratio", "`--kv-offloading-size`", "`--cpu-tier-gib`", "31.66",
    ):
        assert cited in notes, cited
    assert "sharing index" not in notes.lower()


def test_the_real_pipeline_survives_a_kill_and_moves_to_a_new_machine(tmp_path):
    """The real runner, server and stand-in baselines: SIGKILL everything, resume under a new instance id."""
    out = tmp_path / "out"
    config = pipeline_config(tmp_path)
    command = [sys.executable, str(REPO / "bench/run_matrix.py"), "--config", str(config), "--out", str(out), "--plan", "first-pass", "--pipeline-check", "--no-gpu"]
    run = subprocess.Popen(command, cwd=REPO, env={**os.environ, "LAYOVER_INSTANCE_ID": "machine-a"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    started = time.monotonic()
    try:
        while not list((out / "runs").rglob("run-0.json")) and time.monotonic() - started < 300:
            assert run.poll() is None, run.stdout.read()
            time.sleep(0.05)
        # The whole machine goes: the runner and every server under it, with no warning. Stopped
        # first, so it cannot start another server between the listing and the kill.
        os.kill(run.pid, signal.SIGSTOP)
        for pid in _descendants(run.pid):
            os.kill(pid, signal.SIGKILL)
        os.kill(run.pid, signal.SIGKILL)
    finally:
        run.wait(timeout=30)
    assert run.returncode == -signal.SIGKILL
    assert (out / "manifest.json").exists() and (out / "calibration" / "pipeline.json").exists()

    finished = {path: path.stat().st_mtime_ns for path in (out / "runs").rglob("run-*.json") if json.loads(path.read_text())["status"] == "ok"}
    assert finished
    assert not (out / "report.md").exists()
    calibration = (out / "calibration" / "pipeline.json").stat().st_mtime_ns

    # The same command on another machine, with results/ as they were.
    done = subprocess.run(command, cwd=REPO, env={**os.environ, "LAYOVER_INSTANCE_ID": "machine-b"}, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"{len(finished)} already complete" in done.stdout
    records = {path: json.loads(path.read_text()) for path in (out / "runs").rglob("run-*.json")}
    # Two sweep points, each with 2 x 2 runs at 100%, 2 at 50%, and the 1 extra.
    assert len(records) == 2 * (2 * 2 + 2 + 1)
    assert all(record["status"] == "ok" for record in records.values())
    assert {path: path.stat().st_mtime_ns for path in finished} == finished
    assert (out / "calibration" / "pipeline.json").stat().st_mtime_ns == calibration
    for path, record in records.items():
        assert record["instance"]["instance_id"] == ("machine-a" if path in finished else "machine-b")

    text = (out / "report.md").read_text()
    assert "## Missing runs\n\nNone." in text and "instances." in text and "machine-a" in text and "machine-b" in text
    manifest = json.loads((out / "manifest.json").read_text())
    assert [host["instance_id"] for host in manifest["hosts"]] == ["machine-a", "machine-b"] and manifest["plans"] == ["first-pass"]
    assert (out / "plots").exists()
    assert not [c for c in running_commands() if str(config.parent) in c]


@pytest.mark.gpu
def test_nvml_backend_reads_this_gpu():
    pytest.importorskip("pynvml")
    from bench.gpu_sampler import NvmlBackend

    backend = NvmlBackend()
    try:
        value = backend.read()
        assert value.memory_total_bytes > 0 and 0 <= value.memory_used_bytes <= value.memory_total_bytes
        assert backend.device_info()["name"]
        sampler = GpuSampler(backend, period_s=0.05)
        sampler.start()
        time.sleep(0.5)
        sampler.stop()
        assert sampler.summary()["samples"] >= 2
    finally:
        backend.close()


@pytest.mark.gpu
def test_every_baseline_starts_and_answers_at_matched_settings(tmp_path):
    """Preflight on the real systems: flags accepted, budgets held, token ids returned."""
    for venv in (".venvs/vllm", ".venvs/sglang"):
        if not (REPO / venv).exists():
            pytest.skip(f"{venv} is missing; run scripts/setup_baselines.sh")
    code = run_matrix.main(["--config", str(REPO / "bench/configs/matrix.json"), "--out", str(tmp_path), "--preflight-only", "--preflight-kinds", "vllm", "sglang"])
    assert code == 0


# ---- paired ablation -------------------------------------------------------------------------------


def seeded_run(mean: float, seed: int) -> dict:
    return {"status": "ok", "seed": seed, "result": fake_result(mean)}


def test_an_identical_system_reads_one_against_a_reference_with_more_runs():
    """The ablation has three seeds and the reference five. Only the three they share may be compared."""
    manifest = tier_manifest(systems=("layover-combined", "vllm"))
    workload = manifest["workloads"][0]
    # Seeds spread widely, and the unpaired medians (20 against 30) would read 0.67.
    means = [10.0, 20.0, 30.0, 400.0, 500.0]
    runs = {("w", "layover-combined", "load100", i): seeded_run(mean, 1000 + i) for i, mean in enumerate(means)}
    runs |= {("w", "vllm", "load100", i): {**seeded_run(means[i], 1000 + i), "result": {**fake_result(means[i]), "server_metrics": {"vllm:external_prefix_cache_hits_total": 5}}} for i in range(3)}
    table = "\n".join(report.ablation_table(manifest, runs, workload, "layover-combined"))
    assert "| vllm | 3/1 | 1000, 1001, 1002 | 1.00 | 1.00 | 1.00 | 1.00 |" in table
    assert "| layover-combined | 5/1 | 1000, 1001, 1002, 1003, 1004 | 1.00 |" in table
    assert "same seed" in table


def test_a_reference_with_fewer_runs_or_no_shared_seed_is_paired_or_says_so():
    manifest = tier_manifest(systems=("layover-combined", "vllm"))
    workload = manifest["workloads"][0]
    hits = {"vllm:external_prefix_cache_hits_total": 5}
    runs = {("w", "layover-combined", "load100", 1): seeded_run(10.0, 1001)}
    runs |= {("w", "vllm", "load100", i): {**seeded_run(mean, 1000 + i), "result": {**fake_result(mean), "server_metrics": hits}} for i, mean in enumerate([99.0, 30.0, 77.0])}
    assert "| vllm | 3/1 | 1001 | 3.00 |" in "\n".join(report.ablation_table(manifest, runs, workload, "layover-combined"))
    del runs[("w", "vllm", "load100", 1)]
    assert "| vllm | 2/1 | none shared | n/a |" in "\n".join(report.ablation_table(manifest, runs, workload, "layover-combined"))


# ---- a system that keeps failing --------------------------------------------------------------------


def failure_ctx(tmp_path, monkeypatch, runs=5, fail=lambda cell: cell.system == "vllm"):
    ctx, workloads = make_ctx(tmp_path, runs=runs)
    calls = []

    def measure(_, cell, *__):
        calls.append(cell.label)
        if fail(cell):
            raise ServerFailed("wrong token count")
        return fake_result(10.0)

    monkeypatch.setattr(run_matrix, "measure_run", measure)
    return ctx, workloads, calls


def test_three_failed_cells_in_a_row_stop_the_whole_matrix(tmp_path, monkeypatch):
    # From run 2 on every cell fails, whichever system it belongs to.
    ctx, workloads, calls = failure_ctx(tmp_path, monkeypatch, fail=lambda cell: cell.run >= 2)
    cells = run_matrix.build_cells(workloads, 5)
    order = [cell.label for cell in cells]
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 1
    first = next(i for i, cell in enumerate(cells) if cell.run >= 2)
    # Three failed cells at two attempts each, then nothing more.
    assert len(calls) == first + 3 * 2
    session = json.loads((tmp_path / "session.json").read_text())
    stopped = session["stopped_matrix"]
    assert "3 cells in a row failed" in stopped["reason"] and "wrong token count" in stopped["reason"]
    assert stopped["failed"] == order[first:first + 3]
    assert stopped["skipped"] == order[first + 3:]
    assert session["budget_skipped"] == []
    text = report.write_report(tmp_path).read_text()
    assert "**MATRIX STOPPED: 3 cells in a row failed" in text
    assert f"`{order[-1]}`: skipped because the matrix stopped" in text


def test_one_system_failing_between_successes_does_not_stop_the_matrix(tmp_path, monkeypatch):
    ctx, workloads, calls = failure_ctx(tmp_path, monkeypatch)
    cells = run_matrix.build_cells(workloads, 5)
    # The systems alternate, so vllm's failures never make three in a row.
    run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
    assert json.loads((tmp_path / "session.json").read_text())["stopped_matrix"] is None
    assert len(set(calls)) == len(cells)


def test_failures_spread_across_systems_count_as_one_streak(tmp_path, monkeypatch):
    ctx, workloads, calls = failure_ctx(tmp_path, monkeypatch, runs=3, fail=lambda cell: True)
    cells = run_matrix.build_cells(workloads, 3)
    run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
    assert len(set(calls)) == 3


def test_a_success_clears_the_failure_count_and_the_limit_comes_from_the_config(tmp_path, monkeypatch):
    # Failing every other cell is never three in a row.
    ctx, workloads, calls = failure_ctx(tmp_path, monkeypatch, runs=6, fail=lambda cell: cell.system == "vllm" and cell.run % 2 == 0)
    cells = run_matrix.build_cells(workloads, 6)
    run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
    assert json.loads((tmp_path / "session.json").read_text())["stopped_matrix"] is None
    ctx, workloads, calls = failure_ctx(tmp_path / "strict", monkeypatch, runs=3)
    ctx.limits = {**ctx.limits, "max_consecutive_failures": 1}
    run_matrix.run_all(ctx, resume_args(), workloads, run_matrix.build_cells(workloads, 3), None)
    assert sum("vllm" in call for call in calls) == 2
    assert all("vllm" not in call for call in calls[calls.index(next(c for c in calls if "vllm" in c)) + 2:])


def test_a_resume_gives_a_stopped_system_another_chance(tmp_path, monkeypatch):
    ctx, workloads, calls = failure_ctx(tmp_path, monkeypatch)
    cells = run_matrix.build_cells(workloads, 5)
    run_matrix.run_all(ctx, resume_args(), workloads, cells, None)
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: fake_result(10.0))
    assert run_matrix.run_all(ctx, resume_args(), workloads, cells, None) == 0
    assert json.loads((tmp_path / "session.json").read_text())["stopped_matrix"] is None
    assert "STOPPED" not in report.write_report(tmp_path).read_text()


# ---- one definition of KV pool used -----------------------------------------------------------------


class MetricsClient:
    def __init__(self, metrics):
        self.metrics = metrics

    def get(self, _):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.metrics)


def test_layover_counts_blocks_only_the_tree_holds_as_free_and_publishes_them_apart():
    system = published_system("layover-discard")
    used, tree_only = system.kv_fractions(MetricsClient({"blocks_total": 100, "blocks_free": 10, "radix_only_blocks": 70, "radix_blocks": 75}))
    # 90 blocks are not free, but 70 of them are as reusable as vLLM's free queue's: 20 are held.
    assert used == pytest.approx(0.2) and tree_only == pytest.approx(0.7)
    assert system.kv_used_fraction(MetricsClient({"blocks_total": 100, "blocks_free": 10, "radix_only_blocks": 70})) == pytest.approx(0.2)


def test_a_server_without_the_count_gives_no_figure_rather_than_another_definition():
    system = published_system("layover-discard")
    assert system.kv_fractions(MetricsClient({"blocks_total": 100, "blocks_free": 10, "radix_blocks": 75})) == (None, None)


def test_a_baseline_reports_no_tree_share_and_the_table_has_a_column_for_layover():
    assert CommandSystem("c", SETTINGS, TOKEN_BYTES, ["x"]).kv_fractions(None) == (None, None)
    manifest = tier_manifest(systems=("layover-combined", "vllm"))
    record = occupied_run(0.9)
    record["result"]["metrics"]["kv_pool_tree_only_frac_mean"] = 0.625
    runs = {("w", "layover-combined", "load100", 0): record}
    table = "\n".join(report.workload_table(manifest, runs, manifest["workloads"][0], []))
    assert "KV pool held only by a radix tree (fraction)" in table and "0.625" in table
    assert "a cached block nothing holds counts as free" in table


def test_a_low_peak_alone_never_says_the_pool_did_not_fill():
    """vLLM can evict at any peak, so its own figure says nothing; Layover's occupancy can."""
    manifest = tier_manifest()
    workload = manifest["workloads"][0]
    low = tier_run(20.0, 0.3, **{VLLM_READS: 0})
    assert report.tier_flag(manifest, {("w", "vllm", "load100", 0): low}, workload, "vllm") == "CPU tier unverified"
    full = {("w", "vllm", "load100", 0): low, ("w", "layover-combined", "load100", 0): occupied_run(1.0)}
    assert report.tier_flag(manifest, full, workload, "vllm") == "CPU tier unverified"
    # A record from before this figure existed is unknown, not calm.
    assert report.tier_flag(manifest, {("w", "vllm", "load100", 0): low, ("w", "layover-combined", "load100", 0): tier_run(10.0, 0.1)}, workload, "vllm") == "CPU tier unverified"


def test_a_layover_without_its_cache_cannot_vouch_for_the_pool():
    manifest = tier_manifest(systems=("layover-combined", "vllm"))
    manifest["config"]["systems"][0]["prefix_cache"] = False
    runs = {("w", "vllm", "load100", 0): tier_run(20.0, 0.3, **{VLLM_READS: 0}), ("w", "layover-combined", "load100", 0): occupied_run(0.2)}
    assert report.tier_flag(manifest, runs, manifest["workloads"][0], "vllm") == "CPU tier unverified"


def test_an_eviction_counter_decides_for_the_system_that_has_one():
    manifest = tier_manifest()
    workload = manifest["workloads"][0]
    quiet = tier_run(30.0, 0.9, **{SGLANG_READS: 0, "sglang__evicted_tokens_total": 0})
    assert report.tier_flag(manifest, {("w", "sglang", "load100", 0): quiet}, workload, "sglang") == "CPU tier unused, nothing evicted"
    busy = tier_run(30.0, 0.1, **{SGLANG_READS: 0, "sglang__evicted_tokens_total": 512})
    calm_layover = occupied_run(0.2)
    assert report.tier_flag(manifest, {("w", "sglang", "load100", 0): busy, ("w", "layover-combined", "load100", 0): calm_layover}, workload, "sglang") == "CPU tier unverified"


# ---- calibration provenance, retries, and failure ----------------------------------------------------


def calibrated(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay([], [12.0, 16.0, 40.0]))
    return ctx, workload, run_matrix.calibrate(ctx, "g", workload)


def test_a_calibration_names_its_config_trace_and_hardware_and_reaches_each_run(tmp_path, monkeypatch):
    ctx, workload, record = calibrated(tmp_path, monkeypatch)
    assert record["identity"] == {"config_hash": run_matrix.stable_hash(ctx.config), "trace_hash": "h", "hardware": None}
    assert run_matrix.cell_rate(record, "load100")["identity"] == record["identity"]


@pytest.mark.parametrize("where", ["finished", "partial"])
@pytest.mark.parametrize("key, value, match", [("config_hash", "other", "config_hash"), ("trace_hash", "other", "trace_hash"), ("hardware", {"name": "A100"}, "A100")])
def test_a_calibration_from_other_inputs_is_refused(tmp_path, monkeypatch, where, key, value, match):
    ctx, workload = calibration_ctx(tmp_path, None)
    monkeypatch.setattr(run_matrix, "serving", no_server)
    calls = []
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(calls, [12.0, 16.0, 40.0]))
    identity = {**run_matrix.calibration_identity(ctx, workload), key: value}
    if where == "finished":
        run_matrix.write_json(tmp_path / "calibration" / "g.json", {"saturation_rate_per_s": 9.0, "identity": identity})
    else:
        run_matrix.write_json(tmp_path / "calibration" / "partial" / "g.json", {"identity": identity, "unloaded_jct_s": {}, "points": []})
    with pytest.raises(Refused, match=match):
        run_matrix.calibrate(ctx, "g", workload)
    assert not calls


def test_a_calibration_naming_nothing_is_refused(tmp_path):
    ctx, workload = calibration_ctx(tmp_path, None)
    run_matrix.write_json(tmp_path / "calibration" / "g.json", {"saturation_rate_per_s": 9.0})
    with pytest.raises(Refused, match="config_hash"):
        run_matrix.calibrate(ctx, "g", workload)


def test_accepted_mixed_hardware_keeps_the_calibration_and_the_report_flags_it(tmp_path, monkeypatch, capsys):
    ctx, workload, record = calibrated(tmp_path, monkeypatch)
    moved = {**record, "identity": {**record["identity"], "hardware": {"name": "A100", "driver": "1"}}}
    run_matrix.write_json(tmp_path / "calibration" / "g.json", moved)
    with pytest.raises(Refused, match="A100"):
        run_matrix.calibrate(ctx, "g", workload)
    ctx.accept_mixed_hardware = True
    assert run_matrix.calibrate(ctx, "g", workload)["identity"]["hardware"]["name"] == "A100"
    assert "MIXED HARDWARE" in capsys.readouterr().out
    run = {"status": "ok", "arrival": run_matrix.cell_rate(moved, "load100"), "instance": {"gpu_name": "L4", "gpu_driver": "550", "instance_id": "i", "commit": "c"}}
    assert "calibrated on A100, driver 1" in "\n".join(report.banners({}, {("w", "s", "load100", 0): run}))
    same = {**run, "instance": {"gpu_name": "A100", "gpu_driver": "1", "instance_id": "i", "commit": "c"}}
    assert report.banners({}, {("w", "s", "load100", 0): same}) == []


def flaky(inner, failures):
    """A replay whose first calls of a kind fail, then behaves."""
    left = list(failures)

    def run(ctx, system, server, jobs, warmup, seed, timeout):
        kind = "alone" if len(jobs) == 1 else "loaded"
        if kind in left:
            left.remove(kind)
            raise ServerFailed("rung died")
        return inner(ctx, system, server, jobs, warmup, seed, timeout)

    return run


def test_each_step_of_the_ladder_is_retried_and_the_failure_is_kept(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", flaky(fake_replay(calls, [12.0, 16.0]), ["alone", "loaded"]))
    record = run_matrix.calibrate(ctx, "g", workload)
    assert record["bracketed"] and calls.count(("loaded", 6, 2)) == 2
    assert [failure["step"] for failure in record["failures"]] == ["unloaded jobs", "rate 0.2"]
    assert all("rung died" in failure["error"] for failure in record["failures"])


def test_a_retry_serves_from_a_new_log_and_keeps_jobs_already_measured(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    logs = []

    @contextmanager
    def serving(_, __, log):
        logs.append(log.name)
        yield None, {}

    calls = []
    inner = fake_replay(calls, [12.0, 16.0])
    alone = []

    def replay(ctx, system, server, jobs, warmup, seed, timeout):
        if len(jobs) == 1:
            alone.append(jobs[0].job_id)
            if len(alone) == 2:
                raise ServerFailed("second job died")
        return inner(ctx, system, server, jobs, warmup, seed, timeout)

    monkeypatch.setattr(run_matrix, "serving", serving)
    monkeypatch.setattr(run_matrix, "_replay_jobs", replay)
    run_matrix.calibrate(ctx, "g", workload)
    # The first job is not repeated; the second gets its retry on a new server with its own log.
    assert alone == ["j2", "j4", "j4"]
    assert logs[:2] == ["unloaded.attempt-1.log", "unloaded.attempt-2.log"] and logs[2] == "rate-0.2.attempt-1.log"


def test_an_incomparable_rung_cannot_set_the_rate(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    inner = fake_replay(calls, [12.0, 16.0])

    def short(*arguments):
        result = inner(*arguments)
        result.output_count_mismatches = 3
        return result

    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", short)
    with pytest.raises(run_matrix.CalibrationFailed, match="not comparable"):
        run_matrix.calibrate(ctx, "g", workload)
    assert not (tmp_path / "calibration" / "g.json").exists()
    ctx.pipeline_check = True
    assert run_matrix.calibrate(ctx, "g", workload)["bracketed"]


def test_a_failed_calibration_is_recorded_and_reported_and_stands_in_nothing(tmp_path, monkeypatch):
    ctx, workload = calibration_ctx(tmp_path, None)
    calls = []
    monkeypatch.setattr(run_matrix, "serving", no_server)
    monkeypatch.setattr(run_matrix, "_replay_jobs", fake_replay(calls, [12.0], fail_at=1))
    manifest = run_matrix.manifest_for(ctx.config, [workload], [Cell("w", "vllm", 0)], False)
    run_matrix.write_json(tmp_path / "manifest.json", manifest)
    workload.source = "recorded:x"
    assert run_matrix.run_all(ctx, resume_args(), [workload], [Cell("w", "vllm", 0)], None) == 1
    session = json.loads((tmp_path / "session.json").read_text())
    assert "rate 0.4 failed after 2 attempts" in session["calibration_failed"]["reason"]
    assert session["calibration_failed"]["skipped"] == ["w/vllm/load100/0"]
    text = report.write_report(tmp_path).read_text()
    assert "**CALIBRATION FAILED: rate 0.4 failed after 2 attempts" in text and "1 runs were not run" in text
    assert "`w/vllm/load100/0`: skipped because calibration failed" in text
    assert "Calibration `g` is unfinished: 1 rungs and 2 unloaded jobs are saved" in text and "rate 0.4, attempt 2: ServerFailed: rung died" in text


def main_arguments(tmp_path):
    return ["--config", str(pipeline_config(tmp_path)), "--out", str(tmp_path / "out"), "--pipeline-check", "--no-gpu", "--skip-preflight"]


@pytest.mark.parametrize("error", [Refused("SATURATION NOT FOUND for g"), ServerFailed("preflight died")], ids=["refused", "crash"])
def test_a_session_that_ends_badly_still_writes_its_report(tmp_path, monkeypatch, error):
    def stops(*_):
        raise error

    monkeypatch.setattr(run_matrix, "run_all", stops)
    with pytest.raises(type(error)):
        run_matrix.main(main_arguments(tmp_path))
    text = (tmp_path / "out" / "report.md").read_text()
    assert f"**MATRIX STOPPED: {type(error).__name__}: {error}" in text
    assert "## Missing runs" in text and json.loads((tmp_path / "out" / "budget.json").read_text())["sessions"]


def test_a_stale_session_does_not_outlive_the_next_end(tmp_path):
    out = tmp_path / "out"
    run_matrix.write_json(out / "session.json", {"budget_skipped": ["w/vllm/load100/0"]})
    run_matrix.write_session(out, stopped="interrupted")
    assert json.loads((out / "session.json").read_text())["budget_skipped"] == []


def test_a_torn_side_file_does_not_stop_the_report(tmp_path):
    (tmp_path / "torn.json").write_text("{")
    assert report.read_json(tmp_path / "torn.json") == {} and report.read_json(tmp_path / "absent.json") == {}


# ---- the budget outlives a machine --------------------------------------------------------------------


def test_the_budget_counts_earlier_sessions_and_a_new_machine_gets_no_fresh_one(tmp_path):
    first = run_matrix.ElapsedLedger(tmp_path, "machine-a", 1.0)
    first.entry["elapsed_s"] = 0.0
    first.earlier_s = 3000.0
    first.finish()
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["elapsed_s"] >= 3000.0 and saved["max_hours"] == 1.0
    second = run_matrix.ElapsedLedger(tmp_path, "machine-b", 1.0)
    assert second.earlier_s == pytest.approx(saved["elapsed_s"])
    # 600 s of an hour are left, on this machine's own clock.
    assert second.deadline() - time.monotonic() == pytest.approx(600.0, abs=5.0)
    second.finish()
    assert [entry["instance_id"] for entry in json.loads((tmp_path / "budget.json").read_text())["sessions"]] == ["machine-a", "machine-b"]
    assert run_matrix.ElapsedLedger(tmp_path, "c", None).deadline() is None


def test_a_killed_machine_loses_at_most_a_heartbeat_of_budget(tmp_path):
    ledger = run_matrix.ElapsedLedger(tmp_path, "machine-a", 1.0, period_s=0.01)
    ledger.start()
    time.sleep(0.2)
    # Never finished, as under a kill.
    saved = json.loads((tmp_path / "budget.json").read_text())
    ledger.halted.set()
    ledger.join()
    assert saved["elapsed_s"] > 0.05 and saved["sessions"][0]["instance_id"] == "machine-a"


@pytest.mark.parametrize("damage", ["missing", "torn"])
def test_a_lost_ledger_cannot_hand_out_a_fresh_budget(tmp_path, damage):
    run_matrix.write_json(tmp_path / "runs" / "w" / "vllm" / "load100" / "run-0.json", {"attempts": [{"elapsed_s": 1800.0}, {"elapsed_s": 1200.0}]})
    if damage == "torn":
        (tmp_path / "budget.json").write_text('{"elapsed_s"')
    assert run_matrix.ElapsedLedger(tmp_path, "m", 1.0).earlier_s == 3000.0


def test_a_spent_budget_starts_nothing_and_the_report_says_what_was_not_run(tmp_path, monkeypatch):
    ctx, workloads = make_ctx(tmp_path, runs=2)
    calls = []
    monkeypatch.setattr(run_matrix, "measure_run", lambda *_: calls.append("run"))
    monkeypatch.setattr(run_matrix, "preflight", lambda *_: calls.append("preflight"))
    args = argparse.Namespace(skip_preflight=False, preflight_only=False)
    run_matrix.write_json(tmp_path / "budget.json", {"elapsed_s": 7200.0, "max_hours": 2.0, "sessions": [{"instance_id": "a"}]})
    ledger = run_matrix.ElapsedLedger(tmp_path, "b", 2.0)
    ctx.deadline = ledger.deadline()
    assert run_matrix.run_all(ctx, args, workloads, run_matrix.build_cells(workloads, 2), ctx.deadline) == 1
    assert calls == []
    ledger.finish()
    text = report.write_report(tmp_path).read_text()
    assert "TIME BUDGET REACHED: 4 runs were not run: layover-combined load100 x 2, vllm load100 x 2" in text
    assert "`w/vllm/load100/1`: skipped by the time budget" in text
    assert re.search(r"Matrix time so far: 2\.00 hours over 2 sessions on 2 instances, against --max-hours 2\.", text)


def test_main_takes_the_budget_from_the_results_directory(tmp_path):
    """Through the real command line: a replacement machine with a spent budget runs nothing."""
    out = tmp_path / "out"
    run_matrix.write_json(out / "budget.json", {"elapsed_s": 3 * 3600.0, "sessions": [{"instance_id": "lost"}]})
    assert run_matrix.main([*main_arguments(tmp_path), "--max-hours", "3"]) == 1
    assert not (out / "runs").exists()
    session = json.loads((out / "session.json").read_text())
    assert session["budget_skipped"] and "TIME BUDGET REACHED" in (out / "report.md").read_text()
    saved = json.loads((out / "budget.json").read_text())
    assert saved["elapsed_s"] >= 3 * 3600.0 and len(saved["sessions"]) == 2


def test_a_higher_limit_on_a_resume_gives_room_for_what_is_left(tmp_path):
    run_matrix.write_json(tmp_path / "budget.json", {"elapsed_s": 3600.0, "sessions": []})
    assert run_matrix.ElapsedLedger(tmp_path, "m", 4.0).deadline() - time.monotonic() == pytest.approx(3 * 3600.0, abs=5.0)


# ---- plans --------------------------------------------------------------------------------------------


def test_under_20h_is_load100_first_then_load125_then_the_ablations_then_the_low_load_check(qwen3_4b_config):
    config, cells = published_plan(qwen3_4b_config, "under-20h")
    assert len(cells) == 15 + 9 + 15 + 2 and len(set(cells)) == len(cells)
    assert {(c.system, c.run) for c in cells[:15]} == {(s, r) for s in HEADLINE for r in range(5)} and {c.load for c in cells[:15]} == {"load100"}
    assert [c.run for c in cells[:15]] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert {(c.system, c.run) for c in cells[15:24]} == {(s, r) for s in HEADLINE for r in range(3)} and {c.load for c in cells[15:24]} == {"load125"}
    assert {(c.system, c.load, c.run) for c in cells[24:39]} == {(s, "load100", r) for s in ABLATIONS for r in range(3)}
    # Last, so a budget stop drops them first: one 50% run each, a sanity check only.
    assert [(c.system, c.load, c.run) for c in cells[39:]] == [("layover-combined", "load50", 0), ("vllm", "load50", 0)]


def test_three_loads_lean_is_three_runs_at_each_load_then_two_of_each_ablation(qwen3_4b_config):
    _, cells = published_plan(qwen3_4b_config, "three-loads-lean")
    assert len(cells) == 9 + 18 + 10
    assert {c.load for c in cells[:9]} == {"load100"} and {c.system for c in cells[:9]} == set(HEADLINE)
    assert {(c.system, c.load, c.run) for c in cells[9:27]} == {(s, l, r) for s in HEADLINE for l in ("load50", "load125") for r in range(3)}
    assert {(c.system, c.load, c.run) for c in cells[27:]} == {(s, "load100", r) for s in ABLATIONS for r in range(2)}


def test_every_new_plan_can_compare_each_ablation_on_seeds_the_reference_has(qwen3_4b_config):
    for name in ("under-20h", "three-loads-lean", "runpod-l4", "final"):
        _, cells = published_plan(qwen3_4b_config, name)
        reference = {c.run for c in cells if c.system == "layover-combined" and c.load == "load100"}
        for system in ABLATIONS:
            assert {c.run for c in cells if c.system == system} <= reference


def test_runs_per_load_must_name_the_groups_loads(qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    config["plans"]["bad"] = {"description": "x", "groups": [{"role": "headline", "systems": HEADLINE, "loads": [1.0, 1.25], "runs": {"load100": 5}}]}
    with pytest.raises(Refused, match="runs per load"):
        run_matrix.plan_cells(config, [Workload("w", "headline", HEADLINE, 0)], "bad")


def test_the_dry_run_gives_the_breakdown_of_every_workload(capsys):
    """Each per-system line used to overwrite the set of workloads the loop goes on to read."""
    config = matrix_config()
    workloads = [Workload(name, "headline", ["layover-combined", "vllm"], 0, [job()], "recorded:x", "h", arrival_rate_per_s=1.0) for name in ("first", "second")]
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in config["systems"]}
    run_matrix.dry_run(config, SETTINGS, systems, workloads, run_matrix.build_cells(workloads, 1))
    printed = capsys.readouterr().out
    assert "  first: source recorded:x" in printed and "  second: source recorded:x" in printed
    assert printed.count("      vllm: 1 runs x") == 2


def test_the_dry_run_prices_every_plan_in_matrix_hours(capsys, qwen3_4b_config):
    config = run_matrix.load_config(REPO / "bench/configs/matrix.json")
    workloads = run_matrix.load_workloads(config, SETTINGS, False, dry_run=True)
    systems = {spec["name"]: build_system(spec, SETTINGS, TOKEN_BYTES) for spec in run_matrix.system_specs(config)}
    run_matrix.dry_run(config, SETTINGS, systems, workloads, run_matrix.plan_cells(config, workloads, "first-pass"), pool_tokens=87381, plan_name="first-pass")
    printed = capsys.readouterr().out
    for name, runs in (("first-pass", 60), ("under-20h", 41), ("three-loads-lean", 37), ("runpod-l4", 28), ("final", 8)):
        assert re.search(rf"  {name}: {runs} runs, [\d.]+ to [\d.]+ matrix hours\.", printed)



def test_baselines_start_with_their_environment_first_on_path(monkeypatch):
    """vLLM builds kernels at start-up with ninja from its own environment."""
    monkeypatch.setenv("PATH", "/usr/bin")
    for spec in run_matrix.system_specs(run_matrix.load_config(REPO / "bench/configs/matrix.json")):
        system = build_system(spec, SETTINGS, TOKEN_BYTES)
        if system.kind in ("vllm", "sglang"):
            assert system.environment()["PATH"] == f"{Path(spec['venv']) / 'bin'}:/usr/bin"
        elif system.kind == "layover":
            assert "PATH" not in system.environment()

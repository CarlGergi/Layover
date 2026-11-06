import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layover.agent.cost_model import CostModel
from layover.config import ModelConfig
from layover.kv.transfer import NativeBackend, PlainCopyBackend

spec = importlib.util.spec_from_file_location(
    "kv_crossover", Path(__file__).resolve().parents[1] / "bench" / "kv_crossover.py"
)
kv = importlib.util.module_from_spec(spec)
import sys  # noqa: E402

sys.modules["kv_crossover"] = kv
spec.loader.exec_module(kv)

# config.json of Qwen/Qwen3-4B as published on the Hugging Face hub.
QWEN3_4B = {
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "eos_token_id": 151645,
    "bos_token_id": 151643,
}


def row(tokens, swap, recompute):
    return kv.Row(tokens, tokens, swap / 2, swap / 2, swap, swap, swap, recompute, recompute, recompute, "x")


def test_qwen3_4b_kv_bytes_use_config_head_dim():
    cfg = ModelConfig.from_hf_dict(QWEN3_4B)
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.kv_bytes_per_token("bf16") == 36 * 2 * 8 * 128 * 2 == 147_456
    assert cfg.kv_bytes_per_token("fp8") == 73_728
    assert cfg.kv_groups == 4


def test_config_rejects_unsupported_models():
    with pytest.raises(ValueError):
        ModelConfig.from_hf_dict({**QWEN3_4B, "model_type": "llama"})
    with pytest.raises(ValueError):
        ModelConfig.from_hf_dict({**QWEN3_4B, "use_sliding_window": True})


def test_prefill_kv_size_matches_config():
    cfg, model = kv.load_hf_model("micro", torch.device("cpu"), torch.bfloat16, 0)
    tensors = kv.hf_prefill(model, kv.make_ids(cfg, 40, torch.device("cpu"), 0), chunk=16)
    assert len(tensors) == 2 * cfg.num_hidden_layers
    assert sum(t.numel() for t in tensors) * 2 == 40 * cfg.kv_bytes_per_token("bf16")


def test_chunking_does_not_change_kv():
    cfg, model = kv.load_hf_model("micro", torch.device("cpu"), torch.float32, 0)
    ids = kv.make_ids(cfg, 48, torch.device("cpu"), 1)
    whole = kv.hf_prefill(model, ids, chunk=48)
    chunked = kv.hf_prefill(model, ids, chunk=16)
    for a, b in zip(whole, chunked):
        assert torch.allclose(a, b, atol=1e-4)


def test_engine_and_hf_backends_produce_the_same_kv():
    cfg, hf = kv.load_hf_model("micro", torch.device("cpu"), torch.float32, 0)
    _, ours = kv.load_layover_model("micro", torch.device("cpu"), torch.float32, 0)
    ours.load_state_dict(hf.state_dict())
    ids = kv.make_ids(cfg, 48, torch.device("cpu"), 2)
    pool = kv.layover_prefill(ours, ids, 16, torch.float32)[0]
    assert pool.numel() * 2 == 48 * cfg.kv_bytes_per_token("bf16")
    reference = kv.hf_prefill(hf, ids, 48)
    # The pool is [block, kind, layer, head, slot, dim]; blocks fill from block 0.
    rebuilt = pool.permute(2, 1, 3, 0, 4, 5).flatten(3, 4)
    for layer in range(cfg.num_hidden_layers):
        assert torch.allclose(rebuilt[layer, 0], reference[2 * layer][0], atol=1e-4)
        assert torch.allclose(rebuilt[layer, 1], reference[2 * layer + 1][0], atol=1e-4)


def test_layover_backend_rejects_partial_blocks():
    cfg, model = kv.load_layover_model("micro", torch.device("cpu"), torch.float32, 0)
    with pytest.raises(ValueError):
        kv.layover_prefill(model, kv.make_ids(cfg, 40, torch.device("cpu"), 0), 16, torch.float32)


def test_checksum_detects_a_flipped_value():
    a = [torch.ones(4, 4, dtype=torch.bfloat16)]
    b = [a[0].clone()]
    assert kv.checksum(a) == kv.checksum(b)
    b[0][1, 1] = 2.0
    assert kv.checksum(a) != kv.checksum(b)


def test_fit_recovers_polynomial_and_stays_nonnegative():
    x = np.array([512.0, 1024, 2048, 4096, 8192])
    coef = kv.fit_nonnegative(x, 0.5 + 2e-3 * x + 1e-7 * x**2, 2)
    assert coef == pytest.approx([0.5, 2e-3, 1e-7], rel=1e-6)
    # A falling series would fit a negative slope; the term must be dropped.
    coef = kv.fit_nonnegative(x, 10 - 1e-3 * x, 1)
    assert coef[1] == 0.0 and coef[0] > 0


def test_crossover_outcomes():
    assert kv.find_crossover([row(100, 1, 2), row(200, 2, 4)])["outcome"] == "swap_cheaper_at_every_length"
    assert kv.find_crossover([row(100, 3, 2), row(200, 5, 4)])["outcome"] == "recompute_cheaper_at_every_length"
    found = kv.find_crossover([row(100, 2, 1), row(400, 2, 3)])
    assert found["outcome"] == "swap_cheaper_above"
    assert 100 < found["crossover_tokens"] < 400


@pytest.mark.parametrize("backend", ["layover", "hf"])
def test_end_to_end_writes_labelled_outputs(tmp_path, backend):
    meta = kv.main(
        ["--tokens", "32", "64", "128", "--repeats", "2", "--chunk", "32", "--backend", backend, "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is False and meta["backend"] == backend
    saved = json.loads((tmp_path / "cost_model.json").read_text())
    assert saved["kv_bytes_per_token"] == ModelConfig.micro().kv_bytes_per_token("bf16")
    with (tmp_path / "kv_crossover.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["tokens"]) for r in rows] == [32, 64, 128]
    assert all(float(r["swap_s"]) > 0 and float(r["recompute_s"]) > 0 for r in rows)
    assert (tmp_path / "kv_crossover.png").stat().st_size > 0


@pytest.mark.parametrize("transfer", ["plain", "native"])
def test_cost_model_records_the_path_it_measured_and_loads_for_it(tmp_path, transfer):
    meta = kv.main(["--tokens", "32", "64", "--repeats", "2", "--chunk", "32", "--transfer", transfer, "--out", str(tmp_path)])
    assert meta["transfer"] == transfer and "fp8_swap_out_s_coef" not in meta
    path = tmp_path / "cost_model.json"
    assert CostModel.from_json(path, transfer=transfer).transfer == transfer
    other = {"plain": "native", "native": "plain"}[transfer]
    with pytest.raises(ValueError, match="measured through"):
        CostModel.from_json(path, transfer=other)


@pytest.mark.parametrize("transfer", ["plain", "native"])
def test_fp8_offload_adds_its_own_times_next_to_bf16(tmp_path, transfer):
    meta = kv.main(
        ["--tokens", "32", "64", "128", "--repeats", "2", "--chunk", "32", "--transfer", transfer, "--offload-dtype", "fp8", "--out", str(tmp_path)]
    )
    assert meta["offload_dtype"] == "fp8"
    model = CostModel.from_json(tmp_path / "cost_model.json")
    assert model.for_offload(True).swap_out_s_coef == meta["fp8_swap_out_s_coef"]
    assert model.for_offload(False).swap_out_s_coef == meta["swap_out_s_coef"]
    with (tmp_path / "kv_crossover.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(float(r["bf16_swap_out_s"]) > 0 and float(r["swap_out_s"]) > 0 for r in rows)


def test_a_corrupting_transfer_stops_the_benchmark():
    class Corrupting(PlainCopyBackend):
        def start_out(self, cache, blocks, fp8=False):
            ticket = super().start_out(cache, blocks, fp8)
            ticket.payload.memory[0] ^= 1
            return ticket

    cfg, model = kv.load_layover_model("micro", torch.device("cpu"), torch.bfloat16, 0)
    cache = kv.layover_prefill_cache(model, kv.make_ids(cfg, 32, torch.device("cpu"), 0), 32, torch.bfloat16)
    with pytest.raises(RuntimeError, match="bytes changed"):
        kv.measure_transfer(Corrupting(), cache, False, torch.device("cpu"), 0)


def test_hf_backend_refuses_paths_it_cannot_measure(tmp_path):
    with pytest.raises(SystemExit):
        kv.main(["--backend", "hf", "--transfer", "native", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        kv.main(["--dtype", "fp16", "--offload-dtype", "fp8", "--out", str(tmp_path)])


def test_rejects_cuda_without_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(SystemExit):
        kv.main(["--device", "cuda", "--out", str(tmp_path)])


def test_a_late_crash_keeps_what_was_measured_and_no_cost_model_comes_of_it(tmp_path, monkeypatch):
    real, calls = kv.measure, []

    def crash_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("the host ran out at the second length")
        return real(*args, **kwargs)

    monkeypatch.setattr(kv, "measure", crash_on_second)
    with pytest.raises(MemoryError):
        kv.main(["--tokens", "32", "64", "--repeats", "1", "--chunk", "32", "--out", str(tmp_path)])
    with (tmp_path / "kv_crossover.csv").open() as f:
        assert [row["tokens"] for row in csv.DictReader(f)] == ["32"]
    partial = json.loads((tmp_path / "cost_model.partial.json").read_text())
    assert partial["complete"] is False
    # What loads as a cost model must come from a whole sweep.
    assert not (tmp_path / "cost_model.json").exists()


def test_a_finished_sweep_leaves_only_the_complete_cost_model(tmp_path):
    meta = kv.main(["--tokens", "32", "64", "--repeats", "1", "--chunk", "32", "--out", str(tmp_path)])
    assert meta["complete"] is True
    assert not (tmp_path / "cost_model.partial.json").exists()
    assert json.loads((tmp_path / "cost_model.json").read_text())["complete"] is True


def test_recompute_clock_does_not_include_allocating_the_cache(monkeypatch):
    """A server's pool exists before it recomputes, so that cost is never paid while serving."""
    cfg, model = kv.load_layover_model("micro", torch.device("cpu"), torch.bfloat16, 0)
    args = kv.parse_args(["--repeats", "1", "--warmup", "0", "--chunk", "32"])
    real_cache, real_timed = kv.new_cache, kv.timed
    clock_running, allocated_on_clock = [False], []

    def watched_cache(*a, **k):
        allocated_on_clock.append(clock_running[0])
        return real_cache(*a, **k)

    def watched_timed(device, fn):
        clock_running[0] = True
        try:
            return real_timed(device, fn)
        finally:
            clock_running[0] = False

    monkeypatch.setattr(kv, "new_cache", watched_cache)
    monkeypatch.setattr(kv, "timed", watched_timed)
    kv.measure(model, cfg, 32, args, torch.device("cpu"), PlainCopyBackend())
    assert allocated_on_clock == [False]


def test_kept_repeats_reuse_a_pinned_buffer_at_every_length_of_a_sweep(monkeypatch):
    """Earlier lengths' buffers must not crowd out the next, or its every repeat pins on the clock.

    At these sizes the two buffers fit a 4 MiB cache only one at a time, as 16,384 and 32,768
    tokens of BF16 do not both fit 8 GiB once the smaller classes have filled it.
    """
    cfg = ModelConfig.micro()
    model = type("NoModel", (), {"cfg": cfg})()
    monkeypatch.setattr(kv, "prefill_into", lambda *a: None)
    args = kv.parse_args(["--repeats", "2", "--warmup", "1"])
    backend = NativeBackend("cpu", staging_bytes=1 << 20, ids_capacity=4096, host_cache_bytes=4 << 20)
    try:
        rows = [kv.measure(model, cfg, n, args, torch.device("cpu"), backend) for n in (4096, 8192)]
    finally:
        backend.close()
    assert [r.cold_pins for r in rows] == [0, 0]


def test_a_pin_inside_a_kept_repeat_is_reported(tmp_path, capsys):
    meta = kv.main(
        ["--tokens", "32", "--repeats", "1", "--warmup", "0", "--chunk", "32", "--transfer", "native", "--out", str(tmp_path)]
    )
    assert meta["cold_pins_in_kept_repeats"] >= 1 and meta["host_pins"] >= 1 and meta["host_pin_s"] > 0
    assert "host pins inside kept repeats" in capsys.readouterr().out


def test_fp8_verification_widens_a_slice_at_a_time(monkeypatch):
    from layover.kernels import fp8

    cfg, model = kv.load_layover_model("micro", torch.device("cpu"), torch.bfloat16, 0)
    cache = kv.layover_prefill_cache(model, kv.make_ids(cfg, 640, torch.device("cpu"), 0), 640, torch.bfloat16)
    seen, real = [], fp8.round_trip_bound
    monkeypatch.setattr(fp8, "round_trip_bound", lambda bits: (seen.append(bits.numel()), real(bits))[1])
    kv.measure_transfer(PlainCopyBackend(), cache, True, torch.device("cpu"), 0)
    assert sum(seen) == cache.pool.numel()
    assert max(seen) * 2 <= cache.pool.numel()


@pytest.mark.gpu
def test_gpu_swap_and_recompute_measured(tmp_path):
    meta = kv.main(
        ["--device", "cuda", "--tokens", "512", "1024", "--repeats", "2", "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is True


@pytest.mark.gpu
def test_gpu_native_fp8_path_is_measured(tmp_path):
    meta = kv.main(
        ["--device", "cuda", "--tokens", "512", "1024", "--repeats", "2", "--transfer", "native", "--offload-dtype", "fp8", "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is True and meta["transfer"] == "native"
    assert CostModel.from_json(tmp_path / "cost_model.json", require_gpu=True, transfer="native").fp8_swap_out_s_coef

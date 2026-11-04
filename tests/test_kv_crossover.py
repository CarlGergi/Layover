import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layover.config import ModelConfig

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


def test_end_to_end_writes_labelled_outputs(tmp_path):
    meta = kv.main(
        ["--tokens", "32", "64", "128", "--repeats", "2", "--chunk", "32", "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is False
    saved = json.loads((tmp_path / "cost_model.json").read_text())
    assert saved["kv_bytes_per_token"] == ModelConfig.micro().kv_bytes_per_token("bf16")
    with (tmp_path / "kv_crossover.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["tokens"]) for r in rows] == [32, 64, 128]
    assert all(float(r["swap_s"]) > 0 and float(r["recompute_s"]) > 0 for r in rows)
    assert (tmp_path / "kv_crossover.png").stat().st_size > 0


def test_rejects_cuda_without_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(SystemExit):
        kv.main(["--device", "cuda", "--out", str(tmp_path)])


@pytest.mark.gpu
def test_gpu_swap_and_recompute_measured(tmp_path):
    meta = kv.main(
        ["--device", "cuda", "--tokens", "512", "1024", "--repeats", "2", "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is True

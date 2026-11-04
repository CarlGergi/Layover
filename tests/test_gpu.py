"""Paths that only mean something on a CUDA device. Skipped without --gpu."""

import zlib

import pytest
import torch

from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request
from layover.kv import BlockTable, PagedKVCache
from layover.kv.tier import CpuTier
from layover.weights import load_model, resolve_checkpoint

pytestmark = pytest.mark.gpu
CFG = ModelConfig.micro()


def test_cpu_tier_round_trip_is_byte_exact_on_cuda():
    cache = PagedKVCache(CFG, 16, "cuda", torch.bfloat16)
    table = BlockTable()
    cache.prepare_write(table, 0, 64)
    heads, dim = CFG.num_key_value_heads, CFG.head_dim
    for layer in range(CFG.num_hidden_layers):
        k = torch.randn(heads, 64, dim, device="cuda")
        cache.write_range(table, layer, 0, k, -k)

    def crc(blocks):
        ids = torch.tensor(blocks, device="cuda")
        return zlib.crc32(cache.pool[ids].cpu().view(torch.uint16).numpy().tobytes())

    before = crc(table.blocks)
    tier = CpuTier(16)
    handle = tier.offload(cache, table)
    restored = tier.restore(cache, handle)
    assert crc(restored.blocks) == before


def test_engine_generates_on_cuda_with_real_weights(checkpoint):
    model = load_model(str(resolve_checkpoint(checkpoint)), device="cuda", dtype=torch.bfloat16)
    engine = Engine(model, PagedKVCache(model.cfg, 64, "cuda", torch.bfloat16), EngineConfig(prefill_chunk=64))
    requests = [Request(str(i), list(range(1, 40 + i)), max_new_tokens=8) for i in range(4)]
    for request in requests:
        engine.add_request(request)
    engine.run_until_idle()
    assert all(len(r.generated) == 8 for r in requests)
    assert engine.cache.allocator.num_used == 0

"""Paths that only mean something on a CUDA device. Skipped without --gpu."""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from layover.config import ModelConfig
from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig, AgentScheduler, AgentTurn
from layover.engine import Engine, EngineConfig, Request
from layover.kv import BlockTable, PagedKVCache
from layover.kv.tier import CpuTier
from layover.kv.transfer import NativeBackend
from layover.model import Qwen3
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


def test_long_context_prefill_takes_a_fused_attention_kernel_on_cuda(monkeypatch):
    """A 2048-token chunk at a 32,768-token context, the matrix's worst, at Qwen3-4B's shape.

    Torch's math path would build a 32 x 2048 x 32768 FP32 matrix of 8.6 GiB, next to a 12 GiB
    pool and 7.5 GiB of weights on the card. Fails here in seconds, not mid-matrix on a full GPU.
    """
    cfg = ModelConfig(
        vocab_size=1024, hidden_size=2560, intermediate_size=9728, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, max_position_embeddings=40960,
    )
    model = Qwen3(cfg).to(dtype=torch.bfloat16, device="cuda").eval()
    context, chunk = 32768, 2048
    cache = PagedKVCache(cfg, context // 16, "cuda", torch.bfloat16)
    table = BlockTable()
    cache.prepare_write(table, 0, context - chunk)
    table.num_tokens = context - chunk
    choices = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None):
        choices.append(torch._fused_sdp_choice(q, k, v, attn_mask=attn_mask))
        return real(q, k, v, attn_mask=attn_mask)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model.prefill(cache, table, torch.randint(0, 1024, (chunk,), device="cuda"), context - chunk)
    torch.cuda.synchronize()
    assert int(torch.nn.attention.SDPBackend.MATH) not in choices, f"backends chosen: {choices}"
    assert torch.cuda.max_memory_allocated() - base < 3 * 2**30


def test_sanitize_shapes_launch_on_cuda():
    """The sanitize stage's decode launches, without the sanitizer, so a bad case fails here first."""
    spec = importlib.util.spec_from_file_location("sanitize_transfer", Path(__file__).parent / "sanitize_transfer.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.decode_launches("cuda") == len(driver.DECODE_SHAPES)


def test_engine_generates_on_cuda_with_real_weights(checkpoint):
    model = load_model(str(resolve_checkpoint(checkpoint)), device="cuda", dtype=torch.bfloat16)
    engine = Engine(model, PagedKVCache(model.cfg, 64, "cuda", torch.bfloat16), EngineConfig(prefill_chunk=64))
    requests = [Request(str(i), list(range(1, 40 + i)), max_new_tokens=8) for i in range(4)]
    for request in requests:
        engine.add_request(request)
    engine.run_until_idle()
    assert all(len(r.generated) == 8 for r in requests)
    assert engine.cache.allocator.num_used == 0


def test_native_library_and_torch_share_a_context():
    """The one assumption everything in the library rests on, checked first and in isolation."""
    cache = PagedKVCache(CFG, 8, "cuda", torch.bfloat16)
    cache.pool.copy_(torch.randn(cache.pool.shape, device="cuda", dtype=torch.float32).to(torch.bfloat16))
    backend = NativeBackend("cuda", staging_bytes=2 * cache.block_bytes, ids_capacity=16)
    try:
        before = zlib.crc32(cache.pool[[5, 2]].cpu().view(torch.uint16).numpy().tobytes())
        out = backend.start_out(cache, [5, 2], False)
        out.wait()
        assert zlib.crc32(out.payload.memory.numpy().tobytes()) == before
        # A pinned buffer of the library's must look pinned to torch's own runtime.
        target = torch.zeros(cache.block_bytes * 2, dtype=torch.uint8, device="cuda")
        target.copy_(out.payload.memory[: cache.block_bytes * 2], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        assert zlib.crc32(target.cpu().numpy().tobytes()) == before
        # Torch's stream handle must work in the library, which is what FP8 relies on.
        stream = torch.cuda.Stream()
        backend.engine.ext_wait(stream.cuda_stream, 0, 0)
        backend.engine.ext_record(stream.cuda_stream, 0, 0)
        finished = backend.engine.ext_finish(stream.cuda_stream)
        backend.engine.wait(finished)
        backend.engine.release(finished)
    finally:
        backend.close()


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_scheduler_swaps_through_the_native_path_on_real_weights(checkpoint, fp8):
    model = load_model(str(resolve_checkpoint(checkpoint)), device="cuda", dtype=torch.bfloat16)
    costs = CostModel([0.0, 0.0], [0.0, 0.0], [1.0, 1.0], fp8_swap_out_s_coef=[0.0, 0.0], fp8_swap_in_s_coef=[0.0, 0.0])
    agent = AgentConfig(policy=Policy.SWAP, prefix_cache=False)
    cache = PagedKVCache(model.cfg, 64, "cuda", torch.bfloat16)
    backend = NativeBackend("cuda", staging_bytes=8 * cache.block_bytes, ids_capacity=256)
    scheduler = AgentScheduler(model, cache, costs, EngineConfig(prefill_chunk=64), agent, transfer=backend, offload_fp8=fp8)
    try:
        first = AgentTurn("first", list(range(1, 100)), 8, session_id="s")
        scheduler.submit(first)
        scheduler.run_until_idle()
        resumed = AgentTurn("resumed", first.prompt + first.generated + [7, 8], 8, session_id="s")
        scheduler.submit(resumed)
        scheduler.run_until_idle()
        assert resumed.reused_tokens > 0 and len(resumed.generated) == 8
        if not fp8:
            # BF16 is byte-exact, so a resume must match the same turns with KV left in place.
            # Same turns, same code path: a fresh prefill would round differently from decode.
            keep = AgentConfig(policy=Policy.KEEP_TTL, prefix_cache=False)
            kept = AgentScheduler(model, PagedKVCache(model.cfg, 64, "cuda", torch.bfloat16), costs, EngineConfig(prefill_chunk=64), keep)
            kept.submit(AgentTurn("first", list(first.prompt), 8, session_id="s"))
            kept.run_until_idle()
            reference = AgentTurn("resumed", list(resumed.prompt), 8, session_id="s")
            kept.submit(reference)
            kept.run_until_idle()
            assert reference.reused_tokens == resumed.reused_tokens
            assert resumed.generated == reference.generated
        scheduler.drain_caches()
        assert cache.allocator.num_used == 0
    finally:
        backend.close()

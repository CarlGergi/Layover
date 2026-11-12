"""What the decode and prefill steps cost the host, independent of the GPU.

A step's layers must share what is built for the step, read nothing back from the GPU, and
never expand K and V to the query heads. These hold on CPU too, so they are checked here; the
GPU tests check that the fused attention kernels take the result.
"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from layover import profiling
from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request
from layover.kv import BlockTable, PagedKVCache
from layover.model import PrefillChunk, Qwen3, prefill_attention

CFG = ModelConfig.micro()


def model_of_depth(layers: int) -> tuple[Qwen3, ModelConfig]:
    cfg = replace(CFG, num_hidden_layers=layers)
    torch.manual_seed(0)
    return Qwen3(cfg).to(torch.float32).eval(), cfg


def prefilled(model, cfg, lengths, blocks=128):
    cache = PagedKVCache(cfg, blocks, dtype=torch.float32)
    tables = []
    for n in lengths:
        table = BlockTable()
        model.prefill(cache, table, torch.arange(n) % cfg.vocab_size, 0)
        tables.append(table)
    return cache, tables


class Counter:
    """Counts calls to torch.tensor, which is how a list from the host becomes a tensor."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = torch.tensor

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(torch, "tensor", counted)


def host_tensors_per_decode(monkeypatch, layers):
    model, cfg = model_of_depth(layers)
    cache, tables = prefilled(model, cfg, [5, 17, 33])
    tokens = torch.tensor([1, 2, 3])
    with monkeypatch.context() as patch:
        counter = Counter(patch)
        model.decode(cache, tables, tokens)
    return counter.calls


def test_a_decode_step_builds_its_tables_once_whatever_the_depth(monkeypatch):
    shallow, deep = host_tensors_per_decode(monkeypatch, 2), host_tensors_per_decode(monkeypatch, 6)
    assert shallow == deep <= 2


def host_tensors_per_prefill_chunk(monkeypatch, layers):
    model, cfg = model_of_depth(layers)
    cache, (table,) = prefilled(model, cfg, [40])
    with monkeypatch.context() as patch:
        counter = Counter(patch)
        model.prefill(cache, table, torch.arange(24), 40)
    return counter.calls


def test_a_prefill_chunk_builds_its_block_ids_once_whatever_the_depth(monkeypatch):
    shallow, deep = host_tensors_per_prefill_chunk(monkeypatch, 2), host_tensors_per_prefill_chunk(monkeypatch, 6)
    assert shallow == deep <= 1


def test_a_decode_step_reads_nothing_back_from_the_device(monkeypatch):
    """With a backend that checks nothing, as the trusted Triton path does, the model itself
    must not read anything back: on a GPU each read waits for every layer queued before it."""
    model, cfg = model_of_depth(4)
    cache, tables = prefilled(model, cfg, [5, 17])
    model.decode_attention = lambda q, pool, layer, table, lengths, groups: torch.zeros_like(q)
    reads = []
    for name in ("item", "tolist", "__int__", "__bool__", "cpu", "numpy"):
        real = getattr(torch.Tensor, name)
        monkeypatch.setattr(torch.Tensor, name, lambda self, *a, _name=name, _real=real, **k: (reads.append(_name), _real(self, *a, **k))[1])
    model.decode(cache, tables, torch.tensor([1, 2]))
    assert reads == []


def test_the_engine_reads_a_decode_steps_tokens_back_once(monkeypatch):
    model, cfg = model_of_depth(2)
    cache = PagedKVCache(cfg, 64, dtype=torch.float32)
    engine = Engine(model, cache, EngineConfig(max_batch=4))
    for i in range(4):
        engine.add_request(Request(f"r{i}", [1 + i, 2, 3], max_new_tokens=3))
    engine.step()  # prefill, and each request's first token
    # The CPU reference attention reads lengths back per layer; the GPU path does not.
    model.decode_attention = lambda q, pool, layer, table, lengths, groups: torch.zeros_like(q)
    reads = []
    for name in ("item", "tolist", "__int__"):
        real = getattr(torch.Tensor, name)
        monkeypatch.setattr(torch.Tensor, name, lambda self, *a, _name=name, _real=real, **k: (reads.append(_name), _real(self, *a, **k))[1])
    engine.step()
    assert reads == ["tolist"]
    monkeypatch.undo()
    finished = engine.run_until_idle()
    assert all(len(r.generated) == 3 for r in finished)


def test_a_corrupt_table_is_refused_on_the_host_before_any_launch():
    model, cfg = model_of_depth(2)
    cache, tables = prefilled(model, cfg, [5, 17], blocks=16)
    tables[1].blocks[0] = 16
    with pytest.raises(ValueError, match="outside a pool of 16 blocks"):
        cache.decode_batch(tables)
    tables[1].blocks[0] = 3
    tables[1].num_tokens = 10_000
    with pytest.raises(ValueError, match="cannot hold"):
        cache.decode_batch(tables)


def test_the_decode_batch_is_what_the_layers_used_to_build_for_themselves():
    model, cfg = model_of_depth(2)
    cache, tables = prefilled(model, cfg, [5, 17, 33])
    for table in tables:
        cache.prepare_write(table, table.num_tokens, table.num_tokens + 1)
    batch = cache.decode_batch(tables)
    width = max(len(t.blocks) for t in tables)
    assert batch.table.tolist() == [t.blocks + [0] * (width - len(t.blocks)) for t in tables]
    assert batch.lengths.tolist() == [6, 18, 34]
    assert batch.write_blocks.tolist() == [t.blocks[(t.num_tokens - 1) // 16] for t in tables]
    assert batch.write_slots.tolist() == [(t.num_tokens - 1) % 16 for t in tables]


def reference_attention(q, all_k, all_v, start, groups):
    """What prefill attention computed before: K and V expanded to the query heads, and an
    explicit mask, causal within the chunk with everything before it visible."""
    tokens, end = q.shape[1], all_k.shape[1]
    allowed = torch.arange(end)[None] <= (start + torch.arange(tokens))[:, None]
    return F.scaled_dot_product_attention(
        q, all_k.repeat_interleave(groups, dim=0), all_v.repeat_interleave(groups, dim=0), attn_mask=allowed
    )


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("start", [0, 1, 16, 45])
def test_prefill_attention_matches_expanding_k_and_v_with_an_explicit_mask(groups, start):
    gen = torch.Generator().manual_seed(groups * 100 + start)
    kv_heads, tokens, dim = 2, 19, 8
    cfg = replace(CFG, num_attention_heads=kv_heads * groups, num_key_value_heads=kv_heads, head_dim=dim, num_hidden_layers=1)
    cache = PagedKVCache(cfg, 8, dtype=torch.float32)
    table = BlockTable()
    end = start + tokens
    cache.prepare_write(table, 0, end)
    history_k = torch.randn(kv_heads, end, dim, generator=gen)
    history_v = torch.randn(kv_heads, end, dim, generator=gen)
    cache.write_range(table, 0, 0, history_k, history_v)
    q = torch.randn(kv_heads * groups, tokens, dim, generator=gen)
    chunk = PrefillChunk(start, end, *cache.chunk_locations(table, start, tokens))
    got = prefill_attention(q, history_k[:, start:], history_v[:, start:], cache, 0, chunk, groups)
    expected = reference_attention(q, history_k, history_v, start, groups)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_prefill_never_expands_k_and_v_and_the_first_chunk_reads_no_history(monkeypatch):
    model, cfg = model_of_depth(2)
    cache = PagedKVCache(cfg, 32, dtype=torch.float32)
    table = BlockTable()
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", lambda *a, **k: pytest.fail("K or V expanded to the query heads"))
    reads = []
    real = cache.read_blocks
    monkeypatch.setattr(cache, "read_blocks", lambda *a: (reads.append(a[2]), real(*a))[1])
    model.prefill(cache, table, torch.arange(20), 0)
    assert reads == []
    model.prefill(cache, table, torch.arange(12), 20)
    assert reads == [32] * cfg.num_hidden_layers


def test_the_folded_mask_is_built_once_per_chunk():
    model, cfg = model_of_depth(3)
    cache, (table,) = prefilled(model, cfg, [10])
    cache.prepare_write(table, 10, 15)
    chunk = PrefillChunk(10, 15, *cache.chunk_locations(table, 10, 5))
    assert chunk.folded_mask(2) is chunk.folded_mask(2)
    assert chunk.folded_mask(2).shape == (10, 15)


def test_nvtx_ranges_cost_nothing_when_off_and_nest_when_on(monkeypatch):
    model, cfg = model_of_depth(2)
    cache, tables = prefilled(model, cfg, [5])
    events = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append(("pop",)))
    assert profiling.nvtx_range("x") is profiling.nvtx_range("y")
    model.decode(cache, tables, torch.tensor([1]))
    assert events == []
    monkeypatch.setattr(profiling, "ENABLED", True)
    model.decode(cache, tables, torch.tensor([1]))
    model.prefill(cache, tables[0], torch.arange(3), 6)
    names = [e[1] for e in events if e[0] == "push"]
    assert names == ["decode", "layer0.attention", "layer0.mlp", "layer1.attention", "layer1.mlp"] * 1 + [
        "prefill", "layer0.attention", "layer0.mlp", "layer1.attention", "layer1.mlp"
    ]
    depth = 0
    for event in events:
        depth += 1 if event[0] == "push" else -1
        assert depth >= 0
    assert depth == 0


def test_the_engine_marks_its_steps(monkeypatch):
    model, cfg = model_of_depth(2)
    engine = Engine(model, PagedKVCache(cfg, 32, dtype=torch.float32))
    engine.add_request(Request("r", [1, 2, 3], max_new_tokens=2))
    names = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", names.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(profiling, "ENABLED", True)
    engine.step()
    assert names[:2] == ["step", "step.prefill"] and "step.decode" in names

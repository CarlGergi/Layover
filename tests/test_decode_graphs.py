"""Decode through captured graphs: the same logits as without, at every batch size, and padded
rows that touch nothing but their scratch block. On a CPU the graph is a stand-in that runs the
captured step eagerly, which tests the buffers, padding and fallbacks; tests/test_gpu.py runs a
real CUDA graph."""

import pytest
import torch

from layover.config import ModelConfig
from layover.decode_graphs import DecodeGraphs, eager_graph
from layover.engine import Engine, EngineConfig, Request
from layover.kv import BlockTable, PagedKVCache
from layover.model import Qwen3

CFG = ModelConfig.micro()


def model():
    torch.manual_seed(0)
    return Qwen3(CFG).to(torch.float32).eval()


def prefilled(m, lengths):
    cache = PagedKVCache(CFG, 96, dtype=torch.float32)
    tables = []
    for i, n in enumerate(lengths):
        table = BlockTable()
        m.prefill(cache, table, (torch.arange(n) * (i + 3)) % CFG.vocab_size, 0)
        tables.append(table)
    return cache, tables


@pytest.mark.parametrize("lengths", [[5], [5, 33], [1, 16, 40], [7, 8, 9, 10, 11], [3] * 9], ids=["1", "2", "3-pads-to-4", "5-pads-to-8", "9-has-no-graph"])
def test_graph_decode_matches_eager_decode_over_several_steps(lengths):
    m = model()
    eager_cache, eager_tables = prefilled(m, lengths)
    graph_cache, graph_tables = prefilled(m, lengths)
    graphs = DecodeGraphs(m, graph_cache, max_tokens=256, capture=eager_graph)
    tokens = torch.tensor([(7 * i + 1) % CFG.vocab_size for i in range(len(lengths))])
    for _ in range(20):  # past a block boundary for every row
        m.decode_graphs = None
        expected = m.decode(eager_cache, eager_tables, tokens)
        m.decode_graphs = graphs
        got = m.decode(graph_cache, graph_tables, tokens)
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)
        tokens = expected.argmax(-1)
    assert [t.num_tokens for t in graph_tables] == [t.num_tokens for t in eager_tables]


def test_padded_rows_write_only_to_the_scratch_block():
    m = model()
    cache, tables = prefilled(m, [20])
    graphs = DecodeGraphs(m, cache, sizes=(4,), max_tokens=256, capture=eager_graph)
    m.decode_graphs = graphs
    before = cache.pool.clone()
    m.decode(cache, tables, torch.tensor([3]))
    changed = {int(b) for b in (cache.pool != before).flatten(1).any(1).nonzero().flatten()}
    # The real row writes token 20 into its second block. The three padded rows write the scratch
    # block, with the same values each step, which capture already wrote there.
    assert tables[0].blocks[1] in changed and changed <= {tables[0].blocks[1], graphs.scratch}
    assert graphs.scratch not in tables[0].blocks
    assert cache.allocator.refcount(graphs.scratch) == 1


def test_a_table_wider_than_the_graphs_runs_without_them():
    m = model()
    cache, tables = prefilled(m, [70])
    graphs = DecodeGraphs(m, cache, max_tokens=32, capture=eager_graph)
    for table in tables:
        cache.prepare_write(table, table.num_tokens, table.num_tokens + 1)
    assert graphs.run(tables, torch.tensor([1])) is None
    m.decode_graphs = graphs
    m.decode(cache, tables, torch.tensor([1]))  # falls back to the eager step


def test_a_corrupt_table_is_refused_before_a_replay():
    m = model()
    cache, tables = prefilled(m, [5])
    graphs = DecodeGraphs(m, cache, max_tokens=256, capture=eager_graph)
    tables[0].blocks[0] = 10_000
    with pytest.raises(ValueError, match="outside a pool"):
        graphs.run(tables, torch.tensor([1]))


def test_the_logits_returned_survive_the_next_replay():
    m = model()
    cache, tables = prefilled(m, [5])
    m.decode_graphs = DecodeGraphs(m, cache, max_tokens=256, capture=eager_graph)
    first = m.decode(cache, tables, torch.tensor([1]))
    kept = first.clone()
    m.decode(cache, tables, torch.tensor([2]))
    assert torch.equal(first, kept)


def test_engine_output_is_the_same_with_graphs():
    def run(graphs):
        m = model()
        cache = PagedKVCache(CFG, 96, dtype=torch.float32)
        if graphs:
            m.decode_graphs = DecodeGraphs(m, cache, max_tokens=256, capture=eager_graph)
        engine = Engine(m, cache, EngineConfig(max_batch=6, prefill_chunk=9, prefill_budget=9))
        requests = [Request(str(i), [(5 * i + j) % CFG.vocab_size for j in range(3 + 4 * i)], max_new_tokens=12) for i in range(6)]
        for request in requests:
            engine.add_request(request)
        engine.run_until_idle()
        return [r.generated for r in requests]

    assert run(True) == run(False)

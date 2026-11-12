"""CUDA graphs for decode at small batch sizes.

A decode step launches about 2,100 kernels for Qwen3-4B, about 59 per layer. At batch 32 the GPU
is the bottleneck and the host keeps ahead of it, but at batch 1 the GPU's work is little more
than reading the weights once, so issuing those launches from Python sets the pace (62 ms a step
on an L4 against a 27 ms weight-read floor). A CUDA graph records a step's launches once and
replays them as one, as vLLM does for decode (Kwon et al., SOSP 2023; reference design only).

A graph replays the same kernels on the same memory, so each captured batch size gets fixed
input tensors: the tokens, and one packed int32 block holding every row's write block, write
slot, length and a block table of fixed width. A step copies its rows into that block and
replays. A batch smaller than a captured size is padded with rows of length 1 that read and
write a scratch block reserved here and never handed out, so a padded row cannot touch a real
sequence's KV. Batches larger than the largest captured size run without a graph.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from layover.kv.paged_cache import BlockTable, DecodeBatch, PagedKVCache

DEFAULT_SIZES = (1, 2, 4, 8)


def cuda_graph(step: Callable[[], torch.Tensor], pool=None) -> tuple[Callable[[], None], torch.Tensor, object]:
    """Capture `step` after warming it up on a side stream, as torch's graph docs require.
    Returns a replay function, the output tensor it refreshes, and the memory pool to share."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = step()
    return graph.replay, out, graph.pool()


def eager_graph(step: Callable[[], torch.Tensor], pool=None) -> tuple[Callable[[], None], torch.Tensor, object]:
    """A stand-in that runs `step` on every replay, for testing the buffers on a CPU."""
    out = step()
    return (lambda: out.copy_(step())), out, pool


class DecodeGraphs:
    def __init__(
        self,
        model,
        cache: PagedKVCache,
        sizes: tuple[int, ...] = DEFAULT_SIZES,
        max_tokens: int | None = None,
        capture: Callable = cuda_graph,
    ):
        self.model, self.cache = model, cache
        self.sizes = tuple(sorted(sizes))
        max_tokens = max_tokens or model.cfg.max_position_embeddings
        self.width = min(cache.blocks_for(max_tokens), cache.allocator.num_blocks)
        self.scratch = cache.allocator.alloc(1)[0]
        largest = self.sizes[-1]
        self.tokens = torch.zeros(largest, dtype=torch.long, device=cache.device)
        self.packed = torch.zeros(largest, 3 + self.width, dtype=torch.int32, device=cache.device)
        self.packed.copy_(self._padding(largest))
        self._replays: dict[int, tuple[Callable[[], None], torch.Tensor]] = {}
        pool = None
        for size in reversed(self.sizes):  # largest first, so the smaller ones fit in its pool
            replay, out, pool = capture(lambda size=size: self._step(size), pool)
            self._replays[size] = (replay, out)

    def _padding(self, rows: int) -> torch.Tensor:
        """Rows of length 1 that write and read slot 0 of the scratch block."""
        row = torch.zeros(3 + self.width, dtype=torch.int32)
        row[0], row[1], row[2], row[3] = self.scratch, 0, 1, self.scratch
        return row.repeat(rows, 1)

    def _step(self, size: int) -> torch.Tensor:
        packed = self.packed[:size]
        batch = DecodeBatch(packed[:, 0].long(), packed[:, 1].long(), packed[:, 3:], packed[:, 2])
        return self.model.decode_step(self.cache, batch, self.tokens[:size])

    def size_for(self, batch: int) -> int | None:
        return next((size for size in self.sizes if size >= batch), None)

    def run(self, tables: list[BlockTable], tokens: torch.Tensor) -> torch.Tensor | None:
        """Logits for a step whose tables are prepared for writing, or None when no graph fits.
        The rows are checked on the host first, as an eager step's are."""
        size = self.size_for(len(tables))
        if size is None or max(len(t.blocks) for t in tables) > self.width:
            return None
        rows = torch.cat([self.cache.decode_rows(tables, self.width), self._padding(size - len(tables))])
        self.packed[:size].copy_(self.cache._to_device(rows), non_blocking=True)
        self.tokens[: len(tables)].copy_(tokens)
        self.tokens[len(tables) : size].zero_()
        replay, out = self._replays[size]
        replay()
        # The graph refreshes the same tensor on its next replay.
        return out[: len(tables)].clone()

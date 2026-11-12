"""Qwen3 forward pass over a paged KV cache.

Parameter names match the Hugging Face checkpoint so weights load without renaming.
Linear layers run on torch.matmul, which is cuBLAS on CUDA.

What a step's layers share is built once per step, not per layer: a prefill chunk's block ids
and write positions, a decode batch's block table and lengths. Each is one copy to the device,
and the layers read nothing back from it, so the host can queue a whole step ahead of the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from layover.config import ModelConfig
from layover.kernels.decode_attention import paged_decode_attention_torch
from layover.kv.paged_cache import BlockTable, DecodeBatch, PagedKVCache
from layover.profiling import nvtx_range


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The variance is computed in FP32, since squaring BF16 activations loses too
        # much precision. The weight applies after rounding back, as in the reference.
        wide = x.float()
        scaled = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * scaled.to(x.dtype)


@dataclass
class PrefillChunk:
    """Tokens [start, end) of one sequence, as every layer of the chunk sees them."""

    start: int
    end: int
    block_ids: torch.Tensor
    write_blocks: torch.Tensor
    write_slots: torch.Tensor
    _masks: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.end - self.start

    def folded_mask(self, groups: int) -> torch.Tensor:
        """[groups * tokens, end]: query row g * tokens + t sees keys up to start + t. Built once
        per chunk; the layers share it."""
        if groups not in self._masks:
            device = self.block_ids.device
            allowed = torch.arange(self.end, device=device)[None] <= (self.start + torch.arange(self.tokens, device=device))[:, None]
            self._masks[groups] = allowed.repeat(groups, 1)
        return self._masks[groups]


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: PagedKVCache, layer: int, chunk: PrefillChunk, groups: int) -> torch.Tensor:
    """Attention for one prefill chunk. q is [heads, tokens, dim]; k and v are the chunk's own,
    [kv_heads, tokens, dim], already written to the cache. Returns [heads, tokens, dim].

    Neither path expands K and V to the query heads. The first chunk attends only to itself, so
    it is plain causal attention with grouped queries, which FlashAttention takes as is. A later
    chunk also attends to everything before it, which needs a mask that SDPA's causal flag
    cannot express. Folding the query heads that share a KV head into the sequence makes that
    ordinary multi-head attention over the unexpanded K and V, which the fused kernels take with
    a mask. The history itself is still gathered from the pool once per layer; reading it in
    place needs a paged prefill kernel.
    """
    # Through the cache's dtype and back, as a later chunk and decode see these values.
    k, v = k.to(cache.dtype).to(q.dtype), v.to(cache.dtype).to(q.dtype)
    if chunk.start == 0:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=groups > 1)
    all_k, all_v = cache.read_blocks(chunk.block_ids, layer, chunk.end)
    kv_heads, dim = all_k.shape[0], q.shape[-1]
    folded = q.reshape(kv_heads, groups * chunk.tokens, dim)
    out = F.scaled_dot_product_attention(folded, all_k.to(q.dtype), all_v.to(q.dtype), attn_mask=chunk.folded_mask(groups))
    return out.reshape(kv_heads * groups, chunk.tokens, dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Return q, k, v as [..., heads, tokens, head_dim] with RoPE applied."""
        cfg = self.cfg
        batch, tokens, _ = x.shape
        q = self.q_proj(x).view(batch, tokens, cfg.num_attention_heads, cfg.head_dim)
        k = self.k_proj(x).view(batch, tokens, cfg.num_key_value_heads, cfg.head_dim)
        v = self.v_proj(x).view(batch, tokens, cfg.num_key_value_heads, cfg.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k, v

    def forward_prefill(self, x, cos, sin, cache: PagedKVCache, chunk: PrefillChunk):
        q, k, v = self.project(x, cos, sin)
        cache.write_at(chunk.write_blocks, chunk.write_slots, self.layer, k[0], v[0])
        out = prefill_attention(q[0], k[0], v[0], cache, self.layer, chunk, self.cfg.kv_groups)
        return self.o_proj(out.transpose(0, 1).reshape(1, chunk.tokens, -1))

    def forward_decode(self, x, cos, sin, cache: PagedKVCache, batch: DecodeBatch, attend):
        q, k, v = self.project(x, cos, sin)
        cache.pool[batch.write_blocks, 0, self.layer, :, batch.write_slots] = k[:, :, 0].to(cache.dtype)
        cache.pool[batch.write_blocks, 1, self.layer, :, batch.write_slots] = v[:, :, 0].to(cache.dtype)
        out = attend(q[:, :, 0], cache.pool, self.layer, batch.table, batch.lengths, self.cfg.kv_groups)
        return self.o_proj(out.reshape(x.shape[0], 1, -1).to(x.dtype))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", freqs, persistent=False)
        self.decode_attention = paged_decode_attention_torch
        # Names made once, since a range is opened per layer per step.
        self._range_names = [(f"layer{i}.attention", f"layer{i}.mlp") for i in range(cfg.num_hidden_layers)]

    def rope(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin shaped [batch, 1, tokens, head_dim] for broadcasting over heads."""
        angles = positions.float()[..., None] * self.inv_freq.float()
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    @torch.inference_mode()
    def prefill(
        self,
        cache: PagedKVCache,
        table: BlockTable,
        tokens: torch.Tensor,
        start: int,
        all_logits: bool = False,
        layer_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run tokens [count] at position `start`; returns logits of the last token.

        Only the last hidden state goes through lm_head unless all_logits is set: at
        Qwen3-4B's vocabulary, logits for a long chunk would dwarf the KV cache.
        layer_outputs, when given, collects each decoder layer's output for parity tests.
        """
        count = tokens.shape[0]
        with nvtx_range("prefill"):
            cache.prepare_write(table, start, start + count)
            chunk = PrefillChunk(start, start + count, *cache.chunk_locations(table, start, count))
            x = self.model.embed_tokens(tokens)[None]
            positions = torch.arange(start, start + count, device=tokens.device)[None]
            cos, sin = self.rope(positions, x.dtype)
            for layer, (attention_name, mlp_name) in zip(self.model.layers, self._range_names):
                with nvtx_range(attention_name):
                    attended = layer.self_attn.forward_prefill(layer.input_layernorm(x), cos, sin, cache, chunk)
                with nvtx_range(mlp_name):
                    x = layer.finish(x, attended)
                if layer_outputs is not None:
                    layer_outputs.append(x[0].clone())
            x = self.model.norm(x)
            return self.lm_head(x[0] if all_logits else x[0, -1])

    @torch.inference_mode()
    def decode(self, cache: PagedKVCache, tables: list[BlockTable], tokens: torch.Tensor) -> torch.Tensor:
        """Run one token for each table, appending it; returns logits [batch, vocab]."""
        with nvtx_range("decode"):
            for table in tables:
                cache.prepare_write(table, table.num_tokens, table.num_tokens + 1)
            batch = cache.decode_batch(tables)
            x = self.model.embed_tokens(tokens)[:, None]
            # The position of each new token is its length less one.
            cos, sin = self.rope((batch.lengths.long() - 1)[:, None], x.dtype)
            for layer, (attention_name, mlp_name) in zip(self.model.layers, self._range_names):
                with nvtx_range(attention_name):
                    attended = layer.self_attn.forward_decode(layer.input_layernorm(x), cos, sin, cache, batch, self.decode_attention)
                with nvtx_range(mlp_name):
                    x = layer.finish(x, attended)
            return self.lm_head(self.model.norm(x)[:, 0])

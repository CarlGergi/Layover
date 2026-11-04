"""Qwen3 forward pass over a paged KV cache.

Parameter names match the Hugging Face checkpoint so weights load without renaming.
Linear layers run on torch.matmul, which is cuBLAS on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layover.config import ModelConfig
from layover.kernels.decode_attention import paged_decode_attention_torch
from layover.kv.paged_cache import BlockTable, PagedKVCache


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

    def forward_prefill(self, x, cos, sin, cache: PagedKVCache, table: BlockTable, start: int):
        q, k, v = self.project(x, cos, sin)
        tokens = x.shape[1]
        cache.write_range(table, self.layer, start, k[0], v[0])
        end = start + tokens
        all_k, all_v = cache.read_range(table, self.layer, end)
        groups = self.cfg.kv_groups
        # Causal within the chunk, everything before the chunk is visible.
        allowed = torch.arange(end, device=x.device)[None] <= (start + torch.arange(tokens, device=x.device))[:, None]
        out = F.scaled_dot_product_attention(
            q[0],
            all_k.repeat_interleave(groups, dim=0).to(q.dtype),
            all_v.repeat_interleave(groups, dim=0).to(q.dtype),
            attn_mask=allowed,
        )
        return self.o_proj(out.transpose(0, 1).reshape(1, tokens, -1))

    def forward_decode(self, x, cos, sin, cache: PagedKVCache, tables: list[BlockTable], positions, attend):
        q, k, v = self.project(x, cos, sin)
        batch = x.shape[0]
        blocks = torch.tensor([t.blocks[p // cache.block_tokens] for t, p in zip(tables, positions)], device=x.device)
        slots = torch.tensor([p % cache.block_tokens for p in positions], device=x.device)
        cache.pool[blocks, 0, self.layer, :, slots] = k[:, :, 0].to(cache.dtype)
        cache.pool[blocks, 1, self.layer, :, slots] = v[:, :, 0].to(cache.dtype)
        table, lengths = cache.block_table_tensor(tables)
        out = attend(q[:, :, 0], cache.pool, self.layer, table, lengths, self.cfg.kv_groups)
        return self.o_proj(out.reshape(batch, 1, -1).to(x.dtype))


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
        cache.prepare_write(table, start, start + count)
        x = self.model.embed_tokens(tokens)[None]
        positions = torch.arange(start, start + count, device=tokens.device)[None]
        cos, sin = self.rope(positions, x.dtype)
        for layer in self.model.layers:
            attended = layer.self_attn.forward_prefill(layer.input_layernorm(x), cos, sin, cache, table, start)
            x = layer.finish(x, attended)
            if layer_outputs is not None:
                layer_outputs.append(x[0].clone())
        x = self.model.norm(x)
        return self.lm_head(x[0] if all_logits else x[0, -1])

    @torch.inference_mode()
    def decode(self, cache: PagedKVCache, tables: list[BlockTable], tokens: torch.Tensor) -> torch.Tensor:
        """Run one token for each table, appending it; returns logits [batch, vocab]."""
        positions = [t.num_tokens for t in tables]
        for table, position in zip(tables, positions):
            cache.prepare_write(table, position, position + 1)
        x = self.model.embed_tokens(tokens)[:, None]
        cos, sin = self.rope(torch.tensor(positions, device=tokens.device)[:, None], x.dtype)
        for layer in self.model.layers:
            attended = layer.self_attn.forward_decode(
                layer.input_layernorm(x), cos, sin, cache, tables, positions, self.decode_attention
            )
            x = layer.finish(x, attended)
        return self.lm_head(self.model.norm(x)[:, 0])

"""Model configuration and the KV-size arithmetic everything else builds on."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

DTYPE_BYTES = {"fp8": 1, "bf16": 2, "fp16": 2, "fp32": 4}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    eos_token_id: int | None = None
    bos_token_id: int | None = None

    @classmethod
    def from_hf_dict(cls, raw: dict) -> "ModelConfig":
        """Parse a Hugging Face Qwen3 config.json.

        head_dim is read from the config and not derived: Qwen3-4B stores
        head_dim=128 while hidden_size/num_attention_heads is 80, and the derived
        value would understate the KV size by 37.5%.
        """
        if raw.get("model_type", "qwen3") != "qwen3":
            raise ValueError(f"unsupported model_type {raw['model_type']!r}")
        if raw.get("use_sliding_window"):
            raise ValueError("sliding-window attention is not supported")
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError(f"unsupported rope configuration {rope!r}")
        theta = raw.get("rope_theta", rope.get("rope_theta", 10000.0))
        eos = raw.get("eos_token_id")
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads") or raw["num_attention_heads"],
            head_dim=raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"],
            max_position_embeddings=raw["max_position_embeddings"],
            rope_theta=float(theta),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
            eos_token_id=eos[0] if isinstance(eos, list) else eos,
            bos_token_id=raw.get("bos_token_id"),
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "ModelConfig":
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(name_or_path, "config.json")).parent
        return cls.from_hf_dict(json.loads((path / "config.json").read_text()))

    @classmethod
    def micro(cls, **overrides) -> "ModelConfig":
        """A very small random-weight model for fast property tests and smoke runs.

        head_dim deliberately differs from hidden_size/num_attention_heads and
        KV heads are fewer than query heads, so tests exercise the same shape
        relationships as Qwen3-4B.
        """
        base = cls(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=24,
            max_position_embeddings=40960,
            rope_theta=10000.0,
        )
        return replace(base, **overrides)

    @property
    def kv_groups(self) -> int:
        """Number of query heads that read each KV head."""
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype: str = "bf16") -> int:
        """Bytes of K and V one token occupies across all layers."""
        return (
            self.num_hidden_layers * 2 * self.num_key_value_heads * self.head_dim * DTYPE_BYTES[dtype]
        )

    def to_hf_dict(self) -> dict:
        return {
            "model_type": "qwen3",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": self.num_hidden_layers,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
        }

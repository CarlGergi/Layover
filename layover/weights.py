"""Load a Qwen3 checkpoint from safetensors into the engine's model."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from layover.config import ModelConfig
from layover.model import Qwen3


def resolve_checkpoint(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"]))


def load_model(name_or_path: str, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16) -> Qwen3:
    """Build the model on the meta device and adopt checkpoint tensors directly."""
    root = resolve_checkpoint(name_or_path)
    cfg = ModelConfig.from_pretrained(str(root))
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {root}")
    with torch.device("meta"):
        model = Qwen3(cfg)
    state: dict[str, torch.Tensor] = {}
    for file in files:
        for key, tensor in load_file(file, device=str(device)).items():
            state[key] = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    expected = set(model.state_dict()) - ({"lm_head.weight"} if cfg.tie_word_embeddings else set())
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(state, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    # inv_freq is a computed buffer, so it is still on the meta device.
    head = cfg.head_dim
    model.inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, head, 2, dtype=torch.float32, device=device) / head)
    )
    return model.eval()

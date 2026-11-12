"""Logit parity with transformers and continuous-batching behaviour."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request, RequestTooLarge, State
from layover.kv import BlockTable, PagedKVCache
from layover.model import Qwen3
from layover.weights import load_model

CFG = ModelConfig.micro()
BF16_EPS = 2.0**-7


def hf_model(dtype):
    torch.manual_seed(0)
    hf_cfg = Qwen3Config(**{k: v for k, v in CFG.to_hf_dict().items() if k != "model_type"})
    return Qwen3ForCausalLM(hf_cfg).to(dtype=dtype).eval()


def our_model(hf, dtype):
    model = Qwen3(CFG).to(dtype=dtype).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    return model


def prompts(count=20):
    gen = torch.Generator().manual_seed(7)
    lengths = torch.randint(1, 200, (count,), generator=gen).tolist()
    return [torch.randint(0, CFG.vocab_size, (n,), generator=gen) for n in lengths]


@pytest.fixture(scope="module", params=[torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def pair(request):
    hf = hf_model(request.param)
    return hf, our_model(hf, request.param), request.param


def tolerance(reference, dtype):
    """BF16 rounding is relative to the largest logit; FP32 gets a tight bound."""
    eps = BF16_EPS if dtype is torch.bfloat16 else 1e-5
    return 4 * eps * reference.abs().max().item()


@pytest.mark.parametrize("chunk", [10_000, 37, 16])
def test_prefill_logits_match_transformers(pair, chunk):
    hf, model, dtype = pair
    for prompt in prompts():
        with torch.inference_mode():
            expected = hf(prompt[None]).logits[0]
        cache = PagedKVCache(CFG, 64, dtype=dtype)
        table = BlockTable()
        got = []
        for start in range(0, len(prompt), chunk):
            got.append(model.prefill(cache, table, prompt[start : start + chunk], start, all_logits=True))
        got = torch.cat(got).float()
        assert (got - expected.float()).abs().max() <= tolerance(expected.float(), dtype)


def test_prefill_attention_is_eligible_for_a_fused_kernel(monkeypatch):
    """The math path builds the whole attention matrix, which a long context cannot afford.

    The call passes unbatched tensors, which torch gives a batch dimension before it picks a
    backend, on CUDA as on CPU. test_gpu.py checks the choice on the device itself.
    """
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False):
        calls.append(torch._fused_sdp_choice(q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=enable_gqa))
        # Neither path expands K and V to the query heads.
        assert k.shape[-3] == CFG.num_key_value_heads
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=enable_gqa)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    torch.manual_seed(0)
    model = Qwen3(CFG).to(dtype=torch.bfloat16).eval()
    cache = PagedKVCache(CFG, 8, dtype=torch.bfloat16)
    table = BlockTable()
    model.prefill(cache, table, torch.arange(32), 0)
    model.prefill(cache, table, torch.arange(32), 32)
    math = int(torch.nn.attention.SDPBackend.MATH)
    assert len(calls) == 2 * CFG.num_hidden_layers and math not in calls


def test_decode_logits_match_transformers(pair):
    hf, model, dtype = pair
    prompt_set = prompts()
    cache = PagedKVCache(CFG, 512, dtype=dtype)
    tables, sequences, split = [], [], 8
    for prompt in prompt_set:
        table = BlockTable()
        model.prefill(cache, table, prompt, 0)
        tables.append(table)
        sequences.append(prompt.tolist())
    for _ in range(split):
        tokens = torch.randint(0, CFG.vocab_size, (len(tables),))
        logits = model.decode(cache, tables, tokens).float()
        for row, sequence, token in zip(logits, sequences, tokens.tolist()):
            sequence.append(token)
            with torch.inference_mode():
                expected = hf(torch.tensor(sequence)[None]).logits[0, -1].float()
            assert (row - expected).abs().max() <= tolerance(expected, dtype)


def test_kv_size_on_disk_matches_config():
    assert PagedKVCache(CFG, 1).block_bytes == 16 * CFG.kv_bytes_per_token("bf16")


def test_greedy_output_matches_transformers_generate():
    hf = hf_model(torch.float32)
    model = our_model(hf, torch.float32)
    engine = Engine(model, PagedKVCache(CFG, 256, dtype=torch.float32), EngineConfig(prefill_chunk=16, prefill_budget=32))
    requests = [Request(str(i), p.tolist(), max_new_tokens=8) for i, p in enumerate(prompts(5))]
    for request in requests:
        engine.add_request(request)
    engine.run_until_idle()
    for request in requests:
        with torch.inference_mode():
            prompt = torch.tensor(request.prompt)[None]
            out = hf.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0, eos_token_id=None)
        assert request.generated == out[0, prompt.shape[1] :].tolist()


def solo_run(model, prompt, new):
    engine = Engine(model, PagedKVCache(CFG, 64, dtype=torch.float32))
    request = Request("solo", prompt, max_new_tokens=new)
    engine.add_request(request)
    engine.run_until_idle()
    return request.generated


def test_requests_join_a_running_batch_without_changing_output():
    hf = hf_model(torch.float32)
    model = our_model(hf, torch.float32)
    cache = PagedKVCache(CFG, 128, dtype=torch.float32)
    engine = Engine(model, cache, EngineConfig(max_batch=8, prefill_chunk=16, prefill_budget=16))
    all_prompts = [p.tolist() for p in prompts(6)]
    requests = [Request(str(i), p, max_new_tokens=6 + i) for i, p in enumerate(all_prompts)]
    joined_mid_flight = False
    pending = list(requests)
    engine.add_request(pending.pop(0))
    while not engine.idle or pending:
        if pending and (engine.idle or any(r.generated for r in engine.running)):
            joined_mid_flight = joined_mid_flight or not engine.idle
            engine.add_request(pending.pop(0))
        engine.step()
    assert joined_mid_flight
    for request in requests:
        assert request.generated == solo_run(model, request.prompt, request.max_new_tokens)
    assert cache.allocator.num_used == 0


def test_long_prompt_does_not_stall_running_decodes():
    hf = hf_model(torch.float32)
    model = our_model(hf, torch.float32)
    engine = Engine(model, PagedKVCache(CFG, 256, dtype=torch.float32), EngineConfig(prefill_chunk=16, prefill_budget=16))
    short = Request("short", [1, 2, 3], max_new_tokens=5)
    engine.add_request(short)
    engine.step()
    assert short.generated
    long = Request("long", list(range(320)), max_new_tokens=1)
    engine.add_request(long)
    for _ in range(4):
        before = len(short.generated)
        engine.step()
        assert len(short.generated) == before + 1 or short.state is State.FINISHED
    assert long.prefilled < len(long.prompt)
    engine.run_until_idle()
    assert long.state is State.FINISHED


def test_admission_waits_for_memory_and_never_exhausts_the_pool():
    hf = hf_model(torch.float32)
    model = our_model(hf, torch.float32)
    cache = PagedKVCache(CFG, 4, dtype=torch.float32)
    engine = Engine(model, cache, EngineConfig())
    a = Request("a", list(range(40)), max_new_tokens=8)
    b = Request("b", list(range(40)), max_new_tokens=8)
    engine.add_request(a)
    engine.add_request(b)
    engine.step()
    assert a.state is not State.WAITING and b.state is State.WAITING
    engine.run_until_idle()
    assert a.state is State.FINISHED and b.state is State.FINISHED


def test_stop_token_ends_a_request():
    hf = hf_model(torch.float32)
    model = our_model(hf, torch.float32)
    first = solo_run(model, [5, 6, 7], 4)
    cache = PagedKVCache(CFG, 16, dtype=torch.float32)
    engine = Engine(model, cache)
    request = Request("s", [5, 6, 7], max_new_tokens=4, stop_tokens=frozenset({first[1]}))
    engine.add_request(request)
    engine.run_until_idle()
    assert request.generated == first[: first.index(first[1]) + 1]


def test_oversized_and_empty_requests_are_rejected():
    hf = hf_model(torch.float32)
    engine = Engine(our_model(hf, torch.float32), PagedKVCache(CFG, 1, dtype=torch.float32))
    with pytest.raises(RequestTooLarge):
        engine.add_request(Request("big", list(range(40)), max_new_tokens=1))
    with pytest.raises(ValueError):
        engine.add_request(Request("empty", [], max_new_tokens=1))


def test_weights_round_trip_through_safetensors(tmp_path):
    hf = hf_model(torch.bfloat16)
    hf.save_pretrained(tmp_path, safe_serialization=True)
    model = load_model(str(tmp_path), dtype=torch.bfloat16)
    prompt = prompts(1)[0]
    cache = PagedKVCache(CFG, 64, dtype=torch.bfloat16)
    got = model.prefill(cache, BlockTable(), prompt, 0).float()
    with torch.inference_mode():
        expected = hf(prompt[None]).logits[0, -1].float()
    assert (got - expected).abs().max() <= tolerance(expected, torch.bfloat16)


def test_checkpoint_mismatch_is_reported(tmp_path):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(__import__("json").dumps(CFG.to_hf_dict()))
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_model(str(tmp_path))


REAL_PROMPTS = [
    "The capital of France is", "def fibonacci(n):", "In a distant galaxy", "Water boils at",
    "The quick brown fox", "Once upon a time", "To be or not to be", "import torch\n",
    "The mitochondria is", "Machine learning models", "SELECT * FROM users WHERE", "Rust is a language that",
    "The theory of relativity", "<html><body>", "A haiku about autumn:", "The GDP of Japan",
    "Photosynthesis converts", "for i in range(10):", "The largest ocean on Earth", "Dear hiring manager,",
]


@pytest.fixture(scope="module")
def real_pair(checkpoint):
    """Real weights, loaded by both implementations from the same checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from layover.weights import resolve_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = str(resolve_checkpoint(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(root)
    hf = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    return hf, load_model(root, device=device, dtype=torch.bfloat16), tokenizer, device


def real_close(got, expected):
    got, expected = got.float(), expected.float()
    return (got - expected).abs().max() <= 4 * BF16_EPS * expected.abs().max().item()


def test_real_weights_per_layer_outputs_match_transformers(real_pair):
    hf, model, tokenizer, device = real_pair
    for text in REAL_PROMPTS:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = hf(ids, output_hidden_states=True)
        cache = PagedKVCache(model.cfg, 8, device=device, dtype=torch.bfloat16)
        layers = []
        logits = model.prefill(cache, BlockTable(), ids[0], 0, all_logits=True, layer_outputs=layers)
        assert len(layers) == model.cfg.num_hidden_layers
        # hidden_states[0] is the embedding; the last entry is after the final norm.
        for ours, theirs in zip(layers[:-1], out.hidden_states[1:-1]):
            assert real_close(ours, theirs[0])
        assert real_close(model.model.norm(layers[-1]), out.hidden_states[-1][0])
        assert real_close(logits, out.logits[0])


def test_real_weights_greedy_output_matches_at_batch_size_one(real_pair):
    hf, model, tokenizer, device = real_pair
    for text in REAL_PROMPTS[:8]:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = hf.generate(ids, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=None)
        engine = Engine(model, PagedKVCache(model.cfg, 8, device=device, dtype=torch.bfloat16), EngineConfig(max_batch=1))
        request = Request("r", ids[0].tolist(), max_new_tokens=16)
        engine.add_request(request)
        engine.run_until_idle()
        assert request.generated == out[0, ids.shape[1] :].tolist(), text


@pytest.mark.gpu
def test_gpu_per_layer_parity_runs_on_cuda(real_pair):
    assert real_pair[3] == "cuda"
    test_real_weights_per_layer_outputs_match_transformers(real_pair)

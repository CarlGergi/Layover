"""The HTTP API over a real socket, on the micro model and on real Qwen3-0.6B weights."""

import json
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig, AgentScheduler
from layover.config import ModelConfig
from layover.engine import Engine, EngineConfig, Request
from layover.kv import PagedKVCache
from layover.model import Qwen3
from layover.kernels.fp8 import Fp8Layout
from layover.kv.transfer import NativeBackend
from layover.server import EngineLoop, HFTokenizer, cpu_tier_capacity_blocks, create_app, parse_tool_calls

CFG = ModelConfig.micro()
COSTS = CostModel([0.001, 5e-6], [0.001, 5e-6], [0.005, 2e-5, 2e-9])


class ByteTokenizer:
    """One token per byte, so tests need no tokenizer files."""

    stop_tokens = frozenset()

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(i % 256 for i in ids).decode("latin-1")

    def apply_chat(self, messages, tools, **_):
        rendered = "".join(f"{m['role']}:{m.get('content') or ''}\n" for m in messages)
        return self.encode(rendered + "assistant:")


class Served:
    def __init__(self, model, tokenizer, name, blocks=128, dtype=torch.float32, agent=AgentConfig(), transfer=None):
        cache = PagedKVCache(model.cfg, blocks, dtype=dtype)
        self.scheduler = AgentScheduler(model, cache, COSTS, EngineConfig(max_batch=4, prefill_chunk=32, prefill_budget=64), agent, transfer=transfer)
        self.loop = EngineLoop(self.scheduler)
        self.loop.start()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = uvicorn.Config(create_app(self.loop, tokenizer, name), host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.02)
        self.client = httpx.Client(base_url=self.url, timeout=120)

    def close(self):
        self.client.close()
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.loop.stop()


@pytest.fixture(scope="module")
def micro_model():
    torch.manual_seed(0)
    return Qwen3(CFG).float().eval()


@pytest.fixture()
def api(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro")
    yield served
    served.close()


def solo(model, prompt, new):
    engine = Engine(model, PagedKVCache(CFG, 64, dtype=torch.float32))
    request = Request("solo", list(prompt), max_new_tokens=new)
    engine.add_request(request)
    engine.run_until_idle()
    return request.generated


def test_completion_with_token_ids_matches_the_engine(api, micro_model):
    reply = api.client.post("/v1/completions", json={"model": "micro", "prompt": list(range(40)), "max_tokens": 5}).json()
    choice = reply["choices"][0]
    assert choice["token_ids"] == solo(micro_model, range(40), 5)
    assert choice["finish_reason"] == "length"
    assert reply["usage"] == {
        "prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_streaming_delivers_the_same_tokens_then_done(api):
    prompt = {"model": "micro", "prompt": list(range(20)), "max_tokens": 6}
    whole = api.client.post("/v1/completions", json=prompt).json()["choices"][0]["token_ids"]
    streamed, lines = [], []
    with api.client.stream("POST", "/v1/completions", json={**prompt, "stream": True}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                lines.append(line[6:])
    assert lines[-1] == "[DONE]"
    for raw in lines[:-1]:
        streamed.extend(json.loads(raw)["choices"][0]["token_ids"])
    assert streamed == whole
    assert json.loads(lines[-2])["choices"][0]["finish_reason"] == "length"


def test_chat_response_shape(api):
    reply = api.client.post(
        "/v1/chat/completions",
        json={"model": "micro", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0.7},
    ).json()
    assert reply["object"] == "chat.completion"
    assert reply["choices"][0]["message"]["role"] == "assistant"
    assert reply["usage"]["completion_tokens"] == 3


def test_session_resume_reuses_kv_and_hints_reach_the_estimator(api, micro_model):
    first = api.client.post(
        "/v1/completions",
        json={"prompt": list(range(40)), "max_tokens": 4, "session_id": "s1", "layover": {"tool_name": "grep", "program_id": "job"}},
    ).json()
    ids = first["choices"][0]["token_ids"]
    time.sleep(0.05)
    next_prompt = list(range(40)) + ids + [9, 9, 9]
    second = api.client.post(
        "/v1/completions",
        json={"prompt": next_prompt, "max_tokens": 4, "session_id": "s1", "layover": {"end_session": True}},
    ).json()
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 40 + len(ids) - 1
    assert second["choices"][0]["token_ids"] == solo(micro_model, next_prompt, 4)
    assert api.scheduler.estimator.estimate("grep").samples == 1
    assert api.scheduler.programs["job"].attained_tokens > 0


def test_session_header_is_accepted(api):
    api.client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 1}, headers={"X-Session-Id": "h1"})
    assert "h1" in api.client.get("/metrics").json() or "h1" in api.scheduler.sessions


def test_one_off_requests_leave_no_session_behind(api):
    api.client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 1})
    assert not api.scheduler.sessions


def test_concurrent_turn_in_one_session_gets_409(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro")
    try:
        results = []
        def long_call():
            results.append(served.client.post("/v1/completions", json={"prompt": list(range(64)), "max_tokens": 400, "session_id": "busy"}))
        thread = threading.Thread(target=long_call)
        thread.start()
        while "busy" not in served.scheduler.sessions:
            time.sleep(0.001)
        conflict = served.client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1, "session_id": "busy"})
        assert conflict.status_code == 409 and conflict.json()["error"]["type"] == "session_busy"
        thread.join()
        assert results[0].status_code == 200
    finally:
        served.close()


def test_errors_use_the_openai_shape(api):
    too_big = api.client.post("/v1/completions", json={"prompt": list(range(100)), "max_tokens": 100000})
    assert too_big.status_code == 400 and too_big.json()["error"]["type"] == "context_length_exceeded"
    missing = api.client.post("/v1/completions", json={"max_tokens": 1})
    assert missing.status_code == 400
    bad = api.client.post("/v1/completions", json={"prompt": [1, "x"]})
    assert bad.status_code == 400
    empty = api.client.post("/v1/chat/completions", json={"messages": []})
    assert empty.status_code == 400


def test_fork_and_close_endpoints(api):
    api.client.post("/v1/completions", json={"prompt": list(range(40)), "max_tokens": 2, "session_id": "p", "layover": {"tool_name": "t"}})
    assert api.client.post("/v1/sessions/p/fork", json={"child_session_id": "c"}).status_code == 200
    assert api.scheduler.sessions["c"].program_id == "p"
    assert api.client.delete("/v1/sessions/p").status_code == 200
    assert api.client.delete("/v1/sessions/c").status_code == 200
    assert "p" not in api.scheduler.sessions and "c" not in api.scheduler.sessions


def test_metrics_and_models(api):
    assert api.client.get("/v1/models").json()["data"][0]["id"] == "micro"
    metrics = api.client.get("/metrics").json()
    assert metrics["blocks_total"] == 128 and metrics["blocks_free"] <= 128
    assert api.client.get("/health").json() == {"status": "ok"}


def test_cpu_tier_budget_is_bytes_so_fp8_holds_about_twice_the_tokens():
    cache = PagedKVCache(ModelConfig.micro(), 8, dtype=torch.bfloat16)
    gib = 0.5
    bf16 = cpu_tier_capacity_blocks(cache, gib, False)
    fp8 = cpu_tier_capacity_blocks(cache, gib, True)
    record = Fp8Layout.from_cache(cache).record_bytes
    assert bf16 == int(gib * 2**30 / cache.block_bytes)
    # A full tier stays inside its budget and leaves no room for another record.
    assert fp8 * record <= gib * 2**30 < (fp8 + 1) * record
    assert 1.5 * bf16 < fp8 <= 2 * bf16


def test_metrics_show_a_failed_pin_and_the_tier_budget(micro_model):
    backend = NativeBackend("cpu", staging_bytes=1 << 20, ids_capacity=256)
    backend.engine.set_pin_limit(0)
    served = Served(micro_model, ByteTokenizer(), "micro", agent=AgentConfig(policy=Policy.SWAP, cpu_capacity_blocks=1000), transfer=backend)
    try:
        reply = served.client.post("/v1/completions", json={"prompt": list(range(40)), "max_tokens": 2, "session_id": "s"})
        assert reply.status_code == 200
        metrics = served.client.get("/metrics").json()
        assert metrics["pin_failure_drops"] == 1 and metrics["dropped"] == 1 and metrics["swapped"] == 0
        assert metrics["cpu_tier_capacity_blocks"] == 1000 and metrics["cpu_tier_peak_blocks"] == 0
        assert "host_pins" in metrics and "host_pin_ns" in metrics
        # The engine survived: another turn still works.
        again = served.client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 1, "session_id": "t"})
        assert again.status_code == 200
    finally:
        served.close()
        backend.close()


def test_metrics_separate_blocks_only_the_radix_tree_holds(api):
    api.client.post("/v1/completions", json={"prompt": list(range(40)), "max_tokens": 1})
    metrics = api.client.get("/metrics").json()
    # A one-off's KV is gone from its session, so its two whole blocks are the tree's alone.
    assert metrics["radix_only_blocks"] == metrics["radix_blocks"] == 2


def test_openai_sdk_can_talk_to_the_server(api):
    from openai import OpenAI

    client = OpenAI(base_url=api.url + "/v1", api_key="unused")
    reply = client.chat.completions.create(
        model="micro", messages=[{"role": "user", "content": "hi"}], max_tokens=2,
        extra_body={"session_id": "sdk", "layover": {"end_session": True}},
    )
    assert reply.choices[0].finish_reason == "length" and reply.usage.completion_tokens == 2
    chunks = list(client.chat.completions.create(model="micro", messages=[{"role": "user", "content": "hi"}], max_tokens=2, stream=True))
    assert chunks[-1].choices[0].finish_reason == "length"


def failing_decode(scheduler):
    """Raises only once a request reaches decode, so the engine is mid-flight when it dies."""
    original = scheduler._decode

    def explode():
        if any(r.generated for r in scheduler.running):
            raise RuntimeError("boom")
        return original()

    return explode


def test_engine_failure_mid_request_reaches_the_client(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro")
    try:
        served.scheduler._decode = failing_decode(served.scheduler)
        reply = served.client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4})
        assert reply.status_code == 500 and "boom" in reply.json()["error"]["message"]
        assert served.client.get("/health").status_code == 500
        # Requests after the failure get the same answer rather than hanging.
        again = served.client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4})
        assert again.status_code == 500 and again.json()["error"]["type"] == "engine_stopped"
    finally:
        served.close()


def test_engine_failure_during_a_stream_sends_an_error_event(micro_model):
    served = Served(micro_model, ByteTokenizer(), "micro")
    try:
        served.scheduler._decode = failing_decode(served.scheduler)
        with served.client.stream("POST", "/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4, "stream": True}) as response:
            lines = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
        assert lines[-1] == "[DONE]" and "boom" in json.loads(lines[-2])["error"]["message"]
    finally:
        served.close()


def run_main_until_scheduler(monkeypatch, tmp_path, argv):
    """Run the server's real main up to the point it would listen, and return what it built."""
    import layover.server as server

    built = {}
    monkeypatch.setattr("layover.weights.resolve_checkpoint", lambda name: name)
    monkeypatch.setattr("layover.weights.load_model", lambda *a, **k: Qwen3(CFG).bfloat16().eval())
    monkeypatch.setattr(server, "HFTokenizer", lambda root: ByteTokenizer())

    class FakeLoop:
        def __init__(self, scheduler):
            built["scheduler"] = scheduler

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(server, "EngineLoop", FakeLoop)
    monkeypatch.setattr(server, "create_app", lambda *a, **k: None)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({
        "swap_out_s_coef": [0.0, 1e-6], "swap_in_s_coef": [0.0, 1e-6], "recompute_s_coef": [0.0, 1e-5, 0.0],
        "fp8_swap_out_s_coef": [0.0, 5e-7], "fp8_swap_in_s_coef": [0.0, 5e-7],
        "measured_on_gpu": False, "transfer": "plain",
    }))
    server.main(["--model", "x", "--cost-model", str(costs), "--pool-gib", "0.01", *argv])
    return built["scheduler"]


def test_server_defaults_use_the_torch_reference_and_plain_copies(monkeypatch, tmp_path):
    from layover.kernels.decode_attention import paged_decode_attention_torch
    from layover.kv.transfer import PlainCopyBackend

    scheduler = run_main_until_scheduler(monkeypatch, tmp_path, [])
    assert scheduler.model.decode_attention is paged_decode_attention_torch
    assert isinstance(scheduler.tier.backend, PlainCopyBackend)
    assert scheduler.tier.fp8 is False


def test_triton_flag_selects_the_kernel(monkeypatch, tmp_path):
    pytest.importorskip("triton")
    from layover.kernels.triton_decode import paged_decode_attention_triton_trusted

    scheduler = run_main_until_scheduler(monkeypatch, tmp_path, ["--decode-backend", "triton"])
    assert scheduler.model.decode_attention is paged_decode_attention_triton_trusted


def test_offload_dtype_flag_reaches_the_tier(monkeypatch, tmp_path):
    scheduler = run_main_until_scheduler(monkeypatch, tmp_path, ["--offload-dtype", "fp8"])
    assert scheduler.tier.fp8 is True


def test_fp8_swap_times_come_from_the_fp8_measurement(monkeypatch, tmp_path):
    scheduler = run_main_until_scheduler(monkeypatch, tmp_path, ["--offload-dtype", "fp8"])
    assert scheduler.costs.swap_out_s(1000) == pytest.approx(5e-4)


def test_a_cost_model_from_another_transfer_path_is_refused(monkeypatch, tmp_path):
    with pytest.raises((ValueError, SystemExit)):
        run_main_until_scheduler(monkeypatch, tmp_path, ["--transfer", "native"])


def test_parse_tool_calls():
    text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me check."
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_malformed_tool_call_stays_in_the_content():
    text = "<tool_call>{not json}</tool_call>"
    assert parse_tool_calls(text) == (text, [])


def test_real_weights_tool_call_round_trip(checkpoint):
    from layover.weights import load_model, resolve_checkpoint

    root = str(resolve_checkpoint(checkpoint))
    model = load_model(root, dtype=torch.bfloat16)
    served = Served(model, HFTokenizer(root), "qwen", blocks=256, dtype=torch.bfloat16)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    try:
        messages = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
        first = served.client.post(
            "/v1/chat/completions",
            json={
                "messages": messages, "tools": tools, "max_tokens": 64, "session_id": "agent",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ).json()
        choice = first["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"])["city"] == "Paris"
        # The tool name came from the output, so the pause is attributed to it.
        assert served.scheduler.sessions["agent"].tool_name == "get_weather"

        messages += [choice["message"], {"role": "tool", "content": "18 C and sunny"}]
        second = served.client.post(
            "/v1/chat/completions",
            json={
                "messages": messages, "tools": tools, "max_tokens": 48, "session_id": "agent",
                "chat_template_kwargs": {"enable_thinking": False}, "layover": {"end_session": True},
            },
        ).json()
        assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > first["usage"]["prompt_tokens"] / 2
        assert "18" in second["choices"][0]["message"]["content"]
    finally:
        served.close()


def test_cuda_graphs_are_refused_where_they_cannot_run(monkeypatch, tmp_path):
    """The torch decode backend reads lengths back, which a graph cannot capture, and a CPU has none."""
    with pytest.raises(SystemExit, match="--cuda-graphs needs"):
        run_main_until_scheduler(monkeypatch, tmp_path, ["--cuda-graphs"])

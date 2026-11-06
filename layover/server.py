"""OpenAI-compatible HTTP API with session ids and tool-call boundary hints.

Beyond the OpenAI request fields, a client may send:

    session_id: ties turns of one agent together (also the X-Session-Id header)
    layover: {
        "program_id":      groups sessions of one job, for program-level priority
        "tool_name":       the tool this turn ends on, when the client knows it
        "expected_tool_s": the client's estimate of that tool's duration
        "end_session":     this is the session's last turn; free its KV
    }

Without a session_id a request is one-off. Sampling is greedy, so temperature and
top_p are accepted for compatibility and have no effect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict

import torch
from fastapi import FastAPI, Request as HttpRequest
from fastapi.responses import JSONResponse, StreamingResponse

from layover.agent.scheduler import AgentScheduler, AgentTurn, SessionBusy
from layover.engine import RequestTooLarge
from layover.kernels.fp8 import Fp8Layout

log = logging.getLogger("layover")

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split Qwen3 output into plain content and OpenAI-shaped tool calls.

    A block that is not valid JSON stays in the content, so nothing the model said is
    lost and the client sees exactly why no call was made.
    """
    calls = []

    def take(match: re.Match) -> str:
        try:
            parsed = json.loads(match.group(1))
            name = parsed["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return match.group(0)
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(parsed.get("arguments", {}))},
            }
        )
        return ""

    return TOOL_CALL.sub(take, text).strip(), calls


class EngineStopped(RuntimeError):
    """The engine thread has died; no request can be served."""


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.inner = AutoTokenizer.from_pretrained(name_or_path)
        self.stop_tokens = frozenset({self.inner.eos_token_id} - {None})

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=True)

    def apply_chat(self, messages: list[dict], tools: list[dict] | None, **template_kwargs) -> list[int]:
        text = self.inner.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        return self.encode(text)


class EngineLoop(threading.Thread):
    """Owns the scheduler: every mutation runs here, between steps."""

    def __init__(self, scheduler: AgentScheduler, idle_sleep_s: float = 0.002):
        super().__init__(daemon=True, name="layover-engine")
        self.scheduler = scheduler
        self.idle_sleep_s = idle_sleep_s
        self.commands: queue.Queue = queue.Queue()
        self.stopping = threading.Event()
        self.fatal: BaseException | None = None

    def call(self, fn: Callable, *args) -> Future:
        future: Future = Future()
        self.commands.put((future, fn, args))
        if self.fatal is not None:
            self._fail_commands()
        return future

    def stop(self) -> None:
        self.stopping.set()
        self.join(timeout=10)

    def _run_commands(self) -> None:
        while True:
            try:
                future, fn, args = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(fn(*args))
            except Exception as exc:  # Delivered to the caller, which maps it to HTTP.
                future.set_exception(exc)

    def _fail_commands(self) -> None:
        while True:
            try:
                future, _, _ = self.commands.get_nowait()
            except queue.Empty:
                return
            future.set_exception(EngineStopped(f"engine stopped: {self.fatal}"))

    def _fail_all(self) -> None:
        """Wake every waiting client instead of leaving them hanging on a dead engine."""
        pending = list(self.scheduler.running) + list(self.scheduler.waiting)
        for turn in pending:
            turn.error = str(self.fatal)
            if turn.on_finish is not None:
                turn.on_finish(turn)
        self._fail_commands()

    def run(self) -> None:
        try:
            while not self.stopping.is_set():
                self._run_commands()
                if not self.scheduler.step():
                    time.sleep(self.idle_sleep_s)
        except BaseException as exc:
            self.fatal = exc
            log.exception("engine loop stopped")
            self._fail_all()


def cpu_tier_capacity_blocks(cache, gib: float, fp8: bool) -> int:
    """How many blocks a tier of this size holds, by what its records really occupy.

    The budget is bytes, so FP8's smaller records fit about twice the tokens of BF16's.
    """
    record_bytes = Fp8Layout.from_cache(cache).record_bytes if fp8 else cache.block_bytes
    return int(gib * 2**30 / record_bytes)


def error_response(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def create_app(loop: EngineLoop, tokenizer, model_name: str, default_max_tokens: int = 256) -> FastAPI:
    app = FastAPI(title="Layover")

    def detect_tool(ids: list[int]) -> str | None:
        _, calls = parse_tool_calls(tokenizer.decode(ids))
        return calls[0]["function"]["name"] if calls else None

    async def start_turn(body: dict, http: HttpRequest, prompt: list[int], events: asyncio.Queue) -> AgentTurn:
        event_loop = asyncio.get_running_loop()
        hints = body.get("layover") or {}
        session_id = body.get("session_id") or http.headers.get("x-session-id")
        one_off = session_id is None
        session_id = session_id or f"oneoff-{uuid.uuid4().hex}"
        ignore_eos = bool(body.get("ignore_eos"))

        def on_token(_, token: int) -> None:
            event_loop.call_soon_threadsafe(events.put_nowait, ("token", token))

        def on_finish(_) -> None:
            event_loop.call_soon_threadsafe(events.put_nowait, ("finish", None))

        turn = AgentTurn(
            request_id=f"cmpl-{uuid.uuid4().hex}",
            prompt=prompt,
            max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens),
            stop_tokens=frozenset() if ignore_eos else tokenizer.stop_tokens,
            on_token=on_token,
            on_finish=on_finish,
            session_id=session_id,
            program_id=hints.get("program_id") or session_id,
            tool_name=hints.get("tool_name"),
            expected_tool_s=hints.get("expected_tool_s"),
            end_session=one_off or bool(hints.get("end_session")),
            detect_tool=detect_tool,
        )
        await asyncio.wrap_future(loop.call(loop.scheduler.submit, turn))
        return turn

    def finish_reason(turn: AgentTurn, calls: list[dict] | None = None) -> str:
        if calls:
            return "tool_calls"
        return "stop" if turn.generated and turn.generated[-1] in turn.stop_tokens else "length"

    def usage(turn: AgentTurn) -> dict:
        return {
            "prompt_tokens": len(turn.prompt),
            "completion_tokens": len(turn.generated),
            "total_tokens": len(turn.prompt) + len(turn.generated),
            "prompt_tokens_details": {"cached_tokens": turn.reused_tokens},
        }

    async def collect(events: asyncio.Queue) -> None:
        while (await events.get())[0] != "finish":
            pass

    async def checked(body_fn):
        try:
            return await body_fn()
        except EngineStopped as exc:
            return error_response(500, str(exc), "engine_stopped")
        except SessionBusy as exc:
            return error_response(409, f"session {exc} already has a turn in flight", "session_busy")
        except RequestTooLarge as exc:
            return error_response(400, str(exc), "context_length_exceeded")
        except (ValueError, TypeError, KeyError) as exc:
            return error_response(400, f"invalid request: {exc}", "invalid_request_error")

    def sse(payload: dict | str) -> str:
        return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"

    @app.get("/health")
    async def health():
        if loop.fatal is not None:
            return error_response(500, str(loop.fatal), "engine_stopped")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "layover"}]}

    @app.get("/metrics")
    async def metrics():
        def read():
            scheduler = loop.scheduler
            return {
                **asdict(scheduler.stats),
                "blocks_total": scheduler.cache.allocator.num_blocks,
                "blocks_free": scheduler.cache.allocator.num_free,
                "radix_blocks": scheduler.tree.num_blocks,
                # Evictable on demand, so a comparison of pool use leaves these out.
                "radix_only_blocks": sum(scheduler.cache.allocator.refcount(b) == 1 for b in scheduler.tree.block_ids()),
                "cpu_tier_blocks": scheduler.tier.used_blocks,
                "cpu_tier_capacity_blocks": scheduler.tier.capacity_blocks,
                "cpu_tier_peak_blocks": scheduler.tier.peak_used_blocks,
                "cpu_tier_bytes_moved": scheduler.tier.bytes_moved,
                "waiting": len(scheduler.waiting),
                "running": len(scheduler.running),
            }

        return await asyncio.wrap_future(loop.call(read))

    @app.post("/v1/sessions/{session_id:path}/fork")
    async def fork(session_id: str, http: HttpRequest):
        body = await http.json()

        async def run():
            child = body["child_session_id"]
            await asyncio.wrap_future(loop.call(loop.scheduler.fork_session, session_id, child))
            return {"session_id": child, "forked_from": session_id}

        return await checked(run)

    @app.delete("/v1/sessions/{session_id:path}")
    async def close(session_id: str):
        async def run():
            await asyncio.wrap_future(loop.call(loop.scheduler.close_session, session_id))
            return {"session_id": session_id, "closed": True}

        return await checked(run)

    @app.post("/v1/completions")
    async def completions(http: HttpRequest):
        body = await http.json()

        async def run():
            prompt = body["prompt"]
            if isinstance(prompt, str):
                prompt = tokenizer.encode(prompt)
            if not (isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)):
                raise ValueError("prompt must be a string or a list of token ids")
            events: asyncio.Queue = asyncio.Queue()
            turn = await start_turn(body, http, prompt, events)
            created = int(time.time())

            if not body.get("stream"):
                await collect(events)
                if turn.error:
                    return error_response(500, turn.error, "engine_stopped")
                return {
                    "id": turn.request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": tokenizer.decode(turn.generated),
                            "token_ids": turn.generated,
                            "finish_reason": finish_reason(turn),
                        }
                    ],
                    "usage": usage(turn),
                }

            async def stream():
                while True:
                    kind, token = await events.get()
                    if kind == "finish":
                        break
                    choice = {"index": 0, "text": tokenizer.decode([token]), "token_ids": [token], "finish_reason": None}
                    yield sse({"id": turn.request_id, "object": "text_completion", "created": created, "model": model_name, "choices": [choice]})
                if turn.error:
                    yield sse({"error": {"message": turn.error, "type": "engine_stopped"}})
                    yield sse("[DONE]")
                    return
                last = {"index": 0, "text": "", "token_ids": [], "finish_reason": finish_reason(turn)}
                yield sse({"id": turn.request_id, "object": "text_completion", "created": created, "model": model_name, "choices": [last], "usage": usage(turn)})
                yield sse("[DONE]")

            return StreamingResponse(stream(), media_type="text/event-stream")

        return await checked(run)

    @app.post("/v1/chat/completions")
    async def chat(http: HttpRequest):
        body = await http.json()

        async def run():
            if not body.get("messages"):
                raise ValueError("messages must not be empty")
            prompt = tokenizer.apply_chat(body["messages"], body.get("tools"), **(body.get("chat_template_kwargs") or {}))
            events: asyncio.Queue = asyncio.Queue()
            turn = await start_turn(body, http, prompt, events)
            created = int(time.time())

            if not body.get("stream"):
                await collect(events)
                if turn.error:
                    return error_response(500, turn.error, "engine_stopped")
                content, calls = parse_tool_calls(tokenizer.decode(turn.generated))
                message = {"role": "assistant", "content": content or None}
                if calls:
                    message["tool_calls"] = calls
                return {
                    "id": turn.request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(turn, calls)}],
                    "usage": usage(turn),
                }

            async def stream():
                base = {"id": turn.request_id, "object": "chat.completion.chunk", "created": created, "model": model_name}
                yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                emitted = ""
                while True:
                    kind, _ = await events.get()
                    if kind == "finish":
                        break
                    # Decoding the whole output keeps multi-token characters intact.
                    text = tokenizer.decode(turn.generated)
                    if text.endswith("�"):
                        continue
                    delta, emitted = text[len(emitted) :], text
                    if delta:
                        yield sse({**base, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
                if turn.error:
                    yield sse({"error": {"message": turn.error, "type": "engine_stopped"}})
                    yield sse("[DONE]")
                    return
                _, calls = parse_tool_calls(tokenizer.decode(turn.generated))
                delta = {}
                if calls:
                    delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(calls)]
                yield sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason(turn, calls)}], "usage": usage(turn)})
                yield sse("[DONE]")

            return StreamingResponse(stream(), media_type="text/event-stream")

        return await checked(run)

    return app


def main(argv=None) -> None:
    import uvicorn

    from layover.agent.cost_model import CostModel, Policy
    from layover.agent.scheduler import AgentConfig
    from layover.engine import EngineConfig
    from layover.kv.paged_cache import PagedKVCache
    from layover.kv.transfer import make_backend
    from layover.weights import load_model, resolve_checkpoint

    p = argparse.ArgumentParser(description="Serve a model over the OpenAI-compatible API")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--pool-gib", type=float, default=2.0, help="KV pool size")
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--policy", default="combined", choices=[x.value for x in Policy])
    p.add_argument("--cost-model", required=True, help="cost_model.json from bench/kv_crossover.py")
    p.add_argument("--ttl-cap-s", type=float, default=30.0)
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--cpu-tier-gib", type=float, default=32.0)
    p.add_argument("--decode-backend", default="torch", choices=["torch", "triton"], help="triton needs a Linux install of it")
    p.add_argument("--transfer", default="plain", choices=["plain", "native"], help="native needs make -C native cuda or host")
    p.add_argument("--offload-dtype", default="bf16", choices=["bf16", "fp8"], help="fp8 needs triton")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--prefill-chunk", type=int, default=512)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    root = str(resolve_checkpoint(args.model))
    model = load_model(root, device=args.device, dtype=dtype)
    if args.decode_backend == "triton":
        from layover.kernels.triton_decode import paged_decode_attention_triton

        model.decode_attention = paged_decode_attention_triton
    token_bytes = model.cfg.kv_bytes_per_token(args.dtype)
    block_bytes = token_bytes * args.block_tokens
    cache = PagedKVCache(model.cfg, int(args.pool_gib * 2**30 / block_bytes), args.device, dtype, args.block_tokens)
    agent = AgentConfig(
        policy=Policy(args.policy),
        ttl_cap_s=args.ttl_cap_s,
        prefix_cache=not args.no_prefix_cache,
        cpu_capacity_blocks=cpu_tier_capacity_blocks(cache, args.cpu_tier_gib, args.offload_dtype == "fp8"),
    )
    costs = CostModel.from_json(args.cost_model, require_gpu=args.device == "cuda", transfer=args.transfer)
    engine = EngineConfig(args.max_batch, args.prefill_chunk, args.prefill_chunk)
    transfer = make_backend(args.transfer, args.device)
    scheduler = AgentScheduler(model, cache, costs, engine, agent, transfer=transfer, offload_fp8=args.offload_dtype == "fp8")
    loop = EngineLoop(scheduler)
    loop.start()
    try:
        uvicorn.run(create_app(loop, HFTokenizer(root), args.model), host=args.host, port=args.port)
    finally:
        loop.stop()


if __name__ == "__main__":
    main()

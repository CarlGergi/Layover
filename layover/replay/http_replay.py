"""Replay a trace against any OpenAI-compatible server.

One driver serves every system under test, so all of them see the same prompts and the
same pauses. Prompts are token ids sent to /v1/completions, which is standard OpenAI API.
session_id and layover fields go only to Layover: session_id is a declared request field
in vLLM 0.33.0 and SGLang 0.5.23 with meanings of their own, so hints=False omits it.

Generated ids come back through the return_token_ids request field, which vLLM 0.33.0 and
SGLang 0.5.23 both answer with a token_ids delta on each stream chunk. Their first chunk
may also carry prompt_token_ids and their last, with stream_options.include_usage, may
have no choices at all.
"""

from __future__ import annotations

import asyncio
import json
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from layover.replay.results import ReplayResult, TurnRecord, summarize
from layover.replay.trace import ContextBuilder, TraceJob, TraceSession


@dataclass
class TurnOutput:
    tokens: list[int]
    text: str
    first_s: float
    end_s: float
    usage: dict = field(default_factory=dict)


# Finish reasons that mean the server gave up on the request.
FAILED_FINISH = ("abort", "error", "content_filter")


async def _run_turn(client, url, payload) -> TurnOutput:
    """Stream one completion; the first token is the first chunk with text or ids."""
    tokens: list[int] = []
    text = []
    first = None
    usage: dict = {}
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise RuntimeError(f"{response.status_code} from {url}: {body[:400]!r}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            # Before parsing: a first chunk carrying the whole prompt's ids takes milliseconds
            # to parse, which is the client's time and not the server's.
            received = time.perf_counter()
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"server error: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                piece = choice.get("text") or ""
                ids = choice.get("token_ids") or []
                if (piece or ids) and first is None:
                    first = received
                text.append(piece)
                tokens.extend(ids)
                if choice.get("finish_reason") in FAILED_FINISH:
                    raise RuntimeError(f"server ended a turn with finish_reason={choice['finish_reason']}")
    end = time.perf_counter()
    return TurnOutput(tokens, "".join(text), first if first is not None else end, end, usage)


def _cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def job_tool_s(job: TraceJob) -> float:
    """Tool time on the job's longest chain of pauses: sub-agents run beside their parent."""

    def chain(session: TraceSession, before: float) -> float:
        longest = before + sum(turn.tool_s for turn in session.turns[:-1])
        elapsed = before
        for index, turn in enumerate(session.turns):
            for child in job.children_of(session.session_id, index):
                longest = max(longest, chain(child, elapsed))
            elapsed += turn.tool_s
        return longest

    return chain(job.main, 0.0)


async def replay(
    jobs: list[TraceJob],
    base_url: str,
    model: str,
    vocab_size: int,
    system: str = "layover",
    time_scale: float = 1.0,
    warmup_jobs: int = 0,
    seed: int = 0,
    system_tokens: int = 256,
    hints: bool = True,
    encode: Callable[[str], list[int]] | None = None,
    timeout_s: float = 3600.0,
) -> ReplayResult:
    """Run every job to completion. Arrivals and tool pauses are multiplied by time_scale.

    A server that does not return token ids needs `encode`, so the next prompt holds what
    the model actually generated; without that, its KV could never match. Each turn
    records which of the two happened. Re-encoding text is not exact, so such a run's
    contexts can drift from the trace's token counts, and that shows in its turn records.
    """
    builder = ContextBuilder(vocab_size, seed, system_tokens)
    # The same seed makes a shorter shared prefix a prefix of a longer one.
    prefixes: dict[int, list[int]] = {}

    def initial_context(job: TraceJob) -> list[int]:
        if job.shared_prefix_tokens is None:
            return list(builder.system_prompt())
        if job.shared_prefix_tokens not in prefixes:
            prefixes[job.shared_prefix_tokens] = ContextBuilder(vocab_size, seed, job.shared_prefix_tokens).system_prompt()
        return list(prefixes[job.shared_prefix_tokens])

    url = base_url.rstrip("/") + "/v1/completions"
    records: list[TurnRecord] = []
    arrivals: dict[str, float] = {}
    finishes: dict[str, float] = {}

    async with httpx.AsyncClient(timeout=timeout_s, limits=httpx.Limits(max_connections=None)) as client:
        start = time.perf_counter()

        async def run_session(job: TraceJob, session: TraceSession, context: list[int]) -> None:
            children = []
            for index, spec in enumerate(session.turns):
                context = context + builder.new_tokens(session.session_id, index, spec.new_tokens)
                last = index == len(session.turns) - 1
                has_children = bool(job.children_of(session.session_id, index))
                payload = {
                    "model": model,
                    "prompt": context,
                    "max_tokens": spec.output_tokens,
                    "temperature": 0,
                    "top_p": 1.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
                if hints:
                    payload["session_id"] = session.session_id
                    payload["layover"] = {
                        "program_id": job.job_id,
                        "tool_name": None if last else spec.tool_name,
                        # A session with children must keep its KV long enough to fork.
                        "end_session": last and not has_children,
                    }
                submitted = time.perf_counter()
                output = await _run_turn(client, url, payload)
                tokens = output.tokens
                token_source = "server"
                if not tokens:
                    if encode is None:
                        raise RuntimeError("server returned no token ids; pass an encode function")
                    tokens = encode(output.text)
                    token_source = "tokenizer"
                records.append(
                    TurnRecord(
                        job.job_id, session.session_id, index, submitted - start, output.first_s - start,
                        output.end_s - start, prompt_tokens=len(context), output_tokens_requested=spec.output_tokens,
                        output_tokens=len(tokens), token_source=token_source,
                        cached_tokens=_cached_tokens(output.usage),
                        output_crc=zlib.crc32(json.dumps(tokens).encode()),
                    )
                )
                context = context + tokens
                for child in job.children_of(session.session_id, index):
                    if hints:
                        response = await client.post(
                            f"{base_url.rstrip('/')}/v1/sessions/{session.session_id}/fork",
                            json={"child_session_id": child.session_id},
                        )
                        response.raise_for_status()
                    children.append(asyncio.create_task(run_session(job, child, list(context))))
                if not last:
                    await asyncio.sleep(spec.tool_s * time_scale)
            if hints and has_children:
                await client.delete(f"{base_url.rstrip('/')}/v1/sessions/{session.session_id}")
            await asyncio.gather(*children)

        async def run_job(job: TraceJob) -> None:
            await asyncio.sleep(job.arrival_s * time_scale)
            arrivals[job.job_id] = time.perf_counter() - start
            await run_session(job, job.main, initial_context(job))
            finishes[job.job_id] = time.perf_counter() - start

        await asyncio.gather(*(run_job(job) for job in jobs))

    planned = [job.job_id for job in sorted(jobs, key=lambda job: job.arrival_s)]
    tool_s = {job.job_id: job_tool_s(job) * time_scale for job in jobs}
    return summarize(jobs[0].source, system, arrivals, finishes, records, warmup_jobs, planned, start, tool_s)

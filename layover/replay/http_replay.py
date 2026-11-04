"""Replay a trace against any OpenAI-compatible server.

One driver serves every system under test, so all of them see the same prompts and the
same pauses. Prompts are token ids sent to /v1/completions, which is standard OpenAI
API; session_id and layover fields are extra body fields other servers ignore.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import httpx

from layover.replay.results import ReplayResult, TurnRecord, summarize
from layover.replay.trace import ContextBuilder, TraceJob, TraceSession


async def _run_turn(client, url, payload) -> tuple[list[int], str, float, float]:
    """Stream one completion; returns token ids, text, and first-token and end times."""
    tokens: list[int] = []
    text = []
    first = None
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
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"server error: {event['error']}")
            choice = event["choices"][0]
            piece = choice.get("text") or ""
            ids = choice.get("token_ids") or []
            if (piece or ids) and first is None:
                first = time.perf_counter()
            text.append(piece)
            tokens.extend(ids)
    end = time.perf_counter()
    return tokens, "".join(text), first if first is not None else end, end


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
    the model actually generated; without that, its KV could never match.
    """
    builder = ContextBuilder(vocab_size, seed, system_tokens)
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
                    "stream": True,
                    "ignore_eos": True,
                    "session_id": session.session_id,
                }
                if hints:
                    payload["layover"] = {
                        "program_id": job.job_id,
                        "tool_name": None if last else spec.tool_name,
                        # A session with children must keep its KV long enough to fork.
                        "end_session": last and not has_children,
                    }
                submitted = time.perf_counter()
                tokens, text, first, end = await _run_turn(client, url, payload)
                if not tokens:
                    if encode is None:
                        raise RuntimeError("server returned no token ids; pass an encode function")
                    tokens = encode(text)
                records.append(
                    TurnRecord(job.job_id, session.session_id, index, submitted - start, first - start, end - start)
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
            await run_session(job, job.main, list(builder.system_prompt()))
            finishes[job.job_id] = time.perf_counter() - start

        await asyncio.gather(*(run_job(job) for job in jobs))

    return summarize(jobs[0].source, system, arrivals, finishes, records, warmup_jobs)

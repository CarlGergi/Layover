"""Metrics for a replay, shared by every driver."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field


@dataclass
class TurnRecord:
    job_id: str
    session_id: str
    turn: int
    submitted_s: float
    first_token_s: float
    finished_s: float
    queue_s: float | None = None
    prompt_tokens: int = 0
    # What the trace asked for and what came back. ignore_eos should keep them equal, on
    # every system; report.py counts the turns where it did not.
    output_tokens_requested: int = 0
    output_tokens: int = 0
    # "server" when the system returned ids, "tokenizer" when the text was re-encoded.
    token_source: str = "server"
    # The server's own prefix-cache count, when it reports one.
    cached_tokens: int | None = None
    # Compares generated ids between systems without keeping the ids.
    output_crc: int | None = None


@dataclass
class ReplayResult:
    source: str
    system: str
    jobs_total: int
    jobs_measured: int
    mean_jct_s: float
    p95_jct_s: float
    programs_per_hour: float
    mean_first_token_s: float
    p95_first_token_s: float = 0.0
    token_sources: dict[str, int] = field(default_factory=dict)
    output_count_mismatches: int = 0
    prompt_tokens_total: int = 0
    output_tokens_total: int = 0
    cached_tokens_total: int | None = None
    jct_s: dict[str, float] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    # What the system controls: JCT less the tool time every system pays alike. None when
    # the driver did not say how long each job's tools take.
    mean_jct_minus_tool_s: float | None = None
    p95_jct_minus_tool_s: float | None = None
    # When the measured jobs started and ended, relative to the start of the replay. A GPU
    # sampler reads only this window, which still holds any warm-up job running inside it.
    window_start_s: float = 0.0
    window_end_s: float = 0.0
    # The driver's perf_counter at the start of the replay.
    clock_origin_s: float = 0.0

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("turns")
        data.pop("jct_s")
        return data


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(-(-fraction * len(ordered) // 1)) - 1))
    return ordered[rank]


def summarize(
    source: str,
    system: str,
    arrivals: dict[str, float],
    finishes: dict[str, float],
    turns: list[TurnRecord],
    warmup_jobs: int = 0,
    planned_order: list[str] | None = None,
    clock_origin_s: float = 0.0,
    tool_s: dict[str, float] | None = None,
) -> ReplayResult:
    """Build the result, leaving out the first warmup_jobs jobs by arrival.

    planned_order is the trace's arrival order. Observed arrivals can swap when two jobs
    arrive together, which would measure different jobs on different systems. tool_s is
    each job's tool time, for the figures that leave it out.
    """
    ordered = planned_order if planned_order is not None else sorted(arrivals, key=arrivals.get)
    measured = ordered[warmup_jobs:]
    if not measured:
        raise ValueError("warm-up excluded every job")
    jct = {job: finishes[job] - arrivals[job] for job in measured}
    window_start, window_end = min(arrivals[j] for j in measured), max(finishes[j] for j in measured)
    window = window_end - window_start
    kept = set(measured)
    kept_turns = [t for t in turns if t.job_id in kept]
    first_tokens = [t.first_token_s - t.submitted_s for t in kept_turns]
    sources: dict[str, int] = {}
    for turn in kept_turns:
        sources[turn.token_source] = sources.get(turn.token_source, 0) + 1
    cached = [t.cached_tokens for t in kept_turns if t.cached_tokens is not None]
    without_tools = [jct[job] - tool_s[job] for job in measured] if tool_s is not None else None
    return ReplayResult(
        source=source,
        system=system,
        jobs_total=len(ordered),
        jobs_measured=len(measured),
        mean_jct_s=statistics.fmean(jct.values()),
        p95_jct_s=percentile(list(jct.values()), 0.95),
        programs_per_hour=len(measured) / window * 3600 if window > 0 else float("inf"),
        mean_first_token_s=statistics.fmean(first_tokens) if first_tokens else 0.0,
        p95_first_token_s=percentile(first_tokens, 0.95) if first_tokens else 0.0,
        token_sources=sources,
        output_count_mismatches=sum(t.output_tokens != t.output_tokens_requested for t in kept_turns),
        prompt_tokens_total=sum(t.prompt_tokens for t in kept_turns),
        output_tokens_total=sum(t.output_tokens for t in kept_turns),
        cached_tokens_total=sum(cached) if cached else None,
        jct_s=jct,
        turns=kept_turns,
        mean_jct_minus_tool_s=statistics.fmean(without_tools) if without_tools else None,
        p95_jct_minus_tool_s=percentile(without_tools, 0.95) if without_tools else None,
        window_start_s=window_start,
        window_end_s=window_end,
        clock_origin_s=clock_origin_s,
    )

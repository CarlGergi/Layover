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
    jct_s: dict[str, float] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)

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
) -> ReplayResult:
    """Build the result, leaving out the first warmup_jobs jobs by arrival."""
    ordered = sorted(arrivals, key=arrivals.get)
    measured = ordered[warmup_jobs:]
    if not measured:
        raise ValueError("warm-up excluded every job")
    jct = {job: finishes[job] - arrivals[job] for job in measured}
    window = max(finishes[j] for j in measured) - min(arrivals[j] for j in measured)
    kept = set(measured)
    first_tokens = [t.first_token_s - t.submitted_s for t in turns if t.job_id in kept]
    return ReplayResult(
        source=source,
        system=system,
        jobs_total=len(ordered),
        jobs_measured=len(measured),
        mean_jct_s=statistics.fmean(jct.values()),
        p95_jct_s=percentile(list(jct.values()), 0.95),
        programs_per_hour=len(measured) / window * 3600 if window > 0 else float("inf"),
        mean_first_token_s=statistics.fmean(first_tokens) if first_tokens else 0.0,
        jct_s=jct,
        turns=[t for t in turns if t.job_id in kept],
    )

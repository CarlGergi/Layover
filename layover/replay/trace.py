"""Agent workload traces: the canonical JSONL format and how to obtain one.

One JSON object per line describes a job (an agent program):

    {"job_id": "j0", "arrival_s": 0.0, "source": "...", "sessions": [
        {"session_id": "j0/main", "parent": null, "fork_after_turn": null, "turns": [
            {"new_tokens": 900, "output_tokens": 120, "tool_name": "bash", "tool_s": 0.8}]}]}

new_tokens is what the client appends to the context before that turn (a tool result,
or the initial prompt). tool_s is the pause after the turn; the last turn's is ignored.
A child session starts when its parent finishes turn fork_after_turn, with the parent's
context, which is how parallel sub-agents appear.

The source field travels with every result, so a synthetic trace can never be mistaken
for a recorded one.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TraceTurn:
    new_tokens: int
    output_tokens: int
    tool_name: str | None = None
    tool_s: float = 0.0


@dataclass
class TraceSession:
    session_id: str
    turns: list[TraceTurn]
    parent: str | None = None
    fork_after_turn: int | None = None


@dataclass
class TraceJob:
    job_id: str
    arrival_s: float
    sessions: list[TraceSession]
    source: str = "unknown"

    @property
    def main(self) -> TraceSession:
        return next(s for s in self.sessions if s.parent is None)

    def children_of(self, session_id: str, turn: int) -> list[TraceSession]:
        return [s for s in self.sessions if s.parent == session_id and s.fork_after_turn == turn]


def validate(job: TraceJob) -> None:
    roots = [s for s in job.sessions if s.parent is None]
    if len(roots) != 1:
        raise ValueError(f"{job.job_id}: expected one root session, found {len(roots)}")
    by_id = {s.session_id: s for s in job.sessions}
    if len(by_id) != len(job.sessions):
        raise ValueError(f"{job.job_id}: duplicate session ids")
    for session in job.sessions:
        if not session.turns:
            raise ValueError(f"{session.session_id}: no turns")
        if any(t.new_tokens < 0 or t.output_tokens < 1 or t.tool_s < 0 for t in session.turns):
            raise ValueError(f"{session.session_id}: invalid turn")
        if session.turns[0].new_tokens < 1 and session.parent is None:
            raise ValueError(f"{session.session_id}: first turn needs a prompt")
        if session.parent is not None:
            parent = by_id.get(session.parent)
            if parent is None:
                raise ValueError(f"{session.session_id}: unknown parent {session.parent}")
            if session.fork_after_turn is None or not 0 <= session.fork_after_turn < len(parent.turns):
                raise ValueError(f"{session.session_id}: fork_after_turn out of range")


def job_from_dict(raw: dict) -> TraceJob:
    sessions = [
        TraceSession(
            s["session_id"],
            [TraceTurn(**t) for t in s["turns"]],
            s.get("parent"),
            s.get("fork_after_turn"),
        )
        for s in raw["sessions"]
    ]
    job = TraceJob(raw["job_id"], float(raw["arrival_s"]), sessions, raw.get("source", "unknown"))
    validate(job)
    return job


def load_jsonl(path: str | Path) -> list[TraceJob]:
    with Path(path).open() as f:
        jobs = [job_from_dict(json.loads(line)) for line in f if line.strip()]
    return sorted(jobs, key=lambda j: j.arrival_s)


def save_jsonl(jobs: Iterable[TraceJob], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for job in jobs:
            f.write(json.dumps(asdict(job)) + "\n")


def from_mini_swe_agent(
    raw: dict,
    count_tokens: Callable[[str], int],
    job_id: str,
    arrival_s: float = 0.0,
    default_tool_s: float = 1.0,
) -> TraceJob:
    """Convert a mini-swe-agent trajectory (format `mini-swe-agent-1`).

    This is the public trajectory format of the agent Continuum evaluated with. Tool
    gaps come from message timestamps when present. The format does not require them,
    so a trajectory without them gets default_tool_s and a source label that says so.
    """
    if raw.get("trajectory_format") != "mini-swe-agent-1":
        raise ValueError(f"unsupported trajectory format {raw.get('trajectory_format')!r}")
    turns: list[TraceTurn] = []
    pending_tokens = 0
    timed = True
    messages = raw["messages"]
    for index, message in enumerate(messages):
        text = message.get("content") or ""
        if message["role"] != "assistant":
            pending_tokens += count_tokens(text)
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        tool_s = default_tool_s
        if following is not None:
            start, end = message.get("timestamp"), following.get("timestamp")
            if start is not None and end is not None:
                tool_s = max(float(end) - float(start), 0.0)
            else:
                timed = False
        turns.append(TraceTurn(pending_tokens, max(count_tokens(text), 1), "bash", tool_s))
        pending_tokens = 0
    source = "mini-swe-agent-1" if timed else "mini-swe-agent-1 (tool times assumed)"
    job = TraceJob(job_id, arrival_s, [TraceSession(f"{job_id}/main", turns)], source)
    validate(job)
    return job


@dataclass(frozen=True)
class SyntheticWorkload:
    """Parameters for a generated trace. Nothing generated is a recorded measurement."""

    name: str = "synthetic"
    jobs: int = 20
    arrival_rate_per_s: float = 0.2
    turns_mean: float = 10.9
    turns_sd: float = 2.1
    tool_s_median: float = 0.5
    tool_s_sigma: float = 1.0
    tokens_per_job_mean: float = 70_126
    tokens_per_job_sd: float = 19_732
    output_tokens_mean: float = 120
    # Sub-agents forked by a job, for the branching-factor sweep.
    branching: int = 0
    branch_turns: int = 3
    tool_names: tuple[str, ...] = ("bash", "edit", "search")


# Workload shape as tabulated in the Continuum paper (arXiv 2511.02230): turns, tool time,
# and total tokens per program. Only these summary statistics are public there, and the
# medians and spreads below are fitted to them, so these remain synthetic traces.
SWE_BENCH_SHAPE = SyntheticWorkload(name="synthetic-swebench-shape")
BFCL_SHAPE = SyntheticWorkload(
    name="synthetic-bfcl-shape",
    turns_mean=6.3,
    turns_sd=2.3,
    tool_s_median=1.2,
    tool_s_sigma=0.9,
    tokens_per_job_mean=93_256,
    tokens_per_job_sd=68_687,
)


def _tool_duration(rng: random.Random, w: SyntheticWorkload) -> float:
    return rng.lognormvariate(_ln(w.tool_s_median), w.tool_s_sigma)


def _ln(x: float) -> float:
    import math

    return math.log(x)


def generate(workload: SyntheticWorkload, seed: int = 0) -> list[TraceJob]:
    """Generate jobs with Poisson arrivals whose shape follows the workload parameters."""
    rng = random.Random(seed)
    jobs, now = [], 0.0
    for index in range(workload.jobs):
        now += rng.expovariate(workload.arrival_rate_per_s)
        job_id = f"{workload.name}-{seed}-{index}"
        count = max(1, round(rng.gauss(workload.turns_mean, workload.turns_sd)))
        total = max(count * 32, rng.gauss(workload.tokens_per_job_mean, workload.tokens_per_job_sd))
        # A context grows each turn, so tokens per program counts every turn's new tokens once.
        prompt = max(16, int(total * 0.3))
        step = max(16, int((total - prompt) / max(count, 1)))
        output = max(1, int(rng.expovariate(1 / workload.output_tokens_mean)))
        turns = [
            TraceTurn(
                prompt if i == 0 else max(1, step - output),
                max(1, int(rng.expovariate(1 / workload.output_tokens_mean))),
                rng.choice(workload.tool_names),
                _tool_duration(rng, workload),
            )
            for i in range(count)
        ]
        sessions = [TraceSession(f"{job_id}/main", turns)]
        for child in range(workload.branching):
            sessions.append(
                TraceSession(
                    f"{job_id}/sub{child}",
                    [
                        TraceTurn(
                            max(1, step // 4),
                            max(1, int(rng.expovariate(1 / workload.output_tokens_mean))),
                            rng.choice(workload.tool_names),
                            _tool_duration(rng, workload),
                        )
                        for _ in range(workload.branch_turns)
                    ],
                    parent=f"{job_id}/main",
                    fork_after_turn=rng.randrange(count),
                )
            )
        job = TraceJob(job_id, now, sessions, workload.name)
        validate(job)
        jobs.append(job)
    return jobs


@dataclass
class ContextBuilder:
    """Deterministic token ids for a trace, so every system sees identical prompts.

    Jobs share a system prompt, as real agent fleets do, which gives the prefix cache
    its honest workload.
    """

    vocab_size: int
    seed: int = 0
    system_tokens: int = 256
    _system: list[int] = field(default_factory=list)

    def system_prompt(self) -> list[int]:
        if not self._system:
            rng = random.Random(self.seed)
            self._system = [rng.randrange(self.vocab_size) for _ in range(self.system_tokens)]
        return self._system

    def new_tokens(self, session_id: str, turn: int, count: int) -> list[int]:
        rng = random.Random(f"{self.seed}/{session_id}/{turn}")
        return [rng.randrange(self.vocab_size) for _ in range(count)]

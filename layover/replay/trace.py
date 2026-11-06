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
for a recorded one. Sources are "synthetic..." for the generator and
"recorded:swebench-experiments/<submission>:reexecuted:<machine type>" for public
trajectories whose commands were re-executed (layover/replay/reexecute.py). Only the
latter may stand behind a headline number.

Optional fields, absent from older files:

    job: shared_prefix_tokens, the context every job starts with before its first turn's
         new_tokens; absent means the replayer's default system prompt. meta, a dict of
         facts about the trace's origin, never task text.
    turn: provider_new_tokens and provider_output_tokens, the provider's own counts when
         new_tokens and output_tokens are in another tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class TraceTurn:
    new_tokens: int
    output_tokens: int
    tool_name: str | None = None
    tool_s: float = 0.0
    provider_new_tokens: int | None = None
    provider_output_tokens: int | None = None


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
    shared_prefix_tokens: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def main(self) -> TraceSession:
        return next(s for s in self.sessions if s.parent is None)

    def children_of(self, session_id: str, turn: int) -> list[TraceSession]:
        return [s for s in self.sessions if s.parent == session_id and s.fork_after_turn == turn]


def validate(job: TraceJob) -> None:
    roots = [s for s in job.sessions if s.parent is None]
    if len(roots) != 1:
        raise ValueError(f"{job.job_id}: expected one root session, found {len(roots)}")
    if job.shared_prefix_tokens is not None and job.shared_prefix_tokens < 0:
        raise ValueError(f"{job.job_id}: negative shared prefix")
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
    job = TraceJob(
        raw["job_id"], float(raw["arrival_s"]), sessions, raw.get("source", "unknown"),
        raw.get("shared_prefix_tokens"), raw.get("meta") or {},
    )
    validate(job)
    return job


def job_to_dict(job: TraceJob) -> dict:
    """The JSONL form. Optional fields that are unset stay absent, as in older files."""
    raw = asdict(job)
    for name in ("shared_prefix_tokens", "meta"):
        if raw[name] is None or raw[name] == {}:
            del raw[name]
    for session in raw["sessions"]:
        for turn in session["turns"]:
            for name in ("provider_new_tokens", "provider_output_tokens"):
                if turn[name] is None:
                    del turn[name]
    return raw


def load_jsonl(path: str | Path) -> list[TraceJob]:
    with Path(path).open() as f:
        jobs = [job_from_dict(json.loads(line)) for line in f if line.strip()]
    return sorted(jobs, key=lambda j: j.arrival_s)


def save_jsonl(jobs: Iterable[TraceJob], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for job in jobs:
            f.write(json.dumps(job_to_dict(job)) + "\n")


def _message_text(message: dict) -> str:
    """A message's text, including the tool calls a v2 assistant message carries."""
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    calls = message.get("tool_calls") or []
    return content + "".join(
        json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}) for c in calls
    )


def _message_time(message: dict) -> float | None:
    stamp = message.get("timestamp")
    return stamp if stamp is not None else (message.get("extra") or {}).get("timestamp")


def from_mini_swe_agent(
    raw: dict,
    count_tokens: Callable[[str], int],
    job_id: str,
    arrival_s: float = 0.0,
    default_tool_s: float = 1.0,
) -> TraceJob:
    """Convert a saved mini-swe-agent trajectory (`mini-swe-agent-1` or `-1.1`).

    This is the public trajectory format of the agent Continuum evaluated with. Tool
    gaps come from message timestamps when present. The format does not require them,
    so a trajectory without them gets default_tool_s and a source label that says so.
    Those gaps are the submitter's, on an unknown machine, so the source never starts
    with "recorded:". For headline traces see layover/replay/trajectories.py.
    """
    trajectory_format = raw.get("trajectory_format")
    if trajectory_format not in ("mini-swe-agent-1", "mini-swe-agent-1.1"):
        raise ValueError(f"unsupported trajectory format {trajectory_format!r}")
    turns: list[TraceTurn] = []
    pending_tokens = 0
    timed = True
    # The exit message is bookkeeping, never sent to the model.
    messages = [m for m in raw["messages"] if m["role"] != "exit"]
    for index, message in enumerate(messages):
        text = _message_text(message)
        if message["role"] != "assistant":
            pending_tokens += count_tokens(text)
            continue
        # Parallel tool calls are several messages; the pause ends with the last one.
        last = index
        while last + 1 < len(messages) and messages[last + 1]["role"] != "assistant":
            last += 1
        tool_s = default_tool_s
        if last > index:
            start, end = _message_time(message), _message_time(messages[last])
            if start is not None and end is not None:
                tool_s = max(float(end) - float(start), 0.0)
            else:
                timed = False
        actions = (message.get("extra") or {}).get("actions") or []
        tool_name = tool_category(actions[0].get("command", "")) if actions else "bash"
        turns.append(TraceTurn(pending_tokens, max(count_tokens(text), 1), tool_name, tool_s))
        pending_tokens = 0
    source = trajectory_format if timed else f"{trajectory_format} (tool times assumed)"
    job = TraceJob(job_id, arrival_s, [TraceSession(f"{job_id}/main", turns)], source)
    validate(job)
    return job


# The stable tool vocabulary for recorded traces: what a duration estimator keys on. The
# first effective word of a command decides, after skipping a leading "cd DIR &&" and
# VAR=value assignments, because the agent's shell state does not persist between calls.
TOOL_VOCABULARY = ("read", "search", "edit", "python", "test", "build", "vcs", "submit", "other", "none")
_TOOL_WORDS = {
    **dict.fromkeys(("cat", "head", "tail", "nl", "less", "more", "wc", "file", "stat", "sed", "diff"), "read"),
    **dict.fromkeys(("grep", "egrep", "fgrep", "rg", "ag", "find", "ls", "tree", "pwd", "which", "awk"), "search"),
    **dict.fromkeys(("tee", "patch", "touch", "mkdir", "mv", "cp", "rm", "ln", "chmod"), "edit"),
    **dict.fromkeys(("python", "python2", "python3"), "python"),
    **dict.fromkeys(("pytest", "py.test", "tox", "nosetests", "runtests.py", "bin/test"), "test"),
    **dict.fromkeys(("pip", "pip3", "conda", "apt", "apt-get", "make", "cmake", "gcc", "npm"), "build"),
    "git": "vcs",
}
_SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def tool_category(command: str) -> str:
    """Map a shell command to TOOL_VOCABULARY."""
    if not command.strip():
        return "none"
    if _SUBMIT_MARK in command:
        return "submit"
    try:
        words = shlex.split(command, posix=True, comments=False)
    except ValueError:
        words = command.split()
    while words:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]
        elif words[0] == "cd" and len(words) >= 3 and words[2] in ("&&", ";"):
            words = words[3:]
        else:
            break
    if not words:
        return "other"
    word = words[0]
    category = _TOOL_WORDS.get(word) or _TOOL_WORDS.get(word.rsplit("/", 1)[-1], "other")
    redirects = any(w.startswith((">", "<<")) for w in words[1:6])
    if word == "cat" and redirects:
        return "edit"
    if word in ("echo", "printf"):
        return "edit" if any(w.startswith(">") for w in words) else "other"
    if word == "sed" and any(w.startswith("-i") for w in words[1:]):
        return "edit"
    if category == "python" and "pytest" in command:
        return "test"
    if category == "python" and any(w.endswith(("runtests.py", "manage.py", "bin/test")) for w in words[1:3]):
        return "test"
    return category


def lognormal_from_moments(mean: float, sd: float) -> tuple[float, float]:
    """The median and sigma of the log-normal with this mean and standard deviation."""
    if mean <= 0 or sd < 0:
        raise ValueError(f"no log-normal has mean {mean} and sd {sd}")
    variance = math.log1p((sd / mean) ** 2)
    return mean * math.exp(-variance / 2), math.sqrt(variance)


# Means and standard deviations tabulated in the Continuum paper (arXiv 2511.02230). Only
# these summary statistics are public there. Everything derived from them stays synthetic.
SWE_BENCH_PUBLISHED = {"turns": (10.9, 2.1), "tool_s": (0.925, 3.550), "tokens_per_job": (70_126, 19_732)}
BFCL_PUBLISHED = {"turns": (6.3, 2.3), "tool_s": (1.923, 2.133), "tokens_per_job": (93_256, 68_687)}
_SWE_TOOL_MEDIAN, _SWE_TOOL_SIGMA = lognormal_from_moments(*SWE_BENCH_PUBLISHED["tool_s"])


@dataclass(frozen=True)
class SyntheticWorkload:
    """Parameters for a generated trace. Nothing generated is a recorded measurement."""

    name: str = "synthetic"
    jobs: int = 20
    arrival_rate_per_s: float = 0.2
    turns_mean: float = SWE_BENCH_PUBLISHED["turns"][0]
    turns_sd: float = SWE_BENCH_PUBLISHED["turns"][1]
    tool_s_median: float = _SWE_TOOL_MEDIAN
    tool_s_sigma: float = _SWE_TOOL_SIGMA
    tokens_per_job_mean: float = SWE_BENCH_PUBLISHED["tokens_per_job"][0]
    tokens_per_job_sd: float = SWE_BENCH_PUBLISHED["tokens_per_job"][1]
    # What the tokens-per-job statistic counts, which the paper's table leaves open. "unique":
    # every token once, the size of the final context. "cumulative": each turn's whole
    # prompt plus its output, added up over the job, which is what a token bill counts.
    tokens_basis: str = "unique"
    # Not published; an assumption about generated tokens per turn.
    output_tokens_mean: float = 120
    # Sigma of a per-job log-normal factor on output length, median 1. Zero unless a
    # calibration sets it; see layover/replay/calibrate.py for what it stands for.
    job_scale_sigma: float = 0.0
    # Sub-agents forked by a job, for the branching-factor sweep.
    branching: int = 0
    branch_turns: int = 3
    tool_names: tuple[str, ...] = ("bash", "edit", "search")


def workload_from_published(name: str, published: dict) -> SyntheticWorkload:
    turns_mean, turns_sd = published["turns"]
    tokens_mean, tokens_sd = published["tokens_per_job"]
    tool_median, tool_sigma = lognormal_from_moments(*published["tool_s"])
    return SyntheticWorkload(
        name=name, turns_mean=turns_mean, turns_sd=turns_sd, tool_s_median=tool_median, tool_s_sigma=tool_sigma,
        tokens_per_job_mean=tokens_mean, tokens_per_job_sd=tokens_sd,
    )


SWE_BENCH_SHAPE = workload_from_published("synthetic-swebench-shape", SWE_BENCH_PUBLISHED)
BFCL_SHAPE = workload_from_published("synthetic-bfcl-shape", BFCL_PUBLISHED)

CALIBRATION_SCHEMA = "layover-calibration-1"


def calibrated_workload(shape: SyntheticWorkload, path: str | Path) -> SyntheticWorkload:
    """Apply a calibration file. The hash goes in the name, and so in every source."""
    data = Path(path).read_bytes()
    raw = json.loads(data)
    if raw.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"{path}: not a {CALIBRATION_SCHEMA} file")
    digest = hashlib.sha256(data).hexdigest()[:12]
    overrides = raw["generator_overrides"]
    return replace(
        shape, name=f"{shape.name}-calibrated-{digest}", job_scale_sigma=overrides["job_scale_sigma"],
        tokens_basis=overrides["tokens_basis"],
    )


# How much of a job's input arrives as its first prompt. Not published.
_FIRST_PROMPT_FRACTION = 0.3


def _tool_duration(rng: random.Random, w: SyntheticWorkload) -> float:
    return rng.lognormvariate(math.log(w.tool_s_median), w.tool_s_sigma)


def _job_turns(rng: random.Random, w: SyntheticWorkload, count: int, total: int, output_mean: float) -> list[TraceTurn]:
    """Turns whose new and output tokens add up to the job's total exactly."""
    outputs = [max(1, int(rng.expovariate(1 / output_mean))) for _ in range(count)]
    inputs = max(16 + count - 1, total - sum(outputs))
    prompt = max(16, int(inputs * _FIRST_PROMPT_FRACTION)) if count > 1 else inputs
    step, extra = divmod(inputs - prompt, max(count - 1, 1))
    new = [prompt] + [step + (1 if i < extra else 0) for i in range(count - 1)]
    return [TraceTurn(new[i], outputs[i], rng.choice(w.tool_names), _tool_duration(rng, w)) for i in range(count)]


def _unique_tokens(cumulative: float, count: int) -> int:
    """Final context size when turn prompts, growing linearly, add up to cumulative tokens."""
    if count == 1:
        return int(cumulative)
    return int(cumulative / (count * (_FIRST_PROMPT_FRACTION + (1 - _FIRST_PROMPT_FRACTION) / 2)))


def generate(workload: SyntheticWorkload, seed: int = 0) -> list[TraceJob]:
    """Generate jobs with Poisson arrivals whose shape follows the workload parameters."""
    rng = random.Random(seed)
    jobs, now = [], 0.0
    tokens_median, tokens_sigma = lognormal_from_moments(workload.tokens_per_job_mean, workload.tokens_per_job_sd)
    for index in range(workload.jobs):
        now += rng.expovariate(workload.arrival_rate_per_s)
        job_id = f"{workload.name}-{seed}-{index}"
        count = max(1, round(rng.gauss(workload.turns_mean, workload.turns_sd)))
        # Log-normal, not normal: BFCL's spread is 74% of its mean, and a normal clipped at
        # zero would no longer have the published moments.
        total = max(count * 32, int(rng.lognormvariate(math.log(tokens_median), tokens_sigma)))
        if workload.tokens_basis == "cumulative":
            total = max(count * 32, _unique_tokens(total, count))
        elif workload.tokens_basis != "unique":
            raise ValueError(f"unknown tokens_basis {workload.tokens_basis!r}")
        output_mean = workload.output_tokens_mean * rng.lognormvariate(0.0, workload.job_scale_sigma)
        turns = _job_turns(rng, workload, count, total, output_mean)
        step = max(1, sum(t.new_tokens for t in turns[1:]) // max(count - 1, 1))
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
    its honest workload. A job from a real trajectory states how long that shared part really was.
    """

    vocab_size: int
    seed: int = 0
    system_tokens: int = 256
    _prefix: list[int] = field(default_factory=list)
    _rng: random.Random | None = None

    def _shared(self, count: int) -> list[int]:
        # One stream, so a shorter prefix is always the start of a longer one.
        if self._rng is None:
            self._rng = random.Random(self.seed)
        while len(self._prefix) < count:
            self._prefix.append(self._rng.randrange(self.vocab_size))
        return self._prefix[:count]

    def system_prompt(self) -> list[int]:
        return self._shared(self.system_tokens)

    def prefix_for(self, job: TraceJob) -> list[int]:
        """The context a job starts with, before its first turn's new tokens."""
        return self.system_prompt() if job.shared_prefix_tokens is None else self._shared(job.shared_prefix_tokens)

    def new_tokens(self, session_id: str, turn: int, count: int) -> list[int]:
        rng = random.Random(f"{self.seed}/{session_id}/{turn}")
        return [rng.randrange(self.vocab_size) for _ in range(count)]

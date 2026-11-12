#!/usr/bin/env python3
"""Replay traces through every system at matched settings, and save every run.

    python bench/run_matrix.py --config bench/configs/matrix.json --out results/matrix --plan first-pass

Cells are workloads x systems x loads x runs. A load is a fraction of the arrival rate that
saturates vLLM, found once by calibration and held for every system, so a cell's label is
workload/system/load/run: headline-swebench/vllm/load100/3. A server is restarted for every
run, so no cache state carries over. A failed run is retried once and recorded with its error;
the report lists whatever is missing. After limits.max_consecutive_failures failed cells in a
row, whichever systems they belong to, the whole matrix stops: something is broken that a person
must look at, and the machine shuts itself down rather than bill for attempts. session.json and
the report say why; a resume tries again.

A run's record is fsynced and renamed into place the moment the run ends, and a calibration
rung's likewise. Rerunning the same command resumes, on this machine or, with results/ restored,
on a new one after a spot reclaim: a kill at any point costs at most the run in progress. A
different GPU or driver is refused unless --accept-mixed-hardware says otherwise, and the report
then flags it. Each record names its instance, and each table how many instances it spans.

A plan in the config lists runs in the order they should happen, so a budget stop keeps the most
valuable. --dry-run prices every plan. --only narrows the chosen plan further. --max-hours counts
the matrix hours kept in budget.json beside the runs, so neither a resume nor a spot replacement
gets a fresh budget; the report lists what the budget left out.

Headline workloads must be recorded trajectories: their trace's source must start with
"recorded:". Only --pipeline-check relaxes that, and it marks every output as not a result.

--preflight-only needs no trace, and launches each kind of system once with a real completion,
so a wrong flag shows up before anything long starts. With --preflight-kinds vllm sglang it also
needs no Layover cost model, so it can run right after the baselines are installed.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import hashlib
import json
import math
import os
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.gpu_sampler import GIB, GpuSampler, NvmlBackend  # noqa: E402
from bench.systems import (  # noqa: E402
    BLOCK_TOKENS, BudgetMismatch, GpuNotReleased, ManagedServer, MatchedSettings, ServerFailed, System, build_system,
    check_cpu_tier, check_gpu_memory, check_kv_capacity, offload_record_bytes, wait_gpu_released,
)
from layover.config import ModelConfig  # noqa: E402
from layover.replay.http_replay import _cached_tokens, _run_turn, replay  # noqa: E402
from layover.replay.results import ReplayResult  # noqa: E402
from layover.replay.trace import (  # noqa: E402
    BFCL_SHAPE, ContextBuilder, SWE_BENCH_SHAPE, TraceJob, TraceSession, TraceTurn, calibrated_workload, generate, job_to_dict,
    load_jsonl,
)

SCHEMA = 1
RECORDED_PREFIX = "recorded:"
PIPELINE_BANNER = "PIPELINE CHECK: NOT A RESULT. These numbers describe the test pipeline and nothing else."
SHAPES = {"swebench": SWE_BENCH_SHAPE, "bfcl": BFCL_SHAPE}
PREPARED_DIR = "traces/prepared"
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 3
HEARTBEAT_S = 30.0
FIRST_TOKEN_NOTE = (
    "first_token_s is the client's time from submitting a turn to its first streamed token, "
    "measured the same way for every system. It holds queueing and prefill together; it is not pure queueing delay."
)


class Refused(SystemExit):
    """The configuration would produce a result that must not exist."""


class BudgetReached(Exception):
    """--max-hours ran out between two steps of a calibration. Everything measured is saved."""


class CalibrationFailed(ServerFailed):
    """A calibration step failed every attempt. Every rate depends on it, so nothing else can run."""


@dataclass
class Workload:
    name: str
    role: str
    systems: list[str]
    warmup_jobs: int
    jobs: list[TraceJob] = field(default_factory=list)
    source: str | None = None
    trace_hash: str | None = None
    trace_path: str | None = None
    arrival_rate_per_s: float | None = None
    calibration_group: str | None = None
    # A workload can be a point on several sweeps: {"sweep": name, "x": value}.
    sweeps: list[dict] = field(default_factory=list)
    calibration_reference: bool = False
    dropped_jobs: list[str] = field(default_factory=list)
    unavailable: str | None = None
    # How many jobs to plan for while the trace does not exist yet.
    estimate_jobs: int | None = None
    trace_meta: dict = field(default_factory=dict)
    # Which multiples of the group's saturation rate this workload runs at.
    load_fractions: list[float] = field(default_factory=lambda: [1.0])


def load_label(fraction: float) -> str:
    """A load's name in cell labels and paths: percent, so 1.25 is load125 whatever the rate."""
    return f"load{round(fraction * 100)}"


def load_fraction(label: str) -> float:
    return int(label.removeprefix("load")) / 100


@dataclass(frozen=True)
class Cell:
    workload: str
    system: str
    run: int
    load: str = "load100"

    @property
    def label(self) -> str:
        return f"{self.workload}/{self.system}/{self.load}/{self.run}"


def write_json(path: Path, data: dict) -> None:
    """Durable and atomic: a kill at any point leaves the old file or the whole new one.

    Nothing ever reads a .tmp file, so one left by a killed writer does not count as anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(data, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # The rename is only durable once its directory is.
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def stable_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def refuse_unrecorded(workload: str, jobs: list[TraceJob], pipeline_check: bool) -> None:
    """The headline table holds recorded agent trajectories and nothing else."""
    bad = sorted({job.source for job in jobs if not job.source.startswith(RECORDED_PREFIX)})
    if bad and not pipeline_check:
        raise Refused(
            f"headline workload {workload!r} has traces whose source does not start with {RECORDED_PREFIX!r}: {bad}. "
            "Build them with make traces on the x86 GPU machine, without reexecute's pipeline-check option, "
            "or pass --pipeline-check to test the pipeline and label every output as such."
        )


def peak_context(job: TraceJob, system_tokens: int) -> int:
    """The longest prompt plus output any session reaches, through its forks."""
    start = job.shared_prefix_tokens if job.shared_prefix_tokens is not None else system_tokens
    by_parent: dict[str, list[TraceSession]] = {}
    for session in job.sessions:
        if session.parent is not None:
            by_parent.setdefault(session.parent, []).append(session)

    def walk(session: TraceSession, context: int) -> int:
        peak = context
        for index, turn in enumerate(session.turns):
            context += turn.new_tokens + turn.output_tokens
            peak = max(peak, context)
            for child in by_parent.get(session.session_id, []):
                if child.fork_after_turn == index:
                    peak = max(peak, walk(child, context))
        return peak

    return walk(job.main, start)


def enforce_context(workload: str, jobs: list[TraceJob], limit: int, system_tokens: int, drop: bool) -> tuple[list[TraceJob], list[str]]:
    """A job over the matched context limit would fail on some systems and not others."""
    over = {job.job_id: peak_context(job, system_tokens) for job in jobs if peak_context(job, system_tokens) > limit}
    if not over:
        return jobs, []
    if not drop:
        worst = max(over.values())
        raise Refused(
            f"{workload}: {len(over)} of {len(jobs)} jobs exceed max_context_tokens={limit} (largest {worst}). "
            "Raise the limit, or pass --drop-over-context to leave them out; the report then says how many."
        )
    return [job for job in jobs if job.job_id not in over], sorted(over)


def retime(jobs: list[TraceJob], rate_per_s: float, seed: int) -> list[TraceJob]:
    """Poisson arrivals at the matrix's rate, in the trace's job order.

    A recorded trace's arrival times say when the recording ran, not what load to apply.
    The same seed gives every system the same arrivals.
    """
    rng = random.Random(f"arrivals/{seed}")
    now = 0.0
    timed = []
    for job in jobs:
        timed.append(replace(job, arrival_s=now))
        now += rng.expovariate(rate_per_s)
    return timed


def repo_path(path: str | Path) -> Path:
    """Paths in a config are relative to the repository, wherever the command runs."""
    return REPO / path


def _synthetic_jobs(sweeps: dict, overrides: dict) -> list[TraceJob]:
    base = sweeps["base"]
    shape = SHAPES[base.get("shape", "swebench")]
    calibration = base.get("calibration_file")
    if calibration and repo_path(calibration).exists():
        # The calibration's hash goes in the shape's name, and so in every source.
        shape = calibrated_workload(shape, repo_path(calibration))
    workload = replace(shape, jobs=base["jobs"], **base.get("overrides", {}), **overrides)
    return generate(workload, base.get("seed", 0))


def trace_meta_summary(jobs: list[TraceJob]) -> dict:
    """Facts a whole trace has in common, such as its submission and machine type."""
    values: dict[str, set[str]] = {}
    for job in jobs:
        for key, value in job.meta.items():
            if isinstance(value, str):
                values.setdefault(key, set()).add(value)
            elif isinstance(value, dict):
                values.setdefault(key, set()).add(json.dumps(value, sort_keys=True))
    return {key: sorted(found) for key, found in sorted(values.items()) if len(found) <= 3}


def load_workloads(
    config: dict, settings: MatchedSettings, pipeline_check: bool, drop_over_context: bool = False, dry_run: bool = False,
    roles: set[str] | None = None,
) -> list[Workload]:
    """Resolve every workload to jobs, refusing what the rules do not allow.

    dry_run accepts a trace that does not exist yet, for plans and preflights made before it is
    built. roles leaves out what a plan does not run, so first-pass needs no sweep inputs.
    """
    all_systems = [s["name"] for s in config["systems"]]
    system_tokens = config.get("system_tokens", 256)
    workloads: list[Workload] = []
    load = config.get("load", {})

    for entry in config.get("workloads", []):
        if roles is not None and entry.get("role", "headline") not in roles:
            continue
        workload = Workload(
            entry["name"], entry.get("role", "headline"), entry.get("systems", all_systems),
            entry.get("warmup_jobs", config.get("warmup_jobs", 0)), trace_path=entry["trace"],
            arrival_rate_per_s=entry.get("arrival_rate_per_s"), calibration_group=entry.get("calibration_group"),
            estimate_jobs=entry.get("estimate_jobs"), load_fractions=entry.get("load_fractions", load.get("fractions", [1.0])),
        )
        path = repo_path(entry["trace"])
        if not path.exists():
            if not dry_run:
                raise Refused(f"workload {workload.name}: trace {path} does not exist")
            workload.unavailable = f"trace {path} does not exist yet"
            workloads.append(workload)
            continue
        jobs = load_jsonl(path)
        if entry.get("max_jobs"):
            jobs = jobs[: entry["max_jobs"]]
        if not jobs:
            raise Refused(f"workload {workload.name}: trace {path} has no jobs")
        if workload.role == "headline":
            refuse_unrecorded(workload.name, jobs, pipeline_check)
        workload.jobs, workload.dropped_jobs = enforce_context(
            workload.name, jobs, settings.max_context_tokens, system_tokens, drop_over_context
        )
        workload.source = _source_label(workload.jobs)
        workload.trace_meta = trace_meta_summary(workload.jobs)
        workload.trace_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        workloads.append(workload)

    sweeps = config.get("sweeps")
    if sweeps and (roles is None or "sweep" in roles):
        calibration = sweeps["base"].get("calibration_file")
        if calibration and not repo_path(calibration).exists():
            message = f"synthetic calibration file {calibration} does not exist; run bench/calibrate_synthetic.py first"
            if not dry_run:
                raise Refused(message)
            print(f"WARNING: {message}. The estimate below uses the uncalibrated shape.")
        tool_base = sweeps["base"].get("tool_median_s", 0.5)
        points: dict[tuple[float, int], list[dict]] = {}
        for value in sweeps.get("tool_median_s", []):
            points.setdefault((value, 0), []).append({"sweep": "tool_median_s", "x": value})
        for value in sweeps.get("branching", []):
            points.setdefault((tool_base, value), []).append({"sweep": "branching", "x": value})
        for (tool, branching), memberships in sorted(points.items()):
            name = f"sweep-tool{tool:g}-branch{branching}"
            jobs = _synthetic_jobs(sweeps, {"tool_s_median": tool, "branching": branching})
            jobs, dropped = enforce_context(name, jobs, settings.max_context_tokens, system_tokens, drop_over_context)
            workloads.append(
                Workload(
                    name, "sweep", sweeps.get("systems", all_systems), sweeps.get("warmup_jobs", config.get("warmup_jobs", 0)),
                    jobs, _source_label(jobs), stable_hash([job_to_dict(j) for j in jobs]),
                    arrival_rate_per_s=sweeps.get("arrival_rate_per_s"), calibration_group=sweeps.get("calibration_group"),
                    sweeps=memberships, dropped_jobs=dropped, calibration_reference=(tool, branching) == (tool_base, 0),
                    load_fractions=load.get("sweep_fractions", load.get("fractions", [1.0])),
                )
            )

    for workload in workloads:
        unknown = set(workload.systems) - set(all_systems)
        if unknown:
            raise Refused(f"workload {workload.name} names unknown systems {sorted(unknown)}")
        if workload.jobs and len(workload.jobs) <= workload.warmup_jobs:
            raise Refused(f"workload {workload.name}: {workload.warmup_jobs} warm-up jobs leave none of {len(workload.jobs)} to measure")
    return workloads


def _source_label(jobs: list[TraceJob]) -> str:
    """One source, or several recorded ones. A synthetic job never sits beside anything else."""
    sources = sorted({job.source for job in jobs})
    if len(sources) > 1 and not all(source.startswith(RECORDED_PREFIX) for source in sources):
        raise Refused(f"a workload mixes trace sources: {sources}")
    return "; ".join(sources)


def build_cells(workloads: list[Workload], runs: int) -> list[Cell]:
    """Every workload's run 0 for all systems and loads before anyone's run 1.

    If the budget ends the session early, what remains is evenly short rather than one
    system complete and another absent. Nearest the saturation rate goes first in each
    round, since that load is the one a shortened table can least do without.
    """
    cells = []
    for workload in workloads:
        loads = sorted(workload.load_fractions, key=lambda fraction: (abs(fraction - 1.0), -fraction))
        for run in range(runs):
            for fraction in loads:
                cells.extend(Cell(workload.name, system, run, load_label(fraction)) for system in workload.systems)
    return cells


def plan_groups(config: dict, name: str) -> list[dict]:
    """A plan's groups, in order, after those of any plan it extends."""
    plans = config.get("plans", {})
    if name not in plans:
        raise Refused(f"no plan {name!r}; the config has {sorted(plans)}")
    entry = plans[name]
    return (plan_groups(config, entry["extends"]) if "extends" in entry else []) + entry.get("groups", [])


def plan_matched(config: dict, name: str | None) -> dict:
    """A plan's changes to the matched settings, after those of any plan it extends."""
    entry = config.get("plans", {}).get(name) if name else None
    if entry is None:
        return {}
    return {**(plan_matched(config, entry["extends"]) if "extends" in entry else {}), **entry.get("matched", {})}


def plan_workloads(config: dict, name: str | None) -> dict:
    """A plan's changes to workloads, by name, after those of any plan it extends."""
    entry = config.get("plans", {}).get(name) if name else None
    if entry is None:
        return {}
    merged = plan_workloads(config, entry["extends"]) if "extends" in entry else {}
    for workload, changes in entry.get("workloads", {}).items():
        merged[workload] = {**merged.get(workload, {}), **changes}
    return merged


WORKLOAD_CHANGES = {"max_jobs", "warmup_jobs"}


def effective_config(config: dict, name: str | None) -> dict:
    """The config as a plan runs it. A plan may change matched settings, every system's at once,
    for example a smaller CPU tier on a machine with less RAM, and a workload's size: how many of
    its jobs run and how many of those warm up. The manifest records and hashes the result, so a
    resume at another plan's settings is refused like any other changed config."""
    changes, workloads = plan_matched(config, name), plan_workloads(config, name)
    if not changes and not workloads:
        return config
    unknown = sorted(set(changes) - set(config["matched"]))
    if unknown:
        raise Refused(f"plan {name!r} changes matched settings the config does not have: {unknown}")
    names = {entry["name"] for entry in config["workloads"]}
    for workload, edits in workloads.items():
        if workload not in names:
            raise Refused(f"plan {name!r} changes workload {workload!r}, which the config does not have")
        if set(edits) - WORKLOAD_CHANGES:
            raise Refused(f"plan {name!r} may change only {sorted(WORKLOAD_CHANGES)} of a workload, not {sorted(set(edits) - WORKLOAD_CHANGES)}")
    return {
        **config,
        "matched": {**config["matched"], **changes},
        "workloads": [{**entry, **workloads.get(entry["name"], {})} for entry in config["workloads"]],
    }


def plan_roles(config: dict, name: str) -> set[str]:
    return {group["role"] for group in plan_groups(config, name)}


def plan_cells(config: dict, workloads: list[Workload], name: str) -> list[Cell]:
    """A plan's runs, in order: group by group, and each group's run 0 before anyone's run 1.

    If a budget ends the session, the groups listed first are complete and the group it stopped
    in is evenly short, rather than one system complete and another absent. A group may give runs
    per load, {"load100": 5, "load125": 3}; it then finishes each load, in the order listed, before
    the next, since those loads are not equally valuable.
    """
    known = {spec["name"] for spec in config["systems"]}
    cells: dict[Cell, None] = {}
    for group in plan_groups(config, name):
        unknown = set(group.get("systems", [])) - known
        if unknown:
            raise Refused(f"plan {name} names unknown systems {sorted(unknown)}")
        per_load = group["runs"] if isinstance(group["runs"], dict) else None
        if per_load is not None and set(per_load) != {load_label(fraction) for fraction in group["loads"]}:
            raise Refused(f"plan {name}: runs per load {sorted(per_load)} do not match its loads {group['loads']}")
        for workload in workloads:
            if workload.role != group["role"]:
                continue
            if per_load is not None:
                order = [(run, fraction) for fraction in group["loads"] for run in range(per_load[load_label(fraction)])]
            else:
                order = [(run, fraction) for run in range(group["runs"]) for fraction in group["loads"]]
            for run, fraction in order:
                for system in group.get("systems", workload.systems):
                    cells[Cell(workload.name, system, run, load_label(fraction))] = None
    return list(cells)


def select_cells(cells: list[Cell], patterns: list[str] | None) -> list[Cell]:
    """--only takes shell patterns over workload/system/load/run."""
    if not patterns:
        return cells
    return [cell for cell in cells if any(fnmatch.fnmatch(cell.label, pattern) for pattern in patterns)]


def _prefilled_tokens(job: TraceJob, recompute: bool) -> int:
    """Tokens a job prefills: new ones only, or its whole context on every turn."""
    total = 0
    for session in job.sessions:
        context = 0
        for turn in session.turns:
            context += turn.new_tokens
            total += context if recompute else turn.new_tokens
            context += turn.output_tokens
    return total


def estimate_run_s(jobs: list[TraceJob], rate_per_s: float, assumed: dict, recompute: bool = False, pool_tokens: int | None = None) -> float:
    """A rough duration for one run: trace tool times plus assumed token rates.

    The rates are assumptions in the config, never measurements. The arrival process
    contributes its mean span; the bound on the total covers a saturated server. With
    recompute, every turn prefills its whole context, which is the slow end: a system
    that loses its KV at every tool call. With pool_tokens, only as many jobs decode at
    once as their contexts fit in the pool, which is usually far below a full batch.
    """
    prefill = assumed["prefill_tokens_per_s"]
    decode_stream = assumed["decode_tokens_per_s_per_stream"]
    decode_total = assumed["decode_tokens_per_s_total"]
    if pool_tokens is not None:
        mean_peak = statistics.fmean(peak_context(job, 0) for job in jobs)
        decode_total = min(decode_total, decode_stream * max(1.0, pool_tokens / mean_peak))
    longest, serial = 0.0, 0.0
    for index, job in enumerate(jobs):
        new = _prefilled_tokens(job, recompute)
        output = sum(t.output_tokens for s in job.sessions for t in s.turns)
        tool = sum(t.tool_s for t in job.main.turns[:-1])
        serial += new / prefill + output / decode_total
        longest = max(longest, index / rate_per_s + tool + new / prefill + output / decode_stream)
    return max(longest, serial) + assumed["startup_s"] + assumed.get("stop_s", 0.0)


class MatrixContext:
    """Everything one invocation shares."""

    def __init__(
        self, config: dict, settings: MatchedSettings, systems: dict[str, System], workloads: dict[str, Workload],
        out: Path, pipeline_check: bool, gpu, weights_bytes: int | None, token_limit: int, encode: Callable | None,
        instance: dict | None = None,
    ):
        self.config = config
        self.settings = settings
        self.systems = systems
        self.workloads = workloads
        self.out = Path(out)
        self.pipeline_check = pipeline_check
        self.gpu = gpu
        self.weights_bytes = weights_bytes
        self.token_limit = token_limit
        self.encode = encode
        self.limits = config["limits"]
        self.tolerances = config["tolerances"]
        self.deadline: float | None = None
        self.accept_mixed_hardware = False
        # What this machine is, for each record it writes.
        self.instance = instance or {}

    def run_path(self, cell: Cell) -> Path:
        return self.out / "runs" / cell.workload / cell.system / cell.load / f"run-{cell.run}.json"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def verify_budgets(ctx: MatrixContext, system: System, server: ManagedServer, idle_bytes: int | None) -> dict:
    """Check a freshly started server against its budget, before any measurement."""
    translation = system.translation
    checks: dict = {"translation": asdict(translation) if translation else None}
    if translation is not None:
        reported = system.reported_kv_tokens(server.client, server.log_text())
        checks.update(check_kv_capacity(system.name, reported, translation.kv_tokens_expected, ctx.tolerances["kv_capacity_frac"]))
    if ctx.gpu is not None and translation is not None:
        used = ctx.gpu.read().memory_used_bytes
        checks.update(
            check_gpu_memory(
                system.name, used, idle_bytes, ctx.weights_bytes, int(ctx.settings.kv_gib * GIB), ctx.tolerances["gpu_memory_gib"]
            )
        )
    if translation is not None:
        checks.update(tier_budget(ctx, system, server))
    return checks


def tier_budget(ctx: MatrixContext, system: System, server: ManagedServer) -> dict:
    """The CPU tier a running server says it has, against the budget. Same bytes for everyone."""
    reported = system.reported_cpu_tier_gib(server.client, server.log_text())
    frac = ctx.tolerances.get("cpu_tier_frac", 0.01)
    return {**check_cpu_tier(system.name, reported, ctx.settings.cpu_offload_gib, frac), "cpu_tier_source": system.cpu_tier_source}


@contextmanager
def serving(ctx: MatrixContext, system: System, log_path: Path):
    """A fresh server, checked. Always stopped, and the GPU is checked to be empty again."""
    idle_bytes = None
    limit = int(ctx.tolerances["gpu_idle_gib"] * GIB)
    if ctx.gpu is not None:
        idle_bytes = wait_gpu_released(ctx.gpu, limit, timeout_s=ctx.limits["gpu_release_timeout_s"])
    server = ManagedServer(system, log_path, free_port(), stop_timeout_s=ctx.limits["stop_timeout_s"])
    try:
        server.start()
        server.wait_ready(ctx.limits["startup_timeout_s"])
        yield server, verify_budgets(ctx, system, server, idle_bytes)
    finally:
        server.stop()
        if ctx.gpu is not None:
            wait_gpu_released(ctx.gpu, limit, timeout_s=ctx.limits["gpu_release_timeout_s"])


def _replay_jobs(ctx: MatrixContext, system: System, server: ManagedServer, jobs: list[TraceJob], warmup: int, seed: int, timeout_s: float) -> ReplayResult:
    return asyncio.run(
        asyncio.wait_for(
            replay(
                jobs, server.url, ctx.settings.model, ctx.token_limit, system.name, ctx.config.get("time_scale", 1.0),
                warmup, seed, ctx.config.get("system_tokens", 256), system.hints, ctx.encode, ctx.limits["request_timeout_s"],
            ),
            timeout_s,
        )
    )


def not_comparable(result: ReplayResult) -> str | None:
    """Why a run cannot sit in a table beside the others, if so.

    A turn that returned fewer tokens did less work, and re-encoded text changes the
    context the server sees, so both would quietly favor that system.
    """
    if result.output_count_mismatches:
        return f"{result.output_count_mismatches} turns did not return the requested token count; ignore_eos did not hold"
    if result.token_sources.get("tokenizer"):
        return f"{result.token_sources['tokenizer']} turns returned no token ids, so their output was re-encoded from text"
    return None


class KvUsagePoller(threading.Thread):
    """Polls each server's own KV usage at a low rate; NVML sees only the preallocated pool."""

    def __init__(self, system: System, url: str, period_s: float):
        super().__init__(daemon=True, name="kv-usage")
        self.system = system
        self.client = httpx.Client(base_url=url, timeout=10.0)
        self.period_s = period_s
        # (when, KV pool used, the share held only by a radix tree or None if the system has none to report)
        self.values: list[tuple[float, float, float | None]] = []
        self.errors = 0
        self.halted = threading.Event()

    def run(self) -> None:
        while not self.halted.wait(self.period_s):
            try:
                used, tree_only = self.system.kv_fractions(self.client)
                if used is not None:
                    self.values.append((time.perf_counter(), float(used), None if tree_only is None else float(tree_only)))
            except (httpx.HTTPError, KeyError, ValueError):
                self.errors += 1

    def finish(self) -> None:
        self.halted.set()
        self.join(timeout=15)
        self.client.close()

    def summary(self, start_s: float, end_s: float) -> dict:
        window = [(used, tree) for at, used, tree in self.values if start_s <= at <= end_s]
        kept = [used for used, _ in window]
        trees = [tree for _, tree in window if tree is not None]
        return {
            "kv_pool_used_frac_mean": statistics.fmean(kept) if kept else None,
            "kv_pool_used_frac_peak": max(kept) if kept else None,
            "kv_pool_tree_only_frac_mean": statistics.fmean(trees) if trees else None,
            "kv_pool_tree_only_frac_peak": max(trees) if trees else None,
            # Held by anyone at all: the one figure that says the free list really ran out.
            "kv_pool_occupied_frac_peak": max(used + tree for used, tree in window if tree is not None) if trees else None,
            "kv_usage_samples": len(kept),
            "kv_usage_errors": self.errors,
        }


def measure_run(ctx: MatrixContext, cell: Cell, log_path: Path, rate: dict) -> dict:
    """One attempt: fresh server, replay, measurements. Raises when anything fails."""
    system = ctx.systems[cell.system]
    workload = ctx.workloads[cell.workload]
    seed = ctx.config.get("seed_base", 0) + cell.run
    jobs = retime(workload.jobs, rate["arrival_rate_per_s"], seed)
    with serving(ctx, system, log_path) as (server, checks):
        sampler = GpuSampler(ctx.gpu, ctx.config.get("sampler_period_s", 0.25)) if ctx.gpu is not None else None
        poller = KvUsagePoller(system, server.url, ctx.config.get("kv_usage_poll_s", 2.0))
        started = time.perf_counter()
        if sampler is not None:
            sampler.start()
        poller.start()
        try:
            result = _replay_jobs(ctx, system, server, jobs, workload.warmup_jobs, seed, ctx.limits["run_timeout_s"])
        finally:
            elapsed = time.perf_counter() - started
            poller.finish()
            if sampler is not None:
                sampler.stop()
        try:
            reported = system.server_metrics(server.client)
        except httpx.HTTPError as exc:
            # A finished replay is worth keeping when only the counters are lost.
            reported = {"error": str(exc)}
        if system.translation is not None:
            try:
                checks.update(tier_budget(ctx, system, server))
            except httpx.HTTPError:
                pass
    if not ctx.pipeline_check:
        reason = not_comparable(result)
        if reason is not None:
            raise ServerFailed(f"{cell.label} is not comparable: {reason}")
    # A time window, so warm-up jobs still running in it are counted. Only job times leave them out exactly.
    window = result.clock_origin_s + result.window_start_s, result.clock_origin_s + result.window_end_s
    usage = poller.summary(*window)
    gpu = sampler.summary(*window) if sampler is not None else {}
    if sampler is not None:
        if not gpu["samples"]:
            raise ServerFailed(f"no GPU samples in the measured part of {cell.label}: {sampler.read_errors} read errors")
        sampler.write_csv(log_path.with_suffix(".gpu.csv"))
    summary = result.summary()
    return {
        "metrics": {
            key: summary[key]
            for key in (
                "mean_jct_s", "p95_jct_s", "mean_jct_minus_tool_s", "p95_jct_minus_tool_s", "programs_per_hour",
                "mean_first_token_s", "p95_first_token_s",
            )
        } | {key: gpu.get(key) for key in (
            "gpu_mem_frac_mean", "gpu_mem_frac_peak", "gpu_mem_used_gib_mean", "gpu_util_pct_mean", "gpu_mem_busy_pct_mean",
            "pcie_gb", "pcie_tx_gb", "pcie_rx_gb",
        )} | {key: usage[key] for key in (
            "kv_pool_used_frac_mean", "kv_pool_used_frac_peak", "kv_pool_tree_only_frac_mean", "kv_pool_tree_only_frac_peak",
            "kv_pool_occupied_frac_peak",
        )},
        "source": result.source,
        "jobs_total": result.jobs_total,
        "jobs_measured": result.jobs_measured,
        "elapsed_s": elapsed,
        "jct_s": result.jct_s,
        "token_sources": result.token_sources,
        "output_count_mismatches": result.output_count_mismatches,
        "prompt_tokens_total": result.prompt_tokens_total,
        "output_tokens_total": result.output_tokens_total,
        "cached_tokens_total": result.cached_tokens_total,
        "server_metrics": reported,
        "budget_checks": checks,
        "gpu": gpu,
        "kv_usage": usage,
        "turns": [
            {
                "session": t.session_id, "turn": t.turn, "crc": t.output_crc, "source": t.token_source,
                "first_token_s": t.first_token_s - t.submitted_s,
            }
            for t in result.turns
        ],
        "log": str(log_path.relative_to(ctx.out)),
    }


def load_run(path: Path) -> dict | None:
    """A saved JSON file, or None if there is none that can be read.

    write_json cannot leave a torn file, but a copy cut off on its way back from a machine can.
    A torn record is an unfinished run, so a resume repeats it rather than crashing.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except ValueError:
        print(f"WARNING: {path} is not valid JSON, so it counts as not done", flush=True)
        return None
    return data if isinstance(data, dict) else None


def run_finished(record: dict | None) -> bool:
    """The one test of whether a resume may skip a run."""
    return bool(record) and record.get("status") == "ok" and isinstance(record.get("result"), dict) and "metrics" in record["result"]


def expected_provenance(ctx: MatrixContext, cell: Cell) -> dict:
    """What a run's record must agree with to sit in the same table as this session's runs."""
    return {
        "config_hash": stable_hash(ctx.config),
        # Read for each run: this is the file its server is about to price placements with.
        "cost_model_hash": cost_model_hash(ctx.config),
        "trace_hash": ctx.workloads[cell.workload].trace_hash,
    }


def run_cell(ctx: MatrixContext, cell: Cell, rate: dict) -> dict:
    """Up to MAX_ATTEMPTS attempts, saving after each. Earlier attempts stay in the file."""
    path = ctx.run_path(cell)
    record = load_run(path) or {
        "schema": SCHEMA,
        "workload": cell.workload,
        "system": cell.system,
        "run": cell.run,
        "load": cell.load,
        "seed": ctx.config.get("seed_base", 0) + cell.run,
        "pipeline_check": ctx.pipeline_check,
        "attempts": [],
    }
    record.pop("result", None)
    record.update(
        status="failed", role=ctx.workloads[cell.workload].role, trace_source=ctx.workloads[cell.workload].source, arrival=rate,
        instance=ctx.instance, **expected_provenance(ctx, cell),
    )
    for attempt_index in range(MAX_ATTEMPTS):
        # A retry can take as long as run_timeout_s; once the budget is gone it waits for a resume.
        if attempt_index and ctx.deadline is not None and time.monotonic() >= ctx.deadline:
            print("  time budget reached: not retrying", flush=True)
            break
        number = len(record["attempts"]) + 1
        log_path = ctx.out / "logs" / cell.workload / cell.system / cell.load / f"run-{cell.run}.attempt-{number}.log"
        started = time.time()
        attempt = {"attempt": number, "started_unix": started, "log": str(log_path.relative_to(ctx.out)), "instance_id": ctx.instance.get("instance_id")}
        try:
            record["result"] = measure_run(ctx, cell, log_path, rate)
        except GpuNotReleased as exc:
            attempt.update(status="failed", error=f"GpuNotReleased: {exc}", elapsed_s=time.time() - started)
            record["attempts"].append(attempt)
            write_json(path, record)
            raise
        except KeyboardInterrupt:
            attempt.update(status="failed", error="interrupted", elapsed_s=time.time() - started)
            record["attempts"].append(attempt)
            write_json(path, record)
            raise
        except Exception as exc:  # Whatever a system does wrong is recorded, never lost.
            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_s=time.time() - started)
            record["attempts"].append(attempt)
            write_json(path, record)
            print(f"  attempt {number} failed: {attempt['error'][:300]}", flush=True)
            continue
        attempt.update(status="ok", elapsed_s=time.time() - started)
        record["attempts"].append(attempt)
        record["status"] = "ok"
        record["finished_unix"] = time.time()
        write_json(path, record)
        break
    return record


def saturation_rate(points: list[dict], target_slowdown: float) -> dict:
    """Where the measured slowdown crosses the target, between the rungs that bracket it.

    The slowdown is taken as linear in log rate between two rungs, so with s_low < target <=
    s_high, log(rate) = log(r_low) + (target - s_low) / (s_high - s_low) * log(r_high / r_low).
    With no bracket there is no such rate: the result names the nearest rung and says which
    edge it is, and a caller must not treat that as a saturation rate without saying so.
    """
    ordered = sorted(points, key=lambda point: point["rate_per_s"])
    for low, high in zip(ordered, ordered[1:]):
        if low["slowdown"] < target_slowdown <= high["slowdown"]:
            share = (target_slowdown - low["slowdown"]) / (high["slowdown"] - low["slowdown"])
            rate = math.exp(math.log(low["rate_per_s"]) + share * math.log(high["rate_per_s"] / low["rate_per_s"]))
            return {"saturation_rate_per_s": rate, "bracketed": True, "edge": None, "bracket": [low, high]}
    if ordered[0]["slowdown"] >= target_slowdown:
        return {"saturation_rate_per_s": ordered[0]["rate_per_s"], "bracketed": False, "edge": "below_ladder", "bracket": None}
    return {"saturation_rate_per_s": ordered[-1]["rate_per_s"], "bracketed": False, "edge": "above_ladder", "bracket": None}


def next_rung(ladder: list[float], points: list[dict], target_slowdown: float, start: float) -> float | None:
    """The next rate to replay: outward from start until two neighbours bracket the target.

    Each rung costs a whole replay of the workload, so walking from a good guess pays for two
    or three of them where climbing from the bottom pays for every slow one below the knee.
    None when it is bracketed or the ladder has run out.
    """
    if not points:
        return start
    if saturation_rate(points, target_slowdown)["bracketed"]:
        return None
    measured = [point["rate_per_s"] for point in points]
    if all(point["slowdown"] < target_slowdown for point in points):
        return next((rate for rate in sorted(ladder) if rate > max(measured)), None)
    return next((rate for rate in sorted(ladder, reverse=True) if rate < min(measured)), None)


def check_budget(ctx: MatrixContext) -> None:
    if ctx.deadline is not None and time.monotonic() >= ctx.deadline:
        raise BudgetReached


def calibration_identity(ctx: MatrixContext, workload: Workload) -> dict:
    """What a calibration describes. The rate it finds means nothing for another trace or GPU."""
    return {"config_hash": stable_hash(ctx.config), "trace_hash": workload.trace_hash, "hardware": hardware_facts(ctx.gpu)}


def refuse_other_calibration(ctx: MatrixContext, path: Path, saved: dict, identity: dict) -> None:
    """A saved calibration, or part of one, sets every system's rate, so it must be this session's.

    The manifest and the run records have their own checks, but a restored directory with no
    finished run has only this one.
    """
    found = saved.get("identity") or {}
    for key in ("config_hash", "trace_hash"):
        if found.get(key) != identity[key]:
            raise Refused(f"{path} was measured with {key} {found.get(key)}, but this session has {identity[key]}; use a new --out directory")
    if found.get("hardware") != identity["hardware"]:
        message = f"{path} was measured on {found.get('hardware')}, and this machine is {identity['hardware']}"
        if not ctx.accept_mixed_hardware:
            raise Refused(f"{message}. Use a new --out directory, or pass --accept-mixed-hardware to keep it with every output flagged.")
        print(f"WARNING: MIXED HARDWARE: {message}. Every output says so.", flush=True)


def calibration_step(ctx: MatrixContext, label: str, progress: dict, progress_path: Path, attempt: Callable[[int], ReplayResult | None]):
    """One step of the ladder, retried as a measured run is. Each failure stays in the partial file."""
    last = None
    for number in range(1, MAX_ATTEMPTS + 1):
        check_budget(ctx)
        try:
            return attempt(number)
        except (GpuNotReleased, BudgetReached):
            raise
        except Exception as exc:  # Whatever a system does wrong is recorded, as in run_cell.
            last = f"{type(exc).__name__}: {exc}"
            progress.setdefault("failures", []).append({"step": label, "attempt": number, "error": last})
            write_json(progress_path, progress)
            print(f"  calibration {label}, attempt {number} failed: {last[:300]}", flush=True)
    raise CalibrationFailed(f"{label} failed after {MAX_ATTEMPTS} attempts: {last}")


def comparable_replay(ctx: MatrixContext, result: ReplayResult, label: str) -> ReplayResult:
    """A rung of short outputs would set every system's rate from a replay no table would accept."""
    reason = None if ctx.pipeline_check else not_comparable(result)
    if reason is not None:
        raise ServerFailed(f"calibration {label} is not comparable: {reason}")
    return result


def calibrate(ctx: MatrixContext, group: str, workload: Workload) -> dict:
    """Find a group's saturation rate, on one system, held for all systems.

    A rate per system would hide the differences this benchmark exists to show. Each rung
    replays the whole workload with its warm-up jobs left out, exactly as a measured run
    does, because a few jobs do not fill the pool at a rate where all of them would. Replaying
    every job alone would take as long as a measured session, so the slowdown compares a
    spread-out sample of them, run alone, with those same jobs in each rung.
    """
    path = ctx.out / "calibration" / f"{group}.json"
    identity = calibration_identity(ctx, workload)
    saved = load_run(path)
    if saved is not None:
        refuse_other_calibration(ctx, path, saved, identity)
        return saved
    settings = ctx.config["calibration"]
    target = settings["target_slowdown"]
    system = ctx.systems[settings["system"]]
    ladder = sorted(rate_ladder(settings, group))
    start = start_rung(settings, group)
    sample = unloaded_sample(workload.jobs, workload.warmup_jobs, settings["unloaded_jobs"])
    seed = ctx.config.get("seed_base", 0)
    logs = ctx.out / "logs" / "calibration" / group
    print(
        f"calibrating {group} on {system.name} over {len(workload.jobs)} jobs of {workload.name}, "
        f"unloaded on {len(sample)} of them",
        flush=True,
    )
    # The ladder takes hours on a full workload, so a resume keeps every job and rung already paid for.
    progress_path = ctx.out / "calibration" / "partial" / f"{group}.json"
    progress = load_run(progress_path)
    if progress is not None:
        refuse_other_calibration(ctx, progress_path, progress, identity)
    else:
        progress = {"identity": identity, "unloaded_jct_s": {}, "points": []}

    def unloaded_jobs(number: int) -> None:
        with serving(ctx, system, logs / f"unloaded.attempt-{number}.log") as (server, _):
            # One job at a time is the unloaded time.
            for job in sample:
                if job.job_id in progress["unloaded_jct_s"]:
                    continue
                check_budget(ctx)
                result = _replay_jobs(ctx, system, server, [replace(job, arrival_s=0.0)], 0, seed, ctx.limits["run_timeout_s"])
                progress["unloaded_jct_s"][job.job_id] = comparable_replay(ctx, result, f"unloaded job {job.job_id}").mean_jct_s
                write_json(progress_path, progress)

    if any(job.job_id not in progress["unloaded_jct_s"] for job in sample):
        calibration_step(ctx, "unloaded jobs", progress, progress_path, unloaded_jobs)
    unloaded_mean = statistics.fmean(progress["unloaded_jct_s"][job.job_id] for job in sample)
    print(f"  unloaded mean JCT {unloaded_mean:.2f}s", flush=True)
    points: list[dict] = progress["points"]
    while (rate := next_rung(ladder, points, target, start)) is not None:
        started = time.monotonic()

        def rung(number: int, rate: float = rate) -> ReplayResult:
            with serving(ctx, system, logs / f"rate-{rate:g}.attempt-{number}.log") as (server, _):
                result = _replay_jobs(
                    ctx, system, server, retime(workload.jobs, rate, seed), workload.warmup_jobs, seed, ctx.limits["run_timeout_s"]
                )
            return comparable_replay(ctx, result, f"rate {rate:g}")

        result = calibration_step(ctx, f"rate {rate:g}", progress, progress_path, rung)
        # The same jobs as the unloaded mean, so the ratio is not one of two job mixes.
        mean = statistics.fmean(result.jct_s[job.job_id] for job in sample)
        points.append(
            {
                "rate_per_s": rate,
                "mean_jct_s": mean,
                "slowdown": mean / unloaded_mean,
                "measured_mean_jct_s": result.mean_jct_s,
                "mean_first_token_s": result.mean_first_token_s,
                "elapsed_s": time.monotonic() - started,
            }
        )
        write_json(progress_path, progress)
        print(f"  rate {rate:g}/s mean JCT {result.mean_jct_s:.2f}s slowdown {points[-1]['slowdown']:.2f}", flush=True)
    found = saturation_rate(points, target)
    points.sort(key=lambda point: point["rate_per_s"])
    measured = len(workload.jobs) - workload.warmup_jobs
    if found["bracketed"]:
        low, high = found["bracket"]
        outcome = (
            f"it crosses between {low['rate_per_s']:g} jobs/s ({low['slowdown']:.2f}x) and {high['rate_per_s']:g} jobs/s "
            f"({high['slowdown']:.2f}x), log-interpolated to {found['saturation_rate_per_s']:.4g} jobs/s"
        )
    elif found["edge"] == "below_ladder":
        outcome = (
            f"NOT BRACKETED: the lowest rung, {points[0]['rate_per_s']:g} jobs/s, already gave {points[0]['slowdown']:.2f}x, "
            f"so saturation lies below this ladder and {found['saturation_rate_per_s']:g} jobs/s is only its edge"
        )
    else:
        outcome = (
            f"NOT BRACKETED: the top rung, {points[-1]['rate_per_s']:g} jobs/s, gave only {points[-1]['slowdown']:.2f}x, "
            f"so {system.name} never reached it and {found['saturation_rate_per_s']:g} jobs/s is only the ladder's edge"
        )
    record = {
        **found,
        # The partial file's, so points from another machine the operator accepted still say so.
        "identity": progress["identity"],
        "failures": progress.get("failures", []),
        "definition": settings.get("definition"),
        "chosen_by": (
            f"the rate where {system.name}'s mean JCT reaches {target}x unloaded, over rungs {', '.join(f'{r:g}' for r in ladder)} "
            f"jobs/s, replaying all {len(workload.jobs)} jobs of {workload.name} with {workload.warmup_jobs} as warm-up "
            f"({measured} measured) and comparing {len(sample)} of them with the same jobs run alone; {outcome}; held fixed for every system"
        ),
        "group": group,
        "system": system.name,
        "workload": workload.name,
        "target_slowdown": target,
        "unloaded_jobs": len(sample),
        "unloaded_job_ids": [job.job_id for job in sample],
        "unloaded_mean_jct_s": unloaded_mean,
        "points": points,
        "pipeline_check": ctx.pipeline_check,
    }
    write_json(path, record)
    return record


def rate_ladder(settings: dict, group: str | None) -> list[float]:
    """A group's rungs: the knee sits elsewhere for a workload with other job sizes."""
    return settings.get("rates_by_group", {}).get(group, settings["rates_per_s"])


def start_rung(settings: dict, group: str | None) -> float:
    """Where a group's ladder walk begins: a rung near the expected knee, or the middle one."""
    ladder = rate_ladder(settings, group)
    start = settings.get("start_rate_by_group", {}).get(group, statistics.median_low(ladder))
    if start not in ladder:
        raise Refused(f"calibration start rate {start} for {group} is not one of its rungs {ladder}")
    return start


def unloaded_sample(jobs: list[TraceJob], warmup: int, count: int) -> list[TraceJob]:
    """Measured jobs spread across the workload, since job size can follow trace order."""
    measured = jobs[warmup:]
    return measured[:: max(1, len(measured) // count)][:count]


def refuse_unbracketed(record: dict, accepted: bool) -> None:
    """Stop before every system runs at a rate that is only the end of a ladder."""
    if record.get("bracketed", True) or accepted:
        return
    raise Refused(
        f"SATURATION NOT FOUND for {record['group']}: {record['chosen_by']}. Nothing else has run, and the rungs are saved. "
        "Widen the ladder for this group in the config and use a new --out directory, or pass --accept-saturation-edge "
        "to run at that edge with every output flagged."
    )


def resolve_rates(ctx: MatrixContext, workloads: list[Workload], needed: set[str], accept_edge: bool = False) -> dict[str, dict]:
    """Each needed workload's rate at 100% load: fixed in the config, or calibrated once per group.

    A group's reference does not depend on --only, so a filter cannot change a rate.
    """
    groups: dict[str, dict] = {}
    rates: dict[str, dict] = {}
    for workload in workloads:
        if workload.name not in needed:
            continue
        if workload.arrival_rate_per_s is not None:
            rates[workload.name] = {"saturation_rate_per_s": workload.arrival_rate_per_s, "bracketed": True, "chosen_by": "fixed in the config"}
            continue
        group = workload.calibration_group
        if group is None:
            raise Refused(f"workload {workload.name} has neither arrival_rate_per_s nor calibration_group")
        if group not in groups:
            members = [w for w in workloads if w.calibration_group == group]
            reference = next((w for w in members if w.calibration_reference), members[0])
            groups[group] = calibrate(ctx, group, reference)
            refuse_unbracketed(groups[group], accept_edge or ctx.pipeline_check)
        rates[workload.name] = groups[group]
    return rates


def cell_rate(saturation: dict, load: str) -> dict:
    """What a run records about its arrivals. The ladder itself stays in the calibration file."""
    fraction = load_fraction(load)
    kept = {key: value for key, value in saturation.items() if key not in ("points", "bracket", "definition", "unloaded_job_ids", "failures")}
    return {**kept, "load": load, "load_fraction": fraction, "arrival_rate_per_s": saturation["saturation_rate_per_s"] * fraction}


def refuse_mixed_results(ctx: MatrixContext, cells: list[Cell], rates: dict[str, dict]) -> None:
    """A saved run from other inputs, or at another rate, must not share a table with these.

    The manifest says the same at a higher level, but a restored directory can lose or replace
    its manifest, and each record is what actually gets skipped.
    """
    for cell in cells:
        saved = load_run(ctx.run_path(cell))
        if not run_finished(saved):
            continue
        for key, value in expected_provenance(ctx, cell).items():
            if saved.get(key) != value:
                raise Refused(f"{cell.label} was recorded with {key} {saved.get(key)}, but this session has {value}; use a new --out directory")
        if cell.workload not in rates:
            continue
        expected = cell_rate(rates[cell.workload], cell.load)["arrival_rate_per_s"]
        recorded = saved["arrival"]["arrival_rate_per_s"]
        if not math.isclose(recorded, expected, rel_tol=1e-9):
            raise Refused(f"{cell.label} ran at {recorded:g} jobs/s, but this calibration gives {expected:g}; use a new --out directory")


def check_prefix_cache(name: str, cached_tokens: int | None, at_least: int = 1) -> None:
    """A baseline whose prefix cache does not work would lose for the wrong reason."""
    if not cached_tokens or cached_tokens < at_least:
        raise Refused(
            f"preflight {name}: a repeated prompt reported cached_tokens={cached_tokens}, wanted at least {at_least}, "
            "so its prefix cache is off or unreported"
        )


def check_versions(systems: dict[str, System], names: list[str]) -> dict[str, str]:
    """Each baseline at its pinned version, before any time is spent. Returns what it found."""
    found = {}
    for name in names:
        expected = systems[name].expected_version()
        if expected is None:
            continue
        installed = systems[name].installed_version()
        if installed != expected:
            raise Refused(f"{name}: installed version {installed}, pinned {expected}; run scripts/setup_baselines.sh")
        found[name] = installed
    return found


def preflight_names(systems: dict[str, System], in_use: set[str], kinds: list[str] | None) -> list[str]:
    """The first system of each kind asked for, in config order."""
    if kinds is None:
        return [name for name in systems if name in in_use]
    unknown = set(kinds) - {system.kind for system in systems.values()}
    if unknown:
        raise Refused(f"--preflight-kinds {sorted(unknown)}: the config has no system of that kind")
    return [next(name for name, system in systems.items() if system.kind == kind) for kind in kinds]


def preflight(ctx: MatrixContext, names: list[str], out: Path) -> None:
    """One short token-id completion through each kind of system, before any long run.

    This is where a wrong flag, a missing venv, or a surprise in a response format shows
    up, minutes into a rented session and not hours. The server's start checks its KV and
    GPU budgets; this checks token ids, ignore_eos, and that a repeated prompt hits its cache.
    """
    seen = set()
    length = ctx.config.get("preflight_output_tokens", 8)
    for name in names:
        system = ctx.systems[name]
        if system.kind in seen:
            continue
        seen.add(system.kind)
        prompt = ContextBuilder(ctx.token_limit, 0, 256).system_prompt()
        payload = {
            "model": ctx.settings.model, "prompt": prompt, "max_tokens": length, "temperature": 0, "stream": True,
            "ignore_eos": True, "return_token_ids": True, "stream_options": {"include_usage": True},
        }

        async def probe(url: str):
            async with httpx.AsyncClient(timeout=ctx.limits["request_timeout_s"]) as client:
                return await _run_turn(client, url + "/v1/completions", payload)

        baseline = system.kind in ("vllm", "sglang")
        with serving(ctx, system, out / "logs" / "preflight" / f"{name}.log") as (server, checks):
            output = asyncio.run(probe(server.url))
            repeated = asyncio.run(probe(server.url)) if baseline else None
            evidence = {}
            if baseline:
                save_metric_names(server, out / "logs" / "preflight" / f"{name}.metrics.txt")
                # Printed, not enforced: when a counter updates depends on the build.
                evidence = {key: value for key, value in system.server_metrics(server.client).items() if "cache_hit" in key}
            if system.translation is not None and checks.get("reported_cpu_tier_gib") is None:
                # Off budget raises. Only a system that says nothing gets through, and it should be heard.
                print(f"WARNING: preflight {name}: its CPU tier size cannot be read back ({system.cpu_tier_source})", flush=True)
        got = output.tokens or (ctx.encode(output.text) if ctx.encode else [])
        source = "server" if output.tokens else "tokenizer"
        cached = _cached_tokens(repeated.usage) if repeated is not None else None
        found = {
            "system": name, "kind": system.kind, "output_tokens": len(got), "asked_tokens": length, "token_source": source,
            "repeat_cached_tokens": cached, "prompt_tokens": len(prompt), "cache_hit_metrics": evidence, "checks": checks,
        }
        write_json(out / "preflight" / f"{name}.json", found)
        print(f"preflight {name}: {found}", flush=True)
        if len(got) != length:
            raise Refused(f"preflight {name}: asked for {length} tokens, got {len(got)} ({source}). ignore_eos did not hold.")
        if baseline and source != "server":
            raise Refused(f"preflight {name}: no token ids came back through return_token_ids at this version")
        if repeated is not None:
            # Every whole block of the prompt should come back. Half is a floor that allows for block rounding.
            check_prefix_cache(name, cached, len(prompt) // 2)


def save_metric_names(server: ManagedServer, path: Path) -> None:
    """The metric names in the report were read from source; this shows what this build serves."""
    try:
        response = server.client.get("/metrics")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"preflight: no /metrics from {server.system.name}: {exc}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text)


def checkpoint_weight_bytes(model: str, dtype: str) -> int:
    """Bytes of weights at the run's precision, from the checkpoint's tensor headers."""
    from safetensors import safe_open

    from layover.config import DTYPE_BYTES
    from layover.weights import resolve_checkpoint

    total = 0
    seen: set[str] = set()
    for path in sorted(resolve_checkpoint(model).glob("*.safetensors")):
        with safe_open(path, framework="pt") as tensors:
            for key in tensors.keys():
                # A tied head shares the embedding's memory.
                if key in seen or key == "lm_head.weight":
                    continue
                seen.add(key)
                count = 1
                for size in tensors.get_slice(key).get_shape():
                    count *= size
                total += count * DTYPE_BYTES[dtype]
    return total


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def system_specs(config: dict) -> list[dict]:
    """Config entries with the shared Layover settings and absolute paths filled in."""
    specs = []
    for raw in config["systems"]:
        spec = dict(raw)
        if spec["kind"] == "layover":
            spec = {**config.get("layover", {}), **spec}
            spec["cost_model"] = str(repo_path(spec["cost_model"]))
        if spec.get("venv"):
            spec["venv"] = str(repo_path(spec["venv"]))
        specs.append(spec)
    return specs


def cost_model_hash(config: dict) -> str | None:
    """The contents of every cost model the matrix prices with; the config holds only their paths."""
    paths = sorted({spec["cost_model"] for spec in system_specs(config) if spec["kind"] == "layover"})
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes() if Path(path).exists() else b"missing")
    return digest.hexdigest()[:16]


def check_cost_model_chunk(config: dict, settings: MatchedSettings, names: set[str]) -> None:
    """Recompute prices must describe the prefill the matrix serves, not another chunk size."""
    for spec in system_specs(config):
        if spec["kind"] != "layover" or spec["name"] not in names:
            continue
        chunk = json.loads(Path(spec["cost_model"]).read_text()).get("chunk")
        if chunk != settings.prefill_chunk:
            raise Refused(
                f"{spec['name']}: {spec['cost_model']} was measured with --chunk {chunk} but the matrix serves "
                f"prefill_chunk={settings.prefill_chunk}; rerun bench/kv_crossover.py with --chunk {settings.prefill_chunk}"
            )


def instance_id() -> str:
    """Which machine this is: a spot replacement gets another id, so a table can say what it spans."""
    override = os.environ.get("LAYOVER_INSTANCE_ID")
    if override:
        return override
    # Local files only: no metadata request leaves this process.
    for path in ("/var/lib/cloud/data/instance-id", "/etc/machine-id"):
        try:
            found = Path(path).read_text().strip()
        except OSError:
            continue
        if found:
            return found
    return socket.gethostname()


def hardware_facts(gpu) -> dict | None:
    """What must match across instances for their runs to share a table."""
    return gpu.device_info() if gpu is not None else None


def manifest_for(config: dict, workloads: list[Workload], cells: list[Cell], pipeline_check: bool) -> dict:
    return {
        "schema": SCHEMA,
        "pipeline_check": pipeline_check,
        "config_hash": stable_hash(config),
        "cost_model_hash": cost_model_hash(config),
        "config": config,
        "first_token_note": FIRST_TOKEN_NOTE,
        "workloads": [
            {
                "name": w.name, "role": w.role, "source": w.source, "trace_hash": w.trace_hash, "trace_path": w.trace_path,
                "systems": w.systems, "warmup_jobs": w.warmup_jobs, "jobs": len(w.jobs), "sweeps": w.sweeps,
                "dropped_jobs": w.dropped_jobs, "trace_meta": w.trace_meta, "load_fractions": w.load_fractions,
                "calibration_group": w.calibration_group,
            }
            for w in workloads
        ],
        "systems": [s["name"] for s in config["systems"]],
        "runs": config["runs"],
        "cells": [asdict(cell) for cell in cells],
    }


def merge_manifest(manifest: dict, earlier: dict | None, selected: list[Cell]) -> dict:
    """What every invocation here asked for, so a narrower one's report is not all gaps, and a
    resume keeps what its predecessors recorded."""
    earlier = earlier or {}
    cells = {tuple(sorted(cell.items())): cell for cell in earlier.get("cells", [])}
    for cell in selected:
        cells.setdefault(tuple(sorted(asdict(cell).items())), asdict(cell))
    workloads = {w["name"]: w for w in earlier.get("workloads", [])}
    # The newest description wins, so a workload built since an earlier preflight has its hash.
    workloads.update({w["name"]: w for w in manifest["workloads"]})
    hosts = {host.get("instance_id"): host for host in earlier.get("hosts", [])}
    host = manifest.get("host")
    if host:
        hosts.setdefault(host.get("instance_id"), host)
    plans = list(earlier.get("plans", []))
    if manifest.get("plan") and manifest["plan"] not in plans:
        plans.append(manifest["plan"])
    return {
        **manifest, "cells": list(cells.values()), "workloads": list(workloads.values()), "hosts": list(hosts.values()), "plans": plans,
        # The first machine's hardware stays the reference for every later resume.
        "hardware": earlier.get("hardware", manifest.get("hardware")),
    }


def check_resume(out: Path, manifest: dict, accept_mixed_hardware: bool = False) -> None:
    """Results from another configuration must never be mixed with these.

    A different instance is fine, which is how a spot replacement resumes. A different GPU or
    driver changes every number, so it stops here unless the operator accepts it; the report then
    flags every table it touches.
    """
    path = out / "manifest.json"
    saved = load_run(path)
    if saved is None:
        return
    for key in ("config_hash", "pipeline_check"):
        if saved.get(key) != manifest.get(key):
            raise Refused(f"{out} holds results with a different {key}; use a new --out directory")
    if saved.get("cost_model_hash") != manifest.get("cost_model_hash"):
        # Placement prices would differ between rows of one table, with nothing to show it.
        raise Refused(
            f"{out} holds results priced by a different cost model; a kv-crossover rerun overwrites it. "
            "Restore the file the saved runs used, or use a new --out directory"
        )
    saved_hashes = {w["name"]: w["trace_hash"] for w in saved["workloads"]}
    for workload in manifest["workloads"]:
        # A workload not seen before, as when a plan grows, has no saved runs to disagree with.
        if workload["name"] in saved_hashes and saved_hashes[workload["name"]] != workload["trace_hash"]:
            raise Refused(f"{out}: the trace for {workload['name']} changed since the saved runs; use a new --out directory")
    if saved.get("hardware") != manifest.get("hardware"):
        message = f"{out} holds runs on {saved.get('hardware')}, and this machine is {manifest.get('hardware')}"
        if not accept_mixed_hardware:
            raise Refused(f"{message}. Their numbers would not be comparable: use a new --out directory, or pass --accept-mixed-hardware to mix them with every table flagged.")
        print(f"WARNING: MIXED HARDWARE: {message}. Every table it touches is flagged.", flush=True)


def host_facts(gpu) -> dict:
    def command_output(args: list[str]) -> str | None:
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=REPO)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    # A rented machine gets the tree without .git; the sync records what it sent.
    synced = REPO / ".layover-run" / "synced_commit"
    if not (REPO / ".git").exists() and synced.exists():
        commit = synced.read_text().strip()
        dirty = "-dirty-" in commit
    else:
        commit = command_output(["git", "rev-parse", "--short", "HEAD"])
        dirty = bool(command_output(["git", "status", "--porcelain"]))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commit": commit,
        "dirty": dirty,
        "gpu": gpu.device_info() if gpu is not None else None,
    }


def save_baseline_freezes(ctx: MatrixContext, names: list[str]) -> None:
    """The exact dependency set of each baseline venv travels with the results."""
    for name in names:
        venv = getattr(ctx.systems[name], "venv", None)
        if venv is None or not (venv / "bin" / "python").exists():
            continue
        done = subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        if done.returncode == 0:
            path = ctx.out / "env" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(done.stdout)


def prepared_jobs(directory: Path) -> list[TraceJob]:
    """The chosen trajectories with their recorded tool gaps, before they are re-executed."""
    from layover.replay.trajectories import load_manifest, load_prepared

    jobs = []
    for task in load_manifest(directory)["tasks"]:
        turns = [
            TraceTurn(turn["new_tokens"], turn["output_tokens"], "tool", turn["recorded_gap_s"] or 0.0)
            for turn in load_prepared(directory, task)["turns"]
        ]
        jobs.append(TraceJob(task, 0.0, [TraceSession(f"{task}/main", turns)], "prepared"))
    return jobs


def planning_jobs(workload: Workload) -> tuple[list[TraceJob], str | None]:
    """The jobs to plan with, and what stands in if the trace does not exist yet.

    The published synthetic shape has a quarter of the real turns, so it is the last resort.
    """
    if workload.jobs or not workload.estimate_jobs:
        return workload.jobs, None
    prepared = repo_path(PREPARED_DIR)
    if (prepared / "manifest.json").exists():
        return prepared_jobs(prepared)[: workload.estimate_jobs], f"jobs of {PREPARED_DIR} stand in, with their recorded tool gaps"
    return generate(replace(SWE_BENCH_SHAPE, jobs=workload.estimate_jobs), 0), "jobs of the synthetic SWE-bench shape stand in"


ASSUMPTION_KEYS = {
    "startup_s", "stop_s", "prefill_tokens_per_s", "decode_tokens_per_s_per_stream", "decode_tokens_per_s_total",
}


def apply_measured_rates(config: dict, path: Path) -> str:
    """Overlay a measured-rates file on a copy of the config's assumptions. Returns where they came from.

    {"source": "what measured these",
     "estimate_assumptions": {"prefill_tokens_per_s": 15000, ...},       any of ASSUMPTION_KEYS
     "by_system": {"vllm": {"prefill_tokens_per_s": 12000}},             optional, same keys
     "expected_saturation_per_s": {"recorded": [0.03, 0.05]}}            optional, [low, high] by group

    A typo in a key would leave an assumed rate standing while the plan claims to be measured.
    """
    raw = json.loads(Path(path).read_text())
    unknown = set(raw) - {"source", "estimate_assumptions", "by_system", "expected_saturation_per_s"}
    if unknown:
        raise Refused(f"{path}: unknown keys {sorted(unknown)}")

    def checked(values: dict, where: str) -> dict:
        wrong = set(values) - ASSUMPTION_KEYS
        if wrong:
            raise Refused(f"{path}: unknown keys in {where}: {sorted(wrong)}; expected some of {sorted(ASSUMPTION_KEYS)}")
        for key, value in values.items():
            positive = not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0
            if not positive and not (key in ("startup_s", "stop_s") and value == 0):
                raise Refused(f"{path}: {where}.{key} must be a positive number, got {value!r}")
        return values

    config["estimate_assumptions"] = {**config["estimate_assumptions"], **checked(raw.get("estimate_assumptions", {}), "estimate_assumptions")}
    unknown = set(raw.get("by_system", {})) - {spec["name"] for spec in config["systems"]}
    if unknown:
        raise Refused(f"{path}: by_system names unknown systems {sorted(unknown)}")
    config["estimate_assumptions_by_system"] = {name: checked(values, f"by_system.{name}") for name, values in raw.get("by_system", {}).items()}
    if "expected_saturation_per_s" in raw:
        for group, pair in raw["expected_saturation_per_s"].items():
            if not (isinstance(pair, list) and len(pair) == 2 and 0 < pair[0] <= pair[1]):
                raise Refused(f"{path}: expected_saturation_per_s.{group} must be [low, high], got {pair!r}")
        config["calibration"] = {**config["calibration"], "expected_saturation_per_s": raw["expected_saturation_per_s"]}
    return f"{raw.get('source', 'no source given')} (from {path})"


def saturation_range(config: dict, workload: Workload, out: Path | None) -> tuple[tuple[float, float], str]:
    """The rate at 100% load this estimate plans on: low, high, and where they come from."""
    if workload.arrival_rate_per_s is not None:
        return (workload.arrival_rate_per_s,) * 2, "fixed in the config"
    group = workload.calibration_group
    saved = load_run(out / "calibration" / f"{group}.json") if out is not None else None
    if saved is not None:
        return (saved["saturation_rate_per_s"],) * 2, "calibrated in this directory"
    calibration = config["calibration"]
    expected = calibration.get("expected_saturation_per_s", {}).get(group)
    if expected:
        return tuple(expected), "expected, not yet calibrated"
    rate = start_rung(calibration, group)
    return (rate, rate), "the ladder's start rung, not yet calibrated"


def calibration_estimate(config: dict, workload: Workload, jobs: list[TraceJob], assumed: dict, pool_tokens: int | None, out: Path | None) -> tuple[float, float]:
    """Hours a group's ladder takes: bracketed by the first neighbour, or walked to its far end."""
    calibration = config["calibration"]
    group = workload.calibration_group
    if out is not None and (out / "calibration" / f"{group}.json").exists():
        return 0.0, 0.0
    sample = unloaded_sample(jobs, workload.warmup_jobs, calibration["unloaded_jobs"])
    # One server takes every unloaded job, so one startup.
    idle = assumed["startup_s"] + assumed.get("stop_s", 0.0)
    unloaded = sum(estimate_run_s([job], 1.0, assumed) - idle for job in sample) + idle
    ladder = sorted(rate_ladder(calibration, group))
    start = ladder.index(start_rung(calibration, group))

    def cost(rates: list[float], recompute: bool) -> float:
        return sum(estimate_run_s(jobs, rate, assumed, recompute=recompute, pool_tokens=pool_tokens) for rate in rates)

    near = ladder[start:start + 2] if start + 1 < len(ladder) else ladder[start - 1:start + 1]
    return unloaded + cost(near, False), unloaded + max(cost(ladder[: start + 1], True), cost(ladder[start:], True))


def estimate_plan(
    config: dict, systems: dict[str, System], workloads: list[Workload], plans: dict, cells: list[Cell],
    pool_tokens: int | None = None, out: Path | None = None,
) -> dict:
    """Runs and a low and high duration for one set of cells, in seconds.

    Low: each run reuses its KV across tool calls, unless the system never does, at the higher
    expected saturation rate, and the ladder brackets the target on its first neighbour. High:
    every turn prefills its whole context, at the lower expected saturation rate, and the ladder
    walks to its far end. No retries, and none of make reproduce's other stages.
    """
    by_name = {w.name: w for w in workloads}
    by_system = config.get("estimate_assumptions_by_system", {})
    base = config["estimate_assumptions"]
    memo: dict = {}

    def run_range(cell: Cell) -> tuple[float, float] | None:
        system = systems[cell.system]
        key = (cell.workload, cell.load, system.recomputes_context, cell.system if cell.system in by_system else None)
        if key not in memo:
            jobs = plans[cell.workload][0]
            workload = by_name[cell.workload]
            if not jobs:
                memo[key] = None
            else:
                assumed = {**base, **by_system.get(cell.system, {})}
                (slow_rate, fast_rate), _ = saturation_range(config, workload, out)
                fraction = load_fraction(cell.load)
                memo[key] = (
                    estimate_run_s(jobs, fast_rate * fraction, assumed, recompute=system.recomputes_context, pool_tokens=pool_tokens),
                    estimate_run_s(jobs, slow_rate * fraction, assumed, recompute=True, pool_tokens=pool_tokens),
                )
        return memo[key]

    ranges = {cell: run_range(cell) for cell in cells}
    low = sum(r[0] for r in ranges.values() if r)
    high = sum(r[1] for r in ranges.values() if r)
    chosen = {cell.workload for cell in cells}
    groups = {by_name[name].calibration_group for name in chosen if not by_name[name].arrival_rate_per_s and plans[name][0]}
    calibration_low, calibration_high = 0.0, 0.0
    for group in groups:
        members = [w for w in workloads if w.calibration_group == group]
        reference = next((w for w in members if w.calibration_reference), members[0])
        assumed = {**base, **by_system.get(config["calibration"]["system"], {})}
        pair = calibration_estimate(config, reference, plans[reference.name][0], assumed, pool_tokens, out)
        calibration_low += pair[0]
        calibration_high += pair[1]
    preflight_s = base["startup_s"] * len({systems[cell.system].kind for cell in cells})
    return {
        "runs": len(cells), "ranges": ranges, "runs_low_s": low, "runs_high_s": high,
        "calibration_low_s": calibration_low, "calibration_high_s": calibration_high, "preflight_s": preflight_s,
        "low_s": low + calibration_low + preflight_s, "high_s": high + calibration_high + preflight_s,
    }


def dry_run(
    config: dict, settings: MatchedSettings, systems: dict[str, System], workloads: list[Workload], cells: list[Cell],
    max_hours: float | None = None, pool_tokens: int | None = None, out: Path | None = None, rates_source: str | None = None,
    plan_name: str | None = None,
) -> float:
    """Print every command and the planned time. Returns the low estimate in seconds.

    A run's estimate comes from the trace's own tool durations and token counts, at the
    assumed token rates. It is a range: see estimate_plan for what each end assumes.
    """
    assumed = config["estimate_assumptions"]
    print("Commands:")
    for name, system in systems.items():
        print(f"\n# {name}\n{' '.join(system.command('127.0.0.1', 8000))}")
    plans: dict[str, tuple[list[TraceJob], str | None]] = {w.name: planning_jobs(w) for w in workloads}
    plan = estimate_plan(config, systems, workloads, plans, cells, pool_tokens, out)
    chosen = {cell.workload for cell in cells}
    print("\nPlan (estimate from trace durations and token rates, not a measurement):")
    print(f"  {len(cells)} runs over {len(chosen)} workloads, {len({cell.system for cell in cells})} systems")
    if rates_source:
        print(f"  token rates MEASURED, {rates_source}")
    else:
        print("  token rates ASSUMED in the config; the GPU smoke test can refine them through --measured-rates")
    print(f"  rates: {assumed}" + (f", by system {config['estimate_assumptions_by_system']}" if config.get("estimate_assumptions_by_system") else ""))
    missing = [f"{w.name}: {w.unavailable}" for w in workloads if w.name in chosen and not plans[w.name][0]]
    timeout_s = config["limits"]["run_timeout_s"]
    for workload in workloads:
        if workload.name not in chosen or not plans[workload.name][0]:
            continue
        (slow_rate, fast_rate), origin = saturation_range(config, workload, out)
        jobs, proxy = plans[workload.name]
        source = f"NO TRACE YET: {len(jobs)} {proxy}" if proxy else f"source {workload.source}"
        print(f"  {workload.name}: {source}; 100% load is {slow_rate:g} to {fast_rate:g} jobs/s, {origin}")
        for load in sorted({cell.load for cell in cells if cell.workload == workload.name}, key=load_fraction):
            found = {cell: r for cell, r in plan["ranges"].items() if (cell.workload, cell.load) == (workload.name, load)}
            print(f"    {load}: {load_fraction(load) * slow_rate:.3g} to {load_fraction(load) * fast_rate:.3g} jobs/s, {len(found)} runs")
            for system in {cell.system: None for cell in found}:
                durations = [r for cell, r in found.items() if cell.system == system]
                print(f"      {system}: {len(durations)} runs x {min(r[0] for r in durations) / 60:.1f} to {max(r[1] for r in durations) / 60:.1f} min")
            if max(r[1] for r in found.values()) > timeout_s:
                print(f"      WARNING: the slow end is over limits.run_timeout_s={timeout_s}; such a run would fail, twice")
    for line in missing:
        print(f"  NOT IN THE ESTIMATE, {line}")
    for name in sorted(name for name in {cell.system for cell in cells} if systems[name].recomputes_context):
        print(f"  {name} has no prefix cache, so it recomputes after every tool call: its runs are counted at the slow end throughout")
    print(
        f"  preflight about {plan['preflight_s'] / 60:.0f} min, "
        f"calibration about {plan['calibration_low_s'] / 60:.0f} to {plan['calibration_high_s'] / 60:.0f} min "
        f"({plan['calibration_low_s'] / 3600:.1f} to {plan['calibration_high_s'] / 3600:.1f} hours)"
    )
    print(
        f"  estimated total: {plan['low_s'] / 3600:.1f} hours, up to {plan['high_s'] / 3600:.1f} hours if every turn recomputes its context"
        + (" (excludes the missing traces above)" if missing else "")
    )
    print("  Not included: retries, and the setup, trace, crossover, decode-bandwidth and transfer stages of make reproduce.")
    budget = config.get("plans", {}).get(plan_name, {}).get("budget_hours") if plan_name else None
    if budget is not None:
        other = config["plans"][plan_name].get("other_stages_hours", 0.0)
        room = budget - other
        print(f"  Budget: {budget:g} hours less about {other:g} hours of other stages leaves {room:g} hours for this plan.")
        if plan["high_s"] / 3600 > room:
            print(f"  OVER BUDGET AT THE SLOW END: {plan['high_s'] / 3600:.1f} hours is {plan['high_s'] / 3600 - room:.1f} hours over.")
        if plan["low_s"] / 3600 > room:
            print(f"  OVER BUDGET EVEN AT THE LOW END: {plan['low_s'] / 3600:.1f} hours is {plan['low_s'] / 3600 - room:.1f} hours over.")
    if config.get("plans"):
        print("\nPlans, each one flag (--plan NAME). Same assumptions; matrix hours are a range, not a measurement:")
        for name, entry in config["plans"].items():
            priced = estimate_plan(config, systems, workloads, plans, plan_cells(config, workloads, name), pool_tokens, out)
            print(f"  {name}: {priced['runs']} runs, {priced['low_s'] / 3600:.1f} to {priced['high_s'] / 3600:.1f} matrix hours. {entry['description']}")
    if max_hours is not None and plan["high_s"] > max_hours * 3600:
        print(f"  WARNING: --max-hours {max_hours:g} may end the session before every run; what is left stays listed as missing")
    return plan["low_s"]


def simulate_point(
    config: dict, settings: MatchedSettings, cfg: ModelConfig, jobs: list[TraceJob], warmup: int, rate: float, seed: int,
    policies: tuple[str, ...] = ("combined", "swap", "discard"),
) -> dict:
    """One sweep point on the real scheduler and a virtual clock, at the assumed rates. Never a result.

    Whether a point can separate systems is a matter of how far its load overflows the
    pool, which this shows before any GPU time is spent on it. The discard row is layover-discard:
    finished blocks stay in the radix tree until pressure evicts them, the nearest stand-in
    for a prefix cache with no working CPU tier.
    """
    from layover.agent.cost_model import CostModel, Policy
    from layover.agent.scheduler import AgentConfig
    from layover.engine import EngineConfig
    from layover.replay.simulate import simulate

    assumed = config["estimate_assumptions"]
    swap = [0.001, config["sweeps"]["selection"]["swap_s_per_token"]]
    costs = CostModel(swap, swap, [0.0, 1 / assumed["prefill_tokens_per_s"]])
    block_bytes = cfg.kv_bytes_per_token(settings.dtype) * BLOCK_TOKENS
    # By the bytes a record occupies, as layover/server.py sizes the tier this models.
    record_bytes = offload_record_bytes(cfg, config.get("layover", {}).get("offload_dtype"))
    # Step time at batch 1 and at a full batch, from the two decode rates.
    full_step_s = settings.max_running / assumed["decode_tokens_per_s_total"]
    per_seq_s = (full_step_s - 1 / assumed["decode_tokens_per_s_per_stream"]) / (settings.max_running - 1)
    found: dict = {}
    for policy in policies:
        run = simulate(
            retime(jobs, rate, seed), costs, int(settings.kv_gib * GIB / block_bytes),
            AgentConfig(policy=Policy(policy), cpu_capacity_blocks=int(settings.cpu_offload_gib * GIB / record_bytes)),
            EngineConfig(settings.max_running, settings.prefill_chunk, settings.prefill_chunk),
            decode_step_s=full_step_s - settings.max_running * per_seq_s, decode_per_seq_s=per_seq_s,
            warmup_jobs=warmup, seed=seed, system_tokens=config.get("system_tokens", 256),
        )
        found[f"{policy}_mean_jct_s"] = run.result.mean_jct_s
        found[f"{policy}_early_demotions"] = run.scheduler.stats.early_demotions
    found["gap"] = found["discard_mean_jct_s"] / found["combined_mean_jct_s"]
    return found


def point_separates(rows: list[dict], min_gap: float) -> bool:
    """A gap in every run, and early demotions in the median run, which is what a table cell shows."""
    return (
        all(row["gap"] >= min_gap for row in rows)
        and statistics.median(row["combined_early_demotions"] for row in rows) > 0
    )


def print_sweep_separation(config: dict, settings: MatchedSettings, cfg: ModelConfig, workloads: list[Workload]) -> None:
    """The table a sweep's job count and points are chosen from."""
    selection = config["sweeps"]["selection"]
    seeds = [config.get("seed_base", 0) + run for run in range(config["runs"])]
    print("SIMULATED at the assumed rates in the config, not a measurement. gap is prefix-cache-only JCT over combined JCT.")
    for workload in workloads:
        if workload.role != "sweep":
            continue
        for rate in rate_ladder(config["calibration"], workload.calibration_group):
            rows = [simulate_point(config, settings, cfg, workload.jobs, workload.warmup_jobs, rate, seed) for seed in seeds]
            print(
                f"{workload.name} at {rate:g}/s: gap {min(r['gap'] for r in rows):.2f} to {max(r['gap'] for r in rows):.2f}, "
                f"combined {statistics.fmean(r['combined_mean_jct_s'] for r in rows):.0f} s, "
                f"swap {statistics.fmean(r['swap_mean_jct_s'] for r in rows):.0f} s, "
                f"discard {statistics.fmean(r['discard_mean_jct_s'] for r in rows):.0f} s, "
                f"early demotions {min(r['combined_early_demotions'] for r in rows)} to {max(r['combined_early_demotions'] for r in rows)}: "
                + ("separates" if point_separates(rows, selection["min_gap"]) else "does not separate"),
                flush=True,
            )


def recorded_attempt_s(out: Path) -> float:
    """The time the run records account for. They travel with the ledger but are written apart from it."""
    total = 0.0
    for path in (out / "runs").rglob("run-*.json") if (out / "runs").exists() else []:
        total += sum(attempt.get("elapsed_s", 0.0) for attempt in (load_run(path) or {}).get("attempts", []))
    return total


class ElapsedLedger(threading.Thread):
    """Matrix hours spent in one results directory, kept in it so they outlive this machine.

    A spot reclaim gives no chance to save, hence the heartbeat: a kill loses at most one period
    here, or the fetch period once the machine is gone. A lost or torn ledger cannot hand out a
    fresh budget either, because the run records put a floor under what it starts from.
    """

    def __init__(self, out: Path, instance_id: str | None, max_hours: float | None, period_s: float = HEARTBEAT_S):
        super().__init__(daemon=True, name="elapsed-ledger")
        self.path = Path(out) / "budget.json"
        saved = load_run(self.path) or {}
        self.earlier_s = max(float(saved.get("elapsed_s", 0.0)), recorded_attempt_s(Path(out)))
        self.sessions = list(saved.get("sessions", []))
        self.entry = {"instance_id": instance_id, "started_unix": time.time(), "elapsed_s": 0.0}
        self.max_hours = max_hours
        self.period_s = period_s
        self.started_at = time.monotonic()
        self.halted = threading.Event()
        self.save()

    def save(self) -> None:
        self.entry["elapsed_s"] = time.monotonic() - self.started_at
        write_json(
            self.path,
            {"elapsed_s": self.earlier_s + self.entry["elapsed_s"], "max_hours": self.max_hours, "sessions": [*self.sessions, self.entry]},
        )

    def deadline(self) -> float | None:
        """When, on this machine's clock, the budget runs out. Already past if earlier sessions spent it."""
        return None if self.max_hours is None else self.started_at + self.max_hours * 3600 - self.earlier_s

    def run(self) -> None:
        while not self.halted.wait(self.period_s):
            try:
                self.save()
            except OSError as exc:  # A full disk must not end a run; the next write tries again.
                print(f"WARNING: could not save {self.path}: {exc}", flush=True)

    def finish(self) -> None:
        self.halted.set()
        if self.is_alive():
            self.join(timeout=15)
        self.save()


def write_session(out: Path, **facts) -> None:
    """How the latest invocation ended. Each one replaces the last, since a resume reconsiders all of it."""
    write_json(Path(out) / "session.json", {"budget_skipped": [], "stopped_matrix": None, **facts, "finished_unix": time.time()})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO / "bench" / "configs" / "matrix.json")
    p.add_argument("--out", type=Path, default=Path("results/matrix"))
    p.add_argument("--dry-run", action="store_true", help="print every command and the estimated time, run nothing")
    p.add_argument("--only", nargs="+", help="narrow the plan: shell patterns over workload/system/load/run, for example 'headline-*/vllm/load100/*'")
    p.add_argument("--plan", help="which of the config's plans to run, default its default_plan; --dry-run prices them all")
    p.add_argument("--measured-rates", type=Path, help="with --dry-run: a JSON file of measured token rates to plan with instead of the config's assumptions")
    p.add_argument("--accept-saturation-edge", action="store_true", help="run even though calibration never bracketed vLLM's saturation; every output says so")
    p.add_argument("--max-hours", type=float, help="stop cleanly after the current run once this directory's results hold this many matrix hours, counting earlier sessions and machines")
    p.add_argument("--pipeline-check", action="store_true", help="test the pipeline; every output says it is not a result")
    p.add_argument("--no-gpu", action="store_true", help="with --pipeline-check: run without NVML")
    p.add_argument("--drop-over-context", action="store_true", help="leave out jobs over the context limit, and say so")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="launch each kind of system once, then stop")
    p.add_argument("--print-baseline-kinds", action="store_true", help="print the baseline kinds the plan runs, vllm or sglang, one per line, and stop")
    p.add_argument("--preflight-kinds", nargs="+", help="with --preflight-only: only these kinds, for example vllm sglang, which need no Layover cost model")
    p.add_argument("--accept-mixed-hardware", action="store_true", help="resume on a different GPU or driver; the report flags every table it touches")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--sweep-separation", action="store_true", help="simulate each sweep point at each rung, run nothing on a server")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.no_gpu and not args.pipeline_check:
        raise Refused("--no-gpu measures nothing about a GPU; it is allowed only with --pipeline-check")
    config = load_config(args.config)
    if args.measured_rates and not args.dry_run:
        raise Refused("--measured-rates only changes a --dry-run's estimate")
    if args.preflight_kinds and not args.preflight_only:
        raise Refused("--preflight-kinds means something only with --preflight-only")
    plan_name = args.plan or config.get("default_plan")
    if plan_name:
        plan_groups(config, plan_name)
    config = effective_config(config, plan_name)
    if args.print_baseline_kinds:
        # What the baselines stage installs: a plan without SGLang does not pay for its install.
        names = {system for group in plan_groups(config, plan_name) for system in group.get("systems", [])} if plan_name else None
        kinds = {spec["kind"] for spec in config["systems"] if names is None or spec["name"] in names}
        print("\n".join(kind for kind in ("vllm", "sglang") if kind in kinds))
        return 0
    settings = MatchedSettings.from_config(config["matched"])
    rates_source = apply_measured_rates(config, args.measured_rates) if args.measured_rates else None
    if not args.pipeline_check and settings.device != "cuda":
        raise Refused(f"device {settings.device!r} gives no comparable numbers; use --pipeline-check to test the pipeline")
    cfg = ModelConfig.from_pretrained(settings.model)
    token_bytes = cfg.kv_bytes_per_token(settings.dtype)
    if settings.max_context_tokens * token_bytes > settings.kv_gib * GIB:
        raise Refused(f"max_context_tokens={settings.max_context_tokens} does not fit in a {settings.kv_gib} GiB KV budget")
    systems = {spec["name"]: build_system(spec, settings, token_bytes, cfg) for spec in system_specs(config)}
    command_systems = [name for name, system in systems.items() if system.kind == "command"]
    if command_systems and not args.pipeline_check:
        raise Refused(f"{command_systems} are stand-in commands, allowed only with --pipeline-check")
    # A dry run prices every plan, and a separation table needs the sweeps. Anything else loads what it runs.
    roles = plan_roles(config, plan_name) if plan_name and not (args.dry_run or args.sweep_separation) else None
    workloads = load_workloads(
        config, settings, args.pipeline_check, args.drop_over_context, args.dry_run or args.preflight_only or args.sweep_separation, roles
    )
    if args.sweep_separation:
        print_sweep_separation(config, settings, cfg, workloads)
        return 0
    cells = plan_cells(config, workloads, plan_name) if plan_name else build_cells(workloads, config["runs"])
    selected = select_cells(cells, args.only)
    if not selected:
        raise Refused(f"--only {args.only} matches no run")
    if args.dry_run:
        dry_run(config, settings, systems, workloads, selected, args.max_hours, int(settings.kv_gib * GIB / token_bytes), args.out, rates_source, plan_name)
        return 0

    in_use = {cell.system for cell in selected}
    to_preflight = preflight_names(systems, in_use, args.preflight_kinds) if args.preflight_only else []
    if args.preflight_only:
        in_use = set(to_preflight)
    for name in in_use:
        cost_model = getattr(systems[name], "cost_model", None)
        if cost_model is not None and not Path(cost_model).exists():
            raise Refused(f"{name}: cost model {cost_model} does not exist; run bench/kv_crossover.py on this machine first")
    if not args.pipeline_check:
        check_cost_model_chunk(config, settings, in_use)

    out = args.out
    versions = check_versions(systems, sorted(in_use))
    gpu = None if args.no_gpu else NvmlBackend(config.get("gpu_index", 0))
    manifest = manifest_for(config, workloads, selected, args.pipeline_check)
    manifest["plan"] = plan_name
    manifest["hardware"] = hardware_facts(gpu)
    # A preflight may precede the trace, so it leaves nothing a later run must agree with.
    if not args.preflight_only:
        check_resume(out, manifest, args.accept_mixed_hardware)
    facts = host_facts(gpu)
    instance = {
        "instance_id": instance_id(), "commit": facts["commit"], "dirty": facts["dirty"],
        "gpu_name": (facts["gpu"] or {}).get("name"), "gpu_driver": (facts["gpu"] or {}).get("driver"),
    }
    manifest["host"] = {**facts, "baseline_versions": versions, "instance_id": instance["instance_id"]}
    if not args.preflight_only:
        write_json(out / "manifest.json", merge_manifest(manifest, load_run(out / "manifest.json"), selected))
    if args.pipeline_check:
        print(PIPELINE_BANNER, flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.model)
    token_limit = min(cfg.vocab_size, len(tokenizer))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    weights = checkpoint_weight_bytes(settings.model, settings.dtype) if gpu is not None else None
    ctx = MatrixContext(
        config, settings, systems, {w.name: w for w in workloads}, out, args.pipeline_check, gpu, weights, token_limit, encode, instance,
    )
    save_baseline_freezes(ctx, list(systems))

    # SIGTERM from a closing session becomes an exception, so servers get stopped.
    def on_terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_terminate)
    ctx.accept_mixed_hardware = args.accept_mixed_hardware
    if args.preflight_only:
        try:
            preflight(ctx, to_preflight, out)
        except GpuNotReleased as exc:
            print(f"STOPPED: {exc}", file=sys.stderr, flush=True)
            return 3
        except KeyboardInterrupt:
            print("Interrupted. Rerun the same command to resume.", file=sys.stderr, flush=True)
            return 130
        return 0
    ledger = ElapsedLedger(out, instance["instance_id"], args.max_hours)
    ledger.start()
    ctx.deadline = ledger.deadline()
    code = 0
    fatal: BaseException | None = None
    try:
        code = run_all(ctx, args, workloads, selected, ctx.deadline)
    except GpuNotReleased as exc:
        print(f"STOPPED: {exc}", file=sys.stderr, flush=True)
        write_session(out, stopped=f"GpuNotReleased: {exc}")
        code = 3
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr, flush=True)
        write_session(out, stopped="interrupted")
        code = 130
    except (Exception, Refused) as exc:
        # A refusal or a crash ends the session, but the report should still say what is there and why.
        write_session(out, stopped=f"{type(exc).__name__}: {exc}")
        fatal = exc
    finally:
        ledger.finish()
    if not args.no_report:
        from bench.report import write_report

        print(f"report: {write_report(out)}", flush=True)
    if fatal is not None:
        raise fatal
    return code


def run_all(ctx: MatrixContext, args, workloads: list[Workload], selected: list[Cell], deadline: float | None) -> int:
    pending = [cell for cell in selected if not run_finished(load_run(ctx.run_path(cell)))]
    print(f"{len(selected)} runs selected, {len(selected) - len(pending)} already complete", flush=True)
    needed = {cell.workload for cell in pending}
    # Before any server starts: a resume that would be refused should not spend a startup first.
    refuse_mixed_results(ctx, selected, {})
    if pending and deadline is not None and time.monotonic() >= deadline:
        # An earlier session or machine spent it, so not even a preflight is left to pay for.
        print(f"Budget already spent: skipping {len(pending)} runs.", flush=True)
        write_session(ctx.out, budget_skipped=[cell.label for cell in pending])
        return 1
    if not args.skip_preflight and pending:
        preflight(ctx, sorted({cell.system for cell in pending}, key=list(ctx.systems).index), ctx.out)
    try:
        rates = resolve_rates(ctx, workloads, needed, getattr(args, "accept_saturation_edge", False)) if pending else {}
    except BudgetReached:
        print(f"Budget reached during calibration: skipping {len(pending)} runs. What it measured is saved.", flush=True)
        write_session(ctx.out, budget_skipped=[cell.label for cell in pending])
        return 1
    except CalibrationFailed as exc:
        print(f"CALIBRATION FAILED: {exc}. What it measured is saved; rerun to resume.", flush=True)
        write_session(ctx.out, calibration_failed={"reason": str(exc), "skipped": [cell.label for cell in pending]})
        return 1
    refuse_mixed_results(ctx, selected, rates)

    skipped: list[str] = []
    stopped: dict | None = None
    # Counted afresh in each invocation: a resume may be on a machine, or with a fix, that cures it.
    streak: list[str] = []
    limit = ctx.limits.get("max_consecutive_failures", MAX_CONSECUTIVE_FAILURES)
    failed = 0
    for index, cell in enumerate(pending):
        if stopped is not None:
            stopped["skipped"].append(cell.label)
            continue
        if deadline is not None and time.monotonic() >= deadline:
            skipped.append(cell.label)
            continue
        rate = cell_rate(rates[cell.workload], cell.load)
        print(f"[{index + 1}/{len(pending)}] {cell.label} at {rate['arrival_rate_per_s']:.4g} jobs/s", flush=True)
        record = run_cell(ctx, cell, rate)
        if record["status"] == "ok":
            metrics = record["result"]["metrics"]
            print(f"  ok: mean JCT {metrics['mean_jct_s']:.2f}s, p95 {metrics['p95_jct_s']:.2f}s", flush=True)
            streak = []
            continue
        failed += 1
        error = record["attempts"][-1]["error"]
        print(f"  FAILED after {len(record['attempts'])} attempts: {error[:300]}", flush=True)
        streak.append(cell.label)
        if len(streak) >= limit:
            print(f"  STOPPING the matrix: {len(streak)} cells in a row failed", flush=True)
            stopped = {"reason": f"{len(streak)} cells in a row failed; the last: {error[:500]}", "failed": list(streak), "skipped": []}
    if skipped:
        print(f"Budget reached: skipping {len(skipped)} runs.", flush=True)
    write_session(ctx.out, budget_skipped=skipped, stopped_matrix=stopped)
    left_out = len(stopped["skipped"]) if stopped else 0
    print(f"Done: {len(pending) - failed - len(skipped) - left_out} ok, {failed} failed, {len(skipped)} skipped by budget, {left_out} skipped after the stop.", flush=True)
    return 1 if failed or skipped or stopped else 0


if __name__ == "__main__":
    sys.exit(main())

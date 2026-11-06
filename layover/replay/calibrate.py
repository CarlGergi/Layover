"""Fit the synthetic generator to the per-job durations Continuum published.

What the data can say. Continuum's results.json gives one number per job: last request
departure minus first request arrival, for Llama 70B on 4 H100s under their serving stack
(Li et al., ICLR 2026, arXiv 2511.02230). That number mixes turns, tool time, prefill,
decode, and queueing. Their published tool time is about 10 s per job against a median job
of 180 s, so tool latency cannot be recovered from it, and a job's decode time only fixes
the product of output length and step time. So the tool-time and token statistics stay at
their published values, and three parameters are fitted by running generated traces
through the production scheduler on a virtual clock (layover/replay/simulate.py):

    decode_step_s, prefill_s_per_token: their hardware's speed. Nuisance parameters that
        describe Llama 70B on 4 H100s and are never applied to a Qwen3 replay.
    job_scale_sigma: per-job spread of output length, the one thing not published. It
        also absorbs whatever else spreads their durations that this model lacks, and it
        is the only fitted number a generated trace inherits.

Arrivals are not fitted: the rate comes from num_jobs / total_duration. Whether their run
used a fixed worker pool or Poisson arrivals is not in the results; the replay here is
Poisson. Every statement of this kind also goes in the output file's "means" and
"does_not_mean".

Both runs pile up just above 900 s (16% of jobs for vLLM), which looks like a client-side
limit rather than workload, so durations are compared after censoring at censor_s.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from layover.agent.cost_model import CostModel, Policy
from layover.agent.scheduler import AgentConfig
from layover.engine import EngineConfig
from layover.replay.simulate import simulate
from layover.replay.trace import CALIBRATION_SCHEMA, SWE_BENCH_SHAPE, SyntheticWorkload, generate

REPOSITORY = "https://github.com/Hanchenli/vllm-continuum"
PINNED_COMMIT = "316a58794a6ff86b216e579b74fd56ed0c5a911f"
PUBLISHED_FILES = {
    "continuum": (
        "continuum_exp/swebench_llama70B_continuum/results.json",
        "5089094f861ead1f8fed855cbe5fb80b1f5e10032cc493579e7a94c329213df6",
    ),
    "vllm": (
        "continuum_exp/swebench_llama70B_vllm/results.json",
        "afa74314b02bc419df3ef4ebcaf4b504cbce17f808b1414360000d7c07961b4d",
    ),
}
# The closest Layover policy to each run's: a kept KV for Continuum, a dropped one for vLLM.
POLICIES = {"continuum": Policy.KEEP_TTL, "vllm": Policy.DISCARD}
DEFAULT_CENSOR_S = 850.0

# Llama 70B on 4 H100s. KV memory is what is left of 80 GB cards at vLLM's default 0.9
# utilization after 140 GB of BF16 weights: an assumption, as their settings are not published.
LLAMA70B_KV_BYTES_PER_TOKEN = 2 * 80 * 8 * 128 * 2
ASSUMED_KV_BYTES = 4 * 80e9 * 0.9 - 140e9


@dataclass(frozen=True)
class Published:
    target: str
    durations: list[float]
    total_duration_s: float
    sha256: str

    @property
    def mean_concurrency(self) -> float:
        return sum(self.durations) / self.total_duration_s

    @property
    def arrival_rate_per_s(self) -> float:
        return len(self.durations) / self.total_duration_s


def fetch_published(target: str, cache_dir: Path) -> Path:
    """Download one results file at the pinned commit, unless already cached."""
    relative, digest = PUBLISHED_FILES[target]
    path = Path(cache_dir) / PINNED_COMMIT[:12] / f"{target}.json"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/Hanchenli/vllm-continuum/{PINNED_COMMIT}/{relative}"
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != digest:
            raise ValueError(f"{url}: content does not match the pinned sha256 {digest}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def load_published(target: str, path: str | Path, verify: bool = True) -> Published:
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify and digest != PUBLISHED_FILES[target][1]:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {PUBLISHED_FILES[target][1]}")
    raw = json.loads(data)
    by_id = raw["job_durations"]
    return Published(target, [by_id[k] for k in sorted(by_id, key=int)], raw["total_duration"], digest)


def split_jobs(values: Sequence[float], seed: int) -> tuple[list[float], list[float]]:
    """A random half split. Random, because durations trend with job id (see report)."""
    order = list(range(len(values)))
    random.Random(seed).shuffle(order)
    middle = len(order) // 2
    return [values[i] for i in order[:middle]], [values[i] for i in order[middle:]]


def _censored(values: Sequence[float], censor_s: float | None) -> list[float]:
    return sorted(values) if censor_s is None else sorted(min(v, censor_s) for v in values)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def log_wasserstein(a: Sequence[float], b: Sequence[float], censor_s: float | None = None, points: int = 400) -> float:
    """1-Wasserstein distance between log durations: the mean gap between quantile curves.

    A value of 0.1 means durations are off by about 10% on average, in either direction.
    """
    x, y = _censored(a, censor_s), _censored(b, censor_s)
    grid = [(i + 0.5) / points for i in range(points)]
    return statistics.fmean(abs(math.log(_quantile(x, p)) - math.log(_quantile(y, p))) for p in grid)


def ks_distance(a: Sequence[float], b: Sequence[float], censor_s: float | None = None) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, looking only below the censoring point."""
    x, y = _censored(a, censor_s), _censored(b, censor_s)
    worst, i, j = 0.0, 0, 0
    for point in sorted(set(x) | set(y)):
        if censor_s is not None and point >= censor_s:
            break
        while i < len(x) and x[i] <= point:
            i += 1
        while j < len(y) and y[j] <= point:
            j += 1
        worst = max(worst, abs(i / len(x) - j / len(y)))
    return worst


def ks_critical(n: int, m: int, alpha_coefficient: float = 1.358) -> float:
    """Asymptotic 5% critical value, which assumes independent samples from fixed laws."""
    return alpha_coefficient * math.sqrt((n + m) / (n * m))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


@dataclass(frozen=True)
class Params:
    decode_step_s: float
    prefill_s_per_token: float
    job_scale_sigma: float

    def as_list(self) -> list[float]:
        return [self.decode_step_s, self.prefill_s_per_token, self.job_scale_sigma]


# (lower, upper, searched on a log scale)
BOUNDS = ((0.005, 0.5, True), (1e-5, 2e-3, True), (0.0, 1.5, False))


@dataclass(frozen=True)
class Scenario:
    """Everything a simulated run fixes, as opposed to fits."""

    arrival_rate_per_s: float
    jobs: int = 500
    policy: Policy = Policy.KEEP_TTL
    workload: SyntheticWorkload = SWE_BENCH_SHAPE
    pool_tokens: int = int(ASSUMED_KV_BYTES / LLAMA70B_KV_BYTES_PER_TOKEN)
    max_batch: int = 256
    prefill_chunk: int = 4096
    swap_s_per_token: float = LLAMA70B_KV_BYTES_PER_TOKEN / 50e9
    # Tokens are simulated in groups of this size with times scaled up to match, which keeps
    # a fit tractable. The output file reports the same parameters at a finer grain.
    coarsen: int = 8


def simulate_durations(params: Params, scenario: Scenario, seed: int) -> list[float]:
    """Job durations for one generated trace on a virtual clock. Deterministic in seed."""
    return _simulate(params, scenario, seed)[0]


def _simulate(params: Params, scenario: Scenario, seed: int) -> tuple[list[float], float]:
    k = scenario.coarsen
    w = scenario.workload
    coarse = replace(
        w,
        jobs=scenario.jobs,
        arrival_rate_per_s=scenario.arrival_rate_per_s,
        job_scale_sigma=params.job_scale_sigma,
        tokens_per_job_mean=w.tokens_per_job_mean / k,
        tokens_per_job_sd=w.tokens_per_job_sd / k,
        output_tokens_mean=w.output_tokens_mean / k,
    )
    jobs = generate(coarse, seed)
    swap = [0.001, scenario.swap_s_per_token * k]
    costs = CostModel(swap, swap, [0.0, params.prefill_s_per_token * k])
    chunk = max(1, scenario.prefill_chunk // k)
    run = simulate(
        jobs, costs, max(1, scenario.pool_tokens // (16 * k)),
        AgentConfig(policy=scenario.policy, prefix_cache=True),
        EngineConfig(max_batch=scenario.max_batch, prefill_chunk=chunk, prefill_budget=chunk),
        decode_step_s=params.decode_step_s * k,
        decode_per_seq_s=0.0,
        seed=seed,
        system_tokens=max(1, 256 // k),
    )
    durations = [run.result.jct_s[j.job_id] for j in jobs]
    return durations, max(j.arrival_s + d for j, d in zip(jobs, durations))


def nelder_mead(
    objective: Callable[[list[float]], float], start: list[float], step: float = 0.15, max_evals: int = 60
) -> tuple[list[float], float, int]:
    """Plain Nelder-Mead on the unit cube, clipped at the walls. Deterministic."""
    clip = lambda v: [min(max(x, 0.0), 1.0) for x in v]  # noqa: E731
    evals = 0

    def f(v):
        nonlocal evals
        evals += 1
        return objective(v)

    simplex = [list(start)]
    for axis in range(len(start)):
        point = list(start)
        point[axis] += step if point[axis] + step <= 1.0 else -step
        simplex.append(point)
    scored = [(f(p), p) for p in simplex]
    while evals < max_evals:
        scored.sort(key=lambda s: s[0])
        best, second_worst, worst = scored[0][0], scored[-2][0], scored[-1][0]
        if worst - best < 1e-5:
            break
        center = [statistics.fmean(s[1][i] for s in scored[:-1]) for i in range(len(start))]
        along = lambda t: clip([c + t * (c - w) for c, w in zip(center, scored[-1][1])])  # noqa: E731
        reflected = along(1.0)
        score = f(reflected)
        if score < best:
            expanded = along(2.0)
            expanded_score = f(expanded)
            scored[-1] = (expanded_score, expanded) if expanded_score < score else (score, reflected)
        elif score < second_worst:
            scored[-1] = (score, reflected)
        else:
            contracted = along(-0.5)
            contracted_score = f(contracted)
            if contracted_score < worst:
                scored[-1] = (contracted_score, contracted)
            else:
                anchor = scored[0][1]
                scored = [scored[0]] + [
                    (f(p), p) for p in (clip([(a + x) / 2 for a, x in zip(anchor, s[1])]) for s in scored[1:])
                ]
    scored.sort(key=lambda s: s[0])
    return scored[0][1], scored[0][0], evals


def from_unit(point: Sequence[float], bounds=BOUNDS) -> list[float]:
    values = []
    for u, (low, high, log_scale) in zip(point, bounds):
        values.append(math.exp(math.log(low) + u * (math.log(high) - math.log(low))) if log_scale else low + u * (high - low))
    return values


@dataclass
class FitResult:
    params: Params
    objective: float
    evaluations: int
    trail: list[tuple[list[float], float]] = field(default_factory=list)


def fit_params(
    train: Sequence[float],
    scenario: Scenario,
    seed: int,
    censor_s: float | None = DEFAULT_CENSOR_S,
    grid: tuple[int, int, int] = (5, 5, 4),
    simplex_evals: int = 50,
    log: Callable[[str], None] = lambda _: None,
) -> FitResult:
    """Minimize the log Wasserstein distance to train: a grid, then a simplex from its best.

    The same seed throughout gives common random numbers, so the objective is a fixed
    deterministic function of the parameters.
    """
    trail: list[tuple[list[float], float]] = []

    def objective(point: list[float]) -> float:
        params = Params(*from_unit(point))
        try:
            score = log_wasserstein(simulate_durations(params, scenario, seed), train, censor_s)
        except (RuntimeError, ValueError) as error:
            log(f"  {params}: {type(error).__name__}")
            score = float("inf")
        trail.append((params.as_list(), score))
        return score

    best_point, best_score = None, float("inf")
    for a in range(grid[0]):
        for b in range(grid[1]):
            for c in range(grid[2]):
                point = [(a + 0.5) / grid[0], (b + 0.5) / grid[1], (c + 0.5) / grid[2]]
                score = objective(point)
                if score < best_score:
                    best_point, best_score = point, score
    log(f"  grid best {best_score:.4f} at {from_unit(best_point)}")
    if best_point is None:
        raise RuntimeError("no parameter setting produced a usable simulation")
    point, score, _ = nelder_mead(objective, best_point, max_evals=simplex_evals)
    params = Params(*from_unit(point))
    log(f"  simplex best {score:.4f} at {params}")
    return FitResult(params, score, len(trail), trail)


def lognormal_baseline(train: Sequence[float], seed: int, count: int = 2000) -> list[float]:
    """Samples from a two-parameter log-normal fitted to train: a reference for shape."""
    logs = [math.log(v) for v in train]
    rng = random.Random(seed)
    mu, sigma = statistics.fmean(logs), statistics.pstdev(logs)
    return [rng.lognormvariate(mu, sigma) for _ in range(count)]


def quantile_table(published: Sequence[float], simulated: Sequence[float]) -> dict:
    fractions = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    a, b = sorted(published), sorted(simulated)
    return {f"p{round(f * 100)}": {"published_s": _quantile(a, f), "simulated_s": _quantile(b, f)} for f in fractions}


def evaluate(params: Params, scenario: Scenario, seed: int, published: Sequence[float], censor_s: float | None) -> dict:
    simulated, makespan = _simulate(params, scenario, seed)
    return {
        "seed": seed,
        "log_wasserstein": log_wasserstein(simulated, published, censor_s),
        "ks": ks_distance(simulated, published, censor_s),
        "simulated_mean_s": statistics.fmean(simulated),
        "simulated_makespan_s": makespan,
    }


def calibrate(
    published: dict[str, Published],
    seed: int = 0,
    censor_s: float = DEFAULT_CENSOR_S,
    coarsen: int = 8,
    verify_coarsen: int = 2,
    grid: tuple[int, int, int] = (5, 5, 4),
    simplex_evals: int = 50,
    log: Callable[[str], None] = print,
) -> dict:
    """The whole calibration. Returns the document that calibration.json holds."""
    main = published["continuum"]
    train, heldout = split_jobs(main.durations, seed)
    noise_ks = ks_distance(train, heldout, censor_s)
    noise_w1 = log_wasserstein(train, heldout, censor_s)

    def scenario_for(data: Published, basis: str, **changes) -> Scenario:
        return Scenario(
            arrival_rate_per_s=data.arrival_rate_per_s,
            jobs=len(data.durations),
            policy=POLICIES[data.target],
            workload=replace(SWE_BENCH_SHAPE, tokens_basis=basis),
            coarsen=changes.pop("coarsen", coarsen),
            **changes,
        )

    # The paper's table does not say what tokens per program counts. Fit both readings.
    log("fitting tokens_basis=cumulative")
    cumulative = fit_params(train, scenario_for(main, "cumulative"), seed, censor_s, grid, simplex_evals, log)
    log("fitting tokens_basis=unique")
    # Unique tokens mean ten times the prefill and memory pressure, so a coarser search.
    unique = fit_params(train, scenario_for(main, "unique", coarsen=coarsen * 4), seed, censor_s, (4, 4, 2), simplex_evals // 2, log)
    basis, fit = ("cumulative", cumulative) if cumulative.objective <= unique.objective else ("unique", unique)
    scenario = scenario_for(main, basis) if basis == "cumulative" else scenario_for(main, basis, coarsen=coarsen * 4)
    params = fit.params

    log(f"chosen basis {basis}: refitting on the other half for stability")
    other = fit_params(heldout, scenario, seed, censor_s, grid, simplex_evals, log)

    log("scoring on held-out jobs and fresh seeds")
    seeds = [seed + 1, seed + 2, seed + 3]
    scored = [evaluate(params, scenario, s, heldout, censor_s) for s in seeds]
    train_scored = evaluate(params, scenario, seed, train, censor_s)
    all_scored = evaluate(params, scenario, seed + 1, main.durations, censor_s)
    simulated = simulate_durations(params, scenario, seed + 1)
    finer = evaluate(params, replace(scenario, coarsen=verify_coarsen), seed + 1, main.durations, censor_s)

    baseline = lognormal_baseline(train, seed)
    uncalibrated = Params(params.decode_step_s, params.prefill_s_per_token, 0.0)

    log("sensitivity")
    base_score = log_wasserstein(simulate_durations(params, scenario, seed), train, censor_s)

    def changed(new_params=params, **workload_changes) -> dict:
        probe = replace(scenario, workload=replace(scenario.workload, **workload_changes))
        score = log_wasserstein(simulate_durations(new_params, probe, seed), train, censor_s)
        return {"train_log_wasserstein": score, "change": score - base_score}

    w = scenario.workload
    sensitivity = {
        "base_train_log_wasserstein": base_score,
        "tool_time_x0.25": changed(tool_s_median=w.tool_s_median * 0.25),
        "tool_time_x4": changed(tool_s_median=w.tool_s_median * 4),
        "output_x2_decode_step_x0.5": changed(replace(params, decode_step_s=params.decode_step_s / 2), output_tokens_mean=w.output_tokens_mean * 2),
        "decode_step_x1.3": changed(replace(params, decode_step_s=params.decode_step_s * 1.3)),
        "decode_step_x0.7": changed(replace(params, decode_step_s=params.decode_step_s * 0.7)),
        "prefill_x1.3": changed(replace(params, prefill_s_per_token=params.prefill_s_per_token * 1.3)),
        "prefill_x0.7": changed(replace(params, prefill_s_per_token=params.prefill_s_per_token * 0.7)),
        "sigma_x1.3": changed(replace(params, job_scale_sigma=params.job_scale_sigma * 1.3)),
        "sigma_x0.7": changed(replace(params, job_scale_sigma=params.job_scale_sigma * 0.7)),
    }

    # vLLM's run is the same workload and hardware with a different scheduler, at a lower
    # arrival rate. Parameters fitted to Continuum's run have never seen it.
    log("vLLM cross-check and separate fit")
    vllm = published["vllm"]
    vllm_scenario = scenario_for(vllm, basis, coarsen=scenario.coarsen)
    cross = evaluate(params, vllm_scenario, seed, vllm.durations, censor_s)
    vllm_train, vllm_heldout = split_jobs(vllm.durations, seed)
    vllm_fit = fit_params(vllm_train, vllm_scenario, seed, censor_s, grid, simplex_evals, log)
    vllm_scored = evaluate(vllm_fit.params, vllm_scenario, seed + 1, vllm_heldout, censor_s)
    vllm_sim = simulate_durations(vllm_fit.params, vllm_scenario, seed + 1)

    total_tool_s = w.turns_mean * math.exp(math.log(w.tool_s_median) + w.tool_s_sigma**2 / 2)
    return {
        "schema": CALIBRATION_SCHEMA,
        "seed": seed,
        "target": "continuum",
        "source": {
            "repository": REPOSITORY,
            "commit": PINNED_COMMIT,
            "files": {
                name: {
                    "path": PUBLISHED_FILES[name][0],
                    "sha256": data.sha256,
                    "num_jobs": len(data.durations),
                    "total_duration_s": data.total_duration_s,
                    "mean_concurrency": data.mean_concurrency,
                    "arrival_rate_per_s": data.arrival_rate_per_s,
                    "mean_duration_s": statistics.fmean(data.durations),
                    "fraction_above_censor": sum(d >= censor_s for d in data.durations) / len(data.durations),
                    "spearman_job_id_vs_duration": spearman(range(len(data.durations)), data.durations),
                }
                for name, data in published.items()
            },
        },
        "generator_overrides": {"job_scale_sigma": params.job_scale_sigma, "tokens_basis": basis},
        "fitted": {
            "decode_step_s": params.decode_step_s,
            "prefill_s_per_token": params.prefill_s_per_token,
            "job_scale_sigma": params.job_scale_sigma,
            "refit_on_other_half": other.params.__dict__,
            "objective": "log Wasserstein-1 on durations censored at censor_s, train half",
            "evaluations": fit.evaluations,
        },
        "assumptions": {
            "censor_s": censor_s,
            "policy": scenario.policy.value,
            "tokens_basis_compared": {"cumulative": cumulative.objective, "unique": unique.objective},
            "output_tokens_mean": w.output_tokens_mean,
            "pool_tokens": scenario.pool_tokens,
            "max_batch": scenario.max_batch,
            "prefill_chunk": scenario.prefill_chunk,
            "arrivals": "Poisson at num_jobs / total_duration",
            "coarsen": scenario.coarsen,
            "mean_tool_s_per_job_from_published_moments": total_tool_s,
        },
        "fit": {
            "train": train_scored,
            "heldout_fresh_seeds": scored,
            "all_jobs": all_scored,
            "at_finer_grain": {**finer, "coarsen": verify_coarsen},
            "noise_floor_train_vs_heldout": {"ks": noise_ks, "log_wasserstein": noise_w1},
            "ks_5pct_critical_heldout_vs_simulated": ks_critical(len(heldout), scenario.jobs),
            "lognormal_two_parameter_baseline_on_heldout": {
                "ks": ks_distance(baseline, heldout, censor_s),
                "log_wasserstein": log_wasserstein(baseline, heldout, censor_s),
            },
            "without_job_spread_on_heldout": evaluate(uncalibrated, scenario, seed + 1, heldout, censor_s),
            "published_makespan_s": main.total_duration_s,
            "quantiles_all_jobs": quantile_table(main.durations, simulated),
        },
        "sensitivity": sensitivity,
        "vllm": {
            "continuum_parameters_on_vllm_run": cross,
            "own_fit": {
                **vllm_fit.params.__dict__,
                "train_log_wasserstein": vllm_fit.objective,
                "heldout": vllm_scored,
                "quantiles_all_jobs": quantile_table(vllm.durations, vllm_sim),
            },
        },
        "means": [
            "With tool-time, turn and token statistics held at the paper's values, job_scale_sigma is the per-job "
            "spread of output length that makes the production scheduler's simulated durations resemble Continuum's "
            "published run, given an effective model of their hardware.",
            "tokens_basis records which reading of the paper's tokens-per-program statistic fits better under that model.",
        ],
        "does_not_mean": [
            "It does not measure tool latency, which stays at the published moments; sensitivity shows how far the "
            "objective moves when tool time is scaled.",
            "decode_step_s is not a measured speed: only its product with output_tokens_mean is constrained, as "
            "sensitivity output_x2_decode_step_x0.5 tests. decode_step_s and prefill_s_per_token are nuisance "
            "parameters for Llama 70B on 4 H100s and do not describe Qwen3 or any Layover GPU.",
            "job_scale_sigma is not a measured output-length distribution. It absorbs every source of spread this "
            "model lacks, including differences between SWE-bench repositories and scheduler details.",
            "A generated trace is not a recording. It inherits only generator_overrides; the headline benchmark uses "
            "recorded trajectories and this is for sweeps.",
            "A good fit does not validate the simulator: three parameters are matched to one marginal distribution, "
            "and jobs in the published runs trend with job id (see spearman_job_id_vs_duration), which an i.i.d. "
            "generator cannot reproduce.",
        ],
    }

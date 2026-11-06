#!/usr/bin/env python3
"""Turn saved matrix runs into tables, an ablation comparison, and plots.

    python bench/report.py results/matrix

Everything comes from the per-run JSON and the manifest beside it. A cell with a missing
run says so in its table, and the report lists each missing run with the reason. Each table
says how many instances its runs came from, since a spot session can span several.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RECORDED_PREFIX = "recorded:"
DEFAULT_LOAD = "load100"
PIPELINE_BANNER = "PIPELINE CHECK: NOT A RESULT. These numbers describe the test pipeline and nothing else."
# Fixed order, never cycled: a system keeps its color in every plot.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#52514e"
GRID = "#e6e6e6"

# (key, label, format)
METRICS = [
    ("mean_jct_s", "Mean JCT (s)", "{:.2f}"),
    ("p95_jct_s", "p95 JCT (s)", "{:.2f}"),
    ("p95_jct_minus_tool_s", "p95 JCT minus tool time (s)", "{:.2f}"),
    ("programs_per_hour", "Programs/hour", "{:.1f}"),
    ("mean_first_token_s", "Mean first_token_s", "{:.3f}"),
    ("p95_first_token_s", "p95 first_token_s", "{:.3f}"),
    ("gpu_mem_frac_mean", "GPU memory used (fraction)", "{:.3f}"),
    ("kv_pool_used_frac_mean", "KV pool used (fraction)", "{:.3f}"),
    ("kv_pool_tree_only_frac_mean", "KV pool held only by a radix tree (fraction)", "{:.3f}"),
    ("pcie_gb", "PCIe moved (GB, NVML estimate)", "{:.2f}"),
]
HIGHER_IS_BETTER = {"programs_per_hour"}
KV_USED_NOTE = (
    "KV pool used is the share that requests and transfers hold, for every system: a cached block nothing holds counts as "
    "free, because it is reusable and the first to go under pressure, as in vLLM's free queue. Layover's such blocks, held "
    "only by its radix tree, have their own column. So a low figure does not mean the pool never ran out."
)


@dataclass(frozen=True)
class Spread:
    n: int
    median: float
    minimum: float
    maximum: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def spread(values: list[float]) -> Spread:
    """Median, range, and quartiles, using the inclusive (linear interpolation) method."""
    if not values:
        raise ValueError("no values")
    if len(values) == 1:
        return Spread(1, values[0], values[0], values[0], values[0], values[0])
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return Spread(len(values), statistics.median(values), min(values), max(values), q1, q3)


def load_results(out: Path) -> tuple[dict, dict[tuple[str, str, str, int], dict]]:
    manifest = json.loads((out / "manifest.json").read_text())
    runs = {}
    # Only finished files: a writer's leftover .tmp never matches, and a torn copy is reported missing.
    for path in sorted((out / "runs").rglob("run-*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            print(f"WARNING: {path} is not valid JSON, so it counts as missing", file=sys.stderr)
            continue
        runs[(record["workload"], record["system"], record.get("load", DEFAULT_LOAD), record["run"])] = record
    return manifest, runs


def read_json(path: Path) -> dict:
    """A side file, or nothing: a report is most needed when a session ended badly, and it must not fail on one."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_fraction(label: str) -> float:
    return int(label.removeprefix("load")) / 100


def loads_of(manifest: dict, workload: str) -> list[str]:
    """The loads a workload runs at, lightest first."""
    found = {cell.get("load", DEFAULT_LOAD) for cell in manifest.get("cells", []) if cell["workload"] == workload}
    return sorted(found or {DEFAULT_LOAD}, key=load_fraction)


def expected_runs(manifest: dict, workload: str, system: str, load: str) -> int:
    cells = manifest.get("cells")
    if cells is None:
        return manifest["runs"]
    return sum(1 for cell in cells if (cell["workload"], cell["system"], cell.get("load", DEFAULT_LOAD)) == (workload, system, load))


def systems_at(manifest: dict, workload: dict, load: str) -> list[str]:
    """The workload's systems this load was asked of, in manifest order."""
    if manifest.get("cells") is None:
        return workload["systems"]
    return [system for system in manifest["systems"] if expected_runs(manifest, workload["name"], system, load)]


def system_label(manifest: dict, system: str) -> str:
    """A system's name, with whatever it must always be read with, such as being a lower bound."""
    for spec in (manifest.get("config") or {}).get("systems", []):
        if spec["name"] == system and spec.get("label"):
            return f"{system} ({spec['label']})"
    return system


def records_of(runs: dict, workload: str, system: str | None = None, load: str | None = None) -> list[dict]:
    return [
        record for (w, s, l, _), record in runs.items()
        if w == workload and system in (None, s) and load in (None, l) and record.get("status") == "ok"
    ]


def missing_runs(manifest: dict, runs: dict, session: dict) -> list[dict]:
    """Every expected run without a successful result, and why."""
    skipped = set(session.get("budget_skipped", []))
    stopped = {label: name for name, entry in session.get("stopped_systems", {}).items() for label in entry["skipped"]}
    blocked = set((session.get("calibration_failed") or {}).get("skipped", []))
    missing = []
    for cell in manifest["cells"]:
        load = cell.get("load", DEFAULT_LOAD)
        record = runs.get((cell["workload"], cell["system"], load, cell["run"]))
        if record is not None and record.get("status") == "ok":
            continue
        label = f"{cell['workload']}/{cell['system']}/{load}/{cell['run']}"
        if record is not None:
            reason = f"failed after {len(record['attempts'])} attempts: {record['attempts'][-1]['error'][:200]}"
        elif label in skipped:
            reason = "skipped by the time budget"
        elif label in stopped:
            reason = f"skipped because {stopped[label]} was stopped"
        elif label in blocked:
            reason = "skipped because calibration failed"
        else:
            reason = "not run"
        missing.append({**cell, "label": label, "reason": reason})
    return missing


def _title(text: str, pipeline: bool) -> str:
    return f"PIPELINE CHECK, NOT A RESULT: {text}" if pipeline else text


def _values(runs: dict, workload: str, system: str, metric: str, load: str = DEFAULT_LOAD) -> list[float]:
    values = [record["result"]["metrics"].get(metric) for record in records_of(runs, workload, system, load)]
    return [value for value in values if value is not None]


def instances_note(records: list[dict]) -> str | None:
    """How many machines a table's runs came from, and a flag if they were not alike."""
    facts = [record["instance"] for record in records if record.get("instance")]
    if not facts:
        return None
    instances = {fact.get("instance_id") for fact in facts}
    note = f"Runs here came from {len(instances)} instance{'s' if len(instances) != 1 else ''}."
    hardware = {(fact.get("gpu_name"), fact.get("gpu_driver")) for fact in facts}
    if len(hardware) > 1:
        note += " **MIXED HARDWARE: " + "; ".join(f"{name}, driver {driver}" for name, driver in sorted(hardware, key=str)) + ".**"
    commits = {fact.get("commit") for fact in facts}
    if len(commits) > 1:
        note += " **MIXED CODE: " + ", ".join(sorted(map(str, commits))) + ".**"
    return note


def _cell_text(values: list[float], fmt: str) -> str:
    if not values:
        return "n/a"
    s = spread(values)
    return (
        f"{fmt.format(s.median)} ({fmt.format(s.minimum)} to {fmt.format(s.maximum)}, IQR {fmt.format(s.iqr)})"
    )


def p95_note(measured: int) -> str:
    """What the nearest-rank p95 is at this many jobs: one job, not a distribution's tail."""
    from_top = measured - math.ceil(0.95 * measured) + 1
    if from_top == 1:
        job = "the longest job, the maximum"
    else:
        job = {2: "the second-longest job", 3: "the third-longest job"}.get(from_top, f"job {from_top} from the longest")
    return (
        f"p95 is nearest-rank over {measured} measured jobs, so it is {job}, and much of that job's time can be tool time "
        "every system pays alike. p95 JCT minus tool time takes each job's trace tool time off first."
    )


def tier_flag(manifest: dict, runs: dict, workload: dict, system: str, load: str = DEFAULT_LOAD) -> str | None:
    """Why a baseline's cell cannot be read as "with CPU offload", if so.

    A configured tier proves nothing: only a counter of what came back from it does. It is per
    load, because a light load may never overflow the pool that a heavy one does.
    """
    specs = {spec["name"]: spec for spec in manifest["config"]["systems"]}
    counter = CPU_TIER_EVIDENCE.get(specs[system]["kind"])
    if counter is None or specs[system]["kind"] == "layover":
        return None
    records = records_of(runs, workload["name"], system, load)
    if not records:
        return None
    if all((r["result"].get("server_metrics") or {}).get(counter) for r in records):
        return None
    if _layover_under_pressure(runs, workload, load):
        return "CPU tier unverified"
    # A system's own KV-used peak cannot answer this: it leaves out cached blocks, so eviction can run below any peak.
    name = EVICTION_EVIDENCE.get(specs[system]["kind"])
    evictions = [(r["result"].get("server_metrics") or {}).get(name) for r in records] if name else []
    if evictions and None not in evictions:
        return "CPU tier unverified" if any(evictions) else "CPU tier unused, nothing evicted"
    return "CPU tier unused, KV never filled on Layover" if _layover_pool_stayed_free(manifest, runs, workload, load) else "CPU tier unverified"


def _layover_under_pressure(runs: dict, workload: dict, load: str) -> bool:
    """Early demotions happen only when the pool runs short, so the same load overflowed there."""
    return any((r["result"].get("server_metrics") or {}).get("early_demotions") for r in records_of(runs, workload["name"], load=load))


def _layover_pool_stayed_free(manifest: dict, runs: dict, workload: dict, load: str) -> bool:
    """Whether the free list lasted at this load on the same KV budget, on a Layover that keeps what it can."""
    keeping = [s["name"] for s in manifest["config"]["systems"] if s["kind"] == "layover" and s.get("prefix_cache", True)]
    peaks = [
        r["result"]["metrics"].get("kv_pool_occupied_frac_peak")
        for name in keeping for r in records_of(runs, workload["name"], name, load)
    ]
    return bool(peaks) and all(peak is not None and peak < FULL_POOL_FRAC for peak in peaks)


def _rate_text(runs: dict, workload: str, load: str) -> str:
    rate = next((r["arrival"]["arrival_rate_per_s"] for r in records_of(runs, workload, load=load) if r.get("arrival")), None)
    return f" ({rate:.3g} jobs/s)" if rate is not None else ""


def workload_table(manifest: dict, runs: dict, workload: dict, missing: list[dict], load: str = DEFAULT_LOAD) -> list[str]:
    """One markdown table, for one workload at one load. The title carries the source and any shortfall."""
    pipeline = manifest["pipeline_check"]
    name = workload["name"]
    systems = systems_at(manifest, workload, load)
    gaps = [m for m in missing if m["workload"] == name and m.get("load", DEFAULT_LOAD) == load]
    title = f"{name}, {load}{_rate_text(runs, name, load)}: {workload['role']}, trace source `{workload['source']}`"
    lines = [f"### {_title(title, pipeline)}", ""]
    if gaps:
        total = sum(expected_runs(manifest, name, system, load) for system in systems)
        lines += [f"**INCOMPLETE: {len(gaps)} of {total} runs missing; see Missing runs.**", ""]
    if workload["dropped_jobs"]:
        lines += [f"{len(workload['dropped_jobs'])} jobs over the context limit were left out.", ""]
    for key, values in (workload.get("trace_meta") or {}).items():
        lines += [f"Trace {key}: {', '.join(values)}.", ""]
    lines += [
        f"{workload['jobs']} jobs. The first {workload['warmup_jobs']} are warm-up, left out of the job-time and first-token "
        "columns only: the GPU memory, KV pool, and PCIe columns cover a time window that includes warm-up jobs still "
        "running. Cells: median (min to max, IQR).",
        "",
        p95_note(workload["jobs"] - workload["warmup_jobs"]),
        "",
        KV_USED_NOTE,
        "",
    ]
    lines.append("| System | Runs | " + " | ".join(label for _, label, _ in METRICS) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in METRICS) + " |")
    for system in systems:
        count = len(records_of(runs, name, system, load))
        expected = expected_runs(manifest, name, system, load)
        marker = f"{count}/{expected}" + ("" if count == expected else " **missing runs**")
        cells = [_cell_text(_values(runs, name, system, key, load), fmt) for key, _, fmt in METRICS]
        lines.append(f"| {_system_cell(manifest, runs, workload, system, load)} | {marker} | " + " | ".join(cells) + " |")
    lines.append("")
    note = instances_note(records_of(runs, name, load=load))
    if note:
        lines += [note, ""]
    return lines


def _system_cell(manifest: dict, runs: dict, workload: dict, system: str, load: str = DEFAULT_LOAD) -> str:
    flag = tier_flag(manifest, runs, workload, system, load)
    label = system_label(manifest, system)
    return f"{label} **{flag}**" if flag else label


def _by_run(runs: dict, workload: str, system: str, metric: str, load: str) -> dict[int, float]:
    """A metric's finished values by run index, which is what fixes a run's seed."""
    found = {}
    for (w, s, l, run), record in runs.items():
        if (w, s, l) == (workload, system, load) and record.get("status") == "ok":
            value = record["result"]["metrics"].get(metric)
            if value is not None:
                found[run] = value
    return found


def paired_ratio(ours: dict[int, float], theirs: dict[int, float]) -> float | None:
    """The median of same-run ratios, over the runs both have.

    A seed sets the arrivals and the prompt tokens, so dividing medians over different seed sets
    would mix those differences into what is meant to show the system's decisions alone.
    """
    ratios = [ours[run] / theirs[run] for run in sorted(set(ours) & set(theirs)) if theirs[run] > 0]
    return statistics.median(ratios) if ratios else None


def ablation_table(manifest: dict, runs: dict, workload: dict, reference: str, load: str = DEFAULT_LOAD) -> list[str]:
    """Each system against the reference, run by run, for the same workload and load."""
    name = workload["name"]
    systems = systems_at(manifest, workload, load)
    if reference not in systems or not _values(runs, name, reference, "mean_jct_s", load):
        return []
    keys = ["mean_jct_s", "p95_jct_s", "programs_per_hour", "mean_first_token_s"]
    labels = {key: label for key, label, _ in METRICS}
    title = f"{name}, {load}{_rate_text(runs, name, load)}, relative to {reference}, trace source `{workload['source']}`"
    lines = [f"### {_title(title, manifest['pipeline_check'])}", ""]
    lines.append(
        "Median of per-run ratios: each run is divided by the reference's run with the same seed, over the seeds both have, "
        "listed in the table, however many more the reference ran. 1.00 is the reference. Above 1 is more time, except Programs/hour, where it is more work. "
        "A ratio marked with * is against a baseline with no evidence its CPU tier returned anything; see CPU tier evidence."
    )
    lines.append("")
    lines.append("| System | Runs | Seeds compared | " + " | ".join(labels[key] for key in keys) + " |")
    lines.append("| --- | --- | --- | " + " | ".join("---" for _ in keys) + " |")
    for system in systems:
        cells = []
        mark = "*" if tier_flag(manifest, runs, workload, system, load) else ""
        for key in keys:
            ratio = paired_ratio(_by_run(runs, name, system, key, load), _by_run(runs, name, reference, key, load))
            cells.append("n/a" if ratio is None else f"{ratio:.2f}{mark}")
        ours = _by_run(runs, name, system, "mean_jct_s", load)
        shared = sorted(set(ours) & set(_by_run(runs, name, reference, "mean_jct_s", load)))
        # Old records name no seed, and the run index still says which pairs were compared.
        seeds = ", ".join(str(runs[(name, system, load, run)].get("seed", f"run {run}")) for run in shared) or "none shared"
        lines.append(f"| {_system_cell(manifest, runs, workload, system, load)} | {len(ours)}/{expected_runs(manifest, name, system, load)} | {seeds} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_csv(path: Path, manifest: dict, runs: dict, missing: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["pipeline_check", "workload", "role", "source", "load", "arrival_rate_per_s", "system", "system_label", "metric", "runs", "expected_runs", "missing_runs", "median", "min", "max", "q1", "q3", "iqr", "cpu_tier_flag"]
        )
        for workload in manifest["workloads"]:
            name = workload["name"]
            for load in loads_of(manifest, name):
                rate = next((r["arrival"]["arrival_rate_per_s"] for r in records_of(runs, name, load=load) if r.get("arrival")), "")
                for system in systems_at(manifest, workload, load):
                    gaps = sum(m["workload"] == name and m["system"] == system and m.get("load", DEFAULT_LOAD) == load for m in missing)
                    flag = tier_flag(manifest, runs, workload, system, load) or ""
                    expected = expected_runs(manifest, name, system, load)
                    for key, _, _ in METRICS:
                        values = _values(runs, name, system, key, load)
                        head = [manifest["pipeline_check"], name, workload["role"], workload["source"], load, rate, system, system_label(manifest, system), key]
                        if values:
                            s = spread(values)
                            writer.writerow([*head, s.n, expected, gaps, s.median, s.minimum, s.maximum, s.q1, s.q3, s.iqr, flag])
                        else:
                            writer.writerow([*head, 0, expected, gaps, "", "", "", "", "", "", flag])


def token_id_section(manifest: dict, runs: dict, reference: str) -> list[str]:
    """How token ids came back, whether counts held, and how far outputs differ."""
    lines = [
        "## Token ids and output comparison",
        "",
        "Same seed gives every system the same arrivals and the same non-generated tokens. Generated ids differ when "
        "numerics differ, so contexts diverge after the first generated turn; only the counts are held equal. "
        "Identical turns compares each run with the same run of the reference.",
        "",
        "| Workload | Load | System | Turns with ids from server | Turns re-encoded from text | Output count mismatches | Prompt tokens | Prompt tokens the server reports cached | Turns identical to reference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for workload in manifest["workloads"]:
        for load in loads_of(manifest, workload["name"]):
            for system in systems_at(manifest, workload, load):
                server, encoded, mismatch, prompt, cached = 0, 0, 0, [], []
                same, compared = 0, 0
                for (w, s, l, run), record in runs.items():
                    if (w, s, l) != (workload["name"], system, load) or record.get("status") != "ok":
                        continue
                    result = record["result"]
                    server += result["token_sources"].get("server", 0)
                    encoded += result["token_sources"].get("tokenizer", 0)
                    mismatch += result["output_count_mismatches"]
                    prompt.append(result["prompt_tokens_total"])
                    if result.get("cached_tokens_total") is not None:
                        cached.append(result["cached_tokens_total"])
                    base = runs.get((w, reference, l, run))
                    if base and base.get("status") == "ok":
                        theirs = {(t["session"], t["turn"]): t["crc"] for t in base["result"]["turns"]}
                        for turn in result["turns"]:
                            key = (turn["session"], turn["turn"])
                            if key in theirs:
                                compared += 1
                                same += theirs[key] == turn["crc"]
                identical = f"{same}/{compared}" if compared else "n/a"
                prompt_text = f"{statistics.median(prompt):.0f}" if prompt else "n/a"
                cached_text = f"{statistics.median(cached):.0f}" if cached else "n/a"
                lines.append(f"| {workload['name']} | {load} | {system_label(manifest, system)} | {server} | {encoded} | {mismatch} | {prompt_text} | {cached_text} | {identical} |")
    lines.append("")
    return lines


# The counter that moves when something came back from a system's CPU tier, by system kind.
# A host usage gauge will not do: under write-through it rises with no reload at all.
CPU_TIER_EVIDENCE = {
    "vllm": "vllm:external_prefix_cache_hits_total",
    "sglang": "sglang:load_back_tokens_total",
    "layover": "cpu_tier_bytes_moved",
}
# What counts evictions, where a system has it. An eviction is the tier's reason to be used.
EVICTION_EVIDENCE = {"sglang": "sglang:evicted_tokens_total"}
# Occupancy, tree-only blocks included, at or above this means the free list ran out.
FULL_POOL_FRAC = 0.98


def budget_section(manifest: dict, runs: dict) -> list[str]:
    """What each server was checked to hold, and whether its CPU tier ever did anything."""
    lines = [
        "## Budget checks",
        "",
        "Read from each freshly started server before its run; a value off budget fails the run, so every row here passed. "
        "The counter is the server's own at the end of a run, warm-up included, as a median across all of a system's "
        "workloads. CPU tier evidence below gives it for each workload.",
        "",
        "| System | KV tokens reported | KV tokens expected | GPU held (GiB) | Weights + KV budget (GiB) | CPU tier reported (GiB) | CPU tier budget (GiB) | How the tier was read | CPU tier counter | Value |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    specs = {spec["name"]: spec for spec in manifest["config"]["systems"]}
    for system in manifest["systems"]:
        records = [r for (_, s, _, _), r in runs.items() if s == system and r.get("status") == "ok"]
        if not records:
            continue

        def median_check(key: str, fmt: str) -> str:
            found = [r["result"].get("budget_checks", {}).get(key) for r in records]
            found = [value for value in found if value is not None]
            return fmt.format(statistics.median(found)) if found else "n/a"

        counter = CPU_TIER_EVIDENCE.get(specs[system]["kind"])
        values = [r["result"].get("server_metrics", {}).get(counter) for r in records] if counter else []
        values = [value for value in values if value is not None]
        evidence = f"{statistics.median(values):g}" if values else "n/a"
        source = next((r["result"].get("budget_checks", {}).get("cpu_tier_source") for r in records), None) or "n/a"
        lines.append(
            f"| {system_label(manifest, system)} | {median_check('reported_kv_tokens', '{:.0f}')} | {median_check('expected_kv_tokens', '{:.0f}')} "
            f"| {median_check('gpu_held_gib', '{:.2f}')} | {median_check('gpu_expected_gib', '{:.2f}')} "
            f"| {median_check('reported_cpu_tier_gib', '{:.2f}')} | {median_check('expected_cpu_tier_gib', '{:g}')} | {source} | {counter or 'n/a'} | {evidence} |"
        )
    lines.append("")
    return lines


def tier_section(manifest: dict, runs: dict) -> list[str]:
    """For each workload apart, since a sweep's counter says nothing about the headline's tier."""
    lines = [
        "## CPU tier evidence",
        "",
        "Each server's own counter of what came back from its CPU tier, at the end of a run, warm-up included. A zero or "
        "missing counter flags that baseline's cells as unverified, unless something shows the tier had no reason to be "
        "used: its own eviction counter stayed at zero, or, with no such counter, Layover's pool on the same budget and "
        "load never ran out of free blocks and it demoted nothing early. The peak here leaves out cached blocks, so it is "
        "not that evidence.",
        "",
        "| Workload | Load | System | Counter | Runs with no evidence | Median counter | Peak KV pool used | Flag |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    specs = {spec["name"]: spec for spec in manifest["config"]["systems"]}
    for workload in manifest["workloads"]:
        for load in loads_of(manifest, workload["name"]):
            for system in systems_at(manifest, workload, load):
                records = records_of(runs, workload["name"], system, load)
                counter = CPU_TIER_EVIDENCE.get(specs[system]["kind"])
                if not records or counter is None:
                    continue
                values = [(r["result"].get("server_metrics") or {}).get(counter) for r in records]
                known = [value for value in values if value is not None]
                peaks = [r["result"]["metrics"].get("kv_pool_used_frac_peak") for r in records]
                peaks = [peak for peak in peaks if peak is not None]
                lines.append(
                    f"| {workload['name']} | {load} | {system_label(manifest, system)} | {counter} | {sum(not value for value in values)}/{len(values)} "
                    f"| {f'{statistics.median(known):g}' if known else 'n/a'} | {f'{max(peaks):.3f}' if peaks else 'n/a'} "
                    f"| {tier_flag(manifest, runs, workload, system, load) or ''} |"
                )
    lines.append("")
    return lines


def server_queue_section(manifest: dict, runs: dict) -> list[str]:
    """Servers' own queue metrics. Their definitions differ, so they are never compared in a table above."""
    lines = [
        "## Server-reported queue time",
        "",
        "From each server's own metrics, which count the whole run including warm-up. The definitions differ between "
        "systems, so this table is not comparable across them the way first_token_s is. Median across runs of the mean "
        "per request.",
        "",
        "| Workload | Load | System | Metric | Mean queue time per request (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    names = {"vllm": "vllm:request_queue_time_seconds", "sglang": "sglang:queue_time_seconds"}
    specs = {spec["name"]: spec for spec in manifest["config"]["systems"]}
    for workload in manifest["workloads"]:
        for load in loads_of(manifest, workload["name"]):
            for system in systems_at(manifest, workload, load):
                metric = names.get(specs[system]["kind"])
                if metric is None:
                    lines.append(f"| {workload['name']} | {load} | {system_label(manifest, system)} | not exposed | n/a |")
                    continue
                means = []
                for record in records_of(runs, workload["name"], system, load):
                    reported = record["result"]["server_metrics"]
                    count = reported.get(f"{metric}_count")
                    if count:
                        means.append(reported.get(f"{metric}_sum", 0.0) / count)
                value = f"{statistics.median(means):.4f}" if means else "n/a"
                lines.append(f"| {workload['name']} | {load} | {system_label(manifest, system)} | {metric} | {value} |")
    lines.append("")
    return lines


def unbracketed(arrival: dict) -> str | None:
    """A sentence for a rate that is only the end of a ladder, if it is one."""
    if arrival.get("bracketed", True):
        return None
    where = "already past it at the lowest rung" if arrival.get("edge") == "below_ladder" else "never reached it, even at the top rung"
    return f"SATURATION NOT FOUND on {arrival.get('workload')}: {arrival.get('system')} {where}, so this rate is only the ladder's edge and every load here is relative to that edge"


def arrival_section(manifest: dict, runs: dict, out: Path) -> list[str]:
    lines = ["## Arrival rates", ""]
    definition = ((manifest.get("config") or {}).get("calibration") or {}).get("definition")
    if definition:
        lines += [f"Saturation: {definition}", ""]
    lines += ["Every system ran at the same absolute rates, so a load means the same arrivals whichever system it is.", ""]
    for workload in manifest["workloads"]:
        arrivals = {load: next((r["arrival"] for r in records_of(runs, workload["name"], load=load) if r.get("arrival")), None) for load in loads_of(manifest, workload["name"])}
        arrivals = {load: arrival for load, arrival in arrivals.items() if arrival}
        if not arrivals:
            lines.append(f"- {workload['name']}: no run recorded")
            continue
        rates = ", ".join(f"{load} {arrival['arrival_rate_per_s']:.4g} jobs/s" for load, arrival in arrivals.items())
        first = next(iter(arrivals.values()))
        warning = unbracketed(first)
        lines.append(f"- {workload['name']}: {rates}. {first.get('chosen_by', '')}." + (f" **{warning}.**" if warning else ""))
    lines.append("")
    calibrations = sorted((out / "calibration").glob("*.json")) if (out / "calibration").exists() else []
    finished = {path.stem for path in calibrations}
    for path in sorted((out / "calibration" / "partial").glob("*.json")) if (out / "calibration" / "partial").exists() else []:
        if path.stem not in finished:
            data = read_json(path)
            lines += [f"Calibration `{path.stem}` is unfinished: {len(data.get('points', []))} rungs and {len(data.get('unloaded_jct_s', {}))} unloaded jobs are saved. A resume keeps them.", ""]
            lines += [f"- {failure['step']}, attempt {failure['attempt']}: {failure['error'][:200]}" for failure in data.get("failures", [])]
            lines += [""] if data.get("failures") else []
    for path in calibrations:
        data = read_json(path)
        if not data:
            lines += [f"Calibration `{path.stem}` cannot be read.", ""]
            continue
        lines += [
            f"Calibration `{path.stem}` on {data['system']} over {data['workload']}, target {data['target_slowdown']}x. "
            f"Unloaded mean JCT {data['unloaded_mean_jct_s']:.2f} s, from {data['unloaded_jobs']} jobs run alone; the slowdown "
            "compares those same jobs in each rung, which replays the whole workload:",
            "",
            "| Rate (jobs/s) | Mean JCT, same jobs (s) | Mean JCT, all measured jobs (s) | Slowdown |",
            "| --- | --- | --- | --- |",
        ]
        lines += [f"| {p['rate_per_s']:g} | {p['mean_jct_s']:.2f} | {p['measured_mean_jct_s']:.2f} | {p['slowdown']:.2f} |" for p in data["points"]]
        lines.append("")
        lines += [f"- Retried: {failure['step']}, attempt {failure['attempt']}: {failure['error'][:200]}" for failure in data.get("failures", [])]
        lines += [""] if data.get("failures") else []
    return lines


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9, loc="left", color=TEXT)
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=8, width=0.6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)


def _legend(ax) -> None:
    legend = ax.legend(frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT)


def system_colors(systems: list[str]) -> dict[str, str]:
    """The manifest's system order fixes the colors. A ninth system is a design problem, not a new hue."""
    if len(systems) > len(PALETTE):
        raise ValueError(f"{len(systems)} systems but only {len(PALETTE)} colors; plot fewer systems")
    return dict(zip(systems, PALETTE))


def bar_plot(path: Path, manifest: dict, runs: dict, workload: dict, metric: str, label: str, colors: dict[str, str], load: str = DEFAULT_LOAD) -> bool:
    """Median per system with a min-max whisker. One metric, one y-axis."""
    systems = [s for s in systems_at(manifest, workload, load) if _values(runs, workload["name"], s, metric, load)]
    if not systems:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for index, system in enumerate(systems):
        s = spread(_values(runs, workload["name"], system, metric, load))
        ax.bar(index, s.median, width=0.45, color=colors[system], label=system_label(manifest, system))
        ax.errorbar(index, s.median, yerr=[[s.median - s.minimum], [s.maximum - s.median]], color=TEXT, linewidth=0.8, capsize=2)
    ax.set_xticks(range(len(systems)), [system_label(manifest, system) for system in systems], rotation=25, ha="right")
    title = f"{workload['name']}, {load}{_rate_text(runs, workload['name'], load)}, source {workload['source']}"
    _style(ax, _title(title, manifest["pipeline_check"]) + "\nMedian across runs, whisker min to max", label)
    if len(systems) > 1:
        _legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def load_points(runs: dict, workload: str, system: str, metric: str, loads: list[str]) -> list[tuple[str, float, Spread]]:
    """(load, the rate it ran at, the metric's spread) for every load this system has a value at."""
    points = []
    for load in loads:
        values = _values(runs, workload, system, metric, load)
        arrival = next((r.get("arrival") for r in records_of(runs, workload, system, load) if r.get("arrival")), None)
        if values and arrival:
            points.append((load, arrival["arrival_rate_per_s"], spread(values)))
    return points


def load_table(manifest: dict, runs: dict, workload: dict, metric: str, label: str, fmt: str) -> list[str]:
    """A metric at each load, for every system that ran at more than one."""
    name = workload["name"]
    loads = loads_of(manifest, name)
    systems = [s for s in manifest["systems"] if sum(1 for load in loads if expected_runs(manifest, name, s, load)) > 1]
    if len(loads) < 2 or not systems:
        return []
    rates = {load: _rate_text(runs, name, load) for load in loads}
    lines = [
        f"### {_title(f'{label} against load, {name}', manifest['pipeline_check'])}", "",
        "Same absolute rates for every system. Cells: median (min to max, IQR).", "",
        "| System | " + " | ".join(f"{load}{rates[load]}" for load in loads) + " |",
        "| --- | " + " | ".join("---" for _ in loads) + " |",
    ]
    for system in systems:
        cells = [_cell_text(_values(runs, name, system, metric, load), fmt) for load in loads]
        lines.append(f"| {system_label(manifest, system)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def load_plot(path: Path, manifest: dict, runs: dict, workload: dict, metric: str, label: str, colors: dict[str, str]) -> bool:
    """A metric against the arrival rate, one line per system, band min to max. One y-axis."""
    name = workload["name"]
    loads = loads_of(manifest, name)
    if len(loads) < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    drawn = 0
    for system in manifest["systems"]:
        points = load_points(runs, name, system, metric, loads)
        if len(points) < 2:
            continue
        xs = [rate for _, rate, _ in points]
        ax.plot(xs, [s.median for _, _, s in points], color=colors[system], linewidth=1.2, marker="o", markersize=3, label=system_label(manifest, system))
        ax.fill_between(xs, [s.minimum for _, _, s in points], [s.maximum for _, _, s in points], color=colors[system], alpha=0.12, linewidth=0)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xlabel("Arrival rate (jobs/s), the same for every system", fontsize=8, color=TEXT)
    title = f"{name}, source {workload['source']}"
    _style(ax, _title(title, manifest["pipeline_check"]) + "\nMedian across runs, band min to max", label)
    if drawn > 1:
        _legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def sweep_plot(path: Path, manifest: dict, runs: dict, sweep: str, metric: str, label: str, colors: dict[str, str]) -> bool:
    """One line per system over a sweep, with a min-max band. One metric, one y-axis."""
    points = sorted(
        ((m["x"], w) for w in manifest["workloads"] for m in w["sweeps"] if m["sweep"] == sweep), key=lambda p: p[0]
    )
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    drawn = 0
    for system in systems_at(manifest, points[0][1], DEFAULT_LOAD):
        xs, medians, lows, highs = [], [], [], []
        for x, workload in points:
            values = _values(runs, workload["name"], system, metric)
            if values:
                s = spread(values)
                xs.append(x)
                medians.append(s.median)
                lows.append(s.minimum)
                highs.append(s.maximum)
        if xs:
            ax.plot(xs, medians, color=colors[system], linewidth=1.2, marker="o", markersize=3, label=system_label(manifest, system))
            ax.fill_between(xs, lows, highs, color=colors[system], alpha=0.12, linewidth=0)
            drawn += 1
    if not drawn:
        plt.close(fig)
        return False
    if sweep == "tool_median_s":
        ax.set_xscale("log")
        ax.set_xlabel("Median tool latency (s)", fontsize=8, color=TEXT)
    else:
        ax.set_xlabel("Sub-agents per job", fontsize=8, color=TEXT)
    source = points[0][1]["source"]
    _style(ax, _title(f"Sweep over {sweep}, source {source}", manifest["pipeline_check"]) + "\nMedian across runs, band min to max", label)
    if drawn > 1:
        _legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def provenance_section(manifest: dict, runs: dict) -> list[str]:
    """Which machines and code produced these tables. A spot session can span several."""
    facts = [r["instance"] for r in runs.values() if r.get("status") == "ok" and r.get("instance")]
    lines = ["## Provenance", ""]
    if not facts:
        return [*lines, "No finished run names its instance.", ""]
    instances = sorted({fact.get("instance_id") for fact in facts}, key=str)
    lines += [f"{len(facts)} finished runs on {len(instances)} instance{'s' if len(instances) != 1 else ''}.", ""]
    lines += ["| Instance | GPU | Driver | Commit | Finished runs |", "| --- | --- | --- | --- | --- |"]
    for found in sorted({(f.get("instance_id"), f.get("gpu_name"), f.get("gpu_driver"), f.get("commit")) for f in facts}, key=str):
        count = sum((f.get("instance_id"), f.get("gpu_name"), f.get("gpu_driver"), f.get("commit")) == found for f in facts)
        lines.append("| " + " | ".join(str(value) for value in found) + f" | {count} |")
    lines.append("")
    return lines


def banners(manifest: dict, runs: dict) -> list[str]:
    """What would make every table misleading if it were found only on page three."""
    finished = [r for r in runs.values() if r.get("status") == "ok"]
    found = []
    warnings = {unbracketed(r["arrival"]) for r in finished if r.get("arrival")} - {None}
    found += [f"**{warning}.**" for warning in sorted(warnings)]
    facts = [r["instance"] for r in finished if r.get("instance")]
    hardware = {(f.get("gpu_name"), f.get("gpu_driver")) for f in facts}
    if len(hardware) > 1:
        found.append(f"**MIXED HARDWARE: these runs came from {len(hardware)} different GPU or driver versions. See Provenance.**")
    commits = {f.get("commit") for f in facts}
    if len(commits) > 1:
        found.append(f"**MIXED CODE: these runs came from {len(commits)} code versions. See Provenance.**")
    # The rate every run depends on has its own provenance, which the runs' instances do not show.
    calibrated = {
        (device.get("name"), device.get("driver"))
        for r in finished if (device := ((r.get("arrival") or {}).get("identity") or {}).get("hardware"))
    }
    if facts and calibrated - hardware:
        names = "; ".join(f"{name}, driver {driver}" for name, driver in sorted(calibrated - hardware, key=str))
        found.append(f"**MIXED HARDWARE: the arrival rates were calibrated on {names}, which ran none of these runs.**")
    return [line for warning in found for line in (warning, "")]


def _count_by_system_and_load(labels: list[str]) -> str:
    """workload/system/load/run labels, as a line a reader can act on."""
    counts: dict[str, int] = {}
    for label in labels:
        _, system, load, _ = label.rsplit("/", 3)
        counts[f"{system} {load}"] = counts.get(f"{system} {load}", 0) + 1
    return ", ".join(f"{name} x {count}" for name, count in counts.items())


def session_section(session: dict, ledger: dict) -> list[str]:
    """How the session ended and what that left out, before any table is read as complete."""
    lines = []
    if ledger.get("sessions"):
        instances = {entry.get("instance_id") for entry in ledger["sessions"]}
        limit = f", against --max-hours {ledger['max_hours']:g}" if ledger.get("max_hours") is not None else ""
        lines += [f"Matrix time so far: {ledger['elapsed_s'] / 3600:.2f} hours over {len(ledger['sessions'])} sessions on {len(instances)} instances{limit}.", ""]
    skipped = session.get("budget_skipped", [])
    if skipped:
        lines += [f"**TIME BUDGET REACHED: {len(skipped)} runs were not run: {_count_by_system_and_load(skipped)}. Each is listed under Missing runs.**", ""]
    for system, entry in session.get("stopped_systems", {}).items():
        left = f"{len(entry['skipped'])} more of its runs were skipped: {_count_by_system_and_load(entry['skipped'])}" if entry["skipped"] else "No more of its runs were left"
        lines += [f"**{system} STOPPED: {entry['reason']}. {left}. The other systems went on.**", ""]
    failed = session.get("calibration_failed")
    if failed:
        lines += [f"**CALIBRATION FAILED: {failed['reason']}. {len(failed['skipped'])} runs were not run. What it measured is saved, and a rerun resumes it.**", ""]
    if session.get("stopped"):
        lines += [f"**MATRIX STOPPED: {session['stopped'][:1000]}. Rerun the same command to resume.**", ""]
    return lines


def write_report(out: str | Path) -> Path:
    out = Path(out)
    manifest, runs = load_results(out)
    pipeline = manifest["pipeline_check"]
    for workload in manifest["workloads"]:
        # The runner refuses this too. A report edited by hand or a stale directory must not get past either.
        if workload["role"] == "headline" and not pipeline and not (workload["source"] or "").startswith(RECORDED_PREFIX):
            raise SystemExit(f"refusing to report headline workload {workload['name']} with source {workload['source']!r}")
    session = read_json(out / "session.json")
    missing = missing_runs(manifest, runs, session)
    reference = manifest["config"].get("reference_system", manifest["systems"][0])
    colors = system_colors(manifest["systems"])

    lines: list[str] = []
    if pipeline:
        lines += [f"# {PIPELINE_BANNER}", ""]
    lines += ["# Benchmark matrix", ""]
    lines += banners(manifest, runs)
    lines += session_section(session, read_json(out / "budget.json"))
    host = manifest.get("host", {})
    gpu = host.get("gpu") or {}
    matched = manifest["config"]["matched"]
    lines += [
        f"Model {matched['model']}, {matched['dtype']}, on {gpu.get('name', 'no GPU')}. "
        f"KV budget {matched['kv_gib']} GiB, CPU tier {matched['cpu_offload_gib']} GiB, "
        f"max context {matched['max_context_tokens']:,} tokens, at most {matched['max_running']} running requests. "
        f"Plan: {', '.join(manifest.get('plans') or []) or 'every cell'}. Cost model {manifest.get('cost_model_hash')}. "
        f"Commit {host.get('commit')}{' with uncommitted changes' if host.get('dirty') else ''}.",
        "",
        "Baseline versions: " + (", ".join(f"{k} {v}" for k, v in sorted(host.get("baseline_versions", {}).items())) or "none") + ".",
        "",
        f"Config: `manifest.json` here holds it in full; flag reasons are in `bench/configs/baseline_flags.md`. "
        f"Dependency versions of each baseline: `env/`.",
        "",
        manifest["first_token_note"],
        "",
    ]
    sample = next((r["result"]["gpu"].get("pcie_note") for r in runs.values() if r.get("status") == "ok" and r["result"]["gpu"]), None)
    if sample:
        lines += [sample, ""]

    headline = [w for w in manifest["workloads"] if w["role"] == "headline"]
    sweeps = [w for w in manifest["workloads"] if w["role"] != "headline"]
    labels = {key: label for key, label, _ in METRICS}
    formats = {key: fmt for key, _, fmt in METRICS}
    lines += ["## Results", ""]
    for workload in headline + sweeps:
        if workload["source"] is None:
            lines += [f"### {_title(workload['name'], pipeline)}", "", "**No trace was available; nothing ran.**", ""]
            continue
        lines += load_table(manifest, runs, workload, "mean_jct_s", labels["mean_jct_s"], formats["mean_jct_s"])
        for load in loads_of(manifest, workload["name"]):
            if systems_at(manifest, workload, load):
                lines += workload_table(manifest, runs, workload, missing, load)
    lines += [f"## Ablation: relative to {reference}", ""]
    for workload in headline + sweeps:
        if workload["source"] is not None:
            for load in loads_of(manifest, workload["name"]):
                lines += ablation_table(manifest, runs, workload, reference, load)
    lines += arrival_section(manifest, runs, out)
    lines += provenance_section(manifest, runs)
    lines += token_id_section(manifest, runs, reference)
    lines += budget_section(manifest, runs)
    lines += tier_section(manifest, runs)
    lines += server_queue_section(manifest, runs)

    lines += ["## Missing runs", ""]
    if missing:
        lines += [f"{len(missing)} of {len(manifest['cells'])} expected runs have no result:", ""]
        lines += [f"- `{m['label']}`: {m['reason']}" for m in missing]
    else:
        lines.append("None.")
    lines.append("")

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    made = []
    wanted = ("mean_jct_s", "p95_jct_s", "programs_per_hour", "mean_first_token_s")
    for workload in headline + sweeps:
        if workload["source"] is None:
            continue
        path = plots / f"{workload['name']}-jct-vs-load.png"
        if load_plot(path, manifest, runs, workload, "mean_jct_s", labels["mean_jct_s"], colors):
            made.append(path)
    for workload in headline:
        if workload["source"] is None:
            continue
        for load in loads_of(manifest, workload["name"]):
            for metric in wanted:
                path = plots / f"{workload['name']}-{load}-{metric}.png"
                if bar_plot(path, manifest, runs, workload, metric, labels[metric], colors, load):
                    made.append(path)
    for sweep in sorted({m["sweep"] for w in manifest["workloads"] for m in w["sweeps"]}):
        for metric in wanted:
            path = plots / f"sweep-{sweep}-{metric}.png"
            if sweep_plot(path, manifest, runs, sweep, metric, labels[metric], colors):
                made.append(path)
    if made:
        lines += ["## Plots", ""]
        lines += [f"![{path.stem}](plots/{path.name})" for path in made]
        lines.append("")

    write_csv(out / "results.csv", manifest, runs, missing)
    report = out / "report.md"
    report.write_text("\n".join(lines))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/matrix"))
    args = parser.parse_args(argv)
    print(write_report(args.results))


if __name__ == "__main__":
    sys.exit(main())

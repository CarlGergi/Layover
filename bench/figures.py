#!/usr/bin/env python3
"""The README's figures, from measured results. Each figure is drawn only when its inputs exist.

    python bench/figures.py                      results/ to docs/images/
    python bench/figures.py --results R --out O

One style for all of them: white background, the same four colors in the same order for the same
systems, thin marks, direct labels, and no titles, since each figure has a caption in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Slots 1 to 4 of the validated categorical order (blue, orange, aqua, yellow); a system keeps its color.
COLORS = {"layover": "#2a78d6", "vllm": "#eb6834", "discard": "#1baf7a", "fourth": "#eda100"}
# The blue ramp's steps 250, 400, 550 and 700, for ordered values such as batch size; the light end clears 2:1.
ORDINAL_BLUES = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
}


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def prefill_figure(results: Path, out: Path) -> Path | None:
    """Prefill rate of a 2048-token chunk after a context: before the hot-path work, after it, and vLLM."""
    series = [
        ("Layover before", read_json(results / "profile/before/rates.json"), "#9a9890"),
        ("Layover after", read_json(results / "profile/after2/rates.json"), COLORS["layover"]),
        ("vLLM", read_json(results / "profile/vllm/rates_quick.json"), COLORS["vllm"]),
    ]
    series = [(label, data["prefill_tokens_per_s"], color) for label, data, color in series if data]
    if len(series) < 2:
        return None
    contexts = sorted({int(c) for _, rates, _ in series for c in rates})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / len(series)
    for i, (label, rates, color) in enumerate(series):
        xs = [j + (i - (len(series) - 1) / 2) * width for j, c in enumerate(contexts) if str(c) in rates]
        ys = [rates[str(c)] for c in contexts if str(c) in rates]
        bars = ax.bar(xs, ys, width * 0.92, color=color, label=label)
        ax.bar_label(bars, labels=[f"{y:,.0f}" for y in ys], fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(range(len(contexts)), [f"{c:,}" for c in contexts])
    ax.set_xlabel("tokens already in the context")
    ax.set_ylabel("prefill, tokens/s")
    ax.legend(loc="upper right", fontsize=8)
    return save(fig, out, "prefill.png")


def bandwidth_figure(results: Path, out: Path) -> Path | None:
    """Unique KV bytes read per second by the decode kernel, as a share of the card's peak."""
    path = results / "decode_bandwidth/decode_bandwidth.csv"
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r["layout"] == "fragmented"]
    batches = sorted({int(r["batch"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for batch, color in zip(batches, ORDINAL_BLUES[-len(batches):]):
        points = sorted((int(r["tokens"]), float(r["unique_pct"])) for r in rows if int(r["batch"]) == batch)
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", color=color)
        # A one-point batch is labelled below its point, clear of the line it sits on.
        offset = (6, -3) if len(xs) > 1 else (-4, -14)
        ax.annotate(f"batch {batch}", (xs[-1], ys[-1]), textcoords="offset points", xytext=offset, fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384, 32768], ["1,024", "4,096", "16,384", "32,768"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("tokens of context per sequence")
    ax.set_ylabel("unique KV read, % of peak bandwidth")
    return save(fig, out, "decode_bandwidth.png")


def crossover_figure(results: Path, out: Path) -> Path | None:
    """Time to swap a sequence's KV out and back in, BF16 and FP8, against recomputing it."""
    fp8, bf16 = results / "kv_crossover/kv_crossover.csv", results / "kv_crossover_native_bf16/kv_crossover.csv"
    if not fp8.exists():
        return None
    rows = list(csv.DictReader(fp8.open()))
    tokens = [int(r["tokens"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lines = [("recompute", [float(r["recompute_s"]) for r in rows], MUTED)]
    if bf16.exists():
        lines.append(("swap, BF16", [float(r["swap_s"]) for r in csv.DictReader(bf16.open())], ORDINAL_BLUES[0]))
    lines.append(("swap, FP8", [float(r["swap_s"]) for r in rows], ORDINAL_BLUES[2]))
    for label, ys, color in lines:
        ax.plot(tokens[: len(ys)], [y * 1e3 for y in ys], marker="o", color=color)
        ax.annotate(label, (tokens[len(ys) - 1], ys[-1] * 1e3), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(tokens, [f"{t:,}" for t in tokens], rotation=0)
    ax.set_xlabel("tokens of KV")
    ax.set_ylabel("milliseconds, out and back in")
    return save(fig, out, "crossover.png")


def matrix_runs(results: Path) -> dict[str, list[float]]:
    """Mean job completion time of each finished run, by system."""
    by_system: dict[str, list[float]] = {}
    for path in sorted((results / "matrix/runs").rglob("run-*.json")):
        run = json.loads(path.read_text())
        if run.get("status") != "ok":
            continue
        jct = ((run.get("result") or {}).get("metrics") or {}).get("mean_jct_s")
        if jct is not None:
            by_system.setdefault(run["system"], []).append(float(jct))
    return by_system


def matrix_figure(results: Path, out: Path) -> Path | None:
    """Mean job completion time per run, each system at the same arrival rate."""
    runs = matrix_runs(results)
    order = [s for s in ("layover-combined", "vllm", "layover-discard", "layover-keep_ttl", "layover-swap") if s in runs]
    if not order:
        return None
    color = {"layover-combined": COLORS["layover"], "vllm": COLORS["vllm"], "layover-discard": COLORS["discard"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, system in enumerate(order):
        values = [v / 60 for v in runs[system]]
        mean = statistics.fmean(values)
        ax.barh(i, mean, 0.6, color=color.get(system, COLORS["fourth"]))
        ax.scatter(values, [i] * len(values), color=INK, s=14, zorder=3)
        ax.annotate(f"{mean:.1f} min", (mean, i), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("mean job completion time, minutes (dots: individual runs)")
    ax.grid(axis="y", visible=False)
    return save(fig, out, "matrix_jct.png")


FIGURES = (matrix_figure, prefill_figure, bandwidth_figure, crossover_figure)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--out", type=Path, default=REPO / "docs/images")
    args = p.parse_args(argv)
    with plt.rc_context(STYLE):
        for figure in FIGURES:
            path = figure(args.results, args.out)
            print(f"{figure.__name__}: {path if path else 'skipped, no results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line for traces: generate, convert saved trajectories, or replay over HTTP.

Headline traces come from `python -m layover.replay.trajectories` and
`python -m layover.replay.reexecute`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from layover.replay.trace import (
    BFCL_SHAPE, SWE_BENCH_SHAPE, calibrated_workload, from_mini_swe_agent, generate, load_jsonl, save_jsonl,
)

SHAPES = {"swebench": SWE_BENCH_SHAPE, "bfcl": BFCL_SHAPE}


def _expand(paths: list[Path]) -> list[Path]:
    """A directory means every trajectory directly in it."""
    expanded = []
    for path in paths:
        expanded += sorted(path.glob("*.traj.json")) if path.is_dir() else [path]
    return expanded


def _convert(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    jobs = []
    for path in _expand(args.trajectories):
        raw = json.loads(path.read_text())
        jobs.append(from_mini_swe_agent(raw, count, path.stem, len(jobs) / args.arrival_rate))
    save_jsonl(jobs, args.out)


def main(argv=None) -> None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="python -m layover.replay", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="write a synthetic trace")
    gen.add_argument("--shape", choices=sorted(SHAPES), default="swebench")
    gen.add_argument("--jobs", type=int, default=20)
    gen.add_argument("--arrival-rate", type=float, default=0.2, help="jobs per second")
    gen.add_argument("--tool-median-s", type=float, help="override, for the tool-latency sweep")
    gen.add_argument("--branching", type=int, default=0, help="sub-agents per job, for the branching sweep")
    gen.add_argument("--calibration", type=Path, help="a file from bench/calibrate_synthetic.py; swebench shape only")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", type=Path, required=True)

    convert = commands.add_parser(
        "convert", help="convert saved mini-swe-agent trajectories to a trace with the submitter's tool gaps; never a headline"
    )
    convert.add_argument("trajectories", type=Path, nargs="+", help=".traj.json files, or directories of them")
    convert.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    convert.add_argument("--arrival-rate", type=float, default=0.2)
    convert.add_argument("--out", type=Path, required=True)

    http = commands.add_parser("http", help="replay a trace against a server")
    http.add_argument("--trace", type=Path, required=True)
    http.add_argument("--url", default="http://127.0.0.1:8000")
    http.add_argument("--model", required=True)
    http.add_argument("--vocab-size", type=int, required=True)
    http.add_argument("--system", default="layover")
    http.add_argument("--no-hints", action="store_true", help="for servers without session hints")
    http.add_argument("--tokenizer", help="needed when a server returns no token ids")
    http.add_argument("--time-scale", type=float, default=1.0)
    http.add_argument("--warmup-jobs", type=int, default=0)
    http.add_argument("--seed", type=int, default=0)
    http.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(arguments)
    if args.command == "generate":
        shape = replace(SHAPES[args.shape], jobs=args.jobs, arrival_rate_per_s=args.arrival_rate, branching=args.branching)
        if args.tool_median_s is not None:
            shape = replace(shape, tool_s_median=args.tool_median_s)
        if args.calibration is not None:
            if args.shape != "swebench":
                parser.error("the calibration is against a SWE-bench run, so it applies only to --shape swebench")
            shape = calibrated_workload(shape, args.calibration)
        save_jsonl(generate(shape, args.seed), args.out)
    elif args.command == "convert":
        _convert(args)
    else:
        from layover.replay.http_replay import replay

        encode = None
        if args.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        result = asyncio.run(
            replay(
                load_jsonl(args.trace), args.url, args.model, args.vocab_size, args.system, args.time_scale,
                args.warmup_jobs, args.seed, hints=not args.no_hints, encode=encode,
            )
        )
        args.out.write_text(json.dumps(result.summary(), indent=2) + "\n")
        print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()

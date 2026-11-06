#!/usr/bin/env python3
"""Calibrate the synthetic trace generator against Continuum's published job durations.

Downloads the two results files at a pinned commit and checks their sha256, or reads a
cached copy from --cache-dir. The data is never stored in this repository. The output is
read by `python -m layover.replay generate --calibration PATH`; it states in its own
"means" and "does_not_mean" fields what the numbers can and cannot be used for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layover.replay.calibrate import DEFAULT_CENSOR_S, PUBLISHED_FILES, calibrate, fetch_published, load_published  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("traces/continuum"))
    parser.add_argument("--out", type=Path, default=Path("results/calibration/calibration.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--censor-s", type=float, default=DEFAULT_CENSOR_S)
    parser.add_argument("--coarsen", type=int, default=8, help="simulate tokens in groups of this size while fitting")
    args = parser.parse_args(argv)

    published = {target: load_published(target, fetch_published(target, args.cache_dir)) for target in PUBLISHED_FILES}
    document = calibrate(published, args.seed, args.censor_s, args.coarsen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({k: document[k] for k in ("generator_overrides", "fitted", "fit", "does_not_mean")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

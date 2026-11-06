"""Calibration against Continuum's published job durations: metrics, pinning, determinism."""

import hashlib
import math
import statistics
from pathlib import Path

import pytest

from layover.replay import calibrate as cal
from layover.replay.calibrate import (
    PUBLISHED_FILES, Params, Scenario, from_unit, ks_critical, ks_distance, load_published, log_wasserstein, nelder_mead,
    simulate_durations, spearman, split_jobs,
)
from layover.replay.trace import SWE_BENCH_SHAPE, lognormal_from_moments

CACHE = Path(__file__).parents[1] / "traces" / "continuum"


def test_log_wasserstein_reads_as_a_relative_error():
    values = [float(v) for v in range(1, 200)]
    assert log_wasserstein(values, values) == 0
    assert log_wasserstein(values, [v * math.exp(0.1) for v in values]) == pytest.approx(0.1)


def test_censoring_hides_differences_above_the_limit():
    low = [10.0, 20.0, 30.0]
    assert log_wasserstein(low + [1000.0], low + [5000.0], censor_s=100.0) == 0
    assert ks_distance(low + [1000.0], low + [5000.0], censor_s=100.0) == 0
    assert ks_distance(low + [1000.0], low + [5000.0]) > 0


def test_ks_extremes_and_critical_value():
    assert ks_distance([1.0, 2.0], [1.0, 2.0]) == 0
    assert ks_distance([1.0, 2.0], [3.0, 4.0]) == 1
    assert ks_critical(500, 500) == pytest.approx(1.358 * math.sqrt(2 / 500))


def test_spearman_is_rank_based():
    assert spearman([1, 2, 3, 4], [1, 10, 100, 1000]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1)


def test_split_is_deterministic_and_covers_everything_once():
    values = [float(i) for i in range(11)]
    train, heldout = split_jobs(values, 3)
    assert (train, heldout) == split_jobs(values, 3) and (train, heldout) != split_jobs(values, 4)
    assert sorted(train + heldout) == values


def test_from_unit_respects_bounds_and_log_scale():
    assert from_unit([0.0, 0.0, 0.0]) == pytest.approx([0.005, 1e-5, 0.0])
    assert from_unit([1.0, 1.0, 1.0]) == pytest.approx([0.5, 2e-3, 1.5])
    assert from_unit([0.5, 0.5, 0.5])[0] == pytest.approx(math.sqrt(0.005 * 0.5))


def test_nelder_mead_finds_a_known_minimum():
    best, value, evaluations = nelder_mead(lambda p: (p[0] - 0.3) ** 2 + (p[1] - 0.7) ** 2, [0.5, 0.5], max_evals=200)
    assert best == pytest.approx([0.3, 0.7], abs=0.02) and value < 1e-3 and evaluations <= 210


def test_simulation_is_deterministic_and_feels_decode_speed():
    scenario = Scenario(arrival_rate_per_s=0.2, jobs=6, coarsen=16)
    slow, fast = Params(0.1, 1e-4, 0.3), Params(0.01, 1e-4, 0.3)
    first = simulate_durations(slow, scenario, 1)
    assert first == simulate_durations(slow, scenario, 1) and len(first) == 6 and min(first) > 0
    assert statistics.fmean(first) > statistics.fmean(simulate_durations(fast, scenario, 1))


def test_download_is_refused_unless_it_matches_the_pinned_hash(tmp_path, monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    monkeypatch.setattr(cal.httpx, "get", lambda url, **_: Response(b"tampered"))
    with pytest.raises(ValueError, match="pinned sha256"):
        cal.fetch_published("continuum", tmp_path)
    assert not list(tmp_path.rglob("*.json"))
    good = b'{"job_durations": {"0": 1.0}, "total_duration": 1.0}'
    urls = []
    monkeypatch.setitem(PUBLISHED_FILES, "continuum", ("x/results.json", hashlib.sha256(good).hexdigest()))
    monkeypatch.setattr(cal.httpx, "get", lambda url, **_: urls.append(url) or Response(good))
    path = cal.fetch_published("continuum", tmp_path)
    assert cal.PINNED_COMMIT in urls[0] and path.read_bytes() == good
    cal.fetch_published("continuum", tmp_path)
    assert len(urls) == 1  # Served from the cache afterwards.


def test_loading_refuses_a_file_that_changed(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"job_durations": {"0": 1.0}, "total_duration": 1.0}')
    with pytest.raises(ValueError, match="pinned"):
        load_published("continuum", path)
    assert load_published("continuum", path, verify=False).durations == [1.0]


@pytest.mark.parametrize("target", sorted(PUBLISHED_FILES))
def test_cached_published_files_match_their_pins(target):
    path = CACHE / cal.PINNED_COMMIT[:12] / f"{target}.json"
    if not path.exists():
        pytest.skip("not downloaded here; bench/calibrate_synthetic.py fetches it")
    published = load_published(target, path)
    assert len(published.durations) == 500 and published.arrival_rate_per_s > 0


def test_generator_shape_stays_at_the_papers_tool_moments_whatever_calibration_does():
    assert (SWE_BENCH_SHAPE.tool_s_median, SWE_BENCH_SHAPE.tool_s_sigma) == lognormal_from_moments(0.925, 3.55)

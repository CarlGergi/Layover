"""Per-tool duration estimates from observed tool-call gaps.

Tool durations are heavy-tailed, so each tool keeps a running mean and variance of
log-duration (Welford's method), which is a log-normal fit. A tool with few samples
borrows from the pool of all tools, weighted by how little it has seen itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z90 = 1.2815515655446004


@dataclass(frozen=True)
class DurationEstimate:
    mean_s: float
    p50_s: float
    p90_s: float
    samples: int


class _LogStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


class ToolDurationEstimator:
    MIN_SECONDS = 1e-4

    def __init__(self, prior_s: float = 1.0, prior_sigma: float = 1.0, prior_weight: float = 2.0):
        self.prior_mu = math.log(prior_s)
        self.prior_var = prior_sigma**2
        self.prior_weight = prior_weight
        self._tools: dict[str, _LogStats] = {}
        self._all = _LogStats()

    def observe(self, tool: str | None, seconds: float) -> None:
        value = math.log(max(seconds, self.MIN_SECONDS))
        self._tools.setdefault(tool or "", _LogStats()).add(value)
        self._all.add(value)

    def estimate(self, tool: str | None = None, hint_s: float | None = None) -> DurationEstimate:
        """Estimate a tool's duration. A client hint replaces the learned center."""
        stats = self._tools.get(tool or "", _LogStats())
        fallback_mu = self._all.mean if self._all.count else self.prior_mu
        fallback_var = self._all.variance if self._all.count > 1 else self.prior_var
        weight = stats.count + self.prior_weight
        mu = (stats.count * stats.mean + self.prior_weight * fallback_mu) / weight
        var = (stats.count * stats.variance + self.prior_weight * fallback_var) / weight
        if hint_s is not None:
            mu = math.log(max(hint_s, self.MIN_SECONDS))
        sigma = math.sqrt(var)
        return DurationEstimate(
            mean_s=math.exp(mu + var / 2),
            p50_s=math.exp(mu),
            p90_s=math.exp(mu + Z90 * sigma),
            samples=stats.count,
        )

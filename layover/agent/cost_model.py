"""Keep, swap, or drop a paused agent's KV, priced in seconds of delay.

The swap and recompute curves come from bench/kv_crossover.py. InferCept (Abhyankar et
al., ICML 2024) introduced the three-way choice, assuming predictable tools. Continuum
(Li et al., ICLR 2026) pins KV with a time-to-live; here the time-to-live is what caps
the loss from a wrong prediction, and expiry demotes to CPU instead of discarding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from layover.agent.estimator import DurationEstimate


class Placement(Enum):
    KEEP = "keep"
    SWAP = "swap"
    DROP = "drop"


class Policy(Enum):
    """Combined is the contribution; the single policies exist for ablation.

    Discard drops a paused session's KV. What the radix tree keeps of it is the scheduler's
    business: see the AgentScheduler module docstring.
    """

    COMBINED = "combined"
    KEEP_TTL = "keep_ttl"
    SWAP = "swap"
    DISCARD = "discard"


def _poly(coef: list[float], x: float) -> float:
    return sum(c * x**power for power, c in enumerate(coef))


@dataclass(frozen=True)
class CostModel:
    swap_out_s_coef: list[float]
    swap_in_s_coef: list[float]
    recompute_s_coef: list[float]
    measured_on_gpu: bool = False
    # Swap times with FP8 offload, quantizing included, when they were measured.
    fp8_swap_out_s_coef: list[float] | None = None
    fp8_swap_in_s_coef: list[float] | None = None
    # The path the swap times were measured through.
    transfer: str = "plain"

    @classmethod
    def from_json(cls, path: str | Path, require_gpu: bool = False, transfer: str | None = None) -> "CostModel":
        """Load bench/kv_crossover.py's output.

        Given `transfer`, refuses a file measured through a different path: prices from one
        path would quietly drive placement on another.
        """
        raw = json.loads(Path(path).read_text())
        model = cls(
            raw["swap_out_s_coef"],
            raw["swap_in_s_coef"],
            raw["recompute_s_coef"],
            raw["measured_on_gpu"],
            raw.get("fp8_swap_out_s_coef"),
            raw.get("fp8_swap_in_s_coef"),
            raw.get("transfer", "plain"),
        )
        if require_gpu and not model.measured_on_gpu:
            raise ValueError(f"{path} was not measured on a GPU")
        if transfer is not None and model.transfer != transfer:
            raise ValueError(f"{path} was measured through {model.transfer!r} transfers, not {transfer!r}")
        return model

    def for_offload(self, fp8: bool) -> "CostModel":
        """The model whose swap times describe a tier storing BF16 or FP8."""
        if not fp8:
            return self
        if self.fp8_swap_out_s_coef is None or self.fp8_swap_in_s_coef is None:
            raise ValueError("this cost model has no FP8 swap times: measure them with kv_crossover --offload-dtype fp8")
        return replace(self, swap_out_s_coef=self.fp8_swap_out_s_coef, swap_in_s_coef=self.fp8_swap_in_s_coef)

    def swap_out_s(self, tokens: int) -> float:
        return _poly(self.swap_out_s_coef, tokens)

    def swap_in_s(self, tokens: int) -> float:
        return _poly(self.swap_in_s_coef, tokens)

    def swap_s(self, tokens: int) -> float:
        return self.swap_out_s(tokens) + self.swap_in_s(tokens)

    def recompute_s(self, tokens: int) -> float:
        return _poly(self.recompute_s_coef, tokens)


@dataclass(frozen=True)
class Decision:
    placement: Placement
    ttl_s: float = 0.0
    keep_cost_s: float = 0.0
    swap_cost_s: float = 0.0
    drop_cost_s: float = 0.0


@dataclass(frozen=True)
class PlacementPolicy:
    costs: CostModel
    policy: Policy = Policy.COMBINED
    ttl_cap_s: float = 30.0

    def decide(
        self,
        tokens: int,
        estimate: DurationEstimate,
        pool_fraction: float,
        waiting_programs: int,
        cpu_has_room: bool = True,
    ) -> Decision:
        """Choose a placement for a session pausing with `tokens` of context.

        Holding a fraction f of the pool for d seconds delays each waiting program by
        up to f*d, so keeping costs d*f*w seconds of other agents' time; with nobody
        waiting, keeping is free. Swapping and dropping cost this agent's resume delay.
        """
        rent_per_s = pool_fraction * waiting_programs
        swap = self.costs.swap_s(tokens) if cpu_has_room else float("inf")
        drop = self.costs.recompute_s(tokens)
        cheapest_move = min(swap, drop)
        keep = estimate.mean_s * rent_per_s
        moved = Placement.SWAP if swap <= drop else Placement.DROP

        # Past the break-even point, renting has cost more than moving would have.
        ttl = self.ttl_cap_s if rent_per_s == 0 else min(self.ttl_cap_s, cheapest_move / rent_per_s)
        costs = {"keep_cost_s": keep, "swap_cost_s": swap, "drop_cost_s": drop}
        if self.policy is Policy.DISCARD:
            return Decision(Placement.DROP, **costs)
        if self.policy is Policy.SWAP:
            return Decision(Placement.SWAP if cpu_has_room else Placement.DROP, **costs)
        if self.policy is Policy.KEEP_TTL:
            return Decision(Placement.KEEP, ttl_s=self.ttl_cap_s, **costs)
        # A tool that typically outlasts the cap would only burn its time-to-live.
        if estimate.p50_s <= self.ttl_cap_s and keep < cheapest_move:
            return Decision(Placement.KEEP, ttl_s=ttl, **costs)
        return Decision(moved, **costs)

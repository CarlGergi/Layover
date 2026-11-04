"""Duration estimates and the keep / swap / drop decision."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from layover.agent.cost_model import CostModel, Placement, PlacementPolicy, Policy
from layover.agent.estimator import DurationEstimate, ToolDurationEstimator

# Illustrative curves: swap is linear; recompute is costlier and superlinear.
COSTS = CostModel([0.001, 5e-6], [0.001, 5e-6], [0.005, 2e-5, 2e-9])
QUICK = DurationEstimate(mean_s=0.5, p50_s=0.4, p90_s=1.0, samples=5)
SLOW = DurationEstimate(mean_s=300.0, p50_s=200.0, p90_s=600.0, samples=5)


def test_estimator_learns_a_tool_and_keeps_tools_apart():
    estimator = ToolDurationEstimator()
    for _ in range(50):
        estimator.observe("grep", 0.1)
        estimator.observe("build", 100.0)
    assert estimator.estimate("grep").p50_s == pytest.approx(0.1, rel=0.2)
    assert estimator.estimate("build").p50_s == pytest.approx(100.0, rel=0.2)


def test_unseen_tool_falls_back_to_all_tools_then_prior():
    estimator = ToolDurationEstimator(prior_s=2.0)
    assert estimator.estimate("new").p50_s == pytest.approx(2.0)
    for _ in range(20):
        estimator.observe("grep", 0.1)
    assert estimator.estimate("new").p50_s == pytest.approx(0.1, rel=0.01)


def test_client_hint_replaces_the_center():
    estimator = ToolDurationEstimator()
    estimator.observe("t", 1.0)
    assert estimator.estimate("t", hint_s=40.0).p50_s == pytest.approx(40.0)


@given(st.lists(st.floats(1e-3, 1e4), min_size=1, max_size=30))
def test_estimates_are_ordered_and_positive(values):
    estimator = ToolDurationEstimator()
    for value in values:
        estimator.observe("t", value)
    e = estimator.estimate("t")
    assert 0 < e.p50_s <= e.mean_s * (1 + 1e-9) and e.p50_s <= e.p90_s * (1 + 1e-9)


def test_keeps_when_nobody_needs_the_memory():
    decision = PlacementPolicy(COSTS).decide(8192, QUICK, pool_fraction=0.5, waiting_programs=0)
    assert decision.placement is Placement.KEEP and decision.ttl_s == 30.0


def test_moves_a_long_tool_call_off_the_gpu_even_when_idle():
    decision = PlacementPolicy(COSTS).decide(8192, SLOW, pool_fraction=0.1, waiting_programs=0)
    assert decision.placement is Placement.SWAP


def test_memory_pressure_turns_keep_into_a_move():
    policy = PlacementPolicy(COSTS)
    assert policy.decide(8192, QUICK, 0.5, 0).placement is Placement.KEEP
    assert policy.decide(8192, QUICK, 0.5, 8).placement is Placement.SWAP


def test_short_context_may_be_cheaper_to_recompute():
    costs = CostModel([0.5, 1e-6], [0.5, 1e-6], [0.0, 1e-5, 0.0])
    assert PlacementPolicy(costs).decide(64, SLOW, 0.01, 4).placement is Placement.DROP


def test_full_cpu_tier_falls_back_to_drop():
    assert PlacementPolicy(COSTS).decide(8192, SLOW, 0.1, 4, cpu_has_room=False).placement is Placement.DROP


def test_ttl_stops_at_the_break_even_point():
    decision = PlacementPolicy(COSTS, ttl_cap_s=1000.0).decide(8192, QUICK, 0.5, 2)
    assert decision.placement is Placement.KEEP or decision.placement is Placement.SWAP
    tiny = DurationEstimate(0.001, 0.001, 0.002, 5)
    kept = PlacementPolicy(COSTS, ttl_cap_s=1000.0).decide(8192, tiny, 0.5, 2)
    assert kept.placement is Placement.KEEP
    assert kept.ttl_s == pytest.approx(min(COSTS.swap_s(8192), COSTS.recompute_s(8192)) / (0.5 * 2))


@pytest.mark.parametrize(
    "policy,expected",
    [(Policy.DISCARD, Placement.DROP), (Policy.SWAP, Placement.SWAP), (Policy.KEEP_TTL, Placement.KEEP)],
)
def test_single_policies_ignore_the_cost_model(policy, expected):
    assert PlacementPolicy(COSTS, policy).decide(8192, SLOW, 0.5, 8).placement is expected
    assert PlacementPolicy(COSTS, policy).decide(8192, QUICK, 0.5, 0).placement is expected


def test_cost_model_loads_kv_crossover_output(tmp_path):
    import json

    path = tmp_path / "cost_model.json"
    path.write_text(json.dumps({
        "swap_out_s_coef": [0.0, 1e-6], "swap_in_s_coef": [0.0, 1e-6],
        "recompute_s_coef": [0.0, 1e-5, 0.0], "measured_on_gpu": False,
    }))
    model = CostModel.from_json(path)
    assert model.swap_s(1000) == pytest.approx(2e-3)
    assert model.recompute_s(1000) == pytest.approx(1e-2)
    with pytest.raises(ValueError):
        CostModel.from_json(path, require_gpu=True)

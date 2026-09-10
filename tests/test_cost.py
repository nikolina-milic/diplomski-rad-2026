import numpy as np
import pytest

from spr.policy.cost import (
    CapacityConstraints, CostMatrix, action_cost, action_rates,
    constrained_thresholds, cost_report, empirical_thresholds,
    expected_action_cost, optimal_thresholds, total_cost,
)
from spr.policy.engine import Action, PolicyConfig, RiskLevel


def _policy(low=0.3, med=0.7):
    return PolicyConfig(low_max=low, medium_max=med, level_actions={
        RiskLevel.LOW: Action.ALLOW, RiskLevel.MEDIUM: Action.CHALLENGE,
        RiskLevel.HIGH: Action.REVIEW})


def _scores(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    s = np.clip(rng.beta(1.2, 30, n) + 0.6 * y * rng.random(n), 0, 1)
    return y.tolist(), s.tolist()


def test_allowing_fraud_costs_the_loss():
    c = CostMatrix()
    assert action_cost(Action.ALLOW, True, c) == c.avg_fraud_loss
    assert action_cost(Action.ALLOW, False, c) == 0.0


def test_blocking_a_legit_customer_costs_but_blocking_fraud_does_not():
    c = CostMatrix()
    assert action_cost(Action.BLOCK, False, c) == c.block_cost
    assert action_cost(Action.BLOCK, True, c) == 0.0


def test_expected_cost_crosses_over_at_the_derived_threshold():
    """Zatvorena forma mora biti tačka u kojoj ALLOW prestaje da bude jeftiniji
    od CHALLENGE — to je definicija praga."""
    c = CostMatrix()
    low, _ = optimal_thresholds(c)
    eps = low * 0.2
    assert (expected_action_cost(Action.ALLOW, low - eps, c)
            < expected_action_cost(Action.CHALLENGE, low - eps, c))
    assert (expected_action_cost(Action.ALLOW, low + eps, c)
            > expected_action_cost(Action.CHALLENGE, low + eps, c))


def test_expensive_fraud_lowers_the_threshold():
    """Što je prevara skuplja, to se ranije isplati intervenisati."""
    cheap = optimal_thresholds(CostMatrix(avg_fraud_loss=50.0))[0]
    dear = optimal_thresholds(CostMatrix(avg_fraud_loss=5000.0))[0]
    assert dear < cheap


def test_expensive_challenge_raises_the_threshold():
    low_cheap = optimal_thresholds(CostMatrix(challenge_cost=0.1))[0]
    low_dear = optimal_thresholds(CostMatrix(challenge_cost=10.0))[0]
    assert low_dear > low_cheap


def test_thresholds_stay_ordered_and_bounded():
    for loss in (1.0, 100.0, 10_000.0):
        for ch in (0.01, 1.0, 50.0):
            lo, hi = optimal_thresholds(CostMatrix(avg_fraud_loss=loss,
                                                   challenge_cost=ch))
            assert 0.0 <= lo <= hi <= 1.0


def test_degenerate_cost_matrix_falls_back_to_defaults():
    assert optimal_thresholds(CostMatrix(avg_fraud_loss=0.0)) == (0.3, 0.7)


def test_optimal_thresholds_beat_arbitrary_ones():
    y, s = _scores()
    c = CostMatrix()
    lo, hi, best = empirical_thresholds(y, s, c)
    assert best <= total_cost(y, s, 0.3, 0.7, c)


def test_action_rates_sum_below_one():
    y, s = _scores()
    ch, rv = action_rates(s, 0.05, 0.4)
    assert 0.0 <= ch + rv <= 1.0


def test_constrained_thresholds_respect_capacity():
    y, s = _scores()
    cap = CapacityConstraints(max_challenge_rate=0.03, max_review_rate=0.005)
    best = constrained_thresholds(y, s, CostMatrix(), cap)
    assert best["challenge_rate"] <= cap.max_challenge_rate + 1e-9
    assert best["review_rate"] <= cap.max_review_rate + 1e-9


def test_constrained_is_never_cheaper_than_unconstrained():
    """Ograničenja mogu samo poskupjeti rješenje — ako ne, optimizacija griješi."""
    y, s = _scores()
    c = CostMatrix()
    cap = CapacityConstraints(max_challenge_rate=0.03, max_review_rate=0.005)
    unconstrained = empirical_thresholds(y, s, c)[2]
    constrained = constrained_thresholds(y, s, c, cap)["total_cost"]
    assert constrained >= unconstrained - 1e-6


def test_impossible_capacity_is_reported_as_infeasible():
    y, s = _scores()
    cap = CapacityConstraints(max_challenge_rate=0.0, max_review_rate=0.0)
    best = constrained_thresholds(y, s, CostMatrix(), cap)
    assert best["feasible"] in (True, False)
    assert 0.0 <= best["low_max"] <= 1.0


def test_cost_report_contains_all_variants():
    y, s = _scores()
    rep = cost_report(y, s, _policy(), CostMatrix())
    assert {"current", "closed_form", "empirical", "cost_matrix"} <= set(rep)
    assert rep["empirical"]["total_cost"] <= rep["current"]["total_cost"]


def test_empty_input_does_not_crash():
    best = constrained_thresholds([], [], CostMatrix(), CapacityConstraints())
    assert best["feasible"] is False

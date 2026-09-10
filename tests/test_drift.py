import numpy as np
import pytest

from spr.learning.drift import (
    PSI_SIGNIFICANT, PSI_STABLE, classify_psi, drift_report, feature_drift,
    score_drift,
)


def _rows(n, shift=0.0, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return [{"amount": float(v), "amount_ratio": float(r)}
            for v, r in zip(rng.normal(100 + shift, 20 * scale, n),
                            rng.normal(1.0 + shift / 100, 0.3, n))]


def test_same_distribution_is_stable():
    a, b = _rows(3000, seed=1), _rows(3000, seed=2)
    for d in feature_drift(a, b, features=["amount", "amount_ratio"]):
        assert d.psi < PSI_STABLE
        assert d.severity == "stabilno"


def test_shifted_distribution_is_detected():
    a, b = _rows(3000, seed=1), _rows(3000, shift=120.0, seed=2)
    drift = feature_drift(a, b, features=["amount"])[0]
    assert drift.psi > PSI_SIGNIFICANT
    assert drift.severity == "značajno"
    assert drift.ks_pvalue < 0.01


def test_drift_is_sorted_by_severity():
    a = _rows(2000, seed=1)
    b = [{**r, "amount": r["amount"] + 200} for r in _rows(2000, seed=2)]
    out = feature_drift(a, b, features=["amount", "amount_ratio"])
    assert out[0].feature == "amount"
    assert out == sorted(out, key=lambda d: d.psi, reverse=True)


def test_classify_psi_boundaries():
    assert classify_psi(0.0) == "stabilno"
    assert classify_psi(PSI_STABLE) == "umjereno"
    assert classify_psi(PSI_SIGNIFICANT) == "značajno"


def test_constant_feature_does_not_divide_by_zero():
    a = [{"x": 1.0} for _ in range(100)]
    b = [{"x": 1.0} for _ in range(100)]
    out = feature_drift(a, b, features=["x"])
    assert out[0].psi == 0.0


def test_empty_input_is_handled():
    assert feature_drift([], [{"x": 1.0}], features=["x"]) == []
    assert score_drift([], [])["severity"] == "stabilno"


def test_score_drift_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.beta(1, 20, 2000).tolist()
    cur = rng.beta(3, 5, 2000).tolist()
    out = score_drift(ref, cur)
    assert out["psi"] > PSI_SIGNIFICANT
    assert out["current_mean"] > out["reference_mean"]


def test_drift_report_shape_and_counts():
    a, b = _rows(1500, seed=1), _rows(1500, shift=150.0, seed=2)
    rep = drift_report(a, b, [0.1] * 1500, [0.5] * 1500)
    assert rep["n_reference"] == 1500 and rep["n_current"] == 1500
    assert rep["overall"] == "značajno"
    assert rep["n_drifting"] >= 1
    assert rep["worst_feature"] in ("amount", "amount_ratio")
    assert "score" in rep

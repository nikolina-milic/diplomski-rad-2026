from datetime import datetime, timezone

from spr.engine.factory import build_default_engine
from spr.evaluation.report import (
    APPROACHES, evaluate_approaches, fit_fusion_models, run_evaluation,
    save_report, significance_tests,
)
from spr.generator.generator import generate_batch
from spr.ml.dataset import build_dataset

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_evaluate_approaches_structure():
    eng = build_default_engine(n_normal=400, days=7)
    ev = generate_batch(n_users=15, n_normal=400, fraud_rate=0.05, seed=3,
                        start=START, days=7)
    X, y = build_dataset(ev)
    res = evaluate_approaches(eng.model, eng.rules, eng.fusion_config, X, y)
    assert set(res) == set(APPROACHES)
    for v in res.values():
        m = v["metrics"]
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert len(v["pr_curve"]) > 0
        assert 0.0 <= v["calibration"]["brier"] <= 1.0
        assert 0.0 <= v["calibration"]["ece"] <= 1.0
    assert res["ml"]["metrics"]["pr_auc"] > 0.4  # model uči signal


def test_hybrids_use_fitted_fusion_when_provided():
    """Sa kalibrisanim fusion modelom hibrid daje drugačije skorove nego
    sirova kombinacija — inače kalibracija ne bi ništa mijenjala."""
    eng = build_default_engine(n_normal=600, days=10)
    tr = generate_batch(n_users=15, n_normal=600, fraud_rate=0.05, seed=1,
                        start=START, days=10)
    te = generate_batch(n_users=15, n_normal=400, fraud_rate=0.05, seed=9,
                        start=START, days=10)
    Xtr, ytr = build_dataset(tr)
    Xte, yte = build_dataset(te)
    fms = fit_fusion_models(eng.model, eng.rules, eng.fusion_config, Xtr, ytr)
    with_fit = evaluate_approaches(eng.model, eng.rules, eng.fusion_config,
                                   Xte, yte, fms)
    without = evaluate_approaches(eng.model, eng.rules, eng.fusion_config,
                                  Xte, yte)
    assert (with_fit["hybrid_stacking"]["metrics"]["pr_auc"]
            != without["hybrid_stacking"]["metrics"]["pr_auc"])


def test_run_evaluation_and_save(tmp_path):
    rep = run_evaluation(n_train=400, n_test=300, days=7)
    assert rep["n_test"] > 0 and rep["n_fraud"] > 0
    assert rep["latency"]["mean_ms"] > 0
    assert set(rep["approaches"]) == set(APPROACHES)
    assert "cost" in rep and "drift" in rep
    p = str(tmp_path / "r.json")
    save_report(rep, p)
    import json, os
    assert os.path.exists(p)
    assert json.load(open(p, encoding="utf-8"))["n_test"] == rep["n_test"]


def test_multiseed_reports_spread_and_significance():
    rep = run_evaluation(n_train=400, n_test=300, days=7, n_seeds=3)
    agg = rep["aggregated"]
    assert set(agg) == set(APPROACHES)
    for metrics in agg.values():
        assert len(metrics["pr_auc"]["values"]) == 3
        assert metrics["pr_auc"]["std"] >= 0.0
    assert rep["significance"], "sa više seedova mora postojati poređenje"
    for t in rep["significance"]:
        assert t["a"] != t["b"]
        assert "mean_diff" in t


def test_significance_needs_enough_seeds():
    """Sa dva seeda Wilcoxon nema snagu — test to mora prijaviti, ne izmisliti."""
    per_seed = {"ml": {"pr_auc": [0.9, 0.91]},
                "rules": {"pr_auc": [0.4, 0.41]}}
    out = significance_tests(per_seed)
    assert out[0]["p_value"] is None
    assert out[0]["significant"] is False

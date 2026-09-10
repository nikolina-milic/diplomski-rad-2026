"""Eksperimenti iz `spr.evaluation.experiments`.

Testovi su mali (brzi) — provjeravaju strukturu i ključna svojstva, ne tačne
brojeve, koji zavise od seeda.
"""

import pytest

from spr.evaluation.experiments import (
    drift_scenarios, feature_ablation, feedback_value, label_scarcity,
    model_zoo, novel_attack, rule_ablation,
)

SMALL = dict(seed=1, n_train=500, n_test=400, days=10, fraud_rate=0.05)


@pytest.fixture(scope="module")
def zoo():
    return model_zoo(**SMALL)


def test_model_zoo_covers_supervised_and_unsupervised(zoo):
    assert {"random_forest", "logistic", "hist_gb", "isolation_forest"} <= set(zoo)
    assert zoo["isolation_forest"]["supervised"] is False
    assert zoo["random_forest"]["supervised"] is True


def test_supervised_models_beat_the_unsupervised_baseline(zoo):
    """Ako nenadzirani detektor ne zaostaje, labele ne donose ništa —
    to bi značilo da nadzirani pristup nema opravdanje."""
    unsup = zoo["isolation_forest"]["metrics"]["pr_auc"]
    best = max(zoo[a]["metrics"]["pr_auc"]
               for a in ("random_forest", "logistic", "hist_gb"))
    assert best > unsup


def test_rule_ablation_ranks_rules_by_impact():
    out = rule_ablation(**SMALL)
    deltas = [r["delta"] for r in out["rules"]]
    assert deltas == sorted(deltas, reverse=True)
    assert all({"rule", "weight", "pr_auc_without", "delta"} <= set(r)
               for r in out["rules"])


def test_feature_ablation_ranks_features_by_impact():
    out = feature_ablation(**SMALL)
    deltas = [r["delta"] for r in out["features"]]
    assert deltas == sorted(deltas, reverse=True)
    assert len(out["features"]) == 12   # svih 12 obilježja


def test_novel_attack_reports_every_pattern_it_could_test():
    out = novel_attack(**SMALL)
    assert out, "nijedan obrazac nije mogao biti testiran"
    for pattern, row in out.items():
        assert {"rules", "ml", "hybrid_stacking", "hybrid_cascade"} <= set(row)
        assert row["n_test_fraud"] >= 3


def test_rules_hold_up_on_impossible_travel_without_training_on_it():
    """Pravilo pisano za fizički nemoguće putovanje ne treba trening primjere —
    to je suština argumenta za hibrid."""
    out = novel_attack(**SMALL)
    row = out.get("impossible_travel")
    if row is None:
        pytest.skip("premalo primjera ovog obrasca pri ovoj veličini skupa")
    assert row["rules"] >= row["ml"]


def test_label_scarcity_is_monotone_in_information():
    out = label_scarcity(**SMALL, fractions=(0.1, 1.0))
    levels = out["levels"]
    assert levels[0]["n_labels"] <= levels[-1]["n_labels"]
    # rezultat pravila ne zavisi od količine labela
    assert len({r["rules"] for r in levels}) == 1


def test_feedback_value_compares_three_label_sources():
    out = feedback_value(**SMALL)
    assert {"full_ground_truth", "analyst_feedback", "random_sample"} <= set(out)
    assert out["n_reviewed"] > 0


def test_drift_detector_is_quiet_on_same_distribution_and_loud_on_change():
    out = drift_scenarios(**SMALL)
    assert out["ista_raspodjela"]["max_psi"] < out["promijenjen_napad"]["max_psi"]

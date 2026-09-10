import numpy as np
import pandas as pd

from spr.realdata.ulb import MODEL_FEATURES, add_time_features, rule_scores, to_matrix


def _frame():
    data = {
        "Time": [0.0, 7200.0, 90000.0],
        "Amount": [10.0, 2500.0, 6000.0],
        "Class": [0, 0, 1],
    }
    data.update({f"V{i}": [0.0, float(i), -float(i)] for i in range(1, 29)})
    return pd.DataFrame(data)


def test_ulb_time_features_and_matrix_are_finite():
    frame = add_time_features(_frame())
    matrix = to_matrix(frame)
    assert matrix.shape == (3, len(MODEL_FEATURES))
    assert np.isfinite(matrix).all()


def test_ulb_rules_use_only_amount_and_time():
    frame = add_time_features(_frame())
    scores, rates = rule_scores(frame)
    assert scores[0] == 0.0
    assert scores[1] >= 0.7
    assert scores[2] > scores[1]
    assert rates["very_large_amount"] == 2 / 3


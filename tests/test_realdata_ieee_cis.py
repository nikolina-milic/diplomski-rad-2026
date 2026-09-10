import numpy as np
import pandas as pd

from spr.realdata.ieee_cis import (
    RealDataEncoder, add_behavioral_features, load_ieee_cis, rule_scores,
)


def _frame():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "isFraud": [0, 0, 0, 0, 0, 1],
        "TransactionDT": [0, 100, 200, 300, 400, 500],
        "TransactionAmt": [10, 10, 10, 10, 10, 200],
        "card1": [100] * 6,
        "addr1": [20] * 6,
        "P_emaildomain": ["a.test"] * 6,
        "R_emaildomain": ["a.test"] * 6,
        "DeviceInfo": ["phone"] * 5 + ["new-phone"],
        "ProductCD": ["W"] * 6,
    })


def test_behavioral_features_use_only_prior_rows():
    frame = add_behavioral_features(_frame())
    assert frame.loc[0, "amount_ratio"] == 1.0
    assert frame.loc[5, "amount_ratio"] == 20.0
    assert frame.loc[5, "txn_count_1h"] == 5.0
    assert frame.loc[5, "is_new_device"] == 1.0


def test_rule_score_fires_for_large_rapid_new_device_event():
    frame = add_behavioral_features(_frame())
    scores, rates = rule_scores(frame)
    assert scores[5] > 0.9
    assert rates["very_large_amount"] > 0
    assert rates["rapid_fire"] > 0


def test_encoder_learns_imputation_and_unknown_categories_from_training_only():
    frame = add_behavioral_features(_frame())
    encoder = RealDataEncoder().fit(frame.iloc[:5])
    matrix = encoder.transform(frame.iloc[5:])
    assert matrix.shape == (1, len(encoder.feature_names))
    assert np.isfinite(matrix).all()


def test_loader_merges_identity_file(tmp_path):
    transaction = _frame().drop(columns=["DeviceInfo"])
    identity = pd.DataFrame({
        "TransactionID": [1, 6],
        "DeviceInfo": ["phone", "new-phone"],
        "DeviceType": ["mobile", "mobile"],
    })
    transaction.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    loaded = load_ieee_cis(tmp_path)
    assert len(loaded) == 6
    assert loaded.loc[5, "DeviceInfo"] == "new-phone"


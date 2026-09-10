"""Vremenska evaluacija na stvarnom ULB/Worldline credit-card skupu.

Skup sadrži 284.807 transakcija evropskih korisnika kartica, od kojih su 492
označene kao prevara. V1--V28 su PCA komponente čije značenje nije objavljeno;
zato poslovna pravila namjerno koriste samo javno razumljive kolone ``Time`` i
``Amount``. Ova evaluacija provjerava spoljašnju validnost klasifikacije, ali ne
može testirati konkretne tipove prevare niti korisnički kontekst.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

from spr.fusion.model import FusionModel
from spr.fusion.strategies import FusionConfig
from spr.learning.metrics import compute_metrics
from spr.ml.calibration import calibration_report

PCA_FEATURES = tuple(f"V{i}" for i in range(1, 29))
REQUIRED_COLUMNS = ("Time", *PCA_FEATURES, "Amount", "Class")
MODEL_FEATURES = (*PCA_FEATURES, "Amount", "log_amount", "hour_sin", "hour_cos")
APPROACHES = (
    "rules", "ml", "hybrid_weighted", "hybrid_cascade", "hybrid_stacking",
)
METRICS = ("pr_auc", "f1", "precision", "recall")

RULE_DEFINITIONS = (
    {
        "name": "very_large_amount",
        "condition": "Amount > 2000",
        "weight": 0.70,
        "explanation": "Veoma visok iznos transakcije",
    },
    {
        "name": "extreme_amount",
        "condition": "Amount > 5000",
        "weight": 0.90,
        "explanation": "Ekstremno visok iznos transakcije",
    },
    {
        "name": "large_night_transaction",
        "condition": "01:00 <= hour < 06:00 and Amount > 500",
        "weight": 0.50,
        "explanation": "Visok iznos u noćnom periodu",
    },
)


def load_ulb(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Nedostaju obavezne kolone: {', '.join(missing)}")
    frame = pd.read_csv(source, usecols=list(REQUIRED_COLUMNS))
    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("ULB skup neočekivano sadrži nedostajuće vrijednosti")
    labels = set(frame["Class"].astype(int).unique().tolist())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Neočekivane klase: {sorted(labels)}")
    return add_time_features(frame)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hour = np.mod(out["Time"].to_numpy(float) / 3600.0, 24.0)
    angle = 2.0 * np.pi * hour / 24.0
    out["hour_of_day"] = hour
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["log_amount"] = np.log1p(np.maximum(out["Amount"].to_numpy(float), 0.0))
    return out


def to_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, MODEL_FEATURES].to_numpy(np.float32)


def rule_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    amount = frame["Amount"].to_numpy(float)
    hour = frame["hour_of_day"].to_numpy(float)
    masks = {
        "very_large_amount": amount > 2000.0,
        "extreme_amount": amount > 5000.0,
        "large_night_transaction": (
            (hour >= 1.0) & (hour < 6.0) & (amount > 500.0)
        ),
    }
    weights = {rule["name"]: rule["weight"] for rule in RULE_DEFINITIONS}
    survival = np.ones(len(frame), dtype=float)
    rates = {}
    for name, mask in masks.items():
        survival[mask] *= 1.0 - weights[name]
        rates[name] = float(mask.mean()) if len(mask) else 0.0
    return 1.0 - survival, rates


def _fit_model(frame: pd.DataFrame, y: np.ndarray, seed: int,
               n_estimators: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(to_matrix(frame), y)
    return model


def _positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return model.predict_proba(to_matrix(frame))[:, classes.index(1)]


def _oof_scores(frame: pd.DataFrame, y: np.ndarray, seed: int,
                n_estimators: int, splits: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(
        TimeSeriesSplit(n_splits=splits).split(frame)
    ):
        model = _fit_model(
            frame.iloc[train_idx], y[train_idx], seed + fold, n_estimators,
        )
        scores[valid_idx] = _positive_probability(model, frame.iloc[valid_idx])
    valid = np.isfinite(scores)
    return scores, valid


def _best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _fit_fusions(rules_oof, ml_oof, y_oof) -> dict[str, FusionModel]:
    strategies = {
        "hybrid_weighted": "weighted",
        "hybrid_cascade": "cascade",
        "hybrid_stacking": "stacking",
    }
    return {
        name: FusionModel(FusionConfig(strategy=strategy)).fit(
            rules_oof, ml_oof, y_oof,
        )
        for name, strategy in strategies.items()
    }


def _evaluate_once(frame: pd.DataFrame, seed: int, test_fraction: float,
                   n_estimators: int, oof_splits: int) -> dict:
    split = int(len(frame) * (1.0 - test_fraction))
    train, test = frame.iloc[:split], frame.iloc[split:]
    y_train = train["Class"].astype(int).to_numpy()
    y_test = test["Class"].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        raise ValueError("Vremenski trening i test moraju sadržati obje klase")

    rules_all, trigger_rates = rule_scores(frame)
    rules_train, rules_test = rules_all[:split], rules_all[split:]
    ml_oof, valid = _oof_scores(
        train, y_train, seed, n_estimators, oof_splits,
    )
    y_oof = y_train[valid]
    rules_oof = rules_train[valid]
    ml_oof = ml_oof[valid]

    model = _fit_model(train, y_train, seed, n_estimators)
    ml_test = _positive_probability(model, test)
    fusions = _fit_fusions(rules_oof, ml_oof, y_oof)

    train_scores = {"rules": rules_oof, "ml": ml_oof}
    test_scores = {"rules": rules_test, "ml": ml_test}
    for name, fusion in fusions.items():
        train_scores[name] = np.asarray(
            fusion.combine_batch(rules_oof, ml_oof), dtype=float,
        )
        test_scores[name] = np.asarray(
            fusion.combine_batch(rules_test, ml_test), dtype=float,
        )

    approaches = {}
    for name in APPROACHES:
        threshold = _best_f1_threshold(y_oof, train_scores[name])
        approaches[name] = {
            "threshold": threshold,
            "metrics": compute_metrics(y_test, test_scores[name], threshold),
            "calibration": calibration_report(y_test, test_scores[name]),
        }

    importance = sorted(
        zip(MODEL_FEATURES, model.feature_importances_),
        key=lambda pair: -pair[1],
    )
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_train_fraud": int(y_train.sum()),
        "n_test_fraud": int(y_test.sum()),
        "train_end_time": float(train["Time"].iloc[-1]),
        "test_start_time": float(test["Time"].iloc[0]),
        "approaches": approaches,
        "rule_trigger_rates": trigger_rates,
        "feature_importance": [
            {"feature": name, "importance": float(value)}
            for name, value in importance
        ],
        "meta_weights": fusions["hybrid_stacking"].weights(),
    }


def _aggregate(runs: list[dict]) -> dict:
    result = {}
    for approach in APPROACHES:
        result[approach] = {}
        for metric in METRICS:
            values = np.asarray([
                run["approaches"][approach]["metrics"][metric] for run in runs
            ], dtype=float)
            result[approach][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return result


def _significance(runs: list[dict], alpha: float = 0.05) -> list[dict]:
    if len(runs) < 3:
        return []
    tests = []
    for i, left in enumerate(APPROACHES):
        for right in APPROACHES[i + 1:]:
            a = np.asarray([
                run["approaches"][left]["metrics"]["pr_auc"] for run in runs
            ])
            b = np.asarray([
                run["approaches"][right]["metrics"]["pr_auc"] for run in runs
            ])
            diff = a - b
            if np.allclose(diff, 0.0):
                p_value = None
            else:
                _, p_value = wilcoxon(a, b)
                p_value = float(p_value)
            tests.append({
                "a": left,
                "b": right,
                "mean_diff": float(diff.mean()),
                "p_value": p_value,
                "significant": bool(p_value is not None and p_value < alpha),
                "note": "ponavljanja mijenjaju slučajno stanje modela na istoj vremenskoj podjeli",
            })
    return tests


def run_ulb_evaluation(
    path: str | Path,
    *,
    seeds: int = 10,
    test_fraction: float = 0.20,
    n_estimators: int = 100,
    oof_splits: int = 3,
) -> dict:
    frame = load_ulb(path)
    runs = [
        _evaluate_once(frame, seed, test_fraction, n_estimators, oof_splits)
        for seed in range(seeds)
    ]
    first = runs[0]
    return {
        "dataset": {
            "name": "ULB/Worldline Credit Card Fraud Detection",
            "source": "https://www.openml.org/d/1597",
            "real_world": True,
            "n_rows": len(frame),
            "n_fraud": int(frame["Class"].sum()),
            "fraud_rate": float(frame["Class"].mean()),
            "time_min": float(frame["Time"].min()),
            "time_max": float(frame["Time"].max()),
        },
        "protocol": {
            "split": f"prvih {1-test_fraction:.0%} trening, posljednjih {test_fraction:.0%} test",
            "oof": f"TimeSeriesSplit({oof_splits}) za kalibraciju, pragove i stacking",
            "seeds": list(range(seeds)),
            "n_estimators": n_estimators,
            "primary_metric": "PR-AUC",
        },
        "split": {key: first[key] for key in (
            "n_train", "n_test", "n_train_fraud", "n_test_fraud",
            "train_end_time", "test_start_time",
        )},
        "approaches": first["approaches"],
        "aggregated": _aggregate(runs),
        "significance": _significance(runs),
        "rule_definitions": list(RULE_DEFINITIONS),
        "rule_trigger_rates": first["rule_trigger_rates"],
        "feature_importance": first["feature_importance"],
        "meta_weights": first["meta_weights"],
        "limitations": [
            "V1--V28 su anonimizovane PCA komponente bez objavljenog semantičkog značenja.",
            "Skup ne sadrži identifikator korisnika, uređaj, lokaciju ni tip prevare.",
            "Rules-only koristi samo Time i Amount i zato nije ekvivalentan punom bankarskom domain pack-u.",
            "Leave-one-pattern-out nije moguć jer tipovi prevare nijesu označeni.",
            "Ponavljanja koriste istu vremensku podjelu i mjere stabilnost slučajne šume, ne varijabilnost uzorka.",
        ],
    }


def save_report(report: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPR evaluacija na ULB skupu")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--oof-splits", type=int, default=3)
    parser.add_argument("--out", default="reports/ulb_evaluation.json")
    args = parser.parse_args()
    report = run_ulb_evaluation(
        args.data, seeds=args.seeds, test_fraction=args.test_fraction,
        n_estimators=args.n_estimators, oof_splits=args.oof_splits,
    )
    save_report(report, args.out)
    ds, split = report["dataset"], report["split"]
    print(f"ULB: {ds['n_rows']} događaja, {ds['n_fraud']} prevara "
          f"({ds['fraud_rate']:.4%})")
    print(f"Vremenska podjela: {split['n_train']} trening / {split['n_test']} test")
    print(f"{'Pristup':<22}{'PR-AUC':>16}{'F1':>16}{'Preciznost':>16}{'Odziv':>16}")
    for name, metrics in report["aggregated"].items():
        cells = [
            f"{metrics[key]['mean']:.3f}±{metrics[key]['std']:.3f}"
            for key in ("pr_auc", "f1", "precision", "recall")
        ]
        print(f"{name:<22}" + "".join(f"{cell:>16}" for cell in cells))
    print(f"\nIzvještaj zapisan: {args.out}")


if __name__ == "__main__":
    main()


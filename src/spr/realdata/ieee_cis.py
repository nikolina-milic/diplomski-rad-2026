"""Evaluacija na stvarnom IEEE-CIS/Vesta skupu transakcija.

Skup se preuzima sa zvanične Kaggle stranice i zbog uslova licence nije dio
repozitorijuma. Ovaj modul koristi samo označeni ``train`` dio, sortira ga po
``TransactionDT`` i posljednjih 20% događaja čuva za testiranje.

Važna ograničenja skupa:

* nema pravi ``user_id``; entitet je aproksimiran sa ``card1 + addr1``;
* nema geografsku širinu/dužinu, pa pravilo nemogućeg putovanja nije primjenljivo;
* nema tip prevare, pa leave-one-pattern-out eksperiment nije moguć.

Zbog toga je ovo spoljašnja validacija sistema na stvarnim podacima, a ne
zamjena za kontrolisani eksperiment sa označenim obrascima prevare.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
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

APPROACHES = (
    "rules", "ml", "hybrid_weighted", "hybrid_cascade", "hybrid_stacking",
)
METRICS = ("pr_auc", "f1", "precision", "recall")

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

REQUIRED_TRANSACTION_COLUMNS = (
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
)

OPTIONAL_TRANSACTION_COLUMNS = (
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"M{i}" for i in range(1, 10)),
)

OPTIONAL_IDENTITY_COLUMNS = (
    "TransactionID", "DeviceType", "DeviceInfo", "id_01", "id_02",
)

BEHAVIORAL_NUMERIC = (
    "amount", "log_amount", "amount_zscore", "amount_ratio",
    "txn_count_1h", "seconds_since_last", "is_new_device",
    "is_new_p_email", "hour_of_day", "hour_sin", "hour_cos",
    "is_unusual_hour", "email_mismatch",
)

RAW_NUMERIC = (
    "dist1", "dist2", "id_01", "id_02",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
)

CATEGORICAL = (
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", *(f"M{i}" for i in range(1, 10)),
)

RULE_DEFINITIONS = (
    ("very_large_amount", 0.70,
     "Iznos je više od osam puta veći od dotadašnjeg prosjeka entiteta"),
    ("high_amount_deviation", 0.60,
     "Iznos odstupa više od četiri standardne devijacije"),
    ("rapid_fire", 0.60,
     "Najmanje pet ranijih transakcija istog entiteta u prethodnom satu"),
    ("new_device_high_amount", 0.50,
     "Novi uređaj i iznos veći od tri dotadašnja prosjeka"),
    ("new_email_high_amount", 0.40,
     "Nova e-mail oznaka i iznos veći od tri dotadašnja prosjeka"),
    ("unusual_hour_spending", 0.50,
     "Povišen iznos između 01:00 i 05:59"),
    ("email_mismatch_high_amount", 0.30,
     "Različite platna i prijemna e-mail oznaka uz povišen iznos"),
)


def _available_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _select_columns(path: Path, requested) -> list[str]:
    available = set(_available_columns(path))
    return [name for name in requested if name in available]


def load_ieee_cis(data_dir: str | os.PathLike, max_rows: int | None = None) -> pd.DataFrame:
    """Učitaj označene IEEE-CIS transakcije i dostupne identity kolone."""
    root = Path(data_dir)
    transaction_path = root / TRANSACTION_FILE
    identity_path = root / IDENTITY_FILE
    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Nedostaje {transaction_path}. Preuzmi IEEE-CIS podatke sa "
            "zvanične Kaggle stranice i raspakuj ih u ovaj direktorijum."
        )

    transaction_columns = _select_columns(
        transaction_path, REQUIRED_TRANSACTION_COLUMNS + OPTIONAL_TRANSACTION_COLUMNS,
    )
    missing = sorted(set(REQUIRED_TRANSACTION_COLUMNS) - set(transaction_columns))
    if missing:
        raise ValueError(f"Nedostaju obavezne kolone: {', '.join(missing)}")

    frame = pd.read_csv(
        transaction_path, usecols=transaction_columns, nrows=max_rows,
        low_memory=False,
    )
    if identity_path.exists():
        identity_columns = _select_columns(identity_path, OPTIONAL_IDENTITY_COLUMNS)
        identity = pd.read_csv(identity_path, usecols=identity_columns, low_memory=False)
        frame = frame.merge(identity, how="left", on="TransactionID", sort=False)

    frame = frame.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable",
    ).reset_index(drop=True)
    return frame


def _category_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), "__missing__", dtype=object)
    return frame[name].astype("string").fillna("__missing__").astype(str).to_numpy()


def add_behavioral_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Dodaj obilježja koristeći isključivo istoriju prije trenutnog reda."""
    out = frame.copy()
    n = len(out)
    amount = pd.to_numeric(out["TransactionAmt"], errors="coerce").fillna(0.0).to_numpy(float)
    timestamp = pd.to_numeric(out["TransactionDT"], errors="coerce").fillna(0.0).to_numpy(float)
    transaction_id = out["TransactionID"].astype(str).to_numpy()
    card1 = _category_array(out, "card1")
    addr1 = _category_array(out, "addr1")
    device = _category_array(out, "DeviceInfo")
    p_email = _category_array(out, "P_emaildomain")
    r_email = _category_array(out, "R_emaildomain")

    keys = np.char.add(np.char.add(card1.astype(str), "|"), addr1.astype(str))
    unknown = (card1 == "__missing__") & (addr1 == "__missing__")
    keys[unknown] = np.char.add("unknown|", transaction_id[unknown].astype(str))
    entity_codes, _ = pd.factorize(keys, sort=False)

    amount_zscore = np.zeros(n, dtype=np.float32)
    amount_ratio = np.ones(n, dtype=np.float32)
    txn_count_1h = np.zeros(n, dtype=np.float32)
    seconds_since_last = np.zeros(n, dtype=np.float32)
    is_new_device = np.zeros(n, dtype=np.float32)
    is_new_p_email = np.zeros(n, dtype=np.float32)

    count = defaultdict(int)
    total = defaultdict(float)
    total_sq = defaultdict(float)
    last_ts: dict[int, float] = {}
    recent: dict[int, deque] = defaultdict(deque)
    devices: dict[int, set[str]] = defaultdict(set)
    emails: dict[int, set[str]] = defaultdict(set)

    for i in range(n):
        entity = int(entity_codes[i])
        a = float(amount[i])
        t = float(timestamp[i])
        seen = count[entity]
        if seen:
            mu = total[entity] / seen
            variance = max(total_sq[entity] / seen - mu * mu, 0.0)
            sigma = max(math.sqrt(variance), 1.0)
            amount_zscore[i] = (a - mu) / sigma
            amount_ratio[i] = a / max(mu, 1e-9)
            seconds_since_last[i] = max(0.0, t - last_ts[entity])

        queue = recent[entity]
        while queue and queue[0] < t - 3600.0:
            queue.popleft()
        txn_count_1h[i] = len(queue)

        current_device = device[i]
        if current_device != "__missing__" and devices[entity]:
            is_new_device[i] = float(current_device not in devices[entity])
        current_email = p_email[i]
        if current_email != "__missing__" and emails[entity]:
            is_new_p_email[i] = float(current_email not in emails[entity])

        count[entity] += 1
        total[entity] += a
        total_sq[entity] += a * a
        last_ts[entity] = t
        queue.append(t)
        if current_device != "__missing__":
            devices[entity].add(current_device)
        if current_email != "__missing__":
            emails[entity].add(current_email)

    hour = np.mod(timestamp / 3600.0, 24.0)
    angle = 2.0 * np.pi * hour / 24.0
    out["amount"] = amount
    out["log_amount"] = np.log1p(np.maximum(amount, 0.0))
    out["amount_zscore"] = amount_zscore
    out["amount_ratio"] = amount_ratio
    out["txn_count_1h"] = txn_count_1h
    out["seconds_since_last"] = seconds_since_last
    out["is_new_device"] = is_new_device
    out["is_new_p_email"] = is_new_p_email
    out["hour_of_day"] = hour.astype(np.float32)
    out["hour_sin"] = np.sin(angle).astype(np.float32)
    out["hour_cos"] = np.cos(angle).astype(np.float32)
    out["is_unusual_hour"] = ((hour >= 1.0) & (hour < 6.0)).astype(np.float32)
    out["email_mismatch"] = (
        (p_email != "__missing__") & (r_email != "__missing__") & (p_email != r_email)
    ).astype(np.float32)
    return out


def rule_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Vektorizovani noisy-OR skor pravila prilagođenih IEEE-CIS kolonama."""
    ratio = frame["amount_ratio"].to_numpy(float)
    masks = {
        "very_large_amount": ratio > 8.0,
        "high_amount_deviation": frame["amount_zscore"].to_numpy(float) > 4.0,
        "rapid_fire": frame["txn_count_1h"].to_numpy(float) >= 5.0,
        "new_device_high_amount": (
            (frame["is_new_device"].to_numpy(float) >= 1.0) & (ratio > 3.0)
        ),
        "new_email_high_amount": (
            (frame["is_new_p_email"].to_numpy(float) >= 1.0) & (ratio > 3.0)
        ),
        "unusual_hour_spending": (
            (frame["is_unusual_hour"].to_numpy(float) >= 1.0) & (ratio > 3.0)
        ),
        "email_mismatch_high_amount": (
            (frame["email_mismatch"].to_numpy(float) >= 1.0) & (ratio > 2.0)
        ),
    }
    survival = np.ones(len(frame), dtype=float)
    rates: dict[str, float] = {}
    weights = {name: weight for name, weight, _ in RULE_DEFINITIONS}
    for name, mask in masks.items():
        survival[mask] *= 1.0 - weights[name]
        rates[name] = float(np.mean(mask)) if len(mask) else 0.0
    return 1.0 - survival, rates


class RealDataEncoder:
    """Trening-only imputacija i kodiranje kategorija bez korišćenja labela."""

    def __init__(self) -> None:
        self.numeric = []
        self.categorical = []
        self.medians: dict[str, float] = {}
        self.codes: dict[str, dict[str, int]] = {}
        self.frequencies: dict[str, dict[str, float]] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "RealDataEncoder":
        self.numeric = [c for c in BEHAVIORAL_NUMERIC + RAW_NUMERIC if c in frame]
        self.categorical = [c for c in CATEGORICAL if c in frame]
        for name in self.numeric:
            values = pd.to_numeric(frame[name], errors="coerce")
            median = values.median()
            self.medians[name] = 0.0 if pd.isna(median) else float(median)
        n = max(len(frame), 1)
        for name in self.categorical:
            values = frame[name].astype("string").fillna("__missing__").astype(str)
            unique = pd.Index(values.unique())
            self.codes[name] = {value: i for i, value in enumerate(unique, start=1)}
            counts = values.value_counts(dropna=False)
            self.frequencies[name] = {str(k): float(v / n) for k, v in counts.items()}
        self.feature_names = list(self.numeric)
        for name in self.categorical:
            self.feature_names.extend((f"{name}_code", f"{name}_frequency"))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name in self.numeric:
            values = pd.to_numeric(frame[name], errors="coerce").fillna(
                self.medians[name]
            )
            columns.append(values.to_numpy(np.float32))
        for name in self.categorical:
            values = frame[name].astype("string").fillna("__missing__").astype(str)
            columns.append(values.map(self.codes[name]).fillna(0).to_numpy(np.float32))
            columns.append(
                values.map(self.frequencies[name]).fillna(0.0).to_numpy(np.float32)
            )
        if not columns:
            raise ValueError("Nijedno obilježje nije dostupno za model")
        return np.column_stack(columns)


@dataclass
class EncodedForest:
    encoder: RealDataEncoder
    model: RandomForestClassifier

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.encoder.transform(frame)
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(len(frame), dtype=float)
        return self.model.predict_proba(matrix)[:, classes.index(1)]


def _fit_model(frame: pd.DataFrame, y: np.ndarray, seed: int,
               n_estimators: int) -> EncodedForest:
    encoder = RealDataEncoder().fit(frame)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(encoder.transform(frame), y)
    return EncodedForest(encoder, model)


def _oof_scores(frame: pd.DataFrame, y: np.ndarray, seed: int,
                n_estimators: int, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(frame), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame)):
        model = _fit_model(frame.iloc[train_idx], y[train_idx],
                           seed + fold, n_estimators)
        scores[valid_idx] = model.predict_proba(frame.iloc[valid_idx])
    mask = np.isfinite(scores)
    return scores, mask


def _best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _fusion_models(rules_oof, ml_oof, y_oof) -> dict[str, FusionModel]:
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
    if split < 100 or split >= len(frame):
        raise ValueError("Skup je premali ili test_fraction nije validan")
    train = frame.iloc[:split]
    test = frame.iloc[split:]
    y_train = train["isFraud"].astype(int).to_numpy()
    y_test = test["isFraud"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("I trening i test moraju sadržati legitimne i prevarne događaje")

    all_rules, trigger_rates = rule_scores(frame)
    rules_train = all_rules[:split]
    rules_test = all_rules[split:]
    ml_oof, valid = _oof_scores(train, y_train, seed, n_estimators, oof_splits)
    y_oof = y_train[valid]
    rules_oof = rules_train[valid]
    ml_oof_valid = ml_oof[valid]

    model = _fit_model(train, y_train, seed, n_estimators)
    ml_test = model.predict_proba(test)
    fusions = _fusion_models(rules_oof, ml_oof_valid, y_oof)

    train_scores = {"rules": rules_oof, "ml": ml_oof_valid}
    test_scores = {"rules": rules_test, "ml": ml_test}
    for name, fusion in fusions.items():
        train_scores[name] = np.asarray(
            fusion.combine_batch(rules_oof, ml_oof_valid), dtype=float,
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
    return {
        "split_index": split,
        "n_train": len(train),
        "n_test": len(test),
        "n_train_fraud": int(y_train.sum()),
        "n_test_fraud": int(y_test.sum()),
        "train_end_transaction_dt": float(train["TransactionDT"].iloc[-1]),
        "test_start_transaction_dt": float(test["TransactionDT"].iloc[0]),
        "approaches": approaches,
        "rule_trigger_rates": trigger_rates,
        "model_features": model.encoder.feature_names,
        "meta_weights": fusions["hybrid_stacking"].weights(),
    }


def _aggregate(seed_runs: list[dict]) -> dict:
    out = {}
    for approach in APPROACHES:
        out[approach] = {}
        for metric in METRICS:
            values = np.asarray([
                run["approaches"][approach]["metrics"][metric]
                for run in seed_runs
            ], dtype=float)
            out[approach][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return out


def _significance(seed_runs: list[dict], alpha: float = 0.05) -> list[dict]:
    if len(seed_runs) < 3:
        return []
    out = []
    for i, left in enumerate(APPROACHES):
        for right in APPROACHES[i + 1:]:
            a = np.asarray([
                run["approaches"][left]["metrics"]["pr_auc"] for run in seed_runs
            ])
            b = np.asarray([
                run["approaches"][right]["metrics"]["pr_auc"] for run in seed_runs
            ])
            diff = a - b
            if np.allclose(diff, 0.0):
                p_value = None
            else:
                _, p_value = wilcoxon(a, b)
                p_value = float(p_value)
            out.append({
                "a": left,
                "b": right,
                "mean_diff": float(diff.mean()),
                "p_value": p_value,
                "significant": bool(p_value is not None and p_value < alpha),
                "note": "uparivanje po slučajnom stanju modela na istoj vremenskoj podjeli",
            })
    return out


def run_ieee_cis_evaluation(
    data_dir: str | os.PathLike,
    *,
    seeds: int = 5,
    test_fraction: float = 0.20,
    n_estimators: int = 100,
    oof_splits: int = 3,
    max_rows: int | None = None,
) -> dict:
    raw = load_ieee_cis(data_dir, max_rows=max_rows)
    frame = add_behavioral_features(raw)
    seed_runs = [
        _evaluate_once(frame, seed, test_fraction, n_estimators, oof_splits)
        for seed in range(seeds)
    ]
    first = seed_runs[0]
    return {
        "dataset": {
            "name": "IEEE-CIS Fraud Detection (Vesta)",
            "source": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
            "real_world": True,
            "n_rows": len(frame),
            "n_fraud": int(frame["isFraud"].sum()),
            "fraud_rate": float(frame["isFraud"].mean()),
            "entity_proxy": "card1 + addr1",
        },
        "protocol": {
            "split": f"prvih {1-test_fraction:.0%} trening, posljednjih {test_fraction:.0%} test",
            "oof": f"TimeSeriesSplit({oof_splits}) za kalibraciju i stacking",
            "seeds": list(range(seeds)),
            "n_estimators": n_estimators,
            "threshold": "maksimum F1 na out-of-fold trening predikcijama",
        },
        "split": {k: first[k] for k in (
            "n_train", "n_test", "n_train_fraud", "n_test_fraud",
            "train_end_transaction_dt", "test_start_transaction_dt",
        )},
        "approaches": first["approaches"],
        "aggregated": _aggregate(seed_runs),
        "significance": _significance(seed_runs),
        "rule_trigger_rates": first["rule_trigger_rates"],
        "rule_definitions": [
            {"name": name, "weight": weight, "explanation": explanation}
            for name, weight, explanation in RULE_DEFINITIONS
        ],
        "model_features": first["model_features"],
        "meta_weights": first["meta_weights"],
        "limitations": [
            "Skup nema pravi identifikator korisnika; card1+addr1 je aproksimacija entiteta.",
            "Većina izvornih atributa je anonimizovana, pa pravila ne treba tumačiti kao produkcionu politiku banke.",
            "Skup nema tip prevare; leave-one-pattern-out hipoteza se njime ne može neposredno testirati.",
            "Ponavljanja mijenjaju slučajno stanje modela, ali koriste istu vremensku podjelu podataka.",
        ],
    }


def save_report(report: dict, path: str | os.PathLike) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


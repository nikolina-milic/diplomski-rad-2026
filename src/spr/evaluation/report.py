"""Evaluacioni izvještaj.

Ranija verzija je mjerila na *jednom* seedu, pa se razlika između pristupa nije
mogla razlikovati od šuma. Ovdje se svaki pristup mjeri na više nezavisnih
generisanih skupova, izvještava se srednja vrijednost ± standardna devijacija,
a razlike se provjeravaju Wilcoxonovim testom uparenih rangova.

Uz to: kvalitet kalibracije (Brier, ECE), trošak politike, pomjeranje
raspodjele i latencija.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_curve

from spr.context.store import ContextStore
from spr.domain.banking_pack import (
    BANKING_CAPACITY, BANKING_COST, BANKING_FUSION, BANKING_GUARDRAILS,
    BANKING_POLICY, BANKING_RULES,
)
from spr.engine.engine import DecisionEngine
from spr.evaluation.experiments import (
    TEST_START, TRAIN_START, drift_scenarios, feedback_value, feature_ablation,
    label_scarcity, model_zoo, novel_attack, rule_ablation,
)
from spr.fusion.model import FusionModel
from spr.fusion.strategies import FusionConfig, fuse
from spr.generator.generator import generate_batch
from spr.learning.drift import drift_report
from spr.learning.metrics import compute_metrics
from spr.ml.calibration import calibration_report
from spr.ml.dataset import build_dataset
from spr.ml.model import RiskModel
from spr.policy.cost import (
    action_rates, constrained_thresholds, cost_report, total_cost,
)
from spr.rules.engine import evaluate_rules

APPROACHES = ("rules", "ml", "hybrid_weighted", "hybrid_cascade",
              "hybrid_stacking")
METRIC_NAMES = ("pr_auc", "f1", "precision", "recall")
HYBRID_STRATEGIES = {"hybrid_weighted": "weighted", "hybrid_cascade": "cascade",
                     "hybrid_stacking": "stacking"}


def _pr_curve(y, scores, points: int = 40) -> list[dict]:
    p, r, _ = precision_recall_curve(y, scores)
    idx = np.linspace(0, len(p) - 1, min(points, len(p))).astype(int)
    pts = [{"precision": float(p[i]), "recall": float(r[i])} for i in idx]
    pts.sort(key=lambda d: d["recall"])
    return pts


def fit_fusion_models(model: RiskModel, rules, fusion_cfg, X, y) -> dict:
    """Po jedan kalibrisan FusionModel za svaku strategiju, na trening skupu."""
    rules_scores = [evaluate_rules(x, rules).score for x in X]
    ml_scores = model.oof_scores(X, y)
    out = {}
    for name, strategy in HYBRID_STRATEGIES.items():
        cfg = FusionConfig(strategy=strategy,
                           rules_weight=fusion_cfg.rules_weight,
                           ml_weight=fusion_cfg.ml_weight,
                           cascade_threshold=fusion_cfg.cascade_threshold)
        out[name] = FusionModel(cfg).fit(rules_scores, ml_scores, y)
    return out


def approach_scores(model, rules, fusion_cfg, X, fusion_models=None) -> dict:
    """Skorovi svakog pristupa na istom skupu."""
    rules_scores = [evaluate_rules(x, rules).score for x in X]
    ml_scores = model.predict_proba_batch(X)
    out = {"rules": rules_scores, "ml": ml_scores}
    for name, strategy in HYBRID_STRATEGIES.items():
        fm = (fusion_models or {}).get(name)
        if fm is not None:
            out[name] = fm.combine_batch(rules_scores, ml_scores)
        else:
            # bez kalibracije: gruba kombinacija sirovih skorova (referentno
            # stanje prije uvođenja fusion modela)
            cfg = FusionConfig(strategy=strategy,
                               rules_weight=fusion_cfg.rules_weight,
                               ml_weight=fusion_cfg.ml_weight,
                               cascade_threshold=fusion_cfg.cascade_threshold)
            out[name] = [fuse(r, m, cfg)
                         for r, m in zip(rules_scores, ml_scores)]
    return out


def evaluate_approaches(model, rules, fusion_cfg, X, y, fusion_models=None) -> dict:
    """Metrike, PR kriva i kvalitet kalibracije po pristupu."""
    scores = approach_scores(model, rules, fusion_cfg, X, fusion_models)
    return {name: {"metrics": compute_metrics(y, s),
                   "pr_curve": _pr_curve(y, s),
                   "calibration": calibration_report(y, s)}
            for name, s in scores.items()}


def measure_latency(model, rules, fusion_cfg, policy, guardrails, events,
                    sample: int = 300, fusion=None) -> dict:
    eng = DecisionEngine(model=model, rules=rules, policy=policy,
                         fusion_config=fusion_cfg, guardrails=guardrails,
                         store=ContextStore(), fusion=fusion)
    lat = []
    for e in events[:sample]:
        t0 = time.perf_counter()
        eng.score(e)
        lat.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(lat)
    return {"n": int(arr.size), "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max())}


# --- više seedova + značajnost ------------------------------------------
def _aggregate(per_seed: dict) -> dict:
    """{pristup: {metrika: [po seedu]}} -> srednja vrijednost ± std."""
    out = {}
    for approach, metrics in per_seed.items():
        agg = {}
        for metric, values in metrics.items():
            arr = np.array(values, dtype=float)
            agg[metric] = {
                "mean": float(arr.mean()), "std": float(arr.std(ddof=1))
                if arr.size > 1 else 0.0,
                "min": float(arr.min()), "max": float(arr.max()),
                "values": [float(v) for v in arr],
            }
        out[approach] = agg
    return out


def significance_tests(per_seed: dict, metric: str = "pr_auc",
                       alpha: float = 0.05) -> list[dict]:
    """Wilcoxonov test uparenih rangova između svih parova pristupa.

    Uparen je zato što svi pristupi vide *isti* generisani skup po seedu, pa se
    varijansa zbog seeda poništava. Neparametarski je jer se raspodjela razlika
    PR-AUC vrijednosti ne može smatrati normalnom pri malom broju seedova.
    """
    names = [n for n in APPROACHES if n in per_seed]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va = np.array(per_seed[a][metric], dtype=float)
            vb = np.array(per_seed[b][metric], dtype=float)
            diff = va - vb
            entry = {"a": a, "b": b, "metric": metric,
                     "mean_diff": float(diff.mean()), "n_seeds": int(va.size)}
            if va.size < 3 or np.allclose(diff, 0):
                entry.update(p_value=None, significant=False,
                             note="premalo seedova ili identični rezultati")
            else:
                stat, p = wilcoxon(va, vb)
                entry.update(statistic=float(stat), p_value=float(p),
                             significant=bool(p < alpha))
            out.append(entry)
    return out


def run_multiseed(seeds, n_train: int, n_test: int, days: int,
                  fraud_rate: float, n_users: int = 25) -> tuple[dict, dict]:
    """Evaluacija po seedovima; vraća (per_seed_metrike, detalji_prvog_seeda)."""
    per_seed: dict = {a: {m: [] for m in METRIC_NAMES} for a in APPROACHES}
    first_detail: dict = {}
    for seed in seeds:
        train_events = generate_batch(n_users=n_users, n_normal=n_train,
                                      fraud_rate=fraud_rate, seed=seed,
                                      start=TRAIN_START, days=days)
        test_events = generate_batch(n_users=n_users, n_normal=n_test,
                                     fraud_rate=fraud_rate, seed=seed + 10_000,
                                     start=TEST_START, days=days)
        Xtr, ytr = build_dataset(train_events)
        Xte, yte = build_dataset(test_events)
        model = RiskModel(version=f"eval-s{seed}", random_state=0)
        model.train(Xtr, ytr)
        fusion_models = fit_fusion_models(model, BANKING_RULES, BANKING_FUSION,
                                          Xtr, ytr)
        detail = evaluate_approaches(model, BANKING_RULES, BANKING_FUSION,
                                     Xte, yte, fusion_models)
        for approach, res in detail.items():
            for metric in METRIC_NAMES:
                per_seed[approach][metric].append(res["metrics"][metric])
        if not first_detail:
            first_detail = {
                "seed": seed, "detail": detail, "model": model,
                "fusion_models": fusion_models,
                "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
                "test_events": test_events,
            }
    return per_seed, first_detail


def run_evaluation(train_seed: int = 1, test_seed: int = 2024,
                   n_train: int = 4000, n_test: int = 3000, days: int = 30,
                   fraud_rate: float = 0.02, n_seeds: int = 1,
                   model=None, rules=None, fusion_cfg=None, policy=None,
                   guardrails=None, fusion=None, full: bool = False) -> dict:
    """Evaluacioni izvještaj.

    Sa `model` (npr. živi engine iz API-ja) mjeri se taj model na jednom
    skupu. Bez njega se trenira svjež model po seedu; `n_seeds > 1` uključuje
    agregaciju i test značajnosti, a `full=True` i sve eksperimente iz
    `spr.evaluation.experiments` (traje znatno duže).
    """
    rules = rules or BANKING_RULES
    fusion_cfg = fusion_cfg or BANKING_FUSION
    policy = policy or BANKING_POLICY
    guardrails = guardrails or BANKING_GUARDRAILS

    if model is not None:
        # evaluacija živog modela iz enginea — jedan skup, bez retreninga
        test_events = generate_batch(n_users=25, n_normal=n_test,
                                     fraud_rate=fraud_rate, seed=test_seed,
                                     start=TEST_START, days=days)
        Xte, yte = build_dataset(test_events)
        fusion_models = ({name: fusion for name in HYBRID_STRATEGIES}
                         if fusion is not None and fusion.fitted else None)
        approaches = evaluate_approaches(model, rules, fusion_cfg, Xte, yte,
                                         fusion_models)
        report = {
            "n_test": len(yte), "n_fraud": int(sum(yte)),
            "fraud_rate": fraud_rate, "seeds": [test_seed],
            "approaches": approaches,
            "latency": measure_latency(model, rules, fusion_cfg, policy,
                                       guardrails, test_events, fusion=fusion),
        }
        scores = approach_scores(model, rules, fusion_cfg, Xte,
                                 fusion_models)["hybrid_stacking"]
        report["cost"] = _cost_section(yte, scores, policy)
        return report

    seeds = [train_seed + i for i in range(max(1, n_seeds))]
    per_seed, first = run_multiseed(seeds, n_train, n_test, days, fraud_rate)

    report = {
        "config": {"n_train": n_train, "n_test": n_test, "days": days,
                   "fraud_rate": fraud_rate, "n_seeds": len(seeds),
                   "seeds": seeds},
        "n_test": len(first["yte"]), "n_fraud": int(sum(first["yte"])),
        "fraud_rate": fraud_rate, "seeds": seeds,
        "approaches": first["detail"],
        "aggregated": _aggregate(per_seed),
        "significance": significance_tests(per_seed) if len(seeds) > 1 else [],
        "latency": measure_latency(first["model"], rules, fusion_cfg, policy,
                                   guardrails, first["test_events"],
                                   fusion=first["fusion_models"]["hybrid_stacking"]),
    }

    scores = approach_scores(first["model"], rules, fusion_cfg, first["Xte"],
                             first["fusion_models"])["hybrid_stacking"]
    report["cost"] = _cost_section(first["yte"], scores, policy)
    report["drift"] = drift_report(first["Xtr"], first["Xte"])

    if full:
        kwargs = dict(seed=seeds[0], n_train=n_train, n_test=n_test, days=days,
                      fraud_rate=fraud_rate)
        report["model_zoo"] = model_zoo(**kwargs)
        report["rule_ablation"] = rule_ablation(**kwargs)
        report["feature_ablation"] = feature_ablation(**kwargs)
        report["novel_attack"] = novel_attack(**kwargs)
        report["label_scarcity"] = label_scarcity(**kwargs)
        report["feedback_value"] = feedback_value(**kwargs)
        report["drift_scenarios"] = drift_scenarios(**kwargs)
        report["meta_weights"] = first["fusion_models"]["hybrid_stacking"].weights()
    return report


def _cost_section(y, scores, policy) -> dict:
    base = cost_report(y, scores, policy, BANKING_COST)
    constrained = constrained_thresholds(y, scores, BANKING_COST, BANKING_CAPACITY)
    ch, rv = action_rates(scores, policy.low_max, policy.medium_max)
    n = max(len(list(y)), 1)
    base["capacity"] = BANKING_CAPACITY.to_dict()
    base["constrained"] = {
        **constrained, "cost_per_event": constrained["total_cost"] / n}
    base["current"]["challenge_rate"] = ch
    base["current"]["review_rate"] = rv
    base["allow_all_baseline"] = {
        "total_cost": total_cost(y, scores, 1.0, 1.0, BANKING_COST),
        "note": "ništa se ne provjerava — referentna gornja granica gubitka",
    }
    return base


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

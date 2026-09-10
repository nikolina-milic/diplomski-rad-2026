"""Eksperimenti koji odgovaraju na pitanja rada.

Svaki od njih postoji zato što nešto *tvrdimo*, a tvrdnja bez mjerenja nije
rezultat:

  model_zoo         — zašto baš RandomForest? (poređenje sa linearnim,
                      boosting i nenadziranim baselineom)
  rule_ablation     — koje pravilo stvarno nosi odluku
  feature_ablation  — koje obilježje stvarno nosi odluku
  novel_attack      — šta se dešava sa obrascem napada kojeg nema u trening
                      podacima (ovdje se vidi prava vrijednost pravila)
  label_scarcity    — koliko labela treba da ML prestigne pravila
  feedback_value    — koliko doprinosi analitičarski feedback u odnosu na
                      ground truth
"""

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from spr.domain.banking_pack import BANKING_FUSION, BANKING_RULES
from spr.fusion.model import FusionModel
from spr.fusion.strategies import FusionConfig
from spr.generator.generator import generate_batch
from spr.generator.patterns import FRAUD_PATTERNS
from spr.learning.metrics import compute_metrics
from spr.ml.dataset import build_dataset
from spr.ml.model import SUPERVISED_ALGORITHMS, RiskModel
from spr.rules.engine import evaluate_rules

TRAIN_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
TEST_START = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _rules_scores(X, rules) -> list[float]:
    return [evaluate_rules(x, rules).score for x in X]


def _fit_stack(model: RiskModel, rules, X, y, strategy: str = "stacking"
               ) -> FusionModel:
    cfg = FusionConfig(strategy=strategy,
                       rules_weight=BANKING_FUSION.rules_weight,
                       ml_weight=BANKING_FUSION.ml_weight,
                       cascade_threshold=BANKING_FUSION.cascade_threshold)
    return FusionModel(cfg).fit(_rules_scores(X, rules), model.oof_scores(X, y), y)


def make_split(seed: int, n_train: int, n_test: int, days: int,
               fraud_rate: float, n_users: int = 25,
               train_exclude: list[str] | None = None,
               test_patterns: list[str] | None = None):
    """Vremenski razdvojen trening/test par (test je uvijek kasniji period)."""
    train_events = generate_batch(n_users=n_users, n_normal=n_train,
                                  fraud_rate=fraud_rate, seed=seed,
                                  start=TRAIN_START, days=days,
                                  exclude_patterns=train_exclude)
    test_events = generate_batch(n_users=n_users, n_normal=n_test,
                                 fraud_rate=fraud_rate, seed=seed + 10_000,
                                 start=TEST_START, days=days,
                                 patterns=test_patterns)
    return build_dataset(train_events) + build_dataset(test_events)


# --- zašto ovaj algoritam ------------------------------------------------
def model_zoo(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
              days: int = 30, fraud_rate: float = 0.02) -> dict:
    """Poređenje algoritama na istom skupu — opravdanje izbora modela."""
    Xtr, ytr, Xte, yte = make_split(seed, n_train, n_test, days, fraud_rate)
    out = {}
    for algo in SUPERVISED_ALGORITHMS + ("isolation_forest",):
        model = RiskModel(algorithm=algo, random_state=0)
        model.train(Xtr, ytr)
        scores = model.predict_proba_batch(Xte)
        out[algo] = {
            "metrics": compute_metrics(yte, scores),
            "supervised": algo != "isolation_forest",
        }
    return out


# --- šta zaista nosi odluku ---------------------------------------------
def rule_ablation(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                  days: int = 30, fraud_rate: float = 0.02) -> dict:
    """Izbaci jedno po jedno pravilo i izmjeri pad PR-AUC skora pravila."""
    Xtr, ytr, Xte, yte = make_split(seed, n_train, n_test, days, fraud_rate)
    full = compute_metrics(yte, _rules_scores(Xte, BANKING_RULES))["pr_auc"]
    rows = []
    for rule in BANKING_RULES:
        subset = [r for r in BANKING_RULES if r.name != rule.name]
        pr = compute_metrics(yte, _rules_scores(Xte, subset))["pr_auc"]
        rows.append({"rule": rule.name, "weight": rule.weight,
                     "pr_auc_without": pr, "delta": full - pr})
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return {"pr_auc_full": full, "rules": rows}


def feature_ablation(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                     days: int = 30, fraud_rate: float = 0.02) -> dict:
    """Zamijeni jedno obilježje njegovom srednjom vrijednošću i izmjeri pad.

    Neutralisanje srednjom vrijednošću (umjesto ponovnog treniranja bez
    obilježja) izoluje doprinos *tog* obilježja pri fiksnom modelu.
    """
    Xtr, ytr, Xte, yte = make_split(seed, n_train, n_test, days, fraud_rate)
    model = RiskModel(random_state=0)
    model.train(Xtr, ytr)
    base = compute_metrics(yte, model.predict_proba_batch(Xte))["pr_auc"]
    names = list(Xte[0].keys()) if Xte else []
    rows = []
    for name in names:
        mean_val = float(np.mean([x[name] for x in Xtr]))
        muted = [{**x, name: mean_val} for x in Xte]
        pr = compute_metrics(yte, model.predict_proba_batch(muted))["pr_auc"]
        rows.append({"feature": name, "pr_auc_muted": pr, "delta": base - pr})
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return {"pr_auc_full": base, "features": rows}


# --- nov napad -----------------------------------------------------------
def novel_attack(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                 days: int = 30, fraud_rate: float = 0.02) -> dict:
    """Leave-one-pattern-out: model ne vidi obrazac koji će biti testiran.

    Ovo je scenario u kojem hibrid ima smisla: ML generalizuje na ono što je
    vidio, a pravilo napisano za poznat rizik radi i za napad koji se prvi put
    pojavljuje.
    """
    rows = {}
    for held in FRAUD_PATTERNS:
        Xtr, ytr, Xte, yte = make_split(
            seed, n_train, n_test, days, fraud_rate,
            train_exclude=[held], test_patterns=[held])
        if sum(yte) < 3 or len(set(ytr)) < 2:
            continue
        model = RiskModel(random_state=0)
        model.train(Xtr, ytr)
        ml = model.predict_proba_batch(Xte)
        rules = _rules_scores(Xte, BANKING_RULES)
        entry = {
            "n_test_fraud": int(sum(yte)),
            "rules": compute_metrics(yte, rules)["pr_auc"],
            "ml": compute_metrics(yte, ml)["pr_auc"],
        }
        for strategy in ("stacking", "cascade"):
            fusion = _fit_stack(model, BANKING_RULES, Xtr, ytr, strategy)
            entry[f"hybrid_{strategy}"] = compute_metrics(
                yte, fusion.combine_batch(rules, ml))["pr_auc"]
        rows[held] = entry
    return rows


# --- koliko labela treba -------------------------------------------------
def label_scarcity(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                   days: int = 30, fraud_rate: float = 0.02,
                   fractions=(0.02, 0.05, 0.10, 0.25, 0.50, 1.0)) -> dict:
    """Kako pristupi degradiraju kad ima malo labeliranih podataka.

    Pravila ne troše nijednu labelu, pa im rezultat ne zavisi od `frac` —
    to je referentna linija ispod koje ML ne smije pasti da bi se isplatio.
    """
    Xtr, ytr, Xte, yte = make_split(seed, n_train, n_test, days, fraud_rate)
    rules_te = _rules_scores(Xte, BANKING_RULES)
    rules_pr = compute_metrics(yte, rules_te)["pr_auc"]
    rows = []
    for frac in fractions:
        k = max(50, int(len(Xtr) * frac))
        Xs, ys = Xtr[:k], ytr[:k]
        if sum(ys) < 3:
            rows.append({"fraction": frac, "n_labels": k, "n_fraud": int(sum(ys)),
                         "ml": None, "hybrid_stacking": None, "rules": rules_pr})
            continue
        model = RiskModel(random_state=0)
        model.train(Xs, ys)
        ml = model.predict_proba_batch(Xte)
        fusion = _fit_stack(model, BANKING_RULES, Xs, ys)
        rows.append({
            "fraction": frac, "n_labels": k, "n_fraud": int(sum(ys)),
            "ml": compute_metrics(yte, ml)["pr_auc"],
            "hybrid_stacking": compute_metrics(
                yte, fusion.combine_batch(rules_te, ml))["pr_auc"],
            "rules": rules_pr,
        })
    return {"rules_pr_auc": rules_pr, "levels": rows}


# --- vrijednost analitičarskog feedbacka ---------------------------------
def feedback_value(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                   days: int = 30, fraud_rate: float = 0.02,
                   review_fraction: float = 0.05,
                   analyst_error_rate: float = 0.05) -> dict:
    """Koliko vrijedi feedback analitičara u odnosu na potpuni ground truth.

    Realan sistem ne dobija labelu za svaki događaj — analitičar pregleda samo
    ono što je označeno za REVIEW, i pri tom griješi. Ovdje se porede tri
    izvora labela pri istom broju trening primjera.
    """
    Xtr, ytr, Xte, yte = make_split(seed, n_train, n_test, days, fraud_rate)
    rng = np.random.default_rng(seed)

    # 1) potpun ground truth (gornja granica, u praksi nedostupna)
    full = RiskModel(random_state=0)
    full.train(Xtr, ytr)

    # 2) samo najrizičniji dio, labeliran od analitičara koji ponekad griješi
    screener = RiskModel(random_state=0)
    screener.train(Xtr[:len(Xtr) // 2], ytr[:len(ytr) // 2])
    scores = np.array(screener.predict_proba_batch(Xtr))
    k = max(50, int(len(Xtr) * review_fraction))
    reviewed = np.argsort(-scores)[:k]
    y_analyst = []
    for i in reviewed:
        label = ytr[i]
        if rng.random() < analyst_error_rate:
            label = 1 - label
        y_analyst.append(int(label))
    out = {"n_reviewed": int(k), "analyst_error_rate": analyst_error_rate,
           "review_fraction": review_fraction}
    if len(set(y_analyst)) > 1:
        analyst = RiskModel(random_state=0)
        analyst.train([Xtr[i] for i in reviewed], y_analyst)
        out["analyst_feedback"] = compute_metrics(
            yte, analyst.predict_proba_batch(Xte))["pr_auc"]
    else:
        out["analyst_feedback"] = None

    # 3) isti broj primjera, ali biranih nasumično — da se vidi da li dobitak
    #    dolazi od feedbacka ili prosto od količine podataka
    idx = rng.choice(len(Xtr), size=k, replace=False)
    y_rand = [ytr[i] for i in idx]
    if len(set(y_rand)) > 1:
        rand = RiskModel(random_state=0)
        rand.train([Xtr[i] for i in idx], y_rand)
        out["random_sample"] = compute_metrics(
            yte, rand.predict_proba_batch(Xte))["pr_auc"]
    else:
        out["random_sample"] = None

    out["full_ground_truth"] = compute_metrics(
        yte, full.predict_proba_batch(Xte))["pr_auc"]
    return out


# --- provjera da detektor drifta zaista reaguje --------------------------
def drift_scenarios(seed: int = 1, n_train: int = 4000, n_test: int = 3000,
                    days: int = 30, fraud_rate: float = 0.02) -> dict:
    """Detektor drifta na kontrolisanim scenarijima.

    „Stabilno" na podacima iz iste raspodjele ništa ne dokazuje — detektor koji
    uvijek ćuti izgleda isto tako. Ovdje se mjeri i scenario sa *poznatim*
    pomjeranjem, pa se vidi da PSI stvarno reaguje.
    """
    from spr.learning.drift import drift_report as _drift

    reference = generate_batch(n_users=25, n_normal=n_train,
                               fraud_rate=fraud_rate, seed=seed,
                               start=TRAIN_START, days=days)
    Xref, _ = build_dataset(reference)

    scenarios = {}

    # 1) ista raspodjela, drugi seed — detektor mora ćutati
    same = generate_batch(n_users=25, n_normal=n_test, fraud_rate=fraud_rate,
                          seed=seed + 777, start=TEST_START, days=days)
    Xsame, _ = build_dataset(same)
    scenarios["ista_raspodjela"] = _drift(Xref, Xsame)

    # 2) promijenjena struktura napada — samo jedan obrazac umjesto pet
    shifted = generate_batch(n_users=25, n_normal=n_test, fraud_rate=fraud_rate * 4,
                             seed=seed + 778, start=TEST_START, days=days,
                             patterns=["card_testing"])
    Xshift, _ = build_dataset(shifted)
    scenarios["promijenjen_napad"] = _drift(Xref, Xshift)

    # 3) druga populacija korisnika (drugi broj i profil persona)
    other = generate_batch(n_users=6, n_normal=n_test, fraud_rate=fraud_rate,
                           seed=seed + 779, start=TEST_START, days=days // 3)
    Xother, _ = build_dataset(other)
    scenarios["druga_populacija"] = _drift(Xref, Xother)

    return {name: {"overall": d["overall"], "max_psi": d["max_psi"],
                   "worst_feature": d["worst_feature"],
                   "n_drifting": d["n_drifting"],
                   "top": [{"feature": f["feature"], "psi": f["psi"],
                            "severity": f["severity"]} for f in d["features"][:5]]}
            for name, d in scenarios.items()}

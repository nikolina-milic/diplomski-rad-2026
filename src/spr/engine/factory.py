from copy import deepcopy
from datetime import datetime, timezone

from spr.context.store import ContextStore
from spr.domain.banking_pack import (
    BANKING_CAPACITY, BANKING_COST, BANKING_FUSION, BANKING_GUARDRAILS,
    BANKING_POLICY, BANKING_RULES,
)
from spr.engine.engine import DecisionEngine
from spr.fusion.model import FusionModel
from spr.generator.generator import generate_batch
from spr.ml.dataset import build_dataset
from spr.ml.model import RiskModel
from spr.policy.cost import apply_constrained_to_policy
from spr.rules.engine import evaluate_rules

DEFAULT_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fit_fusion(model: RiskModel, rules, fusion_config, X, y,
               calibration: str = "isotonic") -> tuple[FusionModel, list[float]]:
    """Kalibriši oba skora i nauči meta-model nad out-of-fold predikcijama.

    Skorovi baznog modela na sopstvenom trening skupu su preoptimistični, pa
    bi kalibracija na njima naučila pogrešnu vezu skor → vjerovatnoća.
    """
    rules_scores = [evaluate_rules(x, rules).score for x in X]
    ml_scores = model.oof_scores(X, y)
    fusion = FusionModel(fusion_config, calibration=calibration).fit(
        rules_scores, ml_scores, y)
    # kombinovani out-of-fold skorovi — podloga za biranje pragova bez curenja
    return fusion, fusion.combine_batch(rules_scores, ml_scores)


def build_default_engine(seed: int = 1, n_users: int = 25, n_normal: int = 4000,
                         fraud_rate: float = 0.02, days: int = 30,
                         algorithm: str = "random_forest",
                         calibration: str = "isotonic",
                         cost_based_thresholds: bool = True) -> DecisionEngine:
    """Istrenirani engine spreman za serviranje.

    Redoslijed je bitan: model → out-of-fold skorovi → kalibracija i fusion →
    pragovi iz matrice troška (koji imaju smisla tek nad kalibrisanim skorom).
    """
    events = generate_batch(n_users=n_users, n_normal=n_normal,
                            fraud_rate=fraud_rate, seed=seed,
                            start=DEFAULT_START, days=days)
    X, y = build_dataset(events)
    model = RiskModel(version="v1", random_state=0, algorithm=algorithm)
    model.train(X, y)

    rules = deepcopy(BANKING_RULES)
    policy = deepcopy(BANKING_POLICY)
    fusion_config = deepcopy(BANKING_FUSION)
    guardrails = deepcopy(BANKING_GUARDRAILS)

    fusion, oof_final = fit_fusion(model, rules, fusion_config, X, y, calibration)
    if cost_based_thresholds:
        apply_constrained_to_policy(policy, y, oof_final,
                                    deepcopy(BANKING_COST),
                                    deepcopy(BANKING_CAPACITY))

    # svjež store za serviranje (cold start korisnika); deepcopy konfiguracije
    # da izmjene uživo ne mutiraju globalni banking pack ni druge engine-e
    return DecisionEngine(
        model=model, rules=rules, policy=policy,
        fusion_config=fusion_config, guardrails=guardrails,
        store=ContextStore(), fusion=fusion,
        cost=deepcopy(BANKING_COST), capacity=deepcopy(BANKING_CAPACITY),
    )

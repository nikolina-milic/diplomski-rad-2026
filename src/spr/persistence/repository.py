from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spr.engine.decision import Decision
from spr.persistence.models import (
    DecisionRecord, FeedbackRecord, ModelVersionRecord, RuleConfigRecord,
)


def save_decision(session: Session, decision: Decision, event_time: datetime,
                  truth: int | None) -> DecisionRecord:
    rec = DecisionRecord(
        event_id=decision.event_id, user_id=decision.user_id,
        first_name=decision.event.first_name if decision.event else None,
        last_name=decision.event.last_name if decision.event else None,
        event_time=event_time,
        final_score=decision.final_score, level=decision.level, action=decision.action,
        executed_action=decision.executed_action, mode=decision.mode, rules_score=decision.rules_score, ml_score=decision.ml_score,
        model_version=decision.model_version, guardrail=decision.guardrail,
        explanation=decision.explanation,
        fired_rules=[fr.model_dump() for fr in decision.fired_rules],
        shap_top=[sc.model_dump() for sc in decision.shap_top],
        features=decision.features,
        event_info=decision.event.model_dump(mode="json") if decision.event else {},
        truth=truth,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def list_decisions(session: Session, limit: int = 50,
                   level: str | None = None) -> list[DecisionRecord]:
    stmt = select(DecisionRecord).order_by(DecisionRecord.id.desc()).limit(limit)
    if level:
        stmt = stmt.where(DecisionRecord.level == level)
    return list(session.scalars(stmt))


def count_decisions(session: Session) -> dict:
    """Ukupan broj odluka, po nivou rizika i po akciji.

    Radi se SQL prebrojavanjem, a ne učitavanjem redova. Brojači i zaglavlja
    kolona moraju dolaziti iz ISTOG izvora — `/stats` broji samo posljednjih
    N odluka, pa je uz ukupne brojeve po nivou davao brojke koje se ne slažu.
    """
    total = session.scalar(select(func.count()).select_from(DecisionRecord)) or 0
    levels = session.execute(
        select(DecisionRecord.level, func.count())
        .group_by(DecisionRecord.level)
    ).all()
    actions = session.execute(
        select(DecisionRecord.action, func.count())
        .group_by(DecisionRecord.action)
    ).all()
    return {"total": int(total),
            "by_level": {level: int(n) for level, n in levels},
            "by_action": {action: int(n) for action, n in actions}}


def get_review_queue(session: Session) -> list[DecisionRecord]:
    stmt = select(DecisionRecord).where(DecisionRecord.action == "REVIEW")
    return [d for d in session.scalars(stmt) if not d.feedback]


def add_feedback(session: Session, decision_id: int, label: int,
                 source: str) -> FeedbackRecord:
    fb = FeedbackRecord(decision_id=decision_id, label=label, source=source)
    session.add(fb)
    session.commit()
    session.refresh(fb)
    return fb


def collect_training_data(session: Session) -> tuple[list[dict], list[int]]:
    """Feature-i + label: analitičarski feedback ako postoji, inače ground truth."""
    X: list[dict] = []
    y: list[int] = []
    for d in session.scalars(select(DecisionRecord)):
        if d.feedback:
            label = d.feedback[-1].label
        elif d.truth is not None:
            label = d.truth
        else:
            continue
        X.append(d.features)
        y.append(int(label))
    return X, y


def upsert_rule_configs(session: Session, rules) -> None:
    """Snimi trenutno stanje pravila (weight, enabled, condition) po imenu."""
    existing = {r.name: r for r in session.scalars(select(RuleConfigRecord))}
    for rule in rules:
        rec = existing.get(rule.name)
        if rec is None:
            session.add(RuleConfigRecord(
                name=rule.name, weight=rule.weight,
                enabled=getattr(rule, "enabled", True), condition=rule.condition))
        else:
            rec.weight = rule.weight
            rec.enabled = getattr(rule, "enabled", True)
            rec.condition = rule.condition
    session.commit()


def load_rule_configs(session: Session) -> dict[str, dict]:
    """Učitaj sačuvane override-e pravila: {name: {weight, enabled, condition}}."""
    return {
        r.name: {"weight": r.weight, "enabled": r.enabled, "condition": r.condition}
        for r in session.scalars(select(RuleConfigRecord))
    }


def save_model_version(session: Session, version: str, status: str,
                       metrics: dict, path: str | None = None) -> ModelVersionRecord:
    rec = ModelVersionRecord(version=version, status=status, metrics=metrics, path=path)
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def get_champion(session: Session) -> ModelVersionRecord | None:
    stmt = select(ModelVersionRecord).where(ModelVersionRecord.status == "champion")
    return session.scalars(stmt).first()


def list_model_versions(session: Session) -> list[ModelVersionRecord]:
    return list(session.scalars(select(ModelVersionRecord).order_by(ModelVersionRecord.id)))


def retire_champions(session: Session) -> None:
    for rec in session.scalars(
        select(ModelVersionRecord).where(ModelVersionRecord.status == "champion")
    ):
        rec.status = "retired"
    session.commit()


def next_model_version(session: Session) -> str:
    """Sljedeća slobodna oznaka verzije (v1, v2, …).

    Verziju određuje backend, a ne klijent: frontend je ranije slao
    `v{broj_verzija + 1}`, pa je na praznom registryju tražio „v1" i pregazio
    artefakt početnog modela.
    """
    import re
    numbers = []
    for rec in session.scalars(select(ModelVersionRecord)):
        m = re.fullmatch(r"v(\d+)", rec.version or "")
        if m:
            numbers.append(int(m.group(1)))
    return f"v{max(numbers) + 1 if numbers else 1}"


def version_exists(session: Session, version: str) -> bool:
    stmt = select(ModelVersionRecord).where(ModelVersionRecord.version == version)
    return session.scalars(stmt).first() is not None

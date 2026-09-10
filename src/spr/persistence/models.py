from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from spr.persistence.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionRecord(Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String, index=True)
    first_name: Mapped[str | None] = mapped_column(String, nullable=True)
    last_name: Mapped[str | None] = mapped_column(String, nullable=True)
    event_time: Mapped[datetime] = mapped_column(DateTime)
    final_score: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String, index=True)
    # stvarno izvršena akcija: u shadow režimu uvijek ALLOW, u enforce = action
    executed_action: Mapped[str | None] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String)
    rules_score: Mapped[float] = mapped_column(Float)
    ml_score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String)
    guardrail: Mapped[str | None] = mapped_column(String, nullable=True)
    explanation: Mapped[str] = mapped_column(String)
    fired_rules: Mapped[list] = mapped_column(JSON, default=list)
    shap_top: Mapped[list] = mapped_column(JSON, default=list)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    event_info: Mapped[dict] = mapped_column(JSON, default=dict)
    truth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    feedback: Mapped[list["FeedbackRecord"]] = relationship(
        back_populates="decision", cascade="all, delete-orphan")


class FeedbackRecord(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id"))
    label: Mapped[int] = mapped_column(Integer)      # 1 = prevara, 0 = legitimno
    source: Mapped[str] = mapped_column(String)       # "analyst" | "ground_truth"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    decision: Mapped["DecisionRecord"] = relationship(back_populates="feedback")


class RuleConfigRecord(Base):
    """Trajni override poslovnog pravila (preživljava restart)."""
    __tablename__ = "rule_configs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    weight: Mapped[float] = mapped_column(Float)
    enabled: Mapped[bool] = mapped_column(Boolean)
    condition: Mapped[str] = mapped_column(String)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String, unique=True)
    status: Mapped[str] = mapped_column(String)       # champion | challenger | retired
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

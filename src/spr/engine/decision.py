from datetime import datetime

from pydantic import BaseModel


class EventInfo(BaseModel):
    """Poslovni detalji transakcije uz odluku (za prikaz i audit)."""
    first_name: str = ""
    last_name: str = ""
    amount: float
    currency: str
    merchant_category: str
    country: str
    city: str | None
    latitude: float
    longitude: float
    device_id: str
    channel: str
    timestamp: datetime
    home_country: str | None
    distance_from_home_km: float


class ShapContribution(BaseModel):
    feature: str
    value: float


class FiredRuleOut(BaseModel):
    name: str
    weight: float
    explanation: str


class Decision(BaseModel):
    event_id: str
    user_id: str
    final_score: float
    level: str
    action: str            # predložena akcija po politici/guardrail-u
    executed_action: str = "ALLOW"   # stvarno izvršena (u shadow režimu: ALLOW)
    mode: str
    rules_score: float
    ml_score: float
    fired_rules: list[FiredRuleOut]
    shap_top: list[ShapContribution]
    model_version: str
    guardrail: str | None
    explanation: str
    features: dict[str, float] = {}
    event: EventInfo | None = None

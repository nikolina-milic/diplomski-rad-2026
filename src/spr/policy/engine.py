from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Action(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class PolicyConfig:
    low_max: float
    medium_max: float
    level_actions: dict
    mode: str = "shadow"  # "shadow" (samo loguj) | "enforce" (izvrši)


def classify(score: float, policy: PolicyConfig) -> RiskLevel:
    if score <= policy.low_max:
        return RiskLevel.LOW
    if score <= policy.medium_max:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def decide_action(level: RiskLevel, policy: PolicyConfig) -> Action:
    return policy.level_actions[level]


def execute(action: Action, policy: PolicyConfig) -> Action:
    """Efektivna akcija nakon primjene režima.

    `shadow`  — sistem samo posmatra: predložena akcija se loguje, ali se ne
                izvršava, pa je efektivno svaka transakcija propuštena.
    `enforce` — predložena akcija se stvarno izvršava.

    Razlika je vidljiva u `Decision.executed_action` i u audit logu, pa se
    shadow period može mjeriti („šta bi sistem uradio da je bio uključen").
    """
    if policy.mode == "enforce":
        return action
    return Action.ALLOW

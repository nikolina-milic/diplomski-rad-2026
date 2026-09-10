from dataclasses import dataclass

from spr.policy.engine import Action


@dataclass
class Guardrail:
    name: str
    condition: str          # izraz nad imenima feature-a
    force_action: Action    # akcija koja nadjačava policy
    explanation: str


@dataclass
class GuardrailHit:
    name: str
    force_action: Action
    explanation: str


def apply_guardrails(features: dict, guardrails: list) -> GuardrailHit | None:
    """Vrati prvi okinuti guardrail (nadjačava odluku), ili None."""
    for g in guardrails:
        try:
            ok = bool(eval(g.condition, {"__builtins__": {}}, features))
        except Exception:
            ok = False
        if ok:
            return GuardrailHit(g.name, g.force_action, g.explanation)
    return None

from spr.fusion.guardrails import Guardrail, apply_guardrails
from spr.policy.engine import Action
from spr.domain.banking_pack import BANKING_GUARDRAILS


def test_compliance_limit_forces_block():
    hit = apply_guardrails({"amount": 20000.0, "is_new_device": 0.0}, BANKING_GUARDRAILS)
    assert hit is not None
    assert hit.force_action == Action.BLOCK


def test_no_match_returns_none():
    hit = apply_guardrails({"amount": 50.0, "is_new_device": 0.0}, BANKING_GUARDRAILS)
    assert hit is None


def test_bad_condition_does_not_crash():
    g = [Guardrail("bad", "nema_takav_feature > 1", Action.BLOCK, "x")]
    assert apply_guardrails({"amount": 1.0}, g) is None


def test_first_match_wins():
    g = [
        Guardrail("a", "amount > 10", Action.BLOCK, "a"),
        Guardrail("b", "amount > 10", Action.ALLOW, "b"),
    ]
    hit = apply_guardrails({"amount": 100.0}, g)
    assert hit.name == "a"

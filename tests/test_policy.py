from spr.policy.engine import RiskLevel, Action, PolicyConfig, classify, decide_action

P = PolicyConfig(low_max=0.3, medium_max=0.7, level_actions={
    RiskLevel.LOW: Action.ALLOW, RiskLevel.MEDIUM: Action.CHALLENGE, RiskLevel.HIGH: Action.REVIEW})


def test_classify_boundaries():
    assert classify(0.0, P) == RiskLevel.LOW
    assert classify(0.3, P) == RiskLevel.LOW
    assert classify(0.31, P) == RiskLevel.MEDIUM
    assert classify(0.7, P) == RiskLevel.MEDIUM
    assert classify(0.71, P) == RiskLevel.HIGH
    assert classify(1.0, P) == RiskLevel.HIGH


def test_decide_action_mapping():
    assert decide_action(RiskLevel.LOW, P) == Action.ALLOW
    assert decide_action(RiskLevel.MEDIUM, P) == Action.CHALLENGE
    assert decide_action(RiskLevel.HIGH, P) == Action.REVIEW

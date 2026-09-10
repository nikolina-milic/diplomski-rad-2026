from spr.rules.engine import (
    Rule, evaluate_rules, split_condition, apply_thresholds, threshold_features,
)
from spr.domain.banking_pack import BANKING_RULES


def test_fires_and_explains():
    rules = [Rule("big", "amount_zscore > 3", 0.6, "Veliko odstupanje iznosa")]
    res = evaluate_rules({"amount_zscore": 5.0}, rules)
    assert len(res.fired) == 1
    assert res.fired[0].name == "big"
    assert res.fired[0].explanation == "Veliko odstupanje iznosa"
    assert 0.0 < res.score <= 1.0


def test_no_fire_gives_zero():
    rules = [Rule("big", "amount_zscore > 3", 0.6, "x")]
    res = evaluate_rules({"amount_zscore": 0.0}, rules)
    assert res.fired == []
    assert res.score == 0.0


def test_noisy_or_bounded_and_monotonic():
    r1 = [Rule("a", "x > 0", 0.6, "a")]
    r2 = [Rule("a", "x > 0", 0.6, "a"), Rule("b", "x > 0", 0.5, "b")]
    s1 = evaluate_rules({"x": 1.0}, r1).score
    s2 = evaluate_rules({"x": 1.0}, r2).score
    assert 0.0 <= s1 <= s2 <= 1.0


def test_bad_condition_does_not_crash():
    rules = [Rule("bad", "nonexistent_feature > 1", 0.5, "x")]
    res = evaluate_rules({"amount": 1.0}, rules)
    assert res.fired == []  # greška u uslovu => pravilo se ne okida


def test_split_condition_single_threshold():
    parts, thr = split_condition("amount_ratio > 8")
    assert thr == [8]
    assert len(parts) == len(thr) + 1
    assert parts[0] + "8" + parts[1] == "amount_ratio > 8"


def test_split_condition_ignores_digits_in_feature_names():
    # txn_count_1h / txn_count_24h: cifre su dio imena, ne pragovi
    parts, thr = split_condition("txn_count_1h >= 5")
    assert thr == [5]
    _, thr2 = split_condition("txn_count_24h >= 10")
    assert thr2 == [10]


def test_split_condition_multiple_thresholds():
    cond = "is_unusual_hour >= 1 and amount_ratio > 3"
    parts, thr = split_condition(cond)
    assert thr == [1, 3]
    assert len(parts) == 3


def test_apply_thresholds_roundtrip_and_change():
    cond = "txn_count_1h >= 5"
    _, thr = split_condition(cond)
    assert apply_thresholds(cond, thr) == cond          # round-trip
    assert apply_thresholds(cond, [3]) == "txn_count_1h >= 3"


def test_apply_thresholds_multi_and_integer_format():
    cond = "is_unusual_hour >= 1 and amount_ratio > 3"
    assert apply_thresholds(cond, [1, 5]) == \
        "is_unusual_hour >= 1 and amount_ratio > 5"
    # cjelobrojna vrijednost se renderuje bez '.0'
    assert apply_thresholds("amount_ratio > 8", [10.0]) == "amount_ratio > 10"


def test_apply_thresholds_wrong_count_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_thresholds("amount_ratio > 8", [1, 2])


def test_all_banking_rules_split_roundtrip():
    for r in BANKING_RULES:
        parts, thr = split_condition(r.condition)
        assert len(parts) == len(thr) + 1
        assert apply_thresholds(r.condition, thr) == r.condition


def test_threshold_features_alignment():
    assert threshold_features("amount_ratio > 8") == ["amount_ratio"]
    assert threshold_features("txn_count_1h >= 5") == ["txn_count_1h"]
    assert threshold_features("is_unusual_hour >= 1 and amount_ratio > 3") == \
        ["is_unusual_hour", "amount_ratio"]
    # poravnanje sa split_condition za sva pravila
    for r in BANKING_RULES:
        _, thr = split_condition(r.condition)
        assert len(threshold_features(r.condition)) == len(thr)


def test_banking_rules_impossible_travel():
    feats = {n: 0.0 for n in [
        "amount", "amount_zscore", "amount_ratio", "txn_count_1h", "txn_count_24h",
        "is_new_device", "is_new_country", "hour_of_day", "is_unusual_hour",
        "seconds_since_last", "impossible_travel_speed_kmh", "distance_from_home_km"]}
    feats["impossible_travel_speed_kmh"] = 8000.0
    res = evaluate_rules(feats, BANKING_RULES)
    assert any(f.name == "impossible_travel" for f in res.fired)
    assert res.score > 0.5

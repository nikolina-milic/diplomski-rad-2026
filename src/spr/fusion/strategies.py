from dataclasses import dataclass


@dataclass
class FusionConfig:
    strategy: str = "weighted"        # "cascade" | "weighted" | "stacking"
    rules_weight: float = 0.5
    ml_weight: float = 0.5
    cascade_threshold: float = 0.7    # ako rules_score >= prag, vjeruj pravilima


def _cascade(rules_score, ml_score, cfg):
    return rules_score if rules_score >= cfg.cascade_threshold else ml_score


def _weighted(rules_score, ml_score, cfg):
    total = cfg.rules_weight + cfg.ml_weight
    if total <= 0:
        return 0.0
    return (cfg.rules_weight * rules_score + cfg.ml_weight * ml_score) / total


def _stacking(rules_score, ml_score, cfg):
    # meta-model dolazi u learning ciklusu; do tada degradira na weighted
    return _weighted(rules_score, ml_score, cfg)


_STRATEGIES = {"cascade": _cascade, "weighted": _weighted, "stacking": _stacking}


def fuse(rules_score: float, ml_score: float, cfg: FusionConfig) -> float:
    fn = _STRATEGIES.get(cfg.strategy)
    if fn is None:
        raise ValueError(f"Nepoznata fusion strategija: {cfg.strategy}")
    return max(0.0, min(1.0, fn(rules_score, ml_score, cfg)))

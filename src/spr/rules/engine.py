import ast
from dataclasses import dataclass


def _threshold_nodes(condition: str) -> list[ast.Constant]:
    """Numeričke konstante (pragovi) u uslovu, sortirane po poziciji.

    Koristi ast da razlikuje broj-prag od cifre u imenu feature-a
    (npr. txn_count_1h, txn_count_24h nisu pragovi).
    """
    tree = ast.parse(condition, mode="eval")
    nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    ]
    return sorted(nodes, key=lambda n: n.col_offset)


def _fmt_threshold(value: float) -> str:
    """Cjelobrojne vrijednosti bez '.0' (da uslov ostane čist)."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def split_condition(condition: str) -> tuple[list[str], list[float]]:
    """Razloži uslov na tekstualne dijelove i numeričke pragove.

    parts ima dužinu len(thresholds) + 1; naizmjenično preplitanje
    parts[0], thresholds[0], parts[1], ... rekonstruiše uslov (za inline UI).
    """
    nodes = _threshold_nodes(condition)
    parts: list[str] = []
    thresholds: list[float] = []
    pos = 0
    for n in nodes:
        parts.append(condition[pos:n.col_offset])
        thresholds.append(n.value)
        pos = n.end_col_offset
    parts.append(condition[pos:])
    return parts, thresholds


def threshold_features(condition: str) -> list[str]:
    """Ime feature-a uz koji se poredi svaki prag (poravnato sa split_condition).

    Npr. "is_unusual_hour >= 1 and amount_ratio > 3" ->
    ["is_unusual_hour", "amount_ratio"].
    """
    tree = ast.parse(condition, mode="eval")
    const_to_feat: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            names = [o.id for o in operands if isinstance(o, ast.Name)]
            feat = names[0] if names else ""
            for o in operands:
                if (isinstance(o, ast.Constant)
                        and isinstance(o.value, (int, float))
                        and not isinstance(o.value, bool)):
                    const_to_feat[o.col_offset] = feat
    return [const_to_feat.get(n.col_offset, "") for n in _threshold_nodes(condition)]


def apply_thresholds(condition: str, values: list[float]) -> str:
    """Vrati uslov sa zamijenjenim numeričkim pragovima.

    Zamjena ide zdesna nalijevo da offseti ostanu validni. Baca ValueError
    ako se broj vrijednosti ne poklapa ili rezultat više ne parsira.
    """
    nodes = _threshold_nodes(condition)
    if len(values) != len(nodes):
        raise ValueError(
            f"očekivano {len(nodes)} pragova, dobijeno {len(values)}")
    out = condition
    for n, v in sorted(zip(nodes, values), key=lambda p: p[0].col_offset,
                       reverse=True):
        out = out[:n.col_offset] + _fmt_threshold(v) + out[n.end_col_offset:]
    ast.parse(out, mode="eval")  # sanity: rezultat mora biti validan izraz
    return out


@dataclass
class Rule:
    name: str
    condition: str      # Python izraz nad imenima feature-a, npr. "amount_zscore > 4"
    weight: float       # doprinos riziku [0,1]
    explanation: str    # čitljiv razlog kad se okine
    enabled: bool = True  # isključena pravila se preskaču pri evaluaciji


@dataclass
class FiredRule:
    name: str
    weight: float
    explanation: str


@dataclass
class RulesResult:
    score: float
    fired: list[FiredRule]


def evaluate_rules(features: dict[str, float], rules: list[Rule]) -> RulesResult:
    """Izvrši pravila nad feature vektorom.

    Uslov je Python izraz nad imenima feature-a, izvršen u sandboxu bez
    builtins-a (ulaz su pouzdani domain packovi). Skor je noisy-OR
    kombinacija težina okinutih pravila: 1 - Π(1 - w_i), ograničeno na [0,1].
    """
    fired: list[FiredRule] = []
    for r in rules:
        if not getattr(r, "enabled", True):
            continue
        try:
            ok = bool(eval(r.condition, {"__builtins__": {}}, features))
        except Exception:
            ok = False
        if ok:
            fired.append(FiredRule(r.name, r.weight, r.explanation))
    prod = 1.0
    for f in fired:
        prod *= (1.0 - f.weight)
    return RulesResult(score=1.0 - prod, fired=fired)

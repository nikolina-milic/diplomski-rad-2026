"""Pragovi izvedeni iz matrice troška, umjesto proizvoljno izabranih.

Pragovi 0.3 / 0.7 nemaju uporište — biraju se „jer izgledaju razumno". Ako je
skor *kalibrisana vjerovatnoća* prevare (vidi `spr.ml.calibration`), optimalan
prag se može izvesti: za dati p bira se akcija sa najmanjim očekivanim troškom.

Model troška po akciji, uz p = P(prevara) i L = prosječan gubitak po prevari:

    ALLOW      p · L
    CHALLENGE  c_ch + p · (1 − s_ch) · L
    REVIEW     c_rv + p · (1 − s_rv) · L
    BLOCK      (1 − p) · c_bl

gdje su c_* troškovi akcije, a s_* vjerovatnoće da akcija zaustavi prevaru.
Tačke u kojima se mijenja argmin daju pragove u zatvorenoj formi:

    low_max    = c_ch / (L · s_ch)
    medium_max = (c_rv − c_ch) / (L · (s_rv − s_ch))

`empirical_thresholds` istu stvar radi pretragom po labeliranom skupu i služi
kao provjera da zatvorena forma nije promašila.
"""

from dataclasses import dataclass, asdict

import numpy as np

from spr.policy.engine import Action, PolicyConfig, RiskLevel


@dataclass
class CostMatrix:
    """Troškovi u novčanim jedinicama (EUR), po jednom događaju."""
    avg_fraud_loss: float = 250.0   # L — prosječan gubitak od propuštene prevare
    challenge_cost: float = 0.5     # c_ch — trenje korisnika (2FA, SMS)
    review_cost: float = 8.0        # c_rv — vrijeme analitičara
    block_cost: float = 120.0       # c_bl — blokiran legitiman korisnik
    challenge_stop_rate: float = 0.7   # s_ch — udio prevara koje 2FA zaustavi
    review_stop_rate: float = 0.95     # s_rv — udio koji ručni pregled zaustavi

    def to_dict(self) -> dict:
        return asdict(self)


def action_cost(action: Action, is_fraud: bool, cost: CostMatrix) -> float:
    """Stvarni trošak jedne odluke kad se zna ishod."""
    L = cost.avg_fraud_loss
    if action is Action.ALLOW:
        return L if is_fraud else 0.0
    if action is Action.CHALLENGE:
        return cost.challenge_cost + ((1 - cost.challenge_stop_rate) * L if is_fraud else 0.0)
    if action is Action.REVIEW:
        return cost.review_cost + ((1 - cost.review_stop_rate) * L if is_fraud else 0.0)
    return 0.0 if is_fraud else cost.block_cost


def expected_action_cost(action: Action, p: float, cost: CostMatrix) -> float:
    """Očekivani trošak akcije pri vjerovatnoći prevare p."""
    return (p * action_cost(action, True, cost)
            + (1 - p) * action_cost(action, False, cost))


def optimal_thresholds(cost: CostMatrix) -> tuple[float, float]:
    """Zatvorena forma: (low_max, medium_max)."""
    L = cost.avg_fraud_loss
    s_ch, s_rv = cost.challenge_stop_rate, cost.review_stop_rate
    if L <= 0 or s_ch <= 0:
        return 0.3, 0.7
    low_max = cost.challenge_cost / (L * s_ch)
    if s_rv <= s_ch:
        medium_max = 1.0
    else:
        medium_max = (cost.review_cost - cost.challenge_cost) / (L * (s_rv - s_ch))
    low_max = float(np.clip(low_max, 0.0, 1.0))
    medium_max = float(np.clip(max(medium_max, low_max), 0.0, 1.0))
    return low_max, medium_max


def total_cost(y_true, scores, low_max: float, medium_max: float,
               cost: CostMatrix) -> float:
    """Ukupan trošak politike na labeliranom skupu."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    total = 0.0
    for label, score in zip(y, s):
        if score <= low_max:
            action = Action.ALLOW
        elif score <= medium_max:
            action = Action.CHALLENGE
        else:
            action = Action.REVIEW
        total += action_cost(action, bool(label), cost)
    return float(total)


def empirical_thresholds(y_true, scores, cost: CostMatrix,
                         grid: int = 60) -> tuple[float, float, float]:
    """Pretraga po mreži: (low_max, medium_max, ukupan_trošak).

    Nezavisna provjera zatvorene forme — na dobro kalibrisanim skorovima dvije
    metode moraju dati blizak rezultat.
    """
    candidates = np.linspace(0.0, 1.0, grid)
    best = (0.3, 0.7, float("inf"))
    for low in candidates:
        for med in candidates:
            if med < low:
                continue
            c = total_cost(y_true, scores, float(low), float(med), cost)
            if c < best[2]:
                best = (float(low), float(med), c)
    return best


def apply_to_policy(policy: PolicyConfig, cost: CostMatrix) -> PolicyConfig:
    """Postavi pragove politike na vrijednosti optimalne za datu matricu troška."""
    policy.low_max, policy.medium_max = optimal_thresholds(cost)
    return policy


def cost_report(y_true, scores, policy: PolicyConfig, cost: CostMatrix) -> dict:
    """Poređenje trenutnih pragova sa optimalnim — koliko politika košta."""
    closed_low, closed_med = optimal_thresholds(cost)
    emp_low, emp_med, emp_cost = empirical_thresholds(y_true, scores, cost)
    current = total_cost(y_true, scores, policy.low_max, policy.medium_max, cost)
    closed = total_cost(y_true, scores, closed_low, closed_med, cost)
    n = max(len(list(y_true)), 1)
    return {
        "cost_matrix": cost.to_dict(),
        "current": {"low_max": policy.low_max, "medium_max": policy.medium_max,
                    "total_cost": current, "cost_per_event": current / n},
        "closed_form": {"low_max": closed_low, "medium_max": closed_med,
                        "total_cost": closed, "cost_per_event": closed / n},
        "empirical": {"low_max": emp_low, "medium_max": emp_med,
                      "total_cost": emp_cost, "cost_per_event": emp_cost / n},
        "savings_vs_current": current - closed,
    }


LEVEL_ACTIONS_DEFAULT = {
    RiskLevel.LOW: Action.ALLOW,
    RiskLevel.MEDIUM: Action.CHALLENGE,
    RiskLevel.HIGH: Action.REVIEW,
}


@dataclass
class CapacityConstraints:
    """Operativna ograničenja koja čista minimizacija troška ignoriše.

    Bez njih optimizacija daje ekonomski tačan, ali neupotrebljiv odgovor: kad
    prevara košta 250 EUR, a 2FA izazov 0.50 EUR, isplati se izazvati korisnika
    i pri vjerovatnoći od 0.3% — dakle gotovo svaku transakciju. U praksi
    postoje budžet trenja (koliko korisnika smijemo uznemiriti) i kapacitet
    analitičara (koliko slučajeva stvarno možemo pregledati).
    """
    max_challenge_rate: float = 0.05   # udio transakcija koje smiju dobiti 2FA
    max_review_rate: float = 0.01      # dnevni kapacitet analitičara

    def to_dict(self) -> dict:
        return asdict(self)


def action_rates(scores, low_max: float, medium_max: float) -> tuple[float, float]:
    """(udio CHALLENGE, udio REVIEW) pri datim pragovima."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.0, 0.0
    review = float(np.mean(s > medium_max))
    challenge = float(np.mean((s > low_max) & (s <= medium_max)))
    return challenge, review


def constrained_thresholds(y_true, scores, cost: CostMatrix,
                           capacity: CapacityConstraints,
                           grid: int = 80) -> dict:
    """Minimizuj očekivani trošak uz poštovanje kapaciteta.

    Pretraga ide po kvantilima skora (a ne po ravnomjernoj mreži) jer su
    kalibrisani skorovi jako zgusnuti pri nuli — ravnomjerna mreža bi
    promašila sve zanimljive pragove.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        lo, hi = optimal_thresholds(cost)
        return {"low_max": lo, "medium_max": hi, "total_cost": 0.0,
                "challenge_rate": 0.0, "review_rate": 0.0, "feasible": False}

    # Kalibrisani skorovi su zgusnuti pri nuli, a granice akcija leže u gornjem
    # repu. Ravnomjerna mreža kvantila tamo je pregruba i preskoči jeftinije
    # pragove, pa se rep dodatno progušćuje, plus tačke same granice kapaciteta.
    qs = np.concatenate([
        np.linspace(0.0, 1.0, grid),
        np.linspace(0.90, 1.0, grid),
        np.linspace(0.99, 1.0, grid),
        [1.0 - capacity.max_review_rate,
         max(0.0, 1.0 - capacity.max_challenge_rate - capacity.max_review_rate)],
    ])
    candidates = np.unique(np.quantile(s, np.clip(np.unique(qs), 0.0, 1.0)))
    best = None
    for low in candidates:
        for med in candidates:
            if med < low:
                continue
            ch, rv = action_rates(s, float(low), float(med))
            if ch > capacity.max_challenge_rate or rv > capacity.max_review_rate:
                continue
            c = total_cost(y_true, s, float(low), float(med), cost)
            if best is None or c < best["total_cost"]:
                best = {"low_max": float(low), "medium_max": float(med),
                        "total_cost": c, "challenge_rate": ch,
                        "review_rate": rv, "feasible": True}
    if best is None:
        # nijedna kombinacija ne staje u kapacitet — uzmi najstrože pragove
        low = float(np.quantile(s, 1.0 - capacity.max_challenge_rate
                                - capacity.max_review_rate))
        med = float(np.quantile(s, 1.0 - capacity.max_review_rate))
        ch, rv = action_rates(s, low, med)
        best = {"low_max": low, "medium_max": med,
                "total_cost": total_cost(y_true, s, low, med, cost),
                "challenge_rate": ch, "review_rate": rv, "feasible": False}
    return best


def apply_constrained_to_policy(policy: PolicyConfig, y_true, scores,
                                cost: CostMatrix,
                                capacity: CapacityConstraints) -> dict:
    """Postavi pragove na trošak-optimalne *uz* ograničenja kapaciteta."""
    best = constrained_thresholds(y_true, scores, cost, capacity)
    policy.low_max = best["low_max"]
    policy.medium_max = best["medium_max"]
    return best

"""Bankarski domain pack: domenski-specifične konstante."""

DEFAULT_CURRENCY = "EUR"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "electronics",
    "travel", "clothing", "utilities", "entertainment",
]

# Reprezentativne (lat, lon) koordinate zemalja za simulaciju lokacije.
COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "RS": (44.82, 20.46),   # Beograd
    "HR": (45.81, 15.98),   # Zagreb
    "DE": (52.52, 13.40),   # Berlin
    "FR": (48.85, 2.35),    # Pariz
    "US": (40.71, -74.01),  # New York
    "JP": (35.68, 139.69),  # Tokio
    "BR": (-23.55, -46.63),  # Sao Paulo
    "AU": (-33.87, 151.21),  # Sydney
}

# Reprezentativni grad po zemlji (za prikaz "odakle je transakcija").
COUNTRY_CITY: dict[str, str] = {
    "RS": "Beograd", "HR": "Zagreb", "DE": "Berlin", "FR": "Pariz",
    "US": "New York", "JP": "Tokio", "BR": "Sao Paulo", "AU": "Sydney",
}

# Realistična imena i prezimena po zemlji (za dodjelu identiteta personi).
FIRST_NAMES: dict[str, list[str]] = {
    "RS": ["Marko", "Jelena", "Nikola", "Ana", "Stefan", "Milica", "Luka", "Ivana"],
    "HR": ["Ivan", "Ana", "Marko", "Ivana", "Josip", "Marija", "Luka", "Petra"],
    "DE": ["Lukas", "Anna", "Felix", "Laura", "Jonas", "Lena", "Paul", "Marie"],
    "FR": ["Louis", "Emma", "Hugo", "Léa", "Lucas", "Chloé", "Jules", "Manon"],
    "US": ["James", "Emily", "Michael", "Sarah", "David", "Jessica", "Daniel", "Ashley"],
    "JP": ["Haruto", "Yui", "Sota", "Aoi", "Ren", "Sakura", "Yuto", "Hina"],
    "BR": ["João", "Maria", "Pedro", "Ana", "Lucas", "Julia", "Gabriel", "Beatriz"],
    "AU": ["Jack", "Charlotte", "Oliver", "Amelia", "William", "Olivia", "Noah", "Mia"],
}

LAST_NAMES: dict[str, list[str]] = {
    "RS": ["Jovanović", "Petrović", "Nikolić", "Marković", "Đorđević", "Stojanović", "Ilić", "Pavlović"],
    "HR": ["Horvat", "Kovačević", "Babić", "Marić", "Novak", "Jurić", "Knežević", "Vuković"],
    "DE": ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Wagner", "Becker", "Hoffmann"],
    "FR": ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand"],
    "US": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"],
    "JP": ["Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito", "Yamamoto", "Nakamura"],
    "BR": ["Silva", "Santos", "Oliveira", "Souza", "Lima", "Pereira", "Costa", "Almeida"],
    "AU": ["Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "White", "Martin"],
}

from spr.rules.engine import Rule

# Deterministička poslovna pravila bankarskog domena.
# Uslovi su izrazi nad imenima feature-a iz spr.context.features.FEATURE_NAMES.
BANKING_RULES: list[Rule] = [
    Rule("impossible_travel", "impossible_travel_speed_kmh > 1000", 0.9,
         "Fizički nemoguće putovanje između uzastopnih transakcija"),
    Rule("very_large_amount", "amount_ratio > 8", 0.7,
         "Iznos višestruko veći od uobičajenog za korisnika"),
    Rule("high_amount_deviation", "amount_zscore > 4", 0.6,
         "Iznos znatno odstupa od korisnikovog prosjeka"),
    Rule("rapid_fire", "txn_count_1h >= 5", 0.6,
         "Neuobičajeno mnogo transakcija u kratkom periodu"),
    Rule("new_country", "is_new_country >= 1", 0.4,
         "Transakcija iz zemlje koju korisnik ranije nije koristio"),
    Rule("new_device", "is_new_device >= 1", 0.3,
         "Transakcija sa novog uređaja"),
    Rule("unusual_hour_spending", "is_unusual_hour >= 1 and amount_ratio > 3", 0.5,
         "Povišen iznos u neuobičajeno doba dana"),
]

from spr.policy.engine import Action, RiskLevel, PolicyConfig
from spr.policy.cost import CapacityConstraints, CostMatrix
from spr.fusion.strategies import FusionConfig
from spr.fusion.guardrails import Guardrail

# Meta-model kombinuje kalibrisane skorove pravila i ML modela.
BANKING_FUSION = FusionConfig(strategy="stacking", rules_weight=0.5, ml_weight=0.5)

# Trošak po događaju u EUR — iz ovoga se izvode pragovi politike.
BANKING_COST = CostMatrix(
    avg_fraud_loss=250.0,
    challenge_cost=0.5,
    review_cost=8.0,
    block_cost=120.0,
    challenge_stop_rate=0.7,
    review_stop_rate=0.95,
)

# Operativna ograničenja: budžet trenja i kapacitet analitičara.
BANKING_CAPACITY = CapacityConstraints(
    max_challenge_rate=0.05,
    max_review_rate=0.01,
)

# Pragovi su polazna vrijednost; `build_default_engine` ih prepisuje
# vrijednostima izvedenim iz BANKING_COST (spr.policy.cost.optimal_thresholds).
BANKING_POLICY = PolicyConfig(
    low_max=0.3,
    medium_max=0.7,
    level_actions={
        RiskLevel.LOW: Action.ALLOW,
        RiskLevel.MEDIUM: Action.CHALLENGE,
        RiskLevel.HIGH: Action.REVIEW,
    },
    mode="shadow",
)

BANKING_GUARDRAILS: list[Guardrail] = [
    Guardrail("compliance_hard_limit", "amount > 15000", Action.BLOCK,
              "Iznos prelazi regulatorni prag — obavezno blokiranje"),
    Guardrail("trusted_micro_payment", "amount < 1 and is_new_device < 1", Action.ALLOW,
              "Mikro-plaćanje sa poznatog uređaja — automatski propušteno"),
]

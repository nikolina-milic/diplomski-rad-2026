import numpy as np

from spr.context.features import FEATURE_NAMES, compute_features
from spr.context.store import ContextStore
from spr.domain.schema import Event


def build_dataset(events: list[Event]) -> tuple[list[dict], list[int]]:
    """Hronološki izgradi (feature-i, label) parove bez curenja podataka."""
    store = ContextStore()
    X: list[dict] = []
    y: list[int] = []
    for e in events:
        feats = compute_features(e, store.get(e.user_id))
        store.update(e)
        X.append(feats)
        y.append(1 if e.is_fraud else 0)
    return X, y


def to_matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )

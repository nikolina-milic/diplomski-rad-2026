from datetime import datetime, timezone
from spr.generator.generator import generate_batch
from spr.context.features import FEATURE_NAMES
from spr.ml.dataset import build_dataset, to_matrix

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_build_dataset_shapes_and_labels():
    events = generate_batch(n_users=10, n_normal=300, fraud_rate=0.1, seed=1, start=START, days=7)
    X, y = build_dataset(events)
    assert len(X) == len(y) == len(events)
    assert set(X[0].keys()) == set(FEATURE_NAMES)
    assert set(y) <= {0, 1}
    assert sum(y) > 0  # ima prevara


def test_to_matrix_shape_and_order():
    events = generate_batch(n_users=5, n_normal=50, fraud_rate=0.1, seed=2, start=START, days=7)
    X, _ = build_dataset(events)
    mat = to_matrix(X)
    assert mat.shape == (len(X), len(FEATURE_NAMES))
    # kolona 0 mora odgovarati FEATURE_NAMES[0] == "amount"
    assert mat[0, 0] == X[0][FEATURE_NAMES[0]]

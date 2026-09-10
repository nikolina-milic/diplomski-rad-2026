from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = list(y_true)
    y_score = list(y_score)
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    pr_auc = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0
    return {
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ROC-AUC on probabilities. Returns np.nan if y_true has only one class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) on probabilities. Returns np.nan if y_true has only one class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def binarize(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert probabilities to {0,1} predictions at a fixed threshold.
    """
    y_prob = np.asarray(y_prob)
    return (y_prob >= threshold).astype(int)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    @property
    def specificity(self) -> float:
        denom = (self.tn + self.fp)
        return float(self.tn / denom) if denom else float("nan")


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """
    Confusion matrix + standard threshold metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = binarize(y_prob, threshold)

    # confusion_matrix returns [[tn, fp], [fn, tp]] for labels [0,1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ThresholdMetrics(
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: Optional[Iterable[float]] = None,
) -> ThresholdMetrics:
    """
    Search over thresholds and return the best one according to metric.
    Supported metrics: "f1", "precision", "recall", "accuracy"
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    metric = metric.lower().strip()
    allowed = {"f1", "precision", "recall", "accuracy"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")

    best: Optional[ThresholdMetrics] = None
    best_val = -float("inf")

    for thr in thresholds:
        m = metrics_at_threshold(y_true, y_prob, float(thr))
        val = getattr(m, metric)
        if val > best_val:
            best_val = val
            best = m

    assert best is not None
    return best

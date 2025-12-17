from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ConfusionResult:
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    recall: float


def pick_threshold_for_target_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    target_recall: float = 0.64,
    grid: Optional[np.ndarray] = None,
) -> ConfusionResult:
    """
    Choose threshold whose recall is closest to target_recall.
    Tie-breaker: fewer false positives.
    Designed to match report’s “recall-prioritising” confusion matrix narrative. :contentReference[oaicite:7]{index=7}
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best = None
    best_dist = float("inf")
    best_fp = float("inf")

    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) else 0.0

        dist = abs(rec - target_recall)
        if (dist < best_dist) or (dist == best_dist and fp < best_fp):
            best_dist = dist
            best_fp = fp
            best = (float(t), int(tn), int(fp), int(fn), int(tp), float(rec))

    assert best is not None
    t, tn, fp, fn, tp, rec = best
    return ConfusionResult(threshold=t, tn=tn, fp=fp, fn=fn, tp=tp, recall=rec)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    out_path: Path,
    *,
    title: str = "Confusion Matrix",
) -> None:
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Non-churn", "Churn"])
    plt.yticks([0, 1], ["Non-churn", "Churn"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

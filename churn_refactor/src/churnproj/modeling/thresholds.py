from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: Dict[str, float]


def pick_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: Literal["f1", "precision", "recall"] = "f1",
    grid: Optional[np.ndarray] = None,
) -> ThresholdResult:
    """Choose a probability threshold that maximizes the selected metric."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_val = -np.inf

    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            val = precision_score(y_true, y_pred, zero_division=0)
        else:
            val = recall_score(y_true, y_pred, zero_division=0)

        if val > best_val:
            best_val = val
            best_t = float(t)

    y_pred = (y_proba >= best_t).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return ThresholdResult(threshold=best_t, metrics=metrics)
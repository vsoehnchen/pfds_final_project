from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvalResult:
    roc_auc: float
    f1: float
    precision: float
    recall: float
    confusion: np.ndarray


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> EvalResult:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    return EvalResult(
        roc_auc=float(roc_auc_score(y, proba)),
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        confusion=confusion_matrix(y, pred),
    )
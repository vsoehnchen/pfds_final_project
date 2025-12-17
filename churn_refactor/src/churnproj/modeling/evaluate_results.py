from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from churnproj.modeling.evaluate import evaluate_classifier
from churnproj.modeling.split import SnapshotTimeSplit


def _fresh_model(model_or_factory: Any) -> Any:
    """
    Accepts either:
      - a scikit-learn estimator instance (will be cloned), or
      - a zero-arg factory function returning a new estimator.
    """
    if callable(model_or_factory) and not hasattr(model_or_factory, "fit"):
        return model_or_factory()
    return clone(model_or_factory)


def evaluate_models_temporal_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    models_dict: Mapping[str, Any],
    splitter: Any | None = None,
    *,
    y_col: str = "churn",
    snapshot_col: str = "snapshot_date",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Temporal CV evaluation (by snapshot blocks).

    Returns a tidy DataFrame with per-fold metrics for each model, plus a summary
    (mean/std) appended at the bottom for convenience.

    Metrics:
      - roc_auc (from evaluate_classifier)
      - pr_auc  (average_precision_score)
      - f1, precision, recall, confusion matrix at `threshold`
    """
    required = set(feature_cols) | {y_col, snapshot_col}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df is missing required columns: {sorted(missing)}")

    X_all = train_df[feature_cols]
    y_all = train_df[y_col].astype(int)

    if splitter is None:
        splitter = SnapshotTimeSplit(train_df[snapshot_col])

    rows: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_all, y_all), start=1):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all.iloc[va_idx]

        for model_name, model_or_factory in models_dict.items():
            model = _fresh_model(model_or_factory)
            model.fit(X_tr, y_tr)

            # threshold metrics + ROC-AUC via your teammate's evaluator
            ev = evaluate_classifier(model, X_va, y_va, threshold=threshold)

            # PR-AUC (Average Precision) on probabilities
            proba = model.predict_proba(X_va)[:, 1]
            pr = float("nan")
            if np.unique(y_va).size >= 2:
                pr = float(average_precision_score(y_va, proba))

            # flatten confusion matrix if it’s 2x2
            cm = ev.confusion
            tn = fp = fn = tp = None
            if cm.shape == (2, 2):
                tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "n_train": int(len(tr_idx)),
                    "n_val": int(len(va_idx)),
                    "roc_auc": ev.roc_auc,
                    "pr_auc": pr,
                    "f1": ev.f1,
                    "precision": ev.precision,
                    "recall": ev.recall,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "threshold": float(threshold),
                }
            )

    fold_df = pd.DataFrame(rows)

    # Summary stats per model (mean/std) – helpful for the report table
    summary = (
        fold_df.groupby("model")[["roc_auc", "pr_auc", "f1", "precision", "recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    # flatten multiindex columns
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in summary.columns.to_flat_index()
    ]
    summary.insert(1, "fold", "summary")

    # Return fold rows + summary appended (still one DataFrame)
    out = pd.concat([fold_df, summary], ignore_index=True)
    return out

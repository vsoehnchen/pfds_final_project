from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from churnproj.modeling.evaluate import EvalResult, evaluate_classifier
from churnproj.modeling.split import SnapshotTimeSplit
from churnproj.modeling.thresholds import ThresholdResult, pick_threshold
from churnproj.modeling.train import TrainResult, train_xgb_gridsearch


@dataclass
class FinalModelResult:
    gridsearch: TrainResult
    threshold: ThresholdResult
    eval_last_fold: EvalResult
    last_fold_date: pd.Timestamp

    # NEW: for SHAP + report-style confusion matrix without retraining
    model: Any
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.Series
    y_proba: np.ndarray



def train_final_xgb_and_threshold_on_last_fold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    snapshot_col: str = "snapshot_date",
    y_col: str = "churn",
    param_grid: Optional[Dict[str, list]] = None,
    scoring: str = "roc_auc",
    threshold_metric: str = "f1",
    seed: int = 42,
) -> FinalModelResult:
    """
    1) GridSearchCV over XGB using temporal CV (SnapshotTimeSplit).
    2) Take the LAST temporal fold as a final validation split.
    3) Fit best-params XGB on last-fold train, tune threshold on last-fold val.
    4) Evaluate confusion matrix + ROC-AUC + f1/precision/recall on last-fold val.
    """
    required = set(feature_cols) | {snapshot_col, y_col}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df missing required columns: {sorted(missing)}")

    X_all = train_df[feature_cols]
    y_all = train_df[y_col].astype(int)
    splitter = SnapshotTimeSplit(train_df[snapshot_col])

    # Default grid (small, safe) if none provided
    if param_grid is None:
        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [4, 5],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        }

    # 1) Grid search using temporal CV
    gs_result = train_xgb_gridsearch(
        X=X_all,
        y=y_all,
        cv=splitter,
        param_grid=param_grid,
        scoring=scoring,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        verbose=0,
    )

    # 2) Last fold split
    splits = list(splitter.split(X_all, y_all))
    if not splits:
        raise RuntimeError("Not enough snapshot dates for temporal CV (need >= 2).")

    tr_idx, va_idx = splits[-1]
    X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
    X_va, y_va = X_all.iloc[va_idx], y_all.iloc[va_idx]

    # Figure out which snapshot date is the validation block
    last_fold_date = pd.to_datetime(train_df.iloc[va_idx][snapshot_col].unique()[0])

    # 3) Fit best params on last-fold train
    # (Don’t reuse gs_result.best_estimator directly; it was refit on all data.)
    best_params = dict(gs_result.best_params)
    model = xgb.XGBClassifier(
        **best_params,
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        enable_categorical=False,
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)

    # 4) Pick threshold on last-fold val
    y_proba = model.predict_proba(X_va)[:, 1]
    thr_res = pick_threshold(
        y_true=y_va.to_numpy(),
        y_proba=y_proba,
        metric=threshold_metric,  # "f1" | "precision" | "recall"
    )

    # 5) Evaluate at that threshold
    eval_res = evaluate_classifier(model, X_va, y_va, threshold=thr_res.threshold)

    return FinalModelResult(
        gridsearch=gs_result,
        threshold=thr_res,
        eval_last_fold=eval_res,
        last_fold_date=last_fold_date,

        model=model,
        X_train=X_tr,
        X_val=X_va,
        y_val=y_va,
        y_proba=y_proba,
    )

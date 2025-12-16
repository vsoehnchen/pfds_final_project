from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb


@dataclass
class TrainResult:
    best_estimator: Any
    best_params: Dict[str, Any]
    cv_results_: Dict[str, Any]


def train_xgb_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    param_grid: Dict[str, list],
    scoring: str = "roc_auc",
    eval_metric: str = "logloss",
    tree_method: str = "hist",
    n_jobs: int = -1,
    verbose: int = 0,
) -> TrainResult:
    """
    Train an XGBoost classifier via GridSearchCV with a provided CV splitter.
    """
    model = xgb.XGBClassifier(
        tree_method=tree_method,
        eval_metric=eval_metric,
        n_jobs=n_jobs,
        enable_categorical=False,
        use_label_encoder=False,
    )

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )
    gs.fit(X, y)

    return TrainResult(
        best_estimator=gs.best_estimator_,
        best_params=gs.best_params_,
        cv_results_=gs.cv_results_,
    )
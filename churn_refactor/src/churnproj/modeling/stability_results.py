from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from churnproj.modeling.split import SnapshotTimeSplit


@dataclass(frozen=True)
class StabilityResult:
    counts: pd.Series  # feature -> occurrence count
    top_k: int


def _gain_importance_per_fold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    y_col: str = "churn",
    snapshot_col: str = "snapshot_date",
    seed: int = 42,
) -> pd.DataFrame:
    X = train_df[feature_cols]
    y = train_df[y_col].astype(int)
    splitter = SnapshotTimeSplit(train_df[snapshot_col])

    gain_series = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
            tree_method="hist",
            random_state=seed + fold_idx,
        )
        model.fit(X_tr, y_tr)

        gain = model.get_booster().get_score(importance_type="gain")
        gain_series.append(pd.Series(gain, name=f"Fold {fold_idx}"))

    gain_df = pd.concat(gain_series, axis=1).fillna(0.0)
    return gain_df


def stability_topk_across_folds(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    top_k: int = 3,
    seed: int = 42,
) -> StabilityResult:
    """
    Count how often each feature appears in the top-k gain features across folds.
    Intended to reproduce Fig 5 stability comparison (full vs reduced). :contentReference[oaicite:5]{index=5}
    """
    gain_df = _gain_importance_per_fold(train_df, feature_cols, seed=seed)
    fold_cols = [c for c in gain_df.columns if c.startswith("Fold ")]

    counts: Dict[str, int] = {}
    for col in fold_cols:
        top = gain_df[col].sort_values(ascending=False).head(top_k).index
        for f in top:
            counts[f] = counts.get(f, 0) + 1

    s = pd.Series(counts).sort_values(ascending=False)
    return StabilityResult(counts=s, top_k=top_k)


def save_stability_comparison_plot(
    full: StabilityResult,
    reduced: StabilityResult,
    out_path: Path,
    *,
    top_n: int = 15,
) -> None:
    """
    Single figure with two panels: (a) full, (b) reduced.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160, constrained_layout=True)

    full_top = full.counts.head(top_n).iloc[::-1]
    axes[0].barh(full_top.index, full_top.values)
    axes[0].set_title(f"Full model stability (top-{full.top_k})")
    axes[0].set_xlabel("Occurrence count across folds")

    red_top = reduced.counts.head(top_n).iloc[::-1]
    axes[1].barh(red_top.index, red_top.values)
    axes[1].set_title(f"Reduced model stability (top-{reduced.top_k})")
    axes[1].set_xlabel("Occurrence count across folds")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

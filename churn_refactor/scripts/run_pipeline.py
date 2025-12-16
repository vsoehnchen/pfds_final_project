from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from churnproj.config import ChurnConfig
from churnproj.data.load import load_parquet, ensure_datetime, basic_schema_check
from churnproj.data.snapshots import slice_history
from churnproj.features.build import build_dual_window_features
from churnproj.features.labels import label_churn_inactivity, label_churn_cancellation_event
from churnproj.modeling.split import SnapshotTimeSplit
from churnproj.modeling.train import train_xgb_gridsearch
from churnproj.modeling.thresholds import pick_threshold
from churnproj.modeling.evaluate import evaluate_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to parquet logs")
    ap.add_argument("--out", type=str, default="reports", help="Output directory")
    args = ap.parse_args()

    cfg = ChurnConfig()

    df = load_parquet(args.data)
    df = ensure_datetime(df, "time")
    basic_schema_check(df)

    # Example: pick one snapshot date (latest full snapshot)
    T = df["time"].max().floor("D")

    df_hist = slice_history(df, T, history_days=cfg.history_days_long)

    feats = build_dual_window_features(
        df_hist_long=df_hist,
        snapshot_T=T,
        window_long=cfg.history_days_long,
        window_short=cfg.history_days_short,
    )

    # attach labels
    if cfg.churn_rule == "cancellation_event":
        churn = label_churn_cancellation_event(df, T, cfg.horizon_days, cfg.cancellation_page_value)
    else:
        churn = label_churn_inactivity(df, T, cfg.horizon_days)

    X = feats.set_index("userId")
    y = churn.reindex(X.index).fillna(0).astype(int)

    # for demo: treat single snapshot as one block
    snapshot_series = pd.Series([T] * len(X), index=X.index)
    cv = SnapshotTimeSplit(snapshot_series)

    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    result = train_xgb_gridsearch(X, y, cv=cv, param_grid=param_grid, scoring="roc_auc")
    proba = result.best_estimator.predict_proba(X)[:, 1]
    thr = pick_threshold(y.to_numpy(), proba, metric="f1")

    metrics = evaluate_classifier(result.best_estimator, X, y, threshold=thr.threshold)
    print("Best params:", result.best_params)
    print("Chosen threshold:", thr.threshold, thr.metrics)
    print("Eval:", metrics)

if __name__ == "__main__":
    main()
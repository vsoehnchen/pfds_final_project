from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model evaluation + results pipeline")

    p.add_argument(
        "--train-parquet",
        type=str,
        default=os.environ.get("CHURN_TRAIN_PARQUET", ""),
        help="Path to train.parquet (or set env var CHURN_TRAIN_PARQUET)",
    )
    p.add_argument(
        "--test-parquet",
        type=str,
        default=os.environ.get("CHURN_TEST_PARQUET", ""),
        help="Path to test.parquet (optional; or set env var CHURN_TEST_PARQUET)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="reports",
        help="Output directory for figures/tables",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def infer_feature_cols(train_df: pd.DataFrame) -> list[str]:
    drop_cols = {"userId", "snapshot_date", "churn"}
    cols = [c for c in train_df.columns if c not in drop_cols]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    if not cols:
        raise ValueError(f"No numeric feature columns inferred. Columns: {list(train_df.columns)}")
    return cols


def main() -> None:
    args = parse_args()

    if not args.train_parquet:
        raise SystemExit("Missing --train-parquet (or set CHURN_TRAIN_PARQUET environment variable).")

    train_path = Path(args.train_parquet)
    if not train_path.exists():
        raise SystemExit(f"Train parquet not found: {train_path}")

    if args.test_parquet:
        test_path = Path(args.test_parquet)
        if not test_path.exists():
            raise SystemExit(f"Test parquet not found: {test_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_src_on_path()

    # imports that depend on src/ being on sys.path
    from churnproj.data.load import basic_schema_check, ensure_datetime, load_parquet
    from churnproj.data.snapshots import make_snapshot_dates
    from churnproj.modeling.model_eval_results_pipeline import build_snapshot_training_set
    import numpy as np

    from churnproj.modeling.shap_results import compute_shap_global_importance, save_shap_bar_plot
    from churnproj.modeling.stability_results import stability_topk_across_folds, save_stability_comparison_plot
    from churnproj.modeling.confusion_results import pick_threshold_for_target_recall, save_confusion_matrix_plot

    from churnproj.modeling.evaluate_results import evaluate_models_temporal_cv
    from churnproj.modeling.split import SnapshotTimeSplit
    from churnproj.modeling.final_model_results import train_final_xgb_and_threshold_on_last_fold

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Load train
    df_train_total = load_parquet(train_path)
    df_train_total = ensure_datetime(df_train_total, time_col="time")
    basic_schema_check(df_train_total)

    # Build snapshot dataset
    snapshot_dates = make_snapshot_dates(df_train_total, freq_days=7)

    train_df = build_snapshot_training_set(
        df_train_total,
        snapshot_dates=snapshot_dates,
        history_days=10,
        horizon_days=11,
    )

    print("Snapshot training set:", train_df.shape)
    print(train_df.head())

    feature_cols = infer_feature_cols(train_df)
    print("Inferred feature_cols:", feature_cols)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # A) Temporal CV comparison
    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000),
        "rf": lambda: RandomForestClassifier(n_estimators=400, random_state=args.seed),
        "xgb": lambda: XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
            tree_method="hist",
            random_state=args.seed,
        ),
    }

    splitter = SnapshotTimeSplit(train_df["snapshot_date"])
    results_df = evaluate_models_temporal_cv(
        train_df=train_df,
        feature_cols=feature_cols,
        models_dict=models,
        splitter=splitter,
        threshold=0.5,
    )

    results_path = tables_dir / "temporal_cv_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print("Saved:", results_path)

    # B) Final XGB + threshold + confusion matrix
    # B) Final XGB + threshold + confusion matrix
    final = train_final_xgb_and_threshold_on_last_fold(
        train_df=train_df,
        feature_cols=feature_cols,
        seed=args.seed,
        scoring="roc_auc",
        threshold_metric="f1",
    )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig 4: SHAP mean |SHAP| + elbow top10 ---
    shap_res = compute_shap_global_importance(
        model=final.model,
        X_background=final.X_train,  # background = last fold train
        X_explain=final.X_val,       # explain = last fold val
    )
    shap_res.shap_importance.to_csv(tables_dir / "shap_mean_abs.csv", index=False)
    save_shap_bar_plot(shap_res.shap_importance, fig_dir / "fig4_shap_mean_abs.png", top_n=20)

    reduced_cols = shap_res.top10_features
    pd.Series(reduced_cols, name="feature").to_csv(
        tables_dir / "selected_top10_features.csv", index=False
    )

    # --- Fig 5: Stability full vs reduced ---
    full_stab = stability_topk_across_folds(train_df, feature_cols, top_k=3, seed=args.seed)
    red_stab = stability_topk_across_folds(train_df, reduced_cols, top_k=3, seed=args.seed)
    save_stability_comparison_plot(full_stab, red_stab, fig_dir / "fig5_stability_comparison.png")

    # --- Fig 7: Confusion matrix with recall-prioritising threshold (report style) ---
    conf = pick_threshold_for_target_recall(
        final.y_val.to_numpy(), final.y_proba, target_recall=0.64
    )
    cm_report = np.array([[conf.tn, conf.fp], [conf.fn, conf.tp]])
    pd.DataFrame(cm_report).to_csv(tables_dir / "confusion_matrix_report_style.csv", index=False)
    pd.DataFrame([conf.__dict__]).to_csv(tables_dir / "confusion_threshold_report_style.csv", index=False)
    save_confusion_matrix_plot(cm_report, fig_dir / "fig7_confusion_matrix.png")
    print("Report-style confusion:", conf)

    # Also save the F1-optimized confusion matrix from your existing threshold picker
    pd.DataFrame(final.eval_last_fold.confusion).to_csv(
        tables_dir / "confusion_matrix_f1_opt.csv", index=False
    )
    pd.DataFrame([{"threshold": final.threshold.threshold, **final.threshold.metrics}]).to_csv(
        tables_dir / "threshold_metrics_f1_opt.csv", index=False
    )

    print("Best params:", final.gridsearch.best_params)
    print("Last fold date:", final.last_fold_date)
    print("Picked threshold (F1-opt):", final.threshold.threshold)
    print("Threshold metrics (F1-opt):", final.threshold.metrics)
    print("ROC-AUC (last fold):", final.eval_last_fold.roc_auc)

    print("Loaded train rows:", len(df_train_total))
    print("Output dir:", out_dir.resolve())


if __name__ == "__main__":
    main()

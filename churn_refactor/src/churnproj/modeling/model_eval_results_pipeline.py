from __future__ import annotations

from typing import Iterable, Sequence
import pandas as pd

from churnproj.data.snapshots import slice_history
from churnproj.data.snapshots import slice_horizon  # used if you don't have a label helper


def build_snapshot_training_set(
    df_events: pd.DataFrame,
    snapshot_dates: Sequence[pd.Timestamp],
    *,
    history_days: int,
    horizon_days: int,
    time_col: str = "time",
    user_col: str = "userId",
) -> pd.DataFrame:
    """
    Glue function: creates the snapshot-level training table:
      userId, snapshot_date, engineered features..., churn

    Uses:
      - slice_history / slice_horizon (data.snapshots)
      - your teammate's feature builder (features.build)
      - your teammate's label function (features.labels) if available
    """

    # --- imports from your teammate's modules (adjust names if needed) ---
    # feature engineering (should return a DataFrame indexed by userId)
    from churnproj.features.build import build_features_for_window

    # labeling: try to use teammate's label helper; fallback to Cancellation Confirmation rule
    try:
        from churnproj.features.labels import get_churn_users_in_horizon  # type: ignore
        has_label_helper = True
    except Exception:
        has_label_helper = False

    df_events = df_events.copy()
    df_events[time_col] = pd.to_datetime(df_events[time_col])

    all_snapshots: list[pd.DataFrame] = []

    for T in snapshot_dates:
        T = pd.to_datetime(T)

        # (T-history_days, T]
        df_hist = slice_history(df_events, T, history_days=history_days, time_col=time_col)

        feats = build_features_for_window(df_hist, T, history_days=history_days)

        # churn users in (T, T+horizon_days]
        if has_label_helper:
            churn_users = set(get_churn_users_in_horizon(df_events, T, horizon_days=horizon_days))
        else:
            df_future = slice_horizon(df_events, T, horizon_days=horizon_days, time_col=time_col)
            churn_users = set(
                df_future.loc[df_future["page"] == "Cancellation Confirmation", user_col]
            )

        feats["churn"] = feats.index.isin(churn_users).astype(int)
        feats["snapshot_date"] = T

        all_snapshots.append(feats)

    train_df = (
        pd.concat(all_snapshots)
        .reset_index()  # userId becomes column
        .sort_values(["snapshot_date", user_col])
        .reset_index(drop=True)
    )
    return train_df

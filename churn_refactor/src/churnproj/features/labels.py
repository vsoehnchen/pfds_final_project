from __future__ import annotations

from typing import Literal, Optional
import pandas as pd


def build_snapshot_user_index(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per user present in the history window."""
    users = pd.Index(df_hist["userId"].unique(), name="userId")
    return pd.DataFrame(index=users).reset_index()


def label_churn_inactivity(
    df_all: pd.DataFrame,
    snapshot_T: pd.Timestamp,
    horizon_days: int = 11,
    time_col: str = "time",
) -> pd.Series:
    """Churn=1 if user has *no* events in (T, T+horizon]."""
    snapshot_T = pd.to_datetime(snapshot_T)
    t = pd.to_datetime(df_all[time_col])
    horizon_mask = (t > snapshot_T) & (t <= snapshot_T + pd.Timedelta(days=horizon_days))
    active_in_horizon = df_all.loc[horizon_mask].groupby("userId").size()
    # If missing => no activity => churn
    all_users = pd.Index(df_all["userId"].unique())
    churn = (~all_users.isin(active_in_horizon.index)).astype(int)
    return pd.Series(churn, index=all_users, name="churn")


def label_churn_cancellation_event(
    df_all: pd.DataFrame,
    snapshot_T: pd.Timestamp,
    horizon_days: int = 11,
    cancellation_page_value: str = "Cancellation Confirmation",
    time_col: str = "time",
    page_col: str = "page",
) -> pd.Series:
    """Churn=1 if user has a cancellation event in (T, T+horizon]."""
    snapshot_T = pd.to_datetime(snapshot_T)
    t = pd.to_datetime(df_all[time_col])
    mask = (t > snapshot_T) & (t <= snapshot_T + pd.Timedelta(days=horizon_days)) & (df_all[page_col] == cancellation_page_value)
    canc = df_all.loc[mask].groupby("userId").size()
    all_users = pd.Index(df_all["userId"].unique())
    churn = all_users.isin(canc.index).astype(int)
    return pd.Series(churn, index=all_users, name="churn")
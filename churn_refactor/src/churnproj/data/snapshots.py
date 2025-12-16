from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import pandas as pd


def make_snapshot_dates(
    df: pd.DataFrame,
    time_col: str = "time",
    freq_days: int = 7,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> List[pd.Timestamp]:
    """Create snapshot dates on a fixed frequency between start and end (inclusive start, inclusive end if aligned)."""
    t = pd.to_datetime(df[time_col])
    if start is None:
        start = t.min().floor("D")
    else:
        start = pd.to_datetime(start).floor("D")
    if end is None:
        end = t.max().floor("D")
    else:
        end = pd.to_datetime(end).floor("D")

    # Snapshots are midnight timestamps
    dates = pd.date_range(start=start, end=end, freq=f"{freq_days}D")
    return list(pd.to_datetime(dates))


def slice_history(
    df: pd.DataFrame,
    T: pd.Timestamp,
    history_days: int,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return events in (T-history_days, T]."""
    T = pd.to_datetime(T)
    t = pd.to_datetime(df[time_col])
    mask = (t > T - pd.Timedelta(days=history_days)) & (t <= T)
    return df.loc[mask].copy()


def slice_horizon(
    df: pd.DataFrame,
    T: pd.Timestamp,
    horizon_days: int,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return events in (T, T+horizon_days]."""
    T = pd.to_datetime(T)
    t = pd.to_datetime(df[time_col])
    mask = (t > T) & (t <= T + pd.Timedelta(days=horizon_days))
    return df.loc[mask].copy()
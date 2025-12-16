from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional, Dict
import numpy as np
import pandas as pd


DEFAULT_PAGE_COUNTS = (
    "Thumbs Up",
    "Thumbs Down",
    "Add Friend",
    "Add to Playlist",
    "Downgrade",
    "Roll Advert",
    "Home",
    "Logout",
    "Settings",
)


def _count_page(df: pd.DataFrame, page_value: str) -> pd.Series:
    return (df["page"] == page_value).groupby(df["userId"]).sum()


def build_dual_window_features(
    df_hist_long: pd.DataFrame,
    snapshot_T: pd.Timestamp,
    window_long: int = 14,
    window_short: int = 7,
    pages_to_count: Sequence[str] = DEFAULT_PAGE_COUNTS,
) -> pd.DataFrame:
    """
    Build per-user features from a long history window (T-window_long, T].
    Also compute short-window features for the last `window_short` days and ratios short/long.

    Expected columns (best-effort):
      userId, time, page, artist, song, length, itemInSession
    """
    df = df_hist_long.copy()
    df["time"] = pd.to_datetime(df["time"])
    snapshot_T = pd.to_datetime(snapshot_T)

    df_short = df[df["time"] > snapshot_T - pd.Timedelta(days=window_short)].copy()

    def agg_basic(d: pd.DataFrame, suffix: str) -> pd.DataFrame:
        g = d.groupby("userId")
        out = pd.DataFrame(index=g.size().index)
        out[f"events_{suffix}"] = g.size()
        if "page" in d.columns:
            out[f"songs_{suffix}"] = (d["page"] == "NextSong").groupby(d["userId"]).sum()
        if "artist" in d.columns:
            out[f"unique_artists_{suffix}"] = g["artist"].nunique()
        if "song" in d.columns:
            out[f"unique_songs_{suffix}"] = g["song"].nunique()
        if "length" in d.columns:
            out[f"total_time_{suffix}"] = g["length"].sum()
        if "itemInSession" in d.columns:
            out[f"max_itemInSession_{suffix}"] = g["itemInSession"].max()
        # active days
        d2 = d.copy()
        d2["date"] = d2["time"].dt.floor("D")
        out[f"days_active_{suffix}"] = d2.groupby("userId")["date"].nunique()
        # page counts
        if "page" in d.columns:
            for p in pages_to_count:
                out[f"{p.lower().replace(' ', '_')}_{suffix}"] = _count_page(d, p)
        return out.fillna(0)

    long_feats = agg_basic(df, f"{window_long}d")
    short_feats = agg_basic(df_short, f"{window_short}d")

    feats = long_feats.join(short_feats, how="outer").fillna(0)

    # ratios (avoid division by zero)
    for col in feats.columns:
        if col.endswith(f"_{window_short}d"):
            base = col.replace(f"_{window_short}d", f"_{window_long}d")
            if base in feats.columns:
                feats[col.replace(f"_{window_short}d", f"_ratio_{window_short}d_over_{window_long}d")] = (
                    feats[col] / (feats[base] + 1e-9)
                )

    # recency: days since last event in the long window
    last_event = df.groupby("userId")["time"].max()
    feats["days_since_last_event"] = (snapshot_T - last_event).dt.total_seconds() / 86400.0

    feats = feats.reset_index()
    return feats


def build_simple_recency_windows(
    df_hist: pd.DataFrame,
    snapshot_T: pd.Timestamp,
    windows: Sequence[int] = (1, 3, 7, 10, 14),
) -> pd.DataFrame:
    """
    For each user and each window in `windows`, compute:
      - total events
      - NextSong events
      - active days
    Uses df_hist expected to cover at least max(windows) days.
    """
    df = df_hist.copy()
    df["time"] = pd.to_datetime(df["time"])
    snapshot_T = pd.to_datetime(snapshot_T)

    df["date"] = df["time"].dt.floor("D")
    df["rel_day"] = (df["date"].dt.normalize() - snapshot_T.normalize()).dt.days

    users = pd.Index(df["userId"].unique(), name="userId")
    feats = pd.DataFrame(index=users)

    for w in windows:
        mask = (df["rel_day"] >= -w + 1) & (df["rel_day"] <= 0)
        d = df.loc[mask]
        g = d.groupby("userId")

        feats[f"events_{w}d"] = g.size()
        feats[f"songs_{w}d"] = (d["page"] == "NextSong").groupby(d["userId"]).sum()
        feats[f"days_active_{w}d"] = g["date"].nunique()

    return feats.fillna(0).reset_index()
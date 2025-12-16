from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(Path(path))


def ensure_datetime(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Ensure `time_col` is a pandas datetime64 column."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out


def basic_schema_check(df: pd.DataFrame) -> None:
    """Raise an error if required columns are missing."""
    required = {"userId", "time", "page"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class SnapshotTimeSplit(BaseCrossValidator):
    """
    Time-aware CV splitter by snapshot_date blocks.

    For unique snapshot dates d1 < d2 < ... < dk:
      fold i trains on {d1..d_{i}} and validates on d_{i+1}.
    """

    def __init__(self, snapshot_series):
        self.snapshot_series = pd.to_datetime(snapshot_series)
        self.unique_dates = sorted(self.snapshot_series.unique())

    def get_n_splits(self, X=None, y=None, groups=None):
        return max(len(self.unique_dates) - 1, 0)

    def split(self, X, y=None, groups=None):
        dates = self.snapshot_series
        for i in range(1, len(self.unique_dates)):
            train_dates = set(self.unique_dates[:i])
            val_date = self.unique_dates[i]

            train_idx = np.where(dates.isin(train_dates))[0]
            val_idx = np.where(dates == val_date)[0]

            yield train_idx, val_idx
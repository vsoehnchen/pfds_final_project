from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class ChurnConfig:
    # Time logic
    history_days_long: int = 14
    history_days_short: int = 7
    horizon_days: int = 11
    snapshot_freq_days: int = 7

    # Feature windows for simple recency features
    window_sizes_days: Sequence[int] = (1, 3, 7, 10, 14)

    # Default churn labeling rule
    churn_rule: str = "inactivity"  # "inactivity" or "cancellation_event"
    cancellation_page_value: str = "Cancellation Confirmation"

    # Model defaults
    random_state: int = 42
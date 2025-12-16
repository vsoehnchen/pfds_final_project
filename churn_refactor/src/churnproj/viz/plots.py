from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: np.ndarray, ax: Optional[plt.Axes] = None, title: str = "Confusion matrix"):
    """Simple confusion matrix plot (no seaborn dependency)."""
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    return ax
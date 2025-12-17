from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShapGlobalResult:
    shap_importance: pd.DataFrame  # columns: feature, mean_abs_shap
    top10_features: list[str]


def compute_shap_global_importance(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> ShapGlobalResult:
    """
    Compute global SHAP importance as mean(|SHAP|) over X_explain.
    Intended to reproduce report Fig 4 (mean |SHAP| elbow at 10). :contentReference[oaicite:3]{index=3}
    """
    import shap

    explainer = shap.Explainer(model, X_background)
    sv = explainer(X_explain)

    vals = np.asarray(sv.values)  # (n, p)
    mean_abs = np.abs(vals).mean(axis=0)

    imp = (
        pd.DataFrame({"feature": list(X_explain.columns), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top10 = imp["feature"].head(10).tolist()
    return ShapGlobalResult(shap_importance=imp, top10_features=top10)


def save_shap_bar_plot(
    shap_importance: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int = 20,
    title: str = "Global Feature Importance (mean |SHAP|)",
) -> None:
    top = shap_importance.head(top_n).iloc[::-1]  # reverse for barh
    plt.figure(figsize=(10, 6), dpi=160)
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean(|SHAP|)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

"""Permutation importance, SHAP values, per-alert explanations and partial dependence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from .evaluation import pr_auc

try:  # SHAP is optional at import time; permutation importance does not need it
    import shap
    HAVE_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAVE_SHAP = False

IMP_FEATURE = "feature"
IMP_MEAN = "mean_drop"
IMP_STD = "std_drop"


def permutation_importance(detector, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10, seed: int = 0,
                           metric=pr_auc) -> pd.DataFrame:
    """Mean drop in `metric` when one feature column is shuffled (breaks time order and link to y)."""
    rng = np.random.default_rng(seed)
    base = metric(y, detector.score(X))
    rows = []
    for col in detector.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - metric(y, detector.score(Xp)))
        rows.append({IMP_FEATURE: col, IMP_MEAN: float(np.mean(drops)), IMP_STD: float(np.std(drops))})
    return pd.DataFrame(rows).sort_values(IMP_MEAN, ascending=False).reset_index(drop=True)


def shap_values(detector, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """TreeExplainer values in log-odds for a fitted tree detector; returns (values, base value)."""
    if not HAVE_SHAP:
        raise RuntimeError("shap is not installed; use permutation_importance instead")
    explainer = shap.TreeExplainer(detector.model)
    vals = explainer.shap_values(X[list(detector.columns)])
    if isinstance(vals, list):  # older shap: one array per class
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:  # newer shap: (n, p, classes)
        vals = vals[..., -1]
    base = np.asarray(explainer.expected_value).ravel()[-1]
    return vals, float(base)


def explain_days(values: np.ndarray, X: pd.DataFrame, columns: tuple[str, ...], days: pd.DatetimeIndex,
                 top_k: int = 3) -> pd.DataFrame:
    """Top contributing features per day (signed log-odds), as text and as a ranked list."""
    pos = X.index.get_indexer(days)
    if (pos < 0).any():
        raise KeyError("some requested days are not in X")
    rows = []
    for d, p in zip(days, pos):
        order = np.argsort(-np.abs(values[p]))[:top_k]
        parts = [f"{columns[j]}={X.iloc[p][columns[j]]:.2f} ({values[p, j]:+.2f})" for j in order]
        rows.append({"date": d, "top_features": [columns[j] for j in order], "explanation": "; ".join(parts)})
    return pd.DataFrame(rows)


def mean_abs_shap(values: np.ndarray, columns: tuple[str, ...]) -> pd.Series:
    return pd.Series(np.abs(values).mean(axis=0), index=list(columns)).sort_values(ascending=False)


def pd_curves(detector, X: pd.DataFrame, features: list[str], grid_resolution: int = 25) -> dict[str, pd.DataFrame]:
    """One-way partial dependence of P(stress) on each feature."""
    out = {}
    for f in features:
        res = partial_dependence(detector.model, X[list(detector.columns)], [f], grid_resolution=grid_resolution,
                                 response_method="predict_proba", kind="average", method="brute")
        out[f] = pd.DataFrame({"value": res["grid_values"][0], "pd": res["average"][0]})
    return out

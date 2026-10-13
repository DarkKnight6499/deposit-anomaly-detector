"""Model-risk stability checks: population stability index (PSI) on features and scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_OK = 0.10
PSI_ACTION = 0.25
EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, edges: np.ndarray | None = None) -> float:
    """PSI with reference-quantile bin edges (open-ended outer bins); EPS floors empty bins.

    Pass explicit `edges` (e.g. equal-width on [0, 1]) for bimodal probability scores, where
    quantile bins collapse onto a few repeated values."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    if edges is None:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.maximum(r, EPS), np.maximum(c, EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    return "OK" if value < PSI_OK else ("WATCH" if value < PSI_ACTION else "ACTION")


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(reference[c].to_numpy(), current[c].to_numpy(), bins)} for c in reference.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(psi_status)
    return out


def psi_by_period(reference: pd.DataFrame, current: pd.DataFrame, freq: str = "QE") -> pd.DataFrame:
    """Feature x period PSI matrix (periods taken from current's DatetimeIndex)."""
    cols = {}
    for period, chunk in current.groupby(current.index.to_period(freq[0])):
        if len(chunk) >= 20:
            cols[str(period)] = {c: psi(reference[c].to_numpy(), chunk[c].to_numpy()) for c in reference.columns}
    return pd.DataFrame(cols)


def probability_edges(bins: int = 10) -> np.ndarray:
    return np.linspace(0.0, 1.0, bins + 1)[1:-1]


def score_drift(reference_scores: np.ndarray, scores: pd.Series, freq: str = "Q",
                edges: np.ndarray | None = None) -> pd.DataFrame:
    """Per-period mean score and PSI against reference scores (pass probability_edges() for probabilities)."""
    rows = []
    for period, chunk in scores.groupby(scores.index.to_period(freq)):
        if len(chunk) >= 20:
            rows.append({"period": str(period), "n": len(chunk), "mean_score": float(chunk.mean()),
                         "psi_vs_reference": psi(reference_scores, chunk.to_numpy(), edges=edges)})
    return pd.DataFrame(rows)

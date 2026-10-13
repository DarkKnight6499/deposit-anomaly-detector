"""Expanding-window time-series CV with a purge gap, plus a final held-out period."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray  # positional indices
    val_idx: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def expanding_window_folds(index: pd.DatetimeIndex, boundaries: list[str], purge_days: int,
                           end_exclusive: str | None = None) -> list[Fold]:
    """Fold i validates on [boundaries[i], boundaries[i+1]); the last block ends at end_exclusive.

    Training uses every day strictly before (val_start - purge_days), so no training row's
    rolling window (length <= purge_days) overlaps a validation row's window."""
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative")
    bounds = [pd.Timestamp(b) for b in boundaries]
    if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
        raise ValueError("boundaries must be strictly increasing")
    stop = pd.Timestamp(end_exclusive) if end_exclusive else index.max() + pd.Timedelta(days=1)
    edges = bounds + [stop]
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cutoff = lo - pd.Timedelta(days=purge_days)
        train = np.flatnonzero(index < cutoff)
        val = np.flatnonzero((index >= lo) & (index < hi))
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"empty train or validation block for validation start {lo.date()}")
        folds.append(Fold(train, val, index[train[-1]], index[val[0]], index[val[-1]]))
    return folds


def holdout_split(index: pd.DatetimeIndex, holdout_start: str, purge_days: int) -> Fold:
    """Train on everything before (holdout_start - purge_days); test on [holdout_start, end]."""
    lo = pd.Timestamp(holdout_start)
    train = np.flatnonzero(index < lo - pd.Timedelta(days=purge_days))
    test = np.flatnonzero(index >= lo)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("empty train or test block")
    return Fold(train, test, index[train[-1]], index[test[0]], index[test[-1]])

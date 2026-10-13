"""Data-quality router: flags days whose operational indicators say the flows cannot be trusted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import columns as C
from .features import MAD_TO_SD

DQ_REASON_RECON = "reconciliation_break"
DQ_REASON_QUEUE = "payment_queue_backlog"


class DataQualityRouter:
    """Robust upper limits (median + k * MAD-sd) on the reconciliation gap and payment queue."""

    def __init__(self, k: float = 10.0):
        self.k = k
        self.limits: dict[str, float] = {}

    def fit(self, X: pd.DataFrame) -> "DataQualityRouter":
        for col in (C.RECON_GAP, C.PAYMENT_QUEUE):
            v = X[col].to_numpy()
            med = np.median(v)
            sd = MAD_TO_SD * np.median(np.abs(v - med))
            self.limits[col] = float(med + self.k * max(sd, 1e-6))
        return self

    def reasons(self, X: pd.DataFrame) -> pd.Series:
        recon = X[C.RECON_GAP] > self.limits[C.RECON_GAP]
        queue = X[C.PAYMENT_QUEUE] > self.limits[C.PAYMENT_QUEUE]
        out = pd.Series("", index=X.index)
        out[recon] = DQ_REASON_RECON
        out[queue & ~recon] = DQ_REASON_QUEUE
        return out

    def flag(self, X: pd.DataFrame) -> pd.Series:
        return self.reasons(X) != ""

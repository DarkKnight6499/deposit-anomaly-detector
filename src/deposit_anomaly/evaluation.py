"""Event-level and day-level evaluation, cost-sensitive thresholds, calibration, intervals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from . import columns as C

GRACE_DAYS = 5        # alerts up to this many days after a liquidity window are not false alarms
FA_BLOCK_DAYS = 7     # a false alarm is a 7-day block containing at least one unjustified alert day
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class CostConfig:
    """Assumed costs (stylized): a missed liquidity event is 100x a false-alarm week."""
    miss: float = 100.0
    false_alarm: float = 1.0


class EventScorer:
    """Scores a boolean alert series against labelled events on a fixed date index."""

    def __init__(self, index: pd.DatetimeIndex, events: pd.DataFrame, grace: int = GRACE_DAYS,
                 block_days: int = FA_BLOCK_DAYS):
        self.index = pd.DatetimeIndex(index)
        n = len(self.index)
        lo, hi = self.index[0], self.index[-1]
        ev = events[(events[C.EV_END] >= lo) & (events[C.EV_START] <= hi)]
        full = ev[(ev[C.EV_START] >= lo) & (ev[C.EV_END] <= hi)].reset_index(drop=True)
        self.events = full  # events wholly inside the evaluation index
        self.start_pos = self.index.searchsorted(full[C.EV_START])
        self.end_pos = self.index.searchsorted(full[C.EV_END], side="right")
        self.is_liq = full[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES).to_numpy()
        self.justified = np.zeros(n, bool)
        self.dq_days = np.zeros(n, bool)
        for r in ev.itertuples():
            s = self.index.searchsorted(getattr(r, C.EV_START))
            e = self.index.searchsorted(getattr(r, C.EV_END), side="right")
            if getattr(r, C.EVENT_TYPE) in C.LIQUIDITY_TYPES:
                self.justified[s:min(n, e + grace)] = True
            else:
                self.dq_days[s:e] = True
        self.block = ((self.index - lo).days // block_days).to_numpy()
        self.years = n / DAYS_PER_YEAR

    def evaluate(self, alerts: np.ndarray) -> dict:
        a = np.asarray(alerts, bool)
        rows = []
        for i, r in enumerate(self.events.itertuples()):
            s, e = self.start_pos[i], self.end_pos[i]
            hit = np.flatnonzero(a[s:e])
            first = self.index[s + hit[0]] if len(hit) else pd.NaT
            lead = (getattr(r, C.EV_CRITICAL) - first).days if len(hit) else np.nan
            rows.append({"event_id": r.event_id, C.EVENT_TYPE: getattr(r, C.EVENT_TYPE),
                         "detected": bool(len(hit)), "first_alert": first, "lead_days": lead,
                         "alert_days": int(len(hit))})
        per_event = pd.DataFrame(rows, columns=["event_id", C.EVENT_TYPE, "detected", "first_alert",
                                                "lead_days", "alert_days"])
        false_days = a & ~self.justified
        fa_weeks = len(np.unique(self.block[false_days]))
        fa_weeks_dq = len(np.unique(self.block[false_days & self.dq_days]))
        liq = per_event[per_event[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES)]
        return {"per_event": per_event, "missed": int((~liq["detected"]).sum()),
                "n_liquidity_events": int(len(liq)), "fa_weeks": int(fa_weeks),
                "fa_weeks_dq_windows": int(fa_weeks_dq), "fa_per_year": fa_weeks / self.years,
                "years": self.years}

    def cost(self, alerts: np.ndarray, costs: CostConfig) -> float:
        ev = self.evaluate(alerts)
        return costs.miss * ev["missed"] + costs.false_alarm * ev["fa_weeks"]


def select_threshold(scores: pd.Series, scorer: EventScorer, costs: CostConfig = CostConfig(),
                     grid: int = 200) -> tuple[float, float]:
    """Cost-minimising threshold on a score grid.

    Many thresholds often tie at the minimum cost (a plateau), so the midpoint of the lowest and
    highest minimisers is used rather than either knife-edge. Both ends are non-increasing in the
    miss cost, hence so is the midpoint. Returns (threshold, cost at that threshold)."""
    s = np.asarray(scores, float)
    cands = np.unique(np.quantile(s, np.linspace(0.0, 1.0, grid)))
    cands = np.append(cands, np.nextafter(s.max(), np.inf))
    cost = np.array([scorer.cost(s >= t, costs) for t in cands])
    best = cands[cost == cost.min()]
    thr = float((best.min() + best.max()) / 2.0)
    return thr, float(scorer.cost(s >= thr, costs))


def day_metrics(y: np.ndarray, score: np.ndarray, alerts: np.ndarray) -> dict:
    y, a = np.asarray(y, int), np.asarray(alerts, bool)
    tp = int((a & (y == 1)).sum())
    return {"precision": tp / a.sum() if a.sum() else np.nan,
            "recall": tp / (y == 1).sum() if (y == 1).sum() else np.nan,
            "pr_auc": pr_auc(y, score), "alert_days": int(a.sum())}


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    return float(average_precision_score(y, score)) if 0 < y.sum() < len(y) else np.nan


# ---- calibration ------------------------------------------------------------------
class PlattCalibrator:
    """Logistic map from a raw score to a probability, fitted on out-of-fold scores."""

    def __init__(self):
        self.model = LogisticRegression(C=1e6, max_iter=1000)
        self.mu = 0.0
        self.sd = 1.0

    def fit(self, score: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        s = np.asarray(score, float)
        self.mu, self.sd = s.mean(), s.std() or 1.0
        self.model.fit(((s - self.mu) / self.sd).reshape(-1, 1), np.asarray(y, int))
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        s = (np.asarray(score, float) - self.mu) / self.sd
        return self.model.predict_proba(s.reshape(-1, 1))[:, 1]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": k, "n": int((b == k).sum()), "mean_pred": p[b == k].mean(), "frac_pos": y[b == k].mean()}
            for k in range(bins) if (b == k).any()]
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = reliability_table(y, p, bins)
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


# ---- intervals ----------------------------------------------------------------------
def block_bootstrap_ci(y: np.ndarray, score: np.ndarray, metric=pr_auc, block: int = 10,
                       n_boot: int = 500, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap over days (keeps serial dependence inside blocks)."""
    y, s = np.asarray(y), np.asarray(score)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for a detection rate k/n."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else stats.beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


def poisson_rate_ci(count: int, exposure: float, level: float = 0.95) -> tuple[float, float]:
    """Exact Poisson interval for a rate (count per unit exposure)."""
    a = (1 - level) / 2
    lo = 0.0 if count == 0 else stats.chi2.ppf(a, 2 * count) / 2
    hi = stats.chi2.ppf(1 - a, 2 * (count + 1)) / 2
    return float(lo / exposure), float(hi / exposure)

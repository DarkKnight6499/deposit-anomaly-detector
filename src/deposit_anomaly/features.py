"""Feature engineering. Every feature at day t uses data observed up to and including day t
(end-of-day scoring); nothing from t+1 or later. Expanding/rolling statistics are shifted so
a day never enters its own baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import calendar as cal
from . import columns as C

MAD_TO_SD = 1.4826
SIGMA_FLOOR = 1e-3  # percent of balance; avoids dividing by ~0 on calm stretches

# Column-name builders (features are always addressed through these names).
TOTAL_OUT = "total_outflow_pct"
Z_TOTAL = "z_total"
CUM5_Z = "cum5_z"
BREADTH_Z2 = "breadth_z2"
BREADTH_Z1 = "breadth_z1"
MAX_SEG_Z = "max_segment_z"
VELOCITY = "outflow_velocity"
ACCEL = "outflow_acceleration"
TOP10 = "top10_share"
TOP10_Z = "top10_share_z"
CDS_CHG5 = "bank_cds_chg5"
CDS_Z = "bank_cds_z"
MKT_CDS_CHG5 = "market_cds_chg5"
BANK_DD = "bank_equity_drawdown_pct"
MKT_DD = "market_equity_drawdown_pct"
RATE_CHG5 = "rate_2y_chg5"
SPREAD_CHG10 = "rate_spread_chg10"


def out_col(seg: str) -> str:
    return f"out_pct_{seg}"


def z_col(seg: str) -> str:
    return f"z_{seg}"


def lag_col(base: str, k: int) -> str:
    return f"{base}_lag{k}"


def cum_col(k: int) -> str:
    return f"cum_out_{k}d"


@dataclass(frozen=True)
class FeatureConfig:
    refit_days: int = 21         # seasonal calendar model refit cadence
    min_fit_rows: int = 120
    trim_mads: float = 5.0       # residuals beyond this many robust sds are dropped before the final fit
    sigma_window: int = 90       # the longest rolling window: defines the CV purge gap
    sigma_min: int = 40
    cum_windows: tuple[int, ...] = (5, 10, 20)
    lags: tuple[int, ...] = (1, 2, 5)
    market_window: int = 60


DEFAULT_CONFIG = FeatureConfig()
LOOKBACK_DAYS = DEFAULT_CONFIG.sigma_window


def _mad_sd(x: np.ndarray) -> float:
    return MAD_TO_SD * float(np.nanmedian(np.abs(x - np.nanmedian(x))))


def _design(calf: pd.DataFrame) -> np.ndarray:
    """Calendar design matrix: 5 business-weekday dummies, weekend, holiday, then the special flags."""
    dow = calf.index.dayofweek.to_numpy()
    bus = calf[cal.IS_BUSINESS].to_numpy()
    cols = [(bus & (dow == k)) for k in range(5)]
    cols += [calf[cal.IS_WEEKEND].to_numpy(), calf[cal.IS_HOLIDAY].to_numpy()]
    cols += [calf[f].to_numpy() for f in cal.SPECIAL_FLAGS]
    return np.column_stack(cols).astype(float)


def _robust_ols(D: np.ndarray, y: np.ndarray, trim: float) -> np.ndarray:
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    res = y - D @ beta
    cut = trim * MAD_TO_SD * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res) <= max(cut, 1e-12)
    return np.linalg.lstsq(D[keep], y[keep], rcond=None)[0] if keep.sum() > D.shape[1] else beta


def seasonal_expected(series: pd.Series, calf: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    """Expected outflow from calendar effects, refit every `refit_days` on PAST rows only.

    Rows before the first fit (min_fit_rows) are NaN. A row in [r, r + refit_days) uses a fit on
    rows < r, so no row ever sees its own or any later value."""
    D, y = _design(calf), series.to_numpy(dtype=float)
    n = len(y)
    exp = np.full(n, np.nan)
    for r in range(cfg.min_fit_rows, n, cfg.refit_days):
        ok = ~np.isnan(y[:r])
        beta = _robust_ols(D[:r][ok], y[:r][ok], cfg.trim_mads)
        exp[r:r + cfg.refit_days] = D[r:r + cfg.refit_days] @ beta
    return pd.Series(exp, index=series.index)


def seasonal_z(series: pd.Series, calf: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    """Seasonality-adjusted residual z-score, positive = more outflow than usual.

    Scale is a trailing robust sd (MAD) of past residuals."""
    resid = series - seasonal_expected(series, calf, cfg)
    sigma = resid.shift(1).rolling(cfg.sigma_window, min_periods=cfg.sigma_min).apply(_mad_sd, raw=True)
    return resid / sigma.clip(lower=SIGMA_FLOOR)


def _robust_z(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    past = x.shift(1)
    med = past.rolling(window, min_periods=min_periods).median()
    sd = past.rolling(window, min_periods=min_periods).apply(_mad_sd, raw=True)
    return (x - med) / sd.clip(lower=1e-9)


def build_features(observed: pd.DataFrame, cfg: FeatureConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return the feature frame (rows with an incomplete warm-up window keep NaN)."""
    idx = observed.index
    calf = cal.calendar_frame(idx)
    feats = pd.DataFrame(index=idx)

    total_bal_prev = observed[list(C.BALANCE_COLUMNS)].sum(axis=1).shift(1)
    total_flow = observed[list(C.FLOW_COLUMNS)].sum(axis=1)
    feats[TOTAL_OUT] = -total_flow / total_bal_prev * 100.0
    for seg in C.SEGMENTS:
        feats[out_col(seg)] = -observed[C.flow_col(seg)] / observed[C.balance_col(seg)].shift(1) * 100.0

    feats[Z_TOTAL] = seasonal_z(feats[TOTAL_OUT], calf, cfg)
    for seg in C.SEGMENTS:
        feats[z_col(seg)] = seasonal_z(feats[out_col(seg)], calf, cfg)
    for k in cfg.lags:
        feats[lag_col(TOTAL_OUT, k)] = feats[TOTAL_OUT].shift(k)
        feats[lag_col(Z_TOTAL, k)] = feats[Z_TOTAL].shift(k)

    for w in cfg.cum_windows:
        feats[cum_col(w)] = feats[TOTAL_OUT].rolling(w, min_periods=w).sum()
    cum5 = feats[cum_col(5)]
    feats[CUM5_Z] = _robust_z(cum5, cfg.sigma_window, cfg.sigma_min)

    seg_z = feats[[z_col(s) for s in C.SEGMENTS]]
    feats[MAX_SEG_Z] = seg_z.max(axis=1, skipna=False)
    feats[BREADTH_Z2] = (seg_z > 2.0).sum(axis=1) / len(C.SEGMENTS)
    feats[BREADTH_Z1] = (seg_z > 1.0).sum(axis=1) / len(C.SEGMENTS)

    mean5 = feats[TOTAL_OUT].rolling(5, min_periods=5).mean()
    feats[VELOCITY] = mean5 - mean5.shift(3)
    feats[ACCEL] = feats[VELOCITY] - feats[VELOCITY].shift(3)

    feats[TOP10] = observed[C.TOP10_SHARE]
    feats[TOP10_Z] = _robust_z(observed[C.TOP10_SHARE], cfg.market_window, 30)
    feats[C.RECON_GAP] = observed[C.RECON_GAP]
    feats[C.PAYMENT_QUEUE] = observed[C.PAYMENT_QUEUE]

    feats[C.BANK_CDS] = observed[C.BANK_CDS]
    feats[CDS_CHG5] = observed[C.BANK_CDS].diff(5)
    feats[CDS_Z] = _robust_z(np.log(observed[C.BANK_CDS]), cfg.sigma_window, cfg.sigma_min)
    feats[MKT_CDS_CHG5] = observed[C.MARKET_CDS].diff(5)
    for col, name in ((C.BANK_EQUITY, BANK_DD), (C.MARKET_EQUITY, MKT_DD)):
        peak = observed[col].rolling(cfg.market_window, min_periods=1).max()
        feats[name] = (observed[col] / peak - 1.0) * 100.0
    feats[RATE_CHG5] = observed[C.RATE_2Y].diff(5)
    feats[C.RATE_SPREAD] = observed[C.RATE_SPREAD]
    feats[SPREAD_CHG10] = observed[C.RATE_SPREAD].diff(10)
    return feats




def feature_columns(cfg: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    """Feature names in a stable order, derived from the builder (single source of truth)."""
    cols = [TOTAL_OUT] + [out_col(s) for s in C.SEGMENTS] + [Z_TOTAL] + [z_col(s) for s in C.SEGMENTS]
    for k in cfg.lags:
        cols += [lag_col(TOTAL_OUT, k), lag_col(Z_TOTAL, k)]
    cols += [cum_col(w) for w in cfg.cum_windows] + [CUM5_Z, MAX_SEG_Z, BREADTH_Z2, BREADTH_Z1, VELOCITY, ACCEL,
                                                     TOP10, TOP10_Z, C.RECON_GAP, C.PAYMENT_QUEUE,
                                                     C.BANK_CDS, CDS_CHG5, CDS_Z, MKT_CDS_CHG5, BANK_DD, MKT_DD,
                                                     RATE_CHG5, C.RATE_SPREAD, SPREAD_CHG10]
    return cols


FEATURE_COLUMNS = tuple(feature_columns())

# Subsets used by the transparent baselines.
RULE_COLUMNS = (TOTAL_OUT,) + tuple(out_col(s) for s in C.SEGMENTS)
STAT_COLUMNS = (Z_TOTAL, CUM5_Z) + tuple(z_col(s) for s in C.SEGMENTS)
MARKET_COLUMNS = (C.BANK_CDS, CDS_CHG5, CDS_Z, MKT_CDS_CHG5, BANK_DD, MKT_DD, RATE_CHG5, C.RATE_SPREAD, SPREAD_CHG10)
DQ_COLUMNS = (C.RECON_GAP, C.PAYMENT_QUEUE)


def model_frame(observed: pd.DataFrame, analysis_start: str = "2020-01-01") -> pd.DataFrame:
    """Features restricted to the analysis window; raises if any NaN remains after warm-up."""
    feats = build_features(observed)[list(FEATURE_COLUMNS)]
    out = feats.loc[analysis_start:]
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaN features after warm-up: {bad}")
    return out

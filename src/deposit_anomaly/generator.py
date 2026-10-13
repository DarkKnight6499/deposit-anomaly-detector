"""Seeded synthetic daily deposit generator with injected, labelled stress events.

Independent of the detector: it only produces an observed frame (what a bank would
see), a label frame and an events table. See docs/methodology.md for label rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import columns as C
from . import calendar as cal

N_SEG = len(C.SEGMENTS)
SEG_IX = {s: i for i, s in enumerate(C.SEGMENTS)}

INIT_BALANCE_M = {
    "retail_insured": 18000.0, "retail_uninsured": 6000.0, "small_business": 5000.0,
    "corp_operational": 8000.0, "corp_nonoperational": 7000.0, "brokered": 4000.0,
}
ANNUAL_GROWTH = {
    "retail_insured": 0.04, "retail_uninsured": 0.02, "small_business": 0.03,
    "corp_operational": 0.03, "corp_nonoperational": 0.01, "brokered": 0.0,
}
# Daily noise sd of net flow, in percent of prior balance.
NOISE_SD_PCT = {
    "retail_insured": 0.05, "retail_uninsured": 0.12, "small_business": 0.15,
    "corp_operational": 0.20, "corp_nonoperational": 0.35, "brokered": 0.30,
}
# Gross payment turnover per day as a share of balance (sizes the outage backlog).
TURNOVER = {"small_business": 0.025, "corp_operational": 0.03, "corp_nonoperational": 0.02}
OUTAGE_SEGMENTS = tuple(TURNOVER)
COMMON_LOADING = 0.4
WEEKEND_NOISE_SCALE = 0.15
HOLIDAY_NOISE_SCALE = 0.25
MAX_DAILY_DROP = 0.6
# Mean reversion of each segment towards its growth trend (deposits slowly rebuild after stress).
REVERSION_RATE = 0.004
REVERSION_CAP = 0.01
AFTER_PAYROLL = cal.AFTER_PAYROLL

# (calendar flag, segment, effect in percent of prior balance on that day)
SEASONAL_EFFECTS = (
    (cal.PAYROLL, "corp_operational", -0.40), (cal.PAYROLL, "small_business", -0.30),
    (AFTER_PAYROLL, "corp_operational", 0.20),
    (cal.MONTH_END, "corp_operational", -0.30), (cal.MONTH_END, "brokered", -0.20),
    (cal.MONTH_END, "corp_nonoperational", -0.15),
    (cal.FIRST_BUSINESS, "corp_operational", 0.25), (cal.FIRST_BUSINESS, "brokered", 0.15),
    (cal.FIRST_BUSINESS, "corp_nonoperational", 0.10),
    (cal.TAX_DAY, "retail_uninsured", -0.35), (cal.TAX_DAY, "retail_insured", -0.08),
    (cal.TAX_DAY, "corp_operational", -0.35), (cal.TAX_DAY, "small_business", -0.30),
    (cal.SB_PAYDAY, "small_business", -0.25),
    (cal.HOLIDAY_SEASON, "retail_insured", -0.02), (cal.HOLIDAY_SEASON, "retail_uninsured", -0.03),
)

# Nominal start dates; each is jittered by +-2 days with the seeded generator.
EVENT_SCHEDULE: tuple[tuple[str, str], ...] = (
    (C.FLIGHT, "2020-03-09"), (C.RUN, "2020-09-08"), (C.BLEED, "2021-01-11"),
    (C.RUN, "2021-05-10"), (C.RUN, "2021-11-15"), (C.RUN, "2022-05-09"),
    (C.FLIGHT, "2022-09-12"), (C.BLEED, "2022-10-24"), (C.RUN, "2023-03-13"),
    (C.BLEED, "2023-05-08"), (C.FLIGHT, "2023-10-02"),
    (C.BLEED, "2024-01-15"), (C.RUN, "2024-04-15"), (C.FLIGHT, "2024-08-05"),
    (C.RUN, "2024-10-14"),
    (C.GLITCH, "2020-06-03"), (C.GLITCH, "2020-12-14"), (C.GLITCH, "2021-04-06"),
    (C.GLITCH, "2021-09-08"), (C.GLITCH, "2022-01-26"), (C.GLITCH, "2022-07-19"),
    (C.GLITCH, "2023-02-06"), (C.GLITCH, "2023-08-22"), (C.GLITCH, "2024-06-12"),
    (C.GLITCH, "2024-09-18"), (C.GLITCH, "2024-12-05"),
    (C.OUTAGE, "2020-07-21"), (C.OUTAGE, "2021-08-10"), (C.OUTAGE, "2022-03-15"),
    (C.OUTAGE, "2023-01-17"), (C.OUTAGE, "2023-07-25"), (C.OUTAGE, "2024-05-21"),
    (C.OUTAGE, "2024-11-12"),
)

# Benign market noise with NO outflow effect and NO label: (kind, nominal start, length in days).
NUISANCE_SCHEDULE: tuple[tuple[str, str, int], ...] = (
    ("spread_bump", "2020-10-26", 40), ("spread_bump", "2021-06-14", 40), ("spread_bump", "2023-11-01", 40),
    ("spread_bump", "2024-06-24", 35),
    ("cds_bump", "2020-05-04", 12), ("cds_bump", "2021-10-04", 12), ("cds_bump", "2023-04-17", 12),
    ("cds_bump", "2024-07-01", 12),
)


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20261012
    start: str = "2019-06-01"
    end: str = "2024-12-31"
    schedule: tuple = field(default=EVENT_SCHEDULE)
    nuisance: tuple = field(default=NUISANCE_SCHEDULE)
    jitter_days: int = 2


@dataclass
class SyntheticDeposits:
    observed: pd.DataFrame
    labels: pd.DataFrame
    events: pd.DataFrame
    config: GeneratorConfig
    nuisance: pd.DataFrame


def _profile(d: int, ramp: int = 3, tail: int = 3) -> np.ndarray:
    k = np.arange(d)
    p = np.minimum(1.0, (k + 1) / ramp)
    decay = np.where(k >= d - tail, 0.5 ** (k - (d - tail) + 1), 1.0)
    return p * decay


def _shape(n: int, s: int, e: int, ramp: int, half_life: float) -> np.ndarray:
    """Ramp up over [s, e], then exponential decay after e (a lingering market effect)."""
    out = np.zeros(n)
    t = np.arange(s, min(e, n - 1) + 1)
    out[t] = np.minimum(1.0, (t - s + 1) / ramp)
    after = np.arange(e + 1, n)
    out[after] = np.exp(-np.log(2) * (after - e) / half_life)
    return out


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _next_business(is_bus: np.ndarray, i: int) -> int:
    while not is_bus[i]:
        i += 1
    return i


def _block(act: int, amp: dict, prof: np.ndarray) -> np.ndarray:
    blk = np.zeros((act, N_SEG))
    for sg, a in amp.items():
        blk[:, SEG_IX[sg]] = a * prof / 100.0
    return blk


def generate(config: GeneratorConfig | None = None) -> SyntheticDeposits:
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, cfg.end, freq="D")
    n = len(dates)
    calf = cal.calendar_frame(dates)
    is_bus = calf[cal.IS_BUSINESS].to_numpy()

    # ---- base flows (fractions of prior balance) ---------------------------------
    scale = np.where(calf[cal.IS_WEEKEND], WEEKEND_NOISE_SCALE,
                     np.where(calf[cal.IS_HOLIDAY], HOLIDAY_NOISE_SCALE, 1.0))
    t5 = np.sqrt(3.0 / 5.0)  # unit-variance Student t(5)
    common = rng.standard_t(5, n) * t5
    f = np.zeros((n, N_SEG))
    for s, i in SEG_IX.items():
        e = rng.standard_t(5, n) * t5
        shock = COMMON_LOADING * common + np.sqrt(1 - COMMON_LOADING**2) * e
        f[:, i] = ANNUAL_GROWTH[s] / 365.0 + scale * NOISE_SD_PCT[s] / 100.0 * shock
    flags = {k: calf[k].to_numpy() for k in calf.columns if calf[k].dtype == bool}
    for flag, seg, pct in SEASONAL_EFFECTS:
        f[flags[flag], SEG_IX[seg]] += pct / 100.0

    # ---- events ---------------------------------------------------------------------
    extra = np.zeros((n, N_SEG))
    glitch = np.zeros((n, N_SEG))
    glitch_rev_seg = np.full(n, -1)  # day -> segment whose spurious record is reversed that day
    d_bank_cds, d_mkt_cds = np.zeros(n), np.zeros(n)
    r_bank, r_mkt = np.zeros(n), np.zeros(n)
    d_rate, d_spread = np.zeros(n), np.zeros(n)
    top10_level = np.zeros(n)
    outage_delay, outage_release = np.zeros(n, bool), np.zeros(n, bool)
    rows = []
    ev_extra: dict[int, tuple[int, np.ndarray]] = {}
    occupied = np.zeros(n, bool)

    sched = sorted(((pd.Timestamp(d), t) for t, d in cfg.schedule), key=lambda x: x[0])
    for eid, (nominal, etype) in enumerate(sched):
        jit = int(rng.integers(-cfg.jitter_days, cfg.jitter_days + 1))
        s = int((nominal - dates[0]).days) + jit
        if etype in C.DQ_TYPES:
            s = _next_business(is_bus, s)
        crit_block = None
        if etype == C.RUN:
            pre, act = 3, int(rng.integers(10, 15))
            e = s + pre + act - 1
            prof = _profile(act)
            amp = {"corp_nonoperational": rng.uniform(1.6, 2.8), "retail_uninsured": rng.uniform(0.7, 1.4),
                   "brokered": 0.4, "corp_operational": 0.25, "small_business": 0.2, "retail_insured": 0.03}
            blk = _block(act, amp, prof)
            extra[s + pre:s + pre + act] += blk
            crit_block = (s + pre, blk)
            shp = _shape(n, s, e, pre + 3, 6.0)
            d_bank_cds += rng.uniform(150, 300) * shp
            d_mkt_cds += rng.uniform(5, 15) * shp
            w = shp[s:e + 1]
            r_bank[s:e + 1] += -rng.uniform(0.18, 0.32) * w / w.sum()
            sl = slice(s + pre, s + pre + act)
            top10_level[sl] = np.maximum(top10_level[sl], 0.30 + 0.40 * prof)
        elif etype == C.FLIGHT:
            pre, act = 2, int(rng.integers(12, 21))
            e = s + pre + act - 1
            prof = _profile(act)
            amp = {"corp_nonoperational": rng.uniform(0.4, 0.8), "brokered": rng.uniform(0.4, 0.7),
                   "retail_uninsured": rng.uniform(0.15, 0.3), "corp_operational": 0.05,
                   "retail_insured": -0.05}
            blk = _block(act, amp, prof)
            extra[s + pre:s + pre + act] += blk
            crit_block = (s + pre, blk)
            shp = _shape(n, s, e, pre + 4, 8.0)
            mk = rng.uniform(60, 110)
            d_mkt_cds += mk * shp
            d_bank_cds += 0.4 * mk * shp
            w = shp[s:e + 1]
            r_mkt[s:e + 1] += -rng.uniform(0.10, 0.18) * w / w.sum()
            r_bank[s:e + 1] += -rng.uniform(0.12, 0.22) * w / w.sum()
            d_rate -= rng.uniform(0.3, 0.6) * shp
            sl = slice(s + pre, s + pre + act)
            top10_level[sl] = np.maximum(top10_level[sl], 0.30 + 0.20 * prof)
        elif etype == C.BLEED:
            act = int(rng.integers(55, 71))
            e = s + act - 1
            lag = 5
            prof = np.minimum(1.0, (np.arange(act - lag) + 1) / 10.0)
            amp = {"retail_uninsured": rng.uniform(0.07, 0.11), "brokered": rng.uniform(0.06, 0.10),
                   "corp_nonoperational": rng.uniform(0.05, 0.08), "small_business": 0.03,
                   "retail_insured": 0.01}
            blk = _block(act - lag, amp, prof)
            extra[s + lag:e + 1] += blk
            crit_block = (s + lag, blk)
            d_spread += rng.uniform(80, 140) * _shape(n, s, e, 20, 15.0)
            d_bank_cds += rng.uniform(8, 18) * _shape(n, s, e, 30, 10.0)
        elif etype == C.GLITCH:
            e = s + 1
            seg = SEG_IX[str(rng.choice(["corp_operational", "corp_nonoperational", "brokered"]))]
            x = rng.uniform(2.5, 5.0) / 100.0
            glitch[s, seg] = x
            glitch_rev_seg[e] = seg
            top10_level[s], top10_level[e] = 0.95, 0.90
        elif etype == C.OUTAGE:
            dur = int(rng.integers(1, 4))
            e = s + dur
            outage_delay[s:s + dur] = True
            outage_release[e] = True
        else:
            raise ValueError(etype)
        if e >= n:
            raise ValueError(f"event {etype} at {nominal.date()} runs past the sample end")
        if occupied[s:e + 1].any():
            raise ValueError(f"event windows overlap near {nominal.date()}")
        occupied[s:e + 1] = True
        rows.append({"event_id": eid, C.EVENT_TYPE: etype, C.EV_START: dates[s], C.EV_END: dates[e],
                     "s_idx": s, "e_idx": e})
        if crit_block is not None:
            ev_extra[eid] = crit_block

    # Benign market noise (unlabelled): looks like stress in one indicator, causes no outflow.
    nrng = np.random.default_rng([cfg.seed, 1])
    nrows = []
    for kind, nominal, length in cfg.nuisance:
        s0 = int((pd.Timestamp(nominal) - dates[0]).days)
        e0 = s0 + length - 1
        if e0 >= n or occupied[max(0, s0 - 5):e0 + 6].any():
            raise ValueError(f"nuisance window {kind} at {nominal} overlaps an event or the sample end")
        if kind == "spread_bump":
            d_spread += nrng.uniform(40, 80) * _shape(n, s0, e0, 10, 10.0)
        elif kind == "cds_bump":
            d_bank_cds += nrng.uniform(40, 90) * _shape(n, s0, e0, 3, 5.0)
        else:
            raise ValueError(kind)
        nrows.append({"kind": kind, "start": dates[s0], "end": dates[e0]})
    nuisance = pd.DataFrame(nrows, columns=["kind", "start", "end"])

    f = np.maximum(f - extra, -MAX_DAILY_DROP)

    # ---- balance simulation ---------------------------------------------------------
    bal = np.zeros((n, N_SEG))
    prev_bal = np.zeros((n, N_SEG))
    reported = np.zeros((n, N_SEG))
    queue_pct = np.zeros(n)
    recon = np.zeros(n)
    prev = np.array([INIT_BALANCE_M[s] for s in C.SEGMENTS])
    aff = np.array([SEG_IX[s] for s in OUTAGE_SEGMENTS])
    turn = np.array([TURNOVER[s] for s in OUTAGE_SEGMENTS])
    backlog_net = np.zeros(N_SEG)
    queue_usd = 0.0
    delay_frac = 0.9
    last_spurious = np.zeros(N_SEG)
    init = prev.copy()
    growth = np.array([ANNUAL_GROWTH[s] for s in C.SEGMENTS])
    for t in range(n):
        prev_bal[t] = prev
        trend = init * np.exp(growth * t / 365.0)
        pull = np.clip(REVERSION_RATE * (trend / prev - 1.0), -REVERSION_CAP, REVERSION_CAP)
        true_flow = (f[t] + pull) * prev
        posted = true_flow.copy()
        total_prev = prev.sum()
        if outage_delay[t]:
            held = delay_frac * true_flow[aff]
            posted[aff] -= held
            backlog_net[aff] += held
            queue_usd += float((delay_frac * turn * prev[aff]).sum())
        if outage_release[t]:
            posted += backlog_net
            queue_pct[t] = queue_usd / total_prev * 100.0
            backlog_net[:] = 0.0
            queue_usd = 0.0
        elif outage_delay[t]:
            queue_pct[t] = queue_usd / total_prev * 100.0
        spurious = glitch[t] * prev
        if glitch[t].any():
            last_spurious = spurious.copy()
        if glitch_rev_seg[t] >= 0:
            spurious[glitch_rev_seg[t]] = -last_spurious[glitch_rev_seg[t]]  # exact dollar reversal
        reported[t] = posted + spurious
        recon[t] = abs(spurious.sum()) / total_prev * 100.0
        prev = prev + posted
        bal[t] = prev
    if (bal <= 0).any():
        raise ValueError("generator produced a non-positive balance")
    queue_pct += np.abs(rng.normal(0.0, 0.01, n)) + 0.02
    recon += np.abs(rng.normal(0.0, 0.0005, n))

    # ---- critical days and event table ------------------------------------------------
    ev = pd.DataFrame(rows)
    crit, out_start, total_pct = [], [], []
    for r in ev.itertuples():
        if r.event_id in ev_extra:
            s0, blk = ev_extra[r.event_id]
            usd = (blk * prev_bal[s0:s0 + len(blk)]).sum(axis=1)
            cum = np.cumsum(usd)
            crit.append(dates[s0 + int(np.argmax(cum >= 0.5 * cum[-1]))])
            out_start.append(dates[s0])
            total_pct.append(float(cum[-1] / prev_bal[s0].sum() * 100.0))
        else:
            crit.append(dates[r.s_idx])
            out_start.append(dates[r.s_idx])
            total_pct.append(0.0)
    ev[C.EV_CRITICAL], ev[C.EV_OUTFLOW_START], ev[C.EV_TOTAL_OUTFLOW] = crit, out_start, total_pct
    ev["n_days"] = ev["e_idx"] - ev["s_idx"] + 1
    events = ev.drop(columns=["s_idx", "e_idx"])

    # ---- market and operational series -----------------------------------------------
    bank_cds = np.exp(np.log(80.0) + _ar1(rng, n, 0.97, 0.03)) + d_bank_cds
    mkt_cds = np.exp(np.log(60.0) + _ar1(rng, n, 0.97, 0.025)) + d_mkt_cds
    mret = rng.normal(0.0004, 0.010, n)
    bret = 0.8 * mret + rng.normal(0.0002, 0.010, n)
    mret = mret + r_mkt
    bret = bret + r_bank + 0.8 * r_mkt
    mkt_px = 100.0 * np.exp(np.cumsum(mret))
    bank_px = 50.0 * np.exp(np.cumsum(bret))
    knots = pd.to_datetime(["2019-06-01", "2020-04-01", "2022-03-01", "2023-07-01", "2024-06-01", "2024-12-31"])
    kv = [1.6, 0.2, 0.5, 4.9, 4.6, 4.1]
    rate = np.interp(np.array([d.toordinal() for d in dates]), np.array([d.toordinal() for d in knots]), kv)
    rate = rate + _ar1(rng, n, 0.98, 0.015) + d_rate
    spread = 25.0 + _ar1(rng, n, 0.95, 1.5) + d_spread
    top10 = np.clip(0.30 + 0.025 * rng.normal(size=n), 0.15, 0.6)
    top10 = np.where(top10_level > 0, np.maximum(top10, top10_level + 0.02 * rng.normal(size=n)), top10)
    top10 = np.clip(top10, 0.0, 1.0)

    observed = pd.DataFrame(index=dates)
    observed.index.name = "date"
    for s, i in SEG_IX.items():
        observed[C.balance_col(s)] = bal[:, i]
    for s, i in SEG_IX.items():
        observed[C.flow_col(s)] = reported[:, i]
    observed[C.BANK_CDS], observed[C.MARKET_CDS] = bank_cds, mkt_cds
    observed[C.BANK_EQUITY], observed[C.MARKET_EQUITY] = bank_px, mkt_px
    observed[C.RATE_2Y], observed[C.RATE_SPREAD] = rate, spread
    observed[C.TOP10_SHARE], observed[C.RECON_GAP], observed[C.PAYMENT_QUEUE] = top10, recon, queue_pct

    labels = pd.DataFrame(index=dates)
    labels.index.name = "date"
    labels[C.EVENT_TYPE] = C.NO_EVENT
    labels[C.EVENT_ID] = -1
    for r in events.itertuples():
        m = (dates >= getattr(r, C.EV_START)) & (dates <= getattr(r, C.EV_END))
        labels.loc[m, C.EVENT_TYPE] = getattr(r, C.EVENT_TYPE)
        labels.loc[m, C.EVENT_ID] = r.event_id
    labels[C.Y_LIQUIDITY] = labels[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES).astype(int)
    labels[C.Y_DQ] = labels[C.EVENT_TYPE].isin(C.DQ_TYPES).astype(int)
    return SyntheticDeposits(observed, labels, events, cfg, nuisance)

"""Daily scoring: severity tiers, data-quality routing, per-alert explanations and drift report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import interpret as I
from . import risk as R
from .evaluation import PlattCalibrator

QUIET, WATCH, ALERT, CRITICAL, DATA_QUALITY = "none", "WATCH", "ALERT", "CRITICAL", "DATA_QUALITY"
SEVERITY_RANK = {QUIET: 0, WATCH: 1, ALERT: 2, CRITICAL: 3}

LOG_COLUMNS = ["date", "score", "probability", "severity", "route", "dq_reason", "consecutive_alert_days",
               "explanation"]


@dataclass(frozen=True)
class TierConfig:
    """WATCH at watch_frac of the alert threshold; CRITICAL on high probability or persistence."""
    watch_frac: float = 0.5
    critical_prob: float = 0.90
    critical_run_days: int = 3


def assign_severity(score: float, threshold: float, prob: float, consecutive: int,
                    cfg: TierConfig = TierConfig()) -> str:
    if score >= threshold:
        if prob >= cfg.critical_prob or consecutive >= cfg.critical_run_days:
            return CRITICAL
        return ALERT
    if score >= cfg.watch_frac * threshold:
        return WATCH
    return QUIET


def _consecutive(flags: np.ndarray) -> np.ndarray:
    out = np.zeros(len(flags), int)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run
    return out


def build_alert_log(detector, X: pd.DataFrame, threshold: float, calibrator: PlattCalibrator, router,
                    tiers: TierConfig = TierConfig(), explain: bool = True) -> pd.DataFrame:
    """Score every day in X; one row per day with severity, route and explanation.

    Days failing the data-quality router are always reported as DATA_QUALITY and never raise a
    liquidity severity (a suppressed liquidity severity is noted in the explanation text)."""
    scores = detector.score(X)
    prob = calibrator.predict(scores)
    consec = _consecutive(scores >= threshold)
    reasons = router.reasons(X)
    sev = [assign_severity(s, threshold, p, c, tiers) for s, p, c in zip(scores, prob, consec)]
    log = pd.DataFrame({"date": X.index, "score": scores, "probability": prob, "severity": sev,
                        "consecutive_alert_days": consec, "dq_reason": reasons.to_numpy()})
    log["route"] = np.where(log["dq_reason"] != "", "data_quality", "liquidity")
    log["explanation"] = ""
    flagged = log["severity"] != QUIET
    if explain and flagged.any() and I.HAVE_SHAP and hasattr(detector.model, "estimators_"):
        vals, _ = I.shap_values(detector, X)
        exp = I.explain_days(vals, X, detector.columns, X.index[flagged.to_numpy()])
        log.loc[flagged, "explanation"] = exp["explanation"].to_numpy()
    dq = (log["route"] == "data_quality").to_numpy()
    suppressed = np.where(flagged.to_numpy(), "; liquidity severity " + log["severity"] + " suppressed", "")
    log.loc[dq, "explanation"] = (log["dq_reason"] + suppressed)[dq]
    log.loc[dq, "severity"] = DATA_QUALITY
    return log[LOG_COLUMNS]


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, reference_scores: np.ndarray,
                 current_scores: np.ndarray, probability_scores: bool = False) -> pd.DataFrame:
    """Feature PSI table plus a final row for the model score itself."""
    rep = R.feature_psi(reference, current)
    s = R.psi(reference_scores, current_scores, edges=R.probability_edges() if probability_scores else None)
    row = pd.DataFrame([{"feature": "MODEL_SCORE", "psi": s, "status": R.psi_status(s)}])
    return pd.concat([rep, row], ignore_index=True)

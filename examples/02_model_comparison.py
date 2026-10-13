"""Time-series CV, cost-based thresholds, held-out year comparison, intervals and calibration."""
from pathlib import Path

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E
from deposit_anomaly import pipeline as P
from deposit_anomaly.plotting import plot_pr_curves, plot_reliability, plot_score_timeline
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
BOOTSTRAP_BLOCK_DAYS = 10
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 11
PLOT_MODEL = "gradient_boosting"
TABLE_COLUMNS = ["model", "threshold", "pr_auc", "precision", "recall", "events_detected", "events_total",
                 "detection_rate", "mean_lead_days", "fa_per_year", "dq_events_alerted", "dq_events_total", "cost"]
# -----------------------------------------------------------------------------------


def main() -> None:
    exp = P.run_experiment()
    costs = exp.costs
    cost_note = (f"Assumed costs (stylized): missed liquidity event = {costs.miss:g}, false-alarm week = "
                 f"{costs.false_alarm:g}. Thresholds are chosen on out-of-fold scores only.")

    write_outputs(exp.fold_pr_auc.pivot(index="model", columns="fold", values="pr_auc").reset_index(),
                  OUT_DIR, "cv_pr_auc_by_fold", "Expanding-window CV: day-level PR-AUC by fold", digits=3,
                  note=f"Purge gap = {P.PURGE_DAYS} days. Fold validation blocks: 2021H2, 2022H1, 2022H2, 2023H1, 2023H2.")
    for period, route in (("oof", False), ("holdout", False), ("holdout", True)):
        table, by_type = P.comparison_tables(exp, period, route)
        tag = f"{period}{'_routed' if route else ''}"
        write_outputs(table[TABLE_COLUMNS], OUT_DIR, f"model_comparison_{tag}",
                      f"Model comparison, {period}{' with data-quality routing' if route else ''}", digits=3, note=cost_note)
        write_outputs(by_type, OUT_DIR, f"by_event_type_{tag}", f"Detection by event type, {period}"
                      f"{' with data-quality routing' if route else ''}", digits=2)
        print(f"== {tag}")
        print(table[TABLE_COLUMNS].round(3).to_string(index=False))

    # Intervals on the held-out year.
    table, _ = P.comparison_tables(exp, "holdout", False)
    y = exp.y(exp.holdout_scores[P.MODEL_ORDER[0]].index)
    rows = []
    for r in table.itertuples():
        sc = exp.holdout_scores["rules" if r.model == P.RULES_FIXED else r.model].to_numpy()
        lo, hi = E.block_bootstrap_ci(y, sc, E.pr_auc, BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        dlo, dhi = E.clopper_pearson(r.events_detected, r.events_total)
        n_fa = int(round(r.fa_per_year * exp.holdout_scorer.years))
        flo, fhi = E.poisson_rate_ci(n_fa, exp.holdout_scorer.years)
        rows.append({"model": r.model, "pr_auc": r.pr_auc, "pr_auc_lo": lo, "pr_auc_hi": hi,
                     "detection_rate": r.detection_rate, "det_lo": dlo, "det_hi": dhi,
                     "fa_per_year": r.fa_per_year, "fa_lo": flo, "fa_hi": fhi})
    write_outputs(pd.DataFrame(rows), OUT_DIR, "holdout_intervals", "Held-out year, 95% intervals", digits=3,
                  note=f"PR-AUC: moving-block bootstrap ({BOOTSTRAP_BLOCK_DAYS}-day blocks, {BOOTSTRAP_REPS} reps). "
                       "Detection rate: exact binomial (Clopper-Pearson) on only a handful of events. "
                       "False alarms: exact Poisson.")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Calibration: raw vs Platt scaling fitted on out-of-fold scores.
    cal_rows, rel = [], {}
    for m in P.MODEL_ORDER:
        oof_y = exp.y(exp.oof_scores[m].index)
        cal = E.PlattCalibrator().fit(exp.oof_scores[m].to_numpy(), oof_y)
        p = cal.predict(exp.holdout_scores[m].to_numpy())
        raw = exp.holdout_scores[m].to_numpy()
        row = {"model": m, "brier_platt": E.brier(y, p), "ece_platt": E.expected_calibration_error(y, p)}
        if m in ("gradient_boosting", "random_forest"):
            row.update({"brier_raw": E.brier(y, raw), "ece_raw": E.expected_calibration_error(y, raw)})
            rel[f"{m} raw"] = E.reliability_table(y, raw)
        rel[f"{m} platt"] = E.reliability_table(y, p)
        cal_rows.append(row)
    cal_df = pd.DataFrame(cal_rows)
    write_outputs(cal_df, OUT_DIR, "calibration", "Held-out year calibration (10 equal-width bins)", digits=4,
                  note="Raw columns apply only to the probabilistic classifiers; others have arbitrary score scales.")
    print(cal_df.round(4).to_string(index=False))

    plot_pr_curves(y, {m: exp.holdout_scores[m].to_numpy() for m in P.MODEL_ORDER}, OUT_DIR / "pr_curves_holdout.png")
    plot_reliability({k: v for k, v in rel.items() if k.startswith(("gradient", "random")) }, OUT_DIR / "calibration.png")
    hold_events = exp.events[exp.events[C.EV_END] >= exp.holdout_scores[PLOT_MODEL].index[0]]
    plot_score_timeline(exp.holdout_scores[PLOT_MODEL], exp.thresholds[PLOT_MODEL], hold_events,
                        f"{PLOT_MODEL} score, held-out year (shaded = injected events)", OUT_DIR / "score_timeline_holdout.png")


if __name__ == "__main__":
    main()

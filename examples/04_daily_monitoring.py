"""Daily scoring over the held-out year: alert log with severity tiers, explanations, drift report."""
from pathlib import Path

import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E
from deposit_anomaly import monitoring as M
from deposit_anomaly import pipeline as P
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
MODEL = "gradient_boosting"
DRIFT_WINDOW_DAYS = 90
# -----------------------------------------------------------------------------------


def main() -> None:
    exp = P.run_experiment()
    det = exp.final_detectors[MODEL]
    idx = exp.holdout_scores[MODEL].index
    X = exp.X.loc[idx]
    cal = E.PlattCalibrator().fit(exp.oof_scores[MODEL].to_numpy(), exp.y(exp.oof_scores[MODEL].index))
    log = M.build_alert_log(det, X, exp.thresholds[MODEL], cal, exp.router)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(OUT_DIR / "alert_log.csv", index=False)
    flagged = log[log["severity"] != M.QUIET]
    write_outputs(flagged.head(25), OUT_DIR, "alert_log_excerpt", "Alert log excerpt (first 25 non-quiet days)", digits=3,
                  note="Synthetic data. Severity: WATCH (>= 0.5x threshold), ALERT, CRITICAL, DATA_QUALITY (routed).")

    truth = exp.labels.loc[log["date"], C.EVENT_TYPE].to_numpy()
    cross = pd.crosstab(pd.Series(truth, name="true_event_type"), log["severity"].rename("severity")).reset_index()
    write_outputs(cross, OUT_DIR, "alert_severity_by_truth", "Held-out year: days by severity and true event type", digits=0)
    print(cross.to_string(index=False))

    ref = exp.X.iloc[exp.holdout_fold.train_idx]
    rep = M.drift_report(ref, X.iloc[-DRIFT_WINDOW_DAYS:], exp.oof_scores[MODEL].to_numpy(), det.score(X.iloc[-DRIFT_WINDOW_DAYS:]),
                         probability_scores=True)
    shown = pd.concat([rep.head(14), rep[rep["feature"] == "MODEL_SCORE"]])
    write_outputs(shown, OUT_DIR, "drift_report", f"Drift report: last {DRIFT_WINDOW_DAYS} days vs training (top 14 features plus score)",
                  digits=3, note="PSI thresholds: OK < 0.10, WATCH < 0.25, ACTION otherwise. MODEL_SCORE is the score PSI.")
    print(rep.head(8).round(3).to_string(index=False))
    print(log["severity"].value_counts().to_string())


if __name__ == "__main__":
    main()

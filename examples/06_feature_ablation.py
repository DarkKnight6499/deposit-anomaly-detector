"""Retrain the gradient-boosting detector with feature groups removed to see what drives the result."""
from pathlib import Path

import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E
from deposit_anomaly import features as F
from deposit_anomaly import pipeline as P
from deposit_anomaly.models import SupervisedDetector
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
OPS_COLUMNS = (F.TOP10, F.TOP10_Z, C.RECON_GAP, C.PAYMENT_QUEUE)
GROUPS = {
    "all features": (),
    "without ops and concentration signals": OPS_COLUMNS,
    "without market and rate-spread signals": F.MARKET_COLUMNS,
    "flows only (no market, no ops)": F.MARKET_COLUMNS + OPS_COLUMNS,
}
# -----------------------------------------------------------------------------------


def make(cols):
    det = SupervisedDetector("gbm")
    det.columns = tuple(cols)
    return det


def main() -> None:
    exp = P.run_experiment()
    rows = []
    for name, dropped in GROUPS.items():
        cols = [c for c in F.FEATURE_COLUMNS if c not in dropped]
        parts = []
        for fold in exp.folds:
            det = P._fit(make(cols), exp.X.iloc[fold.train_idx], exp.labels)
            parts.append(pd.Series(det.score(exp.X.iloc[fold.val_idx]), index=exp.X.index[fold.val_idx]))
        oof = pd.concat(parts)
        thr, _ = E.select_threshold(oof, exp.oof_scorer, exp.costs)
        det = P._fit(make(cols), exp.X.iloc[exp.holdout_fold.train_idx], exp.labels)
        hold = pd.Series(det.score(exp.X.iloc[exp.holdout_fold.val_idx]), index=exp.holdout_scores["rules"].index)
        row = P.evaluate_scores(exp, name, hold, thr, exp.holdout_scorer)["row"]
        rows.append({"features": name, "n_features": len(cols), "pr_auc": row["pr_auc"],
                     "detection_rate": row["detection_rate"], "mean_lead_days": row["mean_lead_days"],
                     "fa_per_year": row["fa_per_year"], "dq_events_alerted": row["dq_events_alerted"],
                     "dq_events_total": row["dq_events_total"]})
    out = pd.DataFrame(rows)
    write_outputs(out, OUT_DIR, "feature_ablation", "Gradient boosting, held-out year, feature groups removed", digits=3,
                  note="Thresholds re-selected on out-of-fold scores for each variant.")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Permutation importance, SHAP, per-alert explanations, partial dependence, PSI and score drift."""
from pathlib import Path

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import interpret as I
from deposit_anomaly import pipeline as P
from deposit_anomaly import risk as R
from deposit_anomaly.plotting import plot_bar, plot_pd, plot_psi_heatmap
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
MODEL = "gradient_boosting"
PERM_REPEATS = 10
PERM_SEED = 3
TOP_FEATURES_FOR_PD = 3
EXPLAIN_TOP_K = 3
N_ALERT_EXAMPLES = 12
# -----------------------------------------------------------------------------------


def main() -> None:
    exp = P.run_experiment()
    det = exp.final_detectors[MODEL]
    hold_idx = exp.holdout_scores[MODEL].index
    Xh = exp.X.loc[hold_idx]
    y = exp.y(hold_idx)

    perm = I.permutation_importance(det, Xh, y, PERM_REPEATS, PERM_SEED)
    write_outputs(perm.head(20), OUT_DIR, "permutation_importance", f"Permutation importance ({MODEL}, held-out year)",
                  digits=4, note="Drop in day-level PR-AUC when one feature is shuffled.")
    plot_bar(perm.set_index(I.IMP_FEATURE)[I.IMP_MEAN], "Permutation importance (held-out year)", "drop in PR-AUC",
             OUT_DIR / "permutation_importance.png")
    print(perm.head(10).round(4).to_string(index=False))

    explainer_used = "shap.TreeExplainer" if I.HAVE_SHAP else "permutation importance only (shap unavailable)"
    print("interpretability backend:", explainer_used)
    vals, base = I.shap_values(det, Xh)
    msh = I.mean_abs_shap(vals, det.columns)
    write_outputs(msh.head(20).rename_axis("feature").reset_index(name="mean_abs_shap"), OUT_DIR, "shap_importance",
                  f"Mean |SHAP| ({MODEL}, held-out year, log-odds)", digits=4)
    plot_bar(msh, "Mean |SHAP| (held-out year)", "mean |SHAP|, log-odds", OUT_DIR / "shap_importance.png")
    print(msh.head(10).round(4).to_string())

    alert_days = hold_idx[exp.holdout_scores[MODEL].to_numpy() >= exp.thresholds[MODEL]]
    expl = I.explain_days(vals, Xh, det.columns, alert_days, EXPLAIN_TOP_K)
    truth = exp.labels.loc[expl["date"], C.EVENT_TYPE].to_numpy()
    expl.insert(1, "true_event_type", truth)
    write_outputs(expl[["date", "true_event_type", "explanation"]].head(N_ALERT_EXAMPLES), OUT_DIR, "alert_explanations",
                  "Per-alert explanations (first alert days, held-out year)", digits=2,
                  note="Each entry: feature=value (signed SHAP contribution in log-odds).")
    print(expl[["date", "true_event_type", "explanation"]].head(6).to_string(index=False))

    top = list(msh.index[:TOP_FEATURES_FOR_PD])
    plot_pd(I.pd_curves(det, Xh, top), OUT_DIR / "partial_dependence.png")

    # Model risk: stability across time.
    ref = exp.X.iloc[exp.holdout_fold.train_idx]
    ref_clean = ref[(exp.labels.loc[ref.index, C.EVENT_TYPE] == C.NO_EVENT).to_numpy()]
    cur_clean = Xh[(exp.labels.loc[hold_idx, C.EVENT_TYPE] == C.NO_EVENT).to_numpy()]
    psi_all = R.feature_psi(ref, Xh)
    psi_clean = R.feature_psi(ref_clean, cur_clean).rename(columns={"psi": "psi_event_free", "status": "status_event_free"})
    merged = psi_all.merge(psi_clean, on="feature")
    write_outputs(merged.head(15), OUT_DIR, "feature_psi", "Feature PSI, training vs held-out year (top 15)", digits=3,
                  note="psi: all days (what monitoring would see). psi_event_free: only days with no injected event.")
    print(merged.head(8).round(3).to_string(index=False))
    n_action = int((merged["status"] == "ACTION").sum())
    n_action_clean = int((merged["status_event_free"] == "ACTION").sum())
    print(f"features at ACTION: {n_action} of {len(merged)} (all days), {n_action_clean} (event-free days)")

    plot_psi_heatmap(R.psi_by_period(ref, Xh), OUT_DIR / "psi_by_quarter.png")
    ref_scores = exp.oof_scores[MODEL].to_numpy()  # out-of-sample reference; in-sample train scores would inflate PSI
    drift = R.score_drift(ref_scores, exp.holdout_scores[MODEL], edges=R.probability_edges())
    write_outputs(drift, OUT_DIR, "score_drift", "Model score drift by quarter (held-out year)", digits=3)
    print(drift.round(3).to_string(index=False))

    # Stability of the CV result across time.
    stab = exp.fold_pr_auc[exp.fold_pr_auc["model"] == MODEL][["fold", "val_start", "val_end", "n_train", "pr_auc"]]
    write_outputs(stab, OUT_DIR, "cv_stability", f"{MODEL}: PR-AUC by CV fold", digits=3)


if __name__ == "__main__":
    main()

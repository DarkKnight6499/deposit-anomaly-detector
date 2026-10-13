"""Repeat the full experiment on several generator seeds to see how much the held-out result moves."""
from dataclasses import replace
from pathlib import Path

import pandas as pd

from deposit_anomaly import pipeline as P
from deposit_anomaly.generator import GeneratorConfig
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
METRICS = ["pr_auc", "detection_rate", "mean_lead_days", "fa_per_year", "dq_events_alerted"]
# -----------------------------------------------------------------------------------


def main() -> None:
    frames = []
    for seed in SEEDS:
        exp = P.run_experiment(replace(GeneratorConfig(), seed=seed))
        table, _ = P.comparison_tables(exp, "holdout", False)
        table["seed"] = seed
        frames.append(table)
        print("seed", seed, "done")
    allr = pd.concat(frames)
    agg = allr.groupby("model")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    write_outputs(agg, OUT_DIR, "seed_robustness", f"Held-out year across {len(SEEDS)} generator seeds", digits=3,
                  note="Mean and standard deviation across seeds of the held-out metrics. Thresholds re-selected per seed.")
    allr.to_csv(OUT_DIR / "seed_robustness_raw.csv", index=False)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

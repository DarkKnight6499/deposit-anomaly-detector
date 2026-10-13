"""Generate the synthetic dataset, check label alignment and balance reconciliation, save a chart."""
from pathlib import Path

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly.generator import generate
from deposit_anomaly.plotting import plot_balances
from deposit_anomaly.report import write_outputs

# ---- config -----------------------------------------------------------------------
OUT_DIR = Path(__file__).parent / "output"
# -----------------------------------------------------------------------------------


def main() -> None:
    data = generate()
    obs, lab, ev = data.observed, data.labels, data.events
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT_DIR / "synthetic_observed.csv")
    lab.to_csv(OUT_DIR / "synthetic_labels.csv")

    ev_out = ev.copy()
    write_outputs(ev_out, OUT_DIR, "events", "Injected events (ground truth)", digits=2)

    counts = lab[C.EVENT_TYPE].value_counts().rename_axis(C.EVENT_TYPE).reset_index(name="days")
    n_events = ev[C.EVENT_TYPE].value_counts().rename_axis(C.EVENT_TYPE).reset_index(name="events")
    summary = counts.merge(n_events, on=C.EVENT_TYPE, how="left").fillna({"events": 0})
    summary["events"] = summary["events"].astype(int)
    write_outputs(summary, OUT_DIR, "data_summary", "Labelled days and events by type", digits=0)

    # Reconciliation: balance change equals reported flows except on glitch days.
    bal = obs[list(C.BALANCE_COLUMNS)].sum(axis=1)
    flow = obs[list(C.FLOW_COLUMNS)].sum(axis=1)
    gap = (bal.diff() - flow).abs() / bal.shift(1) * 100
    glitch_day = lab[C.EVENT_TYPE] == C.GLITCH
    checks = pd.DataFrame([
        {"check": "days", "value": len(obs)},
        {"check": "first day", "value": str(obs.index[0].date())},
        {"check": "last day", "value": str(obs.index[-1].date())},
        {"check": "min balance, $m", "value": float(obs[list(C.BALANCE_COLUMNS)].min().min())},
        {"check": "max abs recon gap on non-glitch days, % of balance", "value": float(gap[~glitch_day].max())},
        {"check": "max abs recon gap on glitch days, % of balance", "value": float(gap[glitch_day].max())},
        {"check": "share of days labelled liquidity stress", "value": float(lab[C.Y_LIQUIDITY].mean())},
    ])
    write_outputs(checks, OUT_DIR, "data_checks", "Dataset checks", digits=4)
    plot_balances(obs, ev, OUT_DIR / "balances_events.png")
    print(summary.to_string(index=False))
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

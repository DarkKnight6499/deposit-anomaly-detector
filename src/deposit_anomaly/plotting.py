"""Matplotlib charts saved as PNG (Agg backend, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from . import columns as C  # noqa: E402

EVENT_COLORS = {C.RUN: "#c0392b", C.FLIGHT: "#e67e22", C.BLEED: "#8e44ad", C.GLITCH: "#2980b9", C.OUTAGE: "#7f8c8d"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def shade_events(ax, events: pd.DataFrame, xlim=None) -> None:
    for r in events.itertuples():
        s, e = getattr(r, C.EV_START), getattr(r, C.EV_END)
        if xlim and (e < xlim[0] or s > xlim[1]):
            continue
        ax.axvspan(s, e + pd.Timedelta(days=1), color=EVENT_COLORS[getattr(r, C.EVENT_TYPE)], alpha=0.35, lw=0)


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=c, alpha=0.5, label=t) for t, c in EVENT_COLORS.items()]


def plot_balances(observed: pd.DataFrame, events: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for seg in C.SEGMENTS:
        axes[0].plot(observed.index, observed[C.balance_col(seg)] / 1000.0, lw=1, label=seg)
    axes[0].set_ylabel("balance, $bn")
    axes[0].legend(ncol=3, fontsize=7)
    axes[1].plot(observed.index, observed[C.BANK_CDS], lw=0.9, label="bank CDS proxy (bps)")
    axes[1].plot(observed.index, observed[C.MARKET_CDS], lw=0.9, label="market CDS proxy (bps)")
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        shade_events(ax, events)
    axes[1].legend(handles=axes[1].get_legend_handles_labels()[0] + legend_handles(), fontsize=6, ncol=4, loc="upper left")
    axes[0].set_title("Synthetic deposits with injected events (shaded)")
    return _save(fig, path)


def plot_pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name, lw=1.4)
    ax.axhline(np.mean(y), color="grey", ls=":", lw=1)
    ax.set_xlabel("recall (days)")
    ax.set_ylabel("precision (days)")
    ax.set_title("Held-out year: day-level precision-recall")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_score_timeline(scores: pd.Series, threshold: float, events: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(scores.index, scores.values, lw=0.9)
    ax.axhline(threshold, color="k", ls="dashed", lw=0.9, label=f"threshold {threshold:.3g}")
    shade_events(ax, events, (scores.index[0], scores.index[-1]))
    ax.set_title(title)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles(), fontsize=6, ncol=3)
    return _save(fig, path)


def plot_reliability(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for name, t in tables.items():
        ax.plot(t["mean_pred"], t["frac_pos"], marker="o", ms=4, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction of stress days")
    ax.set_title("Reliability, held-out year")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_bar(series: pd.Series, title: str, xlabel: str, path: Path, top: int = 15) -> Path:
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(s.index, s.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _save(fig, path)


def plot_pd(curves: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(curves), figsize=(3.6 * len(curves), 3.2), sharey=True)
    for ax, (f, c) in zip(np.atleast_1d(axes), curves.items()):
        ax.plot(c["value"], c["pd"])
        ax.set_xlabel(f, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("average P(stress)")
    return _save(fig, path)


def plot_psi_heatmap(psi_matrix: pd.DataFrame, path: Path, top: int = 15) -> Path:
    m = psi_matrix.loc[psi_matrix.max(axis=1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(m.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="PSI")
    ax.set_title("Feature PSI by quarter (held-out year vs training)")
    return _save(fig, path)

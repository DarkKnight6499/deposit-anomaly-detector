"""End-to-end experiment: generate, featurise, cross-validate, pick thresholds, evaluate holdout."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import columns as C
from . import evaluation as E
from . import features as F
from . import splits as S
from .dq import DataQualityRouter
from .generator import GeneratorConfig, SyntheticDeposits, generate
from .models import Detector, make_detectors

DEV_BOUNDARIES = ["2021-07-01", "2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01"]
HOLDOUT_START = "2024-01-01"
PURGE_DAYS = F.LOOKBACK_DAYS
RULES_FIXED = "rules_fixed_policy"
MODEL_ORDER = ("rules", "robust_z", "isolation_forest", "autoencoder", "gradient_boosting", "random_forest")


@dataclass
class Experiment:
    data: SyntheticDeposits
    X: pd.DataFrame
    labels: pd.DataFrame
    folds: list
    holdout_fold: S.Fold
    oof_scores: dict[str, pd.Series]
    holdout_scores: dict[str, pd.Series]
    thresholds: dict[str, float]
    final_detectors: dict[str, Detector]
    router: DataQualityRouter
    oof_scorer: E.EventScorer
    holdout_scorer: E.EventScorer
    fold_pr_auc: pd.DataFrame
    costs: E.CostConfig = field(default_factory=E.CostConfig)

    @property
    def events(self) -> pd.DataFrame:
        return self.data.events

    def y(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.labels.loc[index, C.Y_LIQUIDITY].to_numpy()


def _fit(det: Detector, X: pd.DataFrame, labels: pd.DataFrame) -> Detector:
    lab = labels.loc[X.index]
    return det.fit(X, lab[C.Y_LIQUIDITY], lab[C.EVENT_TYPE] == C.NO_EVENT)


def run_experiment(config: GeneratorConfig | None = None, costs: E.CostConfig | None = None) -> Experiment:
    costs = costs or E.CostConfig()
    data = generate(config)
    X = F.model_frame(data.observed)
    labels = data.labels.loc[X.index]
    folds = S.expanding_window_folds(X.index, DEV_BOUNDARIES, PURGE_DAYS, end_exclusive=HOLDOUT_START)
    holdout = S.holdout_split(X.index, HOLDOUT_START, PURGE_DAYS)

    parts: dict[str, list[pd.Series]] = {m: [] for m in MODEL_ORDER}
    fold_rows = []
    for k, fold in enumerate(folds):
        dets = make_detectors()
        for name in MODEL_ORDER:
            _fit(dets[name], X.iloc[fold.train_idx], labels)
            sc = pd.Series(dets[name].score(X.iloc[fold.val_idx]), index=X.index[fold.val_idx])
            parts[name].append(sc)
            fold_rows.append({"fold": k + 1, "val_start": fold.val_start, "val_end": fold.val_end,
                              "model": name, "n_train": len(fold.train_idx),
                              "pr_auc": E.pr_auc(labels[C.Y_LIQUIDITY].iloc[fold.val_idx].to_numpy(), sc.to_numpy())})
    oof = {m: pd.concat(parts[m]) for m in MODEL_ORDER}
    oof_index = oof[MODEL_ORDER[0]].index
    oof_scorer = E.EventScorer(oof_index, data.events)

    dets = make_detectors()
    train_X = X.iloc[holdout.train_idx]
    holdout_scores = {}
    for name in MODEL_ORDER:
        _fit(dets[name], train_X, labels)
        holdout_scores[name] = pd.Series(dets[name].score(X.iloc[holdout.val_idx]), index=X.index[holdout.val_idx])
    router = DataQualityRouter().fit(train_X)
    thresholds = {m: E.select_threshold(oof[m], oof_scorer, costs)[0] for m in MODEL_ORDER}
    return Experiment(data, X, labels, folds, holdout, oof, holdout_scores, thresholds, dets, router,
                      oof_scorer, E.EventScorer(X.index[holdout.val_idx], data.events),
                      pd.DataFrame(fold_rows), costs)


def evaluate_scores(exp: Experiment, name: str, scores: pd.Series, threshold: float, scorer: E.EventScorer,
                    route_dq: bool = False) -> dict:
    """One comparison-table row for a model on an evaluation period."""
    alerts = (scores.to_numpy() >= threshold)
    if route_dq:
        alerts = alerts & ~exp.router.flag(exp.X.loc[scores.index]).to_numpy()
    ev = scorer.evaluate(alerts)
    y = exp.y(scores.index)
    dm = E.day_metrics(y, scores.to_numpy(), alerts)
    pe = ev["per_event"]
    liq = pe[pe[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES)]
    dq = pe[pe[C.EVENT_TYPE].isin(C.DQ_TYPES)]
    row = {"model": name, "threshold": threshold, "pr_auc": dm["pr_auc"], "precision": dm["precision"],
           "recall": dm["recall"], "alert_days": dm["alert_days"],
           "events_detected": int(liq["detected"].sum()), "events_total": int(len(liq)),
           "detection_rate": liq["detected"].mean() if len(liq) else np.nan,
           "mean_lead_days": liq["lead_days"].mean(), "fa_per_year": ev["fa_per_year"],
           "dq_events_alerted": int(dq["detected"].sum()), "dq_events_total": int(len(dq)),
           "cost": exp.costs.miss * ev["missed"] + exp.costs.false_alarm * ev["fa_weeks"]}
    return {"row": row, "eval": ev}


def comparison_tables(exp: Experiment, period: str = "holdout", route_dq: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(model comparison table, per-event-type detection table) for 'oof' or 'holdout'."""
    scores = exp.oof_scores if period == "oof" else exp.holdout_scores
    scorer = exp.oof_scorer if period == "oof" else exp.holdout_scorer
    rows, type_rows = [], []
    items = [(m, scores[m], exp.thresholds[m]) for m in MODEL_ORDER]
    items.append((RULES_FIXED, scores["rules"], 1.0))
    for name, sc, thr in items:
        out = evaluate_scores(exp, name, sc, thr, scorer, route_dq)
        rows.append(out["row"])
        pe = out["eval"]["per_event"]
        for t, g in pe.groupby(C.EVENT_TYPE):
            type_rows.append({"model": name, C.EVENT_TYPE: t, "events": len(g), "alerted": int(g["detected"].sum()),
                              "rate": g["detected"].mean(), "mean_lead_days": g["lead_days"].mean()})
    return pd.DataFrame(rows), pd.DataFrame(type_rows)

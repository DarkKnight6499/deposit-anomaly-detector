"""Detectors. Every detector exposes fit(X, y, clean_mask) and score(X); a higher score means
more anomalous / more likely a liquidity-stress day."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import features as F

DEFAULT_SEED = 7


class NotFittedError(RuntimeError):
    pass


class Detector:
    name = "detector"
    columns: tuple[str, ...] = F.FEATURE_COLUMNS
    fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None,
            clean_mask: pd.Series | None = None) -> "Detector":
        raise NotImplementedError

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _check(self) -> None:
        if not self.fitted:
            raise NotFittedError(f"{self.name} is not fitted")


class RulesBaseline(Detector):
    """Static thresholds on daily outflow percent of balances (total and per segment).

    Thresholds are the `quantile` of each outflow series over the training period, then frozen.
    Score = largest ratio of an outflow to its threshold; an alert fires when the score >= 1."""
    name = "rules"
    columns = F.RULE_COLUMNS

    def __init__(self, quantile: float = 0.995):
        self.quantile = quantile
        self.thresholds: pd.Series | None = None

    def fit(self, X, y=None, clean_mask=None):
        self.thresholds = X[list(self.columns)].quantile(self.quantile).clip(lower=1e-6)
        self.fitted = True
        return self

    def score(self, X):
        self._check()
        ratio = X[list(self.columns)] / self.thresholds
        return ratio.max(axis=1).to_numpy()


class RobustZBaseline(Detector):
    """Seasonality-adjusted robust z-scores: score = largest outflow z across total and segments."""
    name = "robust_z"
    columns = F.STAT_COLUMNS

    def fit(self, X, y=None, clean_mask=None):
        self.fitted = True
        return self

    def score(self, X):
        self._check()
        return X[list(self.columns)].max(axis=1).to_numpy()


class IsolationForestDetector(Detector):
    name = "isolation_forest"

    def __init__(self, seed: int = DEFAULT_SEED, n_estimators: int = 300):
        self.seed, self.n_estimators = seed, n_estimators
        self.model: IsolationForest | None = None

    def fit(self, X, y=None, clean_mask=None):
        self.model = IsolationForest(n_estimators=self.n_estimators, max_samples=256,
                                     random_state=self.seed).fit(X[list(self.columns)])
        self.fitted = True
        return self

    def score(self, X):
        self._check()
        return -self.model.score_samples(X[list(self.columns)])


class AutoencoderDetector(Detector):
    """MLPRegressor trained to reconstruct standardised features on clean days (no labelled event)."""
    name = "autoencoder"
    CLIP = 8.0

    def __init__(self, seed: int = DEFAULT_SEED, hidden: tuple[int, ...] = (24, 8, 24)):
        self.seed, self.hidden = seed, hidden
        self.scaler: StandardScaler | None = None
        self.model: MLPRegressor | None = None

    def _prep(self, X):
        return np.clip(self.scaler.transform(X[list(self.columns)]), -self.CLIP, self.CLIP)

    def fit(self, X, y=None, clean_mask=None):
        train = X if clean_mask is None else X[clean_mask.to_numpy()]
        self.scaler = StandardScaler().fit(train[list(self.columns)])
        Z = self._prep(train)
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-3, max_iter=600,
                                  random_state=self.seed).fit(Z, Z)
        self.fitted = True
        return self

    def score(self, X):
        self._check()
        Z = self._prep(X)
        return ((self.model.predict(Z) - Z) ** 2).mean(axis=1)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights (class imbalance handling for boosting)."""
    y = np.asarray(y)
    n = len(y)
    w = np.ones(n)
    for c in np.unique(y):
        w[y == c] = n / (len(np.unique(y)) * (y == c).sum())
    return w


class SupervisedDetector(Detector):
    """Gradient boosting (kind='gbm') or random forest (kind='rf') on the liquidity-stress label."""

    def __init__(self, kind: str = "gbm", seed: int = DEFAULT_SEED):
        if kind not in ("gbm", "rf"):
            raise ValueError("kind must be 'gbm' or 'rf'")
        self.kind, self.seed = kind, seed
        self.name = {"gbm": "gradient_boosting", "rf": "random_forest"}[kind]
        self.model = None

    def fit(self, X, y=None, clean_mask=None):
        if y is None or len(np.unique(y)) < 2:
            raise ValueError("supervised detector needs both classes in the training labels")
        Xm, yv = X[list(self.columns)], np.asarray(y)
        if self.kind == "gbm":
            self.model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                                                    subsample=0.8, random_state=self.seed)
            self.model.fit(Xm, yv, sample_weight=balanced_weights(yv))
        else:
            self.model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=self.seed)
            self.model.fit(Xm, yv)
        self.fitted = True
        return self

    def score(self, X):
        self._check()
        return self.model.predict_proba(X[list(self.columns)])[:, 1]


def make_detectors(seed: int = DEFAULT_SEED) -> dict[str, Detector]:
    return {
        "rules": RulesBaseline(),
        "robust_z": RobustZBaseline(),
        "isolation_forest": IsolationForestDetector(seed),
        "autoencoder": AutoencoderDetector(seed),
        "gradient_boosting": SupervisedDetector("gbm", seed),
        "random_forest": SupervisedDetector("rf", seed),
    }

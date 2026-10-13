"""Shared cached fixtures so slow objects are built once per test run."""
from functools import lru_cache

import numpy as np
import pandas as pd

from deposit_anomaly import features as F
from deposit_anomaly import pipeline as P
from deposit_anomaly.generator import generate


@lru_cache(maxsize=1)
def data():
    return generate()


@lru_cache(maxsize=1)
def experiment():
    return P.run_experiment()


def random_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Gaussian frame with the real feature columns and a daily index (for model unit tests)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="D")
    return pd.DataFrame(rng.normal(size=(n, len(F.FEATURE_COLUMNS))), index=idx, columns=list(F.FEATURE_COLUMNS))

import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import pipeline as P
from deposit_anomaly import splits as S

IDX = pd.date_range("2020-01-01", "2024-12-31")
BOUNDS = P.DEV_BOUNDARIES
PURGE = 90


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.folds = S.expanding_window_folds(IDX, BOUNDS, PURGE, end_exclusive="2024-01-01")

    def test_purge_gap_respected(self):
        for f in self.folds:
            self.assertLessEqual(f.train_end, f.val_start - pd.Timedelta(days=PURGE + 1))
            self.assertGreater(f.val_start - f.train_end, pd.Timedelta(days=PURGE))

    def test_train_strictly_before_validation(self):
        for f in self.folds:
            self.assertLess(IDX[f.train_idx].max(), IDX[f.val_idx].min())

    def test_expanding_window(self):
        sizes = [len(f.train_idx) for f in self.folds]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), len(sizes))
        for f in self.folds:
            self.assertEqual(IDX[f.train_idx][0], IDX[0])  # always starts at the first day

    def test_validation_blocks_contiguous_and_disjoint(self):
        val = np.concatenate([f.val_idx for f in self.folds])
        self.assertEqual(len(val), len(np.unique(val)))
        self.assertTrue(np.all(np.diff(val) == 1))

    def test_validation_stays_out_of_holdout(self):
        for f in self.folds:
            self.assertLess(f.val_end, pd.Timestamp("2024-01-01"))

    def test_holdout_split_is_last_year_with_purge(self):
        h = S.holdout_split(IDX, "2024-01-01", PURGE)
        self.assertEqual(IDX[h.val_idx][0], pd.Timestamp("2024-01-01"))
        self.assertEqual(IDX[h.val_idx][-1], IDX[-1])
        self.assertLessEqual(h.train_end, pd.Timestamp("2024-01-01") - pd.Timedelta(days=PURGE + 1))

    def test_zero_purge_allowed_and_negative_rejected(self):
        folds = S.expanding_window_folds(IDX, BOUNDS, 0, end_exclusive="2024-01-01")
        self.assertEqual(folds[0].train_end, pd.Timestamp(BOUNDS[0]) - pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            S.expanding_window_folds(IDX, BOUNDS, -1)

    def test_bad_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            S.expanding_window_folds(IDX, ["2022-01-01", "2021-01-01"], PURGE)
        with self.assertRaises(ValueError):
            S.expanding_window_folds(IDX, ["2020-01-10"], PURGE)  # nothing to train on

    def test_pipeline_purge_equals_feature_lookback(self):
        from deposit_anomaly import features as F
        self.assertEqual(P.PURGE_DAYS, F.LOOKBACK_DAYS)


if __name__ == "__main__":
    unittest.main()

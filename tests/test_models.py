import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import features as F
from deposit_anomaly import models as M
from deposit_anomaly.dq import DQ_REASON_QUEUE, DQ_REASON_RECON, DataQualityRouter
from deposit_anomaly import columns as C
from helpers import random_frame


def planted(n=300, seed=0):
    X = random_frame(n, seed)
    y = pd.Series((X[F.Z_TOTAL] + 0.3 * X[F.CUM5_Z] > 1.0).astype(int).to_numpy(), index=X.index)
    return X, y


class BaselineTests(unittest.TestCase):
    def test_unfitted_raises(self):
        for det in M.make_detectors().values():
            with self.assertRaises(M.NotFittedError):
                det.score(random_frame(10))

    def test_rules_thresholds_are_training_quantiles(self):
        X = random_frame()
        det = M.RulesBaseline(quantile=0.9).fit(X)
        self.assertAlmostEqual(det.thresholds[F.TOTAL_OUT], max(X[F.TOTAL_OUT].quantile(0.9), 1e-6))

    def test_rules_score_is_max_ratio_and_flags_extremes(self):
        X = random_frame()
        det = M.RulesBaseline(0.95).fit(X)
        Xt = X.copy()
        Xt.iloc[0, Xt.columns.get_loc(F.TOTAL_OUT)] = 1000.0
        s = det.score(Xt)
        self.assertEqual(int(np.argmax(s)), 0)
        self.assertGreater(s[0], 100.0)

    def test_rules_are_static_after_fit(self):
        X = random_frame()
        det = M.RulesBaseline().fit(X)
        before = det.thresholds.copy()
        det.score(X * 5)
        pd.testing.assert_series_equal(before, det.thresholds)

    def test_robust_z_is_row_max_of_z_columns(self):
        X = random_frame()
        s = M.RobustZBaseline().fit(X).score(X)
        np.testing.assert_allclose(s, X[list(F.STAT_COLUMNS)].max(axis=1).to_numpy())


class UnsupervisedTests(unittest.TestCase):
    def setUp(self):
        self.X = random_frame(400, 3)
        self.out = self.X.copy()
        self.out.iloc[10] = 12.0  # a gross multivariate outlier

    def test_isolation_forest_ranks_outlier_first(self):
        det = M.IsolationForestDetector(seed=1).fit(self.X)
        s = det.score(self.out)
        self.assertEqual(int(np.argmax(s)), 10)

    def test_isolation_forest_deterministic_by_seed(self):
        a = M.IsolationForestDetector(seed=5).fit(self.X).score(self.X)
        b = M.IsolationForestDetector(seed=5).fit(self.X).score(self.X)
        np.testing.assert_array_equal(a, b)

    def test_autoencoder_scores_outlier_higher(self):
        det = M.AutoencoderDetector(seed=1).fit(self.X)
        s = det.score(self.out)
        self.assertGreater(s[10], 5 * np.median(s))

    def test_autoencoder_uses_clean_mask_only(self):
        mask = pd.Series(np.arange(len(self.X)) % 2 == 0, index=self.X.index)
        det = M.AutoencoderDetector(seed=1).fit(self.X, clean_mask=mask)
        self.assertEqual(det.scaler.n_samples_seen_, int(mask.sum()))

    def test_autoencoder_deterministic(self):
        a = M.AutoencoderDetector(seed=2).fit(self.X).score(self.X)
        b = M.AutoencoderDetector(seed=2).fit(self.X).score(self.X)
        np.testing.assert_allclose(a, b)


class SupervisedTests(unittest.TestCase):
    def test_learns_separable_signal_out_of_sample(self):
        X, y = planted(600, 1)
        for kind in ("gbm", "rf"):
            det = M.SupervisedDetector(kind, seed=0).fit(X.iloc[:400], y.iloc[:400])
            s = det.score(X.iloc[400:])
            from sklearn.metrics import roc_auc_score
            self.assertGreater(roc_auc_score(y.iloc[400:], s), 0.9, kind)

    def test_scores_are_probabilities(self):
        X, y = planted()
        s = M.SupervisedDetector("gbm").fit(X, y).score(X)
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_deterministic_by_seed(self):
        X, y = planted()
        a = M.SupervisedDetector("rf", seed=4).fit(X, y).score(X)
        b = M.SupervisedDetector("rf", seed=4).fit(X, y).score(X)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_single_class_labels_rejected(self):
        X, _ = planted()
        with self.assertRaises(ValueError):
            M.SupervisedDetector("gbm").fit(X, pd.Series(np.zeros(len(X), int), index=X.index))

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            M.SupervisedDetector("svm")

    def test_balanced_weights_equalise_class_mass(self):
        w = M.balanced_weights(np.array([0] * 90 + [1] * 10))
        self.assertAlmostEqual(w[:90].sum(), w[90:].sum())
        self.assertAlmostEqual(w.sum(), 100.0)

    def test_registry_has_all_models(self):
        self.assertEqual(set(M.make_detectors()), {"rules", "robust_z", "isolation_forest", "autoencoder",
                                                   "gradient_boosting", "random_forest"})


class DataQualityRouterTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2022-01-01", periods=200)
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame({C.RECON_GAP: np.abs(rng.normal(0, 0.0005, 200)),
                               C.PAYMENT_QUEUE: 0.02 + np.abs(rng.normal(0, 0.01, 200))}, index=idx)
        self.router = DataQualityRouter().fit(self.X)

    def test_normal_days_not_flagged_often(self):
        self.assertLess(self.router.flag(self.X).mean(), 0.02)

    def test_recon_break_is_flagged_with_reason(self):
        X = self.X.copy()
        X.iloc[5, X.columns.get_loc(C.RECON_GAP)] = 0.5
        self.assertEqual(self.router.reasons(X).iloc[5], DQ_REASON_RECON)

    def test_queue_backlog_is_flagged_with_reason(self):
        X = self.X.copy()
        X.iloc[7, X.columns.get_loc(C.PAYMENT_QUEUE)] = 1.0
        self.assertEqual(self.router.reasons(X).iloc[7], DQ_REASON_QUEUE)


if __name__ == "__main__":
    unittest.main()

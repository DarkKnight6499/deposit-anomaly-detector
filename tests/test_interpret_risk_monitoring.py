import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E
from deposit_anomaly import features as F
from deposit_anomaly import interpret as I
from deposit_anomaly import models as M
from deposit_anomaly import monitoring as MON
from deposit_anomaly import risk as R
from deposit_anomaly.dq import DataQualityRouter
from helpers import random_frame

SIGNAL = F.Z_TOTAL
NOISE = F.BANK_DD


def fitted_gbm():
    X = random_frame(500, 2)
    y = pd.Series((X[SIGNAL] > 0.8).astype(int).to_numpy(), index=X.index)
    det = M.SupervisedDetector("gbm", seed=0).fit(X.iloc[:350], y.iloc[:350])
    return det, X.iloc[350:], y.iloc[350:].to_numpy()


class InterpretTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det, cls.X, cls.y = fitted_gbm()

    def test_permutation_importance_finds_the_driver(self):
        imp = I.permutation_importance(self.det, self.X, self.y, n_repeats=4, seed=0).set_index(I.IMP_FEATURE)
        self.assertEqual(imp[I.IMP_MEAN].idxmax(), SIGNAL)
        self.assertGreater(imp.loc[SIGNAL, I.IMP_MEAN], 0.1)
        self.assertLess(abs(imp.loc[NOISE, I.IMP_MEAN]), 0.05)

    def test_permutation_importance_deterministic_and_does_not_mutate_input(self):
        before = self.X.copy()
        a = I.permutation_importance(self.det, self.X, self.y, n_repeats=2, seed=1)
        b = I.permutation_importance(self.det, self.X, self.y, n_repeats=2, seed=1)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(before, self.X)

    @unittest.skipUnless(I.HAVE_SHAP, "shap not installed")
    def test_shap_values_are_additive(self):
        vals, base = I.shap_values(self.det, self.X)
        raw = self.det.model.decision_function(self.X[list(self.det.columns)])
        np.testing.assert_allclose(vals.sum(axis=1) + base, raw, atol=1e-4)

    @unittest.skipUnless(I.HAVE_SHAP, "shap not installed")
    def test_shap_ranks_the_driver_first(self):
        vals, _ = I.shap_values(self.det, self.X)
        self.assertEqual(I.mean_abs_shap(vals, self.det.columns).index[0], SIGNAL)

    @unittest.skipUnless(I.HAVE_SHAP, "shap not installed")
    def test_explanations_list_top_features_in_order(self):
        vals, _ = I.shap_values(self.det, self.X)
        days = self.X.index[:3]
        ex = I.explain_days(vals, self.X, self.det.columns, days, top_k=3)
        self.assertEqual(len(ex), 3)
        for r, d in zip(ex.itertuples(), days):
            p = self.X.index.get_loc(d)
            mags = [abs(vals[p, self.det.columns.index(f)]) for f in r.top_features]
            self.assertEqual(mags, sorted(mags, reverse=True))
            self.assertEqual(len(r.top_features), 3)
            self.assertIn(r.top_features[0], r.explanation)

    def test_explain_unknown_day_raises(self):
        with self.assertRaises(KeyError):
            I.explain_days(np.zeros((len(self.X), len(self.det.columns))), self.X, self.det.columns,
                           pd.DatetimeIndex(["1999-01-01"]))

    def test_partial_dependence_rises_with_the_driver(self):
        c = I.pd_curves(self.det, self.X, [SIGNAL], grid_resolution=15)[SIGNAL]
        self.assertGreater(c["pd"].iloc[-1], c["pd"].iloc[0] + 0.3)


class RiskTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = rng.normal(size=5000)

    def test_psi_identical_is_zero(self):
        self.assertAlmostEqual(R.psi(self.ref, self.ref), 0.0, places=9)

    def test_psi_grows_with_shift_and_is_non_negative(self):
        vals = [R.psi(self.ref, self.ref + d) for d in (0.0, 0.2, 0.5, 1.0, 2.0)]
        self.assertTrue(all(v >= 0 for v in vals))
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])))

    def test_psi_hand_case_two_bins(self):
        ref = np.array([0.0] * 50 + [1.0] * 50)
        cur = np.array([0.0] * 80 + [1.0] * 20)
        expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
        self.assertAlmostEqual(R.psi(ref, cur, bins=2), expected, places=9)

    def test_psi_status_thresholds(self):
        self.assertEqual(R.psi_status(0.05), "OK")
        self.assertEqual(R.psi_status(0.15), "WATCH")
        self.assertEqual(R.psi_status(0.30), "ACTION")

    def test_feature_psi_sorted_and_flags_shifted_column(self):
        a = random_frame(500, 0)
        b = a.copy()
        b[F.TOTAL_OUT] = b[F.TOTAL_OUT] + 3.0
        t = R.feature_psi(a, b)
        self.assertEqual(t["feature"].iloc[0], F.TOTAL_OUT)
        self.assertEqual(t["status"].iloc[0], "ACTION")
        self.assertTrue(t["psi"].is_monotonic_decreasing)

    def test_psi_by_period_and_score_drift_shapes(self):
        a = random_frame(400, 0)
        m = R.psi_by_period(a.iloc[:200], a.iloc[200:])
        self.assertEqual(m.shape[0], len(F.FEATURE_COLUMNS))
        sc = pd.Series(np.random.default_rng(0).normal(size=400), index=a.index)
        d = R.score_drift(sc.to_numpy()[:200], sc.iloc[200:])
        self.assertEqual(list(d.columns), ["period", "n", "mean_score", "psi_vs_reference"])


class MonitoringTests(unittest.TestCase):
    def test_severity_is_monotone_in_score(self):
        ranks = [MON.SEVERITY_RANK[MON.assign_severity(s, 1.0, 0.5, 1)] for s in np.linspace(0, 3, 61)]
        self.assertEqual(ranks, sorted(ranks))

    def test_severity_levels(self):
        self.assertEqual(MON.assign_severity(0.2, 1.0, 0.1, 0), MON.QUIET)
        self.assertEqual(MON.assign_severity(0.6, 1.0, 0.1, 0), MON.WATCH)
        self.assertEqual(MON.assign_severity(1.2, 1.0, 0.5, 1), MON.ALERT)
        self.assertEqual(MON.assign_severity(1.2, 1.0, 0.95, 1), MON.CRITICAL)
        self.assertEqual(MON.assign_severity(1.2, 1.0, 0.5, 3), MON.CRITICAL)

    def test_consecutive_run_counter(self):
        np.testing.assert_array_equal(MON._consecutive(np.array([0, 1, 1, 0, 1, 1, 1], bool)),
                                      [0, 1, 2, 0, 1, 2, 3])

    def test_alert_log_routes_data_quality_and_keeps_columns(self):
        class Fake:
            columns = F.FEATURE_COLUMNS
            model = None

            def score(self, X):
                return X[F.Z_TOTAL].to_numpy()

        X = random_frame(120, 4)
        X[C.RECON_GAP] = 0.0005
        X[C.PAYMENT_QUEUE] = 0.03
        X.iloc[10, X.columns.get_loc(F.Z_TOTAL)] = 5.0     # genuine stress day
        X.iloc[20, X.columns.get_loc(F.Z_TOTAL)] = 5.0     # same score but a reconciliation break
        X.iloc[20, X.columns.get_loc(C.RECON_GAP)] = 0.8
        X.iloc[30, X.columns.get_loc(C.RECON_GAP)] = 0.8   # quiet score, break still reported
        cal = E.PlattCalibrator().fit(X[F.Z_TOTAL].to_numpy(), (X[F.Z_TOTAL] > 1).astype(int).to_numpy())
        router = DataQualityRouter().fit(X)
        log = MON.build_alert_log(Fake(), X, 2.0, cal, router)
        self.assertEqual(list(log.columns), MON.LOG_COLUMNS)
        self.assertEqual(len(log), len(X))
        self.assertIn(log["severity"].iloc[10], (MON.ALERT, MON.CRITICAL))
        self.assertEqual(log["severity"].iloc[20], MON.DATA_QUALITY)
        self.assertIn("suppressed", log["explanation"].iloc[20])
        self.assertEqual(log["severity"].iloc[30], MON.DATA_QUALITY)
        self.assertEqual(log["route"].iloc[20], "data_quality")

    def test_drift_report_has_score_row(self):
        a = random_frame(300, 0)
        rep = MON.drift_report(a.iloc[:150], a.iloc[150:], np.arange(150.0), np.arange(150.0) + 80)
        self.assertEqual(rep["feature"].iloc[-1], "MODEL_SCORE")
        self.assertEqual(len(rep), len(F.FEATURE_COLUMNS) + 1)


if __name__ == "__main__":
    unittest.main()

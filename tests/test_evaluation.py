import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E

IDX = pd.date_range("2024-01-01", periods=60)


def events(rows):
    """rows: (id, type, start_pos, end_pos, critical_pos)."""
    return pd.DataFrame([{"event_id": i, C.EVENT_TYPE: t, C.EV_START: IDX[s], C.EV_END: IDX[e], C.EV_CRITICAL: IDX[c]}
                         for i, t, s, e, c in rows])


def alerts_at(*pos):
    a = np.zeros(len(IDX), bool)
    a[list(pos)] = True
    return a


class EventScorerTests(unittest.TestCase):
    def setUp(self):
        self.ev = events([(0, C.RUN, 10, 14, 13), (1, C.GLITCH, 30, 31, 30), (2, C.BLEED, 40, 49, 45)])
        self.sc = E.EventScorer(IDX, self.ev)

    def test_detection_and_lead_time(self):
        r = self.sc.evaluate(alerts_at(11, 12))
        pe = r["per_event"].set_index("event_id")
        self.assertTrue(pe.loc[0, "detected"])
        self.assertEqual(pe.loc[0, "lead_days"], 2)  # critical pos 13 minus first alert pos 11
        self.assertFalse(pe.loc[2, "detected"])
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["n_liquidity_events"], 2)

    def test_alert_outside_window_is_not_a_detection(self):
        r = self.sc.evaluate(alerts_at(9))
        self.assertEqual(r["missed"], 2)

    def test_late_alert_has_negative_lead(self):
        r = self.sc.evaluate(alerts_at(14))
        self.assertEqual(r["per_event"].set_index("event_id").loc[0, "lead_days"], -1)

    def test_false_alarm_blocks(self):
        # days 3 and 4 share a 7-day block; day 25 is another block
        r = self.sc.evaluate(alerts_at(3, 4, 25))
        self.assertEqual(r["fa_weeks"], 2)

    def test_grace_period_after_window(self):
        r = self.sc.evaluate(alerts_at(15, 19))   # window ends pos 14, grace covers 15..19
        self.assertEqual(r["fa_weeks"], 0)
        r = self.sc.evaluate(alerts_at(20))
        self.assertEqual(r["fa_weeks"], 1)

    def test_alert_in_data_quality_window_is_false_alarm_and_tracked(self):
        r = self.sc.evaluate(alerts_at(30))
        self.assertEqual(r["fa_weeks"], 1)
        self.assertEqual(r["fa_weeks_dq_windows"], 1)
        self.assertTrue(r["per_event"].set_index("event_id").loc[1, "detected"])
        self.assertEqual(r["n_liquidity_events"], 2)  # glitch is not a liquidity event

    def test_no_alerts_no_false_alarms(self):
        r = self.sc.evaluate(alerts_at())
        self.assertEqual((r["fa_weeks"], r["missed"]), (0, 2))

    def test_fa_per_year_scaling(self):
        r = self.sc.evaluate(alerts_at(3))
        self.assertAlmostEqual(r["fa_per_year"], 1 / (60 / 365.25))

    def test_events_outside_index_are_ignored(self):
        far = pd.DataFrame([{"event_id": 9, C.EVENT_TYPE: C.RUN, C.EV_START: pd.Timestamp("2030-01-01"),
                             C.EV_END: pd.Timestamp("2030-01-10"), C.EV_CRITICAL: pd.Timestamp("2030-01-05")}])
        sc = E.EventScorer(IDX, far)
        self.assertEqual(len(sc.events), 0)

    def test_cost_matches_components(self):
        costs = E.CostConfig(miss=10.0, false_alarm=2.0)
        a = alerts_at(11, 3, 25)
        r = self.sc.evaluate(a)
        self.assertEqual(self.sc.cost(a, costs), 10.0 * r["missed"] + 2.0 * r["fa_weeks"])


class DayMetricTests(unittest.TestCase):
    def test_precision_recall_hand_case(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        a = np.array([1, 0, 1, 0, 1, 0], bool)
        m = E.day_metrics(y, np.array([.9, .4, .8, .1, .7, .2]), a)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertEqual(m["alert_days"], 3)

    def test_pr_auc_perfect_and_degenerate(self):
        self.assertAlmostEqual(E.pr_auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])), 1.0)
        self.assertTrue(np.isnan(E.pr_auc(np.zeros(5, int), np.arange(5.0))))

    def test_pr_auc_hand_value(self):
        # ranking: pos, neg, pos -> AP = (1/1 + 2/3) / 2
        self.assertAlmostEqual(E.pr_auc(np.array([1, 0, 1]), np.array([.9, .8, .7])), (1 + 2 / 3) / 2)

    def test_no_alerts_gives_nan_precision(self):
        m = E.day_metrics(np.array([1, 0]), np.array([.2, .1]), np.array([False, False]))
        self.assertTrue(np.isnan(m["precision"]))
        self.assertEqual(m["recall"], 0.0)


class CalibrationTests(unittest.TestCase):
    def test_brier_and_ece_hand_cases(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(E.brier(y, p), 0.01)
        self.assertAlmostEqual(E.expected_calibration_error(y, p), 0.1)

    def test_perfectly_calibrated_bins(self):
        y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        p = np.where(np.arange(10) < 4, 0.5, 0.8)
        t = E.reliability_table(y, p)
        self.assertEqual(int(t["n"].sum()), 10)
        self.assertAlmostEqual(float(t.loc[t["mean_pred"] == 0.5, "frac_pos"].iloc[0]), 0.5)

    def test_platt_is_monotone_and_bounded(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=500)
        y = (s + rng.normal(0, 0.7, 500) > 0).astype(int)
        cal = E.PlattCalibrator().fit(s, y)
        p = cal.predict(np.sort(s))
        self.assertTrue(np.all(np.diff(p) >= -1e-12))
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class IntervalTests(unittest.TestCase):
    def test_clopper_pearson_known_values(self):
        lo, hi = E.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 10), places=6)
        lo, hi = E.clopper_pearson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.025 ** (1 / 10), places=6)

    def test_clopper_pearson_contains_point_estimate(self):
        lo, hi = E.clopper_pearson(3, 4)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)

    def test_poisson_zero_count_upper_bound(self):
        lo, hi = E.poisson_rate_ci(0, 1.0)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, -np.log(0.025), places=6)

    def test_poisson_exposure_scaling(self):
        a = E.poisson_rate_ci(5, 1.0)
        b = E.poisson_rate_ci(5, 2.0)
        self.assertAlmostEqual(a[0] / 2, b[0])
        self.assertAlmostEqual(a[1] / 2, b[1])

    def test_block_bootstrap_deterministic_and_brackets(self):
        rng = np.random.default_rng(2)
        y = np.r_[np.zeros(200, int), np.ones(100, int)]
        y = rng.permutation(y)
        s = y + rng.normal(0, 0.4, len(y))
        a = E.block_bootstrap_ci(y, s, n_boot=200, seed=5)
        b = E.block_bootstrap_ci(y, s, n_boot=200, seed=5)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], E.pr_auc(y, s) + 0.05)
        self.assertGreaterEqual(a[1], E.pr_auc(y, s) - 0.05)
        self.assertLess(a[0], a[1])


if __name__ == "__main__":
    unittest.main()

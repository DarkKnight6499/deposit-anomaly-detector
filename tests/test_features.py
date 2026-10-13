import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import calendar as cal
from deposit_anomaly import columns as C
from deposit_anomaly import features as F
from helpers import data

CUTOFF = "2022-06-30"


class NoLookAheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = data().observed
        cls.full = F.build_features(cls.obs)

    def test_changing_future_data_changes_no_past_feature(self):
        rng = np.random.default_rng(1)
        pert = self.obs.copy()
        future = pert.index > CUTOFF
        for col in pert.columns:
            pert.loc[future, col] = pert.loc[future, col] * (1 + rng.normal(0, 0.5, future.sum()))
        other = F.build_features(pert)
        pd.testing.assert_frame_equal(self.full.loc[:CUTOFF], other.loc[:CUTOFF])

    def test_truncating_the_sample_gives_identical_past_features(self):
        trunc = F.build_features(self.obs.loc[:CUTOFF])
        pd.testing.assert_frame_equal(self.full.loc[:CUTOFF], trunc)

    def test_future_features_do_change(self):
        pert = self.obs.copy()
        pert.loc[pert.index > CUTOFF, C.flow_col("brokered")] *= 3.0
        other = F.build_features(pert)
        self.assertFalse(self.full.loc["2022-07-15":].equals(other.loc["2022-07-15":]))


class FeatureValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = data()
        cls.obs, cls.lab = d.observed, d.labels
        cls.full = F.build_features(cls.obs)
        cls.X = F.model_frame(cls.obs)

    def test_no_nan_after_warmup_and_columns_unique(self):
        self.assertFalse(self.X.isna().any().any())
        self.assertEqual(len(set(F.FEATURE_COLUMNS)), len(F.FEATURE_COLUMNS))
        self.assertEqual(list(self.X.columns), list(F.FEATURE_COLUMNS))

    def test_total_outflow_hand_check(self):
        t = pd.Timestamp("2022-03-09")
        prev = self.obs.index.get_loc(t) - 1
        bal_prev = self.obs[list(C.BALANCE_COLUMNS)].iloc[prev].sum()
        flow = self.obs.loc[t, list(C.FLOW_COLUMNS)].sum()
        self.assertAlmostEqual(self.full.loc[t, F.TOTAL_OUT], -flow / bal_prev * 100, places=10)

    def test_segment_outflow_hand_check(self):
        t = pd.Timestamp("2022-03-09")
        seg = "corp_operational"
        prev_bal = self.obs[C.balance_col(seg)].shift(1).loc[t]
        self.assertAlmostEqual(self.full.loc[t, F.out_col(seg)], -self.obs.loc[t, C.flow_col(seg)] / prev_bal * 100)

    def test_lag_features_are_shifts(self):
        pd.testing.assert_series_equal(self.full[F.lag_col(F.TOTAL_OUT, 2)], self.full[F.TOTAL_OUT].shift(2),
                                       check_names=False)

    def test_cumulative_outflow_is_rolling_sum(self):
        manual = self.full[F.TOTAL_OUT].rolling(5).sum()
        pd.testing.assert_series_equal(self.full[F.cum_col(5)], manual, check_names=False)

    def test_breadth_is_share_of_segments(self):
        b = self.X[F.BREADTH_Z2]
        self.assertTrue(((b >= 0) & (b <= 1)).all())
        self.assertTrue(np.allclose(b * len(C.SEGMENTS), np.round(b * len(C.SEGMENTS))))
        self.assertGreaterEqual(self.X[F.BREADTH_Z1].min(), 0)
        self.assertTrue((self.X[F.BREADTH_Z1] >= b).all())

    def test_outflow_z_is_high_on_run_days(self):
        run = self.lab.loc[self.X.index, C.EVENT_TYPE] == C.RUN
        none = self.lab.loc[self.X.index, C.EVENT_TYPE] == C.NO_EVENT
        self.assertGreater(self.X.loc[run, F.Z_TOTAL].mean(), 2.0)
        self.assertLess(abs(self.X.loc[none, F.Z_TOTAL].mean()), 0.5)

    def test_breadth_wider_in_run_than_normal(self):
        run = self.lab.loc[self.X.index, C.EVENT_TYPE] == C.RUN
        none = self.lab.loc[self.X.index, C.EVENT_TYPE] == C.NO_EVENT
        self.assertGreater(self.X.loc[run, F.BREADTH_Z2].mean(), 3 * self.X.loc[none, F.BREADTH_Z2].mean())

    def test_calendar_adjustment_shrinks_month_end_effect(self):
        cf = cal.calendar_frame(self.X.index)
        none = (self.lab.loc[self.X.index, C.EVENT_TYPE] == C.NO_EVENT).to_numpy()
        me = cf[cal.MONTH_END].to_numpy() & none
        col, zc = F.out_col("corp_operational"), F.z_col("corp_operational")
        raw_shift = abs(self.X.loc[me, col].mean() - self.X.loc[none, col].mean())
        sd = self.X.loc[none, col].std()
        self.assertGreater(raw_shift / sd, 0.5)           # raw flow shows a month-end effect
        self.assertLess(abs(self.X.loc[me, zc].mean()), 0.5)  # seasonal z removes most of it

    def test_velocity_hand_cases(self):
        idx = pd.date_range("2022-01-03", periods=40)
        obs = data().observed.iloc[:40].copy()
        obs.index = idx
        for s in C.SEGMENTS:  # constant balances, zero flows -> zero outflow, zero velocity
            obs[C.balance_col(s)] = 1000.0
            obs[C.flow_col(s)] = 0.0
        f = F.build_features(obs)
        self.assertAlmostEqual(float(f[F.VELOCITY].dropna().abs().max()), 0.0)
        for i, s in enumerate(C.SEGMENTS):  # linearly growing outflow -> positive velocity
            obs[C.flow_col(s)] = -np.arange(40) * 0.1
        f = F.build_features(obs)
        self.assertGreater(float(f[F.VELOCITY].dropna().iloc[-1]), 0.0)
        self.assertAlmostEqual(float(f[F.ACCEL].dropna().iloc[-1]), 0.0, places=6)


class SeasonalZTests(unittest.TestCase):
    def setUp(self):
        self.cfg = F.FeatureConfig()
        n = 400
        self.idx = pd.date_range("2022-01-03", periods=n)
        self.calf = cal.calendar_frame(self.idx)
        self.key = (self.idx.dayofweek == 4)
        rng = np.random.default_rng(0)
        self.x = pd.Series(np.where(self.idx.dayofweek == 4, 2.0, 0.0), index=self.idx) + rng.normal(0, 0.1, n)  # Friday spike

    def test_recurring_pattern_is_absorbed(self):
        z = F.seasonal_z(self.x, self.calf, self.cfg)
        fridays = z[self.key].iloc[-20:]
        self.assertLess(fridays.abs().mean(), 3.0)

    def test_unexpected_spike_is_flagged(self):
        x = self.x.copy()
        x.iloc[-1] += 3.0  # a Monday-type day gets an out-of-pattern jump
        z = F.seasonal_z(x, self.calf, self.cfg)
        self.assertGreater(z.iloc[-1], 8.0)

    def test_sign_convention_positive_is_more_outflow(self):
        x = self.x.copy()
        x.iloc[-1] -= 3.0
        z = F.seasonal_z(x, self.calf, self.cfg)
        self.assertLess(z.iloc[-1], -8.0)

    def test_lookback_constant_matches_longest_window(self):
        self.assertEqual(F.LOOKBACK_DAYS, self.cfg.sigma_window)
        self.assertGreaterEqual(F.LOOKBACK_DAYS, max(self.cfg.cum_windows))


if __name__ == "__main__":
    unittest.main()

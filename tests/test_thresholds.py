import unittest

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import evaluation as E

IDX = pd.date_range("2023-01-01", periods=400)


def make_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    spans = [(50, 60, 57), (150, 165, 160), (300, 312, 306)]
    ev = pd.DataFrame([{"event_id": i, C.EVENT_TYPE: C.RUN, C.EV_START: IDX[s], C.EV_END: IDX[e],
                        C.EV_CRITICAL: IDX[c]} for i, (s, e, c) in enumerate(spans)])
    score = rng.gamma(2.0, 1.0, len(IDX))
    for i, (s, e, _) in enumerate(spans):
        score[s:e + 1] += [2.5, 6.0, 4.0][i] + rng.normal(0, 0.5, e - s + 1)  # events of unequal visibility
    return pd.Series(score, index=IDX), E.EventScorer(IDX, ev)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.scores, self.scorer = make_case()

    def test_threshold_is_non_increasing_in_miss_cost(self):
        thr = [E.select_threshold(self.scores, self.scorer, E.CostConfig(miss=m, false_alarm=1.0))[0]
               for m in (0.5, 1, 3, 10, 30, 100, 1000)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(thr, thr[1:])), thr)

    def test_monotone_across_several_seeds(self):
        for seed in range(5):
            sc, scorer = make_case(seed)
            thr = [E.select_threshold(sc, scorer, E.CostConfig(miss=m))[0] for m in (1, 5, 25, 125, 625)]
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(thr, thr[1:])), (seed, thr))

    def test_huge_miss_cost_detects_every_event(self):
        thr, _ = E.select_threshold(self.scores, self.scorer, E.CostConfig(miss=1e6, false_alarm=1.0))
        self.assertEqual(self.scorer.evaluate(self.scores.to_numpy() >= thr)["missed"], 0)

    def test_zero_miss_cost_gives_no_false_alarms(self):
        thr, cost = E.select_threshold(self.scores, self.scorer, E.CostConfig(miss=0.0, false_alarm=1.0))
        self.assertEqual(cost, 0.0)
        self.assertEqual(self.scorer.evaluate(self.scores.to_numpy() >= thr)["fa_weeks"], 0)

    def test_reported_cost_matches_recomputation(self):
        costs = E.CostConfig(miss=50.0, false_alarm=2.0)
        thr, cost = E.select_threshold(self.scores, self.scorer, costs)
        self.assertAlmostEqual(cost, self.scorer.cost(self.scores.to_numpy() >= thr, costs))

    def test_selected_cost_is_not_worse_than_any_grid_point(self):
        costs = E.CostConfig(miss=20.0, false_alarm=1.0)
        _, cost = E.select_threshold(self.scores, self.scorer, costs)
        for t in np.quantile(self.scores, np.linspace(0, 1, 25)):
            self.assertLessEqual(cost, self.scorer.cost(self.scores.to_numpy() >= t, costs) + 1e-9)

    def test_missed_events_non_decreasing_and_false_alarms_non_increasing_in_threshold(self):
        s = self.scores.to_numpy()
        prev_missed, prev_fa = -1, 10**9
        for t in np.quantile(s, np.linspace(0, 1, 40)):
            ev = self.scorer.evaluate(s >= t)
            self.assertGreaterEqual(ev["missed"], prev_missed)
            self.assertLessEqual(ev["fa_weeks"], prev_fa)
            prev_missed, prev_fa = ev["missed"], ev["fa_weeks"]

    def test_deterministic(self):
        a = E.select_threshold(self.scores, self.scorer)
        b = E.select_threshold(self.scores, self.scorer)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly import pipeline as P
from helpers import experiment


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = experiment()

    def test_oof_covers_dev_period_contiguously_and_excludes_holdout(self):
        idx = self.e.oof_scores["rules"].index
        self.assertEqual(idx[0], pd.Timestamp("2021-07-01"))
        self.assertEqual(idx[-1], pd.Timestamp("2023-12-31"))
        self.assertTrue((idx.to_series().diff().dropna() == pd.Timedelta(days=1)).all())

    def test_holdout_is_the_last_calendar_year(self):
        idx = self.e.holdout_scores["rules"].index
        self.assertEqual((idx[0], idx[-1]), (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")))

    def test_no_training_row_within_purge_of_validation_or_holdout(self):
        for f in self.e.folds:
            self.assertLessEqual(f.train_end, f.val_start - pd.Timedelta(days=P.PURGE_DAYS + 1))
        h = self.e.holdout_fold
        self.assertLessEqual(h.train_end, pd.Timestamp(P.HOLDOUT_START) - pd.Timedelta(days=P.PURGE_DAYS + 1))

    def test_every_model_has_scores_and_threshold(self):
        for m in P.MODEL_ORDER:
            self.assertEqual(len(self.e.holdout_scores[m]), 366)
            self.assertIn(m, self.e.thresholds)
            self.assertFalse(self.e.oof_scores[m].isna().any())

    def test_holdout_event_counts(self):
        sc = self.e.holdout_scorer
        self.assertEqual(int(sc.is_liq.sum()), 4)
        self.assertEqual(int((~sc.is_liq).sum()), 5)

    def test_comparison_table_internal_consistency(self):
        t, by = P.comparison_tables(self.e, "holdout", False)
        self.assertEqual(len(t), len(P.MODEL_ORDER) + 1)
        for r in t.itertuples():
            self.assertLessEqual(r.events_detected, r.events_total)
            self.assertAlmostEqual(r.detection_rate, r.events_detected / r.events_total)
        liq = by[by[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES)].groupby("model")["alerted"].sum()
        self.assertTrue((liq.reindex(t["model"]).to_numpy() == t["events_detected"].to_numpy()).all())

    def test_dq_routing_never_adds_false_alarms_and_clears_dq_alerts(self):
        raw, _ = P.comparison_tables(self.e, "holdout", False)
        routed, _ = P.comparison_tables(self.e, "holdout", True)
        self.assertTrue((routed["fa_per_year"].to_numpy() <= raw["fa_per_year"].to_numpy() + 1e-9).all())
        self.assertTrue((routed["events_detected"].to_numpy() <= raw["events_detected"].to_numpy()).all())
        self.assertEqual(int(routed["dq_events_alerted"].sum()), 0)

    def test_supervised_beats_static_rules_on_pr_auc(self):
        t, _ = P.comparison_tables(self.e, "holdout", False)
        pr = t.set_index("model")["pr_auc"]
        self.assertGreater(pr["gradient_boosting"], pr["rules"])
        self.assertGreater(pr["random_forest"], pr["rules"])

    def test_threshold_selection_uses_only_oof_scores(self):
        from deposit_anomaly import evaluation as E
        for m in ("isolation_forest", "gradient_boosting"):
            thr, _ = E.select_threshold(self.e.oof_scores[m], self.e.oof_scorer, self.e.costs)
            self.assertEqual(thr, self.e.thresholds[m])

    def test_label_columns_never_enter_features(self):
        from deposit_anomaly import features as F
        banned = {C.EVENT_TYPE, C.EVENT_ID, C.Y_LIQUIDITY, C.Y_DQ}
        self.assertTrue(banned.isdisjoint(F.FEATURE_COLUMNS))
        self.assertTrue(banned.isdisjoint(self.e.X.columns))


if __name__ == "__main__":
    unittest.main()

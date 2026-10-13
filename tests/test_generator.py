import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from deposit_anomaly import columns as C
from deposit_anomaly.generator import EVENT_SCHEDULE, GeneratorConfig, generate
from helpers import data


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = data()

    def test_determinism_by_seed(self):
        other = generate(GeneratorConfig())
        pd.testing.assert_frame_equal(self.d.observed, other.observed)
        pd.testing.assert_frame_equal(self.d.labels, other.labels)

    def test_different_seed_differs(self):
        other = generate(replace(GeneratorConfig(), seed=99))
        self.assertFalse(np.allclose(self.d.observed[C.flow_col("corp_operational")],
                                     other.observed[C.flow_col("corp_operational")]))

    def test_span_about_five_years_daily(self):
        idx = self.d.observed.index
        self.assertEqual(len(idx), (idx[-1] - idx[0]).days + 1)
        self.assertGreater((idx[-1] - pd.Timestamp("2020-01-01")).days / 365.25, 4.9)

    def test_no_negative_balances(self):
        self.assertTrue((self.d.observed[list(C.BALANCE_COLUMNS)] > 0).all().all())

    def test_all_segments_present(self):
        self.assertEqual(len(C.SEGMENTS), 6)
        for s in C.SEGMENTS:
            self.assertIn(C.balance_col(s), self.d.observed.columns)
            self.assertIn(C.flow_col(s), self.d.observed.columns)

    def test_event_counts_match_schedule(self):
        want = pd.Series([t for t, _ in EVENT_SCHEDULE]).value_counts()
        got = self.d.events[C.EVENT_TYPE].value_counts()
        pd.testing.assert_series_equal(got.sort_index(), want.sort_index(), check_names=False)

    def test_labels_aligned_to_event_windows(self):
        lab = self.d.labels
        for r in self.d.events.itertuples():
            window = lab.loc[getattr(r, C.EV_START):getattr(r, C.EV_END)]
            self.assertTrue((window[C.EVENT_TYPE] == getattr(r, C.EVENT_TYPE)).all())
            self.assertTrue((window[C.EVENT_ID] == r.event_id).all())
        self.assertEqual(int((lab[C.EVENT_TYPE] != C.NO_EVENT).sum()), int(self.d.events["n_days"].sum()))

    def test_binary_labels_consistent_with_type(self):
        lab = self.d.labels
        self.assertTrue((lab[C.Y_LIQUIDITY] == lab[C.EVENT_TYPE].isin(C.LIQUIDITY_TYPES)).all())
        self.assertTrue((lab[C.Y_DQ] == lab[C.EVENT_TYPE].isin(C.DQ_TYPES)).all())
        self.assertEqual(int((lab[C.Y_LIQUIDITY] & lab[C.Y_DQ]).sum()), 0)

    def test_event_windows_do_not_overlap(self):
        ev = self.d.events.sort_values(C.EV_START)
        self.assertTrue((ev[C.EV_START].iloc[1:].to_numpy() > ev[C.EV_END].iloc[:-1].to_numpy()).all())

    def test_overlapping_schedule_is_rejected(self):
        sched = ((C.RUN, "2021-05-10"), (C.GLITCH, "2021-05-12"))
        with self.assertRaises(ValueError):
            generate(replace(GeneratorConfig(), schedule=sched, nuisance=()))

    def test_critical_day_inside_window(self):
        ev = self.d.events
        self.assertTrue(((ev[C.EV_CRITICAL] >= ev[C.EV_START]) & (ev[C.EV_CRITICAL] <= ev[C.EV_END])).all())

    def test_glitch_is_not_a_real_outflow(self):
        o, lab = self.d.observed, self.d.labels
        bal = o[list(C.BALANCE_COLUMNS)].sum(axis=1)
        flow = o[list(C.FLOW_COLUMNS)].sum(axis=1)
        gap = (flow - bal.diff()).iloc[1:]
        glitch = (lab[C.EVENT_TYPE] == C.GLITCH).iloc[1:]
        self.assertLess(gap[~glitch].abs().max(), 1e-6)  # ledger reconciles off glitch days
        for r in self.d.events[self.d.events[C.EVENT_TYPE] == C.GLITCH].itertuples():
            w = gap.loc[getattr(r, C.EV_START):getattr(r, C.EV_END)]
            self.assertGreater(abs(w.iloc[0]), 50.0)  # spurious record, $m
            self.assertAlmostEqual(float(w.sum()), 0.0, places=6)  # reversed next day

    def test_recon_gap_feature_flags_glitch_only(self):
        o, lab = self.d.observed, self.d.labels
        g = o[C.RECON_GAP]
        first_days = self.d.events[self.d.events[C.EVENT_TYPE] == C.GLITCH][C.EV_START]
        self.assertTrue((g.loc[first_days] > 0.1).all())
        self.assertLess(g[lab[C.EVENT_TYPE] != C.GLITCH].max(), 0.01)

    def test_outage_raises_payment_queue_and_damps_flows(self):
        o, lab = self.d.observed, self.d.labels
        q = o[C.PAYMENT_QUEUE]
        out = lab[C.EVENT_TYPE] == C.OUTAGE
        self.assertGreater(q[out].max(), 0.3)
        self.assertLess(q[lab[C.EVENT_TYPE] == C.NO_EVENT].max(), 0.2)
        delay_days = []
        for r in self.d.events[self.d.events[C.EVENT_TYPE] == C.OUTAGE].itertuples():
            delay_days += list(pd.date_range(getattr(r, C.EV_START), getattr(r, C.EV_END) - pd.Timedelta(days=1)))
        col = C.flow_col("corp_nonoperational")
        normal = o.loc[(lab[C.EVENT_TYPE] == C.NO_EVENT) & (o.index.dayofweek < 5), col].abs().mean()
        self.assertLess(o.loc[delay_days, col].abs().mean(), 0.5 * normal)

    def test_run_hits_uninsured_segments_hardest(self):
        o, lab = self.d.observed, self.d.labels
        run_out = o[lab[C.EVENT_TYPE] == C.RUN]

        def pct(s):
            return (run_out[C.flow_col(s)] / o[C.balance_col(s)].shift(1).loc[run_out.index]).mean()

        self.assertLess(pct("corp_nonoperational"), -0.005)
        self.assertLess(pct("corp_nonoperational"), pct("retail_insured"))
        self.assertGreater(pct("retail_insured"), -0.002)

    def test_run_widens_bank_cds(self):
        o, lab = self.d.observed, self.d.labels
        self.assertGreater(o.loc[lab[C.EVENT_TYPE] == C.RUN, C.BANK_CDS].mean(),
                           1.5 * o.loc[lab[C.EVENT_TYPE] == C.NO_EVENT, C.BANK_CDS].mean())

    def test_flight_to_quality_is_market_wide(self):
        o, lab = self.d.observed, self.d.labels
        m = lab[C.EVENT_TYPE] == C.FLIGHT
        self.assertGreater(o.loc[m, C.MARKET_CDS].mean(),
                           1.3 * o.loc[lab[C.EVENT_TYPE] == C.NO_EVENT, C.MARKET_CDS].mean())

    def test_top10_concentration_ordering(self):
        o, lab = self.d.observed, self.d.labels

        def mean(t):
            return o.loc[lab[C.EVENT_TYPE] == t, C.TOP10_SHARE].mean()

        self.assertGreater(mean(C.GLITCH), mean(C.RUN))
        self.assertGreater(mean(C.RUN), mean(C.FLIGHT))
        self.assertGreater(mean(C.FLIGHT), mean(C.NO_EVENT))

    def test_weekend_flows_are_quieter(self):
        o = self.d.observed
        col = C.flow_col("corp_nonoperational")
        lab_none = self.d.labels[C.EVENT_TYPE] == C.NO_EVENT
        wk = o.loc[lab_none & (o.index.dayofweek >= 5), col].std()
        bd = o.loc[lab_none & (o.index.dayofweek < 5), col].std()
        self.assertLess(wk, 0.5 * bd)

    def test_nuisance_windows_are_unlabelled_but_visible(self):
        nu, lab, o = self.d.nuisance, self.d.labels, self.d.observed
        self.assertEqual(len(nu), 8)
        for r in nu.itertuples():
            self.assertTrue((lab.loc[r.start:r.end, C.EVENT_TYPE] == C.NO_EVENT).all())
        sp = nu[nu["kind"] == "spread_bump"].iloc[0]
        self.assertGreater(o.loc[sp.start:sp.end, C.RATE_SPREAD].max(), 55.0)

    def test_run_precursors_are_in_window_before_outflows(self):
        for r in self.d.events[self.d.events[C.EVENT_TYPE] == C.RUN].itertuples():
            self.assertGreater(getattr(r, C.EV_OUTFLOW_START), getattr(r, C.EV_START))

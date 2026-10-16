import unittest
import numpy as np
import pandas as pd
from src.detector_contribution import (SECOND, QuoteGrid, causal_onsets, calibrate,
    core_scores, hash_training_sample, match_times, ratio_hac)
from src.event_audit import FEATURES, fit_detector


class DetectorContributionTests(unittest.TestCase):
    def test_score_identity_and_separate_thresholds(self):
        x = np.random.default_rng(20).normal(size=(1000, 3))
        x[:, 1] = .8*x[:, 0]+.3*x[:, 1]
        fit = fit_detector(x)
        s = core_scores(pd.DataFrame(x, columns=FEATURES), fit)
        inverse = np.linalg.inv(np.cov(x, rowvar=False))
        dx = x-x.mean(axis=0)
        direct = np.sqrt(np.einsum('ij,jk,ik->i', dx, inverse, dx))
        np.testing.assert_allclose(s[:, 0], direct)
        thresholds, counts, ties = calibrate(s)
        self.assertAlmostEqual(thresholds[0], fit['threshold'])
        self.assertTrue((counts == 1000).all())
        self.assertNotAlmostEqual(thresholds[0], thresholds[1])

    def test_onset_gap_and_unknown(self):
        sid = np.zeros(25, int); known = np.ones(25, bool)
        a = np.zeros(25, bool); a[[5, 10, 16, 20, 24]] = True
        known[18] = False
        onset, fragment, _ = causal_onsets(sid, known, a)
        np.testing.assert_array_equal(np.flatnonzero(onset), [5, 16])
        np.testing.assert_array_equal(np.flatnonzero(fragment), [5, 16, 20])

    def test_onset_chunk_invariance_and_support_change(self):
        rng = np.random.default_rng(1)
        sid = np.repeat(np.arange(10), 100)
        known = rng.random(1000) > .03
        a = (rng.random(1000) < .08) & known
        whole = causal_onsets(sid, known, a)
        carry = None; alerts = []; fragments = []
        for start in range(0, 1000, 37):
            o, f, carry = causal_onsets(sid[start:start+37], known[start:start+37], a[start:start+37], carry)
            alerts.extend(o); fragments.extend(f)
        np.testing.assert_array_equal(alerts, whole[0]); np.testing.assert_array_equal(fragments, whole[1])

    def test_onsets_against_independent_state_machine(self):
        rng = np.random.default_rng(66)
        sid = np.repeat(np.arange(4), 70)
        known = rng.random(280) > .08
        trig = (rng.random(280) < .25) & known
        clear = 0; opened = False; expected = []; fragments = []
        for i in range(len(sid)):
            if i == 0 or sid[i] != sid[i-1]:
                clear = 0; opened = False
            onset = frag = False
            if not known[i]:
                clear = 0; opened = False
            elif trig[i]:
                if not opened:
                    frag = True; onset = clear >= 5; opened = True
                clear = 0
            else:
                clear = min(5, clear+1)
                if clear == 5:
                    opened = False
            expected.append(onset); fragments.append(frag)
        a, f, _ = causal_onsets(sid, known, trig)
        np.testing.assert_array_equal(a, expected); np.testing.assert_array_equal(f, fragments)

    def make_grid(self, split=False):
        base = pd.Timestamp('2024-01-01', tz='UTC')
        times = pd.date_range(base, periods=1000, freq='s')
        spread = np.arange(1000) % 10 + 1
        quotes = pd.DataFrame({'timestamp': times, 'bid': np.ones(1000), 'ask': (100000+spread)/100000})
        if split:
            support = pd.DataFrame({'start': [base, base+pd.Timedelta(seconds=600)],
                'end_exclusive': [base+pd.Timedelta(seconds=500), base+pd.Timedelta(seconds=1000)]})
        else:
            support = pd.DataFrame({'start': [base], 'end_exclusive': [base+pd.Timedelta(seconds=1000)]})
        return QuoteGrid(quotes, support, base, base+pd.Timedelta(seconds=1000)), base, spread

    def test_outcome_strict_left_and_excludes_detection_mark(self):
        grid, base, spread = self.make_grid()
        origins = pd.DataFrame({'bin_start': [base+pd.Timedelta(seconds=350)], 'support_id': [0], 'quote_ofi': [1.]})
        result = grid.outcomes(origins).iloc[0]
        # S(s+2) uses quote s+1, not s+2; detection mark uses quote s.
        self.assertAlmostEqual(result.y300, spread[351:651].mean()/10)
        self.assertAlmostEqual(result.y60, spread[351:411].mean()/10)
        self.assertAlmostEqual(result.J+result.C, result.I)

    def test_outcome_crossing_gap_missing_and_boundary_left_limit_valid(self):
        grid, base, _ = self.make_grid(split=True)
        f = pd.DataFrame({'bin_start': [base+pd.Timedelta(seconds=s) for s in [199, 200, 650]],
                          'support_id': [0, 0, 1], 'quote_ofi': [1., 1., 0.]})
        r = grid.outcomes(f)
        self.assertTrue(r.outcome_ready.iloc[0])  # final mark exactly at support end
        self.assertFalse(r.outcome_ready.iloc[1])
        self.assertTrue(r.outcome_ready.iloc[2])
        self.assertTrue(np.isnan(r.C.iloc[2]))

    def test_invalid_quote_rejected(self):
        grid, base, _ = self.make_grid()
        quotes = pd.DataFrame({'timestamp': pd.date_range(base, periods=2, freq='s'), 'bid': [1., 1.], 'ask': [.9, 1.]})
        support = pd.DataFrame({'start': [base], 'end_exclusive': [base+pd.Timedelta(seconds=1000)]})
        with self.assertRaises(ValueError):
            QuoteGrid(quotes, support, base, base+pd.Timedelta(seconds=1000))

    def test_feature_history_is_strict_and_future_invariant(self):
        grid, base, spread = self.make_grid()
        clock = pd.DataFrame({'bin_start': pd.date_range(base, periods=1000, freq='s'),
            'recorded_update_count': np.ones(1000, int), 'flow_status_structural_only': np.full(1000, 2)})
        origin = pd.DataFrame({'bin_start': [base+pd.Timedelta(seconds=350)],
            'support_id': [0], 'recorded_update_count': [1]})
        first = grid.features(origin, clock)
        self.assertAlmostEqual(first.current_spread.iloc[0], spread[350]/10)
        self.assertAlmostEqual(first.pre_spread.iloc[0], spread[349]/10)
        self.assertAlmostEqual(first.history_spread.iloc[0], spread[50:350].mean()/10)
        grid.spread[352:] += 10000
        grid.mid[352:] += 10000
        grid.age[352:] += 10000
        pd.testing.assert_frame_equal(first, grid.features(origin, clock))

    def test_calendar_ratio_covariance_brute_pair_sum(self):
        rng = np.random.default_rng(13)
        counts = rng.integers(0, 50, (40, 3)); counts[[2, 9]] = 0
        sums = counts*rng.normal(size=(40, 3))
        means, cov, influence = ratio_hac(sums, counts, 5)
        np.testing.assert_allclose(means, sums.sum(axis=0)/counts.sum(axis=0))
        brute = np.zeros((3, 3))
        for i in range(40):
            for j in range(40):
                if abs(i-j) <= 5:
                    brute += (1-abs(i-j)/6)*np.outer(influence[i], influence[j])
        np.testing.assert_allclose(cov, brute, atol=1e-15)

    def test_hash_selection_is_chunk_independent(self):
        times = np.arange(10000, dtype=np.int64)*SECOND
        all_ = hash_training_sample(times)
        np.testing.assert_array_equal(all_, np.r_[hash_training_sample(times[:377]), hash_training_sample(times[377:])])
        self.assertTrue(.03 < all_.mean() < .07)

    def test_matching_is_one_to_one(self):
        np.testing.assert_array_equal(match_times([10, 12, 25], [11, 28]), [1, 3])


if __name__ == '__main__':
    unittest.main()

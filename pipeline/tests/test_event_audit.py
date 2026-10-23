"""Counterexamples for chronology, endpoint aggregation, and detector algebra."""
import unittest

import numpy as np
import pandas as pd

from src.event_audit import (fit_detector, score_detector, completed_duration_bins,
                             quote_diagnostics, diagnostics_tables, METRICS,
                             classify_bin_eligibility)
from src.verify_event_audit import (reference_detector, reference_duration_bins,
                                   reference_quote_diagnostics, reference_psi,
                                   compare_numeric_frame)


def quote_fixture(length=620):
    origin = pd.Timestamp('2024-04-22', tz='UTC')
    seconds = np.arange(-1, length+1)
    quotes = pd.DataFrame({'timestamp': origin+pd.to_timedelta(seconds, unit='s'),
                           'bid': 1.10+seconds*.000001,
                           'ask': 1.1001+seconds*.000001,
                           'bid_volume': 2.+(seconds % 7)*.1,
                           'ask_volume': 3.+(seconds % 5)*.1})
    bins = origin+pd.to_timedelta(np.arange(length), unit='s')
    clock = pd.DataFrame({'bin_start': bins, 'available_at': bins+pd.Timedelta(seconds=1),
                          'support_id': 0, 'flow_status': 2,
                          'quote_ofi': np.sin(np.arange(length)/13.)})
    support = pd.DataFrame({'start': [origin-pd.Timedelta(seconds=1)],
                            'end_exclusive': [origin+pd.Timedelta(seconds=length+1)]})
    return clock, quotes, support


class DetectorTests(unittest.TestCase):
    def test_raw_covariance_reference_and_frozen_future_scoring(self):
        rng = np.random.default_rng(51)
        train = rng.normal(size=(1200, 3)) @ np.array([[1., .6, .1], [.2, 2., .4], [.4, -.1, 1.]])
        later = rng.normal(size=(400, 3))*3
        fit = fit_detector(train)
        expected, scores = reference_detector(train, later)
        for key in ['mean', 'sample_std', 'covariance_z', 'threshold', 'quantile']:
            np.testing.assert_allclose(fit[key], expected[key], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(score_detector(later, fit), scores, rtol=1e-11, atol=1e-11)
        earlier = score_detector(later[:200], fit)
        poisoned = later.copy(); poisoned[200:] *= 1e8
        np.testing.assert_array_equal(score_detector(poisoned, fit)[:200], earlier)

    def test_sign_units_and_threshold_ties(self):
        x = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (25, 1))
        fitted = fit_detector(x)
        score = score_detector(x, fitted)
        self.assertEqual(int((score >= fitted['threshold']).sum()), len(x))
        changed = x*np.array([-13., 7.])+np.array([22., -43.])
        fit_changed = fit_detector(changed)
        np.testing.assert_allclose(score_detector(changed, fit_changed), score, atol=1e-12)
        np.testing.assert_allclose(fit_changed['threshold'], fitted['threshold'], atol=1e-12)

    def test_invalid_covariance_and_nonfinite_inputs_are_refused(self):
        v = np.arange(1., 301.)
        fixtures = [np.column_stack([v, v]), np.column_stack([v, np.ones(len(v))]),
                    np.column_stack([v, v+1e-8*np.sin(v)])]
        bad = np.column_stack([v, np.sin(v)]); bad[30, 0] = np.nan
        fixtures.append(bad)
        for fixture in fixtures:
            with self.subTest(kind=fixture.shape):
                with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                    fit_detector(fixture)


class DurationTests(unittest.TestCase):
    def test_partial_duration_quiet_and_unsupported_stay_distinct(self):
        status = np.array([0, 3, 2, 1, 2, 2, 4])
        ready = np.array([False, True, False, True, True, True, True])
        recorded = np.array([0, 2, 2, 0, 3, 3, 1])
        completed = np.array([0, 2, 2, 0, 2, 3, 1])
        np.testing.assert_array_equal(classify_bin_eligibility(status, ready, recorded, completed),
                                      [0, 1, 2, 3, 4, 5, 1])
        with self.assertRaises(ValueError):
            classify_bin_eligibility([2], [True], [1], [2])

    def test_endpoint_assignment_mean_before_log_and_censor_exclusion(self):
        origin = pd.Timestamp('2024-04-22', tz='UTC')
        path = pd.DataFrame({'origin_quote_id': np.arange(5),
                             'origin_at': origin+pd.to_timedelta([0, .1, .7, 1., 1.3], unit='s'),
                             'end_at': origin+pd.to_timedelta([.1, .7, 1., 1.3, 1.8], unit='s'),
                             'normalized_exposure': [1., 9., 2., 8., 9999.],
                             'event_observed': [True, True, True, True, False]})
        predictions = pd.DataFrame({'origin_quote_id': np.arange(5), 'psi': np.ones(5)})
        actual = completed_duration_bins(path, predictions)
        expected = reference_duration_bins(path, predictions.psi)
        compare_numeric_frame(actual, expected)
        self.assertEqual(actual.completed_residual_count.tolist(), [2, 2])
        self.assertEqual(actual.mean_duration_residual.tolist(), [5., 5.])
        self.assertNotAlmostEqual(np.log(actual.mean_duration_residual.iloc[0]), np.mean(np.log([1., 9.])))
        self.assertEqual(actual.bin_start.iloc[1], origin+pd.Timedelta(seconds=1))

    def test_reference_recursion_against_scalar_state_with_resets(self):
        path = pd.DataFrame({'normalized_exposure': [1., 3., 8., 2., 4., 9.],
                             'recurrence_reset': [True, False, False, True, False, False]})
        parameters = {'omega': .1, 'alpha': .2, 'beta': .6}
        expected = []
        for i, row in path.iterrows():
            expected.append(.5 if row.recurrence_reset else .1+.2*path.normalized_exposure.iloc[i-1]+.6*expected[-1])
        np.testing.assert_allclose(reference_psi(path, parameters), expected, atol=1e-14)


class QuoteTimingTests(unittest.TestCase):
    def test_strict_boundary_asof_and_history_excludes_current_bin(self):
        clock, quotes, support = quote_fixture()
        actual = quote_diagnostics(clock, quotes, support, policy='structural_only')
        expected = reference_quote_diagnostics(clock, quotes, support)
        compare_numeric_frame(actual, expected)
        self.assertEqual(actual.pre_quote_age_sec.iloc[0], 1.)
        self.assertEqual(actual.end_quote_age_sec.iloc[0], 1.)
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[:300].isna().all())
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[300:].notna().all())
        s = 400
        # direct scalar reconstruction has neither the production rolling sums
        # nor the vector reference's five-phase grouping.
        marks = []
        for lag in range(300, -1, -5):
            cutoff = clock.bin_start.iloc[s]-pd.Timedelta(seconds=lag)
            row = quotes.loc[quotes.timestamp < cutoff].iloc[-1]
            marks.append(np.log((row.bid+row.ask)/2))
        self.assertAlmostEqual(actual.pre_rv_300s_5s_bps.iloc[s],
                               np.sqrt(np.sum(np.diff(marks)**2))*10000, places=11)
        self.assertAlmostEqual(actual.pre_quote_ofi_300s.iloc[s],
                               sum(clock.quote_ofi.iloc[s-300:s]), places=11)
        changed_clock = clock.copy(); changed_clock.loc[s:, 'quote_ofi'] = 1e8
        changed_quotes = quotes.copy()
        current = changed_quotes.timestamp >= clock.bin_start.iloc[s]
        changed_quotes.loc[current, ['bid', 'ask']] *= 2
        changed = quote_diagnostics(changed_clock, changed_quotes, support, policy='structural_only')
        for column in ['pre_spread_pips', 'pre_displayed_size', 'pre_quote_age_sec',
                       'pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']:
            np.testing.assert_allclose(changed[column].iloc[:s+1], actual[column].iloc[:s+1], equal_nan=True, err_msg=column)
        self.assertNotEqual(changed.end_spread_pips.iloc[s], actual.end_spread_pips.iloc[s])

    def test_invalid_history_support_break_and_partial_first_mark(self):
        clock, quotes, support = quote_fixture()
        clock.loc[50, 'flow_status'] = 3
        clock.loc[50, 'quote_ofi'] = np.nan
        actual = quote_diagnostics(clock, quotes, support, policy='structural_only')
        self.assertTrue(actual.pre_quote_ofi_300s.iloc[300:351].isna().all())
        self.assertTrue(np.isfinite(actual.pre_quote_ofi_300s.iloc[351]))
        # a quote before the support start cannot supply the first history mark.
        clock, quotes, support = quote_fixture()
        support.loc[0, 'start'] = clock.bin_start.iloc[0]
        actual = quote_diagnostics(clock, quotes, support, policy='structural_only')
        self.assertTrue(np.isnan(actual.pre_spread_pips.iloc[0]))
        self.assertTrue(np.isnan(actual.pre_rv_300s_5s_bps.iloc[300]))
        self.assertTrue(np.isnan(actual.pre_quote_ofi_300s.iloc[300]))
        self.assertTrue(np.isfinite(actual.pre_rv_300s_5s_bps.iloc[301]))
        # start a new support interval; 300 seconds of history must not bridge it.
        clock, quotes, support = quote_fixture()
        split = clock.bin_start.iloc[310]
        support = pd.DataFrame({'start': [support.start.iloc[0], split],
                                'end_exclusive': [split, support.end_exclusive.iloc[0]]})
        clock.loc[310:, 'support_id'] = 1
        actual = quote_diagnostics(clock, quotes, support, policy='structural_only')
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[310:611].isna().all())
        self.assertTrue(np.isfinite(actual.pre_rv_300s_5s_bps.iloc[611]))


class DescriptiveComparisonTests(unittest.TestCase):
    def test_metric_specific_event_weights_and_unmatched_cells(self):
        origin = pd.Timestamp('2024-04-22', tz='UTC')
        seconds = [0, 1, 2, 3, 900, 901, 902, 1800, 1801]
        f = pd.DataFrame({'bin_start': origin+pd.to_timedelta(seconds, unit='s'),
                          'period': 'development',
                          'candidate_event': [True, True, False, False, True, False, False, True, True],
                          'candidate_without_duration': False, 'distance': 1.,
                          'z_ofi': 0., 'z_intensity': 0., 'z_duration': 0.})
        for metric in METRICS:
            f[metric] = [10., 20., 0., 0., 100., 50., 50., 999., np.nan]
        row = diagnostics_tables(f)['economic_comparison'].iloc[0]
        self.assertEqual(row.matched_events, 3)
        self.assertEqual(row.unmatched_events, 1)
        self.assertEqual(row.event_missing, 1)
        self.assertAlmostEqual(row.matched_event_mean, 130/3)
        self.assertAlmostEqual(row.event_weighted_control_mean, 50/3)
        self.assertAlmostEqual(row.matched_difference, 80/3)


if __name__ == '__main__':
    unittest.main()

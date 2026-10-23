"""Independent small examples for past-only variation and cancellation audits."""
import unittest

import numpy as np
import pandas as pd

from src.state_measurement import (build_windows, measure_windows,
                                   summarize_measurements, direction_diagnostics,
                                   direction_summary)


ORIGIN = pd.Timestamp('2024-04-22 00:00:00', tz='UTC')
FREQUENCIES = [1, 5, 10, 15, 30, 60]


def at(seconds):
    return pd.Timestamp(ORIGIN.value + round(float(seconds) * 1_000_000_000),
                        unit='ns', tz='UTC')


def onsets(ends=(300,), nets=(1.,), confirmed=None):
    size = len(ends)
    return pd.DataFrame({
        'episode_id': np.arange(size, dtype='int64'),
        'bin_start': pd.DatetimeIndex([at(t) for t in ends]),
        'available_at': pd.DatetimeIndex([at(t + 1) for t in ends]),
        'support_id': np.zeros(size, dtype='int64'),
        'quote_ofi': np.asarray(nets, dtype=float),
        'onset_confirmed': np.ones(size, dtype=bool) if confirmed is None else np.asarray(confirmed, dtype=bool),
        'state_history_available': np.zeros(size, dtype=bool),
        'pre_rv_300s_5s_bps': np.full(size, np.nan),
    })


def quotes(seconds, mids):
    values = np.asarray(mids, dtype=float)
    return pd.DataFrame({
        'timestamp': pd.DatetimeIndex([at(t) for t in seconds], tz='UTC'),
        'bid': values - .00001,
        'ask': values + .00001,
    })


def support(starts=(-3600,), ends=(3600,)):
    return pd.DataFrame({
        'start': pd.DatetimeIndex([at(t) for t in starts]),
        'end_exclusive': pd.DatetimeIndex([at(t) for t in ends]),
    })


def ticks(seconds, values, codes=None):
    return pd.DataFrame({
        'timestamp': pd.DatetimeIndex([at(t) for t in seconds], tz='UTC'),
        'quote_ofi': np.asarray(values, dtype=float),
        'exclusion_code': np.zeros(len(seconds), dtype='uint8') if codes is None else np.asarray(codes, dtype='uint8'),
    })


def dev_windows(events=None, intervals=None):
    events = onsets() if events is None else events
    intervals = support() if intervals is None else intervals
    all_windows = build_windows(events, intervals, training_start=at(-300), parameter_cutoff=at(0))
    return all_windows.loc[all_windows.kind.eq('development')].reset_index(drop=True)


def analytic_quotes(rate=1e-5, last=300):
    marks = np.arange(last + 1)
    return quotes(marks - .1, np.exp(.1 + rate * marks))


def measure(q=None, events=None, intervals=None):
    intervals = support() if intervals is None else intervals
    q = analytic_quotes() if q is None else q
    return measure_windows(dev_windows(events, intervals), q, intervals)


class WindowConstructionTests(unittest.TestCase):
    def test_deterministic_calibration_endpoints_and_development_conservation(self):
        o = onsets([300, 401], [1., -1.], confirmed=[True, False])
        w = build_windows(o, support(), training_start=at(-900), parameter_cutoff=at(0))
        self.assertEqual(w.loc[w.kind.eq('calibration'), 'end_at'].tolist(), [at(-600), at(-300), at(0)])
        d = w.loc[w.kind.eq('development')]
        self.assertEqual(d.end_at.tolist(), o.bin_start.tolist())
        self.assertEqual(d.episode_id.tolist(), o.episode_id.tolist())
        self.assertEqual(d.onset_confirmed.tolist(), [True, False])
        self.assertFalse(w.window_id.duplicated().any())

    def test_calendar_endpoints_without_support_are_retained(self):
        s = support([-1000, -200], [-700, 3600])
        o = onsets()
        o['support_id'] = 1
        w = build_windows(o, s, training_start=at(-900), parameter_cutoff=at(0))
        self.assertEqual(len(w.loc[w.kind.eq('calibration')]), 3)
        self.assertEqual(w.loc[w.kind.eq('calibration'), 'support_id'].tolist(), [-1, -1, 1])

    def test_calibration_left_limit_uses_previous_interval_at_adjacent_start(self):
        s = support([-1000, 0], [0, 3600])
        o = onsets()
        o['support_id'] = 1
        w = build_windows(o, s, training_start=at(-300), parameter_cutoff=at(0))
        c = w.loc[w.kind.eq('calibration')].iloc[0]
        self.assertEqual(c.support_id, 0)
        self.assertEqual(c.end_at, at(0))

    def test_invalid_calibration_boundaries_and_early_onset_rejected(self):
        for start, end in [(0, 0), (0, -300), (-301, 0), (-300, 1)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    build_windows(onsets(), support(), training_start=at(start), parameter_cutoff=at(end))
        with self.assertRaises(ValueError):
            build_windows(onsets([-1], [1.]), support(), training_start=at(-300), parameter_cutoff=at(0))


class VariationMeasurementTests(unittest.TestCase):
    def test_analytic_loglinear_path_checks_units_and_return_counts(self):
        rate = 1e-5
        m = measure(q=analytic_quotes(rate)).set_index('frequency_sec')
        self.assertEqual(m.index.tolist(), FREQUENCIES)
        for f in FREQUENCIES:
            expected_v = 300 * f * rate ** 2 * 1e8
            self.assertEqual(m.loc[f, 'n_returns'], 300 // f)
            self.assertEqual(m.loc[f, 'status'], 'available')
            self.assertAlmostEqual(m.loc[f, 'rv_bps2'], expected_v, places=8)
            self.assertAlmostEqual(m.loc[f, 'rv_bps'], np.sqrt(expected_v), places=8)
            self.assertAlmostEqual(m.loc[f, 'max_mark_age_sec'], .1)
            self.assertEqual(m.loc[f, 'repeat_fraction'], 0.)
            self.assertEqual(m.loc[f, 'zero_return_fraction'], 0.)

    def test_quote_exactly_at_window_end_is_excluded(self):
        q = analytic_quotes()
        future = pd.concat([q, quotes([300, 300.1, 400], [4., 5., 6.])], ignore_index=True)
        pd.testing.assert_frame_equal(measure(q=q), measure(q=future))

    def test_strict_initial_boundary_and_subsequent_grid_boundary(self):
        q = quotes([-1, 0, 1, 300], [1., 2., 4., 10.])
        m = measure(q=q).set_index('frequency_sec')
        self.assertEqual(m.loc[1, 'first_quote_at'], at(-1))
        self.assertEqual(m.loc[1, 'last_quote_at'], at(1))
        self.assertAlmostEqual(m.loc[1, 'rv_bps2'], 2 * np.log(2) ** 2 * 1e8, places=5)
        self.assertAlmostEqual(m.loc[5, 'rv_bps2'], np.log(4) ** 2 * 1e8, places=5)

    def test_zero_variation_is_not_replaced_by_floor(self):
        m = measure(q=quotes([-1], [1.1]))
        self.assertTrue(m.rv_bps2.eq(0).all())
        self.assertTrue(m.rv_bps.eq(0).all())
        self.assertTrue(m.repeat_fraction.eq(1).all())
        self.assertTrue(m.zero_return_fraction.eq(1).all())
        self.assertTrue(m.max_mark_age_sec.eq(301).all())

    def test_zero_returns_and_repeated_recorded_quotes_are_distinct(self):
        q = quotes(np.arange(301) - .1, np.full(301, 1.1))
        m = measure(q=q)
        self.assertTrue(m.rv_bps.eq(0).all())
        self.assertTrue(m.zero_return_fraction.eq(1).all())
        self.assertTrue(m.repeat_fraction.eq(0).all())

    def test_missing_initial_quote_retains_every_frequency_with_missing_values(self):
        m = measure(q=quotes([0, 1, 299], [1.1, 1.2, 1.3]))
        self.assertEqual(len(m), 6)
        self.assertTrue(m.status.eq('no_prior_initial_quote').all())
        self.assertTrue(m[['rv_bps', 'rv_bps2', 'max_mark_age_sec', 'repeat_fraction']].isna().all().all())
        self.assertTrue(m.first_quote_at.isna().all())
        self.assertFalse(m.common_age_le_60.any())

    def test_initial_quote_outside_interval_is_not_carried_through_boundary(self):
        m = measure(q=quotes([-1, 1, 299], [1.1, 1.2, 1.3]), intervals=support([0], [3600]))
        self.assertTrue(m.status.eq('initial_quote_outside_support').all())
        self.assertTrue(m.rv_bps.isna().all())

    def test_window_crosses_support_even_with_quotes_at_both_ends(self):
        s = support([-100, 200], [150, 3600])
        o = onsets()
        o['support_id'] = 1
        m = measure(q=quotes([-1, 210, 299], [1.1, 1.2, 1.3]), events=o, intervals=s)
        self.assertTrue(m.status.eq('window_crosses_support').all())

    def test_exact_support_end_is_valid_for_calibration_left_limit(self):
        s = support([-100], [300])
        w = build_windows(onsets().iloc[:0], s, training_start=at(0), parameter_cutoff=at(300))
        m = measure_windows(w, analytic_quotes(), s)
        self.assertTrue(m.status.eq('available').all())
        self.assertTrue(m.last_quote_at.eq(at(299.9)).all())

    def test_pretraining_initial_quote_is_permitted_as_boundary_initializer(self):
        s = support()
        w = build_windows(onsets().iloc[:0], s, training_start=at(0), parameter_cutoff=at(300))
        m = measure_windows(w, analytic_quotes(), s)
        self.assertTrue(m.status.eq('available').all())
        self.assertTrue(m.first_quote_at.eq(at(-.1)).all())

    def test_inherited_five_second_reference_is_checked_without_replacement(self):
        o = onsets()
        o['state_history_available'] = True
        o['pre_rv_300s_5s_bps'] = np.sqrt(15.)
        m = measure(events=o)
        self.assertAlmostEqual(m.loc[m.frequency_sec.eq(5), 'rv_bps'].iloc[0], np.sqrt(15.))
        o['pre_rv_300s_5s_bps'] = 99.
        with self.assertRaisesRegex(ValueError, 'reconciliation'):
            measure(events=o)

    def test_duplicate_quote_times_and_invalid_prices_fail(self):
        for q in [quotes([-1, -1], [1.1, 1.2]), quotes([0, -1], [1.1, 1.2]), quotes([-1], [np.nan])]:
            with self.subTest(quotes=q.to_dict()):
                with self.assertRaises(ValueError):
                    measure(q=q)


class AgeComparisonTests(unittest.TestCase):
    def test_age_cap_equality_passes_and_one_nanosecond_older_fails(self):
        for cap in [15, 30, 60]:
            with self.subTest(cap=cap):
                q = quotes(np.r_[-cap, np.arange(300) + .5], np.ones(301))
                good = measure(q=q)
                self.assertTrue(good[f'common_age_le_{cap}'].all())
                q.loc[0, 'timestamp'] = pd.Timestamp(q.timestamp.iloc[0].value - 1, unit='ns', tz='UTC')
                bad = measure(q=q)
                self.assertFalse(bad[f'common_age_le_{cap}'].any())

    def test_common_grid_screen_prevents_frequency_specific_sample_changes(self):
        times = np.r_[-.1, .1, np.arange(19, 300) + .9]
        m = measure(q=quotes(times, np.ones(len(times)))).set_index('frequency_sec')
        self.assertFalse(m.loc[1, 'own_age_le_15'])
        self.assertTrue(m.loc[5, 'own_age_le_15'])
        self.assertFalse(m.common_age_le_15.any())
        result = summarize_measurements(m.reset_index())
        s = result['summary'].set_index(['frequency_sec', 'sample'])
        self.assertEqual(s.loc[(5, 'common_age_15'), 'own_grid_age_pass_windows'], 1)
        self.assertEqual(s.loc[(5, 'common_age_15'), 'selected_windows'], 0)
        c = result['comparisons'].loc[lambda x: x['sample'].eq('common_age_15')]
        self.assertTrue(c.paired_windows.eq(0).all())


def synthetic_measurements():
    """Known window-level values isolate aggregation from quote construction."""
    rows = []
    reference = [0., 1., 1., 3.]
    alternate = [0., 2., 2., 6.]
    for window_id in range(4):
        for f in FREQUENCIES:
            r = alternate[window_id] if f == 1 else reference[window_id]
            rows.append({'window_id': window_id, 'kind': 'development',
                'end_at': at(300 + window_id * 300), 'episode_id': window_id,
                'onset_confirmed': True, 'frequency_sec': f, 'status': 'available',
                'rv_bps': r, 'rv_bps2': r*r, 'repeat_fraction': .25,
                'zero_return_fraction': .5, 'max_mark_age_sec': 1.,
                **{f'{prefix}_age_le_{cap}': window_id != 3 if cap == 15 else True
                   for prefix in ['own', 'common'] for cap in [15, 30, 60]}})
    return pd.DataFrame(rows)


class MeasurementSummaryTests(unittest.TestCase):
    def test_variance_means_sqrt_means_ties_and_zero_pairs(self):
        out = summarize_measurements(synthetic_measurements())
        s = out['summary'].set_index(['frequency_sec', 'sample'])
        self.assertEqual(s.loc[(5, 'all_available'), 'mean_variance_bps2'], 11/4)
        self.assertEqual(s.loc[(5, 'all_available'), 'mean_rv_bps'], 5/4)
        self.assertNotEqual(np.sqrt(11/4), s.loc[(5, 'all_available'), 'mean_rv_bps'])
        c = out['comparisons'].set_index(['frequency_sec', 'sample'])
        self.assertEqual(c.loc[(1, 'all_available'), 'paired_windows'], 4)
        self.assertEqual(c.loc[(1, 'all_available'), 'mean_variance_ratio_to_5s'], 4.)
        self.assertEqual(c.loc[(1, 'all_available'), 'mean_absolute_rv_difference_bps'], 1.25)
        self.assertAlmostEqual(c.loc[(1, 'all_available'), 'spearman_rv_to_5s'], 1.)
        self.assertEqual(c.loc[(1, 'all_available'), 'zero_variance_windows'], 1)
        self.assertEqual(c.loc[(1, 'all_available'), 'reference_zero_variance_windows'], 1)
        self.assertEqual(c.loc[(1, 'common_age_15'), 'paired_windows'], 3)

    def test_all_zero_reference_has_no_ratio_or_rank_correlation(self):
        m = synthetic_measurements()
        m[['rv_bps', 'rv_bps2']] = 0.
        out = summarize_measurements(m)
        self.assertTrue(out['comparisons'].mean_variance_ratio_to_5s.isna().all())
        self.assertTrue(out['comparisons'].spearman_rv_to_5s.isna().all())
        self.assertTrue(out['comparisons'].mean_absolute_rv_difference_bps.eq(0).all())

    def test_unconfirmed_fragments_remain_in_counts_but_not_descriptive_cohort(self):
        m = synthetic_measurements()
        m.loc[m.window_id.eq(3), 'onset_confirmed'] = False
        out = summarize_measurements(m)
        s = out['summary'].set_index(['frequency_sec', 'sample']).loc[(5, 'all_available')]
        self.assertEqual(s.windows, 4)
        self.assertEqual(s.cohort_windows, 3)
        self.assertEqual(s.selected_windows, 3)
        self.assertAlmostEqual(s.mean_rv_bps, 2/3)

    def test_calibration_and_development_are_not_pooled(self):
        m = synthetic_measurements()
        cal = m.copy()
        cal['kind'] = 'calibration'
        cal['onset_confirmed'] = False
        cal['rv_bps'] *= 3
        cal['rv_bps2'] *= 9
        out = summarize_measurements(pd.concat([m, cal], ignore_index=True))
        s = out['summary'].set_index(['kind', 'frequency_sec', 'sample'])
        self.assertEqual(s.loc[('development', 5, 'all_available'), 'mean_rv_bps'], 1.25)
        self.assertEqual(s.loc[('calibration', 5, 'all_available'), 'mean_rv_bps'], 3.75)
        self.assertEqual(set(out['daily'].kind), {'calibration', 'development'})
        self.assertEqual(set(out['hourly'].kind), {'calibration', 'development'})

    def test_mismatched_frequency_sample_is_rejected(self):
        m = synthetic_measurements()
        m = m.loc[~(m.window_id.eq(0) & m.frequency_sec.eq(1))]
        with self.assertRaisesRegex(ValueError, 'samples differ'):
            summarize_measurements(m)


class DirectionCancellationTests(unittest.TestCase):
    def test_half_open_onset_bin_includes_start_and_excludes_end(self):
        o = onsets([10], [.5])
        t = ticks([9.9, 10, 10.5, 11], [100., 1., -.5, -999.])
        d = direction_diagnostics(o, t).iloc[0]
        self.assertEqual(d.tick_contributions, 2)
        self.assertEqual(d.stable_net, .5)
        self.assertEqual(d.gross_contributions, 1.5)
        self.assertAlmostEqual(d.cancellation_ratio, 1/3)
        self.assertTrue(d.direction_agrees)

    def test_zero_gross_is_undefined_but_exact_cancellation_ratio_is_zero(self):
        o = onsets([10, 20], [0., 0.])
        t = ticks([10, 10.5, 20, 20.5], [0., 0., 1., -1.])
        d = direction_diagnostics(o, t)
        self.assertTrue(np.isnan(d.cancellation_ratio.iloc[0]))
        self.assertEqual(d.cancellation_ratio.iloc[1], 0.)
        s = direction_summary(d)['all_onsets']
        self.assertEqual(s['gross_zero'], 1)
        self.assertEqual(s['zero_ratio'], 1)
        self.assertEqual(s['stable_net_zero'], 2)
        self.assertEqual(s['positive_ratio_le_10pct'], 0)

    def test_positive_cancellation_bands_do_not_count_zero_or_zero_gross(self):
        o = onsets([10, 20, 30, 40, 50], [0., 0., 1., .5, 1.])
        t = ticks([10, 20, 20.5, 30, 30.5, 40, 40.5, 50, 50.5],
                  [0., 1., -1., 100., -99., 10., -9.5, 10., -9.])
        s = direction_summary(direction_diagnostics(o, t))['all_onsets']
        self.assertEqual(s['ratio_available'], 4)
        self.assertEqual(s['positive_ratio'], 3)
        self.assertEqual(s['positive_ratio_le_1pct'], 1)
        self.assertEqual(s['positive_ratio_le_5pct'], 2)
        self.assertEqual(s['positive_ratio_le_10pct'], 3)

    def test_stable_sum_sign_difference_is_retained_within_error_bound(self):
        o = onsets([10], [0.])
        before = o.copy(deep=True)
        t = ticks([10, 10.1, 10.2], [1e16, 1., -1e16])
        d = direction_diagnostics(o, t).iloc[0]
        self.assertEqual(d.stable_net, 1.)
        self.assertEqual(d.inherited_net, 0.)
        self.assertEqual(d.inherited_direction, 0)
        self.assertEqual(d.stable_direction, 1)
        self.assertFalse(d.direction_agrees)
        self.assertLessEqual(abs(d.aggregation_difference), d.summation_error_bound)
        pd.testing.assert_frame_equal(o, before)

    def test_difference_beyond_numerical_bound_stops_reconciliation(self):
        with self.assertRaisesRegex(ValueError, 'beyond summation bound'):
            direction_diagnostics(onsets([10], [2.]), ticks([10, 10.5], [1., -.5]))

    def test_invalid_missing_or_nonfinite_contributions_are_not_silently_dropped(self):
        cases = [ticks([9], [1.]), ticks([10], [1.], [1]), ticks([10], [np.nan])]
        for t in cases:
            with self.subTest(ticks=t.to_dict()):
                with self.assertRaises(ValueError):
                    direction_diagnostics(onsets([10], [1.]), t)

    def test_recorded_count_mismatch_fails(self):
        o = onsets([10], [1.])
        o['recorded_update_count'] = 2
        with self.assertRaisesRegex(ValueError, 'update count'):
            direction_diagnostics(o, ticks([10], [1.]))

    def test_empty_direction_sample_has_usable_zero_count_summary(self):
        d = direction_diagnostics(onsets().iloc[:0], ticks([], []))
        self.assertEqual(len(d), 0)
        s = direction_summary(d)
        self.assertEqual(s['all_onsets']['onsets'], 0)
        self.assertEqual(s['confirmed_onsets']['ratio_available'], 0)


if __name__ == '__main__':
    unittest.main()

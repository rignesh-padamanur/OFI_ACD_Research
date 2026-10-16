"""Independent boundary and numerical examples for fixed-onset state checks."""
import math
import unittest

import numpy as np
import pandas as pd

from src.panel_measurement import (FREQUENCIES, make_windows, measure_windows,
                                   summarize, MEASUREMENT_COLUMNS)


ORIGIN = pd.Timestamp('2024-04-22', tz='UTC')


def at(seconds):
    return pd.Timestamp(ORIGIN.value + round(float(seconds) * 1_000_000_000), unit='ns', tz='UTC')


def onsets(ends=(300,)):
    n = len(ends)
    return pd.DataFrame({'episode_id': np.arange(n, dtype='int64'),
        'bin_start': pd.DatetimeIndex([at(x) for x in ends], tz='UTC'),
        'available_at': pd.DatetimeIndex([at(x + 1) for x in ends], tz='UTC'),
        'support_id': np.zeros(n, dtype='int64'), 'onset_confirmed': np.ones(n, dtype=bool),
        'state_history_available': np.zeros(n, dtype=bool),
        'state_sample_eligible': np.zeros(n, dtype=bool),
        'pre_rv_300s_5s_bps': np.full(n, np.nan)})


def support(starts=(-3600,), ends=(3600,)):
    return pd.DataFrame({'start': pd.DatetimeIndex([at(x) for x in starts], tz='UTC'),
        'end_exclusive': pd.DatetimeIndex([at(x) for x in ends], tz='UTC')})


def quotes(seconds, mids):
    mids = np.asarray(mids, dtype=float)
    return pd.DataFrame({'timestamp': pd.DatetimeIndex([at(x) for x in seconds], tz='UTC'),
                         'bid': mids - 1e-5, 'ask': mids + 1e-5})


def analytic_quotes(rate=1e-5, last=300):
    steps = np.arange(last + 1)
    return quotes(steps - .1, np.exp(.1 + steps * rate))


def measure(q=None, o=None, s=None, batch_size=512):
    return measure_windows(make_windows(onsets() if o is None else o),
        analytic_quotes() if q is None else q, support() if s is None else s, batch_size=batch_size)


def synthetic_measurements():
    m = measure(q=analytic_quotes(last=1200), o=onsets([300, 600, 900, 1200]))
    reference = np.array([0., 1., 1., 3.])
    m['rv_bps'] = reference[m.window_id] * np.where(m.frequency_sec.eq(15), 2., 1.)
    m['rv_bps2'] = m.rv_bps ** 2
    m['max_mark_age_sec'] = np.where(m.window_id.eq(3), 31., 1.)
    m['common_max_mark_age_sec'] = m.max_mark_age_sec
    for cap in [15, 30, 60]:
        m[f'own_age_le_{cap}'] = m.max_mark_age_sec.le(cap)
        m[f'common_age_le_{cap}'] = m.common_max_mark_age_sec.le(cap)
    return m


class WindowTests(unittest.TestCase):
    def test_all_onsets_kept_with_same_episode_identity_and_flags(self):
        o = onsets([300, 400])
        o['episode_id'] = [100, 104]
        o.loc[1, 'onset_confirmed'] = False
        w = make_windows(o)
        self.assertEqual(w.window_id.tolist(), [100, 104])
        self.assertEqual(w.episode_id.tolist(), [100, 104])
        self.assertEqual(w.end_at.tolist(), o.bin_start.tolist())
        self.assertEqual(w.onset_confirmed.tolist(), [True, False])
        self.assertTrue(w.kind.eq('chronological_onset').all())

    def test_bad_onset_time_rejected(self):
        for ends in [[300, 300], [400, 300], [300.1]]:
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                make_windows(onsets(ends))
        o = onsets()
        o['available_at'] = o.bin_start
        with self.assertRaisesRegex(ValueError, 'available_at'):
            make_windows(o)

    def test_bad_id_support_and_flags_rejected(self):
        for column, value in [('episode_id', -1), ('episode_id', 1.),
                              ('support_id', -1), ('onset_confirmed', 1)]:
            o = onsets()
            o[column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                make_windows(o)
        o = onsets([300, 400])
        o['episode_id'] = 1
        with self.assertRaises(ValueError):
            make_windows(o)

    def test_inherited_eligibility_and_reference_are_validated(self):
        o = onsets()
        o['state_sample_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            make_windows(o)
        o['state_history_available'] = True
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            make_windows(o)
        o['pre_rv_300s_5s_bps'] = -1.
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            make_windows(o)

    def test_naive_timezone_rejected(self):
        o = onsets()
        o['bin_start'] = o.bin_start.dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, 'UTC'):
            make_windows(o)

    def test_invalid_window_edit_cannot_bypass_identity_checks(self):
        w = make_windows(onsets())
        w['window_id'] = 5
        with self.assertRaisesRegex(ValueError, 'identity'):
            measure_windows(w, analytic_quotes(), support())


class StrictMarkTests(unittest.TestCase):
    def test_analytic_path_correct_units_counts_and_only_frozen_frequencies(self):
        m = measure().set_index('frequency_sec')
        self.assertEqual(m.index.tolist(), [5, 15, 30])
        for f in FREQUENCIES:
            self.assertAlmostEqual(m.loc[f, 'rv_bps2'], 300 * f * 1e-10 * 1e8, places=8)
            self.assertEqual(m.loc[f, 'n_returns'], 300 // f)
            self.assertAlmostEqual(m.loc[f, 'common_max_mark_age_sec'], .1)
            self.assertEqual(m.loc[f, 'repeat_fraction'], 0.)

    def test_onset_and_future_quotes_cannot_change_measurements(self):
        q = pd.concat([analytic_quotes(), quotes([300, 300.1, 400], [4., 5., 6.])], ignore_index=True)
        pd.testing.assert_frame_equal(measure(q), measure())

    def test_every_grid_boundary_uses_strictly_prior_quote(self):
        m = measure(quotes([-1, 0, 5, 300], [1., 2., 4., 20.])).set_index('frequency_sec')
        self.assertEqual(m.loc[5, 'first_quote_at'], at(-1))
        self.assertEqual(m.loc[5, 'last_quote_at'], at(5))
        self.assertAlmostEqual(m.loc[5, 'rv_bps2'], 2 * math.log(2) ** 2 * 1e8, places=5)
        self.assertAlmostEqual(m.loc[15, 'rv_bps2'], math.log(4) ** 2 * 1e8, places=5)

    def test_initial_quote_may_precede_processing_chunk(self):
        q = quotes([-500, 1, 299], [1.1, 1.2, 1.3])
        m = measure(q)
        self.assertTrue(m.status.eq('available').all())
        self.assertTrue(m.first_quote_at.eq(at(-500)).all())

    def test_missing_initial_quote_retains_three_unavailable_rows(self):
        for q in [quotes([0, 1, 299], [1.1, 1.2, 1.3]), quotes([], [])]:
            with self.subTest(n=len(q)):
                m = measure(q)
                self.assertEqual(len(m), 3)
                self.assertTrue(m.status.eq('no_prior_initial_quote').all())
                self.assertTrue(m[['rv_bps', 'rv_bps2', 'common_max_mark_age_sec']].isna().all().all())
                self.assertTrue(m.first_quote_at.isna().all())
                self.assertFalse(m.common_age_le_60.any())

    def test_pre_window_crossing_support_is_not_bridged(self):
        o = onsets()
        o['support_id'] = 1
        m = measure(quotes([-1, 210, 299], [1.1, 1.2, 1.3]), o,
                    support([-100, 200], [150, 3600]))
        self.assertTrue(m.status.eq('window_crosses_support').all())

    def test_initializer_outside_support_is_not_carried(self):
        m = measure(quotes([-1, 1, 299], [1.1, 1.2, 1.3]), s=support([0], [3600]))
        self.assertTrue(m.status.eq('initial_quote_outside_support').all())

    def test_complete_onset_bin_must_fit_support_even_though_window_is_prior(self):
        with self.assertRaisesRegex(ValueError, 'onset bin outside'):
            measure(s=support([-100], [300]))
        self.assertTrue(measure(s=support([-100], [301])).status.eq('available').all())

    def test_quotes_or_support_invalid_order_fail(self):
        for q in [quotes([-1, -1], [1.1, 1.2]), quotes([1, -1], [1.1, 1.2]),
                  quotes([-1], [np.nan])]:
            with self.subTest(q=q.to_dict()), self.assertRaises(ValueError):
                measure(q)
        with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
            measure(s=support([-100, 100], [200, 400]))
        o = onsets()
        o['support_id'] = 2
        with self.assertRaisesRegex(ValueError, 'support identity'):
            measure(o=o)


class NumericalAndBatchTests(unittest.TestCase):
    def test_flat_paths_keep_exact_zero_without_floor(self):
        m = measure(quotes([-1], [1.1]))
        self.assertTrue(m.rv_bps.eq(0).all())
        self.assertTrue(m.rv_bps2.eq(0).all())
        self.assertTrue(m.repeat_fraction.eq(1).all())
        self.assertTrue(m.zero_return_fraction.eq(1).all())

    def test_tiny_current_returns_not_contaminated_by_earlier_large_return(self):
        t = np.arange(-100, 301)
        mids = np.where(t < 0, 2., np.exp(.1 + np.maximum(t, 0) * 1e-14))
        q = quotes(t - .1, mids)
        m = measure(q).set_index('frequency_sec')
        actual_mid = (q.bid.to_numpy() / 2 + q.ask.to_numpy() / 2)
        for f in FREQUENCIES:
            logs = [math.log(float(actual_mid[100 + j])) for j in range(0, 301, f)]
            reference = math.fsum((v - u) ** 2 for u, v in zip(logs[:-1], logs[1:])) * 1e8
            self.assertAlmostEqual(m.loc[f, 'rv_bps2'] / reference, 1., places=4)
            self.assertLess(m.loc[f, 'rv_bps'], 1e-6)

    def test_repeated_quote_identity_and_zero_return_are_different(self):
        m = measure(quotes(np.arange(301) - .1, np.ones(301)))
        self.assertTrue(m.zero_return_fraction.eq(1).all())
        self.assertTrue(m.repeat_fraction.eq(0).all())

    def test_five_second_reference_checked_at_unchanged_tolerance(self):
        o = onsets()
        o['state_history_available'] = True
        o['state_sample_eligible'] = True
        o['pre_rv_300s_5s_bps'] = math.sqrt(15)
        self.assertTrue(measure(o=o).status.eq('available').all())
        o['pre_rv_300s_5s_bps'] += 1e-6
        with self.assertRaisesRegex(ValueError, 'reconciliation'):
            measure(o=o)

    def test_variation_available_when_inherited_joint_state_is_missing(self):
        m = measure()
        self.assertTrue(m.status.eq('available').all())
        self.assertFalse(m.state_sample_eligible.any())
        s = summarize(m)['summary'].set_index(['cohort', 'frequency_sec', 'sample'])
        self.assertEqual(s.loc[('confirmed_onsets', 5, 'all_available'), 'selected_windows'], 1)
        self.assertEqual(s.loc[('state_eligible_onsets', 5, 'all_available'), 'selected_windows'], 0)

    def test_batch_size_does_not_change_any_output(self):
        o, q = onsets([300, 400, 500, 600, 700]), analytic_quotes(last=700)
        reference = measure(q, o, batch_size=512)
        for size in [1, 2, 3, 5, 7]:
            with self.subTest(size=size):
                pd.testing.assert_frame_equal(reference, measure(q, o, batch_size=size))

    def test_batch_validation(self):
        for size in [0, -1, 2.5, True]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'batch size'):
                measure(batch_size=size)

    def test_empty_windows_have_full_usable_schema(self):
        m = measure(quotes([], []), onsets([]))
        self.assertEqual(m.columns.tolist(), MEASUREMENT_COLUMNS)
        self.assertEqual(len(m), 0)
        for name, table in summarize(m).items():
            self.assertEqual(len(table), 0, name)
            self.assertTrue(len(table.columns) > 5)


class AgeScreenTests(unittest.TestCase):
    def test_age_equality_passes_and_one_nanosecond_older_fails(self):
        for cap in [15, 30, 60]:
            with self.subTest(cap=cap):
                q = quotes(np.r_[-cap, np.arange(300) + .5], np.ones(301))
                self.assertTrue(measure(q)[f'common_age_le_{cap}'].all())
                q.loc[0, 'timestamp'] -= pd.Timedelta(1, unit='ns')
                self.assertFalse(measure(q)[f'common_age_le_{cap}'].any())

    def test_common_one_second_screen_catches_gap_omitted_by_own_grid(self):
        times = np.r_[-.1, .1, np.arange(19, 300) + .9]
        m = measure(quotes(times, np.ones(len(times)))).set_index('frequency_sec')
        self.assertAlmostEqual(m.loc[5, 'common_max_mark_age_sec'], 18.9)
        self.assertTrue(m.loc[5, 'own_age_le_15'])
        self.assertFalse(m.common_age_le_15.any())
        s = summarize(m.reset_index())['summary'].set_index(['cohort', 'frequency_sec', 'sample'])
        row = s.loc[('confirmed_onsets', 5, 'common_age_15')]
        self.assertEqual(row.own_grid_age_pass_windows, 1)
        self.assertEqual(row.selected_windows, 0)

    def test_same_common_screen_and_nested_caps_at_every_frequency(self):
        m = measure(quotes(np.r_[-20, np.arange(300) + .1], np.ones(301)))
        self.assertFalse(m.common_age_le_15.any())
        self.assertTrue(m.common_age_le_30.all())
        self.assertTrue(m.common_age_le_60.all())
        self.assertTrue(m.common_max_mark_age_sec.eq(20).all())


class SummaryTests(unittest.TestCase):
    def test_two_cohorts_use_explicit_denominators_without_dropping_fragments(self):
        m = synthetic_measurements()
        m.loc[m.window_id.eq(3), 'onset_confirmed'] = False
        m['state_sample_eligible'] = m.window_id.isin([0, 2])
        s = summarize(m)['summary'].set_index(['cohort', 'frequency_sec', 'sample'])
        confirmed = s.loc[('confirmed_onsets', 5, 'all_available')]
        eligible = s.loc[('state_eligible_onsets', 5, 'all_available')]
        self.assertEqual(confirmed.windows, 4)
        self.assertEqual(confirmed.cohort_windows, 3)
        self.assertEqual(confirmed.selected_windows, 3)
        self.assertAlmostEqual(confirmed.mean_rv_bps, 2/3)
        self.assertEqual(eligible.windows, 4)
        self.assertEqual(eligible.cohort_windows, 2)
        self.assertEqual(eligible.selected_windows, 2)
        self.assertEqual(eligible.mean_rv_bps, .5)

    def test_ties_zeros_and_variance_mean_ratio_use_same_window_pairs(self):
        out = summarize(synthetic_measurements())
        s = out['summary'].set_index(['cohort', 'frequency_sec', 'sample'])
        c = out['comparisons'].set_index(['cohort', 'frequency_sec', 'sample'])
        row = s.loc[('confirmed_onsets', 5, 'all_available')]
        self.assertEqual(row.mean_variance_bps2, 11/4)
        self.assertEqual(row.mean_rv_bps, 5/4)
        row = c.loc[('confirmed_onsets', 15, 'all_available')]
        self.assertEqual(row.paired_windows, 4)
        self.assertEqual(row.mean_variance_ratio_to_5s, 4)
        self.assertEqual(row.mean_absolute_rv_difference_bps, 1.25)
        self.assertAlmostEqual(row.spearman_rv_to_5s, 1.)
        self.assertEqual(row.zero_variance_windows, 1)
        self.assertEqual(c.loc[('confirmed_onsets', 15, 'common_age_15'), 'paired_windows'], 3)

    def test_zero_reference_has_undefined_ratio_and_correlation(self):
        m = synthetic_measurements()
        m[['rv_bps', 'rv_bps2']] = 0.
        c = summarize(m)['comparisons']
        self.assertTrue(c.mean_variance_ratio_to_5s.isna().all())
        self.assertTrue(c.spearman_rv_to_5s.isna().all())
        self.assertTrue(c.mean_absolute_rv_difference_bps.dropna().eq(0).all())

    def test_daily_hourly_populations_and_frequency_count(self):
        m = synthetic_measurements()
        out = summarize(m)
        self.assertEqual(len(out['summary']), 3 * 2 * 4)
        self.assertEqual(len(out['comparisons']), 3 * 2 * 4)
        for key in ['daily', 'hourly']:
            self.assertEqual(set(out[key].cohort), {'confirmed_onsets', 'state_eligible_onsets'})
            self.assertEqual(set(out[key].frequency_sec), {5, 15, 30})
            self.assertTrue(out[key].windows.eq(4).all())

    def test_missing_or_duplicate_frequency_rows_are_rejected(self):
        m = synthetic_measurements()
        for bad in [m.iloc[1:], pd.concat([m, m.iloc[:1]], ignore_index=True)]:
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                summarize(bad)

    def test_disagreeing_frequency_cohort_or_age_screen_fails(self):
        for column, value in [('onset_confirmed', False), ('common_max_mark_age_sec', 2.)]:
            m = synthetic_measurements()
            m.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'samples differ'):
                summarize(m)

    def test_invalid_measurement_status_or_cap_is_rejected(self):
        for column, value in [('status', 'made_up'), ('frequency_sec', 10), ('common_age_le_15', False),
                              ('rv_bps2', -1.), ('rv_bps', np.nan), ('n_returns', 1)]:
            m = synthetic_measurements()
            m.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                summarize(m)


if __name__ == '__main__':
    unittest.main()

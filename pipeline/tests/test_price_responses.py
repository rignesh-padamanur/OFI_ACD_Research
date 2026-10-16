"""Adversarial tests for onset-anchored recorded-midquote responses.

Expected values below come from small explicit price paths, not a second copy
of the production join or its implementation details.
"""
import unittest

import numpy as np
import pandas as pd

from src.price_responses import construct_price_responses, summarize_responses


ORIGIN = pd.Timestamp('2024-04-22 12:00:00', tz='UTC')


def at(seconds):
    return pd.Timestamp(ORIGIN.value + round(float(seconds) * 1_000_000_000),
                        unit='ns', tz='UTC')


def onset_table(starts=(10,), directions=(2.,), confirmed=None, state=None):
    n = len(starts)
    return pd.DataFrame({
        'episode_id': np.arange(n, dtype='int64'),
        'bin_start': pd.DatetimeIndex([at(s) for s in starts]),
        'available_at': pd.DatetimeIndex([at(s + 1) for s in starts]),
        'support_id': np.zeros(n, dtype='int64'),
        'quote_ofi': np.asarray(directions, dtype=float),
        'onset_confirmed': np.ones(n, dtype=bool) if confirmed is None else np.asarray(confirmed, dtype=bool),
        'state_sample_eligible': np.ones(n, dtype=bool) if state is None else np.asarray(state, dtype=bool),
    })


def quote_table(seconds=(9., 10.5, 11.5), mids=(1.1000, 1.1002, 1.1001)):
    values = np.asarray(mids, dtype=float)
    return pd.DataFrame({'timestamp': pd.DatetimeIndex([at(s) for s in seconds]),
                         'bid': values - .00001, 'ask': values + .00001})


def support_table(starts=(-3600,), ends=(3600,)):
    return pd.DataFrame({'start': pd.DatetimeIndex([at(s) for s in starts]),
                         'end_exclusive': pd.DatetimeIndex([at(s) for s in ends])})


def construct(onsets=None, quotes=None, support=None, horizons=(0, 1, 5),
              policy='structural_only', development_end=None):
    return construct_price_responses(
        onset_table() if onsets is None else onsets,
        quote_table() if quotes is None else quotes,
        support_table() if support is None else support,
        policy=policy, development_end=at(3600) if development_end is None else development_end,
        horizons=horizons)


class PriceResponseTimingTests(unittest.TestCase):
    def test_strict_boundaries_exclude_quotes_exactly_at_each_mark(self):
        q = quote_table([9, 10, 10.5, 11, 11.5, 12],
                        [1.1, 1.101, 1.102, 1.103, 1.104, 1.105])
        result = construct(quotes=q, horizons=[0, 1])
        a = result['anchors'].iloc[0]
        p = result['paths'].set_index('horizon_sec')
        self.assertEqual(a.pre_mark_at, at(9))
        self.assertEqual(a.detection_mark_at, at(10.5))
        self.assertEqual(p.loc[1, 'endpoint_quote_at'], at(11.5))
        self.assertAlmostEqual(a.raw_initial_pips, 20.)
        self.assertAlmostEqual(p.loc[1, 'raw_continuation_pips'], 20.)
        self.assertAlmostEqual(p.loc[1, 'raw_total_pips'], 40.)

    def test_horizon_zero_is_detection_and_decomposition_holds(self):
        result = construct()
        a = result['anchors'].iloc[0]
        p = result['paths'].set_index('horizon_sec')
        self.assertEqual(p.loc[0, 'target_at'], at(11))
        self.assertEqual(p.loc[0, 'endpoint_quote_at'], a.detection_mark_at)
        self.assertEqual(p.loc[0, 'raw_continuation_pips'], 0.)
        self.assertEqual(p.loc[0, 'continuation_pips'], 0.)
        self.assertEqual(p.loc[0, 'total_pips'], a.initial_pips)
        np.testing.assert_allclose(p.total_pips, p.initial_pips + p.continuation_pips,
                                   rtol=0, atol=1e-9)

    def test_initial_jump_without_continuation_is_not_called_decay(self):
        result = construct(quotes=quote_table([9, 10.5], [1.1, 1.2]), horizons=[0, 1, 300])
        a, p = result['anchors'].iloc[0], result['paths']
        self.assertAlmostEqual(a.initial_pips, 1000.)
        self.assertTrue(p.continuation_pips.eq(0).all())
        np.testing.assert_allclose(p.total_pips, 1000.)
        self.assertTrue(p.endpoint_reuses_detection_quote.all())

    def test_negative_raw_continuation_remains_signed(self):
        p = construct()['paths'].set_index('horizon_sec')
        self.assertAlmostEqual(p.loc[1, 'raw_continuation_pips'], -1.)
        self.assertAlmostEqual(p.loc[1, 'continuation_pips'], -1.)
        self.assertAlmostEqual(p.loc[1, 'raw_total_pips'], 1.)

    def test_all_default_301_horizons_retained_for_each_onset(self):
        o = onset_table([10, 30], [1., -1.])
        result = construct(onsets=o, horizons=None)
        self.assertEqual(len(result['anchors']), 2)
        self.assertEqual(len(result['paths']), 602)
        for _, part in result['paths'].groupby('episode_id'):
            self.assertEqual(part.horizon_sec.tolist(), list(range(301)))

    def test_custom_requested_horizons_are_exact_not_row_positions(self):
        p = construct(horizons=[0, 5, 30, 300])['paths']
        self.assertEqual(p.horizon_sec.tolist(), [0, 5, 30, 300])
        self.assertEqual(p.target_at.tolist(), [at(11), at(16), at(41), at(311)])

    def test_stale_repeated_mark_is_reported_with_increasing_age(self):
        p = construct(quotes=quote_table([9, 10.5], [1.1, 1.1002]),
                      horizons=[0, 15, 300])['paths']
        self.assertEqual(p.endpoint_age_sec.tolist(), [.5, 15.5, 300.5])
        self.assertTrue(p.endpoint_reuses_detection_quote.all())
        self.assertTrue(p.continuation_pips.eq(0).all())
        self.assertTrue(p.triplet_quote_available.all())


class PriceResponseDirectionTests(unittest.TestCase):
    def test_positive_and_negative_directions_orient_all_three_responses(self):
        for sign in [1., -1.]:
            with self.subTest(sign=sign):
                result = construct(onsets=onset_table(directions=[7 * sign]))
                a, p = result['anchors'].iloc[0], result['paths'].iloc[1]
                self.assertEqual(a.direction, sign)
                self.assertEqual(a.direction_status, 'positive' if sign > 0 else 'negative')
                self.assertTrue(a.direction_available)
                self.assertEqual(a.abs_onset_quote_ofi, 7.)
                self.assertAlmostEqual(a.initial_pips, 2 * sign)
                self.assertAlmostEqual(p.continuation_pips, -sign)
                self.assertAlmostEqual(p.total_pips, sign)

    def test_zero_direction_keeps_raw_changes_and_no_aligned_fallback(self):
        result = construct(onsets=onset_table(directions=[0.]))
        a, p = result['anchors'].iloc[0], result['paths']
        self.assertEqual(a.direction, 0.)
        self.assertEqual(a.direction_status, 'zero')
        self.assertFalse(a.direction_available)
        self.assertFalse(a.confirmed_direction_cohort)
        self.assertTrue(np.isnan(a.initial_pips))
        self.assertAlmostEqual(a.raw_initial_pips, 2.)
        self.assertTrue(p[['initial_pips', 'continuation_pips', 'total_pips']].isna().all().all())
        self.assertTrue(p[['raw_initial_pips', 'raw_continuation_pips', 'raw_total_pips']].notna().all().all())
        self.assertTrue(p.triplet_quote_available.all())

    def test_nonfinite_direction_is_missing_without_discarding_quotes(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                result = construct(onsets=onset_table(directions=[value]))
                a, p = result['anchors'].iloc[0], result['paths']
                self.assertEqual(a.direction_status, 'missing')
                self.assertTrue(np.isnan(a.direction))
                self.assertFalse(a.direction_available)
                self.assertTrue(p.total_pips.isna().all())
                self.assertTrue(p.raw_total_pips.notna().all())

    def test_tiny_nonzero_recorded_sign_has_no_unannounced_tolerance(self):
        for value in [-5.960464477539063e-8, np.nextafter(0., 1.)]:
            with self.subTest(value=value):
                a = construct(onsets=onset_table(directions=[value]))['anchors'].iloc[0]
                self.assertEqual(a.direction, -1. if value < 0 else 1.)
                self.assertTrue(a.direction_available)
                self.assertEqual(a.abs_onset_quote_ofi, abs(value))

    def test_uncertain_and_missing_state_onsets_stay_in_outputs(self):
        o = onset_table([10, 30, 50], [1., 1., 0.],
                        confirmed=[False, True, True], state=[False, False, True])
        r = construct(onsets=o)
        self.assertEqual(len(r['anchors']), 3)
        self.assertEqual(len(r['paths']), 9)
        self.assertEqual(r['anchors'].confirmed_direction_cohort.tolist(), [False, True, False])
        self.assertEqual(r['anchors'].state_direction_cohort.tolist(), [False, False, False])


class PriceResponseAvailabilityTests(unittest.TestCase):
    def test_missing_premark_keeps_continuation_available(self):
        r = construct(quotes=quote_table([10.5, 11.5], [1.1, 1.1001]),
                      support=support_table([10], [3600]))
        a, p = r['anchors'].iloc[0], r['paths']
        self.assertEqual(a.pre_mark_status, 'no_prior_quote')
        self.assertFalse(a.baseline_quote_available)
        self.assertTrue(np.isnan(a.raw_initial_pips))
        self.assertTrue(p.continuation_quote_available.all())
        self.assertFalse(p.total_quote_available.any())
        self.assertFalse(p.triplet_quote_available.any())
        self.assertAlmostEqual(p.loc[p.horizon_sec.eq(1), 'continuation_pips'].iloc[0], 1.)

    def test_quote_before_onset_support_is_not_imported_as_premark(self):
        r = construct(support=support_table([10], [3600]))
        a = r['anchors'].iloc[0]
        self.assertEqual(a.pre_mark_status, 'prior_quote_outside_support')
        self.assertTrue(pd.isna(a.pre_mark_at))
        self.assertTrue(np.isnan(a.pre_mid))
        self.assertTrue(np.isnan(a.pre_age_sec))
        self.assertTrue(r['paths'].continuation_quote_available.all())

    def test_no_prior_detection_quote_is_explicit_missing(self):
        r = construct(quotes=quote_table([20], [1.1]), horizons=[0, 1, 30])
        a, p = r['anchors'].iloc[0], r['paths']
        self.assertEqual(a.detection_mark_status, 'no_prior_quote')
        self.assertFalse(a.baseline_quote_available)
        self.assertFalse(p.continuation_quote_available.any())
        self.assertFalse(p.triplet_quote_available.any())

    def test_target_at_support_end_uses_left_limit_but_cannot_resume(self):
        support = support_table([-100, 21], [20, 100])
        q = quote_table([9, 10.5, 19.5, 22.5], [1.1, 1.1002, 1.1004, 1.2])
        p = construct(quotes=q, support=support, horizons=[0, 9, 10, 12])['paths'].set_index('horizon_sec')
        self.assertEqual(p.loc[9, 'endpoint_status'], 'available')
        self.assertEqual(p.loc[9, 'endpoint_quote_at'], at(19.5))
        for h in [10, 12]:
            self.assertEqual(p.loc[h, 'endpoint_status'], 'outside_onset_support')
            self.assertTrue(pd.isna(p.loc[h, 'endpoint_quote_at']))
            self.assertTrue(np.isnan(p.loc[h, 'endpoint_mid']))
            self.assertTrue(np.isnan(p.loc[h, 'endpoint_age_sec']))
        self.assertEqual(len(p), 4)

    def test_target_at_development_end_allowed_but_later_unavailable(self):
        q = quote_table([9, 10.5, 19.5, 20, 20.5], [1.1, 1.1002, 1.1004, 1.2, 1.3])
        p = construct(quotes=q, horizons=[0, 9, 10], development_end=at(20))['paths'].set_index('horizon_sec')
        self.assertEqual(p.loc[9, 'endpoint_status'], 'available')
        self.assertEqual(p.loc[9, 'endpoint_quote_at'], at(19.5))
        self.assertEqual(p.loc[10, 'endpoint_status'], 'beyond_development_end')
        self.assertTrue(pd.isna(p.loc[10, 'endpoint_quote_at']))
        self.assertTrue(pd.isna(p.loc[10, 'later_observed_episode_by_horizon']))

    def test_last_horizon_missing_does_not_erase_short_paths_or_initial_jump(self):
        r = construct(horizons=[0, 1, 300], development_end=at(20))
        p = r['paths'].set_index('horizon_sec')
        self.assertEqual(len(r['anchors']), 1)
        self.assertEqual(len(p), 3)
        self.assertTrue(p.loc[[0, 1], 'triplet_quote_available'].all())
        self.assertFalse(p.loc[300, 'triplet_quote_available'])
        self.assertFalse(p.common_path_available.any())
        self.assertTrue(p.initial_pips.notna().all())

    def test_fresh_caps_accept_exact_age_and_reject_older_marks(self):
        for cap in [15, 30, 60]:
            with self.subTest(cap=cap):
                q = quote_table([11 - cap], [1.1])
                r = construct(quotes=q, horizons=[0, 1], policy=f'fresh_{cap}s')
                a, p = r['anchors'].iloc[0], r['paths'].set_index('horizon_sec')
                self.assertEqual(a.detection_age_sec, cap)
                self.assertEqual(a.detection_mark_status, 'available')
                self.assertEqual(p.loc[1, 'endpoint_status'], 'stale_under_policy')
                self.assertTrue(pd.isna(p.loc[1, 'endpoint_quote_at']))
                self.assertTrue(np.isnan(p.loc[1, 'endpoint_age_sec']))

    def test_fresh_interval_extra_nanosecond_does_not_relax_age_cap(self):
        for cap in [15, 30, 60]:
            with self.subTest(cap=cap):
                q = quote_table([11 - cap], [1.1])
                q['timestamp'] = pd.to_datetime(q.timestamp.astype('int64') - 1, utc=True)
                support = support_table([-100], [11])
                r = construct(quotes=q, support=support, horizons=[0], policy=f'fresh_{cap}s')
                self.assertEqual(r['anchors'].detection_mark_status.iloc[0], 'stale_under_policy')
                self.assertEqual(r['paths'].endpoint_status.iloc[0], 'stale_under_policy')

    def test_fixed_structural_age_caps_do_not_change_primary_availability(self):
        q = quote_table([9, 10.5], [1.1, 1.1002])
        p = construct(quotes=q, horizons=[0, 15, 30, 60])['paths'].set_index('horizon_sec')
        self.assertTrue(p.triplet_quote_available.all())
        self.assertEqual(p.triplet_age_le_15.tolist(), [True, False, False, False])
        self.assertEqual(p.triplet_age_le_30.tolist(), [True, True, False, False])
        self.assertEqual(p.triplet_age_le_60.tolist(), [True, True, True, False])
        self.assertFalse(p.common_path_age_le_15.any())
        self.assertFalse(p.common_path_age_le_30.any())
        self.assertFalse(p.common_path_age_le_60.any())
        self.assertTrue(p.common_path_available.all())

    def test_structural_cap_checks_premark_as_well_as_endpoint(self):
        q = quote_table([-10, 10.5, 11.5], [1.1, 1.1002, 1.1001])
        p = construct(quotes=q, horizons=[0, 1])['paths']
        self.assertTrue(p.triplet_quote_available.all())
        self.assertFalse(p.triplet_age_le_15.any())
        self.assertTrue(p.triplet_age_le_30.all())


class PriceResponseInvarianceTests(unittest.TestCase):
    def test_future_quotes_cannot_change_anchor_information_or_earlier_path(self):
        q = quote_table([9, 10.5, 11.5], [1.1, 1.1002, 1.1001])
        extended = pd.concat([q, quote_table([20, 40], [1.5, .9])], ignore_index=True)
        a = construct(quotes=q)
        b = construct(quotes=extended)
        pd.testing.assert_frame_equal(a['anchors'], b['anchors'])
        pd.testing.assert_frame_equal(a['paths'], b['paths'])

    def test_later_opposite_episode_does_not_change_current_direction_or_jump(self):
        old = construct(onsets=onset_table([10], [2.]))
        new = construct(onsets=onset_table([10, 30], [2., -999.]))
        columns = ['episode_id', 'bin_start', 'available_at', 'quote_ofi', 'direction',
                   'direction_status', 'pre_mark_at', 'detection_mark_at',
                   'raw_initial_pips', 'initial_pips']
        pd.testing.assert_frame_equal(old['anchors'][columns].reset_index(drop=True),
                                      new['anchors'].iloc[:1][columns].reset_index(drop=True))

    def test_overlap_uses_later_detection_and_includes_exact_target(self):
        o = onset_table([10, 20, 40], [1., -1., 1.],
                        confirmed=[True, False, True], state=[True, False, True])
        p = construct(onsets=o, horizons=[0, 9, 10, 11])['paths']
        first = p.loc[p.episode_id.eq(0)].set_index('horizon_sec')
        self.assertEqual(first.later_observed_episode_by_horizon.tolist(), [False, False, True, True])
        self.assertTrue(first.triplet_quote_available.all())
        self.assertEqual(len(p), 12)

    def test_overlap_after_cutoff_is_unknown_even_when_next_row_is_absent(self):
        p = construct(horizons=[0, 300], development_end=at(20))['paths']
        self.assertFalse(p.later_observed_episode_by_horizon.iloc[0])
        self.assertTrue(pd.isna(p.later_observed_episode_by_horizon.iloc[1]))

    def test_input_tables_are_unchanged(self):
        o, q, s = onset_table(), quote_table(), support_table()
        copies = [x.copy(deep=True) for x in [o, q, s]]
        construct(o, q, s)
        for original, before in zip([o, q, s], copies):
            pd.testing.assert_frame_equal(original, before)


class PriceResponseValidationTests(unittest.TestCase):
    def test_legacy_persistence_entry_points_fail_before_processing_data(self):
        from src.persistence import (build_coalesced_event_table,
                                     compute_volatility_standardized_paths,
                                     compute_decay_metrics)
        empty = pd.DataFrame()
        for function, args in [(build_coalesced_event_table, (empty,)),
                               (compute_volatility_standardized_paths, (empty, empty)),
                               (compute_decay_metrics, (empty,))]:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, 'Legacy persistence'):
                    function(*args)

    def test_empty_onsets_preserve_both_output_schemas(self):
        empty = onset_table().iloc[:0].copy()
        result = construct(onsets=empty)
        self.assertEqual(len(result['anchors']), 0)
        self.assertEqual(len(result['paths']), 0)
        self.assertIn('initial_pips', result['anchors'])
        self.assertIn('horizon_sec', result['paths'])
        self.assertIn('triplet_quote_available', result['paths'])

    def test_bad_horizons_rejected(self):
        for horizons in [[], [1], [-1, 0], [0, 1, 1], [0, 2, 1], [0, .5],
                         [0, 301], [False, 1], ['0', '1']]:
            with self.subTest(horizons=horizons):
                with self.assertRaises((ValueError, TypeError)):
                    construct(horizons=horizons)

    def test_onset_requires_whole_second_and_completed_bin(self):
        for column, offset in [('bin_start', pd.Timedelta(milliseconds=1)),
                               ('available_at', pd.Timedelta(seconds=1))]:
            with self.subTest(column=column):
                o = onset_table()
                o[column] = o[column] + offset
                with self.assertRaises(ValueError):
                    construct(onsets=o)

    def test_unordered_or_duplicate_onsets_and_identifiers_rejected(self):
        for kind in ['duplicate_time', 'reversed_time', 'duplicate_id', 'fractional_id']:
            with self.subTest(kind=kind):
                o = onset_table([10, 30], [1., 1.])
                if kind == 'duplicate_time':
                    o.loc[1, ['bin_start', 'available_at']] = o.loc[0, ['bin_start', 'available_at']]
                elif kind == 'reversed_time':
                    o = o.iloc[::-1].reset_index(drop=True)
                elif kind == 'duplicate_id':
                    o['episode_id'] = 0
                else:
                    o['episode_id'] = [0., .5]
                with self.assertRaises(ValueError):
                    construct(onsets=o)

    def test_non_boolean_flags_rejected(self):
        for column in ['onset_confirmed', 'state_sample_eligible']:
            with self.subTest(column=column):
                o = onset_table()
                o[column] = 1
                with self.assertRaises(ValueError):
                    construct(onsets=o)

    def test_non_numeric_imbalance_rejected_but_missing_numeric_allowed(self):
        o = onset_table()
        o['quote_ofi'] = '1.0'
        with self.assertRaises((ValueError, TypeError)):
            construct(onsets=o)

    def test_invalid_support_identifier_and_uncovered_bin_rejected(self):
        for sid in [-1, 1, .5]:
            with self.subTest(sid=sid):
                o = onset_table()
                o['support_id'] = sid
                with self.assertRaises(ValueError):
                    construct(onsets=o)
        for start, end in [(10.5, 100), (-100, 10.5)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    construct(support=support_table([start], [end]))

    def test_support_overlap_and_nonpositive_interval_rejected(self):
        for s in [support_table([-100, 0], [20, 100]), support_table([10], [10])]:
            with self.subTest(support=s.to_dict()):
                with self.assertRaises(ValueError):
                    construct(support=s)

    def test_duplicate_or_unordered_quote_timestamps_rejected(self):
        for q in [quote_table([9, 9], [1.1, 1.2]), quote_table([10, 9], [1.1, 1.2])]:
            with self.subTest(quotes=q.to_dict()):
                with self.assertRaises(ValueError):
                    construct(quotes=q)

    def test_invalid_quote_prices_rejected(self):
        for bid, ask in [(0., 1.), (-1., 1.), (1.2, 1.1), (1., 1.),
                         (np.nan, 1.), (1., np.inf)]:
            with self.subTest(bid=bid, ask=ask):
                q = quote_table()
                q.loc[0, ['bid', 'ask']] = [bid, ask]
                with self.assertRaises(ValueError):
                    construct(quotes=q)

    def test_naive_time_and_wrong_period_rejected(self):
        o = onset_table()
        o['bin_start'] = o.bin_start.dt.tz_localize(None)
        with self.assertRaises(ValueError):
            construct(onsets=o)
        o = onset_table()
        o['period'] = 'calibration'
        with self.assertRaises(ValueError):
            construct(onsets=o)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            construct(policy='automatic')


class PriceResponseSummaryTests(unittest.TestCase):
    def test_joint_means_use_same_events_and_common_path_changes_all_horizons(self):
        onsets = onset_table([10, 30], [1., 1.], state=[True, False])
        quotes = quote_table([9, 10.5, 11.5, 29, 30.5, 31.5],
                             [1.1, 1.1002, 1.1001, 1.2, 1.2004, 1.2008])
        tables = construct(onsets, quotes, horizons=[0, 1, 5], development_end=at(32))
        summary, stats = summarize_responses(tables)
        s = summary.set_index(['horizon_sec', 'sample'])
        self.assertEqual(s.loc[(1, 'horizon_confirmed'), 'observations'], 2)
        self.assertAlmostEqual(s.loc[(1, 'horizon_confirmed'), 'mean_initial_pips'], 3.)
        self.assertAlmostEqual(s.loc[(1, 'horizon_confirmed'), 'mean_continuation_pips'], 1.5)
        self.assertAlmostEqual(s.loc[(1, 'horizon_confirmed'), 'mean_total_pips'], 4.5)
        self.assertEqual(s.loc[(5, 'horizon_confirmed'), 'observations'], 1)
        for horizon in [0, 1, 5]:
            self.assertEqual(s.loc[(horizon, 'common_path_confirmed'), 'observations'], 1)
            self.assertAlmostEqual(s.loc[(horizon, 'common_path_confirmed'), 'mean_initial_pips'], 2.)
            self.assertEqual(s.loc[(horizon, 'horizon_state'), 'observations'], 1)
        available = summary.observations.gt(0)
        np.testing.assert_allclose(summary.loc[available, 'mean_total_pips'],
                                   summary.loc[available, 'mean_initial_pips'] +
                                   summary.loc[available, 'mean_continuation_pips'], atol=1e-9)
        self.assertEqual(stats['horizons']['5']['confirmed_directed_cohort'], 2)
        self.assertEqual(stats['horizons']['5']['joint_available'], 1)
        self.assertEqual(stats['horizons']['5']['joint_missing'], 1)

    def test_missing_premark_continuation_is_counted_outside_joint_means(self):
        tables = construct(quotes=quote_table([10.5, 11.5], [1.1, 1.1002]),
                           support=support_table([10], [3600]), horizons=[0, 1])
        summary, stats = summarize_responses(tables)
        self.assertTrue(summary.observations.eq(0).all())
        self.assertTrue(summary[['mean_initial_pips', 'mean_continuation_pips',
                                 'mean_total_pips']].isna().all().all())
        for h in ['0', '1']:
            self.assertEqual(stats['horizons'][h]['joint_available'], 0)
            self.assertEqual(stats['horizons'][h]['continuation_available_without_premark'], 1)


if __name__ == '__main__':
    unittest.main()

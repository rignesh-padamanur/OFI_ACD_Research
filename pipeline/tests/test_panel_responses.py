"""Counterexamples for continuous full-panel response paths and denominators."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.panel_responses import (
    anchor_statistics, component_summary, construct_panel_responses, summarize_horizon,
)
from src.price_responses import construct_price_responses, summarize_responses

ORIGIN = pd.Timestamp('2024-04-22', tz='UTC')


def at(seconds):
    return ORIGIN + pd.Timedelta(seconds=seconds)


def onsets(starts=(10,), directions=None, *, confirmed=None, state=None):
    n = len(starts)
    return pd.DataFrame({
        'episode_id': np.arange(n, dtype='int64'),
        'bin_start': pd.DatetimeIndex([at(s) for s in starts]),
        'available_at': pd.DatetimeIndex([at(s + 1) for s in starts]),
        'support_id': np.zeros(n, dtype='int64'),
        'quote_ofi': np.ones(n) if directions is None else np.asarray(directions, dtype=float),
        'onset_confirmed': np.ones(n, dtype=bool) if confirmed is None else np.array(confirmed, dtype=bool),
        'state_sample_eligible': np.ones(n, dtype=bool) if state is None else np.array(state, dtype=bool),
        'period': ['development'] * n,
    })


def quotes(seconds=(9, 10.5, 11.5), mids=(1.1, 1.1002, 1.1001)):
    values = np.asarray(mids, dtype=float)
    return pd.DataFrame({'timestamp': pd.DatetimeIndex([at(s) for s in seconds]),
                         'bid': values - .00001, 'ask': values + .00001})


def support(starts=(-4000,), ends=(4000,)):
    return pd.DataFrame({'start': pd.DatetimeIndex([at(s) for s in starts]),
                         'end_exclusive': pd.DatetimeIndex([at(s) for s in ends])})


def construct(o=None, q=None, s=None, *, stop=4000, all_o=None):
    o = onsets() if o is None else o
    return construct_panel_responses(
        o, quotes() if q is None else q, support() if s is None else s,
        sample_end=at(stop), all_onsets=o if all_o is None else all_o)


class PanelResponseTimingTests(unittest.TestCase):
    def test_calibration_cutoff_is_not_an_observation_boundary(self):
        # Treat 20 seconds as a processing cutoff. The endpoint at 21 still
        # observes the quote at 20.5, and strictly excludes the quote at 21.
        q = quotes([9, 10.5, 19.5, 20.5, 21], [1.1, 1.1002, 1.1003, 1.1004, 1.2])
        p = construct(q=q)['paths'].set_index('horizon_sec')
        self.assertEqual(p.loc[10, 'target_at'], at(21))
        self.assertEqual(p.loc[10, 'endpoint_quote_at'], at(20.5))
        self.assertAlmostEqual(p.loc[10, 'continuation_pips'], 2.)
        self.assertEqual(p.loc[10, 'endpoint_status'], 'available')

    def test_global_next_detection_in_another_batch_is_seen_at_equality(self):
        o = onsets([10, 20])
        p = construct(o.iloc[:1], all_o=o)['paths'].set_index('horizon_sec')
        self.assertFalse(p.loc[9, 'later_observed_episode_by_horizon'])
        self.assertTrue(p.loc[10, 'later_observed_episode_by_horizon'])
        self.assertTrue(p.loc[300, 'later_observed_episode_by_horizon'])
        last = construct(o.iloc[1:], all_o=o)['paths']
        self.assertFalse(last.later_observed_episode_by_horizon.any())

    def test_uncertain_undirected_next_fragment_still_counts_as_observed(self):
        o = onsets([10, 20], [1., 0.], confirmed=[True, False], state=[True, False])
        p = construct(o.iloc[:1], all_o=o)['paths'].set_index('horizon_sec')
        self.assertTrue(p.loc[10, 'later_observed_episode_by_horizon'])
        self.assertTrue(p.loc[10, 'triplet_quote_available'])

    def test_chunking_preserves_every_value_and_global_overlap(self):
        o = onsets([10, 20, 40], [1., -1., 0.])
        q = quotes([9, 10.5, 19.5, 20.5, 39.5, 40.5, 50.5],
                   [1.1, 1.1002, 1.1003, 1.1004, 1.1001, 1.1002, 1.1003])
        full = construct(o, q)
        chunks = [construct(o.iloc[i:i + 1], q, all_o=o) for i in range(3)]
        for key in ['anchors', 'paths']:
            assert_frame_equal(full[key], pd.concat([part[key] for part in chunks], ignore_index=True))
        self.assertEqual(len(full['paths']), 3 * 301)

    def test_sample_end_and_support_end_are_left_limits(self):
        q = quotes([9, 10.5, 19.5, 20, 20.5], [1.1, 1.1002, 1.1004, 1.2, 1.3])
        for same_support_end in [False, True]:
            with self.subTest(same_support_end=same_support_end):
                p = construct(q=q, s=support(ends=[20 if same_support_end else 4000]),
                              stop=20)['paths'].set_index('horizon_sec')
                self.assertEqual(p.loc[9, 'endpoint_quote_at'], at(19.5))
                self.assertEqual(p.loc[9, 'endpoint_status'], 'available')
                self.assertEqual(p.loc[10, 'endpoint_status'], 'beyond_sample_end')
                self.assertTrue(pd.isna(p.loc[10, 'endpoint_mid']))
                self.assertTrue(pd.isna(p.loc[10, 'later_observed_episode_by_horizon']))
                self.assertFalse(p.common_path_available.any())

    def test_support_gap_cannot_resume_at_the_next_support(self):
        q = quotes([9, 10.5, 19.5, 22], [1.1, 1.1002, 1.1004, 1.3])
        p = construct(q=q, s=support([-4000, 21], [20, 4000]))['paths'].set_index('horizon_sec')
        self.assertEqual(p.loc[9, 'endpoint_status'], 'available')
        self.assertEqual(p.loc[11, 'endpoint_status'], 'outside_onset_support')
        self.assertTrue(pd.isna(p.loc[12, 'total_pips']))

    def test_old_initializer_is_retained_without_structural_age_filter(self):
        r = construct(q=quotes([-3000, 10.5], [1.1, 1.1002]))
        a = r['anchors'].iloc[0]
        self.assertEqual(a.pre_mark_at, at(-3000))
        self.assertEqual(a.pre_age_sec, 3010)
        self.assertAlmostEqual(a.initial_pips, 2.)
        self.assertTrue(r['paths'].triplet_quote_available.all())
        self.assertFalse(r['paths'].triplet_age_le_60.any())

    def test_future_quote_changes_leave_earlier_marks_and_responses_unchanged(self):
        q = quotes([9, 10.5, 11.5, 50], [1.1, 1.1002, 1.1001, 1.2])
        changed = q.copy()
        changed.loc[3, ['bid', 'ask']] += .2
        first, second = construct(q=q)['paths'], construct(q=changed)['paths']
        assert_frame_equal(first.loc[first.target_at.le(at(50))],
                           second.loc[second.target_at.le(at(50))])


class PanelResponseComponentTests(unittest.TestCase):
    def test_missing_premark_does_not_erase_continuation(self):
        r = construct(q=quotes([10.5, 11.5], [1.1, 1.1001]), s=support([10], [4000]))
        p = r['paths'].loc[lambda f: f.horizon_sec.eq(1)]
        self.assertTrue(p.initial_pips.isna().all())
        self.assertAlmostEqual(p.continuation_pips.iloc[0], 1.)
        table = component_summary(p).set_index(['cohort', 'component'])
        for cohort in ['confirmed_directed', 'state_directed']:
            self.assertEqual(table.loc[(cohort, 'initial'), 'available_onsets'], 0)
            self.assertEqual(table.loc[(cohort, 'continuation'), 'available_onsets'], 1)
            self.assertEqual(table.loc[(cohort, 'total'), 'available_onsets'], 0)
            self.assertEqual(table.loc[(cohort, 'joint'), 'available_onsets'], 0)
            self.assertTrue(pd.isna(table.loc[(cohort, 'continuation'), 'mean_initial_pips']))
        _, stats = summarize_horizon(p)
        self.assertEqual(stats['continuation_available_without_premark'], 1)
        self.assertEqual(stats['joint_missing'], 1)

    def test_initial_remains_available_after_sample_censoring(self):
        p = construct(stop=20)['paths'].loc[lambda f: f.horizon_sec.eq(300)]
        table = component_summary(p).set_index(['cohort', 'component'])
        self.assertEqual(table.loc[('confirmed_directed', 'initial'), 'available_onsets'], 1)
        self.assertEqual(table.loc[('confirmed_directed', 'continuation'), 'available_onsets'], 0)
        self.assertEqual(table.loc[('confirmed_directed', 'total'), 'available_onsets'], 0)
        self.assertAlmostEqual(table.loc[('confirmed_directed', 'initial'), 'mean_initial_pips'], 2.)

    def test_joint_sample_decomposition_and_denominators(self):
        o = onsets([10, 20], [1., -1.], state=[True, False])
        r = construct(o)
        p = r['paths'].loc[lambda f: f.horizon_sec.eq(1)]
        table = component_summary(p).set_index(['cohort', 'component'])
        for cohort, n in [('confirmed_directed', 2), ('state_directed', 1)]:
            row = table.loc[(cohort, 'joint')]
            self.assertEqual(row.available_onsets, n)
            self.assertAlmostEqual(row.mean_total_pips, row.mean_initial_pips + row.mean_continuation_pips)
        np.testing.assert_allclose(r['paths'].total_pips,
                                   r['paths'].initial_pips + r['paths'].continuation_pips,
                                   rtol=0, atol=1e-9)

    def test_zero_nonfinite_and_tiny_direction_keep_all_rows(self):
        directions = [0., np.nan, np.inf, np.nextafter(0., 1.)]
        r = construct(onsets([10, 20, 30, 40], directions))
        self.assertEqual(len(r['paths']), 1204)
        self.assertEqual(r['anchors'].direction_status.tolist(), ['zero', 'missing', 'missing', 'positive'])
        self.assertEqual(r['anchors'].direction_available.tolist(), [False, False, False, True])
        p = r['paths'].loc[lambda f: f.horizon_sec.eq(0)]
        table = component_summary(p)
        self.assertTrue(table.cohort_onsets.eq(1).all())
        stats = anchor_statistics(r['anchors'])
        self.assertEqual(stats['direction_counts'], {'positive': 1, 'negative': 0, 'zero': 1, 'missing': 2})
        self.assertEqual(stats['minimum_nonzero_abs_onset_ofi'], np.nextafter(0., 1.))

    def test_empty_cohorts_preserve_summary_rows_and_missing_means(self):
        r = construct(onsets(confirmed=[False], state=[False]))
        p = r['paths'].loc[lambda f: f.horizon_sec.eq(1)]
        summary, stats = summarize_horizon(p)
        self.assertEqual(len(summary), 10)
        self.assertTrue(summary.observations.eq(0).all())
        self.assertTrue(summary.mean_total_pips.isna().all())
        self.assertIsNone(stats['samples']['horizon_confirmed']['mean_total_pips'])
        components = component_summary(p)
        self.assertEqual(len(components), 8)
        self.assertTrue(components.cohort_onsets.eq(0).all())
        self.assertTrue(components.mean_continuation_pips.isna().all())

    def test_horizon_summary_matches_existing_frozen_definition(self):
        o = onsets([10, 30], [1., -1.], state=[True, False])
        legacy = construct_price_responses(o, quotes(), support(), policy='structural_only',
                                          development_end=at(4000))
        expected, old_stats = summarize_responses(legacy)
        actual = construct(o)
        for horizon in [0, 60, 300]:
            frame, stats = summarize_horizon(actual['paths'].loc[lambda f: f.horizon_sec.eq(horizon)])
            assert_frame_equal(frame, expected.loc[expected.horizon_sec.eq(horizon)].reset_index(drop=True))
            prior = old_stats['horizons'][str(horizon)]
            prior['endpoint_status_all_fragments']['beyond_sample_end'] = prior['endpoint_status_all_fragments'].pop('beyond_development_end')
            self.assertEqual(stats, prior)


class PanelResponseValidationTests(unittest.TestCase):
    def test_parent_columns_and_development_label_are_preserved(self):
        original = onsets()
        original['inherited_measurement'] = 17.25
        original['inherited_missing'] = np.nan
        output = construct(original)['anchors']
        assert_frame_equal(output[original.columns], original)

    def test_absent_or_modified_selected_onset_is_rejected(self):
        full = onsets([10, 20])
        for column, value in [('episode_id', 99), ('bin_start', at(9)), ('available_at', at(12)),
                              ('quote_ofi', -1.), ('support_id', 1), ('state_sample_eligible', False)]:
            with self.subTest(column=column):
                selected = full.iloc[:1].copy()
                selected.loc[selected.index[0], column] = value
                with self.assertRaises(ValueError):
                    construct(selected, all_o=full)

    def test_invalid_global_stream_is_rejected_even_outside_selected_batch(self):
        full = onsets([10, 20])
        selected = full.iloc[:1]
        for changed in [full.iloc[::-1], pd.concat([full, full.iloc[1:]], ignore_index=True)]:
            with self.assertRaises(ValueError):
                construct(selected, all_o=changed)
        with self.assertRaises(ValueError):
            construct(selected, all_o=full, stop=15)

    def test_horizon_summaries_reject_duplicate_ids_or_mixed_horizons(self):
        paths = construct()['paths']
        for frame in [paths, pd.concat([paths.iloc[:1], paths.iloc[:1]]), paths.iloc[:0]]:
            for function in [summarize_horizon, component_summary]:
                with self.assertRaises(ValueError):
                    function(frame)


if __name__ == '__main__':
    unittest.main()

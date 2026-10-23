"""Small adversarial examples for the frozen analysis and overlap ledger."""
import unittest

import numpy as np
import pandas as pd

from src.analysis_ledger import (specification, assemble_ledger,
                                 dependence_tables, summarize_ledger)


ORIGIN = pd.Timestamp('2024-04-28 00:00:00', tz='UTC')


def at(seconds):
    return pd.Timestamp(ORIGIN.value + round(float(seconds) * 1_000_000_000),
                        unit='ns', tz='UTC')


def parent_frames(ends=(0, 400), ids=(9, 2)):
    anchor_rows, measurement_rows, direction_rows, path_rows = [], [], [], []
    for endpoint, event_id in zip(ends, ids):
        anchor_rows.append({'episode_id': event_id, 'bin_start': at(endpoint),
            'available_at': at(endpoint + 1), 'support_id': 0,
            'pre_rv_300s_5s_bps': 1., 'pre_quote_ofi_300s': -2.,
            'pre_spread_pips': .3, 'pre_displayed_size': 10., 'pre_quote_age_sec': .1,
            'state_history_available': True, 'state_sample_eligible': True,
            'pre_mark_max_age_sec': 1., 'pre_mark_repeat_fraction': .2,
            'pre_recorded_updates_300s': 100, 'pre_max_observed_silence_sec': 5.,
            'quote_ofi': 2., 'distance': 5., 'z_ofi': 1., 'z_intensity': 2., 'z_duration': 3.,
            'direction': 1., 'direction_available': True, 'onset_confirmed': True,
            'pre_mark_at': at(endpoint - .1), 'detection_mark_at': at(endpoint + .5),
            'baseline_quote_available': True})
        for frequency in [1, 5, 10, 15, 30, 60]:
            measurement_rows.append({'kind': 'development', 'episode_id': event_id,
                'frequency_sec': frequency, 'end_at': at(endpoint), 'onset_confirmed': True,
                'rv_bps': 1. if frequency == 5 else 2., 'status': 'available',
                'first_quote_at': at(endpoint - 301), 'last_quote_at': at(endpoint - .1),
                'common_age_le_15': True, 'common_age_le_30': True, 'common_age_le_60': True})
        direction_rows.append({'episode_id': event_id, 'bin_start': at(endpoint),
            'onset_confirmed': True, 'inherited_direction': 1., 'inherited_net': 2.,
            'cancellation_ratio': 1.})
        for horizon in [0, 60, 300]:
            path_rows.append({'episode_id': event_id, 'bin_start': at(endpoint),
                'detected_at': at(endpoint + 1), 'horizon_sec': horizon,
                'target_at': at(endpoint + 1 + horizon), 'endpoint_status': 'available',
                'endpoint_quote_at': at(endpoint + .5 if horizon == 0 else endpoint + horizon + .75),
                'endpoint_age_sec': .5 if horizon == 0 else .25,
                'pre_age_sec': .1, 'detection_age_sec': .5,
                'initial_pips': 1., 'continuation_pips': horizon / 300.,
                'total_pips': 1. + horizon / 300., 'continuation_quote_available': True,
                'total_quote_available': True, 'triplet_quote_available': True,
                'triplet_age_le_15': True, 'triplet_age_le_30': True, 'triplet_age_le_60': True,
                'direction': 1., 'direction_available': True,
                'state_sample_eligible': True, 'onset_confirmed': True})
    return tuple(pd.DataFrame(rows) for rows in
                 [anchor_rows, measurement_rows, direction_rows, path_rows])


def set_direction(frames, event_id, value):
    a, _, d, p = frames
    available = bool(np.isfinite(value) and value != 0)
    a.loc[a.episode_id.eq(event_id), ['direction', 'direction_available']] = [value, available]
    d.loc[d.episode_id.eq(event_id), 'inherited_direction'] = value
    p.loc[p.episode_id.eq(event_id), ['direction', 'direction_available']] = [value, available]
    if not available:
        p.loc[p.episode_id.eq(event_id), ['initial_pips', 'continuation_pips', 'total_pips']] = np.nan


def direct_dependence(starts, ends, ids=None, onset_times=None):
    """Explicit intervals isolate overlap accounting from the parent-data joins."""
    count = len(starts)
    ids = list(range(count)) if ids is None else list(ids)
    onset_times = starts if onset_times is None else onset_times
    rows, paths = [], []
    for event_id, left, right, onset in zip(ids, starts, ends, onset_times):
        stamp = at(onset)
        day = stamp.normalize()
        week = day - pd.Timedelta(days=(stamp.dayofweek + 1) % 7)
        rows.append({'episode_id': event_id, 'nominal_start_at': at(left + 1),
                     'source_start_at': at(left), 'onset_day': day, 'onset_week_start': week})
        paths.append({'episode_id': event_id, 'horizon_sec': 300,
                      'target_at': at(right), 'eligible_continuation': True})
    return dependence_tables(pd.DataFrame(rows), pd.DataFrame(paths))


class LedgerJoinTests(unittest.TestCase):
    def test_ids_drive_join_despite_nonmonotonic_ids_and_shuffled_parent_rows(self):
        frames = parent_frames()
        result = assemble_ledger(*[f.iloc[::-1].reset_index(drop=True) for f in frames])
        self.assertEqual(result['anchors'].episode_id.tolist(), [2, 9])
        self.assertEqual(len(result['outcomes']), 6)
        indexed = result['anchors'].set_index('episode_id')
        self.assertEqual(indexed.loc[9, 'bin_start'], at(0))
        self.assertEqual(indexed.loc[2, 'bin_start'], at(400))
        self.assertEqual(indexed.loc[9, 'pre_quote_ofi_300s'], -2.)

    def test_parent_frames_and_preserved_anchor_fields_are_unchanged(self):
        frames = parent_frames()
        originals = tuple(f.copy(deep=True) for f in frames)
        result = assemble_ledger(*frames)
        for frame, before in zip(frames, originals):
            pd.testing.assert_frame_equal(frame, before)
        common = [name for name in frames[0] if name in result['anchors']]
        pd.testing.assert_frame_equal(result['anchors'][common].reset_index(drop=True),
                                      originals[0].sort_values('episode_id')[common].reset_index(drop=True))

    def test_missing_or_duplicate_join_rows_fail(self):
        for parent in range(4):
            for change in ['missing', 'duplicate']:
                with self.subTest(parent=parent, change=change):
                    frames = list(parent_frames())
                    if parent == 1:
                        row = frames[parent].index[frames[parent].frequency_sec.eq(5)][0]
                    else:
                        row = frames[parent].index[0]
                    if change == 'missing':
                        frames[parent] = frames[parent].drop(index=row)
                    else:
                        frames[parent] = pd.concat([frames[parent], frames[parent].loc[[row]]], ignore_index=True)
                    with self.assertRaises(ValueError):
                        assemble_ledger(*frames)

    def test_inconsistent_parent_times_and_values_fail(self):
        mutations = [(1, 'end_at', at(99)), (2, 'bin_start', at(99)),
                     (3, 'detected_at', at(99)), (3, 'target_at', at(99)),
                     (2, 'inherited_direction', -1.), (2, 'inherited_net', 999.)]
        for parent, column, value in mutations:
            with self.subTest(parent=parent, column=column):
                frames = list(parent_frames())
                row = frames[1].index[frames[1].frequency_sec.eq(5)][0] if parent == 1 else 0
                frames[parent].loc[row, column] = value
                with self.assertRaises(ValueError):
                    assemble_ledger(*frames)

    def test_quote_at_boundary_is_not_treated_as_strictly_past(self):
        for parent, column, value in [(0, 'pre_mark_at', at(0)),
                                      (0, 'detection_mark_at', at(1)),
                                      (1, 'first_quote_at', at(-300)),
                                      (1, 'last_quote_at', at(0)),
                                      (3, 'endpoint_quote_at', at(1))]:
            with self.subTest(column=column):
                frames = list(parent_frames())
                if parent == 1:
                    row = frames[1].index[(frames[1].episode_id == 9) & frames[1].frequency_sec.eq(5)][0]
                else:
                    row = 0
                frames[parent].loc[row, column] = value
                with self.assertRaises(ValueError):
                    assemble_ledger(*frames)

    def test_reference_measurement_must_reconcile_with_inherited_state(self):
        frames = list(parent_frames())
        frames[1].loc[frames[1].frequency_sec.eq(5), 'rv_bps'] = 3.
        with self.assertRaisesRegex(ValueError, 'reference differs'):
            assemble_ledger(*frames)

    def test_common_grid_age_masks_must_agree_across_frequencies(self):
        frames = list(parent_frames())
        frames[1].loc[frames[1].frequency_sec.eq(15), 'common_age_le_15'] = False
        with self.assertRaisesRegex(ValueError, 'common-grid'):
            assemble_ledger(*frames)

    def test_calibration_measurements_are_not_added_as_control_events(self):
        frames = list(parent_frames())
        cal = frames[1].copy()
        cal['kind'] = 'calibration'
        cal['episode_id'] = -1
        frames[1] = pd.concat([frames[1], cal], ignore_index=True)
        ledger = assemble_ledger(*frames)
        self.assertEqual(len(ledger['anchors']), 2)
        self.assertEqual(len(ledger['outcomes']), 6)

    def test_fractional_episode_ids_are_rejected_before_dependence_truncation(self):
        frames = list(parent_frames())
        for f in frames:
            f['episode_id'] = f.episode_id.astype(float) + .5
        with self.assertRaises(ValueError):
            assemble_ledger(*frames)

    def test_missing_or_nonboolean_quality_flags_cannot_be_coerced_to_true(self):
        fields = [(0, 'baseline_quote_available'), (0, 'state_history_available'),
                  (0, 'state_sample_eligible'), (0, 'direction_available'),
                  (0, 'onset_confirmed'), (1, 'onset_confirmed'),
                  (1, 'common_age_le_15'), (2, 'onset_confirmed'),
                  (3, 'continuation_quote_available'), (3, 'total_quote_available'),
                  (3, 'triplet_quote_available'), (3, 'triplet_age_le_15')]
        for parent, column in fields:
            for invalid in [np.nan, 1, 'False']:
                with self.subTest(parent=parent, column=column, invalid=invalid):
                    frames = list(parent_frames())
                    frames[parent][column] = frames[parent][column].astype(object)
                    row = frames[1].index[frames[1].frequency_sec.eq(5)][0] if parent == 1 else 0
                    frames[parent].loc[row, column] = invalid
                    with self.assertRaises(ValueError):
                        assemble_ledger(*frames)

    def test_dependency_eligibility_flag_must_be_known_boolean(self):
        tables = assemble_ledger(*parent_frames())
        for invalid in [np.nan, 1, 'False']:
            with self.subTest(invalid=invalid):
                outcomes = tables['outcomes'].copy()
                outcomes['eligible_continuation'] = outcomes.eligible_continuation.astype(object)
                outcomes.loc[0, 'eligible_continuation'] = invalid
                with self.assertRaises(ValueError):
                    dependence_tables(tables['anchors'], outcomes)


class LedgerEligibilityTests(unittest.TestCase):
    def test_valid_zero_variation_is_retained(self):
        frames = list(parent_frames())
        frames[0]['pre_rv_300s_5s_bps'] = 0.
        frames[1].loc[frames[1].frequency_sec.eq(5), 'rv_bps'] = 0.
        ledger = assemble_ledger(*frames)
        self.assertTrue(ledger['anchors'].pre_state_eligible.all())
        self.assertTrue(ledger['outcomes'].eligible_continuation.all())

    def test_zero_or_nonfinite_direction_keeps_rows_without_alignment_eligibility(self):
        for value in [0., np.nan, np.inf]:
            with self.subTest(direction=value):
                frames = parent_frames()
                set_direction(frames, 9, value)
                result = assemble_ledger(*frames)
                self.assertEqual(len(result['anchors']), 2)
                self.assertEqual(len(result['outcomes']), 6)
                self.assertFalse(result['anchors'].set_index('episode_id').loc[9, 'pre_state_eligible'])
                self.assertFalse(result['outcomes'].loc[lambda x: x.episode_id.eq(9), 'eligible_continuation'].any())

    def test_continuation_does_not_require_initial_or_total_price_pair(self):
        frames = list(parent_frames())
        frames[0].loc[0, 'baseline_quote_available'] = False
        frames[0].loc[0, 'pre_mark_at'] = pd.NaT
        selected = frames[3].episode_id.eq(9)
        frames[3].loc[selected, ['initial_pips', 'total_pips']] = np.nan
        frames[3].loc[selected, ['total_quote_available', 'triplet_quote_available',
                                 'triplet_age_le_15', 'triplet_age_le_30', 'triplet_age_le_60']] = False
        result = assemble_ledger(*frames)['outcomes'].loc[lambda x: x.episode_id.eq(9)]
        self.assertTrue(result.eligible_continuation.all())
        self.assertFalse(result.eligible_initial.any())
        self.assertFalse(result.eligible_total.any())
        self.assertFalse(result.eligible_joint.any())

    def test_future_missingness_and_age_do_not_change_anchor_cohort(self):
        frames = list(parent_frames())
        before = assemble_ledger(*frames)
        selected = frames[3].horizon_sec.eq(300)
        frames[3].loc[selected, ['continuation_quote_available', 'total_quote_available',
                                 'triplet_quote_available', 'triplet_age_le_15']] = False
        frames[3].loc[selected, ['continuation_pips', 'total_pips']] = np.nan
        frames[3].loc[selected, 'endpoint_quote_at'] = pd.NaT
        after = assemble_ledger(*frames)
        pd.testing.assert_frame_equal(before['anchors'], after['anchors'])
        self.assertFalse(after['outcomes'].loc[selected.to_numpy(), 'eligible_continuation'].any())
        self.assertTrue(after['outcomes'].loc[after['outcomes'].horizon_sec.eq(60), 'eligible_continuation'].all())

    def test_pre_age_response_age_and_intersection_are_distinct(self):
        frames = list(parent_frames())
        frames[1].loc[frames[1].episode_id.eq(9), 'common_age_le_15'] = False
        frames[3].loc[frames[3].episode_id.eq(2), 'triplet_age_le_15'] = False
        ledger = assemble_ledger(*frames)
        a = ledger['anchors']
        self.assertTrue(a.pre_state_eligible.all())
        o = ledger['outcomes'].loc[lambda x: x.horizon_sec.eq(300)].set_index('episode_id')
        self.assertFalse(o.loc[9, 'eligible_pre_age_15'])
        self.assertTrue(o.loc[9, 'eligible_response_age_15'])
        self.assertTrue(o.loc[2, 'eligible_pre_age_15'])
        self.assertFalse(o.loc[2, 'eligible_response_age_15'])
        self.assertFalse(o.eligible_both_age_15.any())

    def test_fixed_direction_sensitivity_uses_strict_threshold(self):
        frames = list(parent_frames(ends=[0, 400, 800], ids=[9, 2, 12]))
        frames[2]['cancellation_ratio'] = [.01, .05, .10]
        o = assemble_ledger(*frames)['outcomes'].loc[lambda x: x.horizon_sec.eq(300)].set_index('episode_id')
        self.assertFalse(o.loc[9, 'eligible_direction_ratio_gt_1pct'])
        self.assertTrue(o.loc[2, 'eligible_direction_ratio_gt_1pct'])
        self.assertFalse(o.loc[2, 'eligible_direction_ratio_gt_5pct'])
        self.assertTrue(o.loc[12, 'eligible_direction_ratio_gt_5pct'])
        self.assertFalse(o.eligible_direction_ratio_gt_10pct.any())
        self.assertTrue(o.eligible_continuation.all())

    def test_frequency_sensitivity_failure_does_not_reselect_primary(self):
        frames = list(parent_frames())
        selected = frames[1].frequency_sec.eq(15) & frames[1].episode_id.eq(9)
        frames[1].loc[selected, 'rv_bps'] = np.nan
        frames[1].loc[selected, 'status'] = 'unavailable'
        result = assemble_ledger(*frames)['outcomes'].loc[lambda x: x.episode_id.eq(9)]
        self.assertTrue(result.eligible_continuation.all())
        self.assertFalse(result.eligible_frequency_15.any())
        self.assertTrue(result.eligible_frequency_30.all())

    def test_nonfinite_prior_flow_is_missing_not_zero(self):
        frames = list(parent_frames())
        frames[0].loc[0, 'pre_quote_ofi_300s'] = np.nan
        a = assemble_ledger(*frames)['anchors'].set_index('episode_id')
        self.assertFalse(a.loc[9, 'pre_state_eligible'])
        self.assertTrue(np.isnan(a.loc[9, 'pre_quote_ofi_300s']))

    def test_source_start_uses_actual_earliest_quote_before_nominal_window(self):
        frames = list(parent_frames())
        frames[1].loc[frames[1].episode_id.eq(9), 'first_quote_at'] = at(-350)
        a = assemble_ledger(*frames)['anchors'].set_index('episode_id')
        self.assertEqual(a.loc[9, 'nominal_start_at'], at(-300))
        self.assertEqual(a.loc[9, 'source_start_at'], at(-350))


class TimingRoleTests(unittest.TestCase):
    def test_state_values_and_detection_cohort_have_different_roles(self):
        s = specification()
        self.assertEqual(s['working_state'], 'pre_rv_bps_5s')
        self.assertEqual(s['frequency_sensitivities_seconds'], [15, 30])
        roles = s['anchor_column_roles']
        self.assertEqual(roles['pre_quote_ofi_300s'], 'strictly_before_onset')
        self.assertEqual(roles['direction'], 'detection_time')
        for field in ['pre_state_eligible', 'state_sample_eligible']:
            self.assertEqual(roles[field], 'detection_time_cohort_flag_not_predictor')
        self.assertFalse(s['inference_approved'])
        self.assertFalse(s['latent_volatility_validated'])
        self.assertFalse(s['causal_identification'])

    def test_onset_date_not_detection_date_and_sunday_week_boundary(self):
        a = assemble_ledger(*parent_frames(ends=[-1, 0, 86400], ids=[9, 2, 12]))['anchors'].set_index('episode_id')
        self.assertEqual(a.loc[9, 'onset_day'], pd.Timestamp('2024-04-27', tz='UTC'))
        self.assertEqual(a.loc[9, 'onset_week_start'], pd.Timestamp('2024-04-21', tz='UTC'))
        self.assertEqual(a.loc[2, 'onset_week_start'], ORIGIN)
        self.assertEqual(a.loc[12, 'onset_week_start'], ORIGIN)
        self.assertEqual(a.loc[9, 'onset_slot_15m_utc'], 95)
        self.assertEqual(a.loc[2, 'onset_hour_utc'], 0)
        self.assertEqual(a.loc[2, 'onset_weekday_utc'], 6)


class DependenceAccountingTests(unittest.TestCase):
    def test_one_nanosecond_intersection_is_retained_as_positive_overlap(self):
        d = direct_dependence([0, 10 - 1e-9], [10, 20], ids=[9, 2])
        self.assertEqual(len(d['pairs']), 1)
        self.assertEqual(d['pairs'].overlap_seconds.iloc[0], 1e-9)
        self.assertEqual(d['components'].events.tolist(), [2])

    def test_exact_half_open_touching_is_not_overlap(self):
        d = direct_dependence([0, 10], [10, 20], ids=[9, 2])
        self.assertTrue(d['pairs'].empty)
        self.assertEqual(len(d['components']), 2)
        self.assertTrue(d['intervals'].degree.eq(0).all())

    def test_chain_connects_component_without_fabricating_all_pairs(self):
        d = direct_dependence([0, 9, 18], [10, 19, 28], ids=[50, 2, 10])
        pairs = d['pairs']
        self.assertEqual(set(zip(pairs.episode_id_left, pairs.episode_id_right)), {(2, 50), (2, 10)})
        self.assertTrue(pairs.overlap_seconds.eq(1).all())
        self.assertEqual(d['components'].events.tolist(), [3])
        degrees = d['intervals'].set_index('episode_id').degree.to_dict()
        self.assertEqual(degrees, {2: 2, 10: 1, 50: 1})

    def test_nested_intervals_and_nonmonotonic_ids(self):
        d = direct_dependence([0, 10, 30], [100, 20, 40], ids=[90, 3, 7])
        pairs = d['pairs']
        self.assertEqual(set(zip(pairs.episode_id_left, pairs.episode_id_right)), {(3, 90), (7, 90)})
        self.assertTrue(pairs.overlap_seconds.eq(10).all())
        self.assertEqual(d['components'].events.tolist(), [3])

    def test_source_extension_catches_overlap_missed_by_nominal_starts(self):
        frames = parent_frames(ends=[0, 602], ids=[9, 2])
        ledger = assemble_ledger(*frames)
        d = dependence_tables(ledger['anchors'], ledger['outcomes'])
        # first h300 interval ends301; second actualstart301 merely touches.
        self.assertTrue(d['pairs'].loc[d['pairs'].horizon_sec.eq(300)].empty)
        ledger['anchors'].loc[ledger['anchors'].episode_id.eq(2), 'source_start_at'] = at(300)
        d = dependence_tables(ledger['anchors'], ledger['outcomes'])
        p = d['pairs'].loc[d['pairs'].horizon_sec.eq(300)]
        self.assertEqual(len(p), 1)
        self.assertEqual(p.overlap_seconds.iloc[0], 1.)

    def test_cross_day_and_week_pairs_are_reported_from_onset_labels(self):
        d = direct_dependence([-10, -5], [10, 15], ids=[9, 2], onset_times=[-1, 0])
        self.assertEqual(len(d['pairs']), 1)
        self.assertTrue(d['pairs'].cross_onset_day.iloc[0])
        self.assertTrue(d['pairs'].cross_onset_week.iloc[0])
        calendar = d['calendar']
        self.assertEqual(len(calendar.loc[calendar.block_type.eq('day')]), 2)
        self.assertEqual(len(calendar.loc[calendar.block_type.eq('week')]), 2)
        self.assertTrue(calendar.share.eq(.5).all())

    def test_dependence_excludes_missing_continuation_but_keeps_anchor(self):
        frames = list(parent_frames(ends=[0, 100], ids=[9, 2]))
        selected = frames[3].episode_id.eq(2) & frames[3].horizon_sec.eq(300)
        frames[3].loc[selected, 'continuation_quote_available'] = False
        frames[3].loc[selected, 'continuation_pips'] = np.nan
        tables = assemble_ledger(*frames)
        d = dependence_tables(tables['anchors'], tables['outcomes'])
        self.assertEqual(len(tables['anchors']), 2)
        self.assertEqual(d['intervals'].loc[d['intervals'].horizon_sec.eq(300), 'episode_id'].tolist(), [9])

    def test_empty_eligible_cohort_has_schema_and_zero_summary(self):
        frames = parent_frames()
        set_direction(frames, 9, 0.)
        set_direction(frames, 2, 0.)
        tables = assemble_ledger(*frames)
        d = dependence_tables(tables['anchors'], tables['outcomes'])
        for frame in d.values():
            self.assertTrue(frame.empty)
            self.assertGreater(len(frame.columns), 0)
        summary = summarize_ledger(tables, d)
        self.assertEqual(summary['anchor_rows'], 2)
        self.assertEqual(summary['pre_state_eligible'], 0)
        for horizon in summary['horizons']:
            self.assertEqual(horizon['dependence']['events'], 0)
            self.assertEqual(horizon['dependence']['day']['blocks'], 0)
            self.assertIsNone(horizon['continuation_mean_own_sample_pips'])

    def test_joint_summary_does_not_mix_component_samples(self):
        frames = list(parent_frames())
        frames[0].loc[0, 'baseline_quote_available'] = False
        selected = frames[3].episode_id.eq(9)
        frames[3].loc[selected, ['initial_pips', 'total_pips']] = np.nan
        frames[3].loc[selected, ['total_quote_available', 'triplet_quote_available']] = False
        frames[3].loc[selected & frames[3].horizon_sec.eq(300), 'continuation_pips'] = 99.
        tables = assemble_ledger(*frames)
        summary = summarize_ledger(tables, dependence_tables(tables['anchors'], tables['outcomes']))
        h = next(row for row in summary['horizons'] if row['horizon_sec'] == 300)
        self.assertEqual(h['continuation_mean_own_sample_pips'], 50.)
        self.assertEqual(h['joint_component_means_pips'], {'initial': 1., 'continuation': 1., 'total': 2.})
        self.assertEqual(h['mask_counts']['eligible_joint'], 1)
        self.assertEqual(h['mask_counts']['eligible_continuation'], 2)
        self.assertFalse(summary['production_inference_approved'])


if __name__ == '__main__':
    unittest.main()

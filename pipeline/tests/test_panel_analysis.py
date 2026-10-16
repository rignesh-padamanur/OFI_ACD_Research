"""Counterexamples for the declared chronological ledger/input extension."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.analysis_ledger import assemble_ledger, dependence_tables
from src.estimation_inputs import build_inputs, CORE, CALENDAR, CURVATURE
from src.panel_analysis import (
    PANEL_START, ANALYSIS_START, PANEL_END, SEGMENTS, analysis_segments,
    assemble_panel_ledger, build_panel_inputs, panel_dependence,
    dependence_summary, daily_calendar,
)

_fixture_spec = importlib.util.spec_from_file_location(
    '_panel_ledger_parent_fixture', Path(__file__).with_name('test_analysis_ledger.py'))
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)


def parents(times=None):
    if times is None:
        times = [PANEL_START, ANALYSIS_START-pd.Timedelta(seconds=1),
                 ANALYSIS_START, ANALYSIS_START+pd.Timedelta(seconds=400)]
    offsets = [(time-_fixture_module.ORIGIN).total_seconds() for time in times]
    a, m, d, p = _fixture_module.parent_frames(offsets, tuple(range(len(times))))
    m = m.loc[m.frequency_sec.isin([5, 15, 30])].reset_index(drop=True)
    m['kind'] = 'chronological_onset'
    m['window_id'] = m.episode_id
    m['state_sample_eligible'] = True
    m['n_returns'] = 300 // m.frequency_sec
    m['rv_bps2'] = m.rv_bps**2
    m['max_mark_age_sec'] = 1.
    m['common_max_mark_age_sec'] = 1.
    m['repeat_fraction'] = .2
    m['zero_return_fraction'] = .1
    for cap in [15, 30, 60]:
        m['own_age_le_'+str(cap)] = True
    return a, m, d, p


def scaling():
    return {'cQ': 2., 'cV': 1., 'sV': .5,
            'reference_start': '2024-02-26T00:00:00+00:00',
            'reference_cutoff': PANEL_START.isoformat(),
            'reference_population': 'synthetic fixed test reference',
            'Q_mean_subtracted': False, 'response_or_candidate_filter': False}


def inputs(frames=None):
    ledger = assemble_panel_ledger(*(parents() if frames is None else frames))
    return build_panel_inputs(ledger['anchors'], ledger['outcomes'], scaling())


class PanelLedgerAdapterTests(unittest.TestCase):
    def test_adapter_preserves_parents_and_matches_old_join(self):
        frames = parents()
        before = tuple(frame.copy(deep=True) for frame in frames)
        result = assemble_panel_ledger(*frames)
        for actual, expected in zip(frames, before):
            pd.testing.assert_frame_equal(actual, expected)
        old_measurements = frames[1].copy(); old_measurements['kind'] = 'development'
        old = assemble_ledger(frames[0], old_measurements, frames[2], frames[3])
        for name in ['anchors', 'outcomes']:
            pd.testing.assert_frame_equal(result[name].drop(columns='analysis_segment'), old[name])
        self.assertTrue(frames[1].kind.eq('chronological_onset').all())
        self.assertFalse(result['adapter']['saved_parent_labels_changed'])

    def test_fixed_half_open_partition_uses_onset_not_detection_or_state_source(self):
        result = assemble_panel_ledger(*parents())
        self.assertEqual(result['anchors'].analysis_segment.tolist(),
                         ['reused_development', 'reused_development', 'later_reanalysis', 'later_reanalysis'])
        row = result['anchors'].iloc[1]
        self.assertEqual(row.available_at, ANALYSIS_START)
        self.assertEqual(row.analysis_segment, 'reused_development')
        self.assertLess(result['anchors'].iloc[2].source_start_at, ANALYSIS_START)

    def test_only_declared_full_panel_dates_and_whole_seconds_accepted(self):
        for invalid in [PANEL_START-pd.Timedelta(seconds=1), PANEL_END,
                        ANALYSIS_START+pd.Timedelta(milliseconds=500), pd.NaT,
                        ANALYSIS_START.tz_localize(None)]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analysis_segments([invalid])
        self.assertEqual(analysis_segments([PANEL_END-pd.Timedelta(seconds=1)]).tolist(), ['later_reanalysis'])

    def test_measurement_kind_mixture_and_legacy_labels_fail(self):
        for value in ['development', 'calibration', 'unexpected', None]:
            frames = list(parents())
            frames[1].loc[0, 'kind'] = value
            with self.subTest(kind=value), self.assertRaises(ValueError):
                assemble_panel_ledger(*frames)

    def test_measurement_identity_frequency_and_inherited_flags_must_match(self):
        for column, value in [('window_id', 999), ('frequency_sec', 10),
                               ('state_sample_eligible', False), ('end_at', PANEL_START+pd.Timedelta(seconds=10))]:
            frames = list(parents()); frames[1].loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                assemble_panel_ledger(*frames)
        for modification in ['drop', 'duplicate']:
            frames = list(parents())
            frames[1] = frames[1].iloc[1:] if modification == 'drop' else pd.concat([frames[1], frames[1].iloc[:1]])
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                assemble_panel_ledger(*frames)

    def test_only_three_selected_response_horizons_are_accepted(self):
        frames = list(parents()); extra = frames[3].iloc[:1].copy(); extra['horizon_sec'] = 1
        frames[3] = pd.concat([frames[3], extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Read only'):
            assemble_panel_ledger(*frames)


class PanelInputTests(unittest.TestCase):
    def test_all_rows_and_scaling_retained_main_designs_exclude_development(self):
        ledger = assemble_panel_ledger(*parents())
        before_a, before_o = ledger['anchors'].copy(deep=True), ledger['outcomes'].copy(deep=True)
        scale = scaling(); frozen = deepcopy(scale)
        result = build_panel_inputs(ledger['anchors'], ledger['outcomes'], scale)
        pd.testing.assert_frame_equal(ledger['anchors'], before_a)
        pd.testing.assert_frame_equal(ledger['outcomes'], before_o)
        self.assertEqual(scale, frozen)
        self.assertEqual(len(result['features']), 4)
        self.assertEqual(len(result['samples']), 12)
        self.assertEqual(set(result['design_rows'].episode_id), {2, 3})
        self.assertEqual(result['statistics']['full_panel_anchor_rows'], 4)
        self.assertEqual(result['statistics']['main_input_statistics']['anchor_rows'], 2)
        self.assertEqual(result['development_statistics']['anchor_rows'], 2)
        self.assertEqual(result['specification']['scaling'], frozen)
        self.assertFalse(result['specification']['chronology']['untouched_holdout'])

    def test_segmentwise_feature_math_and_masks_equal_unchanged_07a(self):
        ledger = assemble_panel_ledger(*parents())
        result = build_panel_inputs(ledger['anchors'], ledger['outcomes'], scaling())
        for segment in SEGMENTS:
            a = ledger['anchors'].loc[ledger['anchors'].analysis_segment.eq(segment)]
            o = ledger['outcomes'].loc[ledger['outcomes'].analysis_segment.eq(segment)]
            expected = build_inputs(a, o, scaling())
            actual_f = result['features'].loc[result['features'].analysis_segment.eq(segment)].reset_index(drop=True)
            actual_s = result['samples'].loc[result['samples'].analysis_segment.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_f, expected['features'])
            pd.testing.assert_frame_equal(actual_s.drop(columns=['main_eligible_'+name for name in ['initial', 'continuation', 'total', 'joint']]), expected['samples'])
            if segment == 'later_reanalysis':
                pd.testing.assert_frame_equal(result['design_rows'], expected['design_rows'])
                pd.testing.assert_frame_equal(result['calendar_support'], expected['calendar_support'])
                self.assertEqual(result['schemas'], expected['schemas'])

    def test_development_extremes_and_responses_cannot_affect_main_features_or_designs(self):
        original = inputs()
        frames = list(parents())
        frames[0].loc[frames[0].episode_id.lt(2), 'pre_quote_ofi_300s'] = 1e9
        frames[3].loc[frames[3].episode_id.lt(2), 'continuation_pips'] = -1000.
        frames[3].loc[frames[3].episode_id.lt(2), 'total_pips'] = -999.
        changed = inputs(frames)
        pd.testing.assert_frame_equal(original['design_rows'], changed['design_rows'], check_exact=True)
        self.assertEqual(original['schemas'], changed['schemas'])

    def test_return_values_never_select_main_inputs(self):
        frames = list(parents()); before = inputs(frames)
        frames[3]['initial_pips'] *= -3
        frames[3]['continuation_pips'] *= -3
        frames[3]['total_pips'] *= -3
        after = inputs(frames)
        pd.testing.assert_frame_equal(before['features'], after['features'])
        pd.testing.assert_frame_equal(before['design_rows'], after['design_rows'])

    def test_continuation_does_not_require_initial_or_joint_availability(self):
        frames = list(parents())
        frames[0].loc[2, 'baseline_quote_available'] = False
        frames[0].loc[2, 'pre_mark_at'] = pd.NaT
        selected = frames[3].episode_id.eq(2)
        frames[3].loc[selected, ['initial_pips', 'total_pips']] = np.nan
        frames[3].loc[selected, ['total_quote_available', 'triplet_quote_available',
                                 'triplet_age_le_15', 'triplet_age_le_30', 'triplet_age_le_60']] = False
        result = inputs(frames)
        own = result['samples'].loc[result['samples'].episode_id.eq(2)]
        self.assertTrue(own.main_eligible_continuation.all())
        self.assertFalse(own.main_eligible_initial.any())
        self.assertFalse(own.main_eligible_joint.any())
        design = result['design_rows']
        self.assertTrue(design.loc[design.sample_key.eq('c_300')].episode_id.eq(2).any())
        self.assertFalse(design.loc[design.sample_key.eq('joint_300')].episode_id.eq(2).any())

    def test_invalid_control_and_zero_direction_remain_visible_without_main_membership(self):
        frames = list(parents())
        frames[0].loc[2, 'pre_displayed_size'] = 0.
        _fixture_module.set_direction(frames, 3, 0.)
        result = inputs(frames)
        self.assertEqual(len(result['features']), 4)
        self.assertEqual(len(result['samples']), 12)
        self.assertFalse(result['samples'].main_eligible_continuation.any())
        self.assertTrue(result['design_rows'].empty)
        self.assertEqual(result['features'].set_index('episode_id').loc[2, 'input_exclusion_reason'], 'invalid_pre_size')

    def test_mislabeled_chronology_and_refitted_cutoff_fail(self):
        ledger = assemble_panel_ledger(*parents())
        for which in ['anchors', 'outcomes']:
            a, o = ledger['anchors'].copy(), ledger['outcomes'].copy()
            target = a if which == 'anchors' else o
            target.loc[0, 'analysis_segment'] = 'later_reanalysis'
            with self.subTest(which=which), self.assertRaises(ValueError):
                build_panel_inputs(a, o, scaling())
        changed = scaling(); changed['reference_cutoff'] = ANALYSIS_START.isoformat()
        with self.assertRaises(ValueError):
            build_panel_inputs(ledger['anchors'], ledger['outcomes'], changed)

    def test_empty_segment_remains_auditable_and_rank_failure_does_not_delete_terms(self):
        for times in [[PANEL_START], [ANALYSIS_START]]:
            with self.subTest(times=times):
                result = inputs(parents(times))
                self.assertEqual(len(result['features']), 1)
                self.assertFalse(result['schemas']['c_300']['baseline']['usable_design'])
                self.assertTrue(set(CORE).issubset(result['schemas']['c_300']['baseline']['columns']))
                self.assertEqual(result['specification']['baseline_columns_canonical'], CORE+CALENDAR)
                self.assertEqual(result['specification']['curvature_added_columns'], CURVATURE)


class PanelDependenceCalendarTests(unittest.TestCase):
    def test_main_sweep_matches_original_and_retains_pre_analysis_source_quotes(self):
        result = inputs()
        before = result['samples'].copy(deep=True)
        seen = []
        for horizon, tables in panel_dependence(result):
            seen.append(horizon)
            selected = result['samples'].loc[result['samples'].horizon_sec.eq(horizon)].copy()
            selected['eligible_continuation'] = selected.main_eligible_continuation
            expected = dependence_tables(result['features'], selected)
            for key in expected:
                pd.testing.assert_frame_equal(tables[key], expected[key])
            summary = dependence_summary(horizon, tables)
            self.assertEqual(summary['events'], 2)
            self.assertEqual(summary['source_start_before_analysis_start'], 1)
            self.assertFalse(summary['inference_approved'])
            self.assertTrue(set(tables['intervals'].episode_id).issubset({2, 3}))
        self.assertEqual(seen, [0, 60, 300])
        pd.testing.assert_frame_equal(result['samples'], before)

    def test_calendar_includes_every_zero_day_and_uses_onset_segment(self):
        result = inputs()
        calendar = daily_calendar(result)
        self.assertEqual(len(calendar), 831*3)
        self.assertEqual(calendar.date.nunique(), 831)
        self.assertEqual(calendar.iloc[0].date, PANEL_START)
        self.assertEqual(calendar.iloc[-1].date, PANEL_END-pd.Timedelta(days=1))
        self.assertEqual(calendar.horizon_sec.iloc[:6].tolist(), [0, 60, 300, 0, 60, 300])
        self.assertEqual(int(calendar.loc[calendar.horizon_sec.eq(300), 'all_onsets'].sum()), 4)
        self.assertTrue(calendar.loc[calendar.analysis_segment.eq('reused_development'), 'main_continuation'].eq(0).all())
        self.assertEqual(int(calendar.loc[calendar.horizon_sec.eq(300), 'main_continuation'].sum()), 2)
        self.assertTrue(calendar.loc[calendar.date.eq(PANEL_START+pd.Timedelta(days=1)), 'all_onsets'].eq(0).all())

    def test_tampered_main_mask_fails_calendar_and_dependence(self):
        result = inputs()
        result['samples'].loc[0, 'main_eligible_continuation'] = True
        with self.assertRaises(ValueError):
            daily_calendar(result)
        with self.assertRaises(ValueError):
            list(panel_dependence(result))

    def test_no_main_eligible_rows_retains_typed_empty_audits_and_zero_calendar(self):
        frames = list(parents())
        frames[0]['pre_spread_pips'] = 0.
        result = inputs(frames)
        for horizon, tables in panel_dependence(result):
            self.assertTrue(all(frame.empty for frame in tables.values()))
            self.assertEqual(dependence_summary(horizon, tables)['events'], 0)
        calendar = daily_calendar(result)
        self.assertEqual(calendar.main_continuation.sum(), 0)
        self.assertEqual(calendar.all_onsets.sum(), 12)


if __name__ == '__main__':
    unittest.main()

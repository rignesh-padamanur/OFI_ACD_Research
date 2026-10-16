"""Synthetic endpoint, attrition, calendar and fail-closed arithmetic checks."""
import unittest

import numpy as np
import pandas as pd

from src.joint_response import build_joint_response, PATTERNS

START = '2024-05-06T00:00:00+00:00'
END = '2024-05-09T00:00:00+00:00'


def fixture():
    triples = [(2., 0., 2.), (2., 1., 3.), (2., -1., 1.), (2., -2., 0.),
               (2., -3., -1.), (0., 1., 1.), (0., -1., -1.), (0., 0., 0.)]
    triples += [tuple(-v for v in t) for t in triples[:5]]
    triples += [(np.nan, 1., np.nan), (1., 1., 2.), (1., 1., np.nan)]
    rows = []
    for event, triple in enumerate(triples):
        onset = pd.Timestamp(START) + pd.Timedelta(seconds=event * 10)
        for horizon in (60, 300):
            j, c, i = triple
            if event == 2 and horizon == 300:
                c, i = -3., -1.  # The same event has different endpoint behavior.
            if event == 14 and horizon == 300:
                c = i = np.nan
            row = dict(episode_id=event, horizon_sec=horizon, bin_start=onset,
                detected_at=onset + pd.Timedelta(seconds=1),
                target_at=onset + pd.Timedelta(seconds=1 + horizon),
                analysis_segment='later_reanalysis', regressor_eligible=True,
                initial_pips=j, continuation_pips=c, total_pips=i,
                inherited_unused_field='keep exactly')
            for name, value in [('initial', j), ('continuation', c), ('total', i)]:
                for prefix in ('eligible_', 'estimation_eligible_', 'main_eligible_'):
                    row[prefix + name] = bool(np.isfinite(value))
            for prefix in ('eligible_', 'estimation_eligible_', 'main_eligible_'):
                row[prefix + 'joint'] = all(row[prefix + n] for n in ('initial', 'continuation', 'total'))
            rows.append(row)
    return pd.DataFrame(rows)


def build(samples=None):
    return build_joint_response(fixture() if samples is None else samples, START, END)


class JointResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = fixture()
        cls.output = build(cls.source)

    def test_parent_fields_preserved_and_input_not_mutated(self):
        source = fixture().sample(frac=1, random_state=4).reset_index(drop=True)
        before = source.copy(deep=True)
        out = build(source)
        pd.testing.assert_frame_equal(source, before)
        expected = source.loc[source.main_eligible_continuation].sort_values(['horizon_sec', 'episode_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(out['membership'][source.columns], expected)
        self.assertEqual(out['membership'].columns[-3:].tolist(), ['joint_included', 'initial_sign', 'endpoint_pattern'])

    def test_all_eight_patterns_and_both_initial_signs(self):
        rows = self.output['membership'].query('horizon_sec == 60').set_index('episode_id')
        self.assertEqual(rows.loc[range(8), 'endpoint_pattern'].tolist(), list(PATTERNS))
        self.assertEqual(rows.loc[range(8, 13), 'endpoint_pattern'].tolist(), list(PATTERNS[:5]))
        self.assertTrue(rows.loc[range(8, 13), 'initial_sign'].eq('negative').all())
        self.assertTrue(rows.loc[[5, 6, 7], 'initial_sign'].eq('zero').all())
        self.assertEqual(rows.loc[13, 'initial_sign'], 'missing')

    def test_attrition_uses_inherited_joint_mask_and_keeps_main_c(self):
        by_h = self.output['statistics']['by_horizon']
        self.assertEqual(by_h['60'], {'main_continuation_rows': 16, 'joint_rows': 14, 'missing_joint_rows': 2})
        self.assertEqual(by_h['300'], {'main_continuation_rows': 15, 'joint_rows': 13, 'missing_joint_rows': 2})
        self.assertEqual(self.output['statistics']['common_joint_events'], 13)
        self.assertEqual(self.output['statistics']['set_relation'], {'only_60': 1, 'only_300': 0,
                         'equal': False, 'joint_300_subset_60': True})
        missing = self.output['membership'].loc[~self.output['membership'].joint_included]
        self.assertTrue(missing.endpoint_pattern.eq('unavailable').all())
        self.assertEqual(set(missing.episode_id), {13, 15})

    def test_nonfinite_ineligible_initial_is_missing_not_directional(self):
        source = fixture()
        source.loc[source.episode_id.eq(13), 'initial_pips'] = np.inf
        output = build(source)
        rows = output['membership'].loc[output['membership'].episode_id.eq(13)]
        self.assertTrue(rows.initial_sign.eq('missing').all())
        self.assertTrue(rows.endpoint_pattern.eq('unavailable').all())
        from src.verify_joint_response import verify_joint_response
        self.assertEqual(verify_joint_response(output, source, START, END)['status'], 'passed')

    def test_nonmain_eligible_nonfinite_component_is_rejected(self):
        source = fixture()
        source.loc[source.episode_id.eq(0), 'regressor_eligible'] = False
        for prefix in ['estimation_eligible_', 'main_eligible_']:
            for part in ['initial', 'continuation', 'total', 'joint']:
                source.loc[source.episode_id.eq(0), prefix + part] = False
        source.loc[source.episode_id.eq(0), 'initial_pips'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonfinite main or other eligible'):
            build(source)

    def test_joint_summary_counts_units_quantiles_and_common_cohort(self):
        s = self.output['summary']
        self.assertEqual(len(s), 48)
        selected = s.query("horizon_sec == 60 and cohort == 'horizon_joint' and initial_group == 'all'").set_index('component')
        self.assertTrue(selected.n.eq(14).all())
        self.assertAlmostEqual(selected.loc['I', 'sum'], selected.loc['J', 'sum'] + selected.loc['C', 'sum'])
        j = self.source.query('horizon_sec == 60 and main_eligible_joint').initial_pips
        np.testing.assert_array_equal(selected.loc['J', ['p25', 'median', 'p75']], np.quantile(j, [.25, .5, .75]))
        common = s.query("cohort == 'common_horizons_joint' and initial_group == 'all' and component == 'J'")
        self.assertTrue(common.n.eq(13).all())
        self.assertEqual(common['sum'].nunique(), 1)

    def test_endpoint_grid_and_denominators_are_complete(self):
        e = self.output['endpoint_counts']
        self.assertEqual(len(e), 96)
        for (_, _), frame in e.groupby(['horizon_sec', 'cohort']):
            self.assertEqual(frame.n.sum(), frame.denominator_joint.iloc[0])
            self.assertAlmostEqual(frame.share_joint.sum(), 1.)
            for _, group in frame.groupby('initial_group'):
                self.assertEqual(group.n.sum(), group.denominator_initial_group.iloc[0])
                self.assertAlmostEqual(group.share_initial_group.sum(), 1.)

    def test_transitions_link_identical_events_and_include_zero_cells(self):
        t = self.output['transition_counts']
        self.assertEqual(len(t), 64)
        self.assertEqual(t.n.sum(), 13)
        self.assertTrue(t.denominator_common.eq(13).all())
        cell = t.query("from_pattern == 'retraced' and to_pattern == 'beyond_pre_onset'").iloc[0]
        self.assertEqual(cell.n, 1)
        self.assertEqual(cell.denominator_from, 2)
        self.assertEqual(cell.share_from, .5)
        self.assertAlmostEqual(t.share_common.sum(), 1.)

    def test_all_calendar_days_included_zero_days_have_nan_means(self):
        d = self.output['daily']
        self.assertEqual(len(d), 12)
        empty = d.loc[d.n.eq(0)]
        self.assertEqual(len(empty), 8)
        self.assertTrue(empty[['sum_J', 'sum_C', 'sum_I']].eq(0).all().all())
        self.assertTrue(empty[['mean_J', 'mean_C', 'mean_I']].isna().all().all())
        self.assertTrue(d[['n_' + p for p in PATTERNS]].sum(axis=1).eq(d.n).all())
        out = build_joint_response(fixture())
        self.assertEqual(len(out['daily']), 4 * 817)
        self.assertEqual(out['statistics']['calendar_days'], 817)

    def test_zero_denominators_and_empty_summary_never_fabricate_zero_mean(self):
        s = fixture().loc[lambda x: x.episode_id.eq(0)].copy()
        out = build(s)
        empty = out['summary'].query("initial_group == 'zero'")
        self.assertTrue(empty.n.eq(0).all())
        self.assertTrue(empty[['sum', 'mean', 'median', 'p25', 'p75']].isna().all().all())
        self.assertTrue(out['endpoint_counts'].query("initial_group == 'zero'").share_initial_group.isna().all())
        self.assertTrue(out['transition_counts'].query("from_pattern == 'retraced'").share_from.isna().all())

    def test_stable_sum_preserves_cancellation_remainder(self):
        s = fixture().loc[lambda x: x.episode_id.isin([0, 1, 2])].copy()
        for event, value in enumerate([1e16, 1., -1e16]):
            s.loc[s.episode_id.eq(event), ['initial_pips', 'total_pips']] = value
            s.loc[s.episode_id.eq(event), 'continuation_pips'] = 0.
        out = build(s)
        row = out['summary'].query("horizon_sec == 60 and cohort == 'horizon_joint' and initial_group == 'all' and component == 'J'").iloc[0]
        self.assertEqual(row['sum'], 1.)
        self.assertEqual(row['mean'], 1 / 3)

    def test_no_fit_or_uncertainty_claims(self):
        stats = self.output['statistics']
        self.assertTrue(stats['descriptive_only'])
        self.assertEqual(stats['empirical_models_fitted'], 0)
        self.assertFalse(stats['uncertainty_computed'])
        self.assertFalse(stats['inference_approved'])

    def test_duplicate_event_horizon_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate event/horizon'):
            build(pd.concat([self.source, self.source.iloc[[0]]], ignore_index=True))

    def test_nonboolean_and_missing_masks_rejected(self):
        for value in (1, None):
            s = fixture()
            s['main_eligible_joint'] = s.main_eligible_joint.astype(object)
            s.loc[0, 'main_eligible_joint'] = value
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                build(s)

    def test_component_joint_and_main_masks_cannot_be_changed(self):
        for field in ('eligible_joint', 'estimation_eligible_joint', 'main_eligible_joint', 'main_eligible_continuation'):
            s = fixture()
            s.loc[0, field] = False
            with self.assertRaisesRegex(ValueError, 'mask|eligibility'):
                build(s)

    def test_regressor_ineligibility_cannot_remain_main(self):
        s = fixture()
        s.loc[s.episode_id.eq(0), 'regressor_eligible'] = False
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            build(s)

    def test_wrong_timing_or_chronology_rejected(self):
        for field, value in [('detected_at', pd.Timestamp(START)),
                             ('target_at', pd.Timestamp(START)),
                             ('analysis_segment', 'reused_development')]:
            s = fixture()
            s.loc[0, field] = value
            with self.assertRaises(ValueError):
                build(s)
        s = fixture()
        s['bin_start'] = s.bin_start.dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, 'UTC'):
            build(s)

    def test_initial_response_and_onset_invariant_across_horizons(self):
        s = fixture()
        s.loc[1, ['initial_pips', 'total_pips']] = 3.
        with self.assertRaisesRegex(ValueError, 'across horizons'):
            build(s)

    def test_nonfinite_main_c_and_missing_horizon_rejected(self):
        s = fixture()
        s.loc[0, 'continuation_pips'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonfinite main'):
            build(s)
        with self.assertRaisesRegex(ValueError, 'one 60-second'):
            build(fixture().iloc[1:])

    def test_additive_identity_failure_rejected(self):
        s = fixture()
        s.loc[0, 'total_pips'] = 20.
        with self.assertRaisesRegex(ValueError, 'J.C=I'):
            build(s)

    def test_roundoff_cannot_relax_endpoint_order_or_exact_zero(self):
        # All these triples pass the additive tolerance, but contradict the
        # endpoint label implied by their signs or zero values.
        for j, c, i in [(1., -1e-12, 1. + 1e-12), (1., 1e-12, 1. - 1e-12),
                        (1., -1e-16, 1.), (1., 0., 1. + 1e-12),
                        (0., 1e-12, 2e-12), (1e-12, 1e-12, -1e-12)]:
            s = fixture().loc[lambda x: x.episode_id.eq(0)].copy()
            s[['initial_pips', 'continuation_pips', 'total_pips']] = [j, c, i]
            with self.assertRaisesRegex(ValueError, 'Endpoint signs/order'):
                build(s)

    def test_small_exact_moves_are_not_zeroed(self):
        s = fixture().loc[lambda x: x.episode_id.eq(0)].copy()
        s[['initial_pips', 'continuation_pips', 'total_pips']] = [1e-15, -1e-15, 0.]
        self.assertTrue(build(s)['membership'].endpoint_pattern.eq('at_pre_onset').all())


if __name__ == '__main__':
    unittest.main()

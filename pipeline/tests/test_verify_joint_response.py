"""Adversarial independent checks for descriptive initial/continuation accounting."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from src.joint_response import build_joint_response
from src.verify_joint_response import verify_joint_response, reference_joint_response


START = pd.Timestamp('2024-05-06', tz='UTC')
END = START + pd.Timedelta(days=4)


def refresh_masks(samples):
    samples['eligible_joint'] = samples.eligible_initial & samples.eligible_continuation & samples.eligible_total
    for name in ['initial', 'continuation', 'total', 'joint']:
        samples['estimation_eligible_' + name] = samples['eligible_' + name] & samples.regressor_eligible
        samples['main_eligible_' + name] = (samples['estimation_eligible_' + name]
            & samples.analysis_segment.eq('later_reanalysis'))
    return samples


def fixture():
    # Includes every endpoint label, changes across horizons, incomplete joint
    # observations, development rows and a date with no events.
    movements = [(1., .5, -2.), (-1., -.5, .5), (1., -.2, 0.), (0., 1., -1.),
                 (0., 0., 0.), (1., -1., -1.), (-1., 2., 2.), (-1., 0., -.5),
                 (1., .5, .5), (1., .5, .5), (np.nan, 1., -1.), (1., .5, .5)]
    rows = []
    for eid, (j, c60, c300) in enumerate(movements):
        onset = START + pd.Timedelta(days=eid % 3, seconds=10 + eid)
        if eid == 11:
            onset = START - pd.Timedelta(days=1)
        for horizon, c in [(0, 0.), (60, c60), (300, c300)]:
            row = {'episode_id': eid, 'horizon_sec': horizon,
                'bin_start': onset, 'detected_at': onset + pd.Timedelta(seconds=1),
                'target_at': onset + pd.Timedelta(seconds=horizon + 1),
                'analysis_segment': 'later_reanalysis' if onset >= START else 'reused_development',
                'regressor_eligible': True, 'initial_pips': j,
                'continuation_pips': c, 'total_pips': j + c,
                'eligible_initial': bool(np.isfinite(j)), 'eligible_continuation': True,
                'eligible_total': bool(np.isfinite(j)),
                'eligible_pre_age_15': False, 'eligible_direction_ratio_gt_10pct': False,
                'immutable_extra': f'source-{eid}'}
            if eid == 8 and horizon == 60:
                row['eligible_total'] = False
                row['total_pips'] = np.nan
            if eid == 9 and horizon == 300:
                row['eligible_continuation'] = row['eligible_total'] = False
                row['continuation_pips'] = row['total_pips'] = np.nan
            rows.append(row)
    return refresh_masks(pd.DataFrame(rows))


class IndependentJointResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = fixture()
        cls.actual = build_joint_response(cls.samples, START, END)

    def verify(self, actual=None, samples=None):
        return verify_joint_response(self.actual if actual is None else actual,
            self.samples if samples is None else samples, START, END)

    def test_independent_complete_reconstruction_and_immutable_fields(self):
        result = self.verify()
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['statistics_verified'])
        self.assertEqual(result['table_rows_verified'], {'membership': 21, 'summary': 48,
            'endpoint_counts': 96, 'transition_counts': 64, 'daily': 16})
        self.assertEqual(self.actual['statistics']['common_joint_events'], 8)
        self.assertEqual(self.actual['statistics']['set_relation'],
            {'only_60': 1, 'only_300': 1, 'equal': False, 'joint_300_subset_60': False})
        self.assertFalse(self.actual['membership'].eligible_pre_age_15.any())
        self.assertFalse(self.actual['membership'].eligible_direction_ratio_gt_10pct.any())

    def test_negative_continuation_can_extend_initial_movement(self):
        member = self.actual['membership'].set_index(['episode_id', 'horizon_sec'])
        self.assertEqual(member.at[(1, 60), 'endpoint_pattern'], 'extended')
        self.assertEqual(member.at[(2, 60), 'endpoint_pattern'], 'retraced')
        self.assertEqual(member.at[(3, 60), 'endpoint_pattern'], 'initial_zero_positive')
        self.assertEqual(member.at[(5, 60), 'endpoint_pattern'], 'at_pre_onset')
        self.assertEqual(member.at[(0, 300), 'endpoint_pattern'], 'beyond_pre_onset')

    def test_missing_initial_is_not_zero_and_joint_losses_are_preserved(self):
        member = self.actual['membership'].set_index(['episode_id', 'horizon_sec'])
        self.assertEqual(member.at[(10, 60), 'initial_sign'], 'missing')
        self.assertEqual(member.at[(10, 60), 'endpoint_pattern'], 'unavailable')
        self.assertFalse(member.at[(8, 60), 'joint_included'])
        self.assertTrue(member.at[(8, 300), 'joint_included'])
        self.assertNotIn((9, 300), member.index)
        self.assertEqual(self.actual['statistics']['by_horizon']['60']['missing_joint_rows'], 2)

    def test_full_calendar_zero_days_and_nonadditive_medians(self):
        self.verify()
        empty = self.actual['daily'].loc[self.actual['daily'].date.eq(END - pd.Timedelta(days=1))]
        self.assertEqual(len(empty), 4)
        self.assertTrue(empty[['n', 'sum_J', 'sum_C', 'sum_I']].eq(0).all().all())
        self.assertTrue(empty[['mean_J', 'mean_C', 'mean_I']].isna().all().all())
        rows = self.actual['summary'].query('horizon_sec == 60 and cohort == "horizon_joint" and initial_group == "all"').set_index('component')
        self.assertAlmostEqual(rows.at['J', 'mean'] + rows.at['C', 'mean'], rows.at['I', 'mean'])
        self.assertNotAlmostEqual(rows.at['J', 'median'] + rows.at['C', 'median'], rows.at['I', 'median'])

    def test_empty_initial_groups_and_no_common_observations(self):
        samples = self.samples.loc[self.samples.episode_id.isin([8, 9])].copy()
        actual = build_joint_response(samples, START, END)
        self.verify(actual, samples)
        self.assertEqual(actual['statistics']['common_joint_events'], 0)
        common = actual['summary'].loc[actual['summary'].cohort.eq('common_horizons_joint')]
        self.assertTrue(common.n.eq(0).all())
        self.assertTrue(common[['sum', 'mean', 'median', 'p25', 'p75']].isna().all().all())
        self.assertTrue(actual['transition_counts'].share_common.isna().all())

    def test_no_tolerance_category_and_tiny_nonzero_initial(self):
        samples = self.samples.loc[self.samples.episode_id.eq(0)].copy()
        samples['initial_pips'] = 1e-200
        samples['continuation_pips'] = 1e-200
        samples['total_pips'] = 2e-200
        actual = build_joint_response(samples, START, END)
        self.verify(actual, samples)
        self.assertTrue(actual['membership'].endpoint_pattern.eq('extended').all())
        # Numerically near an identity, but endpoints do not strictly extend.
        samples['total_pips'] = samples.initial_pips
        with self.assertRaises(AssertionError):
            reference_joint_response(samples, START, END)

    def test_independent_stable_sums_preserve_cancellation_residual(self):
        samples = self.samples.loc[self.samples.episode_id.isin([0, 1, 2])].copy()
        values = samples.episode_id.map({0: 1e16, 1: 1., 2: -1e16})
        samples['initial_pips'] = values
        samples['continuation_pips'] = 0.
        samples['total_pips'] = values
        # Put all three on a single date to check the daily aggregation as well.
        samples['bin_start'] = START + pd.Timedelta(seconds=10)
        samples['detected_at'] = samples.bin_start + pd.Timedelta(seconds=1)
        samples['target_at'] = samples.detected_at + pd.to_timedelta(samples.horizon_sec, unit='s')
        actual = build_joint_response(samples, START, END)
        self.verify(actual, samples)
        rows = actual['summary'].query('initial_group == "all" and component == "J"')
        self.assertTrue(rows['sum'].eq(1.).all())
        self.assertTrue(actual['daily'].loc[actual['daily'].n.gt(0), 'sum_J'].eq(1.).all())

    def test_all_joint_observations_unavailable_preserves_main_C_and_zero_counts(self):
        samples = self.samples.copy(deep=True)
        samples['eligible_initial'] = False
        refresh_masks(samples)
        actual = build_joint_response(samples, START, END)
        self.verify(actual, samples)
        self.assertEqual(len(actual['membership']), len(self.actual['membership']))
        self.assertTrue(actual['membership'].endpoint_pattern.eq('unavailable').all())
        self.assertTrue(actual['summary'].n.eq(0).all())
        self.assertTrue(actual['daily'].n.eq(0).all())

    def test_source_keys_horizon_invariance_and_masks_fail(self):
        for corruption in ['key', 'missing_horizon', 'J', 'clock', 'mainmask', 'jointmask', 'segment']:
            with self.subTest(corruption=corruption):
                samples = self.samples.copy(deep=True)
                if corruption == 'key': samples = pd.concat([samples, samples.iloc[:1]], ignore_index=True)
                elif corruption == 'missing_horizon': samples = samples.drop(index=1)
                elif corruption == 'J': samples.loc[1, 'initial_pips'] += .25
                elif corruption == 'clock': samples.loc[1, 'target_at'] += pd.Timedelta(nanoseconds=1)
                elif corruption == 'mainmask': samples.loc[1, 'main_eligible_continuation'] = False
                elif corruption == 'jointmask': samples.loc[1, 'eligible_joint'] = False
                elif corruption == 'segment': samples.loc[samples.episode_id.eq(0), 'analysis_segment'] = 'reused_development'
                with self.assertRaises(AssertionError): self.verify(samples=samples)

    def test_bad_identity_exact_unchanged_and_zero_J_fail(self):
        for eid, field, value in [(0, 'total_pips', 12.), (7, 'total_pips', -1. + 1e-12),
                                  (3, 'total_pips', 1. + 1e-12)]:
            with self.subTest(eid=eid):
                samples = self.samples.copy(deep=True)
                samples.loc[samples.episode_id.eq(eid) & samples.horizon_sec.eq(60), field] = value
                with self.assertRaises(AssertionError): self.verify(samples=samples)

    def test_membership_keys_labels_and_every_source_column_are_checked(self):
        for field, replacement in [('episode_id', 999), ('initial_pips', 2.),
            ('immutable_extra', 'changed'), ('joint_included', False),
            ('initial_sign', 'negative'), ('endpoint_pattern', 'retraced')]:
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual['membership'].loc[0, field] = replacement
                with self.assertRaises(AssertionError): self.verify(actual)

    def test_component_summaries_counts_and_quantiles_are_checked(self):
        for field in ['n', 'sum', 'mean', 'median', 'p25', 'p75']:
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual['summary'].loc[0, field] += 1
                with self.assertRaises(AssertionError): self.verify(actual)

    def test_denominator_transition_and_daily_corruptions_are_checked(self):
        changes = [('endpoint_counts', 'n'), ('endpoint_counts', 'denominator_joint'),
            ('endpoint_counts', 'denominator_initial_group'), ('endpoint_counts', 'share_joint'),
            ('transition_counts', 'n'), ('transition_counts', 'denominator_common'),
            ('transition_counts', 'denominator_from'), ('transition_counts', 'share_from'),
            ('daily', 'n'), ('daily', 'sum_I'), ('daily', 'mean_C'), ('daily', 'n_extended')]
        for table, field in changes:
            with self.subTest(table=table, field=field):
                actual = deepcopy(self.actual)
                index = actual[table][field].first_valid_index()
                actual[table].loc[index, field] += 1
                with self.assertRaises(AssertionError): self.verify(actual)

    def test_duplicate_missing_rows_and_statistics_are_checked(self):
        for table in ['summary', 'endpoint_counts', 'transition_counts', 'daily']:
            actual = deepcopy(self.actual)
            actual[table] = pd.concat([actual[table], actual[table].iloc[:1]], ignore_index=True)
            with self.subTest(table=table), self.assertRaises(AssertionError): self.verify(actual)
        for key in ['common_joint_events', 'calendar_days', 'empirical_models_fitted', 'inference_approved']:
            actual = deepcopy(self.actual)
            actual['statistics'][key] = not actual['statistics'][key] if key == 'inference_approved' else actual['statistics'][key] + 1
            with self.subTest(key=key), self.assertRaises(AssertionError): self.verify(actual)
        actual = deepcopy(self.actual)
        actual['statistics']['cohorts']['horizon_joint_60']['mean']['J'] += .1
        with self.assertRaises(AssertionError): self.verify(actual)


if __name__ == '__main__':
    unittest.main()

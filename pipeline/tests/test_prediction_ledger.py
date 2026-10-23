"""Synthetic chronology and future-information separation for Notebook 5."""
import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.prediction_ledger import build_origins, attach_labels, build_folds, PRE_PREDICTORS


def fixture():
    onset = pd.to_datetime(['2024-04-22T00:00:01Z', '2024-05-05T23:54:59Z',
        '2024-05-05T23:55:00Z', '2024-05-05T23:59:59Z', '2024-05-06T00:00:02Z',
        '2024-05-06T00:00:05Z', '2024-05-06T00:00:08Z', '2024-05-19T23:54:59Z',
        '2024-05-20T00:00:00Z', '2024-06-02T23:59:58Z'])
    n = len(onset)
    d = np.ones(n); d[5] = 0; d[6] = -1
    raw_j = np.array([0.00001, 0., -0.00001, 0.00002, 0.00003, 0.00001,
                      0.00001, 0.00001, 0.00001, 0.00001])
    a = pd.DataFrame({'episode_id': np.arange(n, dtype='int64'), 'bin_start': onset,
        'available_at': onset + pd.Timedelta(1, unit='s'), 'support_id': np.zeros(n, dtype='int64'),
        'onset_confirmed': True, 'state_history_available': True, 'state_sample_eligible': True,
        'pre_quote_ofi_300s': np.linspace(-2., 2., n), 'pre_rv_300s_5s_bps': np.linspace(.1, 1., n),
        'pre_spread_pips': .3, 'pre_displayed_size': 2., 'direction': d,
        'direction_available': d != 0, 'pre_mark_at': onset - pd.Timedelta(500, unit='ms'),
        'detection_mark_at': onset + pd.Timedelta(500, unit='ms'), 'quote_ofi': d * 2.,
        'pre_mark_status': 'available', 'detection_mark_status': 'available',
        'pre_mid': 1.1, 'detection_mid': 1.1 + raw_j})
    a['initial_pips'] = np.where(d != 0, d * (a.detection_mid - a.pre_mid) * 10000, np.nan)
    f = a.drop(columns=['pre_mark_status', 'detection_mark_status', 'pre_mid', 'detection_mid', 'initial_pips']).copy()
    f['pre_rv_bps_5s'] = f.pre_rv_300s_5s_bps
    f['pre_status_5s'] = 'available'
    f['pre_first_quote_at_5s'] = onset - pd.Timedelta(301, unit='s')
    f['pre_last_quote_at_5s'] = a.pre_mark_at
    scaling = {'cQ': 2., 'cV': .5, 'sV': .25, 'reference_cutoff': '2024-04-22T00:00:00Z'}
    stop = pd.Timestamp('2024-06-03T00:00:00Z')
    pieces = []
    for h in [0, 60, 300]:
        o = a[['episode_id', 'bin_start', 'available_at', 'initial_pips']].rename(columns={'available_at': 'detected_at'}).copy()
        o['horizon_sec'] = h
        o['target_at'] = o.detected_at + pd.Timedelta(h, unit='s')
        o['endpoint_quote_at'] = o.target_at - pd.Timedelta(500, unit='ms')
        o['endpoint_age_sec'] = .5
        o['endpoint_status'] = 'available'
        o['continuation_pips'] = np.where(d != 0, h / 3000., np.nan)
        o['total_pips'] = o.initial_pips + o.continuation_pips
        late = o.target_at.gt(stop)
        o.loc[late, 'endpoint_status'] = 'beyond_sample_end'
        o.loc[late, 'endpoint_quote_at'] = pd.NaT
        o.loc[late, ['endpoint_age_sec', 'continuation_pips', 'total_pips']] = np.nan
        pieces.append(o)
    outcomes = pd.concat(pieces, ignore_index=True)
    schedule = pd.DataFrame({'cycle_id': [0, 1, 2],
        'parameter_cutoff': pd.to_datetime(['2024-04-22T00:00:00Z', '2024-05-06T00:00:00Z', '2024-05-20T00:00:00Z']),
        'deployment_end': pd.to_datetime(['2024-05-06T00:00:00Z', '2024-05-20T00:00:00Z', '2024-06-03T00:00:00Z'])})
    return a, f, scaling, outcomes, schedule, stop


class PredictionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.a, self.f, self.scaling, self.o, self.schedule, self.stop = fixture()

    def origins(self):
        return build_origins(self.a, self.f, self.scaling)

    def all_tables(self):
        first = self.origins()
        labels = attach_labels(first['origins'], self.o, sample_end=self.stop)
        folds = build_folds(first['origins'], labels, self.schedule, sample_end=self.stop)
        return first, labels, folds

    def test_origin_count_signs_and_predictor_allowlist(self):
        result = self.origins(); origins = result['origins']; x = result['predictors']
        self.assertEqual(len(origins), len(self.a))
        self.assertEqual(int(origins.origin_eligible.sum()), len(self.a) - 1)
        self.assertEqual(list(x), ['episode_id', 'bin_start', 'forecast_at', *PRE_PREDICTORS])
        self.assertTrue(origins.loc[1, 'origin_eligible'])
        self.assertEqual(origins.loc[1, 'initial_sign'], 'zero')
        self.assertTrue(origins.loc[2, 'origin_eligible'])
        self.assertEqual(origins.loc[2, 'initial_sign'], 'negative')
        self.assertFalse(origins.loc[5, 'origin_eligible'])
        self.assertEqual(origins.loc[5, 'origin_exclusion_reason'], 'defined_detection_direction')
        self.assertTrue(origins.loc[6, 'origin_eligible'])
        self.assertEqual(origins.loc[6, 'initial_sign'], 'negative')
        np.testing.assert_allclose(x.signed_pressure, self.a.pre_quote_ofi_300s / 2)
        np.testing.assert_allclose(x.pressure_magnitude, abs(x.signed_pressure))
        np.testing.assert_allclose(x.variation_x_magnitude, x.variation_standardized * x.pressure_magnitude)
        self.assertTrue(origins.loc[origins.origin_eligible, 'reference_continuation_pips'].eq(0).all())
        np.testing.assert_allclose(origins.loc[origins.origin_eligible, 'reference_total_pips'],
                                   origins.loc[origins.origin_eligible, 'initial_pips'])
        json.dumps(result['statistics'], allow_nan=False)

    def test_future_columns_and_episode_closure_do_not_change_origins(self):
        expected = self.origins()
        for frame in [self.a, self.f]:
            for name, value in [('main_eligible_joint', False), ('continuation_pips', 1e100),
                                ('endpoint_status', 'missing'), ('closure_known_at', self.stop),
                                ('last_candidate_bin', self.stop), ('initial_sign', 'invented')]:
                frame[name] = value
        changed = self.origins()
        for name in ['origins', 'predictors']:
            assert_frame_equal(expected[name], changed[name])
        self.assertEqual(expected['statistics'], changed['statistics'])

    def test_permuted_keys_reconstruct_identically(self):
        expected = self.origins()
        self.a = self.a.sample(frac=1, random_state=3)
        self.f = self.f.sample(frac=1, random_state=9)
        actual = self.origins()
        assert_frame_equal(expected['origins'], actual['origins'])
        assert_frame_equal(expected['predictors'], actual['predictors'])

    def test_missing_future_record_is_retained(self):
        first = self.origins(); origins = first['origins'].copy(deep=True)
        self.o = self.o.loc[~(self.o.episode_id.eq(4) & self.o.horizon_sec.eq(300))]
        labels = attach_labels(origins, self.o, sample_end=self.stop)
        row = labels.loc[labels.episode_id.eq(4) & labels.horizon_sec.eq(300)].iloc[0]
        self.assertEqual(len(labels), 2 * len(origins))
        self.assertTrue(row.origin_eligible)
        self.assertFalse(row.label_available)
        self.assertFalse(row.evaluation_eligible)
        self.assertEqual(row.label_status, 'missing_record')
        assert_frame_equal(origins, first['origins'])

    def test_late_and_undefined_labels_preserve_origin_eligibility(self):
        first, labels, folds = self.all_tables()
        late = labels.loc[labels.episode_id.eq(9)]
        self.assertTrue(late.origin_eligible.all())
        self.assertTrue(late.endpoint_status.eq('beyond_sample_end').all())
        self.assertFalse(late.evaluation_eligible.any())
        zero = labels.loc[labels.episode_id.eq(5)]
        self.assertTrue(zero.label_status.eq('undefined_direction').all())
        self.assertFalse(zero.label_available.any())
        predictions = folds['membership'].loc[lambda x: x.role.eq('prediction')]
        self.assertIn(9, predictions.episode_id.to_list())
        self.assertIn(5, predictions.episode_id.to_list())

    def test_maturity_uses_target_not_stale_source_quote(self):
        selected = self.o.episode_id.eq(2) & self.o.horizon_sec.eq(300)
        self.o.loc[selected, 'endpoint_quote_at'] = pd.Timestamp('2024-05-05T23:54:00Z')
        self.o.loc[selected, 'endpoint_age_sec'] = 361.
        _, labels, result = self.all_tables()
        training = result['membership'].loc[lambda x: x.fold_id.eq(1) & x.role.eq('training')]
        self.assertIn(1, training.episode_id.to_list())
        self.assertNotIn(2, training.episode_id.to_list())
        maturity = labels.loc[labels.episode_id.eq(2) & labels.horizon_sec.eq(300)].iloc[0]
        self.assertEqual(maturity.label_ready_at, pd.Timestamp('2024-05-06T00:00:01Z'))
        self.assertEqual(result['folds'].loc[0, 'earlier_labels_not_yet_mature'], 1)

    def test_detection_boundary_assigns_next_block_and_expands_training(self):
        _, _, result = self.all_tables(); members = result['membership']
        p = members.loc[members.role.eq('prediction')].set_index('episode_id')
        self.assertEqual(p.loc[3, 'fold_id'], 1)
        self.assertEqual(p.loc[8, 'fold_id'], 2)
        self.assertEqual(len(p), 7)
        t1 = set(members.loc[members.fold_id.eq(1) & members.role.eq('training'), 'episode_id'])
        t2 = set(members.loc[members.fold_id.eq(2) & members.role.eq('training'), 'episode_id'])
        self.assertEqual(t1, {0, 1})
        self.assertTrue(t1.issubset(t2))
        self.assertNotIn(5, t2)
        self.assertIn(7, t2)
        self.assertFalse(members.loc[members.role.eq('training'), 'evaluation_eligible'].any())
        json.dumps(result['statistics'], allow_nan=False)

    def test_future_missingness_cannot_remove_prediction_memberships(self):
        _, _, expected = self.all_tables()
        self.o = self.o.loc[self.o.episode_id.lt(3)]
        _, _, actual = self.all_tables()
        columns = ['fold_id', 'episode_id', 'forecast_at', 'origin_eligible']
        for result in [expected, actual]:
            result['prediction'] = result['membership'].loc[lambda x: x.role.eq('prediction'), columns].reset_index(drop=True)
        assert_frame_equal(expected['prediction'], actual['prediction'])
        self.assertEqual(int(actual['folds'].prediction_evaluable.sum()), 0)

    def test_parent_key_time_and_direction_failures(self):
        changes = [lambda a, f: f.__setitem__('episode_id', f.episode_id + 1),
                   lambda a, f: f.__setitem__('bin_start', f.bin_start + pd.Timedelta(1, unit='s')),
                   lambda a, f: a.__setitem__('quote_ofi', -a.quote_ofi),
                   lambda a, f: a.__setitem__('available_at', a.available_at + pd.Timedelta(1, unit='s'))]
        for mutate in changes:
            with self.subTest(change=mutate):
                a, f = self.a.copy(), self.f.copy(); mutate(a, f)
                with self.assertRaises(ValueError): build_origins(a, f, self.scaling)
        with self.assertRaises(ValueError): build_origins(pd.concat([self.a, self.a.iloc[:1]]), self.f, self.scaling)

    def test_large_integer_key_comparison_is_exact(self):
        self.a['support_id'] = np.int64(2**55)
        self.f['support_id'] = np.int64(2**55 + 1)
        with self.assertRaises(ValueError): self.origins()

    def test_source_timing_and_timezone_failures(self):
        for field, value in [('pre_first_quote_at_5s', self.f.bin_start - pd.Timedelta(300, unit='s')),
                             ('pre_last_quote_at_5s', self.f.bin_start)]:
            with self.subTest(field=field):
                f = self.f.copy(); f[field] = value
                with self.assertRaises(ValueError): build_origins(self.a, f, self.scaling)
        self.a['bin_start'] = self.a.bin_start.dt.tz_localize(None)
        with self.assertRaises(ValueError): self.origins()

    def test_scaling_and_invalid_numeric_fail(self):
        for name, value in [('reference_cutoff', '2024-04-23T00:00:00Z'), ('cQ', 0),
                            ('sV', np.inf), ('cQ', True)]:
            with self.subTest(name=name, value=value):
                scaling = {**self.scaling, name: value}
                with self.assertRaises(ValueError): build_origins(self.a, self.f, scaling)
        self.f['pre_rv_bps_5s'] = self.f.pre_rv_bps_5s.astype(str)
        with self.assertRaises(ValueError): self.origins()

    def test_missing_prior_state_is_explicit_abstention(self):
        for frame in [self.a, self.f]:
            frame.loc[4, 'pre_rv_300s_5s_bps'] = np.nan
            frame.loc[4, ['state_history_available', 'state_sample_eligible']] = False
        self.f.loc[4, 'pre_status_5s'] = 'window_crosses_support'
        self.f.loc[4, 'pre_rv_bps_5s'] = np.nan
        self.f.loc[4, ['pre_first_quote_at_5s', 'pre_last_quote_at_5s']] = pd.NaT
        result = self.origins()['origins']
        self.assertEqual(len(result), len(self.a))
        self.assertFalse(result.loc[4, 'origin_eligible'])
        self.assertEqual(result.loc[4, 'origin_exclusion_reason'], 'prior_state_available')

    def test_missing_direction_is_distinct_from_zero_direction(self):
        for frame in [self.a, self.f]:
            frame.loc[4, 'direction'] = np.nan
            frame.loc[4, 'direction_available'] = False
        self.a.loc[4, ['quote_ofi', 'initial_pips']] = np.nan
        result = self.origins()['origins']
        self.assertFalse(result.loc[4, 'origin_eligible'])
        self.assertEqual(result.loc[4, 'direction_status'], 'missing')
        self.assertEqual(result.loc[5, 'direction_status'], 'zero')

    def test_invalid_label_identity_value_and_timing_fail(self):
        origins = self.origins()['origins']
        changes = [('episode_id', 999), ('horizon_sec', 301),
                   ('detected_at', pd.Timestamp('2024-04-23T00:00:00Z')),
                   ('target_at', pd.Timestamp('2024-04-23T00:00:00Z')),
                   ('endpoint_status', 'available_but_unmeasured'), ('endpoint_age_sec', -1.),
                   ('total_pips', np.nan), ('continuation_pips', np.nan)]
        for field, value in changes:
            with self.subTest(field=field):
                outcomes = self.o.copy(); outcomes.loc[0, field] = value
                with self.assertRaises(ValueError): attach_labels(origins, outcomes, sample_end=self.stop)
        with self.assertRaises(ValueError): attach_labels(origins, pd.concat([self.o, self.o.iloc[:1]]), sample_end=self.stop)

    def test_sample_end_equality_is_observable(self):
        origins = self.origins()['origins']
        selected = origins.loc[origins.episode_id.eq(0)].copy()
        outcomes = self.o.loc[self.o.episode_id.eq(0)].copy()
        end = selected.forecast_at.iloc[0] + pd.Timedelta(300, unit='s')
        labels = attach_labels(selected, outcomes, sample_end=end)
        self.assertTrue(labels.label_available.all())
        self.assertEqual(labels.loc[labels.horizon_sec.eq(300), 'label_ready_at'].iloc[0], end)

    def test_fold_schedule_and_label_tampering_fail(self):
        first, labels, _ = self.all_tables()
        schedule = self.schedule.copy(); schedule.loc[1, 'parameter_cutoff'] += pd.Timedelta(1, unit='s')
        with self.assertRaises(ValueError): build_folds(first['origins'], labels, schedule, sample_end=self.stop)
        for field, value in [('label_ready_at', pd.Timestamp('2024-01-01T00:00:00Z')),
                             ('origin_eligible', False), ('evaluation_eligible', False)]:
            with self.subTest(field=field):
                bad = labels.copy(); bad.loc[0, field] = value
                with self.assertRaises(ValueError): build_folds(first['origins'], bad, self.schedule, sample_end=self.stop)
        missing = labels.loc[~(labels.episode_id.eq(4) & labels.horizon_sec.eq(300))]
        with self.assertRaises(ValueError): build_folds(first['origins'], missing, self.schedule, sample_end=self.stop)

    def test_endpoint_values_cannot_enter_predictor_matrix(self):
        result = self.origins()
        self.o.loc[np.isfinite(self.o.continuation_pips), 'continuation_pips'] *= 1000
        self.o['total_pips'] = self.o.initial_pips + self.o.continuation_pips
        attach_labels(result['origins'], self.o, sample_end=self.stop)
        again = self.origins()
        assert_frame_equal(result['predictors'], again['predictors'])
        self.assertNotIn('direction', again['predictors'])
        self.assertNotIn('initial_pips', again['predictors'])


if __name__ == '__main__':
    unittest.main()

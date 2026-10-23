"""Adversarial checks for fixed, outcome-independent stage-07a input construction."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis_ledger import assemble_ledger
from src.estimation_inputs import make_scaling, build_inputs

# load the existing independent parent fixture without requiring tests to be a package.
_fixture_spec = importlib.util.spec_from_file_location(
    '_stage06c_test_fixture', Path(__file__).with_name('test_analysis_ledger.py'))
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
parent_frames = _fixture_module.parent_frames
set_direction = _fixture_module.set_direction

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp('2024-02-26', tz='UTC')
CUTOFF = pd.Timestamp('2024-04-22', tz='UTC')
DERIVED = ['variation_standardized', 'pressure_magnitude', 'aligned_prior_pressure',
           'variation_x_magnitude', 'variation_x_alignment', 'log_pre_displayed_size',
           'magnitude_squared', 'variation_squared']


def reference():
    bins = pd.date_range(START + pd.Timedelta(seconds=300), periods=3, freq='s')
    return pd.DataFrame({'bin_start': bins,
        'available_at': bins + pd.Timedelta(seconds=1), 'period': 'calibration',
        'pre_quote_ofi_300s': [-3., 0., 3.], 'pre_rv_300s_5s_bps': [0., 1., 2.]})


def scaling():
    return make_scaling(reference(), START, CUTOFF)[1]


def ledger(frames=None):
    return assemble_ledger(*(parent_frames() if frames is None else frames))


def inputs(tables=None):
    tables = ledger() if tables is None else tables
    return build_inputs(tables['anchors'], tables['outcomes'], scaling())


class ReferenceScalingTests(unittest.TestCase):
    def test_joint_reference_uses_rms_without_demeaning_and_population_variance(self):
        r = reference()
        r['pre_quote_ofi_300s'] = [2., 3., 4.]
        selected, constants = make_scaling(r, START, CUTOFF)
        self.assertEqual(len(selected), 3)
        self.assertAlmostEqual(constants['cQ'], np.sqrt(29./3.))
        self.assertAlmostEqual(constants['cV'], 1.)
        self.assertAlmostEqual(constants['sV'], np.sqrt(2./3.))

    def test_zero_flow_and_zero_variation_remain_in_reference(self):
        selected, constants = make_scaling(reference(), START, CUTOFF)
        self.assertEqual(len(selected), 3)
        self.assertEqual(int(selected.pre_quote_ofi_300s.eq(0).sum()), 1)
        self.assertEqual(int(selected.pre_rv_300s_5s_bps.eq(0).sum()), 1)
        self.assertAlmostEqual(constants['cQ'], np.sqrt(6.))

    def test_development_values_cannot_change_scaling(self):
        r = reference()
        extra = r.iloc[[0]].copy()
        extra['period'] = 'development'
        extra['bin_start'] = CUTOFF + pd.Timedelta(seconds=1)
        extra['available_at'] = CUTOFF + pd.Timedelta(seconds=2)
        extra[['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']] = 1.e100
        chosen, constants = make_scaling(pd.concat([r, extra], ignore_index=True), START, CUTOFF)
        self.assertEqual(len(chosen), 3)
        for key in ['cQ', 'cV', 'sV']:
            self.assertEqual(constants[key], scaling()[key])

    def test_missing_or_negative_reference_states_are_jointly_excluded(self):
        r = reference()
        additions = []
        for j, pair in enumerate([(np.nan, 100.), (100., np.inf), (100., -1.)]):
            row = r.iloc[[0]].copy()
            row['bin_start'] += pd.Timedelta(seconds=10+j)
            row['available_at'] += pd.Timedelta(seconds=10+j)
            row[['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']] = pair
            additions.append(row)
        chosen, constants = make_scaling(pd.concat([r, *additions], ignore_index=True), START, CUTOFF)
        self.assertEqual(len(chosen), 3)
        for key in ['cQ', 'cV', 'sV']:
            self.assertEqual(constants[key], scaling()[key])

    def test_reference_cutoff_allows_completed_boundary_bin(self):
        r = reference()
        r.loc[2, 'bin_start'] = CUTOFF-pd.Timedelta(seconds=1)
        r.loc[2, 'available_at'] = CUTOFF
        selected, _ = make_scaling(r, START, CUTOFF)
        self.assertEqual(len(selected), 3)
        self.assertTrue(selected.available_at.le(CUTOFF).all())

    def test_no_scale_is_invented_for_zero_pressure_or_constant_variation(self):
        for col in ['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']:
            r = reference()
            r[col] = 0.
            with self.subTest(column=col), self.assertRaises(ValueError):
                make_scaling(r, START, CUTOFF)

    def test_reference_requires_unique_utc_bin_identity(self):
        r = reference()
        with self.assertRaises(ValueError):
            make_scaling(pd.concat([r, r.iloc[[0]]], ignore_index=True), START, CUTOFF)
        for col in ['bin_start', 'available_at']:
            altered = r.copy()
            altered[col] = altered[col].dt.tz_localize(None)
            with self.subTest(column=col), self.assertRaises(ValueError):
                make_scaling(altered, START, CUTOFF)

    def test_reference_is_not_mutated(self):
        r = reference()
        original = r.copy(deep=True)
        make_scaling(r, START, CUTOFF)
        pd.testing.assert_frame_equal(r, original)

    def test_mislabeled_future_calibration_and_incomplete_bin_fail(self):
        for start, available in [(CUTOFF, CUTOFF+pd.Timedelta(seconds=1)),
                                 (START, START+pd.Timedelta(seconds=2)),
                                 (START-pd.Timedelta(seconds=1), START)]:
            r = reference()
            r.loc[2, ['bin_start', 'available_at']] = [start, available]
            with self.subTest(start=start, available=available), self.assertRaises(ValueError):
                make_scaling(r, START, CUTOFF)


class EstimationInputTests(unittest.TestCase):
    def test_parents_are_unchanged_and_every_row_is_preserved(self):
        tables = ledger()
        a, o = tables['anchors'], tables['outcomes']
        before_a, before_o = a.copy(deep=True), o.copy(deep=True)
        result = inputs(tables)
        pd.testing.assert_frame_equal(a, before_a)
        pd.testing.assert_frame_equal(o, before_o)
        got_a = result['features'].set_index('episode_id').sort_index()
        got_o = result['samples'].set_index(['episode_id', 'horizon_sec']).sort_index()
        pd.testing.assert_frame_equal(got_a[a.columns.drop('episode_id')],
            a.set_index('episode_id').sort_index())
        pd.testing.assert_frame_equal(got_o[o.columns.drop(['episode_id', 'horizon_sec'])],
            o.set_index(['episode_id', 'horizon_sec']).sort_index())

    def test_exact_state_transformations(self):
        result = inputs()['features']
        self.assertTrue(np.allclose(result.pressure_magnitude, 2/np.sqrt(6)))
        self.assertTrue(np.allclose(result.aligned_prior_pressure, -2/np.sqrt(6)))
        self.assertTrue(result.variation_standardized.eq(0).all())
        self.assertTrue(result.variation_x_magnitude.eq(0).all())
        self.assertTrue(result.variation_x_alignment.eq(0).all())
        self.assertTrue(np.allclose(result.log_pre_displayed_size, np.log(10.)))
        self.assertTrue(np.allclose(result.magnitude_squared, 4/6))
        self.assertTrue(result.variation_squared.eq(0).all())

    def test_mirrored_direction_and_flow_preserve_aligned_states(self):
        frames = parent_frames()
        frames[0].loc[frames[0].episode_id.eq(9), 'pre_quote_ofi_300s'] = 2.
        set_direction(frames, 9, -1.)
        f = inputs(ledger(frames))['features'].set_index('episode_id')
        for col in DERIVED:
            self.assertAlmostEqual(f.loc[9, col], f.loc[2, col])
        self.assertEqual(f.loc[9, 'direction'], -f.loc[2, 'direction'])

    def test_zero_prior_flow_and_variation_are_valid_states(self):
        frames = parent_frames()
        frames[0]['pre_quote_ofi_300s'] = 0.
        frames[0]['pre_rv_300s_5s_bps'] = 0.
        frames[1].loc[frames[1].frequency_sec.eq(5), 'rv_bps'] = 0.
        f = inputs(ledger(frames))['features']
        self.assertTrue(f.regressor_eligible.all())
        self.assertTrue(f.pressure_magnitude.eq(0).all())
        self.assertTrue(f.aligned_prior_pressure.eq(0).all())
        self.assertTrue((f.variation_standardized < 0).all())

    def test_invalid_spread_or_size_excludes_without_flooring(self):
        for col, flag in [('pre_spread_pips', 'valid_spread'), ('pre_displayed_size', 'valid_size')]:
            for invalid in [0., -1., np.nan, np.inf]:
                with self.subTest(column=col, invalid=invalid):
                    tables = ledger()
                    tables['anchors'].loc[tables['anchors'].episode_id.eq(9), col] = invalid
                    r = inputs(tables)
                    bad = r['features'].set_index('episode_id').loc[9]
                    self.assertFalse(bad[flag])
                    self.assertFalse(bad.regressor_eligible)
                    self.assertEqual(len(r['features']), 2)
                    self.assertEqual(len(r['samples']), 6)
                    self.assertFalse(r['samples'].loc[r['samples'].episode_id.eq(9),
                                                      'estimation_eligible_continuation'].any())
                    if col == 'pre_displayed_size':
                        self.assertTrue(pd.isna(bad.log_pre_displayed_size))

    def test_exclusion_priority_is_fixed_and_not_return_sign_dependent(self):
        tables = ledger()
        tables['anchors'].loc[:, ['pre_spread_pips', 'pre_displayed_size']] = 0.
        reason_both = inputs(tables)['features'].input_exclusion_reason.tolist()
        tables['anchors']['pre_displayed_size'] = 10.
        reason_spread = inputs(tables)['features'].input_exclusion_reason.tolist()
        self.assertEqual(reason_both, reason_spread)

    def test_outcome_changes_cannot_change_features_or_control_eligibility(self):
        tables = ledger()
        original = inputs(tables)
        tables['outcomes']['initial_pips'] *= -8.
        tables['outcomes']['continuation_pips'] *= 13.
        tables['outcomes']['total_pips'] = (tables['outcomes'].initial_pips+
                                           tables['outcomes'].continuation_pips)
        altered = inputs(tables)
        pd.testing.assert_frame_equal(original['features'], altered['features'])
        pd.testing.assert_frame_equal(original['design_rows'], altered['design_rows'])

    def test_continuation_sample_does_not_require_initial_mark(self):
        frames = parent_frames()
        frames[0].loc[frames[0].episode_id.eq(9), 'baseline_quote_available'] = False
        frames[0].loc[frames[0].episode_id.eq(9), 'pre_mark_at'] = pd.NaT
        selected = frames[3].episode_id.eq(9)
        frames[3].loc[selected, ['initial_pips', 'total_pips']] = np.nan
        frames[3].loc[selected, ['total_quote_available', 'triplet_quote_available',
            'triplet_age_le_15', 'triplet_age_le_30', 'triplet_age_le_60']] = False
        r = inputs(ledger(frames))
        sample = r['samples'].loc[r['samples'].episode_id.eq(9)]
        self.assertTrue(sample.estimation_eligible_continuation.all())
        self.assertFalse(sample.estimation_eligible_initial.any())
        self.assertFalse(sample.estimation_eligible_total.any())
        self.assertFalse(sample.estimation_eligible_joint.any())
        self.assertEqual(len(sample), 3)

    def test_parent_zero_direction_is_retained_but_ineligible(self):
        frames = parent_frames()
        set_direction(frames, 9, 0.)
        r = inputs(ledger(frames))
        self.assertEqual(len(r['features']), 2)
        self.assertEqual(r['features'].set_index('episode_id').loc[9, 'direction'], 0.)
        self.assertFalse(r['features'].set_index('episode_id').loc[9, 'regressor_eligible'])

    def test_future_sensitivity_flags_are_not_baseline_predictors_or_exclusions(self):
        tables = ledger()
        original = inputs(tables)
        for name in tables['outcomes']:
            if name.startswith('triplet_age_le_'):
                tables['outcomes'][name] = False
        changed = inputs(tables)
        pd.testing.assert_frame_equal(original['features'], changed['features'])
        pd.testing.assert_frame_equal(original['design_rows'], changed['design_rows'])
        for flag in ['initial', 'continuation', 'total', 'joint']:
            self.assertTrue(changed['samples']['estimation_eligible_'+flag].all())

    def test_duplicate_keys_and_missing_horizon_fail_closed(self):
        for which in ['anchors', 'outcomes']:
            tables = ledger()
            tables[which] = pd.concat([tables[which], tables[which].iloc[[0]]], ignore_index=True)
            with self.subTest(table=which), self.assertRaises(ValueError):
                inputs(tables)
        tables = ledger()
        tables['outcomes'] = tables['outcomes'].iloc[1:].copy()
        with self.assertRaises(ValueError):
            inputs(tables)

    def test_invalid_parent_boolean_does_not_coerce_to_true(self):
        for table, column in [('anchors', 'pre_state_eligible'), ('outcomes', 'eligible_continuation')]:
            for value in [np.nan, 'False', 1]:
                with self.subTest(table=table, value=value):
                    tables = ledger()
                    tables[table][column] = tables[table][column].astype(object)
                    tables[table].loc[tables[table].index[0], column] = value
                    with self.assertRaises(ValueError):
                        inputs(tables)

    def test_calendar_is_encoded_from_consistent_utc_timestamp(self):
        r = inputs()
        f = r['features']
        for level in range(1, 24):
            self.assertTrue(np.array_equal(f[f'hour_{level:02d}'], f.bin_start.dt.hour.eq(level)))
        for level in range(1, 7):
            self.assertTrue(np.array_equal(f[f'weekday_{level}'], f.bin_start.dt.dayofweek.eq(level)))
        tables = ledger()
        tables['anchors']['onset_hour_utc'] = 17
        with self.assertRaises(ValueError):
            inputs(tables)

    def test_scaling_cutoff_cannot_be_after_any_onset(self):
        tables = ledger()
        s = scaling()
        s['reference_cutoff'] = (tables['anchors'].bin_start.max()+pd.Timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            build_inputs(tables['anchors'], tables['outcomes'], s)

    def test_shuffled_parent_rows_match_by_keys(self):
        tables = ledger()
        expected = inputs(tables)
        tables['anchors'] = tables['anchors'].iloc[::-1].reset_index(drop=True)
        tables['outcomes'] = tables['outcomes'].sample(frac=1, random_state=63).reset_index(drop=True)
        result = inputs(tables)
        for table, keys in [('features', ['episode_id']), ('samples', ['episode_id', 'horizon_sec'])]:
            pd.testing.assert_frame_equal(expected[table].sort_values(keys).reset_index(drop=True),
                                          result[table].sort_values(keys).reset_index(drop=True))

    def test_only_all_zero_calendar_columns_are_removed_from_schema(self):
        r = inputs()
        schema = r['schemas']['c_300']['baseline']
        # both fixture events are Sunday hour00; Sunday must stay despite
        # being collinear with the intercept when Monday is absent.
        self.assertIn('weekday_6', schema['columns'])
        self.assertNotIn('hour_01', schema['columns'])
        self.assertIn('hour_01', schema['unsupported_calendar_columns'])
        self.assertIn('hour_01', r['features'])
        self.assertIn('hour_01', r['design_rows'])
        for column in ['intercept', 'direction', 'variation_standardized',
                       'pressure_magnitude', 'aligned_prior_pressure',
                       'variation_x_magnitude', 'variation_x_alignment',
                       'pre_spread_pips', 'log_pre_displayed_size']:
            self.assertIn(column, schema['columns'])
        self.assertFalse(schema['full_column_rank'])
        self.assertFalse(schema['reference_present']['weekday_0_monday'])
        self.assertFalse(schema['usable_design'])

    def test_empty_eligible_cohort_remains_auditable(self):
        frames = parent_frames()
        for event_id in [2, 9]:
            set_direction(frames, event_id, 0.)
        r = inputs(ledger(frames))
        self.assertEqual(len(r['features']), 2)
        self.assertEqual(len(r['samples']), 6)
        self.assertEqual(len(r['design_rows']), 0)
        self.assertEqual(r['statistics']['regressor_eligible'], 0)
        for sample in r['schemas'].values():
            for model in ['baseline', 'curvature']:
                self.assertEqual(sample[model]['rows'], 0)
                self.assertFalse(sample[model]['usable_design'])

    def test_curvature_overflow_does_not_shrink_baseline_sample(self):
        tables = ledger()
        tables['anchors']['pre_quote_ofi_300s'] = 1.e200
        r = inputs(tables)
        self.assertTrue(r['features'].baseline_finite.all())
        self.assertTrue(r['features'].regressor_eligible.all())
        self.assertFalse(r['features'].curvature_finite.any())
        self.assertEqual(r['statistics']['curvature_nonfinite_on_baseline_cohort'], 2)
        for sample in r['schemas'].values():
            self.assertEqual(sample['baseline']['rows'], 2)
            self.assertEqual(sample['curvature']['rows'], 2)
            self.assertFalse(sample['curvature']['finite'])
            self.assertFalse(sample['curvature']['usable_design'])

    def test_baseline_overflow_is_explicit_attrition_not_imputation(self):
        tables = ledger()
        tables['anchors']['pre_quote_ofi_300s'] = 1.e308
        tables['anchors']['pre_rv_bps_5s'] = 1.e100
        r = inputs(tables)
        self.assertFalse(r['features'].baseline_finite.any())
        self.assertFalse(r['features'].regressor_eligible.any())
        self.assertTrue(r['features'].input_exclusion_reason.eq('nonfinite_baseline').all())
        self.assertTrue(np.isinf(r['features'].variation_x_magnitude).all())
        self.assertEqual(len(r['features']), 2)
        self.assertEqual(len(r['samples']), 6)

    def test_timing_roles_distinguish_prior_state_from_detected_alignment(self):
        roles = inputs()['specification']['predictor_timing_roles']
        for name in ['variation_standardized', 'pressure_magnitude',
                     'variation_x_magnitude', 'pre_spread_pips', 'log_pre_displayed_size']:
            self.assertEqual(roles[name], 'strictly_before_onset')
        for name in ['direction', 'aligned_prior_pressure', 'variation_x_alignment']:
            self.assertEqual(roles[name], 'detection_time')
        self.assertNotIn('initial_pips', roles)
        self.assertNotIn('continuation_pips', roles)

    def test_tampered_joint_mask_or_price_identity_fails(self):
        tables = ledger()
        tables['outcomes'].loc[0, 'eligible_joint'] = False
        with self.assertRaises(ValueError):
            inputs(tables)
        tables = ledger()
        tables['outcomes'].loc[0, 'total_pips'] += 1.
        with self.assertRaises(ValueError):
            inputs(tables)

    def test_full_rank_supported_design_and_leverage_accounting(self):
        rng = np.random.default_rng(306)
        ends = [86400*(1+j//20)+3600*(j % 2)+5*j for j in range(40)]
        frames = parent_frames(ends=ends, ids=range(40))
        for j in range(40):
            q, rv = rng.normal()*5, rng.uniform(.1, 3.)
            direction = [-1., 1.][(j // 2) % 2]
            set_direction(frames, j, direction)
            frames[0].loc[j, ['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps',
                'pre_spread_pips', 'pre_displayed_size', 'quote_ofi']] = [
                    q, rv, rng.uniform(.1, .8), rng.uniform(.2, 15.), 2*direction]
            frames[1].loc[frames[1].episode_id.eq(j)&frames[1].frequency_sec.eq(5), 'rv_bps'] = rv
            frames[2].loc[frames[2].episode_id.eq(j), 'inherited_net'] = 2*direction
        r = inputs(ledger(frames))
        for sample in r['schemas'].values():
            for model in ['baseline', 'curvature']:
                audit = sample[model]
                self.assertTrue(audit['usable_design'])
                self.assertEqual(audit['rank'], audit['column_count'])
                self.assertAlmostEqual(audit['leverage']['sum'], audit['rank'], places=10)
                self.assertTrue(0 <= audit['leverage']['max'] <= 1+1.e-12)


class LegacyRegressionGuardTests(unittest.TestCase):
    def test_direct_execution_blocks_before_heavy_imports_or_output_writes(self):
        script = ROOT/'src/run_corrected_regressions.py'
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp)/'sentinel_result.txt'
            sentinel.write_text('historical outputs stay intact', encoding='utf-8')
            before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, '-I', '-S', str(script)],
                cwd=temp, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Legacy regression pipeline is blocked', result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
            self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), before)
            self.assertEqual([p.name for p in Path(temp).iterdir()], ['sentinel_result.txt'])

    def test_import_guard_prevents_legacy_data_access_or_function_exposure(self):
        script = str(ROOT/'src/run_corrected_regressions.py')
        code = '''import importlib.util, sys
attempts=[]
def audit(event, args):
    if event == 'open' and any(token in str(args[0]) for token in ['final_event_database', 'data/results']):
        attempts.append(str(args[0]))
        raise AssertionError('legacy file accessed before guard')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('blocked_regression', sys.argv[1])
module=importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except RuntimeError as error:
    assert 'Legacy regression pipeline is blocked' in str(error)
else:
    raise AssertionError('legacy import succeeded')
assert not attempts
assert not hasattr(module, 'load_unified_dataset')
assert not hasattr(module, 'run_probit_with_session_fe')
assert 'pandas' not in sys.modules and 'numpy' not in sys.modules
print('guard verified')
'''
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, script],
                cwd=temp, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('guard verified', result.stdout)


if __name__ == '__main__':
    unittest.main()

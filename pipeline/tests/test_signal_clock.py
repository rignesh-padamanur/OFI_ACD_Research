import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.data_preparation import atomic_json, sha256
from src.intensity import compute_tick_intensity, materialize_clock_policy
from src.observation_eligibility import POLICIES
from src.signal_clock import SECOND as S, HOUR, SparseCursor, prepare_clock_inputs, build_clock_chunk, ClockStatistics
from src.verify_signal_clock import reference_chunk, verify_clock_run
from src.pipeline_state import iter_signal_clock

BASE = pd.Timestamp('2024-01-07T22:00:00Z')  # Sunday


def hours():
    return pd.DataFrame({'hour': pd.date_range(BASE, periods=4, freq='h'),
        'nominal_weekly_open': [True, True, True, False],
        'structurally_observed_hour': [True, True, False, False], 'rows': [5, 2, 0, 0]})


def intervals(bounds=None):
    bounds = [(0, 10*S), (20*S, 30*S)] if bounds is None else bounds
    return pd.DataFrame([{'policy': p, 'start': pd.Timestamp(BASE.value+a, tz='UTC'),
                         'end_exclusive': pd.Timestamp(BASE.value+b, tz='UTC')} for p in POLICIES for a,b in bounds])


def sparse(rows=()):
    return pd.DataFrame({'bin_start': pd.to_datetime([BASE.value+r[0]*S for r in rows], utc=True),
        'n_quotes': pd.Series([r[1] for r in rows], dtype='int64'),
        'quote_ofi': pd.Series([r[2] for r in rows], dtype=float),
        'n_invalid_transitions': pd.Series([r[3] for r in rows], dtype='int64'),
        'max_incoming_gap_sec': pd.Series([r[4] for r in rows], dtype=float)})


def build(rows=(), bounds=None, start=0, stop=40):
    h = hours(); support = intervals(bounds)
    origin, _, codes, prepared = prepare_clock_inputs(h, support)
    return build_clock_chunk(BASE.value+start*S, BASE.value+stop*S, sparse(rows),
                             origin=origin, hour_codes=codes, intervals=prepared)


def model(frame, policy='structural_only'):
    return materialize_clock_policy(frame, policy=policy)


class ClockContract(unittest.TestCase):
    def test_supported_silence_zero_unsupported_silence_missing(self):
        out = model(build())
        self.assertEqual(out.tick_count.iloc[0], 0)
        self.assertEqual(out.quote_ofi.iloc[0], 0)
        self.assertTrue(out[['tick_count', 'quote_ofi']].iloc[10:20].isna().all().all())

    def test_zero_ofi_with_activity_distinct_from_empty(self):
        out = model(build([(2, 3, 0., 0, .2)]))
        self.assertEqual(out.quote_ofi.iloc[2], 0)
        self.assertEqual(out.tick_count.iloc[2], 3)
        self.assertEqual(out.flow_status.iloc[2], 2)
        self.assertEqual(out.flow_status.iloc[3], 1)

    def test_invalid_transition_preserves_count_not_partial_flow(self):
        out = model(build([(2, 3, np.nan, 1, .2)]))
        self.assertEqual(out.tick_count.iloc[2], 3)
        self.assertTrue(np.isnan(out.quote_ofi.iloc[2]))
        self.assertEqual(out.flow_status.iloc[2], 3)

    def test_stale_incoming_transition_does_not_erase_count(self):
        frame = build([(2, 1, 7., 0, 16.)])
        self.assertEqual(model(frame).quote_ofi.iloc[2], 7)
        out = model(frame, 'fresh_15s')
        self.assertEqual(out.tick_count.iloc[2], 1)
        self.assertTrue(np.isnan(out.quote_ofi.iloc[2]))
        self.assertEqual(out.flow_status.iloc[2], 4)

    def test_gap_exact_threshold_allowed(self):
        out = model(build([(2, 1, 7., 0, 15.)]), 'fresh_15s')
        self.assertEqual(out.quote_ofi.iloc[2], 7)

    def test_unknown_incoming_gap_not_assumed_fresh(self):
        out = model(build([(2, 1, 7., 0, np.nan)]), 'fresh_15s')
        self.assertEqual(out.flow_status.iloc[2], 4)

    def test_invalid_transition_precedes_gap_status(self):
        out = model(build([(2, 1, np.nan, 1, 80.)]), 'fresh_15s')
        self.assertEqual(out.flow_status.iloc[2], 3)

    def test_exact_support_end_allows_full_preceding_bin(self):
        out = model(build(bounds=[(0, 3*S)]))
        self.assertEqual(out.quote_ofi.iloc[2], 0)
        self.assertTrue(np.isnan(out.quote_ofi.iloc[3]))

    def test_nanosecond_short_support_end_disallows_bin(self):
        out = model(build(bounds=[(0, 3*S-1)]))
        self.assertTrue(np.isnan(out.quote_ofi.iloc[2]))

    def test_fractional_support_start_disallows_first_partial_bin(self):
        out = model(build(bounds=[(1, 3*S)]))
        self.assertTrue(np.isnan(out.quote_ofi.iloc[0]))
        self.assertEqual(out.quote_ofi.iloc[1], 0)

    def test_two_partial_intervals_cannot_fill_one_bin(self):
        out = model(build(bounds=[(0, S//2), (S//2, 2*S)]))
        self.assertTrue(np.isnan(out.quote_ofi.iloc[0]))
        self.assertEqual(out.quote_ofi.iloc[1], 0)

    def test_adjacent_full_components_remain_distinct(self):
        out = model(build(bounds=[(0, 2*S), (2*S, 4*S)]))
        self.assertEqual(out.support_id.iloc[:4].tolist(), [0, 0, 1, 1])

    def test_exact_second_and_availability_not_backdated(self):
        out = model(build([(2, 1, 7., 0, .1)]))
        self.assertEqual(out.quote_ofi.iloc[1], 0)
        self.assertEqual(out.quote_ofi.iloc[2], 7)
        self.assertTrue((out.available_at == out.bin_start + pd.Timedelta(seconds=1)).all())

    def test_sunday_is_kept(self):
        out = model(build([(2, 1, 7., 0, .1)]))
        self.assertTrue((out.bin_start.dt.dayofweek == 6).all())
        self.assertEqual(out.quote_ofi.iloc[2], 7)

    def test_closed_and_missing_hours_have_missing_model_columns(self):
        out = build(start=7198, stop=10802)
        self.assertEqual(out.hour_status.iloc[2], 1)
        self.assertEqual(out.hour_status.iloc[-1], 0)
        self.assertTrue(model(out)[['tick_count', 'quote_ofi']].isna().all().all())

    def test_no_support_no_count_even_if_quote_present(self):
        out = model(build([(12, 1, 7., 0, .1)]))
        self.assertTrue(out[['tick_count', 'quote_ofi']].iloc[12].isna().all())

    def test_no_hidden_price_or_duration_fill(self):
        self.assertEqual(set(model(build()).columns), {'bin_start', 'available_at', 'tick_count', 'quote_ofi',
            'abs_quote_ofi', 'support_id', 'flow_status', 'joint_usable'})

    def test_arbitrary_chunk_boundary_invariance(self):
        full = build([(2, 1, 7., 0, .1), (20, 2, -3., 0, 18.)])
        split = pd.concat([build([(2, 1, 7., 0, .1)], stop=13),
                           build([(20, 2, -3., 0, 18.)], start=13)], ignore_index=True)
        pd.testing.assert_frame_equal(full, split)

    def test_future_quotes_do_not_change_past_values_given_fixed_support(self):
        full = build([(2, 1, 7., 0, .1), (20, 2, -3., 0, 18.)])
        prefix = build([(2, 1, 7., 0, .1)], stop=20)
        pd.testing.assert_frame_equal(full.iloc[:20], prefix)

    def test_brute_force_interval_membership_random_nanosecond_boundaries(self):
        rng = np.random.default_rng(723)
        for _ in range(20):
            edges = np.sort(rng.choice(np.arange(80)*S//2+1, size=12, replace=False))
            bounds = list(zip(edges[::2], edges[1::2]))
            out = build(bounds=bounds)
            expected = [next((j for j,(a,b) in enumerate(bounds) if a <= i*S and (i+1)*S <= b), -1) for i in range(40)]
            np.testing.assert_array_equal(out.support_id_structural_only, expected)

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError): model(build(), 'invented')
        with self.assertRaises(ValueError): next(iter_signal_clock(Path('/none'), policy=None))

    def test_legacy_intensity_api_blocked(self):
        with self.assertRaises(RuntimeError): compute_tick_intensity(pd.DataFrame())

    def test_discontinuous_ledger_rejected(self):
        with self.assertRaises(ValueError): prepare_clock_inputs(hours().drop(index=1), intervals())

    def test_overlapping_support_rejected(self):
        with self.assertRaises(ValueError): prepare_clock_inputs(hours(), intervals([(0, 3*S), (2*S, 4*S)]))

    def test_support_in_missing_hour_rejected(self):
        with self.assertRaises(ValueError): build(bounds=[(7200*S, 7202*S)], start=7200, stop=7203)

    def test_duplicate_sparse_second_rejected(self):
        with self.assertRaises(ValueError): build([(2, 1, 7., 0, .1), (2, 2, -3., 0, .1)])

    def test_materializer_rejects_corrupted_status(self):
        frame = build(); frame.loc[0, 'flow_status_structural_only'] = 2
        with self.assertRaises(ValueError): model(frame)


class ClockArtifactVerification(unittest.TestCase):
    def fixture(self, path):
        # coverage rows include a raw record removed by cleaning; signal rows do not.
        h = hours(); h['rows'] = [4, 0, 0, 0]; h['structurally_observed_hour'] = [True, False, False, False]
        support = intervals([(0, 5*S), (9*S, 12*S)])
        source = sparse([(0, 1, np.nan, 1, np.nan), (9, 2, 7., 0, 20.)])
        h.to_parquet(path/'hours.parquet', index=False); support.to_parquet(path/'support.parquet', index=False)
        source.to_parquet(path/'source.parquet', index=False)
        from repair_signal_clock import build_artifacts
        inputs = {key: {'path': str(path/file), 'sha256': sha256(path/file)} for key,file in
                  [('hourly_eligibility', 'hours.parquet'), ('support', 'support.parquet'), ('sparse_bins', 'source.parquet')]}
        build_artifacts(inputs, path, batch_size=997)
        manifest = {'inputs': inputs, 'outputs': {file: sha256(path/file) for file in
            ['signal_clock_1s.parquet', 'statistics.json', 'monthly_statistics.parquet', 'utc_hour_statistics.parquet']}}
        atomic_json(path/'manifest.json', manifest)
        return manifest

    def test_full_verifier_and_statistics_with_different_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); self.fixture(p)
            result = verify_clock_run(p, batch_size=1103)
            self.assertEqual(result['clock_rows_checked'], 14400)
            self.assertEqual(result['recorded_updates_checked'], 3)

    def corrupt(self, kind):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); manifest = self.fixture(p)
            if kind == 'statistics':
                file = 'statistics.json'; stats = json.loads((p/file).read_text())
                stats['policies']['structural_only']['supported_seconds'] += 1; atomic_json(p/file, stats)
            else:
                file = 'signal_clock_1s.parquet'; frame = pd.read_parquet(p/file)
                if kind == 'value': frame.loc[9, 'observed_quote_ofi'] += 1
                elif kind == 'status': frame.loc[6, 'flow_status_structural_only'] = 1
                elif kind == 'timing': frame.loc[9, 'available_at'] += pd.Timedelta(nanoseconds=1)
                elif kind == 'support_id': frame.loc[9, 'support_id_structural_only'] = 0
                frame.to_parquet(p/file, index=False)
            manifest['outputs'][file] = sha256(p/file); atomic_json(p/'manifest.json', manifest)
            with self.assertRaises(AssertionError): verify_clock_run(p, batch_size=1103)

    def test_value_corruption_detected_after_hash_refresh(self): self.corrupt('value')
    def test_status_corruption_detected_after_hash_refresh(self): self.corrupt('status')
    def test_timing_corruption_detected_after_hash_refresh(self): self.corrupt('timing')
    def test_support_identity_corruption_detected_after_hash_refresh(self): self.corrupt('support_id')
    def test_statistics_corruption_detected_after_hash_refresh(self): self.corrupt('statistics')

    def test_sparse_reader_does_not_drop_tail_or_bridge_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'sparse.parquet'
            sparse([(0, 1, 2., 0, 1.), (2, 1, 3., 0, 2.), (9, 1, 4., 0, 7.)]).to_parquet(p, index=False)
            cursor = SparseCursor(p, batch_size=1)
            self.assertEqual(cursor.take(BASE.value, BASE.value+2*S).n_quotes.sum(), 1)
            self.assertEqual(cursor.take(BASE.value+2*S, BASE.value+9*S).n_quotes.sum(), 1)
            self.assertFalse(cursor.exhausted())
            self.assertEqual(cursor.take(BASE.value+9*S, BASE.value+10*S).n_quotes.sum(), 1)
            self.assertTrue(cursor.exhausted())

    def test_registered_loader_masks_and_rejects_changed_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'data/signals/clock'; run.mkdir(parents=True)
            frame = build([(2, 1, 7., 0, .1)]); frame.to_parquet(run/'signal_clock_1s.parquet', index=False)
            parent = root/'parent'; parent.write_text('verified parent')
            parents = {'parent_data_run': 'data', 'parent_eligibility_run': 'elig', 'parent_signal_run': 'signal'}
            manifest = {'status': 'verified_supported_signal_clock', **parents,
                'inputs': {'parent': {'path': str(parent), 'sha256': sha256(parent)}},
                'outputs': {'signal_clock_1s.parquet': sha256(run/'signal_clock_1s.parquet')}}
            atomic_json(run/'manifest.json', manifest); atomic_json(run/'verification.json', {'status': 'passed'})
            atomic_json(root/'data/active_signal_clock.json', {'clock_run_id': 'clock', **parents})
            for file,key,value in [('active_run.json', 'data_run_id', 'data'), ('active_eligibility.json', 'eligibility_run_id', 'elig'),
                                   ('active_quote_imbalance.json', 'signal_run_id', 'signal')]:
                atomic_json(root/'data'/file, {key: value})
            out = pd.concat(iter_signal_clock(root, policy='structural_only', batch_size=7), ignore_index=True)
            pd.testing.assert_frame_equal(out, model(frame))
            atomic_json(root/'data/active_quote_imbalance.json', {'signal_run_id': 'different'})
            with self.assertRaises(RuntimeError): next(iter_signal_clock(root, policy='structural_only'))
            atomic_json(root/'data/active_quote_imbalance.json', {'signal_run_id': 'signal'})
            parent.write_text('changed parent')
            with self.assertRaises(RuntimeError): next(iter_signal_clock(root, policy='structural_only'))


if __name__ == '__main__':
    unittest.main()

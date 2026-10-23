"""Falsification checks for streaming materialization and quote-window history."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.duration_calibration import (
    DAY, DurationAuditSpec, apply_scales, fit_scales, source_spells, sufficient_statistics,
)
from src.panel_observations import iter_source_spells, build_seasonal_panel, read_quote_window
from tests.test_duration_observation import build_fixture

ORIGIN = pd.Timestamp('2024-01-01', tz='UTC')


class StreamingSpellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        build_fixture(self.directory)
        self.path = self.directory / 'quote_durations.parquet'
        self.tails = pd.read_parquet(self.directory / 'terminal_spells.parquet')

    def materialize(self, *, policy='structural_only', stop=300., batch_size=2, tails=None):
        return pd.concat(iter_source_spells(self.path, self.tails if tails is None else tails,
            policy=policy, stop_ns=(ORIGIN + pd.Timedelta(seconds=stop)).value,
            batch_size=batch_size), ignore_index=True)

    def test_prefix_equivalence_for_every_policy_cutoff_and_batch_size(self):
        for policy in ['structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s']:
            for stop in [1., 15., 15.000000001, 45., 165.000000001, 166., 170., 200., 300.]:
                expected = source_spells(self.path, self.tails, policy=policy,
                    stop_ns=(ORIGIN + pd.Timedelta(seconds=stop)).value)
                for size in [1, 2, 3, 7, 20]:
                    with self.subTest(policy=policy, stop=stop, size=size):
                        pd.testing.assert_frame_equal(self.materialize(policy=policy, stop=stop,
                                                                     batch_size=size), expected)

    def test_terminal_zero_exposure_is_preserved(self):
        final = self.materialize().iloc[-1]
        self.assertEqual(final.origin_quote_id, 7)
        self.assertEqual(final.exposure_ns, 0)
        self.assertFalse(final.event_observed)
        self.assertEqual(final.stop_code, 4)

    def test_equal_cutoff_completion_is_censored(self):
        final = self.materialize(stop=45.).iloc[-1]
        self.assertEqual(final.origin_quote_id, 1)
        self.assertEqual(final.end_at, ORIGIN + pd.Timedelta(seconds=45))
        self.assertFalse(final.event_observed)
        self.assertEqual(final.stop_code, 8)

    def test_missing_terminal_rejected_at_internal_component_boundary(self):
        bad = self.tails.loc[~(self.tails.policy.eq('structural_only') & self.tails.last_quote_id.eq(4))]
        with self.assertRaisesRegex(ValueError, 'Unaccounted terminal'):
            self.materialize(tails=bad)

    def test_duplicate_terminal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unique terminal'):
            self.materialize(tails=pd.concat([self.tails, self.tails.iloc[[0]]]))

    def test_negative_terminal_exposure_rejected(self):
        bad = self.tails.copy()
        bad.loc[bad.policy.eq('structural_only') & bad.last_quote_id.eq(4), 'stop'] = ORIGIN
        with self.assertRaisesRegex(ValueError, 'Negative observed exposure'):
            self.materialize(tails=bad)

    def test_reordered_or_gapped_global_identities_rejected(self):
        original = pd.read_parquet(self.path)
        for ids in [[0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 4, 6, 7, 8], list(range(1, 9))]:
            bad = original.copy()
            bad['quote_id'] = ids
            bad.to_parquet(self.path, index=False)
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'global quote identities'):
                self.materialize(batch_size=1)

    def test_duplicate_and_decreasing_times_rejected_across_batch_boundary(self):
        original = pd.read_parquet(self.path)
        for value in [original.available_at.iloc[1], original.available_at.iloc[1] - pd.Timedelta(seconds=1)]:
            bad = original.copy()
            bad.loc[2, 'available_at'] = value
            bad.to_parquet(self.path, index=False)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'increasing quote timestamps'):
                self.materialize(batch_size=2)

    def test_invalid_batch_size_policy_and_empty_prefix_rejected(self):
        for kwargs in [{'batch_size': 0}, {'batch_size': True}, {'batch_size': 1.5},
                       {'policy': 'unknown'}, {'stop': 0.}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.materialize(**kwargs)

    def test_yielded_batches_never_exceed_requested_bound(self):
        for size in [1, 2, 3]:
            frames = list(iter_source_spells(self.path, self.tails, policy='structural_only',
                stop_ns=(ORIGIN + pd.Timedelta(days=1)).value, batch_size=size))
            self.assertTrue(all(0 < len(frame) <= size for frame in frames))
            np.testing.assert_array_equal(pd.concat(frames).origin_quote_id, np.arange(8))


class SeasonalPanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / 'output'
        self.output.mkdir()
        day = np.repeat(np.arange(113), 120)
        times = ORIGIN.value + day * DAY + np.tile(np.arange(120), 113) * 15_000_000_000
        source = pd.DataFrame({'quote_id': np.arange(len(times)),
            'available_at': pd.to_datetime(times, utc=True), 'component_id_structural_only': day})
        self.path = self.directory / 'durations.parquet'
        source.to_parquet(self.path, index=False, row_group_size=173)
        tails = np.arange(119, len(source), 120)
        self.tails = pd.DataFrame({'policy': 'structural_only', 'last_quote_id': tails,
            'stop': pd.to_datetime(times[tails] + 15_000_000_000, utc=True),
            'stop_code': np.full(len(tails), 2, dtype='uint8')})
        self.tails.loc[0, 'stop'] = ORIGIN + pd.Timedelta(days=1)
        self.stop = ORIGIN + pd.Timedelta(days=113)

    def test_streamed_artifacts_match_unbatched_statistical_contract(self):
        expected_raw = source_spells(self.path, self.tails, policy='structural_only', stop_ns=self.stop.value)
        expected_table, expected_corrections = sufficient_statistics(expected_raw, ORIGIN.value, 113)
        seasonal, scales, status = fit_scales(expected_table, expected_corrections, ORIGIN.value,
                                             DurationAuditSpec(validation_days=1))
        expected_scaled = apply_scales(expected_raw, scales, status, ORIGIN.value)
        with patch('src.panel_observations.BATCH_SIZE', 257):
            actual = build_seasonal_panel(self.path, self.tails, self.output, origin=ORIGIN, stop=self.stop)
        for name, expected in [('raw_spells', expected_raw),
                               ('origin_daily_sufficient_statistics', expected_table),
                               ('historical_cutoff_corrections', expected_corrections),
                               ('seasonal_parameters', seasonal), ('seasonal_spells', expected_scaled)]:
            with self.subTest(name=name):
                pd.testing.assert_frame_equal(pd.read_parquet(self.output / (name + '.parquet')), expected)
        self.assertGreater(actual['source_batches'], 1)
        self.assertEqual(actual['source_spell_rows'], len(expected_raw))
        self.assertEqual(actual['source_exposure_ns'], int(expected_raw.exposure_ns.sum()))
        self.assertEqual(actual['source_completed_events'], int(expected_raw.event_observed.sum()))
        self.assertEqual(actual['seasonal_ready_rows'], int(expected_scaled.seasonal_status.eq(5).sum()))
        self.assertGreater(actual['seasonal_ready_rows'], 0)
        self.assertEqual(sum(actual['seasonal_status_counts'].values()), len(expected_raw))
        for name in ['raw_spells', 'seasonal_spells']:
            metadata = pq.ParquetFile(self.output / (name + '.parquet')).metadata
            self.assertTrue(all(metadata.row_group(i).num_rows <= 257 for i in range(metadata.num_row_groups)))

    def test_existing_artifact_is_not_overwritten(self):
        target = self.output / 'raw_spells.parquet'
        target.write_bytes(b'preserved')
        with self.assertRaises(FileExistsError):
            build_seasonal_panel(self.path, self.tails, self.output, origin=ORIGIN, stop=self.stop)
        self.assertEqual(target.read_bytes(), b'preserved')

    def test_incomplete_calendar_and_non_utc_midnight_rejected(self):
        for start, stop in [(ORIGIN, ORIGIN + pd.Timedelta(days=112)),
                            (ORIGIN.tz_localize(None), self.stop),
                            (ORIGIN, self.stop + pd.Timedelta(seconds=1))]:
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                build_seasonal_panel(self.path, self.tails, self.output, origin=start, stop=stop)

    def test_missing_output_directory_rejected(self):
        with self.assertRaises(ValueError):
            build_seasonal_panel(self.path, self.tails, self.directory / 'absent',
                                 origin=ORIGIN, stop=self.stop)


class QuoteWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'quotes.parquet'
        self.frame = pd.DataFrame({'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-04-01',
            '2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05'], utc=True),
            'bid': np.arange(8, dtype=float) + 1., 'ask': np.arange(8, dtype=float) + 1.1})
        self.frame.to_parquet(self.path, index=False, row_group_size=2)

    def window(self, start, stop, columns=('timestamp', 'bid')):
        return read_quote_window(self.path, pd.Timestamp(start, tz='UTC'),
                                 pd.Timestamp(stop, tz='UTC'), columns)

    def test_old_predecessor_is_retained_without_arbitrary_age_cap(self):
        actual = self.window('2024-03-01', '2024-04-02')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[[2, 3]][['timestamp', 'bid']].reset_index(drop=True))

    def test_exact_start_and_stop_have_strict_predecessor_and_exclusive_end(self):
        actual = self.window('2024-04-02', '2024-04-04')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[[3, 4, 5]][['timestamp', 'bid']].reset_index(drop=True))

    def test_empty_interior_still_returns_nearest_prior_quote(self):
        actual = self.window('2024-02-01', '2024-03-01')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[[2]][['timestamp', 'bid']].reset_index(drop=True))

    def test_before_and_after_source_boundaries(self):
        before = self.window('2023-01-01', '2023-12-01')
        self.assertEqual(len(before), 0)
        after = self.window('2025-01-01', '2025-02-01')
        pd.testing.assert_frame_equal(after, self.frame.iloc[[-1]][['timestamp', 'bid']].reset_index(drop=True))

    def test_window_before_first_quote_preserves_all_available_interior(self):
        actual = self.window('2023-12-01', '2024-01-03')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[:2][['timestamp', 'bid']].reset_index(drop=True))

    def test_missing_rowgroup_statistics_use_bounded_search_reads(self):
        self.frame.to_parquet(self.path, index=False, row_group_size=2, write_statistics=False)
        actual = self.window('2024-04-02', '2024-04-04')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[[3, 4, 5]][['timestamp', 'bid']].reset_index(drop=True))

    def test_requested_column_order_preserved(self):
        actual = self.window('2024-04-02', '2024-04-03', columns=['ask', 'timestamp'])
        self.assertEqual(list(actual.columns), ['ask', 'timestamp'])

    def test_late_window_does_not_load_prefix_rowgroups(self):
        real = pq.ParquetFile(self.path)
        reads = []

        class Spy:
            def __getattr__(self, name):
                return getattr(real, name)

            def read_row_groups(self, groups, **kwargs):
                reads.extend(groups)
                return real.read_row_groups(groups, **kwargs)

            def read_row_group(self, group, **kwargs):
                reads.append(group)
                return real.read_row_group(group, **kwargs)

        with patch('src.panel_observations.pq.ParquetFile', return_value=Spy()):
            actual = self.window('2024-04-04', '2024-04-06')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[[5, 6, 7]][['timestamp', 'bid']].reset_index(drop=True))
        self.assertEqual(reads, [2, 3])

    def test_invalid_window_and_column_contract_rejected(self):
        for start, stop, columns in [(ORIGIN, ORIGIN, ['timestamp']),
                                     (ORIGIN.tz_localize(None), ORIGIN + pd.Timedelta(days=1), ['timestamp']),
                                     (ORIGIN, ORIGIN + pd.Timedelta(days=1), ['bid']),
                                     (ORIGIN, ORIGIN + pd.Timedelta(days=1), ['timestamp', 'timestamp'])]:
            with self.subTest(start=start, stop=stop, columns=columns), self.assertRaises(ValueError):
                read_quote_window(self.path, start, stop, columns)

    def test_empty_registered_source_returns_typed_empty_window(self):
        self.frame.iloc[:0].to_parquet(self.path, index=False)
        actual = self.window('2024-01-01', '2024-02-01')
        pd.testing.assert_frame_equal(actual, self.frame.iloc[:0][['timestamp', 'bid']])


if __name__ == '__main__':
    unittest.main()

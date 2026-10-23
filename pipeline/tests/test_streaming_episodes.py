"""Chunk boundaries must never become economic episode boundaries."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from src.event_episodes import build_episode_tables
from src.streaming_episodes import EpisodeStream
from src.verify_event_episodes import reference_episode_tables


ORIGIN = pd.Timestamp('2024-05-05 23:59:50', tz='UTC')


def fixture(length=30, candidates=(), active=()):
    times = ORIGIN + pd.to_timedelta(np.arange(length), unit='s')
    o = pd.DataFrame({'bin_start': times, 'available_at': times + pd.Timedelta(seconds=1),
                      'support_id': 0, 'recorded_update_count': 0,
                      'completed_residual_count': 0, 'flow_status': 1,
                      'seasonal_ready': True, 'reason': 'quiet_no_duration', 'period': 'development'})
    active = sorted(set(active) | set(candidates))
    o.loc[active, ['recorded_update_count', 'completed_residual_count', 'flow_status']] = [1, 1, 2]
    o.loc[active, 'reason'] = 'eligible'
    s = o.loc[active, ['bin_start', 'available_at', 'support_id', 'recorded_update_count',
                       'completed_residual_count', 'period']].reset_index(drop=True).copy()
    s['candidate_event'] = pd.Series([i in candidates for i in active], dtype=bool)
    s['pre_rv_300s_5s_bps'], s['pre_quote_ofi_300s'] = 2., 10.
    s['quote_ofi'] = 1.
    return o, s


def partition(o, s, cuts, restore=False):
    stream = EpisodeStream()
    outputs = {key: [] for key in ['clock', 'membership', 'onsets', 'episodes']}
    for start, stop in zip([0] + cuts, cuts + [len(o)]):
        part = o.iloc[start:stop]
        selected = s.loc[s.bin_start.isin(part.bin_start)]
        result = stream.consume(part, selected)
        for key in ['clock', 'membership', 'onsets']:
            outputs[key].append(result[key])
        outputs['episodes'].append(result['closed_episodes'])
        if restore:
            stream = EpisodeStream(json.loads(json.dumps(stream.state_dict(), allow_nan=False)))
    outputs['episodes'].append(stream.finish())
    return {key: pd.concat(values, ignore_index=True) for key, values in outputs.items()}


class StreamingEpisodeTests(unittest.TestCase):
    def assert_batch_and_independent(self, o, s, cuts, restore=False):
        actual = partition(o, s, cuts, restore)
        expected = build_episode_tables(o, s)
        for key in expected:
            pd.testing.assert_frame_equal(actual[key], expected[key], obj=key)
        if len(o):
            reference = reference_episode_tables(o, s, o.bin_start.iloc[0], o.available_at.iloc[-1])
            for key, frame in zip(['clock', 'membership', 'episodes', 'onsets'], reference):
                # the older independent verifier leaves empty membership times
                # as object. Exact typed output equivalence was checked above.
                pd.testing.assert_frame_equal(actual[key], frame, check_dtype=bool(len(frame)),
                                              obj='independent ' + key)
        return actual

    def test_every_possible_single_cut_and_per_second_chunks(self):
        o, s = fixture(candidates=[0, 4, 10, 15, 21, 28], active=[2, 9, 18, 25])
        for cut in range(31):
            with self.subTest(cut=cut):
                self.assert_batch_and_independent(o, s, [cut], restore=True)
        self.assert_batch_and_independent(o, s, list(range(1, len(o))), restore=True)

    def test_transitive_chain_across_many_chunks_is_one_episode(self):
        o, s = fixture(1008, candidates=range(5, 1006, 5))
        result = self.assert_batch_and_independent(o, s, list(range(1, 1008, 7)), restore=True)
        self.assertEqual(len(result['episodes']), 1)
        self.assertEqual(result['episodes'].candidate_bins.iloc[0], 201)
        self.assertEqual(result['episodes'].span_seconds.iloc[0], 1001)

    def test_exact_five_clear_seconds_at_boundary_and_final_stop(self):
        for length in [10, 11, 12, 17]:
            for second in [9, 10, 11]:
                if second >= length:
                    continue
                with self.subTest(length=length, second=second):
                    o, s = fixture(length, candidates=[5, second])
                    self.assert_batch_and_independent(o, s, [6, 10] if length >= 10 else [6])

    def test_unknown_and_support_barrier_at_chunk_boundary(self):
        for index in [6, 7, 9, 10, 11]:
            for kind in ['incomplete_duration', 'seasonal_unavailable', 'invalid_flow', 'support']:
                with self.subTest(index=index, kind=kind):
                    o, s = fixture(candidates=[5, 12, 17, 25])
                    if kind == 'support':
                        o.loc[index:, 'support_id'] = 1
                        s.loc[s.bin_start.ge(o.bin_start.iloc[index]), 'support_id'] = 1
                    else:
                        o.loc[index, ['flow_status', 'recorded_update_count', 'completed_residual_count']] = [2, 1, 0]
                        o.loc[index, 'reason'] = kind
                        if kind == 'seasonal_unavailable':
                            o.loc[index, 'seasonal_ready'] = False
                        if kind == 'invalid_flow':
                            o.loc[index, 'flow_status'] = 3
                    self.assert_batch_and_independent(o, s, [index, index + 1], restore=True)

    def test_support_change_precedes_unknown_censor_reason(self):
        o, s = fixture(candidates=[5, 12])
        o.loc[8:10, ['support_id', 'flow_status', 'reason']] = [-1, 0, 'unsupported']
        o.loc[11:, 'support_id'] = 1
        s.loc[s.bin_start.ge(o.bin_start.iloc[11]), 'support_id'] = 1
        actual = self.assert_batch_and_independent(o, s, [8, 9, 11, 12], restore=True)
        self.assertEqual(actual['episodes'].termination.iloc[0], 'censored_support_change')

    def test_quiet_without_seasonal_factors_remains_clear(self):
        o, s = fixture(candidates=[5, 10])
        quiet = o.flow_status.eq(1)
        o.loc[quiet, 'seasonal_ready'] = False
        o.loc[quiet, 'reason'] = 'seasonal_unavailable'
        self.assert_batch_and_independent(o, s, [3, 5, 8, 10, 14], restore=True)

    def test_candidate_free_and_empty_chunks_preserve_typed_tables(self):
        for length in [0, 30]:
            for active in [[], [3, 8] if length else []]:
                with self.subTest(length=length, active=active):
                    o, s = fixture(length, active=active)
                    self.assert_batch_and_independent(o, s, [0, length // 2, length], restore=True)

    def test_unknown_chunks_preserve_no_clear_history(self):
        o, s = fixture(candidates=[5, 20])
        o.loc[6:19, ['flow_status', 'recorded_update_count', 'completed_residual_count']] = [2, 1, 0]
        o.loc[6:19, 'reason'] = 'incomplete_duration'
        result = self.assert_batch_and_independent(o, s, [6, 10, 20], restore=True)
        self.assertFalse(result['onsets'].onset_confirmed.iloc[1])

    def test_missing_prior_states_do_not_remove_candidates(self):
        o, s = fixture(candidates=[0, 6, 12, 18, 24])
        s.loc[1, 'pre_quote_ofi_300s'] = np.nan
        s.loc[2, 'pre_rv_300s_5s_bps'] = np.inf
        result = self.assert_batch_and_independent(o, s, [6, 12, 18, 24], restore=True)
        self.assertEqual(result['onsets'].state_history_available.tolist(), [True, False, False, True, True])

    def test_onsets_emitted_immediately_never_reissued_or_mutated(self):
        o, s = fixture(candidates=[5, 10])
        stream = EpisodeStream()
        first = stream.consume(o.iloc[:6], s.iloc[:1])
        saved = first['onsets'].copy(deep=True)
        self.assertEqual(len(first['onsets']), 1)
        self.assertTrue(first['closed_episodes'].empty)
        second = stream.consume(o.iloc[6:], s.iloc[1:])
        self.assertTrue(second['onsets'].empty)
        self.assertEqual(second['closed_episodes'].candidate_bins.iloc[0], 2)
        pd.testing.assert_frame_equal(first['onsets'], saved)
        self.assertTrue(stream.finish().empty)

    def test_only_finish_censors_at_last_observed_availability(self):
        o, s = fixture(8, candidates=[5])
        stream = EpisodeStream()
        self.assertTrue(stream.consume(o, s)['closed_episodes'].empty)
        final = stream.finish()
        self.assertEqual(final.termination.iloc[0], 'censored_sample_end')
        self.assertEqual(final.observation_stop_at.iloc[0], o.available_at.iloc[-1])
        self.assertTrue(stream.finish().empty)
        with self.assertRaises(ValueError):
            stream.consume(o.iloc[:0], s.iloc[:0])

    def test_checkpoint_is_detached_native_json_and_constant_size(self):
        o, s = fixture(1000, candidates=range(5, 1000, 5))
        stream = EpisodeStream()
        stream.consume(o, s)
        state = stream.state_dict()
        payload = json.dumps(state, allow_nan=False)
        self.assertLess(len(payload), 1000)
        restored = EpisodeStream(json.loads(payload))
        self.assertEqual(restored.state_dict(), state)
        state['open_episode']['candidate_bins'] = -99
        self.assertEqual(stream.state_dict()['open_episode']['candidate_bins'], 199)
        pd.testing.assert_frame_equal(stream.finish(), restored.finish())

    def test_chunk_gaps_overlaps_and_internal_bad_clock_are_atomic(self):
        o, s = fixture(candidates=[5, 12], active=[15])
        stream = EpisodeStream()
        stream.consume(o.iloc[:8], s.iloc[:1])
        before = stream.state_dict()
        for bad in [o.iloc[7:], o.iloc[9:], o.iloc[8:].drop(index=10), o.iloc[8:].iloc[::-1]]:
            with self.assertRaises(ValueError):
                stream.consume(bad, s.iloc[1:])
            self.assertEqual(stream.state_dict(), before)
        valid = stream.consume(o.iloc[8:], s.iloc[1:])
        self.assertEqual(len(valid['onsets']), 1)

    def test_invalid_fields_and_scores_leave_state_unchanged(self):
        o, s = fixture(candidates=[5], active=[10])
        cases = []
        for column, value in [('support_id', 0.5), ('recorded_update_count', -1),
                              ('completed_residual_count', 2), ('flow_status', 6),
                              ('seasonal_ready', None), ('reason', None), ('period', 'calibration')]:
            bad = o.copy()
            if column in ['support_id']:
                bad[column] = bad[column].astype(float)
            if column == 'seasonal_ready':
                bad[column] = bad[column].astype(object)
            bad.loc[5, column] = value
            cases.append((bad, s))
        bad = s.copy(); bad['candidate_event'] = [1, 0]; cases.append((o, bad))
        bad = s.copy(); bad['candidate_event'] = [True, None]; cases.append((o, bad))
        bad = s.copy(); bad['pre_quote_ofi_300s'] = ['bad', 'input']; cases.append((o, bad))
        bad = s.copy(); bad.loc[0, 'support_id'] = 1; cases.append((o, bad))
        bad = s.copy(); bad.loc[0, 'recorded_update_count'] = 2; cases.append((o, bad))
        bad = s.copy(); bad.loc[0, 'available_at'] += pd.Timedelta(seconds=1); cases.append((o, bad))
        cases.extend([(o, s.iloc[:1]), (o, pd.concat([s, s.iloc[:1]], ignore_index=True)),
                      (o.drop(columns=['reason']), s), (o, s.drop(columns=['pre_quote_ofi_300s']))])
        for bad_o, bad_s in cases:
            with self.subTest(columns=list(bad_s.columns)):
                stream = EpisodeStream()
                before = stream.state_dict()
                with self.assertRaises(ValueError):
                    stream.consume(bad_o, bad_s)
                self.assertEqual(stream.state_dict(), before)

    def test_malformed_checkpoint_rejected(self):
        o, s = fixture(8, candidates=[5])
        stream = EpisodeStream(); stream.consume(o, s)
        state = stream.state_dict()
        changes = [('version', 2), ('next_episode_id', True), ('next_timestamp_ns', 1),
                   ('previous_support_id', -1), ('clear_run', 5), ('finished', True)]
        for field, value in changes:
            bad = copy.deepcopy(state); bad[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    EpisodeStream(bad)
        for field, value in [('span_seconds', 999), ('candidate_bins', 0),
                              ('onset_confirmed', False), ('first_candidate_bin', 1)]:
            bad = copy.deepcopy(state); bad['open_episode'][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    EpisodeStream(bad)

    def test_missing_numeric_boolean_and_timestamp_fields_rejected(self):
        o, s = fixture(candidates=[5], active=[10])
        for name in ['support_id', 'recorded_update_count', 'completed_residual_count', 'flow_status']:
            for kind in ['missing', 'boolean']:
                bad = o.copy()
                if kind == 'missing':
                    bad[name] = bad[name].astype('Int64')
                    bad.loc[5, name] = pd.NA
                else:
                    bad[name] = bad[name].astype(bool)
                with self.subTest(name=name, kind=kind):
                    with self.assertRaises(ValueError):
                        EpisodeStream().consume(bad, s)
        for name in ['bin_start', 'available_at']:
            bad = o.copy(); bad.loc[5, name] = pd.NaT
            with self.assertRaises(ValueError):
                EpisodeStream().consume(bad, s)
        bad = s.copy(); bad['candidate_event'] = bad.candidate_event.astype('boolean')
        bad.loc[0, 'candidate_event'] = pd.NA
        with self.assertRaises(ValueError):
            EpisodeStream().consume(o, bad)

    def test_seeded_adversarial_partitions_match_both_references(self):
        rng = np.random.default_rng(71023)
        for _ in range(12):
            length = 240
            candidates = np.flatnonzero(rng.uniform(size=length) < .14).tolist()
            active = np.flatnonzero(rng.uniform(size=length) < .3).tolist()
            o, s = fixture(length, candidates, active)
            unknown = np.flatnonzero(rng.uniform(size=length) < .08)
            o.loc[unknown, ['flow_status', 'recorded_update_count', 'completed_residual_count']] = [2, 1, 0]
            o.loc[unknown, 'reason'] = 'incomplete_duration'
            s = s.loc[~s.bin_start.isin(o.bin_start.iloc[unknown])].reset_index(drop=True)
            for change in [70, 115, 200]:
                o.loc[change:, 'support_id'] += 1
                s.loc[s.bin_start.ge(o.bin_start.iloc[change]), 'support_id'] += 1
            cuts = sorted(rng.choice(np.arange(1, length), 35, replace=False).tolist())
            self.assert_batch_and_independent(o, s, cuts, restore=True)


if __name__ == '__main__':
    unittest.main()

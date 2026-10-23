"""Adversarial examples for observable episode timing, censorship and history."""
import unittest

import numpy as np
import pandas as pd

from src.event_episodes import build_episode_tables, attach_history_quality


ORIGIN = pd.Timestamp('2024-04-22 12:00:00', tz='UTC')


def fixture(length=25, candidates=(), active=(), origin=ORIGIN):
    """Complete known clock; scores are deliberately supplied only for active bins."""
    times = origin + pd.to_timedelta(np.arange(length), unit='s')
    opportunities = pd.DataFrame({
        'bin_start': times, 'available_at': times + pd.Timedelta(seconds=1),
        'period': 'development', 'recorded_update_count': 0,
        'support_id': 0, 'flow_status': 1, 'seasonal_ready': True,
        'completed_residual_count': 0, 'eligible_without_duration': True,
        'reason': 'quiet_no_duration',
    })
    active = sorted(set(active) | set(candidates))
    opportunities.loc[active, ['flow_status', 'recorded_update_count',
                               'completed_residual_count']] = [2, 1, 1]
    opportunities.loc[active, 'reason'] = 'eligible'
    scored = opportunities.loc[active, ['bin_start', 'available_at', 'support_id',
                                        'period']].copy().reset_index(drop=True)
    scored['candidate_event'] = pd.Series([i in candidates for i in active], dtype=bool)
    scored['pre_rv_300s_5s_bps'] = 2.
    scored['pre_quote_ofi_300s'] = 10.
    scored['quote_ofi'] = 1.
    return opportunities, scored


def unknown_at(opportunities, scored, index, reason='incomplete_duration'):
    opportunities = opportunities.copy()
    scored = scored.copy()
    opportunities.loc[index, ['flow_status', 'recorded_update_count',
                              'completed_residual_count']] = [2, 1, 0]
    opportunities.loc[index, 'reason'] = reason
    if reason == 'seasonal_unavailable':
        opportunities.loc[index, 'seasonal_ready'] = False
        opportunities.loc[index, 'eligible_without_duration'] = False
        opportunities.loc[index, 'completed_residual_count'] = 1
    elif reason == 'invalid_flow':
        opportunities.loc[index, 'flow_status'] = 3
        opportunities.loc[index, 'eligible_without_duration'] = False
    elif reason == 'unsupported':
        opportunities.loc[index, 'support_id'] = -1
        opportunities.loc[index, 'flow_status'] = 0
        opportunities.loc[index, 'eligible_without_duration'] = False
    timestamp = opportunities.bin_start.iloc[index]
    return opportunities, scored.loc[scored.bin_start.ne(timestamp)].reset_index(drop=True)


class EpisodeTimingTests(unittest.TestCase):
    def test_candidate_gap_four_five_and_six(self):
        for gap, expected in [(4, 1), (5, 1), (6, 2)]:
            with self.subTest(gap=gap):
                o, s = fixture(candidates=[5, 5 + gap])
                result = build_episode_tables(o, s)
                self.assertEqual(len(result['episodes']), expected)
                self.assertEqual(result['episodes'].candidate_bins.sum(), 2)
                self.assertTrue(result['episodes'].onset_confirmed.all())
                self.assertTrue(result['episodes'].termination.eq('closed_no_trigger').all())

    def test_transitive_chain_and_exact_closure_available_time(self):
        o, s = fixture(candidates=[5, 10, 15])
        episode = build_episode_tables(o, s)['episodes'].iloc[0]
        self.assertEqual(episode.candidate_bins, 3)
        self.assertEqual(episode.first_candidate_bin, ORIGIN + pd.Timedelta(seconds=5))
        self.assertEqual(episode.detected_at, ORIGIN + pd.Timedelta(seconds=6))
        self.assertEqual(episode.last_candidate_bin, ORIGIN + pd.Timedelta(seconds=15))
        self.assertEqual(episode.last_candidate_available_at, ORIGIN + pd.Timedelta(seconds=16))
        self.assertEqual(episode.closure_known_at, ORIGIN + pd.Timedelta(seconds=21))

    def test_five_preceding_clear_bins_required_for_clean_onset(self):
        for onset, confirmed in [(0, False), (4, False), (5, True), (8, True)]:
            with self.subTest(onset=onset):
                o, s = fixture(candidates=[onset])
                episode = build_episode_tables(o, s)['episodes'].iloc[0]
                self.assertEqual(bool(episode.onset_confirmed), confirmed)
                self.assertEqual(episode.preceding_clear_seconds, min(onset, 5))

    def test_missing_seasonal_factor_does_not_turn_observed_quiet_into_unknown(self):
        o, s = fixture(candidates=[5, 10])
        quiet = o.recorded_update_count.eq(0)
        o.loc[quiet, 'seasonal_ready'] = False
        o.loc[quiet, 'eligible_without_duration'] = False
        o.loc[quiet, 'reason'] = 'seasonal_unavailable'
        result = build_episode_tables(o, s)
        self.assertTrue(result['clock'].loc[quiet, 'observation_state'].eq('clear_quiet').all())
        self.assertEqual(len(result['episodes']), 1)
        self.assertTrue(result['episodes'].onset_confirmed.iloc[0])
        self.assertEqual(result['episodes'].closure_known_at.iloc[0], ORIGIN + pd.Timedelta(seconds=16))

    def test_below_threshold_active_bins_are_clear(self):
        o, s = fixture(candidates=[5, 10], active=[0, 3, 6, 7, 8, 9, 11, 13])
        result = build_episode_tables(o, s)
        actual = result['clock'].set_index('bin_start').observation_state
        for row in s.loc[~s.candidate_event].itertuples():
            self.assertEqual(actual.loc[row.bin_start], 'clear_active')
        self.assertEqual(len(result['episodes']), 1)

    def test_unknown_active_bins_split_and_do_not_certify_onset(self):
        for reason in ['incomplete_duration', 'seasonal_unavailable', 'invalid_flow']:
            with self.subTest(reason=reason):
                o, s = fixture(candidates=[5, 8, 14])
                o, s = unknown_at(o, s, 7, reason)
                result = build_episode_tables(o, s)
                episodes = result['episodes']
                self.assertEqual(len(episodes), 3)
                self.assertEqual(episodes.onset_confirmed.tolist(), [True, False, True])
                self.assertEqual(episodes.preceding_clear_seconds.tolist(), [5, 0, 5])
                self.assertEqual(episodes.termination.iloc[0], 'censored_unknown')
                self.assertEqual(episodes.observation_stop_at.iloc[0], ORIGIN + pd.Timedelta(seconds=7))
                self.assertTrue(pd.isna(episodes.closure_known_at.iloc[0]))
                self.assertEqual(result['clock'].observation_state.iloc[7], 'unknown')

    def test_unknown_at_each_required_closure_bin_censors(self):
        for barrier in range(6, 11):
            with self.subTest(barrier=barrier):
                o, s = fixture(candidates=[5])
                o, s = unknown_at(o, s, barrier)
                episode = build_episode_tables(o, s)['episodes'].iloc[0]
                self.assertEqual(episode.termination, 'censored_unknown')
                self.assertTrue(pd.isna(episode.closure_known_at))
                self.assertEqual(episode.observation_stop_at, ORIGIN + pd.Timedelta(seconds=barrier))

    def test_barrier_after_completed_closure_does_not_retroactively_censor(self):
        o, s = fixture(candidates=[5])
        o, s = unknown_at(o, s, 11)
        episode = build_episode_tables(o, s)['episodes'].iloc[0]
        self.assertEqual(episode.termination, 'closed_no_trigger')
        self.assertEqual(episode.closure_known_at, ORIGIN + pd.Timedelta(seconds=11))

    def test_support_change_breaks_episode_and_resets_history(self):
        o, s = fixture(candidates=[5, 10])
        boundary = ORIGIN + pd.Timedelta(seconds=8)
        o.loc[o.bin_start.ge(boundary), 'support_id'] = 1
        s.loc[s.bin_start.ge(boundary), 'support_id'] = 1
        episodes = build_episode_tables(o, s)['episodes']
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.termination.iloc[0], 'censored_support_change')
        self.assertEqual(episodes.observation_stop_at.iloc[0], boundary)
        self.assertFalse(episodes.onset_confirmed.iloc[1])
        self.assertEqual(episodes.preceding_clear_seconds.iloc[1], 2)

    def test_unsupported_interval_is_unknown_even_without_recorded_quotes(self):
        o, s = fixture(candidates=[5, 10])
        o, s = unknown_at(o, s, 8, 'unsupported')
        o.loc[8, 'recorded_update_count'] = 0
        result = build_episode_tables(o, s)
        self.assertEqual(result['clock'].observation_state.iloc[8], 'unknown')
        self.assertEqual(len(result['episodes']), 2)
        self.assertFalse(result['episodes'].onset_confirmed.iloc[1])

    def test_sample_end_before_and_exactly_at_closure(self):
        for length, termination in [(6, 'censored_sample_end'),
                                    (10, 'censored_sample_end'),
                                    (11, 'closed_no_trigger')]:
            with self.subTest(length=length):
                o, s = fixture(length=length, candidates=[5])
                episode = build_episode_tables(o, s)['episodes'].iloc[0]
                self.assertEqual(episode.termination, termination)
                if termination == 'censored_sample_end':
                    self.assertEqual(episode.observation_stop_at, ORIGIN + pd.Timedelta(seconds=length))
                    self.assertTrue(pd.isna(episode.closure_known_at))
                else:
                    self.assertEqual(episode.closure_known_at, ORIGIN + pd.Timedelta(seconds=11))

    def test_calendar_midnight_alone_does_not_split(self):
        origin = pd.Timestamp('2024-04-22 23:59:52', tz='UTC')
        o, s = fixture(candidates=[5, 10], origin=origin)
        episodes = build_episode_tables(o, s)['episodes']
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.candidate_bins.iloc[0], 2)

    def test_appending_future_changes_no_onset_information(self):
        o, s = fixture(candidates=[5, 10, 16])
        full = build_episode_tables(o, s)
        short_o = o.iloc[:9].copy()
        short_s = s.loc[s.bin_start.lt(short_o.available_at.iloc[-1])].copy()
        short = build_episode_tables(short_o, short_s)
        columns = ['bin_start', 'available_at', 'support_id', 'episode_id',
                   'onset_confirmed', 'preceding_clear_seconds',
                   'pre_rv_300s_5s_bps', 'pre_quote_ofi_300s',
                   'state_history_available', 'state_sample_eligible']
        pd.testing.assert_frame_equal(short['onsets'][columns].reset_index(drop=True),
                                      full['onsets'].iloc[:1][columns].reset_index(drop=True))
        self.assertEqual(short['episodes'].termination.iloc[0], 'censored_sample_end')
        self.assertEqual(full['episodes'].termination.iloc[0], 'closed_no_trigger')

    def test_every_trigger_assigned_once_and_no_nontrigger_assigned(self):
        o, s = fixture(candidates=[0, 2, 8, 14], active=[3, 9, 15])
        result = build_episode_tables(o, s)
        membership, clock = result['membership'], result['clock']
        self.assertEqual(len(membership), 4)
        self.assertFalse(membership.bin_start.duplicated().any())
        self.assertEqual(result['episodes'].candidate_bins.sum(), 4)
        self.assertTrue(clock.loc[clock.observation_state.ne('trigger'), 'episode_id'].eq(-1).all())
        self.assertTrue(clock.loc[clock.observation_state.eq('trigger'), 'episode_id'].ge(0).all())

    def test_missing_prestate_is_separate_from_detection_and_clean_onset(self):
        o, s = fixture(candidates=[0, 6, 12, 18])
        s.loc[1, 'pre_rv_300s_5s_bps'] = np.nan
        s.loc[2, 'pre_quote_ofi_300s'] = np.inf
        result = build_episode_tables(o, s)
        onsets = result['onsets']
        self.assertEqual(len(onsets), 4)
        self.assertEqual(onsets.onset_confirmed.tolist(), [False, True, True, True])
        self.assertEqual(onsets.state_history_available.tolist(), [True, False, False, True])
        self.assertEqual(onsets.state_sample_eligible.tolist(), [False, False, False, True])

    def test_no_candidates_yields_no_episodes(self):
        o, s = fixture(active=[3, 8, 10])
        result = build_episode_tables(o, s)
        for table in ['episodes', 'onsets', 'membership']:
            self.assertEqual(len(result[table]), 0)
        self.assertEqual(len(result['clock']), len(o))

    def test_all_quiet_and_empty_clocks_keep_typed_empty_outputs(self):
        for length in [0, 12]:
            with self.subTest(length=length):
                o, s = fixture(length=length)
                result = build_episode_tables(o, s)
                self.assertEqual(len(result['clock']), length)
                for table in ['episodes', 'onsets', 'membership']:
                    self.assertTrue(result[table].empty)
                self.assertEqual(str(result['onsets'].bin_start.dt.tz), 'UTC')

    def test_all_unknown_clock_preserved_without_fabricating_events(self):
        for kind in ['unsupported', 'incomplete_duration']:
            with self.subTest(kind=kind):
                o, s = fixture(length=12)
                o['reason'] = kind
                if kind == 'unsupported':
                    o['support_id'] = -1
                    o['flow_status'] = 0
                else:
                    o['flow_status'] = 2
                    o['recorded_update_count'] = 1
                result = build_episode_tables(o, s)
                self.assertEqual(len(result['clock']), 12)
                self.assertTrue(result['clock'].observation_state.eq('unknown').all())
                self.assertTrue(result['episodes'].empty)
                self.assertTrue(result['onsets'].empty)


class EpisodeValidationTests(unittest.TestCase):
    def test_missing_duplicate_unsorted_and_naive_clock_are_rejected(self):
        o, s = fixture(candidates=[5])
        bads = [o.drop(index=9), pd.concat([o, o.iloc[[9]]], ignore_index=True),
                o.iloc[::-1].reset_index(drop=True)]
        naive = o.copy()
        for name in ['bin_start', 'available_at']:
            naive[name] = naive[name].dt.tz_localize(None)
        bads.append(naive)
        delayed = o.copy(); delayed.loc[9, 'available_at'] += pd.Timedelta(seconds=1)
        bads.append(delayed)
        for bad in bads:
            with self.subTest(rows=len(bad)):
                with self.assertRaises(ValueError):
                    build_episode_tables(bad, s)

    def test_missing_extra_duplicate_and_misaligned_scores_are_rejected(self):
        o, s = fixture(candidates=[5], active=[10])
        extra = s.iloc[[0]].copy()
        extra['bin_start'] = ORIGIN + pd.Timedelta(seconds=7)
        extra['available_at'] = extra.bin_start + pd.Timedelta(seconds=1)
        bads = [s.iloc[:1], pd.concat([s, extra], ignore_index=True),
                pd.concat([s, s.iloc[[0]]], ignore_index=True)]
        sid = s.copy(); sid.loc[0, 'support_id'] = 1; bads.append(sid)
        time = s.copy(); time.loc[0, 'available_at'] += pd.Timedelta(seconds=1); bads.append(time)
        missing_flag = s.copy(); missing_flag['candidate_event'] = pd.Series([True, None], dtype='object')
        bads.append(missing_flag)
        for bad in bads:
            with self.subTest(rows=len(bad)):
                with self.assertRaises(ValueError):
                    build_episode_tables(o, bad)

    def test_calibration_rows_not_accepted_as_prospective_development(self):
        o, s = fixture(candidates=[5])
        o.loc[0, 'period'] = 'calibration'
        with self.assertRaises(ValueError):
            build_episode_tables(o, s)

    def test_fractional_integer_fields_and_nonboolean_readiness_are_rejected(self):
        o, s = fixture(candidates=[5])
        bads = []
        for column in ['support_id', 'recorded_update_count',
                       'completed_residual_count', 'flow_status']:
            bad = o.copy()
            bad[column] = bad[column].astype(float)
            bad.loc[5, column] += .5
            bads.append((column, bad))
        bad = o.copy(); bad['seasonal_ready'] = 'False'
        bads.append(('seasonal_ready', bad))
        for column, bad in bads:
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    build_episode_tables(bad, s)


class HistoryQualityTests(unittest.TestCase):
    def history_fixture(self):
        o, s = fixture(candidates=[5])
        onsets = build_episode_tables(o, s)['onsets']
        onset = onsets.bin_start.iloc[0]
        offsets = np.array([-301., -300., -200., -100., -1., 0., .5, 10.])
        quotes = pd.DataFrame({
            'timestamp': onset + pd.to_timedelta(offsets, unit='s'),
            'bid': 1.1 + np.arange(len(offsets)) * .00001,
            'ask': 1.1001 + np.arange(len(offsets)) * .00001,
        })
        support = pd.DataFrame({'start': [onset - pd.Timedelta(seconds=302)],
                                'end_exclusive': [onset + pd.Timedelta(seconds=100)]})
        marks = []
        for offset in range(-300, 1, 5):
            cutoff = onset + pd.Timedelta(seconds=offset)
            quote = quotes.loc[quotes.timestamp.lt(cutoff)].iloc[-1]
            marks.append(np.log((quote.bid + quote.ask) / 2))
        onsets['pre_rv_300s_5s_bps'] = np.sqrt(np.square(np.diff(marks)).sum()) * 10000
        return onsets, quotes, support

    def test_quote_boundaries_and_scalar_history_reference(self):
        onsets, quotes, support = self.history_fixture()
        actual = attach_history_quality(onsets, quotes, support).iloc[0]
        onset = onsets.bin_start.iloc[0]
        marks, ages, quote_times = [], [], []
        for offset in range(-300, 1, 5):
            cutoff = onset + pd.Timedelta(seconds=offset)
            q = quotes.loc[quotes.timestamp.lt(cutoff)].iloc[-1]
            marks.append(np.log((q.bid + q.ask) / 2))
            ages.append((cutoff - q.timestamp).total_seconds())
            quote_times.append(q.timestamp)
        rv = np.sqrt(np.square(np.diff(marks)).sum()) * 10000
        repeats = sum(a == b for a, b in zip(quote_times[:-1], quote_times[1:])) / 60
        self.assertAlmostEqual(actual.raw_pre_rv_300s_5s_bps, rv, places=11)
        self.assertEqual(actual.pre_mark_max_age_sec, max(ages))
        self.assertAlmostEqual(actual.pre_mark_repeat_fraction, repeats)
        self.assertEqual(actual.pre_recorded_updates_300s, 4)
        self.assertEqual(actual.pre_max_observed_silence_sec, 100)

    def test_history_diagnostics_ignore_current_and_future_quotes(self):
        onsets, quotes, support = self.history_fixture()
        base = attach_history_quality(onsets, quotes, support)
        changed = quotes.copy()
        future = changed.timestamp.ge(onsets.bin_start.iloc[0])
        changed.loc[future, ['bid', 'ask']] *= 10
        actual = attach_history_quality(onsets, changed, support)
        fields = ['raw_pre_rv_300s_5s_bps', 'pre_mark_max_age_sec',
                  'pre_mark_repeat_fraction', 'pre_recorded_updates_300s',
                  'pre_max_observed_silence_sec']
        pd.testing.assert_frame_equal(base[fields], actual[fields])

    def test_unavailable_history_does_not_receive_computed_quality_numbers(self):
        onsets, quotes, support = self.history_fixture()
        onsets['state_history_available'] = False
        onsets['state_sample_eligible'] = False
        actual = attach_history_quality(onsets, quotes, support)
        for field in ['raw_pre_rv_300s_5s_bps', 'pre_mark_max_age_sec',
                      'pre_mark_repeat_fraction', 'pre_recorded_updates_300s',
                      'pre_max_observed_silence_sec']:
            self.assertTrue(actual[field].isna().all(), field)

    def test_quote_before_support_cannot_supply_first_history_mark(self):
        onsets, quotes, support = self.history_fixture()
        support.loc[0, 'start'] = onsets.bin_start.iloc[0] - pd.Timedelta(seconds=300)
        # the advertised state history is internally inconsistent with its support;
        # fail closed instead of attaching numbers from a pre-support quote.
        with self.assertRaises(ValueError):
            attach_history_quality(onsets, quotes, support)

    def test_saved_state_must_reconcile_to_raw_quote_marks(self):
        onsets, quotes, support = self.history_fixture()
        onsets['pre_rv_300s_5s_bps'] += 1
        with self.assertRaises(ValueError):
            attach_history_quality(onsets, quotes, support)

    def test_empty_onsets_preserve_quality_schema(self):
        onsets, quotes, support = self.history_fixture()
        actual = attach_history_quality(onsets.iloc[:0], quotes, support)
        self.assertTrue(actual.empty)
        for field in ['raw_pre_rv_300s_5s_bps', 'pre_mark_max_age_sec',
                      'pre_mark_repeat_fraction', 'pre_recorded_updates_300s',
                      'pre_max_observed_silence_sec']:
            self.assertIn(field, actual.columns)


if __name__ == '__main__':
    unittest.main()

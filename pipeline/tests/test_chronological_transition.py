"""Chronological boundary checks independent of empirical return outcomes."""
import unittest

import numpy as np
import pandas as pd

from src.chronological_transition import cycle_schedule, carry_forecasts, join_deployments
from src.chronological_acd import predict
from src.duration_calibration import model_sample
from src.event_episodes import build_episode_tables


CUTOFF = pd.Timestamp('2024-05-06', tz='UTC')
PREVIOUS = CUTOFF - pd.Timedelta(days=14)
SECOND = pd.Timedelta(seconds=1)


def forecast_frames():
    origins = CUTOFF + pd.to_timedelta([-3, -2, 0, 1], unit='s')
    ends = CUTOFF + pd.to_timedelta([-2, 0, 1, 2], unit='s')
    path = pd.DataFrame({'origin_quote_id': [10, 11, 12, 13],
        'origin_at': origins, 'end_at': ends, 'event_observed': True})
    new = pd.DataFrame({'origin_quote_id': [10, 11, 12, 13], 'psi': [2., 3., 4., 5.]})
    old = pd.DataFrame({'origin_quote_id': [10, 11], 'psi': [20., 30.]})
    return path, new, old


def carried(path=None, new=None, old=None):
    default_path, default_new, default_old = forecast_frames()
    return carry_forecasts(default_path if path is None else path,
        default_new if new is None else new, default_old if old is None else old,
        cutoff=CUTOFF, previous_cutoff=PREVIOUS)


def deployment_frames(candidates=(-1, 2), *, unknown=(), quiet=()):
    offset = np.arange(-8, 10)
    times = CUTOFF + pd.to_timedelta(offset, unit='s')
    opportunities = pd.DataFrame({'bin_start': times, 'available_at': times + SECOND,
        'support_id': 0, 'recorded_update_count': 1, 'completed_residual_count': 1,
        'flow_status': 2, 'seasonal_ready': True, 'reason': 'eligible', 'period': 'development'})
    for value in unknown:
        mask = offset == value
        opportunities.loc[mask, 'seasonal_ready'] = False
        opportunities.loc[mask, 'reason'] = 'seasonal_unavailable'
    for value in quiet:
        mask = offset == value
        opportunities.loc[mask, ['recorded_update_count', 'completed_residual_count']] = 0
        opportunities.loc[mask, 'flow_status'] = 1
        opportunities.loc[mask, 'reason'] = 'quiet_no_duration'
    scored = opportunities.loc[opportunities.reason.eq('eligible'),
        ['bin_start', 'available_at', 'support_id', 'period']].copy()
    scored['candidate_event'] = scored.bin_start.isin(CUTOFF + pd.to_timedelta(candidates, unit='s'))
    scored['pre_rv_300s_5s_bps'] = 1.
    scored['pre_quote_ofi_300s'] = -2.
    return tuple(f.loc[f.bin_start.lt(CUTOFF)].reset_index(drop=True) for f in [opportunities, scored]) + tuple(
        f.loc[f.bin_start.ge(CUTOFF)].reset_index(drop=True) for f in [opportunities, scored])


def joined(*args):
    frames = deployment_frames() if not args else args
    opportunities, scored = join_deployments(*frames, cutoff=CUTOFF)
    return {'opportunities': opportunities, 'scored': scored}


class CycleScheduleTests(unittest.TestCase):
    def test_fifty_six_day_history_and_fourteen_day_cutoffs(self):
        schedule = cycle_schedule(CUTOFF, CUTOFF + pd.Timedelta(days=42))
        self.assertEqual(list(schedule.cycle_id), [0, 1, 2])
        self.assertEqual(list(schedule.parameter_cutoff), [CUTOFF + pd.Timedelta(days=14*j) for j in range(3)])
        self.assertTrue((schedule.parameter_cutoff - schedule.training_start).eq(pd.Timedelta(days=56)).all())
        self.assertTrue((schedule.deployment_end - schedule.parameter_cutoff).eq(pd.Timedelta(days=14)).all())
        self.assertTrue(schedule.deployment_end.iloc[:-1].reset_index(drop=True).equals(
            schedule.parameter_cutoff.iloc[1:].reset_index(drop=True)))

    def test_partial_final_cycle_stops_at_requested_end(self):
        stop = CUTOFF + pd.Timedelta(days=17)
        schedule = cycle_schedule(CUTOFF, stop)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule.deployment_end.iloc[-1], stop)
        self.assertEqual(schedule.parameter_cutoff.iloc[-1], CUTOFF + pd.Timedelta(days=14))

    def test_exact_end_does_not_create_zero_length_cycle(self):
        schedule = cycle_schedule(CUTOFF, CUTOFF + pd.Timedelta(days=14))
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule.deployment_end.iloc[0], CUTOFF + pd.Timedelta(days=14))

    def test_invalid_or_ambiguous_clock_boundaries_rejected(self):
        for first, stop in [(CUTOFF, CUTOFF), (CUTOFF, CUTOFF - SECOND),
                            (CUTOFF.tz_localize(None), CUTOFF + pd.Timedelta(days=14)),
                            (CUTOFF, (CUTOFF + pd.Timedelta(days=14)).tz_localize(None)),
                            (CUTOFF + SECOND, CUTOFF + pd.Timedelta(days=14)),
                            (CUTOFF, CUTOFF + pd.Timedelta(days=14, seconds=1)),
                            (pd.NaT, CUTOFF + pd.Timedelta(days=14))]:
            with self.subTest(first=first, stop=stop), self.assertRaises(ValueError):
                cycle_schedule(first, stop)


class OriginForecastCarryTests(unittest.TestCase):
    def test_completion_exactly_at_cutoff_keeps_origin_forecast(self):
        output = carried().set_index('origin_quote_id')
        self.assertEqual(output.loc[11, 'forecast_role'], 'carried_origin_forecast')
        self.assertEqual(output.loc[11, 'psi_for_residual'], 30.)
        self.assertEqual(output.loc[11, 'psi_refiltered'], 3.)
        self.assertEqual(output.loc[11, 'residual_parameter_cutoff'], PREVIOUS)

    def test_origin_exactly_at_cutoff_uses_new_forecast(self):
        output = carried().set_index('origin_quote_id')
        self.assertEqual(output.loc[12, 'forecast_role'], 'new_origin_forecast')
        self.assertEqual(output.loc[12, 'psi_for_residual'], 4.)
        self.assertEqual(output.loc[12, 'residual_parameter_cutoff'], CUTOFF)

    def test_historical_calibration_uses_refiltered_new_parameters(self):
        output = carried().set_index('origin_quote_id')
        self.assertEqual(output.loc[10, 'forecast_role'], 'calibration_refilter')
        self.assertEqual(output.loc[10, 'psi_for_residual'], 2.)
        self.assertEqual(output.loc[10, 'residual_parameter_cutoff'], CUTOFF)

    def test_censored_terminal_crossing_cutoff_also_keeps_origin_forecast(self):
        path, new, old = forecast_frames()
        path.loc[1, 'event_observed'] = False
        path.loc[1, 'end_at'] = CUTOFF + SECOND
        output = carried(path, new, old).set_index('origin_quote_id')
        self.assertEqual(output.loc[11, 'psi_for_residual'], 30.)
        self.assertEqual(output.loc[11, 'forecast_role'], 'carried_origin_forecast')

    def test_carry_never_replaces_refiltered_state_for_later_origins(self):
        _, new, old = forecast_frames()
        old.loc[old.origin_quote_id.eq(11), 'psi'] = 1.e12
        output = carried(old=old)
        np.testing.assert_array_equal(output.psi_refiltered, new.psi)
        np.testing.assert_array_equal(output.loc[output.origin_quote_id.ge(12), 'psi_for_residual'], [4., 5.])

    def test_old_lookup_is_by_identity_and_not_row_position(self):
        path, new, old = forecast_frames()
        old = old.iloc[::-1].reset_index(drop=True)
        output = carried(path, new, old).set_index('origin_quote_id')
        self.assertEqual(output.loc[11, 'psi_for_residual'], 30.)

    def test_future_old_forecast_cannot_replace_new_origin(self):
        old = forecast_frames()[2]
        old = pd.concat([old, pd.DataFrame({'origin_quote_id': [12, 13], 'psi': [1000., 2000.]})], ignore_index=True)
        output = carried(old=old).set_index('origin_quote_id')
        np.testing.assert_array_equal(output.loc[[12, 13], 'psi_for_residual'], [4., 5.])

    def test_carry_requires_previous_deployment_origin(self):
        path, new, old = forecast_frames()
        path, new = path.iloc[[1]].copy(), new.iloc[[1]].copy()
        path.loc[1, 'origin_at'] = PREVIOUS - SECOND
        with self.assertRaises(ValueError):
            carried(path, new, old)

    def test_missing_carried_identity_is_rejected(self):
        old = forecast_frames()[2].iloc[[0]].copy()
        with self.assertRaises(ValueError):
            carried(old=old)

    def test_duplicate_previous_forecasts_rejected_even_when_values_agree(self):
        old = forecast_frames()[2]
        with self.assertRaises(ValueError):
            carried(old=pd.concat([old, old.iloc[[1]]], ignore_index=True))

    def test_new_forecast_identity_must_match_path(self):
        path, new, old = forecast_frames()
        for invalid in [new.iloc[::-1].reset_index(drop=True), new.iloc[:-1].copy(),
                        pd.concat([new, new.iloc[[0]]], ignore_index=True),
                        new.assign(origin_quote_id=[10, 11, 12, 99])]:
            with self.subTest(ids=invalid.origin_quote_id.to_list()), self.assertRaises(ValueError):
                carried(path, invalid, old)

    def test_nonpositive_nonfinite_forecasts_rejected(self):
        for label in ['new', 'old']:
            for bad in [0., -1., np.inf, -np.inf, np.nan]:
                path, new, old = forecast_frames()
                target = new if label == 'new' else old
                target.loc[1, 'psi'] = bad
                with self.subTest(source=label, value=bad), self.assertRaises(ValueError):
                    carried(path, new, old)

    def test_input_frames_are_preserved(self):
        frames = forecast_frames()
        originals = [frame.copy(deep=True) for frame in frames]
        carried(*frames)
        for actual, expected in zip(frames, originals):
            pd.testing.assert_frame_equal(actual, expected)

    def test_previous_cutoff_must_precede_current_cutoff(self):
        for invalid in [CUTOFF, CUTOFF + SECOND]:
            with self.subTest(previous=invalid), self.assertRaises(ValueError):
                carry_forecasts(*forecast_frames(), cutoff=CUTOFF, previous_cutoff=invalid)


class HistoricalCensoringTests(unittest.TestCase):
    @staticmethod
    def source(end, observed=True, code=0):
        origin = CUTOFF - SECOND
        return pd.DataFrame({'origin_quote_id': [1], 'component_id': [0],
            'origin_at': [origin], 'end_at': [end], 'event_observed': [observed],
            'stop_code': [code], 'exposure_ns': [(end-origin).value], 'scale_sec': [2.],
            'normalized_exposure': [(end-origin).total_seconds()/2.]})

    @staticmethod
    def training(frame):
        return model_sample(frame, start_ns=(CUTOFF-pd.Timedelta(days=56)).value, stop_ns=CUTOFF.value)

    def test_future_completion_time_cannot_change_training_exposure(self):
        early = self.training(self.source(CUTOFF + SECOND))
        late = self.training(self.source(CUTOFF + pd.Timedelta(days=3)))
        pd.testing.assert_frame_equal(early, late)
        self.assertFalse(early.event_observed.iloc[0])
        self.assertEqual(early.exposure_ns.iloc[0], SECOND.value)
        self.assertEqual(early.normalized_exposure.iloc[0], .5)

    def test_event_at_cutoff_is_censored_in_parameter_training(self):
        frame = self.training(self.source(CUTOFF))
        self.assertFalse(frame.event_observed.iloc[0])
        self.assertEqual(frame.end_at.iloc[0], CUTOFF)
        self.assertEqual(frame.stop_code.iloc[0], 8)

    def test_future_structural_cause_cannot_change_training_record(self):
        observed = self.training(self.source(CUTOFF + SECOND, True, 0))
        terminal = self.training(self.source(CUTOFF + SECOND, False, 2))
        pd.testing.assert_frame_equal(observed, terminal)

    def test_forecast_at_origin_cannot_use_its_own_or_later_duration(self):
        frame = forecast_frames()[0]
        frame['normalized_exposure'] = [1., 2., 1., 1.]
        frame['scale_sec'] = 1.
        frame['recurrence_reset'] = [True, False, False, False]
        params = np.array([.1, .2, .5, 1.])
        before = predict(frame, params, weibull=True)
        changed = frame.copy()
        changed.loc[2:, 'normalized_exposure'] = [20., 30.]
        after = predict(changed, params, weibull=True)
        np.testing.assert_array_equal(before.psi.iloc[:3], after.psi.iloc[:3])
        self.assertNotEqual(before.psi.iloc[3], after.psi.iloc[3])
        self.assertAlmostEqual(after.psi.iloc[3], .1 + .2*20. + .5*after.psi.iloc[2])

    def test_next_forecast_uses_refiltered_recurrence_not_carried_residual_forecast(self):
        frame = forecast_frames()[0]
        frame['normalized_exposure'] = [1., 2., 1., 1.]
        frame['scale_sec'] = 1.
        frame['recurrence_reset'] = [True, False, False, False]
        new = predict(frame, np.array([.1, .2, .5, 1.]), weibull=True)
        old = pd.DataFrame({'origin_quote_id': [10, 11], 'psi': [20., 30.]})
        result = carry_forecasts(frame, new, old, cutoff=CUTOFF, previous_cutoff=PREVIOUS)
        self.assertEqual(result.psi_for_residual.iloc[1], 30.)
        expected = .1 + .2*2. + .5*new.psi.iloc[1]
        contaminated = .1 + .2*2. + .5*30.
        self.assertAlmostEqual(result.psi_for_residual.iloc[2], expected)
        self.assertNotEqual(result.psi_for_residual.iloc[2], contaminated)


class DeploymentContinuityTests(unittest.TestCase):
    def test_historical_categorical_storage_preserves_exact_deployed_values(self):
        old_o, old_s, new_o, new_s = deployment_frames()
        for frame in [old_o, old_s]:
            frame['period'] = pd.Categorical(frame.period, categories=['calibration', 'development'])
        result = joined(old_o, old_s, new_o, new_s)
        for key, parent in [('opportunities', old_o), ('scored', old_s)]:
            pd.testing.assert_frame_equal(result[key].iloc[:len(parent)].reset_index(drop=True),
                parent, check_exact=True, check_dtype=False, check_categorical=False)

    def test_join_preserves_every_bin_and_exact_eligibility(self):
        frames = deployment_frames(quiet=(-3, 1), unknown=(6,))
        result = joined(*frames)
        self.assertEqual(len(result['opportunities']), 18)
        self.assertEqual(len(result['scored']), 15)
        pd.testing.assert_series_equal(result['opportunities'].loc[
            result['opportunities'].reason.eq('eligible'), 'bin_start'].reset_index(drop=True),
            result['scored'].bin_start.reset_index(drop=True))
        self.assertTrue(result['opportunities'].period.eq('development').all())

    def test_recalibration_does_not_split_episode_across_cutoff(self):
        result = joined()
        tables = build_episode_tables(result['opportunities'], result['scored'])
        self.assertEqual(len(tables['episodes']), 1)
        episode = tables['episodes'].iloc[0]
        self.assertEqual(episode.candidate_bins, 2)
        self.assertEqual(episode.first_candidate_bin, CUTOFF-SECOND)
        self.assertEqual(episode.last_candidate_bin, CUTOFF+2*SECOND)
        self.assertTrue(episode.onset_confirmed)
        self.assertEqual(episode.termination, 'closed_no_trigger')

    def test_unknown_barrier_still_splits_at_boundary(self):
        result = joined(*deployment_frames(unknown=(0,)))
        episodes = build_episode_tables(result['opportunities'], result['scored'])['episodes']
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.termination.iloc[0], 'censored_unknown')
        self.assertEqual(episodes.observation_stop_at.iloc[0], CUTOFF)
        self.assertFalse(episodes.onset_confirmed.iloc[1])

    def test_support_change_still_splits_at_boundary(self):
        old_o, old_s, new_o, new_s = deployment_frames()
        new_o['support_id'] = 1
        new_s['support_id'] = 1
        result = joined(old_o, old_s, new_o, new_s)
        episodes = build_episode_tables(result['opportunities'], result['scored'])['episodes']
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.termination.iloc[0], 'censored_support_change')
        self.assertFalse(episodes.onset_confirmed.iloc[1])

    def test_five_clear_seconds_rule_does_not_reset_at_cutoff(self):
        for second_candidate, expected_count in [(4, 1), (5, 2)]:
            result = joined(*deployment_frames(candidates=(-1, second_candidate)))
            episodes = build_episode_tables(result['opportunities'], result['scored'])['episodes']
            with self.subTest(second_candidate=second_candidate):
                self.assertEqual(len(episodes), expected_count)
                if expected_count == 2:
                    self.assertTrue(episodes.onset_confirmed.iloc[1])

    def test_extending_sample_preserves_existing_onset_and_detection_time(self):
        old_o, old_s, new_o, new_s = deployment_frames()
        before = build_episode_tables(old_o, old_s)['onsets']
        result = joined(old_o, old_s, new_o, new_s)
        after = build_episode_tables(result['opportunities'], result['scored'])['onsets']
        for column in ['episode_id', 'bin_start', 'available_at', 'onset_confirmed',
                       'preceding_clear_seconds', 'state_sample_eligible']:
            pd.testing.assert_series_equal(before[column], after[column])

    def test_only_true_final_end_censors_open_episode(self):
        result = joined(*deployment_frames(candidates=(-1, 2, 9)))
        episodes = build_episode_tables(result['opportunities'], result['scored'])['episodes']
        self.assertEqual(episodes.termination.to_list(), ['closed_no_trigger', 'censored_sample_end'])
        self.assertEqual(episodes.observation_stop_at.iloc[-1], CUTOFF + 10*SECOND)

    def test_clock_gap_overlap_or_duplicate_rejected(self):
        base = deployment_frames()
        cases = []
        frames = [frame.copy() for frame in base]
        frames[2] = frames[2].iloc[1:].reset_index(drop=True)
        frames[3] = frames[3].iloc[1:].reset_index(drop=True)
        cases.append(frames)
        frames = [frame.copy() for frame in base]
        frames[0] = pd.concat([frames[0], frames[2].iloc[[0]]], ignore_index=True)
        frames[1] = pd.concat([frames[1], frames[3].iloc[[0]]], ignore_index=True)
        cases.append(frames)
        frames = [frame.copy() for frame in base]
        frames[2] = pd.concat([frames[2].iloc[[0]], frames[2]], ignore_index=True)
        cases.append(frames)
        for index, frames in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                joined(*frames)

    def test_scored_rows_cannot_be_missing_extra_or_wrong_support(self):
        base = deployment_frames()
        for case in ['missing', 'duplicate', 'support']:
            frames = [frame.copy() for frame in base]
            if case == 'missing':
                frames[3] = frames[3].iloc[1:].reset_index(drop=True)
            elif case == 'duplicate':
                frames[3] = pd.concat([frames[3], frames[3].iloc[[-1]]], ignore_index=True)
            else:
                frames[3].loc[0, 'support_id'] = 3
            with self.subTest(case=case), self.assertRaises(ValueError):
                joined(*frames)

    def test_calibration_rows_never_enter_deployment_episodes(self):
        for index in range(4):
            frames = [frame.copy() for frame in deployment_frames()]
            frames[index].loc[0, 'period'] = 'calibration'
            with self.subTest(frame=index), self.assertRaises(ValueError):
                joined(*frames)

    def test_invalid_availability_clock_rejected(self):
        frames = [frame.copy() for frame in deployment_frames()]
        frames[2].loc[0, 'available_at'] += SECOND
        with self.assertRaises(ValueError):
            joined(*frames)

    def test_join_does_not_mutate_sources(self):
        frames = deployment_frames()
        originals = [frame.copy(deep=True) for frame in frames]
        joined(*frames)
        for actual, expected in zip(frames, originals):
            pd.testing.assert_frame_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

"""Bounded-memory, resumable version of the verified episode state machine.

Computational chunk boundaries are not observation boundaries. Onsets are emitted
once, as soon as observed; only completed/censored episode summaries can arrive in
a later chunk. ``finish`` alone declares the actual end of the supplied panel.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from src.event_episodes import EPISODE_COLUMNS, GAP, SECOND, TIME_COLUMNS, integer_values, ns


_INTEGER_COLUMNS = ['episode_id', 'support_id', 'candidate_bins', 'span_seconds',
                    'preceding_clear_seconds']
_OPEN_COLUMNS = set(EPISODE_COLUMNS) - {
    'termination', 'closure_known_at', 'observation_stop_at', 'censor_reason'}
_STATE_COLUMNS = {'version', 'next_episode_id', 'next_timestamp_ns',
                  'previous_support_id', 'clear_run', 'open_episode', 'finished'}


def _integers(series):
    try:
        return integer_values(series)
    except (TypeError, OverflowError) as error:
        raise ValueError('Nonmissing finite integer-valued fields required') from error


def _episode_frame(rows):
    """Preserve integer nanosecond precision and typed nullable time columns."""
    frame = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    for name in TIME_COLUMNS:
        frame[name] = pd.to_datetime(pd.array(frame[name], dtype='Int64'), unit='ns', utc=True)
    for name in _INTEGER_COLUMNS:
        frame[name] = frame[name].astype('int64')
    frame['onset_confirmed'] = frame.onset_confirmed.astype(bool)
    for name in ['termination', 'censor_reason']:
        frame[name] = frame[name].astype(str)
    return frame


def _strict_int(value, name, minimum=None, maximum=None):
    if type(value) is not int or (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum):
        raise ValueError('Invalid checkpoint integer: ' + name)
    return value


def _validate_checkpoint(state):
    if not isinstance(state, dict) or set(state) != _STATE_COLUMNS or state['version'] != 1:
        raise ValueError('Unsupported or incomplete episode checkpoint')
    _strict_int(state['version'], 'version', 1, 1)
    next_id = _strict_int(state['next_episode_id'], 'next_episode_id', 0, np.iinfo('int64').max - 1)
    clear = _strict_int(state['clear_run'], 'clear_run', 0, GAP)
    if type(state['finished']) is not bool:
        raise ValueError('Checkpoint finished must be Boolean')
    next_time, previous = state['next_timestamp_ns'], state['previous_support_id']
    if next_time is None:
        if previous is not None or next_id != 0 or clear != 0 or state['open_episode'] is not None:
            raise ValueError('Empty checkpoint has observation state')
    else:
        _strict_int(next_time, 'next_timestamp_ns', np.iinfo('int64').min + 1,
                    np.iinfo('int64').max)
        _strict_int(previous, 'previous_support_id', -1, np.iinfo('int64').max - 1)
        if next_time % SECOND:
            raise ValueError('Checkpoint clock must align to seconds')
    current = state['open_episode']
    if current is not None:
        if not isinstance(current, dict) or set(current) != _OPEN_COLUMNS:
            raise ValueError('Invalid open-episode checkpoint schema')
        if state['finished'] or next_time is None or clear == GAP:
            raise ValueError('Closed or unobserved checkpoint has an open episode')
        for name in _INTEGER_COLUMNS:
            _strict_int(current[name], name, 0)
        for name in ['first_candidate_bin', 'detected_at', 'last_candidate_bin',
                     'last_candidate_available_at']:
            _strict_int(current[name], name, np.iinfo('int64').min + 1,
                        np.iinfo('int64').max)
        first, last = current['first_candidate_bin'], current['last_candidate_bin']
        if (first % SECOND or last % SECOND or first > last or last >= next_time
                or current['detected_at'] != first + SECOND
                or current['last_candidate_available_at'] != last + SECOND
                or current['episode_id'] != next_id - 1
                or current['support_id'] != previous
                or current['candidate_bins'] < 1
                or current['candidate_bins'] > current['span_seconds']
                or current['span_seconds'] != (last - first) // SECOND + 1
                or next_time != last + (clear + 1) * SECOND
                or current['preceding_clear_seconds'] > GAP
                or type(current['onset_confirmed']) is not bool
                or current['onset_confirmed'] != (current['preceding_clear_seconds'] == GAP)):
            raise ValueError('Contradictory open-episode checkpoint')


def _validated_inputs(opportunities, scored):
    """Validate the whole chunk before any stream state can change."""
    required_o = {'bin_start', 'available_at', 'support_id', 'recorded_update_count',
                  'completed_residual_count', 'flow_status', 'seasonal_ready', 'reason'}
    required_s = {'bin_start', 'available_at', 'support_id', 'candidate_event',
                  'pre_rv_300s_5s_bps', 'pre_quote_ofi_300s'}
    if (not isinstance(opportunities, pd.DataFrame) or not isinstance(scored, pd.DataFrame)
            or not opportunities.columns.is_unique or not scored.columns.is_unique
            or not required_o.issubset(opportunities.columns)
            or not required_s.issubset(scored.columns)):
        raise ValueError('Complete, unique opportunity and scored columns required')
    o, s = opportunities.reset_index(drop=True), scored.reset_index(drop=True)
    t, at, st, sat = ns(o.bin_start), ns(o.available_at), ns(s.bin_start), ns(s.available_at)
    if ((t % SECOND != 0).any() or (st % SECOND != 0).any()
            or (t > np.iinfo('int64').max - SECOND).any()
            or (st > np.iinfo('int64').max - SECOND).any()
            or (np.diff(t) != SECOND).any() or (at != t + SECOND).any()
            or (np.diff(st) <= 0).any() or (sat != st + SECOND).any()):
        raise ValueError('Complete one-second clock and unique aligned scored bins required')
    for frame in [o, s]:
        if 'period' in frame and not frame.period.eq('development').all():
            raise ValueError('Only development bins are eligible for prospective episodes')
    sid = _integers(o.support_id)
    count = _integers(o.recorded_update_count)
    completed = _integers(o.completed_residual_count)
    flow = _integers(o.flow_status)
    scored_sid = _integers(s.support_id)
    if not pd.api.types.is_bool_dtype(o.seasonal_ready.dtype) or o.seasonal_ready.isna().any():
        raise ValueError('Seasonal readiness must be nonmissing Boolean')
    if o.reason.isna().any():
        raise ValueError('Every opportunity must have a nonmissing eligibility reason')
    if ((sid < -1).any() or (count < 0).any() or (completed < 0).any()
            or (completed > count).any() or not np.isin(flow, [0, 1, 2, 3, 4]).all()):
        raise ValueError('Invalid support, flow or update counts')
    eligible = o.reason.eq('eligible').to_numpy(dtype=bool)
    ix = np.flatnonzero(eligible)
    if not np.array_equal(t[ix], st) or not np.array_equal(sid[ix], scored_sid):
        raise ValueError('Scored rows must exactly match eligible opportunity rows and support')
    for name, values in [('recorded_update_count', count), ('completed_residual_count', completed)]:
        if name in s and not np.array_equal(_integers(s[name]), values[ix]):
            raise ValueError('Scored counts must match eligible opportunity rows')
    if (eligible & ((sid < 0) | (flow != 2) | (count <= 0) | (count != completed)
                    | ~o.seasonal_ready.to_numpy(dtype=bool))).any():
        raise ValueError('Contradictory eligible active row')
    if s.candidate_event.isna().any() or s.candidate_event.dtype != bool:
        raise ValueError('Candidate flags must be nonmissing booleans')
    for name in ['pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']:
        if (not pd.api.types.is_numeric_dtype(s[name].dtype)
                or pd.api.types.is_bool_dtype(s[name].dtype)):
            raise ValueError('Pre-onset states must be numeric, with missing states permitted')
    # Missing or infinite prior states mark an onset unavailable; they do not
    # exclude a detector candidate. Conversion also validates nullable dtypes.
    history = np.isfinite(s[['pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']].to_numpy(
        dtype='float64', na_value=np.nan)).all(axis=1)
    trigger = np.zeros(len(o), dtype=bool)
    trigger[ix] = s.candidate_event.to_numpy()
    quiet = (sid >= 0) & (flow == 1) & (count == 0) & (completed == 0)
    observation = np.full(len(o), 'unknown', dtype=object)
    observation[quiet], observation[eligible], observation[trigger] = 'clear_quiet', 'clear_active', 'trigger'
    return o, s, t, at, sid, ix, trigger, observation, history


class EpisodeStream:
    """Carry only the unfinished episode and at most five seconds of clear state."""

    def __init__(self, state=None):
        if state is None:
            state = {'version': 1, 'next_episode_id': 0, 'next_timestamp_ns': None,
                     'previous_support_id': None, 'clear_run': 0,
                     'open_episode': None, 'finished': False}
        _validate_checkpoint(state)
        self._state = deepcopy(state)

    def state_dict(self):
        """JSON-native detached checkpoint, with timestamps as integer nanoseconds."""
        return deepcopy(self._state)

    def consume(self, opportunities, scored):
        """Emit this chunk's observations/onsets and any newly finalized episodes."""
        if self._state['finished']:
            raise ValueError('Cannot consume observations after the final panel stop')
        o, s, t, at, sid, ix, trigger, observation, history = _validated_inputs(opportunities, scored)
        state = self.state_dict()
        if len(t) and state['next_timestamp_ns'] is not None and t[0] != state['next_timestamp_ns']:
            raise ValueError('Consecutive chunks required: overlap or missing seconds')
        reason = o.reason.astype(str).to_numpy()
        assignment = np.full(len(o), -1, dtype='int64')
        completed_rows, onset_positions, onset_records = [], [], []
        score_positions = np.full(len(o), -1, dtype='int64')
        score_positions[ix] = np.arange(len(s), dtype='int64')

        def close(kind, stop=None, known=None, why=''):
            current = state['open_episode']
            if current is not None:
                current.update(termination=kind, closure_known_at=known,
                               observation_stop_at=stop, censor_reason=why)
                completed_rows.append(current)
                state['open_episode'] = None

        for i in range(len(o)):
            current_sid = int(sid[i])
            if state['previous_support_id'] is not None and current_sid != state['previous_support_id']:
                close('censored_support_change', stop=int(t[i]), why='support_id_change')
                state['clear_run'] = 0
            if observation[i] == 'unknown':
                close('censored_unknown', stop=int(t[i]), why=reason[i])
                state['clear_run'] = 0
            elif trigger[i]:
                if state['open_episode'] is None:
                    current = {'episode_id': state['next_episode_id'], 'support_id': current_sid,
                               'first_candidate_bin': int(t[i]), 'detected_at': int(at[i]),
                               'last_candidate_bin': int(t[i]), 'last_candidate_available_at': int(at[i]),
                               'candidate_bins': 0, 'span_seconds': 1,
                               'onset_confirmed': state['clear_run'] >= GAP,
                               'preceding_clear_seconds': state['clear_run']}
                    state['next_episode_id'] += 1
                    state['open_episode'] = current
                    onset_positions.append(int(score_positions[i]))
                    onset_records.append((current['episode_id'], current['onset_confirmed'],
                                          current['preceding_clear_seconds']))
                current = state['open_episode']
                current['last_candidate_bin'], current['last_candidate_available_at'] = int(t[i]), int(at[i])
                current['candidate_bins'] += 1
                current['span_seconds'] = (int(t[i]) - current['first_candidate_bin']) // SECOND + 1
                assignment[i] = current['episode_id']
                state['clear_run'] = 0
            else:
                state['clear_run'] = min(state['clear_run'] + 1, GAP)
                if state['open_episode'] is not None and state['clear_run'] == GAP:
                    close('closed_no_trigger', known=int(at[i]))
            state['previous_support_id'] = current_sid
        if len(t):
            state['next_timestamp_ns'] = int(at[-1])

        clock = o[['bin_start', 'available_at', 'support_id']].copy()
        clock['observation_state'] = observation
        clock['barrier_reason'] = np.where(observation == 'unknown', reason, '')
        clock['episode_id'] = assignment
        membership = clock.loc[trigger, ['bin_start', 'available_at', 'episode_id']].reset_index(drop=True)
        onsets = s.iloc[onset_positions].reset_index(drop=True).copy()
        onsets['episode_id'] = pd.Series([r[0] for r in onset_records], dtype='int64')
        onsets['onset_confirmed'] = pd.Series([r[1] for r in onset_records], dtype=bool)
        onsets['preceding_clear_seconds'] = pd.Series([r[2] for r in onset_records], dtype='int64')
        onsets['state_history_available'] = history[onset_positions]
        onsets['state_sample_eligible'] = onsets.onset_confirmed & onsets.state_history_available
        result = {'clock': clock, 'membership': membership, 'onsets': onsets,
                  'closed_episodes': _episode_frame(completed_rows)}
        _validate_checkpoint(state)
        self._state = state
        return result

    def finish(self):
        """Declare the actual panel stop; return its remaining censored episode."""
        state = self.state_dict()
        rows = []
        if state['open_episode'] is not None:
            current = state['open_episode']
            current.update(termination='censored_sample_end', closure_known_at=None,
                           observation_stop_at=state['next_timestamp_ns'], censor_reason='development_end')
            rows.append(current)
            state['open_episode'] = None
        state['finished'] = True
        result = _episode_frame(rows)
        _validate_checkpoint(state)
        self._state = state
        return result

"""Independent reconstruction of observable stage-05b event fragments.

This verifier deliberately does not import the production episode builder.
It joins candidate identities to the complete opportunity clock, constructs
connected components using barrier prefix sums, and uses strict backward
quote joins for the pre-onset quality calculations.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

SECOND = 1_000_000_000
POLICIES = ('structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s')
QUALITY_COLUMNS = ('pre_mark_max_age_sec', 'pre_mark_repeat_fraction',
                   'pre_recorded_updates_300s', 'pre_max_observed_silence_sec',
                   'raw_pre_rv_300s_5s_bps')
STATES = ('trigger', 'clear_active', 'clear_quiet', 'unknown')
TERMINATIONS = ('closed_no_trigger', 'censored_unknown',
                'censored_support_change', 'censored_sample_end')


def _ns(values, *, missing=False):
    index = pd.DatetimeIndex(values)
    if index.tz is None or (not missing and index.hasnans):
        raise AssertionError('Timezone-aware timestamps required')
    if str(index.tz) != 'UTC':
        raise AssertionError('UTC timestamps required')
    return index.as_unit('ns').asi8


def _compare(actual, expected):
    """Check expected columns including row order, preserving exact identities."""
    if len(actual) != len(expected):
        raise AssertionError('Frame lengths disagree')
    for name in expected.columns:
        if name not in actual:
            raise AssertionError('Missing column: '+name)
        a, e = actual[name], expected[name]
        if pd.api.types.is_datetime64_any_dtype(e.dtype):
            np.testing.assert_array_equal(_ns(a, missing=True), _ns(e, missing=True), err_msg=name)
        elif pd.api.types.is_float_dtype(e.dtype):
            np.testing.assert_allclose(a.to_numpy(dtype=float), e.to_numpy(dtype=float),
                                       rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(a.to_numpy(), e.to_numpy(), err_msg=name)


def _json_compare(actual, expected, location='statistics'):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(location+': missing '+key)
            _json_compare(actual[key], value, location+'.'+key)
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(location+': list length')
        for i, value in enumerate(expected):
            _json_compare(actual[i], value, location+'.'+str(i))
    elif expected is None:
        if actual is not None:
            raise AssertionError(location+': expected null')
    elif isinstance(expected, (bool, int, str)):
        if actual != expected:
            raise AssertionError(location+': '+repr(actual)+' != '+repr(expected))
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, err_msg=location)


def _registered_hashes(run):
    manifest = json.loads((run/'manifest.json').read_text())
    cache = {}

    def digest(path):
        path = Path(path)
        key = str(path.resolve())
        if key not in cache:
            value = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    value.update(chunk)
            cache[key] = value.hexdigest()
        return cache[key]

    checked = 0
    for name, record in manifest['inputs'].items():
        if digest(record['path']) != record['sha256']:
            raise AssertionError('Changed input: '+name)
        checked += 1
    for name, expected in manifest.get('outputs', {}).items():
        if digest(run/name) != expected:
            raise AssertionError('Changed output: '+name)
        checked += 1
    for name, expected in manifest.get('code_sha256', {}).items():
        if digest(run/'code'/name) != expected:
            raise AssertionError('Changed executed source: '+name)
        checked += 1
    for name, expected in manifest.get('ancestor_hashes', {}).items():
        if digest(name) != expected:
            raise AssertionError('Changed ancestor: '+name)
        checked += 1
    for name, expected in manifest.get('previous_files', {}).items():
        if digest(Path(manifest['snapshot_path'])/name) != expected:
            raise AssertionError('Changed before-repair snapshot: '+name)
        checked += 1
    return manifest, checked


def reference_episode_tables(opportunities, scored, start, stop):
    """Construct candidate connected components using vector barrier counts.

    A pair of adjacent candidate nodes is connected exactly when its gap is at
    most five seconds and the intervening clock contains no observation or
    support barrier. This differs from the production chronological state machine.
    """
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    o = opportunities.loc[opportunities.bin_start.ge(start) & opportunities.bin_start.lt(stop)].reset_index(drop=True)
    s = scored.loc[scored.bin_start.ge(start) & scored.bin_start.lt(stop)].reset_index(drop=True)
    times = _ns(o.bin_start)
    np.testing.assert_array_equal(times, np.arange(start.value, stop.value, SECOND))
    np.testing.assert_array_equal(_ns(o.available_at), times+SECOND)
    if not o.period.eq('development').all() or not s.period.eq('development').all():
        raise AssertionError('Development selection or parameter cutoff disagrees')
    if s.bin_start.duplicated().any() or not s.bin_start.is_monotonic_increasing:
        raise AssertionError('Duplicate or unordered scored bins')
    eligible = o.reason.eq('eligible').to_numpy()
    np.testing.assert_array_equal(_ns(s.bin_start), times[eligible])
    np.testing.assert_array_equal(_ns(s.available_at), times[eligible]+SECOND)
    for column in ('support_id', 'recorded_update_count', 'completed_residual_count'):
        np.testing.assert_array_equal(s[column], o.loc[eligible, column], err_msg=column)
    if not pd.api.types.is_bool_dtype(s.candidate_event.dtype):
        raise AssertionError('Candidate selection must be Boolean')
    sid = o.support_id.to_numpy(dtype=np.int64)
    quiet = ((sid >= 0) & o.flow_status.eq(1).to_numpy()
             & o.recorded_update_count.eq(0).to_numpy()
             & o.completed_residual_count.eq(0).to_numpy())
    if (eligible & quiet).any():
        raise AssertionError('An eligible active bin cannot be a quiet bin')
    state = np.full(len(o), 'unknown', dtype=object)
    state[quiet] = 'clear_quiet'
    state[eligible] = np.where(s.candidate_event.to_numpy(), 'trigger', 'clear_active')
    reason = np.where(state == 'unknown', o.reason.astype(str), '')
    unknown = state == 'unknown'
    changed = np.r_[False, sid[1:] != sid[:-1]]
    barrier = unknown | changed
    cumulative_barriers = np.cumsum(barrier, dtype=np.int64)
    candidates = np.flatnonzero(state == 'trigger')
    if len(candidates):
        disconnected = ((np.diff(candidates) > 5)
                        | (cumulative_barriers[candidates[1:]] != cumulative_barriers[candidates[:-1]]))
        identifiers = np.cumsum(np.r_[0, disconnected.astype(np.int64)], dtype=np.int64)
    else:
        identifiers = np.empty(0, dtype=np.int64)
    clock_ids = np.full(len(o), -1, dtype=np.int64)
    clock_ids[candidates] = identifiers
    clock = pd.DataFrame({'bin_start': o.bin_start, 'available_at': o.available_at,
                          'support_id': sid, 'observation_state': state,
                          'barrier_reason': reason, 'episode_id': clock_ids})
    membership = pd.DataFrame({'bin_start': o.bin_start.iloc[candidates].to_numpy(),
                               'available_at': o.available_at.iloc[candidates].to_numpy(),
                               'episode_id': identifiers})
    clear = np.isin(state, ['clear_active', 'clear_quiet'])
    barrier_positions = np.flatnonzero(barrier)
    rows = []
    first_positions = []
    for episode_id in np.unique(identifiers):
        members = candidates[identifiers == episode_id]
        first, last = int(members[0]), int(members[-1])
        first_positions.append(first)
        preceding = 0
        for lag in range(1, 6):
            prior = first-lag
            if prior < 0 or not clear[prior] or sid[prior] != sid[first]:
                break
            preceding += 1
        position = np.searchsorted(barrier_positions, last, side='right')
        next_barrier = int(barrier_positions[position]) if position < len(barrier_positions) else len(o)
        closure, observation_stop, censor_reason = pd.NaT, pd.NaT, ''
        if next_barrier < len(o) and next_barrier <= last+5:
            observation_stop = o.bin_start.iloc[next_barrier]
            if changed[next_barrier]:
                termination, censor_reason = 'censored_support_change', 'support_id_change'
            else:
                termination, censor_reason = 'censored_unknown', str(o.reason.iloc[next_barrier])
        elif last+5 >= len(o):
            termination, observation_stop, censor_reason = 'censored_sample_end', stop, 'development_end'
        else:
            termination, closure = 'closed_no_trigger', pd.Timestamp(times[last]+6*SECOND, tz='UTC')
        rows.append({'episode_id': int(episode_id), 'support_id': int(sid[first]),
                     'first_candidate_bin': o.bin_start.iloc[first],
                     'detected_at': o.available_at.iloc[first],
                     'last_candidate_bin': o.bin_start.iloc[last],
                     'last_candidate_available_at': o.available_at.iloc[last],
                     'candidate_bins': int(len(members)), 'span_seconds': int(last-first+1),
                     'onset_confirmed': preceding == 5, 'preceding_clear_seconds': preceding,
                     'termination': termination, 'closure_known_at': closure,
                     'observation_stop_at': observation_stop, 'censor_reason': censor_reason})
    episode_columns = ['episode_id', 'support_id', 'first_candidate_bin', 'detected_at',
                       'last_candidate_bin', 'last_candidate_available_at', 'candidate_bins',
                       'span_seconds', 'onset_confirmed', 'preceding_clear_seconds',
                       'termination', 'closure_known_at', 'observation_stop_at', 'censor_reason']
    episodes = pd.DataFrame(rows, columns=episode_columns)
    for name in ('first_candidate_bin', 'detected_at', 'last_candidate_bin',
                 'last_candidate_available_at', 'closure_known_at', 'observation_stop_at'):
        episodes[name] = pd.to_datetime(episodes[name], utc=True)
    for name in ('episode_id', 'support_id', 'candidate_bins', 'span_seconds', 'preceding_clear_seconds'):
        episodes[name] = episodes[name].astype('int64')
    episodes['onset_confirmed'] = episodes.onset_confirmed.astype(bool)
    first_times = pd.DatetimeIndex(o.bin_start.iloc[first_positions])
    onsets = s.loc[s.bin_start.isin(first_times)].reset_index(drop=True).copy()
    onsets['episode_id'] = episodes.episode_id.to_numpy()
    onsets['onset_confirmed'] = episodes.onset_confirmed.to_numpy()
    onsets['preceding_clear_seconds'] = episodes.preceding_clear_seconds.to_numpy()
    available = np.isfinite(onsets[['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']].to_numpy()).all(axis=1)
    onsets['state_history_available'] = available
    onsets['state_sample_eligible'] = available & onsets.onset_confirmed.to_numpy()
    return clock, membership, episodes, onsets


def reference_quote_quality(onsets, quotes, support_intervals):
    """Independently inspect each common-support 300-second pre-onset window.

    Backward joins exclude quotes exactly at every mark. The update count and
    silence use [onset-300s, onset), including an update at its left boundary.
    Silence is measured between the two clipped window endpoints and interior
    update timestamps, including the elapsed tail to onset.
    """
    if not len(onsets) or not onsets.state_history_available.any():
        return pd.DataFrame(np.nan, index=np.arange(len(onsets)), columns=QUALITY_COLUMNS)
    q = quotes[['timestamp', 'bid', 'ask']].copy()
    qt = _ns(q.timestamp)
    if not len(q) or (np.diff(qt) <= 0).any():
        raise AssertionError('Raw quotes must be nonempty, unique and time ordered')
    q['quote_timestamp'] = q.timestamp
    q['log_mid'] = np.log((q.bid.to_numpy()+q.ask.to_numpy())/2.)
    support = support_intervals.reset_index(drop=True)
    result = pd.DataFrame(np.nan, index=np.arange(len(onsets)), columns=QUALITY_COLUMNS)
    valid_rows = np.flatnonzero(onsets.state_history_available.to_numpy())
    if not len(valid_rows):
        return result
    onset_times = _ns(onsets.bin_start.iloc[valid_rows])
    mark_times = (onset_times[:, None] + np.arange(-300, 1, 5)[None, :]*SECOND).ravel()
    requests = pd.DataFrame({'timestamp': pd.to_datetime(mark_times, utc=True),
                             'row': np.repeat(valid_rows, 61), 'mark': np.tile(np.arange(61), len(valid_rows))})
    joined = pd.merge_asof(requests.sort_values('timestamp'),
                           q[['timestamp', 'quote_timestamp', 'log_mid']], on='timestamp',
                           direction='backward', allow_exact_matches=False)
    joined = joined.sort_values(['row', 'mark']).reset_index(drop=True)
    if joined.quote_timestamp.isna().any():
        raise AssertionError('Missing raw quote for available pre-event history')
    all_quote_times = _ns(joined.quote_timestamp).reshape(-1, 61)
    all_mark_times = _ns(joined.timestamp).reshape(-1, 61)
    all_log_mid = joined.log_mid.to_numpy().reshape(-1, 61)
    for block, row_number in enumerate(valid_rows):
        row = onsets.iloc[row_number]
        onset = pd.Timestamp(row.bin_start)
        left = onset-pd.Timedelta(seconds=300)
        if row.support_id < 0 or row.support_id >= len(support):
            raise AssertionError('State history references invalid support')
        interval = support.iloc[row.support_id]
        if not (interval.start <= left and onset+pd.Timedelta(seconds=1) <= interval.end_exclusive):
            raise AssertionError('Claimed complete state history crosses support')
        mark_quotes, mark_times = all_quote_times[block], all_mark_times[block]
        if (mark_quotes < pd.Timestamp(interval.start).value).any() or (mark_quotes >= mark_times).any():
            raise AssertionError('Pre-event mark crosses support or includes a boundary quote')
        age = (mark_times-mark_quotes)/SECOND
        lo, hi = np.searchsorted(qt, [left.value, onset.value], side='left')
        interior = qt[lo:hi]
        clipped = np.r_[left.value, interior, onset.value]
        variation = float(np.sqrt(np.square(np.diff(all_log_mid[block])).sum())*10000)
        result.loc[row_number] = [float(age.max()), float(np.mean(mark_quotes[1:] == mark_quotes[:-1])),
                                  int(len(interior)), float(np.max(np.diff(clipped))/SECOND), variation]
        np.testing.assert_allclose(variation, row.pre_rv_300s_5s_bps, rtol=1e-8, atol=1e-8,
                                   err_msg='Raw-quote realized variation')
    return result


def _linear_order_quantile(values, probability):
    values = np.sort(np.asarray(values, dtype=float))
    location = (len(values)-1)*probability
    lo, hi = int(np.floor(location)), int(np.ceil(location))
    return float(values[lo] + (location-lo)*(values[hi]-values[lo]))


def reference_policy_statistics(clock, episodes, onsets):
    """Aggregate independently using Python row identities and sorted values."""
    states = clock.observation_state.to_numpy()
    term = episodes.termination.to_numpy()
    confirmed = set(np.flatnonzero(onsets.onset_confirmed.to_numpy()))
    available = set(np.flatnonzero(onsets.state_history_available.to_numpy()))
    eligible = confirmed & available
    right_censored = set(np.flatnonzero(term != 'closed_no_trigger'))
    output = {'clock_seconds': len(clock),
              'observation_states': {name: int(np.count_nonzero(states == name)) for name in STATES},
              'candidate_bins': int(np.count_nonzero(states == 'trigger')),
              'episode_fragments': len(episodes), 'confirmed_onsets': len(confirmed),
              'uncertain_onsets': len(episodes)-len(confirmed),
              'terminations': {name: int(np.count_nonzero(term == name)) for name in TERMINATIONS},
              'state_history_available': len(available), 'state_history_missing': len(episodes)-len(available),
              'state_sample_eligible': len(eligible), 'confirmed_onsets_missing_history': len(confirmed-available),
              'state_eligible_right_censored': len(eligible & right_censored),
              'multiple_candidate_fragments': sum(int(x) > 1 for x in episodes.candidate_bins),
              'max_candidate_bins': int(max(episodes.candidate_bins, default=0)),
              'max_span_seconds': int(max(episodes.span_seconds, default=0)),
              'quality': {}, 'mark_age_sensitivity': {}}
    eligible_index = sorted(eligible)
    for name in QUALITY_COLUMNS:
        values = onsets.iloc[eligible_index][name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        output['quality'][name] = {'available': len(values),
                                   'median': _linear_order_quantile(values, .5) if len(values) else None,
                                   'p95': _linear_order_quantile(values, .95) if len(values) else None,
                                   'max': float(max(values)) if len(values) else None}
    ages = onsets.iloc[eligible_index].pre_mark_max_age_sec.to_numpy(dtype=float)
    ages = ages[np.isfinite(ages)]
    for cutoff in (15, 30, 60):
        output['mark_age_sensitivity'][str(cutoff)] = {
            'state_sample_with_quality': len(ages),
            'max_mark_age_at_most_cap': sum(float(x) <= cutoff for x in ages),
            'max_mark_age_above_cap': sum(float(x) > cutoff for x in ages)}
    return output


def reference_cross_policy(onsets_by_policy):
    """Exact confirmed-onset sets; no approximate event alignment."""
    sets, available = {}, {}
    for policy, frame in onsets_by_policy.items():
        sets[policy] = set(_ns(frame.loc[frame.onset_confirmed, 'bin_start']))
        available[policy] = set(_ns(frame.loc[frame.state_history_available, 'bin_start']))
    base = sets['structural_only']
    result = {}
    for policy in POLICIES[1:]:
        comparison = sets[policy]
        common, union = base & comparison, base | comparison
        result[policy] = {'structural_confirmed_onsets': len(base),
                           'sensitivity_confirmed_onsets': len(comparison),
                           'exact_onset_intersection': len(common), 'exact_onset_union': len(union),
                           'jaccard': len(common)/len(union) if union else None,
                           'common_onsets_both_states_available': len(common & available['structural_only'] & available[policy]),
                           'common_onsets_structural_state_missing': len(common-available['structural_only']),
                           'common_onsets_sensitivity_state_missing': len(common-available[policy])}
    return result


def reference_daily(membership, onsets, start, stop):
    dates = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(stop)-pd.Timedelta(nanoseconds=1), freq='1D')
    rows = []
    candidate_dates = [stamp.date() for stamp in membership.bin_start]
    onset_dates = [stamp.date() for stamp in onsets.bin_start]
    certified = onsets.onset_confirmed.to_numpy()
    eligible = onsets.state_sample_eligible.to_numpy()
    for date in dates:
        day = date.date()
        ix = [i for i, value in enumerate(onset_dates) if value == day]
        rows.append({'date': date, 'candidate_bins': candidate_dates.count(day),
                     'episode_fragments': len(ix), 'confirmed_onsets': sum(bool(certified[i]) for i in ix),
                     'state_sample_eligible': sum(bool(eligible[i]) for i in ix)})
    return pd.DataFrame(rows, columns=['date', 'candidate_bins', 'episode_fragments', 'confirmed_onsets', 'state_sample_eligible'])


def verify_event_episodes(run):
    """Verify one immutable stage-05b run and return an auditable result."""
    run = Path(run)
    manifest, hash_count = _registered_hashes(run)
    inputs = manifest['inputs']
    parent = json.loads(Path(inputs['parent_statistics']['path']).read_text())
    parent_manifest = json.loads(Path(inputs['parent_manifest']['path']).read_text())
    for policy in POLICIES:
        for key, filename in [('opportunities', 'opportunities.parquet'), ('scored', 'scored_bins.parquet'), ('parameters', 'parameters.json')]:
            expected = parent_manifest['outputs'][policy+'/'+filename]
            if inputs[policy+'_'+key]['sha256'] != expected:
                raise AssertionError('Parent output lineage differs: '+policy+'_'+key)
    for key in ('clean_quotes', 'support_intervals'):
        if inputs[key]['sha256'] != parent_manifest['inputs'][key]['sha256']:
            raise AssertionError('Parent raw-input lineage differs: '+key)
    prior_verification = json.loads(Path(inputs['parent_verification']['path']).read_text())
    if prior_verification.get('status') != 'passed':
        raise AssertionError('Stage-05a independent verification did not pass')
    start, stop = [pd.Timestamp(parent[key]) for key in ('parameter_cutoff', 'development_end_exclusive')]
    summary = json.loads((run/'statistics.json').read_text())
    _json_compare(summary, {'scope': 'reused_development_episode_audit_not_paper_estimates',
                            'parameter_cutoff': start.isoformat(), 'development_end_exclusive': stop.isoformat(),
                            'primary_policy': 'structural_only', 'candidate_gap_seconds': 5,
                            'production_events_approved': False})
    _json_compare(manifest, {'primary_policy': 'structural_only', 'candidate_gap_seconds': 5,
                             'scope': 'reused_development_only', 'production_events_approved': False,
                             'downstream_ready': False})
    quotes = pd.read_parquet(inputs['clean_quotes']['path'], columns=['timestamp', 'bid', 'ask'],
                            filters=[('timestamp', '<', stop)])
    support = pd.read_parquet(inputs['support_intervals']['path'])
    counts = {'hashes_verified': hash_count, 'clock_rows_verified': 0, 'candidate_rows_verified': 0,
              'episodes_verified': 0, 'onset_rows_verified': 0, 'raw_quote_quality_rows_verified': 0,
              'policies_verified': 0, 'daily_table_rows_verified': 0}
    all_onsets = {}
    for policy in POLICIES:
        opportunity = pd.read_parquet(inputs[policy+'_opportunities']['path'])
        scored = pd.read_parquet(inputs[policy+'_scored']['path'])
        clock, membership, episodes, onsets = reference_episode_tables(opportunity, scored, start, stop)
        directory = run/policy
        for name, expected in [('episode_clock', clock), ('candidate_membership', membership), ('episodes', episodes)]:
            _compare(pd.read_parquet(directory/(name+'.parquet')), expected)
        quality = reference_quote_quality(onsets, quotes, support.loc[support.policy.eq(policy)])
        for name in QUALITY_COLUMNS:
            onsets[name] = quality[name].to_numpy()
        _compare(pd.read_parquet(directory/'onsets.parquet'), onsets)
        statistics = json.loads((directory/'statistics.json').read_text())
        if statistics != summary['policies'][policy]:
            raise AssertionError('Policy/root statistics differ: '+policy)
        _json_compare(statistics, reference_policy_statistics(clock, episodes, onsets))
        daily = reference_daily(membership, onsets, start, stop)
        _compare(pd.read_parquet(directory/'daily.parquet'), daily)
        counts['daily_table_rows_verified'] += len(daily)
        all_onsets[policy] = onsets
        counts['clock_rows_verified'] += len(clock)
        counts['candidate_rows_verified'] += len(membership)
        counts['episodes_verified'] += len(episodes)
        counts['onset_rows_verified'] += len(onsets)
        counts['raw_quote_quality_rows_verified'] += int(onsets.state_history_available.sum())
        counts['policies_verified'] += 1
        print('Independently verified '+policy+': '+str(len(episodes))+' episodes/fragments', flush=True)
    _json_compare(summary['cross_policy_exact_onsets'], reference_cross_policy(all_onsets))
    return {'status': 'passed', **counts,
            'methods': 'Candidate connected components from barrier prefix counts; explicit five-bin onset histories; independently calculated closure/censor boundaries; strict backward quote joins and clipped-window silence; exact parent-row conservation and registered hashes.',
            'full_panel_events_approved': False}

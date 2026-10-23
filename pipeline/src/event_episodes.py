"""Observable recorded-quote episodes; no price outcomes or detector fitting."""
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
GAP = 5
POLICIES = ['structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s']
EPISODE_COLUMNS = ['episode_id', 'support_id', 'first_candidate_bin', 'detected_at',
    'last_candidate_bin', 'last_candidate_available_at', 'candidate_bins', 'span_seconds',
    'onset_confirmed', 'preceding_clear_seconds', 'termination', 'closure_known_at',
    'observation_stop_at', 'censor_reason']
TIME_COLUMNS = ['first_candidate_bin', 'detected_at', 'last_candidate_bin',
    'last_candidate_available_at', 'closure_known_at', 'observation_stop_at']
QUALITY_COLUMNS = ['pre_mark_max_age_sec', 'pre_mark_repeat_fraction',
    'pre_recorded_updates_300s', 'pre_max_observed_silence_sec', 'raw_pre_rv_300s_5s_bps']


def ns(values):
    t = pd.DatetimeIndex(values)
    if t.tz is None or t.hasnans:
        raise ValueError('Timezone-aware nonmissing timestamps required')
    return t.as_unit('ns').asi8


def integer_values(series):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError('Integer-valued numeric fields required')
    values = series.to_numpy()
    bounds = np.iinfo('int64')
    if (not np.isfinite(values).all() or (values != np.floor(values)).any()
            or (values < bounds.min).any() or (values >= float(bounds.max)).any()):
        raise ValueError('Nonfinite, fractional or overflowing integer field')
    return values.astype('int64')


def build_episode_tables(opportunities, scored):
    """Scan development bins once. Future termination never selects an onset."""
    o, s = opportunities.reset_index(drop=True).copy(), scored.reset_index(drop=True).copy()
    t, at = ns(o.bin_start), ns(o.available_at)
    st, sat = ns(s.bin_start), ns(s.available_at)
    if ((t % SECOND != 0).any() or (st % SECOND != 0).any()
            or (np.diff(t) != SECOND).any() or (at != t + SECOND).any()
            or (np.diff(st) <= 0).any() or (sat != st + SECOND).any()):
        raise ValueError('Complete one-second clock and unique aligned scored bins required')
    for f in (o, s):
        if 'period' in f and not f.period.eq('development').all():
            raise ValueError('Only development bins are eligible for prospective episodes')
    sid = integer_values(o.support_id)
    count = integer_values(o.recorded_update_count)
    completed = integer_values(o.completed_residual_count)
    flow = integer_values(o.flow_status)
    scored_sid = integer_values(s.support_id)
    if not pd.api.types.is_bool_dtype(o.seasonal_ready.dtype) or o.seasonal_ready.isna().any():
        raise ValueError('Seasonal readiness must be nonmissing boolean')
    if (sid < -1).any() or (count < 0).any() or (completed < 0).any() or (completed > count).any() or not np.isin(flow, [0, 1, 2, 3, 4]).all():
        raise ValueError('Invalid support, flow or update counts')
    eligible = o.reason.eq('eligible').to_numpy()
    ix = np.flatnonzero(eligible)
    if not np.array_equal(t[ix], st) or not np.array_equal(sid[ix], scored_sid):
        raise ValueError('Scored rows must exactly match eligible opportunity rows and support')
    if (eligible & ((sid < 0) | (flow != 2) | (count <= 0) | (count != completed) | ~o.seasonal_ready.to_numpy(dtype=bool))).any():
        raise ValueError('Contradictory eligible active row')
    if s.candidate_event.isna().any() or s.candidate_event.dtype != bool:
        raise ValueError('Candidate flags must be nonmissing booleans')
    trigger = np.zeros(len(o), dtype=bool)
    trigger[ix] = s.candidate_event.to_numpy()
    quiet = (sid >= 0) & (flow == 1) & (count == 0) & (completed == 0)
    state = np.full(len(o), 'unknown', dtype=object)
    state[quiet] = 'clear_quiet'
    state[eligible] = 'clear_active'
    state[trigger] = 'trigger'
    reason = o.reason.astype(str).to_numpy()
    clock = o[['bin_start', 'available_at', 'support_id']].copy()
    clock['observation_state'] = state
    clock['barrier_reason'] = np.where(state == 'unknown', reason, '')
    assignment = np.full(len(o), -1, dtype='int64')
    rows, current = [], None
    clear_run = 0

    def finish(kind, stop=None, known=None, why=''):
        nonlocal current
        if current is not None:
            current.update(termination=kind, closure_known_at=known,
                           observation_stop_at=stop, censor_reason=why)
            rows.append(current)
            current = None

    for i in range(len(o)):
        if i and sid[i] != sid[i-1]:
            finish('censored_support_change', stop=t[i], why='support_id_change')
            clear_run = 0
        if state[i] == 'unknown':
            finish('censored_unknown', stop=t[i], why=reason[i])
            clear_run = 0
        elif trigger[i]:
            if current is None:
                current = {'episode_id': len(rows), 'support_id': int(sid[i]),
                    'first_candidate_bin': t[i], 'detected_at': at[i],
                    'last_candidate_bin': t[i], 'last_candidate_available_at': at[i],
                    'candidate_bins': 0, 'span_seconds': 1,
                    'onset_confirmed': clear_run >= GAP,
                    'preceding_clear_seconds': min(clear_run, GAP)}
            current['last_candidate_bin'] = t[i]
            current['last_candidate_available_at'] = at[i]
            current['candidate_bins'] += 1
            current['span_seconds'] = int((t[i] - current['first_candidate_bin']) // SECOND + 1)
            assignment[i] = current['episode_id']
            clear_run = 0
        else:
            clear_run = min(clear_run + 1, GAP)
            if current is not None and clear_run == GAP:
                finish('closed_no_trigger', known=at[i])
    if len(o):
        finish('censored_sample_end', stop=at[-1], why='development_end')
    ep = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    for col in TIME_COLUMNS:
        # nullable integers preserve nanoseconds when timestamps are missing
        ep[col] = pd.to_datetime(pd.array(ep[col], dtype='Int64'), unit='ns', utc=True)
    for col in ['episode_id', 'support_id', 'candidate_bins', 'span_seconds', 'preceding_clear_seconds']:
        ep[col] = ep[col].astype('int64')
    ep['onset_confirmed'] = ep.onset_confirmed.astype(bool)
    for col in ['termination', 'censor_reason']:
        ep[col] = ep[col].astype(str)
    clock['episode_id'] = assignment
    membership = clock.loc[trigger, ['bin_start', 'available_at', 'episode_id']].reset_index(drop=True)
    onset_ix = pd.Index(s.bin_start).get_indexer(ep.first_candidate_bin)
    if (onset_ix < 0).any():
        raise ValueError('Missing first-candidate score')
    onsets = s.iloc[onset_ix].reset_index(drop=True).copy()
    onsets['episode_id'] = ep.episode_id
    onsets['onset_confirmed'] = ep.onset_confirmed
    onsets['preceding_clear_seconds'] = ep.preceding_clear_seconds
    onsets['state_history_available'] = np.isfinite(onsets[['pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']]).all(axis=1)
    onsets['state_sample_eligible'] = onsets.onset_confirmed & onsets.state_history_available
    if int(ep.candidate_bins.sum()) != int(trigger.sum()) or len(membership) != int(trigger.sum()):
        raise ValueError('Candidate conservation failure')
    return {'clock': clock, 'membership': membership, 'episodes': ep, 'onsets': onsets}


def attach_history_quality(onsets, quotes, support):
    """Pre-onset quality only. Silence is clipped to the stated history window."""
    out = onsets.copy()
    for col in QUALITY_COLUMNS:
        out[col] = np.nan
    if not len(out) or not out.state_history_available.any():
        return out
    q = ns(quotes.timestamp)
    if not len(q) or (np.diff(q) <= 0).any():
        raise ValueError('Unique sorted raw quotes required')
    mid = (quotes.bid.to_numpy() + quotes.ask.to_numpy()) / 2
    if not np.isfinite(mid).all() or (mid <= 0).any():
        raise ValueError('Positive finite midquotes required')
    a, b = ns(support.start), ns(support.end_exclusive)
    for i in np.flatnonzero(out.state_history_available.to_numpy()):
        row = out.iloc[i]
        start, end = row.bin_start.value - 300 * SECOND, row.bin_start.value
        sid = int(row.support_id)
        if sid < 0 or sid >= len(a) or a[sid] > start or b[sid] < end:
            raise ValueError('Claimed state history crosses observation support')
        marks = np.arange(start, end + SECOND, 5 * SECOND, dtype='int64')
        ix = np.searchsorted(q, marks, side='left') - 1
        if (ix < 0).any() or (q[ix] < a[sid]).any():
            raise ValueError('Claimed state history lacks strictly preceding support marks')
        ages = (marks - q[ix]) / SECOND
        lo, hi = np.searchsorted(q, [start, end], side='left')
        edges = np.concatenate(([start], q[lo:hi], [end]))
        rv = np.sqrt(np.sum(np.diff(np.log(mid[ix])) ** 2)) * 10000
        if not np.isclose(rv, row.pre_rv_300s_5s_bps, rtol=1e-8, atol=1e-8):
            raise ValueError('Pre-state realized variation does not match raw marks')
        out.loc[out.index[i], QUALITY_COLUMNS] = [float(ages.max()),
            float(np.mean(ix[1:] == ix[:-1])), int(hi - lo),
            float(np.diff(edges).max() / SECOND), float(rv)]
    return out


def policy_statistics(tables):
    c, e, o = (tables[k] for k in ['clock', 'episodes', 'onsets'])
    number = lambda mask: int(np.asarray(mask).sum())
    summary = {'clock_seconds': len(c),
        'observation_states': {k: number(c.observation_state.eq(k)) for k in ['trigger', 'clear_active', 'clear_quiet', 'unknown']},
        'candidate_bins': int(e.candidate_bins.sum()), 'episode_fragments': len(e),
        'confirmed_onsets': number(e.onset_confirmed), 'uncertain_onsets': number(~e.onset_confirmed),
        'terminations': {k: number(e.termination.eq(k)) for k in ['closed_no_trigger', 'censored_unknown', 'censored_support_change', 'censored_sample_end']},
        'state_history_available': number(o.state_history_available),
        'state_history_missing': number(~o.state_history_available),
        'state_sample_eligible': number(o.state_sample_eligible),
        'confirmed_onsets_missing_history': number(o.onset_confirmed & ~o.state_history_available),
        'state_eligible_right_censored': number(o.state_sample_eligible & ~e.termination.eq('closed_no_trigger')),
        'multiple_candidate_fragments': number(e.candidate_bins.gt(1)),
        'max_candidate_bins': int(e.candidate_bins.max()) if len(e) else 0,
        'max_span_seconds': int(e.span_seconds.max()) if len(e) else 0,
        'quality': {}, 'mark_age_sensitivity': {}}
    for col in QUALITY_COLUMNS:
        values = o.loc[o.state_sample_eligible, col].dropna()
        summary['quality'][col] = {'available': len(values), 'median': float(values.median()) if len(values) else None,
            'p95': float(values.quantile(.95)) if len(values) else None,
            'max': float(values.max()) if len(values) else None}
    for cap in [15, 30, 60]:
        available = o.state_sample_eligible & o.pre_mark_max_age_sec.notna()
        summary['mark_age_sensitivity'][str(cap)] = {
            'state_sample_with_quality': number(available),
            'max_mark_age_at_most_cap': number(available & o.pre_mark_max_age_sec.le(cap)),
            'max_mark_age_above_cap': number(available & o.pre_mark_max_age_sec.gt(cap))}
    return summary


def cross_policy_statistics(onsets):
    base = onsets['structural_only'].loc[lambda f: f.onset_confirmed].set_index('bin_start')
    rows = {}
    for policy in POLICIES[1:]:
        part = onsets[policy].loc[lambda f: f.onset_confirmed].set_index('bin_start')
        common = base.index.intersection(part.index)
        union = base.index.union(part.index)
        both = base.loc[common].state_history_available & part.loc[common].state_history_available
        rows[policy] = {'structural_confirmed_onsets': len(base), 'sensitivity_confirmed_onsets': len(part),
            'exact_onset_intersection': len(common), 'exact_onset_union': len(union),
            'jaccard': len(common) / len(union) if len(union) else None,
            'common_onsets_both_states_available': int(both.sum()),
            'common_onsets_structural_state_missing': int((~base.loc[common].state_history_available).sum()),
            'common_onsets_sensitivity_state_missing': int((~part.loc[common].state_history_available).sum())}
    return rows

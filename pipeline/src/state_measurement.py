"""Past-only recorded quote variation; measurement diagnostics, not inference."""
import math
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
FREQUENCIES = [1, 5, 10, 15, 30, 60]
CAPS = [15, 30, 60]
SAMPLES = ['all_available', 'common_age_15', 'common_age_30', 'common_age_60']
STATUSES = ['available', 'no_endpoint_support', 'window_crosses_support',
            'no_prior_initial_quote', 'initial_quote_outside_support']


def ns(values):
    t = pd.DatetimeIndex(values)
    if t.tz is None or str(t.tz) != 'UTC' or t.hasnans:
        raise ValueError('Nonmissing UTC timestamps required')
    return t.as_unit('ns').asi8


def ints(values):
    if not pd.api.types.is_integer_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype) or values.isna().any():
        raise ValueError('Nonmissing integer identifiers required')
    return values.to_numpy(dtype='int64')


def intervals(support):
    a, b = ns(support.start), ns(support.end_exclusive)
    if not len(a) or (a >= b).any() or (a[1:] < b[:-1]).any():
        raise ValueError('Nonempty ordered nonoverlapping support required')
    return a, b


def build_windows(onsets, support, *, training_start, parameter_cutoff):
    start, cutoff = ns([training_start, parameter_cutoff])
    if start >= cutoff or start % (300 * SECOND) or cutoff % (300 * SECOND):
        raise ValueError('Ordered five-minute calibration boundaries required')
    a, b = intervals(support)
    end = np.arange(start + 300 * SECOND, cutoff + SECOND, 300 * SECOND, dtype='int64')
    sid = np.searchsorted(a, end, side='left') - 1
    safe = np.maximum(sid, 0)
    sid[(sid < 0) | (end > b[safe])] = -1
    cal = pd.DataFrame({'kind': 'calibration', 'end_at': pd.to_datetime(end, utc=True),
        'episode_id': -1, 'onset_confirmed': False, 'support_id': sid,
        'inherited_state_available': False, 'reference_rv_bps': np.nan})
    t, td = ns(onsets.bin_start), ns(onsets.available_at)
    ids, osid = ints(onsets.episode_id), ints(onsets.support_id)
    if (np.diff(t) <= 0).any() or (t % SECOND).any() or (t < cutoff).any() or not np.array_equal(td, t + SECOND):
        raise ValueError('Unique chronological development onset bins required')
    if (ids < 0).any() or len(np.unique(ids)) != len(ids) or ((osid < 0) | (osid >= len(a))).any():
        raise ValueError('Invalid onset identity/support')
    if ((t < a[osid]) | (td > b[osid])).any():
        raise ValueError('Onset bin outside registered support')
    for field in ['onset_confirmed', 'state_history_available']:
        if not pd.api.types.is_bool_dtype(onsets[field].dtype) or onsets[field].isna().any():
            raise ValueError('Nonmissing boolean onset flags required')
    if 'period' in onsets and not onsets.period.eq('development').all():
        raise ValueError('Only development onsets are allowed')
    dev = pd.DataFrame({'kind': 'development', 'end_at': pd.to_datetime(t, utc=True),
        'episode_id': ids, 'onset_confirmed': onsets.onset_confirmed.to_numpy(), 'support_id': osid,
        'inherited_state_available': onsets.state_history_available.to_numpy(),
        'reference_rv_bps': onsets.pre_rv_300s_5s_bps.to_numpy(dtype=float)})
    result = pd.concat([cal, dev], ignore_index=True)
    result.insert(0, 'window_id', np.arange(len(result), dtype='int64'))
    return result


def measure_windows(windows, quotes, support):
    a, b = intervals(support)
    t, q = ns(windows.end_at), ns(quotes.timestamp)
    ids, sid = ints(windows.window_id), ints(windows.support_id)
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or ((sid < -1) | (sid >= len(a))).any() or (t % SECOND).any():
        raise ValueError('Invalid window identity, support or whole-second endpoint')
    if (np.diff(q) <= 0).any():
        raise ValueError('Unique chronological quotes required')
    price = quotes[['bid', 'ask']].to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price[:, 0] <= 0).any() or (price[:, 1] <= price[:, 0]).any():
        raise ValueError('Clean positive bid/ask prices required')
    mid = price.mean(axis=1)
    safe = np.maximum(sid, 0)
    status = np.full(len(t), 'available', dtype=object)
    status[sid < 0] = 'no_endpoint_support'
    status[(status == 'available') & ((t - 300 * SECOND < a[safe]) | (t > b[safe]))] = 'window_crosses_support'
    first_ix = np.searchsorted(q, t - 300 * SECOND, side='left') - 1
    status[(status == 'available') & (first_ix < 0)] = 'no_prior_initial_quote'
    first_q = q[np.maximum(first_ix, 0)] if len(q) else np.zeros(len(t), dtype='int64')
    status[(status == 'available') & (first_q < a[safe])] = 'initial_quote_outside_support'
    good = status == 'available'
    result, common_caps = [], {}
    for freq in FREQUENCIES:
        targets = t[:, None] + np.arange(-300, 1, freq, dtype='int64')[None, :] * SECOND
        ix = np.searchsorted(q, targets, side='left') - 1
        safe_ix = np.maximum(ix, 0)
        stamps = q[safe_ix] if len(q) else np.zeros(targets.shape, dtype='int64')
        ages = (targets - stamps) / SECOND
        marks = mid[safe_ix] if len(q) else np.ones(targets.shape)
        returns = np.diff(np.log(marks), axis=1)
        variance = np.sum(returns ** 2, axis=1) * 1e8
        max_age = ages.max(axis=1)
        out = windows[['window_id', 'kind', 'end_at', 'episode_id', 'onset_confirmed']].copy()
        out['frequency_sec'], out['n_returns'], out['status'] = freq, 300 // freq, status
        for name, values in [('rv_bps2', variance), ('rv_bps', np.sqrt(variance)),
                             ('max_mark_age_sec', max_age),
                             ('repeat_fraction', np.mean(ix[:, 1:] == ix[:, :-1], axis=1)),
                             ('zero_return_fraction', np.mean(returns == 0, axis=1))]:
            out[name] = np.where(good, values, np.nan)
        for name, column in [('first_quote_at', 0), ('last_quote_at', -1)]:
            out[name] = pd.to_datetime(np.where(good, stamps[:, column], np.iinfo('int64').min), utc=True)
        for cap in CAPS:
            own = good & (max_age <= cap)
            if freq == 1:
                common_caps[cap] = own
            out[f'own_age_le_{cap}'] = own
            out[f'common_age_le_{cap}'] = common_caps[cap]
        if freq == 5:
            check = windows.inherited_state_available.to_numpy(dtype=bool)
            if not good[check].all() or not np.allclose(out.loc[check, 'rv_bps'], windows.loc[check, 'reference_rv_bps'], rtol=1e-8, atol=1e-8):
                raise ValueError('Inherited five-second state fails raw-quote reconciliation')
        result.append(out)
    return pd.concat(result, ignore_index=True).sort_values(['window_id', 'frequency_sec']).reset_index(drop=True)


def sample_mask(frame, sample):
    mask = (frame.kind.eq('calibration') | frame.onset_confirmed) & frame.status.eq('available')
    if sample != 'all_available':
        mask &= frame['common_age_le_' + sample.rsplit('_', 1)[1]]
    return mask


def summary_row(group, sample):
    cohort = group.kind.eq('calibration') | group.onset_confirmed
    selected = group.loc[sample_mask(group, sample)]
    own = group.status.eq('available') if sample == 'all_available' else group['own_age_le_' + sample.rsplit('_', 1)[1]]
    row = {'sample': sample, 'windows': len(group), 'cohort_windows': int(cohort.sum()),
        'available_windows': int((cohort & group.status.eq('available')).sum()),
        'selected_windows': len(selected), 'own_grid_age_pass_windows': int((cohort & own).sum()),
        'zero_variance_windows': int(selected.rv_bps2.eq(0).sum())}
    for name, col, method in [('mean_variance_bps2', 'rv_bps2', 'mean'), ('mean_rv_bps', 'rv_bps', 'mean'),
                              ('median_rv_bps', 'rv_bps', 'median'), ('mean_repeat_fraction', 'repeat_fraction', 'mean'),
                              ('mean_zero_return_fraction', 'zero_return_fraction', 'mean'),
                              ('median_max_mark_age_sec', 'max_mark_age_sec', 'median')]:
        row[name] = float(getattr(selected[col], method)()) if len(selected) else np.nan
    row['p95_max_mark_age_sec'] = float(selected.max_mark_age_sec.quantile(.95)) if len(selected) else np.nan
    return row


def summarize_measurements(measurements):
    frame = measurements.copy()
    frame['endpoint_date'] = frame.end_at.dt.normalize()
    frame['endpoint_hour_utc'] = frame.end_at.dt.hour
    tables = {}
    for name, keys in [('summary', ['kind', 'frequency_sec']),
                       ('daily', ['kind', 'endpoint_date', 'frequency_sec']),
                       ('hourly', ['kind', 'endpoint_hour_utc', 'frequency_sec'])]:
        rows = []
        for values, group in frame.groupby(keys, sort=True):
            for sample in SAMPLES:
                rows.append({**dict(zip(keys, values)), **summary_row(group, sample)})
        tables[name] = pd.DataFrame(rows)
    rows = []
    for kind, g in frame.groupby('kind', sort=True):
        for sample in SAMPLES:
            selected = g.loc[sample_mask(g, sample)]
            ref = selected.loc[selected.frequency_sec.eq(5), ['window_id', 'rv_bps2', 'rv_bps']]
            for freq in FREQUENCIES:
                f = selected.loc[selected.frequency_sec.eq(freq), ['window_id', 'rv_bps2', 'rv_bps']]
                pair = f.merge(ref, on='window_id', suffixes=('', '_reference'), validate='one_to_one')
                if len(pair) != len(ref) or len(pair) != len(f):
                    raise ValueError('Common-grid comparison samples differ')
                x, y = pair.rv_bps, pair.rv_bps_reference
                mx = float(pair.rv_bps2.mean()) if len(pair) else np.nan
                my = float(pair.rv_bps2_reference.mean()) if len(pair) else np.nan
                correlation = float(x.rank(method='average').corr(y.rank(method='average'))) if len(pair) > 1 and x.nunique() > 1 and y.nunique() > 1 else np.nan
                rows.append({'kind': kind, 'sample': sample, 'frequency_sec': freq, 'reference_frequency_sec': 5,
                    'paired_windows': len(pair), 'zero_variance_windows': int(pair.rv_bps2.eq(0).sum()),
                    'reference_zero_variance_windows': int(pair.rv_bps2_reference.eq(0).sum()),
                    'mean_variance_ratio_to_5s': mx / my if my > 0 else np.nan,
                    'mean_absolute_rv_difference_bps': float((x - y).abs().mean()) if len(pair) else np.nan,
                    'spearman_rv_to_5s': correlation})
    tables['comparisons'] = pd.DataFrame(rows)
    return tables


def direction_diagnostics(onsets, ticks):
    t, q = ns(onsets.bin_start), ns(ticks.timestamp)
    if (np.diff(q) <= 0).any() or (np.diff(t) <= 0).any():
        raise ValueError('Unique chronological quotes and onsets required')
    rows = []
    for onset, s in zip(onsets.itertuples(index=False), t):
        lo, hi = np.searchsorted(q, [s, s + SECOND], side='left')
        g = ticks.iloc[lo:hi]
        e = g.quote_ofi.to_numpy(dtype=float)
        if not len(g) or not g.exclusion_code.eq(0).all() or not np.isfinite(e).all():
            raise ValueError('Onset requires finite valid tick contributions')
        if hasattr(onset, 'recorded_update_count') and len(g) != onset.recorded_update_count:
            raise ValueError('Onset update count differs from tick reconstruction')
        inherited = float(onset.quote_ofi)
        if not np.isfinite(inherited):
            raise ValueError('Inherited onset imbalance must be finite')
        net, gross = math.fsum(e), math.fsum(abs(float(x)) for x in e)
        neps = len(e) * np.finfo(float).eps
        if neps >= 1:
            raise ValueError('Summation bound unavailable')
        bound = neps / (1 - neps) * gross + abs(float(np.spacing(net)))
        error = inherited - net
        if abs(error) > bound:
            raise ValueError('Inherited onset net differs beyond summation bound')
        rows.append({'episode_id': int(onset.episode_id), 'bin_start': onset.bin_start,
            'onset_confirmed': bool(onset.onset_confirmed), 'tick_contributions': len(e),
            'inherited_net': inherited, 'stable_net': net, 'gross_contributions': gross,
            'cancellation_ratio': abs(net) / gross if gross > 0 else np.nan,
            'aggregation_difference': error, 'summation_error_bound': bound,
            'inherited_direction': int(np.sign(inherited)), 'stable_direction': int(np.sign(net)),
            'direction_agrees': bool(np.sign(inherited) == np.sign(net))})
    columns = ['episode_id', 'bin_start', 'onset_confirmed', 'tick_contributions',
        'inherited_net', 'stable_net', 'gross_contributions', 'cancellation_ratio',
        'aggregation_difference', 'summation_error_bound', 'inherited_direction',
        'stable_direction', 'direction_agrees']
    result = pd.DataFrame(rows, columns=columns)
    for col in ['onset_confirmed', 'direction_agrees']:
        result[col] = result[col].astype(bool)
    for col in ['episode_id', 'tick_contributions', 'inherited_direction', 'stable_direction']:
        result[col] = result[col].astype('int64')
    result['bin_start'] = pd.to_datetime(result.bin_start, utc=True)
    for col in ['inherited_net', 'stable_net', 'gross_contributions', 'cancellation_ratio',
                'aggregation_difference', 'summation_error_bound']:
        result[col] = result[col].astype(float)
    return result


def direction_summary(frame):
    out = {}
    for name, group in [('all_onsets', frame), ('confirmed_onsets', frame.loc[frame.onset_confirmed])]:
        r = group.cancellation_ratio
        values = {'onsets': len(group), 'gross_zero': int(group.gross_contributions.eq(0).sum()),
            'stable_net_zero': int(group.stable_net.eq(0).sum()), 'inherited_net_zero': int(group.inherited_net.eq(0).sum()),
            'direction_disagreements': int((~group.direction_agrees).sum()), 'ratio_available': int(r.notna().sum()),
            'zero_ratio': int(r.eq(0).sum()), 'positive_ratio': int(r.gt(0).sum()),
            'max_absolute_aggregation_difference': float(group.aggregation_difference.abs().max()) if len(group) else None}
        for q, label in [(.01, 'p01'), (.05, 'p05'), (.5, 'median')]:
            values[label + '_cancellation_ratio'] = float(r.quantile(q)) if r.notna().any() else None
        for cap in [.01, .05, .10]:
            values[f'positive_ratio_le_{int(cap*100)}pct'] = int((r.gt(0) & r.le(cap)).sum())
        out[name] = values
    return out

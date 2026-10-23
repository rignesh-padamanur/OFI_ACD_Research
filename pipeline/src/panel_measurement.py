"""Frozen 5/15/30-second quote-state checks on chronological episode onsets.

Each row is a measurement of the same 300-second pre-onset window.  The
one-second grid is used only for a common quote-age screen, never as another
variation specification.  Nothing here selects or redetects episodes.
"""
import numpy as np
import pandas as pd

from src.state_measurement import ns, ints, intervals


SECOND = 1_000_000_000
FREQUENCIES = [5, 15, 30]
CAPS = [15, 30, 60]
SAMPLES = ['all_available', 'common_age_15', 'common_age_30', 'common_age_60']
COHORTS = ['confirmed_onsets', 'state_eligible_onsets']
STATUSES = ['available', 'window_crosses_support', 'no_prior_initial_quote',
            'initial_quote_outside_support']
WINDOW_COLUMNS = ['window_id', 'kind', 'end_at', 'episode_id', 'onset_confirmed',
                  'support_id', 'inherited_state_available', 'reference_rv_bps',
                  'state_sample_eligible']
MEASUREMENT_COLUMNS = ['window_id', 'kind', 'end_at', 'episode_id', 'onset_confirmed',
    'state_sample_eligible', 'frequency_sec', 'n_returns', 'status', 'rv_bps2',
    'rv_bps', 'max_mark_age_sec', 'common_max_mark_age_sec', 'repeat_fraction',
    'zero_return_fraction', 'first_quote_at', 'last_quote_at',
    *[name for cap in CAPS for name in (f'own_age_le_{cap}', f'common_age_le_{cap}')]]
SUMMARY_COLUMNS = ['sample', 'windows', 'cohort_windows', 'available_windows',
    'selected_windows', 'own_grid_age_pass_windows', 'zero_variance_windows',
    'mean_variance_bps2', 'mean_rv_bps', 'median_rv_bps', 'mean_repeat_fraction',
    'mean_zero_return_fraction', 'median_max_mark_age_sec', 'p95_max_mark_age_sec']
COMPARISON_COLUMNS = ['kind', 'cohort', 'sample', 'frequency_sec',
    'reference_frequency_sec', 'paired_windows', 'zero_variance_windows',
    'reference_zero_variance_windows', 'mean_variance_ratio_to_5s',
    'mean_absolute_rv_difference_bps', 'spearman_rv_to_5s']


def _bools(frame, columns):
    for name in columns:
        if not pd.api.types.is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError('Nonmissing boolean flags required: ' + name)


def _validate_windows(windows):
    t, ids, eid, sid = (ns(windows.end_at), ints(windows.window_id),
                        ints(windows.episode_id), ints(windows.support_id))
    if ((np.diff(t) <= 0).any() or (t % SECOND).any() or
            (ids < 0).any() or len(np.unique(ids)) != len(ids) or
            not np.array_equal(ids, eid) or (sid < 0).any()):
        raise ValueError('Unique chronological onset identity and whole-second endpoints required')
    if not windows.kind.eq('chronological_onset').all():
        raise ValueError('Only inherited chronological onset windows are allowed')
    _bools(windows, ['onset_confirmed', 'inherited_state_available', 'state_sample_eligible'])
    expected = windows.onset_confirmed & windows.inherited_state_available
    if not windows.state_sample_eligible.equals(expected):
        raise ValueError('Inherited state eligibility disagrees with confirmation/history flags')
    reference = windows.reference_rv_bps.to_numpy(dtype=float)
    available = windows.inherited_state_available.to_numpy(dtype=bool)
    if not (np.isfinite(reference[available]) & (reference[available] >= 0)).all():
        raise ValueError('Available inherited states require finite nonnegative variation')
    return t, sid


def make_windows(onsets):
    """Retain every inherited onset, using its episode ID as the window ID."""
    t, td = ns(onsets.bin_start), ns(onsets.available_at)
    if not np.array_equal(td, t + SECOND):
        raise ValueError('Onset available_at must equal bin_start plus one second')
    _bools(onsets, ['onset_confirmed', 'state_history_available', 'state_sample_eligible'])
    result = pd.DataFrame({
        'window_id': ints(onsets.episode_id), 'kind': 'chronological_onset',
        'end_at': pd.to_datetime(t, utc=True), 'episode_id': ints(onsets.episode_id),
        'onset_confirmed': onsets.onset_confirmed.to_numpy(dtype=bool),
        'support_id': ints(onsets.support_id),
        'inherited_state_available': onsets.state_history_available.to_numpy(dtype=bool),
        'reference_rv_bps': onsets.pre_rv_300s_5s_bps.to_numpy(dtype=float),
        'state_sample_eligible': onsets.state_sample_eligible.to_numpy(dtype=bool)})
    _validate_windows(result)
    return result[WINDOW_COLUMNS]


def _empty_measurements():
    out = pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    for name in ['window_id', 'episode_id', 'frequency_sec', 'n_returns']:
        out[name] = out[name].astype('int64')
    for name in ['onset_confirmed', 'state_sample_eligible',
                 *[f'{prefix}_age_le_{cap}' for cap in CAPS for prefix in ['own', 'common']]]:
        out[name] = out[name].astype(bool)
    for name in ['end_at', 'first_quote_at', 'last_quote_at']:
        out[name] = pd.to_datetime(out[name], utc=True)
    for name in ['rv_bps2', 'rv_bps', 'max_mark_age_sec', 'common_max_mark_age_sec',
                 'repeat_fraction', 'zero_return_fraction']:
        out[name] = out[name].astype(float)
    return out


def measure_windows(windows, quotes, support, batch_size=512):
    """Measure fixed onsets, retaining three rows even when history is unavailable.

    Quote input must include the latest quote strictly before the earliest
    initial boundary when that quote exists.  It may precede a processing
    chunk; support membership, rather than chunk membership, governs its use.
    """
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    a, b = intervals(support)
    t, sid = _validate_windows(windows)
    if (sid >= len(a)).any():
        raise ValueError('Invalid inherited support identity')
    if ((t < a[sid]) | (t + SECOND > b[sid])).any():
        raise ValueError('Inherited onset bin outside registered support')
    q = ns(quotes.timestamp)
    if (np.diff(q) <= 0).any():
        raise ValueError('Unique chronological quotes required')
    price = quotes[['bid', 'ask']].to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price[:, 0] <= 0).any() or (price[:, 1] <= price[:, 0]).any():
        raise ValueError('Clean positive bid/ask prices required')
    # halve first to avoid overflow when adding finite positive prices
    mid = price[:, 0] / 2 + price[:, 1] / 2
    if not len(t):
        return _empty_measurements()
    result = []
    for left in range(0, len(t), batch_size):
        right = min(left + batch_size, len(t))
        w = windows.iloc[left:right].reset_index(drop=True)
        end, s = t[left:right], sid[left:right]
        status = np.full(len(w), 'available', dtype=object)
        status[end - 300 * SECOND < a[s]] = 'window_crosses_support'
        # one strict-prior lookup per one-second boundary. Coarser grids are
        # exact subsets, ensuring a common endpoint and age-screen population.
        targets = end[:, None] + np.arange(-300, 1, dtype='int64')[None, :] * SECOND
        ix = np.searchsorted(q, targets, side='left') - 1
        stamps = q[np.maximum(ix, 0)] if len(q) else np.zeros(targets.shape, dtype='int64')
        status[(status == 'available') & (ix[:, 0] < 0)] = 'no_prior_initial_quote'
        status[(status == 'available') & (stamps[:, 0] < a[s])] = 'initial_quote_outside_support'
        good = status == 'available'
        ages = (targets - stamps) / SECOND
        common_max_age = ages.max(axis=1)
        marks = mid[np.maximum(ix, 0)] if len(q) else np.ones(targets.shape)
        for freq in FREQUENCIES:
            own_marks, own_ix, own_stamps, own_ages = (x[:, ::freq] for x in [marks, ix, stamps, ages])
            returns = np.diff(np.log(own_marks), axis=1)
            variance = np.sum(returns * returns, axis=1) * 1e8
            max_age = own_ages.max(axis=1)
            out = w[['window_id', 'kind', 'end_at', 'episode_id', 'onset_confirmed', 'state_sample_eligible']].copy()
            out['frequency_sec'], out['n_returns'], out['status'] = freq, 300 // freq, status
            for name, values in [('rv_bps2', variance), ('rv_bps', np.sqrt(variance)),
                ('max_mark_age_sec', max_age), ('common_max_mark_age_sec', common_max_age),
                ('repeat_fraction', np.mean(own_ix[:, 1:] == own_ix[:, :-1], axis=1)),
                ('zero_return_fraction', np.mean(returns == 0, axis=1))]:
                out[name] = np.where(good, values, np.nan)
            for name, column in [('first_quote_at', 0), ('last_quote_at', -1)]:
                out[name] = pd.to_datetime(np.where(good, own_stamps[:, column], np.iinfo('int64').min), utc=True)
            for cap in CAPS:
                out[f'own_age_le_{cap}'] = good & (max_age <= cap)
                out[f'common_age_le_{cap}'] = good & (common_max_age <= cap)
            if freq == 5:
                check = w.inherited_state_available.to_numpy(dtype=bool)
                if not good[check].all() or not np.allclose(out.loc[check, 'rv_bps'], w.loc[check, 'reference_rv_bps'], rtol=1e-9, atol=1e-9):
                    raise ValueError('Inherited five-second state fails raw-quote reconciliation')
            result.append(out[MEASUREMENT_COLUMNS])
    return pd.concat(result, ignore_index=True).sort_values(['window_id', 'frequency_sec']).reset_index(drop=True)


def _cohort_mask(frame, cohort):
    return frame.onset_confirmed if cohort == 'confirmed_onsets' else frame.state_sample_eligible


def _sample_mask(frame, cohort, sample):
    mask = _cohort_mask(frame, cohort) & frame.status.eq('available')
    if sample != 'all_available':
        mask &= frame['common_age_le_' + sample.rsplit('_', 1)[1]]
    return mask


def _summary_row(group, cohort, sample):
    population = _cohort_mask(group, cohort)
    selected = group.loc[_sample_mask(group, cohort, sample)]
    own = group.status.eq('available') if sample == 'all_available' else group['own_age_le_' + sample.rsplit('_', 1)[1]]
    row = {'sample': sample, 'windows': len(group), 'cohort_windows': int(population.sum()),
        'available_windows': int((population & group.status.eq('available')).sum()),
        'selected_windows': len(selected), 'own_grid_age_pass_windows': int((population & own).sum()),
        'zero_variance_windows': int(selected.rv_bps2.eq(0).sum())}
    for name, col, method in [('mean_variance_bps2', 'rv_bps2', 'mean'), ('mean_rv_bps', 'rv_bps', 'mean'),
        ('median_rv_bps', 'rv_bps', 'median'), ('mean_repeat_fraction', 'repeat_fraction', 'mean'),
        ('mean_zero_return_fraction', 'zero_return_fraction', 'mean'), ('median_max_mark_age_sec', 'max_mark_age_sec', 'median')]:
        row[name] = float(getattr(selected[col], method)()) if len(selected) else np.nan
    row['p95_max_mark_age_sec'] = float(selected.max_mark_age_sec.quantile(.95)) if len(selected) else np.nan
    return row


def _validate_measurements(frame):
    ns(frame.end_at)
    ids, eid = ints(frame.window_id), ints(frame.episode_id)
    if (ids < 0).any() or not np.array_equal(ids, eid) or not frame.kind.eq('chronological_onset').all():
        raise ValueError('Invalid chronological measurement identity')
    _bools(frame, ['onset_confirmed', 'state_sample_eligible',
                  *[f'{prefix}_age_le_{cap}' for cap in CAPS for prefix in ['own', 'common']]])
    if (frame.state_sample_eligible & ~frame.onset_confirmed).any():
        raise ValueError('State eligible onsets must be confirmed')
    if not frame.status.isin(STATUSES).all() or not frame.frequency_sec.isin(FREQUENCIES).all():
        raise ValueError('Unexpected measurement status or frequency')
    if frame.duplicated(['window_id', 'frequency_sec']).any():
        raise ValueError('Duplicate measurement identity/frequency')
    if not frame.groupby('window_id', sort=False).size().eq(len(FREQUENCIES)).all():
        raise ValueError('Common-grid comparison samples differ: three rows per onset required')
    identity = ['kind', 'end_at', 'episode_id', 'onset_confirmed', 'state_sample_eligible', 'status',
                'common_max_mark_age_sec', *[f'common_age_le_{cap}' for cap in CAPS]]
    if frame.groupby('window_id', sort=False)[identity].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Common-grid comparison samples differ across frequencies')
    if not frame.n_returns.eq(300 // frame.frequency_sec).all():
        raise ValueError('Incorrect return counts')
    good = frame.status.eq('available')
    for column in ['rv_bps2', 'rv_bps', 'max_mark_age_sec', 'common_max_mark_age_sec',
                   'repeat_fraction', 'zero_return_fraction']:
        values = frame[column].to_numpy(dtype=float)
        if not (np.isfinite(values[good]) & (values[good] >= 0)).all() or not np.isnan(values[~good]).all():
            raise ValueError('Inconsistent measurement availability: ' + column)
    if (frame.loc[good, ['repeat_fraction', 'zero_return_fraction']] > 1).any().any():
        raise ValueError('Invalid fractions')
    if not np.allclose(frame.loc[good, 'rv_bps'] ** 2, frame.loc[good, 'rv_bps2'], rtol=1e-12, atol=1e-12):
        raise ValueError('Variation/variance units disagree')
    for cap in CAPS:
        if not frame[f'own_age_le_{cap}'].equals(good & frame.max_mark_age_sec.le(cap)) or not frame[f'common_age_le_{cap}'].equals(good & frame.common_max_mark_age_sec.le(cap)):
            raise ValueError('Quote-age cap disagrees with measured ages')


def summarize(measurements):
    """Fixed-cohort summaries with explicit inherited and selected denominators."""
    frame = measurements.copy()
    _validate_measurements(frame)
    frame['endpoint_date'] = frame.end_at.dt.normalize()
    frame['endpoint_hour_utc'] = frame.end_at.dt.hour
    tables = {}
    for name, keys in [('summary', ['kind', 'frequency_sec']),
                       ('daily', ['kind', 'endpoint_date', 'frequency_sec']),
                       ('hourly', ['kind', 'endpoint_hour_utc', 'frequency_sec'])]:
        rows = []
        for values, group in frame.groupby(keys, sort=True):
            for cohort in COHORTS:
                for sample in SAMPLES:
                    rows.append({**dict(zip(keys, values)), 'cohort': cohort, **_summary_row(group, cohort, sample)})
        tables[name] = pd.DataFrame(rows, columns=[*keys, 'cohort', *SUMMARY_COLUMNS])
    rows = []
    for kind, group in frame.groupby('kind', sort=True):
        for cohort in COHORTS:
            for sample in SAMPLES:
                selected = group.loc[_sample_mask(group, cohort, sample)]
                reference = selected.loc[selected.frequency_sec.eq(5), ['window_id', 'rv_bps2', 'rv_bps']]
                for freq in FREQUENCIES:
                    own = selected.loc[selected.frequency_sec.eq(freq), ['window_id', 'rv_bps2', 'rv_bps']]
                    pair = own.merge(reference, on='window_id', suffixes=('', '_reference'), validate='one_to_one')
                    if len(pair) != len(reference) or len(pair) != len(own):
                        raise ValueError('Common-grid comparison samples differ')
                    x, y = pair.rv_bps, pair.rv_bps_reference
                    mx, my = (float(pair[c].mean()) if len(pair) else np.nan for c in ['rv_bps2', 'rv_bps2_reference'])
                    correlation = float(x.rank(method='average').corr(y.rank(method='average'))) if len(pair) > 1 and x.nunique() > 1 and y.nunique() > 1 else np.nan
                    rows.append({'kind': kind, 'cohort': cohort, 'sample': sample, 'frequency_sec': freq,
                        'reference_frequency_sec': 5, 'paired_windows': len(pair),
                        'zero_variance_windows': int(pair.rv_bps2.eq(0).sum()),
                        'reference_zero_variance_windows': int(pair.rv_bps2_reference.eq(0).sum()),
                        'mean_variance_ratio_to_5s': mx / my if my > 0 else np.nan,
                        'mean_absolute_rv_difference_bps': float((x - y).abs().mean()) if len(pair) else np.nan,
                        'spearman_rv_to_5s': correlation})
    tables['comparisons'] = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return tables

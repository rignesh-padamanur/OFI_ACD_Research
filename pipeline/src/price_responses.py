"""Left-limit recorded-midquote responses, with explicit timing and attrition."""
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
POLICIES = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}
MARK_STATUSES = ['available', 'beyond_development_end', 'outside_onset_support',
                 'no_prior_quote', 'prior_quote_outside_support', 'stale_under_policy']
MEASURES = ['initial_pips', 'continuation_pips', 'total_pips']
SAMPLE_NAMES = ['horizon_confirmed', 'common_path_confirmed', 'horizon_state', 'common_path_state',
                'horizon_age_15', 'common_path_age_15', 'horizon_age_30', 'common_path_age_30',
                'horizon_age_60', 'common_path_age_60']


def ns(values):
    t = pd.DatetimeIndex(values)
    if t.tz is None or str(t.tz) != 'UTC' or t.hasnans:
        raise ValueError('Nonmissing UTC timestamps required')
    return t.as_unit('ns').asi8


def integer_values(series):
    if not pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype) or series.isna().any():
        raise ValueError('Nonmissing integer identity required')
    return series.to_numpy(dtype='int64')


def _marks(target, sid, q, mid, a, b, stop, age_cap):
    """Each mark is q(target-); equality to support/sample end is allowed."""
    status = np.full(len(target), 'available', dtype=object)
    status[target > stop] = 'beyond_development_end'
    outside = (target < a[sid]) | (target > b[sid])
    status[(status == 'available') & outside] = 'outside_onset_support'
    ix = np.searchsorted(q, target, side='left') - 1
    status[(status == 'available') & (ix < 0)] = 'no_prior_quote'
    safe = np.maximum(ix, 0)
    qt = q[safe] if len(q) else np.zeros(len(target), dtype='int64')
    status[(status == 'available') & (qt < a[sid])] = 'prior_quote_outside_support'
    age_ns = target - qt
    if age_cap is not None:
        status[(status == 'available') & (age_ns > age_cap * SECOND)] = 'stale_under_policy'
    good = status == 'available'
    stamps = np.where(good, qt, np.iinfo('int64').min)
    prices = mid[safe] if len(q) else np.full(len(target), np.nan)
    return {'status': status, 'at': pd.to_datetime(stamps, utc=True),
            'mid': np.where(good, prices, np.nan),
            'age_sec': np.where(good, age_ns / SECOND, np.nan)}


def construct_price_responses(onsets, quotes, support, *, policy, development_end, horizons=None):
    if policy not in POLICIES:
        raise ValueError('Unknown support policy')
    h = np.arange(301, dtype='int64') if horizons is None else np.asarray(horizons)
    if horizons is not None and h.ndim == 1 and any(isinstance(x, (bool, np.bool_)) for x in horizons):
        raise ValueError('Boolean values are not horizon seconds')
    if (h.ndim != 1 or not len(h) or h.dtype.kind not in 'iu' or h[0] != 0
            or (np.diff(h.astype('int64')) <= 0).any() or (h > 300).any()):
        raise ValueError('Horizons must be sorted unique integers starting at zero, at most 300 seconds')
    h = h.astype('int64')
    anchors = onsets.reset_index(drop=True).copy()
    ids = integer_values(anchors.episode_id)
    sid = integer_values(anchors.support_id)
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique nonnegative episode IDs required')
    s, td = ns(anchors.bin_start), ns(anchors.available_at)
    if (s % SECOND).any() or (np.diff(s) <= 0).any() or not np.array_equal(td, s + SECOND):
        raise ValueError('Unique chronological onset bins and right-endpoint detection required')
    if 'period' in anchors and not anchors.period.eq('development').all():
        raise ValueError('Only registered development onsets are allowed')
    for col in ['onset_confirmed', 'state_sample_eligible']:
        if not pd.api.types.is_bool_dtype(anchors[col].dtype) or anchors[col].isna().any():
            raise ValueError('Nonmissing boolean sample flags required')
    if (anchors.state_sample_eligible & ~anchors.onset_confirmed).any():
        raise ValueError('State sample cannot contain an uncertain onset')
    if not pd.api.types.is_numeric_dtype(anchors.quote_ofi.dtype) or pd.api.types.is_bool_dtype(anchors.quote_ofi.dtype):
        raise ValueError('Numeric signed onset imbalance required')
    stop = ns([development_end])[0]
    if (td > stop).any():
        raise ValueError('Onset detection occurs after development end')
    a, b = ns(support.start), ns(support.end_exclusive)
    if (a >= b).any() or (a[1:] < b[:-1]).any() or ((sid < 0) | (sid >= len(a))).any():
        raise ValueError('Valid nonoverlapping support and interval identifiers required')
    if ((s < a[sid]) | (td > b[sid])).any():
        raise ValueError('Onset bin crosses its registered support interval')
    q = ns(quotes.timestamp)
    if (np.diff(q) <= 0).any():
        raise ValueError('Unique chronological quotes required')
    price = quotes[['bid', 'ask']].to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price[:, 0] <= 0).any() or (price[:, 1] <= price[:, 0]).any():
        raise ValueError('Clean positive bid/ask prices required')
    mid = price.mean(axis=1)
    ofi = anchors.quote_ofi.to_numpy(dtype=float)
    finite = np.isfinite(ofi)
    direction = np.where(finite, np.sign(ofi), np.nan)
    directed = finite & (ofi != 0)
    anchors['direction'] = direction
    anchors['direction_status'] = np.select([~finite, ofi == 0, ofi > 0], ['missing', 'zero', 'positive'], default='negative')
    anchors['abs_onset_quote_ofi'] = np.where(finite, np.abs(ofi), np.nan)
    anchors['direction_available'] = directed
    pre = _marks(s, sid, q, mid, a, b, stop, POLICIES[policy])
    det = _marks(td, sid, q, mid, a, b, stop, POLICIES[policy])
    for name, mark in [('pre', pre), ('detection', det)]:
        anchors[name + '_mark_status'] = mark['status']
        anchors[name + '_mark_at'] = mark['at']
        anchors[name + '_mid'] = mark['mid']
        anchors[name + '_age_sec'] = mark['age_sec']
    base = (pre['status'] == 'available') & (det['status'] == 'available')
    anchors['baseline_quote_available'] = base
    anchors['raw_initial_pips'] = (det['mid'] - pre['mid']) * 10000
    anchors['initial_pips'] = np.where(directed, direction * anchors.raw_initial_pips, np.nan)
    anchors['confirmed_direction_cohort'] = anchors.onset_confirmed & directed
    anchors['state_direction_cohort'] = anchors.state_sample_eligible & directed
    n, width = len(anchors), len(h)
    repeat = lambda x: np.repeat(np.asarray(x), width)
    target = (td[:, None] + h[None, :] * SECOND).ravel()
    hs = np.tile(h, n)
    rsid = repeat(sid)
    end = _marks(target, rsid, q, mid, a, b, stop, POLICIES[policy])
    pre_good = repeat(pre['status'] == 'available')
    det_good = repeat(det['status'] == 'available')
    end_good = end['status'] == 'available'
    direction_long, directed_long = repeat(direction), repeat(directed)
    paths = pd.DataFrame({'episode_id': repeat(ids), 'bin_start': pd.to_datetime(repeat(s), utc=True),
        'detected_at': pd.to_datetime(repeat(td), utc=True), 'horizon_sec': hs,
        'target_at': pd.to_datetime(target, utc=True), 'support_id': rsid,
        'onset_confirmed': repeat(anchors.onset_confirmed), 'state_sample_eligible': repeat(anchors.state_sample_eligible),
        'direction_available': directed_long, 'direction': direction_long,
        'endpoint_status': end['status'], 'endpoint_quote_at': end['at'],
        'endpoint_mid': end['mid'], 'endpoint_age_sec': end['age_sec'],
        'pre_age_sec': repeat(pre['age_sec']), 'detection_age_sec': repeat(det['age_sec']),
        'raw_initial_pips': repeat(anchors.raw_initial_pips), 'initial_pips': repeat(anchors.initial_pips),
        'raw_continuation_pips': (end['mid'] - repeat(det['mid'])) * 10000,
        'raw_total_pips': (end['mid'] - repeat(pre['mid'])) * 10000,
        'continuation_quote_available': det_good & end_good,
        'total_quote_available': pre_good & end_good,
        'triplet_quote_available': pre_good & det_good & end_good})
    for col in ['continuation_pips', 'total_pips']:
        paths[col] = np.where(directed_long, direction_long * paths['raw_' + col], np.nan)
    paths['endpoint_reuses_detection_quote'] = det_good & end_good & (pd.DatetimeIndex(end['at']).asi8 == repeat(pd.DatetimeIndex(det['at']).asi8))
    # later detections, not future episode prices, provide a descriptive overlap
    # flag. It never enters onset eligibility or direction.
    next_detection = np.r_[td[1:], np.iinfo('int64').max] if n else np.empty(0, dtype='int64')
    overlap = pd.array(repeat(next_detection) <= target, dtype='boolean')
    overlap[target > stop] = pd.NA
    paths['later_observed_episode_by_horizon'] = overlap
    paths['common_path_available'] = repeat(paths.triplet_quote_available.to_numpy().reshape(n, width).all(axis=1))
    for cap in [15, 30, 60]:
        within = (paths.triplet_quote_available & paths.pre_age_sec.le(cap)
                  & paths.detection_age_sec.le(cap) & paths.endpoint_age_sec.le(cap)).to_numpy()
        paths['triplet_age_le_' + str(cap)] = within
        paths['common_path_age_le_' + str(cap)] = repeat(within.reshape(n, width).all(axis=1))
    joint = paths.triplet_quote_available & paths.direction_available
    if not np.allclose(paths.loc[joint, 'total_pips'], paths.loc[joint, 'initial_pips'] + paths.loc[joint, 'continuation_pips'], rtol=1e-10, atol=1e-9):
        raise ValueError('Price-response decomposition failed')
    if len(paths) != n * width:
        raise ValueError('Onset-horizon conservation failed')
    return {'anchors': anchors, 'paths': paths}


def _sample_mask(frame, name):
    mask = frame.onset_confirmed & frame.direction_available & frame.triplet_quote_available
    if name.endswith('_state'):
        mask &= frame.state_sample_eligible
    if name.startswith('common_path'):
        mask &= frame.common_path_available
    if '_age_' in name:
        cap = name.rsplit('_', 1)[1]
        mask &= frame[('common_path_age_le_' if name.startswith('common_path') else 'triplet_age_le_') + cap]
    return mask


def summarize_responses(tables):
    """Descriptive joint samples only; no independence or inference assumed."""
    a, p = tables['anchors'], tables['paths']
    rows = []
    for h, group in p.groupby('horizon_sec', sort=True):
        for sample in SAMPLE_NAMES:
            g = group.loc[_sample_mask(group, sample)]
            row = {'horizon_sec': int(h), 'sample': sample, 'observations': len(g),
                'mean_endpoint_age_sec': float(g.endpoint_age_sec.mean()) if len(g) else None,
                'p95_endpoint_age_sec': float(g.endpoint_age_sec.quantile(.95)) if len(g) else None,
                'reused_detection_quote': int(g.endpoint_reuses_detection_quote.sum()),
                'later_observed_episode': int(g.later_observed_episode_by_horizon.fillna(False).sum())}
            for col in MEASURES:
                row['mean_' + col] = float(g[col].mean()) if len(g) else None
                row['median_' + col] = float(g[col].median()) if len(g) else None
            rows.append(row)
    columns = ['horizon_sec', 'sample', 'observations', 'mean_endpoint_age_sec', 'p95_endpoint_age_sec',
        'reused_detection_quote', 'later_observed_episode', *[x + col for col in MEASURES for x in ['mean_', 'median_']]]
    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        summary['horizon_sec'] = summary.horizon_sec.astype('int64')
    directions = {k: int(a.direction_status.eq(k).sum()) for k in ['positive', 'negative', 'zero', 'missing']}
    magnitudes = a.loc[a.direction_available, 'abs_onset_quote_ofi']
    stats = {'onset_rows': len(a), 'path_rows': len(p), 'confirmed_onsets': int(a.onset_confirmed.sum()),
        'direction_counts': directions, 'confirmed_directed_onsets': int(a.confirmed_direction_cohort.sum()),
        'state_directed_onsets': int(a.state_direction_cohort.sum()),
        'baseline_quotes_available': int(a.baseline_quote_available.sum()),
        'minimum_nonzero_abs_onset_ofi': float(magnitudes.min()) if len(magnitudes) else None,
        'pre_mark_status': {k: int(a.pre_mark_status.eq(k).sum()) for k in MARK_STATUSES},
        'detection_mark_status': {k: int(a.detection_mark_status.eq(k).sum()) for k in MARK_STATUSES},
        'horizons': {}}
    for h, g in p.groupby('horizon_sec', sort=True):
        cohort = g.onset_confirmed & g.direction_available
        joint = cohort & g.triplet_quote_available
        no_pre_cont = cohort & g.continuation_quote_available & ~g.triplet_quote_available
        values = summary.loc[summary.horizon_sec.eq(h)].set_index('sample')
        stats['horizons'][str(int(h))] = {'endpoint_status_all_fragments': {k: int(g.endpoint_status.eq(k).sum()) for k in MARK_STATUSES},
            'confirmed_directed_cohort': int(cohort.sum()), 'joint_available': int(joint.sum()),
            'joint_missing': int((cohort & ~g.triplet_quote_available).sum()),
            'continuation_available_without_premark': int(no_pre_cont.sum()),
            'samples': {sample: {k: None if pd.isna(v) else (int(v) if k in ['observations', 'reused_detection_quote', 'later_observed_episode'] else float(v))
                               for k, v in values.loc[sample].items() if k != 'horizon_sec'} for sample in SAMPLE_NAMES}}
    return summary, stats


def daily_responses(tables, *, development_start, development_end):
    a, p = tables['anchors'], tables['paths']
    days = pd.date_range(pd.Timestamp(development_start).normalize(), pd.Timestamp(development_end).normalize(), inclusive='left', freq='D')
    rows = []
    last = int(p.horizon_sec.max()) if len(p) else 300
    endpoints = p.loc[p.horizon_sec.eq(last)]
    for day in days:
        daily_a = a.loc[a.bin_start.dt.normalize().eq(day)]
        g = endpoints.loc[endpoints.bin_start.dt.normalize().eq(day)]
        g = g.loc[_sample_mask(g, 'horizon_confirmed')]
        rows.append({'date': day, 'onset_rows': len(daily_a),
            'confirmed_directed_onsets': int(daily_a.confirmed_direction_cohort.sum()),
            'last_horizon_sec': last, 'joint_available_last_horizon': len(g),
            'mean_initial_pips': float(g.initial_pips.mean()) if len(g) else np.nan,
            'mean_continuation_pips': float(g.continuation_pips.mean()) if len(g) else np.nan,
            'mean_total_pips': float(g.total_pips.mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)

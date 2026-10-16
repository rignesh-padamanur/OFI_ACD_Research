"""Independent audit of finite-horizon recorded-quote price responses.

This module does not import the production response builder. Price marks are
reconstructed with strict backward ``merge_asof`` joins, preserving the complete
onset-by-horizon grid. Aggregates use explicit groups of reference row identities.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

SECOND = 1_000_000_000
POLICIES = ('structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s')
CAPS = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}


def _ns(values, *, missing=False):
    index = pd.DatetimeIndex(values)
    if index.tz is None or str(index.tz) != 'UTC' or (index.hasnans and not missing):
        raise AssertionError('Nonmissing UTC timestamps required')
    return index.as_unit('ns').asi8


def _compare(actual, expected):
    """Compare every expected column, including masks, identities and row order."""
    if len(actual) != len(expected):
        raise AssertionError('Frame lengths disagree')
    if set(actual.columns) != set(expected.columns):
        raise AssertionError('Unexpected or omitted frame columns: '+repr(set(actual.columns) ^ set(expected.columns)))
    for name in expected.columns:
        if name not in actual:
            raise AssertionError('Missing column: '+name)
        a, e = actual[name], expected[name]
        if pd.api.types.is_datetime64_any_dtype(e.dtype):
            np.testing.assert_array_equal(_ns(a, missing=True), _ns(e, missing=True), err_msg=name)
        elif pd.api.types.is_bool_dtype(e.dtype):
            np.testing.assert_array_equal(a.isna(), e.isna(), err_msg=name+' missingness')
            np.testing.assert_array_equal(a.fillna(False).to_numpy(dtype=bool), e.fillna(False).to_numpy(dtype=bool), err_msg=name)
        elif pd.api.types.is_float_dtype(e.dtype):
            np.testing.assert_allclose(a.to_numpy(dtype=float), e.to_numpy(dtype=float),
                                      rtol=1e-12, atol=1e-9, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(a.to_numpy(), e.to_numpy(), err_msg=name)


def _json_compare(actual, expected, location='statistics', *, exact=False):
    if isinstance(expected, dict):
        if exact and set(actual) != set(expected):
            raise AssertionError(location+': unexpected or omitted keys')
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(location+': missing '+key)
            _json_compare(actual[key], value, location+'.'+key, exact=exact)
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(location+': list length')
        for i, value in enumerate(expected):
            _json_compare(actual[i], value, location+'.'+str(i), exact=exact)
    elif expected is None:
        if actual is not None:
            raise AssertionError(location+': expected null')
    elif isinstance(expected, (bool, int, str)):
        if actual != expected:
            raise AssertionError(location+': '+repr(actual)+' != '+repr(expected))
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9, err_msg=location)


def _registered_hashes(run):
    manifest = json.loads((run/'manifest.json').read_text())
    cache = {}

    def digest(path):
        path = Path(path)
        key = str(path.resolve())
        if key not in cache:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
            cache[key] = h.hexdigest()
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


def _strict_marks(targets, interval_starts, interval_ends, quotes, stop, cap):
    """Join all targets strictly backward before applying interval/age support.

    The quote join is global; interval membership is tested afterwards. Thus a
    missing mark in the original interval cannot be replaced by a later segment.
    A left-limit target exactly at an interval or study end is permitted.
    """
    requests = pd.DataFrame({'target': pd.to_datetime(targets, utc=True),
                             '_order': np.arange(len(targets))})
    q = quotes[['timestamp', 'bid', 'ask']].rename(columns={'timestamp': 'quote_timestamp'}).copy()
    q['mid'] = q[['bid', 'ask']].mean(axis=1)
    if len(q):
        joined = pd.merge_asof(requests.sort_values('target'), q[['quote_timestamp', 'mid']],
                               left_on='target', right_on='quote_timestamp',
                               direction='backward', allow_exact_matches=False)
        joined = joined.sort_values('_order').reset_index(drop=True)
    else:
        joined = requests.copy()
        joined['quote_timestamp'] = pd.Series(pd.NaT, index=joined.index, dtype='datetime64[ns, UTC]')
        joined['mid'] = np.nan
    age = (joined.target-joined.quote_timestamp).dt.total_seconds().to_numpy()
    in_interval = (joined.quote_timestamp.ge(pd.Series(pd.to_datetime(interval_starts, utc=True))).to_numpy()
                   & joined.quote_timestamp.lt(pd.Series(pd.to_datetime(interval_ends, utc=True))).to_numpy())
    target_start_ok = joined.target.ge(pd.Series(pd.to_datetime(interval_starts, utc=True))).to_numpy()
    target_end_ok = joined.target.le(pd.Series(pd.to_datetime(interval_ends, utc=True))).to_numpy()
    sample_ok = joined.target.le(pd.Timestamp(stop)).to_numpy()
    prior_ok = joined.quote_timestamp.notna().to_numpy() & in_interval
    age_ok = np.ones(len(joined), dtype=bool) if cap is None else age <= cap
    return joined, age, target_start_ok, target_end_ok, sample_ok, prior_ok, age_ok


def _reference_core(onsets, quotes, support, stop, policy, horizons):
    """Reconstruct timestamps and economically separate price changes.

    The independent representation is a wide (onset, mark) panel: mark zero is
    pre-onset, mark one is detection, and subsequent marks are future horizons.
    No production function or saved outcome is used in constructing this panel.
    """
    n, k = len(onsets), len(horizons)
    intervals = support.reset_index(drop=True).iloc[onsets.support_id.to_numpy(dtype=int)].reset_index(drop=True)
    starts = _ns(onsets.bin_start)
    detected = _ns(onsets.available_at)
    targets = np.column_stack([starts, detected, detected[:, None] + np.asarray(horizons)[None, :]*SECOND])
    lefts = np.repeat(_ns(intervals.start), k+2)
    rights = np.repeat(_ns(intervals.end_exclusive), k+2)
    parts = _strict_marks(targets.ravel(), lefts, rights, quotes, stop, CAPS[policy])
    joined, age, start_ok, end_ok, sample_ok, prior_ok, age_ok = parts
    available = start_ok & end_ok & sample_ok & prior_ok & age_ok
    mids = joined.mid.to_numpy(dtype=float).copy()
    mids[~available] = np.nan
    mids = mids.reshape(n, k+2)
    ofi = onsets.quote_ofi.to_numpy(dtype=float)
    direction = np.full(n, np.nan)
    finite = np.isfinite(ofi)
    direction[finite] = np.sign(ofi[finite])
    oriented_direction = np.where(finite & (ofi != 0), direction, np.nan)
    displacement = (mids[:, 1]-mids[:, 0])*10000.
    continuation = (mids[:, 2:]-mids[:, 1, None])*10000.
    total = (mids[:, 2:]-mids[:, 0, None])*10000.
    return {'targets': targets, 'intervals': intervals, 'joined': joined,
            'age': age.reshape(n, k+2), 'start_ok': start_ok.reshape(n, k+2),
            'end_ok': end_ok.reshape(n, k+2), 'sample_ok': sample_ok.reshape(n, k+2),
            'prior_ok': prior_ok.reshape(n, k+2), 'age_ok': age_ok.reshape(n, k+2),
            'available': available.reshape(n, k+2), 'mids': mids,
            'direction': direction, 'oriented_direction': oriented_direction,
            'raw_initial': displacement, 'raw_continuation': continuation, 'raw_total': total,
            'initial': displacement*oriented_direction,
            'continuation': continuation*oriented_direction[:, None],
            'total': total*oriented_direction[:, None]}


STATUSES = ('available', 'beyond_development_end', 'outside_onset_support',
            'no_prior_quote', 'prior_quote_outside_support', 'stale_under_policy')
SAMPLES = ('horizon_confirmed', 'common_path_confirmed', 'horizon_state', 'common_path_state',
           'horizon_age_15', 'common_path_age_15', 'horizon_age_30', 'common_path_age_30',
           'horizon_age_60', 'common_path_age_60')
MEASURES = ('initial_pips', 'continuation_pips', 'total_pips')


def reference_response_tables(onsets, quotes, support, *, policy, development_end, horizons):
    core = _reference_core(onsets, quotes, support, development_end, policy, horizons)
    n, width = len(onsets), len(horizons)
    quote_at = pd.DatetimeIndex(core['joined'].quote_timestamp).to_numpy().reshape(n, width+2)
    absent = core['joined'].quote_timestamp.isna().to_numpy().reshape(n, width+2)
    statuses = np.select([~core['sample_ok'], ~core['start_ok'] | ~core['end_ok'],
                          absent, ~core['prior_ok'], ~core['age_ok']],
                         STATUSES[1:], default='available')
    ages = np.where(core['available'], core['age'], np.nan)
    quote_at[~core['available']] = pd.NaT
    a = onsets.reset_index(drop=True).copy()
    signed_flow = a.quote_ofi.to_numpy(dtype=float)
    usable_direction = np.isfinite(signed_flow) & (signed_flow != 0)
    a['direction'] = core['direction']
    a['direction_status'] = ['missing' if not np.isfinite(x) else 'zero' if x == 0
                             else 'positive' if x > 0 else 'negative' for x in signed_flow]
    a['abs_onset_quote_ofi'] = np.where(np.isfinite(signed_flow), abs(signed_flow), np.nan)
    a['direction_available'] = usable_direction
    for mark, column in [('pre', 0), ('detection', 1)]:
        a[mark+'_mark_status'] = statuses[:, column]
        a[mark+'_mark_at'] = pd.to_datetime(quote_at[:, column], utc=True)
        a[mark+'_mid'] = core['mids'][:, column]
        a[mark+'_age_sec'] = ages[:, column]
    a['baseline_quote_available'] = core['available'][:, :2].all(axis=1)
    a['raw_initial_pips'] = core['raw_initial']
    a['initial_pips'] = core['initial']
    a['confirmed_direction_cohort'] = usable_direction & a.onset_confirmed.to_numpy()
    a['state_direction_cohort'] = usable_direction & a.state_sample_eligible.to_numpy()
    identities = pd.MultiIndex.from_product([range(n), horizons], names=['onset_row', 'horizon_sec']).to_frame(index=False)
    p = a.iloc[identities.onset_row.to_numpy()][['episode_id', 'bin_start', 'available_at', 'support_id',
        'onset_confirmed', 'state_sample_eligible', 'direction_available', 'direction',
        'pre_age_sec', 'detection_age_sec', 'raw_initial_pips', 'initial_pips']].reset_index(drop=True)
    p = p.rename(columns={'available_at': 'detected_at'})
    p['horizon_sec'] = identities.horizon_sec.to_numpy(dtype='int64')
    p['target_at'] = pd.to_datetime(core['targets'][:, 2:].ravel(), utc=True)
    p['endpoint_status'] = statuses[:, 2:].ravel()
    p['endpoint_quote_at'] = pd.to_datetime(quote_at[:, 2:].ravel(), utc=True)
    p['endpoint_mid'] = core['mids'][:, 2:].ravel()
    p['endpoint_age_sec'] = ages[:, 2:].ravel()
    p['raw_continuation_pips'] = core['raw_continuation'].ravel()
    p['raw_total_pips'] = core['raw_total'].ravel()
    p['continuation_pips'] = core['continuation'].ravel()
    p['total_pips'] = core['total'].ravel()
    future_available = core['available'][:, 2:]
    p['continuation_quote_available'] = (future_available & core['available'][:, [1]]).ravel()
    p['total_quote_available'] = (future_available & core['available'][:, [0]]).ravel()
    p['triplet_quote_available'] = (future_available & core['available'][:, :2].all(axis=1)[:, None]).ravel()
    detection_stamps = a.detection_mark_at.iloc[identities.onset_row.to_numpy()].reset_index(drop=True)
    p['endpoint_reuses_detection_quote'] = p.continuation_quote_available & p.endpoint_quote_at.eq(detection_stamps)
    # Independent forward join to another observed detection. This does not use
    # episode closing times and allows uncertain fragments as later events.
    later = pd.DataFrame({'next_detected_at': pd.to_datetime(onsets.available_at, utc=True)})
    if n:
        joined = pd.merge_asof(pd.DataFrame({'detected_at': pd.to_datetime(onsets.available_at, utc=True)}),
                               later, left_on='detected_at', right_on='next_detected_at',
                               direction='forward', allow_exact_matches=False)
        future_detection = joined.next_detected_at.iloc[identities.onset_row.to_numpy()].reset_index(drop=True)
        warning = pd.array(future_detection.le(p.target_at), dtype='boolean')
    else:
        warning = pd.array([], dtype='boolean')
    warning[p.target_at.gt(pd.Timestamp(development_end)).to_numpy()] = pd.NA
    p['later_observed_episode_by_horizon'] = warning
    p['common_path_available'] = p.groupby('episode_id', sort=False).triplet_quote_available.transform('all').astype(bool)
    for cap in (15, 30, 60):
        mask = p.triplet_quote_available & p[['pre_age_sec', 'detection_age_sec', 'endpoint_age_sec']].le(cap).all(axis=1)
        p['triplet_age_le_'+str(cap)] = mask
        p['common_path_age_le_'+str(cap)] = mask.groupby(p.episode_id, sort=False).transform('all').astype(bool)
    return {'anchors': a, 'paths': p}


def _quantile(values, probability):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return None
    point = (len(ordered)-1)*probability
    lower, upper = int(np.floor(point)), int(np.ceil(point))
    return ordered[lower] + (point-lower)*(ordered[upper]-ordered[lower])


def _summary_for_rows(frame, rows):
    g = frame.loc[sorted(rows)]
    count = len(rows)
    values = {'observations': count,
              'mean_endpoint_age_sec': float(np.mean(g.endpoint_age_sec.to_numpy())) if count else None,
              'p95_endpoint_age_sec': _quantile(g.endpoint_age_sec, .95),
              'reused_detection_quote': int(np.count_nonzero(g.endpoint_reuses_detection_quote)),
              'later_observed_episode': int(np.count_nonzero(g.later_observed_episode_by_horizon.fillna(False)))}
    for col in MEASURES:
        numbers = g[col].to_numpy(dtype=float)
        values['mean_'+col] = float(np.mean(numbers)) if count else None
        values['median_'+col] = _quantile(numbers, .5)
    return values


def _cohort_indices(frame):
    """Build cohorts by set intersection, independently of production masks."""
    indices = lambda column: set(frame.index[frame[column].to_numpy(dtype=bool)])
    joint = indices('onset_confirmed') & indices('direction_available') & indices('triplet_quote_available')
    state, common = indices('state_sample_eligible'), indices('common_path_available')
    result = {'horizon_confirmed': joint, 'common_path_confirmed': joint & common,
              'horizon_state': joint & state, 'common_path_state': joint & state & common}
    for cap in (15, 30, 60):
        result['horizon_age_'+str(cap)] = joint & indices('triplet_age_le_'+str(cap))
        result['common_path_age_'+str(cap)] = joint & common & indices('common_path_age_le_'+str(cap))
    return result


def reference_summaries(tables, *, development_start, development_end):
    a, p = tables['anchors'], tables['paths']
    nonzero = a.loc[a.direction_available, 'abs_onset_quote_ofi'].tolist()
    status_counts = lambda values: {name: list(values).count(name) for name in STATUSES}
    statistics = {'onset_rows': len(a), 'path_rows': len(p),
                  'confirmed_onsets': int(np.count_nonzero(a.onset_confirmed)),
                  'direction_counts': {name: a.direction_status.tolist().count(name) for name in ('positive','negative','zero','missing')},
                  'confirmed_directed_onsets': int(np.count_nonzero(a.confirmed_direction_cohort)),
                  'state_directed_onsets': int(np.count_nonzero(a.state_direction_cohort)),
                  'baseline_quotes_available': int(np.count_nonzero(a.baseline_quote_available)),
                  'minimum_nonzero_abs_onset_ofi': min(nonzero) if nonzero else None,
                  'pre_mark_status': status_counts(a.pre_mark_status),
                  'detection_mark_status': status_counts(a.detection_mark_status), 'horizons': {}}
    records = []
    for horizon in sorted(p.horizon_sec.unique()):
        g = p.loc[p.horizon_sec.eq(horizon)]
        groups = _cohort_indices(g)
        samples = {name: _summary_for_rows(p, groups[name]) for name in SAMPLES}
        for name in SAMPLES:
            records.append({'horizon_sec': int(horizon), 'sample': name, **samples[name]})
        directed = set(g.index[g.direction_available & g.onset_confirmed])
        triplet = set(g.index[g.triplet_quote_available])
        continuation = set(g.index[g.continuation_quote_available])
        statistics['horizons'][str(int(horizon))] = {
            'endpoint_status_all_fragments': status_counts(g.endpoint_status),
            'confirmed_directed_cohort': len(directed), 'joint_available': len(directed & triplet),
            'joint_missing': len(directed-triplet),
            'continuation_available_without_premark': len((directed & continuation)-triplet), 'samples': samples}
    summary = pd.DataFrame(records, columns=['horizon_sec', 'sample', 'observations', 'mean_endpoint_age_sec',
        'p95_endpoint_age_sec', 'reused_detection_quote', 'later_observed_episode',
        *[prefix+col for col in MEASURES for prefix in ('mean_', 'median_')]])
    daily_rows = []
    last = int(max(p.horizon_sec)) if len(p) else 300
    day = pd.Timestamp(development_start).normalize()
    end = pd.Timestamp(development_end).normalize()
    while day < end:
        next_day = day+pd.Timedelta(days=1)
        same_a = a.loc[a.bin_start.ge(day) & a.bin_start.lt(next_day)]
        endpoint = p.loc[p.horizon_sec.eq(last) & p.bin_start.ge(day) & p.bin_start.lt(next_day)]
        rows = _cohort_indices(endpoint)['horizon_confirmed']
        means = _summary_for_rows(p, rows)
        daily_rows.append({'date': day, 'onset_rows': len(same_a),
            'confirmed_directed_onsets': int(np.count_nonzero(same_a.confirmed_direction_cohort)),
            'last_horizon_sec': last, 'joint_available_last_horizon': len(rows),
            **{'mean_'+col: means['mean_'+col] for col in MEASURES}})
        day = next_day
    daily = pd.DataFrame(daily_rows)
    for column in MEASURES:
        if 'mean_'+column in daily:
            daily['mean_'+column] = daily['mean_'+column].astype(float)
    return summary, statistics, daily


def _check_identities(paths):
    for prefix in ('', 'raw_'):
        complete = paths.triplet_quote_available.to_numpy().copy()
        if not prefix:
            complete &= paths.direction_available.to_numpy()
        j, c, total = [paths.loc[complete, prefix+name].to_numpy(dtype=float) for name in MEASURES]
        if not (np.isfinite(j).all() and np.isfinite(c).all() and np.isfinite(total).all()):
            raise AssertionError('Available price triplet contains nonfinite values')
        np.testing.assert_allclose(total, j+c, rtol=0, atol=1e-9, err_msg='J+C=I '+prefix)
        zero = paths.loc[paths.horizon_sec.eq(0)]
        observable = zero.continuation_quote_available.to_numpy().copy()
        if not prefix:
            observable &= zero.direction_available.to_numpy()
        np.testing.assert_allclose(zero.loc[observable, prefix+'continuation_pips'], 0, rtol=0, atol=1e-9)
        common = zero.triplet_quote_available.to_numpy().copy()
        if not prefix:
            common &= zero.direction_available.to_numpy()
        np.testing.assert_allclose(zero.loc[common, prefix+'total_pips'], zero.loc[common, prefix+'initial_pips'], rtol=0, atol=1e-9)


def verify_price_responses(run):
    """Independently verify one immutable stage-06a development response run."""
    run = Path(run)
    manifest, hash_count = _registered_hashes(run)
    inputs = manifest['inputs']
    parent_manifest = json.loads(Path(inputs['parent_manifest']['path']).read_text())
    parent_statistics = json.loads(Path(inputs['parent_statistics']['path']).read_text())
    parent_verification = json.loads(Path(inputs['parent_verification']['path']).read_text())
    if parent_verification.get('status') != 'passed':
        raise AssertionError('Parent episode verification did not pass')
    for key, filename in (('parent_statistics', 'statistics.json'), ('parent_verification', 'verification.json')):
        if inputs[key]['sha256'] != parent_manifest['outputs'][filename]:
            raise AssertionError('Parent summary or verification lineage differs: '+key)
    for key in ('clean_quotes', 'support_intervals'):
        if inputs[key]['sha256'] != parent_manifest['inputs'][key]['sha256']:
            raise AssertionError('Raw-input lineage differs: '+key)
    for policy in POLICIES:
        for name in ('onsets', 'episodes'):
            if inputs[policy+'_'+name]['sha256'] != parent_manifest['outputs'][policy+'/'+name+'.parquet']:
                raise AssertionError('Parent event-output lineage differs: '+policy+'_'+name)
    horizons = list(range(301))
    _json_compare(manifest, {'primary_policy': 'structural_only', 'horizons_seconds': horizons,
                             'scope': 'reused_development_only', 'production_inference_approved': False,
                             'downstream_ready': False, 'parent_episode_audit': parent_manifest['run_id']})
    start, stop = [pd.Timestamp(parent_statistics[key]) for key in ('parameter_cutoff', 'development_end_exclusive')]
    statistics = json.loads((run/'statistics.json').read_text())
    _json_compare(statistics, {'parameter_cutoff': start.isoformat(), 'development_end_exclusive': stop.isoformat(),
                               'primary_policy': 'structural_only', 'production_inference_approved': False})
    quotes = pd.read_parquet(inputs['clean_quotes']['path'], columns=['timestamp', 'bid', 'ask'],
                            filters=[('timestamp', '<', stop)])
    support = pd.read_parquet(inputs['support_intervals']['path'])
    if not quotes.timestamp.is_monotonic_increasing or quotes.timestamp.duplicated().any():
        raise AssertionError('Clean quote timestamps are not unique and chronological')
    price = quotes[['bid','ask']].to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price[:, 0] <= 0).any() or (price[:, 1] <= price[:, 0]).any():
        raise AssertionError('Invalid clean-quote prices')
    counts = {'hashes_verified': hash_count, 'policies_verified': 0, 'onset_rows_verified': 0,
              'path_rows_verified': 0, 'price_mark_requests_verified': 0,
              'horizon_summary_rows_verified': 0, 'daily_table_rows_verified': 0}
    for policy in POLICIES:
        original = pd.read_parquet(inputs[policy+'_onsets']['path'])
        episodes = pd.read_parquet(inputs[policy+'_episodes']['path'])
        if len(original) != len(episodes) or original.episode_id.duplicated().any():
            raise AssertionError('Parent onset/episode identity conservation failed')
        for left, right in (('episode_id','episode_id'), ('support_id','support_id'),
                            ('bin_start','first_candidate_bin'), ('available_at','detected_at'),
                            ('onset_confirmed','onset_confirmed')):
            _compare(original[[left]], episodes[[right]].rename(columns={right:left}))
        if not original.period.eq('development').all() or not original.bin_start.ge(start).all() or not original.bin_start.lt(stop).all():
            raise AssertionError('An onset falls outside the frozen development window')
        tables = reference_response_tables(original, quotes, support.loc[support.policy.eq(policy)],
            policy=policy, development_end=stop, horizons=horizons)
        directory = run/policy
        actual_a, actual_p = [pd.read_parquet(directory/(name+'.parquet')) for name in ('anchors','paths')]
        _compare(actual_a, tables['anchors'])
        _compare(actual_p, tables['paths'])
        if len(actual_p) != len(original)*301:
            raise AssertionError('The complete onset-by-horizon grid was not preserved')
        _check_identities(actual_p)
        summary, policy_statistics, daily = reference_summaries(tables, development_start=start, development_end=stop)
        _compare(pd.read_parquet(directory/'horizon_summary.parquet'), summary)
        _compare(pd.read_parquet(directory/'daily.parquet'), daily)
        saved_policy = json.loads((directory/'statistics.json').read_text())
        if saved_policy != statistics['policies'][policy]:
            raise AssertionError('Policy/root statistics differ: '+policy)
        _json_compare(saved_policy, policy_statistics, exact=True)
        # Means must share the exact same available row identities; medians are
        # not additive and are deliberately excluded from this mean identity.
        present = summary.observations.gt(0)
        np.testing.assert_allclose(summary.loc[present, 'mean_total_pips'],
            summary.loc[present, 'mean_initial_pips']+summary.loc[present, 'mean_continuation_pips'],
            rtol=0, atol=1e-9, err_msg='Joint-sample mean decomposition')
        counts['policies_verified'] += 1
        counts['onset_rows_verified'] += len(original)
        counts['path_rows_verified'] += len(actual_p)
        counts['price_mark_requests_verified'] += len(original)*303
        counts['horizon_summary_rows_verified'] += len(summary)
        counts['daily_table_rows_verified'] += len(daily)
        print('Independently verified '+policy+': '+str(len(actual_p))+' onset-horizon rows', flush=True)
    return {'status': 'passed', **counts,
        'methods': 'Strict backward merge_asof raw-quote joins; separate endpoint-pair J/C/I arithmetic; forward joins for later detections; explicit set-intersection cohorts and sorted-value quantiles; every stored anchor/path/summary/daily cell; exact parent-row conservation, left-limit boundaries, identities and registered hashes.',
        'production_inference_approved': False, 'downstream_ready': False}

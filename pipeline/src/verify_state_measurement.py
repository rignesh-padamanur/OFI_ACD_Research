"""Independent raw-quote verification of stage-06b state measurement.

No production state-measurement functions are imported. Full boundary grids
are reconstructed by strict backward joins; saved tables are only comparison
targets after independent raw-data calculations.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

SECOND = 1_000_000_000
FREQUENCIES = (1, 5, 10, 15, 30, 60)


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



def _quote_grid_batches(endpoints, quotes, batch_size=768):
    """Yield independently joined one-second grids with strict left limits.

    Every coarser grid is a subset of this boundary grid. Requests are sorted
    only for the join and then restored to their original window/mark order.
    Batching limits temporary memory without changing quote-boundary semantics.
    """
    ends = pd.DatetimeIndex(endpoints)
    q = pd.DataFrame({'quote_at': quotes.timestamp,
                      'log_mid': np.log((quotes.bid.to_numpy(dtype=float)+quotes.ask.to_numpy(dtype=float))/2.)})
    for left in range(0, len(ends), batch_size):
        right = min(len(ends), left+batch_size)
        boundary = (_ns(ends[left:right])[:, None] + np.arange(-300, 1, dtype='int64')[None, :]*SECOND).ravel()
        requests = pd.DataFrame({'target': pd.to_datetime(boundary, utc=True), '_position': np.arange(len(boundary))})
        if len(q):
            joined = pd.merge_asof(requests.sort_values('target'), q,
                                   left_on='target', right_on='quote_at',
                                   direction='backward', allow_exact_matches=False)
            joined = joined.sort_values('_position')
        else:
            joined = requests.copy()
            joined['quote_at'] = pd.Series(pd.NaT, index=joined.index, dtype='datetime64[ns, UTC]')
            joined['log_mid'] = np.nan
        quote_times = _ns(joined.quote_at, missing=True).reshape(right-left, 301)
        logs = joined.log_mid.to_numpy(dtype=float).reshape(right-left, 301)
        ages = (joined.target-joined.quote_at).dt.total_seconds().to_numpy().reshape(right-left, 301)
        yield left, right, quote_times, logs, ages

CAPS = (15, 30, 60)
SAMPLES = ('all_available', 'common_age_15', 'common_age_30', 'common_age_60')
STATUSES = ('available', 'no_endpoint_support', 'window_crosses_support',
            'no_prior_initial_quote', 'initial_quote_outside_support')


def reference_windows(onsets, support, *, training_start, parameter_cutoff):
    """Construct the fixed calendar and inherited-onset populations afresh."""
    support = support.reset_index(drop=True)
    start, stop = pd.Timestamp(training_start), pd.Timestamp(parameter_cutoff)
    endpoints = pd.date_range(start+pd.Timedelta(seconds=300), stop, freq='300s')
    intervals = support[['start', 'end_exclusive']].copy()
    intervals['support_id'] = np.arange(len(intervals), dtype='int64')
    matched = pd.merge_asof(pd.DataFrame({'end_at': endpoints}), intervals,
                            left_on='end_at', right_on='start', direction='backward',
                            allow_exact_matches=False)
    covered = matched.support_id.notna() & matched.end_at.le(matched.end_exclusive)
    records = []
    for pos, endpoint in enumerate(endpoints):
        records.append({'window_id': pos, 'kind': 'calibration', 'end_at': endpoint,
            'episode_id': -1, 'onset_confirmed': False,
            'support_id': int(matched.support_id.iloc[pos]) if covered.iloc[pos] else -1,
            'inherited_state_available': False, 'reference_rv_bps': np.nan})
    for row in onsets.itertuples(index=False):
        records.append({'window_id': len(records), 'kind': 'development', 'end_at': row.bin_start,
            'episode_id': int(row.episode_id), 'onset_confirmed': bool(row.onset_confirmed),
            'support_id': int(row.support_id), 'inherited_state_available': bool(row.state_history_available),
            'reference_rv_bps': float(row.pre_rv_300s_5s_bps)})
    return pd.DataFrame(records)


def reference_measurements(windows, quotes, support):
    """Reconstruct nested mark grids and every per-window measure."""
    support = support.reset_index(drop=True)
    interval_start, interval_end = _ns(support.start), _ns(support.end_exclusive)
    window_support = windows.support_id.to_numpy(dtype='int64')
    endpoints = _ns(windows.end_at)
    pieces = []
    for lo, hi, stamps, logs, ages in _quote_grid_batches(windows.end_at, quotes):
        sid = window_support[lo:hi]
        endpoint = endpoints[lo:hi]
        a, b = interval_start[np.maximum(sid, 0)], interval_end[np.maximum(sid, 0)]
        absent = stamps[:, 0] == np.iinfo('int64').min
        status = np.select([sid < 0, (endpoint-300*SECOND < a) | (endpoint > b), absent,
                            stamps[:, 0] < a], STATUSES[1:], default='available')
        good = status == 'available'
        if ((stamps[good] < a[good, None]).any() or (stamps[good] >= endpoint[good, None]).any()):
            raise AssertionError('A used quote leaves the original support or reaches the future')
        one_second_max_age = np.max(ages, axis=1)
        for freq in FREQUENCIES:
            quote_grid = stamps[:, ::freq]
            log_grid = logs[:, ::freq]
            differences = log_grid[:, 1:]-log_grid[:, :-1]
            # Multiplication after the sum preserves the prescribed units.
            variance = np.add.reduce(np.square(differences), axis=1)*100_000_000.
            grid_age = np.max(ages[:, ::freq], axis=1)
            out = windows.iloc[lo:hi][['window_id','kind','end_at','episode_id','onset_confirmed']].copy()
            out['frequency_sec'] = freq
            out['n_returns'] = 300//freq
            out['status'] = status
            calculated = {'rv_bps2': variance, 'rv_bps': np.sqrt(variance),
                          'max_mark_age_sec': grid_age,
                          'repeat_fraction': np.count_nonzero(quote_grid[:,1:] == quote_grid[:,:-1], axis=1)/(300//freq),
                          'zero_return_fraction': np.count_nonzero(differences == 0, axis=1)/(300//freq)}
            for name, values in calculated.items():
                out[name] = np.where(good, values, np.nan)
            out['first_quote_at'] = pd.to_datetime(np.where(good, stamps[:,0], np.iinfo('int64').min), utc=True)
            out['last_quote_at'] = pd.to_datetime(np.where(good, stamps[:,-1], np.iinfo('int64').min), utc=True)
            for cap in CAPS:
                out['own_age_le_'+str(cap)] = good & (grid_age <= cap)
                out['common_age_le_'+str(cap)] = good & (one_second_max_age <= cap)
            pieces.append(out)
    result = pd.concat(pieces, ignore_index=True).sort_values(['window_id','frequency_sec']).reset_index(drop=True)
    five = result.loc[result.frequency_sec.eq(5)].reset_index(drop=True)
    inherited = windows.inherited_state_available.to_numpy(dtype=bool)
    if not five.loc[inherited, 'status'].eq('available').all():
        raise AssertionError('An inherited available state cannot be reconstructed')
    np.testing.assert_allclose(five.loc[inherited,'rv_bps'], windows.loc[inherited,'reference_rv_bps'],
                               rtol=1e-8, atol=1e-8, err_msg='Inherited 5s pre-state')
    return result


def _quantile(values, probability):
    values = sorted(float(x) for x in values if pd.notna(x))
    if not values:
        return np.nan
    point = (len(values)-1)*probability
    lower, upper = int(math.floor(point)), int(math.ceil(point))
    return values[lower]+(point-lower)*(values[upper]-values[lower])


def _selected_rows(group, sample):
    cohort = set(group.index[group.kind.eq('calibration') | group.onset_confirmed])
    available = set(group.index[group.status.eq('available')])
    selected = cohort & available
    if sample != 'all_available':
        cap = sample.split('_')[-1]
        selected &= set(group.index[group['common_age_le_'+cap]])
        own = set(group.index[group['own_age_le_'+cap]])
    else:
        own = available
    return cohort, available, selected, own


def _aggregate(group, sample):
    cohort, available, selected, own = _selected_rows(group, sample)
    chosen = group.loc[sorted(selected)]
    output = {'sample': sample, 'windows': len(group), 'cohort_windows': len(cohort),
              'available_windows': len(cohort & available), 'selected_windows': len(selected),
              'own_grid_age_pass_windows': len(cohort & own),
              'zero_variance_windows': int(np.count_nonzero(chosen.rv_bps2.to_numpy() == 0))}
    for name, column in [('mean_variance_bps2','rv_bps2'), ('mean_rv_bps','rv_bps'),
                         ('mean_repeat_fraction','repeat_fraction'), ('mean_zero_return_fraction','zero_return_fraction')]:
        output[name] = float(np.mean(chosen[column].to_numpy())) if len(chosen) else np.nan
    output['median_rv_bps'] = _quantile(chosen.rv_bps, .5)
    output['median_max_mark_age_sec'] = _quantile(chosen.max_mark_age_sec, .5)
    output['p95_max_mark_age_sec'] = _quantile(chosen.max_mark_age_sec, .95)
    return output


def _average_ranks(values):
    """Assign average tied ranks without the production pandas rank/corr path."""
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = np.empty(len(values), dtype=float)
    first = 0
    while first < len(ordered):
        end = first+1
        while end < len(ordered) and ordered[end][1] == ordered[first][1]:
            end += 1
        rank = (first+1+end)/2.
        for position in range(first, end):
            ranks[ordered[position][0]] = rank
        first = end
    return ranks


def reference_summary_tables(measurements):
    frame = measurements.copy()
    frame['endpoint_date'] = pd.to_datetime(frame.end_at.dt.strftime('%Y-%m-%d'), utc=True)
    frame['endpoint_hour_utc'] = frame.end_at.dt.hour
    tables = {}
    for name, columns in [('summary',['kind','frequency_sec']),
                          ('daily',['kind','endpoint_date','frequency_sec']),
                          ('hourly',['kind','endpoint_hour_utc','frequency_sec'])]:
        rows = []
        for key, group in frame.groupby(columns, sort=True):
            identity = dict(zip(columns, key))
            rows.extend({**identity, **_aggregate(group,sample)} for sample in SAMPLES)
        tables[name] = pd.DataFrame(rows)
    pairs = []
    for kind in sorted(frame.kind.unique()):
        group = frame.loc[frame.kind.eq(kind)]
        for sample in SAMPLES:
            selected = group.loc[sorted(_selected_rows(group,sample)[2])]
            reference = selected.loc[selected.frequency_sec.eq(5)].set_index('window_id')
            for freq in FREQUENCIES:
                candidate = selected.loc[selected.frequency_sec.eq(freq)].set_index('window_id')
                ids = sorted(set(reference.index) & set(candidate.index))
                if len(ids) != len(reference) or len(ids) != len(candidate):
                    raise AssertionError('Compared frequencies use different window identities')
                x = candidate.loc[ids,'rv_bps'].to_numpy(dtype=float)
                y = reference.loc[ids,'rv_bps'].to_numpy(dtype=float)
                vx = candidate.loc[ids,'rv_bps2'].to_numpy(dtype=float)
                vy = reference.loc[ids,'rv_bps2'].to_numpy(dtype=float)
                rx, ry = _average_ranks(x), _average_ranks(y)
                correlation = np.nan
                if len(ids) >= 2:
                    rx, ry = rx-rx.mean(), ry-ry.mean()
                    denominator = np.sqrt(np.dot(rx,rx)*np.dot(ry,ry))
                    if denominator > 0:
                        correlation = float(np.dot(rx,ry)/denominator)
                ref_mean = float(np.mean(vy)) if len(ids) else np.nan
                pairs.append({'kind':kind, 'sample':sample, 'frequency_sec':freq, 'reference_frequency_sec':5,
                    'paired_windows':len(ids), 'zero_variance_windows':int(np.count_nonzero(vx == 0)),
                    'reference_zero_variance_windows':int(np.count_nonzero(vy == 0)),
                    'mean_variance_ratio_to_5s':float(np.mean(vx)/ref_mean) if ref_mean > 0 else np.nan,
                    'mean_absolute_rv_difference_bps':float(np.mean(np.abs(x-y))) if len(ids) else np.nan,
                    'spearman_rv_to_5s':correlation})
    tables['comparisons'] = pd.DataFrame(pairs)
    return tables


def reference_direction(onsets, ticks):
    """Group saved tick contributions by their calendar bin independently."""
    keys = ticks.timestamp.dt.floor('s')
    selected = ticks.loc[keys.isin(onsets.bin_start)].copy()
    selected['onset_bin'] = keys.loc[selected.index]
    grouped = {key: group for key, group in selected.groupby('onset_bin',sort=False)}
    rows = []
    for onset in onsets.itertuples(index=False):
        group = grouped.get(onset.bin_start)
        if group is None or not len(group):
            raise AssertionError('Onset has no saved tick contributions')
        values = group.quote_ofi.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not group.exclusion_code.eq(0).all():
            raise AssertionError('Onset includes an invalid saved contribution')
        if hasattr(onset,'recorded_update_count') and len(group) != onset.recorded_update_count:
            raise AssertionError('Onset quote count differs from saved tick count')
        stable = math.fsum(float(value) for value in values)
        gross = math.fsum(abs(float(value)) for value in values)
        eps = np.finfo('float64').eps
        if len(group)*eps >= 1:
            raise AssertionError('Floating-point summation bound is undefined')
        gamma = len(group)*eps/(1-len(group)*eps)
        spacing = abs(float(np.spacing(stable)))
        inherited = float(onset.quote_ofi)
        difference = inherited-stable
        bound = gamma*gross+spacing
        if not np.isfinite(inherited) or abs(difference) > bound:
            raise AssertionError('Inherited onset net fails the registered summation bound')
        sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0
        rows.append({'episode_id': int(onset.episode_id), 'bin_start':onset.bin_start,
            'onset_confirmed':bool(onset.onset_confirmed), 'tick_contributions':len(group),
            'inherited_net':inherited, 'stable_net':stable, 'gross_contributions':gross,
            'cancellation_ratio':abs(stable)/gross if gross else np.nan,
            'aggregation_difference':difference, 'summation_error_bound':bound,
            'inherited_direction':sign(inherited), 'stable_direction':sign(stable),
            'direction_agrees':sign(inherited) == sign(stable)})
    return pd.DataFrame(rows)


def reference_direction_summary(direction):
    result = {}
    for name, group in [('all_onsets',direction), ('confirmed_onsets',direction.loc[direction.onset_confirmed])]:
        ratios = group.cancellation_ratio.to_numpy(dtype=float)
        valid = ratios[np.isfinite(ratios)]
        values = {'onsets':len(group), 'gross_zero':int(np.count_nonzero(group.gross_contributions.to_numpy() == 0)),
            'stable_net_zero':int(np.count_nonzero(group.stable_net.to_numpy() == 0)),
            'inherited_net_zero':int(np.count_nonzero(group.inherited_net.to_numpy() == 0)),
            'direction_disagreements':int(np.count_nonzero(~group.direction_agrees.to_numpy())),
            'ratio_available':len(valid), 'zero_ratio':int(np.count_nonzero(valid == 0)),
            'positive_ratio':int(np.count_nonzero(valid > 0)),
            'max_absolute_aggregation_difference':max(abs(float(x)) for x in group.aggregation_difference) if len(group) else None}
        for label, probability in [('p01',.01),('p05',.05),('median',.5)]:
            values[label+'_cancellation_ratio'] = _quantile(valid,probability) if len(valid) else None
        for cap in (.01,.05,.10):
            values['positive_ratio_le_'+str(int(cap*100))+'pct'] = int(np.count_nonzero((valid > 0) & (valid <= cap)))
        result[name] = values
    return result


def _records(frame):
    records = []
    for record in frame.to_dict(orient='records'):
        normalized = {}
        for key,value in record.items():
            if pd.isna(value):
                value = None
            elif isinstance(value,np.generic):
                value = value.item()
            normalized[key] = value
        records.append(normalized)
    return records


def verify_state_measurement(run):
    """Verify deterministic state measurements and their full stored lineage."""
    run = Path(run)
    manifest, hash_count = _registered_hashes(run)
    inputs = manifest['inputs']
    parent = json.loads(Path(inputs['parent_manifest']['path']).read_text())
    parent_stats = json.loads(Path(inputs['parent_statistics']['path']).read_text())
    parent_verification = json.loads(Path(inputs['parent_verification']['path']).read_text())
    episode = json.loads(Path(inputs['episode_manifest']['path']).read_text())
    imbalance = json.loads(Path(inputs['ofi_manifest']['path']).read_text())
    parameters = json.loads(Path(inputs['detector_parameters']['path']).read_text())
    if parent_verification.get('status') != 'passed':
        raise AssertionError('Parent response verification did not pass')
    for name,filename in [('parent_statistics','statistics.json'),('parent_verification','verification.json')]:
        if inputs[name]['sha256'] != parent['outputs'][filename]:
            raise AssertionError('Parent output lineage differs: '+name)
    for name in ('clean_quotes','support_intervals'):
        if inputs[name]['sha256'] != parent['inputs'][name]['sha256']:
            raise AssertionError('Parent raw-input lineage differs: '+name)
    links = [('onsets',parent['inputs']['structural_only_onsets']['sha256']),
             ('onsets',episode['outputs']['structural_only/onsets.parquet']),
             ('episode_manifest',parent['inputs']['parent_manifest']['sha256']),
             ('detector_parameters',episode['inputs']['structural_only_parameters']['sha256']),
             ('ofi_ticks',imbalance['outputs']['quote_ofi_ticks.parquet']),
             ('clean_quotes',imbalance['inputs']['parent_clean']['sha256']),
             ('support_intervals',imbalance['inputs']['support']['sha256'])]
    for key,expected in links:
        if inputs[key]['sha256'] != expected:
            raise AssertionError('Cross-stage lineage differs: '+key)
    if episode['run_id'] != parent['parent_episode_audit']:
        raise AssertionError('Parent episode run differs')
    scope = 'calibration_time_windows_and_reused_development_onsets'
    _json_compare(manifest, {'parent_price_response_audit':parent['run_id'], 'primary_policy':'structural_only',
        'frequencies_seconds':list(FREQUENCIES), 'window_seconds':300, 'scope':scope,
        'production_inference_approved':False, 'final_state_specification_approved':False, 'downstream_ready':False})
    start = pd.Timestamp(parameters['training_start'])
    cutoff = pd.Timestamp(parameters['parameter_cutoff'])
    stop = pd.Timestamp(parent_stats['development_end_exclusive'])
    if cutoff != pd.Timestamp(parent_stats['parameter_cutoff']):
        raise AssertionError('Calibration and development cutoff mismatch')
    quotes = pd.read_parquet(inputs['clean_quotes']['path'],columns=['timestamp','bid','ask'],filters=[('timestamp','<',stop)])
    support = pd.read_parquet(inputs['support_intervals']['path'],filters=[('policy','==','structural_only')]).reset_index(drop=True)
    onsets = pd.read_parquet(inputs['onsets']['path'])
    ticks = pd.read_parquet(inputs['ofi_ticks']['path'],columns=['timestamp','exclusion_code','quote_ofi'],
                            filters=[('timestamp','>=',cutoff),('timestamp','<',stop)])
    for name,frame in [('quotes',quotes),('contributions',ticks)]:
        if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise AssertionError('Unordered or duplicated '+name)
        _ns(frame.timestamp)
    prices = quotes[['bid','ask']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices[:,0] <= 0).any() or (prices[:,1] <= prices[:,0]).any():
        raise AssertionError('Invalid raw prices')
    if not (onsets.bin_start.ge(cutoff) & onsets.available_at.le(stop)).all():
        raise AssertionError('Onset outside the frozen development interval')
    windows = reference_windows(onsets,support,training_start=start,parameter_cutoff=cutoff)
    _compare(pd.read_parquet(run/'windows.parquet'),windows)
    expected_calibration = int((cutoff-start).total_seconds()//300)
    if int(windows.kind.eq('calibration').sum()) != expected_calibration or int(windows.kind.eq('development').sum()) != len(onsets):
        raise AssertionError('Calibration-grid or onset conservation failed')
    measurements = reference_measurements(windows,quotes,support)
    _compare(pd.read_parquet(run/'measurements.parquet'),measurements)
    if len(measurements) != len(windows)*len(FREQUENCIES):
        raise AssertionError('Full window-frequency grid was not retained')
    print('Independently reconstructed '+str(len(windows))+' windows and '+str(len(measurements))+' measurements',flush=True)
    tables = reference_summary_tables(measurements)
    for name,table in tables.items():
        _compare(pd.read_parquet(run/(name+'.parquet')),table)
    direction = reference_direction(onsets,ticks)
    actual_direction = pd.read_parquet(run/'direction.parquet')
    _compare(actual_direction,direction)
    for name in ('inherited_net','stable_net','gross_contributions','aggregation_difference','summation_error_bound','cancellation_ratio'):
        np.testing.assert_allclose(actual_direction[name],direction[name],rtol=0,atol=0,equal_nan=True,
                                   err_msg='Exact direction arithmetic: '+name)
    statuses = {}
    for kind in sorted(measurements.kind.unique()):
        statuses[kind] = {}
        for freq in FREQUENCIES:
            group = measurements.loc[measurements.kind.eq(kind) & measurements.frequency_sec.eq(freq)]
            statuses[kind][str(freq)] = {status:int(np.count_nonzero(group.status.to_numpy() == status)) for status in STATUSES}
    expected_stats = {'scope':scope, 'training_start':start.isoformat(), 'parameter_cutoff':cutoff.isoformat(),
        'development_end_exclusive':stop.isoformat(), 'primary_policy':'structural_only',
        'window_counts':{kind:int(np.count_nonzero(windows.kind.to_numpy() == kind)) for kind in sorted(windows.kind.unique())},
        'measurement_rows':len(measurements), 'status_counts_by_kind_frequency':statuses,
        'summary':_records(tables['summary']), 'comparisons':_records(tables['comparisons']),
        'direction_summary':reference_direction_summary(direction), 'final_state_specification_approved':False}
    _json_compare(json.loads((run/'statistics.json').read_text()),expected_stats,exact=True)
    evidence = json.loads((run/'test_evidence.json').read_text())
    if evidence.get('status') != 'passed' or evidence.get('failures') != 0 or evidence.get('errors') != 0:
        raise AssertionError('Stored full-suite test evidence did not pass')
    for name,sha in evidence.get('test_source_sha256',{}).items():
        if manifest['outputs'].get('full_test_sources/'+name) != sha:
            raise AssertionError('Executed test source differs from archived source: '+name)
    return {'status':'passed','hashes_verified':hash_count, 'window_rows_verified':len(windows),
        'calibration_windows_verified':expected_calibration, 'development_onsets_verified':len(onsets),
        'measurement_rows_verified':len(measurements), 'strict_quote_mark_requests_verified':len(windows)*301,
        'inherited_5s_states_verified':int(onsets.state_history_available.sum()),
        'summary_rows_verified':len(tables['summary']), 'daily_rows_verified':len(tables['daily']),
        'hourly_rows_verified':len(tables['hourly']), 'comparison_rows_verified':len(tables['comparisons']),
        'direction_rows_verified':len(direction), 'tick_contributions_verified':int(direction.tick_contributions.sum()),
        'methods':'Strict backward merge_asof raw-quote grids, independently reconstructed deterministic windows and inherited onsets, nested-frequency calculations, set-based cohorts, hand-ranked Spearman correlations, calendar-bin fsum reconciliation, every table/statistics cell, exact input/output/source/snapshot hashes.',
        'production_inference_approved':False,'final_state_specification_approved':False,'downstream_ready':False}

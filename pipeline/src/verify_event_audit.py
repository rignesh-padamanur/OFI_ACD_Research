"""Independent arithmetic for the frozen stage-05a development event audit.

Reference calculations intentionally do not import the detector implementation.
They reconstruct the empirical quantile, raw-feature covariance distances,
endpoint residual aggregation, and strictly preceding quote measurements.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter

SECOND = 1_000_000_000
POLICIES = ('structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s')


def ns(values):
    return pd.DatetimeIndex(values).as_unit('ns').asi8


def linear_quantile(values, quantile=.99):
    ordered = np.sort(np.asarray(values, dtype=float))
    position = (len(ordered)-1)*quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    return float(ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower]))


def reference_detector(values, scoring=None):
    """Use pandas moments and raw covariance, not standardized Cholesky scoring."""
    frame = pd.DataFrame(np.asarray(values, dtype=float))
    mean = frame.mean().to_numpy()
    sd = frame.std(ddof=1).to_numpy()
    covariance_raw = frame.cov().to_numpy()
    standardized = (frame-mean)/sd
    covariance_z = standardized.cov().to_numpy()
    inverse = np.linalg.inv(covariance_raw)
    centered = frame.to_numpy()-mean
    train_score = np.sqrt(np.einsum('ij,jk,ik->i', centered, inverse, centered))
    selected = frame.to_numpy() if scoring is None else np.asarray(scoring, dtype=float)
    delta = selected-mean
    score = np.sqrt(np.einsum('ij,jk,ik->i', delta, inverse, delta))
    return {'mean': mean, 'sample_std': sd, 'covariance_z': covariance_z,
            'threshold': linear_quantile(train_score), 'quantile': .99}, score


def reference_duration_bins(path, psi):
    """Endpoint resampling, with explicit observed-completion selection."""
    event = path.event_observed.to_numpy(dtype=bool)
    ratio = path.normalized_exposure.to_numpy(dtype=float)/np.asarray(psi, dtype=float)
    frame = pd.DataFrame({'endpoint': path.end_at.to_numpy()[event],
                          'residual': ratio[event]})
    grouped = frame.set_index('endpoint').resample('1s').residual.agg(['count', 'mean'])
    grouped = grouped.loc[grouped['count'] > 0]
    return grouped.rename(columns={'count': 'completed_residual_count',
                                    'mean': 'mean_duration_residual'}).rename_axis('bin_start').reset_index()


def reference_psi(path, parameters):
    """A filter within each reset fragment, independent of the compiled loop."""
    omega, alpha, beta = [float(parameters[k]) for k in ('omega', 'alpha', 'beta')]
    y = path.normalized_exposure.to_numpy(dtype=float)
    starts = np.flatnonzero(path.recurrence_reset.to_numpy(dtype=bool))
    if len(path) and (not len(starts) or starts[0] != 0):
        raise AssertionError('Missing first recurrence reset')
    out = np.empty(len(path))
    for first, stop in zip(starts, np.r_[starts[1:], len(path)], strict=True):
        out[first] = omega/(1-alpha-beta)
        if stop > first+1:
            out[first+1:stop] = lfilter([1.], [1., -beta], omega+alpha*y[first:stop-1],
                                      zi=[beta*out[first]])[0]
    return out


def reference_quote_diagnostics(clock, quotes, support_intervals):
    """Independent backward merges and rolling-sum reconstruction.

    The clock must include every second, including unavailable seconds, and use
    the materialized policy columns support_id/flow_status/quote_ofi.
    """
    times = ns(clock.bin_start)
    q = quotes[['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume']].copy()
    q['quote_timestamp'] = q.timestamp
    support_start = ns(support_intervals.start)
    support_stop = ns(support_intervals.end_exclusive)
    sid = clock.support_id.to_numpy(dtype=int)
    safe = np.maximum(sid, 0)

    def marks(mark_time):
        request = pd.DataFrame({'timestamp': pd.to_datetime(mark_time, utc=True)})
        joined = pd.merge_asof(request, q, on='timestamp', direction='backward', allow_exact_matches=False)
        qt = ns(joined.quote_timestamp)
        valid = ((sid >= 0) & joined.quote_timestamp.notna().to_numpy()
                 & (qt >= support_start[safe]) & (qt < support_stop[safe]))
        return joined, valid

    pre, pre_valid = marks(times)
    end, end_valid = marks(times+SECOND)
    pre_size = (pre.bid_volume+pre.ask_volume).to_numpy()
    end_size = (end.bid_volume+end.ask_volume).to_numpy()
    pre_spread = (pre.ask-pre.bid).to_numpy()*10000
    end_spread = (end.ask-end.bid).to_numpy()*10000
    out = pd.DataFrame({'bin_start': clock.bin_start.to_numpy(),
                        'pre_spread_pips': np.where(pre_valid, pre_spread, np.nan),
                        'end_spread_pips': np.where(end_valid, end_spread, np.nan),
                        'spread_change_pips': np.where(pre_valid & end_valid, end_spread-pre_spread, np.nan),
                        'pre_displayed_size': np.where(pre_valid, pre_size, np.nan),
                        'end_displayed_size': np.where(end_valid, end_size, np.nan),
                        'log_size_ratio': np.nan,
                        'pre_quote_age_sec': np.where(pre_valid, (times-ns(pre.quote_timestamp))/SECOND, np.nan),
                        'end_quote_age_sec': np.where(end_valid, (times+SECOND-ns(end.quote_timestamp))/SECOND, np.nan)})
    size_ok = pre_valid & end_valid & (pre_size > 0) & (end_size > 0)
    out.loc[size_ok, 'log_size_ratio'] = np.log(end_size[size_ok]/pre_size[size_ok])
    history_valid = np.isin(clock.flow_status.to_numpy(), [1, 2]) & np.isfinite(clock.quote_ofi)
    flow = pd.Series(clock.quote_ofi.to_numpy()).where(history_valid).shift(1).rolling(300, min_periods=300).sum().to_numpy()
    enough = np.arange(len(clock)) >= 300
    same = np.zeros(len(clock), dtype=bool)
    if len(clock) > 300:
        same[300:] = ((sid[300:] >= 0) & (sid[300:] == sid[:-300])
                      & (times[300:]-times[:-300] == 300*SECOND))
    ready = enough & same & np.isfinite(flow)
    # Five disjoint second phases contain the required five-second return grid.
    # Each rolling window has 60 returns and therefore exactly 61 marks.
    mid = (pre.bid+pre.ask).to_numpy()/2
    log_mid = pd.Series(np.where(pre_valid & (mid > 0), np.log(mid), np.nan))
    squared = (log_mid-log_mid.shift(5)).pow(2)
    total = np.full(len(clock), np.nan)
    for phase in range(5):
        index = np.arange(phase, len(clock), 5)
        total[index] = squared.iloc[index].rolling(60, min_periods=60).sum()
    valid_returns = ready & np.isfinite(total)
    out['pre_quote_ofi_300s'] = np.where(valid_returns, flow, np.nan)
    out['pre_rv_300s_5s_bps'] = np.where(valid_returns, np.sqrt(total)*10000, np.nan)
    return out


def compare_numeric_frame(actual, expected, *, rtol=1e-9, atol=1e-9):
    if len(actual) != len(expected):
        raise AssertionError('Frame row count differs')
    for column in expected.columns:
        a, e = actual[column], expected[column]
        if pd.api.types.is_datetime64_any_dtype(e.dtype):
            np.testing.assert_array_equal(ns(a), ns(e), err_msg=column)
        elif pd.api.types.is_numeric_dtype(e.dtype):
            np.testing.assert_allclose(a, e, rtol=rtol, atol=atol, equal_nan=True, err_msg=column)
        else:
            np.testing.assert_array_equal(a, e, err_msg=column)


def check_detector_fit(values, fitted, score):
    reference, expected_score = reference_detector(values)
    for key in ('mean', 'sample_std', 'covariance_z', 'threshold', 'quantile'):
        np.testing.assert_allclose(fitted[key], reference[key], rtol=1e-10, atol=1e-10, err_msg=key)
    np.testing.assert_allclose(score, expected_score, rtol=1e-9, atol=1e-9)


def check_registered_hashes(run):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    cache = {}
    def digest(path):
        key = str(Path(path))
        if key in cache:
            return cache[key]
        h = hashlib.sha256()
        with Path(path).open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024), b''):
                h.update(chunk)
        cache[key] = h.hexdigest()
        return cache[key]
    for key, record in manifest.get('inputs', {}).items():
        if digest(record['path']) != record['sha256']:
            raise AssertionError('Changed input: '+key)
    for name, fingerprint in manifest.get('outputs', {}).items():
        if digest(run/name) != fingerprint:
            raise AssertionError('Changed output: '+name)
    for name, fingerprint in manifest.get('code_sha256', {}).items():
        if digest(run/'code'/name) != fingerprint:
            raise AssertionError('Changed executed source: '+name)
    for name, fingerprint in manifest.get('ancestor_hashes', {}).items():
        if digest(name) != fingerprint:
            raise AssertionError('Changed ancestor: '+name)
    for name, fingerprint in manifest.get('previous_files', {}).items():
        if digest(Path(manifest['snapshot_path'])/name) != fingerprint:
            raise AssertionError('Changed before-repair snapshot: '+name)
    return manifest


def reference_clock(raw, factors, policy):
    """Materialize masks directly and join each past-only factor by date/slot."""
    status = raw['flow_status_'+policy].to_numpy()
    sid = raw['support_id_'+policy].to_numpy()
    count = np.where(sid >= 0, raw.recorded_update_count.to_numpy(), np.nan)
    flow = np.full(len(raw), np.nan)
    flow[status == 1] = 0.
    flow[status == 2] = raw.observed_quote_ofi.to_numpy()[status == 2]
    out = pd.DataFrame({'bin_start': raw.bin_start, 'available_at': raw.available_at,
                        'support_id': sid, 'flow_status': status, 'quote_ofi': flow,
                        'tick_count': count, 'abs_quote_ofi': np.abs(flow)})
    keys = pd.DataFrame({'effective_date': raw.bin_start.dt.normalize(),
                         'slot': raw.bin_start.dt.hour*4+raw.bin_start.dt.minute//15})
    for channel in ('tick_count', 'abs_quote_ofi'):
        selected = factors.loc[factors.policy.eq(policy) & factors.channel.eq(channel),
                               ['effective_date', 'slot', 'phi', 'status', 'training_end_exclusive']]
        merged = keys.merge(selected, how='left', on=['effective_date', 'slot'], validate='many_to_one', sort=False)
        if len(merged) != len(raw) or not (merged.training_end_exclusive == merged.effective_date).all():
            raise AssertionError('Missing/nonchronological seasonal calibration')
        value = out[channel].to_numpy()/merged.phi.to_numpy()
        out['adjusted_'+channel] = np.where(merged.status.eq(5), value, np.nan)
    return out


def check_json_values(actual, expected, path='statistics'):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check_json_values(actual[key], value, path+'.'+key)
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(path+': sequence length')
        for i, value in enumerate(expected):
            check_json_values(actual[i], value, path+'['+str(i)+']')
    elif expected is None:
        if actual is not None:
            raise AssertionError(path)
    elif isinstance(expected, (bool, int, str)):
        if actual != expected:
            raise AssertionError(path+': '+repr(actual)+' != '+repr(expected))
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=path)


def reference_tables(scored):
    """Count/sum from group row identities; no production table function reuse."""
    f = scored.copy()
    f['date'] = f.bin_start.dt.normalize()
    f['weekday'] = f.bin_start.dt.dayofweek
    f['slot'] = f.bin_start.dt.hour*4+f.bin_start.dt.minute//15
    daily, cells, profiles = [], [], []
    for key, indices in f.groupby(['period', 'date'], observed=True, sort=True).indices.items():
        rows = f.iloc[indices]
        daily.append({'period': key[0], 'date': key[1], 'eligible_bins': len(rows),
                      'candidate_bins': int(np.count_nonzero(rows.candidate_event)),
                      'comparator_bins': int(np.count_nonzero(rows.candidate_without_duration)),
                      'mean_distance': float(np.sum(rows.distance)/len(rows))})
    for key, indices in f.groupby(['period', 'weekday', 'slot'], observed=True, sort=True).indices.items():
        rows = f.iloc[indices]
        cells.append({'period': key[0], 'weekday': key[1], 'slot': key[2],
                      'eligible_bins': len(rows), 'candidate_bins': int(np.count_nonzero(rows.candidate_event))})
    f['sign_code'] = ((f.z_ofi >= 0).astype(int)*4+(f.z_intensity >= 0).astype(int)*2
                      +(f.z_duration >= 0).astype(int))
    for key, indices in f.groupby(['period', 'candidate_event', 'sign_code'], observed=True, sort=True).indices.items():
        rows = f.iloc[indices]
        profiles.append({'period': key[0], 'candidate_event': key[1], 'sign_code': key[2], 'bins': len(rows),
                         **{'mean_'+v: float(np.sum(rows[v])/len(rows)) for v in ['z_ofi', 'z_intensity', 'z_duration']}})
    metrics = ['pre_spread_pips', 'end_spread_pips', 'spread_change_pips',
               'pre_displayed_size', 'end_displayed_size', 'log_size_ratio',
               'pre_quote_age_sec', 'end_quote_age_sec', 'pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']
    d = f.loc[f.period.eq('development')].reset_index(drop=True)
    groups = list(d.groupby(['weekday', 'slot']).indices.values())
    event = d.candidate_event.to_numpy(dtype=bool)
    economic = []
    for metric in metrics:
        values = d[metric].to_numpy()
        finite = np.isfinite(values)
        ev, control = values[event & finite], values[~event & finite]
        matched_n, sum_events, sum_controls = 0, 0., 0.
        for indices in groups:
            e = indices[event[indices] & finite[indices]]
            c = indices[~event[indices] & finite[indices]]
            if len(e) and len(c):
                matched_n += len(e)
                sum_events += np.sum(values[e])
                sum_controls += len(e)*np.mean(values[c])
        mean_event = sum_events/matched_n if matched_n else np.nan
        mean_control = sum_controls/matched_n if matched_n else np.nan
        economic.append({'metric': metric, 'event_available': len(ev), 'control_available': len(control),
                         'event_missing': int(np.count_nonzero(event & ~finite)),
                         'control_missing': int(np.count_nonzero(~event & ~finite)),
                         'matched_events': matched_n, 'unmatched_events': len(ev)-matched_n,
                         'event_mean': float(np.mean(ev)) if len(ev) else np.nan,
                         'event_median': float(np.median(ev)) if len(ev) else np.nan,
                         'control_mean_unmatched': float(np.mean(control)) if len(control) else np.nan,
                         'control_median': float(np.median(control)) if len(control) else np.nan,
                         'matched_event_mean': mean_event, 'event_weighted_control_mean': mean_control,
                         'matched_difference': mean_event-mean_control})
    return {'daily': pd.DataFrame(daily), 'time_cells': pd.DataFrame(cells),
            'feature_profiles': pd.DataFrame(profiles), 'economic_comparison': pd.DataFrame(economic)}


def verify_event_audit(run):
    """Reconstruct each registered development artifact from verified ancestors."""
    from src.verify_chronological_acd import reference_sample
    run = Path(run)
    manifest = check_registered_hashes(run)
    inputs = manifest['inputs']
    summary = json.loads((run/'statistics.json').read_text())
    parent = json.loads(Path(inputs['acd_statistics']['path']).read_text())
    start, cutoff, stop = [pd.Timestamp(parent[k]) for k in
                           ['training_start', 'parameter_cutoff', 'validation_end_exclusive']]
    raw = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    np.testing.assert_array_equal(ns(raw.bin_start), np.arange(start.value, stop.value, SECOND))
    np.testing.assert_array_equal(ns(raw.available_at), ns(raw.bin_start)+SECOND)
    quotes = pd.read_parquet(inputs['clean_quotes']['path'], columns=['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume'],
                            filters=[('timestamp', '>=', start-pd.Timedelta(days=1)), ('timestamp', '<', stop)])
    supports = pd.read_parquet(inputs['support_intervals']['path'])
    factors = pd.read_parquet(inputs['calibration_parameters']['path'],
                             filters=[('effective_date', '>=', start), ('effective_date', '<', stop)])
    reasons = ['unsupported', 'invalid_flow', 'seasonal_unavailable', 'quiet_no_duration', 'incomplete_duration', 'eligible']
    counts = {'clock_rows_verified': 0, 'duration_spells_verified': 0,
              'completed_duration_bins_verified': 0, 'scored_bins_verified': 0, 'tables_verified': 0}
    for policy in POLICIES:
        directory = run/policy
        params = json.loads((directory/'parameters.json').read_text())
        model = parent['policies'][policy]['models']['weibull_acd']
        check_json_values(params, {'parameter_cutoff': cutoff.isoformat(), 'training_start': start.isoformat(),
                                  'acd_parameters': model['parameters'],
                                  'acd_constraint_warning': model['numerical_constraint_warning'],
                                  'original_acd_diagnostic_flags': model['validation']['diagnostic_flags']})
        seasonal = pd.read_parquet(inputs['seasonal_spells_'+policy]['path'])
        path = reference_sample(seasonal, start.value, stop.value)
        del seasonal
        psi = reference_psi(path, model['parameters'])
        saved_prediction = pd.read_parquet(inputs['saved_predictions_'+policy]['path'])
        past = path.origin_at.ge(cutoff).to_numpy()
        np.testing.assert_array_equal(path.origin_quote_id.to_numpy()[past], saved_prediction.origin_quote_id)
        np.testing.assert_allclose(psi[past], saved_prediction.psi, rtol=1e-10, atol=1e-10)
        duration = reference_duration_bins(path, psi)
        compare_numeric_frame(pd.read_parquet(directory/'duration_bins.parquet'), duration)
        cross = int(np.count_nonzero((ns(path.origin_at) < cutoff.value)
                                     & (ns(path.end_at) >= cutoff.value) & path.event_observed.to_numpy()))
        counts['duration_spells_verified'] += len(path)
        counts['completed_duration_bins_verified'] += len(duration)
        del path, psi, saved_prediction
        clock = reference_clock(raw, factors, policy)
        joined = clock[['bin_start']].merge(duration, on='bin_start', how='left', validate='one_to_one', sort=False)
        completed = joined.completed_residual_count.fillna(0).to_numpy(dtype='int64')
        recorded = raw.recorded_update_count.to_numpy()
        if (completed > recorded).any():
            raise AssertionError('Excess completed residual coverage')
        ready = np.isfinite(clock.adjusted_abs_quote_ofi) & np.isfinite(clock.adjusted_tick_count)
        reason = np.full(len(clock), 'eligible', dtype=object)
        reason[completed != recorded] = 'incomplete_duration'
        reason[recorded == 0] = 'quiet_no_duration'
        reason[~ready] = 'seasonal_unavailable'
        valid_flow = np.isin(clock.flow_status, [1, 2])
        reason[~valid_flow] = 'invalid_flow'
        reason[clock.flow_status.eq(0)] = 'unsupported'
        period = np.where(clock.bin_start.lt(cutoff), 'calibration', 'development')
        expected_opportunity = pd.DataFrame({'bin_start': raw.bin_start, 'available_at': raw.available_at,
                                             'period': period, 'recorded_update_count': recorded,
                                             'support_id': clock.support_id, 'flow_status': clock.flow_status,
                                             'seasonal_ready': ready, 'completed_residual_count': completed,
                                             'eligible_without_duration': valid_flow & ready, 'reason': reason})
        opportunity = pd.read_parquet(directory/'opportunities.parquet')
        compare_numeric_frame(opportunity, expected_opportunity)
        eligible = reason == 'eligible'
        expected = clock.loc[eligible, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                                       'adjusted_abs_quote_ofi', 'adjusted_tick_count']].reset_index(drop=True)
        expected['recorded_update_count'] = recorded[eligible]
        expected['completed_residual_count'] = completed[eligible]
        expected['mean_duration_residual'] = joined.mean_duration_residual.to_numpy()[eligible]
        expected['negative_log_mean_residual'] = -np.log(expected.mean_duration_residual)
        expected['period'] = period[eligible]
        economic = reference_quote_diagnostics(clock, quotes, supports.loc[supports.policy.eq(policy)].reset_index(drop=True))
        for column in economic.columns:
            if column != 'bin_start':
                expected[column] = economic[column].to_numpy()[eligible]
        del economic, clock, joined, duration, expected_opportunity
        features = ['adjusted_abs_quote_ofi', 'adjusted_tick_count', 'negative_log_mean_residual']
        if params['features'] != features:
            raise AssertionError('Changed detector feature definition')
        values = expected[features].to_numpy()
        train = expected.period.eq('calibration').to_numpy()
        for width, name, score_column, event_column in [
                (3, 'three_feature', 'distance', 'candidate_event'),
                (2, 'two_feature', 'distance_without_duration', 'candidate_without_duration')]:
            fit, scores = reference_detector(values[train, :width], values[:, :width])
            check_json_values(params[name], fit)
            check_json_values(params[name], {'calibration_rows': int(train.sum()), 'quantile_method': 'linear'})
            np.testing.assert_allclose(params[name]['condition_number'], np.linalg.cond(fit['covariance_z']), rtol=1e-9)
            expected[score_column] = scores
            expected[event_column] = scores >= fit['threshold']
        z = (values-np.asarray(params['three_feature']['mean']))/np.asarray(params['three_feature']['sample_std'])
        expected[['z_ofi', 'z_intensity', 'z_duration']] = z
        scored = pd.read_parquet(directory/'scored_bins.parquet')
        compare_numeric_frame(scored, expected, rtol=1e-8, atol=1e-8)
        # Selection must be exact for the saved arithmetic, including ties.
        np.testing.assert_array_equal(scored.candidate_event, scored.distance >= params['three_feature']['threshold'])
        np.testing.assert_array_equal(scored.candidate_without_duration,
                                      scored.distance_without_duration >= params['two_feature']['threshold'])
        for name, table in reference_tables(scored).items():
            compare_numeric_frame(pd.read_parquet(directory/(name+'.parquet')), table, rtol=1e-8, atol=1e-8)
            counts['tables_verified'] += 1
        statistics = json.loads((directory/'statistics.json').read_text())
        check_json_values(statistics, {'policy': policy, 'cross_cutoff_completed_waits_included': cross,
                                      'three_feature_threshold': params['three_feature']['threshold'],
                                      'two_feature_threshold': params['two_feature']['threshold'],
                                      'acd_constraint_warning': model['numerical_constraint_warning']})
        for label in ['calibration', 'development']:
            o = opportunity.loc[opportunity.period.eq(label)]
            s = scored.loc[scored.period.eq(label)]
            a = s.candidate_event.to_numpy(dtype=bool)
            b = s.candidate_without_duration.to_numpy(dtype=bool)
            union = np.count_nonzero(a | b)
            intersection = np.count_nonzero(a & b)
            event_rows = s.loc[s.candidate_event]
            if len(event_rows):
                event_time = ns(event_rows.bin_start)
                runs = 1+int(np.count_nonzero((np.diff(event_time) != SECOND) | (np.diff(event_rows.support_id) != 0)))
            else:
                runs = 0
            expected_stats = {'clock_seconds': len(o), 'attrition': {r: int(np.count_nonzero(o.reason == r)) for r in reasons},
                              'eligible_without_duration': int(np.count_nonzero(o.eligible_without_duration)),
                              'eligible_bins': len(s), 'candidate_bins': int(np.count_nonzero(a)),
                              'candidate_rate': float(np.count_nonzero(a)/len(s)),
                              'candidate_without_duration_bins': int(np.count_nonzero(b)),
                              'candidate_overlap_bins': int(intersection), 'candidate_union_bins': int(union),
                              'candidate_jaccard': float(intersection/union) if union else None,
                              'consecutive_exceedance_runs': runs,
                              'threshold_ties': int(np.count_nonzero(s.distance == params['three_feature']['threshold'])),
                              'candidate_pre_history_missing': int(event_rows.pre_rv_300s_5s_bps.isna().sum())}
            check_json_values(statistics['periods'][label], expected_stats)
        if statistics != summary['policies'][policy]:
            raise AssertionError('Root/policy statistics disagree')
        counts['clock_rows_verified'] += len(opportunity)
        counts['scored_bins_verified'] += len(scored)
        print('Independently verified '+policy+': '+str(len(scored))+' scored bins', flush=True)
        del opportunity, scored, expected, values
    check_json_values(summary, {'training_start': start.isoformat(), 'parameter_cutoff': cutoff.isoformat(),
                                'development_end_exclusive': stop.isoformat(),
                                'production_events_approved': False, 'primary_freshness_policy': None})
    return {'status': 'passed', **counts,
            'methods': 'Pandas strict backward joins; raw covariance inverse; explicit linear empirical quantile; independent reset filtering and endpoint resampling; row-identity aggregate and metric-matched reference summaries.',
            'production_events_approved': False}

"""Independent reconstruction of the first chronological recalibration transition.

No production transition, duration-calibration, detector or episode builder is
imported.  Prior independent reference implementations supply incoming-spell
construction, historical exposure clipping, SciPy filtering, raw covariance
arithmetic and graph-based episode reconstruction.
"""
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd

from src.verify_chronological_acd import (
    reference_spells, reference_scales, reference_sample,
    reference_predictions, check_diagnostics,
)
from src.verify_event_audit import (
    reference_detector, reference_duration_bins, reference_clock,
    reference_quote_diagnostics, compare_numeric_frame, check_json_values,
)
from src.verify_event_episodes import (
    reference_episode_tables, reference_quote_quality,
    reference_policy_statistics, reference_daily, QUALITY_COLUMNS,
)

SECOND = 1_000_000_000
DAY = 86400 * SECOND
POLICY = 'structural_only'
FEATURES = ['adjusted_abs_quote_ofi', 'adjusted_tick_count',
            'negative_log_mean_residual']
METRICS = ['pre_spread_pips', 'end_spread_pips', 'spread_change_pips',
           'pre_displayed_size', 'end_displayed_size', 'log_size_ratio',
           'pre_quote_age_sec', 'end_quote_age_sec',
           'pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']
REASONS = ('unsupported', 'invalid_flow', 'seasonal_unavailable',
           'quiet_no_duration', 'incomplete_duration', 'eligible')


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _ns(values):
    index = pd.DatetimeIndex(values)
    _require(str(index.tz) == 'UTC' and not index.hasnans,
             'Nonmissing UTC timestamps required')
    return index.as_unit('ns').asi8


def _frame(actual, expected, name, *, exact_columns=True):
    if exact_columns:
        _require(set(actual.columns) == set(expected.columns),
                 name + ': columns differ ' + repr(set(actual.columns) ^ set(expected.columns)))
    _require(len(actual) == len(expected), name + ': row count differs')
    for column in expected:
        _require(actual[column].isna().reset_index(drop=True).equals(
            expected[column].isna().reset_index(drop=True)),
            name + '.' + column + ': missingness differs')
    for column in expected:
        if pd.api.types.is_integer_dtype(expected[column].dtype) or pd.api.types.is_bool_dtype(expected[column].dtype):
            np.testing.assert_array_equal(actual[column].to_numpy(), expected[column].to_numpy(),
                                          err_msg=name + '.' + column)
    compare_numeric_frame(actual, expected, rtol=1e-9, atol=1e-9)


def _registered_tree(run):
    """Stream each physical file once and verify all registered parent runs.

    Parent manifests are discovered only through registered paths.  An input
    that names a parent output must match that parent's own output registration;
    all repeated references to the same physical path must agree.
    """
    cache, expected_by_path, manifests = {}, {}, {}
    contextual_audits = set()
    registrations = 0

    def check(path, expected):
        nonlocal registrations
        path = Path(path).resolve()
        key = str(path)
        if isinstance(expected, dict):
            expected = expected['sha256']
        _require(key not in expected_by_path or expected_by_path[key] == expected,
                 'Conflicting hash registrations: ' + key)
        expected_by_path[key] = expected
        if key not in cache:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            cache[key] = h.hexdigest()
        _require(cache[key] == expected, 'Registered file changed: ' + key)
        registrations += 1

    def visit(directory, *, root=False):
        directory = Path(directory).resolve()
        key = str(directory)
        if key in manifests:
            return
        m = _read(directory / 'manifest.json')
        manifests[key] = m
        if not root:
            _require(str(m.get('status', '')).startswith('verified'),
                     'Unverified ancestor: ' + key)
            verification = _read(directory / 'verification.json')
            _require(verification.get('status') == 'passed',
                     'Unpassed ancestor verification: ' + key)
        parents = []
        for entry in m.get('inputs', m.get('input_hashes', {})).values():
            check(entry['path'], entry['sha256'])
            if Path(entry['path']).name == 'manifest.json':
                parents.append(Path(entry['path']).parent)
        for entry in m.get('input_files', []):
            check(entry['path'], entry['sha256'])
        for name, value in m.get('outputs', m.get('output_sha256', {})).items():
            check(directory / name, value)
        for name, value in m.get('code_sha256', m.get('code_files', {})).items():
            check(directory / 'code' / name, value)
        for name, value in m.get('postfit_code_sha256', {}).items():
            check(directory / 'postfit_code' / name, value)
        for name, value in m.get('ancestor_hashes', {}).items():
            check(name, value)
            if Path(name).name == 'manifest.json':
                parents.append(Path(name).parent)
        for name, value in m.get('previous_files', {}).items():
            check(Path(m['snapshot_path']) / name, value)
        for parent in parents:
            context = _read(Path(parent) / 'manifest.json')
            if context.get('stage') == '02a_whole_hour_coverage':
                _require(context.get('status') == 'coverage_audited_recovery_waiting_for_vendor_access'
                         and context.get('coverage_certified') is False,
                         'Coverage-context scope changed')
                _require(_read(Path(parent) / 'coverage_verification.json').get('status') == 'passed',
                         'Coverage-context timestamp verification did not pass')
                contextual_audits.add(str(Path(parent).resolve()))
            else:
                visit(parent)

    visit(run, root=True)
    root = manifests[str(Path(run).resolve())]
    output_owners = {}
    for directory, m in manifests.items():
        for name in m.get('outputs', m.get('output_sha256', {})):
            output_owners[str((Path(directory) / name).resolve())] = directory
    required = ['quote_durations', 'terminal_spells', 'clock',
                'calibration_parameters', 'clean_quotes', 'support_intervals',
                'previous_predictions', 'previous_seasonal_spells',
                'previous_opportunities', 'previous_scored',
                'previous_onsets', 'previous_episodes', 'frozen_regression_scaling']
    for key in required:
        _require(key in root['inputs'], 'Missing input registration: ' + key)
        path = str(Path(root['inputs'][key]['path']).resolve())
        _require(path in output_owners, 'Input is not a registered parent output: ' + key)
    records = root['inputs']
    named = {}
    for key in ('event', 'duration', 'acd', 'episode', 'regression'):
        record = records[key + '_manifest_json']
        directory = str(Path(record['path']).parent.resolve())
        _require(directory in manifests, 'Missing named parent: ' + key)
        named[key] = manifests[directory]

    def same(actual, expected, label):
        _require(Path(actual['path']).resolve() == Path(expected['path']).resolve() and
                 actual['sha256'] == expected['sha256'], 'Inconsistent parent lineage: ' + label)

    event = named['event']
    for key in ('clock', 'calibration_parameters', 'clean_quotes', 'support_intervals'):
        same(records[key], event['inputs'][key], key)
    same(records['previous_predictions'], event['inputs']['saved_predictions_structural_only'], 'issued forecasts')
    same(records['previous_seasonal_spells'], event['inputs']['seasonal_spells_structural_only'], 'origin scales')
    _require(Path(event['inputs']['acd_statistics']['path']).parent.resolve() ==
             Path(records['acd_manifest_json']['path']).parent.resolve(), 'Event ACD parent differs')
    for name in ('duration', 'acd'):
        same(records['clean_quotes'], named[name]['inputs']['clean'], name + ' clean quotes')
        same(records['support_intervals'], named[name]['inputs']['support'], name + ' observation support')
    for key in ('quote_durations', 'terminal_spells'):
        same(records[key], named['acd']['inputs'][key], key)
    _require(named['episode']['parent_event_audit'] == event['run_id'], 'Episode detector ancestor differs')
    for key, old in [('previous_opportunities', 'structural_only_opportunities'),
                     ('previous_scored', 'structural_only_scored')]:
        same(records[key], named['episode']['inputs'][old], key)
    for key in ('clean_quotes', 'support_intervals'):
        same(records[key], named['episode']['inputs'][key], 'episode ' + key)
    _require(named['regression']['parent_reference_detector'] == event['run_id'],
             'Frozen regression scales use another detector ancestor')
    same(records['previous_scored'], named['regression']['inputs']['reference_bins'], 'regression scaling reference')
    same(records['episode_manifest_json'], named['regression']['inputs']['episode_manifest'], 'regression episode ancestor')
    for key, owner in [('previous_onsets', 'episode'), ('previous_episodes', 'episode'),
                        ('frozen_regression_scaling', 'regression'),
                        ('quote_durations', 'duration'), ('terminal_spells', 'duration')]:
        _require(output_owners[str(Path(records[key]['path']).resolve())] ==
                 str(Path(records[owner + '_manifest_json']['path']).parent.resolve()),
                 'Named parent does not own input: ' + key)
    return root, {'hash_registrations_verified': registrations,
                  'unique_files_hashed': len(cache),
                  'parent_runs_verified': len(manifests) - 1,
                  'contextual_coverage_audits_hash_checked': len(contextual_audits)}


def _duration_reference(run, manifest, parameters):
    inputs = manifest['inputs']
    origin = pd.Timestamp(manifest['calendar_origin']).value
    start = pd.Timestamp(manifest['training_start']).value
    cutoff = pd.Timestamp(manifest['parameter_cutoff']).value
    old_cutoff = pd.Timestamp(manifest['previous_cutoff']).value
    stop = pd.Timestamp(manifest['deployment_end']).value
    cfg = manifest['seasonal_configuration']
    _require((origin, start, old_cutoff, cutoff, stop) == tuple(pd.Timestamp(t, tz='UTC').value
        for t in ['2024-01-01', '2024-03-11', '2024-04-22', '2024-05-06', '2024-05-20']),
        'This verifier certifies only the frozen first transition calendar')
    _require(cutoff - start == 56 * DAY and stop - cutoff == 14 * DAY,
             'Chronological 56/14 calendar changed')
    _require(cutoff - old_cutoff == 14 * DAY,
             'Previous deployment is not immediately preceding')
    _require((cfg['seasonal_days'], cfg['training_days'], cfg['validation_days'],
              cfg['min_prior_dates'], cfg['min_exposure_seconds'], cfg['min_completions']) ==
             (56, 56, 28, 4, 1800, 100), 'Seasonal configuration changed')
    _require(origin + 140 * DAY == stop, 'Seasonal domain differs')
    for key, timestamp in [('training_start', start), ('parameter_cutoff', cutoff),
                           ('deployment_end', stop)]:
        _require(pd.Timestamp(parameters[key]).value == timestamp,
                 'Parameter calendar differs: ' + key)
    tails = pd.read_parquet(inputs['terminal_spells']['path'])
    raw = reference_spells(inputs['quote_durations']['path'], tails, POLICY, stop)
    del tails
    saved = pd.read_parquet(run / 'seasonal_spells.parquet')
    _frame(saved[list(raw.columns)], raw, 'raw spells')
    day = (_ns(raw.origin_at) - origin) // DAY
    slot = (_ns(raw.origin_at) % DAY) // (900 * SECOND)
    grouped = raw.assign(origin_day=day, slot=slot)
    grid = pd.MultiIndex.from_product([range(140), range(96)], names=['origin_day', 'slot'])
    sufficient = grouped.groupby(['origin_day', 'slot']).agg(
        origins=('origin_quote_id', 'size'), completed_events=('event_observed', 'sum'),
        full_exposure_ns=('exposure_ns', 'sum')).reindex(grid, fill_value=0).reset_index()
    _frame(pd.read_parquet(run / 'origin_daily_sufficient_statistics.parquet'), sufficient,
           'origin-day sufficient statistics')
    rows = []
    begin, end = _ns(raw.origin_at), _ns(raw.end_at)
    observed = raw.event_observed.to_numpy()
    for index in np.flatnonzero(end >= origin + (day + 1) * DAY):
        for date in range(int(day[index]) + 1, min(139, int((end[index] - origin) // DAY)) + 1):
            rows.append((date, int(day[index]), int(slot[index]),
                         int(end[index] - origin - date * DAY), int(observed[index])))
    correction = pd.DataFrame(rows, columns=['cutoff_day', 'origin_day', 'slot',
                                           'future_exposure_ns', 'future_completions'])
    if len(correction):
        correction = correction.groupby(['cutoff_day', 'origin_day', 'slot'], as_index=False, sort=True).sum()
    _frame(pd.read_parquet(run / 'historical_cutoff_corrections.parquet'), correction,
           'historical completion/exposure corrections')
    del grouped, sufficient, correction, rows, day, slot, begin, end, observed
    scales = reference_scales(raw, origin, cfg)
    _frame(pd.read_parquet(run / 'seasonal_parameters.parquet'), scales, 'past-only duration seasons')
    dates = (_ns(raw.origin_at) - origin) // DAY
    slots = (_ns(raw.origin_at) % DAY) // (900 * SECOND)
    keys = dates * 96 + slots
    expected = raw.copy()
    expected['seasonal_status'] = scales.status.to_numpy()[keys]
    expected['scale_sec'] = scales.working_scale_sec.to_numpy()[keys]
    expected['normalized_exposure'] = raw.exposure_ns.to_numpy() / SECOND / expected.scale_sec.to_numpy()
    expected['seasonal_cutoff'] = pd.to_datetime(origin + dates * DAY, utc=True)
    _frame(saved, expected, 'seasonally normalized spells')
    counts = {'source_spell_rows_verified': len(raw), 'seasonal_rows_verified': len(scales)}
    counts.update(source_completed_events=int(raw.event_observed.sum()),
                  source_exposure_ns=int(raw.exposure_ns.sum()),
                  seasonal_ready_spell_rows=int(expected.scale_sec.notna().sum()))
    del saved, raw, scales, dates, slots, keys
    training = reference_sample(expected, start, cutoff)
    _frame(pd.read_parquet(run / 'training_spells.parquet'), training, 'historically clipped training')
    path = reference_sample(expected, start, stop)
    _frame(pd.read_parquet(run / 'model_path.parquet'), path, 'continuous new-parameter filter path')
    del expected
    pmap = parameters['acd_parameters']
    p = np.asarray([pmap[k] for k in ('omega', 'alpha', 'beta', 'shape')], dtype=float)
    _require(np.isfinite(p).all() and p[0] > 0 and min(p[1:3]) >= 0 and
             p[1] + p[2] < .999 and .2 <= p[3] <= 5,
             'Inadmissible fitted Weibull ACD parameters')
    attempts = parameters['optimization_attempts']
    _require(attempts == _read(run / 'optimization_attempts.json'), 'Optimization log differs')
    _require(len(attempts) == 3, 'Three predeclared numerical starts required')
    for attempt, fixed in zip(attempts, [(.05, .85, .7), (.15, .70, 1.), (.30, .50, .5)], strict=True):
        np.testing.assert_allclose([attempt[k] for k in ('start_alpha', 'start_beta', 'start_shape')], fixed,
                                   rtol=0, atol=0)
    successful = [attempt for attempt in attempts if attempt['success'] and np.isfinite(attempt['objective'])]
    _require(bool(successful), 'No successful ACD numerical fit')
    best = min(successful, key=lambda attempt: attempt['objective'])
    np.testing.assert_array_equal(p, [best[k] for k in ('omega', 'alpha', 'beta', 'shape')])
    warning = bool(p[1] + p[2] > .999 - 1e-5 or p[3] < .2001 or p[3] > 4.999)
    _require(parameters['numerical_constraint_warning'] == best['near_constraint'] == warning,
             'Numerical constraint warning differs')
    fitted = reference_predictions(training, p, weibull=True)
    objective = float((fitted.negative_log_score_seconds -
                       training.event_observed * np.log(training.scale_sec)).mean())
    np.testing.assert_allclose([parameters['acd_fit_objective'], best['objective']], objective,
                               rtol=1e-9, atol=1e-9)
    counts.update(training_spell_rows_verified=len(training),
                  training_completed_events=int(training.event_observed.sum()),
                  training_censored_spells=int((~training.event_observed).sum()))
    del fitted, training
    prediction = reference_predictions(path, p, weibull=True)
    forecast = pd.read_parquet(run / 'forecasts.parquet')
    np.testing.assert_array_equal(forecast.origin_quote_id, path.origin_quote_id)
    np.testing.assert_allclose(forecast.psi_refiltered, prediction.psi, rtol=1e-10, atol=1e-10)
    actual = prediction.psi.to_numpy().copy()
    a, b = _ns(path.origin_at), _ns(path.end_at)
    carry = (a < cutoff) & (b >= cutoff)
    prior = pd.read_parquet(inputs['previous_predictions']['path'])
    prior_scales = pd.read_parquet(inputs['previous_seasonal_spells']['path'],
                                  columns=['origin_quote_id', 'origin_at', 'component_id', 'scale_sec'])
    _require(not prior.origin_quote_id.duplicated().any(), 'Duplicate prior forecast identities')
    _require(not prior_scales.origin_quote_id.duplicated().any(), 'Duplicate prior scale identities')
    prior = prior.set_index('origin_quote_id')
    prior_scales = prior_scales.set_index('origin_quote_id')
    carry_ids = path.loc[carry, 'origin_quote_id'].to_numpy()
    _require((a[carry] >= old_cutoff).all(), 'Carry is older than the verified deployment')
    _require(np.isin(carry_ids, prior.index).all(), 'Missing prior issued forecast')
    _require(np.isin(carry_ids, prior_scales.index).all(), 'Missing prior origin-scale record')
    if len(carry_ids):
        previous = prior_scales.loc[carry_ids]
        np.testing.assert_array_equal(_ns(previous.origin_at), a[carry])
        np.testing.assert_array_equal(previous.component_id, path.loc[carry, 'component_id'])
        np.testing.assert_allclose(previous.scale_sec, path.loc[carry, 'scale_sec'], rtol=0, atol=0)
        actual[carry] = prior.loc[carry_ids, 'psi'].to_numpy()
    np.testing.assert_allclose(forecast.psi_for_residual, actual, rtol=1e-10, atol=1e-10)
    vintage = np.full(len(path), cutoff, dtype='int64')
    vintage[carry] = old_cutoff
    np.testing.assert_array_equal(_ns(forecast.residual_parameter_cutoff), vintage)
    roles = np.where(carry, 'carried_origin_forecast',
                     np.where(a >= cutoff, 'new_origin_forecast', 'calibration_refilter'))
    np.testing.assert_array_equal(forecast.forecast_role, roles)
    _require(set(forecast.columns) == {'origin_quote_id', 'psi_refiltered',
        'psi_for_residual', 'residual_parameter_cutoff', 'forecast_role'},
        'Forecast audit schema differs')
    bridge = path.loc[carry].reset_index(drop=True).copy()
    for name in forecast.columns[1:]:
        bridge[name] = forecast.loc[carry, name].to_numpy()
    _frame(pd.read_parquet(run / 'boundary_spells.parquet'), bridge, 'boundary spell audit')
    bins = reference_duration_bins(path, actual)
    _frame(pd.read_parquet(run / 'duration_bins.parquet'), bins, 'completed duration residual bins')
    counts.update(model_path_rows_verified=len(path), forecast_rows_verified=len(forecast),
                  carried_forecast_rows_verified=int(carry.sum()),
                  carried_observed_completions=int((carry & path.event_observed.to_numpy()).sum()),
                  completed_exactly_at_cutoff=int((path.event_observed.to_numpy() & (b == cutoff)).sum()),
                  duration_bins_verified=len(bins))
    new = a >= cutoff
    check_diagnostics(path.loc[new].reset_index(drop=True), prediction.loc[new].reset_index(drop=True),
                      _read(run / 'statistics.json')['duration_working_model_diagnostics'])
    del prior, prior_scales, forecast, path, prediction
    gc.collect()
    return bins, counts


def _feature_reference(run, manifest, parameters, duration):
    inputs = manifest['inputs']
    start, cutoff, stop = [pd.Timestamp(manifest[k]) for k in
                           ('training_start', 'parameter_cutoff', 'deployment_end')]
    raw = pd.read_parquet(inputs['clock']['path'],
                          filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    times = _ns(raw.bin_start)
    np.testing.assert_array_equal(times, np.arange(start.value, stop.value, SECOND))
    np.testing.assert_array_equal(_ns(raw.available_at), times + SECOND)
    factors = pd.read_parquet(inputs['calibration_parameters']['path'],
        filters=[('effective_date', '>=', start), ('effective_date', '<', stop), ('policy', '==', POLICY)])
    clock = reference_clock(raw, factors, POLICY)
    joined = pd.DataFrame({'bin_start': clock.bin_start}).merge(duration, how='left',
                      on='bin_start', sort=False, validate='one_to_one')
    recorded = raw.recorded_update_count.to_numpy(dtype='int64')
    completed = joined.completed_residual_count.fillna(0).to_numpy(dtype='int64')
    _require((completed >= 0).all() and (completed <= recorded).all(),
             'Completion count is incompatible with recorded updates')
    ready = np.isfinite(clock[['adjusted_abs_quote_ofi', 'adjusted_tick_count']]).all(axis=1).to_numpy()
    status = clock.flow_status.to_numpy()
    reason = np.full(len(raw), 'eligible', dtype=object)
    reason[completed != recorded] = 'incomplete_duration'
    reason[recorded == 0] = 'quiet_no_duration'
    reason[~ready] = 'seasonal_unavailable'
    reason[~np.isin(status, [1, 2])] = 'invalid_flow'
    reason[status == 0] = 'unsupported'
    opportunities = pd.DataFrame({'bin_start': clock.bin_start,
        'available_at': clock.available_at,
        'period': np.where(times < cutoff.value, 'calibration', 'development'),
        'recorded_update_count': recorded, 'support_id': clock.support_id,
        'flow_status': status, 'seasonal_ready': ready,
        'completed_residual_count': completed,
        'eligible_without_duration': np.isin(status, [1, 2]) & ready,
        'reason': reason})
    selected = reason == 'eligible'
    scored = clock.loc[selected, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
        'adjusted_abs_quote_ofi', 'adjusted_tick_count']].reset_index(drop=True)
    scored['recorded_update_count'] = recorded[selected]
    scored['completed_residual_count'] = completed[selected]
    scored['mean_duration_residual'] = joined.mean_duration_residual.to_numpy()[selected]
    scored['negative_log_mean_residual'] = -np.log(scored.mean_duration_residual)
    scored['period'] = opportunities.period.to_numpy()[selected]
    quotes = pd.read_parquet(inputs['clean_quotes']['path'],
        columns=['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume'],
        filters=[('timestamp', '>=', start-pd.Timedelta(days=1)), ('timestamp', '<', stop)])
    support = pd.read_parquet(inputs['support_intervals']['path'])
    support = support.loc[support.policy.eq(POLICY)].reset_index(drop=True)
    economic = reference_quote_diagnostics(clock, quotes, support)
    for name in METRICS:
        scored[name] = economic[name].to_numpy()[selected]
    training = scored.period.eq('calibration').to_numpy()
    reference, distance = reference_detector(scored.loc[training, FEATURES], scored[FEATURES])
    fit = parameters['detector']
    for key in ('mean', 'sample_std', 'covariance_z', 'threshold', 'quantile'):
        np.testing.assert_allclose(fit[key], reference[key], rtol=1e-10, atol=1e-10,
                                   err_msg='Detector ' + key)
    eigenvalues = np.linalg.eigvalsh(reference['covariance_z'])
    _require(eigenvalues[0] > 0 and eigenvalues[-1] / eigenvalues[0] <= 1e10,
             'Inadmissible detector covariance')
    np.testing.assert_allclose(fit['condition_number'], eigenvalues[-1] / eigenvalues[0], rtol=1e-10)
    _require(fit['quantile_method'] == 'linear' and fit['calibration_rows'] == int(training.sum()),
             'Detector calibration recipe differs')
    scored['distance'] = distance
    scored['candidate_event'] = distance >= reference['threshold']
    z = (scored[FEATURES].to_numpy() - reference['mean']) / reference['sample_std']
    for index, name in enumerate(['z_ofi', 'z_intensity', 'z_duration']):
        scored[name] = z[:, index]
    _require(fit['training_threshold_ties'] == int(np.count_nonzero(distance[training] == reference['threshold'])),
             'Threshold tie count differs')
    partitions, partition_stats = {}, {}
    for period, prefix in [('calibration', 'calibration'), ('development', 'deployment')]:
        o = opportunities.loc[opportunities.period.eq(period)].reset_index(drop=True)
        s = scored.loc[scored.period.eq(period)].reset_index(drop=True)
        _frame(pd.read_parquet(run / (prefix + '_opportunities.parquet')), o, prefix + ' opportunities')
        _frame(pd.read_parquet(run / (prefix + '_scored.parquet')), s, prefix + ' scored features')
        partitions[prefix] = o, s
        partition_stats[prefix] = _partition_statistics(o, s)
    counts = {'feature_clock_rows_verified': len(clock), 'feature_rows_verified': len(scored),
              'calibration_scored_rows_verified': int(training.sum()),
              'deployment_scored_rows_verified': int((~training).sum())}
    del raw, clock, factors, joined, opportunities, scored, economic
    gc.collect()
    return partitions['deployment'], partition_stats, quotes, support, counts


def _partition_statistics(opportunities, scored):
    counts = {name: int(np.count_nonzero(opportunities.reason.to_numpy() == name)) for name in REASONS}
    _require(sum(counts.values()) == len(opportunities), 'Opportunity reasons fail conservation')
    _require(counts['eligible'] == len(scored), 'Eligible rows fail conservation')
    candidates = int(np.count_nonzero(scored.candidate_event.to_numpy()))
    return {'calendar_seconds': len(opportunities), 'eligible_bins': len(scored),
            'candidate_bins': candidates,
            'candidate_share_of_eligible': candidates / len(scored) if len(scored) else None,
            'reason_counts': counts}


def _episode_reference(run, manifest, deployment, quotes, support):
    inputs = manifest['inputs']
    start, cutoff, stop = [pd.Timestamp(manifest[k]) for k in
                           ('previous_cutoff', 'parameter_cutoff', 'deployment_end')]
    new_opportunities, new_scored = deployment
    old_opportunities = pd.read_parquet(inputs['previous_opportunities']['path'],
                                       filters=[('period', '==', 'development')])
    old_scored = pd.read_parquet(inputs['previous_scored']['path'],
                                filters=[('period', '==', 'development')], columns=list(new_scored.columns))
    np.testing.assert_array_equal(_ns(old_opportunities.bin_start), np.arange(start.value, cutoff.value, SECOND))
    np.testing.assert_array_equal(_ns(new_opportunities.bin_start), np.arange(cutoff.value, stop.value, SECOND))
    combined_o = pd.concat([old_opportunities, new_opportunities], ignore_index=True)
    combined_s = pd.concat([old_scored, new_scored], ignore_index=True)
    saved_o = pd.read_parquet(run / 'combined_opportunities.parquet')
    saved_s = pd.read_parquet(run / 'combined_scored.parquet')
    _frame(saved_o, combined_o, 'combined opportunity clock')
    _frame(saved_s, combined_s, 'combined baseline scores')
    pd.testing.assert_frame_equal(saved_o.iloc[:len(old_opportunities)].reset_index(drop=True),
                                  old_opportunities, check_exact=True, check_dtype=False, check_categorical=False)
    pd.testing.assert_frame_equal(saved_s.iloc[:len(old_scored)].reset_index(drop=True),
                                  old_scored, check_exact=True, check_dtype=False, check_categorical=False)
    clock, membership, episodes, onsets = reference_episode_tables(combined_o, combined_s, start, stop)
    for name, expected in [('episode_clock', clock), ('candidate_membership', membership), ('episodes', episodes)]:
        _frame(pd.read_parquet(run / (name + '.parquet')), expected, name)
    quality = reference_quote_quality(onsets, quotes, support)
    for name in QUALITY_COLUMNS:
        onsets[name] = quality[name].to_numpy()
    saved_onsets = pd.read_parquet(run / 'onsets.parquet')
    _frame(saved_onsets, onsets, 'episode onset history and quality')
    old_onsets = pd.read_parquet(inputs['previous_onsets']['path'], columns=list(saved_onsets.columns))
    prefix = saved_onsets.loc[saved_onsets.bin_start.lt(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, old_onsets, check_exact=True, check_dtype=False, check_categorical=False)
    old_episodes = pd.read_parquet(inputs['previous_episodes']['path'])
    old_prefix = episodes.loc[episodes.first_candidate_bin.lt(cutoff)].reset_index(drop=True)
    immutable = ['episode_id', 'support_id', 'first_candidate_bin', 'detected_at',
                 'onset_confirmed', 'preceding_clear_seconds']
    _frame(old_prefix[immutable], old_episodes[immutable], 'old onset identities')
    closed = old_episodes.termination.ne('censored_sample_end')
    _frame(old_prefix.loc[closed].reset_index(drop=True), old_episodes.loc[closed].reset_index(drop=True),
           'previously terminated episodes')
    daily = reference_daily(membership, onsets, start, stop)
    _frame(pd.read_parquet(run / 'daily.parquet'), daily, 'daily episode conservation')
    _require(int(daily.candidate_bins.sum()) == len(membership) == int(episodes.candidate_bins.sum()),
             'Candidate conservation differs across daily/episode/membership tables')
    new_onsets = onsets.loc[onsets.bin_start.ge(cutoff)]
    cross = episodes.first_candidate_bin.lt(cutoff) & episodes.last_candidate_bin.ge(cutoff)
    stats = {'combined_episodes': reference_policy_statistics(clock, episodes, onsets),
             'previous_onsets_preserved': len(prefix), 'new_onset_fragments': len(new_onsets),
             'new_confirmed_onsets': int(new_onsets.onset_confirmed.sum()),
             'new_state_eligible_onsets': int(new_onsets.state_sample_eligible.sum()),
             'episodes_with_candidates_across_cutoff': int(cross.sum())}
    counts = {'combined_clock_rows_verified': len(clock), 'candidate_rows_verified': len(membership),
              'episode_rows_verified': len(episodes), 'onset_rows_verified': len(onsets),
              'pre_cutoff_onsets_preserved_verified': len(prefix),
              'raw_quote_quality_rows_verified': int(onsets.state_history_available.sum()),
              'daily_rows_verified': len(daily)}
    del combined_o, combined_s, saved_o, saved_s, clock, membership, episodes, onsets
    gc.collect()
    return stats, counts


def verify_transition(run):
    """Verify all stage-07b numerical artifacts without fitting another model."""
    run = Path(run)
    manifest, counts = _registered_tree(run)
    _require(manifest['primary_policy'] == POLICY and
             manifest['scope'] == 'first_chronological_transition_only' and
             manifest['production_inference_approved'] is False and manifest['downstream_ready'] is False,
             'Unapproved pilot scope or downstream promotion')
    evidence = _read(run / 'test_evidence.json')
    _require(evidence['status'] == 'passed' and evidence['failures'] == evidence['errors'] == 0,
             'Tests did not pass')
    for name, expected in evidence['test_source_sha256'].items():
        path = run / 'full_test_sources' / name
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 'Saved test source differs from executed test: ' + name)
    cutoffs = pd.to_datetime(['2024-04-22', '2024-05-06'], utc=True)
    schedule = pd.DataFrame({'cycle_id': [0, 1], 'training_start': cutoffs-pd.Timedelta(days=56),
                             'parameter_cutoff': cutoffs, 'deployment_end': cutoffs+pd.Timedelta(days=14)})
    _frame(pd.read_parquet(run / 'cycle_schedule.parquet'), schedule, 'fixed deployment schedule')
    parameters = _read(run / 'parameters.json')
    statistics = _read(run / 'statistics.json')
    duration, duration_counts = _duration_reference(run, manifest, parameters)
    counts.update(duration_counts)
    deployment, partitions, quotes, support, feature_counts = _feature_reference(run, manifest, parameters, duration)
    counts.update(feature_counts)
    del duration
    episode_stats, episode_counts = _episode_reference(run, manifest, deployment, quotes, support)
    counts.update(episode_counts)
    expected = {'scope': 'first_chronological_transition_only', 'primary_policy': POLICY,
        'source_spell_rows': counts['source_spell_rows_verified'],
        'seasonal_parameter_rows': counts['seasonal_rows_verified'],
        'training_spells': counts['training_spell_rows_verified'],
        'training_events': counts['training_completed_events'],
        'forecast_rows': counts['forecast_rows_verified'],
        'carried_forecasts': counts['carried_forecast_rows_verified'],
        'carried_completed_residuals': counts['carried_observed_completions'],
        **partitions, **episode_stats,
        'fitted_duration_specifications': 1, 'empirical_response_models_fitted': 0,
        'full_panel_extension_performed': False, 'extended_responses_computed': False,
        'production_inference_approved': False, 'downstream_ready': False}
    check_json_values(statistics, expected)
    return {'status': 'passed', **counts,
        'scope': 'First structural-only chronological transition; no later cycles, response estimates or inference certified.',
        'methods': [
            'Recursive verified measurement-parent input/output/source hashes and shared-path lineage; historical coverage context hashes checked without certifying vendor recovery',
            'Independent incoming completed waits plus separate terminal spells and direct past-only same-weekday exposure clipping',
            'Historically censored training and SciPy Weibull likelihood/ACD filtering with origin-identity forecast carry',
            'Raw-feature covariance and sorted empirical q99, independent eligibility partitions and prior-quote measurements',
            'Unchanged old baseline rows; graph-connected candidates over the complete joined clock with true support barriers',
            'Independent onset quality, immutable previous onset histories and calendar/episode/candidate conservation'],
        'full_panel_extension_approved': False, 'production_inference_approved': False}

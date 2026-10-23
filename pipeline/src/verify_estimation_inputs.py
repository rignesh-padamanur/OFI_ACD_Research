"""Independent, outcome-free reconstruction of stage-07a estimation inputs.

This module imports no production estimation-input implementation.  It reads
the immutable parent tables, rebuilds transformations and sample masks, and
checks design geometry without fitting any response.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


REFERENCE_COLUMNS = ['bin_start', 'available_at', 'period',
                     'pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']
CALENDAR_COLUMNS = ['hour_%02d' % h for h in range(1, 24)] + [
    'weekday_%d' % d for d in range(1, 7)]
SCIENCE_COLUMNS = ['intercept', 'direction', 'variation_standardized',
                   'pressure_magnitude', 'aligned_prior_pressure',
                   'variation_x_magnitude', 'variation_x_alignment',
                   'pre_spread_pips', 'log_pre_displayed_size']
BASELINE_COLUMNS = SCIENCE_COLUMNS + CALENDAR_COLUMNS
CURVATURE_COLUMNS = BASELINE_COLUMNS + ['magnitude_squared', 'variation_squared']


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _registered_hashes(run, cache, *, include_inputs=True):
    """Check registered files using streaming digests, including saved sources."""
    run = Path(run)
    manifest = _read(run / 'manifest.json')

    def digest(path):
        path = Path(path)
        key = str(path.resolve())
        if key not in cache:
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(chunk)
            cache[key] = value.hexdigest()
        return cache[key]

    count = 0
    for key, entry in (manifest.get('inputs', {}) if include_inputs else {}).items():
        _require(digest(entry['path']) == entry['sha256'], 'Changed input: ' + key)
        count += 1
    for key, expected in manifest.get('outputs', {}).items():
        _require(digest(run / key) == expected, 'Changed output: ' + key)
        count += 1
    for key, expected in manifest.get('code_sha256', {}).items():
        _require(digest(run / 'code' / key) == expected, 'Changed source snapshot: ' + key)
        count += 1
    for key, expected in manifest.get('postfit_code_sha256', {}).items():
        _require(digest(run / 'postfit_code' / key) == expected, 'Changed postfit source snapshot: ' + key)
        count += 1
    for key, expected in (manifest.get('ancestor_hashes', {}) if include_inputs else {}).items():
        _require(digest(key) == expected, 'Changed ancestor: ' + key)
        count += 1
    for key, expected in manifest.get('previous_files', {}).items():
        _require(digest(Path(manifest['snapshot_path']) / key) == expected,
                 'Changed previous-file snapshot: ' + key)
        count += 1
    return manifest, count


def _compare_frame(actual, expected, label):
    _require(set(actual.columns) == set(expected.columns),
             label + ': columns differ')
    _require(len(actual) == len(expected), label + ': row count differs')
    for key in expected:
        a, e = actual[key], expected[key]
        _require(a.isna().equals(e.isna()), label + '.' + key + ': missingness differs')
        if pd.api.types.is_float_dtype(e.dtype):
            np.testing.assert_allclose(a.to_numpy(dtype=float), e.to_numpy(dtype=float),
                                       rtol=1e-12, atol=1e-10, equal_nan=True,
                                       err_msg=label + '.' + key)
        elif pd.api.types.is_datetime64_any_dtype(e.dtype):
            ai, ei = pd.DatetimeIndex(a), pd.DatetimeIndex(e)
            _require(str(ai.tz) == str(ei.tz) == 'UTC', label + '.' + key + ': timezone')
            np.testing.assert_array_equal(ai.as_unit('ns').asi8, ei.as_unit('ns').asi8,
                                          err_msg=label + '.' + key)
        else:
            np.testing.assert_array_equal(a.to_numpy(), e.to_numpy(),
                                          err_msg=label + '.' + key)


def _compare_json(actual, expected, location='json', exact=True):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), location + ': dictionary required')
        if exact:
            _require(set(actual) == set(expected), location + ': keys differ ' +
                     repr(set(actual) ^ set(expected)))
        for key, value in expected.items():
            _require(key in actual, location + ': missing ' + key)
            _compare_json(actual[key], value, location + '.' + key, exact)
    elif isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                 location + ': list length/type differs')
        for index, value in enumerate(expected):
            _compare_json(actual[index], value, location + '.' + str(index), exact)
    elif expected is None or isinstance(expected, (str, bool, int)):
        _require(actual == expected, location + ': value differs')
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10,
                                   err_msg=location)


def _unique(frame, columns, label):
    _require(not frame[columns].isna().any().any(), label + ': missing identity')
    _require(not frame.duplicated(columns).any(), label + ': duplicated identity')


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'n': 0, 'min': None, 'p10': None, 'median': None,
                'p90': None, 'max': None, 'mean': None}
    return {'n': int(len(values)), 'min': float(np.min(values)),
            'p10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'max': float(np.max(values)),
            'mean': float(np.mean(values))}


def _reference_values(bins, start, cutoff):
    """Independent selection from calendar/time predicates and raw state values."""
    for key in ['bin_start', 'available_at']:
        index = pd.DatetimeIndex(bins[key])
        _require(str(index.tz) == 'UTC' and not index.hasnans,
                 'Reference requires complete UTC timestamps')
    _unique(bins, ['bin_start'], 'Reference bins')
    time_eligible = (bins['period'].astype(str).eq('calibration') &
                     bins['bin_start'].ge(start) & bins['bin_start'].lt(cutoff) &
                     bins['available_at'].le(cutoff))
    q = bins['pre_quote_ofi_300s'].to_numpy(dtype=float)
    v = bins['pre_rv_300s_5s_bps'].to_numpy(dtype=float)
    state_eligible = np.isfinite(q) & np.isfinite(v) & (v >= 0)
    selected = bins.loc[time_eligible & state_eligible, REFERENCE_COLUMNS].sort_values('bin_start').reset_index(drop=True)
    _require(len(selected) > 0, 'Empty scaling reference')
    q = selected['pre_quote_ofi_300s'].to_numpy(dtype=float)
    v = selected['pre_rv_300s_5s_bps'].to_numpy(dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        cq = float(np.sqrt(np.mean(np.square(q))))
        cv = float(np.mean(v))
        sv = float(np.sqrt(np.mean(np.square(v - cv))))
    _require(np.isfinite(cq) and cq > 0 and np.isfinite(cv) and np.isfinite(sv) and sv > 0,
             'Invalid scaling constants')
    return selected, (cq, cv, sv), time_eligible, state_eligible


def _feature_reference(anchors, constants):
    """Rebuild one feature record per parent anchor, never reading an outcome."""
    _unique(anchors, ['episode_id'], 'Anchors')
    anchors = anchors.sort_values('episode_id').reset_index(drop=True)
    features = anchors.copy()
    cq, cv, sv = constants
    q = anchors['pre_quote_ofi_300s'].to_numpy(dtype=float)
    v = anchors['pre_rv_bps_5s'].to_numpy(dtype=float)
    direction = anchors['direction'].to_numpy(dtype=float)
    spread = anchors['pre_spread_pips'].to_numpy(dtype=float)
    size = anchors['pre_displayed_size'].to_numpy(dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        standard = (v - cv) / sv
        magnitude = np.abs(q) / cq
        aligned = direction * q / cq
        log_size = np.where(np.isfinite(size) & (size > 0), np.log(size), np.nan)
        computed = {
            'intercept': np.ones(len(anchors), dtype=float),
            'variation_standardized': standard,
            'pressure_magnitude': magnitude,
            'aligned_prior_pressure': aligned,
            'variation_x_magnitude': standard * magnitude,
            'variation_x_alignment': standard * aligned,
            'log_pre_displayed_size': log_size,
            'magnitude_squared': magnitude * magnitude,
            'variation_squared': standard * standard,
        }
    for key, values in computed.items():
        features[key] = values
    for hour in range(1, 24):
        features['hour_%02d' % hour] = anchors['onset_hour_utc'].eq(hour).astype(float)
    for day in range(1, 7):
        features['weekday_%d' % day] = anchors['onset_weekday_utc'].eq(day).astype(float)
    features['valid_spread'] = np.isfinite(spread) & (spread > 0)
    features['valid_size'] = np.isfinite(size) & (size > 0)
    features['baseline_finite'] = np.isfinite(features[BASELINE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    features['curvature_finite'] = np.isfinite(features[CURVATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    features['regressor_eligible'] = (features['pre_state_eligible'].astype(bool) &
        features['valid_spread'] & features['valid_size'] & features['baseline_finite'])
    reasons = []
    for row in features.to_dict('records'):
        if not row['pre_state_eligible']:
            reasons.append('parent_pre_state_ineligible')
        elif not row['valid_spread']:
            reasons.append('invalid_pre_spread')
        elif not row['valid_size']:
            reasons.append('invalid_pre_size')
        elif not row['baseline_finite']:
            reasons.append('nonfinite_baseline')
        else:
            reasons.append('eligible')
    features['input_exclusion_reason'] = reasons
    return features


def _sample_reference(outcomes, features):
    """Keep all outcomes; intersect each parent component mask separately."""
    _unique(outcomes, ['episode_id', 'horizon_sec'], 'Outcomes')
    lookup = features.set_index('episode_id')
    _require(set(outcomes['episode_id']) == set(lookup.index),
             'Outcome and feature episode populations differ')
    _require(outcomes.groupby('episode_id')['horizon_sec'].apply(
        lambda values: set(values) == {0, 60, 300}).all(),
        'Parent outcome horizons differ')
    samples = outcomes.sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True).copy()
    samples['regressor_eligible'] = [bool(lookup.at[key, 'regressor_eligible'])
                                     for key in samples['episode_id']]
    for component in ['initial', 'continuation', 'total', 'joint']:
        samples['estimation_eligible_' + component] = np.logical_and(
            samples['eligible_' + component], samples['regressor_eligible'])
    return samples


def _design_reference(features, samples):
    """Design matrices contain predictors and identity only, never responses."""
    lookup = features.set_index('episode_id', drop=False)
    rows = []
    for horizon in [60, 300]:
        for key, component in [('c', 'continuation'), ('joint', 'joint')]:
            mask = samples['horizon_sec'].eq(horizon) & samples['estimation_eligible_' + component]
            ids = samples.loc[mask, 'episode_id']
            selected = lookup.loc[ids, ['episode_id'] + CURVATURE_COLUMNS].reset_index(drop=True)
            selected.insert(0, 'sample_key', key + '_' + str(horizon))
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _lineage(run, manifest, cache):
    inputs = manifest['inputs']
    parent_directory = Path(inputs['parent_manifest']['path']).parent
    reference_directory = Path(inputs['reference_manifest']['path']).parent
    parent, count_parent = _registered_hashes(parent_directory, cache)
    # the earlier detector's verified outputs and executed sources are the
    # immediate reference parent. Do not rehash/reconstruct its multi-GB raw
    # ancestors during a narrowly scoped transformation-stage verification.
    reference, count_reference = _registered_hashes(reference_directory, cache, include_inputs=False)
    _require(parent['status'] == 'verified_development_analysis_ledger', 'Unverified ledger parent')
    _require(_read(parent_directory / 'verification.json')['status'] == 'passed',
             'Ledger parent verification has not passed')
    _require(_read(reference_directory / 'verification.json')['status'] == 'passed',
             'Reference parent verification has not passed')
    _require(reference['status'] == 'verified_original_detector_development_audit',
             'Reference parent status differs')
    _require(parent['primary_policy'] == 'structural_only', 'Wrong ledger policy')
    # the calibration parent must be the actual detector ancestor of the
    # selected episodes, not merely another run sharing the same time bounds.
    chain = parent['inputs']['response_manifest']
    for step, expected_status in [
        ('responses', 'verified_development_price_responses'),
        ('episodes', 'verified_development_episodes_and_state_sample'),
    ]:
        path = Path(chain['path'])
        key = str(path.resolve())
        if key not in cache:
            cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(cache[key] == chain['sha256'], 'Changed ancestry manifest: ' + step)
        ancestor = _read(path)
        _require(ancestor['status'] == expected_status, 'Unverified ancestry: ' + step)
        if step == 'episodes':
            _require(ancestor['parent_event_audit'] == reference['run_id'],
                     'Episode ancestry uses a different detector run')
        chain = ancestor['inputs']['parent_manifest']
        count_parent += 1
    _require(chain['sha256'] == inputs['reference_manifest']['sha256'] and
             Path(chain['path']).resolve() == Path(inputs['reference_manifest']['path']).resolve(),
             'Scaling reference is not the detector ancestor of the outcome ledger')
    for key, suffix, owner, folder in [
        ('anchors', 'anchors.parquet', parent, parent_directory),
        ('outcomes', 'outcomes.parquet', parent, parent_directory),
        ('reference_bins', 'structural_only/scored_bins.parquet', reference, reference_directory),
        ('reference_parameters', 'structural_only/parameters.json', reference, reference_directory),
    ]:
        _require(inputs[key]['sha256'] == owner['outputs'][suffix], 'Input ancestry differs: ' + key)
        _require(Path(inputs[key]['path']).resolve() == (folder / suffix).resolve(),
                 'Input path ancestry differs: ' + key)
    parameters = _read(inputs['reference_parameters']['path'])
    configuration = manifest['configuration']
    start, cutoff = pd.Timestamp(configuration['reference_start']), pd.Timestamp(configuration['reference_cutoff'])
    _require(start == pd.Timestamp(parameters['training_start']) and
             cutoff == pd.Timestamp(parameters['parameter_cutoff']),
             'Reference bounds differ from recorded earlier detector calibration')
    _require(str(start.tz) == str(cutoff.tz) == 'UTC' and start < cutoff,
             'Invalid reference interval')
    return start, cutoff, count_parent + count_reference


def _scaling_reference(selected, constants, bins, start, cutoff):
    calibration = bins.loc[bins['period'].eq('calibration')]
    q = calibration['pre_quote_ofi_300s'].to_numpy(dtype=float)
    v = calibration['pre_rv_300s_5s_bps'].to_numpy(dtype=float)
    finite_q, valid_v = np.isfinite(q), np.isfinite(v) & (v >= 0)
    return {
        'cQ': constants[0], 'cV': constants[1], 'sV': constants[2],
        'reference_start': start.isoformat(), 'reference_cutoff': cutoff.isoformat(),
        'reference_population': 'detector-eligible structural calibration bins with finite Q and nonnegative finite V',
        'reference_bins': len(calibration), 'selected_reference_bins': len(selected),
        'excluded_nonfinite_Q': int((~finite_q).sum()),
        'excluded_invalid_V': int((~valid_v).sum()),
        'excluded_union': int((~(finite_q & valid_v)).sum()),
        'zero_Q_retained': int(selected['pre_quote_ofi_300s'].eq(0).sum()),
        'zero_V_retained': int(selected['pre_rv_300s_5s_bps'].eq(0).sum()),
        'population_standard_deviation_ddof': 0,
        'response_or_candidate_filter': False, 'Q_mean_subtracted': False,
    }


def _design_geometry(frame, curvature):
    """Numerical linear algebra of X only: no response is used or fitted."""
    unsupported = [key for key in CALENDAR_COLUMNS if np.count_nonzero(frame[key]) == 0]
    included = SCIENCE_COLUMNS + [key for key in CALENDAR_COLUMNS if key not in unsupported]
    if curvature:
        included += ['magnitude_squared', 'variation_squared']
    matrix = np.asarray(frame[included], dtype=float)
    nrows, ncolumns = matrix.shape
    finite = bool(np.isfinite(matrix).all())
    levels = {'hour_00': bool(np.any(frame['onset_hour_utc'] == 0)),
              'weekday_0_monday': bool(np.any(frame['onset_weekday_utc'] == 0))}
    rank, tolerance, condition, singular = 0, None, None, []
    leverage = {'max': None, 'mean': None, 'sum': None,
                'threshold_2rank_over_n': None, 'count_above_threshold': 0,
                'maximum_event_ids': []}
    if nrows > 0 and finite:
        left, values, right = np.linalg.svd(matrix, full_matrices=False)
        tolerance = float(np.finfo(np.float64).eps * max(nrows, ncolumns) * values[0])
        used = values > tolerance
        rank = int(np.sum(used))
        singular = [float(value) for value in values]
        if rank == ncolumns:
            condition = float(values[0] / values[-1])
        # diagonal of the orthogonal projector onto the retained column space.
        diagonal = np.einsum('ij,ij->i', left[:, used], left[:, used])
        limit = float(2 * rank / nrows)
        max_h = float(np.max(diagonal))
        leverage = {
            'max': max_h, 'mean': float(np.mean(diagonal)), 'sum': float(np.sum(diagonal)),
            'threshold_2rank_over_n': limit,
            'count_above_threshold': int(np.sum(diagonal > limit)),
            'maximum_event_ids': frame.loc[np.abs(diagonal - max_h) <= 1e-12,
                                           'episode_id'].astype(int).tolist(),
        }
        np.testing.assert_allclose(np.sum(diagonal), rank, rtol=1e-12, atol=1e-10,
                                   err_msg='Leverage trace does not equal rank')
        _require(np.all(diagonal >= -1e-12) and np.all(diagonal <= 1 + 1e-12),
                 'Projector diagonal out of bounds')
    return {
        'columns': included, 'rows': nrows, 'column_count': ncolumns,
        'unsupported_calendar_columns': unsupported, 'reference_present': levels,
        'finite': finite, 'rank': rank, 'rank_tolerance': tolerance,
        'singular_values': singular, 'condition_number': condition,
        'full_column_rank': bool(nrows > 0 and finite and rank == ncolumns),
        'usable_design': bool(nrows > ncolumns and finite and rank == ncolumns and all(levels.values())),
        'leverage': leverage, 'inference_approved': False,
    }


def _diagnostic_reference(features, samples):
    schemas, support = {}, []
    for horizon in [60, 300]:
        for prefix, component in [('c', 'continuation'), ('joint', 'joint')]:
            key = prefix + '_' + str(horizon)
            ids = set(samples.loc[samples['horizon_sec'].eq(horizon) &
                                  samples['estimation_eligible_' + component], 'episode_id'])
            rows = features.loc[features['episode_id'].isin(ids)]
            schemas[key] = {'horizon_sec': horizon, 'component': component,
                            'baseline': _design_geometry(rows, False),
                            'curvature': _design_geometry(rows, True)}
            for field, levels in [('onset_hour_utc', range(24)), ('onset_weekday_utc', range(7))]:
                counts = rows[field].value_counts()
                for level in levels:
                    count = int(counts.get(level, 0))
                    support.append({'sample_key': key, 'field': field, 'level': level,
                                    'events': count, 'supported': bool(count)})
    return schemas, pd.DataFrame(support)


def _distribution_reference(series):
    finite = np.asarray(series, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {'count': 0, 'min': None, 'q25': None, 'median': None, 'q75': None, 'max': None}
    quantiles = np.quantile(finite, [0, .25, .5, .75, 1])
    return dict(zip(['count', 'min', 'q25', 'median', 'q75', 'max'],
                    [len(finite)] + [float(value) for value in quantiles]))


def _statistics_reference(features, samples, design_rows, schemas):
    # count failures independently as successive set intersections of IDs.
    survivors = set(features['episode_id'])
    waterfall = []
    for flag in ['pre_state_eligible', 'valid_spread', 'valid_size', 'baseline_finite']:
        before = len(survivors)
        survivors.intersection_update(features.loc[features[flag], 'episode_id'])
        waterfall.append({'step': flag, 'remaining': len(survivors),
                          'newly_excluded': before - len(survivors)})
    selected = features.loc[features['regressor_eligible']]
    _require(set(selected['episode_id']) == survivors, 'Sequential attrition differs from final mask')
    horizons = []
    for horizon in [0, 60, 300]:
        rows = samples.loc[samples['horizon_sec'].eq(horizon)]
        counts = {}
        for label, prefix in [('parent_counts', 'eligible_'), ('estimation_counts', 'estimation_eligible_')]:
            counts[label] = {key: int(rows[prefix + key].sum())
                             for key in ['initial', 'continuation', 'total', 'joint']}
        horizons.append({'horizon_sec': horizon, **counts})
    reasons = features['input_exclusion_reason'].value_counts().sort_index()
    return {
        'anchor_rows': len(features), 'outcome_rows': len(samples), 'design_rows': len(design_rows),
        'regressor_eligible': len(selected), 'attrition_sequential': waterfall,
        'exclusion_reason_counts': {str(key): int(value) for key, value in reasons.items()},
        'primitive_invalid_counts_all_anchors': {
            'spread': int((~features['valid_spread']).sum()),
            'size': int((~features['valid_size']).sum()),
            'baseline_nonfinite': int((~features['baseline_finite']).sum()),
        },
        'curvature_nonfinite_on_baseline_cohort': int((features['regressor_eligible'] &
                                                    ~features['curvature_finite']).sum()),
        'horizons': horizons, 'design_diagnostics': schemas,
        'state_distributions': {key: _distribution_reference(selected[key]) for key in [
            'pressure_magnitude', 'aligned_prior_pressure', 'variation_standardized',
            'pre_spread_pips', 'log_pre_displayed_size']},
        'empirical_models_fitted': 0, 'production_inference_approved': False,
    }


def _specification_reference(scaling):
    roles = {key: 'strictly_before_onset' for key in [
        'pressure_magnitude', 'variation_standardized', 'variation_x_magnitude',
        'pre_spread_pips', 'log_pre_displayed_size', 'magnitude_squared', 'variation_squared']}
    roles.update({key: 'detection_time' for key in [
        'direction', 'aligned_prior_pressure', 'variation_x_alignment']})
    roles.update({key: 'deterministic_calendar_or_constant' for key in ['intercept'] + CALENDAR_COLUMNS})
    return {
        'version': '07a.1', 'scope': 'reused_development_input_preparation_only',
        'primary_horizon_seconds': 300, 'secondary_horizon_seconds': 60,
        'baseline_columns_canonical': BASELINE_COLUMNS,
        'curvature_added_columns': ['magnitude_squared', 'variation_squared'],
        'predictor_timing_roles': roles, 'calendar_reference': {'hour': 0, 'weekday': 0},
        'unsupported_calendar_policy': 'retain canonical columns; exclude only all-zero dummies from computational schema; no implicit estimated zero effect',
        'rank_failure_policy': 'record unusable design; do not drop substantive terms or observations',
        'control_policy': 'finite strictly positive prior spread and size; natural log size; no imputation',
        'outcome_policy': 'preserve component-specific availability; joint comparisons use intersection',
        'scaling': scaling, 'empirical_models_fitted': 0, 'production_inference_approved': False,
    }


def verify_estimation_inputs(run):
    """Verify lineage, every saved table and summary without production imports."""
    run = Path(run)
    cache = {}
    manifest, hashes = _registered_hashes(run, cache)
    start, cutoff, parent_hashes = _lineage(run, manifest, cache)
    hashes += parent_hashes
    paths = {key: Path(entry['path']) for key, entry in manifest['inputs'].items()}
    bins = pd.read_parquet(paths['reference_bins'], columns=REFERENCE_COLUMNS)
    reference, constants, time_eligible, state_eligible = _reference_values(bins, start, cutoff)
    calibration = bins['period'].eq('calibration')
    _require(np.array_equal(time_eligible.to_numpy(), calibration.to_numpy()),
             'Calibration contains bins outside recorded reference interval')
    _require(bins.loc[calibration, 'available_at'].eq(
        bins.loc[calibration, 'bin_start'] + pd.Timedelta(seconds=1)).all(),
        'Reference availability is not completed-bin boundary')
    scaling = _scaling_reference(reference, constants, bins, start, cutoff)
    _compare_frame(pd.read_parquet(run / 'reference.parquet'), reference, 'reference')
    _compare_json(_read(run / 'scaling.json'), scaling, 'scaling')
    anchors = pd.read_parquet(paths['anchors'])
    outcomes = pd.read_parquet(paths['outcomes'])
    _require(anchors['bin_start'].ge(cutoff).all(), 'Development overlaps reference')
    _require(anchors['onset_hour_utc'].eq(anchors['bin_start'].dt.hour).all() and
             anchors['onset_weekday_utc'].eq(anchors['bin_start'].dt.dayofweek).all(),
             'Parent calendar fields differ from actual UTC timestamps')
    _require(anchors['available_at'].eq(anchors['bin_start'] + pd.Timedelta(seconds=1)).all(),
             'Parent detection time differs from onset-bin completion')
    features = _feature_reference(anchors, constants)
    samples = _sample_reference(outcomes, features)
    design_rows = _design_reference(features, samples)
    schemas, support = _diagnostic_reference(features, samples)
    for name, expected in [('features', features), ('samples', samples),
                           ('design_rows', design_rows), ('calendar_support', support)]:
        _compare_frame(pd.read_parquet(run / (name + '.parquet')), expected, name)
    _compare_json(_read(run / 'schemas.json'), schemas, 'schemas')
    statistics = _statistics_reference(features, samples, design_rows, schemas)
    _compare_json(_read(run / 'statistics.json'), statistics, 'statistics')
    _compare_json(_read(run / 'specification.json'), _specification_reference(scaling), 'specification')
    _require(samples['estimation_eligible_joint'].eq(samples['estimation_eligible_initial'] &
        samples['estimation_eligible_continuation'] & samples['estimation_eligible_total']).all(),
        'Joint sample differs from component intersection')
    joint = samples.loc[samples['estimation_eligible_joint']]
    np.testing.assert_allclose(joint['initial_pips'] + joint['continuation_pips'],
                               joint['total_pips'], rtol=0, atol=1e-9, err_msg='J+C=I')
    lookup = features.set_index('episode_id')
    for field, source in [('bin_start', 'bin_start'), ('detected_at', 'available_at'),
                          ('pre_state_eligible', 'pre_state_eligible')]:
        np.testing.assert_array_equal(samples[field].to_numpy(),
            lookup.loc[samples['episode_id'], source].to_numpy(), err_msg='Parent identity: ' + field)
    evidence = _read(run / 'test_evidence.json')
    _require(evidence.get('status') == 'passed' and evidence.get('failures') == 0 and
             evidence.get('errors') == 0 and evidence.get('tests_run', 0) > 0,
             'Saved tests did not pass')
    for name, expected in evidence.get('test_source_sha256', {}).items():
        _require(manifest['outputs'].get('full_test_sources/' + name) == expected,
                 'Test evidence is not linked to saved test source: ' + name)
    return {
        'status': 'passed', 'hashes_verified': hashes, 'unique_files_hashed': len(cache),
        'parent_runs_verified': 2, 'reference_rows_verified': len(reference),
        'anchor_rows_verified': len(features), 'outcome_rows_verified': len(samples),
        'design_rows_verified': len(design_rows), 'calendar_support_rows_verified': len(support),
        'designs_verified': len(schemas), 'design_variants_verified': 2 * len(schemas),
        'joint_identities_verified': len(joint), 'all_statistics_fields_verified': True,
        'empirical_models_fitted': 0, 'production_inference_approved': False,
        'methods': 'Independent immutable-parent reconstruction of reference selection and constants, every feature, component-specific mask, design row, calendar support cell, SVD geometry, leverage, timing role and statistics field; no production transformation imports and no response fitting. New-run and direct-ledger registered hashes plus detector output/source/snapshot hashes checked; earlier detector raw ancestors not rehashed or re-estimated.',
    }

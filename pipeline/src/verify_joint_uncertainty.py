"""Independent verification of calendar uncertainty for joint endpoint means.

No production mean, bandwidth, covariance or diagnostic helper is imported.
Daily inputs are reconciled to inherited event records. The reference uses
stable scalar sums for NW1994 pilots and a dense calendar Bartlett matrix.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

COHORTS = ('horizon_joint', 'common_horizons_joint')
HORIZONS = (60, 300)
COMPONENTS = {'J': 'initial_pips', 'C': 'continuation_pips', 'I': 'total_pips'}
EPS = np.finfo(float).eps


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _numeric_equal(actual, expected, label, scale=None):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    magnitude = max(1., float(np.max(abs(b)))) if b.size else 1.
    if scale is not None:
        magnitude = max(magnitude, scale)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=256 * EPS * magnitude,
                               equal_nan=True, err_msg=label)


def _source(daily, membership, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _require(str(start.tz) == str(end.tz) == 'UTC' and start < end
             and start == start.normalize() and end == end.normalize(),
             'Ordered UTC midnight boundaries required')
    dates = pd.date_range(start, end, inclusive='left', freq='D')
    _require(len(dates) >= 3, 'At least three calendar days required')
    fields = ['episode_id', 'horizon_sec', 'bin_start', 'joint_included', *COMPONENTS.values()]
    _require(set(fields).issubset(membership), 'Missing event membership fields')
    _require(membership.columns.is_unique and daily.columns.is_unique, 'Duplicate source columns')
    _require(not membership[['episode_id', 'horizon_sec']].isna().any().any()
             and not membership.duplicated(['episode_id', 'horizon_sec']).any(),
             'Missing or duplicate membership keys')
    ids = membership.episode_id.to_numpy()
    _require(ids.dtype.kind in 'fiu' and np.isfinite(ids).all() and (ids >= 0).all()
             and (ids == np.floor(ids)).all(), 'Nonnegative integer event keys required')
    _require(membership.horizon_sec.isin(HORIZONS).all(), 'Unexpected membership horizon')
    _require(pd.api.types.is_bool_dtype(membership.joint_included)
             and membership.joint_included.notna().all(), 'Boolean joint mask required')
    onset = pd.DatetimeIndex(membership.bin_start)
    _require(str(onset.tz) == 'UTC' and not onset.hasnans
             and ((onset >= start) & (onset < end)).all(), 'Membership outside UTC calendar')
    for column in COMPONENTS.values():
        _require(pd.api.types.is_numeric_dtype(membership[column])
                 and not pd.api.types.is_bool_dtype(membership[column])
                 and np.isfinite(membership.loc[membership.joint_included, column]).all(),
                 'Joint responses must be finite numeric values')
    joint = membership.loc[membership.joint_included].copy()
    _require(joint.groupby('episode_id').initial_pips.nunique(dropna=False).le(1).all()
             and joint.groupby('episode_id').bin_start.nunique(dropna=False).le(1).all(),
             'Cross-horizon event identity differs')
    np.testing.assert_allclose(joint.initial_pips + joint.continuation_pips,
                               joint.total_pips, rtol=1e-12, atol=1e-9,
                               err_msg='Membership additive identity differs')
    required = ['cohort', 'horizon_sec', 'date', 'n', 'sum_J', 'sum_C', 'sum_I']
    _require(set(required).issubset(daily), 'Missing daily inputs')
    _require(not daily[['cohort', 'horizon_sec', 'date']].isna().any().any()
             and not daily.duplicated(['cohort', 'horizon_sec', 'date']).any(),
             'Duplicate or missing calendar keys')
    _require(set(daily.cohort) == set(COHORTS) and set(daily.horizon_sec) == set(HORIZONS),
             'Unexpected daily cohort/horizon')
    times = pd.DatetimeIndex(daily.date)
    _require(str(times.tz) == 'UTC' and not times.hasnans, 'Daily timestamps must be UTC')
    counts = daily.n.to_numpy()
    _require(counts.dtype.kind in 'iu' and np.isfinite(counts).all()
             and (counts >= 0).all() and (counts == np.floor(counts)).all(), 'Invalid daily counts')
    _require(all(pd.api.types.is_numeric_dtype(daily[name])
                 and not pd.api.types.is_bool_dtype(daily[name]) for name in ['sum_J', 'sum_C', 'sum_I']),
             'Real numeric daily sums required')
    _require(np.isfinite(daily[['sum_J', 'sum_C', 'sum_I']].to_numpy(dtype=float)).all(),
             'Nonfinite daily response sums')
    _require(daily.loc[daily.n.eq(0), ['sum_J', 'sum_C', 'sum_I']].eq(0).all().all(),
             'Empty calendar dates require exact zero observed sums')
    sets = {h: set(joint.loc[joint.horizon_sec.eq(h), 'episode_id']) for h in HORIZONS}
    common = sets[60] & sets[300]
    targets, score_columns, count_columns, means = [], [], [], []
    for cohort in COHORTS:
        for horizon in HORIZONS:
            rows = joint.loc[joint.horizon_sec.eq(horizon)]
            if cohort == 'common_horizons_joint':
                rows = rows.loc[rows.episode_id.isin(common)]
            observed = daily.loc[daily.cohort.eq(cohort) & daily.horizon_sec.eq(horizon)].sort_values('date')
            _require(pd.DatetimeIndex(observed.date).equals(dates), 'Incomplete full calendar')
            grouped = {date: frame for date, frame in rows.groupby(rows.bin_start.dt.normalize())}
            expected_counts = np.array([len(grouped.get(date, [])) for date in dates], dtype='int64')
            np.testing.assert_array_equal(observed.n, expected_counts, err_msg='Daily counts differ from event membership')
            n = int(sum(expected_counts))
            _require(n > 0, 'Empty joint cohort cannot support mean uncertainty')
            for component, field in COMPONENTS.items():
                values = np.array([math.fsum(grouped[date][field]) if date in grouped else 0.
                                   for date in dates])
                _numeric_equal(observed['sum_' + component], values, 'Daily sums differ from event membership')
                mean = math.fsum(values) / n
                g = np.array([float(value) - int(count) * mean
                              for value, count in zip(values, expected_counts)])
                _require(np.isfinite(g).all() and math.isfinite(mean), 'Nonfinite score or mean')
                targets.append({'target': f'{cohort}:{horizon}:{component}', 'cohort': cohort,
                                'horizon_sec': horizon, 'component': component, 'n': n})
                means.append(mean)
                score_columns.append(g)
                count_columns.append(expected_counts)
    return dates, targets, np.column_stack(score_columns), np.column_stack(count_columns), np.asarray(means)


def _pilot(values):
    t = len(values)
    m = math.floor(4 * (t / 100) ** (2 / 9))
    _require(1 <= m < t, 'Inadmissible pilot lag')
    scale = max(abs(float(value)) for value in values)
    tolerance = 64 * EPS * scale
    _require(math.isfinite(scale) and scale > tolerance, 'Degenerate scalar coordinate pilot')
    z = [float(value) / scale for value in values]
    gamma = [math.fsum(z[i] * z[i-j] for i in range(j, t)) / t for j in range(m + 1)]
    s0 = gamma[0] + 2 * math.fsum(gamma[1:])
    s1 = 2 * math.fsum(j * gamma[j] for j in range(1, m + 1))
    bound = 64 * EPS * (gamma[0] + 2 * math.fsum(abs(value) for value in gamma[1:]))
    _require(abs(s0) > bound, 'Numerically zero pilot s0')
    bandwidth = 1.1447 * ((s1 / s0) ** 2 * t) ** (1 / 3)
    _require(math.isfinite(bandwidth) and math.floor(bandwidth) < t, 'Automatic lag outside calendar')
    return {'calendar_days': t, 'pilot_lag': m, 'scalar_projection_scale': scale,
            'scalar_projection_tolerance': tolerance, 'scaled_autocovariances': gamma,
            'scaled_s0': s0, 'scaled_s1': s1, 'scaled_s0_tolerance': bound,
            'negative_pilot_s0': bool(s0 < 0), 'bandwidth': bandwidth,
            'lag': math.floor(bandwidth), 'weights': [1.], 'prewhitening': False,
            'daily_score_centering': False, 'moment_denominator': t, 'lag_capped': False}


def _reference(daily, membership, start, end):
    dates, targets, g, counts, means = _source(daily, membership, start, end)
    t, p = g.shape
    pilots = [{'column_index': j, **_pilot(g[:, j])} for j in range(p)]
    bandwidth = max(pilot['bandwidth'] for pilot in pilots)
    lag = math.floor(bandwidth)
    lags = {'automatic': lag, 'longer': max(1, 2 * lag)}
    _require(lags['longer'] < t, 'Longer lag outside calendar; no clipping')
    pilot = {'calendar_days': t, 'pilot_lag': pilots[0]['pilot_lag'],
        'bandwidth': bandwidth, 'lag': lag, 'weights': [1.] * p,
        'selected_columns': list(range(p)), 'component_pilots': pilots,
        'selector': 'maximum_of_fixed_selected_coordinate_NW1994_bandwidths',
        'prewhitening': False, 'daily_score_centering': False,
        'lag_capped': False, 'moment_denominator': t}
    sizes = np.asarray([target['n'] for target in targets], dtype=float)
    influence = g / sizes[None, :]
    intervals, covariance_rows, score_rows, lag_rows, diagnostics = [], [], [], [], {}
    distance = abs(np.arange(t)[:, None] - np.arange(t)[None, :])
    critical = NormalDist().inv_cdf(.975)
    for method, chosen_lag in lags.items():
        kernel = np.maximum(0., 1. - distance / (chosen_lag + 1))
        meat = g.T @ kernel @ g
        covariance = (meat / sizes[:, None]) / sizes[None, :]
        covariance = (covariance + covariance.T) / 2
        _require(np.isfinite(covariance).all(), 'Nonfinite covariance')
        covariance_scale = float(np.max(abs(covariance)))
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = 64 * EPS * p * covariance_scale
        _require(covariance_scale > 0 and eigenvalues[0] >= -tolerance,
                 'Degenerate or indefinite covariance')
        diagnostics[method] = {'covariance': covariance, 'meat': (meat + meat.T) / 2,
                               'min_eigenvalue': float(eigenvalues[0]), 'psd_tolerance': tolerance}
        for j, target in enumerate(targets):
            variance = float(covariance[j, j])
            _require(variance > 0, 'Nonpositive target variance')
            se = math.sqrt(variance)
            intervals.append(dict(target, method=method, lag=chosen_lag, mean=means[j],
                                  se=se, ci95_low=means[j] - critical * se,
                                  ci95_high=means[j] + critical * se))
            for k, other in enumerate(targets):
                covariance_rows.append({'method': method, 'lag': chosen_lag,
                    'target_i': target['target'], 'target_j': other['target'],
                    'value': covariance[j, k]})
    energy = np.asarray([math.fsum(float(value) ** 2 for value in influence[:, j]) for j in range(p)])
    _require(np.isfinite(energy).all() and (energy > 0).all(), 'Degenerate influence energy')
    shares = influence ** 2 / energy[None, :]
    maximum_lag = min(t - 1, max(70, lags['longer']))
    score_energy = np.asarray([math.fsum(float(value) ** 2 for value in g[:, j]) for j in range(p)])
    denominator = np.sqrt(score_energy[:, None]) * np.sqrt(score_energy[None, :])
    normalized = np.empty((maximum_lag + 1, p, p))
    for selected_lag in range(maximum_lag + 1):
        for j, target in enumerate(targets):
            for k, other in enumerate(targets):
                cross = math.fsum(float(g[i, j]) * float(g[i-selected_lag, k])
                                  for i in range(selected_lag, t))
                value = cross / denominator[j, k]
                normalized[selected_lag, j, k] = value
                lag_rows.append({'lag': selected_lag, 'target_i': target['target'],
                                 'target_j': other['target'], 'normalized_product': value})
    for j, target in enumerate(targets):
        for index, date in enumerate(dates):
            score_rows.append({'target': target['target'], 'date': date,
                'n': int(counts[index, j]), 'score': g[index, j],
                'influence': influence[index, j], 'energy_share': shares[index, j]})
    tables = {'intervals': pd.DataFrame(intervals), 'covariances': pd.DataFrame(covariance_rows),
              'scores': pd.DataFrame(score_rows), 'lag_products': pd.DataFrame(lag_rows)}
    return tables, dict(dates=dates, targets=targets, g=g, counts=counts, means=means,
        bandwidth=pilot, lags=lags, influence=influence, shares=shares, energy=energy,
        lag_inspection_max=maximum_lag, normalized_products=normalized, diagnostics=diagnostics)


def _compare_frame(actual, expected, keys, name):
    _require(set(actual) == set(expected), name + ' schema differs')
    _require(not actual[keys].isna().any().any() and not actual.duplicated(keys).any(),
             name + ' duplicate or missing keys')
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    _require(len(left) == len(right), name + ' row count differs')
    for field in right:
        if pd.api.types.is_float_dtype(right[field]):
            _numeric_equal(left[field], right[field], name + '.' + field)
        else:
            pd.testing.assert_series_equal(left[field], right[field], check_dtype=False,
                                            check_exact=True, obj=name + '.' + field)


def _compare_nested(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label + ' keys differ')
        for key, value in expected.items():
            _compare_nested(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), label + ' length differs')
        for index, value in enumerate(expected):
            _compare_nested(actual[index], value, label + f'[{index}]')
    elif isinstance(expected, bool):
        _require(isinstance(actual, bool) and actual is expected, label + ' Boolean differs')
    elif isinstance(expected, float):
        _require(isinstance(actual, (float, int)) and not isinstance(actual, bool), label + ' number required')
        _numeric_equal(actual, expected, label)
    else:
        _require(actual == expected and not (isinstance(expected, int) and isinstance(actual, bool)),
                 label + ' differs')


def _verify_statistics(stats, reference):
    r = reference
    t = len(r['dates'])
    names = [target['target'] for target in r['targets']]
    groups = {}
    for index in range(0, 12, 3):
        target = r['targets'][index]
        counts = r['counts'][:, index]
        groups[f"{target['cohort']}:{target['horizon_sec']}"] = {
            'n': target['n'], 'occupied_days': int(np.count_nonzero(counts)),
            'zero_event_days': int(np.count_nonzero(counts == 0)),
            'mean': {component: float(r['means'][index + j]) for j, component in enumerate(COMPONENTS)}}
    expected = {'calendar_days': t, 'target_order': names, 'groups': groups,
        'bandwidth': r['bandwidth'], 'automatic_lag': r['lags']['automatic'],
        'longer_lag': r['lags']['longer'], 'lag_inspection_max': r['lag_inspection_max'],
        'empirical_models_fitted': 0, 'uncertainty_computed': True,
        'interval_type': 'pointwise_nominal_95_percent_normal', 'inference_approved': False,
        'confidence_interval_coverage_validated': False, 'score_process_assumptions_verified': False,
        'covariance_inverted': False, 'p_values_computed': False, 'new_hypothesis_tests': 0}
    extra = {'covariance_diagnostics', 'score_lag_tail_summary', 'influence_summary', 'mean_intervals'}
    _require(set(expected) | extra <= set(stats)
             and set(stats) <= set(expected) | extra | {'regression_assessment'}, 'Statistics schema differs')
    for key, value in expected.items():
        _compare_nested(stats[key], value, 'statistics.' + key)
    _require(set(stats['covariance_diagnostics']) == set(r['lags']), 'Covariance diagnostic methods differ')
    for method, lag in r['lags'].items():
        saved = stats['covariance_diagnostics'][method]
        fields = {'lag', 'meat_min_eigenvalue', 'meat_psd_tolerance',
                  'covariance_min_eigenvalue', 'covariance_psd_tolerance'}
        _require(set(saved) == fields and saved['lag'] == lag, 'Covariance diagnostic schema or lag differs')
        for prefix, matrix in [('meat', r['diagnostics'][method]['meat']),
                                ('covariance', r['diagnostics'][method]['covariance'])]:
            scale = float(np.max(abs(matrix)))
            tolerance = 64 * EPS * 12 * scale
            _numeric_equal(saved[prefix + '_psd_tolerance'], tolerance,
                           'Covariance PSD tolerance differs')
            minimum = float(np.linalg.eigvalsh(matrix)[0])
            _require(math.isfinite(saved[prefix + '_min_eigenvalue'])
                     and saved[prefix + '_min_eigenvalue'] >= -saved[prefix + '_psd_tolerance'],
                     'Saved covariance diagnostic is indefinite')
            # Matrices are singular by the accounting identities; tiny null-space
            # eigenvalues may differ in sign between two valid summation orders.
            _require(abs(saved[prefix + '_min_eigenvalue'] - minimum) <= 1024 * EPS * 12 * scale,
                     'Covariance minimum eigenvalue differs beyond roundoff')
    _require(set(stats['score_lag_tail_summary']) == set(r['lags']), 'Lag tail methods differ')
    products = r['normalized_products']
    for method, lag in r['lags'].items():
        saved = stats['score_lag_tail_summary'][method]
        if lag >= r['lag_inspection_max']:
            _require(saved is None, 'Uninspected lag tail must be absent')
            continue
        _require(isinstance(saved, dict) and set(saved) == {'lag', 'target_i', 'target_j', 'max_abs_product'},
                 'Lag tail schema differs')
        _require(isinstance(saved['lag'], int) and lag < saved['lag'] <= r['lag_inspection_max']
                 and saved['target_i'] in names and saved['target_j'] in names, 'Lag tail coordinate differs')
        selected = abs(products[saved['lag'], names.index(saved['target_i']), names.index(saved['target_j'])])
        largest = float(np.max(abs(products[lag + 1:])))
        _numeric_equal(selected, largest, 'Reported lag tail coordinate is not a maximum')
        _numeric_equal(saved['max_abs_product'], largest, 'Lag tail maximum differs')
    summaries = []
    for j, name in enumerate(names):
        beyond = {method: (float(np.max(abs(products[lag+1:, j, j])))
                           if lag < r['lag_inspection_max'] else None) for method, lag in r['lags'].items()}
        summaries.append({'target': name, 'total_energy': float(r['energy'][j]),
            'maximum_day': r['dates'][int(np.argmax(r['shares'][:, j]))].isoformat(),
            'maximum_day_energy_share': float(max(r['shares'][:, j])),
            'max_abs_lag_product_beyond': beyond})
    _compare_nested(stats['influence_summary'], summaries, 'statistics.influence_summary')


def verify_joint_uncertainty(actual, daily, membership, start, end):
    """Verify serialized uncertainty artifacts against raw inherited events."""
    expected, reference = _reference(daily, membership, start, end)
    _require(set(actual) == set(expected) | {'statistics'}, 'Unexpected uncertainty artifacts')
    keys = {'intervals': ['method', 'target'], 'covariances': ['method', 'target_i', 'target_j'],
            'scores': ['target', 'date'], 'lag_products': ['lag', 'target_i', 'target_j']}
    for name in expected:
        _compare_frame(actual[name], expected[name], keys[name], name)
    _verify_statistics(actual['statistics'], reference)
    _compare_nested(actual['statistics']['mean_intervals'], expected['intervals'].to_dict('records'),
                    'statistics.mean_intervals')
    return {'status': 'passed', 'target_count': len(reference['targets']),
        'interval_rows_verified': len(expected['intervals']),
        'covariance_rows_verified': len(expected['covariances']),
        'score_rows_verified': len(expected['scores']),
        'lag_product_rows_verified': len(expected['lag_products']),
        'calendar_days_verified': len(reference['dates']),
        'empirical_models_fitted': 0, 'inference_approved': False,
        'method': 'Raw-event daily reconciliation, stable scalar coordinate pilots, dense Bartlett calendar kernel and independent normal quantile.'}

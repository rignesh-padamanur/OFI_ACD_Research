"""Stage 07a: deterministic regression inputs, with no response model fitting."""
import numpy as np
import pandas as pd

REFERENCE_FIELDS = ['bin_start', 'available_at', 'period', 'pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']
CORE = ['intercept', 'direction', 'variation_standardized', 'pressure_magnitude',
        'aligned_prior_pressure', 'variation_x_magnitude', 'variation_x_alignment',
        'pre_spread_pips', 'log_pre_displayed_size']
CALENDAR = [f'hour_{h:02d}' for h in range(1, 24)] + [f'weekday_{d}' for d in range(1, 7)]
CURVATURE = ['magnitude_squared', 'variation_squared']
COMPONENTS = ['initial', 'continuation', 'total', 'joint']


def _require(frame, fields):
    missing = set(fields) - set(frame)
    if missing:
        raise ValueError('Missing fields: ' + str(sorted(missing)))


def _unique(frame, fields):
    if frame[fields].isna().any().any() or frame.duplicated(fields).any():
        raise ValueError('Missing or duplicate keys: ' + str(fields))


def _utc(series):
    idx = pd.DatetimeIndex(series)
    if str(idx.tz) != 'UTC' or idx.hasnans:
        raise ValueError('Nonmissing UTC timestamps required')


def _numeric(series):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError('Numeric, nonboolean field required: ' + str(series.name))


def _boolean(series):
    if not pd.api.types.is_bool_dtype(series.dtype) or series.isna().any():
        raise ValueError('Nonmissing boolean field required: ' + str(series.name))


def make_scaling(reference, start, cutoff):
    """Fixed units from earlier detector-eligible bins, without event/return filters."""
    _require(reference, REFERENCE_FIELDS)
    start, cutoff = pd.Timestamp(start), pd.Timestamp(cutoff)
    _utc([start, cutoff])
    if not start < cutoff:
        raise ValueError('Reference interval must have positive length')
    if reference.period.isna().any() or not reference.period.isin(['calibration', 'development']).all():
        raise ValueError('Unknown reference period')
    f = reference.loc[reference.period.eq('calibration'), REFERENCE_FIELDS].copy()
    _unique(f, ['bin_start'])
    _utc(f.bin_start); _utc(f.available_at)
    if not (f.bin_start.ge(start) & f.bin_start.lt(cutoff) & f.available_at.le(cutoff)
            & f.available_at.eq(f.bin_start + pd.Timedelta(seconds=1))).all():
        raise ValueError('Calibration reference violates its interval or availability cutoff')
    for field in REFERENCE_FIELDS[-2:]:
        _numeric(f[field])
    q, v = f.pre_quote_ofi_300s, f.pre_rv_300s_5s_bps
    finite_q, finite_v = np.isfinite(q), np.isfinite(v) & v.ge(0)
    selected = f.loc[finite_q & finite_v].sort_values('bin_start').reset_index(drop=True)
    if len(selected) < 2:
        raise ValueError('Insufficient valid scaling reference')
    q = selected.pre_quote_ofi_300s.to_numpy(dtype=float)
    v = selected.pre_rv_300s_5s_bps.to_numpy(dtype=float)
    # Scale before squaring to avoid avoidable RMS overflow.
    q_max = float(np.max(np.abs(q)))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        cQ = q_max * float(np.sqrt(np.mean((q / q_max) ** 2))) if q_max > 0 else 0.
        cV = float(np.mean(v))
        deviation = v - cV
        v_max = float(np.max(np.abs(deviation)))
        sV = v_max * float(np.sqrt(np.mean((deviation / v_max) ** 2))) if v_max > 0 else 0.
    if not (np.isfinite([cQ, cV, sV]).all() and cQ > 0 and sV > 0):
        raise ValueError('Scaling constants must be finite with positive cQ and sV')
    scaling = {
        'cQ': cQ, 'cV': cV, 'sV': sV,
        'reference_start': start.isoformat(), 'reference_cutoff': cutoff.isoformat(),
        'reference_population': 'detector-eligible structural calibration bins with finite Q and nonnegative finite V',
        'reference_bins': len(f), 'selected_reference_bins': len(selected),
        'excluded_nonfinite_Q': int((~finite_q).sum()),
        'excluded_invalid_V': int((~finite_v).sum()),
        'excluded_union': int((~(finite_q & finite_v)).sum()),
        'zero_Q_retained': int(np.count_nonzero(q == 0)),
        'zero_V_retained': int(np.count_nonzero(v == 0)),
        'population_standard_deviation_ddof': 0,
        'response_or_candidate_filter': False, 'Q_mean_subtracted': False,
    }
    return selected, scaling


def _matrix_audit(frame, curvature=False):
    unsupported = [name for name in CALENDAR if not frame[name].ne(0).any()]
    columns = CORE + [name for name in CALENDAR if name not in unsupported]
    if curvature:
        columns += CURVATURE
    x = frame[columns].to_numpy(dtype=float)
    n, p = x.shape
    finite = bool(np.isfinite(x).all())
    reference_present = {
        'hour_00': bool(frame.onset_hour_utc.eq(0).any()),
        'weekday_0_monday': bool(frame.onset_weekday_utc.eq(0).any()),
    }
    singular, rank, tolerance, condition = [], 0, None, None
    leverage = {'max': None, 'mean': None, 'sum': None, 'threshold_2rank_over_n': None,
                'count_above_threshold': 0, 'maximum_event_ids': []}
    if n and finite:
        u, values, _ = np.linalg.svd(x, full_matrices=False)
        tolerance = float(np.finfo(float).eps * max(x.shape) * values[0])
        rank = int(np.count_nonzero(values > tolerance))
        singular = values.tolist()
        if rank == p:
            condition = float(values[0] / values[-1])
        h = np.sum(u[:, :rank] ** 2, axis=1)
        limit = 2 * rank / n
        leverage = {'max': float(h.max()), 'mean': float(h.mean()), 'sum': float(h.sum()),
                    'threshold_2rank_over_n': limit,
                    'count_above_threshold': int(np.count_nonzero(h > limit)),
                    'maximum_event_ids': frame.loc[np.isclose(h, h.max(), rtol=0, atol=1e-12), 'episode_id'].astype(int).tolist()}
    return {'columns': columns, 'rows': n, 'column_count': p,
            'unsupported_calendar_columns': unsupported, 'reference_present': reference_present,
            'finite': finite, 'rank': rank, 'rank_tolerance': tolerance,
            'singular_values': singular, 'condition_number': condition,
            'full_column_rank': bool(n and finite and rank == p),
            'usable_design': bool(n > p and finite and rank == p and all(reference_present.values())),
            'leverage': leverage, 'inference_approved': False}


def _distribution(series):
    x = series[np.isfinite(series)].to_numpy(dtype=float)
    return {'count': len(x), 'min': float(np.min(x)) if len(x) else None,
            'q25': float(np.quantile(x, .25)) if len(x) else None,
            'median': float(np.median(x)) if len(x) else None,
            'q75': float(np.quantile(x, .75)) if len(x) else None,
            'max': float(np.max(x)) if len(x) else None}


def build_inputs(anchors, outcomes, scaling):
    """Retain parents; add features and masks without estimating any response."""
    required = ['episode_id', 'bin_start', 'available_at', 'pre_quote_ofi_300s',
                'pre_rv_bps_5s', 'pre_spread_pips', 'pre_displayed_size', 'direction',
                'pre_state_eligible', 'onset_hour_utc', 'onset_weekday_utc']
    _require(anchors, required)
    _require(outcomes, ['episode_id', 'bin_start', 'detected_at', 'horizon_sec',
                       'pre_state_eligible', 'initial_pips', 'continuation_pips', 'total_pips',
                       *['eligible_' + name for name in COMPONENTS]])
    _unique(anchors, ['episode_id']); _unique(outcomes, ['episode_id', 'horizon_sec'])
    for frame in (anchors, outcomes):
        _numeric(frame.episode_id)
        if not (np.isfinite(frame.episode_id) & frame.episode_id.ge(0)
                & frame.episode_id.eq(np.floor(frame.episode_id))).all():
            raise ValueError('Nonnegative integer episode IDs required')
        _utc(frame.bin_start); _boolean(frame.pre_state_eligible)
    _utc(anchors.available_at); _utc(outcomes.detected_at)
    for name in ['pre_quote_ofi_300s', 'pre_rv_bps_5s', 'pre_spread_pips', 'pre_displayed_size', 'direction']:
        _numeric(anchors[name])
    for name in COMPONENTS:
        _boolean(outcomes['eligible_' + name])
    for name in ['initial', 'continuation', 'total']:
        _numeric(outcomes[name + '_pips'])
        if (outcomes['eligible_' + name] & ~np.isfinite(outcomes[name + '_pips'])).any():
            raise ValueError('Eligible outcome is not finite: ' + name)
    f = anchors.sort_values('episode_id').reset_index(drop=True).copy()
    o = outcomes.sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True).copy()
    if list(zip(o.episode_id, o.horizon_sec)) != [(i, h) for i in f.episode_id for h in (0, 60, 300)]:
        raise ValueError('Exactly one parent outcome per onset and 0/60/300 horizon required')
    joined = f.set_index('episode_id').loc[o.episode_id]
    for name, other in [('bin_start', 'bin_start'), ('detected_at', 'available_at'), ('pre_state_eligible', 'pre_state_eligible')]:
        if not np.array_equal(o[name].to_numpy(), joined[other].to_numpy()):
            raise ValueError('Parent outcome identity/timing mismatch: ' + name)
    if not f.available_at.eq(f.bin_start + pd.Timedelta(seconds=1)).all():
        raise ValueError('Detection must equal the completed onset-bin boundary')
    if not (f.onset_hour_utc.eq(f.bin_start.dt.hour).all()
            and f.onset_weekday_utc.eq(f.bin_start.dt.dayofweek).all()):
        raise ValueError('Calendar encoding disagrees with UTC onset')
    for mark, bound in [('pre_mark_at', 'bin_start'), ('detection_mark_at', 'available_at')]:
        if mark in f and (f[mark].notna() & ~f[mark].lt(f[bound])).any():
            raise ValueError('Price mark is not strictly before its boundary: ' + mark)
    q, v, d = f.pre_quote_ofi_300s, f.pre_rv_bps_5s, f.direction
    valid_states = np.isfinite(q) & np.isfinite(v) & v.ge(0) & np.isfinite(d) & d.isin([-1, 1])
    if (f.pre_state_eligible & ~valid_states).any():
        raise ValueError('Parent state eligibility contradicts primitive values')
    cutoff = pd.Timestamp(scaling['reference_cutoff'])
    _utc([cutoff])
    if not f.bin_start.ge(cutoff).all():
        raise ValueError('Scaling reference is later than an onset')
    cQ, cV, sV = (scaling[name] for name in ['cQ', 'cV', 'sV'])
    if not (np.isfinite([cQ, cV, sV]).all() and cQ > 0 and sV > 0):
        raise ValueError('Invalid scaling constants')
    f['intercept'] = 1.
    f['valid_spread'] = np.isfinite(f.pre_spread_pips) & f.pre_spread_pips.gt(0)
    f['valid_size'] = np.isfinite(f.pre_displayed_size) & f.pre_displayed_size.gt(0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        f['pressure_magnitude'] = q.abs() / cQ
        f['aligned_prior_pressure'] = d * (q / cQ)
        f['variation_standardized'] = (v - cV) / sV
        f['variation_x_magnitude'] = f.variation_standardized * f.pressure_magnitude
        f['variation_x_alignment'] = f.variation_standardized * f.aligned_prior_pressure
        f['log_pre_displayed_size'] = np.log(f.pre_displayed_size.where(f.valid_size))
        f['magnitude_squared'] = f.pressure_magnitude ** 2
        f['variation_squared'] = f.variation_standardized ** 2
    for h in range(1, 24):
        f[f'hour_{h:02d}'] = f.onset_hour_utc.eq(h).astype(float)
    for day in range(1, 7):
        f[f'weekday_{day}'] = f.onset_weekday_utc.eq(day).astype(float)
    f['baseline_finite'] = np.isfinite(f[CORE].to_numpy()).all(axis=1)
    f['curvature_finite'] = f.baseline_finite & np.isfinite(f[CURVATURE].to_numpy()).all(axis=1)
    f['regressor_eligible'] = f.pre_state_eligible & f.valid_spread & f.valid_size & f.baseline_finite
    f['input_exclusion_reason'] = np.select(
        [~f.pre_state_eligible, ~f.valid_spread, ~f.valid_size, ~f.baseline_finite],
        ['parent_pre_state_ineligible', 'invalid_pre_spread', 'invalid_pre_size', 'nonfinite_baseline'],
        default='eligible')
    eligible = f.set_index('episode_id').regressor_eligible
    o['regressor_eligible'] = o.episode_id.map(eligible).astype(bool)
    for name in COMPONENTS:
        o['estimation_eligible_' + name] = o['eligible_' + name] & o.regressor_eligible
    expected_joint = o.eligible_initial & o.eligible_continuation & o.eligible_total
    if not o.eligible_joint.eq(expected_joint).all():
        raise ValueError('Parent joint mask disagrees with component intersection')
    joint = o.eligible_joint
    if not np.allclose(o.loc[joint, 'initial_pips'] + o.loc[joint, 'continuation_pips'],
                       o.loc[joint, 'total_pips'], rtol=1e-12, atol=1e-10):
        raise ValueError('Parent J+C=I identity failed')
    designs, schemas, calendar = [], {}, []
    for horizon in (60, 300):
        for label, component in [('c', 'continuation'), ('joint', 'joint')]:
            key = f'{label}_{horizon}'
            ids = o.loc[o.horizon_sec.eq(horizon) & o['estimation_eligible_' + component], 'episode_id']
            sample = f.loc[f.episode_id.isin(ids)].copy()
            matrix = sample[['episode_id', *CORE, *CALENDAR, *CURVATURE]].copy()
            matrix.insert(0, 'sample_key', key)
            designs.append(matrix)
            schemas[key] = {'horizon_sec': horizon, 'component': component,
                            'baseline': _matrix_audit(sample), 'curvature': _matrix_audit(sample, True)}
            for field, levels in [('onset_hour_utc', range(24)), ('onset_weekday_utc', range(7))]:
                for level in levels:
                    n = int(sample[field].eq(level).sum())
                    calendar.append({'sample_key': key, 'field': field, 'level': level,
                                     'events': n, 'supported': n > 0})
    mask = np.ones(len(f), dtype=bool)
    waterfall = []
    for name in ['pre_state_eligible', 'valid_spread', 'valid_size', 'baseline_finite']:
        before = int(mask.sum()); mask &= f[name].to_numpy()
        waterfall.append({'step': name, 'remaining': int(mask.sum()), 'newly_excluded': before - int(mask.sum())})
    by_horizon = []
    for horizon in (0, 60, 300):
        sample = o.loc[o.horizon_sec.eq(horizon)]
        by_horizon.append({'horizon_sec': horizon,
            'parent_counts': {name: int(sample['eligible_' + name].sum()) for name in COMPONENTS},
            'estimation_counts': {name: int(sample['estimation_eligible_' + name].sum()) for name in COMPONENTS}})
    selected = f.loc[f.regressor_eligible]
    statistics = {
        'anchor_rows': len(f), 'outcome_rows': len(o), 'design_rows': sum(len(x) for x in designs),
        'regressor_eligible': len(selected), 'attrition_sequential': waterfall,
        'exclusion_reason_counts': {str(k): int(v) for k, v in f.input_exclusion_reason.value_counts().sort_index().items()},
        'primitive_invalid_counts_all_anchors': {'spread': int((~f.valid_spread).sum()), 'size': int((~f.valid_size).sum()),
                                              'baseline_nonfinite': int((~f.baseline_finite).sum())},
        'curvature_nonfinite_on_baseline_cohort': int((f.regressor_eligible & ~f.curvature_finite).sum()),
        'horizons': by_horizon, 'design_diagnostics': schemas,
        'state_distributions': {name: _distribution(selected[name]) for name in
            ['pressure_magnitude', 'aligned_prior_pressure', 'variation_standardized', 'pre_spread_pips', 'log_pre_displayed_size']},
        'empirical_models_fitted': 0, 'production_inference_approved': False,
    }
    roles = {name: 'strictly_before_onset' for name in ['pressure_magnitude', 'variation_standardized',
        'variation_x_magnitude', 'pre_spread_pips', 'log_pre_displayed_size', *CURVATURE]}
    roles.update({name: 'detection_time' for name in ['direction', 'aligned_prior_pressure', 'variation_x_alignment']})
    roles.update({name: 'deterministic_calendar_or_constant' for name in ['intercept', *CALENDAR]})
    spec = {'version': '07a.1', 'scope': 'reused_development_input_preparation_only',
        'primary_horizon_seconds': 300, 'secondary_horizon_seconds': 60,
        'baseline_columns_canonical': CORE + CALENDAR, 'curvature_added_columns': CURVATURE,
        'predictor_timing_roles': roles, 'calendar_reference': {'hour': 0, 'weekday': 0},
        'unsupported_calendar_policy': 'retain canonical columns; exclude only all-zero dummies from computational schema; no implicit estimated zero effect',
        'rank_failure_policy': 'record unusable design; do not drop substantive terms or observations',
        'control_policy': 'finite strictly positive prior spread and size; natural log size; no imputation',
        'outcome_policy': 'preserve component-specific availability; joint comparisons use intersection',
        'scaling': scaling, 'empirical_models_fitted': 0, 'production_inference_approved': False}
    return {'features': f, 'samples': o, 'design_rows': pd.concat(designs, ignore_index=True),
            'calendar_support': pd.DataFrame(calendar), 'schemas': schemas,
            'statistics': statistics, 'specification': spec}

"""Notebook 5 reconstruction: first-alert origins, labels and refit membership.

This module fits no forecasting model. The prediction clock is the completed
onset second, not an advance warning. Covariates use strictly prior quotes;
direction and J are known at detection and remain metadata. Future labels can
change evaluation availability, never whether an origin is eligible.
"""
import numpy as np
import pandas as pd


SECOND = pd.Timedelta(1, unit='s')
HORIZONS = (60, 300)
PRIMARY_HORIZON = 300
PRE_PREDICTORS = ['signed_pressure', 'pressure_magnitude', 'variation_standardized',
                  'variation_x_magnitude', 'pre_spread_pips', 'log_pre_displayed_size',
                  'onset_hour_utc', 'onset_weekday_utc']
MARK_STATUSES = ('available', 'beyond_sample_end', 'outside_onset_support',
                 'no_prior_quote', 'prior_quote_outside_support', 'stale_under_policy')
PRIOR_STATUSES = ('available', 'window_crosses_support', 'no_prior_quote',
                  'prior_quote_outside_support', 'outside_support', 'invalid_mark')


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _fields(frame, fields, label):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
             and set(fields).issubset(frame.columns), 'Missing or duplicate ' + label + ' fields')


def _ids(series, label='episode_id'):
    _require(pd.api.types.is_integer_dtype(series.dtype)
             and not pd.api.types.is_bool_dtype(series.dtype) and series.notna().all(),
             'Nonmissing integer ' + label + ' required')
    values = series.to_numpy(dtype='int64')
    _require((values >= 0).all(), 'Nonnegative ' + label + ' required')
    return values


def _times(series, label, missing=False):
    _require(pd.api.types.is_datetime64_any_dtype(series.dtype), 'Datetime required: ' + label)
    index = pd.DatetimeIndex(series)
    _require(str(index.tz) == 'UTC' and (missing or not index.hasnans),
             'UTC timestamps required: ' + label)
    return index


def _bound(value, label):
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and str(stamp.tz) == 'UTC', 'UTC boundary required: ' + label)
    return stamp


def _number(series, label):
    _require(pd.api.types.is_numeric_dtype(series.dtype)
             and not pd.api.types.is_bool_dtype(series.dtype), 'Numeric field required: ' + label)
    values = series.to_numpy(dtype=float, na_value=np.nan)
    _require(not np.isinf(values).any(), 'Infinite input: ' + label)
    return values


def _boolean(series, label):
    _require(pd.api.types.is_bool_dtype(series.dtype) and series.notna().all(),
             'Nonmissing Boolean required: ' + label)


def _same(left, right, label):
    if pd.api.types.is_integer_dtype(left.dtype):
        _require(pd.api.types.is_integer_dtype(right.dtype) and np.array_equal(left.to_numpy(), right.to_numpy()),
                 'Parent disagreement: ' + label)
    elif pd.api.types.is_numeric_dtype(left.dtype) and not pd.api.types.is_bool_dtype(left.dtype):
        _require(np.allclose(left.to_numpy(dtype=float, na_value=np.nan),
                            right.to_numpy(dtype=float, na_value=np.nan),
                            rtol=1e-12, atol=1e-10, equal_nan=True), 'Parent disagreement: ' + label)
    else:
        _require(((left.to_numpy() == right.to_numpy())
                  | (left.isna().to_numpy() & right.isna().to_numpy())).all(),
                 'Parent disagreement: ' + label)


def _origin_identity(origins):
    _fields(origins, ['episode_id', 'bin_start', 'forecast_at', 'origin_eligible',
                     'direction', 'initial_pips', 'detection_quote_available'], 'origin')
    _ids(origins.episode_id)
    _require(not origins.episode_id.duplicated().any(), 'Duplicate origin IDs')
    onset = _times(origins.bin_start, 'bin_start')
    _times(origins.forecast_at, 'forecast_at')
    _require(not onset.duplicated().any()
             and (onset.as_unit('ns').asi8 % SECOND.value == 0).all()
             and origins.forecast_at.eq(origins.bin_start + SECOND).all(),
             'Unique whole-second onsets and first-alert origins required')
    _boolean(origins.origin_eligible, 'origin_eligible')
    _boolean(origins.detection_quote_available, 'detection_quote_available')
    _number(origins.direction, 'direction'); _number(origins.initial_pips, 'initial_pips')
    _require((origins.direction.isna() | origins.direction.isin([-1, 0, 1])).all(),
             'Direction must be a sign or missing')
    _require((~origins.origin_eligible | (origins.direction.isin([-1, 1])
              & origins.detection_quote_available & np.isfinite(origins.initial_pips))).all(),
             'Eligible origin lacks known direction or initial quotes')


def build_origins(anchors, features, scaling):
    """Preserve every onset; derive eligibility without reading future fields.

    Required anchor/feature fields are explicitly projected below. Additional
    columns, including future membership masks and episode closure, are ignored.
    Missing primitive prior measurements cause explicit abstention. Contradictory
    identities, source times, flags or finite parent values fail closed.
    """
    common = ['episode_id', 'bin_start', 'available_at', 'support_id', 'onset_confirmed',
              'state_history_available', 'state_sample_eligible', 'pre_quote_ofi_300s',
              'pre_rv_300s_5s_bps', 'pre_spread_pips', 'pre_displayed_size',
              'direction', 'direction_available', 'pre_mark_at', 'detection_mark_at']
    extra_a = ['quote_ofi', 'pre_mark_status', 'detection_mark_status', 'pre_mid',
               'detection_mid', 'initial_pips']
    extra_f = ['pre_rv_bps_5s', 'pre_status_5s', 'pre_first_quote_at_5s', 'pre_last_quote_at_5s']
    _fields(anchors, common + extra_a, 'anchor')
    _fields(features, common + extra_f, 'feature')
    a = anchors[common + extra_a].sort_values('episode_id').reset_index(drop=True).copy()
    f = features[common + extra_f].sort_values('episode_id').reset_index(drop=True).copy()
    for frame in [a, f]:
        _ids(frame.episode_id); _ids(frame.support_id, 'support_id')
        _require(len(frame) and not frame.episode_id.duplicated().any(), 'Nonempty unique origin IDs required')
        for name in ['bin_start', 'available_at']:
            _times(frame[name], name)
        for name in ['pre_mark_at', 'detection_mark_at']:
            _times(frame[name], name, missing=True)
        for name in ['onset_confirmed', 'state_history_available', 'state_sample_eligible', 'direction_available']:
            _boolean(frame[name], name)
        for name in ['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps', 'pre_spread_pips',
                     'pre_displayed_size', 'direction']:
            _number(frame[name], name)
    _require(len(a) == len(f) and np.array_equal(a.episode_id, f.episode_id),
             'Anchor and feature IDs must match exactly')
    for name in common:
        _same(a[name], f[name], name)
    onset = _times(a.bin_start, 'bin_start')
    _require(not onset.duplicated().any() and (np.diff(onset.as_unit('ns').asi8) > 0).all()
             and (onset.as_unit('ns').asi8 % SECOND.value == 0).all()
             and a.available_at.eq(a.bin_start + SECOND).all(), 'Invalid onset/detection chronology')
    for name, boundary in [('pre_mark_at', a.bin_start), ('detection_mark_at', a.available_at)]:
        _require((a[name].isna() | a[name].lt(boundary)).all(), 'Source quote violates boundary: ' + name)
    for name, boundary in [('pre_first_quote_at_5s', a.bin_start - pd.Timedelta(300, unit='s')),
                            ('pre_last_quote_at_5s', a.bin_start)]:
        _times(f[name], name, missing=True)
        _require((f[name].isna() | f[name].lt(boundary)).all(), 'Prior quote violates boundary: ' + name)
    _require(f.pre_status_5s.isin(PRIOR_STATUSES).all(), 'Unknown prior measurement status')
    prior_available = f.pre_status_5s.eq('available')
    _require((~prior_available | (f.pre_first_quote_at_5s.notna() & f.pre_last_quote_at_5s.notna()
              & f.pre_first_quote_at_5s.le(f.pre_last_quote_at_5s))).all(),
             'Available prior state lacks ordered source quotes')
    _require((~prior_available | f.pre_last_quote_at_5s.eq(a.pre_mark_at)).all(),
             'Prior-state final quote differs from pre-onset quote')
    q = _number(a.pre_quote_ofi_300s, 'pre_quote_ofi_300s')
    old_v = _number(a.pre_rv_300s_5s_bps, 'pre_rv_300s_5s_bps')
    v = _number(f.pre_rv_bps_5s, 'pre_rv_bps_5s')
    history = np.isfinite(q) & np.isfinite(old_v)
    _require(np.array_equal(a.state_history_available, history)
             and np.array_equal(a.state_sample_eligible, a.onset_confirmed & history),
             'Inherited history flags contradict origin primitives')
    _require(np.allclose(v[a.state_sample_eligible], old_v[a.state_sample_eligible],
                         rtol=1e-12, atol=1e-9, equal_nan=False), 'Prior variation changed')
    detection_flow = _number(a.quote_ofi, 'quote_ofi')
    d = _number(a.direction, 'direction')
    expected_d = np.sign(detection_flow)
    _require(np.allclose(d, expected_d, rtol=0, atol=0, equal_nan=True), 'Direction differs from detection imbalance')
    directed = np.isfinite(d) & np.isin(d, [-1., 1.])
    _require(np.array_equal(a.direction_available, directed), 'Direction availability mismatch')
    good_marks = {}
    for name, timestamp, midpoint in [('pre', 'pre_mark_at', 'pre_mid'),
                                      ('detection', 'detection_mark_at', 'detection_mid')]:
        status = a[name + '_mark_status']
        _require(status.isin(MARK_STATUSES).all(), 'Unknown origin mark status')
        values = _number(a[midpoint], midpoint)
        available = status.eq('available')
        _require(np.array_equal(available, a[timestamp].notna())
                 and np.array_equal(available, np.isfinite(values))
                 and (values[available] > 0).all(), 'Origin mark status/value contradiction')
        good_marks[name] = available
    expected_j = np.where(directed, d * (a.detection_mid - a.pre_mid) * 10000, np.nan)
    _number(a.initial_pips, 'initial_pips')
    _require(np.allclose(a.initial_pips, expected_j, rtol=1e-12, atol=1e-10, equal_nan=True),
             'Initial displacement differs from origin quotes')
    _require(isinstance(scaling, dict) and {'cQ', 'cV', 'sV', 'reference_cutoff'}.issubset(scaling),
             'Frozen scaling metadata required')
    _require(all(isinstance(scaling[name], (int, float, np.integer, np.floating))
                 and not isinstance(scaling[name], (bool, np.bool_)) for name in ['cQ', 'cV', 'sV']),
             'Numeric nonboolean scaling constants required')
    cutoff = _bound(scaling['reference_cutoff'], 'scaling reference cutoff')
    constants = np.array([scaling[x] for x in ['cQ', 'cV', 'sV']], dtype=float)
    _require(np.isfinite(constants).all() and constants[0] > 0 and constants[2] > 0
             and a.bin_start.ge(cutoff).all(), 'Invalid or future scaling reference')
    c_q, c_v, s_v = constants
    valid_spread = np.isfinite(a.pre_spread_pips) & a.pre_spread_pips.gt(0)
    valid_size = np.isfinite(a.pre_displayed_size) & a.pre_displayed_size.gt(0)
    x = a[['episode_id', 'bin_start']].copy()
    x['forecast_at'] = a.available_at
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        x['signed_pressure'] = q / c_q
        x['pressure_magnitude'] = np.abs(q) / c_q
        x['variation_standardized'] = (v - c_v) / s_v
        x['variation_x_magnitude'] = x.variation_standardized * x.pressure_magnitude
        x['pre_spread_pips'] = a.pre_spread_pips
        x['log_pre_displayed_size'] = np.log(a.pre_displayed_size.where(valid_size))
    x['onset_hour_utc'] = a.bin_start.dt.hour.astype('int64')
    x['onset_weekday_utc'] = a.bin_start.dt.dayofweek.astype('int64')
    finite_x = np.isfinite(x[PRE_PREDICTORS].to_numpy(dtype=float)).all(axis=1)
    prior_ok = history & prior_available & np.isfinite(v) & (v >= 0) & np.isfinite(q)
    steps = [('confirmed_onset', a.onset_confirmed), ('defined_detection_direction', directed),
             ('prior_state_available', prior_ok), ('positive_pre_spread', valid_spread),
             ('positive_pre_size', valid_size), ('known_initial_quotes', good_marks['pre'] & good_marks['detection']),
             ('finite_prior_predictors', finite_x)]
    eligible = np.ones(len(a), dtype=bool)
    reason = np.full(len(a), 'eligible', dtype=object)
    waterfall = []
    for name, valid in steps:
        valid = np.asarray(valid, dtype=bool)
        newly = eligible & ~valid
        reason[newly] = name
        eligible &= valid
        waterfall.append({'step': name, 'remaining': int(eligible.sum()), 'newly_excluded': int(newly.sum())})
    origins = a[['episode_id', 'bin_start']].copy()
    origins['forecast_at'] = a.available_at
    origins['feature_cutoff_at'] = a.bin_start
    origins['scaling_ready_at'] = cutoff
    origins['origin_eligible'] = eligible
    origins['origin_exclusion_reason'] = reason
    origins['onset_confirmed'] = a.onset_confirmed
    origins['direction'] = d
    origins['direction_status'] = np.select([~np.isfinite(d), d == 0, d > 0],
                                           ['missing', 'zero', 'positive'], default='negative')
    origins['initial_pips'] = a.initial_pips
    origins['initial_sign'] = np.select([a.initial_pips.isna(), a.initial_pips.eq(0), a.initial_pips.gt(0)],
                                       ['undefined', 'zero', 'positive'], default='negative')
    origins['initial_quote_available'] = good_marks['pre'] & good_marks['detection']
    origins['detection_quote_available'] = good_marks['detection']
    origins['prior_state_available'] = np.asarray(prior_ok, dtype=bool)
    origins['reference_continuation_pips'] = np.where(eligible, 0., np.nan)
    origins['reference_total_pips'] = np.where(eligible, a.initial_pips, np.nan)
    for name in ['pre_mark_at', 'detection_mark_at']:
        origins[name] = a[name]
    for name in ['pre_first_quote_at_5s', 'pre_last_quote_at_5s']:
        origins[name] = f[name]
    _origin_identity(origins)
    statistics = {'all_origins': len(origins), 'origin_eligible': int(eligible.sum()),
        'origin_ineligible': int((~eligible).sum()), 'waterfall': waterfall,
        'exclusion_reasons': {str(k): int(v) for k, v in origins.origin_exclusion_reason.value_counts().items()},
        'direction_counts': {str(k): int(v) for k, v in origins.direction_status.value_counts().items()},
        'eligible_initial_sign_counts': {str(k): int(v) for k, v in origins.loc[eligible, 'initial_sign'].value_counts().items()},
        'prediction_clock': 'completed_onset_second_first_alert', 'strict_prior_predictor_columns': PRE_PREDICTORS.copy(),
        'scaling_reference_cutoff': cutoff.isoformat(), 'empirical_models_fitted': 0,
        'forecasts_computed': 0, 'advance_warning_supported': False, 'future_eligibility_used': False,
        'live_feed_latency_validated': False, 'untouched_holdout': False}
    return {'origins': origins, 'predictors': x, 'statistics': statistics}


def attach_labels(origins, outcomes, *, sample_end, horizons=HORIZONS):
    """Left-attach outcome records; label maturity is its scheduled horizon.

    Missing future keys are explicit missing_record rows. Missing values are
    preserved. Origin eligibility is copied unchanged and never recomputed from
    an outcome flag. The full parent may also contain the h=0 accounting rows.
    """
    _origin_identity(origins)
    stop = _bound(sample_end, 'sample end')
    _require(origins.forecast_at.le(stop).all(), 'Origin after sample end')
    _require(isinstance(horizons, (tuple, list)) and len(horizons) > 0
             and all(type(h) is int and h in HORIZONS for h in horizons)
             and len(set(horizons)) == len(horizons), 'Fixed unique 60/300 horizons required')
    fields = ['episode_id', 'bin_start', 'detected_at', 'horizon_sec', 'target_at',
              'endpoint_status', 'endpoint_quote_at', 'endpoint_age_sec', 'initial_pips',
              'continuation_pips', 'total_pips']
    _fields(outcomes, fields, 'outcome')
    o = outcomes[fields].copy()
    _ids(o.episode_id); _ids(o.horizon_sec, 'horizon_sec')
    _require(not o.duplicated(['episode_id', 'horizon_sec']).any()
             and o.episode_id.isin(origins.episode_id).all()
             and o.horizon_sec.isin([0, *HORIZONS]).all(), 'Unexpected or duplicate outcome identity')
    for name in ['bin_start', 'detected_at', 'target_at']:
        _times(o[name], name)
    _times(o.endpoint_quote_at, 'endpoint_quote_at', missing=True)
    for name in ['endpoint_age_sec', 'initial_pips', 'continuation_pips', 'total_pips']:
        _number(o[name], name)
    _require(o.endpoint_status.isin(MARK_STATUSES).all(), 'Unknown future endpoint status')
    keyed = origins.set_index('episode_id').loc[o.episode_id].reset_index(drop=True)
    o = o.reset_index(drop=True)
    _same(o.bin_start, keyed.bin_start, 'label onset')
    _same(o.detected_at, keyed.forecast_at, 'label forecast origin')
    _same(o.initial_pips, keyed.initial_pips, 'label initial displacement')
    target = o.detected_at + pd.to_timedelta(o.horizon_sec, unit='s')
    _require(o.target_at.eq(target).all(), 'Wrong label horizon target')
    available = o.endpoint_status.eq('available')
    _require(np.array_equal(available, o.endpoint_quote_at.notna())
             and np.array_equal(available, o.endpoint_age_sec.notna())
             and (o.endpoint_quote_at.isna() | o.endpoint_quote_at.lt(o.target_at)).all(),
             'Future mark timing/status contradiction')
    _require(np.allclose(o.loc[available, 'endpoint_age_sec'],
                         (o.loc[available, 'target_at'] - o.loc[available, 'endpoint_quote_at']).dt.total_seconds(),
                         rtol=1e-12, atol=1e-10), 'Future mark age mismatch')
    _require(o.endpoint_status.eq('beyond_sample_end').eq(o.target_at.gt(stop)).all(),
             'Sample-end status contradicts label target')
    c_finite = np.isfinite(o.continuation_pips)
    i_finite = np.isfinite(o.total_pips)
    expected_c = available & keyed.direction.isin([-1, 1]) & keyed.detection_quote_available
    _require(np.array_equal(c_finite, expected_c), 'Continuation availability contradicts source marks/direction')
    _require(np.array_equal(i_finite, available & np.isfinite(keyed.initial_pips)),
             'Total displacement availability contradicts source marks')
    joint = c_finite & i_finite & np.isfinite(keyed.initial_pips)
    _require(np.allclose(keyed.loc[joint, 'initial_pips'] + o.loc[joint, 'continuation_pips'],
                         o.loc[joint, 'total_pips'], rtol=1e-12, atol=1e-10), 'Price accounting identity failed')
    origin_fields = ['episode_id', 'bin_start', 'forecast_at', 'origin_eligible', 'direction',
                     'initial_pips', 'detection_quote_available']
    long = origins[origin_fields].merge(pd.DataFrame({'horizon_sec': list(horizons)}), how='cross')
    saved = o.loc[o.horizon_sec.isin(horizons), ['episode_id', 'horizon_sec', 'endpoint_status',
                'endpoint_quote_at', 'endpoint_age_sec', 'continuation_pips', 'total_pips']].copy()
    saved['label_record_present'] = True
    labels = long.merge(saved, on=['episode_id', 'horizon_sec'], how='left', validate='one_to_one')
    labels['label_record_present'] = labels.label_record_present.eq(True)
    labels['endpoint_status'] = labels.endpoint_status.fillna('missing_record')
    labels['target_at'] = labels.forecast_at + pd.to_timedelta(labels.horizon_sec, unit='s')
    labels['label_ready_at'] = labels.target_at
    labels['label_available'] = labels.label_record_present & labels.endpoint_status.eq('available') & np.isfinite(labels.continuation_pips)
    labels['evaluation_eligible'] = labels.origin_eligible & labels.label_available
    labels['label_status'] = np.select([~labels.label_record_present, ~labels.direction.isin([-1, 1]),
        ~labels.detection_quote_available, ~labels.endpoint_status.eq('available')],
        ['missing_record', 'undefined_direction', 'missing_detection_quote', labels.endpoint_status], default='available')
    labels['target_role'] = np.where(labels.horizon_sec.eq(PRIMARY_HORIZON), 'primary', 'secondary_description')
    return labels.sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True)


def build_folds(origins, labels, schedule, *, evaluation_start='2024-05-06T00:00:00+00:00', sample_end):
    """Expanding primary-horizon training and untouched-in-function block origins.

    The detector schedule supplies refit dates, not its rolling training_start.
    Response training expands from the first recorded origin. Equality of a
    label's maturity and cutoff is allowed: strict-left marks are complete at
    their boundary under an explicitly idealized zero-latency clock. No further
    overlap purge or independence claim is introduced.
    """
    _origin_identity(origins)
    start, stop = _bound(evaluation_start, 'evaluation start'), _bound(sample_end, 'sample end')
    _require(start < stop and origins.forecast_at.le(stop).all(), 'Invalid evaluation bounds')
    _fields(schedule, ['cycle_id', 'parameter_cutoff', 'deployment_end'], 'schedule')
    s = schedule[['cycle_id', 'parameter_cutoff', 'deployment_end']].sort_values('parameter_cutoff').reset_index(drop=True).copy()
    _ids(s.cycle_id, 'cycle_id')
    _times(s.parameter_cutoff, 'parameter_cutoff'); _times(s.deployment_end, 'deployment_end')
    _require(len(s) and not s.cycle_id.duplicated().any() and not s.parameter_cutoff.duplicated().any()
             and s.parameter_cutoff.lt(s.deployment_end).all()
             and np.array_equal(s.deployment_end.iloc[:-1].to_numpy(), s.parameter_cutoff.iloc[1:].to_numpy())
             and (s.parameter_cutoff == start).any()
             and s.parameter_cutoff.iloc[0] <= origins.forecast_at.min()
             and s.deployment_end.iloc[-1] >= stop, 'Incomplete or overlapping fixed schedule')
    _fields(labels, ['episode_id', 'horizon_sec', 'forecast_at', 'target_at', 'label_ready_at',
                     'origin_eligible', 'label_available', 'evaluation_eligible', 'continuation_pips'], 'label')
    _ids(labels.episode_id); _ids(labels.horizon_sec, 'horizon_sec')
    _require(not labels.duplicated(['episode_id', 'horizon_sec']).any()
             and labels.episode_id.isin(origins.episode_id).all() and labels.horizon_sec.isin(HORIZONS).all(),
             'Unexpected label identities')
    for name in ['forecast_at', 'target_at', 'label_ready_at']:
        _times(labels[name], name)
    for name in ['origin_eligible', 'label_available', 'evaluation_eligible']:
        _boolean(labels[name], name)
    _number(labels.continuation_pips, 'continuation_pips')
    _require(labels.target_at.eq(labels.forecast_at + pd.to_timedelta(labels.horizon_sec, unit='s')).all()
             and labels.label_ready_at.eq(labels.target_at).all()
             and labels.evaluation_eligible.eq(labels.origin_eligible & labels.label_available).all()
             and (~labels.label_available | (np.isfinite(labels.continuation_pips) & labels.target_at.le(stop))).all(),
             'Invalid label maturity or availability')
    identity = origins.set_index('episode_id').loc[labels.episode_id].reset_index(drop=True)
    _same(labels.forecast_at.reset_index(drop=True), identity.forecast_at, 'fold label origin')
    _same(labels.origin_eligible.reset_index(drop=True), identity.origin_eligible, 'fold eligibility')
    primary = labels.loc[labels.horizon_sec.eq(PRIMARY_HORIZON)].set_index('episode_id')
    _require(len(primary) == len(origins) and set(primary.index) == set(origins.episode_id),
             'Every origin needs a primary label row, including missing labels')
    f = origins[['episode_id', 'forecast_at', 'origin_eligible']].copy()
    for name in ['label_ready_at', 'label_available', 'evaluation_eligible']:
        f[name] = primary.loc[f.episode_id, name].to_numpy()
    f['label_ready_at'] = pd.to_datetime(f.label_ready_at, utc=True)
    rows, memberships = [], []
    selected = s.loc[s.parameter_cutoff.ge(start) & s.parameter_cutoff.lt(stop)]
    for row in selected.itertuples(index=False):
        cutoff, end = row.parameter_cutoff, min(row.deployment_end, stop)
        earlier = f.forecast_at.lt(cutoff) & f.origin_eligible
        mature = f.label_ready_at.le(cutoff)
        train = earlier & f.label_available & mature
        prediction = f.forecast_at.ge(cutoff) & f.forecast_at.lt(end)
        for role, mask in [('training', train), ('prediction', prediction)]:
            part = f.loc[mask, ['episode_id', 'forecast_at', 'origin_eligible', 'label_ready_at', 'label_available']].copy()
            part.insert(0, 'fold_id', int(row.cycle_id)); part.insert(1, 'role', role)
            part['horizon_sec'] = PRIMARY_HORIZON
            part['label_mature_at_refit'] = part.label_ready_at.le(cutoff)
            part['evaluation_eligible'] = (role == 'prediction') & part.origin_eligible & part.label_available
            memberships.append(part)
        rows.append({'fold_id': int(row.cycle_id), 'fit_cutoff': cutoff, 'prediction_start': cutoff,
            'prediction_end_exclusive': end, 'primary_horizon_sec': PRIMARY_HORIZON,
            'training_origins': int(train.sum()), 'earlier_origin_eligible': int(earlier.sum()),
            'earlier_labels_not_yet_mature': int((earlier & ~mature).sum()),
            'earlier_mature_labels_unavailable': int((earlier & mature & ~f.label_available).sum()),
            'prediction_origins': int(prediction.sum()),
            'prediction_origin_eligible': int((prediction & f.origin_eligible).sum()),
            'prediction_evaluable': int((prediction & f.evaluation_eligible).sum())})
    folds = pd.DataFrame(rows)
    _require(len(folds), 'No evaluation folds')
    membership = pd.concat(memberships, ignore_index=True)
    expected = f.loc[f.forecast_at.ge(start) & f.forecast_at.lt(stop), 'episode_id']
    actual = membership.loc[membership.role.eq('prediction'), 'episode_id']
    _require(not actual.duplicated().any() and set(actual) == set(expected), 'Prediction-origin conservation failed')
    _require(not membership.duplicated(['fold_id', 'role', 'episode_id']).any(), 'Duplicate fold membership')
    statistics = {'folds': len(folds), 'all_origins': len(origins),
        'warmup_origins': int(origins.forecast_at.lt(start).sum()),
        'origins_at_sample_end_not_in_evaluation_window': int(origins.forecast_at.eq(stop).sum()),
        'prediction_origins': int(len(actual)),
        'prediction_origin_eligible': int(folds.prediction_origin_eligible.sum()),
        'prediction_evaluable': int(folds.prediction_evaluable.sum()),
        'training_membership_rows': int(membership.role.eq('training').sum()),
        'primary_horizon_sec': PRIMARY_HORIZON, 'secondary_labels_used_for_training': False,
        'training_rule': 'expanding origin_eligible with observed C300 and target_at <= fit_cutoff',
        'training_start': s.parameter_cutoff.iloc[0].isoformat(),
        'evaluation_start': start.isoformat(), 'sample_end': stop.isoformat(),
        'boundary_convention': 'label_ready_at <= fit_cutoff; forecast_at in [fit_cutoff, deployment_end)',
        'detector_rolling_training_start_used_for_response_training': False,
        'future_response_availability_used_for_prediction_issuance': False,
        'label_horizon_boundary_purge': True, 'additional_source_overlap_purge': False,
        'independent_errors_assumed': False, 'empirical_models_fitted': 0, 'forecasts_computed': 0,
        'evaluation_losses_computed': False, 'untouched_holdout': False,
        'live_feed_latency_validated': False, 'inference_approved': False}
    return {'folds': folds, 'membership': membership, 'statistics': statistics}

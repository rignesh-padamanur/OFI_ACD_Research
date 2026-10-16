"""Independent first-alert ledger verification; no producer helpers or fitting.

The verifier reconstructs identities, information timing, censoring and refit
membership from the saved parents. It does not certify a live quote feed,
economic identification, untouched holdout status or forecast performance.
"""
import numpy as np
import pandas as pd


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _fields(frame, names, label):
    _require(frame.columns.is_unique, label + ': duplicate columns')
    _require(set(names).issubset(frame), label + ': missing fields ' +
             str(sorted(set(names) - set(frame))))


def _keys(frame, names, label):
    _fields(frame, names, label)
    _require(not frame[names].isna().any().any() and
             not frame.duplicated(names).any(), label + ': missing/duplicate keys')
    for name in names:
        x = frame[name]
        _require(pd.api.types.is_integer_dtype(x) and
                 not pd.api.types.is_bool_dtype(x) and np.isfinite(x).all() and
                 x.ge(0).all() and x.eq(np.floor(x)).all(),
                 label + ': nonnegative integral keys required')


def _utc(values, label, missing=False):
    x = pd.DatetimeIndex(values)
    _require(str(x.tz) == 'UTC' and (missing or not x.hasnans),
             label + ': UTC timestamps required')


def _bool(frame, columns, label):
    for name in columns:
        _require(pd.api.types.is_bool_dtype(frame[name]) and
                 not frame[name].isna().any(), label + ': boolean ' + name)


def _same(actual, expected, keys, label):
    _keys(actual, keys, label)
    _require(set(actual.columns) == set(expected.columns),
             label + ': unexpected or missing columns (possible predictor leakage)')
    left = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    right = expected.sort_values(keys).reset_index(drop=True)
    for key in keys:
        _require(np.array_equal(left[key], right[key]), label + ': changed exact keys')
    for col in expected.columns:
        if pd.api.types.is_bool_dtype(expected[col]):
            _bool(actual, [col], label)
    pd.testing.assert_frame_equal(left, right, check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-10,
                                  obj=label)


def _sources(anchors, features, scaling):
    fields = ['episode_id', 'bin_start', 'available_at', 'quote_ofi', 'direction',
              'pre_mid', 'detection_mid', 'initial_pips', 'raw_initial_pips',
              'pre_mark_at', 'detection_mark_at', 'baseline_quote_available',
              'onset_confirmed', 'state_history_available', 'state_sample_eligible']
    _fields(anchors, fields, 'anchors')
    _keys(anchors, ['episode_id'], 'anchors')
    _keys(features, ['episode_id'], 'features')
    a = anchors.sort_values('episode_id').reset_index(drop=True)
    f = features.sort_values('episode_id').reset_index(drop=True)
    _require(a.episode_id.equals(f.episode_id), 'Full parent ID sets disagree')
    for col in ['bin_start', 'available_at']:
        _utc(a[col], 'anchors ' + col)
        _utc(f[col], 'features ' + col)
        _require(a[col].equals(f[col]), 'Parent clock mismatch: ' + col)
    _require(a.available_at.eq(a.bin_start + pd.Timedelta(1, unit='s')).all(),
             'First alert must equal completed onset-bin boundary')
    _require(not a.bin_start.duplicated().any() and a.bin_start.is_monotonic_increasing,
             'Onset chronology must follow the conserved episode IDs')
    _require((pd.DatetimeIndex(a.bin_start).as_unit('ns').asi8 % 1_000_000_000 == 0).all(),
             'Onsets must lie on whole-second boundaries')
    for col, cutoff in [('pre_mark_at', a.bin_start),
                        ('detection_mark_at', a.available_at)]:
        _utc(a[col], col, missing=True)
        _require((a[col].isna() | a[col].lt(cutoff)).all(),
                 'Price mark not strictly before boundary: ' + col)
    for frame, label in [(a, 'anchors'), (f, 'features')]:
        _bool(frame, ['onset_confirmed', 'state_history_available',
                      'state_sample_eligible'], label)
        _require(frame.state_sample_eligible.eq(
            frame.onset_confirmed & frame.state_history_available).all(),
            label + ': changed onset/history eligibility')
    _bool(a, ['baseline_quote_available'], 'anchors')
    cutoff = pd.Timestamp(scaling['reference_cutoff'])
    _utc([cutoff], 'scaling cutoff')
    _require(a.bin_start.ge(cutoff).all(), 'Scaling is future to an origin')
    scales = np.asarray([scaling[k] for k in ['cQ', 'cV', 'sV']], dtype=float)
    _require(np.isfinite(scales).all() and scales[0] > 0 and scales[2] > 0,
             'Invalid frozen scales')
    q = a.quote_ofi.to_numpy(dtype=float)
    d = np.where(np.isfinite(q), np.sign(q), np.nan)
    np.testing.assert_allclose(a.direction, d, rtol=0, atol=0, equal_nan=True,
                               err_msg='Direction differs from alert quote pressure')
    raw = (a.detection_mid - a.pre_mid).to_numpy(dtype=float) * 10000
    np.testing.assert_allclose(a.raw_initial_pips, raw, rtol=1e-12, atol=1e-9,
                               equal_nan=True, err_msg='Raw initial displacement')
    initial = np.where(np.isfinite(d) & (d != 0), d * raw, np.nan)
    np.testing.assert_allclose(a.initial_pips, initial, rtol=1e-12, atol=1e-9,
                               equal_nan=True, err_msg='Aligned initial displacement')
    return a, f


PREDICTORS = ['signed_pressure', 'pressure_magnitude', 'variation_standardized',
              'variation_x_magnitude', 'pre_spread_pips', 'log_pre_displayed_size',
              'onset_hour_utc', 'onset_weekday_utc']
STATUSES = {'available', 'beyond_sample_end', 'outside_onset_support',
            'no_prior_quote', 'prior_quote_outside_support', 'stale_under_policy'}


def _reference_origins(a, f, scaling):
    common = ['support_id', 'onset_confirmed', 'state_history_available',
              'state_sample_eligible', 'pre_quote_ofi_300s', 'pre_rv_300s_5s_bps',
              'pre_spread_pips', 'pre_displayed_size', 'direction',
              'direction_available', 'pre_mark_at', 'detection_mark_at']
    _fields(f, common + ['pre_rv_bps_5s', 'pre_status_5s',
                         'pre_first_quote_at_5s', 'pre_last_quote_at_5s'], 'features')
    for col in common:
        pd.testing.assert_series_equal(a[col], f[col], check_dtype=False,
                                       check_names=False, rtol=1e-12, atol=1e-10)
    q = a.pre_quote_ofi_300s.to_numpy(dtype=float)
    v = f.pre_rv_bps_5s.to_numpy(dtype=float)
    history = np.isfinite(q) & np.isfinite(a.pre_rv_300s_5s_bps)
    _require(np.array_equal(history, a.state_history_available),
             'History flag differs from prior primitives')
    _require((~a.state_sample_eligible | np.isclose(
        v, a.pre_rv_300s_5s_bps, rtol=1e-12, atol=1e-9)).all(),
        'Measured prior variation disagrees with inherited variation')
    _require(f.pre_status_5s.isin({'available', 'window_crosses_support', 'no_prior_quote',
        'prior_quote_outside_support', 'outside_support', 'invalid_mark'}).all(),
        'Unknown prior measurement status')
    prior = f.pre_status_5s.eq('available') & history & np.isfinite(v) & (v >= 0)
    for col, bound in [('pre_first_quote_at_5s', a.bin_start-pd.Timedelta(300, unit='s')),
                       ('pre_last_quote_at_5s', a.bin_start)]:
        _utc(f[col], col, missing=True)
        _require((f[col].isna() | f[col].lt(bound)).all(), 'Future prior-state quote')
    _require((~f.pre_status_5s.eq('available') | (
        f.pre_first_quote_at_5s.notna() & f.pre_last_quote_at_5s.notna() &
        f.pre_first_quote_at_5s.le(f.pre_last_quote_at_5s) &
        f.pre_last_quote_at_5s.eq(a.pre_mark_at))).all(), 'Invalid prior-state marks')
    good = {}
    for prefix in ['pre', 'detection']:
        status = a[prefix+'_mark_status']
        _require(status.isin(STATUSES).all(), 'Unknown initial mark status')
        good[prefix] = status.eq('available')
        _require(np.array_equal(good[prefix], a[prefix+'_mark_at'].notna()) and
                 np.array_equal(good[prefix], np.isfinite(a[prefix+'_mid'])) and
                 a.loc[good[prefix], prefix+'_mid'].gt(0).all(),
                 'Initial mark status/value mismatch')
    directed = a.direction.isin([-1., 1.])
    _require(directed.equals(a.direction_available), 'Direction mask disagreement')
    positive_spread = np.isfinite(a.pre_spread_pips) & a.pre_spread_pips.gt(0)
    positive_size = np.isfinite(a.pre_displayed_size) & a.pre_displayed_size.gt(0)
    x = a[['episode_id', 'bin_start']].copy()
    x['forecast_at'] = a.available_at
    x['signed_pressure'] = q / scaling['cQ']
    x['pressure_magnitude'] = np.abs(q) / scaling['cQ']
    x['variation_standardized'] = (v-scaling['cV']) / scaling['sV']
    x['variation_x_magnitude'] = x.variation_standardized*x.pressure_magnitude
    x['pre_spread_pips'] = a.pre_spread_pips
    x['log_pre_displayed_size'] = np.log(a.pre_displayed_size.where(positive_size))
    x['onset_hour_utc'] = a.bin_start.dt.hour.astype('int64')
    x['onset_weekday_utc'] = a.bin_start.dt.dayofweek.astype('int64')
    masks = {'confirmed_onset': a.onset_confirmed,
             'defined_detection_direction': directed, 'prior_state_available': prior,
             'positive_pre_spread': positive_spread, 'positive_pre_size': positive_size,
             'known_initial_quotes': good['pre'] & good['detection'],
             'finite_prior_predictors': np.isfinite(x[PREDICTORS]).all(axis=1)}
    reasons = pd.Series('eligible', index=a.index, dtype=object)
    for name, mask in masks.items():
        reasons.loc[reasons.eq('eligible') & ~mask] = name
    eligible = reasons.eq('eligible')
    o = a[['episode_id', 'bin_start']].copy()
    o['forecast_at'] = a.available_at
    o['feature_cutoff_at'] = a.bin_start
    o['scaling_ready_at'] = pd.Timestamp(scaling['reference_cutoff'])
    o['origin_eligible'] = eligible
    o['origin_exclusion_reason'] = reasons
    o['onset_confirmed'] = a.onset_confirmed
    o['direction'] = a.direction
    o['direction_status'] = a.direction.map({-1.: 'negative', 0.: 'zero', 1.: 'positive'}).fillna('missing')
    o['initial_pips'] = a.initial_pips
    o['initial_sign'] = np.sign(a.initial_pips).map({-1.: 'negative', 0.: 'zero', 1.: 'positive'}).fillna('undefined')
    o['initial_quote_available'] = good['pre'] & good['detection']
    o['detection_quote_available'] = good['detection']
    o['prior_state_available'] = prior
    o['reference_continuation_pips'] = pd.Series(0., index=o.index).where(eligible)
    o['reference_total_pips'] = a.initial_pips.where(eligible)
    for col in ['pre_mark_at', 'detection_mark_at']:
        o[col] = a[col]
    for col in ['pre_first_quote_at_5s', 'pre_last_quote_at_5s']:
        o[col] = f[col]
    return o, x


def _reference_labels(origins, outcomes, stop):
    fields = ['episode_id', 'bin_start', 'detected_at', 'horizon_sec', 'target_at',
              'endpoint_status', 'endpoint_quote_at', 'endpoint_age_sec',
              'initial_pips', 'continuation_pips', 'total_pips']
    _fields(outcomes, fields, 'source outcomes')
    _keys(outcomes, ['episode_id', 'horizon_sec'], 'source outcomes')
    _require(outcomes.episode_id.isin(origins.episode_id).all() and
             outcomes.horizon_sec.isin([0, 60, 300]).all(), 'Unexpected source outcome keys')
    source = outcomes[fields].reset_index(drop=True)
    matched = origins.set_index('episode_id').loc[source.episode_id].reset_index()
    for col, peer in [('bin_start', 'bin_start'), ('detected_at', 'forecast_at')]:
        _utc(source[col], col)
        _require(source[col].equals(matched[peer]), 'Source label clock mismatch')
    for col in ['target_at', 'endpoint_quote_at']:
        _utc(source[col], col, missing=col == 'endpoint_quote_at')
    target = source.detected_at + pd.to_timedelta(source.horizon_sec, unit='s')
    _require(source.target_at.equals(target), 'Source label target mismatch')
    _require(source.endpoint_status.isin(STATUSES).all(), 'Unknown source endpoint status')
    _require(source.endpoint_status.eq('beyond_sample_end').equals(target.gt(stop)),
             'Sample end and censoring mismatch')
    available = source.endpoint_status.eq('available')
    _require(available.equals(source.endpoint_quote_at.notna()) and
             available.equals(source.endpoint_age_sec.notna()) and
             (source.endpoint_quote_at.isna() | source.endpoint_quote_at.lt(target)).all(),
             'Endpoint availability or strict-left mark mismatch')
    np.testing.assert_allclose(source.loc[available, 'endpoint_age_sec'],
        (target[available]-source.loc[available, 'endpoint_quote_at']).dt.total_seconds(),
        rtol=1e-12, atol=1e-10, err_msg='Endpoint age')
    np.testing.assert_allclose(source.initial_pips, matched.initial_pips,
        rtol=1e-12, atol=1e-10, equal_nan=True, err_msg='Initial J changed across labels')
    c_good = available & matched.direction.isin([-1., 1.]) & matched.detection_quote_available
    i_good = c_good & np.isfinite(matched.initial_pips)
    _require(np.array_equal(np.isfinite(source.continuation_pips), c_good),
             'Continuation finite mask contradicts marks/direction')
    _require(np.array_equal(np.isfinite(source.total_pips), i_good),
             'Total finite mask contradicts available J and C')
    np.testing.assert_allclose(source.loc[i_good, 'total_pips'],
        source.loc[i_good, 'initial_pips'] + source.loc[i_good, 'continuation_pips'],
        rtol=1e-12, atol=1e-10, err_msg='J+C=I')
    grid = pd.MultiIndex.from_product([origins.episode_id, [60, 300]],
                                     names=['episode_id', 'horizon_sec'])
    chosen = source.set_index(['episode_id', 'horizon_sec'])
    future = chosen.reindex(grid)
    repeated = origins.set_index('episode_id').loc[grid.get_level_values('episode_id')].reset_index()
    origin_cols = ['episode_id', 'bin_start', 'forecast_at', 'origin_eligible',
                   'direction', 'initial_pips', 'detection_quote_available']
    labels = repeated[origin_cols].copy()
    labels['horizon_sec'] = grid.get_level_values('horizon_sec')
    for col in ['endpoint_status', 'endpoint_quote_at', 'endpoint_age_sec',
                'continuation_pips', 'total_pips']:
        labels[col] = future[col].to_numpy()
    labels['endpoint_quote_at'] = pd.to_datetime(labels.endpoint_quote_at, utc=True)
    present = grid.isin(chosen.index)
    labels['label_record_present'] = present
    labels['endpoint_status'] = labels.endpoint_status.fillna('missing_record')
    labels['target_at'] = labels.forecast_at + pd.to_timedelta(labels.horizon_sec, unit='s')
    labels['label_ready_at'] = labels.target_at
    labels['label_available'] = present & labels.endpoint_status.eq('available') & np.isfinite(labels.continuation_pips)
    labels['evaluation_eligible'] = labels.origin_eligible & labels.label_available
    labels['label_status'] = labels.endpoint_status
    labels.loc[~labels.detection_quote_available, 'label_status'] = 'missing_detection_quote'
    labels.loc[~labels.direction.isin([-1., 1.]), 'label_status'] = 'undefined_direction'
    labels.loc[~labels.label_record_present, 'label_status'] = 'missing_record'
    labels['target_role'] = labels.horizon_sec.map({300: 'primary', 60: 'secondary_description'})
    return labels


def _verify_folds(origins, labels, schedule, artifacts, start, stop):
    _fields(schedule, ['cycle_id', 'parameter_cutoff', 'deployment_end'], 'schedule')
    _keys(schedule, ['cycle_id'], 'schedule')
    s = schedule.sort_values('parameter_cutoff').reset_index(drop=True)
    for col in ['parameter_cutoff', 'deployment_end']:
        _utc(s[col], 'schedule '+col)
    _require(not s.parameter_cutoff.duplicated().any() and
             s.parameter_cutoff.lt(s.deployment_end).all() and
             s.deployment_end.iloc[:-1].reset_index(drop=True).equals(
                 s.parameter_cutoff.iloc[1:].reset_index(drop=True)) and
             s.parameter_cutoff.eq(start).any() and
             s.parameter_cutoff.iloc[0] <= origins.forecast_at.min() and
             s.deployment_end.iloc[-1] >= stop, 'Invalid refit schedule')
    actual = artifacts['membership']
    _fields(actual, ['fold_id', 'role', 'episode_id'], 'membership')
    _require(not actual.duplicated(['fold_id', 'role', 'episode_id']).any(),
             'Duplicate fold membership')
    _require(actual.role.isin(['training', 'prediction']).all(), 'Unknown membership role')
    selected = s.loc[s.parameter_cutoff.ge(start) & s.parameter_cutoff.lt(stop)]
    _require(actual.fold_id.isin(selected.cycle_id).all(), 'Unknown fold IDs')
    primary = labels.loc[labels.horizon_sec.eq(300)].set_index('episode_id')
    z = origins[['episode_id', 'forecast_at', 'origin_eligible']].set_index('episode_id')
    z = z.join(primary[['label_ready_at', 'label_available', 'evaluation_eligible']], validate='one_to_one').reset_index()
    summary, total_membership = [], 0
    for row in selected.itertuples(index=False):
        cutoff, end = row.parameter_cutoff, min(row.deployment_end, stop)
        old = z.forecast_at.lt(cutoff) & z.origin_eligible
        mature = z.label_ready_at.le(cutoff)
        training = old & mature & z.label_available
        prediction = z.forecast_at.ge(cutoff) & z.forecast_at.lt(end)
        expected = []
        for role, mask in [('training', training), ('prediction', prediction)]:
            part = z.loc[mask, ['episode_id', 'forecast_at', 'origin_eligible',
                                'label_ready_at', 'label_available']].copy()
            part['fold_id'] = int(row.cycle_id)
            part['role'] = role
            part['horizon_sec'] = 300
            part['label_mature_at_refit'] = part.label_ready_at.le(cutoff)
            part['evaluation_eligible'] = part.origin_eligible & part.label_available & (role == 'prediction')
            expected.append(part)
        expected = pd.concat(expected, ignore_index=True)
        _same(actual.loc[actual.fold_id.eq(row.cycle_id)], expected,
              ['episode_id'], 'membership fold '+str(row.cycle_id))
        total_membership += len(expected)
        summary.append(dict(fold_id=int(row.cycle_id), fit_cutoff=cutoff,
            prediction_start=cutoff, prediction_end_exclusive=end, primary_horizon_sec=300,
            training_origins=int(training.sum()), earlier_origin_eligible=int(old.sum()),
            earlier_labels_not_yet_mature=int((old & ~mature).sum()),
            earlier_mature_labels_unavailable=int((old & mature & ~z.label_available).sum()),
            prediction_origins=int(prediction.sum()),
            prediction_origin_eligible=int((prediction & z.origin_eligible).sum()),
            prediction_evaluable=int((prediction & z.evaluation_eligible).sum())))
    expected_folds = pd.DataFrame(summary)
    _same(artifacts['folds'], expected_folds, ['fold_id'], 'fold summaries')
    _require(len(actual) == total_membership, 'Membership conservation failed')
    predicted = actual.loc[actual.role.eq('prediction'), 'episode_id']
    wanted = origins.loc[origins.forecast_at.ge(start) & origins.forecast_at.lt(stop), 'episode_id']
    _require(not predicted.duplicated().any() and set(predicted) == set(wanted),
             'Prediction issuance dropped, duplicated or used future availability')
    return expected_folds, total_membership


def verify_prediction_ledger(anchors, features, scaling, outcomes, schedule,
                             artifacts, *, sample_end,
                             evaluation_start='2024-05-06T00:00:00+00:00'):
    """Independently verify all five saved ledger tables against their parents."""
    _require({'origins', 'predictors', 'labels', 'folds', 'membership'}.issubset(artifacts),
             'Missing reconstruction artifacts')
    start, stop = pd.Timestamp(evaluation_start), pd.Timestamp(sample_end)
    _utc([start, stop], 'evaluation boundaries')
    _require(start < stop, 'Invalid evaluation interval')
    a, f = _sources(anchors, features, scaling)
    _require(a.available_at.le(stop).all(), 'Origin after sample end')
    origins, predictors = _reference_origins(a, f, scaling)
    _same(artifacts['origins'], origins, ['episode_id'], 'origins')
    _same(artifacts['predictors'], predictors, ['episode_id'], 'strict-prior predictors')
    labels = _reference_labels(origins, outcomes, stop)
    _same(artifacts['labels'], labels, ['episode_id', 'horizon_sec'], 'labels')
    folds, membership_rows = _verify_folds(origins, labels, schedule, artifacts, start, stop)
    expected_counts = {'all_origins': len(origins),
        'origin_eligible': int(origins.origin_eligible.sum()),
        'origin_ineligible': int((~origins.origin_eligible).sum())}
    for name, series in [('exclusion_reasons', origins.origin_exclusion_reason),
        ('direction_counts', origins.direction_status),
        ('eligible_initial_sign_counts', origins.loc[origins.origin_eligible, 'initial_sign'])]:
        expected_counts[name] = {str(k): int(v) for k, v in series.value_counts().items()}
    remaining, waterfall = len(origins), []
    for step in ['confirmed_onset', 'defined_detection_direction', 'prior_state_available',
                 'positive_pre_spread', 'positive_pre_size', 'known_initial_quotes',
                 'finite_prior_predictors']:
        excluded = int(origins.origin_exclusion_reason.eq(step).sum())
        remaining -= excluded
        waterfall.append({'step': step, 'remaining': remaining, 'newly_excluded': excluded})
    expected_counts['waterfall'] = waterfall
    for key, value in expected_counts.items():
        if 'origin_statistics' in artifacts:
            _require(artifacts['origin_statistics'].get(key) == value,
                     'Origin statistics count mismatch: '+key)
    for key, value in {'folds': len(folds), 'all_origins': len(origins),
        'warmup_origins': int(origins.forecast_at.lt(start).sum()),
        'origins_at_sample_end_not_in_evaluation_window': int(origins.forecast_at.eq(stop).sum()),
        'prediction_origins': int(folds.prediction_origins.sum()),
        'prediction_origin_eligible': int(folds.prediction_origin_eligible.sum()),
        'prediction_evaluable': int(folds.prediction_evaluable.sum()),
        'training_membership_rows': int(folds.training_origins.sum())}.items():
        if 'fold_statistics' in artifacts:
            _require(artifacts['fold_statistics'].get(key) == value,
                     'Fold statistics count mismatch: '+key)
    return {'status': 'passed', 'origins_verified': len(origins),
        'origin_eligible_verified': int(origins.origin_eligible.sum()),
        'predictor_rows_verified': len(predictors), 'labels_verified': len(labels),
        'folds_verified': len(folds), 'membership_rows_verified': membership_rows,
        'prediction_origins_verified': int(folds.prediction_origins.sum()),
        'checks': ['full_parent_identity_and_strict_left_timing',
            'strict_prior_predictor_allowlist_and_independent_algebra',
            'origin_eligibility_from_current_and_prior_primitives',
            'zero_and_negative_J_retained', 'all_origin_horizon_pairs_preserved',
            'scheduled_label_maturity_not_quote_timestamp', 'endpoint_status_and_J_C_I',
            'expanding_primary300_training_with_maturity_at_or_before_cutoff',
            'all_origins_issued_in_each_half_open_prediction_block',
            'derived_statistics_counts'],
        'producer_helpers_imported': False, 'empirical_models_fitted': 0,
        'forecast_performance_validated': False, 'live_feed_validated': False,
        'untouched_holdout': False}

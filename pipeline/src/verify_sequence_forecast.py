"""Independent QR and loss-accounting audit for the fixed 10b sequence comparison.

No forecast producer helper is imported. This verifies arithmetic and declared
information timing, not independence, forecast superiority or economic cause.
"""
import json
import math

import numpy as np
import pandas as pd


METHODS = ('zero', 'training_mean', 'baseline', 'sequence')
BASE_NUMERIC = ('direction', 'initial_pips', 'aligned_pressure', 'pressure_magnitude',
    'gross_pressure_scaled', 'variation_standardized', 'pre_spread_pips',
    'log_pre_displayed_size', 'log1p_prior_quote_updates', 'aligned_sign_balance',
    'nonzero_fraction', 'composition_reference', 'aligned_recent_5s_pressure',
    'aligned_recent_30s_pressure', 'recent_5s_gross_scaled', 'recent_30s_gross_scaled',
    'aligned_prior_5s_price_pips', 'aligned_prior_30s_price_pips', 'aligned_prior_300s_price_pips')
PRIOR_PREDICTORS = (*BASE_NUMERIC, 'sequence_z', 'onset_hour_utc', 'onset_weekday_utc')
CALENDAR = tuple([f'hour_{h:02d}' for h in range(1, 24)] +
                 [f'weekday_{d}' for d in (1, 2, 3, 4, 6)])


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _fields(frame, fields, label, exact=False):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique,
             label + ': unique DataFrame columns required')
    _require(set(fields) == set(frame) if exact else set(fields).issubset(frame),
             label + ': unexpected or missing columns')


def _keys(frame, fields, label):
    _fields(frame, fields, label)
    _require(not frame[fields].isna().any().any() and
             not frame.duplicated(fields).any(), label + ': duplicate/missing keys')
    for name in set(fields) & {'episode_id', 'fold_id', 'horizon_sec', 'lag'}:
        _require(pd.api.types.is_integer_dtype(frame[name]) and
                 not pd.api.types.is_bool_dtype(frame[name]) and frame[name].ge(0).all(),
                 label + ': nonnegative integer IDs required')


def _utc(values, label, missing=False):
    stamps = pd.DatetimeIndex(values)
    _require(str(stamps.tz) == 'UTC' and (missing or not stamps.hasnans),
             label + ': UTC timestamps required')


def _compare(actual, expected, keys, label, *, rtol=1e-9, atol=1e-10):
    _fields(actual, expected.columns, label, exact=True)
    _keys(actual, keys, label)
    aa = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    ee = expected.sort_values(keys).reset_index(drop=True)
    _require(len(aa) == len(ee), label + ': row count mismatch')
    for name in keys:
        _require(np.array_equal(aa[name], ee[name]), label + ': exact keys differ')
    for name in ee:
        if pd.api.types.is_bool_dtype(ee[name]):
            _require(pd.api.types.is_bool_dtype(aa[name]) and aa[name].notna().all(),
                     label + ': nonmissing boolean required')
    pd.testing.assert_frame_equal(aa, ee, check_dtype=False, check_exact=False,
                                  rtol=rtol, atol=atol, obj=label)


def _numeric_matrix(frame, columns):
    arrays = []
    for name in columns:
        if name == 'intercept':
            arrays.append(np.ones(len(frame)))
        elif name.startswith('hour_'):
            arrays.append(frame.onset_hour_utc.eq(int(name[5:])).to_numpy(dtype=float))
        elif name.startswith('weekday_'):
            arrays.append(frame.onset_weekday_utc.eq(int(name[8:])).to_numpy(dtype=float))
        else:
            arrays.append(frame[name].to_numpy(dtype=float))
    return np.column_stack(arrays)


def _parents(parent):
    _require({'origins', 'predictors', 'labels', 'folds', 'membership'}.issubset(parent),
             'Missing forecast parent tables')
    origins, predictors, labels, folds, membership = [parent[k].copy() for k in
        ['origins', 'predictors', 'labels', 'folds', 'membership']]
    _keys(origins, ['episode_id'], 'origins')
    _keys(predictors, ['episode_id'], 'predictors')
    _keys(labels, ['episode_id', 'horizon_sec'], 'labels')
    _keys(folds, ['fold_id'], 'folds')
    _keys(membership, ['fold_id', 'role', 'episode_id'], 'membership')
    _require(set(membership.fold_id).issubset(set(folds.fold_id)) and
             set(membership.episode_id).issubset(set(origins.episode_id)), 'Orphan membership record')
    _fields(predictors, ['episode_id', 'bin_start', 'forecast_at', *PRIOR_PREDICTORS],
             'First-alert source predictors', exact=True)
    o = origins.sort_values('episode_id').reset_index(drop=True)
    x = predictors.sort_values('episode_id').reset_index(drop=True)
    _require(np.array_equal(o.episode_id, x.episode_id), 'Missing predictor origin')
    for name in ['bin_start', 'forecast_at']:
        _utc(o[name], name); _utc(x[name], name)
        _require(o[name].equals(x[name]), 'Predictor/metadata clock mismatch')
    _require(o.forecast_at.eq(o.bin_start + pd.Timedelta(1, unit='s')).all(),
             'Changed first-alert clock')
    _require(o.origin_eligible.notna().all() and pd.api.types.is_bool_dtype(o.origin_eligible),
             'Origin eligibility must be nonmissing boolean')
    _require(x.onset_hour_utc.eq(o.bin_start.dt.hour).all() and
             x.onset_weekday_utc.eq(o.bin_start.dt.dayofweek).all(),
             'Calendar metadata differ from prior onset clock')
    _require(x.onset_weekday_utc.isin([0, 1, 2, 3, 4, 6]).all(), 'Unexpected Saturday support')
    eligible = o.origin_eligible
    _require(np.isfinite(x.loc[eligible, list(PRIOR_PREDICTORS)]).all().all(),
             'Eligible origin has a missing first-alert predictor')
    _require(x.loc[eligible, 'pre_spread_pips'].gt(0).all() and
             np.isfinite(o.loc[eligible, 'initial_pips']).all(),
             'Eligible origin lacks positive spread or known J')
    for field in ['direction', 'initial_pips']:
        _require(x[field].equals(o[field]), 'First-alert predictor differs from origin: '+field)
    _require(x.loc[eligible, 'direction'].isin([-1., 1.]).all(), 'Undefined detection direction')
    np.testing.assert_allclose(x.loc[eligible, 'pressure_magnitude'],
        x.loc[eligible, 'aligned_pressure'].abs(), rtol=1e-12, atol=1e-12,
        err_msg='Pressure magnitude differs from absolute aligned pressure')
    _require(x.loc[eligible, 'nonzero_fraction'].between(0., 1.).all() and
             x.loc[eligible, 'aligned_sign_balance'].between(-1., 1.).all(),
             'Invalid sign composition fraction')
    composition = ((60*x.loc[eligible, 'aligned_sign_balance'])**2 -
                   60*x.loc[eligible, 'nonzero_fraction'])/(60*59)
    np.testing.assert_allclose(x.loc[eligible, 'composition_reference'], composition,
        rtol=1e-12, atol=1e-12, err_msg='Composition reference changed')
    _require(x.loc[eligible, 'sequence_z'].between(-2., 2.).all(), 'Invalid sequence range')
    primary = labels.loc[labels.horizon_sec.eq(300)].sort_values('episode_id').reset_index(drop=True)
    _require(np.array_equal(primary.episode_id, o.episode_id), 'Primary label row missing')
    for name in ['forecast_at', 'label_ready_at']:
        _utc(primary[name], 'primary '+name)
    _require(primary.forecast_at.equals(o.forecast_at) and
             primary.label_ready_at.eq(o.forecast_at+pd.Timedelta(300, unit='s')).all(), 'Wrong primary label maturity')
    _require(pd.api.types.is_bool_dtype(primary.label_available) and primary.label_available.notna().all(),
             'Label availability must be nonmissing boolean')
    # response finiteness is examined only after selecting a matured training set.
    for name in ['fit_cutoff', 'prediction_start', 'prediction_end_exclusive']:
        _utc(folds[name], 'fold '+name)
    fs = folds.sort_values('fit_cutoff').reset_index(drop=True)
    _require(not fs.fit_cutoff.duplicated().any() and fs.fit_cutoff.eq(fs.prediction_start).all()
             and fs.prediction_start.lt(fs.prediction_end_exclusive).all(), 'Invalid fold intervals')
    if len(fs) > 1:
        _require(np.array_equal(fs.prediction_end_exclusive.iloc[:-1], fs.prediction_start.iloc[1:]),
                 'Discontinuous forecast blocks')
    return o, x, primary, fs, membership


def verify_source_projection(parent, features):
    """Bind allowlisted model values to the independently verified 10a table.

    This function is a provenance/projection check, not a fresh source reconstruction.
    The runner must separately verify the frozen 10a file hashes.
    """
    _keys(features, ['episode_id'], 'frozen 10a features')
    expected = features[['episode_id', 'bin_start', 'forecast_at', *PRIOR_PREDICTORS]]
    _compare(parent['predictors'], expected, ['episode_id'], 'exact frozen predictor projection',
             rtol=0., atol=0.)
    origins = parent['origins'].sort_values('episode_id').reset_index(drop=True)
    source = features.sort_values('episode_id').reset_index(drop=True)
    _require(np.array_equal(origins.episode_id, source.episode_id), 'Source projection lost origin')
    for field in ['bin_start', 'forecast_at', 'direction', 'initial_pips']:
        pd.testing.assert_series_equal(origins[field], source[field], check_names=False,
                                       check_dtype=True, check_exact=True)
    _require(origins.origin_eligible.equals(source.inherited_origin_eligible) and
             origins.origin_eligible.equals(source.combined_eligible),
             'Source projection changed inherited eligibility')
    eligible = source.inherited_origin_eligible
    np.testing.assert_allclose(source.loc[eligible, 'sequence_z'],
        source.loc[eligible, 'adjacent_agreement']-source.loc[eligible, 'composition_reference'],
        rtol=0., atol=0., err_msg='Frozen sequence centering identity changed')
    return {'status': 'passed', 'origins_verified': len(source),
            'exact_predictor_projection_verified': True,
            'inherited_eligibility_preserved': True, 'future_columns_excluded': True}


def _fold_inputs(o, x, labels, fold, membership):
    cutoff = fold.fit_cutoff
    train = (o.origin_eligible & o.forecast_at.lt(cutoff) & labels.label_available &
             labels.label_ready_at.le(cutoff))
    prediction = o.forecast_at.ge(cutoff) & o.forecast_at.lt(fold.prediction_end_exclusive)
    _require(int(fold.training_origins) == int(train.sum()) and
             int(fold.prediction_origins) == int(prediction.sum()) and
             int(fold.prediction_origin_eligible) == int((prediction & o.origin_eligible).sum()),
             'Fold counts disagree with independently reconstructed origins')
    actual = membership.loc[membership.fold_id.eq(fold.fold_id)]
    _require(actual.role.isin(['training', 'prediction']).all(), 'Unknown fold role')
    for role, mask in [('training', train), ('prediction', prediction)]:
        subset = actual.loc[actual.role.eq(role)]
        _require(set(subset.episode_id) == set(o.loc[mask, 'episode_id']) and
                 len(subset) == int(mask.sum()), 'Changed '+role+' membership')
        if 'horizon_sec' in subset:
            _require(subset.horizon_sec.eq(300).all(), 'Secondary labels used for fitting')
        subset = subset.set_index('episode_id').loc[o.loc[mask, 'episode_id']]
        for field in ['forecast_at', 'origin_eligible']:
            _require(np.array_equal(subset[field], o.loc[mask, field]),
                     'Membership origin metadata differ: '+field)
        if role == 'training':
            _require(np.array_equal(subset.label_ready_at, labels.loc[mask, 'label_ready_at']) and
                     subset.label_available.eq(True).all(), 'Membership training maturity changed')
    tx = x.loc[train].reset_index(drop=True)
    ty = labels.loc[train, 'continuation_pips'].to_numpy(dtype=float)
    _require(len(tx) > 0 and np.isfinite(ty).all(), 'No finite matured training sample')
    _require(tx.onset_hour_utc.eq(0).any() and tx.onset_weekday_utc.eq(0).any(),
             'Training reference hour zero and Monday must both be supported')
    hours = sorted(int(v) for v in tx.onset_hour_utc.unique())
    weekdays = sorted(int(v) for v in tx.onset_weekday_utc.unique())
    kept_calendar = [name for name in CALENDAR if
        (int(name[5:]) in hours if name.startswith('hour_') else int(name[8:]) in weekdays)]
    return train, prediction, tx, ty, hours, weekdays, kept_calendar


def _qr_fit(tx, ty, columns):
    matrix = _numeric_matrix(tx, columns)
    n, p = matrix.shape
    _require(n > p and np.isfinite(matrix).all(), 'Insufficient or nonfinite training design')
    maxima = np.max(np.abs(matrix), axis=0)
    _require((maxima > 0).all(), 'Zero substantive design column')
    # stable RMS units; QR is independently solved without using producer fits.
    units = maxima * (np.linalg.norm(matrix/maxima, axis=0)/math.sqrt(n))
    _require(np.isfinite(units).all() and (units > 0).all(), 'Invalid equilibrated column scale')
    scaled = matrix/units
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = np.finfo(float).eps*max(n, p)*singular[0]
    rank = int((singular > tolerance).sum())
    _require(rank == p, 'Declared state model is rank deficient')
    q, r = np.linalg.qr(scaled, mode='reduced')
    beta = np.linalg.solve(r, q.T@ty)/units
    fitted = matrix@beta
    residual = ty-fitted
    return {'matrix': matrix, 'units': units, 'beta': beta, 'rank': rank,
            'tolerance': float(tolerance), 'condition': float(singular[0]/singular[-1]),
            'residual': residual, 'fitted': fitted, 'singular': singular}


def _forecast_reference(parent):
    o, x, labels, folds, membership = _parents(parent)
    predictions, coefficients, diagnostic_data = [], [], []
    for fold in folds.itertuples(index=False):
        train, prediction, tx, ty, hours, weekdays, calendar = _fold_inputs(
            o, x, labels, fold, membership)
        mean = math.fsum(ty)/len(ty)
        po = o.loc[prediction].reset_index(drop=True)
        px = x.loc[prediction].reset_index(drop=True)
        eligible = po.origin_eligible.to_numpy(dtype=bool)
        unseen_hour = ~px.onset_hour_utc.isin(hours)
        unseen_weekday = ~px.onset_weekday_utc.isin(weekdays)
        fallback = eligible & (unseen_hour | unseen_weekday)
        supported = eligible & ~(unseen_hour | unseen_weekday)
        for method in METHODS:
            estimate = np.full(len(po), np.nan)
            if method == 'zero':
                estimate[eligible] = 0.
                status = np.where(eligible, 'zero_reference', 'origin_ineligible')
            elif method == 'training_mean':
                estimate[eligible] = mean
                status = np.where(eligible, 'training_mean', 'origin_ineligible')
            else:
                columns = ['intercept', *BASE_NUMERIC, *calendar]
                if method == 'sequence':
                    columns.append('sequence_z')
                solved = _qr_fit(tx, ty, columns)
                if method == 'sequence':
                    raw_columns = solved['matrix'].copy()
                    sequence_index = columns.index('sequence_z')
                    composition_index = columns.index('composition_reference')
                    raw_columns[:, sequence_index] += raw_columns[:, composition_index]
                    raw_beta = solved['beta'].copy()
                    raw_beta[composition_index] -= raw_beta[sequence_index]
                    np.testing.assert_allclose(raw_columns @ raw_beta, solved['fitted'],
                        rtol=1e-10, atol=1e-10,
                        err_msg='S/A linear reparameterization failed on training design')
                    px_raw = _numeric_matrix(px.loc[supported], columns)
                    original_prediction = px_raw @ solved['beta']
                    px_raw[:, sequence_index] += px_raw[:, composition_index]
                    np.testing.assert_allclose(px_raw @ raw_beta, original_prediction,
                        rtol=1e-10, atol=1e-10,
                        err_msg='S/A linear reparameterization failed on prediction design')
                estimate[eligible] = mean
                estimate[supported] = _numeric_matrix(px.loc[supported], columns)@solved['beta']
                status = np.where(~eligible, 'origin_ineligible',
                                  np.where(fallback, 'unseen_calendar_mean', 'estimated'))
                for term, value, units in zip(columns, solved['beta'], solved['units']):
                    coefficients.append(dict(fold_id=int(fold.fold_id), method=method,
                        term=term, coefficient=float(value), equilibration=float(units)))
                diagnostic_data.append(dict(fold=fold, method=method, columns=columns,
                    train_mask=train, prediction_mask=prediction, y=ty, mean=mean,
                    hours=hours, weekdays=weekdays, fallback=fallback, supported=supported,
                    unseen_hour=unseen_hour, unseen_weekday=unseen_weekday,
                    eligible=eligible, solution=solved))
            out = po[['episode_id', 'bin_start', 'forecast_at', 'origin_eligible', 'initial_pips']].copy()
            out.insert(0, 'fold_id', int(fold.fold_id))
            out['method'] = method
            out['forecast_status'] = status
            out['unseen_hour'] = ((eligible & unseen_hour) if method in ('baseline', 'sequence')
                                  else np.zeros(len(po), dtype=bool))
            out['unseen_weekday'] = ((eligible & unseen_weekday) if method in ('baseline', 'sequence')
                                     else np.zeros(len(po), dtype=bool))
            out['predicted_C'] = estimate
            out['predicted_I'] = po.initial_pips.to_numpy(dtype=float)+estimate
            predictions.append(out)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients), diagnostic_data


def verify_forecasts(parent, artifacts):
    """Reconstruct fitted coefficients through QR before any loss evaluation."""
    _require({'forecasts', 'coefficients', 'fit_diagnostics', 'statistics'}.issubset(artifacts),
             'Missing forecast artifacts')
    forecasts, coefficients, diagnostics = _forecast_reference(parent)
    _compare(artifacts['forecasts'], forecasts, ['fold_id', 'episode_id', 'method'], 'forecasts')
    _compare(artifacts['coefficients'], coefficients, ['fold_id', 'method', 'term'], 'QR coefficients')
    primary = parent['labels'].loc[parent['labels'].horizon_sec.eq(300)].set_index('episode_id')
    o = parent['origins'].sort_values('episode_id').reset_index(drop=True)
    records = []
    for item in diagnostics:
        solved, fold = item['solution'], item['fold']
        ids = o.loc[item['train_mask'], 'episode_id']
        records.append({'fold_id': int(fold.fold_id), 'method': item['method'],
            'fit_cutoff': fold.fit_cutoff,
            'max_training_label_ready_at': primary.loc[ids, 'label_ready_at'].max(),
            'n_train': len(item['y']), 'n_columns': len(item['columns']),
            'rank': solved['rank'], 'condition_equilibrated': solved['condition'],
            'rank_tolerance': solved['tolerance'], 'training_mean': item['mean'],
            'observed_hours_json': json.dumps(item['hours']),
            'observed_weekdays_json': json.dumps(item['weekdays']),
            'dropped_calendar_terms_json': json.dumps([c for c in CALENDAR if c not in item['columns']]),
            'n_prediction_origin_eligible': int(item['eligible'].sum()),
            'unseen_calendar_fallbacks': int(item['fallback'].sum())})
    expected_diagnostics = pd.DataFrame(records)
    _compare(artifacts['fit_diagnostics'], expected_diagnostics,
             ['fold_id', 'method'], 'fit diagnostics')
    stats = artifacts['statistics']
    origins = forecasts.loc[forecasts.method.eq('zero')]
    counts = {'folds': len(parent['folds']), 'prediction_origins': len(origins),
        'eligible_origins': int(origins.origin_eligible.sum()),
        'ineligible_origins': int((~origins.origin_eligible).sum()),
        'forecast_rows': len(forecasts), 'forecasts_computed': int(origins.origin_eligible.sum())*4,
        'ols_fits': len(expected_diagnostics), 'training_mean_calibrations': len(parent['folds']),
        'empirical_models_fitted': len(expected_diagnostics)+len(parent['folds']),
        'methods': list(METHODS), 'primary_horizon_sec': 300,
        'numeric_baseline_columns': list(BASE_NUMERIC), 'sequence_extension_columns': ['sequence_z'],
        'prediction_clock': 'first_alert_s_plus_1_second',
        'ols_unseen_calendar_fallbacks': {m: int(expected_diagnostics.loc[
            expected_diagnostics.method.eq(m), 'unseen_calendar_fallbacks'].sum())
            for m in ['baseline', 'sequence']}}
    for key, value in counts.items():
        _require(stats.get(key) == value, 'Forecast statistics disagree: '+key)
    np.testing.assert_allclose(stats['maximum_equilibrated_condition'],
        expected_diagnostics.condition_equilibrated.max(), rtol=1e-9, atol=1e-10)
    for key in ['future_evaluation_labels_attached', 'forecast_performance_computed',
                'inference_computed', 'untouched_holdout']:
        _require(stats.get(key) is False, 'Invalid fit-stage claim: '+key)
    expected_stats = dict(counts,
        maximum_equilibrated_condition=float(expected_diagnostics.condition_equilibrated.max()),
        future_evaluation_labels_attached=False, forecast_performance_computed=False,
        inference_computed=False, untouched_holdout=False)
    _json_compare(stats, expected_stats, 'fit statistics')
    return {'status': 'passed', 'coefficient_solver': 'independent equilibrated QR',
        'ols_fits_verified': len(expected_diagnostics), 'coefficients_verified': len(coefficients),
        'forecast_rows_verified': len(forecasts), 'prediction_origins_verified': len(origins),
        'eligible_origins_verified': int(origins.origin_eligible.sum()),
        'all_origin_and_method_keys_preserved': True, 'future_outcomes_in_forecast_table': False,
        'training_membership_reconstructed_from_maturity': True,
        'unseen_marginal_level_fallbacks_verified': True,
        'producer_helpers_imported': False, 'sequence_adjacency_reparameterization_verified': True,
        'forecast_superiority_established': False}


FORECAST_COLUMNS = ['fold_id', 'episode_id', 'bin_start', 'forecast_at',
    'origin_eligible', 'initial_pips', 'method', 'forecast_status', 'unseen_hour',
    'unseen_weekday', 'predicted_C', 'predicted_I']
CONTRASTS = {f'sequence_minus_{m}': m for m in ['baseline', 'training_mean', 'zero']}


def _evaluation_base(forecasts, labels, start, end):
    _fields(forecasts, FORECAST_COLUMNS, 'unscored forecasts', exact=True)
    _keys(forecasts, ['episode_id', 'method'], 'unscored forecasts')
    _require(set(forecasts.method) == set(METHODS), 'Incomplete forecast methods')
    _utc(forecasts.forecast_at, 'forecast origin')
    _require(forecasts.forecast_at.ge(start).all() & forecasts.forecast_at.lt(end).all(),
             'Forecast outside evaluation interval')
    _require(forecasts.groupby('episode_id').size().eq(4).all(), 'Dropped method/origin pair')
    base = forecasts.loc[forecasts.method.eq('zero'),
        ['episode_id', 'fold_id', 'forecast_at', 'origin_eligible', 'initial_pips']].set_index('episode_id').sort_index()
    predictions = pd.DataFrame(index=base.index)
    for method in METHODS:
        part = forecasts.loc[forecasts.method.eq(method)].set_index('episode_id').reindex(base.index)
        for field in ['fold_id', 'forecast_at', 'origin_eligible', 'initial_pips']:
            pd.testing.assert_series_equal(part[field], base[field], check_names=False,
                                            check_dtype=True, check_exact=True)
        issued = part.origin_eligible
        _require(np.isfinite(part.loc[issued, ['predicted_C', 'predicted_I']]).all().all(),
                 'Issued forecast is not finite')
        _require(part.loc[~issued, ['predicted_C', 'predicted_I']].isna().all().all(),
                 'Abstention has a numeric forecast')
        _require(np.array_equal(part.loc[issued, 'predicted_I'],
            part.loc[issued, 'initial_pips']+part.loc[issued, 'predicted_C']), 'Forecast I differs from J+C')
        predictions[method] = part.predicted_C
    selected = labels.loc[labels.horizon_sec.eq(300) & labels.episode_id.isin(base.index)].copy()
    _keys(selected, ['episode_id'], 'primary evaluation labels')
    _require(pd.api.types.is_bool_dtype(selected.label_available) and selected.label_available.notna().all(),
             'Evaluation availability must be nonmissing boolean')
    selected = selected.set_index('episode_id')
    common = selected.index.intersection(base.index)
    _require(selected.loc[common, 'forecast_at'].equals(base.loc[common, 'forecast_at']),
             'Evaluation source origin mismatch')
    _require(selected.loc[common, 'label_ready_at'].equals(
        base.loc[common, 'forecast_at']+pd.Timedelta(300, unit='s')), 'Evaluation source maturity mismatch')
    _require((~selected.label_available | selected.label_ready_at.le(end)).all() and
             np.isfinite(selected.loc[selected.label_available, 'continuation_pips']).all(),
             'Available evaluation label is nonfinite or beyond sample end')
    _require(np.array_equal(selected.label_available, np.isfinite(selected.continuation_pips))
             and not np.isinf(selected.continuation_pips).any(),
             'Evaluation availability differs from finite recorded C')
    base['label_available'] = selected.label_available.reindex(base.index).eq(True)
    base['label_status'] = selected.label_status.reindex(base.index).fillna('missing_record')
    base['observed_C'] = selected.continuation_pips.reindex(base.index).where(base.label_available)
    base['scored'] = base.origin_eligible & base.label_available
    _require(base.scored.any(), 'No evaluable issued origins')
    base['date'] = base.forecast_at.dt.floor('D')
    base['quarter'] = base.forecast_at.dt.tz_localize(None).dt.to_period('Q').astype(str)
    for method in METHODS:
        base['loss_'+method] = ((predictions[method]-base.observed_C)**2).where(base.scored)
    for name, other in CONTRASTS.items():
        stable = ((predictions[other]-predictions.sequence) *
                  (2*base.observed_C-predictions.sequence-predictions[other])).where(base.scored)
        base['delta_'+name] = stable
        ordinary = base['loss_sequence']-base['loss_'+other]
        scale = ((abs(base.observed_C)+abs(predictions.sequence))**2 +
                 (abs(base.observed_C)+abs(predictions[other]))**2)
        bound = 32*np.finfo(float).eps*scale
        _require((abs(stable[base.scored]-ordinary[base.scored]) <= bound[base.scored]).all(),
                 'Stable loss difference disagrees beyond floating-point roundoff')
    _require(np.isfinite(base.loc[base.scored,
        ['loss_'+m for m in METHODS]+['delta_'+c for c in CONTRASTS]]).all().all(),
        'Nonfinite loss or difference')
    return base, predictions


def _loss_aggregate(base, group_column, groups, include_means):
    data = {key: value for key, value in base.groupby(group_column, sort=False)}
    rows = []
    for key in groups:
        part = data.get(key, base.iloc[:0])
        scored = part.loc[part.scored]
        n = len(scored)
        out = {group_column: key, 'n_origins': len(part),
               'n_issued': int(part.origin_eligible.sum()), 'n_scored': n}
        for method in METHODS:
            total = math.fsum(scored['loss_'+method])
            out['sse_'+method] = total
            if include_means:
                out['mse_'+method] = total/n if n else np.nan
        for name in CONTRASTS:
            total = math.fsum(scored['delta_'+name])
            out['sum_delta_'+name] = total
            if include_means:
                out['mean_delta_'+name] = total/n if n else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _evaluation_reference(forecasts, labels, start, end):
    base, predictions = _evaluation_base(forecasts, labels, start, end)
    good = base.scored
    n = int(good.sum())
    metrics, event_parts = [], []
    for method in METHODS:
        sse = math.fsum(base.loc[good, 'loss_'+method])
        metrics.append(dict(method=method, n_origins=len(base),
            n_issued=int(base.origin_eligible.sum()), n_scored=n,
            n_issued_unavailable=int((base.origin_eligible & ~base.label_available).sum()),
            sse=sse, mse=sse/n, rmse=math.sqrt(sse/n)))
        part = forecasts.loc[forecasts.method.eq(method)].copy()
        for field in ['label_available', 'label_status', 'observed_C', 'scored']:
            part[field] = part.episode_id.map(base[field])
        errors = (predictions[method]-base.observed_C).where(good)
        part['forecast_error'] = part.episode_id.map(errors)
        part['squared_error'] = part.episode_id.map(base['loss_'+method])
        event_parts.append(part)
    metrics = pd.DataFrame(metrics)
    comparisons = []
    for name, other in CONTRASTS.items():
        total = math.fsum(base.loc[good, 'delta_'+name])
        full_mse = float(metrics.loc[metrics.method.eq('sequence'), 'mse'].iloc[0])
        other_mse = float(metrics.loc[metrics.method.eq(other), 'mse'].iloc[0])
        comparisons.append(dict(contrast=name, benchmark=other, n=n,
            sum_loss_difference=total, mean_loss_difference=total/n,
            relative_mse_reduction=1-full_mse/other_mse if other_mse > 0 else np.nan,
            rmse_difference=math.sqrt(full_mse)-math.sqrt(other_mse),
            relative_mse_status='available' if other_mse > 0 else 'zero_benchmark_mse'))
    comparisons = pd.DataFrame(comparisons)
    calendar = pd.date_range(start, end, freq='D', inclusive='left')
    daily = _loss_aggregate(base, 'date', calendar, False)
    fold = _loss_aggregate(base, 'fold_id', sorted(base.fold_id.unique()), True)
    quarter = _loss_aggregate(base, 'quarter', sorted(set(calendar.tz_localize(None).to_period('Q').astype(str))), True)
    serial, concentration = [], {}
    for name in CONTRASTS:
        mean = math.fsum(base.loc[good, 'delta_'+name])/n
        totals = daily['sum_delta_'+name].to_numpy(dtype=float)
        psi = totals-daily.n_scored.to_numpy(dtype=float)*mean
        daily['psi_'+name] = psi
        energy = math.fsum(psi**2)
        absolute = math.fsum(abs(totals))
        for lag in range(41):
            pairs = len(calendar)-lag
            product = math.fsum(float(psi[i]*psi[i-lag]) for i in range(lag, len(psi))) if pairs > 0 else np.nan
            serial.append(dict(contrast=name, lag=lag, n_pairs=max(0, pairs), cross_product=product,
                normalized_autocorrelation=product/energy if pairs > 0 and energy > 0 else np.nan,
                status='no_pairs' if pairs <= 0 else 'zero_energy' if energy == 0 else 'available'))
        concentration[name] = {
            'sum_absolute_daily_loss_difference': absolute,
            'maximum_absolute_daily_loss_difference_share': float(np.max(abs(totals))/absolute) if absolute else None,
            'maximum_absolute_difference_date': daily.date.iloc[int(np.argmax(abs(totals)))].isoformat() if absolute else None,
            'sum_squared_daily_centered_difference': energy, 'zero_energy': energy == 0,
            'maximum_daily_energy_share': float(np.max(psi**2)/energy) if energy else None,
            'maximum_energy_date': daily.date.iloc[int(np.argmax(psi**2))].isoformat() if energy else None,
            'daily_centering_residual': math.fsum(psi)}
    return {'scored_events': pd.concat(event_parts, ignore_index=True), 'metrics': metrics,
        'comparisons': comparisons, 'fold_losses': fold, 'quarter_losses': quarter,
        'daily_losses': daily, 'serial_dependence': pd.DataFrame(serial)}, concentration, base


def _json_compare(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label+': keys differ')
        for name in expected:
            _json_compare(actual[name], expected[name], label+'.'+name)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), label+': rows differ')
        for i, value in enumerate(expected):
            _json_compare(actual[i], value, label+f'[{i}]')
    elif isinstance(expected, (float, np.floating)):
        if not math.isfinite(expected):
            _require(actual is None or (isinstance(actual, (float, np.floating)) and not math.isfinite(actual)),
                     label+': undefined statistic became finite')
        else:
            _require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
                     label+': numeric statistic differs')
    elif isinstance(expected, (bool, np.bool_)):
        _require(type(actual) is bool and actual == bool(expected), label+': boolean differs')
    elif isinstance(expected, (int, np.integer)):
        _require(type(actual) is int and actual == int(expected), label+': integer differs')
    else:
        _require(actual == expected, label+': statistic differs')


def verify_evaluation(forecasts, labels, artifacts, *, start, end):
    """Reconstruct event-weighted squared losses and the complete calendar."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _utc([start, end], 'evaluation boundaries')
    _require(start < end and start == start.normalize() and end == end.normalize(),
             'Evaluation boundaries must be ordered UTC midnights')
    expected, diagnostics, base = _evaluation_reference(forecasts, labels, start, end)
    keys = {'scored_events': ['fold_id', 'episode_id', 'method'],
            'metrics': ['method'], 'comparisons': ['contrast'], 'fold_losses': ['fold_id'],
            'daily_losses': ['date'], 'quarter_losses': ['quarter'],
            'serial_dependence': ['contrast', 'lag']}
    _require(set(keys).issubset(artifacts) and 'statistics' in artifacts,
             'Missing evaluation artifacts')
    for table, key in keys.items():
        _compare(artifacts[table], expected[table], key, table)
    daily = expected['daily_losses']
    statistics = {'status': 'descriptive_sequence_forecast_evaluation', 'primary_horizon_sec': 300,
        'calendar_days': len(daily), 'days_with_scored_origins': int(daily.n_scored.gt(0).sum()),
        'prediction_origins': len(base), 'issued_origins': int(base.origin_eligible.sum()),
        'scored_origins': int(base.scored.sum()),
        'issued_unavailable': int((base.origin_eligible & ~base.label_available).sum()),
        'metrics': expected['metrics'].to_dict('records'),
        'comparisons': expected['comparisons'].to_dict('records'),
        'comparison_diagnostics': diagnostics, 'serial_max_lag': 40,
        'loss': 'raw squared C300 error in pips squared', 'empirical_models_fitted': 0,
        'inference_computed': False, 'uncertainty_computed': False, 'untouched_holdout': False,
        'causal_identification_established': False, 'novelty_established': False}
    _json_compare(artifacts['statistics'], statistics, 'evaluation statistics')
    return {'status': 'passed', 'scored_event_rows_verified': len(expected['scored_events']),
        'prediction_origins_verified': len(base), 'evaluable_origins_verified': int(base.scored.sum()),
        'issued_unavailable_verified': int((base.origin_eligible & ~base.label_available).sum()),
        'calendar_days_verified': len(daily), 'metrics_verified': len(expected['metrics']),
        'comparisons_verified': len(expected['comparisons']),
        'serial_rows_verified': len(expected['serial_dependence']),
        'event_weighting_verified': True, 'stable_delta_roundoff_bound_checked': True,
        'centering_uses_daily_event_counts': True, 'empty_calendar_dates_preserved': True,
        'producer_helpers_imported': False, 'inference_validated': False,
        'forecast_superiority_established': False}

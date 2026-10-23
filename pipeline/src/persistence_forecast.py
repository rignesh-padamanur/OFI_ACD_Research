"""Frozen first-alert C300 forecasts and a separate descriptive loss ledger.

Only matured training labels enter fitting. Evaluation is deliberately a separate
entry point so callers can serialize and hash forecasts before exposing labels.
"""
import json
import math

import numpy as np
import pandas as pd


METHODS = ('zero', 'training_mean', 'additive', 'interaction')
CONTINUOUS = ('signed_pressure', 'pressure_magnitude', 'variation_standardized',
              'pre_spread_pips', 'log_pre_displayed_size')
INTERACTION = 'variation_x_magnitude'
PREDICTORS = (*CONTINUOUS, INTERACTION, 'onset_hour_utc', 'onset_weekday_utc')
IDENTIFIERS = ('episode_id', 'bin_start', 'forecast_at')
CONTRASTS = {f'interaction_minus_{m}': m for m in ('additive', 'training_mean', 'zero')}
FORECAST_COLUMNS = ['fold_id', 'episode_id', 'bin_start', 'forecast_at',
                    'origin_eligible', 'initial_pips', 'method', 'forecast_status',
                    'unseen_hour', 'unseen_weekday', 'predicted_C', 'predicted_I']


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _fields(frame, names, label, exact=False):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
             and set(names).issubset(frame.columns), 'Missing or duplicate ' + label + ' fields')
    if exact:
        _require(set(frame.columns) == set(names), 'Unexpected ' + label + ' fields')


def _utc(series, label):
    _require(pd.api.types.is_datetime64_any_dtype(series.dtype)
             and str(series.dt.tz) == 'UTC' and series.notna().all(), 'UTC times required: ' + label)


def _bool(series, label):
    _require(pd.api.types.is_bool_dtype(series.dtype) and series.notna().all(), 'Boolean required: ' + label)


def _ids(series, label):
    _require(pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
             and series.notna().all() and series.ge(0).all(), 'Nonnegative integer IDs required: ' + label)


def _finite(values, label):
    _require(np.isfinite(np.asarray(values, dtype=float)).all(), 'Nonfinite ' + label)


def _sum(values):
    return math.fsum(float(v) for v in values)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _prepare(origins, predictors, folds, membership):
    _fields(origins, (*IDENTIFIERS, 'origin_eligible', 'initial_pips'), 'origin')
    _fields(predictors, (*IDENTIFIERS, *PREDICTORS), 'predictor', exact=True)
    _fields(folds, ('fold_id', 'fit_cutoff', 'prediction_start', 'prediction_end_exclusive',
                   'training_origins', 'prediction_origins', 'prediction_origin_eligible'), 'fold')
    _fields(membership, ('fold_id', 'role', 'episode_id', 'forecast_at', 'origin_eligible',
                        'label_ready_at', 'label_available'), 'membership')
    for frame, name in [(origins, 'origin'), (predictors, 'predictor')]:
        _ids(frame.episode_id, name)
        _require(not frame.episode_id.duplicated().any(), 'Duplicate ' + name + ' IDs')
        for key in ['bin_start', 'forecast_at']:
            _utc(frame[key], key)
    _bool(origins.origin_eligible, 'origin_eligible')
    _require(origins.forecast_at.eq(origins.bin_start + pd.Timedelta(1, unit='s')).all(),
             'First-alert timing mismatch')
    _require(set(predictors.episode_id) == set(origins.episode_id), 'Predictor-origin ID mismatch')
    o = origins.set_index('episode_id', drop=False).sort_index()
    x = predictors.set_index('episode_id', drop=False).loc[o.index]
    _require(x.bin_start.equals(o.bin_start) and x.forecast_at.equals(o.forecast_at), 'Predictor times disagree')
    _require(x.onset_hour_utc.eq(o.bin_start.dt.hour).all()
             and x.onset_weekday_utc.eq(o.bin_start.dt.weekday).all(), 'Predictor calendar disagrees')
    _require(x.onset_weekday_utc.isin([0, 1, 2, 3, 4, 6]).all(), 'Saturday is outside the inherited calendar')
    _finite(x.loc[o.origin_eligible, list(PREDICTORS)], 'eligible predictors')
    _finite(o.loc[o.origin_eligible, 'initial_pips'], 'eligible initial movement')
    eligible_x = x.loc[o.origin_eligible]
    _require(eligible_x.pre_spread_pips.gt(0).all(), 'Nonpositive eligible prior spread')
    _require(np.allclose(eligible_x.pressure_magnitude, abs(eligible_x.signed_pressure), rtol=1e-12, atol=1e-12)
             and np.allclose(eligible_x[INTERACTION], eligible_x.variation_standardized
                             * eligible_x.pressure_magnitude, rtol=1e-12, atol=1e-12), 'Predictor identities disagree')
    _ids(folds.fold_id, 'fold'); _ids(membership.fold_id, 'membership fold')
    _ids(membership.episode_id, 'membership event')
    _require(not folds.fold_id.duplicated().any() and len(folds), 'Unique nonempty folds required')
    _require(membership.role.isin(['training', 'prediction']).all()
             and set(membership.fold_id).issubset(set(folds.fold_id))
             and set(membership.episode_id).issubset(set(o.index))
             and not membership.duplicated(['fold_id', 'role', 'episode_id']).any(), 'Invalid membership keys')
    for key in ['fit_cutoff', 'prediction_start', 'prediction_end_exclusive']:
        _utc(folds[key], key)
    f = folds.sort_values('fit_cutoff').reset_index(drop=True)
    _require(f.fit_cutoff.eq(f.prediction_start).all()
             and f.prediction_end_exclusive.gt(f.prediction_start).all()
             and (len(f) == 1 or f.prediction_start.iloc[1:].reset_index(drop=True).eq(
                 f.prediction_end_exclusive.iloc[:-1].reset_index(drop=True)).all()), 'Invalid refit intervals')
    return o, x, f


def _design(training, prediction, interaction):
    hours = sorted(int(v) for v in training.onset_hour_utc.unique())
    weekdays = sorted(int(v) for v in training.onset_weekday_utc.unique())
    _require(0 in hours and 0 in weekdays, 'Training reference hour 00 and Monday required')
    terms = ['intercept', *CONTINUOUS]
    train_parts = [np.ones(len(training)), *[training[k].to_numpy(float) for k in CONTINUOUS]]
    pred_parts = [np.ones(len(prediction)), *[prediction[k].to_numpy(float) for k in CONTINUOUS]]
    dropped = []
    for key, levels, support, prefix in [('onset_hour_utc', range(1, 24), hours, 'hour'),
                                         ('onset_weekday_utc', [1, 2, 3, 4, 6], weekdays, 'weekday')]:
        for level in levels:
            term = f'hour_{level:02d}' if prefix == 'hour' else f'weekday_{level}'
            if level not in support:
                dropped.append(term)
                continue
            terms.append(term)
            train_parts.append(training[key].eq(level).to_numpy(float))
            pred_parts.append(prediction[key].eq(level).to_numpy(float))
    if interaction:
        terms.append(INTERACTION)
        train_parts.append(training[INTERACTION].to_numpy(float))
        pred_parts.append(prediction[INTERACTION].to_numpy(float))
    return terms, np.column_stack(train_parts), np.column_stack(pred_parts), hours, weekdays, dropped


def _ols(design, target):
    n, p = design.shape
    _require(n > p, 'OLS requires more training rows than columns')
    maximum = abs(design).max(axis=0)
    _require((maximum > 0).all(), 'Rank-deficient continuous design: zero column')
    scales = maximum * np.sqrt(np.mean((design / maximum) ** 2, axis=0))
    _finite(scales, 'column equilibration')
    _require((scales > 0).all(), 'Column equilibration underflow')
    equilibrated = design / scales
    u, singular, vt = np.linalg.svd(equilibrated, full_matrices=False)
    tolerance = np.finfo(float).eps * max(n, p) * singular[0]
    rank = int((singular > tolerance).sum())
    _require(rank == p, 'Rank-deficient training design; no coefficient dropping or fallback')
    coefficients = (vt.T @ ((u.T @ target) / singular)) / scales
    _finite(coefficients, 'OLS coefficients')
    return coefficients, scales, rank, float(singular[0] / singular[-1]), float(tolerance)


def fit_forecasts(origins, predictors, labels, folds, membership):
    """Fit only selected matured C300 training labels; return unscored forecasts."""
    o, x, f = _prepare(origins, predictors, folds, membership)
    _fields(labels, ('episode_id', 'horizon_sec', 'forecast_at', 'label_ready_at',
                     'label_available', 'continuation_pips'), 'label')
    forecast_parts, coefficient_rows, diagnostic_rows = [], [], []
    forecast_ids = []
    for fold in f.itertuples(index=False):
        member = membership.loc[membership.fold_id.eq(fold.fold_id)]
        train_member = member.loc[member.role.eq('training')].set_index('episode_id')
        prediction_member = member.loc[member.role.eq('prediction')].set_index('episode_id')
        train_ids = train_member.index.sort_values()
        prediction_ids = prediction_member.index.sort_values()
        expected = o.index[o.forecast_at.ge(fold.prediction_start) & o.forecast_at.lt(fold.prediction_end_exclusive)]
        _require(set(prediction_ids) == set(expected) and len(prediction_ids) == fold.prediction_origins,
                 'Prediction membership does not preserve all block origins')
        _require(len(train_ids) == fold.training_origins and len(train_ids) > 0, 'Training membership count mismatch')
        train_o = o.loc[train_ids]
        ready = train_o.forecast_at + pd.Timedelta(300, unit='s')
        _require(train_o.origin_eligible.all() and train_o.forecast_at.lt(fold.fit_cutoff).all()
                 and ready.le(fold.fit_cutoff).all(), 'Training origin or label has not matured at cutoff')
        for member_part, ids in [(train_member, train_ids), (prediction_member, prediction_ids)]:
            part = member_part.loc[ids]
            _require(part.forecast_at.equals(o.loc[ids].forecast_at)
                     and part.origin_eligible.equals(o.loc[ids].origin_eligible), 'Membership origin metadata mismatch')
        _require(train_member.loc[train_ids].label_ready_at.equals(ready)
                 and train_member.loc[train_ids].label_available.eq(True).all(), 'Training membership maturity mismatch')
        selected = labels.loc[labels.horizon_sec.eq(300) & labels.episode_id.isin(train_ids),
                              ['episode_id', 'forecast_at', 'label_ready_at', 'label_available', 'continuation_pips']]
        _require(len(selected) == len(train_ids) and not selected.episode_id.duplicated().any(),
                 'Missing or duplicate selected training labels')
        selected = selected.set_index('episode_id').loc[train_ids]
        _bool(selected.label_available, 'training label availability')
        _require(selected.label_available.all() and selected.forecast_at.equals(train_o.forecast_at)
                 and selected.label_ready_at.equals(ready), 'Selected training labels violate maturity contract')
        y = selected.continuation_pips.to_numpy(float)
        _finite(y, 'selected training labels')
        training_mean = _sum(y) / len(y)
        prediction_o = o.loc[prediction_ids]
        eligible_ids = prediction_o.index[prediction_o.origin_eligible]
        _require(len(eligible_ids) == fold.prediction_origin_eligible, 'Prediction eligibility count mismatch')
        prediction_x = x.loc[eligible_ids]
        estimates, status_flags = {}, {}
        for method in ('additive', 'interaction'):
            terms, design, pred_design, hours, weekdays, dropped = _design(
                x.loc[train_ids], prediction_x, method == 'interaction')
            beta, scales, rank, condition, tolerance = _ols(design, y)
            unseen_hour = ~prediction_x.onset_hour_utc.isin(hours)
            unseen_weekday = ~prediction_x.onset_weekday_utc.isin(weekdays)
            unseen = unseen_hour | unseen_weekday
            estimate = np.full(len(prediction_x), training_mean)
            seen = ~unseen.to_numpy()
            estimate[seen] = pred_design[seen] @ beta
            _finite(estimate, 'issued forecasts')
            estimates[method] = estimate
            status_flags[method] = (unseen_hour, unseen_weekday, unseen)
            coefficient_rows.extend({'fold_id': int(fold.fold_id), 'method': method, 'term': term,
                                     'coefficient': float(b), 'equilibration': float(scale)}
                                    for term, b, scale in zip(terms, beta, scales))
            diagnostic_rows.append({'fold_id': int(fold.fold_id), 'method': method,
                'fit_cutoff': fold.fit_cutoff, 'max_training_label_ready_at': ready.max(),
                'n_train': len(y), 'n_columns': len(terms), 'rank': rank,
                'condition_equilibrated': condition, 'rank_tolerance': tolerance,
                'training_mean': training_mean, 'observed_hours_json': json.dumps(hours),
                'observed_weekdays_json': json.dumps(weekdays), 'dropped_calendar_terms_json': json.dumps(dropped),
                'n_prediction_origin_eligible': len(eligible_ids), 'unseen_calendar_fallbacks': int(unseen.sum())})
        base = prediction_o[['episode_id', 'bin_start', 'forecast_at', 'origin_eligible', 'initial_pips']].copy()
        base.insert(0, 'fold_id', int(fold.fold_id))
        for method in METHODS:
            part = base.copy(); part['method'] = method
            part['forecast_status'] = 'origin_ineligible'
            part['unseen_hour'] = False; part['unseen_weekday'] = False
            part['predicted_C'] = np.nan
            if method in ('zero', 'training_mean'):
                part.loc[eligible_ids, 'predicted_C'] = 0. if method == 'zero' else training_mean
                part.loc[eligible_ids, 'forecast_status'] = 'zero_reference' if method == 'zero' else 'training_mean'
            else:
                uh, uw, unseen = status_flags[method]
                part.loc[eligible_ids, 'predicted_C'] = estimates[method]
                part.loc[eligible_ids, 'forecast_status'] = np.where(unseen, 'unseen_calendar_mean', 'estimated')
                part.loc[eligible_ids, 'unseen_hour'] = uh.to_numpy()
                part.loc[eligible_ids, 'unseen_weekday'] = uw.to_numpy()
            part['predicted_I'] = part.initial_pips + part.predicted_C
            _finite(part.loc[eligible_ids, 'predicted_I'], 'issued total-displacement forecasts')
            forecast_parts.append(part[FORECAST_COLUMNS].reset_index(drop=True))
        forecast_ids.extend(prediction_ids.tolist())
    _require(len(forecast_ids) == len(set(forecast_ids)), 'Origin issued in more than one fold')
    forecasts = pd.concat(forecast_parts, ignore_index=True).sort_values(['fold_id', 'episode_id', 'method']).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    n_eligible = int(f.prediction_origin_eligible.sum())
    statistics = {'folds': len(f), 'prediction_origins': len(forecast_ids), 'eligible_origins': n_eligible,
        'ineligible_origins': len(forecast_ids) - n_eligible, 'forecast_rows': len(forecasts),
        'forecasts_computed': n_eligible * len(METHODS), 'ols_fits': len(diagnostics),
        'training_mean_calibrations': len(f), 'empirical_models_fitted': len(diagnostics) + len(f),
        'methods': list(METHODS), 'primary_horizon_sec': 300,
        'ols_unseen_calendar_fallbacks': {m: int(diagnostics.loc[diagnostics.method.eq(m), 'unseen_calendar_fallbacks'].sum())
                                          for m in ('additive', 'interaction')},
        'maximum_equilibrated_condition': float(diagnostics.condition_equilibrated.max()),
        'future_evaluation_labels_attached': False, 'forecast_performance_computed': False,
        'inference_computed': False, 'untouched_holdout': False}
    return {'forecasts': forecasts, 'coefficients': coefficients, 'fit_diagnostics': diagnostics,
            'statistics': _plain(statistics)}


def _aggregate(base, groups, group_column, *, means):
    """Aggregate event sums with fsum; empty calendar dates have zero sums, no loss observations."""
    rows = []
    for key in groups:
        part = base.loc[base[group_column].eq(key)]
        good = part.loc[part.scored]
        row = {group_column: key, 'n_origins': len(part), 'n_issued': int(part.origin_eligible.sum()), 'n_scored': len(good)}
        for method in METHODS:
            total = _sum(good['loss_' + method])
            row['sse_' + method] = total
            if means:
                row['mse_' + method] = total / len(good) if len(good) else np.nan
        for contrast in CONTRASTS:
            total = _sum(good['delta_' + contrast])
            row['sum_delta_' + contrast] = total
            if means:
                row['mean_delta_' + contrast] = total / len(good) if len(good) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_forecasts(forecasts, labels, *, start, end):
    """Attach labels only after forecast serialization; descriptive squared loss, no inference."""
    _fields(forecasts, FORECAST_COLUMNS, 'forecast', exact=True)
    _fields(labels, ('episode_id', 'horizon_sec', 'forecast_at', 'label_ready_at',
                     'label_available', 'label_status', 'continuation_pips'), 'label')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _require(str(start.tz) == 'UTC' and str(end.tz) == 'UTC' and start < end
             and start == start.normalize() and end == end.normalize(), 'UTC whole-day evaluation interval required')
    _require(len(forecasts) and not forecasts.duplicated(['episode_id', 'method']).any()
             and set(forecasts.method) == set(METHODS), 'Invalid forecast method keys')
    _utc(forecasts.forecast_at, 'forecast_at'); _bool(forecasts.origin_eligible, 'origin_eligible')
    _require(forecasts.forecast_at.ge(start).all() and forecasts.forecast_at.lt(end).all(), 'Forecast outside evaluation interval')
    _require(forecasts.groupby('episode_id').size().eq(len(METHODS)).all(), 'Incomplete method coverage')
    wide = forecasts.pivot(index='episode_id', columns='method', values='predicted_C')
    base = forecasts.loc[forecasts.method.eq('zero'), ['episode_id', 'fold_id', 'forecast_at', 'origin_eligible', 'initial_pips']].set_index('episode_id').sort_index()
    for method in METHODS:
        part = forecasts.loc[forecasts.method.eq(method)].set_index('episode_id').loc[base.index]
        for key in ['fold_id', 'forecast_at', 'origin_eligible', 'initial_pips']:
            _require(part[key].equals(base[key]), 'Method origin metadata mismatch: ' + key)
        _require(part.loc[~part.origin_eligible, ['predicted_C', 'predicted_I']].isna().all().all(), 'Ineligible forecast must be missing')
        _finite(part.loc[part.origin_eligible, ['predicted_C', 'predicted_I']], 'eligible forecast')
        _require(np.array_equal(part.loc[part.origin_eligible, 'predicted_I'].to_numpy(),
                               (part.initial_pips + part.predicted_C).loc[part.origin_eligible].to_numpy()), 'Forecast J/C/I identity mismatch')
    selected = labels.loc[labels.horizon_sec.eq(300) & labels.episode_id.isin(base.index),
        ['episode_id', 'forecast_at', 'label_ready_at', 'label_available', 'label_status', 'continuation_pips']]
    _require(not selected.episode_id.duplicated().any(), 'Duplicate evaluation labels')
    _bool(selected.label_available, 'evaluation label availability')
    selected = selected.set_index('episode_id')
    present = base.index.intersection(selected.index)
    _require(selected.loc[present].forecast_at.equals(base.loc[present].forecast_at)
             and selected.loc[present].label_ready_at.equals(base.loc[present].forecast_at + pd.Timedelta(300, unit='s')),
             'Evaluation label timing mismatch')
    _require((~selected.label_available | selected.label_ready_at.le(end)).all(), 'Available label beyond sample end')
    target_values = selected.continuation_pips.to_numpy(float)
    _require(not np.isinf(target_values).any()
             and np.array_equal(selected.label_available.to_numpy(), np.isfinite(target_values)),
             'Evaluation availability must equal finite target availability')
    _finite(selected.loc[selected.label_available, 'continuation_pips'], 'available evaluation target')
    base['label_available'] = selected.label_available.reindex(base.index, fill_value=False).astype(bool)
    base['label_status'] = selected.label_status.reindex(base.index).fillna('missing_record')
    base['observed_C'] = selected.continuation_pips.reindex(base.index).where(base.label_available)
    base['scored'] = base.origin_eligible & base.label_available
    _require(base.scored.any(), 'No issued forecasts with available labels')
    base['date'] = base.forecast_at.dt.normalize()
    base['quarter'] = base.forecast_at.dt.tz_localize(None).dt.to_period('Q').astype(str)
    scored_parts, metric_rows = [], []
    for method in METHODS:
        pred = wide[method].reindex(base.index)
        error = (pred - base.observed_C).where(base.scored)
        loss = error ** 2
        _finite(loss.loc[base.scored], 'squared loss')
        base['loss_' + method] = loss
        part = forecasts.loc[forecasts.method.eq(method)].copy()
        for key in ['label_available', 'label_status', 'observed_C', 'scored']:
            part[key] = part.episode_id.map(base[key])
        part['forecast_error'] = part.episode_id.map(error)
        part['squared_error'] = part.episode_id.map(loss)
        scored_parts.append(part)
        n = int(base.scored.sum()); sse = _sum(loss.loc[base.scored]); mse = sse / n
        metric_rows.append({'method': method, 'n_origins': len(base), 'n_issued': int(base.origin_eligible.sum()),
            'n_scored': n, 'n_issued_unavailable': int((base.origin_eligible & ~base.label_available).sum()),
            'sse': sse, 'mse': mse, 'rmse': math.sqrt(mse)})
    comparison_rows = []
    for contrast, benchmark in CONTRASTS.items():
        full = wide.interaction.reindex(base.index); other = wide[benchmark].reindex(base.index)
        delta = ((other - full) * (2 * base.observed_C - full - other)).where(base.scored)
        _finite(delta.loc[base.scored], 'stable loss difference')
        base['delta_' + contrast] = delta
        total = _sum(delta.loc[base.scored]); n = int(base.scored.sum())
        full_mse = next(row['mse'] for row in metric_rows if row['method'] == 'interaction')
        benchmark_mse = next(row['mse'] for row in metric_rows if row['method'] == benchmark)
        comparison_rows.append({'contrast': contrast, 'benchmark': benchmark, 'n': n,
            'sum_loss_difference': total, 'mean_loss_difference': total / n,
            'relative_mse_reduction': 1 - full_mse / benchmark_mse if benchmark_mse > 0 else np.nan,
            'relative_mse_status': 'available' if benchmark_mse > 0 else 'zero_benchmark_mse'})
    calendar = pd.date_range(start, end, freq='D', inclusive='left')
    daily = _aggregate(base, calendar, 'date', means=False)
    fold_losses = _aggregate(base, sorted(base.fold_id.unique()), 'fold_id', means=True)
    quarters = sorted(pd.Series(calendar.tz_localize(None).to_period('Q').astype(str)).unique())
    quarter_losses = _aggregate(base, quarters, 'quarter', means=True)
    serial_rows, concentration = [], {}
    comparisons = pd.DataFrame(comparison_rows)
    for contrast in CONTRASTS:
        mean = float(comparisons.loc[comparisons.contrast.eq(contrast), 'mean_loss_difference'].iloc[0])
        difference = daily['sum_delta_' + contrast].to_numpy(float)
        psi = difference - daily.n_scored.to_numpy() * mean
        daily['psi_' + contrast] = psi
        energy = _sum(psi * psi); absolute = _sum(abs(difference))
        _finite([energy, absolute], 'loss-difference concentration')
        for lag in range(41):
            pairs = max(0, len(psi) - lag)
            product = _sum(psi[lag:] * psi[:len(psi)-lag]) if pairs else np.nan
            serial_rows.append({'contrast': contrast, 'lag': lag, 'n_pairs': pairs,
                'cross_product': product, 'normalized_autocorrelation': product / energy if pairs and energy > 0 else np.nan,
                'status': 'no_pairs' if not pairs else 'zero_energy' if energy == 0 else 'available'})
        concentration[contrast] = {
            'sum_absolute_daily_loss_difference': absolute,
            'maximum_absolute_daily_loss_difference_share': float(max(abs(difference)) / absolute) if absolute else None,
            'maximum_absolute_difference_date': daily.date.iloc[int(np.argmax(abs(difference)))].isoformat() if absolute else None,
            'sum_squared_daily_centered_difference': energy, 'zero_energy': energy == 0,
            'maximum_daily_energy_share': float(max(psi * psi) / energy) if energy else None,
            'maximum_energy_date': daily.date.iloc[int(np.argmax(psi * psi))].isoformat() if energy else None,
            'daily_centering_residual': _sum(psi)}
    scored_events = pd.concat(scored_parts, ignore_index=True).sort_values(['fold_id', 'episode_id', 'method']).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows)
    statistics = {'status': 'descriptive_forecast_evaluation', 'primary_horizon_sec': 300,
        'calendar_days': len(calendar), 'days_with_scored_origins': int(daily.n_scored.gt(0).sum()),
        'prediction_origins': len(base), 'issued_origins': int(base.origin_eligible.sum()),
        'scored_origins': int(base.scored.sum()),
        'issued_unavailable': int((base.origin_eligible & ~base.label_available).sum()),
        'metrics': metrics.to_dict('records'), 'comparisons': comparisons.to_dict('records'),
        'comparison_diagnostics': concentration, 'serial_max_lag': 40,
        'loss': 'raw squared C300 error in pips squared', 'empirical_models_fitted': 0,
        'inference_computed': False, 'uncertainty_computed': False, 'untouched_holdout': False,
        'causal_identification_established': False, 'novelty_established': False}
    return {'scored_events': scored_events, 'metrics': metrics, 'comparisons': comparisons,
            'fold_losses': fold_losses, 'daily_losses': daily, 'quarter_losses': quarter_losses,
            'serial_dependence': pd.DataFrame(serial_rows), 'statistics': _plain(statistics)}

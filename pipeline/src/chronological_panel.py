"""General origin-vintage carry and unchanged detector inputs for panel cycles."""
import hashlib
import numpy as np
import pandas as pd
from src.event_audit import (FEATURES, METRICS, classify_bin_eligibility,
    fit_detector, score_detector, ns)
from src.panel_state_precision import quote_diagnostics
from repair_event_audit import add_adjusted, REASONS

ISSUED_COLUMNS = ['origin_quote_id', 'component_id', 'origin_at', 'scale_sec', 'psi',
                  'residual_parameter_cutoff']


def integer_fingerprint(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        v = np.asarray(value)
        if not np.isfinite(v).all() or not np.equal(v, v.astype('int64')).all():
            raise ValueError('Exact integer fingerprint input required')
        a = np.ascontiguousarray(v, dtype='<i8')
        h.update(np.asarray(a.shape, dtype='<i8').tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def issued_from_predictions(path, prediction, *, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if len(path) != len(prediction) or not np.array_equal(path.origin_quote_id, prediction.origin_quote_id):
        raise ValueError('Prediction identities differ')
    take = path.origin_at.ge(cutoff)
    f = path.loc[take, ['origin_quote_id', 'component_id', 'origin_at', 'scale_sec']].reset_index(drop=True)
    f['psi'] = prediction.psi.to_numpy()[take.to_numpy()]
    f['residual_parameter_cutoff'] = cutoff
    return f[ISSUED_COLUMNS]


def _check_issued(f):
    if list(f.columns) != ISSUED_COLUMNS or f.origin_quote_id.duplicated().any():
        raise ValueError('Unique issued-forecast schema required')
    if not np.isfinite(f[['scale_sec', 'psi']]).all(axis=None) or (f[['scale_sec', 'psi']] <= 0).any(axis=None):
        raise ValueError('Issued scales and forecasts must be finite and positive')
    if len(f):
        ns(f.origin_at); ns(f.residual_parameter_cutoff)
        if not f.residual_parameter_cutoff.le(f.origin_at).all():
            raise ValueError('Parameter vintage follows forecast origin')


def deployment_durations(new_issued, new_origins, carry_in, required_open, *, cutoff, stop):
    """Carry issued means independently of the current training/refilter window.

    new_origins and required_open retain full recorded endpoint information for
    bookkeeping. Only observed completions before stop enter residual bins.
    """
    cutoff, stop = pd.Timestamp(cutoff), pd.Timestamp(stop)
    if cutoff.tz is None or stop.tz is None or stop <= cutoff:
        raise ValueError('Ordered timezone-aware deployment interval required')
    for issued in [new_issued, carry_in]: _check_issued(issued)
    if not carry_in.origin_at.lt(cutoff).all() or not new_issued.origin_at.ge(cutoff).all() or not new_issued.origin_at.lt(stop).all():
        raise ValueError('Forecast origin outside assigned vintage interval')
    if not new_issued.residual_parameter_cutoff.eq(cutoff).all():
        raise ValueError('New origins need current-cycle parameters')
    for raw in [new_origins, required_open]:
        if raw.origin_quote_id.duplicated().any(): raise ValueError('Duplicate source origin')
        if len(raw) and ((ns(raw.end_at) <= ns(raw.origin_at)).any() or not np.isfinite(raw.scale_sec).all()):
            raise ValueError('Invalid eligible source spell')
    if not required_open.origin_at.lt(cutoff).all() or not required_open.end_at.ge(cutoff).all():
        raise ValueError('Open-spell source does not cross cutoff')
    if set(carry_in.origin_quote_id) != set(required_open.origin_quote_id):
        raise ValueError('Missing or extraneous previously issued open forecast')
    if set(new_issued.origin_quote_id) != set(new_origins.origin_quote_id):
        raise ValueError('New issued forecasts do not cover eligible origins')
    issued = pd.concat([carry_in, new_issued], ignore_index=True).sort_values('origin_quote_id').reset_index(drop=True)
    _check_issued(issued)
    raw = pd.concat([required_open, new_origins], ignore_index=True).set_index('origin_quote_id')
    raw = raw.reindex(issued.origin_quote_id).reset_index()
    for name in ['origin_quote_id', 'component_id', 'origin_at', 'scale_sec']:
        if not np.array_equal(raw[name].to_numpy(), issued[name].to_numpy()):
            raise ValueError('Issued/source identity or scale mismatch: '+name)
    completed = raw.event_observed & raw.end_at.ge(cutoff) & raw.end_at.lt(stop)
    ratio = (ns(raw.end_at[completed])-ns(raw.origin_at[completed]))/1e9/issued.loc[completed, 'scale_sec'].to_numpy()/issued.loc[completed, 'psi'].to_numpy()
    if not np.isfinite(ratio).all() or (ratio <= 0).any(): raise ValueError('Invalid completed residual')
    bins = pd.DataFrame({'bin_start': raw.loc[completed, 'end_at'].dt.floor('s').to_numpy(), 'residual': ratio}).groupby(
        'bin_start', as_index=False, sort=True).agg(completed_residual_count=('residual', 'size'), mean_duration_residual=('residual', 'mean'))
    carry_out = issued.loc[raw.end_at.ge(stop)].reset_index(drop=True)
    counts = {'issued_new': len(new_issued), 'carried_in': len(carry_in), 'carried_out': len(carry_out),
        'completed_residuals': int(completed.sum()),
        'completed_carried_residuals': int((completed & raw.origin_at.lt(cutoff)).sum()),
        'carried_origin_before_training_start': int(carry_in.origin_at.lt(cutoff-pd.Timedelta(days=56)).sum()),
        'oldest_carry_vintage': carry_in.residual_parameter_cutoff.min().isoformat() if len(carry_in) else None}
    return bins, carry_out, counts


def feature_inputs(clock, factors, duration, *, cutoff, policy='structural_only'):
    f = add_adjusted(clock, factors, policy)
    d = duration.set_index('bin_start').reindex(f.bin_start)
    n = d.completed_residual_count.fillna(0).to_numpy(dtype='int64')
    recorded = clock.recorded_update_count.to_numpy()
    ready = np.isfinite(f[['adjusted_abs_quote_ofi', 'adjusted_tick_count']]).all(axis=1).to_numpy()
    codes = classify_bin_eligibility(f.flow_status.to_numpy(), ready, recorded, n)
    o = pd.DataFrame({'bin_start': f.bin_start, 'available_at': f.available_at,
        'period': np.where(f.bin_start < cutoff, 'calibration', 'development'),
        'recorded_update_count': recorded, 'support_id': f.support_id, 'flow_status': f.flow_status,
        'seasonal_ready': ready, 'completed_residual_count': n,
        'eligible_without_duration': np.isin(f.flow_status, [1, 2]) & ready,
        'reason': pd.Categorical(np.asarray(REASONS)[codes], categories=REASONS)})
    keep = codes == 5
    s = f.loc[keep, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                    'adjusted_abs_quote_ofi', 'adjusted_tick_count']].reset_index(drop=True)
    s['recorded_update_count'] = recorded[keep]; s['completed_residual_count'] = n[keep]
    s['mean_duration_residual'] = d.mean_duration_residual.to_numpy()[keep]
    s['negative_log_mean_residual'] = -np.log(s.mean_duration_residual)
    s['period'] = o.period.to_numpy()[keep]
    return f, o, s


def score_cycle(clock, factors, duration, quotes, support, *, cutoff):
    f, o, s = feature_inputs(clock, factors, duration, cutoff=cutoff)
    training = s.period.eq('calibration')
    fit = fit_detector(s.loc[training, FEATURES].to_numpy())
    calibration = partition_statistics(o.loc[o.period.eq('calibration')],
        s.loc[training].assign(candidate_event=lambda x: score_detector(x[FEATURES].to_numpy(), fit) >= fit['threshold']))
    calibration['eligible_bin_identity_sha256'] = integer_fingerprint(ns(s.loc[training, 'bin_start']))
    selected = s.loc[~training].reset_index(drop=True)
    # The first deployed state's full 300-second clock history is explicitly included.
    begin = cutoff-pd.Timedelta(seconds=300)
    history = f.loc[f.bin_start.ge(begin)].reset_index(drop=True)
    economic = quote_diagnostics(history, quotes, support, 'structural_only').set_index('bin_start')
    for name in METRICS: selected[name] = economic[name].reindex(selected.bin_start).to_numpy()
    selected['distance'] = score_detector(selected[FEATURES].to_numpy(), fit)
    selected['candidate_event'] = selected.distance.ge(fit['threshold'])
    z = (selected[FEATURES].to_numpy()-np.asarray(fit['mean']))/np.asarray(fit['sample_std'])
    for i, name in enumerate(['z_ofi', 'z_intensity', 'z_duration']): selected[name] = z[:, i]
    return o.loc[o.period.eq('development')].reset_index(drop=True), selected, fit, calibration


def partition_statistics(o, s):
    return {'calendar_seconds': len(o), 'eligible_bins': len(s),
        'candidate_bins': int(s.candidate_event.sum()),
        'candidate_share_of_eligible': float(s.candidate_event.mean()) if len(s) else None,
        'reason_counts': {r: int(o.reason.eq(r).sum()) for r in REASONS}}

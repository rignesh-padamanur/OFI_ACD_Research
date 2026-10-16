"""Calendar and forecast handoff for the first verified rolling extension."""
import numpy as np
import pandas as pd
from src.event_audit import ns


def _utc_midnight(value):
    t = pd.Timestamp(value)
    if t.tz is None or pd.isna(t) or t.utcoffset().total_seconds() != 0 or t != t.normalize():
        raise ValueError('UTC midnight required')
    return t.tz_convert('UTC')


def cycle_schedule(first_cutoff, stop):
    start, stop = _utc_midnight(first_cutoff), _utc_midnight(stop)
    if stop <= start:
        raise ValueError('Nonempty deployment interval required')
    rows = []
    cutoff = start
    while cutoff < stop:
        end = min(cutoff + pd.Timedelta(days=14), stop)
        rows.append({'cycle_id': len(rows), 'training_start': cutoff-pd.Timedelta(days=56),
                     'parameter_cutoff': cutoff, 'deployment_end': end})
        cutoff = end
    return pd.DataFrame(rows)


def carry_forecasts(path, new_predictions, previous_predictions, *, cutoff, previous_cutoff):
    cutoff, previous_cutoff = _utc_midnight(cutoff), _utc_midnight(previous_cutoff)
    if cutoff-previous_cutoff != pd.Timedelta(days=14):
        raise ValueError('First transition requires adjacent 14-day deployments')
    if (path.origin_quote_id.duplicated().any() or new_predictions.origin_quote_id.duplicated().any()
            or previous_predictions.origin_quote_id.duplicated().any()
            or not np.array_equal(path.origin_quote_id, new_predictions.origin_quote_id)):
        raise ValueError('Unique aligned forecast identities required')
    for f in [new_predictions, previous_predictions]:
        if not np.isfinite(f.psi).all() or (f.psi <= 0).any():
            raise ValueError('Positive finite forecasts required')
    a, b = ns(path.origin_at), ns(path.end_at)
    if (b <= a).any() or (np.diff(a) <= 0).any():
        raise ValueError('Ordered positive spell domain required')
    carry = (a < cutoff.value) & (b >= cutoff.value)
    if (a[carry] < previous_cutoff.value).any():
        raise ValueError('Carry predates the verified previous deployment')
    used = new_predictions.psi.to_numpy(dtype=float).copy()
    old = previous_predictions.set_index('origin_quote_id').psi
    values = old.reindex(path.loc[carry, 'origin_quote_id']).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('Missing previously issued forecast')
    used[carry] = values
    return pd.DataFrame({'origin_quote_id': path.origin_quote_id.to_numpy(),
        'psi_refiltered': new_predictions.psi.to_numpy(), 'psi_for_residual': used,
        'residual_parameter_cutoff': pd.to_datetime(np.where(carry, previous_cutoff.value, cutoff.value), utc=True),
        'forecast_role': np.where(carry, 'carried_origin_forecast',
                          np.where(a >= cutoff.value, 'new_origin_forecast', 'calibration_refilter'))})


def join_deployments(old_opportunities, old_scored, new_opportunities, new_scored, *, cutoff):
    cutoff = _utc_midnight(cutoff)
    for o, s, before in [(old_opportunities, old_scored, True), (new_opportunities, new_scored, False)]:
        if not len(o) or not o.period.eq('development').all() or not s.period.eq('development').all():
            raise ValueError('Nonempty deployment clocks only')
        t = ns(o.bin_start)
        if ((np.diff(t) != 1_000_000_000).any() or (t % 1_000_000_000 != 0).any()
                or not np.array_equal(ns(o.available_at), t+1_000_000_000)):
            raise ValueError('Complete ordered one-second clock required')
        expected = o.loc[o.reason.eq('eligible')]
        for col in ['bin_start', 'available_at', 'support_id']:
            if not np.array_equal(expected[col], s[col]):
                raise ValueError('Scored rows must exactly match eligible clock rows')
        if before and not o.bin_start.lt(cutoff).all():
            raise ValueError('Old deployment overlaps the new cutoff')
        if not before and not o.bin_start.ge(cutoff).all():
            raise ValueError('New deployment precedes its cutoff')
    if old_opportunities.available_at.iloc[-1] != cutoff or new_opportunities.bin_start.iloc[0] != cutoff:
        raise ValueError('Gap or overlap at deployment boundary')
    return (pd.concat([old_opportunities, new_opportunities], ignore_index=True),
            pd.concat([old_scored, new_scored], ignore_index=True))

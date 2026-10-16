"""Fixed representation comparison; no detector replay or parameter search."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.event_audit import ns
from src.detector_contribution import XCOLS, hash_training_sample

SECOND = 1_000_000_000
BASE = XCOLS + ['quote_ofi', 'adjusted_abs_quote_ofi', 'adjusted_tick_count',
               'pre_quote_age_sec', 'raw_speed', 'seasonal_speed']
EXTRA = 'negative_log_mean_residual'
MODEL = dict(loss='squared_error', learning_rate=.05, max_iter=100,
    max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1,
    max_bins=63, early_stopping=False, random_state=20261004)
START = pd.Timestamp('2024-07-01', tz='UTC')
STOP = pd.Timestamp('2026-08-01', tz='UTC')
STATS = ['n', 'sum_y', 'sum_b', 'sum_t', 'sum_lb', 'sum_lt', 'sum_d', 'sum_carry']


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def seasonal_bins(raw):
    require(raw.event_observed.all(), 'Only completed incoming gaps allowed')
    require(raw.seasonal_cutoff.le(raw.origin_at).all(), 'Future seasonal cutoff')
    gap = (ns(raw.end_at)-ns(raw.origin_at))/SECOND
    require(np.array_equal(ns(raw.end_at)-ns(raw.origin_at), raw.exposure_ns), 'Raw exposure identity')
    valid = np.isfinite(raw.scale_sec.to_numpy()) & (raw.scale_sec.to_numpy() > 0)
    require((gap > 0).all(), 'Nonpositive completed gap')
    f = pd.DataFrame({'bin_start': raw.end_at.dt.floor('s').to_numpy(),
        'gap': gap, 'seasonal_gap': np.where(valid, gap/raw.scale_sec.to_numpy(), np.nan),
        'scale_ready': valid})
    g = f.groupby('bin_start', sort=True).agg(n=('gap', 'size'), total=('gap', 'sum'),
        n_scale=('scale_ready', 'sum'), seasonal_mean=('seasonal_gap', 'mean'))
    return g


def predictor_frame(scored, ledger, bins):
    require(np.array_equal(ns(scored.bin_start), ns(ledger.bin_start)), 'Scored/ledger identities')
    require(np.array_equal(scored.support_id, ledger.support_id), 'Support identities')
    require(np.array_equal(scored.recorded_update_count, scored.completed_residual_count), 'Incomplete scored durations')
    require(np.array_equal(ns(scored.available_at), ns(scored.bin_start)+SECOND), 'Feature availability')
    b = bins.reindex(scored.bin_start)
    require(b.notna().all(axis=None), 'Missing seasonal bins')
    n = scored.recorded_update_count.to_numpy()
    require(np.array_equal(b.n, n) and np.array_equal(b.n_scale, n), 'Seasonal gap coverage differs')
    a = 1+scored.pre_quote_age_sec.to_numpy()-scored.end_quote_age_sec.to_numpy()
    require(np.isfinite(a).all() and (a > 0).all(), 'Invalid raw aggregate')
    require(np.allclose(b.total, a, rtol=1e-10, atol=1e-10), 'Incoming exposure differs')
    require(np.allclose(ledger.quote_age, scored.end_quote_age_sec, rtol=0, atol=1e-12), 'End age differs')
    require(np.array_equal(ledger[EXTRA], scored[EXTRA]), 'Archived normalized feature changed')
    f = ledger[['bin_start', 'support_id', 'recorded_update_count']+XCOLS+
        ['quote_ofi', 'adjusted_abs_quote_ofi', 'adjusted_tick_count', EXTRA]].copy()
    f['available_at'] = scored.available_at.to_numpy()
    f['pre_quote_age_sec'] = scored.pre_quote_age_sec.to_numpy()
    f['mean_raw_gap'] = a/n
    f['mean_seasonal_gap'] = b.seasonal_mean.to_numpy()
    require(np.isfinite(f.mean_seasonal_gap).all() and f.mean_seasonal_gap.gt(0).all(), 'Invalid seasonal means')
    f['raw_speed'] = -np.log(f.mean_raw_gap)
    f['seasonal_speed'] = -np.log(f.mean_seasonal_gap)
    f['forecast_ready'] = np.isfinite(f[BASE+[EXTRA]]).all(axis=1)
    return f


def select_training(frame, cutoff):
    cutoff = pd.Timestamp(cutoff)
    t = ns(frame.bin_start)
    return ((t >= (cutoff-pd.Timedelta(days=56)).value) &
        (t+301*SECOND <= cutoff.value) & frame.forecast_ready.to_numpy() &
        frame.outcome_ready.to_numpy() & np.isfinite(frame.y300.to_numpy()) &
        hash_training_sample(t))


def quartile_groups(values, thresholds):
    return np.searchsorted(np.asarray(thresholds), np.asarray(values), side='right')+1


def aggregate(frame, by):
    result = frame.groupby(by, sort=True).agg(n=('d', 'size'), sum_y=('y300', 'sum'),
        sum_b=('prediction_b', 'sum'), sum_t=('prediction_t', 'sum'),
        sum_lb=('loss_b', 'sum'), sum_lt=('loss_t', 'sum'), sum_d=('d', 'sum'),
        sum_carry=('loss_carry', 'sum')).reset_index()
    return result


def daily_hac(influence, lag):
    x = np.asarray(influence, float)
    require(np.isfinite(x).all(), 'Nonfinite influence')
    v = float(x@x)
    for j in range(1, min(lag, len(x)-1)+1):
        v += 2*(1-j/(lag+1))*float(x[j:]@x[:-j])
    require(v >= -1e-16, 'Negative HAC variance')
    return float(np.sqrt(max(v, 0)))


def estimates(daily):
    n = float(daily.n.sum())
    require(n > 0, 'No evaluable forecasts')
    sb, st, sd = map(float, [daily.sum_lb.sum(), daily.sum_lt.sum(), daily.sum_d.sum()])
    require(sb > 0, 'Degenerate benchmark loss')
    delta, relative = sd/n, sd/sb
    influence = (daily.sum_d.to_numpy()-delta*daily.n.to_numpy())/n
    ri = (daily.sum_d.to_numpy()-relative*daily.sum_lb.to_numpy())/sb
    rows = []
    for lag in [56, 28, 112]:
        se, rse = daily_hac(influence, lag), daily_hac(ri, lag)
        z = float(norm.ppf(.975))
        rows.append({'lag_days': lag, 'delta_pips_squared': delta, 'nominal_se': se,
            'nominal_low': delta-z*se, 'nominal_high': delta+z*se,
            'relative_mse_reduction': relative, 'relative_nominal_se': rse,
            'relative_nominal_low': relative-z*rse, 'relative_nominal_high': relative+z*rse})
    leave_out = (sd-daily.sum_d.to_numpy())/(n-daily.n.to_numpy())
    return {'n': int(n), 'mse_b': sb/n, 'mse_t': st/n,
        'rmse_b': float(np.sqrt(sb/n)), 'rmse_t': float(np.sqrt(st/n)),
        'mse_current_spread_carry': float(daily.sum_carry.sum()/n),
        'delta_pips_squared': delta, 'relative_mse_reduction': relative,
        'leave_one_day_out_delta_min': float(leave_out.min()),
        'leave_one_day_out_delta_max': float(leave_out.max()),
        'nominal_intervals': rows}

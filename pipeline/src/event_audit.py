"""Original Mahalanobis detector: development diagnostics, not shock identification."""
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
FEATURES = ['adjusted_abs_quote_ofi', 'adjusted_tick_count', 'negative_log_mean_residual']
METRICS = ['pre_spread_pips', 'end_spread_pips', 'spread_change_pips',
           'pre_displayed_size', 'end_displayed_size', 'log_size_ratio',
           'pre_quote_age_sec', 'end_quote_age_sec', 'pre_rv_300s_5s_bps',
           'pre_quote_ofi_300s']


def classify_bin_eligibility(status, ready, recorded, completed):
    status, ready, recorded, completed = [np.asarray(x) for x in (status, ready, recorded, completed)]
    if not (status.shape == ready.shape == recorded.shape == completed.shape):
        raise ValueError('Eligibility inputs have inconsistent shapes')
    if not np.isin(status, [0, 1, 2, 3, 4]).all() or (recorded < 0).any() or (completed < 0).any() or (completed > recorded).any():
        raise ValueError('Invalid flow status or residual/quote counts')
    return np.select([status == 0, ~np.isin(status, [1, 2]), ~ready.astype(bool),
                      recorded == 0, completed != recorded], np.arange(5), default=5).astype('uint8')


def ns(values):
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError('Timezone-aware nonmissing timestamps required')
    return index.as_unit('ns').asi8


def score_detector(values, fit):
    x = np.asarray(values, dtype=float)
    mean, sd, cov = (np.asarray(fit[k]) for k in ['mean', 'sample_std', 'covariance_z'])
    if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix with fitted dimensions required')
    z = (x - mean) / sd
    white = np.linalg.solve(np.linalg.cholesky(cov), z.T)
    return np.sqrt(np.sum(white * white, axis=0))


def fit_detector(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) <= x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Insufficient or nonfinite calibration features')
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if (sd <= 0).any():
        raise ValueError('Constant calibration feature')
    z = (x - mean) / sd
    cov = np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= 0 or eig[-1] / eig[0] > 1e10:
        raise ValueError('Singular or numerically unstable calibration covariance')
    fit = {'mean': mean.tolist(), 'sample_std': sd.tolist(), 'covariance_z': cov.tolist(),
           'condition_number': float(eig[-1] / eig[0]), 'quantile': .99,
           'quantile_method': 'linear', 'calibration_rows': len(x)}
    d = score_detector(x, fit)
    fit['threshold'] = float(np.quantile(d, .99, method='linear'))
    fit['training_threshold_ties'] = int((d == fit['threshold']).sum())
    return fit


def completed_duration_bins(path, predictions):
    if len(path) != len(predictions) or not np.array_equal(path.origin_quote_id, predictions.origin_quote_id):
        raise ValueError('Duration predictions not aligned with origin identities')
    keep = path.event_observed.to_numpy(dtype=bool)
    residual = path.normalized_exposure.to_numpy()[keep] / predictions.psi.to_numpy()[keep]
    if not np.isfinite(residual).all() or (residual <= 0).any():
        raise ValueError('Completed residuals must be positive and finite')
    return pd.DataFrame({'bin_start': path.end_at[keep].dt.floor('s'), 'residual': residual}).groupby(
        'bin_start', as_index=False, sort=True).agg(
            completed_residual_count=('residual', 'size'), mean_duration_residual=('residual', 'mean'))


def quote_diagnostics(clock, quotes, support_intervals, policy):
    """As-of quote snapshots and strictly preceding states; no future returns."""
    t, q = ns(clock.bin_start), ns(quotes.timestamp)
    if not len(q) or (np.diff(q) <= 0).any() or (np.diff(t) != SECOND).any():
        raise ValueError('Unique sorted quotes and a complete one-second clock required')
    sid = clock.support_id.to_numpy(dtype=int)
    a = ns(support_intervals.start)
    b = ns(support_intervals.end_exclusive)
    if (sid >= len(a)).any() or (sid < -1).any():
        raise ValueError('Support index outside interval table')
    j = np.maximum(sid, 0)
    supported = (sid >= 0) & (a[j] <= t) & (b[j] >= t + SECOND)
    if not np.array_equal(supported, sid >= 0):
        raise ValueError('Clock support disagrees with intervals')
    pre = np.searchsorted(q, t, side='left') - 1
    end = np.searchsorted(q, t + SECOND, side='left') - 1
    before, after = np.maximum(pre, 0), np.maximum(end, 0)
    vp = supported & (pre >= 0) & (q[before] >= a[j])
    ve = supported & (end >= 0) & (q[after] >= a[j])
    spread = (quotes.ask.to_numpy() - quotes.bid.to_numpy()) * 10000
    size = quotes.ask_volume.to_numpy() + quotes.bid_volume.to_numpy()
    mid = (quotes.ask.to_numpy() + quotes.bid.to_numpy()) / 2
    out = pd.DataFrame({'bin_start': clock.bin_start})
    for prefix, ix, valid, cutoff in [('pre', before, vp, t), ('end', after, ve, t + SECOND)]:
        out[prefix + '_spread_pips'] = np.where(valid, spread[ix], np.nan)
        out[prefix + '_displayed_size'] = np.where(valid, size[ix], np.nan)
        out[prefix + '_quote_age_sec'] = np.where(valid, (cutoff - q[ix]) / SECOND, np.nan)
    out['spread_change_pips'] = out.end_spread_pips - out.pre_spread_pips
    ratio = np.full(len(t), np.nan)
    positive = vp & ve & (size[before] > 0) & (size[after] > 0)
    ratio[positive] = np.log(size[after[positive]] / size[before[positive]])
    out['log_size_ratio'] = ratio
    log_mid = pd.Series(np.where(vp, np.log(mid[before]), np.nan))
    r2 = log_mid.diff(5).pow(2).to_numpy()
    rv = np.full(len(t), np.nan)
    for phase in range(5):
        rv[phase::5] = pd.Series(r2[phase::5]).rolling(60, min_periods=60).sum().to_numpy()
    flow = pd.Series(clock.quote_ofi.to_numpy())
    preflow = flow.shift(1).rolling(300, min_periods=300).sum().to_numpy()
    same = np.zeros(len(t), dtype=bool)
    same[300:] = (sid[300:] >= 0) & (sid[300:] == sid[:-300])
    ready = same & np.isfinite(preflow) & np.isfinite(rv)
    out['pre_rv_300s_5s_bps'] = np.where(ready, np.sqrt(np.maximum(rv, 0)) * 10000, np.nan)
    out['pre_quote_ofi_300s'] = np.where(ready, preflow, np.nan)
    return out


def diagnostics_tables(frame):
    """Uncertainty-free descriptive tables, with explicit metric-specific matching."""
    f = frame.copy()
    f['date'] = f.bin_start.dt.normalize()
    f['weekday'] = f.bin_start.dt.dayofweek
    f['slot'] = (f.bin_start.dt.hour * 4 + f.bin_start.dt.minute // 15).astype(int)
    daily = f.groupby(['period', 'date'], observed=True).agg(
        eligible_bins=('candidate_event', 'size'), candidate_bins=('candidate_event', 'sum'),
        comparator_bins=('candidate_without_duration', 'sum'), mean_distance=('distance', 'mean')).reset_index()
    cells = f.groupby(['period', 'weekday', 'slot'], observed=True).agg(
        eligible_bins=('candidate_event', 'size'), candidate_bins=('candidate_event', 'sum')).reset_index()
    signs = (f[['z_ofi', 'z_intensity', 'z_duration']].to_numpy() >= 0).astype(int)
    f['sign_code'] = 4 * signs[:, 0] + 2 * signs[:, 1] + signs[:, 2]
    profiles = f.groupby(['period', 'candidate_event', 'sign_code'], observed=True).agg(
        bins=('distance', 'size'), mean_z_ofi=('z_ofi', 'mean'),
        mean_z_intensity=('z_intensity', 'mean'), mean_z_duration=('z_duration', 'mean')).reset_index()
    d = f.loc[f.period.eq('development')]
    rows = []
    for metric in METRICS:
        finite = np.isfinite(d[metric])
        selected = d.loc[finite, ['weekday', 'slot', 'candidate_event', metric]]
        events = selected.loc[selected.candidate_event].groupby(['weekday', 'slot'])[metric].agg(['size', 'sum'])
        controls = selected.loc[~selected.candidate_event].groupby(['weekday', 'slot'])[metric].mean()
        matched = events.join(controls.rename('control_mean'), how='inner')
        n = int(matched['size'].sum())
        event_values = d.loc[d.candidate_event & finite, metric]
        control_values = d.loc[~d.candidate_event & finite, metric]
        event_mean = float(matched['sum'].sum() / n) if n else np.nan
        control_mean = float((matched['size'] * matched.control_mean).sum() / n) if n else np.nan
        rows.append({'metric': metric, 'event_available': len(event_values), 'control_available': len(control_values),
                     'event_missing': int((d.candidate_event & ~finite).sum()),
                     'control_missing': int((~d.candidate_event & ~finite).sum()),
                     'matched_events': n, 'unmatched_events': len(event_values) - n,
                     'event_mean': event_values.mean(), 'event_median': event_values.median(),
                     'control_mean_unmatched': control_values.mean(), 'control_median': control_values.median(),
                     'matched_event_mean': event_mean, 'event_weighted_control_mean': control_mean,
                     'matched_difference': event_mean - control_mean})
    return {'daily': daily, 'time_cells': cells, 'feature_profiles': profiles,
            'economic_comparison': pd.DataFrame(rows)}

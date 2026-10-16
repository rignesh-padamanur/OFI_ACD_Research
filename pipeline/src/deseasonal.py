"""Past-day seasonal calibration; legacy full-series fitting is disabled."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from src.intensity import materialize_clock_policy
from src.observation_eligibility import POLICIES

SECOND = 1_000_000_000
DAY = 86400 * SECOND
POLICY_NAMES = tuple(POLICIES)
CHANNELS = ('tick_count', 'abs_quote_ofi')
STATUS = {0: 'full_window_warmup', 1: 'too_few_prior_dates', 2: 'too_few_prior_seconds',
          3: 'nonpositive_or_unavailable_mean', 4: 'nonpositive_or_unavailable_scale', 5: 'ready'}


@dataclass(frozen=True)
class CalibrationSpec:
    lookback_days: int = 56
    min_prior_dates: int = 4
    min_prior_seconds: int = 1800
    slot_seconds: int = 900

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Positive integer calibration settings required: '+key)
        if self.slot_seconds != 900 or self.lookback_days < 7 or self.min_prior_seconds < 2:
            raise ValueError('This version requires 15-minute cells, at least seven prior days and two seconds')
        if self.min_prior_dates > self.lookback_days // 7:
            raise ValueError('Distinct-date requirement exceeds matching weekday exposures')


def compute_diurnal_pattern(*args, **kwargs):
    raise RuntimeError('Full-series seasonal estimation is disabled. Use the registered past-day calibration stage.')


def deseasonalize_series(*args, **kwargs):
    raise RuntimeError('Full-series seasonal adjustment is disabled. Use iter_calibrated_clock(root, policy=...).')


def slots(frame):
    t = pd.DatetimeIndex(frame.bin_start)
    if t.tz is None or t.hasnans:
        raise ValueError('Nonmissing timezone-aware clock required')
    ns = t.as_unit('ns').asi8
    if (np.diff(ns) <= 0).any() or (ns % SECOND).any():
        raise ValueError('Ordered second-aligned clock required')
    available = pd.DatetimeIndex(frame.available_at)
    if available.tz is None or available.hasnans:
        raise ValueError('Nonmissing timezone-aware availability required')
    if not np.array_equal(available.as_unit('ns').asi8, ns+SECOND):
        raise ValueError('Completed-bin availability differs')
    return ((ns % DAY) // (900*SECOND)).astype('int16')


def daily_moments(frame):
    """Per-policy/channel/cell count, mean and centered sum of squares."""
    cell = slots(frame)
    days = pd.DatetimeIndex(frame.bin_start).tz_convert('UTC').normalize()
    if days.nunique() != 1:
        raise ValueError('One UTC day required for sufficient statistics')
    result = np.zeros((len(POLICY_NAMES), len(CHANNELS), 96, 3), dtype=float)
    for pi, policy in enumerate(POLICY_NAMES):
        model = materialize_clock_policy(frame, policy=policy)
        for vi, variable in enumerate(CHANNELS):
            x = model[variable].to_numpy(dtype=float); valid = np.isfinite(x)
            if (x[valid] < 0).any():
                raise ValueError('Seasonal channels must be nonnegative')
            s, v = cell[valid], x[valid]
            n = np.bincount(s, minlength=96)
            mean = np.divide(np.bincount(s, weights=v, minlength=96), n, out=np.zeros(96), where=n>0)
            m2 = np.bincount(s, weights=np.square(v-mean[s]), minlength=96)
            result[pi, vi] = np.column_stack([n, mean, m2])
    return result


def fit_day(history, index, spec):
    """Fit using [index-lookback,index), never the application day's moments.

    History is a consecutive UTC-day array from the registered grid origin.
    Counts are sample denominators, not independent observations for inference.
    """
    if index < 0 or index > len(history) or history.shape[1:] != (4, 2, 96, 3):
        raise ValueError('Invalid consecutive-day history shape or application index')
    begin = max(0, index-spec.lookback_days)
    recent = history[begin:index]
    matching = recent[(index-np.arange(begin,index)) % 7 == 0]
    dn = matching[..., 0]; dm = matching[..., 1]; dm2 = matching[..., 2]
    n = dn.sum(axis=0)
    dates = np.count_nonzero(dn, axis=0)
    mean = np.divide((dn*dm).sum(axis=0), n, out=np.zeros_like(n), where=n>0)
    m2 = (dm2+dn*np.square(dm-mean)).sum(axis=0)
    sd = np.sqrt(np.divide(m2, n-1, out=np.full_like(n,np.nan), where=n>1))
    gn = recent[..., 0].sum(axis=(0,3))
    global_mean = np.divide((recent[..., 0]*recent[..., 1]).sum(axis=(0,3)), gn,
                            out=np.full((4,2),np.nan), where=gn>0)
    g = np.broadcast_to(global_mean[..., None], mean.shape).copy()
    mean[n == 0] = np.nan
    code = np.full(n.shape, 5, dtype='uint8')
    code[~np.isfinite(sd) | (sd <= 0)] = 4
    code[~np.isfinite(mean) | (mean <= 0) | ~np.isfinite(g) | (g <= 0)] = 3
    code[n < spec.min_prior_seconds] = 2
    code[dates < spec.min_prior_dates] = 1
    if index < spec.lookback_days:
        code[:] = 0
    phi = np.divide(mean, g, out=np.full_like(mean,np.nan), where=code==5)
    return {'n': n.astype('int64'), 'prior_dates': dates.astype('int16'), 'mean': mean, 'sample_std': sd,
            'global_mean': g, 'phi': phi, 'status': code}


def normalize_policy(frame, fit, *, policy):
    """Location/scale deviations plus multiplicative adjustment preserving flow sign."""
    if policy not in POLICIES:
        raise ValueError('Explicit registered policy required')
    pi = POLICY_NAMES.index(policy); cell = slots(frame)
    out = materialize_clock_policy(frame, policy=policy)
    for vi, variable in enumerate(CHANNELS):
        x = out[variable].to_numpy()
        mean = fit['mean'][pi,vi,cell]; sd = fit['sample_std'][pi,vi,cell]
        phi = fit['phi'][pi,vi,cell]; status = fit['status'][pi,vi,cell]
        valid = (status == 5) & np.isfinite(x)
        if ((status == 5) & (~np.isfinite(phi) | (phi<=0) | ~np.isfinite(sd) | (sd<=0))).any():
            raise ValueError('Ready calibration has an invalid factor or scale')
        out['z_'+variable] = np.divide(x-mean, sd, out=np.full(len(frame),np.nan), where=valid)
        out['adjusted_'+variable] = np.divide(x, phi, out=np.full(len(frame),np.nan), where=valid)
        out['calibration_status_'+variable] = status
        if variable == 'abs_quote_ofi':
            out['adjusted_quote_ofi'] = np.divide(out.quote_ofi.to_numpy(), phi,
                                                  out=np.full(len(frame),np.nan), where=valid)
    out['joint_calibrated'] = out[['z_tick_count','z_abs_quote_ofi']].notna().all(axis=1)
    return out


def normalize_day(frame, fit):
    result = frame[['bin_start','available_at']].reset_index(drop=True).copy()
    for policy in POLICY_NAMES:
        model = normalize_policy(frame, fit, policy=policy)
        for variable in CHANNELS:
            result[f'z_{variable}_{policy}'] = model['z_'+variable].to_numpy()
    return result


def moments_table(day, values):
    rows = []
    for pi,p in enumerate(POLICY_NAMES):
        for vi,v in enumerate(CHANNELS):
            for cell in range(96):
                n, mean, m2 = values[pi,vi,cell]
                rows.append({'date': day, 'policy': p, 'channel': v, 'slot': cell,
                             'n': int(n), 'mean': float(mean) if n else np.nan, 'm2': float(m2)})
    return pd.DataFrame(rows)


def fit_table(day, fit, spec):
    rows = []
    for pi,p in enumerate(POLICY_NAMES):
        for vi,v in enumerate(CHANNELS):
            for cell in range(96):
                row = {'effective_date': day, 'training_start': day-pd.Timedelta(days=spec.lookback_days),
                       'training_end_exclusive': day, 'policy': p, 'channel': v, 'slot': cell}
                for key in fit:
                    value = fit[key][pi,vi,cell]
                    row[key] = value.item()
                rows.append(row)
    return pd.DataFrame(rows)


def table_to_fits(table):
    """Validate a complete ordered per-day parameter table for safe lookup."""
    fits = {}
    for day, frame in table.groupby('effective_date', sort=True):
        if len(frame) != 4*2*96:
            raise ValueError('Incomplete daily calibration table')
        f = {key: np.full((4,2,96), np.nan) for key in ['n','prior_dates','mean','sample_std','global_mean','phi','status']}
        seen = set()
        for row in frame.itertuples(index=False):
            ix = (POLICY_NAMES.index(row.policy), CHANNELS.index(row.channel), row.slot)
            if row.slot not in range(96) or ix in seen or row.training_end_exclusive != day:
                raise ValueError('Invalid calibration cell identity or cutoff')
            seen.add(ix)
            for key in f:
                f[key][ix] = getattr(row,key)
        if not np.isin(f['status'], list(STATUS)).all():
            raise ValueError('Unknown calibration status')
        fits[pd.Timestamp(day).value] = f
    return fits

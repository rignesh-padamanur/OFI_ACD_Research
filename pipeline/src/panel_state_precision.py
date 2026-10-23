"""Panel quote states with direct finite-window recorded-variation sums.

The quote marks, support checks, prior imbalance window, and output schema are
the same as ``event_audit.quote_diagnostics``. Only variance accumulation changes:
each 60-return window is summed directly, without subtracting expired returns or
clamping a rounded negative rolling sum to zero. Historical source is preserved.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.event_audit import SECOND, ns


def direct_square_window_sums(values, window=60):
    """Direct nonnegative sums; every term must be finite and present.

    The overlapping windows are views, not a materialized window-by-time array.
    Each output depends only on its own bounded window, so cancellation from a
    formerly large return cannot remain after that return exits the window.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('One-dimensional values and a positive integer window required')
    finite = np.isfinite(values)
    if (values[finite] < 0).any():
        raise ValueError('Squared-return terms cannot be negative')
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = sliding_window_view(values, window)
        counts = sliding_window_view(finite, window).sum(axis=-1)
        # terms are nonnegative; keep nonfinite windows unavailable
        with np.errstate(invalid='ignore', over='ignore'):
            total = windows.sum(axis=-1)
        result[window-1:] = np.where(counts == window, total, np.nan)
    return result


def quote_diagnostics(clock, quotes, support_intervals, policy):
    """Strictly preceding quote marks and the existing 300-second prior states."""
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
        rv[phase::5] = direct_square_window_sums(r2[phase::5], 60)
    flow = pd.Series(clock.quote_ofi.to_numpy())
    preflow = flow.shift(1).rolling(300, min_periods=300).sum().to_numpy()
    same = np.zeros(len(t), dtype=bool)
    same[300:] = (sid[300:] >= 0) & (sid[300:] == sid[:-300])
    ready = same & np.isfinite(preflow) & np.isfinite(rv)
    out['pre_rv_300s_5s_bps'] = np.where(ready, np.sqrt(rv) * 10000, np.nan)
    out['pre_quote_ofi_300s'] = np.where(ready, preflow, np.nan)
    return out

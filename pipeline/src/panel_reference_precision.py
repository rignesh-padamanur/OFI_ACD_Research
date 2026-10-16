"""Independent quote-state reference with direct, finite-window square sums.

Versioned after the stage-07c numerical build. This only replaces the verifier's
cancellation-prone rolling sum; production artifacts and tolerances are unchanged.
"""
import numpy as np
import pandas as pd

SECOND = 1_000_000_000


def ns(values):
    return pd.DatetimeIndex(values).as_unit('ns').asi8


def direct_square_window_sums(values, window=60):
    """Sum nonnegative terms directly; require every term in the finite window.

    Convolution adds only the window's terms and never subtracts a prior large
    sum. The separate count preserves missing-data semantics. No variance floor
    is applied. Nonfinite terms make their containing windows unavailable.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('One-dimensional values and a positive integer window required')
    finite = np.isfinite(values)
    if (values[finite] < 0).any():
        raise ValueError('Squared-return terms cannot be negative')
    if not len(values):
        return np.empty(0, dtype=float)
    sums = np.convolve(np.where(finite, values, 0.), np.ones(window), mode='full')[:len(values)]
    counts = np.convolve(finite.astype('int64'), np.ones(window, dtype='int64'), mode='full')[:len(values)]
    return np.where(counts == window, sums, np.nan)


def reference_quote_diagnostics(clock, quotes, support_intervals):
    """Strict backward marks and direct five-second-grid realized variation."""
    times = ns(clock.bin_start)
    q = quotes[['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume']].copy()
    q['quote_timestamp'] = q.timestamp
    support_start = ns(support_intervals.start)
    support_stop = ns(support_intervals.end_exclusive)
    sid = clock.support_id.to_numpy(dtype=int)
    safe = np.maximum(sid, 0)

    def marks(mark_time):
        request = pd.DataFrame({'timestamp': pd.to_datetime(mark_time, utc=True)})
        joined = pd.merge_asof(request, q, on='timestamp', direction='backward', allow_exact_matches=False)
        qt = ns(joined.quote_timestamp)
        valid = ((sid >= 0) & joined.quote_timestamp.notna().to_numpy()
                 & (qt >= support_start[safe]) & (qt < support_stop[safe]))
        return joined, valid

    pre, pre_valid = marks(times)
    end, end_valid = marks(times+SECOND)
    pre_size = (pre.bid_volume+pre.ask_volume).to_numpy()
    end_size = (end.bid_volume+end.ask_volume).to_numpy()
    pre_spread = (pre.ask-pre.bid).to_numpy()*10000
    end_spread = (end.ask-end.bid).to_numpy()*10000
    out = pd.DataFrame({'bin_start': clock.bin_start.to_numpy(),
                        'pre_spread_pips': np.where(pre_valid, pre_spread, np.nan),
                        'end_spread_pips': np.where(end_valid, end_spread, np.nan),
                        'spread_change_pips': np.where(pre_valid & end_valid, end_spread-pre_spread, np.nan),
                        'pre_displayed_size': np.where(pre_valid, pre_size, np.nan),
                        'end_displayed_size': np.where(end_valid, end_size, np.nan),
                        'log_size_ratio': np.nan,
                        'pre_quote_age_sec': np.where(pre_valid, (times-ns(pre.quote_timestamp))/SECOND, np.nan),
                        'end_quote_age_sec': np.where(end_valid, (times+SECOND-ns(end.quote_timestamp))/SECOND, np.nan)})
    size_ok = pre_valid & end_valid & (pre_size > 0) & (end_size > 0)
    out.loc[size_ok, 'log_size_ratio'] = np.log(end_size[size_ok]/pre_size[size_ok])
    history_valid = np.isin(clock.flow_status.to_numpy(), [1, 2]) & np.isfinite(clock.quote_ofi)
    flow = pd.Series(clock.quote_ofi.to_numpy()).where(history_valid).shift(1).rolling(300, min_periods=300).sum().to_numpy()
    enough = np.arange(len(clock)) >= 300
    same = np.zeros(len(clock), dtype=bool)
    if len(clock) > 300:
        same[300:] = ((sid[300:] >= 0) & (sid[300:] == sid[:-300])
                      & (times[300:]-times[:-300] == 300*SECOND))
    ready = enough & same & np.isfinite(flow)
    mid = (pre.bid+pre.ask).to_numpy()/2
    log_mid = pd.Series(np.where(pre_valid & (mid > 0), np.log(mid), np.nan))
    squared = (log_mid-log_mid.shift(5)).pow(2)
    total = np.full(len(clock), np.nan)
    for phase in range(5):
        index = np.arange(phase, len(clock), 5)
        total[index] = direct_square_window_sums(squared.iloc[index].to_numpy(), 60)
    valid_returns = ready & np.isfinite(total)
    out['pre_quote_ofi_300s'] = np.where(valid_returns, flow, np.nan)
    out['pre_rv_300s_5s_bps'] = np.where(valid_returns, np.sqrt(total)*10000, np.nan)
    return out

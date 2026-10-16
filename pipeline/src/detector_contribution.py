"""Fixed exploratory detector comparison; no production-pointer mutations."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.event_audit import FEATURES, ns, score_detector
from src.panel_state_precision import direct_square_window_sums

SECOND = 1_000_000_000
RULES = ['full', 'two_feature', 'diagonal', 'marginal', 'ofi', 'activity',
         'duration', 'spread', 'variation']
PRIMARY = ['diagonal', 'two_feature', 'marginal']
XCOLS = ['current_spread', 'pre_spread', 'history_spread', 'variation',
         'log_count', 'log_history_count', 'quote_age', 'history_age', 'hour', 'weekday']


def core_scores(frame, fit):
    x = frame[FEATURES].to_numpy(float)
    z = (x - np.asarray(fit['mean'])) / np.asarray(fit['sample_std'])
    small = {k: np.asarray(fit[k])[:2].tolist() for k in ['mean', 'sample_std']}
    small['covariance_z'] = np.asarray(fit['covariance_z'])[:2, :2].tolist()
    return np.column_stack([score_detector(x, fit), score_detector(x[:, :2], small),
                            np.linalg.norm(z, axis=1), np.abs(z).max(axis=1), np.abs(z)])


def calibrate(scores):
    x = np.asarray(scores, float)
    thresholds = np.array([np.quantile(c[np.isfinite(c)], .99, method='linear') for c in x.T])
    counts = np.isfinite(x).sum(axis=0)
    return thresholds, counts, np.sum(x == thresholds, axis=0)


def hash_training_sample(times):
    """Stable timestamp-only approximately 1/20 selection, independent of outcomes."""
    x = (np.asarray(times, dtype=np.int64) // SECOND).astype(np.uint64)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> 31)
    return x % 20 == 0


def causal_onsets(sid, known, trigger, carry=None):
    """Vectorized five-clear-second onset logic, with chunk-invariant history."""
    sid, known, trigger = np.asarray(sid), np.asarray(known, bool), np.asarray(trigger, bool)
    if len(sid) != len(known) or known.shape != trigger.shape or (trigger & ~known).any():
        raise ValueError('Invalid detector clock')
    n = len(sid)
    if carry is None:
        carry = {'sid': [-1]*5, 'known': [False]*5, 'trigger': [False]*5}
    s = np.r_[carry['sid'], sid]
    k = np.r_[carry['known'], known]
    a = np.r_[carry['trigger'], trigger]
    confirmed = trigger.copy()
    for lag in range(1, 6):
        confirmed &= k[5-lag:5-lag+n] & ~a[5-lag:5-lag+n] & (s[5-lag:5-lag+n] == sid)
    i = np.arange(len(s))
    prev = np.r_[-100, np.maximum.accumulate(np.where(a, i, -100))[:-1]]
    barrier = np.maximum.accumulate(np.where(~k | np.r_[True, np.diff(s) != 0], i, -100))
    fragment = a & ((i-prev > 5) | (prev < barrier))
    next_carry = {'sid': s[-5:].tolist(), 'known': k[-5:].tolist(), 'trigger': a[-5:].tolist()}
    return confirmed, fragment[5:], next_carry


def _sum_windows(values, left, right):
    """Inclusive integer sums, avoiding cancellation of tiny price differences."""
    pref = np.r_[np.int64(0), np.cumsum(values, dtype=np.int64)]
    return pref[right+1] - pref[left]


class QuoteGrid:
    """Strict-left recorded quote marks on a bounded consecutive second grid."""
    def __init__(self, quotes, support, start, stop):
        self.t = np.arange(pd.Timestamp(start).value, pd.Timestamp(stop).value, SECOND)
        if len(self.t) < 302:
            raise ValueError('Grid too short')
        q = ns(quotes.timestamp)
        if not len(q) or (np.diff(q) <= 0).any():
            raise ValueError('Ordered unique quotes required')
        raw = quotes[['bid', 'ask']].to_numpy(float) * 100000
        ticks = np.rint(raw).astype(np.int64)
        if not np.isfinite(raw).all() or not np.allclose(raw, ticks, atol=1e-7, rtol=0):
            raise ValueError('Source prices do not lie on integer vendor grid')
        if (ticks[:, 0] <= 0).any() or (ticks[:, 1] < ticks[:, 0]).any():
            raise ValueError('Invalid source quote')
        a, b = ns(support.start), ns(support.end_exclusive)
        j = np.searchsorted(a, self.t, side='left') - 1
        qi = np.searchsorted(q, self.t, side='left') - 1
        jj, qq = np.maximum(j, 0), np.maximum(qi, 0)
        valid = (j >= 0) & (qi >= 0) & (self.t <= b[jj]) & (q[qq] >= a[jj])
        self.sid = np.where(valid, j, -1)
        self.spread = ticks[qq, 1] - ticks[qq, 0]
        self.mid = ticks[qq].sum(axis=1)
        self.age = (self.t-q[qq])/SECOND
        self.valid = valid
        self.badprefix = np.r_[0, np.cumsum(~valid)]

    def index(self, times):
        times = np.asarray(times, dtype=np.int64)
        i = (times-self.t[0])//SECOND
        if ((times-self.t[0]) % SECOND).any() or (i < 0).any() or (i >= len(self.t)).any():
            raise ValueError('Origins outside aligned quote grid')
        return i

    def window_valid(self, left, right, sid):
        return ((self.badprefix[right+1]-self.badprefix[left] == 0)
                & (self.sid[left] == sid) & (self.sid[right] == sid) & (sid >= 0))

    def features(self, origins, clock):
        t, sid = ns(origins.bin_start), origins.support_id.to_numpy()
        i = self.index(t)
        if (i < 300).any() or (i+1 >= len(self.t)).any():
            raise ValueError('Insufficient feature grid')
        # Quote count/flow history uses the same strict pre-bin 300 seconds.
        ct = ns(clock.bin_start)
        ci = (t-ct[0])//SECOND
        if (np.diff(ct) != SECOND).any() or (ci < 300).any() or (ci >= len(ct)).any():
            raise ValueError('Insufficient clock history')
        good_clock = np.isin(clock.flow_status_structural_only, [1, 2])
        cp = np.r_[0, np.cumsum(~good_clock)]
        hist_ok = self.window_valid(i-300, i, sid) & (cp[ci]-cp[ci-300] == 0)
        # Sum 60 actual five-second return squares, without subtracting old sums.
        logmid = np.where(self.valid, np.log(self.mid/200000), np.nan)
        squares = np.full(len(self.t), np.nan)
        squares[5:] = (logmid[5:]-logmid[:-5])**2
        rv = np.full(len(self.t), np.nan)
        for phase in range(5):
            rv[phase::5] = direct_square_window_sums(squares[phase::5], 60)
        hist_age = pd.Series(np.where(self.valid, self.age, np.nan)).rolling(300, min_periods=300).max().to_numpy()
        current_ok = self.window_valid(i+1, i+1, sid)
        pre_count = _sum_windows(clock.recorded_update_count.to_numpy(), ci-300, ci-1)/300
        out = origins[['bin_start', 'support_id']].reset_index(drop=True).copy()
        out['current_spread'] = np.where(current_ok, self.spread[i+1]/10, np.nan)
        out['pre_spread'] = np.where(hist_ok, self.spread[i]/10, np.nan)
        out['history_spread'] = np.where(hist_ok, _sum_windows(self.spread, i-299, i)/3000, np.nan)
        out['variation'] = np.where(hist_ok, np.sqrt(rv[i])*10000, np.nan)
        out['log_count'] = np.log1p(origins.recorded_update_count.to_numpy())
        out['log_history_count'] = np.where(hist_ok, np.log1p(pre_count), np.nan)
        out['quote_age'] = np.where(current_ok, self.age[i+1], np.nan)
        out['history_age'] = np.where(hist_ok, hist_age[i], np.nan)
        out['hour'] = out.bin_start.dt.hour.astype(float)
        out['weekday'] = out.bin_start.dt.dayofweek.astype(float)
        out['reference_ready'] = np.isfinite(out[XCOLS]).all(axis=1)
        return out

    def outcomes(self, origins):
        t, sid = ns(origins.bin_start), origins.support_id.to_numpy()
        i = self.index(t)
        if (i+301 >= len(self.t)).any():
            raise ValueError('Insufficient future grid')
        out = pd.DataFrame(index=np.arange(len(i)))
        for h in [60, 300]:
            ok = self.window_valid(i+1, i+1+h, sid)
            out[f'y{h}'] = np.where(ok, _sum_windows(self.spread, i+2, i+1+h)/(10*h), np.nan)
        ok = self.window_valid(i+1, i+301, sid)
        out['absolute_move'] = np.where(ok, np.abs(self.mid[i+301]-self.mid[i+1])/20, np.nan)
        direction = np.sign(origins.quote_ofi.to_numpy()) if 'quote_ofi' in origins else np.ones(len(i))
        good_direction = (direction != 0) & np.isfinite(direction)
        initial_ok = self.window_valid(i, i+1, sid) & good_direction
        out['J'] = np.where(initial_ok, direction*(self.mid[i+1]-self.mid[i])/20, np.nan)
        out['C'] = np.where(ok & good_direction, direction*(self.mid[i+301]-self.mid[i+1])/20, np.nan)
        out['I'] = out.J+out.C
        age = pd.Series(np.where(self.valid, self.age, np.nan)).rolling(301, min_periods=301).max().to_numpy()
        out['future_age'] = np.where(ok, age[i+301], np.nan)
        out['outcome_ready'] = np.isfinite(out.y300)
        return out


def ratio_hac(daily_sums, daily_counts, lag):
    sums, counts = np.asarray(daily_sums, float), np.asarray(daily_counts, float)
    if sums.shape != counts.shape or sums.ndim != 2 or not np.isfinite(sums).all() or (counts < 0).any():
        raise ValueError('Invalid daily ratio inputs')
    total = counts.sum(axis=0)
    if (total <= 0).any() or not 0 <= lag < len(sums):
        raise ValueError('Empty ratio or invalid lag')
    means = sums.sum(axis=0)/total
    influence = (sums-counts*means)/total
    cov = influence.T@influence
    for l in range(1, lag+1):
        cross = influence[l:].T@influence[:-l]
        cov += (1-l/(lag+1))*(cross+cross.T)
    return means, cov, influence


def contrast_table(sums, counts, lag):
    mean, cov, influence = ratio_hac(sums, counts, lag)
    rows = []
    for j, name in enumerate(RULES[1:], 1):
        variance = cov[0, 0]+cov[j, j]-2*cov[0, j]
        if variance < -1e-15:
            raise ValueError('Negative contrast variance')
        se = np.sqrt(max(variance, 0))
        d = mean[0]-mean[j]
        z = norm.ppf(1-.05/(2*3)) if name in PRIMARY else norm.ppf(.975)
        g = influence[:, 0]-influence[:, j]
        rows.append({'benchmark': name, 'lag_days': lag, 'difference_pips': d,
                     'se': se, 'lower': d-z*se, 'upper': d+z*se,
                     'interval': 'Bonferroni_3' if name in PRIMARY else 'pointwise',
                     'max_daily_squared_influence_share': float(np.max(g*g)/np.sum(g*g)) if np.any(g) else 0.,
                     'max_influence_day_index': int(np.argmax(g*g))})
    return pd.DataFrame(rows)


def match_times(left, right, tolerance=5):
    """Ordered one-to-one matching within a caller-supplied support interval."""
    left, right = np.asarray(left), np.asarray(right)
    if (np.diff(left) <= 0).any() or (np.diff(right) <= 0).any():
        raise ValueError('Unique sorted onset times required')
    i = j = 0
    offsets = []
    while i < len(left) and j < len(right):
        delta = int(right[j]-left[i])
        if abs(delta) <= tolerance:
            offsets.append(delta); i += 1; j += 1
        elif delta > 0:
            i += 1
        else:
            j += 1
    return np.array(offsets, dtype=np.int64)

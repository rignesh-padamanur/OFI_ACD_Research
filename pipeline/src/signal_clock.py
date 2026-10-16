"""Uniform clock without confusing unsupported silence with zero recorded activity."""
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.observation_eligibility import POLICIES
from src.quote_imbalance_statistics import Moments
from src.intensity import materialize_clock_policy

SECOND = 1_000_000_000
HOUR = 3600 * SECOND
SOURCE_COLUMNS = ['bin_start', 'n_quotes', 'quote_ofi', 'n_invalid_transitions', 'max_incoming_gap_sec']


def ns(series):
    idx = pd.DatetimeIndex(series)
    if idx.tz is None or idx.hasnans:
        raise ValueError('Nonmissing timezone-aware timestamps required')
    return idx.as_unit('ns').asi8


class SparseCursor:
    """Forward-only sparse reader. Gaps are not filled by the reader."""
    def __init__(self, path, batch_size=400_000):
        self.iterator = pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS)
        self.pending = None
        self.last_source_time = None

    def _advance(self):
        batch = next(self.iterator, None)
        if batch is None:
            self.pending = None
            return False
        frame = batch.to_pandas()
        times = ns(frame.bin_start)
        if (times % SECOND).any() or (np.diff(times) <= 0).any():
            raise ValueError('Sparse clock must be unique, sorted and second-aligned')
        if self.last_source_time is not None and times[0] <= self.last_source_time:
            raise ValueError('Sparse clock ordering failed across batches')
        self.last_source_time = times[-1]
        if (frame.n_quotes <= 0).any():
            raise ValueError('Sparse rows must contain quotes')
        self.pending = frame
        return True

    def take(self, start, stop):
        pieces = []
        while True:
            if self.pending is None or self.pending.empty:
                if not self._advance():
                    break
            times = ns(self.pending.bin_start)
            if times[0] < start:
                raise ValueError('Unconsumed sparse row outside requested grid')
            cut = np.searchsorted(times, stop, side='left')
            if cut:
                pieces.append(self.pending.iloc[:cut])
                self.pending = self.pending.iloc[cut:]
            if len(self.pending):
                break
        if pieces:
            return pd.concat(pieces, ignore_index=True)
        return pd.DataFrame({'bin_start': pd.Series([], dtype='datetime64[ns, UTC]'),
                             **{name: pd.Series([], dtype='float64') for name in SOURCE_COLUMNS[1:]}})

    def exhausted(self):
        return (self.pending is None or self.pending.empty) and not self._advance()


def prepare_clock_inputs(hours, support):
    h = ns(hours.hour)
    if not len(h) or (h % HOUR).any() or (np.diff(h) != HOUR).any():
        raise ValueError('Hourly ledger must cover a contiguous hour-aligned range')
    observed = hours.structurally_observed_hour.to_numpy(dtype=bool)
    opened = hours.nominal_weekly_open.to_numpy(dtype=bool)
    if (observed & ~opened).any() or not np.array_equal(observed, hours.rows.to_numpy() > 0):
        raise ValueError('Hourly calendar/observation labels conflict')
    hour_codes = np.where(observed, 2, np.where(opened, 1, 0)).astype('uint8')
    intervals = {}
    if set(support.policy.unique()) != set(POLICIES):
        raise ValueError('All registered policies required')
    for policy in POLICIES:
        part = support.loc[support.policy.eq(policy)]
        a, b = ns(part.start), ns(part.end_exclusive)
        if not len(a) or (a >= b).any() or (a[1:] < b[:-1]).any():
            raise ValueError('Nonempty ordered disjoint support intervals required')
        if a[0] < h[0] or b[-1] > h[-1] + HOUR:
            raise ValueError('Support falls outside requested grid')
        intervals[policy] = (a, b)
    return h[0], h[-1] + HOUR, hour_codes, intervals


def build_clock_chunk(start, stop, sparse, *, origin, hour_codes, intervals):
    if start % SECOND or stop % SECOND or stop <= start or start < origin:
        raise ValueError('Ordered second-aligned grid boundaries required')
    times = np.arange(start, stop, SECOND, dtype='int64')
    index = pd.to_datetime(times, utc=True)
    t = ns(sparse.bin_start)
    if len(t) and ((np.diff(t) <= 0).any() or (t % SECOND).any() or t[0] < start or t[-1] >= stop):
        raise ValueError('Sparse rows lie outside this grid or have duplicate/off-clock timestamps')
    aligned = sparse.set_index('bin_start').reindex(index)
    counts = aligned.n_quotes.fillna(0).to_numpy(dtype='int64')
    invalid = aligned.n_invalid_transitions.fillna(0).to_numpy(dtype='int64')
    flow = aligned.quote_ofi.to_numpy(dtype=float)
    gap = aligned.max_incoming_gap_sec.to_numpy(dtype=float)
    if (counts < 0).any() or (invalid < 0).any() or (invalid > counts).any():
        raise ValueError('Invalid source counts')
    if ((counts > 0) & (invalid == 0) & ~np.isfinite(flow)).any():
        raise ValueError('Missing complete active flow')
    hcodes = hour_codes[(times - origin) // HOUR]
    if ((counts > 0) & (hcodes != 2)).any():
        raise ValueError('Quotes found outside observed hours')
    result = pd.DataFrame({'bin_start': index, 'available_at': pd.to_datetime(times + SECOND, utc=True),
        'hour_status': hcodes, 'recorded_update_count': counts, 'observed_quote_ofi': flow,
        'n_invalid_transitions': invalid, 'max_incoming_gap_sec': gap})
    for policy, freshness in POLICIES.items():
        a, b = intervals[policy]
        j = np.searchsorted(a, times, side='right') - 1
        covered = (j >= 0) & (b[np.maximum(j, 0)] >= times + SECOND)
        if (covered & (hcodes != 2)).any():
            raise ValueError('Support overlaps an unobserved or closed hour')
        active = counts > 0
        status = np.where(covered, np.where(active, 2, 1), 0).astype('uint8')
        if freshness is not None:
            status[covered & active & ~(gap <= freshness)] = 4
        status[covered & active & (invalid > 0)] = 3
        result[f'support_id_{policy}'] = np.where(covered, j, -1).astype('int32')
        result[f'flow_status_{policy}'] = status
    return result


def policy_counts(frame, policy):
    status = frame[f'flow_status_{policy}'].to_numpy()
    count = frame.recorded_update_count.to_numpy()
    supported = status != 0
    active = count > 0
    return {
        'clock_seconds': len(frame), 'supported_seconds': int(supported.sum()),
        'unsupported_seconds': int((~supported).sum()),
        'supported_no_update_seconds': int((status == 1).sum()),
        'supported_active_seconds': int((supported & active).sum()),
        'usable_active_flow_seconds': int((status == 2).sum()),
        'invalid_transition_seconds': int((status == 3).sum()),
        'excess_incoming_gap_seconds': int((status == 4).sum()),
        'usable_flow_seconds': int(np.isin(status, [1, 2]).sum()),
        'supported_recorded_updates': int(count[supported].sum()),
        'supported_sunday_seconds': int((supported & (frame.bin_start.dt.dayofweek.to_numpy() == 6)).sum()),
    }


class ClockStatistics:
    def __init__(self):
        self.total = Counter()
        self.counts = {p: Counter() for p in POLICIES}
        self.moments = {p: {v: Moments() for v in ['tick_count', 'quote_ofi']} for p in POLICIES}
        self.months, self.hours = {}, {}

    def add(self, frame):
        count = frame.recorded_update_count.to_numpy()
        self.total.update({'clock_seconds': len(frame), 'recorded_updates': int(count.sum()),
            'active_seconds': int((count > 0).sum()), 'no_recorded_update_seconds': int((count == 0).sum()),
            'weekly_closed_seconds': int((frame.hour_status == 0).sum()),
            'absent_nominal_open_seconds': int((frame.hour_status == 1).sum()),
            'observed_hour_seconds': int((frame.hour_status == 2).sum())})
        month = frame.bin_start.dt.year.to_numpy() * 100 + frame.bin_start.dt.month.to_numpy()
        hour = frame.bin_start.dt.hour.to_numpy()
        for policy in POLICIES:
            self.counts[policy].update(policy_counts(frame, policy))
            model = materialize_clock_policy(frame, policy=policy)
            for variable in self.moments[policy]:
                self.moments[policy][variable].add(model[variable])
            for key in np.unique(month):
                self.months.setdefault((int(key), policy), Counter()).update(policy_counts(frame.loc[month == key], policy))
            for key in np.unique(hour):
                self.hours.setdefault((int(key), policy), Counter()).update(policy_counts(frame.loc[hour == key], policy))

    def tables(self):
        return (pd.DataFrame([{'month': f'{m//100}-{m%100:02}', 'policy': p, **dict(c)}
                             for (m, p), c in sorted(self.months.items())]),
                pd.DataFrame([{'utc_hour': h, 'policy': p, **dict(c)} for (h, p), c in sorted(self.hours.items())]))

    def result(self):
        return {'totals': dict(self.total), 'policies': {p: dict(c) for p, c in self.counts.items()},
                'descriptive_moments': {p: {v: m.result() for v, m in values.items()} for p, values in self.moments.items()},
                'primary_freshness_policy': None, 'coverage_certified': False, 'downstream_ready': False,
                'new_quotes_acquired': 0, 'calibration_or_models_reestimated': False}

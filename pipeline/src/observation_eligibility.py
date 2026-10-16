"""Observation support, not a certification of feed completeness or an event model.

All interval endpoints use integer UTC nanoseconds and [start, end) semantics.
Freshness permits age <= F. Hard boundaries remain exclusive. Long durations
are retained in the parent panel. Structural-only support has no freshness rule.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.acquisition import nominal_weekly_open

SECOND = 1_000_000_000
HOUR = 3600 * SECOND
POLICIES = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}
LOOKBACKS = (0, 60, 300)
HORIZONS = (0, 1, 5, 30, 60, 300)


def utc_ns(values):
    ix = pd.DatetimeIndex(values)
    if ix.tz is None or ix.hasnans:
        raise ValueError('Explicit timezone-aware, nonmissing timestamps required')
    return ix.tz_convert('UTC').as_unit('ns').asi8


def make_hours(coverage):
    table = coverage.copy().reset_index(drop=True)
    times = utc_ns(table.hour)
    if not len(times) or (times % HOUR).any() or (np.diff(times) != HOUR).any():
        raise ValueError('Coverage must be a complete ordered hourly grid')
    if (table.rows < 0).any():
        raise ValueError('Negative hourly observation counts')
    calendar = nominal_weekly_open(table.hour)
    if not np.array_equal(calendar, table.nominal_weekly_open):
        raise ValueError('Coverage calendar differs from nominal weekly schedule')
    allowed = calendar & table.rows.gt(0).to_numpy()
    if (table.rows.gt(0).to_numpy() & ~calendar).any():
        raise ValueError('Observed quotes outside nominal weekly schedule need explicit reconciliation')
    starts = allowed & ~np.r_[False, allowed[:-1]]
    block = np.cumsum(starts).astype('int64') - 1
    table['observation_block_id'] = np.where(allowed, block, -1)
    table['structurally_observed_hour'] = allowed
    table['eligibility_reason'] = np.where(~calendar, 'weekly_closed',
        np.where(allowed, 'observed_not_completeness_certified', 'absent_nominal_open_unresolved'))
    return table


def load_quote_coordinates(parent, input_files):
    """Read only time/provenance; raw ordinals retain the cleaning omissions."""
    offsets = {}; count = 0
    for item in input_files:
        offsets[item['name']] = (count, item['rows']); count += item['rows']
    times = []; ordinals = []
    for batch in pq.ParquetFile(Path(parent) / 'clean_quotes.parquet').iter_batches(
            batch_size=500_000, columns=['timestamp', 'source_file', 'source_row']):
        frame = batch.to_pandas()
        row = frame.source_row.to_numpy(dtype='int64')
        ordinal = np.empty(len(frame), dtype='int64')
        for name in frame.source_file.unique():
            offset, length = offsets[name]
            mask = frame.source_file.eq(name).to_numpy()
            if ((row[mask] < 0) | (row[mask] >= length)).any():
                raise ValueError('Source row outside its registered file')
            ordinal[mask] = offset + row[mask]
        times.append(utc_ns(frame.timestamp)); ordinals.append(ordinal)
    if not times:
        raise ValueError('No retained quotes')
    ts, raw_ids = np.concatenate(times), np.concatenate(ordinals)
    if (np.diff(ts) <= 0).any():
        raise ValueError('This eligibility contract requires strictly increasing quotes; tied timestamps need a separate ordering policy')
    if (np.diff(raw_ids) <= 0).any():
        raise ValueError('Clean source provenance is not strictly increasing')
    return ts, raw_ids


def quote_structure(ts, raw_ids, hours):
    h = (ts - utc_ns(hours.hour)[0]) // HOUR
    if (h < 0).any() or (h >= len(hours)).any():
        raise ValueError('Quote outside requested coverage')
    block = hours.observation_block_id.to_numpy()[h]
    if (block < 0).any():
        raise ValueError('Retained quote in an unsupported hour')
    calendar_break = np.r_[False, np.diff(block) != 0]
    removed_before = np.r_[raw_ids[0], np.diff(raw_ids) - 1]
    starts = np.flatnonzero(np.r_[True, (calendar_break | (removed_before > 0))[1:]])
    return h, calendar_break, removed_before, starts


def scan_raw_support(parent, hours, required_raw_ids):
    """Read every raw timestamp; independently reconcile hourly counts and fetch boundaries."""
    origin = utc_ns(hours.hour)[0]
    wanted = np.array(sorted(set(map(int, required_raw_ids))), dtype='int64')
    found = {}; counts = np.zeros(len(hours), dtype='int64'); offset = 0; previous = None
    first = np.full(len(hours), np.iinfo('int64').max, dtype='int64')
    last = np.full(len(hours), np.iinfo('int64').min, dtype='int64')
    for batch in pq.ParquetFile(Path(parent) / 'raw_quotes.parquet').iter_batches(
            batch_size=1_000_000, columns=['timestamp']):
        ts = utc_ns(batch.column(0).to_pandas())
        if (np.diff(ts) < 0).any() or (previous is not None and ts[0] < previous):
            raise ValueError('Unordered raw timestamps')
        h = (ts - origin) // HOUR
        if (h < 0).any() or (h >= len(hours)).any():
            raise ValueError('Raw timestamp outside audit scope')
        counts += np.bincount(h, minlength=len(hours))
        edges = np.r_[0, np.flatnonzero(np.diff(h)) + 1]
        ends = np.r_[edges[1:] - 1, len(ts) - 1]
        first[h[edges]] = np.minimum(first[h[edges]], ts[edges])
        last[h[ends]] = np.maximum(last[h[ends]], ts[ends])
        ids = wanted[(wanted >= offset) & (wanted < offset + len(ts))]
        found.update(zip(ids.tolist(), ts[ids - offset].tolist()))
        offset += len(ts); previous = int(ts[-1])
    if len(found) != len(wanted):
        raise ValueError('Missing requested raw boundary')
    np.testing.assert_array_equal(counts, hours.rows.to_numpy())
    observed = counts > 0
    np.testing.assert_array_equal(first[observed], utc_ns(hours.loc[observed, 'first_timestamp']))
    np.testing.assert_array_equal(last[observed], utc_ns(hours.loc[observed, 'last_timestamp']))
    return found, offset, previous + 1


def build_segments(ts, raw_ids, hours, boundary_times, sample_end):
    h, calendar_break, removed_before, starts = quote_structure(ts, raw_ids, hours)
    stops = np.r_[starts[1:], len(ts)]
    hour_ns = utc_ns(hours.hour)
    blocks = hours.observation_block_id.to_numpy()
    good = blocks >= 0
    end_hours = np.flatnonzero(good & ~np.r_[good[1:] & (blocks[1:] == blocks[:-1]), False])
    block_ends = {int(blocks[i]): int(hour_ns[i] + HOUR) for i in end_hours}
    absent = np.r_[0, np.cumsum((hours.nominal_weekly_open & hours.rows.eq(0)).to_numpy())]
    closed = np.r_[0, np.cumsum((~hours.nominal_weekly_open).to_numpy())]
    rows = []
    for segment_id, (a, b) in enumerate(zip(starts, stops)):
        end = min(block_ends[int(blocks[h[a]])], sample_end)
        next_removed_id = int(raw_ids[b - 1] + 1)
        if next_removed_id in boundary_times:
            end = min(end, boundary_times[next_removed_id])
        if not ts[b - 1] < end:
            raise ValueError('Removed raw record ties a retained timestamp; explicit tie policy required')
        left_hour = int(h[a - 1]) + 1 if a else 0
        missing = int(absent[h[a]] - absent[left_hour]) if a else int(absent[h[a]])
        closures = int(closed[h[a]] - closed[left_hour]) if a else int(closed[h[a]])
        reasons = []
        if not a: reasons.append('sample_start')
        if a and closures: reasons.append('weekly_closure')
        if a and missing: reasons.append('absent_nominal_open_hour')
        if removed_before[a] > 0: reasons.append('removed_raw_record')
        if a and not reasons: raise AssertionError('Unexplained structural break')
        rows.append({'segment_id': segment_id, 'quote_start': int(a), 'quote_stop_exclusive': int(b),
            'quote_count': int(b-a), 'start': pd.Timestamp(int(ts[a]), tz='UTC'),
            'last_quote': pd.Timestamp(int(ts[b-1]), tz='UTC'),
            'end_exclusive': pd.Timestamp(end, tz='UTC'),
            'start_reason': '|'.join(reasons), 'removed_raw_records_before': int(removed_before[a]),
            'absent_open_hours_before': missing, 'weekly_closed_hours_before': closures})
    return pd.DataFrame(rows)


def support_intervals(ts, segments):
    base_starts = segments.quote_start.to_numpy(dtype='int64')
    base_stops = segments.quote_stop_exclusive.to_numpy(dtype='int64')
    base_end = utc_ns(segments.end_exclusive)
    gaps = np.diff(ts)
    result = []
    for policy, fresh in POLICIES.items():
        starts = base_starts if fresh is None else np.union1d(base_starts, np.flatnonzero(gaps > fresh * SECOND) + 1)
        stops = np.r_[starts[1:], len(ts)]
        sid = np.searchsorted(base_starts, starts, side='right') - 1
        ends = base_end[sid].copy()
        if fresh is not None:
            ends = np.minimum(ends, ts[stops-1] + fresh * SECOND + 1)
        result.append(pd.DataFrame({'policy': policy, 'segment_id': sid,
            'quote_start': starts, 'quote_stop_exclusive': stops,
            'start': pd.to_datetime(ts[starts], utc=True), 'end_exclusive': pd.to_datetime(ends, utc=True)}))
    return pd.concat(result, ignore_index=True)


def evaluate_windows(intervals, anchors, lookback_seconds, horizons=HORIZONS):
    """Keep past eligibility and future availability distinct; future flags are labels only.

    Call on ONE policy. Full path continuity is required, not only endpoints.
    Segment ends, support ends and outcome flags must never be model predictors.
    """
    if lookback_seconds < 0 or any(h < 0 for h in horizons):
        raise ValueError('Negative window')
    if 'policy' in intervals and intervals.policy.nunique() != 1:
        raise ValueError('Select exactly one support policy')
    t = utc_ns(anchors); a = utc_ns(intervals.start); b = utc_ns(intervals.end_exclusive)
    if not len(a) or (a >= b).any() or (a[1:] < b[:-1]).any():
        raise ValueError('Expected ordered nonoverlapping nonempty support intervals')
    i = np.searchsorted(a, t, side='right') - 1; j = np.maximum(i, 0)
    inside = (i >= 0) & (t < b[j])
    pre = inside & (t - lookback_seconds * SECOND >= a[j])
    out = pd.DataFrame({'timestamp': pd.to_datetime(t, utc=True), 'pre_context_eligible': pre})
    for horizon in horizons:
        available = inside & (t + horizon * SECOND < b[j])
        out[f'outcome_available_{horizon}s'] = available
        out[f'joint_eligible_{horizon}s'] = pre & available
    return out


def grid_ranges(intervals, lookback, horizon):
    a = utc_ns(intervals.start); b = utc_ns(intervals.end_exclusive)
    lo = (a + SECOND - 1) // SECOND + lookback
    hi = (b + SECOND - 1) // SECOND - horizon
    keep = hi > lo
    return lo[keep], hi[keep]


def count_ranges_by_hour(lo, hi, hour_edges):
    if not len(lo): return np.zeros(len(hour_edges)-1, dtype='int64')
    length = hi-lo; cumulative = np.r_[0, np.cumsum(length)]
    i = np.searchsorted(lo, hour_edges, side='right') - 1; j = np.maximum(i, 0)
    total = np.where(i >= 0, cumulative[j] + np.minimum(np.maximum(hour_edges-lo[j], 0), length[j]), 0)
    return np.diff(total)


def window_attrition(intervals, hours):
    hour_edges = np.r_[utc_ns(hours.hour)//SECOND, (utc_ns(hours.hour)[-1]+HOUR)//SECOND]
    tables = []; summary = []
    for policy in POLICIES:
        part = intervals.loc[intervals.policy.eq(policy)]
        for lookback in LOOKBACKS:
            pre_lo, pre_hi = grid_ranges(part, lookback, 0)
            pre_count = int((pre_hi-pre_lo).sum())
            for horizon in HORIZONS:
                lo, hi = grid_ranges(part, lookback, horizon)
                counts = count_ranges_by_hour(lo, hi, hour_edges)
                tables.append(pd.DataFrame({'hour': hours.hour, 'policy': policy,
                    'lookback_seconds': lookback, 'horizon_seconds': horizon, 'eligible_clock_anchors': counts}))
                summary.append({'policy': policy, 'lookback_seconds': lookback, 'horizon_seconds': horizon,
                    'pre_context_clock_anchors': pre_count, 'joint_clock_anchors': int(counts.sum()),
                    'future_unavailable_among_pre_eligible': pre_count-int(counts.sum())})
    return pd.concat(tables, ignore_index=True), summary


def transition_statistics(ts, segments):
    delta = np.diff(ts)
    hard = segments.quote_start.to_numpy(dtype='int64')[1:]-1
    structural = np.ones(len(delta), dtype=bool); structural[hard] = False
    counts = {'retained_quotes': len(ts), 'potential_adjacent_quote_transitions': len(delta),
        'structural_segments': len(segments), 'structurally_eligible_transitions': int(structural.sum()),
        'structurally_excluded_transitions': len(hard), 'initial_quotes_without_predecessor': 1}
    for fresh in (15, 30, 60):
        counts[f'within_segment_gaps_gt_{fresh}s'] = int(((delta > fresh*SECOND) & structural).sum())
        counts[f'transitions_with_structural_support_and_gap_le_{fresh}s'] = int(((delta <= fresh*SECOND) & structural).sum())
    ix = np.flatnonzero(delta > 15*SECOND)
    gaps = pd.DataFrame({'previous_quote_id': ix, 'quote_id': ix+1,
        'previous_timestamp': pd.to_datetime(ts[ix], utc=True), 'timestamp': pd.to_datetime(ts[ix+1], utc=True),
        'delta_sec': delta[ix]/SECOND, 'crosses_structural_boundary': ~structural[ix]})
    return counts, gaps

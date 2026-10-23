"""Bounded-memory quote spells and past-only seasonal inputs for the full panel.

The statistical rules are those of ``duration_calibration``.  This module changes
their materialization, not their window lengths, censoring or readiness rules.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.duration_calibration import (
    DAY, DurationAuditSpec, apply_scales, fit_scales, sufficient_statistics,
)
from src.observation_eligibility import POLICIES

BATCH_SIZE = 500_000


def _ns(values):
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError('Nonmissing timezone-aware timestamps required')
    return index.as_unit('ns').asi8


def _integer(values, name):
    values = np.asarray(values)
    if (not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError(name + ' must contain integers')
    return values


def iter_source_spells(path, terminals, *, policy, stop_ns, batch_size=BATCH_SIZE):
    """Yield the same outgoing spells as the prefix builder, one batch at a time.

    Exactly one prior quote is retained between input batches.  The first quote
    at or after the stop can establish a pending wait, but its completion is not
    admitted at the historical stop.  Quote IDs keep their original global value.
    """
    if policy not in POLICIES:
        raise ValueError('Explicit policy required')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    if isinstance(stop_ns, bool) or not isinstance(stop_ns, (int, np.integer)):
        raise ValueError('Integer nanosecond stop required')
    tails = terminals.loc[terminals.policy.eq(policy)].sort_values('last_quote_id')
    tail_ids = _integer(tails.last_quote_id.to_numpy(), 'Terminal identities')
    if len(tail_ids) and (np.diff(tail_ids) <= 0).any():
        raise ValueError('Unique terminal identities required')
    tail_end = _ns(tails.stop)
    tail_code = _integer(tails.stop_code.to_numpy(), 'Terminal codes')
    if (tail_code < 0).any() or (tail_code > 255).any():
        raise ValueError('Terminal code outside uint8 domain')
    component = 'component_id_' + policy
    pending = None
    expected_id = 0
    previous_time = None
    yielded = False

    def materialize(frame, count):
        times = _ns(frame.available_at)
        ids = frame.quote_id.to_numpy()
        cid = frame[component].to_numpy()
        origins = times[:count]
        end = np.full(count, stop_ns, dtype='int64')
        observed = np.zeros(count, dtype=bool)
        code = np.full(count, 8, dtype='uint8')
        has_next = np.arange(count) + 1 < len(frame)
        same = np.zeros(count, dtype=bool)
        positions = np.flatnonzero(has_next)
        same[positions] = cid[positions] == cid[positions + 1]
        next_times = times[np.flatnonzero(same) + 1]
        end[same] = np.minimum(next_times, stop_ns)
        observed[same] = next_times < stop_ns
        code[observed] = 0
        terminal_positions = np.flatnonzero(~same)
        if len(terminal_positions):
            where = np.searchsorted(tail_ids, ids[terminal_positions])
            if ((where >= len(tail_ids)).any()
                    or not np.array_equal(tail_ids[where], ids[terminal_positions])):
                raise ValueError('Unaccounted terminal spell')
            ends = tail_end[where]
            end[terminal_positions] = np.minimum(ends, stop_ns)
            code[terminal_positions] = np.where(ends < stop_ns, tail_code[where], 8)
        if (end < origins).any():
            raise ValueError('Negative observed exposure')
        return pd.DataFrame({
            'origin_quote_id': ids[:count], 'component_id': cid[:count],
            'origin_at': pd.to_datetime(origins, utc=True),
            'end_at': pd.to_datetime(end, utc=True),
            'event_observed': observed, 'stop_code': code, 'exposure_ns': end - origins,
        })

    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size, columns=['quote_id', 'available_at', component]):
        frame = batch.to_pandas()
        if not len(frame):
            continue
        times = _ns(frame.available_at)
        if (np.diff(times) <= 0).any():
            raise ValueError('Unique increasing quote timestamps required')
        # read the requested prefix plus one lookahead observation
        crossing = int(np.searchsorted(times, stop_ns))
        frame = frame.iloc[:min(len(frame), crossing + 1)].reset_index(drop=True)
        times = _ns(frame.available_at)
        ids = _integer(frame.quote_id.to_numpy(), 'Quote identities')
        _integer(frame[component].to_numpy(), 'Component identities')
        if not np.array_equal(ids, np.arange(expected_id, expected_id + len(frame))):
            raise ValueError('Contiguous global quote identities required')
        if ((np.diff(times) <= 0).any()
                or (previous_time is not None and times[0] <= previous_time)):
            raise ValueError('Unique increasing quote timestamps required')
        expected_id += len(frame)
        previous_time = times[-1]
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
        origins = _ns(frame.available_at)
        reached_stop = origins[-1] >= stop_ns
        count = int(np.searchsorted(origins, stop_ns)) if reached_stop else len(frame) - 1
        if count:
            yield materialize(frame, count)
            yielded = True
        if reached_stop:
            if not yielded:
                raise ValueError('Empty chronological audit prefix')
            return
        pending = frame.iloc[[-1]].reset_index(drop=True)
    if pending is not None:
        yield materialize(pending, 1)
        yielded = True
    if not yielded:
        raise ValueError('Empty chronological audit prefix')


def build_seasonal_panel(quote_durations, terminal_spells, outdir, *, origin, stop,
                         policy='structural_only'):
    """Write the five full-panel duration artifacts without loading all spells.

    Counts marked seasonal-ready include zero-exposure terminal records; positive
    exposure and event counts are separately reported.  No fitted ACD is involved.
    """
    origin, stop = pd.Timestamp(origin), pd.Timestamp(stop)
    for value in (origin, stop):
        if (pd.isna(value) or value.tz is None or value.utcoffset().total_seconds() != 0
                or value != value.normalize()):
            raise ValueError('UTC midnight bounds required')
    ndays = int((stop.value - origin.value) // DAY)
    if ndays <= 112:
        raise ValueError('Full seasonal panel requires more than 112 calendar days')
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise ValueError('Output directory must already exist')
    names = ['raw_spells', 'origin_daily_sufficient_statistics',
             'historical_cutoff_corrections', 'seasonal_parameters', 'seasonal_spells']
    if any((outdir / (name + '.parquet')).exists() for name in names):
        raise FileExistsError('Seasonal panel outputs are immutable')
    spec = DurationAuditSpec(validation_days=ndays - 112)
    tails = (terminal_spells if isinstance(terminal_spells, pd.DataFrame)
             else pd.read_parquet(terminal_spells))
    totals = np.zeros((ndays * 96, 3), dtype='int64')
    correction_totals = {}
    counts = {'calendar_days': ndays, 'stream_batch_size': BATCH_SIZE, 'source_batches': 0,
              'source_spell_rows': 0, 'source_completed_events': 0,
              'source_exposure_ns': 0, 'source_positive_exposure_rows': 0}
    raw_writer = None
    try:
        for frame in iter_source_spells(quote_durations, tails, policy=policy,
                                       stop_ns=stop.value, batch_size=BATCH_SIZE):
            table, corrections = sufficient_statistics(frame, origin.value, ndays)
            totals += table[['origins', 'completed_events', 'full_exposure_ns']].to_numpy()
            for row in corrections.itertuples(index=False):
                key = (int(row.cutoff_day), int(row.origin_day), int(row.slot))
                old = correction_totals.get(key, (0, 0))
                correction_totals[key] = (old[0] + int(row.future_exposure_ns),
                                          old[1] + int(row.future_completions))
            arrow = pa.Table.from_pandas(frame, preserve_index=False)
            if raw_writer is None:
                raw_writer = pq.ParquetWriter(outdir / 'raw_spells.parquet', arrow.schema)
            raw_writer.write_table(arrow, row_group_size=BATCH_SIZE)
            counts['source_batches'] += 1
            counts['source_spell_rows'] += len(frame)
            counts['source_completed_events'] += int(frame.event_observed.sum())
            counts['source_exposure_ns'] += int(frame.exposure_ns.sum())
            counts['source_positive_exposure_rows'] += int(frame.exposure_ns.gt(0).sum())
    finally:
        if raw_writer is not None:
            raw_writer.close()
    table = pd.DataFrame({'origin_day': np.repeat(np.arange(ndays), 96),
                          'slot': np.tile(np.arange(96), ndays),
                          'origins': totals[:, 0], 'completed_events': totals[:, 1],
                          'full_exposure_ns': totals[:, 2]})
    corrections = pd.DataFrame([(*key, *value) for key, value in sorted(correction_totals.items())],
        columns=['cutoff_day', 'origin_day', 'slot', 'future_exposure_ns', 'future_completions'])
    table.to_parquet(outdir / 'origin_daily_sufficient_statistics.parquet', index=False)
    corrections.to_parquet(outdir / 'historical_cutoff_corrections.parquet', index=False)
    seasonal, scales, statuses = fit_scales(table, corrections, origin.value, spec)
    seasonal.to_parquet(outdir / 'seasonal_parameters.parquet', index=False)
    counts['seasonal_parameter_rows'] = len(seasonal)
    counts['seasonal_ready_rows'] = 0
    counts['seasonal_ready_completed_events'] = 0
    status_counts = np.zeros(6, dtype='int64')
    writer = None
    try:
        for batch in pq.ParquetFile(outdir / 'raw_spells.parquet').iter_batches(batch_size=BATCH_SIZE):
            frame = apply_scales(batch.to_pandas(), scales, statuses, origin.value)
            ready = frame.seasonal_status.eq(5)
            counts['seasonal_ready_rows'] += int(ready.sum())
            counts['seasonal_ready_completed_events'] += int((ready & frame.event_observed).sum())
            status_counts += np.bincount(frame.seasonal_status.to_numpy(), minlength=6)
            arrow = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(outdir / 'seasonal_spells.parquet', arrow.schema)
            writer.write_table(arrow, row_group_size=BATCH_SIZE)
    finally:
        if writer is not None:
            writer.close()
    counts['seasonal_status_counts'] = {str(i): int(n) for i, n in enumerate(status_counts)}
    return counts


def read_quote_window(path, start, stop, columns):
    """Return [start, stop) plus the nearest quote strictly preceding start.

    Binary searches use row-group extrema.  Missing extrema are read only for the
    row groups touched by those searches; complete prefix data are never loaded.
    The caller's registered clean input supplies the global ordering guarantee.
    """
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if (pd.isna(start) or pd.isna(stop) or start.tz is None or stop.tz is None or stop <= start):
        raise ValueError('Nonempty timezone-aware quote window required')
    columns = list(columns)
    if 'timestamp' not in columns or len(set(columns)) != len(columns):
        raise ValueError('Unique requested columns must include timestamp')
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index('timestamp')
    if index < 0 or any(c not in parquet.schema_arrow.names for c in columns):
        raise ValueError('Requested quote columns are unavailable')
    def empty_window():
        return pa.Table.from_batches([], schema=parquet.schema_arrow).select(columns).to_pandas()[columns]
    if parquet.metadata.num_rows == 0:
        return empty_window()
    bounds_cache = {}

    def bounds(i):
        if i not in bounds_cache:
            stats = parquet.metadata.row_group(i).column(index).statistics
            if stats is not None and stats.has_min_max:
                lo, hi = pd.Timestamp(stats.min).value, pd.Timestamp(stats.max).value
            else:
                time = parquet.read_row_group(i, columns=['timestamp']).to_pandas().timestamp
                values = _ns(time)
                if not len(values) or (np.diff(values) <= 0).any():
                    raise ValueError('Empty or unordered quote row group')
                lo, hi = int(values[0]), int(values[-1])
            if hi < lo:
                raise ValueError('Invalid quote row-group bounds')
            bounds_cache[i] = (lo, hi)
        return bounds_cache[i]

    def lower_bound(field, value):
        left, right = 0, parquet.num_row_groups
        while left < right:
            middle = (left + right) // 2
            if bounds(middle)[field] < value:
                left = middle + 1
            else:
                right = middle
        return left

    first = lower_bound(1, start.value)
    last = lower_bound(0, stop.value)
    groups = list(range(first, last))
    if first == parquet.num_row_groups or bounds(first)[0] >= start.value:
        if first > 0:
            groups.insert(0, first - 1)
    if not groups:
        return empty_window()
    frame = parquet.read_row_groups(groups, columns=columns).to_pandas()[columns]
    times = _ns(frame.timestamp)
    if (np.diff(times) <= 0).any():
        raise ValueError('Unique increasing quote timestamps required')
    left = int(np.searchsorted(times, start.value, side='left'))
    right = int(np.searchsorted(times, stop.value, side='left'))
    return frame.iloc[max(0, left - 1):right].reset_index(drop=True)

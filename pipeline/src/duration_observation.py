"""Observed quote waiting spells, conditional on audited support; no ACD fitting."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.observation_eligibility import POLICIES
from src.quote_imbalance_statistics import Moments

SECOND = 1_000_000_000
STOP_CODES = {1: 'freshness_cap', 2: 'structural_boundary', 4: 'sample_end'}


def nanos(values):
    return pd.DatetimeIndex(values).as_unit('ns').asi8


def prepare_components(support, segments, n):
    if not n or set(support.policy) != set(POLICIES):
        raise ValueError('Nonempty quotes and all registered policies required')
    if not np.array_equal(segments.segment_id, np.arange(len(segments))):
        raise ValueError('Invalid structural IDs')
    tables = {}
    for policy in POLICIES:
        c = support.loc[support.policy.eq(policy)].reset_index(drop=True).copy()
        a = c.quote_start.to_numpy(); b = c.quote_stop_exclusive.to_numpy()
        if a[0] != 0 or b[-1] != n or (b <= a).any() or not np.array_equal(a[1:], b[:-1]):
            raise ValueError('Components must partition quote IDs: ' + policy)
        starts = nanos(c.start); ends = nanos(c.end_exclusive)
        if (ends <= starts).any() or (starts[1:] < ends[:-1]).any():
            raise ValueError('Invalid/overlapping support: ' + policy)
        sid = c.segment_id.to_numpy()
        if (sid < 0).any() or (sid >= len(segments)).any():
            raise ValueError('Invalid segment reference')
        s = segments.iloc[sid]
        if (a < s.quote_start.to_numpy()).any() or (b > s.quote_stop_exclusive.to_numpy()).any():
            raise ValueError('Component crosses a structural boundary')
        tables[policy] = c
    structural = tables['structural_only']
    for col in ['segment_id', 'quote_start', 'quote_stop_exclusive', 'start', 'end_exclusive']:
        np.testing.assert_array_equal(structural[col], segments[col])
    return tables


def duration_chunk(timestamps, offset, components, previous_ns=None):
    t = nanos(timestamps)
    if not len(t) or (t == pd.NaT.value).any() or offset < 0 or ((offset == 0) != (previous_ns is None)):
        raise ValueError('Contiguous nonempty source chunks and predecessor required')
    gap = np.diff(np.r_[t[0] if previous_ns is None else previous_ns, t])
    if ((gap[1:] if previous_ns is None else gap) <= 0).any():
        raise ValueError('Strictly increasing quote times required')
    if previous_ns is None:
        gap[0] = -1  # Explicit absent-predecessor sentinel, never a duration.
    ids = np.arange(offset, offset+len(t), dtype='int64')
    out = pd.DataFrame({'quote_id': ids, 'available_at': pd.to_datetime(t, utc=True), 'audit_gap_ns': gap})
    for policy, c in components.items():
        a = c.quote_start.to_numpy()
        j = np.searchsorted(a, ids, side='right')-1
        if (j < 0).any() or (ids >= c.quote_stop_exclusive.to_numpy()[j]).any():
            raise ValueError('Quote IDs outside support partition')
        complete = ids > a[j]
        if POLICIES[policy] is not None and (gap[complete] > POLICIES[policy]*SECOND).any():
            raise ValueError('Complete wait exceeds policy cap')
        out['component_id_'+policy] = j.astype('int32')
        out['completed_'+policy] = complete
        out['recurrence_reset_'+policy] = ids == a[j]+1
    return out


def materialize_duration_policy(frame, *, policy):
    """Keep component-start rows unavailable; available_at is the later quote.

    This is only the completed-spell side of the ledger. A likelihood also needs
    terminal_spells.parquet, an explicit observation mechanism, and past-only
    parameter/seasonality fitting. Audit IDs are not predictive covariates.
    """
    if policy not in POLICIES:
        raise ValueError('An explicit registered support policy is required')
    c = frame['completed_'+policy].to_numpy(dtype=bool)
    gap = frame.audit_gap_ns.to_numpy()
    if (gap[c] <= 0).any() or (POLICIES[policy] is not None and (gap[c] > POLICIES[policy]*SECOND).any()):
        raise ValueError('Invalid completed duration')
    ns = pd.array(gap, dtype='Int64'); ns[~c] = pd.NA
    start = nanos(frame.available_at).copy(); start[c] -= gap[c]; start[~c] = pd.NaT.value
    return pd.DataFrame({'quote_id': frame.quote_id.to_numpy(), 'available_at': frame.available_at.to_numpy(),
        'component_id': frame['component_id_'+policy].to_numpy(), 'completed': c,
        'recurrence_reset': frame['recurrence_reset_'+policy].to_numpy(),
        'duration_start': pd.to_datetime(start, utc=True), 'duration_ns': ns,
        'duration_sec': np.where(c, gap/SECOND, np.nan)})


def terminal_spells(components, segments, edge_ids, edge_times, sample_last_ns):
    """Stop flags may combine at exact ties; no independent-censoring assertion."""
    rows = []
    hard = nanos(segments.end_exclusive).copy()
    sample = hard == sample_last_ns+1
    if (hard > sample_last_ns+1).any():
        raise ValueError('Structural stop exceeds source sample')
    hard[sample] = sample_last_ns
    for policy, c in components.items():
        a = c.quote_start.to_numpy(); b = c.quote_stop_exclusive.to_numpy()-1
        sid = c.segment_id.to_numpy(); j = np.searchsorted(edge_ids, b)
        if (j >= len(edge_ids)).any() or not np.array_equal(edge_ids[j], b):
            raise ValueError('Missing component endpoint quotes')
        last = edge_times[j]; start = nanos(c.start)
        ja = np.searchsorted(edge_ids, a)
        if not np.array_equal(edge_ids[ja], a) or not np.array_equal(edge_times[ja], start):
            raise ValueError('Support start differs from source quote')
        cap = np.full(len(c), np.iinfo('int64').max, dtype='int64') if POLICIES[policy] is None else last+POLICIES[policy]*SECOND
        stop = np.minimum(hard[sid], cap)
        expected_end = nanos(segments.end_exclusive)[sid] if POLICIES[policy] is None else np.minimum(nanos(segments.end_exclusive)[sid], cap+1)
        if not np.array_equal(expected_end, nanos(c.end_exclusive)) or (stop < last).any():
            raise ValueError('Registered half-open support and measured elapsed time differ')
        codes = ((cap == stop).astype('uint8') + ((hard[sid] == stop) & ~sample[sid]).astype('uint8')*2
                 + ((hard[sid] == stop) & sample[sid]).astype('uint8')*4)
        rows.append(pd.DataFrame({'policy': policy, 'component_id': np.arange(len(c), dtype='int32'),
            'segment_id': sid, 'first_quote_id': a, 'last_quote_id': b, 'completed_spells': b-a,
            'component_start': pd.to_datetime(start, utc=True), 'origin': pd.to_datetime(last, utc=True),
            'stop': pd.to_datetime(stop, utc=True), 'exposure_ns': stop-last, 'exposure_sec': (stop-last)/SECOND,
            'event_observed': False, 'positive_exposure': stop > last, 'stop_code': codes}))
    return pd.concat(rows, ignore_index=True)


class DurationStatistics:
    def __init__(self):
        self.rows = 0; self.audit = Moments(); self.moments = {p: Moments() for p in POLICIES}
        self.totals = {p: dict(completed=0, unavailable_incoming=0, recurrence_resets=0, completed_exposure_ns=0,
            completed_over_60s=0, legacy_kept_but_ineligible=0, eligible_but_legacy_dropped=0) for p in POLICIES}
        self.months = {}; self.legacy_kept = 0

    def add(self, frame):
        gap = frame.audit_gap_ns.to_numpy(); legacy = (gap > 0) & (gap <= 60*SECOND)
        self.rows += len(frame); self.legacy_kept += int(legacy.sum()); self.audit.add(gap[gap > 0]/SECOND)
        months = pd.DatetimeIndex(frame.available_at).tz_localize(None).to_numpy().astype('datetime64[M]')
        for p in POLICIES:
            ok = frame['completed_'+p].to_numpy(); reset = frame['recurrence_reset_'+p].to_numpy()
            counts = dict(completed=int(ok.sum()), unavailable_incoming=int((~ok).sum()), recurrence_resets=int(reset.sum()),
                completed_exposure_ns=int(gap[ok].sum()), completed_over_60s=int((ok & (gap > 60*SECOND)).sum()),
                legacy_kept_but_ineligible=int((legacy & ~ok).sum()), eligible_but_legacy_dropped=int((ok & ~legacy).sum()))
            for k,v in counts.items(): self.totals[p][k] += v
            self.moments[p].add(gap[ok]/SECOND)
            for month in np.unique(months):
                select = months == month; valid = select & ok
                row = self.months.setdefault((str(month), p), dict(quotes=0, completed=0, unavailable_incoming=0,
                    recurrence_resets=0, completed_exposure_ns=0))
                for k,v in dict(quotes=int(select.sum()), completed=int(valid.sum()), unavailable_incoming=int((select & ~ok).sum()),
                    recurrence_resets=int((select & reset).sum()), completed_exposure_ns=int(gap[valid].sum())).items(): row[k] += v

    def result(self, tails):
        policies = {}
        for p in POLICIES:
            t = tails.loc[tails.policy.eq(p)]; counts = self.totals[p]
            assert counts['completed']+len(t) == self.rows
            assert counts['unavailable_incoming'] == len(t)
            assert counts['recurrence_resets'] == int(t.completed_spells.gt(0).sum())
            exposure = int(t.exposure_ns.sum())
            span = int((nanos(t.stop)-nanos(t.component_start)).sum())
            assert counts['completed_exposure_ns']+exposure == span
            policies[p] = {**counts, 'components': len(t), 'single_quote_components': int(t.completed_spells.eq(0).sum()),
                'terminal_records': len(t), 'positive_terminal_exposures': int(t.positive_exposure.sum()),
                'zero_terminal_exposures': int((~t.positive_exposure).sum()), 'terminal_exposure_ns': exposure,
                'total_observed_span_ns': span, 'stop_code_counts': {str(k):int(v) for k,v in t.stop_code.value_counts().sort_index().items()},
                'completed_duration_seconds': self.moments[p].result()}
        return {'quote_rows': self.rows, 'audit_adjacent_gap_seconds': self.audit.result(), 'legacy_filter_kept': self.legacy_kept,
            'policies': policies, 'units': 'elapsed UTC seconds; exact arithmetic in integer nanoseconds',
            'acd_estimated': False, 'coverage_certified': False, 'primary_freshness_policy': None, 'downstream_ready': False}

    def monthly(self):
        return pd.DataFrame([{'month': m, 'policy': p, **row} for (m,p),row in sorted(self.months.items())])


def build_duration_artifacts(clean_path, raw_path, support, segments, run, *, batch_size=500_000):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    source = pq.ParquetFile(clean_path); n = source.metadata.num_rows
    components = prepare_components(support, segments, n)
    edge_ids = np.unique(np.concatenate([np.r_[c.quote_start, c.quote_stop_exclusive-1] for c in components.values()]))
    edge_times = np.full(len(edge_ids), pd.NaT.value, dtype='int64')
    stats = DurationStatistics(); offset = 0; previous = None; writer = None
    try:
        for batch in source.iter_batches(batch_size=batch_size, columns=['timestamp']):
            t = batch.to_pandas().timestamp; ns = nanos(t)
            frame = duration_chunk(t, offset, components, previous)
            lo, hi = np.searchsorted(edge_ids, [offset, offset+len(t)])
            edge_times[lo:hi] = ns[edge_ids[lo:hi]-offset]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None: writer = pq.ParquetWriter(run/'quote_durations.parquet', table.schema, compression='zstd')
            writer.write_table(table); stats.add(frame); offset += len(frame); previous = int(ns[-1])
            if offset % 5_000_000 == 0: print(f'Built {offset:,} duration rows', flush=True)
    finally:
        if writer is not None: writer.close()
    if offset != n or (edge_times == pd.NaT.value).any(): raise AssertionError('Source row/edge conservation failed')
    raw = pq.ParquetFile(raw_path)
    sample_last = nanos(raw.read_row_group(raw.num_row_groups-1, columns=['timestamp']).to_pandas().timestamp)[-1]
    tails = terminal_spells(components, segments, edge_ids, edge_times, int(sample_last))
    tails.to_parquet(run/'terminal_spells.parquet', index=False)
    stats.monthly().to_parquet(run/'monthly_statistics.parquet', index=False)
    return stats.result(tails)

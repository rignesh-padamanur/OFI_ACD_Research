"""Independent clock reconstruction: interval slices and direct sparse index placement."""
from collections import Counter
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import sha256
from src.signal_clock import SparseCursor  # I/O only; no production alignment/support/statistics logic.
from src.verify_quote_imbalance import Rows, compare_frames

S = 1_000_000_000
RULES = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}


def reference_chunk(start, stop, source, hours, support):
    seconds = np.arange(start // S, stop // S, dtype='int64')
    n = len(seconds)
    count = np.zeros(n, dtype='int64'); invalid = count.copy()
    flow = np.full(n, np.nan); gap = flow.copy()
    ix = pd.DatetimeIndex(source.bin_start).as_unit('ns').asi8 // S - start // S
    count[ix] = source.n_quotes.to_numpy(); invalid[ix] = source.n_invalid_transitions.to_numpy()
    flow[ix] = source.quote_ofi.to_numpy(); gap[ix] = source.max_incoming_gap_sec.to_numpy()
    hix = seconds // 3600 - hours.hour.iloc[0].value // (3600*S)
    ledger = hours.iloc[hix]
    hc = np.zeros(n, dtype='uint8')
    hc[ledger.nominal_weekly_open.to_numpy()] = 1
    hc[ledger.structurally_observed_hour.to_numpy()] = 2
    result = pd.DataFrame({'bin_start': pd.to_datetime(seconds, unit='s', utc=True).as_unit('ns'),
        'available_at': pd.to_datetime(seconds+1, unit='s', utc=True).as_unit('ns'),
        'hour_status': hc, 'recorded_update_count': count, 'observed_quote_ofi': flow,
        'n_invalid_transitions': invalid, 'max_incoming_gap_sec': gap})
    for policy, fresh in RULES.items():
        part = support.loc[support.policy.eq(policy)]
        a = pd.DatetimeIndex(part.start).as_unit('ns').asi8
        b = pd.DatetimeIndex(part.end_exclusive).as_unit('ns').asi8
        ids = np.full(n, -1, dtype='int32')
        for j in np.flatnonzero((b > start) & (a < stop)):
            left = max(-(-int(a[j])//S), int(seconds[0])) - int(seconds[0])
            right = min(int(b[j])//S, stop//S) - int(seconds[0])
            if right > left:
                if (ids[left:right] != -1).any():
                    raise AssertionError('Overlapping full-bin support')
                ids[left:right] = j
        covered = ids >= 0
        codes = np.zeros(n, dtype='uint8')
        codes[covered & (count == 0)] = 1
        codes[covered & (count > 0)] = 2
        if fresh is not None:
            codes[covered & (count > 0) & ((gap > fresh) | ~np.isfinite(gap))] = 4
        codes[covered & (invalid > 0)] = 3
        result[f'support_id_{policy}'] = ids
        result[f'flow_status_{policy}'] = codes
    return result


class ReferenceStatistics:
    def __init__(self):
        self.total = Counter(); self.policies = {p: Counter() for p in RULES}
        self.months = {}; self.hours = {}
        self.moments = {(p, v): np.array([0., 0., 0., np.inf, -np.inf]) for p in RULES for v in ['tick_count', 'quote_ofi']}

    def add(self, frame):
        count = frame.recorded_update_count.to_numpy(); hc = frame.hour_status.to_numpy()
        self.total.update({'clock_seconds': len(frame), 'recorded_updates': int(np.sum(count)),
            'active_seconds': int(np.count_nonzero(count)), 'no_recorded_update_seconds': int(np.count_nonzero(count == 0)),
            'weekly_closed_seconds': int(np.count_nonzero(hc == 0)),
            'absent_nominal_open_seconds': int(np.count_nonzero(hc == 1)),
            'observed_hour_seconds': int(np.count_nonzero(hc == 2))})
        months = frame.bin_start.dt.year.to_numpy()*100 + frame.bin_start.dt.month.to_numpy()
        hours = frame.bin_start.dt.hour.to_numpy(); sunday = frame.bin_start.dt.dayofweek.to_numpy() == 6
        for policy in RULES:
            codes = frame[f'flow_status_{policy}'].to_numpy(); valid = codes != 0
            metrics = {'clock_seconds': np.ones(len(frame), dtype='int8'),
                'supported_seconds': valid, 'unsupported_seconds': ~valid,
                'supported_no_update_seconds': codes == 1,
                'supported_active_seconds': valid & (count > 0),
                'usable_active_flow_seconds': codes == 2,
                'invalid_transition_seconds': codes == 3,
                'excess_incoming_gap_seconds': codes == 4,
                'usable_flow_seconds': (codes == 1) | (codes == 2),
                'supported_recorded_updates': count * valid,
                'supported_sunday_seconds': valid & sunday}
            self.policies[policy].update({name: int(values.sum()) for name, values in metrics.items()})
            for group, target in [(months, self.months), (hours, self.hours)]:
                keys, inv = np.unique(group, return_inverse=True)
                for name, values in metrics.items():
                    sums = np.bincount(inv, weights=values, minlength=len(keys))
                    for k, value in zip(keys, sums):
                        target.setdefault((int(k), policy), Counter())[name] += int(value)
            for variable in ['tick_count', 'quote_ofi']:
                if variable == 'tick_count':
                    x = count[valid].astype(float)
                else:
                    x = frame.observed_quote_ofi.to_numpy()[codes == 2]
                    x = np.r_[x, np.zeros(np.count_nonzero(codes == 1))]
                if len(x):
                    acc = self.moments[policy, variable]
                    acc[:3] += [len(x), x.sum(), np.square(x).sum()]
                    acc[3] = min(acc[3], x.min()); acc[4] = max(acc[4], x.max())

    def check(self, run):
        stats = json.loads((run/'statistics.json').read_text())
        if stats['totals'] != dict(self.total) or stats['policies'] != {p: dict(c) for p, c in self.policies.items()}:
            raise AssertionError('Clock statistics counts differ')
        for (p, variable), a in self.moments.items():
            n, total, squares, low, high = a
            values = {'count': int(n), 'mean': total/n if n else None,
                'sample_std': math.sqrt(max(0., squares-total*total/n)/(n-1)) if n>1 else None,
                'minimum': low if n else None, 'maximum': high if n else None}
            for k, expected in values.items():
                actual = stats['descriptive_moments'][p][variable][k]
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, err_msg=f'{p}/{variable}/{k}')
        month = pd.DataFrame([{'month': f'{m//100}-{m%100:02}', 'policy': p, **dict(c)} for (m,p),c in sorted(self.months.items())])
        hour = pd.DataFrame([{'utc_hour': h, 'policy': p, **dict(c)} for (h,p),c in sorted(self.hours.items())])
        pd.testing.assert_frame_equal(month, pd.read_parquet(run/'monthly_statistics.parquet'), check_dtype=False)
        pd.testing.assert_frame_equal(hour, pd.read_parquet(run/'utc_hour_statistics.parquet'), check_dtype=False)


def verify_clock_run(run, *, batch_size=173_111):
    run = Path(run); manifest = json.loads((run/'manifest.json').read_text())
    for name, record in manifest['inputs'].items():
        if sha256(Path(record['path'])) != record['sha256']:
            raise AssertionError('Input changed: '+name)
    for name, value in manifest['outputs'].items():
        if sha256(run/name) != value:
            raise AssertionError('Output changed: '+name)
    for name, value in manifest.get('code_sha256', {}).items():
        if sha256(run/'code'/name) != value:
            raise AssertionError('Executed source changed: '+name)
    hours = pd.read_parquet(manifest['inputs']['hourly_eligibility']['path'])
    support = pd.read_parquet(manifest['inputs']['support']['path'])
    origin = hours.hour.iloc[0].value; stop = hours.hour.iloc[-1].value + 3600*S
    assert stop > origin and len(hours)*3600*S == stop-origin
    sparse = SparseCursor(manifest['inputs']['sparse_bins']['path'], batch_size=131_071)
    actual = Rows(run/'signal_clock_1s.parquet'); stats = ReferenceStatistics()
    checked = 0; source_rows = 0; source_quotes = 0
    for start in range(origin, stop, batch_size*S):
        end = min(stop, start+batch_size*S); source = sparse.take(start, end)
        expected = reference_chunk(start, end, source, hours, support)
        saved = actual.take(len(expected)); compare_frames(expected, saved); stats.add(expected)
        checked += len(expected); source_rows += len(source); source_quotes += int(source.n_quotes.sum())
        if checked % (batch_size*50) == 0:
            print(f'Independently verified {checked:,} clock rows', flush=True)
    assert actual.exhausted() and sparse.exhausted()
    assert source_rows == pq.ParquetFile(manifest['inputs']['sparse_bins']['path']).metadata.num_rows
    assert stats.total['recorded_updates'] == source_quotes
    if 'signal_statistics' in manifest['inputs']:
        parent = json.loads(Path(manifest['inputs']['signal_statistics']['path']).read_text())
        assert source_quotes == parent['ticks']['quotes']
        assert source_rows == parent['bins']['quote_containing_bins']
    stats.check(run)
    # Additional closed-form interval accounting, independent of any row construction.
    for policy in RULES:
        part = support.loc[support.policy.eq(policy)]
        a = pd.DatetimeIndex(part.start).as_unit('ns').asi8
        b = pd.DatetimeIndex(part.end_exclusive).as_unit('ns').asi8
        bins = int(np.maximum(0, b//S - (-(-a//S))).sum())
        assert bins == stats.policies[policy]['supported_seconds']
    return {'status': 'passed', 'clock_rows_checked': checked, 'sparse_bins_checked': source_rows,
        'recorded_updates_checked': stats.total['recorded_updates'], 'verification_batch_size': batch_size,
        'checks': ['all clock values and availability timestamps', 'all policy statuses and support IDs',
            'independent interval-slice and closed-form support accounting', 'source quote/bin conservation',
            'all counts, descriptive moments, monthly and UTC-hour tables', 'input/output/source fingerprints']}

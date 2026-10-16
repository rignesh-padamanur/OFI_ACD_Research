"""Independent timestamp-difference, stop-search and exposure-sum verification."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import sha256
from src.verify_quote_imbalance import Rows, compare_frames

POLICIES = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}
S = 1_000_000_000


def verify_duration_run(run):
    run = Path(run); m = json.loads((run/'manifest.json').read_text())
    for name, record in m['inputs'].items():
        if sha256(Path(record['path'])) != record['sha256']: raise AssertionError('Input changed: '+name)
    for name, fingerprint in m['outputs'].items():
        if sha256(run/name) != fingerprint: raise AssertionError('Output changed: '+name)
    for name, fingerprint in m.get('code_sha256', {}).items():
        if sha256(run/'code'/name) != fingerprint: raise AssertionError('Code snapshot changed: '+name)
    inputs = {k: Path(v['path']) for k,v in m['inputs'].items()}
    support = pd.read_parquet(inputs['support']); segments = pd.read_parquet(inputs['segments'])
    tables = {p:support.loc[support.policy.eq(p)].reset_index(drop=True) for p in POLICIES}
    states = {p:dict(first=np.full(len(c), np.iinfo('int64').max, dtype='int64'),
        last=np.full(len(c), np.iinfo('int64').min, dtype='int64'), exposure=np.zeros(len(c), dtype='int64'),
        count=np.zeros(len(c), dtype='int64'), resets=0, n=0, sum_sec=0., square_sec=0., minimum=np.inf, maximum=-np.inf,
        over60=0, legacy_bad=0, legacy_dropped=0) for p,c in tables.items()}
    saved = Rows(run/'quote_durations.parquet'); previous = None; offset = 0; monthly = defaultdict(lambda: np.zeros(5, dtype='int64'))
    legacy_count = 0; raw_sum = 0.; raw_square = 0.; raw_n = 0; raw_min = np.inf; raw_max = -np.inf
    for batch in pq.ParquetFile(inputs['clean']).iter_batches(batch_size=333_331, columns=['timestamp']):
        times = batch.to_pandas().timestamp
        joined = times if previous is None else pd.concat([pd.Series([previous]), times], ignore_index=True)
        gap = joined.diff().to_numpy(dtype='timedelta64[ns]').astype('int64')
        if previous is None: gap[0] = -1
        else: gap = gap[1:]
        t = pd.DatetimeIndex(times).as_unit('ns').asi8; ids = np.arange(offset, offset+len(times), dtype='int64')
        legacy = (gap > 0) & (gap <= 60*S); legacy_count += int(legacy.sum())
        x = gap[gap > 0]/S; raw_n += len(x); raw_sum += float(x.sum()); raw_square += float(x@x)
        if len(x): raw_min = min(raw_min, float(x.min())); raw_max = max(raw_max, float(x.max()))
        expected = pd.DataFrame({'quote_id': ids, 'available_at': times.to_numpy(), 'audit_gap_ns': gap})
        months = pd.DatetimeIndex(times).tz_localize(None).to_numpy().astype('datetime64[M]')
        for p,c in tables.items():
            state = states[p]; cid = np.searchsorted(c.quote_stop_exclusive.to_numpy(), ids, side='right')
            first_ids = c.quote_start.to_numpy()[cid]; ok = ids != first_ids; reset = ids-first_ids == 1
            expected['component_id_'+p] = cid.astype('int32'); expected['completed_'+p] = ok; expected['recurrence_reset_'+p] = reset
            edges = np.r_[0, np.flatnonzero(np.diff(cid))+1]; ends = np.r_[edges[1:]-1, len(ids)-1]; unique = cid[edges]
            state['first'][unique] = np.minimum(state['first'][unique], t[edges]); state['last'][unique] = t[ends]
            state['exposure'][unique] += np.add.reduceat(np.where(ok, gap, 0), edges)
            state['count'][unique] += np.add.reduceat(ok.astype('int64'), edges)
            state['resets'] += int(reset.sum()); state['over60'] += int((ok & (gap > 60*S)).sum())
            state['legacy_bad'] += int((legacy & ~ok).sum()); state['legacy_dropped'] += int((ok & ~legacy).sum())
            values = gap[ok]/S; state['n'] += len(values); state['sum_sec'] += float(values.sum()); state['square_sec'] += float(values@values)
            if len(values): state['minimum'] = min(state['minimum'], float(values.min())); state['maximum'] = max(state['maximum'], float(values.max()))
            for month in np.unique(months):
                sel = months == month; complete = sel & ok
                monthly[(str(month),p)] += [int(sel.sum()), int(complete.sum()), int((sel & ~ok).sum()), int((sel & reset).sum()), int(gap[complete].sum())]
        compare_frames(expected, saved.take(len(times)))
        offset += len(times); previous = times.iloc[-1]
        if offset//10_000_000 != (offset-len(times))//10_000_000: print(f'Independently verified {offset:,} duration rows', flush=True)
    if not saved.exhausted(): raise AssertionError('Extra saved duration rows')
    raw = pq.ParquetFile(inputs['raw']); last_raw = pd.Timestamp(raw.read_row_group(raw.num_row_groups-1, columns=['timestamp']).column(0)[-1].as_py()).value
    tails = pd.read_parquet(run/'terminal_spells.parquet'); expected_rows = []
    for p,c in tables.items():
        state = states[p]
        for j,row in enumerate(c.itertuples(index=False)):
            first = int(state['first'][j]); last = int(state['last'][j]); hard_end = pd.Timestamp(segments.end_exclusive.iloc[row.segment_id]).value
            assert first == pd.Timestamp(row.start).value
            hard_sample = hard_end == last_raw+1; hard = last_raw if hard_sample else hard_end
            cap = None if POLICIES[p] is None else last+POLICIES[p]*S
            stop = hard if cap is None else min(hard, cap)
            original_stop = hard_end if cap is None else min(hard_end, cap+1)
            assert original_stop == pd.Timestamp(row.end_exclusive).value
            code = (1 if cap is not None and cap == stop else 0) | ((4 if hard_sample else 2) if hard == stop else 0)
            assert stop >= last and state['exposure'][j] == last-first
            assert state['count'][j] == row.quote_stop_exclusive-row.quote_start-1
            assert state['exposure'][j]+stop-last == stop-first
            expected_rows.append(dict(policy=p, component_id=j, segment_id=row.segment_id, first_quote_id=row.quote_start,
                last_quote_id=row.quote_stop_exclusive-1, completed_spells=int(state['count'][j]),
                component_start=pd.Timestamp(first, tz='UTC'), origin=pd.Timestamp(last, tz='UTC'), stop=pd.Timestamp(stop, tz='UTC'),
                exposure_ns=stop-last, exposure_sec=(stop-last)/S, event_observed=False, positive_exposure=stop > last, stop_code=code))
        if offset > 1_000_000: print(f'Verified {p}: {len(c):,} terminal records and component exposure identities', flush=True)
    compare_frames(pd.DataFrame(expected_rows), tails)
    stats = json.loads((run/'statistics.json').read_text()); assert stats['quote_rows'] == offset and stats['legacy_filter_kept'] == legacy_count
    def check_moments(actual, n, total, square, low, high):
        assert actual['count'] == n
        if n:
            np.testing.assert_allclose([actual['mean'], actual['minimum'], actual['maximum']], [total/n, low, high], rtol=1e-10, atol=1e-10)
        if n > 1: np.testing.assert_allclose(actual['sample_std'], np.sqrt((square-total*total/n)/(n-1)), rtol=1e-9, atol=1e-9)
    check_moments(stats['audit_adjacent_gap_seconds'], raw_n, raw_sum, raw_square, raw_min, raw_max)
    for p,state in states.items():
        actual = stats['policies'][p]; t = tails.loc[tails.policy.eq(p)]; ncomp = len(t)
        counts = dict(completed=state['n'], unavailable_incoming=ncomp, recurrence_resets=state['resets'],
            completed_exposure_ns=int(state['exposure'].sum()), completed_over_60s=state['over60'],
            legacy_kept_but_ineligible=state['legacy_bad'], eligible_but_legacy_dropped=state['legacy_dropped'],
            components=ncomp, single_quote_components=int((state['count']==0).sum()), terminal_records=ncomp,
            positive_terminal_exposures=int(t.positive_exposure.sum()), zero_terminal_exposures=int((~t.positive_exposure).sum()),
            terminal_exposure_ns=int(t.exposure_ns.sum()), total_observed_span_ns=int(state['exposure'].sum()+t.exposure_ns.sum()),
            stop_code_counts={str(k):int(v) for k,v in t.stop_code.value_counts().sort_index().items()})
        for k,v in counts.items(): assert actual[k] == v, (p,k,actual[k],v)
        check_moments(actual['completed_duration_seconds'], state['n'], state['sum_sec'], state['square_sec'], state['minimum'], state['maximum'])
    expected_month = pd.DataFrame([dict(zip(['month','policy','quotes','completed','unavailable_incoming','recurrence_resets','completed_exposure_ns'], [m,p,*values])) for (m,p),values in sorted(monthly.items())])
    compare_frames(expected_month, pd.read_parquet(run/'monthly_statistics.parquet'))
    return {'status':'passed', 'quote_rows_checked':offset, 'policy_row_classifications_checked':offset*4,
        'terminal_records_checked':len(tails), 'component_exposure_identities_checked':len(tails), 'monthly_rows_checked':len(expected_month),
        'reference_batch_size':333331, 'method':'Independent pandas timestamp differences, support-stop search, per-component integer exposure sums, boundary reconstruction and raw-moment/monthly reconciliation.',
        'scope':'Observation ledger only; no ACD estimation, independent censoring, feed completeness or paper readiness certified.'}

"""Independent verification of recovered wire records and monthly reconstruction.

Uses struct.iter_unpack, rather than the acquisition decoder's NumPy layout.
Original cached observations must remain exactly present and unchanged.
"""
from pathlib import Path
import json
import lzma
import struct
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data_preparation import atomic_json, sha256

COLUMNS=['timestamp','ask','bid','ask_volume','bid_volume']


def reference_decode(data, hour):
    raw=lzma.decompress(data)
    if len(raw)%20:raise AssertionError('Reference decoder: incomplete binary record')
    rows=list(struct.iter_unpack('>IIIff',raw))
    offsets=[r[0] for r in rows]
    if any(t>=3600000 for t in offsets) or offsets!=sorted(offsets):raise AssertionError('Reference decoder: bad time/order')
    base=pd.Timestamp(hour)
    return pd.DataFrame({'timestamp':pd.DatetimeIndex([base+pd.Timedelta(milliseconds=t) for t in offsets],tz='UTC'),
        'ask':pd.Series([r[1]/100000 for r in rows],dtype=float),
        'bid':pd.Series([r[2]/100000 for r in rows],dtype=float),
        'ask_volume':pd.Series([r[3] for r in rows],dtype=float),
        'bid_volume':pd.Series([r[4] for r in rows],dtype=float)})


def check_hourly_calendar(ledger):
    ny=ZoneInfo('America/New_York')
    expected=[]
    for ts in ledger.hour:
        local=ts.to_pydatetime().astimezone(ny)
        weekday=local.weekday()
        expected.append(weekday in (0,1,2,3) or (weekday==4 and local.hour<17) or (weekday==6 and local.hour>=17))
    if not np.array_equal(ledger.nominal_weekly_open,expected):raise AssertionError('Weekly calendar mismatch')


def verify_coverage_audit(acq:Path):
    """Recompute every hourly diagnostic and >60s gap from integer timestamps."""
    manifest=json.loads((acq/'manifest.json').read_text())
    ledger=pd.read_parquet(acq/'coverage_before.parquet');check_hourly_calendar(ledger)
    grid=pd.date_range(manifest['start_inclusive'],manifest['end_exclusive'],freq='h',inclusive='left')
    if not np.array_equal(ledger.hour.array.asi8,grid.asi8):raise AssertionError('Coverage ledger bounds changed')
    counts={};first={};last={};maxgap={};gap_from=[];gap_to=[];previous=None;total=0
    for item in manifest['input_files']:
        path=Path(manifest['source_dir'])/item['name']
        if sha256(path)!=item['sha256']:raise AssertionError('Audited source changed')
        ts=pd.read_parquet(path,columns=['timestamp']).timestamp
        ns=ts.astype('datetime64[ns, UTC]').array.asi8
        ns=ns[(ns>=grid[0].value)&(ns<grid[-1].value+3_600_000_000_000)]
        if not len(ns):continue
        if (np.diff(ns)<0).any():raise AssertionError('Timestamp order changed')
        hour_ns=(ns//3_600_000_000_000)*3_600_000_000_000
        unique,starts,n=np.unique(hour_ns,return_index=True,return_counts=True)
        delta=np.diff(ns,prepend=ns[0] if previous is None else previous)
        qualifying=delta>60_000_000_000
        priors=np.concatenate(([ns[0] if previous is None else previous],ns[:-1]))
        gap_from.extend(priors[qualifying].tolist());gap_to.extend(ns[qualifying].tolist())
        inner=delta.copy();inner[starts]=0
        largest=np.maximum.reduceat(inner,starts)/1e9
        for h,i,size,maximum in zip(unique,starts,n,largest):
            counts[int(h)]=int(size);first[int(h)]=int(ns[i]);last[int(h)]=int(ns[i+size-1]);maxgap[int(h)]=float(maximum) if size>1 else np.nan
        total+=len(ns);previous=int(ns[-1])
    hours=ledger.hour.array.asi8
    if not np.array_equal(ledger.rows,[counts.get(int(h),0) for h in hours]):raise AssertionError('Hourly counts differ')
    for field,lookup in [('first_timestamp',first),('last_timestamp',last)]:
        # construct datetime values without converting integer nanoseconds to
        # floats when absent hours are present.
        expected=pd.DatetimeIndex([pd.Timestamp(lookup[int(h)],unit='ns',tz='UTC') if int(h) in lookup else pd.NaT for h in hours])
        if not np.array_equal(ledger[field].array.asi8,expected.asi8):raise AssertionError(field+' differs')
    metrics={
        'max_internal_gap_sec':[maxgap.get(int(h),np.nan) for h in hours],
        'first_quote_delay_sec':[(first[int(h)]-int(h))/1e9 if int(h) in first else np.nan for h in hours],
        'last_quote_to_hour_end_sec':[(int(h)+3_600_000_000_000-last[int(h)])/1e9 if int(h) in last else np.nan for h in hours]}
    for name,values in metrics.items():
        if not np.array_equal(ledger[name],values,equal_nan=True):raise AssertionError(name+' differs')
    gaps=pd.read_parquet(acq/'gaps_before.parquet')
    if not np.array_equal(gaps.previous_timestamp.array.asi8,gap_from) or not np.array_equal(gaps.timestamp.array.asi8,gap_to):raise AssertionError('Gap endpoints differ')
    expected=(np.array(gap_to,dtype=np.int64)-np.array(gap_from,dtype=np.int64))/1e9
    if not np.array_equal(gaps.delta_sec,expected):raise AssertionError('Gap lengths differ')
    result={'status':'passed','raw_timestamps_checked':total,'hourly_rows_checked':len(ledger),
        'long_gaps_checked':len(gaps),'calendar_checked_independently':True,'verifier_sha256':sha256(Path(__file__)),
        'scope':'Complete cached-input timestamp audit; vendor completeness is not certified'}
    atomic_json(acq/'coverage_verification.json',result)
    return result


def verify_recovery(acq:Path):
    manifest=json.loads((acq/'manifest.json').read_text())
    results=json.loads((acq/'acquisition_results.json').read_text())['results']
    targets=json.loads((acq/'targets.json').read_text())['targets']
    wanted={(r['hour'],r['purpose']) for r in targets}
    if len(wanted)!=len(targets) or {(r['hour'],r['purpose']) for r in results}!=wanted or len(results)!=len(targets):
        raise AssertionError('Missing, duplicate, or extraneous acquisition outcomes')
    ledger=pd.read_parquet(acq/'coverage_before.parquet');check_hourly_calendar(ledger)
    grid=pd.date_range(manifest['start_inclusive'],manifest['end_exclusive'],freq='h',inclusive='left')
    if not np.array_equal(ledger.hour.array.asi8,grid.asi8):raise AssertionError('Coverage ledger must use requested bounds')
    absent=set(ledger.loc[ledger.nominal_weekly_open & ledger.rows.eq(0),'hour'])
    targeted={pd.Timestamp(r['hour']) for r in results if r['purpose']=='absent_nominal_open_hour'}
    if absent!=targeted:raise AssertionError('Every absent nominally open hour must have an outcome')
    decoded={};response_counts={}
    for r in results:
        h=pd.Timestamp(r['hour']);saved=json.loads((acq/'receipts'/f'{h.strftime("%Y%m%dT%H")}.json').read_text())
        if {k:v for k,v in r.items() if k!='purpose'}!=saved:raise AssertionError('Outcome differs from request receipt')
        for attempt in saved['attempts']:
            if 'response_file' in attempt and sha256(acq/'receipts'/attempt['response_file'])!=attempt['response_sha256']:
                raise AssertionError('Response bytes changed')
        body=(acq/'receipts'/r['response_file']).read_bytes()
        if r['status']=='ok':
            frame=reference_decode(body,h)
            if not len(frame)==r['rows']>0:raise AssertionError('Wire row count differs')
            decoded[r['hour']]=frame
        elif r['status']=='empty_200':
            if r['http_status']!=200 or body:raise AssertionError('Not an empty successful response')
        elif r['status']=='empty_decoded':
            if len(reference_decode(body,h)):raise AssertionError('Decoded response was not empty')
        elif r['status']=='not_found_404':
            if r['http_status']!=404:raise AssertionError('Not a 404 response')
        else:raise AssertionError('Unresolved acquisition error')
        response_counts[r['status']]=response_counts.get(r['status'],0)+1
    controls=[];old_total=0;new_total=0;added_total=0;recovered_hours=[]
    for item in manifest['input_files']:
        original=Path(manifest['source_dir'])/item['name']
        if sha256(original)!=item['sha256']:raise AssertionError('Original monthly source mutated')
        old=pd.read_parquet(original,columns=COLUMNS).reset_index(drop=True)
        _,year,month=original.stem.split('_');additions=[]
        for r in results:
            hour=pd.Timestamp(r['hour'])
            if (hour.year,hour.month)!=(int(year),int(month)) or r['status']!='ok':continue
            frame=decoded[r['hour']]
            old_hour=old.loc[(old.timestamp>=hour)&(old.timestamp<hour+pd.Timedelta(hours=1))]
            if r['purpose']=='decoder_control_existing_hour':
                # existing inputs intentionally used a lossy price-only key.
                # report whether that transform is needed to reconcile a control.
                direct=len(old_hour)==len(frame) and old_hour.reset_index(drop=True).equals(frame)
                legacy=frame.loc[(frame.ask!=0)&(frame.bid!=0)].drop_duplicates(['timestamp','ask','bid'])
                pd.testing.assert_frame_equal(old_hour.reset_index(drop=True),legacy.reset_index(drop=True),check_exact=True)
                controls.append({'hour':r['hour'],'vendor_rows':len(frame),'legacy_rows':len(old_hour),
                    'exact_full_record_match':direct,'rows_removed_by_legacy_rules':len(frame)-len(old_hour)})
            else:
                if len(old_hour):raise AssertionError('Recovery overlaps an observed hour')
                additions.append(frame);recovered_hours.append(r['hour']);added_total+=len(frame)
        expected=pd.concat([old,*additions],ignore_index=True).sort_values('timestamp',kind='stable').reset_index(drop=True)
        actual=pd.read_parquet(acq/'monthly'/item['name'],columns=COLUMNS)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        observed_hours=old.timestamp.dt.floor('h').value_counts()
        subset=ledger[(ledger.hour.dt.year==int(year))&(ledger.hour.dt.month==int(month))]
        counts=subset.hour.map(observed_hours).fillna(0).astype('int64')
        if not np.array_equal(counts,subset.rows):raise AssertionError('Original hourly ledger count mismatch')
        old_total+=len(old);new_total+=len(actual)
    if new_total!=old_total+added_total:raise AssertionError('Recovery row conservation failed')
    result={'status':'passed','scope':'All input months and recovered wire records; prespecified controls; whole-hour recovery only',
        'original_rows_preserved_exactly':old_total,'new_raw_rows':new_total,'recovered_raw_rows':added_total,
        'recovered_hours':len(recovered_hours),'recovered_hour_list':sorted(recovered_hours),
        'request_outcomes':response_counts,'control_comparisons':controls,'verifier_sha256':sha256(Path(__file__))}
    atomic_json(acq/'recovery_verification.json',result)
    return result

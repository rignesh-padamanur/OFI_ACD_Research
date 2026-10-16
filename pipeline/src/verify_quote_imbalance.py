"""Independent indicator-formula and reduce-at verification of saved quote signals."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import sha256

S=1_000_000_000
POLICIES={'structural_only':None,'fresh_15s':15,'fresh_30s':30,'fresh_60s':60}


def nanos(series):return pd.DatetimeIndex(series).as_unit('ns').asi8


class Rows:
    def __init__(self,path):
        self.iterator=pq.ParquetFile(path).iter_batches(batch_size=400_000)
        self.pending=None;self.position=0
    def take(self,n):
        pieces=[]
        while n:
            if self.pending is None or self.position==len(self.pending):
                try:self.pending=next(self.iterator).to_pandas();self.position=0
                except StopIteration:raise AssertionError('Saved output ended before reference')
            k=min(n,len(self.pending)-self.position)
            pieces.append(self.pending.iloc[self.position:self.position+k]);self.position+=k;n-=k
        return pd.concat(pieces,ignore_index=True)
    def exhausted(self):
        if self.pending is not None and self.position<len(self.pending):return False
        return next(self.iterator,None) is None


def compare_frames(expected,actual):
    if list(expected.columns)!=list(actual.columns):raise AssertionError('Output columns differ')
    for name in expected:
        if pd.api.types.is_float_dtype(expected[name]):
            np.testing.assert_allclose(actual[name],expected[name],rtol=1e-12,atol=1e-10,equal_nan=True,err_msg=name)
        else:np.testing.assert_array_equal(actual[name].to_numpy(),expected[name].to_numpy(),err_msg=name)


def reference_ticks(frame,offset,segments,previous):
    part=pd.concat([previous,frame],ignore_index=True) if previous is not None else frame.reset_index(drop=True)
    start=offset-1 if previous is not None else offset
    ids=np.arange(start,start+len(part),dtype='int64')
    sid=np.searchsorted(segments.quote_stop_exclusive.to_numpy(),ids,side='right')
    t=nanos(part.timestamp);b=part.bid.to_numpy();a=part.ask.to_numpy()
    qb=part.bid_volume.to_numpy();qa=part.ask_volume.to_numpy()
    if (np.diff(t)<=0).any():raise AssertionError('Unsupported timestamp ordering')
    has=np.arange(len(part))>0
    pb=np.r_[np.nan,b[:-1]];pa=np.r_[np.nan,a[:-1]]
    pqb=np.r_[np.nan,qb[:-1]];pqa=np.r_[np.nan,qa[:-1]]
    goodp=np.isfinite(b)&np.isfinite(a)&(b>0)&(a>b)
    goods=np.isfinite(qb)&np.isfinite(qa)&(qb>=0)&(qa>=0)
    code=np.zeros(len(part),dtype='uint8');code[~has]|=1
    code[1:][sid[1:]!=sid[:-1]]|=2
    code[~goodp]|=4;code[1:][~goodp[:-1]]|=8
    code[~goods]|=16;code[1:][~goods[:-1]]|=32
    with np.errstate(invalid='ignore'):
        bp=(b>=pb)*qb-(b<=pb)*pqb
        ap=-qa*(a<=pa)+(a>=pa)*pqa
    bp[code!=0]=np.nan;ap[code!=0]=np.nan
    pair=np.r_[False,goodp[1:]&goodp[:-1]]
    bm=np.where(pair,(b>pb).astype('int8')-(b<pb).astype('int8'),0).astype('int8')
    am=np.where(pair,(a>pa).astype('int8')-(a<pa).astype('int8'),0).astype('int8')
    result=pd.DataFrame({'quote_id':ids,'timestamp':pd.to_datetime(t,utc=True),'segment_id':sid.astype('int64'),
        'delta_sec':np.r_[np.nan,np.diff(t)/S],'exclusion_code':code,'bid_move':bm,'ask_move':am,
        'bid_pressure':bp,'ask_pressure':ap,'quote_ofi':bp+ap})
    return result.iloc[1 if previous is not None else 0:].reset_index(drop=True)


def reference_bins(ticks,support):
    buckets=nanos(ticks.timestamp)//S*S
    idx=np.r_[0,np.flatnonzero(np.diff(buckets))+1];stops=np.r_[idx[1:],len(ticks)]
    count=stops-idx
    valid=ticks.exclusion_code.to_numpy()==0
    valid_count=np.add.reduceat(valid.astype('int64'),idx)
    values=ticks.quote_ofi.to_numpy()
    subtotal=np.add.reduceat(np.nan_to_num(values,nan=0),idx)
    subtotal[valid_count==0]=np.nan
    full=subtotal.copy();full[valid_count!=count]=np.nan
    gap=np.maximum.reduceat(np.nan_to_num(ticks.delta_sec.to_numpy(),nan=-np.inf),idx)
    gap[~np.isfinite(gap)]=np.nan
    sid=ticks.segment_id.to_numpy();first=sid[idx];last=sid[stops-1]
    left=buckets[idx];right=left+S
    out=pd.DataFrame({'bin_start':pd.to_datetime(left,utc=True),'available_at':pd.to_datetime(right,utc=True),
        'n_quotes':count,'n_valid_transitions':valid_count,'max_incoming_gap_sec':gap,
        'observed_subtotal':subtotal,'n_invalid_transitions':count-valid_count,'quote_ofi':full,
        'segment_id':np.where(first==last,first,-1)})
    for policy,fresh in POLICIES.items():
        intervals=support.loc[support.policy.eq(policy)]
        a=nanos(intervals.start);b=nanos(intervals.end_exclusive)
        j=np.searchsorted(b,right,side='left');safe=np.minimum(j,len(a)-1)
        covered=(j<len(a))&(a[safe]<=left)
        out[f'full_bin_{policy}']=covered
        out[f'usable_{policy}']=covered&(valid_count==count)&(True if fresh is None else gap<=fresh)
    return out


def verify_quote_run(run):
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text())
    for name,record in manifest['inputs'].items():
        if sha256(Path(record['path']))!=record['sha256']:raise AssertionError('Input changed: '+name)
    for name,hash_value in manifest['outputs'].items():
        if sha256(run/name)!=hash_value:raise AssertionError('Output changed: '+name)
    for name,hash_value in manifest.get('code_sha256',{}).items():
        if sha256(run/'code'/name)!=hash_value:raise AssertionError('Snapshot changed: '+name)
    parent=Path(manifest['parent_clean_path']);eligibility=Path(manifest['eligibility_directory'])
    segments=pd.read_parquet(eligibility/'segments.parquet');support=pd.read_parquet(eligibility/'support_intervals.parquet')
    total=pq.ParquetFile(parent).metadata.num_rows
    assert segments.quote_start.iloc[0]==0 and segments.quote_stop_exclusive.iloc[-1]==total
    np.testing.assert_array_equal(segments.quote_start.to_numpy()[1:],segments.quote_stop_exclusive.to_numpy()[:-1])
    actual_ticks=Rows(run/'quote_ofi_ticks.parquet');actual_bins=Rows(run/'quote_ofi_1s.parquet')
    offset=0;previous=None;pending=None;nbins=0;count=Counter();codes=Counter();cases=Counter();comparison=Counter();bin_counts=Counter()
    tick_sums={p:np.zeros(3) for p in POLICIES};bin_sums={p:np.zeros(3) for p in POLICIES}
    extremes={('tick',p):[np.inf,-np.inf] for p in POLICIES};extremes.update({('bin',p):[np.inf,-np.inf] for p in POLICIES})
    monthly={}
    def moments(values,acc,extreme):
        if not len(values):return
        acc += [len(values),float(np.sum(values)),float(np.dot(values,values))]
        extreme[0]=min(extreme[0],float(values.min()));extreme[1]=max(extreme[1],float(values.max()))
    def check_bins(ready):
        nonlocal nbins
        ref=reference_bins(ready,support);saved=actual_bins.take(len(ref));compare_frames(ref,saved);nbins+=len(ref)
        bin_counts.update({'quote_containing_bins':len(ref),'quotes_accounted_for':int(ref.n_quotes.sum()),
            'bins_with_excluded_transitions':int((ref.n_invalid_transitions>0).sum()),
            'bins_with_no_valid_transition':int((ref.n_valid_transitions==0).sum()),
            'bins_with_multiple_structural_segments':int((ref.segment_id<0).sum()),
            'zero_complete_observed_sums':int((ref.quote_ofi==0).sum())})
        month=ref.bin_start.dt.year.to_numpy()*100+ref.bin_start.dt.month.to_numpy()
        for p in POLICIES:
            mask=ref[f'usable_{p}'].to_numpy();bin_counts[f'full_bin_{p}']+=int(ref[f'full_bin_{p}'].sum());bin_counts[f'usable_{p}']+=int(mask.sum())
            moments(ref.quote_ofi.to_numpy()[mask],bin_sums[p],extremes['bin',p])
            for k in np.unique(month):monthly.setdefault(f'{k//100}-{k%100:02}',Counter())[f'usable_bins_{p}']+=int((mask&(month==k)).sum())
        for k in np.unique(month):monthly.setdefault(f'{k//100}-{k%100:02}',Counter())['quote_containing_bins']+=int((month==k).sum())
    for batch in pq.ParquetFile(parent).iter_batches(batch_size=333_333,columns=['timestamp','bid','ask','bid_volume','ask_volume']):
        frame=batch.to_pandas();ref=reference_ticks(frame,offset,segments,previous)
        saved=actual_ticks.take(len(ref));compare_frames(ref,saved)
        previous=frame.tail(1).copy();offset+=len(frame)
        valid=ref.exclusion_code.to_numpy()==0;x=ref.quote_ofi.to_numpy();bm=ref.bid_move.to_numpy()[valid];am=ref.ask_move.to_numpy()[valid];flow=x[valid]
        count.update({'quotes':len(ref),'eligible_transitions':int(valid.sum()),'excluded_transitions':int((~valid).sum()),
            'sunday_quotes':int((ref.timestamp.dt.dayofweek==6).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum()),'positive':int((x>0).sum())})
        nums,freq=np.unique(ref.exclusion_code,return_counts=True)
        for k,n in zip(nums,freq):codes[str(k)]+=int(n)
        for b in (-1,0,1):
            for a in (-1,0,1):cases[f'bid_{b}_ask_{a}']+=int(((bm==b)&(am==a)).sum())
        old=np.sign(bm-am);new=np.sign(flow)
        comparison.update({'both_zero':int(((old==0)&(new==0)).sum()),'legacy_zero_corrected_nonzero':int(((old==0)&(new!=0)).sum()),
            'legacy_nonzero_corrected_zero':int(((old!=0)&(new==0)).sum()),'both_nonzero_same_sign':int(((old!=0)&(new!=0)&(old==new)).sum()),
            'both_nonzero_opposite_sign':int(((old!=0)&(new!=0)&(old!=new)).sum()),'unchanged_prices_nonzero_size_flow':int(((bm==0)&(am==0)&(new!=0)).sum())})
        for p,fresh in POLICIES.items():
            mask=valid if fresh is None else valid&(ref.delta_sec.to_numpy()<=fresh)
            moments(x[mask],tick_sums[p],extremes['tick',p])
        month=ref.timestamp.dt.year.to_numpy()*100+ref.timestamp.dt.month.to_numpy()
        for k in np.unique(month):
            mask=month==k;monthly.setdefault(f'{k//100}-{k%100:02}',Counter()).update({'quotes':int(mask.sum()),
                'eligible_transitions':int((valid&mask).sum()),'excluded_transitions':int((~valid&mask).sum()),
                'positive_tick_ofi':int(((x>0)&mask).sum()),'negative_tick_ofi':int(((x<0)&mask).sum()),'zero_tick_ofi':int(((x==0)&mask).sum())})
        combo=pd.concat([pending,ref],ignore_index=True) if pending is not None else ref
        times=nanos(combo.timestamp);cut=int(np.searchsorted(times,times[-1]//S*S,side='left'))
        if cut:check_bins(combo.iloc[:cut])
        pending=combo.iloc[cut:].copy()
        if offset%(333_333*30)==0:print(f'Independently verified {offset:,} quote rows',flush=True)
    if pending is not None and len(pending):check_bins(pending)
    assert offset==total and actual_ticks.exhausted() and actual_bins.exhausted()
    stats=json.loads((run/'statistics.json').read_text())
    for name,expected in [('ticks',count),('transition_exclusion_codes',codes),('price_move_cases',cases),
            ('legacy_sign_comparison_on_common_transitions',comparison),('bins',bin_counts)]:
        assert stats[name]==dict(expected),(name,stats[name],dict(expected))
    assert stats['monthly']=={k:dict(v) for k,v in sorted(monthly.items())}
    expected_monthly=pd.DataFrame([{'month':k,**dict(v)} for k,v in sorted(monthly.items())])
    pd.testing.assert_frame_equal(expected_monthly.sort_index(axis=1),
        pd.read_parquet(run/'monthly_statistics.parquet').sort_index(axis=1))
    for kind,accs,name in [('tick',tick_sums,'tick_ofi_descriptive_by_gap_policy'),('bin',bin_sums,'bin_ofi_descriptive_by_policy')]:
        for p,(n,total_value,squares) in accs.items():
            saved=stats[name][p];assert saved['count']==int(n)
            if n:
                np.testing.assert_allclose(saved['mean'],total_value/n,rtol=1e-11,atol=1e-10)
                np.testing.assert_allclose([saved['minimum'],saved['maximum']],extremes[kind,p],rtol=0,atol=0)
            if n>1:np.testing.assert_allclose(saved['sample_std'],np.sqrt(max(0,(squares-total_value**2/n)/(n-1))),rtol=1e-11,atol=1e-10)
            if n==0:assert all(saved[k] is None for k in ['mean','sample_std','minimum','maximum'])
            if n==1:assert saved['sample_std'] is None
    return {'status':'passed','quote_rows_checked':offset,'one_second_bins_checked':nbins,
        'formula':'independent indicator expression; production uses price-change cases',
        'aggregation':'independent numpy reduce-at; production uses pandas groupby',
        'verification_batch_size':333333,'all_counts_monthly_tables_and_moments_checked':True,
        'float_comparison_rtol':1e-12,'float_comparison_atol':1e-10,'coverage_certified':False,
        'scope':'Recorded-quote proxy arithmetic, missingness, boundaries and bin availability; no inference or feed-completeness claim'}

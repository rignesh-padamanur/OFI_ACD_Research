"""Past-only origin-time duration scale and a fixed chronological audit sample."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.observation_eligibility import POLICIES

SECOND=1_000_000_000
DAY=86400*SECOND
SLOT=900*SECOND


@dataclass(frozen=True)
class DurationAuditSpec:
    seasonal_days:int=56
    training_days:int=56
    validation_days:int=14
    min_prior_dates:int=4
    min_exposure_seconds:int=1800
    min_completions:int=100

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in asdict(self).values()):
            raise ValueError('Positive integer reference settings required')
        if self.seasonal_days<7 or self.min_prior_dates>self.seasonal_days//7:
            raise ValueError('Insufficient prior same-weekday dates')

    @property
    def total_days(self):return self.seasonal_days+self.training_days+self.validation_days


def ns(x):return pd.DatetimeIndex(x).as_unit('ns').asi8


def source_spells(path, terminals, *, policy, stop_ns):
    """One outgoing observed spell per quote before stop; no future endpoint event."""
    if policy not in POLICIES:raise ValueError('Explicit policy required')
    chunks=[];column='component_id_'+policy
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000,columns=['quote_id','available_at',column]):
        f=batch.to_pandas();t=ns(f.available_at);k=np.searchsorted(t,stop_ns)
        chunks.append(f.iloc[:min(len(f),k+1)])
        if k<len(f):break
    f=pd.concat(chunks,ignore_index=True);t=ns(f.available_at);n=int(np.searchsorted(t,stop_ns))
    if not n:raise ValueError('Empty chronological audit prefix')
    ids=f.quote_id.to_numpy();cid=f[column].to_numpy()
    np.testing.assert_array_equal(ids,np.arange(len(f)))
    origin=t[:n];end=np.full(n,stop_ns,dtype='int64');event=np.zeros(n,dtype=bool);code=np.full(n,8,dtype='uint8')
    has_next=np.arange(n)+1<len(f);same=np.zeros(n,dtype=bool)
    same[has_next]=cid[:n][has_next]==cid[np.arange(n)[has_next]+1]
    next_times=t[np.arange(n)[same]+1]
    end[same]=np.minimum(next_times,stop_ns);event[same]=next_times<stop_ns;code[same & event]=0
    positions=np.flatnonzero(~same);tails=terminals.loc[terminals.policy.eq(policy)].sort_values('last_quote_id')
    j=np.searchsorted(tails.last_quote_id.to_numpy(),ids[positions])
    if (j>=len(tails)).any() or not np.array_equal(tails.last_quote_id.to_numpy()[j],ids[positions]):
        raise ValueError('Unaccounted terminal spell')
    ends=ns(tails.stop)[j];end[positions]=np.minimum(ends,stop_ns)
    code[positions]=np.where(ends<stop_ns,tails.stop_code.to_numpy()[j],8)
    if (end<origin).any():raise ValueError('Negative observed exposure')
    return pd.DataFrame({'origin_quote_id':ids[:n],'component_id':cid[:n],
        'origin_at':pd.to_datetime(origin,utc=True),'end_at':pd.to_datetime(end,utc=True),
        'event_observed':event,'stop_code':code,'exposure_ns':end-origin})


def sufficient_statistics(spells, origin_ns, ndays):
    """Origin-day cells plus explicit corrections for every future daily cutoff.

    Full-spell totals are audit records. fit_scales subtracts the part unavailable
    at a historical cutoff before using them. Neither totals nor corrections are
    market-state covariates.
    """
    a=ns(spells.origin_at);b=ns(spells.end_at);event=spells.event_observed.to_numpy()
    day=(a-origin_ns)//DAY;cell=(a%DAY)//SLOT;index=day*96+cell
    if (day<0).any() or (day>=ndays).any() or (b<a).any():raise ValueError('Invalid calendar domain')
    counts=np.bincount(index,weights=event.astype('int64'),minlength=ndays*96).astype('int64').reshape(ndays,96)
    exposure=np.zeros(ndays*96,dtype='int64');np.add.at(exposure,index,b-a);exposure=exposure.reshape(ndays,96)
    origins=np.bincount(index,minlength=ndays*96).reshape(ndays,96)
    corrections=[]
    for i in np.flatnonzero(b>=(origin_ns+(day+1)*DAY)):
        for cutoff in range(int(day[i])+1,min(ndays-1,int((b[i]-origin_ns)//DAY))+1):
            corrections.append((cutoff,int(day[i]),int(cell[i]),int(b[i]-(origin_ns+cutoff*DAY)),int(event[i])))
    corr=pd.DataFrame(corrections,columns=['cutoff_day','origin_day','slot','future_exposure_ns','future_completions'])
    if len(corr):corr=corr.groupby(['cutoff_day','origin_day','slot'],as_index=False,sort=True).sum()
    dates=np.repeat(np.arange(ndays),96);slots=np.tile(np.arange(96),ndays)
    table=pd.DataFrame({'origin_day':dates,'slot':slots,'origins':origins.ravel(),
        'completed_events':counts.ravel(),'full_exposure_ns':exposure.ravel()})
    return table,corr


def fit_scales(table, corrections, origin_ns, spec):
    ndays=spec.total_days
    counts=table.completed_events.to_numpy().reshape(ndays,96)
    exposure=table.full_exposure_ns.to_numpy().reshape(ndays,96)
    output=[];scales=np.full((ndays,96),np.nan);statuses=np.zeros((ndays,96),dtype='uint8')
    for d in range(ndays):
        begin=max(0,d-spec.seasonal_days);dates=np.arange(begin,d)
        c=counts[begin:d].copy();e=exposure[begin:d].copy()
        correction=corrections.loc[(corrections.cutoff_day==d)&(corrections.origin_day>=begin)]
        if len(correction):
            ii=correction.origin_day.to_numpy(dtype='int64')-begin;jj=correction.slot.to_numpy(dtype='int64')
            np.add.at(c,(ii,jj),-correction.future_completions.to_numpy(dtype='int64'))
            np.add.at(e,(ii,jj),-correction.future_exposure_ns.to_numpy(dtype='int64'))
        if (c<0).any() or (e<0).any():raise AssertionError('Invalid chronological correction')
        match=(d-dates)%7==0;c=c[match];e=e[match]
        nc=c.sum(axis=0);ex=e.sum(axis=0);prior=(e>0).sum(axis=0)
        scale=np.divide(ex/SECOND,nc,out=np.full(96,np.nan),where=nc>0)
        status=np.full(96,5,dtype='uint8');status[~np.isfinite(scale)|(scale<=0)]=4
        status[nc<spec.min_completions]=3;status[ex<spec.min_exposure_seconds*SECOND]=2
        status[prior<spec.min_prior_dates]=1
        if d<spec.seasonal_days:status[:]=0
        scales[d]=np.where(status==5,scale,np.nan);statuses[d]=status
        output.append(pd.DataFrame({'effective_date':pd.Timestamp(origin_ns+d*DAY,tz='UTC'),
            'training_start':pd.Timestamp(origin_ns+(d-spec.seasonal_days)*DAY,tz='UTC'),
            'training_end_exclusive':pd.Timestamp(origin_ns+d*DAY,tz='UTC'),'slot':np.arange(96),
            'prior_dates':prior,'completed_events':nc,'observed_exposure_ns':ex,
            'working_scale_sec':scales[d],'status':status}))
    return pd.concat(output,ignore_index=True),scales,statuses


def apply_scales(spells, scales, statuses, origin_ns):
    out=spells.copy();a=ns(out.origin_at);d=(a-origin_ns)//DAY;cell=(a%DAY)//SLOT
    out['seasonal_status']=statuses[d,cell];out['scale_sec']=scales[d,cell]
    out['normalized_exposure']=out.exposure_ns.to_numpy()/SECOND/out.scale_sec.to_numpy()
    out['seasonal_cutoff']=pd.to_datetime(origin_ns+d*DAY,utc=True)
    return out


def model_sample(frame, *, start_ns, stop_ns, omit_structural_tails=False):
    """Training/validation origins in [start,stop), censor all unfinished waits.

    Reset after every unavailable factor, audit gap or component change. Data
    thinning must not reconnect unrelated waits. Missing origins stay in saved
    seasonal artifacts; only this explicit likelihood view excludes them.
    """
    a=ns(frame.origin_at);b=ns(frame.end_at);event=frame.event_observed.to_numpy()
    mask=(a>=start_ns)&(a<stop_ns)&np.isfinite(frame.scale_sec)&(frame.exposure_ns>0)
    if omit_structural_tails:mask&=~((~event)&(b<stop_ns)&((frame.stop_code.to_numpy()&2)!=0))
    idx=np.flatnonzero(mask);out=frame.iloc[idx].copy().reset_index(drop=True)
    end=np.minimum(b[idx],stop_ns);observed=event[idx]&(b[idx]<stop_ns)
    out['end_at']=pd.to_datetime(end,utc=True);out['event_observed']=observed
    out['stop_code']=np.where(b[idx]>=stop_ns,8,out.stop_code).astype('uint8')
    out['exposure_ns']=end-a[idx]
    out['normalized_exposure']=(end-a[idx])/SECOND/out.scale_sec.to_numpy()
    cid=out.component_id.to_numpy();qid=out.origin_quote_id.to_numpy();reset=np.ones(len(out),dtype=bool)
    if len(out)>1:reset[1:]=(cid[1:]!=cid[:-1])|(qid[1:]!=qid[:-1]+1)|(~observed[:-1])|(a[idx][1:]!=end[:-1])
    out['recurrence_reset']=reset
    return out

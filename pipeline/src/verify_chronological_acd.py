"""Independent incoming-spell construction, historical refits and SciPy filtering."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lfilter
from scipy.special import gamma
from scipy.stats import weibull_min
from src.data_preparation import sha256
from src.verify_quote_imbalance import compare_frames

S=1_000_000_000;DAY=86400*S
POLICIES=['structural_only','fresh_15s','fresh_30s','fresh_60s']
EDGES=np.r_[0.,-np.log1p(-np.array([.01,.05,.1,.5,.9,.99])),np.inf]


def ns(x):return pd.DatetimeIndex(x).as_unit('ns').asi8


def reference_spells(path,terminals,policy,stop):
    pieces=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=333_331,columns=['quote_id','available_at','audit_gap_ns','component_id_'+policy,'completed_'+policy]):
        f=batch.to_pandas();b=ns(f.available_at);a=b-f.audit_gap_ns.to_numpy()
        eligible=f['completed_'+policy].to_numpy()&(a<stop)
        pieces.append(pd.DataFrame({'origin_quote_id':f.quote_id.to_numpy()[eligible]-1,
            'component_id':f['component_id_'+policy].to_numpy()[eligible],
            'origin_at':pd.to_datetime(a[eligible],utc=True),'end_at':pd.to_datetime(np.minimum(b[eligible],stop),utc=True),
            'event_observed':b[eligible]<stop,'stop_code':np.where(b[eligible]<stop,0,8).astype('uint8'),
            'exposure_ns':np.minimum(b[eligible],stop)-a[eligible]}))
        if b[-1]>=stop:break
    t=terminals.loc[terminals.policy.eq(policy)&(ns(terminals.origin)<stop)]
    a=ns(t.origin);b=ns(t.stop)
    pieces.append(pd.DataFrame({'origin_quote_id':t.last_quote_id.to_numpy(),'component_id':t.component_id.to_numpy(),
        'origin_at':t.origin.to_numpy(),'end_at':pd.to_datetime(np.minimum(b,stop),utc=True),'event_observed':False,
        'stop_code':np.where(b<stop,t.stop_code.to_numpy(),8).astype('uint8'),'exposure_ns':np.minimum(b,stop)-a}))
    out=pd.concat(pieces,ignore_index=True).sort_values('origin_quote_id').reset_index(drop=True)
    np.testing.assert_array_equal(out.origin_quote_id,np.arange(len(out)))
    return out


def reference_scales(spells,origin,config):
    a=ns(spells.origin_at);b=ns(spells.end_at);event=spells.event_observed.to_numpy();cell=(a%DAY)//(900*S)
    ndays=config['seasonal_days']+config['training_days']+config['validation_days'];out=[]
    for d in range(ndays):
        dates=list(range(d-7,max(-1,d-config['seasonal_days']-1),-7))
        events=np.zeros(96,dtype='int64');expo=np.zeros(96,dtype='int64');prior=np.zeros(96,dtype='int64')
        for date in dates:
            lo,hi=np.searchsorted(a,[origin+date*DAY,origin+(date+1)*DAY]);slots=cell[lo:hi]
            elapsed=np.minimum(b[lo:hi],origin+d*DAY)-a[lo:hi]
            exact=np.zeros(96,dtype='int64');np.add.at(exact,slots,elapsed)
            expo+=exact;prior+=exact>0
            events+=np.bincount(slots,weights=(event[lo:hi]&(b[lo:hi]<origin+d*DAY)).astype('int64'),minlength=96).astype('int64')
        scale=np.divide(expo/S,events,out=np.full(96,np.nan),where=events>0)
        status=np.full(96,5,dtype='uint8')
        for j in range(96):
            if d<config['seasonal_days']:status[j]=0
            elif prior[j]<config['min_prior_dates']:status[j]=1
            elif expo[j]<config['min_exposure_seconds']*S:status[j]=2
            elif events[j]<config['min_completions']:status[j]=3
            elif not np.isfinite(scale[j]) or scale[j]<=0:status[j]=4
        scale[status!=5]=np.nan
        out.append(pd.DataFrame({'effective_date':pd.Timestamp(origin+d*DAY,tz='UTC'),
            'training_start':pd.Timestamp(origin+(d-config['seasonal_days'])*DAY,tz='UTC'),
            'training_end_exclusive':pd.Timestamp(origin+d*DAY,tz='UTC'),'slot':np.arange(96),'prior_dates':prior,
            'completed_events':events,'observed_exposure_ns':expo,'working_scale_sec':scale,'status':status}))
    return pd.concat(out,ignore_index=True)


def reference_sample(frame,start,stop,omit=False):
    a=ns(frame.origin_at);b=ns(frame.end_at);event=frame.event_observed.to_numpy()
    keep=(a>=start)&(a<stop)&frame.scale_sec.notna().to_numpy()&(b>a)
    f=frame.loc[keep].copy().reset_index(drop=True);old_end=ns(f.end_at);end=np.minimum(old_end,stop)
    f['end_at']=pd.to_datetime(end,utc=True);f['event_observed']=f.event_observed.to_numpy()&(old_end<stop)
    f['stop_code']=np.where(old_end>=stop,8,f.stop_code).astype('uint8');f['exposure_ns']=end-ns(f.origin_at)
    # apply sensitivity selection to the already historically clipped view.
    if omit:f=f.loc[f.event_observed | ((f.stop_code & 2)==0)].reset_index(drop=True)
    f['normalized_exposure']=f.exposure_ns.to_numpy()/S/f.scale_sec.to_numpy()
    previous=f.shift(1)
    contiguous=f.component_id.eq(previous.component_id)&f.origin_quote_id.eq(previous.origin_quote_id+1)&f.origin_at.eq(previous.end_at)&previous.event_observed.eq(True)
    f['recurrence_reset']=~contiguous
    return f


def reference_predictions(frame,params,*,weibull=False,unit_init=False):
    y=frame.normalized_exposure.to_numpy();r=frame.recurrence_reset.to_numpy();event=frame.event_observed.to_numpy()
    omega,alpha,beta,k=params;p=np.empty(len(frame));starts=np.flatnonzero(r);ends=np.r_[starts[1:],len(frame)]
    for a,b in zip(starts,ends,strict=True):
        p[a]=1. if unit_init else omega/(1-alpha-beta)
        if b>a+1:p[a+1:b]=lfilter([1.],[1.,-beta],omega+alpha*y[a:b-1],zi=[beta*p[a]])[0]
    if weibull:
        scale=p/gamma(1+1/k);h=-weibull_min.logsf(y,k,scale=scale)
        nll=np.where(event,-weibull_min.logpdf(y,k,scale=scale),h)
    else:h=y/p;nll=h+event*np.log(p)
    return pd.DataFrame({'origin_quote_id':frame.origin_quote_id.to_numpy(),'psi':p,'cumulative_hazard':h,
        'negative_log_score_seconds':nll+event*np.log(frame.scale_sec.to_numpy()),'martingale_residual':event-h})


def check_diagnostics(frame,pred,saved):
    h=pred.cumulative_hazard.to_numpy();e=frame.event_observed.to_numpy();res=e-h
    assert saved['spells']==len(frame) and saved['completed_events']==int(e.sum())
    assert saved['incomplete_positive_exposures']==int((~e).sum())
    assert saved['sequence_resets']==int(frame.recurrence_reset.sum()) and saved['sequence_fragments']==int(1+frame.recurrence_reset.iloc[1:].sum())
    np.testing.assert_allclose([saved['total_hazard'],saved['event_hazard_ratio'],saved['martingale_mean'],saved['negative_log_score_seconds_mean']],
        [h.sum(),e.sum()/h.sum(),res.mean(),pred.negative_log_score_seconds.mean()],rtol=1e-10,atol=1e-10)
    pair=~frame.recurrence_reset.to_numpy()[1:];assert saved['adjacent_pairs']==int(pair.sum())
    corr=float(pd.Series(res[:-1][pair]).corr(pd.Series(res[1:][pair]))) if pair.sum()>1 else None
    if corr is not None and not np.isfinite(corr):corr=None
    if corr is None:assert saved['martingale_lag1_correlation'] is None
    else:np.testing.assert_allclose(saved['martingale_lag1_correlation'],corr,atol=1e-10)
    flags=[]
    if not .9<=e.sum()/h.sum()<=1.1:flags.append('total_hazard_calibration')
    if corr is None or abs(corr)>.05:flags.append('martingale_lag1_dependence')
    flag_band=False
    for i,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:],strict=True)):
        risk=np.clip(h-lo,0,hi-lo).sum();n=int((e&(h>=lo)&(h<hi)).sum());b=saved['hazard_bands'][i]
        assert b['events']==n
        np.testing.assert_allclose(b['hazard_exposure'],risk,rtol=1e-10,atol=1e-10)
        if risk>0:np.testing.assert_allclose(b['event_exposure_ratio'],n/risk,rtol=1e-10,atol=1e-10)
        else:assert b['event_exposure_ratio'] is None
        if risk>=100 and not .8<=n/risk<=1.2:flag_band=True
    if flag_band:flags.append('hazard_band_calibration')
    assert saved['diagnostic_flags']==flags and saved['reference_diagnostic_screen_passed']==(not flags)


def verify_acd_run(run):
    run=Path(run);m=json.loads((run/'manifest.json').read_text());s=json.loads((run/'statistics.json').read_text())
    for name,rec in m['inputs'].items():
        if sha256(Path(rec['path']))!=rec['sha256']:raise AssertionError('Input changed: '+name)
    for name,h in m['outputs'].items():
        if sha256(run/name)!=h:raise AssertionError('Output changed: '+name)
    for name,h in m['code_sha256'].items():
        if sha256(run/'code'/name)!=h:raise AssertionError('Code snapshot changed: '+name)
    origin=pd.Timestamp(m['calendar_origin']).value;cfg=m['configuration'];start=origin+cfg['seasonal_days']*DAY
    cutoff=start+cfg['training_days']*DAY;stop=cutoff+cfg['validation_days']*DAY
    tails=pd.read_parquet(m['inputs']['terminal_spells']['path']);count=0;scale_rows=0;pred_rows=0;model_count=0
    for policy in POLICIES:
        d=run/policy;raw=reference_spells(m['inputs']['quote_durations']['path'],tails,policy,stop)
        saved=pd.read_parquet(d/'seasonal_spells.parquet');compare_frames(raw,saved[list(raw.columns)])
        grouped=raw.assign(origin_day=(ns(raw.origin_at)-origin)//DAY,slot=(ns(raw.origin_at)%DAY)//(900*S))
        full_index=pd.MultiIndex.from_product([range(cfg['seasonal_days']+cfg['training_days']+cfg['validation_days']),range(96)],names=['origin_day','slot'])
        moments=grouped.groupby(['origin_day','slot']).agg(origins=('origin_quote_id','size'),completed_events=('event_observed','sum'),full_exposure_ns=('exposure_ns','sum')).reindex(full_index,fill_value=0).reset_index()
        compare_frames(moments,pd.read_parquet(d/'origin_daily_sufficient_statistics.parquet'))
        scales=reference_scales(raw,origin,cfg);compare_frames(scales,pd.read_parquet(d/'seasonal_parameters.parquet'))
        day=(ns(raw.origin_at)-origin)//DAY;slot=(ns(raw.origin_at)%DAY)//(900*S);key=day*96+slot
        np.testing.assert_allclose(saved.scale_sec,scales.working_scale_sec.to_numpy()[key],equal_nan=True,rtol=1e-12)
        np.testing.assert_array_equal(saved.seasonal_status,scales.status.to_numpy()[key])
        np.testing.assert_array_equal(ns(saved.seasonal_cutoff),origin+day*DAY)
        np.testing.assert_allclose(saved.normalized_exposure,raw.exposure_ns.to_numpy()/S/saved.scale_sec,equal_nan=True,rtol=1e-12)
        ps=s['policies'][policy];assert ps['source_spell_rows']==len(raw)
        assert ps['source_completed_events']==int(raw.event_observed.sum())
        assert ps['source_exposure_ns']==int(raw.exposure_ns.sum())
        assert ps['seasonal_ready_spell_rows']==int(saved.scale_sec.notna().sum())
        assert ps['seasonal_unavailable_spell_rows']==int(saved.scale_sec.isna().sum())
        assert ps['seasonal_status_counts']=={str(k):int(v) for k,v in saved.seasonal_status.value_counts().sort_index().items()}
        assert ps['smallest_completed_duration_sec']==float(raw.loc[raw.event_observed,'exposure_ns'].min()/S)
        for name,item in ps['models'].items():
            if item['status']!='fitted_reference':continue
            omit=item['omit_structural_training_tails'];unit=item['unit_initialization'];weibull=item['distribution']=='weibull'
            training=reference_sample(saved,start,cutoff,omit);path=reference_sample(saved,start,stop)
            params=np.array([item['parameters'][x] for x in ['omega','alpha','beta','shape']])
            trainpred=reference_predictions(training,params,weibull=weibull,unit_init=unit)
            objective=float((trainpred.negative_log_score_seconds-training.event_observed*np.log(training.scale_sec)).mean())
            np.testing.assert_allclose(item['training_objective_normalized'],objective,rtol=1e-9,atol=1e-9)
            assert item['training_spells']==len(training) and item['training_events']==int(training.event_observed.sum())
            prediction=reference_predictions(path,params,weibull=weibull,unit_init=unit)
            mask=ns(path.origin_at)>=cutoff;validation=path.loc[mask].reset_index(drop=True);vp=prediction.loc[mask].reset_index(drop=True)
            compare_frames(validation,pd.read_parquet(d/'validation_spells.parquet'))
            compare_frames(vp,pd.read_parquet(d/(name+'_validation_predictions.parquet')))
            check_diagnostics(validation,vp,item['validation'])
            daily=pd.DataFrame({'date':pd.DatetimeIndex(validation.origin_at).normalize(),'event':validation.event_observed.to_numpy().astype('int64'),
                'hazard':vp.cumulative_hazard.to_numpy(),'score':vp.negative_log_score_seconds.to_numpy(),'residual':vp.martingale_residual.to_numpy()})
            rows=[]
            for date,g in daily.groupby('date',sort=True):rows.append({'date':date,'spells':len(g),'completed_events':int(g.event.sum()),
                'total_hazard':g.hazard.sum(),'mean_negative_log_score':g.score.mean(),'martingale_mean':g.residual.mean()})
            compare_frames(pd.DataFrame(rows),pd.read_parquet(d/(name+'_daily_validation.parquet')))
            pred_rows+=len(vp);model_count+=1
        count+=len(raw);scale_rows+=len(scales)
        print(f'Independent audit passed: {policy}, {len(raw):,} spells and {len(scales):,} seasonal cells',flush=True)
    assert s['fitted_reference_models']==model_count
    return {'status':'passed','policy_spell_rows_checked':count,'seasonal_parameter_rows_checked':scale_rows,
        'validation_prediction_rows_checked':pred_rows,'fitted_models_checked':model_count,
        'scope':'Fixed initial chronological audit only; implementation verification does not certify model adequacy or economic identification.',
        'methods':['incoming completed durations plus separate terminal rows','direct prior-spell exposure clipping and same-weekday refits',
            'SciPy linear filtering and distribution logpdf/logsf','independent aggregate and calendar-day likelihood/hazard diagnostic reconstruction']}

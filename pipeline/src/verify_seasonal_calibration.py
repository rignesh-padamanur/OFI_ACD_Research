"""Independent sequential reference: pandas cell moments and pairwise pooled variance."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import sha256
from src.verify_quote_imbalance import Rows

P = ('structural_only','fresh_15s','fresh_30s','fresh_60s')
V = ('tick_count','abs_quote_ofi')
LABELS = ('full_window_warmup','too_few_prior_dates','too_few_prior_seconds',
          'nonpositive_or_unavailable_mean','nonpositive_or_unavailable_scale','ready')
S = 1_000_000_000; D = 86400*S


def check_frame(expected, actual):
    if list(expected.columns) != list(actual.columns):
        raise AssertionError('Saved table columns differ')
    for c in expected:
        if pd.api.types.is_numeric_dtype(expected[c]) and pd.api.types.is_float_dtype(expected[c]):
            np.testing.assert_allclose(actual[c],expected[c],rtol=1e-10,atol=1e-9,equal_nan=True,err_msg=c)
        else:
            np.testing.assert_array_equal(actual[c].to_numpy(),expected[c].to_numpy(),err_msg=c)


def raw_channels(frame, policy):
    status = frame[f'flow_status_{policy}'].to_numpy()
    count = np.where(status>0,frame.recorded_update_count.to_numpy(),np.nan)
    absolute = np.full(len(frame),np.nan)
    absolute[status==1] = 0.
    absolute[status==2] = np.abs(frame.observed_quote_ofi.to_numpy()[status==2])
    return count, absolute


def reference_daily(frame):
    slot = frame.bin_start.dt.hour.to_numpy()*4 + frame.bin_start.dt.minute.to_numpy()//15
    out = np.zeros((4,2,96,3))
    for pi,p in enumerate(P):
        for vi,x in enumerate(raw_channels(frame,p)):
            grouped = pd.DataFrame({'slot':slot,'x':x}).groupby('slot').x.agg(['count','mean','var'])
            for c,r in grouped.iterrows():
                n = int(r['count'])
                out[pi,vi,int(c)] = [n,float(r['mean']) if n else 0.,float(r['var'])*(n-1) if n>1 else 0.]
    return out


def reference_fit(history, index, spec):
    # Select prior calendar days explicitly; the current/future indices are never read.
    past = list(range(max(0,index-spec.lookback_days),index))
    matched = [j for j in past if (index-j)%7==0]
    shape = (4,2,96)
    result = {k:np.full(shape,np.nan) for k in ['n','prior_dates','mean','sample_std','global_mean','phi','status']}
    result['n'] = np.zeros(shape,dtype='int64'); result['prior_dates'] = np.zeros(shape,dtype='int16')
    result['status'] = np.zeros(shape,dtype='uint8')
    for pi in range(4):
        for vi in range(2):
            nall = sum(float(history[j][pi,vi,:,0].sum()) for j in past)
            sall = sum(float(np.dot(history[j][pi,vi,:,0],history[j][pi,vi,:,1])) for j in past)
            glob = sall/nall if nall else np.nan
            result['global_mean'][pi,vi] = glob
            values = np.array([history[j][pi,vi] for j in matched]).reshape((-1,96,3))
            weights = values[...,0]; means = values[...,1]
            n = weights.sum(axis=0); dates = (weights>0).sum(axis=0)
            mu = np.divide((weights*means).sum(axis=0),n,out=np.full(96,np.nan),where=n>0)
            between = np.zeros(96)
            for j in range(len(values)):
                for k in range(j):
                    between += weights[j]*weights[k]*np.square(means[j]-means[k])
            pooled = values[...,2].sum(axis=0)+np.divide(between,n,out=np.zeros(96),where=n>0)
            sd = np.sqrt(np.divide(pooled,n-1,out=np.full(96,np.nan),where=n>1))
            for c in range(96):
                if index<spec.lookback_days: status=0
                elif dates[c]<spec.min_prior_dates: status=1
                elif n[c]<spec.min_prior_seconds: status=2
                elif not np.isfinite(mu[c]) or mu[c]<=0 or not np.isfinite(glob) or glob<=0: status=3
                elif not np.isfinite(sd[c]) or sd[c]<=0: status=4
                else: status=5
                result['status'][pi,vi,c] = status
                if status==5: result['phi'][pi,vi,c] = mu[c]/glob
            result['n'][pi,vi]=n;result['prior_dates'][pi,vi]=dates
            result['mean'][pi,vi]=mu;result['sample_std'][pi,vi]=sd
    return result


def reference_output(frame, fit):
    slot = frame.bin_start.dt.hour.to_numpy()*4 + frame.bin_start.dt.minute.to_numpy()//15
    out = frame[['bin_start','available_at']].reset_index(drop=True).copy()
    for pi,p in enumerate(P):
        for vi,x in enumerate(raw_channels(frame,p)):
            z = np.full(len(frame),np.nan)
            ok = (fit['status'][pi,vi,slot]==5)&np.isfinite(x)
            z[ok] = (x[ok]-fit['mean'][pi,vi,slot[ok]])/fit['sample_std'][pi,vi,slot[ok]]
            out[f'z_{V[vi]}_{p}'] = z
    return out


def reference_tables(day, daily, fit, spec):
    a=[];b=[]
    for pi,p in enumerate(P):
        for vi,v in enumerate(V):
            for c in range(96):
                n,mean,m2=daily[pi,vi,c]
                a.append({'date':day,'policy':p,'channel':v,'slot':c,'n':int(n),'mean':mean if n else np.nan,'m2':m2})
                b.append({'effective_date':day,'training_start':day-pd.Timedelta(days=spec.lookback_days),
                    'training_end_exclusive':day,'policy':p,'channel':v,'slot':c,
                    **{k:fit[k][pi,vi,c].item() for k in fit}})
    return pd.DataFrame(a),pd.DataFrame(b)


class ReferenceStatistics:
    def __init__(self):
        self.rows=0;self.days=0;self.counts={};self.acc={};self.first={};self.status={};self.month={};self.cell={}
        for p in P:
            for v in V:
                self.counts[p,v]=Counter();self.acc[p,v]=np.array([0.,0.,0.,np.inf,-np.inf])
                self.first[p,v]=None;self.status[p,v]=Counter()

    def add(self,day,frame,output,fit):
        self.rows+=len(frame);self.days+=1
        group=frame.bin_start.dt.hour.to_numpy()*4+frame.bin_start.dt.minute.to_numpy()//15
        for pi,p in enumerate(P):
            for vi,x in enumerate(raw_channels(frame,p)):
                v=V[vi];valid_input=np.isfinite(x);c=fit['status'][pi,vi,group]
                z=output[f'z_{v}_{p}'].to_numpy();ok=np.isfinite(z);values=z[ok]
                counts={'clock_seconds':len(frame),'input_eligible_seconds':int(valid_input.sum()),
                        'unavailable_input_seconds':int((~valid_input).sum()),'calibrated_seconds':int(ok.sum())}
                for k,label in enumerate(LABELS):
                    counts['eligible_'+label]=int(np.count_nonzero(valid_input & (c==k)))
                    self.status[p,v][label]+=int(np.count_nonzero(fit['status'][pi,vi]==k))
                self.counts[p,v].update(counts)
                acc=self.acc[p,v]
                if len(values):
                    acc[:3]+=[len(values),values.sum(),np.square(values).sum()]
                    acc[3]=min(acc[3],values.min());acc[4]=max(acc[4],values.max())
                    if self.first[p,v] is None:self.first[p,v]=frame.available_at.iloc[int(np.flatnonzero(ok)[0])].isoformat()
                key=(day.strftime('%Y-%m'),p,v)
                mc,ma=self.month.setdefault(key,(Counter(),np.zeros(3)))
                mc.update(counts);ma += [len(values),values.sum(),np.square(values).sum()]
                grouped=pd.DataFrame({'slot':group[ok],'z':values,'z2':values*values}).groupby('slot').agg(n=('z','size'),s=('z','sum'),q=('z2','sum'))
                for slot,r in grouped.iterrows():
                    self.cell.setdefault((p,v,day.dayofweek,int(slot)),np.zeros(3))[:] += r[['n','s','q']].to_numpy()

    @staticmethod
    def mean_std(a):
        n,s,q=a[:3]
        return (s/n if n else None,np.sqrt(max(0.,q-s*s/n)/(n-1)) if n>1 else None)

    def check(self,run):
        saved=json.loads((run/'statistics.json').read_text())
        assert saved['clock_rows']==self.rows and saved['utc_days']==self.days
        for p in P:
            for v in V:
                expected=self.counts[p,v];actual=saved['channels'][p][v]
                assert actual['counts']==dict(expected),(p,v,'counts')
                assert actual['parameter_cell_status_counts']==dict(self.status[p,v])
                assert actual['first_calibrated_available_at']==self.first[p,v]
                a=self.acc[p,v];mean,sd=self.mean_std(a)
                m={'count':int(a[0]),'mean':mean,'sample_std':sd,'minimum':a[3] if a[0] else None,'maximum':a[4] if a[0] else None}
                for k,x in m.items():
                    y=actual['z_descriptive'][k]
                    if x is None: assert y is None
                    else: np.testing.assert_allclose(y,x,rtol=1e-9,atol=1e-9,err_msg=f'{p}/{v}/{k}')
        rows=[]
        for (month,p,v),(c,a) in sorted(self.month.items()):
            mean,sd=self.mean_std(a)
            rows.append({'month':month,'policy':p,'channel':v,**dict(c),'z_mean':mean,'z_sample_std':sd})
        check_frame(pd.DataFrame(rows),pd.read_parquet(run/'monthly_statistics.parquet'))
        rows=[]
        for p in P:
            for v in V:
                for w in range(7):
                    for slot in range(96):
                        a=self.cell.get((p,v,w,slot),np.zeros(3));mean,sd=self.mean_std(a)
                        rows.append({'policy':p,'channel':v,'weekday':w,'slot':slot,'n':int(a[0]),
                                     'z_mean':np.nan if mean is None else mean,'z_sample_std':np.nan if sd is None else sd})
        check_frame(pd.DataFrame(rows),pd.read_parquet(run/'weekday_slot_diagnostics.parquet'))


def verify_calibration_run(run):
    from src.deseasonal import CalibrationSpec  # Schema only, no production numerical functions.
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text())
    for key,r in manifest['inputs'].items():
        assert sha256(Path(r['path']))==r['sha256'],'Input changed: '+key
    for key,digest in manifest['outputs'].items():assert sha256(run/key)==digest,'Output changed: '+key
    for key,digest in manifest.get('code_sha256',{}).items():assert sha256(run/'code'/key)==digest,'Source snapshot changed: '+key
    spec=CalibrationSpec(**manifest['configuration']);source=Rows(manifest['inputs']['clock']['path'])
    total=pq.ParquetFile(manifest['inputs']['clock']['path']).metadata.num_rows
    assert total%86400==0 and total>0
    normalized=Rows(run/'standardized_clock_1s.parquet');daily_rows=Rows(run/'daily_sufficient_statistics.parquet')
    fit_rows=Rows(run/'calibration_parameters.parquet');history=[];stats=ReferenceStatistics();origin=None
    for i in range(total//86400):
        frame=source.take(86400);day=pd.Timestamp(frame.bin_start.iloc[0])
        if origin is None:origin=day
        assert day==origin+pd.Timedelta(days=i) and day==day.normalize()
        actual_time=pd.DatetimeIndex(frame.bin_start).as_unit('ns').asi8
        np.testing.assert_array_equal(actual_time,day.value+np.arange(86400)*S)
        np.testing.assert_array_equal(pd.DatetimeIndex(frame.available_at).as_unit('ns').asi8,actual_time+S)
        fit=reference_fit(history,i,spec)
        expected=reference_output(frame,fit);check_frame(expected,normalized.take(86400))
        daily=reference_daily(frame);a,b=reference_tables(day,daily,fit,spec)
        check_frame(a,daily_rows.take(768));check_frame(b,fit_rows.take(768))
        stats.add(day,frame,expected,fit);history.append(daily)
        if (i+1)%100==0:print(f'Independently verified {(i+1)*86400:,} normalized clock rows and {i+1} daily fits',flush=True)
    assert source.exhausted() and normalized.exhausted() and daily_rows.exhausted() and fit_rows.exhausted()
    stats.check(run)
    return {'status':'passed','clock_rows_checked':total,'utc_days_checked':len(history),
        'daily_moment_rows_checked':len(history)*768,'calibration_parameter_rows_checked':len(history)*768,
        'normalized_channel_values_checked':total*8,'source_read_batch_size':400000,
        'checks':['full sequential parent-clock reconstruction','pandas daily cell moments',
                  'prior-date-only parameters with independent pairwise pooled variance',
                  'every serialized normalized value and missingness mask','all counts/moments/monthly/weekday-slot diagnostics',
                  'input/output/executed-source fingerprints'],
        'scope':'Conditional on registered retrospective support; no assertion of online outage detection, optimal seasonal specification, Gaussian scores or event identification.'}

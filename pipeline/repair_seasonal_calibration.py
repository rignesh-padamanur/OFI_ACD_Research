"""Stage 04b: sequential prior-day seasonality and scale calibration on the verified clock."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import io
import json
import platform
import re
import shutil
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_preparation import atomic_json, sha256
from src.deseasonal import CalibrationSpec, DAY, SECOND, fit_day, daily_moments, normalize_day, fit_table, moments_table
from src.calibration_statistics import CalibrationStatistics
from src.verify_seasonal_calibration import verify_calibration_run
from repair_signal_clock import resolve_inputs as resolve_clock_ancestors

ROOT=Path(__file__).resolve().parent
CODE=['repair_seasonal_calibration.py','src/deseasonal.py','src/calibration_statistics.py',
      'src/verify_seasonal_calibration.py','src/event_detection.py','src/pipeline_state.py','src/intensity.py',
      'src/data_preparation.py','src/quote_imbalance_statistics.py','src/observation_eligibility.py',
      'src/verify_quote_imbalance.py','repair_signal_clock.py','tests/test_seasonal_calibration.py']


def run_tests():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'output':stream.getvalue(),
              'test_source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage04b_tests.json',evidence);print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise RuntimeError('Tests failed; build and promotion refused')
    return evidence


def resolve_inputs(root):
    ancestors,parent,elig,signal=resolve_clock_ancestors(root)
    pointer=json.loads((root/'data/active_signal_clock.json').read_text())
    run=root/'data/signals'/pointer['clock_run_id'];m=json.loads((run/'manifest.json').read_text())
    for field,value in [('parent_data_run',parent),('parent_eligibility_run',elig),('parent_signal_run',signal)]:
        if pointer[field]!=value or m[field]!=value:raise ValueError('Clock active lineage changed: '+field)
    if m['status']!='verified_supported_signal_clock' or json.loads((run/'verification.json').read_text())['status']!='passed':
        raise ValueError('Verified parent clock required')
    if m['inputs']!=ancestors:raise ValueError('Clock ancestors changed')
    records=dict(ancestors)
    for key,file in [('clock','signal_clock_1s.parquet'),('clock_manifest','manifest.json'),
                     ('clock_verification','verification.json'),('clock_statistics','statistics.json')]:
        path=run/file;records[key]={'path':str(path),'sha256':sha256(path)}
        if file in m['outputs'] and records[key]['sha256']!=m['outputs'][file]:raise ValueError('Clock artifact changed: '+file)
    return records,{'parent_data_run':parent,'parent_eligibility_run':elig,'parent_signal_run':signal,'parent_clock_run':run.name}


def build_artifacts(clock_path,run,spec):
    source=pq.ParquetFile(clock_path);total=source.metadata.num_rows
    if not total or total%86400:raise ValueError('Registered full calendar days required')
    ndays=total//86400;history=np.zeros((ndays,4,2,96,3));stats=CalibrationStatistics();writers={};origin=None
    def write(file,frame):
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if file not in writers:writers[file]=pq.ParquetWriter(run/file,table.schema,compression='zstd')
        writers[file].write_table(table)
    try:
        for index,batch in enumerate(source.iter_batches(batch_size=86400)):
            frame=batch.to_pandas();day=pd.Timestamp(frame.bin_start.iloc[0])
            if origin is None:origin=day
            if len(frame)!=86400 or day.value%DAY or day!=origin+pd.Timedelta(days=index):
                raise ValueError('Parent clock must comprise consecutive complete UTC days')
            t=pd.DatetimeIndex(frame.bin_start).as_unit('ns').asi8
            if not np.array_equal(t,day.value+np.arange(86400)*SECOND):raise ValueError('Missing/duplicate parent clock row')
            # The application day's measurements have not entered history when fit_day runs.
            fit=fit_day(history,index,spec)
            normalized=normalize_day(frame,fit)
            stats.add(day,frame,normalized,fit)
            moments=daily_moments(frame)
            write('standardized_clock_1s.parquet',normalized)
            write('calibration_parameters.parquet',fit_table(day,fit,spec))
            write('daily_sufficient_statistics.parquet',moments_table(day,moments))
            history[index]=moments
            if (index+1)%100==0:print(f'Calibrated {(index+1)*86400:,} clock rows using completed prior days only',flush=True)
    finally:
        for writer in writers.values():writer.close()
    assert stats.rows==total and stats.days==ndays
    monthly,cells=stats.tables();monthly.to_parquet(run/'monthly_statistics.parquet',index=False)
    cells.to_parquet(run/'weekday_slot_diagnostics.parquet',index=False)
    result=stats.result();atomic_json(run/'statistics.json',result)
    return result


METHOD={
 'scope':'Notebook 2: past-day seasonal calibration for quote counts and absolute recorded quote imbalance; four support scenarios, no primary chosen.',
 'timing':'For UTC day d, fit once from [d-lookback_days,d). Matching cell uses the same UTC weekday and 15-minute slot. No current-day observations enter any mean, scale or factor. Current normalized signal is available at bin completion.',
 'reference_specification':'56 calendar days, full-window warm-up, four prior matching dates, 1800 eligible seconds. Fixed before execution, configurable, not optimized or mandated by cited literature.',
 'weighting':'Equal weight per eligible second, including supported zeros. Channel-specific support: count support for counts, valid-flow support for absolute imbalance. Distinct-date and observation counts are recorded; seconds are not assumed independent for inference.',
 'formula':'phi=historical_cell_mean/historical_global_mean; adjusted_x=x/phi; z_x=(x-historical_cell_mean)/historical_cell_sample_sd. Signed imbalance is divided by positive phi_abs to preserve direction, never divided by its possibly zero signed mean.',
 'variance':'Daily centered sums of squares pooled within and between dates; sample denominator n-1. No zero/NaN denominator fallback, winsorization, quantile threshold, Gaussian transformation or future fitting.',
 'unavailable':'Unsupported input, incomplete warm-up, too few dates/seconds, nonpositive/unavailable mean or scale produce missing calibrated outputs with explicit reasons. No cross-cell interpolation or retrospective backfill.',
 'interpretation':'Rolling unsupervised calibration from past covariates, conditional on fixed retrospective observation masks. Not an online outage detector, frozen supervised train/test split, optimal seasonality model, or evidence of prediction/causal information/permanence.',
 'limits':['UTC weekday/time cells do not fully align changing local-market sessions/DST or holiday effects. New or unsupported cells can lose calibration until enough same-weekday history exists.',
           'Prior mean/scale estimates adapt to changing activity and can include past shocks. Out-of-sample moments need not equal zero/one; scores need not be Gaussian or independent.',
           'A primary support/freshness policy and calibration specification sensitivity must be settled before paper estimation.',
           'Duration/ACD inputs, multivariate scoring/covariance/thresholds, observable event onset, supervised evaluation and persistence inference remain pending.'],
 'sources':[{'title':'Hyndman and Athanasopoulos, rolling-origin evaluation','url':'https://otexts.com/fpp3/tscv.html',
             'scope':'Chronological prior-only evaluation principle; does not select our numerical reference settings.'},
            {'title':'Andersen and Bollerslev (1997), Intraday periodicity and volatility persistence in financial markets',
             'url':'https://public.econ.duke.edu/~boller/Published_Papers/joef_97.pdf',
             'scope':'Importance of intraday seasonality in return volatility; does not validate this quote-count/quote-pressure model.'}],
 'coverage_certified':False,'downstream_ready':False,'primary_freshness_policy':None}


def build_run(root,run_id,evidence,spec):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid run id')
    inputs,parents=resolve_inputs(root);run=root/'data/signals'/run_id;run.mkdir(parents=True,exist_ok=False)
    m={'schema_version':1,'run_id':run_id,'stage':'04b_past_day_seasonal_calibration','status':'building',**parents,
       'created_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':asdict(spec),'inputs':inputs,
       'production_read_batch_size':86400,'code_sha256':{},'primary_freshness_policy':None,
       'coverage_certified':False,'downstream_ready':False,
       'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__}}
    for name in CODE:
        target=run/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target);m['code_sha256'][name]=sha256(target)
    for file in (root/'tests').glob('test_*.py'):
        dest=run/'full_test_sources'/file.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
    atomic_json(run/'manifest.json',m);atomic_json(run/'methodology.json',METHOD);atomic_json(run/'test_evidence.json',evidence)
    stats=build_artifacts(inputs['clock']['path'],run,spec)
    m['outputs']={file:sha256(run/file) for file in ['standardized_clock_1s.parquet','daily_sufficient_statistics.parquet',
        'calibration_parameters.parquet','monthly_statistics.parquet','weekday_slot_diagnostics.parquet',
        'statistics.json','methodology.json','test_evidence.json']}
    m['status']='built_pending_independent_verification';atomic_json(run/'manifest.json',m)
    print('Starting independent sequential calibration verification',flush=True)
    verification=verify_calibration_run(run);atomic_json(run/'verification.json',verification)
    m['status']='verified_past_day_reference_calibration';atomic_json(run/'manifest.json',m)
    return run,stats,verification


def promote(root,run,stats,verification):
    m=json.loads((run/'manifest.json').read_text());inputs,parents=resolve_inputs(root)
    if inputs!=m['inputs'] or any(m[k]!=v for k,v in parents.items()):raise ValueError('Inputs changed during build')
    if m['status']!='verified_past_day_reference_calibration' or verification['status']!='passed':raise ValueError('Unverified calibration')
    path=root/'reports/pipeline_master_statistics.json';master=json.loads(path.read_text())
    entry={'stage':m['stage'],'run_id':run.name,**parents,'status':m['status'],'configuration':m['configuration'],
           'completed_at_utc':datetime.now(timezone.utc).isoformat(),'statistics':stats,'verification':verification,
           'methodology':METHOD,'primary_freshness_policy':None,'downstream_ready':False}
    master['stage_history'].append(entry);master['current_calibration_stage']=entry
    master['pipeline_status']='past_day_reference_calibration_verified_duration_scoring_and_models_pending'
    master['downstream_status']['ofi']='past_day_seasonal_reference_normalization_verified_specification_sensitivity_pending'
    master['downstream_status']['intensity']='past_day_seasonal_reference_normalization_verified_specification_sensitivity_pending'
    master['downstream_status']['seasonality_acd']='count_and_quote_pressure_calibration_verified_duration_ACD_pending'
    master['repair_progress']={'stage':m['stage'],'status':m['status'],
        'next_required_work':'Repair duration/ACD treatment and chronological calibration; then rebuild multivariate score, threshold and onset. Select primary support/specification and evaluate sensitivity before paper estimation.'}
    master['updated_at_utc']=datetime.now(timezone.utc).isoformat();atomic_json(path,master)
    atomic_json(root/'data/active_calibration.json',{'schema_version':1,'calibration_run_id':run.name,**parents,
        'status':m['status'],'configuration':m['configuration'],'coverage_certified':False,'downstream_ready':False,'primary_freshness_policy':None})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='seasonal_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only',action='store_true');parser.add_argument('--verify-only')
    parser.add_argument('--lookback-days',type=int,default=56);parser.add_argument('--min-prior-dates',type=int,default=4)
    parser.add_argument('--min-prior-seconds',type=int,default=1800);args=parser.parse_args()
    if args.verify_only:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.verify_only):parser.error('Invalid run id')
        print(json.dumps(verify_calibration_run(ROOT/'data/signals'/args.verify_only),indent=2));return
    spec=CalibrationSpec(args.lookback_days,args.min_prior_dates,args.min_prior_seconds)
    evidence=run_tests()
    if args.test_only:return
    run,stats,verification=build_run(ROOT,args.run_id,evidence,spec);promote(ROOT,run,stats,verification)
    print(json.dumps({'run':str(run),'statistics':stats,'verification':verification},indent=2),flush=True)


if __name__=='__main__':main()

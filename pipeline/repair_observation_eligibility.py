"""Build and independently verify cached-data observation eligibility; no data acquisition."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import io
import json
import platform
import re
import shutil
import unittest
import numpy as np
import pandas as pd
import pyarrow

from src.data_preparation import atomic_json, sha256
from src.observation_eligibility import (POLICIES, LOOKBACKS, HORIZONS, make_hours,
    load_quote_coordinates, quote_structure, scan_raw_support, build_segments,
    support_intervals, transition_statistics, window_attrition)
from src.verify_observation_eligibility import verify_eligibility

ROOT=Path(__file__).resolve().parent
CODE=['repair_observation_eligibility.py','src/observation_eligibility.py',
      'src/verify_observation_eligibility.py','src/data_preparation.py','src/acquisition.py',
      'src/pipeline_state.py','tests/test_observation_eligibility.py']


def contract_tests():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'output':stream.getvalue(),
        'test_source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage02b_tests.json',evidence)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Tests failed; no build or promotion')
    return evidence


def write_artifacts(parent, coverage, run):
    """Numerical build, also usable in small serialized integration tests."""
    pm=json.loads((parent/'manifest.json').read_text())
    hours=make_hours(pd.read_parquet(coverage))
    ts,rid=load_quote_coordinates(parent,pm['input_files'])
    _,_,removed,starts=quote_structure(ts,rid,hours)
    queries=(rid[starts[1:]-1]+1)[removed[starts[1:]]>0].tolist()
    total=sum(i['rows'] for i in pm['input_files'])
    if rid[-1]<total-1: queries.append(int(rid[-1]+1))
    print('Checking raw coverage and locating cleaning boundaries',flush=True)
    boundary_times,raw_rows,end=scan_raw_support(parent,hours,queries)
    segments=build_segments(ts,rid,hours,boundary_times,end)
    support=support_intervals(ts,segments)
    transition,gaps=transition_statistics(ts,segments)
    del ts,rid,removed;gc.collect()
    hourly,summaries=window_attrition(support,hours)
    composition=hourly.assign(month=hourly.hour.dt.strftime('%Y-%m'),
        utc_six_hour_band=(hourly.hour.dt.hour//6)*6).groupby(
        ['policy','lookback_seconds','horizon_seconds','month','utc_six_hour_band'],as_index=False).eligible_clock_anchors.sum()
    frames={'hourly_eligibility.parquet':hours,'segments.parquet':segments,
        'support_intervals.parquet':support,'long_interquote_gaps.parquet':gaps,
        'window_attrition_by_hour.parquet':hourly,'window_attrition_by_month_band.parquet':composition}
    for name,frame in frames.items(): frame.to_parquet(run/name,index=False)
    statistics={'coverage':{'raw_rows':raw_rows,'requested_hours':len(hours),
        'nominal_open_hours':int(hours.nominal_weekly_open.sum()),
        'observed_hours':int(hours.structurally_observed_hour.sum()),
        'absent_nominal_open_hours':int((hours.nominal_weekly_open & hours.rows.eq(0)).sum())},
        'quote_transitions':transition,'segment_start_reasons':{str(k):int(v) for k,v in segments.start_reason.value_counts().items()},
        'support_interval_counts':{str(k):int(v) for k,v in support.policy.value_counts().items()},
        'window_opportunities':summaries,'new_quotes_acquired':0,'quotes_deleted_by_this_stage':0,
        'coverage_certified':False,'primary_freshness_policy_selected':False,
        'event_counts_computed':False,'model_estimates_computed':False}
    atomic_json(run/'statistics.json',statistics)
    return statistics,list(frames)+['statistics.json']


def build_run(root,run_id,evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): raise ValueError('Invalid run id')
    active=json.loads((root/'data/active_run.json').read_text())
    parent=root/'data/runs'/active['data_run_id']
    acquisition=root/'data/acquisition/step02_hours_20260927'
    pm=json.loads((parent/'manifest.json').read_text()); am=json.loads((acquisition/'manifest.json').read_text())
    if pm['status']!='verified' or am['parent_data_run']!=parent.name:
        raise ValueError('A verified matching parent and audit are required')
    if {i['name']:i['sha256'] for i in pm['input_files']}!={i['name']:i['sha256'] for i in am['input_files']}:
        raise ValueError('Audit and parent cached-input lineages differ')
    paths={'parent_manifest':parent/'manifest.json','parent_verification':parent/'verification.json',
        'parent_clean':parent/'clean_quotes.parquet','parent_raw':parent/'raw_quotes.parquet',
        'coverage':acquisition/'coverage_before.parquet','coverage_manifest':acquisition/'manifest.json',
        'coverage_verification':acquisition/'coverage_verification.json'}
    hashes={name:{'path':str(p),'sha256':sha256(p)} for name,p in paths.items()}
    for name in ['raw_quotes.parquet','clean_quotes.parquet']:
        if sha256(parent/name)!=pm['outputs'][name]['sha256']: raise ValueError('Parent data fingerprint mismatch')
    if json.loads(paths['parent_verification'].read_text())['status']!='passed' or json.loads(paths['coverage_verification'].read_text())['status']!='passed':
        raise ValueError('Prior verification did not pass')
    run=root/'data/eligibility'/run_id;run.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'run_id':run_id,'stage':'02b_observation_eligibility',
        'status':'building','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'parent_data_run':parent.name,'parent_directory':str(parent),'input_hashes':hashes,
        'freshness_diagnostics':POLICIES,'lookback_diagnostics_seconds':list(LOOKBACKS),
        'horizons_seconds':list(HORIZONS),'primary_freshness_policy':None,
        'coverage_certified':False,'downstream_ready':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
            'pandas':pd.__version__,'pyarrow':pyarrow.__version__},
        'limitations':[
            'Observed hours do not certify update completeness; silence can be inactivity or missing recording.',
            'Holiday versus outage causes remain unresolved; absent hours are never automatically labelled outages.',
            'Legacy same-timestamp price deduplication may have lost size updates; no records are recovered here.',
            'Structural-only support is a diagnostic without a stale-quote safeguard, not a selected analysis policy.',
            '15/30/60-second freshness rules and 0/60/300-second contexts are diagnostics, not fitted thresholds or ACD burn-in choices.',
            'Eligibility describes recorded support, not an independently observed true market path.',
            'Outcome availability and interval ends are ex-post audit information and must never be predictors.',
            'Clock anchors are not events; event attrition, sample-selection bias, OFI and economic identification remain to be evaluated.',
            'This stage supports strictly increasing clean timestamps; new tied-timestamp inputs fail until an ordering policy is specified.'
        ]}
    manifest['code_sha256']={}
    for name in CODE:
        dest=run/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
        manifest['code_sha256'][name]=sha256(dest)
    atomic_json(run/'manifest.json',manifest)
    atomic_json(run/'test_evidence.json',evidence)
    statistics,names=write_artifacts(parent,paths['coverage'],run)
    manifest['output_sha256']={name:sha256(run/name) for name in names+['test_evidence.json']}
    manifest['status']='built_pending_independent_verification'
    atomic_json(run/'manifest.json',manifest)
    print('Independently checking every retained timestamp and all window counts',flush=True)
    verification=verify_eligibility(run)
    atomic_json(run/'verification.json',verification)
    manifest['status']='verified_observation_support_only'
    atomic_json(run/'manifest.json',manifest)
    return run,statistics,verification


def promote(root,run,statistics,verification):
    m=json.loads((run/'manifest.json').read_text())
    active=json.loads((root/'data/active_run.json').read_text())
    if active['data_run_id']!=m['parent_data_run'] or verification['status']!='passed' or m['status']!='verified_observation_support_only':
        raise ValueError('Cannot register unmatched or unverified eligibility')
    master_path=root/'reports/pipeline_master_statistics.json'
    master=json.loads(master_path.read_text())
    entry={'stage':m['stage'],'run_id':run.name,'parent_data_run':m['parent_data_run'],
        'status':m['status'],'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'statistics':statistics,'verification':verification,'coverage_certified':False,
        'primary_freshness_policy':None,'downstream_ready':False}
    master['stage_history'].append(entry);master['current_eligibility_stage']=entry
    master['repair_progress']={'stage':m['stage'],'status':m['status'],
        'next_required_work':'Review observation-support attrition, then validate quote-based imbalance; select substantive eligibility and history policies before model fitting.'}
    master['pipeline_status']='data_and_observation_support_verified_economic_pipeline_pending'
    master['downstream_status']['coverage_recovery']='not_pursued_user_requests_cached_data_only_no_paid_access'
    master['downstream_status']['observation_eligibility']='verified_diagnostics_no_primary_freshness_policy_selected'
    master['data_access_policy'].update(paid_access_authorized=False,
        current_plan='Use cached data only; no additional acquisition required for this bounded eligibility repair.')
    master['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(master_path,master)
    atomic_json(root/'data/active_eligibility.json',{'schema_version':1,'eligibility_run_id':run.name,
        'parent_data_run':m['parent_data_run'],'parent_clean_sha256':m['input_hashes']['parent_clean']['sha256'],
        'status':m['status'],'coverage_certified':False,'primary_freshness_policy':None,'downstream_ready':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='eligibility_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--verify-only',metavar='RUN_ID')
    args=parser.parse_args()
    if args.verify_only:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.verify_only): parser.error('Invalid run id')
        print(json.dumps(verify_eligibility(ROOT/'data/eligibility'/args.verify_only),indent=2));return
    evidence=contract_tests()
    if args.test_only:return
    run,stats,verification=build_run(ROOT,args.run_id,evidence)
    promote(ROOT,run,stats,verification)
    print(json.dumps({'run':str(run),'quote_transitions':stats['quote_transitions'],'verification':verification},indent=2),flush=True)


if __name__=='__main__': main()

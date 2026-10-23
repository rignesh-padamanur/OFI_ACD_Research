"""Stage 02a: audit, acquire absent hours, then build a new verified data run.

Run phases in order: audit, fetch, build. Raw receipts are retained. Existing
observed hours are not replaced by this bounded whole-hour repair.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import io
import json
import re
import shutil
import time
import unittest

import pandas as pd
import requests

from src.acquisition import audit_months, decode_hour, fetch_hour
from src.data_preparation import QUOTE_COLUMNS, atomic_json, sha256, write_panel
from src.verify_data_stage import verify_run

ROOT=Path(__file__).resolve().parent
START='2024-01-01 00:00Z'; END='2026-08-01 00:00Z'
CONTROLS=['2024-01-02 12:00Z','2024-03-11 21:00Z','2024-10-28 21:00Z',
          '2025-03-10 21:00Z','2025-11-03 22:00Z','2026-05-06 12:00Z']


def contract_tests():
    suite=unittest.TestSuite();fingerprints={}
    for name in ['test_data_preparation.py','test_acquisition.py']:
        path=ROOT/'tests'/name
        spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module));fingerprints[name]=sha256(path)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'test_source_sha256':fingerprints,'output':stream.getvalue()}
    atomic_json(ROOT/'reports/tests/stage02_tests.json',evidence)
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    return evidence


def audit(acq):
    evidence=contract_tests();acq.mkdir(parents=True,exist_ok=False)
    ledger,gaps,stats,files=audit_months(ROOT/'data/raw/dukascopy',START,END)
    ledger.to_parquet(acq/'coverage_before.parquet',index=False)
    gaps.to_parquet(acq/'gaps_before.parquet',index=False)
    absent=ledger.loc[ledger.nominal_weekly_open & ledger.rows.eq(0),'hour']
    targets=[{'hour':h.isoformat(),'purpose':'absent_nominal_open_hour'} for h in absent]
    for t in pd.to_datetime(CONTROLS):
        if not ledger.loc[ledger.hour.eq(t),'rows'].iloc[0]>0:raise ValueError('Prespecified control hour is absent')
        targets.append({'hour':t.isoformat(),'purpose':'decoder_control_existing_hour'})
    atomic_json(acq/'targets.json',{'targets':targets})
    manifest={'stage':'02a_whole_hour_coverage','status':'audited','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'start_inclusive':START,'end_exclusive':END,'source_dir':str(ROOT/'data/raw/dukascopy'),
        'parent_data_run':json.loads((ROOT/'data/active_run.json').read_text())['data_run_id'],
        'input_files':files,'before_statistics':stats,'test_evidence':evidence,'coverage_certified':False,
        'scope':'Acquire absent nominal weekly-open hours; retain observed cached hours. Holidays and within-hour completeness are not inferred.'}
    manifest['code_files']={}
    for name in ['src/acquisition.py','repair_acquisition.py','tests/test_acquisition.py']:
        out=acq/'code'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,out)
        manifest['code_files'][name]=sha256(out)
    atomic_json(acq/'manifest.json',manifest)
    print(json.dumps(stats,indent=2),flush=True)


def acquire(acq):
    manifest=json.loads((acq/'manifest.json').read_text())
    record_phase_code(acq,manifest,'fetch')
    targets=json.loads((acq/'targets.json').read_text())['targets'];results=[]
    consecutive_failures=0;stop_reason=None
    with requests.Session() as session:
        for target in targets:
            result=fetch_hour(target['hour'],acq/'receipts',session=session)
            result['purpose']=target['purpose'];results.append(result)
            print(target['hour'],result['status'],result.get('rows',0),flush=True)
            consecutive_failures=consecutive_failures+1 if result['status']=='network_error' else 0
            if result['status'] in ['access_denied','rate_limited']:
                stop_reason=result['status'];break
            if consecutive_failures>=3:
                stop_reason='repeated_transport_failures';break
            time.sleep(.1)
    completed={r['hour']:r for r in results}
    for target in targets:
        if target['hour'] not in completed:
            receipt=acq/'receipts'/f'{pd.Timestamp(target["hour"]).strftime("%Y%m%dT%H")}.json'
            if receipt.exists():results.append({**json.loads(receipt.read_text()),'purpose':target['purpose']})
            else:results.append({**target,'status':'not_attempted_after_'+str(stop_reason)})
    atomic_json(acq/'acquisition_results.json',{'results':sorted(results,key=lambda r:r['hour'])})
    manifest['status']='acquisition_attempts_recorded';manifest['response_counts']=pd.Series([r['status'] for r in results]).value_counts().to_dict()
    atomic_json(acq/'manifest.json',manifest)
    print(json.dumps(manifest['response_counts'],indent=2),flush=True)


def record_phase_code(acq,manifest,phase):
    tag=phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    fingerprints={}
    for name in ['src/acquisition.py','src/verify_acquisition.py','repair_acquisition.py','tests/test_acquisition.py']:
        target=acq/tag/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        fingerprints[name]=sha256(target)
    manifest.setdefault('phase_implementations',[]).append({'phase':phase,'directory':tag,'sha256':fingerprints})
    atomic_json(acq/'manifest.json',manifest)


def build(acq,data_run_id):
    from src.verify_acquisition import verify_recovery
    contract_tests()
    manifest=json.loads((acq/'manifest.json').read_text())
    record_phase_code(acq,manifest,'build')
    results=json.loads((acq/'acquisition_results.json').read_text())['results']
    unacceptable=[r for r in results if r['status'] not in ['ok','empty_200','empty_decoded','not_found_404']]
    if unacceptable:raise RuntimeError('Acquisition errors remain; repair/retry them before a data build')
    controls=[r for r in results if r['purpose']=='decoder_control_existing_hour']
    if not all(r['status']=='ok' for r in controls):raise RuntimeError('A decoder control did not return data')
    sources=acq/'monthly';sources.mkdir(exist_ok=False)
    original=Path(manifest['source_dir']);recovered_rows=0
    before=pd.read_parquet(acq/'coverage_before.parquet').set_index('hour')
    for item in manifest['input_files']:
        path=original/item['name']
        if sha256(path)!=item['sha256']:raise ValueError('Original source changed after audit')
        old=pd.read_parquet(path,columns=QUOTE_COLUMNS);additions=[]
        _,year,month=path.stem.split('_')
        for result in results:
            h=pd.Timestamp(result['hour'])
            if result['purpose']!='absent_nominal_open_hour' or result['status']!='ok' or (h.year,h.month)!=(int(year),int(month)):continue
            if before.loc[h,'rows']!=0:raise ValueError('Recovery may only add previously absent hours')
            receipt=acq/'receipts'/result['response_file']
            if sha256(receipt)!=result['response_sha256']:raise ValueError('Acquired bytes changed')
            frame=decode_hour(receipt.read_bytes(),h)[QUOTE_COLUMNS]
            additions.append(frame);recovered_rows+=len(frame)
        combined=pd.concat([old,*additions],ignore_index=True).sort_values('timestamp',kind='stable').reset_index(drop=True)
        combined.to_parquet(sources/path.name,index=False,compression='zstd')
    recovery=verify_recovery(acq)
    run=write_panel(ROOT,sources,data_run_id)
    verified=verify_run(run)
    ledger,gaps,stats,_=audit_months(sources,START,END)
    outcome={r['hour']:r for r in results if r['purpose']=='absent_nominal_open_hour'}
    ledger['acquisition_status']=[outcome[h.isoformat()]['status'] if h.isoformat() in outcome else
        ('legacy_hour_not_refetched' if n else 'not_requested_weekly_closed') for h,n in zip(ledger.hour,ledger.rows)]
    ledger.to_parquet(acq/'coverage_after.parquet',index=False);gaps.to_parquet(acq/'gaps_after.parquet',index=False)
    manifest.update(status='verified_whole_hour_recovery',data_run_id=run.name,recovered_raw_rows=recovered_rows,
        after_statistics=stats,recovery_verification=recovery,data_verification=str((run/'verification.json').relative_to(ROOT)))
    atomic_json(acq/'manifest.json',manifest)
    run_manifest=json.loads((run/'manifest.json').read_text())
    run_manifest.update(stage='02a_whole_hour_coverage',source_scope='legacy_months_plus_verified_absent_hour_recovery',
        acquisition_manifest=str((acq/'manifest.json').relative_to(ROOT)))
    atomic_json(run/'manifest.json',run_manifest)
    master_path=ROOT/'reports/pipeline_master_statistics.json';master=json.loads(master_path.read_text())
    entry={'stage':'02a_whole_hour_coverage','status':'verified_whole_hour_recovery_not_complete_feed',
        'run_id':run.name,'manifest':str((run/'manifest.json').relative_to(ROOT)),
        'acquisition_manifest':str((acq/'manifest.json').relative_to(ROOT)),
        'statistics':verified['statistics'],'coverage_before':manifest['before_statistics'],'coverage_after':stats,
        'recovered_raw_rows':recovered_rows,'recovery_verification':recovery,
        'test_evidence':json.loads((ROOT/'reports/tests/stage02_tests.json').read_text()),
        'limitations':['Observed cached hours were not comprehensively refetched; old deduplication losses remain unquantified.',
            'Empty/404 responses are vendor availability observations, not independent proof of a market closure.',
            'Partial hours, long gaps, and missing size-only update semantics remain subject to stage 02b.',
            'Quote sizes describe offered liquidity; no executed trade direction or dealer inventory is observed.'],
        'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    master['stage_history'].append(entry);master['current_data_stage']=entry
    master['downstream_status']['coverage_recovery']='whole_hour_recovery_verified; detailed_coverage_and_transition_policy_pending'
    master['updated_at_utc']=datetime.now(timezone.utc).isoformat();atomic_json(master_path,master)
    atomic_json(ROOT/'data/active_run.json',{'schema_version':2,'data_run_id':run.name,'stage':'02a_whole_hour_coverage',
        'coverage_certified':False,'downstream_ready':False})
    print(json.dumps({'data_statistics':verified['statistics'],'coverage_statistics':stats,'recovered_rows':recovered_rows},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['audit','fetch','build','test'])
    parser.add_argument('--acquisition-id',default='step02_hours_20260927')
    parser.add_argument('--data-run-id',default='step02_hour_recovery_20260927')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.acquisition_id):parser.error('Invalid acquisition id')
    acq=ROOT/'data/acquisition'/args.acquisition_id
    if args.phase=='test':print(json.dumps(contract_tests(),indent=2))
    elif args.phase=='audit':audit(acq)
    elif args.phase=='fetch':acquire(acq)
    else:build(acq,args.data_run_id)


if __name__=='__main__':main()

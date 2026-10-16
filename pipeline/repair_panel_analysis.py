"""Stage07f: fixed full-panel accounting and later-sample analysis inputs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import io
import json
import platform
import re
import shutil
import traceback
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from src.data_preparation import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
STAGE = '07f_full_panel_analysis_inputs'
STATUS = 'verified_full_panel_analysis_inputs'
SCOPE = 'full_panel_accounting_later_reanalysis_inputs_only'


def read(path): return json.loads(Path(path).read_text())
def require(condition, message):
    if not condition: raise AssertionError(message)


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage07f_tests.json', evidence)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Tests failed; no analysis build')
    return evidence


def resolve_inputs():
    inputs, parents = {}, {}
    def add(key, path, expected=None):
        path = Path(path); h = sha256(path)
        require(expected is None or h == expected, 'Changed parent artifact: '+str(path))
        inputs[key] = {'path': str(path), 'sha256': h}
    for prefix, pointer, identifier, status, files in [
        ('response','active_panel_response_audit.json','response_run_id','verified_full_panel_price_responses',
         {'anchors':'anchors.parquet'}),
        ('measurement','active_panel_measurement_audit.json','measurement_run_id','verified_full_panel_fixed_onset_measurement',
         {'measurements':'measurements.parquet','directions':'direction.parquet'}),
        ('scaling','active_regression_input_audit.json','input_run_id','verified_development_regression_inputs',
         {'scaling':'scaling.json','legacy_features':'features.parquet','legacy_samples':'samples.parquet'}),
    ]:
        run = ROOT/'data/signals'/read(ROOT/'data'/pointer)[identifier]
        m = read(run/'manifest.json')
        require(m['status'] == status and m['primary_policy'] == 'structural_only', 'Parent status/scope differs: '+prefix)
        require(read(run/'verification.json')['status'] == 'passed', 'Parent verification missing: '+prefix)
        for key, name in {prefix+'_manifest':'manifest.json', prefix+'_verification':'verification.json',
                          prefix+'_statistics':'statistics.json', **files}.items():
            add(key, run/name, m.get('outputs',{}).get(name))
        parents[prefix] = m
        if prefix == 'response':
            paths = sorted(name for name in m['outputs'] if name.startswith('paths/') and name.endswith('.parquet'))
            require(paths == [f'paths/cycle_{i:03d}.parquet' for i in range(60)], 'Response partition grid differs')
            for i, name in enumerate(paths): add(f'path_{i:03d}',run/name,m['outputs'][name])
    rm, mm, sm = [parents[key] for key in ['response','measurement','scaling']]
    require(rm['parent_measurement'] == mm['run_id'] and rm['parent_chronological_panel'] == mm['parent_chronological_panel'],
            'Response/measurement panel identities differ')
    require(rm['inputs']['onsets'] == mm['inputs']['onsets'], 'Response/measurement onset lineage differs')
    require(rm['inputs']['measurement_manifest'] == inputs['measurement_manifest'], 'Response measurement manifest differs')
    source = sm['inputs']['parent_manifest']
    add('legacy_ledger_manifest',source['path'],source['sha256'])
    lm = read(source['path'])
    require(lm['parent_price_response_audit'] == Path(rm['inputs']['legacy_manifest']['path']).parent.name,
            'Development ledger/response ancestry differs')
    return parents, inputs


def selected_paths(inputs):
    paths = [inputs[f'path_{i:03d}']['path'] for i in range(60)]
    return ds.dataset(paths,format='parquet').to_table(filter=ds.field('horizon_sec').isin([0,60,300])).to_pandas()


def legacy_reconciliation(inputs, features, samples):
    from src.verify_analysis_ledger import _compare
    old_f, old_s = [pd.read_parquet(inputs[key]['path']) for key in ['legacy_features','legacy_samples']]
    f = features.loc[features.episode_id.isin(old_f.episode_id)].sort_values('episode_id').reset_index(drop=True)
    old_f = old_f.sort_values('episode_id').reset_index(drop=True)
    shared_f = [name for name in old_f if name in f]
    require(set(old_f).issubset(f), 'Legacy feature field disappeared')
    _compare(f[shared_f],old_f[shared_f])
    old_s = old_s.sort_values(['episode_id','horizon_sec']).reset_index(drop=True)
    s = samples.loc[samples.episode_id.isin(old_f.episode_id)].sort_values(['episode_id','horizon_sec']).reset_index(drop=True)
    require(set(old_s).issubset(s), 'Legacy sample field disappeared')
    historical = old_s.target_at.le(pd.Timestamp('2024-05-06',tz='UTC'))
    columns = list(old_s)
    _compare(s.loc[historical,columns].reset_index(drop=True),old_s.loc[historical,columns].reset_index(drop=True))
    restored = ~historical & ~old_s.eligible_continuation & s.eligible_continuation
    return {'legacy_features_checked':len(f),'shared_feature_columns_checked':len(shared_f),
        'legacy_sample_rows':len(old_s),'historical_sample_rows_checked':int(historical.sum()),
        'restored_continuation_rows':int(restored.sum()),
        'restored_event_ids':sorted(s.loc[restored,'episode_id'].astype(int).unique().tolist()),
        'legacy_artifacts_preserved':True,'historical_features_and_outcomes_reconciled':True,
        'development_design_diagnostics_recomputed':True}


def build(run_id, evidence):
    from src.panel_analysis import assemble_panel_ledger, build_panel_inputs, panel_dependence, daily_calendar, dependence_summary
    require(re.fullmatch(r'[A-Za-z0-9_-]+',run_id) is not None,'Invalid run ID')
    parents, inputs = resolve_inputs()
    run = ROOT/'data/signals'/run_id; run.mkdir(exist_ok=False)
    snapshot = ROOT/'reports/repair_history'/('2026-10-01_pre_'+run_id); snapshot.mkdir(exist_ok=False)
    prior = [ROOT/'reports/pipeline_master_statistics.json', ROOT/'reports/PIPELINE_REPAIR_LOG.md', ROOT/'REPAIR_WORKFLOW.md',
             *sorted((ROOT/'data').glob('active_*.json'))]
    require(len(prior)==25,'Expected22 prior pointers and3 registration records')
    previous = {}
    for path in prior:
        name = str(path.relative_to(ROOT)); target = snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target); previous[name]=sha256(target)
    atomic_json(snapshot/'preserved_files.json',previous)
    m = {'run_id':run_id,'stage':STAGE,'status':'building','scope':SCOPE,'primary_policy':'structural_only',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs,
        'parent_response':parents['response']['run_id'],'parent_measurement':parents['measurement']['run_id'],
        'parent_scaling':parents['scaling']['run_id'],
        'deployment_start':'2024-04-22T00:00:00+00:00','analysis_start':'2024-05-06T00:00:00+00:00',
        'deployment_end_exclusive':'2026-08-01T00:00:00+00:00','horizons_seconds':[0,60,300],
        'snapshot_path':str(snapshot),'previous_files':previous,'code_sha256':{},
        'production_inference_approved':False,'downstream_ready':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__}}
    sources = [ROOT/'repair_panel_analysis.py',ROOT/'reports/STAGE07F_PANEL_ANALYSIS_PROTOCOL.md',
               *sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'tests').glob('test_*.py'))]
    for path in sources:
        name=str(path.relative_to(ROOT)); target=run/'code'/name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target); m['code_sha256'][name]=sha256(target)
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        require(sha256(path)==evidence['test_source_sha256'][path.name],'Tests changed after execution')
        dest=run/'full_test_sources'/path.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(path,dest)
    atomic_json(run/'manifest.json',m); atomic_json(run/'test_evidence.json',evidence)
    try:
        ledger=assemble_panel_ledger(pd.read_parquet(inputs['anchors']['path']),
            pd.read_parquet(inputs['measurements']['path']),pd.read_parquet(inputs['directions']['path']),selected_paths(inputs))
        for name in ['anchors','outcomes']: ledger[name].to_parquet(run/('ledger_'+name+'.parquet'),index=False)
        atomic_json(run/'ledger_specification.json',ledger['specification']); atomic_json(run/'adapter.json',ledger['adapter'])
        shutil.copy2(inputs['scaling']['path'],run/'scaling.json')
        output=build_panel_inputs(ledger['anchors'],ledger['outcomes'],read(run/'scaling.json'))
        for name,obj in output.items():
            if name=='statistics':continue
            if isinstance(obj,pd.DataFrame):obj.to_parquet(run/(name+'.parquet'),index=False)
            else:atomic_json(run/(name+'.json'),obj)
        calendar=daily_calendar(output);calendar.to_parquet(run/'daily_calendar.parquet',index=False)
        dep_stats=[]
        for h,dep in panel_dependence(output):
            folder=run/'dependence'/str(h);folder.mkdir(parents=True)
            for name,frame in dep.items():frame.to_parquet(folder/(name+'.parquet'),index=False)
            dep_stats.append(dependence_summary(h,dep))
            print(f'Constructed main-sample dependence horizon{h}: {len(dep["intervals"]):,} events, {len(dep["pairs"]):,} pairs',flush=True)
            del dep;gc.collect()
        legacy=legacy_reconciliation(inputs,output['features'],output['samples']);atomic_json(run/'legacy_reconciliation.json',legacy)
        stats={'scope':SCOPE,**output['statistics'],'dependence':dep_stats,'legacy_reconciliation':legacy,
            'calendar':{'rows':len(calendar),'days':831,'development_days':14,'later_days':817},
            'empirical_models_fitted':0,'production_inference_approved':False,'downstream_ready':False}
        atomic_json(run/'statistics.json',stats)
        m['outputs']={str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()
            and p.name!='manifest.json' and 'code' not in p.relative_to(run).parts and 'full_test_sources' not in p.relative_to(run).parts}
        m['status']='built_pending_independent_verification';atomic_json(run/'manifest.json',m)
        return run
    except Exception as error:
        atomic_json(run/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        m['status']='failed_unregistered';atomic_json(run/'manifest.json',m);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');parser.add_argument('--test-only',action='store_true');parser.add_argument('--verify-only')
    args=parser.parse_args()
    if args.verify_only:run=ROOT/'data/signals'/args.verify_only
    else:
        evidence=run_tests()
        if args.test_only:return
        if not args.run_id:parser.error('--run-id required')
        run=build(args.run_id,evidence)
    from src.verify_panel_analysis_run import verify_panel_analysis_run
    try:verification=verify_panel_analysis_run(run)
    except Exception as error:
        if not args.verify_only:
            atomic_json(run/'verification_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
            m=read(run/'manifest.json');m['status']='failed_independent_verification'
            m['outputs']['verification_failure.json']=sha256(run/'verification_failure.json');atomic_json(run/'manifest.json',m)
        raise
    require(verification['status']=='passed','Verification did not pass')
    if not args.verify_only:
        atomic_json(run/'verification.json',verification);m=read(run/'manifest.json');m['status']=STATUS
        m['outputs']['verification.json']=sha256(run/'verification.json');atomic_json(run/'manifest.json',m)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()

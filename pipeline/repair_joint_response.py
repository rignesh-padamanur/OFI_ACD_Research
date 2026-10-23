"""Stage07m: verified descriptive J/C/I accounting; no empirical model fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import io
import json
import re
import shutil
import traceback
import unittest
import numpy as np
import pandas as pd
from src.data_preparation import atomic_json, sha256 as file_sha256

ROOT = Path(__file__).resolve().parent
STAGE = '07m_joint_response_accounting'
STATUS = 'verified_descriptive_joint_response_accounting'
SCOPE = 'fixed_baseline_joint_endpoint_descriptions_without_new_inference'
POINTER = 'data/active_joint_response_audit.json'
CURRENT = 'current_joint_response_audit'
NEXT = ('Assess dependence and joint uncertainty for the fixed response summaries and existing state equation; '
        'then complete reporting/reproducibility closeout. No detector, model or threshold selection. '
        'Causal identification and paper readiness are not implied by completing these steps.')
FRAMES = ['membership', 'summary', 'endpoint_counts', 'transition_counts', 'daily']
START = '2024-05-06T00:00:00+00:00'
END = '2026-08-01T00:00:00+00:00'


def read(path): return json.loads(Path(path).read_text())
def sha256(path): return file_sha256(Path(path))
def require(ok, message):
    if not ok: raise ValueError(message)


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT/'reports/STAGE07M_JOINT_RESPONSE_PROTOCOL.md',
             *sorted((ROOT/'src').glob('*.py')), *sorted((ROOT/'tests').glob('test_*.py'))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def resolve_inputs():
    prior = ROOT/'data/signals'/read(ROOT/'data/active_direction_cancellation_audit.json')['direction_run_id']
    oldm = read(prior/'manifest.json'); record = read(ROOT/'reports/stage07l_final_integrity.json')
    require(record['status'] == 'research_and_documentation_verified', 'Previous release incomplete')
    master = read(ROOT/'reports/pipeline_master_statistics.json')
    require(record['run_id']==oldm['run_id']==prior.name and oldm['status']=='verified_declared_direction_cancellation_sensitivity'
            and sha256(prior/'manifest.json')==record['input_manifest_sha256']
            and sha256(ROOT/'reports/pipeline_master_statistics.json')==record['registered_master_sha256']
            and sha256(ROOT/'data/active_direction_cancellation_audit.json')==record['registered_pointer_sha256'],
            'Prior registered identity differs')
    for name,digest in oldm['outputs'].items(): require(sha256(prior/name)==digest,'Prior output changed: '+name)
    for key,item in oldm['inputs'].items(): require(sha256(item['path'])==item['sha256'],'Prior input changed: '+key)
    oldtests=read(prior/'test_evidence.json')
    expected_tests={Path(n).name:h for n,h in oldm['code_sha256'].items() if n.startswith('tests/test_')}
    require(oldtests['status']=='passed' and oldtests['sources_unchanged_during_tests'] is True
            and oldtests['test_source_sha256']==expected_tests,'Prior executed-test evidence differs')
    entry=master['current_direction_cancellation_audit']
    require(entry['run_id']==prior.name and entry['statistics']==read(prior/'statistics.json')
            and entry['verification']==read(prior/'verification.json'),'Prior master entry differs')
    for name,digest in record['documentation_records_sha256'].items(): require(sha256(ROOT/name)==digest,'Prior documentation changed: '+name)
    inputs = {}; parents = {}
    def add(key, path, expected=None):
        path = Path(path); digest = sha256(path)
        require(expected is None or digest == expected, 'Changed source: '+key)
        inputs[key] = {'path': str(path.resolve()), 'sha256': digest}
    for label, pointer, key, current in [
        ('panel','active_panel_analysis_audit.json','analysis_run_id','current_panel_analysis_audit'),
        ('response','active_panel_response_audit.json','response_run_id','current_panel_response_audit'),
        ('estimation','active_continuation_estimation_audit.json','estimation_run_id','current_continuation_estimation_audit')]:
        run = ROOT/'data/signals'/read(ROOT/'data'/pointer)[key]; m = read(run/'manifest.json')
        entry = master[current]
        require(m['run_id'] == entry['run_id'] == run.name and m['status'] == entry['status']
                and entry['verification']['status'] == 'passed', 'Parent identity/status differs')
        parents[label] = m
        add(label+'/manifest.json', run/'manifest.json')
        names = ['statistics.json','verification.json']
        if label == 'panel': names += ['samples.parquet']
        if label == 'response': names += ['anchors.parquet']+[n for n in m['outputs'] if n.startswith('paths/') and n.endswith('.parquet')]
        if label == 'estimation': names += [f'c_{h}/baseline/events.parquet' for h in (60,300)]
        for name in names: add(label+'/'+name, run/name, m['outputs'][name])
        require(entry['statistics'] == read(run/'statistics.json') and entry['verification'] == read(run/'verification.json'), 'Parent ledger differs')
    p,r,e = [parents[x] for x in ('panel','response','estimation')]
    require(p['parent_response'] == r['run_id'] and e['parent_analysis'] == p['run_id'], 'Parent chain differs')
    require(p['inputs']['response_manifest'] == inputs['response/manifest.json']
            and p['inputs']['anchors'] == inputs['response/anchors.parquet'], 'Response source binding differs')
    for key,item in p['inputs'].items():
        if key.startswith('path_'):
            require(item == inputs['response/paths/'+Path(item['path']).name], 'Path source binding differs')
    require(e['analysis_start'] == START and e['end_exclusive'] == END, 'Analysis dates differ')
    add('prior/manifest.json', prior/'manifest.json')
    add('prior/final_integrity.json', ROOT/'reports/stage07l_final_integrity.json')
    for name,digest in oldm['code_sha256'].items(): require(sha256(ROOT/name)==digest, 'Prior source changed: '+name)
    return parents, inputs


def source_lineage(samples, inputs):
    """Recheck direct registered prices and original fitted-response keys, not raw ticks."""
    a = pd.read_parquet(inputs['response/anchors.parquet']['path']).sort_values('episode_id').reset_index(drop=True)
    require(not a.episode_id.duplicated().any(), 'Duplicate anchors')
    files = [item['path'] for key,item in sorted(inputs.items()) if key.startswith('response/paths/')]
    paths = pd.concat([pd.read_parquet(f, filters=[('horizon_sec','in',[0,60,300])]) for f in files], ignore_index=True)
    paths = paths.sort_values(['episode_id','horizon_sec']).reset_index(drop=True)
    s = samples.sort_values(['episode_id','horizon_sec']).reset_index(drop=True)
    require(not paths.duplicated(['episode_id','horizon_sec']).any(), 'Duplicate response paths')
    fields = ['episode_id','bin_start','detected_at','horizon_sec','target_at','endpoint_status',
              'endpoint_quote_at','endpoint_age_sec','pre_age_sec','detection_age_sec',
              'initial_pips','continuation_pips','total_pips','continuation_quote_available',
              'total_quote_available','triplet_quote_available']
    pd.testing.assert_frame_equal(s[fields], paths[fields], check_exact=True)
    aa = a.set_index('episode_id').loc[paths.episode_id].reset_index(drop=True)
    require(np.array_equal(paths.bin_start,aa.bin_start) and np.array_equal(paths.detected_at,aa.available_at)
            and np.array_equal(paths.direction,aa.direction,equal_nan=True), 'Anchor clock/direction differs')
    chosen = s.main_eligible_continuation & s.horizon_sec.isin([60,300])
    pp, ap, ss = paths.loc[chosen], aa.loc[chosen], s.loc[chosen]
    require(pp.direction.isin([-1,1]).all(), 'Main sample must be directed')
    for mark,bound in [(ap.pre_mark_at,pp.bin_start),(ap.detection_mark_at,pp.detected_at),(pp.endpoint_quote_at,pp.target_at)]:
        require((mark.isna() | mark.lt(bound)).all(), 'Strict-left source clock differs')
    for field, raw in [('initial_pips', (ap.detection_mid-ap.pre_mid)*10000),
                       ('continuation_pips', (pp.endpoint_mid-ap.detection_mid)*10000),
                       ('total_pips', (pp.endpoint_mid-ap.pre_mid)*10000)]:
        selected = np.isfinite(ss[field])
        expected = pp.direction*raw
        np.testing.assert_array_equal(ss.loc[selected,field].to_numpy(),expected.loc[selected].to_numpy(),err_msg='Direct prices differ: '+field)
    for h in [60,300]:
        old = pd.read_parquet(inputs[f'estimation/c_{h}/baseline/events.parquet']['path'], columns=['episode_id','bin_start','detected_at','target_at','continuation_pips']).sort_values('episode_id').reset_index(drop=True)
        now = s.loc[s.horizon_sec.eq(h)&s.main_eligible_continuation,old.columns].sort_values('episode_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(now,old,check_exact=True)
    return {'status':'passed','registered_selected_horizon_rows':len(s),'source_anchor_rows':len(a),
            'main_continuation_rows':int(chosen.sum()),'path_files':len(files),
            'direct_mark_arithmetic_checked':True,'baseline_estimation_keys_unchanged':True,
            'raw_tick_reconstruction_performed':False}


def integrity(run):
    m = read(Path(run)/'manifest.json')
    for key,item in m['inputs'].items(): require(sha256(item['path'])==item['sha256'],'Input changed: '+key)
    for name,digest in m['code_sha256'].items(): require(sha256(ROOT/name)==digest,'Frozen code changed: '+name)
    for name,digest in m['outputs'].items(): require(sha256(Path(run)/name)==digest,'Output changed: '+name)
    older = {n:h for n,h in m['previous_files'].items() if n.startswith('data/active_')}
    require(len(older)==29 and all(sha256(ROOT/n)==h for n,h in older.items()),'Prior pointer changed')
    return m


def main(run_id):
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]+',run_id)), 'Invalid run ID')
    run=ROOT/'data/signals'/run_id
    require(not run.exists(), 'Run already exists; preserve it')
    parents,inputs=resolve_inputs()
    snapshot=ROOT/'reports/repair_history'/('2026-10-02_pre_'+run_id)
    require(not snapshot.exists(), 'Snapshot already exists')
    previous=sorted((ROOT/'data').glob('active_*.json'))+[ROOT/'reports/pipeline_master_statistics.json',ROOT/'REPAIR_WORKFLOW.md',ROOT/'reports/PIPELINE_REPAIR_LOG.md']
    saved={str(p.relative_to(ROOT)):sha256(p) for p in previous}
    for p in previous:
        dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(snapshot/'preserved_files.json',saved)
    run.mkdir(parents=True)
    m={'run_id':run_id,'stage':STAGE,'status':'started','scope':SCOPE,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'primary_policy':'structural_only','analysis_start':START,'end_exclusive':END,
       **{'parent_'+k:v['run_id'] for k,v in parents.items()},'inputs':inputs,'code_sha256':source_hashes(),
       'previous_files':saved,'snapshot_path':str(snapshot),'outputs':{},'empirical_models_fitted':0,
       'uncertainty_computed':False,'production_inference_approved':False,'downstream_ready':False}
    atomic_json(run/'manifest.json',m)
    try:
        stream=io.StringIO();res=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        evidence={'status':'passed' if res.wasSuccessful() else 'failed','tests_run':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),
                  'output':stream.getvalue(),'executed_source_sha256':m['code_sha256'],'sources_unchanged_during_tests':source_hashes()==m['code_sha256']}
        atomic_json(ROOT/'reports/tests/stage07m_tests.json',evidence);atomic_json(run/'test_evidence.json',evidence)
        print(stream.getvalue(),flush=True)
        require(res.wasSuccessful() and evidence['sources_unchanged_during_tests'],'Tests failed or code changed')
        samples=pd.read_parquet(inputs['panel/samples.parquet']['path'])
        lineage=source_lineage(samples,inputs);atomic_json(run/'source_lineage.json',lineage)
        from src.joint_response import build_joint_response
        from src.verify_joint_response import verify_joint_response
        actual=build_joint_response(samples,START,END)
        for name in FRAMES: actual[name].to_parquet(run/(name+'.parquet'),index=False)
        atomic_json(run/'statistics.json',actual['statistics'])
        loaded={n:pd.read_parquet(run/(n+'.parquet')) for n in FRAMES}
        loaded['statistics']=read(run/'statistics.json')
        verification=verify_joint_response(loaded,samples,START,END)
        require(verification['status']=='passed','Independent verification failed')
        verification.update(source_lineage=lineage,executed_tests=res.testsRun,empirical_models_fitted=0,
                            production_inference_approved=False,downstream_ready=False)
        atomic_json(run/'verification.json',verification)
        for name in [*[n+'.parquet' for n in FRAMES],'statistics.json','verification.json','test_evidence.json','source_lineage.json']:
            m['outputs'][name]=sha256(run/name)
        m['status']=STATUS;atomic_json(run/'manifest.json',m);integrity(run)
        print(json.dumps({'status':STATUS,'run_id':run_id,'tests':res.testsRun,'empirical_models_fitted':0}),flush=True)
    except Exception:
        atomic_json(run/'failure.json',{'status':'failed','traceback':traceback.format_exc(),'empirical_models_fitted':0})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args().run_id)

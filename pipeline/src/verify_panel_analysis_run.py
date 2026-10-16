"""Independent saved-artifact verification for stage07f; never fits responses."""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from src.data_preparation import sha256
from src.verify_analysis_ledger import _compare, _json_compare

ROOT=Path(__file__).resolve().parents[1]
SCOPE='full_panel_accounting_later_reanalysis_inputs_only'


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise AssertionError(message)


def lineage(run):
    run=Path(run).resolve();m=read(run/'manifest.json');cache={}
    require(run.parent==(ROOT/'data/signals').resolve() and run.name==m['run_id'],'Run identity/path differs')
    require(m['stage']=='07f_full_panel_analysis_inputs' and m['scope']==SCOPE and m['primary_policy']=='structural_only','Stage/scope differs')
    require(m['status'] in ['built_pending_independent_verification','verified_full_panel_analysis_inputs'],'Run not built/verified')
    require(m['production_inference_approved'] is False and m['downstream_ready'] is False,'Inference gate promoted')
    def check(path,h):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        require(cache[path]==h,'Changed registered artifact: '+str(path))
    required={'ledger_anchors.parquet','ledger_outcomes.parquet','ledger_specification.json','adapter.json',
        'scaling.json','features.parquet','samples.parquet','design_rows.parquet','calendar_support.parquet',
        'schemas.json','specification.json','development_statistics.json','daily_calendar.parquet',
        'legacy_reconciliation.json','statistics.json','test_evidence.json'}
    required|={f'dependence/{h}/{name}.parquet' for h in [0,60,300] for name in ['intervals','pairs','components','calendar']}
    require(required.issubset(m['outputs']) and set(m['outputs']).issubset(required|{'verification.json','review.md'}),'Missing/unexpected outputs')
    for name,h in m['outputs'].items():check(run/name,h)
    for record in m['inputs'].values():check(record['path'],record['sha256'])
    for name,h in m['code_sha256'].items():check(run/'code'/name,h);check(ROOT/name,h)
    require({'repair_panel_analysis.py','src/panel_analysis.py','src/verify_panel_analysis.py',
        'src/verify_panel_analysis_run.py','reports/STAGE07F_PANEL_ANALYSIS_PROTOCOL.md'}.issubset(m['code_sha256']),'Source snapshot missing')
    t=read(run/'test_evidence.json')
    require(t['status']=='passed' and t['tests_run']>0 and t['failures']==t['errors']==0,'Tests did not pass')
    require(set(t['test_source_sha256'])=={p.name for p in (ROOT/'tests').glob('test_*.py')}
        =={p.name for p in (run/'full_test_sources').glob('test_*.py')},'Test-source set changed')
    for name,h in t['test_source_sha256'].items():
        check(run/'full_test_sources'/name,h);check(ROOT/'tests'/name,h)
        require(m['code_sha256']['tests/'+name]==h,'Test evidence/source differs')
    parents={}
    for prefix,status,files in [
        ('response','verified_full_panel_price_responses',{'anchors':'anchors.parquet',**{f'path_{i:03d}':f'paths/cycle_{i:03d}.parquet' for i in range(60)}}),
        ('measurement','verified_full_panel_fixed_onset_measurement',{'measurements':'measurements.parquet','directions':'direction.parquet'}),
        ('scaling','verified_development_regression_inputs',{'scaling':'scaling.json','legacy_features':'features.parquet','legacy_samples':'samples.parquet'}),
    ]:
        path=Path(m['inputs'][prefix+'_manifest']['path']);parent=read(path)
        require(parent['status']==status and parent['run_id']==m['parent_'+prefix]==path.parent.name,'Parent identity/status differs')
        require(parent['primary_policy']=='structural_only','Parent policy differs')
        for key,name in {prefix+'_verification':'verification.json',prefix+'_statistics':'statistics.json',**files}.items():
            r=m['inputs'][key]
            require(Path(r['path']).resolve()==(path.parent/name).resolve() and r['sha256']==parent['outputs'][name],'Parent output lineage differs: '+key)
        require(read(m['inputs'][prefix+'_verification']['path'])['status']=='passed','Parent verification failed')
        parents[prefix]=parent
    rm,mm,sm=[parents[k] for k in ['response','measurement','scaling']]
    require(rm['parent_measurement']==mm['run_id'] and rm['parent_chronological_panel']==mm['parent_chronological_panel']
        and rm['inputs']['onsets']==mm['inputs']['onsets'] and rm['inputs']['measurement_manifest']==m['inputs']['measurement_manifest'],
        'Response/measurement origin differs')
    require(sm['inputs']['parent_manifest']==m['inputs']['legacy_ledger_manifest'],'Original development ledger origin differs')
    old=read(m['inputs']['legacy_ledger_manifest']['path'])
    require(old['parent_price_response_audit']==Path(rm['inputs']['legacy_manifest']['path']).parent.name,'Legacy response ancestor differs')
    require(sha256(run/'scaling.json')==m['inputs']['scaling']['sha256'],'Frozen scaling bytes changed')
    require(m['horizons_seconds']==[0,60,300] and m['deployment_start']=='2024-04-22T00:00:00+00:00'
        and m['analysis_start']=='2024-05-06T00:00:00+00:00' and m['deployment_end_exclusive']=='2026-08-01T00:00:00+00:00','Chronology differs')
    snapshot=Path(m['snapshot_path']);require(read(snapshot/'preserved_files.json')==m['previous_files'],'Preservation index differs')
    pointers=0
    for name,h in m['previous_files'].items():
        check(snapshot/name,h)
        if name.startswith('data/active_'):check(ROOT/name,h);pointers+=1
    require(pointers==22 and len(m['previous_files'])==25,'Prior pointer/record set differs')
    return m,len(cache)


def reference_legacy(inputs,features,samples):
    old_f=pd.read_parquet(inputs['legacy_features']['path']).sort_values('episode_id').reset_index(drop=True)
    old_s=pd.read_parquet(inputs['legacy_samples']['path']).sort_values(['episode_id','horizon_sec']).reset_index(drop=True)
    f=features.set_index('episode_id',drop=False).loc[old_f.episode_id].reset_index(drop=True)
    require(set(old_f).issubset(f),'Historical feature columns lost')
    _compare(f[list(old_f)],old_f)
    index=pd.MultiIndex.from_frame(old_s[['episode_id','horizon_sec']])
    s=samples.set_index(['episode_id','horizon_sec'],drop=False).loc[index].reset_index(drop=True)
    require(set(old_s).issubset(s),'Historical sample columns lost')
    historical=(old_s.target_at<=pd.Timestamp('2024-05-06T00:00:00Z')).to_numpy()
    _compare(s.loc[historical,list(old_s)].reset_index(drop=True),old_s.loc[historical].reset_index(drop=True))
    restored=[int(s.iloc[i].episode_id) for i in range(len(s)) if not historical[i]
              and not bool(old_s.iloc[i].eligible_continuation) and bool(s.iloc[i].eligible_continuation)]
    return {'legacy_features_checked':len(old_f),'shared_feature_columns_checked':len(old_f.columns),
        'legacy_sample_rows':len(old_s),'historical_sample_rows_checked':int(sum(historical)),
        'restored_continuation_rows':len(restored),'restored_event_ids':sorted(set(restored)),
        'legacy_artifacts_preserved':True,'historical_features_and_outcomes_reconciled':True,
        'development_design_diagnostics_recomputed':True}


def verify_panel_analysis_run(run):
    from src.verify_panel_analysis import (reference_panel_ledger, reference_panel_inputs,
        reference_panel_dependence, reference_daily_calendar, reference_dep_summary)
    run=Path(run);m,hashes=lineage(run);inputs=m['inputs']
    paths=ds.dataset([inputs[f'path_{i:03d}']['path'] for i in range(60)],format='parquet').to_table(
        filter=ds.field('horizon_sec').isin([0,60,300])).to_pandas()
    expected=reference_panel_ledger(pd.read_parquet(inputs['anchors']['path']),pd.read_parquet(inputs['measurements']['path']),
        pd.read_parquet(inputs['directions']['path']),paths)
    del paths;gc.collect()
    for name in ['anchors','outcomes']:_compare(pd.read_parquet(run/('ledger_'+name+'.parquet')),expected[name])
    for name,file in [('specification','ledger_specification.json'),('adapter','adapter.json')]:
        _json_compare(read(run/file),expected[name],exact=True)
    print('Independently reconstructed all panel ledger rows',flush=True)
    output=reference_panel_inputs(expected['anchors'],expected['outcomes'],read(run/'scaling.json'))
    for name,obj in output.items():
        if name=='statistics':continue
        if isinstance(obj,pd.DataFrame):_compare(pd.read_parquet(run/(name+'.parquet')),obj)
        else:_json_compare(read(run/(name+'.json')),obj,exact=True)
    print('Independently reconstructed all features, masks and main matrix diagnostics',flush=True)
    dep_stats=[];intervals=pairs=0
    for h,dep in reference_panel_dependence(output):
        for name,frame in dep.items():_compare(pd.read_parquet(run/'dependence'/str(h)/(name+'.parquet')),frame)
        dep_stats.append(reference_dep_summary(h,dep))
        intervals+=len(dep['intervals']);pairs+=len(dep['pairs'])
        print(f'Independently verified main dependence horizon{h}: {len(dep["pairs"]):,} pairs',flush=True)
        del dep;gc.collect()
    calendar=reference_daily_calendar(output);_compare(pd.read_parquet(run/'daily_calendar.parquet'),calendar)
    legacy=reference_legacy(inputs,output['features'],output['samples']);_json_compare(read(run/'legacy_reconciliation.json'),legacy,exact=True)
    statistics={'scope':SCOPE,**output['statistics'],'dependence':dep_stats,'legacy_reconciliation':legacy,
        'calendar':{'rows':len(calendar),'days':831,'development_days':14,'later_days':817},
        'empirical_models_fitted':0,'production_inference_approved':False,'downstream_ready':False}
    _json_compare(read(run/'statistics.json'),statistics,exact=True)
    require(len(expected['outcomes'])==3*len(expected['anchors'])==len(output['samples']),'Outcome row conservation failed')
    require(len(expected['anchors'])==len(output['features'])==read(inputs['response_statistics']['path'])['onset_rows'],'Parent onset conservation failed')
    require(len(calendar)==2493,'Full calendar grid missing')
    return {'status':'passed','scope':SCOPE,'hashes_verified':hashes,'prior_pointer_snapshots_verified':22,
        'anchor_rows_verified':len(expected['anchors']),'outcome_rows_verified':len(expected['outcomes']),
        'feature_rows_verified':len(output['features']),'sample_rows_verified':len(output['samples']),
        'design_rows_verified':len(output['design_rows']),'dependence_intervals_verified':intervals,
        'dependence_pairs_verified':pairs,'calendar_rows_verified':len(calendar),
        'legacy_feature_rows_verified':legacy['legacy_features_checked'],
        'legacy_historical_sample_rows_verified':legacy['historical_sample_rows_checked'],
        'methods':'Independent parent-key ledger reconstruction; independent feature/mask/reference-scaling and design-only linear algebra; exact sorted-start overlap enumeration and independent components; full zero-calendar and legacy reconciliation. Every saved row/statistic and registered lineage/source/test hash checked. No empirical response fitting.',
        'production_inference_approved':False,'downstream_ready':False}

"""Run the fixed stage07n mean uncertainty and saved-regression assessment."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import io
import json
import re
import shutil
import sys
import traceback
import unittest
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from src.data_preparation import atomic_json, sha256 as file_sha256
from src.joint_uncertainty import FRAMES

ROOT=Path(__file__).resolve().parent
STAGE='07n_joint_uncertainty_assessment'
STATUS='verified_provisional_calendar_score_uncertainty'
SCOPE='fixed_joint_means_and_existing_state_equation_uncertainty_assessment'
POINTER='data/active_joint_uncertainty_audit.json'
CURRENT='current_joint_uncertainty_audit'
START='2024-05-06T00:00:00+00:00'
END='2026-08-01T00:00:00+00:00'
NEXT=('Complete the fixed reporting and reproducibility closeout: map each RQ to its final table, '
      'reconcile original versus repaired methodology and notebook entry points, and record claim limits. '
      'Do not start another model or sensitivity search. Interval coverage, causality and JFE readiness remain unestablished.')


def read(path):return json.loads(Path(path).read_text())
def sha256(path):return file_sha256(Path(path))
def require(ok,message):
    if not ok:raise ValueError(message)


def source_hashes():
    files=[Path(__file__).resolve(),ROOT/'reports/STAGE07N_UNCERTAINTY_PROTOCOL.md',
           *sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'tests').glob('test_*.py'))]
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def resolve_inputs():
    from reports.stage07m_closeout import registered_evidence
    joint=ROOT/'data/signals'/read(ROOT/'data/active_joint_response_audit.json')['joint_response_run_id']
    jm,_,_,_,record=registered_evidence(joint)
    require(record['status']=='research_and_documentation_verified','Previous stage not closed')
    for name,digest in record['documentation_records_sha256'].items():
        require(sha256(ROOT/name)==digest,'Previous documentation changed: '+name)
    est=ROOT/'data/signals'/read(ROOT/'data/active_continuation_estimation_audit.json')['estimation_run_id']
    em=read(est/'manifest.json')
    master=read(ROOT/'reports/pipeline_master_statistics.json')
    require(jm['parent_estimation']==em['run_id']==est.name
            and master['current_continuation_estimation_audit']['run_id']==est.name,
            'Parent estimation identity differs')
    require(sha256(est/'manifest.json')==jm['inputs']['estimation/manifest.json']['sha256'],
            'Estimation manifest source changed')
    inputs={}
    def add(key,path,digest=None):
        require(Path(path).is_file(),'Missing input: '+key)
        actual=sha256(path)
        require(digest is None or digest==actual,'Input changed: '+key)
        inputs[key]={'path':str(Path(path).resolve()),'sha256':actual}
    for label,run,m,names in [('joint',joint,jm,['daily.parquet','membership.parquet','statistics.json','verification.json']),
                             ('estimation',est,em,['statistics.json','verification.json'])]:
        add(label+'/manifest.json',run/'manifest.json')
        for name in names:add(label+'/'+name,run/name,m['outputs'][name])
    for h in [60,300]:
        for model in ['baseline','curvature']:
            for name in ['daily_scores.parquet','bread.parquet','covariances.parquet','coefficients.parquet',
                         'contrasts.parquet','influence_daily.parquet','diagnostics.json','bandwidth.json']:
                key=f'c_{h}/{model}/{name}'
                add('estimation/'+key,est/key,em['outputs'][key])
    add('prior/final_integrity.json',ROOT/'reports/stage07m_final_integrity.json')
    return {'joint':joint.name,'estimation':est.name},inputs


def assess_regressions(inputs):
    """Recheck saved arithmetic, without estimating a coefficient or new lag."""
    records=[];assessment={}
    state=['variation_standardized','pressure_magnitude','aligned_prior_pressure',
           'variation_x_magnitude','variation_x_alignment']
    for h in [60,300]:
        for model in ['baseline','curvature']:
            key=f'c_{h}/{model}'
            def path(name):return inputs['estimation/'+key+'/'+name]['path']
            d=read(path('diagnostics.json'));bw=read(path('bandwidth.json'))
            daily=pd.read_parquet(path('daily_scores.parquet'))
            cols=d['columns'];g=daily[['score::'+c for c in cols]].to_numpy()
            dates=pd.date_range(START,END,freq='D',inclusive='left')
            require(pd.DatetimeIndex(daily.date).equals(dates) and daily.events.sum()==d['rows'],
                    'Saved regression calendar/counts differ')
            require(d['automatic_lag']==bw['lag'] and d['longer_lag']==max(1,2*bw['lag']),
                    'Saved regression lag rule differs')
            bread=pd.read_parquet(path('bread.parquet')).pivot(index='column_i',columns='column_j',values='value').loc[cols,cols].to_numpy()
            saved=pd.read_parquet(path('covariances.parquet'))
            coef=pd.read_parquet(path('coefficients.parquet'))
            contrasts=pd.read_parquet(path('contrasts.parquet'))
            for method in ['automatic','longer']:
                lag=d[method+'_lag'];meat=g.T@g
                for j in range(1,lag+1):
                    cross=g[j:].T@g[:-j];meat+=(1-j/(lag+1))*(cross+cross.T)
                expected=bread@meat@bread;expected=(expected+expected.T)/2
                c=saved.loc[saved.covariance_method.eq(method)].pivot(index='column_i',columns='column_j',values='value').loc[cols,cols].to_numpy()
                np.testing.assert_allclose(c,expected,rtol=1e-10,atol=1e-12)
                f=coef.loc[coef.covariance_method.eq(method)].set_index('column').loc[cols]
                np.testing.assert_allclose(f.se,np.sqrt(np.diag(c)),rtol=1e-10,atol=1e-12)
                selected=[]
                for name in state:
                    row=f.loc[name].to_dict();row['target']='coefficient:'+name;selected.append(row)
                for row in contrasts.loc[contrasts.covariance_method.eq(method)].to_dict('records'):
                    row['target']='branch:'+row['branch'];selected.append(row)
                require(len(selected)==7,'Saved regression target count differs')
                for row in selected:
                    require(row['lag']==lag and row['se']>0,'Saved interval lag/variance differs')
                    np.testing.assert_allclose([row['ci95_low'],row['ci95_high']],
                        [row['estimate']-norm.ppf(.975)*row['se'],row['estimate']+norm.ppf(.975)*row['se']],
                        rtol=1e-12,atol=1e-12)
                    records.append({'sample_key':f'c_{h}','model':model,'target':row['target'],
                        'method':method,'lag':lag,**{k:row[k] for k in ['estimate','se','ci95_low','ci95_high']}})
            influences=pd.read_parquet(path('influence_daily.parquet'))
            require(set(influences.target)=={s['target'] for s in d['influence_summary']},'Influence target set differs')
            for s in d['influence_summary']:
                f=influences.loc[influences.target.eq(s['target'])].sort_values('date')
                require(pd.DatetimeIndex(f.date).equals(dates) and np.array_equal(f.events,daily.events),'Influence dates/counts differ')
                z=f.influence.to_numpy();energy=z*z;shares=energy/energy.sum()
                np.testing.assert_allclose(f.energy,energy,rtol=1e-12,atol=1e-15)
                np.testing.assert_allclose(f.energy_share,shares,rtol=1e-12,atol=1e-15)
                require(s['maximum_day']==dates[int(np.argmax(energy))].isoformat(),'Influential date differs')
                np.testing.assert_allclose([s['maximum_day_energy_share'],s['total_energy']],
                    [shares.max(),energy.sum()],rtol=1e-12,atol=1e-15)
                for method in ['automatic','longer']:
                    maximum=max(abs(float(z[j:]@z[:-j]))/energy.sum()
                                for j in range(d[method+'_lag']+1,d['lag_inspection_max']+1))
                    np.testing.assert_allclose(maximum,s['max_abs_lag_product_beyond'][method],rtol=1e-12,atol=1e-15)
            assessment[key]={'rows':d['rows'],'automatic_lag':d['automatic_lag'],'longer_lag':d['longer_lag'],
                'influence_summary':d['influence_summary'],'score_lag_tail_summary':d['score_lag_tail_summary'],
                'saved_covariance_reconstruction':'passed','saved_concentration_reconstruction':'passed',
                'refitted':False,'confidence_interval_coverage_validated':False}
    return pd.DataFrame(records),assessment


def integrity(run):
    m=read(Path(run)/'manifest.json')
    for key,item in m['inputs'].items():require(sha256(item['path'])==item['sha256'],'Input changed: '+key)
    for name,digest in m['code_sha256'].items():require(sha256(ROOT/name)==digest,'Source changed: '+name)
    for name,digest in m['outputs'].items():require(sha256(Path(run)/name)==digest,'Output changed: '+name)
    old={n:h for n,h in m['previous_files'].items() if n.startswith('data/active_')}
    require(len(old)==30 and all(sha256(ROOT/n)==h for n,h in old.items()),'Prior pointers changed')
    return m


def main(run_id):
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]+',run_id)),'Invalid run ID')
    run=ROOT/'data/signals'/run_id;require(not run.exists(),'Run exists; preserve it')
    parents,inputs=resolve_inputs()
    snapshot=ROOT/'reports/repair_history'/('2026-10-02_pre_'+run_id)
    require(not snapshot.exists(),'Snapshot exists')
    previous=sorted((ROOT/'data').glob('active_*.json'))+[ROOT/'reports/pipeline_master_statistics.json',ROOT/'REPAIR_WORKFLOW.md',ROOT/'reports/PIPELINE_REPAIR_LOG.md']
    saved={str(p.relative_to(ROOT)):sha256(p) for p in previous}
    for p in previous:
        dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(snapshot/'preserved_files.json',saved);run.mkdir(parents=True)
    m={'run_id':run_id,'stage':STAGE,'status':'started','scope':SCOPE,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'primary_policy':'structural_only','analysis_start':START,'end_exclusive':END,
       **{'parent_'+k:v for k,v in parents.items()},'inputs':inputs,'code_sha256':source_hashes(),
       'previous_files':saved,'snapshot_path':str(snapshot),'outputs':{},'empirical_models_fitted':0,
       'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
       'uncertainty_computed':True,'production_inference_approved':False,'downstream_ready':False}
    atomic_json(run/'manifest.json',m)
    try:
        stream=io.StringIO();res=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        evidence={'status':'passed' if res.wasSuccessful() else 'failed','tests_run':res.testsRun,
            'failures':len(res.failures),'errors':len(res.errors),'output':stream.getvalue(),
            'executed_source_sha256':m['code_sha256'],'sources_unchanged_during_tests':source_hashes()==m['code_sha256']}
        atomic_json(ROOT/'reports/tests/stage07n_tests.json',evidence);atomic_json(run/'test_evidence.json',evidence)
        print(stream.getvalue(),flush=True)
        require(res.wasSuccessful() and evidence['sources_unchanged_during_tests'],'Tests failed or sources changed')
        from src.joint_uncertainty import build_joint_uncertainty
        from src.verify_joint_uncertainty import verify_joint_uncertainty
        daily=pd.read_parquet(inputs['joint/daily.parquet']['path']);membership=pd.read_parquet(inputs['joint/membership.parquet']['path'])
        actual=build_joint_uncertainty(daily,START,END)
        regression,assessment=assess_regressions(inputs)
        actual['statistics']['regression_assessment']=assessment
        for name in FRAMES:actual[name].to_parquet(run/(name+'.parquet'),index=False)
        regression.to_parquet(run/'regression_intervals.parquet',index=False)
        atomic_json(run/'statistics.json',actual['statistics'])
        loaded={n:pd.read_parquet(run/(n+'.parquet')) for n in FRAMES};loaded['statistics']=read(run/'statistics.json')
        require(loaded['statistics']['regression_assessment']==assessment,'Serialized regression assessment differs')
        verification=verify_joint_uncertainty(loaded,daily,membership,START,END)
        require(verification['status']=='passed','Independent verification failed')
        pd.testing.assert_frame_equal(pd.read_parquet(run/'regression_intervals.parquet'),regression,check_exact=True)
        verification.update(executed_tests=res.testsRun,empirical_models_fitted=0,
            regression_jobs_rechecked=4,regression_interval_rows_preserved=len(regression),
            regression_verification_scope='source_bound_rechecks_of_previously_independently_verified_results',
            production_inference_approved=False,downstream_ready=False)
        atomic_json(run/'verification.json',verification)
        for name in [*[n+'.parquet' for n in FRAMES],'regression_intervals.parquet','statistics.json','verification.json','test_evidence.json']:
            m['outputs'][name]=sha256(run/name)
        m['status']=STATUS;atomic_json(run/'manifest.json',m);integrity(run)
        print(json.dumps({'status':STATUS,'run_id':run_id,'tests':res.testsRun,'empirical_models_fitted':0}),flush=True)
    except Exception:
        atomic_json(run/'failure.json',{'status':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);main(parser.parse_args().run_id)

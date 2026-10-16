"""Stage10a: frozen sequence measurement, coverage and predictor-only diagnostics.

Never reads labels, response membership, future price responses or forecast
performance. This runner creates no active pointer and never updates the master.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from src.sequence_measurement import (ORIGIN_COLUMNS, PREDICTOR_COLUMNS,
    ANCHOR_COLUMNS, SPARSE_COLUMNS, QUOTE_COLUMNS, NUMERIC_BASELINE,
    CATEGORICAL_BASELINE, Q_COLUMNS, measure_pressure, measure_prior_prices,
    build_features, conditioning_diagnostics, sequence_statistics, require)
from src.panel_observations import read_quote_window

ROOT = Path(__file__).resolve().parent
RUN_ID = 'step10a_sequence_measurement_20261003'
PROTOCOL = 'reports/STEP10A_SEQUENCE_PROTOCOL.md'
SPEC = 'reports/STEP10A_SEQUENCE_SPEC.json'
FROZEN = 'reports/STEP10A_FROZEN_DESIGN.json'
LITERATURE = 'reports/STEP10A_LITERATURE_GATE.md'
START = pd.Timestamp('2024-05-06T00:00:00Z')
SOURCES = ['repair_sequence_measurement.py', 'src/sequence_measurement.py',
    'src/panel_observations.py', 'tests/test_sequence_measurement.py',
    PROTOCOL, SPEC, FROZEN, LITERATURE]
STATUS = 'measurement_complete_pending_independent_verification'


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def fingerprint(path, expected=None):
    p=Path(path);digest=sha(p)
    require(expected is None or digest==expected, 'Source hash changed: '+str(p))
    return {'path':str(p.resolve()),'sha256':digest,'bytes':p.stat().st_size}


def validate_spec():
    spec=read(ROOT/SPEC); frozen=read(ROOT/FROZEN)
    expected={'specification_id':RUN_ID,'status':'frozen_before_production_measurement',
        'scope':'literature_design_and_measurement_only','window_sec':300,'bin_sec':5,'bins':60,
        'clock':'first_alert_s_plus_1_second','feature_window':'[s-300,s)','policy':'structural_only',
        'aggregation':'math.fsum_of_five_stored_one_second_ofi_values',
        'measure':'adjacent_sign_product_minus_fixed_composition_reference',
        'reference':'((sum_signs)^2-sum_squared_signs)/(60*59)', 'zero_sign':0,
        'drop_zero_bins':False,'all_one_sign_feature_value':0,
        'include_composition_reference_in_benchmark':True,'recent_window_seconds':[5,30],
        'return_window_seconds':[5,30,300], 'primary_response_for_later_task':'C300',
        'primary_loss_for_later_task':'squared_error','sequence_candidates':1,
        'numeric_reconciliation_rtol':1e-8,'numeric_reconciliation_atol':1e-8,
        'evaluation_start':START.isoformat(),'empirical_response_models_permitted':0,
        'outcome_association_screening_permitted':False,'automatic_estimation_after_measurement':False,
        'novelty_established':False}
    require(all(spec.get(k)==v for k,v in expected.items()),'Frozen specification differs from implementation')
    require('Protocol status: frozen before production measurement' in (ROOT/PROTOCOL).read_text(), 'Missing protocol freeze')
    for relative,digest in frozen['files_sha256'].items():fingerprint(ROOT/relative,digest)
    require(set(frozen['files_sha256'])=={PROTOCOL,SPEC,LITERATURE},'Unexpected design freeze files')
    require(not frozen['new_outcome_associations_inspected'] and frozen['new_response_models_fitted']==0,'Invalid design freeze scope')
    return spec


def input_metadata():
    paths={name:ROOT/'data/signals'/folder for name,folder in [
        ('origins','step09a_prediction_ledger_20261003'),('anchors','step07e_panel_responses_20261001'),
        ('scaling','step07f_panel_analysis_20261001'),('sparse','step03_quote_imbalance_20260927'),
        ('clock','step04a_signal_clock_20260927_v2')]}
    m={k:read(p/'manifest.json') for k,p in paths.items()}
    expected={'origins':'verified_prediction_input_ledger','anchors':'verified_full_panel_price_responses',
        'scaling':'verified_full_panel_analysis_inputs','sparse':'verified_recorded_quote_proxy',
        'clock':'verified_supported_signal_clock'}
    require(all(m[k]['status']==status for k,status in expected.items()),'Unverified inherited parent')
    result={k+'_manifest':fingerprint(path/'manifest.json') for k,path in paths.items()}
    for key,manifest_key,parentfile in [('origins','origins','origins.parquet'),('predictors','origins','predictors.parquet'),
        ('anchors','anchors','anchors.parquet'),('scaling','scaling','scaling.json'),
        ('sparse','sparse','quote_ofi_1s.parquet'),('clock','clock','signal_clock_1s.parquet')]:
        result[key]=fingerprint(paths[manifest_key]/parentfile,m[manifest_key]['outputs'][parentfile])
    for key in ['clean_quotes','support_intervals']:
        item=m['anchors']['inputs'][key];result[key]=fingerprint(item['path'],item['sha256'])
    require(result['anchors_manifest']['sha256']==m['origins']['inputs']['response_manifest']['sha256'],'Origin response manifest differs')
    require(result['scaling_manifest']['sha256']==m['origins']['inputs']['panel_manifest']['sha256'],'Origin panel manifest differs')
    require(result['sparse_manifest']['sha256']==m['clock']['inputs']['signal_manifest']['sha256'],'Clock signal manifest differs')
    require(result['anchors']['sha256']==m['origins']['inputs']['anchors']['sha256'],'Origin anchor lineage differs')
    require(result['scaling']['sha256']==m['origins']['inputs']['scaling']['sha256'],'Origin scaling lineage differs')
    require(result['clean_quotes']['sha256']==m['sparse']['inputs']['parent_clean']['sha256'],'Clean/sparse lineage differs')
    require(result['support_intervals']['sha256']==m['sparse']['inputs']['support']['sha256'],'Support/sparse lineage differs')
    require(result['sparse']['sha256']==m['clock']['inputs']['sparse_bins']['sha256'],'Sparse/clock lineage differs')
    require(result['support_intervals']['sha256']==m['clock']['inputs']['support']['sha256'],'Support/clock lineage differs')
    for key,path in [('protocol',PROTOCOL),('specification',SPEC),('frozen_design',FROZEN),('literature_gate',LITERATURE)]:
        result[key]=fingerprint(ROOT/path)
    return result


def tests():
    cmd=[sys.executable,'-m','unittest','tests.test_sequence_measurement','-v']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env)
    log=result.stdout+result.stderr
    require(result.returncode==0,'Focused tests failed:\n'+log)
    match=re.search(r'Ran (\d+) tests?',log)
    require(match is not None and '\nOK\n' in log,'Missing successful focused test receipt')
    return log,int(match.group(1))


def distribution(values):
    v=pd.Series(values).dropna().astype(float)
    if not len(v):return {'count':0}
    require(np.isfinite(v).all(),'Nonfinite summary input')
    qs=v.quantile([.01,.25,.5,.75,.99])
    return {'count':len(v),'minimum':float(v.min()),'mean':float(v.mean()),
        'standard_deviation_ddof0':float(v.std(ddof=0)),'p01':float(qs.loc[.01]),
        'p25':float(qs.loc[.25]),'median':float(qs.loc[.5]),'p75':float(qs.loc[.75]),
        'p99':float(qs.loc[.99]),'maximum':float(v.max())}


def coverage_row(name, frame):
    eligible=frame.inherited_origin_eligible
    return {'cohort':name,'origins':len(frame),'inherited_origin_eligible':int(eligible.sum()),
        'pressure_sequence_available':int(frame.pressure_sequence_available.sum()),
        'prior_price_available':int(frame.prior_price_available.sum()),
        'measurement_available':int(frame.measurement_available.sum()),'combined_eligible':int(frame.combined_eligible.sum()),
        'new_measurement_attrition_among_inherited_eligible':int((eligible&~frame.measurement_available).sum()),
        'all_zero_sequences':int((frame.pressure_sequence_available&frame.all_zero_sequence).sum()),
        'constant_nonzero_sequences':int((frame.pressure_sequence_available&frame.constant_nonzero_sequence).sum()),
        'order_degenerate_sequences':int((frame.pressure_sequence_available&frame.order_degenerate).sum()),
        'pressure_status_counts':{str(k):int(v) for k,v in frame.pressure_sequence_status.value_counts().items()},
        'price_status_counts':{str(k):int(v) for k,v in frame.prior_price_status.value_counts().items()},
        'measurement_status_counts':{str(k):int(v) for k,v in frame.measurement_status.value_counts().items()}}


def derive_statistics(frame):
    cohorts={'all':frame,'inherited_eligible':frame.loc[frame.inherited_origin_eligible],
        'initial_training':frame.loc[frame.forecast_at.lt(START)],
        'later_deployment':frame.loc[frame.forecast_at.ge(START)],
        'initial_training_inherited_eligible':frame.loc[frame.forecast_at.lt(START)&frame.inherited_origin_eligible],
        'later_deployment_inherited_eligible':frame.loc[frame.forecast_at.ge(START)&frame.inherited_origin_eligible]}
    coverage=[coverage_row(name,f) for name,f in cohorts.items()]
    columns=['sequence_z','adjacent_agreement','composition_reference', 'sign_sum','nonzero_fraction',
        'gross_pressure_scaled','prior_quote_updates','prior_max_quote_age_seconds','repeated_mark_fraction',
        'pre_pressure_reconstruction_difference','pre_rv_reconstruction_difference']
    summaries={name:{column:distribution(f[column]) for column in columns} for name,f in cohorts.items()}
    numerical={name:{'available_sequences':int(f.pressure_sequence_available.sum()),
        'numpy_fsum_sign_differences':int(f.numpy_fsum_sign_differences.sum()),
        'origins_with_numpy_fsum_sign_difference':int(f.numpy_fsum_sign_differences.gt(0).sum()),
        'bound_sensitive_nonzero_subbins':int(f.bound_sensitive_nonzero_subbins.sum()),
        'origins_with_bound_sensitive_nonzero':int(f.bound_sensitive_nonzero_subbins.gt(0).sum()),
        'exact_zero_subbins_with_nonzero_gross':int(f.exact_zero_subbins_with_nonzero_gross.sum()),
        'maximum_ordinary_fsum_disagreement':float(f.max_sum_disagreement.max()) if f.max_sum_disagreement.notna().any() else None,
        'maximum_absolute_inherited_pressure_disagreement':float(f.pre_pressure_reconstruction_difference.abs().max()) if f.pre_pressure_reconstruction_difference.notna().any() else None,
        'maximum_absolute_inherited_rv_disagreement':float(f.pre_rv_reconstruction_difference.abs().max()) if f.pre_rv_reconstruction_difference.notna().any() else None}
        for name,f in cohorts.items()}
    return {'all_origins':len(frame),'coverage':coverage,'distributions':summaries,'numerical_diagnostics':numerical,
        'design_support':{name:conditioning_diagnostics(cohorts[name]) for name in ['all','initial_training','later_deployment']},
        'numeric_reconciliation_rtol':1e-8,'numeric_reconciliation_atol':1e-8,
        'local_aggregation_bound_covers_inherited_bin_error':False,'economic_zero_threshold_used':False,
        'source_scope':'recorded quote-flow cache under inherited structural support, not certified complete market feed',
        'response_models_fitted':0,'forecasts_computed':0,'outcome_association_tests':0,
        'labels_read':False,'future_availability_read':False,'novelty_established':False,
        'causal_identification_established':False,'automatic_estimation_authorized':False}


def saved_checks(run,manifest,expected_statistics):
    for key,entry in manifest['inputs'].items():fingerprint(entry['path'],entry['sha256'])
    for relative,digest in manifest['source_sha256'].items():
        fingerprint(ROOT/relative,digest);fingerprint(run/'code'/relative,digest)
    for relative,digest in manifest['outputs'].items():fingerprint(run/relative,digest)
    f=pd.read_parquet(run/'features.parquet');q=pd.read_parquet(run/'sequences.parquet')
    marks=pd.read_parquet(run/'prior_price_marks.parquet')
    require(len(f)==135724 and f.episode_id.is_unique and np.array_equal(f.episode_id,q.episode_id)
            and np.array_equal(f.episode_id,marks.episode_id),'Serialized origin conservation failed')
    good=f.pressure_sequence_available.to_numpy()
    expected=sequence_statistics(q.loc[good,Q_COLUMNS].to_numpy())
    for column in expected:
        np.testing.assert_array_equal(f.loc[good,column].to_numpy(),expected[column].to_numpy(),err_msg=column)
    require(q.loc[~good,Q_COLUMNS].isna().all(axis=None),'Unavailable sequence contains values')
    require(derive_statistics(f)==expected_statistics,'Saved statistics do not reproduce')
    return {'status':'passed','verified_at_utc':now(),'all_origins':len(f),'source_files':len(manifest['inputs']),
        'code_files':len(manifest['source_sha256']),'output_files':len(manifest['outputs']),
        'scope':'serialized hashes/identities/statistics; separate independent full-source verification still required',
        'response_models_fitted':0}


def build():
    validate_spec();run=ROOT/'data/signals'/RUN_ID
    require(not run.exists(),'Run exists: never overwrite a production/failed run')
    log,count=tests();meta=input_metadata()
    run.mkdir();hashes={}
    for relative in SOURCES:
        target=run/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);hashes[relative]=sha(target)
    (run/'test_execution.log').write_text(log)
    write(run/'test_evidence.json',{'status':'passed','tests_passed':count,'log_sha256':sha(run/'test_execution.log')})
    m={'run_id':RUN_ID,'stage':'10a_sequence_measurement','status':'running','started_at_utc':now(),
       'inputs':meta,'source_sha256':hashes,'protocol_sha256':meta['protocol']['sha256'],
       'specification_sha256':meta['specification']['sha256'],'frozen_design_sha256':meta['frozen_design']['sha256'],
       'read_allowlists':{'origins':ORIGIN_COLUMNS,'predictors':PREDICTOR_COLUMNS,'anchors':ANCHOR_COLUMNS,
                         'sparse':SPARSE_COLUMNS,'clean_quotes':QUOTE_COLUMNS},
       'numeric_baseline_columns':NUMERIC_BASELINE,'categorical_baseline_columns':CATEGORICAL_BASELINE,
       'extension_columns':['sequence_z'],'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},
       'future_response_values_read':False,'future_outcome_availability_read':False,
       'response_models_fitted':0,'hypothesis_tests':0,'forecast_performance_computed':False,
       'active_pointer_changed':False,'master_statistics_changed':False,'novelty_established':False,
       'automatic_estimation_authorized':False}
    write(run/'manifest.json',m)
    try:
        origins=pd.read_parquet(meta['origins']['path'],columns=ORIGIN_COLUMNS)
        predictors=pd.read_parquet(meta['predictors']['path'],columns=PREDICTOR_COLUMNS)
        anchors=pd.read_parquet(meta['anchors']['path'],columns=ANCHOR_COLUMNS)
        support=pd.read_parquet(meta['support_intervals']['path'],filters=[('policy','==','structural_only')]).reset_index(drop=True)
        scaling=read(meta['scaling']['path']);features=[];sequences=[];marks=[]
        month=origins.bin_start.dt.strftime('%Y-%m')
        for key in sorted(month.unique()):
            o=origins.loc[month.eq(key)].reset_index(drop=True)
            ids=set(o.episode_id);a=anchors.loc[anchors.episode_id.isin(ids)].reset_index(drop=True)
            p=predictors.loc[predictors.episode_id.isin(ids)].reset_index(drop=True)
            lo=o.bin_start.min()-pd.Timedelta(300,unit='s');hi=o.bin_start.max()
            sparse=pd.read_parquet(meta['sparse']['path'],columns=SPARSE_COLUMNS,
                filters=[('bin_start','>=',lo),('bin_start','<',hi)])
            quotes=read_quote_window(meta['clean_quotes']['path'],lo,hi,QUOTE_COLUMNS)
            flow,seq=measure_pressure(a,sparse,support)
            price,grid=measure_prior_prices(a,quotes,support)
            f=build_features(o,p,a,flow,seq,price,scaling)
            features.append(f);sequences.append(seq);marks.append(grid)
            print(f'Measured {key}: {len(o):,} origins; {int(f.combined_eligible.sum()):,} combined eligible',flush=True)
            del sparse,quotes;gc.collect()
        f=pd.concat(features,ignore_index=True);q=pd.concat(sequences,ignore_index=True);grid=pd.concat(marks,ignore_index=True)
        require(np.array_equal(f.episode_id,origins.episode_id),'Monthly processing changed origin order')
        f.to_parquet(run/'features.parquet',index=False);q.to_parquet(run/'sequences.parquet',index=False)
        grid.to_parquet(run/'prior_price_marks.parquet',index=False)
        statistics=derive_statistics(f);write(run/'statistics.json',statistics)
        monthly=[]
        for key,g in f.groupby(f.bin_start.dt.strftime('%Y-%m'),sort=True):
            for name,part in [('all',g),('inherited_eligible',g.loc[g.inherited_origin_eligible])]:
                row=coverage_row(name,part);row['month']=key
                for col in ['pressure_status_counts','price_status_counts','measurement_status_counts']:row[col]=json.dumps(row[col],sort_keys=True)
                row.update({'sequence_z_'+k:v for k,v in distribution(part.sequence_z).items()})
                monthly.append(row)
        pd.DataFrame(monthly).to_parquet(run/'monthly_coverage.parquet',index=False)
        m['outputs']={p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='manifest.json'}
        m.update(status='pending_serialized_verification',tests_passed=count);write(run/'manifest.json',m)
        checked=saved_checks(run,m,statistics);write(run/'saved_verification.json',checked)
        m['outputs']['saved_verification.json']=sha(run/'saved_verification.json')
        m.update(status=STATUS,completed_at_utc=now());write(run/'manifest.json',m)
        print(json.dumps({'run_id':RUN_ID,'status':STATUS,'tests_passed':count,'coverage':statistics['coverage'],
                         'response_models_fitted':0},indent=2),flush=True)
    except Exception as exc:
        m.update(status='failed',failed_at_utc=now(),failure_type=type(exc).__name__,failure=str(exc))
        write(run/'manifest.json',m);raise


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test-only',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.test_only:
        log,_=tests();print(log,end='')
    else:build()


if __name__=='__main__':main()

"""Stage 03: build and independently verify recorded best-quote imbalance."""
from datetime import datetime,timezone
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
from src.data_preparation import atomic_json,sha256
from src.ofi import compute_quote_imbalance,prepare_support,BinStream
from src.quote_imbalance_statistics import SignalStatistics
from src.verify_quote_imbalance import verify_quote_run

ROOT=Path(__file__).resolve().parent
CODE=['repair_quote_imbalance.py','src/ofi.py','src/quote_imbalance_statistics.py',
 'src/verify_quote_imbalance.py','src/pipeline_state.py','src/data_preparation.py','tests/test_quote_imbalance.py']


def run_tests():
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'output':stream.getvalue(),
        'test_source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage03_tests.json',evidence);print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise RuntimeError('Tests failed; no build or promotion')
    return evidence


def build_artifacts(parent_clean,eligibility,run,*,batch_size=500_000):
    segments=pd.read_parquet(eligibility/'segments.parquet')
    support=prepare_support(pd.read_parquet(eligibility/'support_intervals.parquet'))
    source=pq.ParquetFile(parent_clean);n=source.metadata.num_rows
    if not n or segments.quote_start.iloc[0]!=0 or segments.quote_stop_exclusive.iloc[-1]!=n:
        raise ValueError('Segment partition does not match parent quotes')
    np.testing.assert_array_equal(segments.quote_start.to_numpy()[1:],segments.quote_stop_exclusive.to_numpy()[:-1])
    np.testing.assert_array_equal(segments.segment_id.to_numpy(),np.arange(len(segments)))
    starts=segments.quote_start.to_numpy();stream=BinStream(support);stats=SignalStatistics()
    offset=0;previous=None;writers={}
    def write(name,frame):
        if frame is None:return
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if name not in writers:writers[name]=pq.ParquetWriter(run/name,table.schema,compression='zstd')
        writers[name].write_table(table,row_group_size=batch_size)
    try:
        for i,batch in enumerate(source.iter_batches(batch_size=batch_size,columns=['timestamp','bid','ask','bid_volume','ask_volume'])):
            frame=batch.to_pandas();ids=np.arange(offset,offset+len(frame),dtype='int64')
            frame.insert(0,'quote_id',ids);frame.insert(2,'segment_id',np.searchsorted(starts,ids,side='right')-1)
            tick=compute_quote_imbalance(frame,previous=previous)
            write('quote_ofi_ticks.parquet',tick);stats.add_ticks(tick)
            bins=stream.push(tick)
            if bins is not None:write('quote_ofi_1s.parquet',bins);stats.add_bins(bins)
            previous=frame.iloc[-1].to_dict();offset+=len(frame)
            if (i+1)%10==0:print(f'Built {offset:,} quote contributions',flush=True)
        bins=stream.finish()
        if bins is not None:write('quote_ofi_1s.parquet',bins);stats.add_bins(bins)
    finally:
        for writer in writers.values():writer.close()
    result=stats.result()
    if result['ticks']['quotes']!=n or result['bins']['quotes_accounted_for']!=n:raise AssertionError('Quote/bin row conservation failed')
    atomic_json(run/'statistics.json',result)
    pd.DataFrame([{'month':k,**v} for k,v in result['monthly'].items()]).to_parquet(run/'monthly_statistics.parquet',index=False)
    return result


def build_run(root,run_id,evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid run id')
    active=json.loads((root/'data/active_run.json').read_text());ep=json.loads((root/'data/active_eligibility.json').read_text())
    if active['data_run_id']!=ep['parent_data_run']:raise ValueError('Active data and eligibility differ')
    parent=root/'data/runs'/active['data_run_id'];elig=root/'data/eligibility'/ep['eligibility_run_id']
    pm=json.loads((parent/'manifest.json').read_text());em=json.loads((elig/'manifest.json').read_text())
    if pm['status']!='verified' or em['status']!='verified_observation_support_only':raise ValueError('Verified parents required')
    if em['parent_data_run']!=parent.name:raise ValueError('Eligibility lineage differs')
    paths={'parent_clean':parent/'clean_quotes.parquet','parent_manifest':parent/'manifest.json',
        'parent_verification':parent/'verification.json','eligibility_manifest':elig/'manifest.json',
        'eligibility_verification':elig/'verification.json','segments':elig/'segments.parquet',
        'support':elig/'support_intervals.parquet','eligibility_usage_contract':elig/'usage_contract.json'}
    inputs={name:{'path':str(path),'sha256':sha256(path)} for name,path in paths.items()}
    if inputs['parent_clean']['sha256']!=pm['outputs']['clean_quotes.parquet']['sha256'] or inputs['parent_clean']['sha256']!=ep['parent_clean_sha256']:
        raise ValueError('Parent quote fingerprint differs')
    for key,file in [('segments','segments.parquet'),('support','support_intervals.parquet')]:
        if inputs[key]['sha256']!=em['output_sha256'][file]:raise ValueError('Eligibility output fingerprint differs')
    if json.loads(paths['parent_verification'].read_text())['status']!='passed' or json.loads(paths['eligibility_verification'].read_text())['status']!='passed':
        raise ValueError('Parent verification failed')
    run=root/'data/signals'/run_id;run.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'run_id':run_id,'stage':'03_recorded_quote_imbalance','status':'building',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'parent_data_run':parent.name,
        'parent_eligibility_run':elig.name,'parent_clean_path':str(paths['parent_clean']),
        'eligibility_directory':str(elig),'inputs':inputs,'production_batch_size':500000,
        'coverage_certified':False,'downstream_ready':False,'primary_freshness_policy':None,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},
        'code_sha256':{}}
    for name in CODE:
        dest=run/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest);manifest['code_sha256'][name]=sha256(dest)
    method={'measure':'Cont-style recorded best-quote imbalance proxy',
        'formula':'I(b_n>=b_prev)*qb_n - I(b_n<=b_prev)*qb_prev - I(a_n<=a_prev)*qa_n + I(a_n>=a_prev)*qa_prev',
        'units':'Stored size units. Vendor BI5 documentation describes millions of base currency; archived decoder passes float32 size fields through unchanged. No rescaling performed.',
        'timing':'Tick contribution available at its timestamp. Bins are [bin_start,bin_start+1s); aggregate available_at=bin_start+1s. Neither signal uses future quotes.',
        'missingness':'Exclude predecessor/segment/price/size failures, retaining rows with NaN contributions. Bin model sum is NaN if any included transition is excluded; subtotal remains a separately named diagnostic. No-quote bins are absent, not silently zero.',
        'policies':'No primary policy selected. usable_<policy> requires all transitions valid, full-bin support, and incoming gaps within the policy limit. Structural-only imposes no freshness limit.',
        'audit_flags':'Full-bin support/usable flags depend on retrospective coverage audits, are sample-quality labels, and must not be Ft0 explanatory features.',
        'limits':['Not identified transactions, cancellations, individual orders, dealer inventory, or complete decentralized FX flow.',
            'Original same-timestamp price deduplication may have lost size updates; gaps and completeness remain unresolved.',
            'Price changes enter the formula mechanically. Contemporaneous direction agreement is not evidence of causality, information, or prediction.',
            'Arbitrary coarsening/thinning and all closed price-size paths are not claimed invariant.',
            'No spread division, depth normalization, seasonal adjustment, threshold fitting, event detection or outcome estimation in this stage.',
            'Actual real-time receive latency is unavailable; available_at refers to recorded quote timestamps and completed buckets.'],
        'sources':[{'title':'Cont, Kukanov and Stoikov: The Price Impact of Order Book Events, section 2.1','url':'https://arxiv.org/abs/1011.6402'},
            {'title':'Dukascopy ITick: volume available at best bid/ask','url':'https://www.dukascopy.com/client/javadoc/com/dukascopy/api/ITick.html'},
            {'title':'Dukascopy BI5 format: size units and field ordering','url':'https://www.dukascopy.com/wiki/en/development/data-export/'}]}
    atomic_json(run/'manifest.json',manifest);atomic_json(run/'methodology.json',method);atomic_json(run/'test_evidence.json',evidence)
    statistics=build_artifacts(paths['parent_clean'],elig,run)
    names=['quote_ofi_ticks.parquet','quote_ofi_1s.parquet','statistics.json','monthly_statistics.parquet','methodology.json','test_evidence.json']
    manifest['outputs']={name:sha256(run/name) for name in names};manifest['status']='built_pending_independent_verification';atomic_json(run/'manifest.json',manifest)
    print('Starting independent formula, bin, timing and statistics verification',flush=True)
    verification=verify_quote_run(run);atomic_json(run/'verification.json',verification)
    manifest['status']='verified_recorded_quote_proxy';atomic_json(run/'manifest.json',manifest)
    return run,statistics,verification


def promote(root,run,statistics,verification):
    m=json.loads((run/'manifest.json').read_text());data=json.loads((root/'data/active_run.json').read_text());elig=json.loads((root/'data/active_eligibility.json').read_text())
    if data['data_run_id']!=m['parent_data_run'] or elig['eligibility_run_id']!=m['parent_eligibility_run'] or verification['status']!='passed' or m['status']!='verified_recorded_quote_proxy':
        raise ValueError('Cannot promote mismatched/unverified signal run')
    path=root/'reports/pipeline_master_statistics.json';master=json.loads(path.read_text())
    entry={'stage':m['stage'],'run_id':run.name,'parent_data_run':m['parent_data_run'],'parent_eligibility_run':m['parent_eligibility_run'],
        'status':m['status'],'completed_at_utc':datetime.now(timezone.utc).isoformat(),'statistics':statistics,'verification':verification,
        'methodology':json.loads((run/'methodology.json').read_text()),'primary_freshness_policy':None,'downstream_ready':False}
    master['stage_history'].append(entry);master['current_quote_imbalance_stage']=entry
    master['pipeline_status']='quote_imbalance_proxy_verified_remaining_signal_calibration_and_models_pending'
    master['downstream_status']['ofi']='recorded_quote_proxy_verified_normalization_and_primary_sampling_policy_pending'
    master['repair_progress']={'stage':m['stage'],'status':m['status'],
        'next_required_work':'Past-only seasonality, duration treatment and signal normalization on a prespecified observation policy; no reuse of legacy event/model estimates.'}
    master['updated_at_utc']=datetime.now(timezone.utc).isoformat();atomic_json(path,master)
    atomic_json(root/'data/active_quote_imbalance.json',{'schema_version':1,'signal_run_id':run.name,
        'parent_data_run':m['parent_data_run'],'parent_eligibility_run':m['parent_eligibility_run'],
        'status':m['status'],'coverage_certified':False,'primary_freshness_policy':None,'downstream_ready':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='quote_ofi_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only',action='store_true');parser.add_argument('--verify-only',metavar='RUN_ID');args=parser.parse_args()
    if args.verify_only:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.verify_only):parser.error('Invalid run ID')
        print(json.dumps(verify_quote_run(ROOT/'data/signals'/args.verify_only),indent=2));return
    evidence=run_tests()
    if args.test_only:return
    run,statistics,verification=build_run(ROOT,args.run_id,evidence);promote(ROOT,run,statistics,verification)
    print(json.dumps({'run':str(run),'ticks':statistics['ticks'],'bins':statistics['bins'],'verification':verification},indent=2),flush=True)


if __name__=='__main__':main()

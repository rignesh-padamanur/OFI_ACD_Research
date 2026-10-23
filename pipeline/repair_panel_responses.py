"""Stage07e: continuous recorded-quote responses; no regression or inference."""
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
import pyarrow.parquet as pq
from src.data_preparation import atomic_json, sha256
from src.panel_observations import read_quote_window

ROOT = Path(__file__).resolve().parent
STAGE = '07e_full_panel_continuous_price_responses'
STATUS = 'verified_full_panel_price_responses'
SCOPE = 'full_cached_structural_price_responses_only'
HIGHLIGHTS = [0, 60, 300]
SECOND = 1_000_000_000


def read(path): return json.loads(Path(path).read_text())
def records(frame): return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15))
def require(condition, message):
    if not condition: raise AssertionError(message)


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage07e_tests.json', evidence)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Tests failed; no response build')
    return evidence


def resolve_inputs():
    inputs = {}
    def add(key, path, expected=None):
        path = Path(path); h = sha256(path)
        require(expected is None or h == expected, 'Changed parent: '+str(path))
        inputs[key] = {'path': str(path), 'sha256': h}
    panel = ROOT/'data/signals'/read(ROOT/'data/active_chronological_panel_audit.json')['panel_run_id']
    pm = read(panel/'manifest.json')
    require(pm['status'] == 'verified_full_chronological_baseline_panel' and pm['primary_policy'] == 'structural_only', 'Verified structural panel required')
    require(read(panel/'verification.json')['status'] == 'passed', 'Panel verification required')
    for key, name in [('parent_manifest','manifest.json'), ('parent_verification','verification.json'),
        ('parent_statistics','statistics.json'), ('onsets','onsets.parquet'), ('cycle_schedule','cycle_schedule.parquet')]:
        add(key, panel/name, pm.get('outputs',{}).get(name))
    for key in ['clean_quotes','support_intervals']:
        r = pm['inputs'][key]; add(key, r['path'], r['sha256'])
    measurement = ROOT/'data/signals'/read(ROOT/'data/active_panel_measurement_audit.json')['measurement_run_id']
    mm = read(measurement/'manifest.json')
    require(mm['status'] == 'verified_full_panel_fixed_onset_measurement' and mm['parent_chronological_panel'] == panel.name, 'Matching measurement parent required')
    require(mm['inputs']['onsets'] == inputs['onsets'] and read(measurement/'verification.json')['status'] == 'passed', 'Measurement onset lineage differs')
    for key, name in [('measurement_manifest','manifest.json'), ('measurement_verification','verification.json'),
                      ('measurement_statistics','statistics.json'), ('directions','direction.parquet')]:
        add(key, measurement/name, mm.get('outputs',{}).get(name))
    legacy = ROOT/'data/signals'/read(ROOT/'data/active_price_response_audit.json')['response_run_id']
    lm = read(legacy/'manifest.json')
    require(lm['status'] == 'verified_development_price_responses' and read(legacy/'verification.json')['status'] == 'passed', 'Verified legacy development responses required')
    for key in ['clean_quotes','support_intervals']:
        require(lm['inputs'][key] == inputs[key], 'Legacy raw/support source differs')
    for key, name in [('legacy_manifest','manifest.json'), ('legacy_verification','verification.json'),
        ('legacy_statistics','statistics.json'), ('legacy_anchors','structural_only/anchors.parquet'), ('legacy_paths','structural_only/paths.parquet')]:
        add(key, legacy/name, lm.get('outputs',{}).get(name))
    return pm, inputs


def boundary_table(onsets, schedule, support, stop):
    times = pd.DatetimeIndex(onsets.bin_start).as_unit('ns').asi8
    cutoffs = pd.DatetimeIndex(schedule.parameter_cutoff).as_unit('ns').asi8
    ix = np.searchsorted(cutoffs, times, side='right')-1
    ends = pd.DatetimeIndex(schedule.deployment_end).as_unit('ns').asi8[ix]
    require((ix >= 0).all() and (times < ends).all(), 'Onset outside cycle schedule')
    td = pd.DatetimeIndex(onsets.available_at).as_unit('ns').asi8
    target = td+300*SECOND
    next_time = np.r_[td[1:], np.iinfo('int64').min]
    next_cycle = np.r_[ix[1:], -1]
    right = pd.DatetimeIndex(support.end_exclusive).as_unit('ns').asi8[onsets.support_id.to_numpy()]
    return pd.DataFrame({'episode_id': onsets.episode_id.to_numpy(), 'cycle_id': schedule.cycle_id.to_numpy()[ix],
        'detected_at': onsets.available_at.to_numpy(), 'cycle_end': pd.to_datetime(ends, utc=True),
        'target_300': pd.to_datetime(target, utc=True),
        'crosses_internal_cutoff': (target > ends) & (ends < pd.Timestamp(stop).value),
        'target_300_within_support': target <= right,
        'next_detection_at': pd.to_datetime(next_time, utc=True),
        'next_detection_in_later_cycle_within300': (next_cycle > ix) & (next_time <= target)})


def calendar_table(anchors, highlighted, start, stop):
    """Full UTC calendar; component means retain their own explicit populations."""
    calendar = pd.DataFrame({'date': pd.date_range(start, stop, inclusive='left', freq='D')})
    a = anchors.copy(); a['date'] = a.bin_start.dt.normalize()
    for name, mask in [('onsets', np.ones(len(a),bool)), ('confirmed_onsets',a.onset_confirmed),
        ('confirmed_directed_onsets',a.confirmed_direction_cohort), ('state_directed_onsets',a.state_direction_cohort)]:
        calendar[name] = calendar.date.map(a.loc[mask].groupby('date').size()).fillna(0).astype('int64')
    for h in [60,300]:
        p = highlighted[h].copy(); p['date'] = p.bin_start.dt.normalize()
        for cohort in ['confirmed','state']:
            base = p.onset_confirmed & p.direction_available
            if cohort == 'state': base &= p.state_sample_eligible
            c = base & p.continuation_quote_available
            joint = base & p.triplet_quote_available
            for label, mask in [('continuation',c), ('joint',joint)]:
                calendar[f'{cohort}_{label}_{h}'] = calendar.date.map(p.loc[mask].groupby('date').size()).fillna(0).astype('int64')
            if h == 300:
                calendar[f'{cohort}_mean_continuation_300'] = calendar.date.map(p.loc[c].groupby('date').continuation_pips.mean())
                for component in ['initial','continuation','total']:
                    calendar[f'{cohort}_joint_mean_{component}_300'] = calendar.date.map(p.loc[joint].groupby('date')[component+'_pips'].mean())
    return calendar


def attrition_record(h, values):
    return {'horizon_sec': h, **{k:v for k,v in values.items() if k not in ['samples','endpoint_status_all_fragments']},
            **{'status_'+k:v for k,v in values['endpoint_status_all_fragments'].items()}}


def reconcile_legacy(inputs, anchors, first_paths):
    from src.verify_price_responses import _compare
    old_a = pd.read_parquet(inputs['legacy_anchors']['path'])
    old_p = pd.read_parquet(inputs['legacy_paths']['path'])
    old_stop = pd.Timestamp(read(inputs['legacy_statistics']['path'])['development_end_exclusive'])
    a = anchors.set_index('episode_id', drop=False).loc[old_a.episode_id].reset_index(drop=True)
    shared = [c for c in old_a if c in a]
    _compare(a[shared], old_a[shared])
    keys = ['episode_id','horizon_sec']
    old_p = old_p.sort_values(keys).reset_index(drop=True)
    p = first_paths.set_index(keys, drop=False).loc[pd.MultiIndex.from_frame(old_p[keys])].reset_index(drop=True)
    valid = old_p.target_at.le(old_stop)
    future_masks = [c for c in old_p if c.startswith('common_path')]
    columns = [c for c in old_p if c not in future_masks]
    _compare(p.loc[valid,columns].reset_index(drop=True), old_p.loc[valid,columns].reset_index(drop=True))
    mask_changes = {c:int(p[c].ne(old_p[c]).sum()) for c in future_masks}
    extended = old_p.endpoint_status.eq('beyond_development_end') & p.endpoint_status.eq('available')
    return {'legacy_onsets_checked':len(old_a), 'shared_anchor_columns_checked':len(shared),
        'legacy_path_rows':len(old_p), 'pointwise_rows_within_old_cutoff_checked':int(valid.sum()),
        'new_available_rows_beyond_old_cutoff':int(extended.sum()),
        'onsets_with_new_available_rows':int(p.loc[extended,'episode_id'].nunique()),
        'future_dependent_mask_changes':mask_changes, 'legacy_cutoff':old_stop.isoformat(),
        'legacy_artifacts_preserved':True, 'historical_pointwise_responses_reconciled':True}


def build_statistics(anchors, highlights, component, calendar, boundary, cycles, legacy, start, stop, anchor_stats):
    return {'scope':SCOPE,'primary_policy':'structural_only', 'deployment_start':start.isoformat(),
        'deployment_end_exclusive':stop.isoformat(), **anchor_stats,
        'path_rows':int(cycles.path_rows.sum()), 'horizons_seconds':list(range(301)),
        'primary_horizon_seconds':300,'secondary_horizon_seconds':60,
        'horizons':highlights,'component_summary':records(component), 'horizon_summary_rows':3010,
        'calendar':{'days':len(calendar),'days_with_onsets':int(calendar.onsets.gt(0).sum()),
            'totals':{c:int(calendar[c].sum()) for c in calendar if pd.api.types.is_integer_dtype(calendar[c])}},
        'boundary':{'crossing_onsets':int(boundary.crosses_internal_cutoff.sum()),
            'internal_boundaries_crossed':int(boundary.loc[boundary.crosses_internal_cutoff,'cycle_id'].nunique()),
            'crossing_onsets_with_supported_300_target':int((boundary.crosses_internal_cutoff & boundary.target_300_within_support).sum()),
            'next_detections_in_later_cycle_within300':int(boundary.next_detection_in_later_cycle_within300.sum()),
            'available_endpoints_after_cycle_end':int(cycles.available_endpoints_after_cycle_end.sum()),
            'later_episode_flags_recovered_across_cycles':int(cycles.later_flags_recovered.sum())},
        'legacy_reconciliation':legacy,'cycles':len(cycles),'detector_fits':0,'empirical_response_models_fitted':0,
        'baseline_references_preserved':True,'production_inference_approved':False,'downstream_ready':False}


def cycle_counts(cycle, anchors, paths):
    past = paths.target_at.gt(cycle.deployment_end)
    local_last = paths.episode_id.eq(anchors.episode_id.iloc[-1]) if len(anchors) else np.zeros(len(paths),bool)
    return {'cycle_id':int(cycle.cycle_id),'onset_rows':len(anchors),'path_rows':len(paths),
        'targets_after_cycle_end':int(past.sum()),
        'available_endpoints_after_cycle_end':int((past & paths.endpoint_status.eq('available')).sum()),
        'later_flags_recovered':int((local_last & paths.later_observed_episode_by_horizon.fillna(False)).sum())}


def build(run_id, evidence):
    from src.panel_responses import construct_panel_responses, summarize_horizon, component_summary, anchor_statistics
    require(re.fullmatch(r'[A-Za-z0-9_-]+',run_id), 'Invalid run identifier')
    pm, inputs = resolve_inputs()
    run = ROOT/'data/signals'/run_id; run.mkdir(exist_ok=False)
    snapshot = ROOT/'reports/repair_history'/('2026-10-01_pre_'+run_id); snapshot.mkdir(exist_ok=False)
    previous = {}
    for path in [ROOT/'reports/pipeline_master_statistics.json',ROOT/'reports/PIPELINE_REPAIR_LOG.md',ROOT/'REPAIR_WORKFLOW.md',*sorted((ROOT/'data').glob('active_*.json'))]:
        rel = str(path.relative_to(ROOT)); dest = snapshot/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest); previous[rel] = sha256(path)
    atomic_json(snapshot/'preserved_files.json',previous)
    m = {'run_id':run_id,'stage':STAGE,'status':'building','scope':SCOPE,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'inputs':inputs,'parent_chronological_panel':pm['run_id'],
        'parent_measurement':Path(inputs['measurement_manifest']['path']).parent.name,
        'primary_policy':'structural_only','horizons_seconds':list(range(301)),
        'deployment_start':pm['first_cutoff'],'deployment_end_exclusive':pm['deployment_end'],
        'price_mark_rule':'strictly_prior_left_limit','direction_rule':'exact_sign_of_finite_nonzero_onset_quote_ofi',
        'path_storage':'cycle_files_with_one_row_group_per_horizon','production_inference_approved':False,'downstream_ready':False,
        'previous_files':previous,'snapshot_path':str(snapshot),'code_sha256':{},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__}}
    for path in [Path(__file__),ROOT/'reports/STAGE07E_PANEL_RESPONSE_PROTOCOL.md',*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'tests').glob('test_*.py'))]:
        rel = str(path.relative_to(ROOT)); dest = run/'code'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest); m['code_sha256'][rel] = sha256(dest)
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        require(sha256(path)==evidence['test_source_sha256'][path.name], 'Test changed after execution')
        dest = run/'full_test_sources'/path.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(path,dest)
    atomic_json(run/'manifest.json',m); atomic_json(run/'test_evidence.json',evidence)
    try:
        onsets = pd.read_parquet(inputs['onsets']['path']); schedule = pd.read_parquet(inputs['cycle_schedule']['path'])
        support = pd.read_parquet(inputs['support_intervals']['path'],filters=[('policy','==','structural_only')]).reset_index(drop=True)
        start, stop = pd.Timestamp(m['deployment_start']),pd.Timestamp(m['deployment_end_exclusive'])
        (run/'paths').mkdir(); (run/'cycles').mkdir()
        anchors, counts = [],[]
        for cycle in schedule.itertuples(index=False):
            selected = onsets.loc[onsets.bin_start.ge(cycle.parameter_cutoff)&onsets.bin_start.lt(cycle.deployment_end)].reset_index(drop=True)
            quotes = read_quote_window(inputs['clean_quotes']['path'],cycle.parameter_cutoff,min(stop,cycle.deployment_end+pd.Timedelta(seconds=300)),['timestamp','bid','ask'])
            tables = construct_panel_responses(selected,quotes,support,sample_end=stop,all_onsets=onsets)
            a,p = tables['anchors'],tables['paths']; count = cycle_counts(cycle,a,p)
            p = p.sort_values(['horizon_sec','episode_id']).reset_index(drop=True)
            pq.write_table(pa.Table.from_pandas(p,preserve_index=False),run/'paths'/f'cycle_{cycle.cycle_id:03d}.parquet',row_group_size=max(len(a),1),compression='zstd')
            directory = run/'cycles'/f'{cycle.cycle_id:03d}'; directory.mkdir()
            a.to_parquet(directory/'anchors.parquet',index=False); atomic_json(directory/'statistics.json',count)
            anchors.append(a); counts.append(count)
            print(f'Built response cycle {cycle.cycle_id:03d}: {len(a):,} onsets, {len(p):,} paths',flush=True)
            del tables,p,quotes; gc.collect()
        anchors = pd.concat(anchors,ignore_index=True); counts = pd.DataFrame(counts)
        anchors.to_parquet(run/'anchors.parquet',index=False); counts.to_parquet(run/'cycle_statistics.parquet',index=False)
        boundary = boundary_table(onsets,schedule,support,stop); boundary.to_parquet(run/'boundary_audit.parquet',index=False)
        source = ds.dataset(run/'paths',format='parquet')
        summaries,attrition,components,highlights,highlight_paths = [],[],[],{},{}
        for h in range(301):
            p = source.to_table(filter=ds.field('horizon_sec')==h).to_pandas().sort_values('episode_id').reset_index(drop=True)
            require(len(p)==len(onsets) and not p.episode_id.duplicated().any(), 'Horizon onset conservation failed')
            summary,stats = summarize_horizon(p); summaries.append(summary); attrition.append(attrition_record(h,stats))
            if h in HIGHLIGHTS:
                highlights[str(h)] = stats; components.append(component_summary(p)); highlight_paths[h] = p
            if h % 60 == 0: print(f'Pooled full-panel horizon {h}',flush=True)
        summary = pd.concat(summaries,ignore_index=True); component = pd.concat(components,ignore_index=True)
        summary.to_parquet(run/'horizon_summary.parquet',index=False); component.to_parquet(run/'component_summary.parquet',index=False)
        pd.DataFrame(attrition).to_parquet(run/'horizon_attrition.parquet',index=False)
        calendar = calendar_table(anchors,highlight_paths,start,stop); calendar.to_parquet(run/'calendar.parquet',index=False)
        legacy = reconcile_legacy(inputs,anchors,pd.read_parquet(run/'paths/cycle_000.parquet'))
        atomic_json(run/'legacy_reconciliation.json',legacy)
        statistics = build_statistics(anchors,highlights,component,calendar,boundary,counts,legacy,start,stop,anchor_statistics(anchors))
        atomic_json(run/'statistics.json',statistics)
        m['outputs'] = {str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file()
            and p.relative_to(run).parts[0] not in ['code','full_test_sources'] and p != run/'manifest.json'}
        m['status']='built_pending_independent_verification'; atomic_json(run/'manifest.json',m)
        return run
    except Exception as error:
        atomic_json(run/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        m['status']='failed_unregistered'; atomic_json(run/'manifest.json',m); raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id'); parser.add_argument('--test-only',action='store_true'); parser.add_argument('--verify-only')
    args=parser.parse_args()
    if args.verify_only: run=ROOT/'data/signals'/args.verify_only
    else:
        evidence=run_tests()
        if args.test_only:return
        if not args.run_id:parser.error('--run-id required')
        run=build(args.run_id,evidence)
    from src.verify_panel_response_run import verify_panel_response_run
    try: verification=verify_panel_response_run(run)
    except Exception as error:
        if not args.verify_only:
            atomic_json(run/'verification_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
            m=read(run/'manifest.json');m['status']='failed_independent_verification';m['outputs']['verification_failure.json']=sha256(run/'verification_failure.json');atomic_json(run/'manifest.json',m)
        raise
    require(verification['status']=='passed','Verification did not pass')
    if not args.verify_only:
        atomic_json(run/'verification.json',verification);m=read(run/'manifest.json');m['status']=STATUS
        m['outputs']['verification.json']=sha256(run/'verification.json');atomic_json(run/'manifest.json',m)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()

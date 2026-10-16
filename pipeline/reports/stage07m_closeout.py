"""Register verified common-row price accounting, without fitting any model."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repair_joint_response import STAGE, STATUS, SCOPE, POINTER, CURRENT, NEXT, FRAMES, integrity, read, require
from src.data_preparation import sha256, atomic_json
REVIEW = 'reports/STAGE07M_JOINT_RESPONSE_REVIEW.md'


def research_evidence(run):
    run=Path(run).resolve();require(run.parent==ROOT/'data/signals','Named signal run required')
    m=integrity(run);s=read(run/'statistics.json');v=read(run/'verification.json');t=read(run/'test_evidence.json')
    require(m['status']==STATUS and v['status']==t['status']=='passed' and t['sources_unchanged_during_tests'] is True,
            'Verified unchanged-source execution required')
    require(v['executed_tests']==t['tests_run'] and v['source_lineage']['status']=='passed','Verification scope differs')
    require(m['empirical_models_fitted']==s['empirical_models_fitted']==v['empirical_models_fitted']==0
            and s['descriptive_only'] is True and s['uncertainty_computed'] is False and s['inference_approved'] is False,
            'Descriptive boundary differs')
    expected={n+'.parquet' for n in FRAMES}|{'statistics.json','verification.json','test_evidence.json','source_lineage.json'}
    require(set(m['outputs']) in [expected,expected|{'review.md'}],'Output accounting differs')
    return m,s,v,t


def make_review(run,s,t):
    rows=[]
    for h in [300,60]:
        n=s['by_horizon'][str(h)];c=s['cohorts'][f'horizon_joint_{h}']
        rows.append(f"| {h}s | {n['main_continuation_rows']:,} | {n['joint_rows']:,} | {n['missing_joint_rows']:,} | "
                    + ' | '.join(f"{c['mean'][k]:+.6f}" for k in ['J','C','I'])+' |')
    c=s['cohorts']['horizon_joint_300'];n=c['n'];patterns=c['endpoint_pattern_counts']
    names={'unchanged':'Same as detection price (nonzero J)', 'extended':'Farther in initial direction',
      'retraced':'Partial retracement', 'at_pre_onset':'At pre-onset price', 'beyond_pre_onset':'Beyond pre-onset in opposite direction',
      'initial_zero_positive':'No initial move; positive aligned endpoint', 'initial_zero_negative':'No initial move; negative aligned endpoint',
      'initial_zero_flat':'No initial move; unchanged endpoint'}
    table=[f"| {names[k]} | {value:,} | {100*value/n:.2f}% |" for k,value in patterns.items()]
    return '\n'.join([
      '# Stage07m: joint price-response interpretation','',
      f'Run `{run.name}`. This is descriptive accounting across Notebooks3/4 feeding Notebook5, using the original main continuation cohorts. '
      f'{t["tests_run"]} tests passed; independent reconstruction verified every accounting table. Zero new regression models were fitted.','',
      'J is price movement during the detection second, C is subsequent movement, and I=J+C is total movement from just before onset. '
      'All are recorded pips aligned to the inherited quote-imbalance direction. J is not necessarily positive and is not the whole eventual episode impact.','',
      '| Horizon | Original C sample | Joint J/C/I sample | Missing joint | Mean J | Mean C | Mean I |',
      '|---|---:|---:|---:|---:|---:|---:|',*rows,'',
      'These means use identical observations within each row. Medians need not add. Counts, medians and quartiles by positive/zero/negative J are saved, without dividing by J or variation.','',
      f'At300s, initial J is positive for {c["initial_sign_counts"]["positive"]:,} events, zero for {c["initial_sign_counts"]["zero"]:,}, '
      f'and negative for {c["initial_sign_counts"]["negative"]:,}. Endpoint descriptions below use all {n:,} jointly observed events as the displayed denominator.','',
      '| Five-minute endpoint description | Events | Share of joint sample |','|---|---:|---:|',*table,'',
      'Negative C alone does not imply retracement: J=-1,C=-0.5 gives I=-1.5, extending the initial move. '
      'J=1,C=-0.2 gives partial retracement; J=0 has no initial displacement to retain. Exact observed signs and strict endpoint ordering define the descriptions. '
      'Tiny nonzero moves remain mathematically nonzero and do not establish economically material displacement.','',
      f'The common60/300 cohort has {s["common_joint_events"]:,} events; {s["set_relation"]["only_60"]:,} joint events are60-only and '
      f'{s["set_relation"]["only_300"]:,} are300-only. Both horizon means are also saved on the intersection, together with all64 endpoint-transition cells '
      'and explicit total/origin denominators. This cohort conditions on later availability; the transition table is descriptive, not a forecast or regime model.','',
      f'All {s["calendar_days"]} calendar dates remain in each cohort/horizon daily table. Empty dates have zero counts/sums and missing means. '
      'Saved event counts and sums prepare event-weighted uncertainty; equally averaging daily means would change the target.','',
      'Source accounting reconciles the registered anchors/paths and original stage07h response keys. It rechecks direct saved price arithmetic, '
      'not raw ticks or an independent external feed. All29 previous active pointers and the existing fitted equations remain preserved.','',
      'Interpretation boundary: endpoint position cannot show monotonic decay, first crossing, permanence, or private information versus inventory. '
      'A positive mean total displacement can coexist with many opposite-signed individual paths. J-sign conditioning is on a realized detection-time outcome. '
      'No confidence intervals, significance tests, causal identification or new state association is claimed in this step. '
      'Joint uncertainty is still pending and must respect the singular J/C/I accounting identity.','',
      'Next: '+NEXT,'','The pipeline is not yet paper-ready.',''])


def registered_evidence(run):
    m,s,v,t=research_evidence(run);master=read(ROOT/'reports/pipeline_master_statistics.json')
    old=read(Path(m['snapshot_path'])/'reports/pipeline_master_statistics.json');entry=master[CURRENT]
    require(master['stage_history'][:-1]==old['stage_history'] and master['stage_history'][-1]==entry
            and set(master)==set(old)|{CURRENT},'Earlier master history/keys differ')
    for k,value in old.items():
        if k not in ['stage_history','pipeline_status','repair_progress','updated_at_utc']:
            require(master[k]==value,'Earlier master record changed: '+k)
    require(entry['statistics']==s and entry['verification']==v and entry['tests_passed']==t['tests_run'],'Registered evidence differs')
    record=read(ROOT/'reports/stage07m_final_integrity.json')
    for path,key in [('reports/pipeline_master_statistics.json','registered_master_sha256'),(POINTER,'registered_pointer_sha256'),
                     ('reports/stage07m_closeout.py','registration_source_sha256')]:
        require(sha256(ROOT/path)==record[key],'Registered hash differs: '+path)
    require(sha256(Path(run)/'manifest.json')==record['input_manifest_sha256'],'Registered manifest differs')
    require(read(ROOT/POINTER)['joint_response_run_id']==Path(run).name,'Pointer differs')
    require(sha256(ROOT/REVIEW)==m['outputs']['review.md'],'Review differs')
    oldp={n for n in m['previous_files'] if n.startswith('data/active_')}
    require({str(p.relative_to(ROOT)) for p in (ROOT/'data').glob('active_*.json')}==oldp|{POINTER},'Pointer set differs')
    return m,s,v,t,record


def main(run):
    run=Path(run).resolve();m,s,v,t=research_evidence(run)
    masterpath=ROOT/'reports/pipeline_master_statistics.json';master=read(masterpath)
    require(CURRENT not in master and not (ROOT/POINTER).exists() and not (ROOT/REVIEW).exists(),'Already registered')
    for name,digest in m['previous_files'].items():require(sha256(ROOT/name)==digest,'Pre-registration state changed: '+name)
    older={n:h for n,h in m['previous_files'].items() if n.startswith('data/active_')}
    require(len(older)==29 and {str(p.relative_to(ROOT)) for p in (ROOT/'data').glob('active_*.json')}==set(older),'Prior pointer set differs')
    now=datetime.now(timezone.utc).isoformat();parents={k:m[k] for k in m if k.startswith('parent_')}
    entry={'stage':STAGE,'run_id':run.name,'status':STATUS,'scope':SCOPE,'completed_at_utc':now,**parents,
      'statistics':s,'verification':v,'tests_passed':t['tests_run'],'protocol':'reports/STAGE07M_JOINT_RESPONSE_PROTOCOL.md',
      'production_inference_approved':False,'downstream_ready':False}
    master['stage_history'].append(entry);master[CURRENT]=entry;master['pipeline_status']='joint_descriptions_verified_uncertainty_and_reporting_pending'
    master['repair_progress']={'stage':STAGE,'status':STATUS,'next_required_work':NEXT};master['updated_at_utc']=now
    pointer={'joint_response_run_id':run.name,'stage':STAGE,'status':STATUS,'scope':SCOPE,'primary_policy':'structural_only',**parents,
      'analysis_start':m['analysis_start'],'end_exclusive':m['end_exclusive'],'empirical_models_fitted':0,'uncertainty_computed':False,
      'production_inference_approved':False,'downstream_ready':False}
    wp=ROOT/'REPAIR_WORKFLOW.md';workflow=wp.read_text();marker='**Current checkpoint: stage 07l direction-cancellation sensitivity**'
    require(workflow.count(marker)==1,'Checkpoint differs');start=workflow.index(marker);end=workflow.index('**Current methodological priority:')
    workflow=workflow[:start]+('**Current checkpoint: stage 07m joint J/C/I descriptions**\n\n'
      f'Verified `{run.name}`: common-row initial, subsequent and total price movements, exact sign-aware endpoint descriptions, '
      'matched60/300 observations and full-calendar sums. This completes descriptive interpretation; it fits no new models and introduces no inference. '
      'All prior detector, state and regression results remain preserved.\n\nNext: '+NEXT+'\n\n')+workflow[end:]
    start=workflow.index('**Current methodological priority:');end=workflow.index('\n\n',start)
    workflow=workflow[:start]+'**Current methodological priority:** '+NEXT+workflow[end:]
    previous=('The provenance trace and the declared fixed-onset frequency, quote-age and direction-cancellation comparisons are complete. '
              'Complete joint J/C/I interpretation, then the remaining dependence/uncertainty assessment and final reporting/reproducibility closeout. '
              'Preserve the detector and fixed equations; no model, threshold or frequency selection by significance.')
    require(workflow.count(previous)==1,'Progression text differs')
    workflow=workflow.replace(previous,'The provenance trace, declared measurement sensitivities and descriptive joint J/C/I interpretation are complete. '+NEXT,1)
    cp=ROOT/'reports/STAGE07M_CHECKPOINT.md';lp=ROOT/'reports/PIPELINE_REPAIR_LOG.md'
    completion=(f'\n\nStage07m numerical work registered: `{run.name}`; {t["tests_run"]} tests and independent table verification passed. '
                'Zero regression fits and no new uncertainty claims. All29 older pointers preserved. PDF closeout pending.\n\nNext: '+NEXT+'\n')
    report=make_review(run,s,t)
    for x in [master,pointer]:json.dumps(x,allow_nan=False)
    (run/'review.md').write_text(report);(ROOT/REVIEW).write_text(report);m['outputs']['review.md']=sha256(run/'review.md')
    atomic_json(run/'manifest.json',m);atomic_json(masterpath,master);atomic_json(ROOT/POINTER,pointer)
    wp.write_text(workflow);lp.write_text(lp.read_text()+completion);cp.write_text(cp.read_text()+completion)
    require(all(sha256(ROOT/n)==h for n,h in older.items()),'Earlier pointer changed during registration')
    record={'status':'research_registered_documentation_pending','run_id':run.name,'registered_master_sha256':sha256(masterpath),
      'registered_pointer_sha256':sha256(ROOT/POINTER),'input_manifest_sha256':sha256(run/'manifest.json'),
      'registration_source_sha256':sha256(Path(__file__).resolve()),'previous_active_pointers_preserved':29,'active_pointer_count':30,
      'tests_passed':t['tests_run'],'empirical_models_fitted':0,'uncertainty_computed':False,'production_inference_approved':False,'downstream_ready':False}
    atomic_json(ROOT/'reports/stage07m_final_integrity.json',record);registered_evidence(run)
    print(json.dumps({'registered':run.name,'tests':t['tests_run'],'status':record['status']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(ROOT/'data/signals'/p.parse_args().run_id)

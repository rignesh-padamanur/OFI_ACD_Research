"""Stage 04c: verified completed quote waits, sequence resets and terminal exposure."""
from datetime import datetime, timezone
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
from src.data_preparation import atomic_json, sha256
from src.duration_observation import build_duration_artifacts
from src.verify_duration_observation import verify_duration_run

ROOT = Path(__file__).resolve().parent
STATUS = 'verified_duration_observation_only'
CODE = ['repair_duration_observation.py', 'src/duration_observation.py', 'src/verify_duration_observation.py',
    'src/re_estimate_acd.py', 'src/run_calibration_audit.py',
    'src/pipeline_state.py', 'src/eacd.py', 'src/weibull.py', 'src/data_preparation.py', 'src/observation_eligibility.py',
    'src/quote_imbalance_statistics.py', 'src/verify_quote_imbalance.py', 'tests/test_duration_observation.py']
METHOD = {
    'scope': 'Notebook 2, duration observation before seasonal duration normalization and ACD fitting.',
    'process': 'Retained recorded quote updates, not transactions, price changes only, or identified dealer/order arrivals.',
    'completed': 'Exact later timestamp minus preceding timestamp, eligible only within the same registered policy component. Available at the later quote. No duration before a component first quote is invented.',
    'sequences': 'Retain all quote rows. Component starts have unavailable incoming durations. recurrence_reset marks the first completed duration, not an estimated initial conditional mean. No state or lag may cross components.',
    'terminal': 'Separate incomplete waiting spell from the last quote to the earlier of the structural stop and freshness cap. event_observed=False means no retained endpoint quote inside this spell. Positive exposure is not automatic likelihood eligibility.',
    'endpoint': 'Freshness cap equality is allowed for completed waits. Remove indexing-only +1ns from freshness caps and final sample end; all other structural stops retain their exact timestamps. Zero terminal exposure is bookkeeping, not a positive censored observation.',
    'stop_codes': {'1':'freshness_cap','2':'registered_structural_boundary','4':'sample_end','combination':'Bitwise OR when boundaries coincide. Structural codes refer to audited support, not a uniquely identified market/feed cause.'},
    'timing': 'Completed durations arrive at recorded quote timestamps; terminal stops and component membership are conditional on retrospective support audits. True reception latency is unavailable. Audit flags and future end times are not Ft0 state features.',
    'likelihood_pending': 'Do not drop long durations then use an uncensored ACD likelihood. A later fit must specify conditional density contributions for observed completions and any justified survival contributions for positive terminal exposures, observation/censoring assumptions, initialization, seasonal time scale and prior-only calibration. Structural boundary censoring is not assumed independent.',
    'limits': ['These are observed recorded-stream waits conditional on an audited support convention. Missing updates and feed completeness cannot be resolved by timestamp subtraction.',
        'Freshness thresholds remain diagnostic scenarios. Terminal exposures preserve the consequence of the cap; they do not prove that elapsed quiet is a true market liquidity shock.',
        'No count-seasonality factor is automatically reused for durations. Endpoint seasonal fitting, initial-state effects, and rolling chronological fitting require separate validation.',
        'No duration is forward-filled onto empty clock bins. A future clock-age/surprise measure needs its own predictable timing contract.',
        'Completed-spell summaries are selected by support and are descriptive, not censoring-adjusted estimates of market waiting-time distributions.'],
    'sources': [{'title':'Statsmodels documentation: survival and duration analysis', 'url':'https://www.statsmodels.org/stable/duration.html',
        'scope':'Distinction between completed event durations and incomplete/right-censored observations. Does not establish independent censoring for this quote stream or prescribe an ACD fit.'}],
    'primary_freshness_policy':None, 'coverage_certified':False, 'downstream_ready':False}


def resolve_inputs(root):
    root = Path(root); active = json.loads((root/'data/active_run.json').read_text()); ep = json.loads((root/'data/active_eligibility.json').read_text())
    if active['data_run_id'] != ep['parent_data_run']: raise ValueError('Active data/eligibility lineage differs')
    data = root/'data/runs'/active['data_run_id']; elig = root/'data/eligibility'/ep['eligibility_run_id']
    dm = json.loads((data/'manifest.json').read_text()); em = json.loads((elig/'manifest.json').read_text())
    if dm['status'] != 'verified' or em['status'] != 'verified_observation_support_only' or em['parent_data_run'] != data.name:
        raise ValueError('Verified and matching data/eligibility required')
    for run in [data, elig]:
        if json.loads((run/'verification.json').read_text())['status'] != 'passed': raise ValueError('Unverified parent')
    paths = {'clean':data/'clean_quotes.parquet', 'raw':data/'raw_quotes.parquet', 'data_manifest':data/'manifest.json',
        'data_verification':data/'verification.json', 'segments':elig/'segments.parquet', 'support':elig/'support_intervals.parquet',
        'eligibility_manifest':elig/'manifest.json', 'eligibility_verification':elig/'verification.json', 'eligibility_contract':elig/'usage_contract.json'}
    inputs = {k:{'path':str(p), 'sha256':sha256(p)} for k,p in paths.items()}
    for k,file in [('clean','clean_quotes.parquet'),('raw','raw_quotes.parquet')]:
        if inputs[k]['sha256'] != dm['outputs'][file]['sha256']: raise ValueError('Changed quote data: '+k)
    if inputs['clean']['sha256'] != ep['parent_clean_sha256']: raise ValueError('Eligibility quote fingerprint differs')
    for k,file in [('segments','segments.parquet'),('support','support_intervals.parquet')]:
        if inputs[k]['sha256'] != em['output_sha256'][file]: raise ValueError('Changed eligibility: '+k)
    return inputs, {'parent_data_run':data.name, 'parent_eligibility_run':elig.name}


def run_tests():
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status':'passed' if result.wasSuccessful() else 'failed', 'tests_run':result.testsRun,
        'failures':len(result.failures), 'errors':len(result.errors), 'output':stream.getvalue(),
        'test_source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage04c_tests.json', evidence); print(stream.getvalue(), flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Tests failed; build and promotion refused')
    return evidence


def build_run(root, run_id, evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id): raise ValueError('Invalid run ID')
    inputs, parents = resolve_inputs(root); run = root/'data/signals'/run_id; run.mkdir(parents=True, exist_ok=False)
    m = {'schema_version':1, 'run_id':run_id, 'stage':'04c_duration_observation', 'status':'building', **parents,
        'created_at_utc':datetime.now(timezone.utc).isoformat(), 'inputs':inputs, 'production_batch_size':500000,
        'code_sha256':{}, 'primary_freshness_policy':None, 'coverage_certified':False, 'downstream_ready':False,
        'environment':{'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'pyarrow':pa.__version__}}
    for name in CODE:
        dest = run/'code'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, dest); m['code_sha256'][name] = sha256(dest)
    for name in sorted((root/'tests').glob('test_*.py')):
        dest = run/'tests'/name.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(name, dest)
    atomic_json(run/'manifest.json', m); atomic_json(run/'methodology.json', METHOD); atomic_json(run/'test_evidence.json', evidence)
    stats = build_duration_artifacts(inputs['clean']['path'], inputs['raw']['path'], pd.read_parquet(inputs['support']['path']), pd.read_parquet(inputs['segments']['path']), run)
    atomic_json(run/'statistics.json', stats)
    m['outputs'] = {name:sha256(run/name) for name in ['quote_durations.parquet','terminal_spells.parquet','monthly_statistics.parquet','statistics.json','methodology.json','test_evidence.json']}
    m['status'] = 'built_pending_verification'; atomic_json(run/'manifest.json', m)
    verification = verify_duration_run(run); atomic_json(run/'verification.json', verification)
    m['status'] = STATUS; atomic_json(run/'manifest.json', m)
    return run, stats, verification


def promote(root, run, stats, verification):
    m = json.loads((run/'manifest.json').read_text()); inputs, parents = resolve_inputs(root)
    if m['inputs'] != inputs or any(m[k] != v for k,v in parents.items()): raise ValueError('Inputs changed during execution')
    if m['status'] != STATUS or verification['status'] != 'passed': raise ValueError('Unverified run')
    master_path = root/'reports/pipeline_master_statistics.json'; master = json.loads(master_path.read_text())
    entry = {'stage':m['stage'], 'run_id':run.name, **parents, 'status':STATUS, 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'statistics':stats, 'verification':verification, 'methodology':METHOD, 'downstream_ready':False}
    master['stage_history'].append(entry); master['current_duration_stage'] = entry
    master['pipeline_status'] = 'duration_observation_verified_chronological_ACD_and_models_pending'
    master['downstream_status']['seasonality_acd'] = 'count_and_pressure_calibrated_duration_observation_verified_duration_seasonality_ACD_pending'
    master['repair_progress'] = {'stage':m['stage'], 'status':STATUS,
        'next_required_work':'04d: duration seasonal time scale, boundary/censoring assumptions and chronological ACD fitting with sequence resets. Then score/threshold/onset and inference. No primary support chosen.'}
    master['updated_at_utc'] = datetime.now(timezone.utc).isoformat(); atomic_json(master_path, master)
    atomic_json(root/'data/active_duration_observation.json', {'schema_version':1, 'duration_run_id':run.name, **parents,
        'status':STATUS, 'primary_freshness_policy':None, 'coverage_certified':False, 'downstream_ready':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='duration_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only', action='store_true'); parser.add_argument('--verify-only'); args = parser.parse_args()
    if args.verify_only:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.verify_only): parser.error('Invalid run ID')
        print(json.dumps(verify_duration_run(ROOT/'data/signals'/args.verify_only), indent=2)); return
    evidence = run_tests()
    if args.test_only: return
    run, stats, verification = build_run(ROOT, args.run_id, evidence); promote(ROOT, run, stats, verification)
    print(json.dumps({'run':str(run), 'statistics':stats, 'verification':verification}, indent=2), flush=True)


if __name__ == '__main__': main()

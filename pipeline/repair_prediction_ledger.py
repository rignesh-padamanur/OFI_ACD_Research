"""Build/reverify the bounded Notebook 5 prediction ledger; fit no model."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'reports/STEP09A_PREDICTION_CONTRACT.md'
START = pd.Timestamp('2024-05-06', tz='UTC')
STOP = pd.Timestamp('2026-08-01', tz='UTC')
STATUS = 'verified_prediction_input_ledger'
TABLES = ['origins', 'predictors', 'labels', 'folds', 'membership']
SOURCES = ['repair_prediction_ledger.py', 'src/prediction_ledger.py',
           'src/verify_prediction_ledger.py', 'tests/test_prediction_ledger.py',
           'tests/test_verify_prediction_ledger.py', 'reports/STEP09A_PREDICTION_CONTRACT.md']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path, expected=None):
    h = sha(path)
    require(expected is None or h == expected, 'Parent hash differs: ' + str(path))
    return {'path': str(Path(path).resolve()), 'sha256': h}


def parent_inputs():
    response = ROOT / 'data/signals/step07e_panel_responses_20261001'
    panel = ROOT / 'data/signals/step07f_panel_analysis_20261001'
    rm, pm = read(response / 'manifest.json'), read(panel / 'manifest.json')
    require(rm['status'] == 'verified_full_panel_price_responses', 'Unverified response parent')
    require(pm['status'] == 'verified_full_panel_analysis_inputs', 'Unverified feature parent')
    result = {'response_manifest': fingerprint(response / 'manifest.json'),
              'panel_manifest': fingerprint(panel / 'manifest.json')}
    for key, folder, manifest, name in [
        ('anchors', response, rm, 'anchors.parquet'), ('features', panel, pm, 'features.parquet'),
        ('outcomes', panel, pm, 'ledger_outcomes.parquet'), ('scaling', panel, pm, 'scaling.json')]:
        result[key] = fingerprint(folder / name, manifest['outputs'][name])
    value = rm['inputs']['cycle_schedule']
    result['schedule'] = fingerprint(value['path'], value['sha256'])
    # this is reconciliation evidence only. It never determines origin issuance.
    joint = ROOT / 'data/signals/step07m_joint_response_20261002'
    jm = read(joint / 'manifest.json')
    result['joint_manifest'] = fingerprint(joint / 'manifest.json')
    result['old_joint_membership'] = fingerprint(joint / 'membership.parquet', jm['outputs']['membership.parquet'])
    return result


def load_inputs(manifest):
    for entry in manifest['inputs'].values():
        fingerprint(entry['path'], entry['sha256'])
    result = {k: pd.read_parquet(manifest['inputs'][k]['path'])
              for k in ['anchors', 'features', 'outcomes', 'schedule', 'old_joint_membership']}
    result['scaling'] = read(manifest['inputs']['scaling']['path'])
    return result


def derive_statistics(artifacts, old_joint):
    """Summary from saved tables; assertions reconcile earlier selected cohorts."""
    origins, labels, folds, membership = (artifacts[k] for k in ['origins', 'labels', 'folds', 'membership'])
    horizon_stats = {}
    for h in [300, 60]:
        part = labels.loc[labels.horizon_sec.eq(h)]
        later = part.forecast_at.ge(START) & part.forecast_at.lt(STOP)
        eligible = part.origin_eligible
        evaluated = part.evaluation_eligible
        expected = set(old_joint.loc[old_joint.horizon_sec.eq(h), 'episode_id'])
        actual = set(part.loc[later & evaluated, 'episode_id'])
        require(actual == expected, 'Later evaluated IDs do not reconcile with original joint sample')
        horizon_stats[str(h)] = {
            'all_label_rows': len(part), 'origin_eligible': int(eligible.sum()),
            'origin_eligible_label_available': int(evaluated.sum()),
            'origin_eligible_label_unavailable': int((eligible & ~part.label_available).sum()),
            'later_origin_eligible': int((later & eligible).sum()),
            'later_evaluable': int((later & evaluated).sum()),
            'later_eligible_unavailable': int((later & eligible & ~part.label_available).sum()),
            'unavailable_reasons_among_eligible': {str(k): int(v) for k, v in
                part.loc[eligible & ~part.label_available, 'label_status'].value_counts().items()},
            'prior_selected_sample_ids_reconciled': True}
    warmup = origins.forecast_at.lt(START)
    return {
        'origins': artifacts['origin_statistics'], 'refits': artifacts['fold_statistics'],
        'labels': horizon_stats, 'warmup_origin_eligible': int((warmup & origins.origin_eligible).sum()),
        'total_label_rows': len(labels), 'membership_rows': len(membership),
        'earlier_not_yet_mature_fold_event_count': int(folds.earlier_labels_not_yet_mature.sum()),
        'earlier_mature_unavailable_fold_event_count': int(folds.earlier_mature_labels_unavailable.sum()),
        'first_refit_training_origins': int(folds.training_origins.iloc[0]),
        'last_refit_training_origins': int(folds.training_origins.iloc[-1]),
        'minimum_refit_training_origins': int(folds.training_origins.min()),
        'maximum_refit_training_origins': int(folds.training_origins.max()),
        'empirical_models_fitted': 0, 'forecasts_computed': 0, 'forecast_performance_computed': False,
        'forecast_validated': False, 'novelty_established': False,
        'causal_identification_established': False, 'paper_ready': False}


def focused_tests():
    command = [sys.executable, '-m', 'unittest', 'tests.test_prediction_ledger',
               'tests.test_verify_prediction_ledger', '-v']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    log = p.stdout + p.stderr
    if p.returncode:
        print(log)
        raise RuntimeError('Focused prediction-ledger tests failed; no run created')
    import re
    count = re.search(r'Ran (\d+) tests?', log)
    require(count is not None and '\nOK\n' in log, 'Missing successful test receipt')
    return log, int(count.group(1))


def verify_saved(run, *, allow_pending=False):
    from src.verify_prediction_ledger import verify_prediction_ledger
    m = read(run / 'manifest.json')
    permitted = {STATUS, 'pending_reload_verification'} if allow_pending else {STATUS}
    require(m['status'] in permitted, 'Saved run not verified')
    for relative, digest in m['source_sha256'].items():
        require(sha(ROOT / relative) == digest and sha(run / 'code' / relative) == digest,
                'Executed source changed: ' + relative)
    for relative, digest in m['outputs'].items():
        require(sha(run / relative) == digest, 'Saved output changed: ' + relative)
    inputs = load_inputs(m)
    artifacts = {k: pd.read_parquet(run / (k + '.parquet')) for k in TABLES}
    artifacts.update(origin_statistics=read(run / 'origin_statistics.json'), fold_statistics=read(run / 'fold_statistics.json'))
    result = verify_prediction_ledger(inputs['anchors'], inputs['features'], inputs['scaling'],
        inputs['outcomes'], inputs['schedule'], artifacts, sample_end=STOP, evaluation_start=START)
    require(result['status'] == 'passed', 'Independent saved verification failed')
    require(derive_statistics(artifacts, inputs['old_joint_membership']) == read(run / 'statistics.json'),
            'Saved statistics do not reconstruct')
    return {'status': 'passed', 'run_id': run.name, 'verified_at_utc': now(),
            'manifest_sha256': sha(run / 'manifest.json'), 'verification': result,
            'parent_files_verified': len(m['inputs']), 'output_files_verified': len(m['outputs']),
            'source_files_verified': len(m['source_sha256']), 'empirical_models_fitted': 0}


def build(run_id):
    from src.prediction_ledger import build_origins, attach_labels, build_folds
    from src.verify_prediction_ledger import verify_prediction_ledger
    require(run_id.replace('_', '').replace('-', '').isalnum(), 'Simple unique run ID required')
    run = ROOT / 'data/signals' / run_id
    require(not run.exists(), 'Run already exists; use a new ID, never overwrite')
    log, count = focused_tests()
    inputs_meta = parent_inputs()
    run.mkdir()
    hashes = {}
    for relative in SOURCES:
        dest = run / 'code' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
        hashes[relative] = sha(dest)
    (run / 'test_execution.log').write_text(log)
    write(run / 'test_evidence.json', {'status': 'passed', 'tests_passed': count,
        'log_sha256': sha(run / 'test_execution.log'), 'source_sha256': hashes})
    m = {'run_id': run_id, 'stage': '09a_prediction_input_ledger', 'status': 'running',
         'started_at_utc': now(), 'inputs': inputs_meta, 'source_sha256': hashes,
         'protocol_sha256': sha(PROTOCOL), 'evaluation_start': START.isoformat(), 'sample_end': STOP.isoformat(),
         'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__},
         'empirical_models_fitted': 0, 'forecasts_computed': 0, 'forecast_validated': False,
         'novelty_established': False, 'causal_identification_established': False,
         'untouched_holdout': False, 'live_feed_latency_validated': False}
    write(run / 'manifest.json', m)
    try:
        i = load_inputs(m)
        built = build_origins(i['anchors'], i['features'], i['scaling'])
        artifacts = {k: built[k] for k in ['origins', 'predictors']}
        artifacts['origin_statistics'] = built['statistics']
        artifacts['labels'] = attach_labels(artifacts['origins'], i['outcomes'], sample_end=STOP)
        refits = build_folds(artifacts['origins'], artifacts['labels'], i['schedule'],
                            evaluation_start=START, sample_end=STOP)
        artifacts.update(folds=refits['folds'], membership=refits['membership'], fold_statistics=refits['statistics'])
        print('Built all origin, label and fold ledgers; independent verification begins.', flush=True)
        verified = verify_prediction_ledger(i['anchors'], i['features'], i['scaling'], i['outcomes'],
            i['schedule'], artifacts, sample_end=STOP, evaluation_start=START)
        require(verified['status'] == 'passed', 'Independent verification failed')
        for k in TABLES:
            artifacts[k].to_parquet(run / (k + '.parquet'), index=False)
        for k in ['origin_statistics', 'fold_statistics']:
            write(run / (k + '.json'), artifacts[k])
        write(run / 'statistics.json', derive_statistics(artifacts, i['old_joint_membership']))
        write(run / 'verification.json', verified)
        m['outputs'] = {str(p.relative_to(run)): sha(p) for p in sorted(run.iterdir())
                        if p.is_file() and p.name != 'manifest.json'}
        m.update(status='pending_reload_verification', tests_passed=count)
        write(run / 'manifest.json', m)
        result = verify_saved(run, allow_pending=True)
        # persist successful reload evidence before promoting the run. An
        # interruption at any earlier point therefore cannot certify completion.
        reload_evidence = {k: v for k, v in result.items() if k != 'manifest_sha256'}
        write(run / 'saved_verification.json', reload_evidence)
        m['outputs']['saved_verification.json'] = sha(run / 'saved_verification.json')
        m.update(status=STATUS, completed_at_utc=now())
        write(run / 'manifest.json', m)
        result['manifest_sha256'] = sha(run / 'manifest.json')
        receipt = ROOT / 'reports/tests' / (run_id + '_saved_verification.json')
        write(receipt, result)
        print(json.dumps({'run_id': run_id, 'status': STATUS, 'tests_passed': count,
                          'statistics': read(run / 'statistics.json')}, indent=2))
    except Exception as exc:
        m.update(status='failed', failed_at_utc=now(), failure_type=type(exc).__name__, failure=str(exc))
        write(run / 'manifest.json', m)
        raise


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-id')
    group.add_argument('--verify-only')
    group.add_argument('--test-only', action='store_true')
    args = p.parse_args()
    if args.test_only:
        log, count = focused_tests()
        print(log, end='')
    elif args.verify_only:
        require(Path(args.verify_only).name == args.verify_only, 'Run ID only')
        print(json.dumps(verify_saved(ROOT / 'data/signals' / args.verify_only), indent=2))
    else:
        build(args.run_id)


if __name__ == '__main__':
    main()

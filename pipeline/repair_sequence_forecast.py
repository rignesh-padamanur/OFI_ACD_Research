"""Frozen stage10b sequence replay: save forecasts before evaluating future outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
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
import scipy

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'data/signals/step09a_prediction_ledger_20261003'
PROTOCOL = ROOT / 'reports/STEP10B_FORECAST_PROTOCOL.md'
MEASUREMENT = ROOT / 'data/signals/step10a_sequence_measurement_20261003'
START = pd.Timestamp('2024-05-06', tz='UTC')
STOP = pd.Timestamp('2026-08-01', tz='UTC')
STATUS = 'verified_exploratory_sequence_forecasts'
SOURCE_PATHS = ['repair_sequence_forecast.py', 'src/sequence_forecast.py',
    'src/verify_sequence_forecast.py', 'tests/test_sequence_forecast.py',
    'tests/test_verify_sequence_forecast.py', 'reports/STEP10B_FORECAST_PROTOCOL.md', 'reports/STEP10B_LITERATURE_ADDENDUM.md']
PARENT_TABLES = ['origins', 'labels', 'folds', 'membership']
from src.sequence_forecast import (CONTINUOUS, PREDICTORS, IDENTIFIERS, ORIGIN_COLUMNS,
    LABEL_COLUMNS, FOLD_COLUMNS, MEMBERSHIP_COLUMNS)
READ_ALLOWLISTS = {'origins': list(ORIGIN_COLUMNS), 'labels': list(LABEL_COLUMNS),
    'folds': list(FOLD_COLUMNS), 'membership': list(MEMBERSHIP_COLUMNS),
    'features': [*IDENTIFIERS, *PREDICTORS, 'inherited_origin_eligible',
                 'measurement_available', 'combined_eligible', 'adjacent_agreement']}
FIT_TABLES = ['forecasts', 'coefficients', 'fit_diagnostics']
EVALUATION_TABLES = ['scored_events', 'metrics', 'comparisons', 'fold_losses',
                     'daily_losses', 'quarter_losses', 'serial_dependence']


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path, expected=None):
    value = sha(path)
    require(expected is None or value == expected, 'Hash changed: ' + str(path))
    return {'path': str(Path(path).resolve()), 'sha256': value}


def parent_binding():
    m = read(PARENT / 'manifest.json')
    require(m['status'] == 'verified_prediction_input_ledger', 'Unverified09a parent')
    pointer = read(ROOT / 'data/active_prediction_ledger_audit.json')
    require(pointer['prediction_ledger_run_id'] == PARENT.name and
            pointer['manifest_sha256'] == sha(PARENT / 'manifest.json'), 'Wrong active09a parent')
    audit = ROOT / 'reports/tests/step09b_parent_reverification.json'
    require(read(audit)['status'] == 'passed', '09a audit missing')
    require(read(audit)['saved_09a_reverification']['manifest_sha256'] == sha(PARENT / 'manifest.json'),
            '09a re-verification receipt binds another run')
    inputs = {'parent_manifest': fingerprint(PARENT / 'manifest.json'),
              'parent_reverification': fingerprint(audit)}
    for name in PARENT_TABLES:
        inputs[name] = fingerprint(PARENT / (name + '.parquet'), m['outputs'][name + '.parquet'])
    measured = read(MEASUREMENT / 'manifest.json')
    require(measured['status'] == 'verified_recorded_sequence_measurement', 'Unverified10a measurement')
    require(measured['numeric_baseline_columns'] == list(CONTINUOUS)
            and measured['categorical_baseline_columns'] == ['onset_hour_utc', 'onset_weekday_utc']
            and measured['extension_columns'] == ['sequence_z'], 'Frozen10a design differs')
    inputs['measurement_manifest'] = fingerprint(MEASUREMENT / 'manifest.json')
    for name in ['features.parquet', 'statistics.json', 'independent_verification.json']:
        inputs['features' if name == 'features.parquet' else 'measurement_' + name[:-5]] = fingerprint(
            MEASUREMENT / name, measured['outputs'][name])
    receipt = read(MEASUREMENT / 'independent_verification.json')
    require(receipt['status'] == 'passed' and receipt['check_count'] == 91,
            'Missing independent10a source verification')
    require(receipt['serialized_output_sha256']['features.parquet'] == measured['outputs']['features.parquet'],
            'Independent measurement receipt binds another feature table')
    return inputs


def load_parent(manifest):
    for value in manifest['inputs'].values():
        fingerprint(value['path'], value['sha256'])
    frames = {name: pd.read_parquet(manifest['inputs'][name]['path'], columns=READ_ALLOWLISTS[name])
              for name in PARENT_TABLES}
    features = pd.read_parquet(manifest['inputs']['features']['path'], columns=READ_ALLOWLISTS['features'])
    origins = frames['origins'].set_index('episode_id', drop=False).sort_index()
    require(features.episode_id.is_unique and set(features.episode_id) == set(origins.index),
            'Measurement/09a origin identity mismatch')
    features = features.set_index('episode_id', drop=False).loc[origins.index]
    for name in ['bin_start', 'forecast_at', 'direction', 'initial_pips']:
        require(features[name].equals(origins[name]), 'Measurement/09a identity differs: ' + name)
    for name in ['inherited_origin_eligible', 'measurement_available', 'combined_eligible']:
        require(pd.api.types.is_bool_dtype(features[name]) and features[name].notna().all(),
                'Invalid measurement availability: ' + name)
    require(features.inherited_origin_eligible.equals(origins.origin_eligible)
            and features.combined_eligible.equals(origins.origin_eligible)
            and features.loc[origins.origin_eligible, 'measurement_available'].all(),
            'Measurement altered inherited eligibility; sample repair is not authorized')
    require(features.combined_eligible.equals(features.inherited_origin_eligible & features.measurement_available),
            'Measurement eligibility identity differs')
    frames['predictors'] = features[[*IDENTIFIERS, *PREDICTORS]].reset_index(drop=True)
    return frames


def verify_projection(parent, manifest):
    from src.verify_sequence_forecast import verify_source_projection
    source = manifest['inputs']['features']
    fingerprint(source['path'], source['sha256'])
    features = pd.read_parquet(source['path'], columns=READ_ALLOWLISTS['features'])
    return verify_source_projection(parent, features)


def tests():
    command = [sys.executable, '-m', 'unittest', 'tests.test_sequence_forecast',
               'tests.test_verify_sequence_forecast', '-v']
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1'))
    log = p.stdout + p.stderr
    if p.returncode:
        print(log)
        raise RuntimeError('Focused forecast tests failed; no empirical run created')
    match = re.search(r'Ran (\d+) tests?', log)
    require(match is not None and '\nOK\n' in log, 'Test receipt missing')
    return log, int(match.group(1))


def save_tables(run, result, names, statistics_name):
    for name in names:
        require(isinstance(result[name], pd.DataFrame), 'Expected table: ' + name)
        result[name].to_parquet(run / (name + '.parquet'), index=False)
    save(run / (statistics_name + '.json'), result['statistics'])


def load_tables(run, names, statistics_name):
    result = {name: pd.read_parquet(run / (name + '.parquet')) for name in names}
    result['statistics'] = read(run / (statistics_name + '.json'))
    return result


def assert_frozen(run, manifest):
    for relative, digest in manifest['source_sha256'].items():
        require(sha(ROOT / relative) == digest and sha(run / 'code' / relative) == digest,
                'Executed source changed: ' + relative)
    for name, digest in manifest.get('forecast_outputs', {}).items():
        require(sha(run / name) == digest, 'Pre-evaluation forecast artifact changed: ' + name)


def verify_saved(run, *, allow_pending=False):
    from src.verify_sequence_forecast import verify_forecasts, verify_evaluation
    m = read(run / 'manifest.json')
    permitted = {STATUS, 'pending_reload_verification'} if allow_pending else {STATUS}
    require(m['status'] in permitted, 'Run not verified')
    assert_frozen(run, m)
    for name, digest in m['outputs'].items():
        require(sha(run / name) == digest, 'Output changed: ' + name)
    parent = load_parent(m)
    projection = verify_projection(parent, m)
    require(read(run / 'source_projection_verification.json') == projection,
            'Saved source projection receipt differs')
    fit = load_tables(run, FIT_TABLES, 'forecast_statistics')
    evaluated = load_tables(run, EVALUATION_TABLES, 'evaluation_statistics')
    fcheck = verify_forecasts(parent, fit)
    echeck = verify_evaluation(fit['forecasts'], parent['labels'], evaluated, start=START, end=STOP)
    require(fcheck['status'] == echeck['status'] == 'passed', 'Saved reconstruction failed')
    require(read(run / 'statistics.json') == {'forecast': fit['statistics'], 'evaluation': evaluated['statistics']},
            'Combined statistics differ')
    return {'status': 'passed', 'run_id': run.name, 'verified_at_utc': now(),
        'manifest_sha256': sha(run / 'manifest.json'), 'source_projection': projection,
        'forecast': fcheck, 'evaluation': echeck,
        'new_scientific_models_fitted': 0, 'verification_reconstructions_only': True}


def build(run_id):
    before_tests = {path: sha(ROOT / path) for path in SOURCE_PATHS}
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', run_id)), 'Simple unique run ID required')
    run = ROOT / 'data/signals' / run_id
    require(not run.exists(), 'Run exists; never overwrite an empirical attempt')
    log, count = tests()
    inputs = parent_binding()
    run.mkdir()
    hashes = {}
    for path in SOURCE_PATHS:
        dest = run / 'code' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
        hashes[path] = sha(dest)
        require(hashes[path] == before_tests[path] == sha(ROOT / path),
                'Source changed during tests or freezing: ' + path)
    (run / 'test_execution.log').write_text(log)
    save(run / 'test_evidence.json', {'status': 'passed', 'tests_passed': count,
        'source_sha256': hashes, 'log_sha256': sha(run / 'test_execution.log')})
    m = {'run_id': run_id, 'stage': '10b_sequence_forecast', 'status': 'frozen_before_fitting',
        'started_at_utc': now(), 'inputs': inputs, 'source_sha256': hashes,
        'protocol_sha256': sha(PROTOCOL), 'parent_run_id': PARENT.name,
        'measurement_run_id': MEASUREMENT.name, 'read_allowlists': READ_ALLOWLISTS,
        'numeric_baseline_columns': list(CONTINUOUS), 'extension_columns': ['sequence_z'],
        'active_pointer_changed': False, 'master_statistics_changed': False,
        'evaluation_start': START.isoformat(), 'end_exclusive': STOP.isoformat(), 'tests_passed': count,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__},
        'causal_identification_established': False, 'novelty_established': False,
        'untouched_holdout': False, 'formal_forecast_inference_approved': False, 'paper_ready': False}
    save(run / 'manifest.json', m)
    try:
        from src.sequence_forecast import fit_forecasts, evaluate_forecasts
        from src.verify_sequence_forecast import verify_forecasts, verify_evaluation
        assert_frozen(run, m)
        parent = load_parent(m)
        projection = verify_projection(parent, m)
        require(projection['status'] == 'passed', 'Source projection failed')
        save(run / 'source_projection_verification.json', projection)
        require(len(parent['folds']) == 59, 'Inherited59-fold schedule changed')
        fit = fit_forecasts(**parent)
        require(fit['statistics']['ols_fits'] == 118 and fit['statistics']['training_mean_calibrations'] == 59
                and fit['statistics']['empirical_models_fitted'] == 177, 'Fixed model accounting changed')
        save_tables(run, fit, FIT_TABLES, 'forecast_statistics')
        frozen = [name + '.parquet' for name in FIT_TABLES] + ['forecast_statistics.json']
        m['forecast_outputs'] = {name: sha(run / name) for name in frozen}
        m.update(status='forecasts_saved_before_evaluation', forecasts_saved_at_utc=now())
        save(run / 'manifest.json', m)
        print('Forecast artifacts saved; independent QR reconstruction begins.', flush=True)
        fit = load_tables(run, FIT_TABLES, 'forecast_statistics')
        fcheck = verify_forecasts(parent, fit)
        require(fcheck['status'] == 'passed', 'Forecast reconstruction failed')
        save(run / 'forecast_verification.json', fcheck)
        assert_frozen(run, m)
        m.update(status='forecasts_verified_before_evaluation', evaluation_started_at_utc=now())
        save(run / 'manifest.json', m)
        evaluated = evaluate_forecasts(fit['forecasts'], parent['labels'], start=START, end=STOP)
        save_tables(run, evaluated, EVALUATION_TABLES, 'evaluation_statistics')
        evaluated = load_tables(run, EVALUATION_TABLES, 'evaluation_statistics')
        echeck = verify_evaluation(fit['forecasts'], parent['labels'], evaluated, start=START, end=STOP)
        require(echeck['status'] == 'passed', 'Evaluation reconstruction failed')
        save(run / 'evaluation_verification.json', echeck)
        save(run / 'statistics.json', {'forecast': fit['statistics'], 'evaluation': evaluated['statistics']})
        assert_frozen(run, m)
        m['outputs'] = {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name != 'manifest.json'}
        m.update(status='pending_reload_verification')
        save(run / 'manifest.json', m)
        result = verify_saved(run, allow_pending=True)
        save(run / 'saved_verification.json', {k: v for k, v in result.items() if k != 'manifest_sha256'})
        m['outputs']['saved_verification.json'] = sha(run / 'saved_verification.json')
        m.update(status=STATUS, completed_at_utc=now(), empirical_model_accounting=fit['statistics'])
        save(run / 'manifest.json', m)
        result['manifest_sha256'] = sha(run / 'manifest.json')
        save(ROOT / 'reports/tests' / (run_id + '_saved_verification.json'), result)
        print(json.dumps({'status': STATUS, 'run_id': run_id, 'tests_passed': count,
                          'statistics': read(run / 'statistics.json')}, indent=2))
    except Exception as exc:
        m.update(status='failed', failed_at_utc=now(), failure_type=type(exc).__name__, failure=str(exc))
        save(run / 'manifest.json', m)
        raise


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run-id')
    choice.add_argument('--verify-only')
    choice.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.test_only:
        print(tests()[0], end='')
    elif args.verify_only:
        require(Path(args.verify_only).name == args.verify_only, 'Run ID only')
        print(json.dumps(verify_saved(ROOT / 'data/signals' / args.verify_only), indent=2))
    else:
        build(args.run_id)


if __name__ == '__main__':
    main()

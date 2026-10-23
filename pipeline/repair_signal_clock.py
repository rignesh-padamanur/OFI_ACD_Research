"""Stage 04a: build and independently verify a supported one-second signal clock."""
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
import pyarrow.parquet as pq
from src.data_preparation import atomic_json, sha256
from src.signal_clock import SECOND, SparseCursor, prepare_clock_inputs, build_clock_chunk, ClockStatistics
from src.verify_signal_clock import verify_clock_run

ROOT = Path(__file__).resolve().parent
CODE = ['repair_signal_clock.py', 'src/signal_clock.py', 'src/intensity.py', 'src/verify_signal_clock.py',
        'src/pipeline_state.py', 'src/quote_imbalance_statistics.py', 'src/verify_quote_imbalance.py',
        'src/observation_eligibility.py', 'src/data_preparation.py', 'tests/test_signal_clock.py']


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage04a_tests.json', evidence)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('Tests failed; no build or promotion')
    return evidence


def resolve_inputs(root):
    data = json.loads((root/'data/active_run.json').read_text())
    ep = json.loads((root/'data/active_eligibility.json').read_text())
    sp = json.loads((root/'data/active_quote_imbalance.json').read_text())
    if ep['parent_data_run'] != data['data_run_id'] or sp['parent_data_run'] != data['data_run_id'] or sp['parent_eligibility_run'] != ep['eligibility_run_id']:
        raise ValueError('Active parent lineages disagree')
    signal = root/'data/signals'/sp['signal_run_id']; elig = root/'data/eligibility'/ep['eligibility_run_id']
    sm = json.loads((signal/'manifest.json').read_text()); em = json.loads((elig/'manifest.json').read_text())
    if sm['status'] != 'verified_recorded_quote_proxy' or em['status'] != 'verified_observation_support_only':
        raise ValueError('Verified signal and eligibility required')
    if sm['parent_data_run'] != data['data_run_id'] or sm['parent_eligibility_run'] != elig.name or em['parent_data_run'] != data['data_run_id']:
        raise ValueError('Parent manifest lineage differs')
    for parent in [signal, elig]:
        if json.loads((parent/'verification.json').read_text())['status'] != 'passed':
            raise ValueError('Parent verification failed')
    paths = {name: Path(item['path']) for name, item in sm['inputs'].items()}
    paths.update({'signal_manifest': signal/'manifest.json', 'signal_verification': signal/'verification.json',
        'sparse_bins': signal/'quote_ofi_1s.parquet', 'signal_statistics': signal/'statistics.json',
        'hourly_eligibility': elig/'hourly_eligibility.parquet'})
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}
    for name, item in sm['inputs'].items():
        if inputs[name]['sha256'] != item['sha256']:
            raise ValueError('Signal ancestor changed: '+name)
    for name, file in [('sparse_bins', 'quote_ofi_1s.parquet'), ('signal_statistics', 'statistics.json')]:
        if inputs[name]['sha256'] != sm['outputs'][file]:
            raise ValueError('Signal output changed: '+name)
    if inputs['hourly_eligibility']['sha256'] != em['output_sha256']['hourly_eligibility.parquet']:
        raise ValueError('Hourly eligibility changed')
    if inputs['parent_clean']['sha256'] != ep['parent_clean_sha256']:
        raise ValueError('Eligibility quote fingerprint changed')
    return inputs, data['data_run_id'], elig.name, signal.name


def build_artifacts(inputs, run, *, batch_size=216_000):
    hours = pd.read_parquet(inputs['hourly_eligibility']['path'])
    support = pd.read_parquet(inputs['support']['path'])
    origin, stop, hour_codes, intervals = prepare_clock_inputs(hours, support)
    cursor = SparseCursor(inputs['sparse_bins']['path']); stats = ClockStatistics(); writer = None
    source_quotes = 0; source_bins = 0
    try:
        for i, start in enumerate(range(origin, stop, batch_size*SECOND)):
            end = min(stop, start+batch_size*SECOND)
            source = cursor.take(start, end)
            source_quotes += int(source.n_quotes.sum()); source_bins += len(source)
            frame = build_clock_chunk(start, end, source, origin=origin, hour_codes=hour_codes, intervals=intervals)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(run/'signal_clock_1s.parquet', table.schema, compression='zstd')
            writer.write_table(table); stats.add(frame)
            if (i+1) % 40 == 0:
                print(f'Built {(end-origin)//SECOND:,} clock rows', flush=True)
    finally:
        if writer is not None:
            writer.close()
    if not cursor.exhausted() or stats.total['recorded_updates'] != source_quotes or source_bins != pq.ParquetFile(inputs['sparse_bins']['path']).metadata.num_rows:
        raise AssertionError('Clock did not preserve every sparse quote count')
    if 'signal_statistics' in inputs:
        parent = json.loads(Path(inputs['signal_statistics']['path']).read_text())
        if source_quotes != parent['ticks']['quotes'] or source_bins != parent['bins']['quote_containing_bins']:
            raise AssertionError('Clock differs from verified parent signal counts')
    result = stats.result(); atomic_json(run/'statistics.json', result)
    month, hour = stats.tables()
    month.to_parquet(run/'monthly_statistics.parquet', index=False)
    hour.to_parquet(run/'utc_hour_statistics.parquet', index=False)
    return result


METHOD = {
    'scope': 'Notebook 2 Signal Construction: aligned raw quote-update count and recorded quote imbalance; no fitting.',
    'grid': 'All requested calendar seconds in the verified hourly ledger, including unsupported rows. Half-open [s,s+1s), available_at=s+1s. UTC.',
    'support': 'Entire bin contained in one registered support interval. End equality allowed. Retrospective coverage labels are not Ft0 predictors.',
    'no_update': 'Zero recorded updates and zero recorded contributions only in fully supported empty bins. This does not establish zero market activity or feed completeness.',
    'count': 'Number of retained quote records per second (not transactions or price changes). Supported counts remain valid when an imbalance transition is invalid.',
    'coverage_denominator': 'Hourly ledger rows count raw records, whereas signal counts use retained clean records. Quote conservation is against verified stage03, not the raw hourly ledger.',
    'imbalance': 'Active-bin proxy inherited unchanged from verified stage03; unavailable if any invalid transition or policy-exceeding incoming gap. No forward fill.',
    'hour_status_codes': {'0': 'weekly_closed', '1': 'absent_nominal_open_unresolved', '2': 'observed_hour_not_completeness_certified'},
    'flow_status_codes': {'0': 'unsupported', '1': 'supported_no_recorded_updates', '2': 'supported_valid_active_flow',
        '3': 'supported_invalid_transition', '4': 'supported_excess_or_unknown_incoming_gap'},
    'storage': 'recorded_update_count is an audit count (zero if no retained records, including unsupported rows); observed_quote_ofi is missing for empty bins. Neither is a model column. Use materialize_clock_policy or iter_signal_clock with an explicit policy.',
    'support_ids': 'Zero-based per-policy interval ordinals; -1 for unsupported bins. Different IDs must never be merged by rolling operations, even if bins are adjacent. Retrospective metadata, never regressors.',
    'downstream': 'Use contiguous valid bins in one support ID for rolling inputs. Preserve missing rows; do not drop them before lagging. Use available_at for completed signals. Ex-ante features require a separately fixed onset and lag contract. No durations or ACD inputs are created by zero counts.',
    'primary_freshness_policy': None, 'coverage_certified': False, 'downstream_ready': False,
    'limitations': ['Quiet versus missing within observed hours is not identified; 15/30/60-second policies are diagnostics, not validated outage detectors.',
        'Existing cached-source omissions and lost same-timestamp size updates remain unrecovered.',
        'The proxy remains feed-specific recorded quote pressure, not identified executed order flow.',
        'Descriptive moments are not fitted normalization parameters. Seasonal adjustment, duration treatment, onset timing and inference remain pending.']}


def build_run(root, run_id, evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run id')
    inputs, parent, elig, signal = resolve_inputs(root)
    run = root/'data/signals'/run_id; run.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'run_id': run_id, 'stage': '04a_supported_signal_clock', 'status': 'building',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'parent_data_run': parent,
        'parent_eligibility_run': elig, 'parent_signal_run': signal, 'inputs': inputs, 'production_batch_size': 216000,
        'coverage_certified': False, 'downstream_ready': False, 'primary_freshness_policy': None,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pa.__version__},
        'code_sha256': {}}
    for name in CODE:
        dest = run/'code'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, dest)
        manifest['code_sha256'][name] = sha256(dest)
    for file in (root/'tests').glob('test_*.py'):
        dest = run/'full_test_sources'/file.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(file, dest)
    atomic_json(run/'manifest.json', manifest); atomic_json(run/'methodology.json', METHOD)
    atomic_json(run/'test_evidence.json', evidence)
    statistics = build_artifacts(inputs, run)
    manifest['outputs'] = {name: sha256(run/name) for name in ['signal_clock_1s.parquet', 'statistics.json',
        'monthly_statistics.parquet', 'utc_hour_statistics.parquet', 'methodology.json', 'test_evidence.json']}
    manifest['status'] = 'built_pending_independent_verification'; atomic_json(run/'manifest.json', manifest)
    print('Starting independent full-clock verification', flush=True)
    verification = verify_clock_run(run); atomic_json(run/'verification.json', verification)
    manifest['status'] = 'verified_supported_signal_clock'; atomic_json(run/'manifest.json', manifest)
    return run, statistics, verification


def promote(root, run, statistics, verification):
    m = json.loads((run/'manifest.json').read_text())
    # recheck live pointers and every ancestor fingerprint before promotion.
    inputs, parent, elig, signal = resolve_inputs(root)
    if inputs != m['inputs'] or (parent, elig, signal) != (m['parent_data_run'], m['parent_eligibility_run'], m['parent_signal_run']):
        raise ValueError('Active parents changed during build')
    if m['status'] != 'verified_supported_signal_clock' or verification['status'] != 'passed':
        raise ValueError('Unverified clock cannot be promoted')
    path = root/'reports/pipeline_master_statistics.json'; master = json.loads(path.read_text())
    entry = {'stage': m['stage'], 'run_id': run.name, 'parent_data_run': parent, 'parent_eligibility_run': elig,
        'parent_signal_run': signal, 'status': m['status'], 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'statistics': statistics, 'verification': verification, 'methodology': METHOD,
        'primary_freshness_policy': None, 'downstream_ready': False}
    master['stage_history'].append(entry); master['current_signal_clock_stage'] = entry
    master['pipeline_status'] = 'signal_clock_verified_past_only_calibration_and_models_pending'
    master['downstream_status']['intensity'] = 'supported_recorded_quote_counts_verified_seasonal_adjustment_pending'
    master['downstream_status']['ofi'] = 'supported_clock_proxy_verified_normalization_pending'
    master['repair_progress'] = {'stage': m['stage'], 'status': m['status'],
        'next_required_work': 'Past-only seasonal calibration and normalization, with explicit primary support/sampling and duration treatment before event or model fitting.'}
    master['updated_at_utc'] = datetime.now(timezone.utc).isoformat(); atomic_json(path, master)
    atomic_json(root/'data/active_signal_clock.json', {'schema_version': 1, 'clock_run_id': run.name,
        'parent_data_run': parent, 'parent_eligibility_run': elig, 'parent_signal_run': signal, 'status': m['status'],
        'coverage_certified': False, 'primary_freshness_policy': None, 'downstream_ready': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='clock_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only', action='store_true'); parser.add_argument('--verify-only', metavar='RUN_ID')
    args = parser.parse_args()
    if args.verify_only:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.verify_only):
            parser.error('Invalid run id')
        print(json.dumps(verify_clock_run(ROOT/'data/signals'/args.verify_only), indent=2)); return
    evidence = run_tests()
    if args.test_only:
        return
    run, stats, verification = build_run(ROOT, args.run_id, evidence)
    promote(ROOT, run, stats, verification)
    print(json.dumps({'run': str(run), 'statistics': stats, 'verification': verification}, indent=2), flush=True)


if __name__ == '__main__':
    main()

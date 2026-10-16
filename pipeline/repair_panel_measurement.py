"""Stage07d: frozen measurement sensitivities on the verified structural panel."""
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
import pyarrow
from src.data_preparation import atomic_json, sha256
from src.panel_observations import read_quote_window
from src.state_measurement import direction_diagnostics, direction_summary
from src.panel_measurement import make_windows, measure_windows, summarize

ROOT = Path(__file__).resolve().parent
STAGE = '07d_full_panel_fixed_onset_measurement'
STATUS = 'verified_full_panel_fixed_onset_measurement'
SCOPE = 'full_cached_structural_fixed_onset_measurement_only'
CAPS = [15, 30, 60]
FREQUENCIES = [5, 15, 30]


def read(path): return json.loads(Path(path).read_text())
def records(frame): return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15))


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage07d_tests.json', evidence)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Tests failed; no build')
    return evidence


def resolve_inputs():
    panel = ROOT/'data/signals'/read(ROOT/'data/active_chronological_panel_audit.json')['panel_run_id']
    pm = read(panel/'manifest.json')
    if pm['status'] != 'verified_full_chronological_baseline_panel' or read(panel/'verification.json')['status'] != 'passed':
        raise ValueError('Verified full structural panel required')
    if pm['primary_policy'] != 'structural_only': raise ValueError('Changed primary support')
    inputs = {}
    def add(key, path, expected=None):
        path = Path(path); digest = sha256(path)
        if expected is not None and digest != expected: raise ValueError('Changed parent: '+str(path))
        inputs[key] = {'path': str(path), 'sha256': digest}
    for key, name in [('parent_manifest', 'manifest.json'), ('parent_verification', 'verification.json'),
                      ('parent_statistics', 'statistics.json'), ('onsets', 'onsets.parquet'), ('cycle_schedule', 'cycle_schedule.parquet')]:
        add(key, panel/name, pm.get('outputs', {}).get(name))
    for key in ['clean_quotes', 'support_intervals']:
        source = pm['inputs'][key]; add(key, source['path'], source['sha256'])
    op = read(ROOT/'data/active_quote_imbalance.json')
    ofi = ROOT/'data/signals'/op['signal_run_id']; om = read(ofi/'manifest.json')
    if om['status'] != 'verified_recorded_quote_proxy' or read(ofi/'verification.json')['status'] != 'passed':
        raise ValueError('Verified recorded imbalance required')
    for source_key, target in [('parent_clean', 'clean_quotes'), ('support', 'support_intervals')]:
        if om['inputs'][source_key] != inputs[target]: raise ValueError('OFI source/support lineage mismatch')
    add('ofi_manifest', ofi/'manifest.json'); add('ofi_verification', ofi/'verification.json')
    add('ofi_ticks', ofi/'quote_ofi_ticks.parquet', om['outputs']['quote_ofi_ticks.parquet'])
    return panel, pm, inputs


def build_calendar(onsets, measurements, direction, start, stop):
    five = measurements.loc[measurements.frequency_sec.eq(5)].set_index('episode_id').loc[onsets.episode_id]
    direction = direction.set_index('episode_id').loc[onsets.episode_id]
    eligible = onsets.state_sample_eligible.to_numpy()
    directed = eligible & direction.inherited_direction.ne(0).to_numpy()
    masks = {'all_onsets': np.ones(len(onsets), dtype=bool),
        'confirmed_onsets': onsets.onset_confirmed.to_numpy(),
        'state_history_available': onsets.state_history_available.to_numpy(),
        'state_sample_eligible': eligible, 'variation_available': five.status.eq('available').to_numpy(),
        'direction_nonzero': directed}
    for cap in CAPS: masks['common_age_le_'+str(cap)] = eligible & five['common_age_le_'+str(cap)].to_numpy()
    for threshold in [1, 5, 10]:
        masks['direction_ratio_gt_'+str(threshold)+'pct'] = directed & direction['ratio_gt_'+str(threshold)+'pct'].to_numpy()
    dates = onsets.bin_start.dt.normalize()
    calendar = pd.DataFrame({'date': pd.date_range(start, stop, freq='D', inclusive='left')})
    for name, mask in masks.items():
        counts = dates.loc[mask].value_counts()
        calendar[name] = calendar.date.map(counts).fillna(0).astype('int64')
    if int(calendar.all_onsets.sum()) != len(onsets): raise ValueError('Lost onset calendar accounting')
    return calendar


def statistics(onsets, windows, measurements, direction, tables, calendar, start, stop):
    five = measurements.loc[measurements.frequency_sec.eq(5)].set_index('episode_id').loc[onsets.episode_id]
    history = onsets.state_history_available.to_numpy()
    eligible = onsets.state_sample_eligible.to_numpy()
    available = five.status.eq('available').to_numpy()
    difference = np.abs(five.rv_bps.to_numpy()[history]-onsets.pre_rv_300s_5s_bps.to_numpy()[history])
    return {'scope': SCOPE, 'primary_policy': 'structural_only',
        'deployment_start': start.isoformat(), 'deployment_end_exclusive': stop.isoformat(),
        'onsets': len(onsets), 'confirmed_onsets': int(onsets.onset_confirmed.sum()),
        'uncertain_onsets': int((~onsets.onset_confirmed).sum()),
        'inherited_history_available': int(history.sum()), 'state_sample_eligible': int(eligible.sum()),
        'measurement_rows': len(measurements), 'frequencies_seconds': FREQUENCIES,
        'status_counts_by_frequency': {str(f): {str(k): int(v) for k, v in g.status.value_counts().sort_index().items()}
            for f, g in measurements.groupby('frequency_sec')},
        'raw_price_available_onsets': int(available.sum()),
        'raw_price_available_missing_state_history': int((available & ~history).sum()),
        'five_second_reconciliation': {'rows': int(history.sum()), 'max_absolute_difference_bps': float(difference.max(initial=0)),
            'rtol': 1e-9, 'atol': 1e-9},
        'common_age_state_counts': {str(c): int((eligible & five['common_age_le_'+str(c)].to_numpy()).sum()) for c in CAPS},
        'own5_age_state_counts': {str(c): int((eligible & five['own_age_le_'+str(c)].to_numpy()).sum()) for c in CAPS},
        'summary': records(tables['summary']), 'comparisons': records(tables['comparisons']),
        'direction_summary': direction_summary(direction),
        'direction_mask_state_counts': {str(c): int(calendar['direction_ratio_gt_'+str(c)+'pct'].sum()) for c in [1, 5, 10]},
        'calendar': {'days': len(calendar), 'days_with_onsets': int(calendar.all_onsets.gt(0).sum()),
            'days_with_state_eligible_onsets': int(calendar.state_sample_eligible.gt(0).sum()),
            'totals': {k: int(calendar[k].sum()) for k in calendar if k != 'date'}},
        'baseline_references_preserved': True, 'detector_fits': 0, 'empirical_response_models_fitted': 0,
        'price_responses_computed': False, 'detector_support_sensitivities_extended': False,
        'production_inference_approved': False, 'downstream_ready': False}


def build(run_id, evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id): raise ValueError('Invalid run identifier')
    panel, pm, inputs = resolve_inputs()
    run = ROOT/'data/signals'/run_id; run.mkdir(exist_ok=False)
    snapshot = ROOT/'reports/repair_history'/('2026-10-01_pre_'+run_id); snapshot.mkdir(exist_ok=False)
    previous = {}
    for path in [ROOT/'reports/pipeline_master_statistics.json', ROOT/'reports/PIPELINE_REPAIR_LOG.md',
                 ROOT/'REPAIR_WORKFLOW.md', *sorted((ROOT/'data').glob('active_*.json'))]:
        rel = str(path.relative_to(ROOT)); dest = snapshot/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); previous[rel] = sha256(path)
    atomic_json(snapshot/'preserved_files.json', previous)
    m = {'run_id': run_id, 'stage': STAGE, 'status': 'building', 'scope': SCOPE,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
        'parent_chronological_panel': panel.name, 'primary_policy': 'structural_only',
        'frequencies_seconds': FREQUENCIES, 'age_caps_seconds': CAPS, 'window_seconds': 300,
        'direction_ratio_strict_thresholds': [0.01, 0.05, 0.10],
        'deployment_start': pm['first_cutoff'], 'deployment_end_exclusive': pm['deployment_end'],
        'production_inference_approved': False, 'downstream_ready': False,
        'previous_files': previous, 'snapshot_path': str(snapshot), 'code_sha256': {},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'pyarrow': pyarrow.__version__}}
    sources = [Path(__file__), ROOT/'reports/STAGE07D_PANEL_MEASUREMENT_PROTOCOL.md',
        *sorted((ROOT/'src').glob('*.py')), *sorted((ROOT/'tests').glob('test_*.py')),
        *sorted((ROOT/'reports/tests').glob('test_panel_measurement_verifier.py'))]
    for path in sources:
        rel = str(path.relative_to(ROOT)); dest = run/'code'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); m['code_sha256'][rel] = sha256(dest)
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        if sha256(path) != evidence['test_source_sha256'][path.name]: raise ValueError('Test changed after execution')
        dest = run/'full_test_sources'/path.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(path, dest)
    atomic_json(run/'manifest.json', m); atomic_json(run/'test_evidence.json', evidence)
    try:
        onsets = pd.read_parquet(inputs['onsets']['path'])
        schedule = pd.read_parquet(inputs['cycle_schedule']['path'])
        support = pd.read_parquet(inputs['support_intervals']['path']).loc[lambda f: f.policy.eq('structural_only')].reset_index(drop=True)
        windows = make_windows(onsets); windows.to_parquet(run/'windows.parquet', index=False)
        parts, directions = [], []
        (run/'cycles').mkdir()
        for row in schedule.itertuples(index=False):
            selected = onsets.loc[onsets.bin_start.ge(row.parameter_cutoff) & onsets.bin_start.lt(row.deployment_end)].reset_index(drop=True)
            wins = windows.loc[windows.end_at.ge(row.parameter_cutoff) & windows.end_at.lt(row.deployment_end)].reset_index(drop=True)
            quotes = read_quote_window(inputs['clean_quotes']['path'], row.parameter_cutoff-pd.Timedelta(seconds=300),
                row.deployment_end, ['timestamp', 'bid', 'ask'])
            measured = measure_windows(wins, quotes, support)
            ticks = pd.read_parquet(inputs['ofi_ticks']['path'], columns=['timestamp', 'exclusion_code', 'quote_ofi'],
                filters=[('timestamp', '>=', row.parameter_cutoff), ('timestamp', '<', row.deployment_end)])
            direction = direction_diagnostics(selected, ticks)
            for cutoff in [1, 5, 10]: direction['ratio_gt_'+str(cutoff)+'pct'] = direction.cancellation_ratio.gt(cutoff/100)
            directory = run/'cycles'/f'{row.cycle_id:03d}'; directory.mkdir()
            measured.to_parquet(directory/'measurements.parquet', index=False)
            direction.to_parquet(directory/'direction.parquet', index=False)
            atomic_json(directory/'statistics.json', {'cycle_id': int(row.cycle_id), 'onsets': len(selected),
                'measurement_rows': len(measured), 'direction_rows': len(direction)})
            parts.append(measured); directions.append(direction)
            print(f'Built measurement cycle {row.cycle_id:03d}: {len(selected):,} onsets', flush=True)
            del quotes, ticks
            gc.collect()
        measured = pd.concat(parts, ignore_index=True).sort_values(['window_id', 'frequency_sec']).reset_index(drop=True)
        direction = pd.concat(directions, ignore_index=True).reset_index(drop=True)
        measured.to_parquet(run/'measurements.parquet', index=False)
        direction.to_parquet(run/'direction.parquet', index=False)
        tables = summarize(measured)
        for name, table in tables.items(): table.to_parquet(run/(name+'.parquet'), index=False)
        start, stop = pd.Timestamp(pm['first_cutoff']), pd.Timestamp(pm['deployment_end'])
        calendar = build_calendar(onsets, measured, direction, start, stop)
        calendar.to_parquet(run/'calendar.parquet', index=False)
        stats = statistics(onsets, windows, measured, direction, tables, calendar, start, stop)
        atomic_json(run/'statistics.json', stats)
        m['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()
            and p.relative_to(run).parts[0] not in ['code', 'full_test_sources'] and p != run/'manifest.json'}
        m['status'] = 'built_pending_independent_verification'; atomic_json(run/'manifest.json', m)
        return run
    except Exception as error:
        atomic_json(run/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        m['status'] = 'failed_unregistered'; atomic_json(run/'manifest.json', m)
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id'); parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--verify-only'); args = parser.parse_args()
    if args.verify_only: run = ROOT/'data/signals'/args.verify_only
    else:
        evidence = run_tests()
        if args.test_only: return
        if not args.run_id: parser.error('--run-id required')
        run = build(args.run_id, evidence)
    from src.verify_panel_measurement import verify_panel_measurement
    try:
        verification = verify_panel_measurement(run)
        if verification['status'] != 'passed': raise RuntimeError('Independent verification failed')
    except Exception as error:
        if not args.verify_only:
            atomic_json(run/'verification_failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
            m = read(run/'manifest.json'); m['status'] = 'failed_independent_verification'
            m['outputs']['verification_failure.json'] = sha256(run/'verification_failure.json')
            atomic_json(run/'manifest.json', m)
        raise
    if args.verify_only:
        print(json.dumps(verification, indent=2)); return
    atomic_json(run/'verification.json', verification)
    m = read(run/'manifest.json'); m['status'] = STATUS; m['outputs']['verification.json'] = sha256(run/'verification.json')
    atomic_json(run/'manifest.json', m)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__': main()

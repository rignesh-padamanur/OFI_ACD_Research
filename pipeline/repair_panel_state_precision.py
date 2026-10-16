"""Stage07c numerical repair: preserve detector fits; stabilize prior variation.

The failed build is an explicitly named numerical source, never a verified
ancestor. Complete independent verification is mandatory before promotion.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import json
import shutil
import traceback
import numpy as np
import pandas as pd
from src.data_preparation import atomic_json, sha256
from src.panel_observations import read_quote_window
from src.panel_state_precision import quote_diagnostics
from src.intensity import materialize_clock_policy

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data/signals/step07c_chronological_panel_20260930'
SOURCE_HASH = 'a54717a628dd2464f14b3e68197dae535d90fb74e59cc7034bd8eede6496d7b0'
RUN_ID = 'step07c_chronological_panel_20261001_v2'
METRIC = 'pre_rv_300s_5s_bps'


def read(path): return json.loads(Path(path).read_text())


def source_check():
    if sha256(SOURCE/'manifest.json') != SOURCE_HASH:
        raise ValueError('Failed source manifest changed')
    m = read(SOURCE/'manifest.json')
    if m['status'] != 'failed_independent_verification':
        raise ValueError('Expected preserved failed source')
    for name, expected in m['outputs'].items():
        if sha256(SOURCE/name) != expected:
            raise ValueError('Failed source artifact changed: '+name)
    return m


def audit_difference(old, new):
    a, b = old.to_numpy(), new.to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    finite = np.isfinite(a)
    np.testing.assert_array_equal(finite, np.isfinite(b))
    differences = np.abs(a[finite]-b[finite])
    outside = finite & ~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    return {'rows': len(a), 'finite_rows': int(finite.sum()),
        'exact_value_changes': int(np.count_nonzero(a[finite] != b[finite])),
        'existing_tolerance_exceedances': int(outside.sum()),
        'max_absolute_change_bps': float(differences.max(initial=0))}, outside


def build():
    original = source_check()
    run = ROOT/'data/signals'/RUN_ID
    run.mkdir(exist_ok=False)
    m = {k: v for k, v in original.items() if k not in
        ['outputs', 'code_sha256', 'postfit_code_sha256', 'verification_reference_repair', 'verification_failure']}
    source_record = {'path': str(SOURCE/'manifest.json'), 'sha256': SOURCE_HASH}
    m.update(run_id=RUN_ID, status='building', created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={}, state_precision_repair={
            'source_manifest': source_record, 'changed_metric': METRIC,
            'inherited_cycles_preserved': [0, 1], 'model_refits': 0,
            'reference_tolerance': {'rtol': 1e-9, 'atol': 1e-9},
            'test_evidence': 'state_precision_test_evidence.json',
            'reference_test_evidence': 'reference_precision_test_evidence.json',
            'audit': 'state_precision_audit.json',
            'method': 'Direct 60-term nonnegative squared-return sums on each five-second phase; unchanged economic definition'})
    m['inputs'] = dict(original['inputs'], numerical_build_manifest=source_record)
    sources = [Path(__file__), ROOT/'repair_chronological_panel.py', ROOT/'repair_chronological_transition.py',
        ROOT/'repair_event_audit.py', ROOT/'reports/STAGE07C_PANEL_PROTOCOL.md',
        *sorted((ROOT/'src').glob('*.py')), ROOT/'src/acd_kernel.c',
        ROOT/'reports/tests/test_panel_state_precision.py', ROOT/'reports/tests/test_panel_reference_precision.py',
        ROOT/'reports/tests/test_stage07c_v3_guards.py']
    for path in sources:
        name = str(path.relative_to(ROOT)); dest = run/'code'/name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        m['code_sha256'][name] = sha256(dest)
    atomic_json(run/'manifest.json', m)
    try:
        for name in ['observations', 'cycles', 'full_test_sources']:
            shutil.copytree(SOURCE/name, run/name)
        for path in SOURCE.iterdir():
            if path.is_file() and path.name in original['outputs']:
                shutil.copy2(path, run/path.name)
        for source, name in [('stage07c_state_precision_tests.json', 'state_precision_test_evidence.json'),
                              ('stage07c_reference_precision_tests.json', 'reference_precision_test_evidence.json'),
                              ('stage07c_current_precision_regression.json', 'current_precision_regression_evidence.json'),
                              ('stage07c_current_precision_regression.log', 'current_precision_regression.log'),
                              ('stage07c_v3_guard_preflight.log', 'v3_guard_preflight.log')]:
            shutil.copy2(ROOT/'reports/tests'/source, run/name)
        state_test = read(run/'state_precision_test_evidence.json')
        if not (state_test.get('passed') is True or state_test.get('status') == 'passed'):
            raise ValueError('State precision tests did not pass')
        for name, expected in state_test['source_sha256'].items():
            if sha256(ROOT/name) != expected: raise ValueError('State precision tested source changed: '+name)
        if read(run/'reference_precision_test_evidence.json')['status'] != 'passed':
            raise ValueError('Reference precision tests did not pass')
        support = pd.read_parquet(m['inputs']['support_intervals']['path'])
        support = support.loc[support.policy.eq('structural_only')].reset_index(drop=True)
        schedule = pd.read_parquet(run/'cycle_schedule.parquet')
        rows, onset_parts = [], []
        for row in schedule.itertuples(index=False):
            directory = run/'cycles'/f'{row.cycle_id:03d}'
            start = row.parameter_cutoff-pd.Timedelta(seconds=300)
            clock = pd.read_parquet(m['inputs']['clock']['path'],
                filters=[('bin_start', '>=', start), ('bin_start', '<', row.deployment_end)])
            clock = materialize_clock_policy(clock, policy='structural_only')
            quotes = read_quote_window(m['inputs']['clean_quotes']['path'], start, row.deployment_end,
                ['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume'])
            values = quote_diagnostics(clock, quotes, support, 'structural_only').set_index('bin_start')
            scored = pd.read_parquet(directory/'scored.parquet')
            mapped = values.reindex(scored.bin_start)
            for name in values.columns:
                if name != METRIC:
                    pd.testing.assert_series_equal(scored[name].reset_index(drop=True),
                        mapped[name].reset_index(drop=True), check_exact=True)
            delta, outside = audit_difference(scored[METRIC], mapped[METRIC])
            delta.update(cycle_id=int(row.cycle_id), inherited_preserved=row.cycle_id < 2,
                candidate_tolerance_exceedances=int(np.count_nonzero(outside & scored.candidate_event.to_numpy())))
            onsets = pd.read_parquet(directory/'onsets.parquet')
            onset_values = values.reindex(onsets.bin_start)[METRIC]
            onset_delta, onset_outside = audit_difference(onsets[METRIC], onset_values)
            delta['onsets'] = onset_delta
            if row.cycle_id < 2:
                np.testing.assert_allclose(scored[METRIC], mapped[METRIC], rtol=1e-9, atol=1e-9, equal_nan=True)
                np.testing.assert_allclose(onsets[METRIC], onset_values, rtol=1e-9, atol=1e-9, equal_nan=True)
            else:
                scored[METRIC] = mapped[METRIC].to_numpy()
                onsets[METRIC] = onset_values.to_numpy()
                for name, frame in [('scored', scored), ('onsets', onsets)]:
                    frame.to_parquet(directory/(name+'.parquet'), index=False, row_group_size=400000)
                cm = read(directory/'manifest.json')
                for name in ['scored.parquet', 'onsets.parquet']: cm['outputs'][name] = sha256(directory/name)
                atomic_json(directory/'manifest.json', cm)
            onset_parts.append(onsets); rows.append(delta)
            print(f"Stable-state cycle {row.cycle_id:03d}: {delta['rows']:,} scores, {delta['existing_tolerance_exceedances']} old-tolerance discrepancies", flush=True)
            del clock, quotes, values, mapped, scored
            gc.collect()
        combined = pd.concat(onset_parts, ignore_index=True)
        old_onsets = pd.read_parquet(SOURCE/'onsets.parquet')
        pd.testing.assert_frame_equal(combined.drop(columns=METRIC), old_onsets.drop(columns=METRIC), check_exact=True)
        combined.to_parquet(run/'onsets.parquet', index=False, row_group_size=400000)
        audit = {'status': 'built_pending_independent_verification', 'metric': METRIC,
            'cycles_checked': len(rows), 'cycles_recomputed': len(rows)-2,
            'model_refits': 0, 'missingness_unchanged': True,
            'detector_and_episode_columns_unchanged': True,
            'scored_tolerance_exceedances_corrected': sum(r['existing_tolerance_exceedances'] for r in rows[2:]),
            'candidate_tolerance_exceedances_corrected': sum(r['candidate_tolerance_exceedances'] for r in rows[2:]),
            'onset_tolerance_exceedances_corrected': sum(r['onsets']['existing_tolerance_exceedances'] for r in rows[2:]),
            'max_absolute_scored_change_bps': max(r['max_absolute_change_bps'] for r in rows[2:]),
            'cycles': rows}
        atomic_json(run/'state_precision_audit.json', audit)
        m['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()
            and p.relative_to(run).parts[0] not in ['code', 'full_test_sources'] and p != run/'manifest.json'}
        m['status'] = 'built_pending_independent_verification'
        atomic_json(run/'manifest.json', m)
        return run
    except Exception as error:
        atomic_json(run/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        m['status'] = 'failed_unregistered'; atomic_json(run/'manifest.json', m)
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = ROOT/'data/signals'/RUN_ID if args.verify_only else build()
    from src.verify_chronological_panel_v3 import verify_panel
    try:
        result = verify_panel(run)
        if result['status'] != 'passed': raise RuntimeError('Independent verification did not pass')
    except Exception as error:
        if not args.verify_only:
            atomic_json(run/'verification_failure.json', {'status': 'failed',
                'error': str(error), 'traceback': traceback.format_exc()})
            m = read(run/'manifest.json'); m['status'] = 'failed_independent_verification'
            m['outputs']['verification_failure.json'] = sha256(run/'verification_failure.json')
            atomic_json(run/'manifest.json', m)
        raise
    if args.verify_only:
        print(json.dumps(result, indent=2)); return
    atomic_json(run/'verification.json', result)
    m = read(run/'manifest.json'); m['status'] = 'verified_full_chronological_baseline_panel'
    m['outputs']['verification.json'] = sha256(run/'verification.json')
    atomic_json(run/'manifest.json', m)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()

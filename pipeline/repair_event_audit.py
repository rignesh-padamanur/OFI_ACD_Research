"""Stage 05a: frozen original-detector audit on the existing development prefix."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import io
import json
import platform
import re
import shutil
import unittest
import numpy as np
import pandas as pd
import pyarrow
import scipy
from src.data_preparation import atomic_json, sha256
from src.intensity import materialize_clock_policy
from src.duration_calibration import model_sample
from src.chronological_acd import predict, get_kernel
from src.event_audit import (FEATURES, METRICS, completed_duration_bins, fit_detector,
                             score_detector, quote_diagnostics, diagnostics_tables, ns,
                             classify_bin_eligibility)

ROOT = Path(__file__).resolve().parent
POLICIES = ['structural_only', 'fresh_15s', 'fresh_30s', 'fresh_60s']
STATUS = 'verified_original_detector_development_audit'
REASONS = ['unsupported', 'invalid_flow', 'seasonal_unavailable', 'quiet_no_duration',
           'incomplete_duration', 'eligible']
CODE = ['repair_event_audit.py', 'src/event_audit.py', 'src/verify_event_audit.py',
        'tests/test_event_audit.py', 'src/duration_calibration.py', 'src/chronological_acd.py',
        'src/acd_kernel.c', 'src/intensity.py', 'src/observation_eligibility.py',
        'src/data_preparation.py', 'src/acquisition.py', 'src/verify_chronological_acd.py',
        'src/verify_quote_imbalance.py', 'reports/STAGE05A_EVENT_AUDIT_PROTOCOL.md',
        'reports/stage05a_closeout.py']


def read(path):
    return json.loads(Path(path).read_text())


def resolve_inputs():
    ptr = {p.name: read(p) for p in (ROOT / 'data').glob('active_*.json')}
    parent = ROOT / 'data/signals' / ptr['active_duration_model_audit.json']['audit_run_id']
    calibration = ROOT / 'data/signals' / ptr['active_calibration.json']['calibration_run_id']
    clock = ROOT / 'data/signals' / ptr['active_signal_clock.json']['clock_run_id']
    data = ROOT / 'data/runs' / ptr['active_run.json']['data_run_id']
    eligibility = ROOT / 'data/eligibility' / ptr['active_eligibility.json']['eligibility_run_id']
    expected = {}
    for directory in [parent, calibration, clock, data, eligibility]:
        m = read(directory / 'manifest.json')
        if read(directory / 'verification.json')['status'] != 'passed':
            raise ValueError('Unverified input: ' + str(directory))
        for field, value in [('parent_data_run', data.name), ('parent_eligibility_run', eligibility.name)]:
            if field in m and m[field] != value:
                raise ValueError('Mixed parent lineage: ' + str(directory))
        for record in m.get('inputs', {}).values():
            path, digest = record['path'], record['sha256']
            if path in expected and expected[path] != digest:
                raise ValueError('Conflicting ancestor hashes')
            expected[path] = digest
    paths = {'clock': clock / 'signal_clock_1s.parquet',
             'calibration_parameters': calibration / 'calibration_parameters.parquet',
             'clean_quotes': data / 'clean_quotes.parquet',
             'support_intervals': eligibility / 'support_intervals.parquet',
             'acd_statistics': parent / 'statistics.json'}
    for policy in POLICIES:
        paths['seasonal_spells_' + policy] = parent / policy / 'seasonal_spells.parquet'
        paths['saved_predictions_' + policy] = parent / policy / 'weibull_acd_validation_predictions.parquet'
    for key, path in paths.items():
        directory = parent if key.startswith(('seasonal_spells_', 'saved_predictions_')) else path.parent
        manifest = read(directory / 'manifest.json')
        outputs = manifest.get('outputs', manifest.get('output_sha256', {}))
        name = str(path.relative_to(directory))
        item = outputs[name]
        expected[str(path)] = item['sha256'] if isinstance(item, dict) else item
    for directory in [parent, calibration, clock, data, eligibility]:
        for name in ['manifest.json', 'verification.json']:
            paths[directory.name + '_' + name] = directory / name
    for path, digest in expected.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed input: ' + path)
    inputs = {key: {'path': str(path), 'sha256': expected[str(path)] if str(path) in expected else sha256(path)}
              for key, path in paths.items()}
    return inputs, expected, parent


def tests():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed',
                'tests_run': result.testsRun, 'failures': len(result.failures),
                'errors': len(result.errors), 'output': stream.getvalue(),
                'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT / 'tests').glob('test_*.py'))}}
    print(stream.getvalue(), flush=True)
    atomic_json(ROOT / 'reports/tests/stage05a_tests.json', evidence)
    if not result.wasSuccessful():
        raise RuntimeError('Tests failed')
    return evidence


def add_adjusted(clock, parameters, policy):
    out = materialize_clock_policy(clock, policy=policy)
    dates = clock.bin_start.dt.normalize()
    slot = clock.bin_start.dt.hour * 4 + clock.bin_start.dt.minute // 15
    target = pd.MultiIndex.from_arrays([dates, slot])
    for channel in ['tick_count', 'abs_quote_ofi']:
        part = parameters.loc[parameters.policy.eq(policy) & parameters.channel.eq(channel)].set_index(['effective_date', 'slot'])
        if part.index.has_duplicates:
            raise ValueError('Duplicate seasonal cells')
        p = part.reindex(target)
        valid = p.status.to_numpy() == 5
        phi = p.phi.to_numpy()
        if (valid & (~np.isfinite(phi) | (phi <= 0))).any():
            raise ValueError('Invalid ready factor')
        out['adjusted_' + channel] = np.divide(out[channel].to_numpy(), phi,
            out=np.full(len(clock), np.nan), where=valid)
    return out


def build_policy(run, policy, inputs, clock, quotes, support, factors, parent_stats):
    directory = run / policy
    directory.mkdir()
    start, cutoff, stop = [pd.Timestamp(parent_stats[k]) for k in
                           ['training_start', 'parameter_cutoff', 'validation_end_exclusive']]
    frame = pd.read_parquet(inputs['seasonal_spells_' + policy]['path'])
    path = model_sample(frame, start_ns=start.value, stop_ns=stop.value)
    del frame
    model = parent_stats['policies'][policy]['models']['weibull_acd']
    if model['parameter_cutoff'] != cutoff.isoformat():
        raise ValueError('ACD parameter cutoff mismatch')
    p = np.array([model['parameters'][k] for k in ['omega', 'alpha', 'beta', 'shape']])
    predictions = predict(path, p, weibull=True)
    saved = pd.read_parquet(inputs['saved_predictions_' + policy]['path'])
    mask = path.origin_at.ge(cutoff)
    pd.testing.assert_frame_equal(predictions.loc[mask].reset_index(drop=True), saved,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    duration = completed_duration_bins(path, predictions)
    duration.to_parquet(directory / 'duration_bins.parquet', index=False)
    cross_boundary = int((path.origin_at.lt(cutoff) & path.end_at.ge(cutoff) & path.event_observed).sum())
    f = add_adjusted(clock, factors, policy)
    d = duration.set_index('bin_start').reindex(f.bin_start)
    n = d.completed_residual_count.fillna(0).to_numpy(dtype='int64')
    recorded = clock.recorded_update_count.to_numpy()
    if (n > recorded).any():
        raise ValueError('More incoming residuals than recorded quotes')
    ready = np.isfinite(f[['adjusted_abs_quote_ofi', 'adjusted_tick_count']]).all(axis=1).to_numpy()
    status = f.flow_status.to_numpy()
    good_flow = np.isin(status, [1, 2])
    reasons = classify_bin_eligibility(status, ready, recorded, n)
    eligible = reasons == 5
    period = np.where(f.bin_start.lt(cutoff), 0, 1).astype('uint8')
    opportunities = pd.DataFrame({'bin_start': f.bin_start, 'available_at': f.available_at,
        'period': pd.Categorical.from_codes(period, ['calibration', 'development']), 'recorded_update_count': recorded,
        'support_id': f.support_id, 'flow_status': status, 'seasonal_ready': ready,
        'completed_residual_count': n, 'eligible_without_duration': good_flow & ready,
        'reason': pd.Categorical.from_codes(reasons, REASONS)})
    opportunities.to_parquet(directory / 'opportunities.parquet', index=False)
    economic = quote_diagnostics(f, quotes, support.loc[support.policy.eq(policy)].reset_index(drop=True), policy)
    selected = f.loc[eligible, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                               'adjusted_abs_quote_ofi', 'adjusted_tick_count']].copy().reset_index(drop=True)
    selected['recorded_update_count'] = recorded[eligible]
    selected['completed_residual_count'] = n[eligible]
    selected['mean_duration_residual'] = d.mean_duration_residual.to_numpy()[eligible]
    selected['negative_log_mean_residual'] = -np.log(selected.mean_duration_residual)
    selected['period'] = pd.Categorical.from_codes(period[eligible], ['calibration', 'development'])
    for metric in METRICS:
        selected[metric] = economic[metric].to_numpy()[eligible]
    del f, economic, d, path, predictions
    x = selected[FEATURES].to_numpy()
    training = selected.period.eq('calibration').to_numpy()
    fit3, fit2 = fit_detector(x[training]), fit_detector(x[training, :2])
    selected['distance'] = score_detector(x, fit3)
    selected['distance_without_duration'] = score_detector(x[:, :2], fit2)
    selected['candidate_event'] = selected.distance.ge(fit3['threshold'])
    selected['candidate_without_duration'] = selected.distance_without_duration.ge(fit2['threshold'])
    z = (x - fit3['mean']) / fit3['sample_std']
    selected[['z_ofi', 'z_intensity', 'z_duration']] = z
    selected.to_parquet(directory / 'scored_bins.parquet', index=False)
    params = {'parameter_cutoff': cutoff.isoformat(), 'training_start': start.isoformat(),
              'features': FEATURES, 'three_feature': fit3, 'two_feature': fit2,
              'acd_parameters': model['parameters'],
              'acd_constraint_warning': model['numerical_constraint_warning'],
              'original_acd_diagnostic_flags': model['validation']['diagnostic_flags']}
    atomic_json(directory / 'parameters.json', params)
    tables = diagnostics_tables(selected)
    for name, table in tables.items():
        table.to_parquet(directory / (name + '.parquet'), index=False)
    stats = {'policy': policy, 'cross_cutoff_completed_waits_included': cross_boundary,
             'three_feature_threshold': fit3['threshold'], 'two_feature_threshold': fit2['threshold'],
             'acd_constraint_warning': model['numerical_constraint_warning'], 'periods': {}}
    for period_name in ['calibration', 'development']:
        o = opportunities.loc[opportunities.period.eq(period_name)]
        s = selected.loc[selected.period.eq(period_name)]
        counts = o.reason.value_counts()
        a, b = s.candidate_event.to_numpy(), s.candidate_without_duration.to_numpy()
        both, union = int((a & b).sum()), int((a | b).sum())
        starts = a.copy()
        if len(a) > 1:
            connected = (np.diff(ns(s.bin_start)) == 1_000_000_000) & (np.diff(s.support_id) == 0)
            starts[1:] &= ~(a[:-1] & connected)
        stats['periods'][period_name] = {'clock_seconds': len(o),
            'attrition': {r: int(counts.get(r, 0)) for r in REASONS},
            'eligible_without_duration': int(o.eligible_without_duration.sum()),
            'eligible_bins': len(s), 'candidate_bins': int(a.sum()),
            'candidate_rate': float(a.mean()), 'candidate_without_duration_bins': int(b.sum()),
            'candidate_overlap_bins': both, 'candidate_union_bins': union,
            'candidate_jaccard': both / union if union else None,
            'consecutive_exceedance_runs': int(starts.sum()),
            'threshold_ties': int(s.distance.eq(fit3['threshold']).sum()),
            'candidate_pre_history_missing': int(s.loc[s.candidate_event, 'pre_rv_300s_5s_bps'].isna().sum())}
        assert sum(stats['periods'][period_name]['attrition'].values()) == len(o)
    atomic_json(directory / 'statistics.json', stats)
    print(policy + ': ' + json.dumps(stats['periods']['development']), flush=True)
    return stats


def build(run_id, evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run identifier')
    inputs, ancestor_hashes, parent = resolve_inputs()
    run = ROOT / 'data/signals' / run_id
    run.mkdir(exist_ok=False)
    snapshot = ROOT / 'reports/repair_history' / ('2026-09-28_pre_' + run_id)
    snapshot.mkdir(exist_ok=False)
    previous = {}
    for path in [ROOT / 'reports/pipeline_master_statistics.json', ROOT / 'reports/PIPELINE_REPAIR_LOG.md',
                 ROOT / 'REPAIR_WORKFLOW.md', *sorted((ROOT / 'data').glob('active_*.json'))]:
        rel = str(path.relative_to(ROOT)); dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        previous[rel] = sha256(path)
    m = {'run_id': run_id, 'stage': '05a_original_detector_development_audit', 'status': 'building',
         'created_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
         'ancestor_hashes': ancestor_hashes, 'parent_model_audit_path': str(parent),
         'previous_files': previous, 'snapshot_path': str(snapshot), 'code_sha256': {},
         'downstream_ready': False, 'production_events_approved': False, 'primary_freshness_policy': None,
         'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'pyarrow': pyarrow.__version__, 'scipy': scipy.__version__}}
    for name in CODE:
        dest = run / 'code' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest); m['code_sha256'][name] = sha256(dest)
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        dest = run / 'full_test_sources' / path.name; dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        if sha256(dest) != evidence['test_source_sha256'][path.name]:
            raise ValueError('Test source changed after execution')
    atomic_json(run / 'manifest.json', m)
    atomic_json(run / 'test_evidence.json', evidence)
    stats = read(inputs['acd_statistics']['path'])
    start, stop = [pd.Timestamp(stats[k]) for k in ['training_start', 'validation_end_exclusive']]
    clock = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    if len(clock) != int((stop - start).total_seconds()):
        raise ValueError('Incomplete audit clock')
    quotes = pd.read_parquet(inputs['clean_quotes']['path'], columns=['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume'],
        filters=[('timestamp', '>=', start - pd.Timedelta(days=1)), ('timestamp', '<', stop)])
    support = pd.read_parquet(inputs['support_intervals']['path'])
    factors = pd.read_parquet(inputs['calibration_parameters']['path'],
        filters=[('effective_date', '>=', start), ('effective_date', '<', stop)])
    if not (factors.training_end_exclusive == factors.effective_date).all():
        raise ValueError('Seasonal calibration uses current/future data')
    summary = {'scope': 'original_detector_development_audit_not_paper_estimates',
               'training_start': start.isoformat(), 'parameter_cutoff': stats['parameter_cutoff'],
               'development_end_exclusive': stop.isoformat(), 'policies': {},
               'production_events_approved': False, 'primary_freshness_policy': None}
    for policy in POLICIES:
        summary['policies'][policy] = build_policy(run, policy, inputs, clock, quotes, support, factors, stats)
    atomic_json(run / 'kernel_build.json', get_kernel()[1])
    atomic_json(run / 'statistics.json', summary)
    m['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*'))
                    if p.is_file() and 'code' not in p.relative_to(run).parts and p.name != 'manifest.json'}
    m['status'] = 'built_pending_independent_verification'
    atomic_json(run / 'manifest.json', m)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id'); parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--verify-only')
    args = parser.parse_args()
    if args.verify_only:
        from src.verify_event_audit import verify_event_audit
        run = ROOT / 'data/signals' / args.verify_only
        result = verify_event_audit(run)
        atomic_json(run / 'verification.json', result)
        print(json.dumps(result), flush=True)
        return
    evidence = tests()
    if args.test_only:
        return
    if not args.run_id:
        parser.error('--run-id required')
    run = build(args.run_id, evidence)
    from src.verify_event_audit import verify_event_audit
    verification = verify_event_audit(run)
    atomic_json(run / 'verification.json', verification)
    m = read(run / 'manifest.json')
    m['outputs']['verification.json'] = sha256(run / 'verification.json')
    m['status'] = STATUS
    atomic_json(run / 'manifest.json', m)
    print('Verified audit: ' + str(run), flush=True)


if __name__ == '__main__':
    main()

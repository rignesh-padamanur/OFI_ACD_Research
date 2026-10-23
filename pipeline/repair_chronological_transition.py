"""Stage 07b: one rolling detector/episode transition, without response fitting."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import io
import json
import platform
import shutil
import unittest
import numpy as np
import pandas as pd
import pyarrow
import scipy
from src.data_preparation import atomic_json, sha256
from src.duration_calibration import (DurationAuditSpec, source_spells, sufficient_statistics,
    fit_scales, apply_scales, model_sample)
from src.chronological_acd import fit_acd, predict, diagnostics, get_kernel
from src.event_audit import (FEATURES, METRICS, completed_duration_bins, fit_detector,
    score_detector, quote_diagnostics, classify_bin_eligibility)
from src.event_episodes import build_episode_tables, attach_history_quality, policy_statistics
from src.chronological_transition import cycle_schedule, carry_forecasts, join_deployments
from repair_event_audit import add_adjusted, REASONS

ROOT = Path(__file__).resolve().parent
STAGE = '07b_first_chronological_transition'
STATUS = 'verified_first_chronological_transition'
ORIGIN = pd.Timestamp('2024-01-01', tz='UTC')
PREVIOUS = pd.Timestamp('2024-04-22', tz='UTC')
CUTOFF = pd.Timestamp('2024-05-06', tz='UTC')
STOP = pd.Timestamp('2024-05-20', tz='UTC')
START = CUTOFF-pd.Timedelta(days=56)
POLICY = 'structural_only'


def read(path): return json.loads(Path(path).read_text())


def resolve_inputs():
    pointers = {p.name: read(p) for p in (ROOT/'data').glob('active_*.json')}
    names = {'event': pointers['active_event_detector_audit.json']['audit_run_id'],
             'duration': pointers['active_duration_observation.json']['duration_run_id'],
             'acd': pointers['active_duration_model_audit.json']['audit_run_id'],
             'episode': pointers['active_episode_audit.json']['episode_run_id'],
             'regression': pointers['active_regression_input_audit.json']['input_run_id']}
    directories = {k: ROOT/'data/signals'/v for k, v in names.items()}
    manifests = {k: read(d/'manifest.json') for k, d in directories.items()}
    for k, d in directories.items():
        if read(d/'verification.json')['status'] != 'passed' or not manifests[k]['status'].startswith('verified_'):
            raise ValueError('Unverified parent '+k)
    em = manifests['event']
    if Path(em['inputs']['acd_statistics']['path']).parent != directories['acd']:
        raise ValueError('ACD parent mismatch')
    if manifests['episode']['parent_event_audit'] != directories['event'].name:
        raise ValueError('Episode parent mismatch')
    for field, key in [('parent_data_run', 'clean_quotes'), ('parent_eligibility_run', 'support_intervals')]:
        expected = Path(em['inputs'][key]['path']).parent.name
        if manifests['duration'][field] != expected or manifests['acd'][field] != expected:
            raise ValueError('Mixed duration/data/support lineage')
    if manifests['acd']['parent_duration_run'] != directories['duration'].name:
        raise ValueError('Duration observation/model parent mismatch')
    if manifests['regression']['parent_reference_detector'] != directories['event'].name:
        raise ValueError('Regression scaling/detector reference mismatch')
    paths = {k: dict(em['inputs'][k]) for k in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals']}
    paths.update(previous_predictions=dict(em['inputs']['saved_predictions_structural_only']),
                 previous_seasonal_spells=dict(em['inputs']['seasonal_spells_structural_only']))
    def add(key, parent, name):
        d, m = directories[parent], manifests[parent]
        h = m['outputs'][name]
        paths[key] = {'path': str(d/name), 'sha256': h['sha256'] if isinstance(h, dict) else h}
    for name in ['quote_durations', 'terminal_spells']: add(name, 'duration', name+'.parquet')
    for key, name in [('previous_opportunities', 'opportunities.parquet'), ('previous_scored', 'scored_bins.parquet')]:
        add(key, 'event', POLICY+'/'+name)
    for name in ['onsets', 'episodes']: add('previous_'+name, 'episode', POLICY+'/'+name+'.parquet')
    add('frozen_regression_scaling', 'regression', 'scaling.json')
    for k, d in directories.items():
        for name in ['manifest.json', 'verification.json']:
            p = d/name; paths[k+'_'+name.replace('.', '_')] = {'path': str(p), 'sha256': sha256(p)}
    for name, record in paths.items():
        if sha256(Path(record['path'])) != record['sha256']: raise ValueError('Changed parent '+name)
    return paths, names


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage07b_tests.json', evidence); print(stream.getvalue(), flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Tests failed; no build')
    return evidence


def build_features(clock, factors, duration, quotes, support):
    f = add_adjusted(clock, factors, POLICY)
    d = duration.set_index('bin_start').reindex(f.bin_start)
    n = d.completed_residual_count.fillna(0).to_numpy(dtype='int64')
    recorded = clock.recorded_update_count.to_numpy()
    ready = np.isfinite(f[['adjusted_abs_quote_ofi', 'adjusted_tick_count']]).all(axis=1).to_numpy()
    codes = classify_bin_eligibility(f.flow_status.to_numpy(), ready, recorded, n)
    opportunities = pd.DataFrame({'bin_start': f.bin_start, 'available_at': f.available_at,
        'period': np.where(f.bin_start < CUTOFF, 'calibration', 'development'),
        'recorded_update_count': recorded, 'support_id': f.support_id, 'flow_status': f.flow_status,
        'seasonal_ready': ready, 'completed_residual_count': n,
        'eligible_without_duration': np.isin(f.flow_status, [1, 2]) & ready,
        'reason': np.asarray(REASONS)[codes]})
    keep = codes == 5
    selected = f.loc[keep, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                            'adjusted_abs_quote_ofi', 'adjusted_tick_count']].reset_index(drop=True)
    selected['recorded_update_count'] = recorded[keep]
    selected['completed_residual_count'] = n[keep]
    selected['mean_duration_residual'] = d.mean_duration_residual.to_numpy()[keep]
    selected['negative_log_mean_residual'] = -np.log(selected.mean_duration_residual)
    selected['period'] = opportunities.period.to_numpy()[keep]
    economic = quote_diagnostics(f, quotes, support, POLICY)
    for name in METRICS: selected[name] = economic[name].to_numpy()[keep]
    fit = fit_detector(selected.loc[selected.period.eq('calibration'), FEATURES].to_numpy())
    selected['distance'] = score_detector(selected[FEATURES].to_numpy(), fit)
    selected['candidate_event'] = selected.distance >= fit['threshold']
    z = (selected[FEATURES].to_numpy()-np.asarray(fit['mean']))/np.asarray(fit['sample_std'])
    for i, name in enumerate(['z_ofi', 'z_intensity', 'z_duration']): selected[name] = z[:, i]
    return opportunities, selected, fit


def partition_statistics(opportunities, scored):
    return {'calendar_seconds': len(opportunities), 'eligible_bins': len(scored),
        'candidate_bins': int(scored.candidate_event.sum()),
        'candidate_share_of_eligible': float(scored.candidate_event.mean()) if len(scored) else None,
        'reason_counts': {r: int(opportunities.reason.eq(r).sum()) for r in REASONS}}


def build(run_id, evidence):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id): raise ValueError('Invalid run ID')
    inputs, parents = resolve_inputs()
    run = ROOT/'data/signals'/run_id; run.mkdir(exist_ok=False)
    spec = DurationAuditSpec(validation_days=28)
    snapshot = ROOT/'reports/repair_history/2026-09-30_pre_stage07b_source'
    m = {'run_id': run_id, 'stage': STAGE, 'status': 'building', 'inputs': inputs, 'parents': parents,
        'calendar_origin': ORIGIN.isoformat(), 'seasonal_configuration': asdict(spec),
        'training_start': START.isoformat(), 'parameter_cutoff': CUTOFF.isoformat(),
        'previous_cutoff': PREVIOUS.isoformat(), 'deployment_end': STOP.isoformat(),
        'primary_policy': POLICY, 'scope': 'first_chronological_transition_only',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': {},
        'previous_files': read(snapshot/'preserved_files.json'), 'snapshot_path': str(snapshot),
        'production_inference_approved': False, 'downstream_ready': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__}}
    sources = [ROOT/'repair_chronological_transition.py', ROOT/'repair_event_audit.py',
        ROOT/'reports/STAGE07B_TRANSITION_PROTOCOL.md', *sorted((ROOT/'src').glob('*.py')), ROOT/'src/acd_kernel.c']
    for p in sources:
        rel = str(p.relative_to(ROOT)); dest = run/'code'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest); m['code_sha256'][rel] = sha256(dest)
    for p in sorted((ROOT/'tests').glob('test_*.py')):
        if sha256(p) != evidence['test_source_sha256'][p.name]: raise ValueError('Tests changed after execution')
        dest = run/'full_test_sources'/p.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(p, dest)
    atomic_json(run/'manifest.json', m); atomic_json(run/'test_evidence.json', evidence)
    atomic_json(run/'kernel_build.json', get_kernel()[1])
    schedule = cycle_schedule(PREVIOUS, STOP); schedule.to_parquet(run/'cycle_schedule.parquet', index=False)
    tails = pd.read_parquet(inputs['terminal_spells']['path'])
    raw = source_spells(inputs['quote_durations']['path'], tails, policy=POLICY, stop_ns=STOP.value)
    table, corrections = sufficient_statistics(raw, ORIGIN.value, spec.total_days)
    seasonal, scales, status = fit_scales(table, corrections, ORIGIN.value, spec)
    frame = apply_scales(raw, scales, status, ORIGIN.value)
    for name, value in [('origin_daily_sufficient_statistics', table), ('historical_cutoff_corrections', corrections),
                        ('seasonal_parameters', seasonal), ('seasonal_spells', frame)]:
        value.to_parquet(run/(name+'.parquet'), index=False, row_group_size=400000)
    source_rows = len(frame)
    train = model_sample(frame, start_ns=START.value, stop_ns=CUTOFF.value)
    path = model_sample(frame, start_ns=START.value, stop_ns=STOP.value)
    del raw, frame, table, corrections, scales, status; gc.collect()
    train.to_parquet(run/'training_spells.parquet', index=False, row_group_size=400000)
    path.to_parquet(run/'model_path.parquet', index=False, row_group_size=400000)
    print(f'Fitting the existing Weibull ACD: {len(train):,} training spells', flush=True)
    best, attempts = fit_acd(train, weibull=True)
    atomic_json(run/'optimization_attempts.json', attempts)
    if best is None: raise RuntimeError('All fixed optimization starts failed; no deployment')
    parameters = {k: best[k] for k in ['omega', 'alpha', 'beta', 'shape']}
    prediction = predict(path, list(parameters.values()), weibull=True)
    previous = pd.read_parquet(inputs['previous_predictions']['path'])
    forecasts = carry_forecasts(path, prediction, previous, cutoff=CUTOFF, previous_cutoff=PREVIOUS)
    carry = forecasts.forecast_role.eq('carried_origin_forecast')
    prior_spells = pd.read_parquet(inputs['previous_seasonal_spells']['path'],
        filters=[('origin_at', '>=', PREVIOUS)], columns=['origin_quote_id', 'origin_at', 'scale_sec', 'seasonal_cutoff'])
    carried_path = path.loc[carry].reset_index(drop=True)
    old_scale = prior_spells.set_index('origin_quote_id').reindex(carried_path.origin_quote_id)
    for name in ['origin_at', 'scale_sec', 'seasonal_cutoff']:
        np.testing.assert_array_equal(carried_path[name], old_scale[name])
    forecasts.to_parquet(run/'forecasts.parquet', index=False, row_group_size=400000)
    bridge = carried_path.copy()
    for name in forecasts.columns[1:]: bridge[name] = forecasts.loc[carry, name].to_numpy()
    bridge.to_parquet(run/'boundary_spells.parquet', index=False)
    duration = completed_duration_bins(path, forecasts[['origin_quote_id', 'psi_for_residual']].rename(columns={'psi_for_residual': 'psi'}))
    duration.to_parquet(run/'duration_bins.parquet', index=False)
    deployed = path.origin_at.ge(CUTOFF)
    dd = diagnostics(path.loc[deployed].reset_index(drop=True), prediction.loc[deployed].reset_index(drop=True))
    training_rows, training_events = len(train), int(train.event_observed.sum())
    forecast_rows, carried_rows = len(path), int(carry.sum())
    carried_completed = int((carry & path.event_observed).sum())
    del path, train, prediction, previous, prior_spells, forecasts; gc.collect()
    print(f'Forecasts saved; {carried_rows} pending origin forecasts carried', flush=True)
    clock = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', START), ('bin_start', '<', STOP)])
    np.testing.assert_array_equal(pd.DatetimeIndex(clock.bin_start).asi8, pd.date_range(START, STOP, freq='s', inclusive='left').asi8)
    factors = pd.read_parquet(inputs['calibration_parameters']['path'],
        filters=[('effective_date', '>=', START), ('effective_date', '<', STOP), ('policy', '==', POLICY)])
    if not factors.training_end_exclusive.eq(factors.effective_date).all(): raise ValueError('Future seasonal factor')
    quotes = pd.read_parquet(inputs['clean_quotes']['path'],
        columns=['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume'],
        filters=[('timestamp', '>=', START-pd.Timedelta(days=1)), ('timestamp', '<', STOP)])
    support = pd.read_parquet(inputs['support_intervals']['path']).loc[lambda f: f.policy.eq(POLICY)].reset_index(drop=True)
    opportunity, score, fit = build_features(clock, factors, duration, quotes, support)
    parts = {}
    for period, prefix in [('calibration', 'calibration'), ('development', 'deployment')]:
        o, s = opportunity.loc[opportunity.period.eq(period)].reset_index(drop=True), score.loc[score.period.eq(period)].reset_index(drop=True)
        o.to_parquet(run/(prefix+'_opportunities.parquet'), index=False)
        s.to_parquet(run/(prefix+'_scored.parquet'), index=False)
        parts[prefix] = partition_statistics(o, s)
        if period == 'development': new_o, new_s = o, s
    old_o = pd.read_parquet(inputs['previous_opportunities']['path'], filters=[('period', '==', 'development')])
    old_s = pd.read_parquet(inputs['previous_scored']['path'], filters=[('period', '==', 'development')], columns=list(new_s.columns))
    combined_o, combined_s = join_deployments(old_o, old_s, new_o, new_s, cutoff=CUTOFF)
    combined_o.to_parquet(run/'combined_opportunities.parquet', index=False)
    combined_s.to_parquet(run/'combined_scored.parquet', index=False)
    del clock, factors, duration, opportunity, score; gc.collect()
    tables = build_episode_tables(combined_o, combined_s)
    tables['onsets'] = attach_history_quality(tables['onsets'], quotes, support)
    for key, file in [('clock', 'episode_clock'), ('membership', 'candidate_membership'), ('episodes', 'episodes'), ('onsets', 'onsets')]:
        tables[key].to_parquet(run/(file+'.parquet'), index=False)
    old_onsets = pd.read_parquet(inputs['previous_onsets']['path'])
    prefix = tables['onsets'].loc[tables['onsets'].bin_start.lt(CUTOFF)].reset_index(drop=True)
    common = list(prefix.columns)
    pd.testing.assert_frame_equal(prefix[common], old_onsets[common], check_dtype=False,
                                  check_categorical=False, check_exact=True)
    episode_stats = policy_statistics(tables)
    daily = pd.DataFrame({'date': pd.date_range(PREVIOUS, STOP, freq='D', inclusive='left')})
    for col, frame, mask in [('candidate_bins', combined_s, combined_s.candidate_event),
            ('episode_fragments', tables['onsets'], np.ones(len(tables['onsets']), dtype=bool)),
            ('confirmed_onsets', tables['onsets'], tables['onsets'].onset_confirmed),
            ('state_sample_eligible', tables['onsets'], tables['onsets'].state_sample_eligible)]:
        count = frame.loc[mask].groupby(frame.loc[mask, 'bin_start'].dt.normalize()).size()
        daily[col] = daily.date.map(count).fillna(0).astype('int64')
    daily.to_parquet(run/'daily.parquet', index=False)
    new_onsets = tables['onsets'].loc[tables['onsets'].bin_start.ge(CUTOFF)]
    crossing = tables['episodes'].first_candidate_bin.lt(CUTOFF) & tables['episodes'].last_candidate_bin.ge(CUTOFF)
    statistics = {'scope': 'first_chronological_transition_only', 'primary_policy': POLICY,
        'source_spell_rows': source_rows, 'seasonal_parameter_rows': len(seasonal),
        'training_spells': training_rows, 'training_events': training_events, 'forecast_rows': forecast_rows,
        'carried_forecasts': carried_rows, 'carried_completed_residuals': carried_completed,
        'calibration': parts['calibration'], 'deployment': parts['deployment'],
        'duration_working_model_diagnostics': dd, 'combined_episodes': episode_stats,
        'previous_onsets_preserved': len(prefix), 'new_onset_fragments': len(new_onsets),
        'new_confirmed_onsets': int(new_onsets.onset_confirmed.sum()),
        'new_state_eligible_onsets': int(new_onsets.state_sample_eligible.sum()),
        'episodes_with_candidates_across_cutoff': int(crossing.sum()),
        'fitted_duration_specifications': 1, 'empirical_response_models_fitted': 0,
        'full_panel_extension_performed': False, 'extended_responses_computed': False,
        'production_inference_approved': False, 'downstream_ready': False}
    atomic_json(run/'parameters.json', {'training_start': START.isoformat(), 'parameter_cutoff': CUTOFF.isoformat(),
        'deployment_end': STOP.isoformat(), 'acd_parameters': parameters, 'optimization_attempts': attempts,
        'acd_fit_objective': best['objective'], 'numerical_constraint_warning': best['near_constraint'], 'detector': fit})
    atomic_json(run/'statistics.json', statistics)
    m['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*'))
                    if p.is_file() and p.relative_to(run).parts[0] not in ['code', 'full_test_sources'] and p.name != 'manifest.json'}
    m['status'] = 'built_pending_independent_verification'; atomic_json(run/'manifest.json', m)
    print(json.dumps({'run': run.name, 'threshold': fit['threshold'], 'new_candidates': parts['deployment']['candidate_bins'],
                      'new_onsets': len(new_onsets), 'old_onsets_preserved': len(prefix)}), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id'); parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--verify-only'); args = parser.parse_args()
    if args.verify_only:
        from src.verify_chronological_transition import verify_transition
        print(json.dumps(verify_transition(ROOT/'data/signals'/args.verify_only), indent=2)); return
    evidence = run_tests()
    if args.test_only: return
    if not args.run_id: parser.error('--run-id required')
    run = build(args.run_id, evidence)
    from src.verify_chronological_transition import verify_transition
    result = verify_transition(run); atomic_json(run/'verification.json', result)
    m = read(run/'manifest.json'); m['status'] = STATUS
    m['outputs']['verification.json'] = sha256(run/'verification.json'); atomic_json(run/'manifest.json', m)
    print('Verified: '+str(run), flush=True)


if __name__ == '__main__': main()

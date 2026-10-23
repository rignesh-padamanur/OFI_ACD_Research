"""Stage 07c: full cached structural baseline, with continuous issued state."""
from dataclasses import asdict
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
import scipy
from src.data_preparation import atomic_json, sha256
from src.duration_calibration import DurationAuditSpec, model_sample
from src.chronological_acd import fit_acd, predict, diagnostics, get_kernel
from src.chronological_transition import cycle_schedule
from src.chronological_panel import (ISSUED_COLUMNS, issued_from_predictions,
    deployment_durations, score_cycle, partition_statistics, integer_fingerprint)
from src.panel_observations import build_seasonal_panel, read_quote_window
from src.streaming_episodes import EpisodeStream
from src.event_audit import completed_duration_bins
from src.event_episodes import attach_history_quality, policy_statistics
from repair_event_audit import REASONS
from repair_chronological_transition import resolve_inputs as resolve_ancestral_inputs

ROOT = Path(__file__).resolve().parent
ORIGIN = pd.Timestamp('2024-01-01', tz='UTC')
FIRST = pd.Timestamp('2024-04-22', tz='UTC')
CONTINUE = pd.Timestamp('2024-05-20', tz='UTC')
STOP = pd.Timestamp('2026-08-01', tz='UTC')
POLICY = 'structural_only'
STAGE = '07c_full_chronological_baseline_panel'
STATUS = 'verified_full_chronological_baseline_panel'
QUOTE_COLUMNS = ['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume']


def read(path): return json.loads(Path(path).read_text())


def resolve_inputs():
    inputs, parents = resolve_ancestral_inputs()
    parent = ROOT/'data/signals'/read(ROOT/'data/active_chronological_transition_audit.json')['transition_run_id']
    manifest = read(parent/'manifest.json')
    if (manifest['status'] != 'verified_first_chronological_transition'
            or read(parent/'verification.json')['status'] != 'passed'
            or pd.Timestamp(manifest['deployment_end']) != CONTINUE):
        raise ValueError('Verified two-cycle prefix required')
    for k in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals', 'quote_durations', 'terminal_spells']:
        if manifest['inputs'][k] != inputs[k]: raise ValueError('Prefix ancestor mismatch: '+k)
    for name in ['manifest.json', 'verification.json', 'statistics.json', 'parameters.json',
                 'combined_opportunities.parquet', 'combined_scored.parquet', 'onsets.parquet',
                 'episodes.parquet', 'model_path.parquet', 'forecasts.parquet']:
        p = parent/name
        h = sha256(p)
        if name != 'manifest.json' and manifest['outputs'][name] != h:
            raise ValueError('Changed verified prefix: '+name)
        inputs['prefix_'+name.replace('.', '_')] = {'path': str(p), 'sha256': h}
    parents['transition'] = parent.name
    return inputs, parents, parent


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(),
        'test_source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    atomic_json(ROOT/'reports/tests/stage07c_tests.json', evidence)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Tests failed; no build')
    return evidence


def save_frame(directory, name, frame):
    frame.to_parquet(directory/(name+'.parquet'), index=False, row_group_size=400000)


def progress_rows(rows):
    return [{k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in row.items()} for row in rows]


def seed_carry(parent, observations):
    path = pd.read_parquet(parent/'model_path.parquet', filters=[('end_at', '>=', CONTINUE)])
    forecasts = pd.read_parquet(parent/'forecasts.parquet').set_index('origin_quote_id').reindex(path.origin_quote_id)
    if not forecasts.forecast_role.isin(['carried_origin_forecast', 'new_origin_forecast']).all():
        raise ValueError('Pending prefix wait lacks issued deployment forecast')
    carry = path[ISSUED_COLUMNS[:4]].reset_index(drop=True)
    carry['psi'] = forecasts.psi_for_residual.to_numpy()
    carry['residual_parameter_cutoff'] = forecasts.residual_parameter_cutoff.to_numpy()
    opened = open_spells(observations, CONTINUE)
    pd.testing.assert_frame_equal(carry[ISSUED_COLUMNS[:4]], opened[ISSUED_COLUMNS[:4]],
                                  check_exact=True, check_dtype=False)
    return carry[ISSUED_COLUMNS]


def open_spells(observations, cutoff):
    f = pd.read_parquet(observations/'seasonal_spells.parquet',
        filters=[('origin_at', '<', cutoff), ('end_at', '>=', cutoff)])
    return f.loc[np.isfinite(f.scale_sec) & f.scale_sec.gt(0) & f.exposure_ns.gt(0)].reset_index(drop=True)


def build_cycle(directory, row, observations, inputs, carry, support):
    start, cutoff, stop = row.training_start, row.parameter_cutoff, row.deployment_end
    frame = pd.read_parquet(observations/'seasonal_spells.parquet',
        filters=[('origin_at', '>=', start), ('origin_at', '<', stop)])
    train = model_sample(frame, start_ns=start.value, stop_ns=cutoff.value)
    path = model_sample(frame, start_ns=start.value, stop_ns=stop.value)
    print(f'Cycle {row.cycle_id:03d}: fitting {len(train):,} training spells; cutoff {cutoff.date()}', flush=True)
    best, attempts = fit_acd(train, weibull=True)
    atomic_json(directory/'optimization_attempts.json', attempts)
    if best is None: raise RuntimeError('All fixed optimizer starts failed; no deployment')
    params = {k: best[k] for k in ['omega', 'alpha', 'beta', 'shape']}
    prediction = predict(path, list(params.values()), weibull=True)
    issued = issued_from_predictions(path, prediction, cutoff=cutoff)
    raw_new = frame.loc[frame.origin_at.ge(cutoff) & np.isfinite(frame.scale_sec) &
                        frame.scale_sec.gt(0) & frame.exposure_ns.gt(0)].reset_index(drop=True)
    bins, carry_out, carry_counts = deployment_durations(issued, raw_new, carry,
        open_spells(observations, cutoff), cutoff=cutoff, stop=stop)
    calibration_bins = completed_duration_bins(path, prediction)
    duration = pd.concat([calibration_bins.loc[calibration_bins.bin_start.lt(cutoff)], bins], ignore_index=True)
    if duration.bin_start.duplicated().any(): raise ValueError('Duplicate residual second')
    deployed = path.origin_at.ge(cutoff)
    stats = {'training_spells': len(train), 'training_events': int(train.event_observed.sum()),
        'training_identity_sha256': integer_fingerprint(train.origin_quote_id, train.exposure_ns, train.event_observed),
        'duration_working_model_diagnostics': diagnostics(path.loc[deployed].reset_index(drop=True),
                                                        prediction.loc[deployed].reset_index(drop=True)),
        'carry': carry_counts}
    for name, value in [('issued_forecasts', issued), ('carry_in', carry), ('carry_out', carry_out),
                        ('deployment_duration_bins', bins)]: save_frame(directory, name, value)
    del frame, train, path, prediction, raw_new, issued, calibration_bins, bins
    gc.collect()
    clock = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    np.testing.assert_array_equal(pd.DatetimeIndex(clock.bin_start).as_unit('ns').asi8,
                                  np.arange(start.value, stop.value, 1_000_000_000))
    factors = pd.read_parquet(inputs['calibration_parameters']['path'], filters=[
        ('effective_date', '>=', start), ('effective_date', '<', stop), ('policy', '==', POLICY)])
    if not factors.training_end_exclusive.eq(factors.effective_date).all(): raise ValueError('Future seasonal factor')
    quotes = read_quote_window(inputs['clean_quotes']['path'], cutoff-pd.Timedelta(seconds=300), stop, QUOTE_COLUMNS)
    o, s, detector, calibration = score_cycle(clock, factors, duration, quotes, support, cutoff=cutoff)
    stats.update(calibration=calibration, deployment=partition_statistics(o, s))
    atomic_json(directory/'parameters.json', {'training_start': start.isoformat(), 'parameter_cutoff': cutoff.isoformat(),
        'deployment_end': stop.isoformat(), 'acd_parameters': params, 'optimization_attempts': attempts,
        'acd_fit_objective': best['objective'], 'numerical_constraint_warning': best['near_constraint'], 'detector': detector})
    return o, s, carry_out, quotes, stats


def daily_statistics(o, s, onsets, clock, cutoff, stop):
    daily = pd.DataFrame({'date': pd.date_range(cutoff, stop, freq='D', inclusive='left')})
    for reason in REASONS:
        dates = o.loc[o.reason.eq(reason), 'bin_start'].dt.normalize().value_counts()
        daily['reason_'+reason] = daily.date.map(dates).fillna(0).astype('int64')
    for state in ['trigger', 'clear_active', 'clear_quiet', 'unknown']:
        dates = clock.loc[clock.observation_state.eq(state), 'bin_start'].dt.normalize().value_counts()
        daily['state_'+state] = daily.date.map(dates).fillna(0).astype('int64')
    for key, f, mask in [('candidate_bins', s, s.candidate_event),
        ('episode_fragments', onsets, np.ones(len(onsets), dtype=bool)),
        ('confirmed_onsets', onsets, onsets.onset_confirmed),
        ('state_sample_eligible', onsets, onsets.state_sample_eligible)]:
        dates = f.loc[mask, 'bin_start'].dt.normalize().value_counts()
        daily[key] = daily.date.map(dates).fillna(0).astype('int64')
    daily['calendar_seconds'] = daily.filter(like='reason_').sum(axis=1)
    if not daily.calendar_seconds.eq(86400).all(): raise ValueError('Lost calendar denominator')
    return daily


def build(run_id, evidence):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id): raise ValueError('Invalid run ID')
    inputs, parents, parent = resolve_inputs()
    run = ROOT/'data/signals'/run_id
    run.mkdir(exist_ok=False)
    snapshot = ROOT/'reports/repair_history/2026-09-30_pre_stage07c_source'
    spec = DurationAuditSpec(validation_days=int((STOP-ORIGIN).days)-112)
    m = {'run_id': run_id, 'stage': STAGE, 'status': 'building', 'inputs': inputs, 'parents': parents,
        'calendar_origin': ORIGIN.isoformat(), 'first_cutoff': FIRST.isoformat(),
        'continuation_cutoff': CONTINUE.isoformat(), 'deployment_end': STOP.isoformat(),
        'seasonal_configuration': asdict(spec), 'primary_policy': POLICY,
        'scope': 'full_cached_structural_baseline_detector_episode_states_only',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': {},
        'previous_files': read(snapshot/'preserved_files.json'), 'snapshot_path': str(snapshot),
        'production_inference_approved': False, 'downstream_ready': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                       'pandas': pd.__version__, 'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__}}
    sources = [Path(__file__), ROOT/'repair_chronological_transition.py', ROOT/'repair_event_audit.py',
        ROOT/'reports/STAGE07C_PANEL_PROTOCOL.md', *sorted((ROOT/'src').glob('*.py')), ROOT/'src/acd_kernel.c']
    for p in sources:
        rel = str(p.relative_to(ROOT)); dest = run/'code'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest); m['code_sha256'][rel] = sha256(dest)
    for p in sorted((ROOT/'tests').glob('test_*.py')):
        if evidence['test_source_sha256'][p.name] != sha256(p): raise ValueError('Test changed after execution')
        dest = run/'full_test_sources'/p.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(p, dest)
    atomic_json(run/'manifest.json', m); atomic_json(run/'test_evidence.json', evidence)
    atomic_json(run/'kernel_build.json', get_kernel()[1])
    schedule = cycle_schedule(FIRST, STOP); save_frame(run, 'cycle_schedule', schedule)
    active_cycle = None
    try:
        obs = run/'observations'; obs.mkdir()
        print('Streaming full cached source and historical seasonal statistics', flush=True)
        source_stats = build_seasonal_panel(inputs['quote_durations']['path'], inputs['terminal_spells']['path'],
            obs, origin=ORIGIN, stop=STOP, policy=POLICY)
        atomic_json(obs/'statistics.json', source_stats)
        support = pd.read_parquet(inputs['support_intervals']['path']).loc[lambda f: f.policy.eq(POLICY)].reset_index(drop=True)
        stream = EpisodeStream(); carry = seed_carry(parent, obs)
        onset_parts, episode_parts, daily_parts, cycle_rows = [], [], [], []
        (run/'cycles').mkdir()
        for row in schedule.itertuples(index=False):
            active_cycle = int(row.cycle_id); directory = run/'cycles'/f'{active_cycle:03d}'; directory.mkdir()
            inherited = row.parameter_cutoff < CONTINUE
            cm = {'cycle_id': active_cycle, 'training_start': row.training_start.isoformat(),
                'parameter_cutoff': row.parameter_cutoff.isoformat(), 'deployment_end': row.deployment_end.isoformat(),
                'inherited': inherited, 'status': 'building'}
            atomic_json(directory/'manifest.json', cm)
            if inherited:
                filters = [('bin_start', '>=', row.parameter_cutoff), ('bin_start', '<', row.deployment_end)]
                o = pd.read_parquet(parent/'combined_opportunities.parquet', filters=filters)
                s = pd.read_parquet(parent/'combined_scored.parquet', filters=filters)
                quotes = read_quote_window(inputs['clean_quotes']['path'], row.parameter_cutoff-pd.Timedelta(seconds=300),
                                           row.deployment_end, QUOTE_COLUMNS)
                stats = {'inherited': True, 'deployment': partition_statistics(o, s)}
                if active_cycle == 1: save_frame(directory, 'carry_out', carry)
            else:
                o, s, carry, quotes, stats = build_cycle(directory, row, obs, inputs, carry, support)
            tables = stream.consume(o, s)
            tables['onsets'] = attach_history_quality(tables['onsets'], quotes, support)
            for key, name in [('clock', 'episode_clock'), ('membership', 'candidate_membership'),
                              ('onsets', 'onsets'), ('closed_episodes', 'closed_episodes')]:
                save_frame(directory, name, tables[key])
            save_frame(directory, 'opportunities', o); save_frame(directory, 'scored', s)
            atomic_json(directory/'episode_state.json', stream.state_dict())
            daily = daily_statistics(o, s, tables['onsets'], tables['clock'], row.parameter_cutoff, row.deployment_end)
            save_frame(directory, 'daily', daily)
            stats.update(onset_fragments=len(tables['onsets']), confirmed_onsets=int(tables['onsets'].onset_confirmed.sum()),
                state_eligible_onsets=int(tables['onsets'].state_sample_eligible.sum()),
                finalized_episodes=len(tables['closed_episodes']))
            atomic_json(directory/'statistics.json', stats)
            cm['status'] = 'built_pending_independent_verification'
            cm['outputs'] = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'manifest.json'}
            atomic_json(directory/'manifest.json', cm)
            onset_parts.append(tables['onsets']); episode_parts.append(tables['closed_episodes']); daily_parts.append(daily)
            cycle_rows.append({'cycle_id': active_cycle, 'parameter_cutoff': row.parameter_cutoff,
                'deployment_end': row.deployment_end, 'inherited': inherited,
                **{k: v for k, v in stats['deployment'].items() if k != 'reason_counts'},
                'onset_fragments': stats['onset_fragments'], 'confirmed_onsets': stats['confirmed_onsets'],
                'state_eligible_onsets': stats['state_eligible_onsets'],
                'duration_diagnostic_flags': '|'.join(stats.get('duration_working_model_diagnostics', {}).get('diagnostic_flags', []))})
            atomic_json(run/'progress.json', {'status': 'building', 'completed_cycles': len(cycle_rows),
                'planned_cycles': len(schedule), 'covered_through_exclusive': row.deployment_end.isoformat(),
                'candidate_bins': sum(x['candidate_bins'] for x in cycle_rows),
                'onset_fragments': sum(x['onset_fragments'] for x in cycle_rows), 'cycle_statistics': progress_rows(cycle_rows)})
            print(f'Cycle {active_cycle:03d} saved: {stats["deployment"]["candidate_bins"]:,} candidates, '
                  f'{stats["onset_fragments"]:,} onsets', flush=True)
            del tables, quotes, o, s, daily; gc.collect()
        final = stream.finish(); save_frame(run, 'final_censored_episodes', final); episode_parts.append(final)
        atomic_json(run/'final_episode_state.json', stream.state_dict())
        onsets = pd.concat(onset_parts, ignore_index=True).sort_values('episode_id').reset_index(drop=True)
        episodes = pd.concat(episode_parts, ignore_index=True).sort_values('episode_id').reset_index(drop=True)
        np.testing.assert_array_equal(onsets.episode_id, episodes.episode_id)
        np.testing.assert_array_equal(episodes.episode_id, np.arange(len(episodes)))
        old_onsets = pd.read_parquet(parent/'onsets.parquet')
        prefix = onsets.loc[onsets.bin_start.lt(CONTINUE)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix[list(old_onsets.columns)], old_onsets,
            check_exact=True, check_dtype=False, check_categorical=False)
        old_episodes = pd.read_parquet(parent/'episodes.parquet')
        closed = old_episodes.loc[~old_episodes.termination.eq('censored_sample_end')]
        pd.testing.assert_frame_equal(episodes.set_index('episode_id').loc[closed.episode_id].reset_index()[list(closed.columns)],
            closed.reset_index(drop=True), check_exact=True, check_dtype=False, check_categorical=False)
        daily = pd.concat(daily_parts, ignore_index=True)
        monthly = daily.drop(columns='date').groupby(daily.date.dt.strftime('%Y-%m')).sum().rename_axis('month').reset_index()
        for name, frame in [('onsets', onsets), ('episodes', episodes), ('daily', daily),
                            ('monthly', monthly), ('cycle_statistics', pd.DataFrame(cycle_rows))]: save_frame(run, name, frame)
        epstats = policy_statistics({'clock': pd.DataFrame({'observation_state': pd.Series(dtype='object')}),
                                    'onsets': onsets, 'episodes': episodes})
        epstats['clock_seconds'] = int(daily.calendar_seconds.sum())
        epstats['observation_states'] = {k: int(daily['state_'+k].sum()) for k in ['trigger', 'clear_active', 'clear_quiet', 'unknown']}
        reason_counts = {k: int(daily['reason_'+k].sum()) for k in REASONS}
        candidates = int(daily.candidate_bins.sum())
        if candidates != epstats['candidate_bins']: raise ValueError('Candidate membership not conserved')
        new_stats = [read(run/'cycles'/f'{i:03d}'/'statistics.json') for i in range(2, len(schedule))]
        new_params = [read(run/'cycles'/f'{i:03d}'/'parameters.json') for i in range(2, len(schedule))]
        diagnostic_names = sorted({flag for x in new_stats for flag in x['duration_working_model_diagnostics']['diagnostic_flags']})
        crossing = np.zeros(len(episodes), dtype=bool)
        for cutoff in schedule.parameter_cutoff.iloc[1:]:
            crossing |= (episodes.first_candidate_bin.lt(cutoff) & episodes.last_candidate_bin.ge(cutoff)).to_numpy()
        statistics = {'scope': m['scope'], 'primary_policy': POLICY, 'source': source_stats,
            'cycles': len(schedule), 'inherited_cycles': 2, 'new_fitted_cycles': len(schedule)-2,
            'final_cycle_days': int((STOP-schedule.parameter_cutoff.iloc[-1]).days),
            'deployment': {'calendar_seconds': epstats['clock_seconds'], 'eligible_bins': reason_counts['eligible'],
                'candidate_bins': candidates, 'candidate_share_of_eligible': candidates/reason_counts['eligible'],
                'reason_counts': reason_counts}, 'combined_episodes': epstats,
            'previous_onsets_preserved': len(prefix), 'previous_closed_episodes_preserved': len(closed),
            'new_onset_fragments': len(onsets)-len(prefix),
            'episodes_with_candidates_across_cutoffs': int(crossing.sum()),
            'new_cycle_diagnostics': {
                'constraint_warning_cycles': sum(x['numerical_constraint_warning'] for x in new_params),
                'threshold_min': min(x['detector']['threshold'] for x in new_params),
                'threshold_max': max(x['detector']['threshold'] for x in new_params),
                'diagnostic_flag_cycles': {flag: sum(flag in x['duration_working_model_diagnostics']['diagnostic_flags'] for x in new_stats) for flag in diagnostic_names},
                'carried_forecast_transitions': sum(x['carry']['carried_in'] for x in new_stats),
                'completed_carried_residuals': sum(x['carry']['completed_carried_residuals'] for x in new_stats),
                'carry_origins_before_training_start': sum(x['carry']['carried_origin_before_training_start'] for x in new_stats)},
            'fitted_duration_specifications': 1, 'empirical_response_models_fitted': 0,
            'full_panel_extension_performed': True, 'fixed_sensitivities_extended': False,
            'extended_responses_computed': False, 'production_inference_approved': False, 'downstream_ready': False}
        atomic_json(run/'statistics.json', statistics)
        atomic_json(run/'progress.json', {'status': 'built_pending_independent_verification', 'completed_cycles': len(schedule),
            'planned_cycles': len(schedule), 'covered_through_exclusive': STOP.isoformat(), 'cycle_statistics': progress_rows(cycle_rows)})
        m['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob('*')) if p.is_file()
            and p.relative_to(run).parts[0] not in ['code', 'full_test_sources'] and p != run/'manifest.json'}
        m['status'] = 'built_pending_independent_verification'; atomic_json(run/'manifest.json', m)
        return run
    except Exception as error:
        failure = {'status': 'failed_unregistered', 'cycle_id': active_cycle, 'error': str(error), 'traceback': traceback.format_exc()}
        atomic_json(run/'failure.json', failure)
        m['status'] = 'failed_unregistered'; atomic_json(run/'manifest.json', m)
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id'); parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--verify-only'); args = parser.parse_args()
    if args.verify_only:
        run = ROOT/'data/signals'/args.verify_only
    else:
        evidence = run_tests()
        if args.test_only: return
        if not args.run_id: parser.error('--run-id required')
        run = build(args.run_id, evidence)
    from src.verify_chronological_panel_v2 import verify_panel
    result = verify_panel(run)
    if result['status'] != 'passed': raise RuntimeError('Independent verification did not pass')
    if args.verify_only:
        print(json.dumps(result, indent=2)); return
    atomic_json(run/'verification.json', result)
    m = read(run/'manifest.json'); m['status'] = STATUS
    m['outputs']['verification.json'] = sha256(run/'verification.json'); atomic_json(run/'manifest.json', m)
    print('Verified: '+str(run), flush=True)


if __name__ == '__main__': main()

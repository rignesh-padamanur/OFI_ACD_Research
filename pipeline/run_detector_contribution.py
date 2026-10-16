"""Reproducible, resumable execution of the fixed Step 11 research extension."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data_preparation import atomic_json, sha256
from src.event_audit import FEATURES, fit_detector, completed_duration_bins, ns
from src.chronological_acd import predict
from src.duration_calibration import model_sample
from src.chronological_panel import feature_inputs, integer_fingerprint
from src.panel_observations import read_quote_window
from src.detector_contribution import (RULES, PRIMARY, XCOLS, SECOND, QuoteGrid,
    core_scores, calibrate, hash_training_sample, causal_onsets, contrast_table, match_times)

ROOT = Path(__file__).resolve().parent
PANEL = ROOT/'data/signals/step07c_chronological_panel_20261001_v2'
PREFIX = ROOT/'data/signals/step07b_chronological_transition_20260930_v2'
DEV = ROOT/'data/signals/step05a_original_detector_audit_20260928/structural_only'
PROTOCOL = ROOT/'reports/STEP11_DETECTOR_CONTRIBUTION_PROTOCOL.md'
START = pd.Timestamp('2024-05-06', tz='UTC')
END = pd.Timestamp('2026-08-01', tz='UTC')
MODEL = dict(loss='squared_error', learning_rate=.05, max_iter=100,
    max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1,
    max_bins=63, early_stopping=False, random_state=20261004)
CLOCK_COLS = ['bin_start', 'available_at', 'recorded_update_count', 'observed_quote_ofi',
    'n_invalid_transitions', 'support_id_structural_only', 'flow_status_structural_only']
SOURCES = [Path(__file__), ROOT/'src/detector_contribution.py', PROTOCOL,
    ROOT/'tests/test_detector_contribution.py', ROOT/'src/event_audit.py',
    ROOT/'src/chronological_panel.py', ROOT/'src/chronological_acd.py',
    ROOT/'src/acd_kernel.c', ROOT/'src/duration_calibration.py',
    ROOT/'src/panel_observations.py', ROOT/'src/panel_state_precision.py',
    ROOT/'src/intensity.py', ROOT/'repair_event_audit.py']


def read(p):
    return json.loads(Path(p).read_text())


def save(p, frame):
    frame.to_parquet(p, index=False, row_group_size=400000)


def init_run(run):
    if (run/'manifest.json').exists():
        m = read(run/'manifest.json')
        for name, digest in m['source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError('Source changed after freeze: '+name)
        return m
    run.mkdir(parents=True, exist_ok=False)
    source = {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES}
    for p in SOURCES:
        dest = run/'code'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    parent = read(PANEL/'manifest.json')
    inputs = {k: parent['inputs'][k] for k in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals']}
    for key, item in inputs.items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Parent input changed: '+key)
    m = {'run_id': run.name, 'status': 'detector_building', 'created_at': datetime.now(timezone.utc).isoformat(),
         'scope': 'exploratory_fixed_detector_economic_comparison', 'source_sha256': source,
         'inputs': inputs, 'parent_panel_manifest_sha256': sha256(PANEL/'manifest.json'),
         'parent_panel': str(PANEL), 'start': START.isoformat(), 'end_exclusive': END.isoformat(),
         'model': MODEL, 'features': XCOLS, 'rules': RULES,
         'independent_validation': False, 'pipeline_uncertainty_validated': False,
         'economic_materiality_margin': None, 'python': platform.python_version()}
    atomic_json(run/'manifest.json', m)
    (run/'cycles').mkdir()
    return m


def read_clock(inputs, start, stop):
    return pd.read_parquet(inputs['clock']['path'], columns=CLOCK_COLS,
        filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])


def calibration_frame(row, inputs):
    cycle = int(row.cycle_id)
    if cycle == 0:
        params = read(DEV/'parameters.json')
        frame = pd.read_parquet(DEV/'scored_bins.parquet', filters=[('period', '==', 'calibration')])
        return frame, params['three_feature'], None
    if cycle == 1:
        frame = pd.read_parquet(PREFIX/'calibration_scored.parquet')
        return frame, read(PREFIX/'parameters.json')['detector'], None
    directory = PANEL/'cycles'/f'{cycle:03d}'
    params = read(directory/'parameters.json')
    raw = pd.read_parquet(PANEL/'observations/seasonal_spells.parquet', filters=[
        ('origin_at', '>=', row.training_start), ('origin_at', '<', row.parameter_cutoff)])
    path = model_sample(raw, start_ns=row.training_start.value, stop_ns=row.parameter_cutoff.value)
    pred = predict(path, [params['acd_parameters'][k] for k in ['omega', 'alpha', 'beta', 'shape']], weibull=True)
    duration = completed_duration_bins(path, pred)
    del raw, path, pred
    clock = read_clock(inputs, row.training_start, row.parameter_cutoff)
    factors = pd.read_parquet(inputs['calibration_parameters']['path'], filters=[
        ('effective_date', '>=', row.training_start), ('effective_date', '<', row.parameter_cutoff),
        ('policy', '==', 'structural_only')])
    if not factors.training_end_exclusive.eq(factors.effective_date).all():
        raise ValueError('Future seasonal information')
    _, _, s = feature_inputs(clock, factors, duration, cutoff=row.parameter_cutoff)
    expected = read(directory/'statistics.json')['calibration']
    identity = integer_fingerprint(ns(s.bin_start))
    if identity != expected['eligible_bin_identity_sha256']:
        raise ValueError('Calibration population differs')
    return s, params['detector'], identity


def build(run):
    m = init_run(run); inputs = m['inputs']
    support = pd.read_parquet(inputs['support_intervals']['path']).query("policy == 'structural_only'").reset_index(drop=True)
    schedule = pd.read_parquet(PANEL/'cycle_schedule.parquet')
    carry = [None]*len(RULES)
    for row in schedule.itertuples(index=False):
        cycle = int(row.cycle_id); out = run/'cycles'/f'{cycle:03d}'
        if (out/'detector_done.json').exists():
            done = read(out/'detector_done.json'); carry = done['carry']
            continue
        out.mkdir(exist_ok=True)
        tic = time.monotonic()
        cal, fit, identity = calibration_frame(row, inputs)
        fresh = fit_detector(cal[FEATURES].to_numpy())
        for key in ['mean', 'sample_std', 'covariance_z', 'threshold']:
            np.testing.assert_allclose(fresh[key], fit[key], rtol=3e-10, atol=3e-10, err_msg=f'calibration {cycle} {key}')
        if len(cal) != fit['calibration_rows']:
            raise ValueError('Calibration count differs')
        deployed = pd.read_parquet(PANEL/'cycles'/f'{cycle:03d}'/'scored.parquet')
        core_cal, core_dep = core_scores(cal, fit), core_scores(deployed, fit)
        np.testing.assert_allclose(core_dep[:, 0], deployed.distance, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(core_dep[:, 0] >= fit['threshold'], deployed.candidate_event)
        grid_start = row.training_start-pd.Timedelta(seconds=301)
        grid_stop = row.deployment_end+pd.Timedelta(seconds=2)
        clock = read_clock(inputs, grid_start, row.deployment_end)
        quotes = read_quote_window(inputs['clean_quotes']['path'], grid_start, grid_stop, ['timestamp', 'bid', 'ask'])
        grid = QuoteGrid(quotes, support, grid_start, grid_stop)
        cf, df = grid.features(cal, clock), grid.features(deployed, clock)
        np.testing.assert_allclose(df.current_spread, deployed.end_spread_pips, atol=1e-7, rtol=0)
        np.testing.assert_allclose(df.variation, deployed.pre_rv_300s_5s_bps, atol=1e-7, rtol=1e-7, equal_nan=True)
        cal_scores = np.column_stack([core_cal, cf.current_spread, cf.variation])
        dep_scores = np.column_stack([core_dep, df.current_spread, df.variation])
        thresholds, nc, ties = calibrate(cal_scores)
        np.testing.assert_allclose(thresholds[0], fit['threshold'], atol=1e-9, rtol=1e-9)
        thresholds[0] = fit['threshold']
        cand = dep_scores >= thresholds
        opp = pd.read_parquet(PANEL/'cycles'/f'{cycle:03d}'/'opportunities.parquet')
        eligible = opp.reason.eq('eligible').to_numpy()
        ix = np.flatnonzero(eligible)
        np.testing.assert_array_equal(ns(opp.bin_start)[ix], ns(deployed.bin_start))
        sid = opp.support_id.to_numpy()
        quiet = (sid >= 0) & opp.flow_status.eq(1).to_numpy() & opp.recorded_update_count.eq(0).to_numpy() & opp.completed_residual_count.eq(0).to_numpy()
        ledger = df.copy()
        for name in FEATURES+['quote_ofi', 'recorded_update_count']:
            ledger[name] = deployed[name].to_numpy()
        burden = []
        for j, name in enumerate(RULES):
            known = quiet.copy(); known[ix] = np.isfinite(dep_scores[:, j])
            trigger = np.zeros(len(opp), bool); trigger[ix] = cand[:, j]
            onset, fragment, carry[j] = causal_onsets(sid, known, trigger, carry[j])
            ledger['score_'+name] = dep_scores[:, j]
            ledger['candidate_'+name] = cand[:, j]
            ledger['onset_'+name] = onset[ix]
            ledger['fragment_'+name] = fragment[ix]
            burden.append({'cycle': cycle, 'rule': name, 'calibration_rows': int(nc[j]),
                'threshold': float(thresholds[j]), 'threshold_ties': int(ties[j]),
                'scorable_seconds': int(np.isfinite(dep_scores[:, j]).sum()),
                'known_calendar_seconds': int(known.sum()), 'candidate_seconds': int(cand[:, j].sum()),
                'confirmed_onsets': int(onset.sum()), 'uncertain_onsets': int((fragment & ~onset).sum())})
        inherited = pd.read_parquet(PANEL/'cycles'/f'{cycle:03d}'/'onsets.parquet')
        np.testing.assert_array_equal(ns(ledger.loc[ledger.onset_full, 'bin_start']), ns(inherited.loc[inherited.onset_confirmed, 'bin_start']))
        np.testing.assert_array_equal(ns(ledger.loc[ledger.fragment_full, 'bin_start']), ns(inherited.bin_start))
        mature = ns(cf.bin_start)+301*SECOND <= row.parameter_cutoff.value
        train_take = hash_training_sample(ns(cf.bin_start)) & cf.reference_ready.to_numpy() & mature
        train = cf.loc[train_take].copy()
        train['quote_ofi'] = cal.loc[train_take, 'quote_ofi'].to_numpy()
        save(out/'ledger.parquet', ledger)
        save(out/'training_features.parquet', train)
        save(out/'burden.parquet', pd.DataFrame(burden))
        atomic_json(out/'calibration.json', {'full_fit': fit, 'thresholds': dict(zip(RULES, thresholds.tolist())),
            'training_start': row.training_start.isoformat(), 'cutoff': row.parameter_cutoff.isoformat(),
            'deployment_end': row.deployment_end.isoformat(), 'calibration_identity': identity,
            'calibration_rows': len(cal), 'reference_training_selected': len(train)})
        input_paths = [PANEL/'cycles'/f'{cycle:03d}'/'scored.parquet', PANEL/'cycles'/f'{cycle:03d}'/'opportunities.parquet',
                       PANEL/'cycles'/f'{cycle:03d}'/'onsets.parquet']
        atomic_json(out/'detector_done.json', {'status': 'passed', 'carry': carry,
            'source_fingerprints': {str(p): sha256(p) for p in input_paths},
            'outputs': {p.name: sha256(p) for p in out.glob('*.parquet')},
            'full_scores_candidates_onsets_reproduced': True, 'seconds': time.monotonic()-tic})
        atomic_json(run/'progress.json', {'phase': 'detector_reconstruction', 'completed_cycles': cycle+1, 'total_cycles': len(schedule)})
        print(f'Detectors {cycle+1:02d}/{len(schedule)}: {len(ledger):,} common seconds, original scores/onsets reproduced; {time.monotonic()-tic:.1f}s', flush=True)
        del cal, deployed, core_cal, core_dep, clock, quotes, grid, cf, df, cal_scores, dep_scores, cand, opp, ledger, train
        gc.collect()
    all_burden = pd.concat([pd.read_parquet(p/'burden.parquet') for p in sorted((run/'cycles').iterdir())], ignore_index=True)
    save(run/'detector_burden.parquet', all_burden)
    m['status'] = 'detectors_verified_economics_pending'; atomic_json(run/'manifest.json', m)


def stratified_reference(train, y, deployed):
    edges = np.quantile(train.current_spread, np.arange(.1, 1, .1))
    bins = np.searchsorted(edges, train.current_spread.to_numpy(), side='right')
    dbins = np.searchsorted(edges, deployed.current_spread.to_numpy(), side='right')
    work = pd.DataFrame({'bin': bins, 'hour': train.hour.to_numpy(), 'y': y})
    cells = work.groupby(['bin', 'hour']).y.mean()
    fallback = work.groupby('bin').y.mean()
    keys = pd.MultiIndex.from_arrays([dbins, deployed.hour])
    result = cells.reindex(keys).to_numpy()
    coarse = fallback.reindex(dbins).to_numpy()
    return np.where(np.isfinite(result), result, np.where(np.isfinite(coarse), coarse, np.mean(y)))


def analyze(run):
    m = init_run(run)
    if m['status'] not in ['detectors_verified_economics_pending', 'economics_building', 'economic_comparison_complete']:
        raise ValueError('Complete detector verification first')
    m['status'] = 'economics_building'; atomic_json(run/'manifest.json', m)
    support = pd.read_parquet(m['inputs']['support_intervals']['path']).query("policy == 'structural_only'").reset_index(drop=True)
    for out in sorted((run/'cycles').iterdir()):
        cycle = int(out.name)
        if cycle == 0 or (out/'economic_done.json').exists():
            continue
        tic = time.monotonic()
        cal = read(out/'calibration.json')
        train = pd.read_parquet(out/'training_features.parquet')
        ledger = pd.read_parquet(out/'ledger.parquet')
        start = pd.Timestamp(cal['training_start'])-pd.Timedelta(seconds=1)
        stop = pd.Timestamp(cal['deployment_end'])+pd.Timedelta(seconds=303)
        quotes = read_quote_window(m['inputs']['clean_quotes']['path'], start, stop, ['timestamp', 'bid', 'ask'])
        grid = QuoteGrid(quotes, support, start, stop)
        ty = grid.outcomes(train)
        keep = ty.outcome_ready.to_numpy()
        train = train.loc[keep].reset_index(drop=True); y = ty.y300.to_numpy()[keep]
        if len(train) < 1000 or not (ns(train.bin_start)+301*SECOND <= pd.Timestamp(cal['cutoff']).value).all():
            raise ValueError('Insufficient or unmatured reference training')
        model = HistGradientBoostingRegressor(**MODEL).fit(train[XCOLS], y)
        ready = ledger.reference_ready.to_numpy()
        predictions = np.full(len(ledger), np.nan)
        predictions[ready] = model.predict(ledger.loc[ready, XCOLS])
        if not np.isfinite(predictions[ready]).all():
            raise ValueError('Nonfinite reference forecast')
        economic = grid.outcomes(ledger)
        economic.insert(0, 'bin_start', ledger.bin_start)
        economic['prediction'] = predictions
        economic['adjusted'] = economic.y300-predictions
        strat = np.full(len(ledger), np.nan)
        strat[ready] = stratified_reference(train, y, ledger.loc[ready])
        economic['stratified_adjusted'] = economic.y300-strat
        economic['primary_ready'] = ready & economic.outcome_ready.to_numpy()
        economic['max_age'] = np.maximum(ledger.history_age.to_numpy(), economic.future_age.to_numpy())
        save(out/'economics.parquet', economic)
        joblib.dump(model, out/'reference_model.joblib')
        save(out/'training_labels.parquet', pd.DataFrame({'bin_start': train.bin_start, 'y300': y}))
        atomic_json(out/'economic_done.json', {'status': 'passed', 'training_rows': len(train),
            'prediction_origins': int(ready.sum()), 'primary_origins': int(economic.primary_ready.sum()),
            'seconds': time.monotonic()-tic, 'outputs': {p.name: sha256(p) for p in [out/'economics.parquet', out/'reference_model.joblib', out/'training_labels.parquet']}})
        atomic_json(run/'progress.json', {'phase': 'economic_comparison', 'completed_evaluation_cycles': cycle, 'total_evaluation_cycles': 59})
        print(f'Economics {cycle:02d}/59: {len(train):,} prior training labels, {int(economic.primary_ready.sum()):,} evaluation origins; {time.monotonic()-tic:.1f}s', flush=True)
        del train, ledger, quotes, grid, ty, model, economic
        gc.collect()
    summarize(run)
    m['status'] = 'economic_comparison_complete'; atomic_json(run/'manifest.json', m)


def summarize(run):
    dates = pd.date_range(START, END, inclusive='left')
    metrics = ['y300', 'adjusted', 'stratified_adjusted', 'y60', 'absolute_move', 'J', 'C', 'I']
    daily = {key: np.zeros((len(dates), len(RULES), 2)) for key in metrics}
    age_daily = {cap: np.zeros((len(dates), len(RULES), 2)) for cap in [15, 30, 60]}
    group_rows = []; onsets = []; coverage = []; calibration = []
    for out in sorted((run/'cycles').iterdir()):
        cycle = int(out.name)
        if cycle == 0:
            continue
        f = pd.read_parquet(out/'ledger.parquet'); e = pd.read_parquet(out/'economics.parquet')
        np.testing.assert_array_equal(ns(f.bin_start), ns(e.bin_start))
        date_ix = dates.get_indexer(f.bin_start.dt.normalize())
        primary = e.primary_ready.to_numpy()
        allmask = primary & np.isfinite(e.prediction)
        calibration.append({'cycle': cycle, 'n': int(allmask.sum()), 'sum_y': e.loc[allmask, 'y300'].sum(),
            'sum_prediction': e.loc[allmask, 'prediction'].sum(), 'sum_error': e.loc[allmask, 'adjusted'].sum(),
            'sum_squared_error': (e.loc[allmask, 'adjusted']**2).sum(),
            'sum_squared_current_spread_error': ((e.loc[allmask, 'y300']-f.loc[allmask, 'current_spread'])**2).sum()})
        for j, rule in enumerate(RULES):
            alerts = f['onset_'+rule].to_numpy()
            coverage.append({'cycle': cycle, 'rule': rule, 'issued': int(alerts.sum()),
                'missing_reference': int((alerts & ~f.reference_ready.to_numpy()).sum()),
                'missing_y300': int((alerts & ~e.outcome_ready.to_numpy()).sum()),
                'evaluable': int((alerts & primary).sum())})
            oo = f.loc[alerts, ['bin_start', 'support_id']].copy(); oo['rule'] = rule; onsets.append(oo)
            for metric in metrics:
                y = e[metric].to_numpy(); mask = alerts & primary & np.isfinite(y)
                daily[metric][:, j, 0] += np.bincount(date_ix[mask], weights=y[mask], minlength=len(dates))
                daily[metric][:, j, 1] += np.bincount(date_ix[mask], minlength=len(dates))
            for cap in age_daily:
                mask = alerts & primary & e.max_age.le(cap).to_numpy()
                age_daily[cap][:, j, 0] += np.bincount(date_ix[mask], weights=e.adjusted.to_numpy()[mask], minlength=len(dates))
                age_daily[cap][:, j, 1] += np.bincount(date_ix[mask], minlength=len(dates))
        for kind in ['candidate', 'onset']:
            full = f[kind+'_full'].to_numpy()
            for other in RULES[1:]:
                comparator = f[kind+'_'+other].to_numpy()
                code = 2*full.astype(np.uint8)+comparator.astype(np.uint8)
                for g, label in enumerate(['neither', 'benchmark_only', 'full_only', 'both']):
                    mask = code == g; valid = mask & primary
                    r = {'cycle': cycle, 'kind': kind, 'benchmark': other, 'group': label,
                         'n': int(mask.sum()), 'evaluable': int(valid.sum())}
                    for metric in ['y300', 'adjusted', 'stratified_adjusted', 'absolute_move']:
                        r['sum_'+metric] = float(e.loc[valid, metric].sum())
                    for col in ['current_spread', *FEATURES]:
                        r['sum_'+col] = float(f.loc[valid, col].sum())
                    group_rows.append(r)
        del f, e
        gc.collect()
    pd.DataFrame(coverage).to_csv(run/'coverage.csv', index=False)
    pd.DataFrame(calibration).to_csv(run/'reference_calibration.csv', index=False)
    groups = pd.DataFrame(group_rows)
    save(run/'group_sufficient_statistics.parquet', groups)
    agg = groups.groupby(['kind', 'benchmark', 'group'], sort=False).sum(numeric_only=True).drop(columns='cycle').reset_index()
    for col in [c for c in agg if c.startswith('sum_')]:
        agg['mean_'+col[4:]] = agg[col]/agg.evaluable
    agg.to_csv(run/'group_comparison.csv', index=False)
    summary = []; contrast = []; daily_frames = []
    for metric, data in daily.items():
        for j, rule in enumerate(RULES):
            n = data[:, j, 1].sum()
            summary.append({'metric': metric, 'rule': rule, 'n': int(n), 'mean': data[:, j, 0].sum()/n})
            daily_frames.append(pd.DataFrame({'date': dates, 'metric': metric, 'rule': rule,
                'sum': data[:, j, 0], 'count': data[:, j, 1].astype(int)}))
        if metric in ['y300', 'adjusted', 'stratified_adjusted']:
            for lag in [28, 56, 112]:
                c = contrast_table(data[:, :, 0], data[:, :, 1], lag); c['metric'] = metric
                c['max_influence_date'] = dates[c.max_influence_day_index].astype(str)
                contrast.append(c)
    pd.DataFrame(summary).to_csv(run/'economic_summary.csv', index=False)
    pd.concat(contrast, ignore_index=True).to_csv(run/'contrasts.csv', index=False)
    d = pd.concat(daily_frames, ignore_index=True); save(run/'daily_sufficient_statistics.parquet', d)
    d['quarter'] = d.date.dt.tz_localize(None).dt.to_period('Q').astype(str)
    q = d.groupby(['quarter', 'metric', 'rule'])[['sum', 'count']].sum().reset_index()
    q['mean'] = q['sum']/q['count']; q.to_csv(run/'quarterly_summary.csv', index=False)
    age_rows = []
    for cap, data in age_daily.items():
        for j, rule in enumerate(RULES):
            n = data[:, j, 1].sum()
            age_rows.append({'age_cap': cap, 'rule': rule, 'n': int(n), 'mean_adjusted': data[:, j, 0].sum()/n if n else None})
    pd.DataFrame(age_rows).to_csv(run/'age_sensitivity.csv', index=False)
    onset = pd.concat(onsets, ignore_index=True).sort_values(['rule', 'support_id', 'bin_start'])
    save(run/'all_confirmed_onsets.parquet', onset)
    timings = []
    for rule in RULES[1:]:
        left = onset.loc[onset.rule.eq('full')]; right = onset.loc[onset.rule.eq(rule)]
        offsets = []
        right_groups = {int(sid): ns(x.bin_start)//SECOND for sid, x in right.groupby('support_id')}
        for sid, x in left.groupby('support_id'):
            offsets.extend(match_times(ns(x.bin_start)//SECOND, right_groups.get(int(sid), np.array([], np.int64))))
        offsets = np.array(offsets)
        exact = len(np.intersect1d(ns(left.bin_start), ns(right.bin_start)))
        timings.append({'benchmark': rule, 'full_onsets': len(left), 'benchmark_onsets': len(right),
            'exact_matches': exact, 'matched_within_5s': len(offsets),
            'full_unmatched': len(left)-len(offsets), 'benchmark_unmatched': len(right)-len(offsets),
            'mean_benchmark_minus_full_seconds': float(offsets.mean()) if len(offsets) else None})
    pd.DataFrame(timings).to_csv(run/'timing_comparison.csv', index=False)
    atomic_json(run/'summary.json', {'status': 'exploratory_comparison_complete', 'calendar_days': len(dates),
        'cycles': 59, 'common_primary_origins': int(pd.DataFrame(calibration).n.sum()),
        'economic_materiality_established': False, 'independent_validation': False,
        'conditional_nominal_intervals_only': True})
    print('Economic summaries saved; interpretation and independent arithmetic verification pending.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['build', 'analyze', 'summarize'])
    parser.add_argument('--run-id', default='step11_detector_contribution_20261004')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid run id')
    run = ROOT/'data/signals'/args.run_id
    {'build': build, 'analyze': analyze, 'summarize': summarize}[args.phase](run)

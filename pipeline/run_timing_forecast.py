"""New fixed timing-normalization forecast pair, with measurement/fit gates."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_preparation import atomic_json, sha256
from src.event_audit import ns
from src.timing_forecast import (BASE, EXTRA, MODEL, START, STOP, SECOND, STATS,
    require, seasonal_bins, predictor_frame, select_training, quartile_groups,
    aggregate, estimates)

ROOT = Path(__file__).resolve().parent
PANEL = ROOT/'data/signals/step07c_chronological_panel_20261001_v2'
ARCHIVE = ROOT/'data/signals/step11_detector_contribution_20261004'
PROTOCOL = ROOT/'reports/STEP03_TIMING_FORECAST_PROTOCOL_2026_10_08.md'
SOURCES = [Path(__file__), ROOT/'src/timing_forecast.py', PROTOCOL,
    ROOT/'verify_timing_forecast.py', ROOT/'tests/test_timing_forecast.py',
    ROOT/'src/detector_contribution.py']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, frame):
    frame.to_parquet(path, index=False, row_group_size=400000)


def init(run):
    if (run/'manifest.json').exists():
        m = read(run/'manifest.json')
        for name, digest in m['code_sha256'].items():
            require(sha256(ROOT/name) == digest, 'Code changed after freeze: '+name)
        return m
    require(read(ROOT/'reports/STEP02_TIMING_MEASUREMENT_AUDIT_2026_10_08.json')['status'] == 'passed', 'Prior measurement audit not passed')
    run.mkdir(parents=True, exist_ok=False)
    (run/'cycles').mkdir()
    code = {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES}
    for p in SOURCES:
        destination = run/'code'/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    protected = read(ROOT/'data/signals/step12_conditional_response_20261004/preserved_scientific_state.json')
    for name, digest in protected.items():
        require(sha256(ROOT/name) == digest, 'Earlier scientific state changed: '+name)
    for name in ['reports/STEP02_TIMING_MEASUREMENT_AUDIT_2026_10_08.json',
        'reports/STEP02_TIMING_MEASUREMENT_PROTOCOL_2026_10_08.md',
        'paper/STEP01_TIMING_LIQUIDITY_QUESTION_2026_10_08.md',
        'reports/STEP01_TIMING_LIQUIDITY_SOURCE_AUDIT_2026_10_08.json']:
        protected[name] = sha256(ROOT/name)
    atomic_json(run/'preserved_scientific_state.json', protected)
    seasonal = PANEL/'observations/seasonal_spells.parquet'
    m = {'run_id': run.name, 'status': 'features_building',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'one_exploratory_incremental_normalization_forecast_pair',
        'code_sha256': code, 'seasonal_spells_sha256': sha256(seasonal),
        'prior_audit_sha256': sha256(ROOT/'reports/STEP02_TIMING_MEASUREMENT_AUDIT_2026_10_08.json'),
        'feature_archive': str(ARCHIVE), 'panel': str(PANEL),
        'start': START.isoformat(), 'stop': STOP.isoformat(),
        'first_evaluation_cycle': 5, 'last_evaluation_cycle': 59,
        'base_columns': BASE, 'extension_column': EXTRA, 'model': MODEL,
        'independent_validation': False, 'economic_materiality_margin': None,
        'full_pipeline_interval_coverage_established': False,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    atomic_json(run/'manifest.json', m)
    return m


def check_preservation(run):
    protected = read(run/'preserved_scientific_state.json')
    for name, digest in protected.items():
        require(sha256(ROOT/name) == digest, 'Protected file changed: '+name)
    return len(protected)


def build_features(run):
    m = init(run)
    schedule = pd.read_parquet(PANEL/'cycle_schedule.parquet')
    for row in schedule.itertuples(index=False):
        c = int(row.cycle_id)
        if c == 0:
            continue
        out = run/'cycles'/f'{c:03d}'
        if (out/'features_done.json').exists():
            continue
        out.mkdir(exist_ok=True)
        tic = time.monotonic()
        parent = PANEL/'cycles'/f'{c:03d}'
        old = ARCHIVE/'cycles'/f'{c:03d}'
        s = pd.read_parquet(parent/'scored.parquet', columns=['bin_start', 'available_at',
            'support_id', 'recorded_update_count', 'completed_residual_count',
            'pre_quote_age_sec', 'end_quote_age_sec', EXTRA])
        ledger = pd.read_parquet(old/'ledger.parquet', columns=['bin_start', 'support_id',
            'recorded_update_count']+BASE[:13]+[EXTRA])
        raw = pd.read_parquet(PANEL/'observations/seasonal_spells.parquet', columns=[
            'origin_at', 'end_at', 'event_observed', 'scale_sec', 'seasonal_cutoff', 'exposure_ns'],
            filters=[('end_at', '>=', row.parameter_cutoff), ('end_at', '<', row.deployment_end),
                     ('event_observed', '==', True)])
        f = predictor_frame(s, ledger, seasonal_bins(raw))
        require(np.ptp((f[EXTRA]-f.seasonal_speed).to_numpy()) > 1e-10, 'Normalized representation duplicates seasonal feature')
        save(out/'features.parquet', f)
        done = {'cycle': c, 'rows': len(f), 'forecast_ready': int(f.forecast_ready.sum()),
            'start': row.parameter_cutoff.isoformat(), 'stop': row.deployment_end.isoformat(),
            'seconds': time.monotonic()-tic, 'feature_sha256': sha256(out/'features.parquet'),
            'sources': {str(p.relative_to(ROOT)): sha256(p) for p in [parent/'scored.parquet', old/'ledger.parquet']}}
        atomic_json(out/'features_done.json', done)
        print(f'Features {c:02d}/59: {len(f):,} rows; {done["seconds"]:.1f}s', flush=True)
        del raw, s, ledger, f
        gc.collect()
    m['status'] = 'features_built_pending_verification'
    m['protected_files_unchanged'] = check_preservation(run)
    atomic_json(run/'manifest.json', m)


def with_labels(run, c):
    f = pd.read_parquet(run/'cycles'/f'{c:03d}'/'features.parquet')
    p = ARCHIVE/'cycles'/f'{c:03d}'/'economics.parquet'
    e = pd.read_parquet(p, columns=['bin_start', 'y300', 'outcome_ready'])
    require(np.array_equal(ns(f.bin_start), ns(e.bin_start)), 'Archive label identity differs')
    f['y300'], f['outcome_ready'] = e.y300.to_numpy(), e.outcome_ready.to_numpy()
    return f


def fit(run):
    m = init(run)
    gate = read(run/'feature_verification.json')
    require(gate['status'] == 'passed', 'Feature verification must pass before fitting')
    m['status'] = 'forecast_pair_running'
    atomic_json(run/'manifest.json', m)
    schedule = pd.read_parquet(PANEL/'cycle_schedule.parquet')
    for c in range(5, 60):
        out = run/'cycles'/f'{c:03d}'
        if (out/'fit_done.json').exists():
            continue
        tic = time.monotonic()
        row = schedule.iloc[c]
        cutoff = row.parameter_cutoff
        parts = []
        for h in range(c-4, c):
            f = with_labels(run, h)
            parts.append(f.loc[select_training(f, cutoff)].copy())
        train = pd.concat(parts, ignore_index=True)
        require(len(train) >= 1000 and not train.bin_start.duplicated().any(), 'Insufficient/duplicated training')
        require((ns(train.bin_start)+301*SECOND <= cutoff.value).all(), 'Unmatured training target')
        save(out/'training_origins.parquet', train[['bin_start', 'available_at', 'y300']])
        deployment = pd.read_parquet(out/'features.parquet')
        ready = deployment.forecast_ready.to_numpy()
        y = train.y300.to_numpy()
        prediction = {}
        for name, columns in [('b', BASE), ('t', BASE+[EXTRA])]:
            model = HistGradientBoostingRegressor(**MODEL).fit(train[columns], y)
            p = np.full(len(deployment), np.nan)
            p[ready] = model.predict(deployment.loc[ready, columns])
            require(np.isfinite(p[ready]).all(), 'Nonfinite issued prediction')
            prediction[name] = p
            joblib.dump(model, out/f'model_{name}.joblib')
        label_path = ARCHIVE/'cycles'/f'{c:03d}'/'economics.parquet'
        e = pd.read_parquet(label_path, columns=['bin_start', 'y300', 'outcome_ready'])
        require(np.array_equal(ns(deployment.bin_start), ns(e.bin_start)), 'Deployment target identity')
        result = deployment[['bin_start', 'available_at', 'forecast_ready']].copy()
        result['prediction_b'], result['prediction_t'] = prediction['b'], prediction['t']
        result['y300'] = e.y300.to_numpy()
        result['outcome_ready'] = e.outcome_ready.to_numpy()
        result['evaluation_ready'] = ready & e.outcome_ready.to_numpy() & np.isfinite(e.y300.to_numpy())
        for name in ['b', 't']:
            result['loss_'+name] = np.where(result.evaluation_ready,
                (result.y300-result['prediction_'+name])**2, np.nan)
        result['d'] = result.loss_b-result.loss_t
        result['loss_carry'] = np.where(result.evaluation_ready,
            (result.y300-deployment.current_spread)**2, np.nan)
        spread_q = np.quantile(train.current_spread, [.25, .5, .75]).tolist()
        norm_q = np.quantile(train[EXTRA], [.25, .5, .75]).tolist()
        result['spread_group'] = quartile_groups(deployment.current_spread, spread_q)
        result['timing_group'] = quartile_groups(deployment[EXTRA], norm_q)
        result['count_group'] = np.select([deployment.recorded_update_count.eq(1),
            deployment.recorded_update_count.between(2, 4)], [1, 2], default=3)
        result['utc_block'] = (deployment.bin_start.dt.hour//6).astype(int)
        save(out/'predictions.parquet', result)
        label_inputs = {}
        for h in range(c-4, c+1):
            p = ARCHIVE/'cycles'/f'{h:03d}'/'economics.parquet'
            label_inputs[str(p.relative_to(ROOT))] = sha256(p)
        outputs = {p.name: sha256(p) for p in [out/'training_origins.parquet',
            out/'predictions.parquet', out/'model_b.joblib', out/'model_t.joblib']}
        done = {'cycle': c, 'cutoff': cutoff.isoformat(), 'training_rows': len(train),
            'training_start': train.bin_start.min().isoformat(),
            'last_training_label_available_at': (train.bin_start.max()+pd.Timedelta(seconds=301)).isoformat(),
            'issued_forecasts': int(ready.sum()), 'evaluable_forecasts': int(result.evaluation_ready.sum()),
            'missing_features': int((~ready).sum()),
            'issued_without_future_label': int((ready & ~result.evaluation_ready).sum()),
            'training_spread_quartiles': spread_q, 'training_timing_quartiles': norm_q,
            'label_source_sha256': label_inputs, 'outputs': outputs,
            'seconds': time.monotonic()-tic}
        atomic_json(out/'fit_done.json', done)
        atomic_json(run/'progress.json', {'phase': 'fixed_forecast_pair', 'last_completed_cycle': c,
            'completed_evaluation_cycles': c-4, 'total_evaluation_cycles': 55})
        print(f'Pair {c:02d}/59: {len(train):,} training; {done["evaluable_forecasts"]:,} evaluated; {done["seconds"]:.1f}s', flush=True)
        del train, deployment, e, result, model, parts
        gc.collect()
    summarize(run)
    m['status'] = 'comparison_complete_pending_verification'
    m['protected_files_unchanged'] = check_preservation(run)
    atomic_json(run/'manifest.json', m)


def summarize(run):
    days, patterns, coverage = [], [], []
    for c in range(5, 60):
        out = run/'cycles'/f'{c:03d}'
        p = pd.read_parquet(out/'predictions.parquet')
        coverage.append(read(out/'fit_done.json'))
        p = p.loc[p.evaluation_ready].copy()
        p['date'] = p.bin_start.dt.normalize()
        days.append(aggregate(p, ['date']))
        p['quarter'] = p.bin_start.dt.year.astype(str)+'Q'+((p.bin_start.dt.month-1)//3+1).astype(str)
        for dimension in ['quarter', 'utc_block', 'count_group', 'spread_group', 'timing_group']:
            a = aggregate(p, ['date', dimension]).rename(columns={dimension: 'group'})
            a['group'] = a.group.astype(str)
            a['dimension'] = dimension
            patterns.append(a)
    daily = pd.concat(days).groupby('date')[STATS].sum().reindex(
        pd.date_range(START, STOP, inclusive='left'), fill_value=0).rename_axis('date').reset_index()
    save(run/'daily.parquet', daily)
    pattern_daily = pd.concat(patterns, ignore_index=True)
    save(run/'pattern_daily.parquet', pattern_daily)
    groups = pattern_daily.groupby(['dimension', 'group'])[STATS].sum().reset_index()
    for stem in ['y', 'b', 't']:
        groups['mean_'+stem] = groups['sum_'+stem]/groups.n
    groups['mse_b'], groups['mse_t'] = groups.sum_lb/groups.n, groups.sum_lt/groups.n
    groups['delta'] = groups.sum_d/groups.n
    groups['relative_reduction'] = groups.sum_d/groups.sum_lb
    save(run/'patterns.parquet', groups)
    result = estimates(daily)
    result['coverage'] = {'scored_origins': sum(x['issued_forecasts']+x['missing_features'] for x in coverage),
        'issued_forecasts': sum(x['issued_forecasts'] for x in coverage),
        'missing_features': sum(x['missing_features'] for x in coverage),
        'issued_without_future_label': sum(x['issued_without_future_label'] for x in coverage),
        'evaluation_cycles': 55, 'calendar_days': len(daily)}
    result['top_absolute_daily_loss_differences'] = json.loads(daily.assign(
        absolute=daily.sum_d.abs()).nlargest(10, 'absolute')[['date', 'n', 'sum_d']].to_json(orient='records', date_format='iso'))
    result['scope'] = 'Realized exploratory forecast pair; nominal conditional intervals; no field-novelty or causal claim.'
    atomic_json(run/'summary.json', result)
    print('Saved primary estimate and every fixed descriptive pattern.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['features', 'fit'])
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require('/' not in args.run_id and args.run_id not in ['.', '..'], 'Safe run identifier required')
    run = ROOT/'data/signals'/args.run_id
    (build_features if args.phase == 'features' else fit)(run)


if __name__ == '__main__':
    main()

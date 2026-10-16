"""Independent source aggregation, archived training and issued-loss checks."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_preparation import atomic_json, sha256
from src.event_audit import ns
from src.detector_contribution import hash_training_sample
from src.timing_forecast import BASE, EXTRA, START, STOP, STATS, MODEL

ROOT = Path(__file__).resolve().parent
PANEL = ROOT/'data/signals/step07c_chronological_panel_20261001_v2'
ARCHIVE = ROOT/'data/signals/step11_detector_contribution_20261004'
SECOND = 1_000_000_000


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def close(a, b, label, atol=1e-10):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=atol, equal_nan=True, err_msg=label)


def fingerprints(run):
    manifest = read(run/'manifest.json')
    for name, digest in manifest['code_sha256'].items():
        require(sha256(ROOT/name) == digest, 'Frozen code differs: '+name)
    for name, digest in read(run/'preserved_scientific_state.json').items():
        require(sha256(ROOT/name) == digest, 'Earlier scientific state differs: '+name)
    return manifest


def verify_features(run):
    m = fingerprints(run)
    require(sha256(PANEL/'observations/seasonal_spells.parquet') == m['seasonal_spells_sha256'], 'Seasonal source changed')
    count, maximum = 0, 0.0
    for c in range(1, 60):
        out = run/'cycles'/f'{c:03d}'
        done = read(out/'features_done.json')
        require(sha256(out/'features.parquet') == done['feature_sha256'], 'Features changed')
        for name, digest in done['sources'].items():
            require(sha256(ROOT/name) == digest, 'Feature parent changed')
        f = pd.read_parquet(out/'features.parquet')
        start, stop = pd.Timestamp(done['start']), pd.Timestamp(done['stop'])
        raw = pd.read_parquet(PANEL/'observations/seasonal_spells.parquet', columns=[
            'origin_at', 'end_at', 'scale_sec', 'seasonal_cutoff'], filters=[
                ('end_at', '>=', start), ('end_at', '<', stop), ('event_observed', '==', True)])
        require(raw.seasonal_cutoff.le(raw.origin_at).all(), 'Future seasonal vintage')
        ends = ns(raw.end_at)
        seconds, inverse = np.unique(ends//SECOND, return_inverse=True)
        n = np.bincount(inverse)
        scale = raw.scale_sec.to_numpy()
        valid = np.isfinite(scale) & (scale > 0)
        complete_scale = np.bincount(inverse, weights=valid.astype(int))
        scaled = np.zeros(len(raw))
        scaled[valid] = (ends[valid]-ns(raw.origin_at)[valid])/SECOND/scale[valid]
        sums = np.bincount(inverse, weights=scaled)
        positions = np.searchsorted(seconds, ns(f.bin_start)//SECOND)
        require((positions < len(seconds)).all(), 'Missing source bin')
        require(np.array_equal(seconds[positions], ns(f.bin_start)//SECOND), 'Source bin identity differs')
        require(np.array_equal(n[positions], f.recorded_update_count), 'Source completed count differs')
        require(np.array_equal(complete_scale[positions], f.recorded_update_count), 'Incomplete source scales')
        expected = sums[positions]/n[positions]
        close(expected, f.mean_seasonal_gap, 'Independent seasonal mean')
        maximum = max(maximum, float(np.max(np.abs(expected-f.mean_seasonal_gap))))
        close(-np.log(expected), f.seasonal_speed, 'Seasonal log order')
        s = pd.read_parquet(PANEL/'cycles'/f'{c:03d}'/'scored.parquet', columns=[
            'bin_start', 'available_at', 'pre_quote_age_sec', 'end_quote_age_sec', EXTRA])
        require(np.array_equal(ns(s.bin_start), ns(f.bin_start)), 'Scored origin identity')
        require(np.array_equal(ns(f.available_at), ns(f.bin_start)+SECOND), 'Feature clock')
        expected_raw = (1+s.pre_quote_age_sec.to_numpy()-s.end_quote_age_sec.to_numpy())/f.recorded_update_count.to_numpy()
        close(expected_raw, f.mean_raw_gap, 'Raw summary representation')
        close(-np.log(expected_raw), f.raw_speed, 'Raw logarithm')
        close(f[EXTRA], s[EXTRA], 'Archived normalized timing changed', atol=0)
        require(np.array_equal(f.forecast_ready, np.isfinite(f[BASE+[EXTRA]]).all(axis=1)), 'Feature eligibility differs')
        count += len(f)
        print(f'Independent seasonal check {c:02d}/59 passed', flush=True)
    receipt = {'status': 'passed', 'cycles': 59, 'source_reconstructed_rows': count,
        'max_absolute_seasonal_mean_discrepancy': maximum,
        'algorithm': 'Independent numpy unique/bincount aggregation of completed source gaps',
        'future_outcomes_read': 0, 'protected_files_unchanged': len(read(run/'preserved_scientific_state.json'))}
    atomic_json(run/'feature_verification.json', receipt)
    m['status'] = 'features_verified_forecasts_pending'
    atomic_json(run/'manifest.json', m)


def verify_results(run):
    m = fingerprints(run)
    dates = pd.date_range(START, STOP, inclusive='left')
    daily = np.zeros((len(dates), len(STATS)))
    group_parts = []
    training_rows = 0
    prediction_checks = 0
    all_rows = 0
    for c in range(5, 60):
        out = run/'cycles'/f'{c:03d}'
        done = read(out/'fit_done.json')
        for name, digest in done['outputs'].items():
            require(sha256(out/name) == digest, 'Saved output differs: '+name)
        for name, digest in done['label_source_sha256'].items():
            require(sha256(ROOT/name) == digest, 'Archived label source changed')
        cutoff = pd.Timestamp(done['cutoff'])
        train_parts = []
        for h in range(c-4, c):
            f = pd.read_parquet(run/'cycles'/f'{h:03d}'/'features.parquet')
            e = pd.read_parquet(ARCHIVE/'cycles'/f'{h:03d}'/'economics.parquet', columns=['bin_start', 'y300', 'outcome_ready'])
            require(np.array_equal(ns(f.bin_start), ns(e.bin_start)), 'Prior label origin differs')
            times = ns(f.bin_start)
            mask = ((times >= cutoff.value-56*86400*SECOND) &
                    (times+301*SECOND <= cutoff.value) &
                    np.isfinite(f[BASE+[EXTRA]]).all(axis=1).to_numpy() &
                    e.outcome_ready.to_numpy() & np.isfinite(e.y300.to_numpy()) & hash_training_sample(times))
            f = f.loc[mask].copy()
            f['y300'] = e.y300.to_numpy()[mask]
            train_parts.append(f)
        train = pd.concat(train_parts, ignore_index=True)
        saved_train = pd.read_parquet(out/'training_origins.parquet')
        require(np.array_equal(ns(train.bin_start), ns(saved_train.bin_start)), 'Training origins differ')
        require(np.array_equal(train.y300, saved_train.y300), 'Training target differs')
        require(np.array_equal(ns(train.available_at), ns(saved_train.available_at)), 'Training feature clock differs')
        require(train.bin_start.max()+pd.Timedelta(seconds=301) <= cutoff, 'Future training target')
        training_rows += len(train)
        f = pd.read_parquet(out/'features.parquet')
        p = pd.read_parquet(out/'predictions.parquet')
        e = pd.read_parquet(ARCHIVE/'cycles'/f'{c:03d}'/'economics.parquet', columns=['bin_start', 'y300', 'outcome_ready'])
        require(np.array_equal(ns(f.bin_start), ns(p.bin_start)) and np.array_equal(ns(p.bin_start), ns(e.bin_start)), 'Deployment origin mismatch')
        close(p.y300, e.y300, 'Archived evaluation target', atol=0)
        ready = np.isfinite(f[BASE+[EXTRA]]).all(axis=1).to_numpy()
        require(np.array_equal(p.forecast_ready, ready), 'Issued prediction population')
        require(np.array_equal(p.evaluation_ready, ready & e.outcome_ready.to_numpy() & np.isfinite(e.y300)), 'Evaluation availability')
        positions = np.flatnonzero(ready)
        sample = positions[np.linspace(0, len(positions)-1, min(1000, len(positions)), dtype=int)]
        for name, columns in [('b', BASE), ('t', BASE+[EXTRA])]:
            model = joblib.load(out/f'model_{name}.joblib')
            for key, value in MODEL.items():
                require(model.get_params()[key] == value, 'Changed model specification')
            require(list(model.feature_names_in_) == columns, 'Changed model columns')
            close(model.predict(f.iloc[sample][columns]), p.iloc[sample]['prediction_'+name], 'Saved issued prediction', atol=1e-12)
            prediction_checks += len(sample)
            expected_loss = np.where(p.evaluation_ready, (e.y300.to_numpy()-p['prediction_'+name].to_numpy())**2, np.nan)
            close(expected_loss, p['loss_'+name], 'Paired squared loss')
        close(p.loss_b-p.loss_t, p.d, 'Paired loss difference')
        sq = np.quantile(train.current_spread, [.25, .5, .75])
        tq = np.quantile(train[EXTRA], [.25, .5, .75])
        close(sq, done['training_spread_quartiles'], 'Past spread quartiles')
        close(tq, done['training_timing_quartiles'], 'Past timing quartiles')
        require(np.array_equal((f.current_spread.to_numpy()[:, None] >= sq).sum(axis=1)+1, p.spread_group), 'Spread group differs')
        require(np.array_equal((f[EXTRA].to_numpy()[:, None] >= tq).sum(axis=1)+1, p.timing_group), 'Timing group differs')
        require(np.array_equal(np.where(f.recorded_update_count.eq(1), 1, np.where(f.recorded_update_count.le(4), 2, 3)), p.count_group), 'Activity grouping differs')
        require(np.array_equal(f.bin_start.dt.hour//6, p.utc_block), 'UTC grouping differs')
        mask = p.evaluation_ready.to_numpy()
        q = p.loc[mask].copy()
        close((q.y300-f.loc[mask, 'current_spread'])**2, q.loss_carry, 'Carry diagnostic')
        day = ((ns(q.bin_start)-START.value)//(86400*SECOND)).astype(int)
        values = np.column_stack([np.ones(len(q)), q.y300, q.prediction_b, q.prediction_t,
            q.loss_b, q.loss_t, q.d, q.loss_carry])
        for j in range(len(STATS)):
            daily[:, j] += np.bincount(day, weights=values[:, j], minlength=len(dates))
        q['quarter'] = q.bin_start.dt.year.astype(str)+'Q'+((q.bin_start.dt.month-1)//3+1).astype(str)
        for dimension in ['quarter', 'utc_block', 'count_group', 'spread_group', 'timing_group']:
            for group, part in q.groupby(dimension):
                group_parts.append({'dimension': dimension, 'group': str(group), 'n': len(part),
                    'sum_y': part.y300.sum(), 'sum_b': part.prediction_b.sum(), 'sum_t': part.prediction_t.sum(),
                    'sum_lb': part.loss_b.sum(), 'sum_lt': part.loss_t.sum(), 'sum_d': part.d.sum(), 'sum_carry': part.loss_carry.sum()})
        all_rows += len(q)
        print(f'Forecast/training/paired-loss check {c:02d}/59 passed', flush=True)
    stored = pd.read_parquet(run/'daily.parquet')
    require(np.array_equal(ns(stored.date), ns(dates)), 'Missing calendar days')
    close(daily, stored[STATS], 'Independent daily statistics', atol=1e-7)
    patterns = pd.DataFrame(group_parts).groupby(['dimension', 'group'])[STATS].sum().sort_index()
    saved_patterns = pd.read_parquet(run/'patterns.parquet').set_index(['dimension', 'group']).sort_index()
    require(patterns.index.equals(saved_patterns.index), 'Pattern cells differ')
    close(patterns, saved_patterns[STATS], 'Complete fixed patterns', atol=1e-7)
    summary = read(run/'summary.json')
    total_n, sb, st, sd = daily[:, 0].sum(), daily[:, 4].sum(), daily[:, 5].sum(), daily[:, 6].sum()
    delta, relative = sd/total_n, sd/sb
    close([total_n, sb/total_n, st/total_n, delta, relative],
          [summary['n'], summary['mse_b'], summary['mse_t'], summary['delta_pips_squared'], summary['relative_mse_reduction']], 'Primary result')
    influence = (daily[:, 6]-delta*daily[:, 0])/total_n
    ri = (daily[:, 6]-relative*daily[:, 4])/sb
    separation = np.abs(np.arange(len(dates))[:, None]-np.arange(len(dates))[None, :])
    for row in summary['nominal_intervals']:
        lag = row['lag_days']
        weight = np.maximum(1-separation/(lag+1), 0)
        se = np.sqrt(influence@weight@influence)
        rse = np.sqrt(ri@weight@ri)
        z = norm.ppf(.975)
        close([se, delta-z*se, delta+z*se, rse, relative-z*rse, relative+z*rse],
            [row['nominal_se'], row['nominal_low'], row['nominal_high'], row['relative_nominal_se'],
             row['relative_nominal_low'], row['relative_nominal_high']], 'Independent matrix HAC', atol=1e-12)
    require(all_rows == summary['n'], 'Evaluation population differs')
    receipt = {'status': 'passed', 'evaluation_cycles': 55,
        'paired_outcomes_verified': all_rows, 'training_origin_instances_verified': training_rows,
        'saved_model_predictions_verified': prediction_checks, 'calendar_days_verified': len(dates),
        'all_fixed_pattern_cells_verified': len(patterns), 'independent_matrix_hac_lags': [56, 28, 112],
        'protected_files_unchanged': len(read(run/'preserved_scientific_state.json')),
        'statistical_coverage_or_external_validation_certified': False}
    atomic_json(run/'verification.json', receipt)
    m['status'] = 'fixed_comparison_verified'
    atomic_json(run/'manifest.json', m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['features', 'results'])
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require('/' not in args.run_id and args.run_id not in ['.', '..'], 'Safe run identifier')
    run = ROOT/'data/signals'/args.run_id
    (verify_features if args.phase == 'features' else verify_results)(run)


if __name__ == '__main__':
    main()

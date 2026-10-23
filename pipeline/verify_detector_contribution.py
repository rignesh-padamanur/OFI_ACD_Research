"""Independent source-mark and arithmetic checks of the Step 11 artifacts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from scipy.linalg import toeplitz
from scipy.stats import norm
from src.data_preparation import atomic_json, sha256
from src.panel_observations import read_quote_window
from src.event_audit import ns
from src.detector_contribution import RULES, PRIMARY, XCOLS, SECOND

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def verify(run):
    manifest = read(run/'manifest.json')
    panel = Path(manifest['parent_panel']); parent = read(panel/'manifest.json')
    if read(panel/'verification.json')['status'] != 'passed':
        raise ValueError('Parent no longer verified')
    if sha256(panel/'manifest.json') != manifest['parent_panel_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    for name, digest in manifest['source_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(run/'code'/name) != digest:
            raise ValueError('Frozen source mismatch: '+name)
    support = pd.read_parquet(manifest['inputs']['support_intervals']['path']).query("policy == 'structural_only'").reset_index(drop=True)
    starts, ends = ns(support.start), ns(support.end_exclusive)
    dates = pd.date_range(manifest['start'], manifest['end_exclusive'], inclusive='left')
    sums = np.zeros((len(dates), len(RULES))); counts = np.zeros_like(sums)
    tested_marks = 0; reconstructed_rows = 0; fingerprint_count = 0
    extra_fingerprints = {}
    for rel in ['cycle_schedule.parquet', 'observations/seasonal_spells.parquet']:
        p = panel/rel; digest = sha256(p)
        if parent['outputs'][rel] != digest:
            raise ValueError('Parent output changed: '+rel)
        extra_fingerprints[str(p)] = digest
    for out in sorted((run/'cycles').iterdir()):
        cycle = int(out.name); status = read(out/'detector_done.json')
        for source, digest in status['source_fingerprints'].items():
            p = Path(source)
            if sha256(p) != digest or parent['outputs'][str(p.relative_to(panel))] != digest:
                raise ValueError('Changed parent cycle input: '+source)
            fingerprint_count += 1
        for filename, digest in status['outputs'].items():
            if sha256(out/filename) != digest:
                raise ValueError('Changed detector output: '+filename)
        if cycle >= 2:
            p = panel/'cycles'/out.name/'parameters.json'; digest = sha256(p)
            if parent['outputs'][str(p.relative_to(panel))] != digest:
                raise ValueError('Changed ACD/detector parameters')
            extra_fingerprints[str(p)] = digest
        if cycle == 0:
            continue
        estate = read(out/'economic_done.json')
        for filename, digest in estate['outputs'].items():
            if sha256(out/filename) != digest:
                raise ValueError('Changed economic output: '+filename)
        f = pd.read_parquet(out/'ledger.parquet'); e = pd.read_parquet(out/'economics.parquet')
        np.testing.assert_array_equal(ns(f.bin_start), ns(e.bin_start))
        np.testing.assert_allclose(e.adjusted, e.y300-e.prediction, rtol=0, atol=1e-13)
        np.testing.assert_allclose(e.I, e.J+e.C, rtol=0, atol=1e-13)
        mask = np.isfinite(f[XCOLS]).all(axis=1).to_numpy() & np.isfinite(e.y300).to_numpy()
        np.testing.assert_array_equal(e.primary_ready, mask)
        model = joblib.load(out/'reference_model.joblib')
        rng = np.random.default_rng(20261004+cycle)
        ix = np.unique(np.r_[0, len(f)-1, rng.integers(0, len(f), 20),
            *[np.flatnonzero(f['onset_'+name])[:2] for name in RULES]])
        pred_ix = ix[np.isfinite(f.iloc[ix][XCOLS]).all(axis=1)]
        np.testing.assert_allclose(model.predict(f.iloc[pred_ix][XCOLS]), e.prediction.iloc[pred_ix], atol=1e-12, rtol=1e-12)
        cal = read(out/'calibration.json')
        train = pd.read_parquet(out/'training_labels.parquet')
        if not (ns(train.bin_start)+301*SECOND <= pd.Timestamp(cal['cutoff']).value).all():
            raise ValueError('Reference model uses unmatured labels')
        sample = f.iloc[ix]
        # direct searches at the 601 declared marks, without using QuoteGrid or prefix sums.
        quote_start = sample.bin_start.min()-pd.Timedelta(seconds=301)
        quote_stop = sample.bin_start.max()+pd.Timedelta(seconds=302)
        q = read_quote_window(manifest['inputs']['clean_quotes']['path'], quote_start, quote_stop, ['timestamp', 'bid', 'ask'])
        qt = ns(q.timestamp); prices = np.rint(q[['bid', 'ask']].to_numpy()*100000).astype(np.int64)
        for pos in ix:
            row = f.iloc[pos]; origin = pd.Timestamp(row.bin_start).value; sid = int(row.support_id)
            times = origin+np.arange(-299, 302, dtype=np.int64)*SECOND
            qi = np.searchsorted(qt, times, side='left')-1
            good = (qi >= 0) & (qt[np.maximum(qi, 0)] >= starts[sid]) & (times <= ends[sid])
            sp = (prices[np.maximum(qi, 0), 1]-prices[np.maximum(qi, 0), 0])/10
            # s is index 299, detection at 300, the future 300 marks at 301:601.
            if good[300]:
                np.testing.assert_allclose(row.current_spread, sp[300], atol=1e-12)
            if row.reference_ready:
                self_history = np.mean(sp[:300])
                np.testing.assert_allclose(row.history_spread, self_history, atol=1e-12)
            for h in [60, 300]:
                expected = sp[301:301+h].mean() if good[300:301+h].all() else np.nan
                np.testing.assert_allclose(e.loc[pos, f'y{h}'], expected, atol=1e-12, rtol=1e-12, equal_nan=True)
            if good[300:].all():
                mid = prices[np.maximum(qi, 0)].sum(axis=1)
                np.testing.assert_allclose(e.loc[pos, 'absolute_move'], abs(mid[-1]-mid[300])/20, atol=1e-12)
            tested_marks += len(times)
        # complete-data, independently grouped daily residual totals.
        for j, rule in enumerate(RULES):
            selected = mask & f['onset_'+rule].to_numpy()
            part = pd.DataFrame({'date': f.loc[selected, 'bin_start'].dt.normalize(),
                                 'residual': e.loc[selected, 'y300'].to_numpy()-e.loc[selected, 'prediction'].to_numpy()})
            agg = part.groupby('date').residual.agg(['sum', 'count']).reindex(dates, fill_value=0)
            sums[:, j] += agg['sum'].to_numpy(); counts[:, j] += agg['count'].to_numpy()
        reconstructed_rows += len(f)
        print(f'Verified economics cycle {cycle:02d}/59', flush=True)
    saved = pd.read_parquet(run/'daily_sufficient_statistics.parquet')
    for j, rule in enumerate(RULES):
        part = saved.loc[saved.metric.eq('adjusted') & saved.rule.eq(rule)].set_index('date').reindex(dates)
        np.testing.assert_allclose(part['sum'], sums[:, j], atol=1e-9, rtol=1e-11)
        np.testing.assert_array_equal(part['count'], counts[:, j])
    comparisons = pd.read_csv(run/'contrasts.csv')
    mean = sums.sum(axis=0)/counts.sum(axis=0)
    influences = (sums-counts*mean)/counts.sum(axis=0)
    for lag in [28, 56, 112]:
        kernel = np.maximum(0, 1-np.arange(len(dates))/(lag+1))
        covariance = influences.T@toeplitz(kernel)@influences
        for j, rule in enumerate(RULES[1:], 1):
            row = comparisons.loc[comparisons.metric.eq('adjusted') & comparisons.lag_days.eq(lag) & comparisons.benchmark.eq(rule)].iloc[0]
            c = np.zeros(len(RULES)); c[0] = 1; c[j] = -1
            se = np.sqrt(c@covariance@c); diff = mean[0]-mean[j]
            z = norm.ppf(1-.05/(2*3)) if rule in PRIMARY else norm.ppf(.975)
            np.testing.assert_allclose([row.difference_pips, row.se, row.lower, row.upper],
                [diff, se, diff-z*se, diff+z*se], atol=1e-10, rtol=1e-9)
    result = {'status': 'passed', 'source_mark_checks': tested_marks,
        'common_deployment_rows_reconciled': reconstructed_rows,
        'parent_cycle_fingerprints_verified': fingerprint_count,
        'additional_input_fingerprints': extra_fingerprints,
        'all_primary_daily_ratios_and_three_HAC_lags_reproduced': True,
        'prediction_replay': 'fixed samples from every evaluation cycle',
        'outcome_source_checks': 'fixed samples from every evaluation cycle; full synthetic boundary tests',
        'verification_source_sha256': sha256(Path(__file__)),
        'coverage_of_statistical_intervals_validated': False}
    atomic_json(run/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'additional_input_fingerprints'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-id', default='step11_detector_contribution_20261004')
    args = p.parse_args(); verify(ROOT/'data/signals'/args.run_id)

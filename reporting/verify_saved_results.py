"""Reconstruct published arithmetic from bundled saved observations and losses.

This checks saved-result arithmetic, not raw-feed completeness, the fitting
history, causal identification, or confidence-interval coverage. No model is
fitted and no scientific source is rewritten. Outputs go to build/ by default.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Verification must run without Python optimization (-O/-OO or PYTHONOPTIMIZE).")

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence/data/signals'
COMPONENTS = {'J': 'initial_pips', 'C': 'continuation_pips', 'I': 'total_pips'}
CALENDAR = pd.date_range('2024-05-06', '2026-07-31', tz='UTC')
USED = {}


def read(run, name):
    path = DATA / run / name
    USED[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path) if path.suffix == '.parquet' else json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message='', atol=1e-9, rtol=1e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=message)


def total(values):
    return math.fsum(np.asarray(values, dtype=float))


def responses():
    run = 'step07m_joint_response_20261002'
    membership = read(run, 'membership.parquet')
    summary = read(run, 'summary.parquet')
    daily = read(run, 'daily.parquet')
    endpoint = read(run, 'endpoint_counts.parquet')
    chosen = membership.loc[membership.joint_included].copy()
    close(chosen.initial_pips + chosen.continuation_pips, chosen.total_pips, 'J+C=I')
    require(chosen.detected_at.eq(chosen.bin_start + pd.Timedelta(seconds=1)).all(), 'Detection clock')
    require(chosen.target_at.eq(chosen.detected_at + pd.to_timedelta(chosen.horizon_sec, unit='s')).all(), 'Target clock')
    by_horizon = {h: chosen.loc[chosen.horizon_sec.eq(h)] for h in (60, 300)}
    common = set(by_horizon[60].episode_id) & set(by_horizon[300].episode_id)
    require(len(common) == 127013, 'Common event count')
    require(len(by_horizon[60]) == 127052 and len(by_horizon[300]) == 127013, 'Available horizon counts')
    groups = {}
    for horizon in (60, 300):
        for cohort in ('horizon_joint', 'common_horizons_joint'):
            part = by_horizon[horizon]
            if cohort == 'common_horizons_joint':
                part = part.loc[part.episode_id.isin(common)]
            groups[(cohort, horizon)] = part
            for sign in ('all', 'positive', 'zero', 'negative'):
                p = part if sign == 'all' else part.loc[part.initial_sign.eq(sign)]
                for component, column in COMPONENTS.items():
                    row = summary.loc[summary.horizon_sec.eq(horizon) & summary.cohort.eq(cohort)
                        & summary.initial_group.eq(sign) & summary.component.eq(component)]
                    require(len(row) == 1, 'Unique response summary')
                    row = row.iloc[0]
                    require(len(p) == row.n, 'Response denominator')
                    quantiles = np.quantile(p[column], [.25, .5, .75])
                    close([total(p[column]), total(p[column])/len(p), *quantiles],
                          [row['sum'], row['mean'], row.p25, row['median'], row.p75], 'Response summaries')
            p = part.assign(date=part.bin_start.dt.normalize())
            d = daily.loc[daily.horizon_sec.eq(horizon) & daily.cohort.eq(cohort)].set_index('date').reindex(CALENDAR)
            require(len(d) == 817 and d.n.notna().all(), 'Full response calendar')
            counts = p.groupby('date').size().reindex(CALENDAR, fill_value=0)
            close(counts, d.n, 'Daily response counts', atol=0, rtol=0)
            for component, column in COMPONENTS.items():
                sums = p.groupby('date')[column].sum().reindex(CALENDAR, fill_value=0)
                close(sums, d['sum_'+component], 'Daily response sums')
            for row in endpoint.loc[endpoint.horizon_sec.eq(horizon) & endpoint.cohort.eq(cohort)].itertuples():
                subgroup = part.loc[part.initial_sign.eq(row.initial_group)]
                count = int(subgroup.endpoint_pattern.eq(row.endpoint_pattern).sum())
                require(count == row.n and len(part) == row.denominator_joint and len(subgroup) == row.denominator_initial_group,
                        'Endpoint category accounting')
    primary = by_horizon[300].set_index('episode_id').sort_index()
    age_rows = []
    for cap, expected in [(None, 127013), (15,100248), (30,116836), (60,125127)]:
        age = primary if cap is None else primary.loc[primary[['pre_age_sec','detection_age_sec','endpoint_age_sec']].le(cap).all(axis=1)]
        if cap is not None:
            require(primary[f'triplet_age_le_{cap}'].eq(primary.index.isin(age.index)).all(), 'Age flag reconstruction')
        require(len(age) == expected, 'Age-restricted denominator')
        age_rows.append({'quote_age_cap_seconds':cap, 'n':len(age), **{c:total(age[f])/len(age) for c,f in COMPONENTS.items()}})
    return {'status':'passed','units':'direction-aligned pips','summary_rows_checked':len(summary),
            'common_events':len(common),'available_60_events':len(by_horizon[60]),'quote_age_rows':age_rows,
            'primary_continuation_quantiles':dict(zip(['p01','p05','p25','p50','p75','p95','p99'],map(float,np.quantile(primary.continuation_pips,[.01,.05,.25,.5,.75,.95,.99]))))}, groups, primary


def uncertainty(groups):
    run = 'step07n_joint_uncertainty_20261002'
    scores, intervals, covariance, statistics = (read(run, n) for n in ['scores.parquet','intervals.parquet','covariances.parquet','statistics.json'])
    targets = {f'{cohort}:{horizon}:{component}'
               for cohort in ('horizon_joint', 'common_horizons_joint')
               for horizon in (60, 300) for component in COMPONENTS}
    require(set(scores.target) == targets, 'Complete response uncertainty targets')
    require(not scores.duplicated(['target', 'date']).any(), 'Unique response score coordinates')
    require(len(scores) == len(targets)*len(CALENDAR), 'Complete response score calendar')
    expected_intervals = {(target, lag) for target in targets for lag in (20, 40)}
    require(len(intervals) == len(expected_intervals) and
            set(zip(intervals.target, intervals.lag)) == expected_intervals,
            'Complete unique response interval coordinates')
    expected_covariances = {(lag, a, b) for lag in (20, 40) for a in targets for b in targets}
    require(len(covariance) == len(expected_covariances) and
            set(zip(covariance.lag, covariance.target_i, covariance.target_j)) == expected_covariances,
            'Complete unique response covariance coordinates')
    reconstructed = {}
    for target in sorted(targets):
        cohort, horizon, component = target.split(':')
        p = groups[(cohort,int(horizon))]
        counts = p.groupby(p.bin_start.dt.normalize()).size().reindex(CALENDAR,fill_value=0)
        sums = p.groupby(p.bin_start.dt.normalize())[COMPONENTS[component]].sum().reindex(CALENDAR,fill_value=0)
        mean = total(p[COMPONENTS[component]])/len(p)
        score = sums.to_numpy() - counts.to_numpy()*mean
        influence = score/len(p)
        part = scores.loc[scores.target.eq(target)]
        require(pd.DatetimeIndex(part.date).equals(CALENDAR), 'Exact chronological score calendar')
        old = part.set_index('date').reindex(CALENDAR)
        close(old.n, counts, 'Score calendar counts',atol=0,rtol=0)
        close(old.score, score, 'Response score')
        close(old.influence, influence, 'Response influence')
        close(old.energy_share, influence**2/(influence@influence), 'Influence concentration')
        reconstructed[target] = influence
        for row in intervals.loc[intervals.target.eq(target)].itertuples():
            lag=int(row.lag)
            variance = influence@influence + 2*sum((1-j/(lag+1))*(influence[j:]@influence[:-j]) for j in range(1,lag+1))
            se=math.sqrt(variance)
            z=NormalDist().inv_cdf(.975)
            close([row.mean,row.se,row.ci95_low,row.ci95_high],[mean,se,mean-z*se,mean+z*se], 'Nominal interval',atol=1e-12)
    for row in covariance.itertuples():
        u,v=reconstructed[row.target_i],reconstructed[row.target_j]
        lag=int(row.lag)
        cov=u@v+sum((1-j/(lag+1))*(u[j:]@v[:-j]+v[j:]@u[:-j]) for j in range(1,lag+1))
        close(cov,row.value,'Cross covariance',atol=1e-12)
    bandwidths=[]
    for influence in reconstructed.values():
        n=len(influence);pilot=int(np.floor(4*(n/100)**(2/9)))
        gamma=[float(influence@influence/n)]+[float(influence[j:]@influence[:-j]/n) for j in range(1,pilot+1)]
        s0=gamma[0]+2*sum(gamma[1:]);s1=2*sum(j*gamma[j] for j in range(1,pilot+1))
        bandwidths.append(1.1447*(n*(s1/s0)**2)**(1/3))
    require(int(np.floor(max(bandwidths))) == 20, 'Primary lag selection')
    require(set(intervals.lag)=={20,40},'Both declared lags retained')
    return {'status':'passed','intervals_checked':len(intervals),'cross_covariances_checked':len(covariance),
            'primary_lag':20,'alternative_lag':40,'interpretation':'Conditional arithmetic only; coverage and full-pipeline uncertainty are not validated.'}


def forecasts(run, extension, benchmark, primary):
    saved=read(run,'scored_events.parquet')
    metrics=read(run,'metrics.parquet')
    comparisons=read(run,'comparisons.parquet')
    methods=metrics.method.tolist()
    expected_methods = {'zero', 'training_mean', benchmark, extension}
    require(len(methods) == len(expected_methods) and set(methods) == expected_methods,
            'Complete unique forecast methods')
    require(set(saved.method) == expected_methods, 'Saved event methods match metric methods')
    expected_contrasts = {extension+'_minus_'+method for method in expected_methods-{extension}}
    require(len(comparisons) == len(expected_contrasts) and set(comparisons.contrast) == expected_contrasts,
            'Complete unique paired comparisons')
    saved['date']=saved.forecast_at.dt.normalize()
    saved['quarter']=saved.forecast_at.dt.tz_localize(None).dt.to_period('Q').astype(str)
    losses={};summary=[]
    for row in metrics.itertuples():
        p=saved.loc[saved.method.eq(row.method)].copy()
        require(p.episode_id.is_unique,'Unique origins within method')
        issued=np.isfinite(p.predicted_C)
        scored=p.loc[p.scored].set_index('episode_id').sort_index()
        require(len(p)==row.n_origins==133918,'All origins retained')
        require(int(issued.sum())==row.n_issued==127058,'Issued origin count')
        require(len(scored)==row.n_scored==127013,'Scored origin count')
        require(int((issued & ~p.scored).sum())==row.n_issued_unavailable==45,'Unavailable outcomes retained')
        require(scored.index.equals(primary.index),'Common event IDs with response sample')
        close(scored.observed_C,primary.continuation_pips,'Forecast/response outcomes')
        close(scored.initial_pips,primary.initial_pips,'Forecast/response initial displacement')
        require(scored.forecast_at.equals(primary.detected_at.rename('forecast_at')),
                'Forecast time matches response detection time')
        close(scored.predicted_I,scored.predicted_C+scored.initial_pips,'Forecast I=J+C')
        errors=scored.predicted_C-scored.observed_C
        close(errors,scored.forecast_error,'Saved forecast error')
        calculated=errors**2
        close(calculated,scored.squared_error,'Saved squared error')
        close([total(calculated),total(calculated)/len(calculated),math.sqrt(total(calculated)/len(calculated))],
              [row.sse,row.mse,row.rmse],'Pooled forecast metrics')
        losses[row.method]=scored.assign(calculated_loss=calculated)
        summary.append({'method':row.method,'n':len(scored),'mse_pips_squared':total(calculated)/len(calculated)})
    for row in comparisons.itertuples():
        ext,bench=row.contrast.split('_minus_',1)
        diff=losses[ext].calculated_loss-losses[bench].calculated_loss
        require(len(diff)==row.n,'Comparison denominator')
        close([total(diff),total(diff)/len(diff),-total(diff)/total(losses[bench].calculated_loss)],
              [row.sum_loss_difference,row.mean_loss_difference,row.relative_mse_reduction],'Paired pooled difference')
    expected_keys = {
        'quarter': pd.period_range('2024Q2', '2026Q3', freq='Q').astype(str).tolist(),
        'fold_id': list(range(1, 60)),
        'date': CALENDAR.tolist(),
    }
    partition_counts = {}
    for filename,key in [('quarter_losses.parquet','quarter'),('fold_losses.parquet','fold_id'),('daily_losses.parquet','date')]:
        table=read(run,filename)
        require(table[key].is_unique and table[key].tolist() == expected_keys[key],
                'Complete ordered partition keys: '+filename)
        partition_counts[key] = len(table)
        original_parts = dict(tuple(saved.groupby(['method', key], sort=False)))
        loss_parts = {method: dict(tuple(frame.groupby(key, sort=False))) for method, frame in losses.items()}
        for row in table.to_dict('records'):
            for method in methods:
                all_rows=original_parts.get((method, row[key]), saved.iloc[:0])
                p=loss_parts[method].get(row[key], losses[method].iloc[:0])
                require(len(all_rows)==row['n_origins'] and int(np.isfinite(all_rows.predicted_C).sum())==row['n_issued'] and len(p)==row['n_scored'],'Partition counts')
                close(total(p.calculated_loss),row['sse_'+method],'Partition losses')
                if 'mse_'+method in row and len(p):close(total(p.calculated_loss)/len(p),row['mse_'+method],'Partition MSE')
            for contrast in comparisons.contrast:
                ext,bench=contrast.split('_minus_',1)
                e=loss_parts[ext].get(row[key], losses[ext].iloc[:0]).calculated_loss
                b=loss_parts[bench].get(row[key], losses[bench].iloc[:0]).calculated_loss
                delta=total(e-b)
                close(delta,row['sum_delta_'+contrast],'Partition paired difference')
                if 'mean_delta_'+contrast in row and len(e):close(delta/len(e),row['mean_delta_'+contrast],'Partition mean difference')
    daily=read(run,'daily_losses.parquet')
    serial=read(run,'serial_dependence.parquet')
    serial_keys = {(contrast, lag) for contrast in expected_contrasts for lag in range(41)}
    require(len(serial) == len(serial_keys) and set(zip(serial.contrast, serial.lag)) == serial_keys,
            'Complete unique serial loss diagnostics')
    for contrast in comparisons.contrast:
        diff=daily['sum_delta_'+contrast].to_numpy()
        psi=diff-daily.n_scored.to_numpy()*total(diff)/int(daily.n_scored.sum())
        close(psi,daily['psi_'+contrast],'Daily centered loss score')
        for row in serial.loc[serial.contrast.eq(contrast)].itertuples():
            j=int(row.lag); cross=float(psi[j:]@psi[:len(psi)-j])
            require(row.n_pairs==len(psi)-j,'Serial pair denominator')
            close(cross,row.cross_product,'Serial loss cross product')
            close(cross/float(psi@psi),row.normalized_autocorrelation,'Normalized serial loss product')
    difference=total(losses[extension].calculated_loss-losses[benchmark].calculated_loss)
    return {'status':'passed','metrics':summary,'quarter_rows':partition_counts['quarter'],'fold_rows':partition_counts['fold_id'],'calendar_days':partition_counts['date'],
            'primary_contrast':extension+'_minus_'+benchmark,'relative_mse_increase_percent':100*difference/total(losses[benchmark].calculated_loss),
            'limits':'Saved predictions and losses are checked; training maturity, original model fitting and inference validity require the full archive.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/results_verification.json')
    args=parser.parse_args()
    response,groups,primary=responses()
    intervals=uncertainty(groups)
    price=forecasts('step09b_persistence_forecast_20261003','interaction','additive',primary)
    sequence=forecasts('step10b_sequence_forecast_20261003','sequence','baseline',primary)
    from plot_spread_concentration import verify, hashes
    _,spread=verify()
    for name,digest in hashes().items():USED['evidence/data/signals/step03_timing_forecast_20261008/'+name]=digest
    result={'status':'passed','scope':'Independent saved-result arithmetic; no model refits or empirical source mutations',
            'responses':response,'nominal_response_uncertainty':intervals,'price_forecast':price,'sequence_forecast':sequence,
            'spread_forecast':spread,'source_sha256':USED,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','response_events':response['common_events'],'price_relative_mse_increase_percent':price['relative_mse_increase_percent'],
                     'sequence_relative_mse_increase_percent':sequence['relative_mse_increase_percent'],'spread_relative_mse_increase_percent':spread['relative_mse_increase_pct']}))


if __name__=='__main__':main()

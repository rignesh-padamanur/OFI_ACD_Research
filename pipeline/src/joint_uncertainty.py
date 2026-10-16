"""Stage07n: ratio-mean scores and one shared calendar HAC calculation.

No regression fits, p-values, covariance inverse or empirical selection occurs.
The covariance is singular by construction; marginal intervals are provisional.
"""
from math import fsum
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.calendar_score_hac import calendar_hac

COHORTS = ('horizon_joint', 'common_horizons_joint')
HORIZONS = (60, 300)
COMPONENTS = ('J', 'C', 'I')
FRAMES = ('intervals', 'covariances', 'scores', 'lag_products')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def plain(x):
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x


def build_joint_uncertainty(daily, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    require(str(start.tz) == str(end.tz) == 'UTC' and not pd.isna(start) and not pd.isna(end)
            and start == start.normalize() and end == end.normalize() and start < end,
            'Ordered UTC midnight bounds required')
    dates = pd.date_range(start, end, inclusive='left', freq='D')
    fields = ['cohort', 'horizon_sec', 'date', 'n', 'sum_J', 'sum_C', 'sum_I']
    require(set(fields).issubset(daily), 'Missing daily fields')
    require(not daily[fields].isna().any().any()
            and not daily.duplicated(fields[:3]).any(), 'Missing or duplicate daily keys/values')
    require(set(daily.cohort) == set(COHORTS) and set(daily.horizon_sec) == set(HORIZONS)
            and len(daily) == 4*len(dates), 'Exactly four complete daily panels required')
    require(daily.n.dtype.kind in 'iu' and (daily.n >= 0).all(), 'Integer nonnegative counts required')
    require(np.isfinite(daily[['sum_J','sum_C','sum_I']].to_numpy(dtype=float)).all(), 'Finite daily sums required')
    targets, groups, score_columns, count_columns, means, totals = [], {}, [], [], [], []
    for cohort in COHORTS:
        for h in HORIZONS:
            d = daily.loc[daily.cohort.eq(cohort) & daily.horizon_sec.eq(h)].sort_values('date')
            require(pd.DatetimeIndex(d.date).equals(dates), 'Complete UTC calendar required')
            counts = d.n.to_numpy(); sums = d[['sum_J','sum_C','sum_I']].to_numpy(dtype=float)
            require((sums[counts == 0] == 0).all(), 'Empty dates must have zero observed sums')
            require(np.allclose(sums[:,2], sums[:,0]+sums[:,1], rtol=1e-12, atol=1e-9), 'Daily J+C=I differs')
            n = sum(int(x) for x in counts)
            require(n > 0, 'Positive total events required')
            mu = np.array([fsum(sums[:,k])/n for k in range(3)])
            g = sums-counts[:,None]*mu
            require(np.isfinite(g).all() and np.isfinite(mu).all(), 'Nonfinite scores/means')
            groups[f'{cohort}:{h}'] = {'n': n, 'occupied_days': int((counts>0).sum()),
                'zero_event_days': int((counts==0).sum()), 'mean': dict(zip(COMPONENTS, mu.tolist()))}
            for k, component in enumerate(COMPONENTS):
                targets.append(f'{cohort}:{h}:{component}')
                score_columns.append(g[:,k]); count_columns.append(counts)
                means.append(mu[k]); totals.append(n)
    g = np.column_stack(score_columns); counts = np.column_stack(count_columns)
    n = np.array(totals, dtype=float); mu = np.array(means)
    hac = calendar_hac(g, np.diag(1/n), np.ones(len(targets)))
    intervals, covariances, covariance_diagnostics = [], [], {}
    z95 = float(norm.ppf(.975))
    for method in ['automatic','longer']:
        c = hac[method]['covariance']; lag = hac[method]['lag']
        require((np.diag(c)>0).all(), 'Positive marginal variances required')
        for i in range(0,12,3):
            require(np.allclose(c[i+2],c[i]+c[i+1],rtol=1e-10,atol=1e-12), 'Covariance J+C=I differs')
        for i in range(12):
            for j in range(i):
                if np.array_equal(g[:,i]/n[i], g[:,j]/n[j]):
                    require(np.allclose(c[i],c[j],rtol=1e-12,atol=1e-14), 'Identical influences have different covariances')
        covariance_diagnostics[method] = {k: hac[method][k] for k in
            ['lag','meat_min_eigenvalue','meat_psd_tolerance','covariance_min_eigenvalue','covariance_psd_tolerance']}
        for i, target in enumerate(targets):
            cohort, h, component = target.split(':'); se = float(np.sqrt(c[i,i]))
            intervals.append({'target':target,'cohort':cohort,'horizon_sec':int(h),'component':component,
                'n':totals[i],'method':method,'lag':lag,'mean':mu[i],'se':se,
                'ci95_low':mu[i]-z95*se,'ci95_high':mu[i]+z95*se})
            for j, other in enumerate(targets):
                covariances.append({'method':method,'lag':lag,'target_i':target,'target_j':other,'value':c[i,j]})
    influence = g/n; energy = np.sum(influence**2,axis=0)
    require(np.isfinite(energy).all() and (energy>0).all(), 'Positive finite daily influence energy required')
    share = influence**2/energy
    score_frames = [pd.DataFrame({'target':target,'date':dates,'n':counts[:,i],
        'score':g[:,i],'influence':influence[:,i],'energy_share':share[:,i]}) for i,target in enumerate(targets)]
    maxlag = min(len(dates)-1,max(70,hac['longer']['lag']))
    denominator = np.sqrt(energy)[:,None]*np.sqrt(energy)[None,:]; products = []
    for lag in range(maxlag+1):
        products.append((influence[lag:].T@influence[:len(dates)-lag])/denominator)
    products = np.asarray(products)
    require(np.isfinite(products).all(), 'Finite normalized lag products required')
    lag_frames = [pd.DataFrame({'lag':lag,'target_i':np.repeat(targets,12),'target_j':np.tile(targets,12),
        'normalized_product':products[lag].ravel()}) for lag in range(maxlag+1)]
    tails = {}
    for method in ['automatic','longer']:
        first = hac[method]['lag']+1
        if first > maxlag: tails[method] = None
        else:
            a,i,j = np.unravel_index(np.argmax(abs(products[first:])),products[first:].shape)
            tails[method] = {'lag':int(first+a),'target_i':targets[i],'target_j':targets[j],
                            'max_abs_product':float(abs(products[first+a,i,j]))}
    summaries = []
    for i,target in enumerate(targets):
        maxima = {}
        for method in ['automatic','longer']:
            first = hac[method]['lag']+1
            maxima[method] = float(np.max(abs(products[first:,i,i]))) if first <= maxlag else None
        summaries.append({'target':target,'total_energy':float(energy[i]),
            'maximum_day':dates[int(np.argmax(share[:,i]))].isoformat(),
            'maximum_day_energy_share':float(share[:,i].max()),'max_abs_lag_product_beyond':maxima})
    stats = {'calendar_days':len(dates),'target_order':targets,'groups':groups,'mean_intervals':plain(intervals),
        'bandwidth':plain(hac['bandwidth']),'automatic_lag':hac['automatic']['lag'],
        'longer_lag':hac['longer']['lag'],'lag_inspection_max':maxlag,
        'covariance_diagnostics':plain(covariance_diagnostics),
        'score_lag_tail_summary':tails,'influence_summary':summaries,
        'empirical_models_fitted':0,'uncertainty_computed':True,
        'interval_type':'pointwise_nominal_95_percent_normal',
        'inference_approved':False,'confidence_interval_coverage_validated':False,
        'score_process_assumptions_verified':False,'covariance_inverted':False,
        'p_values_computed':False,'new_hypothesis_tests':0}
    return {'intervals':pd.DataFrame(intervals),'covariances':pd.DataFrame(covariances),
        'scores':pd.concat(score_frames,ignore_index=True),'lag_products':pd.concat(lag_frames,ignore_index=True),
        'statistics':stats}

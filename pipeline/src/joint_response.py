"""Stage 07m: descriptive joint endpoints, without fitting or inference.

J is the detection-time initial movement, C the subsequent movement, and I
the total movement, all in inherited direction-aligned pips. Endpoint labels
describe only the recorded endpoints, never the intervening path or a cause.
"""
import math

import numpy as np
import pandas as pd

HORIZONS = (60, 300)
COHORTS = ('horizon_joint', 'common_horizons_joint')
INITIAL_GROUPS = ('positive', 'zero', 'negative')
PATTERNS = ('unchanged', 'extended', 'retraced', 'at_pre_onset',
            'beyond_pre_onset', 'initial_zero_positive',
            'initial_zero_negative', 'initial_zero_flat')
COMPONENTS = {'J': 'initial_pips', 'C': 'continuation_pips', 'I': 'total_pips'}
SUMMARY_COLUMNS = ['horizon_sec', 'cohort', 'initial_group', 'component',
                   'n', 'sum', 'mean', 'median', 'p25', 'p75']


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _utc(values, label):
    _require(isinstance(values.dtype, pd.DatetimeTZDtype)
             and str(values.dt.tz) == 'UTC' and values.notna().all(),
             'Nonmissing UTC timestamps required: ' + label)


def _validate(samples, start, end):
    masks = ['regressor_eligible'] + [prefix + component for prefix in
             ('eligible_', 'estimation_eligible_', 'main_eligible_')
             for component in ('initial', 'continuation', 'total', 'joint')]
    fields = ['episode_id', 'horizon_sec', 'bin_start', 'detected_at', 'target_at',
              'analysis_segment', *COMPONENTS.values(), *masks]
    _require(set(fields).issubset(samples.columns), 'Missing joint-response input fields')
    _require(samples.columns.is_unique, 'Duplicate input column names')
    for field in ('episode_id', 'horizon_sec'):
        x = samples[field]
        _require(pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
                 and np.isfinite(x).all() and x.ge(0).all() and x.eq(np.floor(x)).all(),
                 'Nonnegative integral input required: ' + field)
    _require(not samples.duplicated(['episode_id', 'horizon_sec']).any(), 'Duplicate event/horizon')
    _require(samples.horizon_sec.isin([0, *HORIZONS]).all(), 'Unexpected horizon')
    counts = samples.loc[samples.horizon_sec.isin(HORIZONS)].groupby('episode_id').size()
    _require(len(counts) == samples.episode_id.nunique() and counts.eq(2).all(),
             'Each event requires one 60-second and one 300-second record')
    for field in masks:
        _require(pd.api.types.is_bool_dtype(samples[field]) and samples[field].notna().all(),
                 'Nonmissing Boolean required: ' + field)
    for field in ('bin_start', 'detected_at', 'target_at'):
        _utc(samples[field], field)
    _require(samples.bin_start.lt(end).all(), 'Onset outside fixed end boundary')
    _require(samples.analysis_segment.eq(np.where(samples.bin_start.lt(start),
             'reused_development', 'later_reanalysis')).all(), 'Chronology/segment mismatch')
    _require(samples.detected_at.eq(samples.bin_start + pd.Timedelta(seconds=1)).all(),
             'Detection time differs from completed onset-bin boundary')
    _require(samples.target_at.eq(samples.detected_at
             + pd.to_timedelta(samples.horizon_sec, unit='s')).all(), 'Target timing mismatch')
    grouped = samples.groupby('episode_id', sort=False)
    for field in ('bin_start', 'detected_at', 'analysis_segment', 'regressor_eligible', 'initial_pips'):
        _require(grouped[field].nunique(dropna=False).le(1).all(),
                 'Event field differs across horizons: ' + field)
    for component, field in COMPONENTS.items():
        x = samples[field]
        _require(pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x),
                 'Numeric response required: ' + field)
        name = {'J': 'initial', 'C': 'continuation', 'I': 'total'}[component]
        _require(np.isfinite(x.loc[samples['eligible_' + name]]).all(),
                 'Nonfinite main or other eligible response: ' + field)
    for prefix in ('eligible_', 'estimation_eligible_', 'main_eligible_'):
        expected = samples[prefix + 'initial'] & samples[prefix + 'continuation'] & samples[prefix + 'total']
        _require(samples[prefix + 'joint'].eq(expected).all(), 'Joint/component mask mismatch: ' + prefix)
    for component in ('initial', 'continuation', 'total', 'joint'):
        estimated = samples['eligible_' + component] & samples.regressor_eligible
        _require(samples['estimation_eligible_' + component].eq(estimated).all(),
                 'Estimation eligibility mismatch: ' + component)
        _require(samples['main_eligible_' + component].eq(estimated
                 & samples.analysis_segment.eq('later_reanalysis')).all(),
                 'Main eligibility mismatch: ' + component)
    selected = samples.loc[samples.main_eligible_joint]
    _require(np.allclose(selected.initial_pips + selected.continuation_pips,
                        selected.total_pips, atol=1e-9, rtol=1e-12), 'J+C=I identity failed')


def _classify(joint):
    j, c, i = (joint[field].to_numpy(dtype=float) for field in COMPONENTS.values())
    sj, sc, si = np.sign(j), np.sign(c), np.sign(i)
    nonzero = j != 0
    rules = [nonzero & (c == 0) & (i == j),
             nonzero & (sc == sj) & (si == sj) & (abs(i) > abs(j)),
             nonzero & (sc == -sj) & (si == sj) & (abs(i) < abs(j)),
             nonzero & (sc == -sj) & (i == 0),
             nonzero & (sc == -sj) & (si == -sj),
             (j == 0) & (c == i) & (i > 0),
             (j == 0) & (c == i) & (i < 0),
             (j == 0) & (c == i) & (i == 0)]
    _require((np.sum(rules, axis=0) == 1).all(),
             'Endpoint signs/order inconsistent with exact endpoint category')
    return np.select(rules, PATTERNS, default='invalid')


def _summarize(values):
    x = values.to_numpy(dtype=float)
    if not len(x):
        return dict(n=0, sum=np.nan, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan)
    try:
        total = math.fsum(x)
    except OverflowError as exc:
        raise ValueError('Nonfinite response sum') from exc
    _require(math.isfinite(total), 'Nonfinite response sum')
    q = np.quantile(x, [.25, .5, .75])
    return dict(n=len(x), sum=total, mean=total / len(x), median=float(q[1]),
                p25=float(q[0]), p75=float(q[2]))


def _share(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def build_joint_response(samples, start='2024-05-06T00:00:00+00:00',
                         end='2026-08-01T00:00:00+00:00'):
    """Preserve main-C rows, describe joint endpoints and their common cohort.

    No sample screen is introduced. Missing J/I are explicit attrition from the
    inherited main-C sample. Exact stored zeros/signs define endpoint classes;
    additive tolerance checks arithmetic, never category membership.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _require(str(start.tz) == str(end.tz) == 'UTC' and start < end
             and start == start.normalize() and end == end.normalize(),
             'Ordered UTC midnight calendar boundaries required')
    _validate(samples, start, end)
    added = ['joint_included', 'initial_sign', 'endpoint_pattern']
    _require(not set(added) & set(samples.columns), 'Output fields already present in source')
    membership = samples.loc[samples.horizon_sec.isin(HORIZONS)
                             & samples.main_eligible_continuation].copy()
    membership = membership.sort_values(['horizon_sec', 'episode_id']).reset_index(drop=True)
    membership['joint_included'] = membership.main_eligible_joint
    j = membership.initial_pips
    membership['initial_sign'] = np.select([np.isfinite(j) & j.gt(0), j.eq(0), np.isfinite(j) & j.lt(0)],
                                           INITIAL_GROUPS, default='missing')
    membership['endpoint_pattern'] = 'unavailable'
    chosen = membership.joint_included
    membership.loc[chosen, 'endpoint_pattern'] = _classify(membership.loc[chosen])
    by_horizon = {h: membership.loc[membership.horizon_sec.eq(h) & chosen].copy() for h in HORIZONS}
    sets = {h: set(by_horizon[h].episode_id) for h in HORIZONS}
    common = sets[60] & sets[300]
    calendar = pd.date_range(start, end, inclusive='left', freq='D')
    summary, endpoint, daily, cohort_stats = [], [], [], {}
    for horizon in HORIZONS:
        for cohort in COHORTS:
            part = by_horizon[horizon]
            if cohort == 'common_horizons_joint':
                part = part.loc[part.episode_id.isin(common)]
            key = f'{cohort}_{horizon}'
            cohort_stats[key] = {'n': len(part), 'mean': {}, 'initial_sign_counts': {},
                                 'endpoint_pattern_counts': {p: int(part.endpoint_pattern.eq(p).sum()) for p in PATTERNS}}
            for group in ('all', *INITIAL_GROUPS):
                subset = part if group == 'all' else part.loc[part.initial_sign.eq(group)]
                if group != 'all':
                    cohort_stats[key]['initial_sign_counts'][group] = len(subset)
                for component, field in COMPONENTS.items():
                    stats = _summarize(subset[field])
                    summary.append(dict(horizon_sec=horizon, cohort=cohort, initial_group=group,
                                        component=component, **stats))
                    if group == 'all':
                        cohort_stats[key]['mean'][component] = stats['mean'] if len(subset) else None
                if group != 'all':
                    for pattern in PATTERNS:
                        n = int(subset.endpoint_pattern.eq(pattern).sum())
                        endpoint.append(dict(horizon_sec=horizon, cohort=cohort,
                            initial_group=group, endpoint_pattern=pattern, n=n,
                            denominator_joint=len(part), denominator_initial_group=len(subset),
                            share_joint=_share(n, len(part)), share_initial_group=_share(n, len(subset))))
            dates = part.bin_start.dt.normalize()
            groups = dict(tuple(part.groupby(dates, sort=False)))
            for date in calendar:
                day = groups.get(date, part.iloc[:0])
                row = dict(horizon_sec=horizon, cohort=cohort, date=date, n=len(day))
                for component, field in COMPONENTS.items():
                    stats = _summarize(day[field])
                    row['sum_' + component] = stats['sum'] if len(day) else 0.
                    row['mean_' + component] = stats['mean']
                row.update({'n_' + p: int(day.endpoint_pattern.eq(p).sum()) for p in PATTERNS})
                daily.append(row)
    from_labels = by_horizon[60].set_index('episode_id').loc[sorted(common), 'endpoint_pattern']
    to_labels = by_horizon[300].set_index('episode_id').loc[sorted(common), 'endpoint_pattern']
    transitions = []
    for origin in PATTERNS:
        denominator_from = int(from_labels.eq(origin).sum())
        for destination in PATTERNS:
            n = int((from_labels.eq(origin) & to_labels.eq(destination)).sum())
            transitions.append(dict(from_pattern=origin, to_pattern=destination, n=n,
                denominator_common=len(common), denominator_from=denominator_from,
                share_common=_share(n, len(common)), share_from=_share(n, denominator_from)))
    joint = membership.loc[chosen]
    residual = abs(joint.total_pips - (joint.initial_pips + joint.continuation_pips))
    statistics = dict(status='descriptive_joint_response_complete',
        start=start.isoformat(), end=end.isoformat(), calendar_days=len(calendar),
        by_horizon={str(h): {'main_continuation_rows': int(membership.horizon_sec.eq(h).sum()),
            'joint_rows': len(by_horizon[h]),
            'missing_joint_rows': int(membership.horizon_sec.eq(h).sum()) - len(by_horizon[h])} for h in HORIZONS},
        common_joint_events=len(common),
        set_relation={'only_60': len(sets[60] - sets[300]), 'only_300': len(sets[300] - sets[60]),
                      'equal': sets[60] == sets[300], 'joint_300_subset_60': sets[300] <= sets[60]},
        cohorts=cohort_stats, identity_max_abs=float(residual.max()) if len(joint) else None,
        descriptive_only=True, empirical_models_fitted=0,
        uncertainty_computed=False, inference_approved=False)
    return {'membership': membership, 'summary': pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
            'endpoint_counts': pd.DataFrame(endpoint), 'transition_counts': pd.DataFrame(transitions),
            'daily': pd.DataFrame(daily), 'statistics': statistics}

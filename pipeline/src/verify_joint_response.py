"""Independent endpoint accounting for the inherited J/C/I observations.

No production joint-response helper is imported.  The reference uses pandas
grouping/crosstabs, exact stored signs and independently checked source masks.
An endpoint description is not a path, causal or permanence classification.
"""
import math

import numpy as np
import pandas as pd


START = '2024-05-06T00:00:00+00:00'
STOP = '2026-08-01T00:00:00+00:00'
PATTERNS = ('unchanged', 'extended', 'retraced', 'at_pre_onset',
            'beyond_pre_onset', 'initial_zero_positive',
            'initial_zero_negative', 'initial_zero_flat')
GROUPS = ('positive', 'zero', 'negative')
COHORTS = ('horizon_joint', 'common_horizons_joint')
COMPONENTS = {'J': 'initial_pips', 'C': 'continuation_pips', 'I': 'total_pips'}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _source(samples, start, end):
    required = ['episode_id', 'bin_start', 'detected_at', 'target_at',
                'horizon_sec', 'analysis_segment', 'regressor_eligible',
                *COMPONENTS.values()]
    required += [prefix + name for prefix in ('eligible_', 'estimation_eligible_',
                 'main_eligible_') for name in ('initial', 'continuation', 'total', 'joint')]
    _require(set(required).issubset(samples), 'Missing inherited sample fields')
    _require(samples.columns.is_unique and not {'joint_included', 'initial_sign',
             'endpoint_pattern'} & set(samples), 'Duplicate or overwritten source fields')
    _require(not samples[['episode_id', 'horizon_sec']].isna().any().any()
             and not samples.duplicated(['episode_id', 'horizon_sec']).any(),
             'Duplicate or missing source keys')
    for field in ['episode_id', 'horizon_sec']:
        values = samples[field]
        _require(pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
                 and np.isfinite(values).all() and values.ge(0).all()
                 and values.eq(np.floor(values)).all(), 'Nonnegative integral keys required')
    _require(samples.horizon_sec.isin([0, 60, 300]).all(), 'Unexpected horizon')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for name, series in [('start/end', [start, end]), *[(x, samples[x]) for x in
                          ['bin_start', 'detected_at', 'target_at']]]:
        times = pd.DatetimeIndex(series)
        _require(str(times.tz) == 'UTC' and not times.hasnans, 'Nonmissing UTC clocks: ' + name)
    _require(start < end and start == start.normalize() and end == end.normalize(),
             'Positive whole-day interval required')
    _require(samples.bin_start.lt(end).all() and samples.analysis_segment.eq(
             np.where(samples.bin_start.lt(start), 'reused_development', 'later_reanalysis')).all(),
             'Changed inherited chronology or segment')
    _require(samples.detected_at.eq(samples.bin_start + pd.Timedelta(seconds=1)).all(),
             'Changed detection boundary')
    _require(samples.target_at.eq(samples.detected_at +
             pd.to_timedelta(samples.horizon_sec, unit='s')).all(), 'Changed target boundary')
    for name in ['regressor_eligible', *[x for x in required if x.startswith(
            ('eligible_', 'estimation_eligible_', 'main_eligible_'))]]:
        _require(pd.api.types.is_bool_dtype(samples[name]) and not samples[name].isna().any(),
                 'Nonmissing boolean masks required: ' + name)
    for component in ['initial', 'continuation', 'total', 'joint']:
        expected = samples['eligible_' + component] & samples.regressor_eligible
        _require(samples['estimation_eligible_' + component].equals(expected),
                 'Changed estimation mask: ' + component)
        expected = expected & samples.analysis_segment.eq('later_reanalysis')
        _require(samples['main_eligible_' + component].equals(expected),
                 'Changed main mask: ' + component)
    _require(samples.eligible_joint.eq(samples.eligible_initial & samples.eligible_continuation
             & samples.eligible_total).all(), 'Joint eligibility is not component intersection')
    for field in COMPONENTS.values():
        _require(pd.api.types.is_numeric_dtype(samples[field]) and
                 not pd.api.types.is_bool_dtype(samples[field]), 'Numeric outcomes required')
    for component, field in zip(['initial', 'continuation', 'total'], COMPONENTS.values()):
        _require(np.isfinite(samples.loc[samples['eligible_' + component], field]).all(),
                 'Eligible component is not finite')
    counts = pd.crosstab(samples.episode_id, samples.horizon_sec)
    _require({60, 300}.issubset(counts.columns) and counts[[60, 300]].eq(1).all().all(),
             'Each event requires both future horizons')
    for field in ['bin_start', 'detected_at', 'initial_pips', 'analysis_segment', 'regressor_eligible']:
        _require(samples.groupby('episode_id')[field].nunique(dropna=False).le(1).all(),
                 'Horizon-invariant source field changed: ' + field)
    selected = samples.loc[samples.horizon_sec.isin([60, 300]) &
                           samples.main_eligible_continuation].copy()
    _require(selected.bin_start.ge(start).all() and selected.bin_start.lt(end).all(),
             'Main sample outside frozen calendar')
    selected = selected.sort_values(['horizon_sec', 'episode_id']).reset_index(drop=True)
    selected['joint_included'] = selected.main_eligible_joint
    j = selected.initial_pips
    finite = np.isfinite(j)
    selected['initial_sign'] = np.select([finite & j.gt(0), finite & j.eq(0), finite & j.lt(0)],
                                         GROUPS, default='missing')
    selected['endpoint_pattern'] = 'unavailable'
    joint = selected.loc[selected.joint_included]
    j, c, i = [joint[name].to_numpy(dtype=float) for name in COMPONENTS.values()]
    np.testing.assert_allclose(i, j + c, rtol=1e-12, atol=1e-9,
                               err_msg='Inherited J+C=I identity')
    same_c = np.sign(j) == np.sign(c)
    opposite_c = np.sign(j) == -np.sign(c)
    same_i = np.sign(j) == np.sign(i)
    opposite_i = np.sign(j) == -np.sign(i)
    nonzero = j != 0
    conditions = [
        nonzero & (c == 0) & (i == j),
        nonzero & same_c & (np.abs(i) > np.abs(j)) & same_i,
        nonzero & opposite_c & same_i & (np.abs(i) > 0) & (np.abs(i) < np.abs(j)),
        nonzero & opposite_c & (i == 0),
        nonzero & opposite_c & opposite_i & (i != 0),
        (j == 0) & (c == i) & (i > 0),
        (j == 0) & (c == i) & (i < 0),
        (j == 0) & (c == i) & (i == 0),
    ]
    _require(np.sum(conditions, axis=0).tolist() == [1] * len(joint),
             'Endpoint signs or strict magnitude ordering are inconsistent')
    selected.loc[joint.index, 'endpoint_pattern'] = np.select(conditions, PATTERNS, default='invalid')
    return selected, pd.date_range(start, end, freq='D', inclusive='left')


def _tables(membership, dates):
    joints = {h: membership.loc[membership.horizon_sec.eq(h) & membership.joint_included].copy()
              for h in [60, 300]}
    common = set(joints[60].episode_id) & set(joints[300].episode_id)
    summary, endpoint, daily = [], [], []
    for h in [60, 300]:
        for cohort in COHORTS:
            frame = joints[h] if cohort == 'horizon_joint' else joints[h].loc[joints[h].episode_id.isin(common)]
            for group in ['all', *GROUPS]:
                subset = frame if group == 'all' else frame.loc[frame.initial_sign.eq(group)]
                for component, field in COMPONENTS.items():
                    values = subset[field].astype(float)
                    n = len(values)
                    summary.append({'horizon_sec': h, 'cohort': cohort, 'initial_group': group,
                        'component': component, 'n': n,
                        'sum': math.fsum(values) if n else np.nan,
                        'mean': math.fsum(values) / n if n else np.nan,
                        'median': float(values.median()) if n else np.nan,
                        'p25': float(values.quantile(.25)) if n else np.nan,
                        'p75': float(values.quantile(.75)) if n else np.nan})
            cross = pd.crosstab(frame.initial_sign, frame.endpoint_pattern).reindex(
                index=GROUPS, columns=PATTERNS, fill_value=0)
            for group in GROUPS:
                group_n = int(frame.initial_sign.eq(group).sum())
                for pattern in PATTERNS:
                    n = int(cross.loc[group, pattern])
                    endpoint.append({'horizon_sec': h, 'cohort': cohort, 'initial_group': group,
                        'endpoint_pattern': pattern, 'n': n, 'denominator_joint': len(frame),
                        'denominator_initial_group': group_n,
                        'share_joint': n / len(frame) if len(frame) else np.nan,
                        'share_initial_group': n / group_n if group_n else np.nan})
            frame = frame.assign(date=frame.bin_start.dt.normalize())
            grouped = frame.groupby('date')
            daily_frame = pd.DataFrame({'date': dates})
            daily_frame['horizon_sec'] = h
            daily_frame['cohort'] = cohort
            daily_frame['n'] = grouped.size().reindex(dates, fill_value=0).to_numpy()
            for prefix in ['sum', 'mean']:
                for component, field in COMPONENTS.items():
                    values = grouped[field].agg(lambda x: math.fsum(x) /
                        (len(x) if prefix == 'mean' else 1)).reindex(dates)
                    if prefix == 'sum':
                        values = values.fillna(0)
                    daily_frame[prefix + '_' + component] = values.to_numpy(dtype=float)
            table = (pd.crosstab(frame.date, frame.endpoint_pattern).reindex(
                index=dates, columns=PATTERNS, fill_value=0) if len(frame)
                else pd.DataFrame(0, index=dates, columns=PATTERNS))
            for pattern in PATTERNS:
                daily_frame['n_' + pattern] = table[pattern].to_numpy(dtype='int64')
            daily.append(daily_frame[['horizon_sec', 'cohort', 'date', 'n',
                                     'sum_J', 'sum_C', 'sum_I', 'mean_J', 'mean_C', 'mean_I',
                                     *['n_' + p for p in PATTERNS]]])
    paired = joints[60][['episode_id', 'endpoint_pattern']].merge(
        joints[300][['episode_id', 'endpoint_pattern']], on='episode_id',
        validate='one_to_one', suffixes=('_60', '_300'))
    cross = pd.crosstab(paired.endpoint_pattern_60, paired.endpoint_pattern_300).reindex(
        index=PATTERNS, columns=PATTERNS, fill_value=0)
    transitions = []
    for origin in PATTERNS:
        origin_n = int(cross.loc[origin].sum())
        for destination in PATTERNS:
            n = int(cross.loc[origin, destination])
            transitions.append({'from_pattern': origin, 'to_pattern': destination, 'n': n,
                'denominator_common': len(common), 'denominator_from': origin_n,
                'share_common': n / len(common) if common else np.nan,
                'share_from': n / origin_n if origin_n else np.nan})
    return {'membership': membership, 'summary': pd.DataFrame(summary),
            'endpoint_counts': pd.DataFrame(endpoint),
            'transition_counts': pd.DataFrame(transitions),
            'daily': pd.concat(daily, ignore_index=True)}


def _compare_frame(actual, expected, keys, name, exact=False):
    _require(set(actual.columns) == set(expected.columns), name + ' schema mismatch')
    _require(not actual[keys].isna().any().any() and not actual.duplicated(keys).any(),
             name + ' duplicate or missing keys')
    actual = actual[expected.columns].sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    for column in expected:
        exact_column = exact or not pd.api.types.is_float_dtype(expected[column])
        pd.testing.assert_series_equal(actual[column], expected[column], check_dtype=exact,
            check_exact=exact_column, rtol=1e-12, atol=1e-9, obj=name + '.' + column)


def _statistics(tables, dates, start, end):
    member = tables['membership']
    joint = member.loc[member.joint_included]
    ids = {h: set(joint.loc[joint.horizon_sec.eq(h), 'episode_id']) for h in [60, 300]}
    cohort_stats = {}
    for h in [60, 300]:
        for cohort in COHORTS:
            frame = joint.loc[joint.horizon_sec.eq(h)]
            if cohort == 'common_horizons_joint':
                frame = frame.loc[frame.episode_id.isin(ids[60] & ids[300])]
            cohort_stats[f'{cohort}_{h}'] = {'n': len(frame),
                'mean': {component: math.fsum(frame[field]) / len(frame) if len(frame) else None
                         for component, field in COMPONENTS.items()},
                'initial_sign_counts': frame.initial_sign.value_counts().reindex(
                    GROUPS, fill_value=0).astype(int).to_dict(),
                'endpoint_pattern_counts': frame.endpoint_pattern.value_counts().reindex(
                    PATTERNS, fill_value=0).astype(int).to_dict()}
    return {'status': 'descriptive_joint_response_complete',
        'start': pd.Timestamp(start).isoformat(), 'end': pd.Timestamp(end).isoformat(),
        'calendar_days': len(dates),
        'by_horizon': {str(h): {'main_continuation_rows': int(member.horizon_sec.eq(h).sum()),
            'joint_rows': len(ids[h]),
            'missing_joint_rows': int(member.horizon_sec.eq(h).sum()) - len(ids[h])} for h in [60, 300]},
        'common_joint_events': len(ids[60] & ids[300]),
        'set_relation': {'only_60': len(ids[60] - ids[300]), 'only_300': len(ids[300] - ids[60]),
                         'equal': ids[60] == ids[300], 'joint_300_subset_60': ids[300] <= ids[60]},
        'cohorts': cohort_stats,
        'identity_max_abs': float((joint.total_pips - (joint.initial_pips + joint.continuation_pips)).abs().max())
                            if len(joint) else None,
        'descriptive_only': True, 'empirical_models_fitted': 0,
        'uncertainty_computed': False, 'inference_approved': False}


def _compare_statistics(actual, expected, path='statistics'):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), path + ' schema mismatch')
        for key, value in expected.items():
            _compare_statistics(actual[key], value, path + '.' + key)
    elif isinstance(expected, (bool, str)) or expected is None:
        _require(type(actual) is type(expected) and actual == expected, path + ' differs')
    elif isinstance(expected, (int, np.integer)):
        _require(isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_))
                 and actual == expected, path + ' differs')
    else:
        _require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, (bool, np.bool_))
                 and np.isfinite(actual), path + ' is not a finite number')
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9, err_msg=path)


def reference_joint_response(samples, start=START, end=STOP):
    member, dates = _source(samples, start, end)
    tables = _tables(member, dates)
    tables['statistics'] = _statistics(tables, dates, start, end)
    return tables


def verify_joint_response(actual, samples, start=START, end=STOP):
    expected = reference_joint_response(samples, start, end)
    _require(set(actual) == set(expected), 'Joint-response output keys changed')
    keys = {'membership': ['horizon_sec', 'episode_id'],
            'summary': ['horizon_sec', 'cohort', 'initial_group', 'component'],
            'endpoint_counts': ['horizon_sec', 'cohort', 'initial_group', 'endpoint_pattern'],
            'transition_counts': ['from_pattern', 'to_pattern'],
            'daily': ['horizon_sec', 'cohort', 'date']}
    for name in keys:
        frame = expected[name]
        _compare_frame(actual[name], frame, keys[name], name, exact=name == 'membership')
    _compare_statistics(actual['statistics'], expected['statistics'])
    return {'status': 'passed', 'independent_method': 'source masks, exact signs, grouped summaries and crosstabs',
            'table_rows_verified': {name: len(expected[name]) for name in keys},
            'statistics_verified': True, 'empirical_models_fitted': 0}

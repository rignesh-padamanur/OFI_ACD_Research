"""Independent continuous-panel response reconstruction and bounded summaries.

This module imports no production response implementation. The previously
verified strict-backward quote reference supplies all 301 horizons, while a
separate strict-forward join uses the global detector panel for later-event
warnings across computational partitions. Summaries use explicit row sets.
"""
from collections import Counter

import numpy as np
import pandas as pd

from src.verify_price_responses import (
    _ns, _cohort_indices, _summary_for_rows, reference_response_tables,
)

SECOND = 1_000_000_000
STATUSES = ('available', 'beyond_sample_end', 'outside_onset_support',
            'no_prior_quote', 'prior_quote_outside_support', 'stale_under_policy')
SAMPLES = ('horizon_confirmed', 'common_path_confirmed', 'horizon_state', 'common_path_state',
           'horizon_age_15', 'common_path_age_15', 'horizon_age_30', 'common_path_age_30',
           'horizon_age_60', 'common_path_age_60')
MEASURES = ('initial_pips', 'continuation_pips', 'total_pips')
SUMMARY_COLUMNS = ['horizon_sec', 'sample', 'observations', 'mean_endpoint_age_sec',
    'p95_endpoint_age_sec', 'reused_detection_quote', 'later_observed_episode',
    *[prefix + name for name in MEASURES for prefix in ('mean_', 'median_')]]
COMPONENT_COLUMNS = ['horizon_sec', 'cohort', 'component', 'cohort_onsets',
    'available_onsets', 'missing_onsets', *['mean_' + name for name in MEASURES]]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _onset_clock(frame, stop):
    start, detected = _ns(frame.bin_start), _ns(frame.available_at)
    ids = frame.episode_id
    require(pd.api.types.is_integer_dtype(ids.dtype) and not pd.api.types.is_bool_dtype(ids.dtype)
            and ids.notna().all() and ids.ge(0).all() and not ids.duplicated().any(),
            'Unique nonnegative onset identities required')
    require(not (np.diff(start) <= 0).any() and not (start % SECOND).any()
            and np.array_equal(detected, start + SECOND), 'Chronological whole-second onset clock required')
    require(not (detected > stop).any(), 'Global detections extend beyond sample end')


def reference_panel_responses(onsets, quotes, support, *, sample_end, all_onsets):
    """Strict quote marks and the global next detection, without chunk censoring."""
    stop = _ns([sample_end])[0]
    _onset_clock(onsets, stop)
    _onset_clock(all_onsets, stop)
    global_identity = all_onsets.set_index('episode_id')[['bin_start', 'available_at']]
    require(onsets.episode_id.isin(global_identity.index).all(), 'Local onset absent from global detector panel')
    selected = global_identity.loc[onsets.episode_id].reset_index()
    for name in ['bin_start', 'available_at']:
        np.testing.assert_array_equal(_ns(onsets[name]), _ns(selected[name]),
                                      err_msg='Local/global onset clock differs: ' + name)
    inherited = all_onsets.set_index('episode_id').loc[onsets.episode_id].reset_index()
    columns = ['support_id', 'quote_ofi', 'onset_confirmed', 'state_sample_eligible']
    if 'period' in onsets or 'period' in inherited:
        columns.append('period')
    for name in columns:
        require(name in onsets and name in inherited, 'Missing local/global onset field: ' + name)
        pd.testing.assert_series_equal(onsets[name].reset_index(drop=True), inherited[name],
                                       check_exact=True, obj='Local/global onset field: ' + name)
    result = reference_response_tables(onsets, quotes, support, policy='structural_only',
                                      development_end=sample_end, horizons=np.arange(301, dtype='int64'))
    anchors, paths = result['anchors'], result['paths']
    for frame, columns in [(anchors, ['pre_mark_status', 'detection_mark_status']),
                           (paths, ['endpoint_status'])]:
        for name in columns:
            frame[name] = frame[name].replace({'beyond_development_end': 'beyond_sample_end'})
    if len(onsets):
        queries = onsets[['available_at']].rename(columns={'available_at': 'detected_at'}).reset_index(drop=True)
        global_dates = all_onsets[['available_at']].rename(columns={'available_at': 'next_detected_at'})
        next_detection = pd.merge_asof(queries, global_dates, left_on='detected_at',
            right_on='next_detected_at', direction='forward', allow_exact_matches=False)
        mapping = pd.Series(next_detection.next_detected_at.to_numpy(), index=onsets.episode_id)
        later = pd.to_datetime(paths.episode_id.map(mapping), utc=True)
        warning = pd.array(later.le(paths.target_at), dtype='boolean')
    else:
        warning = pd.array([], dtype='boolean')
    warning[paths.target_at.gt(pd.Timestamp(sample_end)).to_numpy()] = pd.NA
    paths['later_observed_episode_by_horizon'] = warning
    require(len(paths) == 301 * len(onsets), 'Incomplete onset-horizon grid')
    return result


def _horizon_rows(paths):
    require(len(paths) > 0 and paths.horizon_sec.nunique(dropna=False) == 1,
            'Exactly one nonempty horizon slice required')
    h = paths.horizon_sec.iloc[0]
    require(isinstance(h, (int, np.integer)) and not isinstance(h, (bool, np.bool_)) and 0 <= h <= 300,
            'Integer horizon must lie between zero and 300')
    require(not paths.episode_id.duplicated().any(), 'Repeated onset in horizon slice')
    frame = paths.reset_index(drop=True)
    for name in ['onset_confirmed', 'state_sample_eligible', 'direction_available']:
        require(pd.api.types.is_bool_dtype(frame[name].dtype) and frame[name].notna().all(),
                'Nonmissing Boolean cohort flags required: ' + name)
    require(not (frame.state_sample_eligible & ~frame.onset_confirmed).any(),
            'Inherited state cohort contains an uncertain onset')
    return int(h), frame


def _status_counts(values):
    counts = Counter(values)
    require(set(counts).issubset(STATUSES), 'Unknown panel response mark status')
    return {name: counts[name] for name in STATUSES}


def reference_horizon_summary(paths):
    """Return the ten legacy joint samples and one-horizon attrition dictionary."""
    horizon, frame = _horizon_rows(paths)
    groups = _cohort_indices(frame)
    samples = {name: _summary_for_rows(frame, groups[name]) for name in SAMPLES}
    summary = pd.DataFrame([{'horizon_sec': horizon, 'sample': name, **samples[name]}
                            for name in SAMPLES], columns=SUMMARY_COLUMNS)
    for name in ['mean_endpoint_age_sec', 'p95_endpoint_age_sec',
                 *[prefix + value for value in MEASURES for prefix in ('mean_', 'median_')]]:
        summary[name] = summary[name].astype(float)
    directed = set(frame.index[frame.onset_confirmed & frame.direction_available])
    joint = set(frame.index[frame.triplet_quote_available])
    continuation = set(frame.index[frame.continuation_quote_available])
    stats = {'endpoint_status_all_fragments': _status_counts(frame.endpoint_status),
        'confirmed_directed_cohort': len(directed), 'joint_available': len(directed & joint),
        'joint_missing': len(directed - joint),
        'continuation_available_without_premark': len((directed & continuation) - joint),
        'samples': samples}
    return summary, stats


def reference_anchor_statistics(anchors):
    """The legacy anchor accounting, excluding path and horizon aggregates."""
    require(not anchors.episode_id.duplicated().any(), 'Duplicate anchor onset identity')
    directions = Counter(anchors.direction_status)
    require(set(directions).issubset({'positive', 'negative', 'zero', 'missing'}), 'Unknown direction status')
    nonzero = anchors.loc[anchors.direction_available, 'abs_onset_quote_ofi'].to_numpy(dtype=float)
    require(np.isfinite(nonzero).all() and (nonzero > 0).all(), 'Usable direction has nonpositive or missing magnitude')
    return {'onset_rows': len(anchors), 'confirmed_onsets': int(np.count_nonzero(anchors.onset_confirmed)),
        'direction_counts': {name: directions[name] for name in ('positive', 'negative', 'zero', 'missing')},
        'confirmed_directed_onsets': int(np.count_nonzero(anchors.confirmed_direction_cohort)),
        'state_directed_onsets': int(np.count_nonzero(anchors.state_direction_cohort)),
        'baseline_quotes_available': int(np.count_nonzero(anchors.baseline_quote_available)),
        'minimum_nonzero_abs_onset_ofi': float(min(nonzero)) if len(nonzero) else None,
        'pre_mark_status': _status_counts(anchors.pre_mark_status),
        'detection_mark_status': _status_counts(anchors.detection_mark_status)}


def reference_component_summary(paths):
    """Component-specific samples preserve observable C when the premark is absent.

    Initial J needs its two anchor marks only, including when the later target
    has crossed the sample endpoint. The joint row uses the same observations
    for all three means; means in distinct component rows need not add up.
    """
    horizon, frame = _horizon_rows(paths)
    yes = lambda name: set(frame.index[frame[name].to_numpy(dtype=bool)])
    directed = yes('direction_available')
    cohorts = {'confirmed_directed': directed & yes('onset_confirmed'),
               'state_directed': directed & yes('state_sample_eligible')}
    available = {'initial': set(frame.index[np.isfinite(frame.initial_pips.to_numpy(dtype=float))]),
        'continuation': yes('continuation_quote_available'), 'total': yes('total_quote_available'),
        'joint': yes('triplet_quote_available')}
    rows = []
    for cohort, population in cohorts.items():
        for component in ['initial', 'continuation', 'total', 'joint']:
            selected = population & available[component]
            row = {'horizon_sec': horizon, 'cohort': cohort, 'component': component,
                'cohort_onsets': len(population), 'available_onsets': len(selected),
                'missing_onsets': len(population - selected)}
            for name in MEASURES:
                values = frame.loc[sorted(selected), name].to_numpy(dtype=float)
                if component == 'joint' or name == component + '_pips':
                    require(np.isfinite(values).all(), 'Available component has nonfinite aligned response: ' + name)
                    row['mean_' + name] = float(np.mean(values)) if len(values) else None
                else:
                    row['mean_' + name] = None
            rows.append(row)
    result = pd.DataFrame(rows, columns=COMPONENT_COLUMNS)
    for name in MEASURES:
        result['mean_' + name] = result['mean_' + name].astype(float)
    return result

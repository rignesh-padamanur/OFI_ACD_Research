"""Continuous recorded-price responses on fixed full-panel episode onsets.

Processing batches delimit I/O only. Structural support and the full sample
end delimit observable paths; the global onset stream supplies overlap flags.
"""
import numpy as np
import pandas as pd

from src.price_responses import (
    MARK_STATUSES as LEGACY_STATUSES, MEASURES, SAMPLE_NAMES, _sample_mask,
    construct_price_responses, integer_values, ns,
)

MARK_STATUSES = [s.replace('beyond_development_end', 'beyond_sample_end')
                 for s in LEGACY_STATUSES]
SUMMARY_COLUMNS = [
    'horizon_sec', 'sample', 'observations', 'mean_endpoint_age_sec',
    'p95_endpoint_age_sec', 'reused_detection_quote', 'later_observed_episode',
    *[prefix + measure for measure in MEASURES for prefix in ['mean_', 'median_']],
]
COMPONENT_COLUMNS = [
    'horizon_sec', 'cohort', 'component', 'cohort_onsets', 'available_onsets',
    'missing_onsets', *['mean_' + measure for measure in MEASURES],
]


def _global_onset_identity(onsets, sample_end):
    ids = integer_values(onsets.episode_id)
    start, detected = ns(onsets.bin_start), ns(onsets.available_at)
    stop = ns([sample_end])[0]
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Global onset IDs must be unique nonnegative integers')
    if ((start % 1_000_000_000).any() or (np.diff(start) <= 0).any()
            or not np.array_equal(detected, start + 1_000_000_000)):
        raise ValueError('Global onsets need chronological bins and right-endpoint detections')
    if (detected > stop).any():
        raise ValueError('Global onset detection exceeds the sample end')
    return ids, detected, stop


def construct_panel_responses(onsets, quotes, support, *, sample_end, all_onsets):
    """Retain every selected onset and all 301 horizons, without batch censoring."""
    global_ids, global_detection, stop = _global_onset_identity(all_onsets, sample_end)
    selected_ids = integer_values(onsets.episode_id)
    positions = pd.Index(global_ids).get_indexer(selected_ids)
    if (positions < 0).any():
        raise ValueError('Selected onset is absent from the global onset stream')
    inherited = all_onsets.iloc[positions].reset_index(drop=True)
    selected = onsets.reset_index(drop=True)
    # matching values, including missing values, prevent a batch from silently
    # changing its timing, direction, support or inherited sample membership.
    for col in ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                'onset_confirmed', 'state_sample_eligible', 'period']:
        if col == 'period' and col not in inherited and col not in selected:
            continue
        if col not in inherited or col not in selected or not selected[col].equals(inherited[col]):
            raise ValueError('Selected onset differs from the global stream: ' + col)
    result = construct_price_responses(
        onsets, quotes, support, policy='structural_only',
        development_end=sample_end, horizons=np.arange(301, dtype='int64'))
    anchors, paths = result['anchors'], result['paths']
    # the next detection can lie in another processing batch or calibration
    # cycle. It remains a descriptive flag, never an eligibility condition.
    next_position = np.searchsorted(global_detection, ns(anchors.available_at), side='right')
    next_detection = np.full(len(anchors), np.iinfo('int64').max, dtype='int64')
    exists = next_position < len(global_detection)
    next_detection[exists] = global_detection[next_position[exists]]
    target = ns(paths.target_at)
    overlap = pd.array(np.repeat(next_detection, 301) <= target, dtype='boolean')
    overlap[target > stop] = pd.NA
    paths['later_observed_episode_by_horizon'] = overlap
    for frame, columns in [(anchors, ['pre_mark_status', 'detection_mark_status']),
                           (paths, ['endpoint_status'])]:
        for col in columns:
            frame[col] = frame[col].replace('beyond_development_end', 'beyond_sample_end')
    return result


def _horizon(paths):
    if paths.empty or paths.horizon_sec.nunique() != 1 or paths.episode_id.duplicated().any():
        raise ValueError('One nonempty horizon with one row per onset is required')
    return int(paths.horizon_sec.iloc[0])


def _status_counts(values):
    if not values.isin(MARK_STATUSES).all():
        raise ValueError('Unrecognized response-mark status')
    return {status: int(values.eq(status).sum()) for status in MARK_STATUSES}


def anchor_statistics(anchors):
    """Counts for unique anchors, independent of how paths are partitioned."""
    if anchors.episode_id.duplicated().any():
        raise ValueError('Duplicate response anchors')
    magnitudes = anchors.loc[anchors.direction_available, 'abs_onset_quote_ofi']
    return {
        'onset_rows': len(anchors),
        'confirmed_onsets': int(anchors.onset_confirmed.sum()),
        'direction_counts': {k: int(anchors.direction_status.eq(k).sum())
                             for k in ['positive', 'negative', 'zero', 'missing']},
        'confirmed_directed_onsets': int(anchors.confirmed_direction_cohort.sum()),
        'state_directed_onsets': int(anchors.state_direction_cohort.sum()),
        'baseline_quotes_available': int(anchors.baseline_quote_available.sum()),
        'minimum_nonzero_abs_onset_ofi': float(magnitudes.min()) if len(magnitudes) else None,
        'pre_mark_status': _status_counts(anchors.pre_mark_status),
        'detection_mark_status': _status_counts(anchors.detection_mark_status),
    }


def summarize_horizon(paths):
    """Legacy joint-sample descriptions and attrition for one full-panel horizon."""
    h = _horizon(paths)
    rows, sample_stats = [], {}
    for sample in SAMPLE_NAMES:
        group = paths.loc[_sample_mask(paths, sample)]
        row = {
            'horizon_sec': h, 'sample': sample, 'observations': len(group),
            'mean_endpoint_age_sec': float(group.endpoint_age_sec.mean()) if len(group) else np.nan,
            'p95_endpoint_age_sec': float(group.endpoint_age_sec.quantile(.95)) if len(group) else np.nan,
            'reused_detection_quote': int(group.endpoint_reuses_detection_quote.sum()),
            'later_observed_episode': int(group.later_observed_episode_by_horizon.fillna(False).sum()),
        }
        for col in MEASURES:
            row['mean_' + col] = float(group[col].mean()) if len(group) else np.nan
            row['median_' + col] = float(group[col].median()) if len(group) else np.nan
        rows.append(row)
        sample_stats[sample] = {
            k: None if pd.isna(v) else v for k, v in row.items()
            if k not in ['horizon_sec', 'sample']}
    cohort = paths.onset_confirmed & paths.direction_available
    stats = {
        'endpoint_status_all_fragments': _status_counts(paths.endpoint_status),
        'confirmed_directed_cohort': int(cohort.sum()),
        'joint_available': int((cohort & paths.triplet_quote_available).sum()),
        'joint_missing': int((cohort & ~paths.triplet_quote_available).sum()),
        'continuation_available_without_premark': int((
            cohort & paths.continuation_quote_available & ~paths.triplet_quote_available).sum()),
        'samples': sample_stats,
    }
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), stats


def component_summary(paths):
    """Keep component-specific denominators explicit; joint rows share all marks."""
    h = _horizon(paths)
    availability = {
        'initial': paths.initial_pips.notna(),
        'continuation': paths.continuation_quote_available,
        'total': paths.total_quote_available,
        'joint': paths.triplet_quote_available,
    }
    rows = []
    for cohort in ['confirmed_directed', 'state_directed']:
        mask = paths.onset_confirmed & paths.direction_available
        if cohort == 'state_directed':
            mask &= paths.state_sample_eligible
        count = int(mask.sum())
        for component, available in availability.items():
            group = paths.loc[mask & available]
            row = {'horizon_sec': h, 'cohort': cohort, 'component': component,
                   'cohort_onsets': count, 'available_onsets': len(group),
                   'missing_onsets': count - len(group)}
            for col in MEASURES:
                included = component == 'joint' or col == component + '_pips'
                row['mean_' + col] = float(group[col].mean()) if included and len(group) else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=COMPONENT_COLUMNS)

"""Independent references for the frozen full-panel analysis-input extension.

No production panel, ledger, or estimation implementation is imported. The
historical independent row references are reused; interval graphs are rebuilt
with binary-search candidate bounds and interval-union components.
"""
from collections import Counter

import numpy as np
import pandas as pd

from src.verify_analysis_ledger import (
    reference_ledger, reference_specification, _ns,
    INTERVAL_COLUMNS, PAIR_COLUMNS, COMPONENT_COLUMNS, CALENDAR_COLUMNS,
)
from src.verify_estimation_inputs import (
    _feature_reference, _sample_reference, _design_reference, _diagnostic_reference,
    _statistics_reference, _specification_reference,
)

PANEL_START = pd.Timestamp('2024-04-22', tz='UTC')
ANALYSIS_START = pd.Timestamp('2024-05-06', tz='UTC')
PANEL_END = pd.Timestamp('2026-08-01', tz='UTC')
HORIZONS = (0, 60, 300)
COMPONENTS = ('initial', 'continuation', 'total', 'joint')
DAY_NS = 86_400_000_000_000


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _identities(frame, keys, label):
    require(not frame[list(keys)].isna().any().any() and not frame.duplicated(list(keys)).any(),
            label + ': repeated or missing key')
    ids = frame.episode_id
    require(pd.api.types.is_integer_dtype(ids) and not pd.api.types.is_bool_dtype(ids)
            and ids.ge(0).all(), label + ': nonnegative integer episode identities required')


def _segments(times):
    time = _ns(times)
    require(((time >= PANEL_START.value) & (time < PANEL_END.value)).all(), 'Onset outside panel interval')
    require(not (time % 1_000_000_000).any(), 'Whole-second onset bins required')
    return np.where(time < ANALYSIS_START.value, 'reused_development', 'later_reanalysis')


def reference_chronology_specification():
    return {'partition_time': 'bin_start', 'reused_development_start': PANEL_START.isoformat(),
        'later_reanalysis_start': ANALYSIS_START.isoformat(), 'panel_end_exclusive': PANEL_END.isoformat(),
        'main_analysis_segment': 'later_reanalysis', 'later_data_previously_inspected': True,
        'untouched_holdout': False}


def _validate_measurements(anchors, measurements):
    require(measurements.kind.eq('chronological_onset').all(), 'Only chronological_onset measurements are admissible')
    _identities(anchors, ['episode_id'], 'response anchors')
    _identities(measurements, ['episode_id', 'frequency_sec'], 'measurement rows')
    require(pd.api.types.is_integer_dtype(measurements.frequency_sec)
            and not pd.api.types.is_bool_dtype(measurements.frequency_sec), 'Integer measurement frequencies required')
    require(measurements.frequency_sec.isin([5, 15, 30]).all()
            and len(measurements) == 3 * len(anchors), 'Expected three fixed measurement frequencies per onset')
    require(measurements.window_id.eq(measurements.episode_id).all(), 'Window identity differs from episode identity')
    anchor = anchors.set_index('episode_id')
    require(set(measurements.episode_id) == set(anchor.index), 'Measurement population differs')
    for frequency in [5, 15, 30]:
        selected = measurements.loc[measurements.frequency_sec.eq(frequency)].set_index('episode_id').reindex(anchor.index)
        require(selected.end_at.notna().all(), 'Missing frequency row')
        np.testing.assert_array_equal(_ns(selected.end_at), _ns(anchor.bin_start))
        for name in ['onset_confirmed', 'state_sample_eligible']:
            require(pd.api.types.is_bool_dtype(selected[name]) and selected[name].notna().all(),
                    'Nonmissing measurement cohort flags required')
            np.testing.assert_array_equal(selected[name], anchor[name])
        require(selected.n_returns.eq(300 // frequency).all(), 'Measurement return count differs')
        available = selected.status.eq('available')
        require(set(selected.status).issubset({'available', 'no_endpoint_support', 'window_crosses_support',
                'no_prior_initial_quote', 'initial_quote_outside_support'}), 'Unknown measurement status')
        for name in ['rv_bps', 'rv_bps2', 'max_mark_age_sec', 'common_max_mark_age_sec', 'repeat_fraction', 'zero_return_fraction']:
            values = selected[name].to_numpy(dtype=float)
            require(np.isfinite(values[available]).all() and (values[available] >= 0).all()
                    and np.isnan(values[~available]).all(), 'Measurement availability contradicts ' + name)
        np.testing.assert_allclose(selected.loc[available, 'rv_bps'].to_numpy() ** 2,
            selected.loc[available, 'rv_bps2'].to_numpy(), rtol=1e-12, atol=1e-9)
        for name in ['repeat_fraction', 'zero_return_fraction']:
            require(selected.loc[available, name].le(1).all(), 'Measurement fraction exceeds one')
        for cap in [15, 30, 60]:
            for prefix, field in [('own', 'max_mark_age_sec'), ('common', 'common_max_mark_age_sec')]:
                name = f'{prefix}_age_le_{cap}'
                require(pd.api.types.is_bool_dtype(selected[name]) and selected[name].notna().all(),
                        'Nonmissing measurement age flags required')
                np.testing.assert_array_equal(selected[name], available & selected[field].le(cap))


def reference_panel_ledger(anchors, measurements, direction, paths):
    _validate_measurements(anchors, measurements)
    _identities(direction, ['episode_id'], 'direction rows')
    _identities(paths, ['episode_id', 'horizon_sec'], 'response paths')
    require(paths.horizon_sec.isin(HORIZONS).all() and len(paths) == 3 * len(anchors),
            'Expected exactly the 0/60/300 response rows')
    adapted = measurements.copy(deep=True)
    adapted['kind'] = 'development'
    result = reference_ledger(anchors, adapted, direction, paths)
    for name in ['anchors', 'outcomes']:
        result[name]['analysis_segment'] = _segments(result[name].bin_start)
    result['adapter'] = {'source_kind': 'chronological_onset', 'temporary_selector_kind': 'development',
        'purpose': 'Compatibility with the unchanged 06c selector; no chronological relabeling of saved parents',
        'measurement_rows': len(measurements), 'anchor_rows': len(anchors),
        'frequencies_seconds': [5, 15, 30], 'saved_parent_labels_changed': False}
    spec = reference_specification()
    spec.update(version='07f.1', scope='full_panel_ledger_with_later_reanalysis_main_inputs',
        chronology=reference_chronology_specification(), measurement_adapter=result['adapter'].copy())
    spec['anchor_column_roles']['analysis_segment'] = 'fixed_chronological_sample_accounting_not_predictor'
    result['specification'] = spec
    return result


def reference_panel_inputs(anchors, outcomes, scaling):
    _identities(anchors, ['episode_id'], 'ledger anchors')
    _identities(outcomes, ['episode_id', 'horizon_sec'], 'ledger outcomes')
    for frame in [anchors, outcomes]:
        np.testing.assert_array_equal(frame.analysis_segment, _segments(frame.bin_start))
    cq, cv, sv = (scaling[key] for key in ['cQ', 'cV', 'sV'])
    require(np.isfinite([cq, cv, sv]).all() and cq > 0 and sv > 0, 'Invalid frozen scaling')
    cutoff = pd.Timestamp(scaling['reference_cutoff'])
    require(cutoff == PANEL_START and anchors.bin_start.ge(cutoff).all(), 'Frozen scaling cutoff differs')
    results = {}
    for segment in ['reused_development', 'later_reanalysis']:
        a = anchors.loc[anchors.analysis_segment.eq(segment)].reset_index(drop=True)
        o = outcomes.loc[outcomes.analysis_segment.eq(segment)].reset_index(drop=True)
        features = _feature_reference(a, (cq, cv, sv))
        samples = _sample_reference(o, features)
        design = _design_reference(features, samples)
        schemas, calendar = _diagnostic_reference(features, samples)
        results[segment] = {'features': features, 'samples': samples, 'design_rows': design,
            'schemas': schemas, 'calendar_support': calendar,
            'statistics': _statistics_reference(features, samples, design, schemas)}
    all_features = pd.concat([part['features'] for part in results.values()], ignore_index=True).sort_values('episode_id').reset_index(drop=True)
    all_samples = pd.concat([part['samples'] for part in results.values()], ignore_index=True).sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True)
    later = all_samples.analysis_segment.eq('later_reanalysis').to_numpy()
    for component in COMPONENTS:
        all_samples['main_eligible_'+component] = later & all_samples['estimation_eligible_'+component].to_numpy(dtype=bool)
    main = results['later_reanalysis']
    stats = {'full_panel_anchor_rows': len(all_features), 'full_panel_outcome_rows': len(all_samples),
        'main_design_rows': len(main['design_rows']),
        'segment_counts': {name: {'anchor_rows': len(part['features']), 'outcome_rows': len(part['samples']),
            'regressor_eligible': int(np.count_nonzero(part['features'].regressor_eligible))}
            for name, part in results.items()},
        'main_input_statistics': main['statistics'], 'empirical_models_fitted': 0,
        'production_inference_approved': False, 'downstream_ready': False}
    spec = _specification_reference(scaling)
    spec.update(version='07f.1', scope='full_panel_rows_later_reanalysis_computational_designs',
        chronology=reference_chronology_specification(),
        main_sample_rule='estimation_eligible_component and analysis_segment == later_reanalysis',
        frozen_scaling_reestimated=False,
        development_diagnostics_role='retained input audit; excluded from main computational matrices')
    return {'features': all_features, 'samples': all_samples, 'design_rows': main['design_rows'],
        'calendar_support': main['calendar_support'], 'schemas': main['schemas'], 'statistics': stats,
        'development_statistics': results['reused_development']['statistics'], 'specification': spec}


def reference_dependence_horizon(features, samples, horizon):
    """Enumerate exact intersections using ordered starts and binary bounds.

    For j>i in start order, valid intervals satisfy end[j]>start[j]>=start[i].
    Consequently start[j]<end[i] is the remaining strict overlap condition.
    Connected components follow the union of overlapping intervals, independently
    of either the pair enumeration or the production graph union-find.
    """
    require(horizon in HORIZONS, 'Undeclared dependence horizon')
    _identities(features, ['episode_id'], 'dependence features')
    _identities(samples, ['episode_id', 'horizon_sec'], 'dependence samples')
    require(pd.api.types.is_bool_dtype(samples.main_eligible_continuation)
            and samples.main_eligible_continuation.notna().all(), 'Complete main-C eligibility flags required')
    chosen = samples.loc[samples.horizon_sec.eq(horizon) & samples.main_eligible_continuation]
    require(chosen.episode_id.isin(features.episode_id).all(), 'Dependence event absent from features')
    frame = features.set_index('episode_id', drop=False).loc[chosen.episode_id].copy()
    frame['end_at'] = chosen.set_index('episode_id').target_at
    frame = frame.reset_index(drop=True).sort_values(['source_start_at', 'episode_id']).reset_index(drop=True)
    require(frame.analysis_segment.eq('later_reanalysis').all(), 'Development event entered main dependence sample')
    n = len(frame)
    output = {name: pd.DataFrame(columns=columns) for name, columns in [
        ('intervals', INTERVAL_COLUMNS), ('pairs', PAIR_COLUMNS),
        ('components', COMPONENT_COLUMNS), ('calendar', CALENDAR_COLUMNS)]}
    if not n:
        return output
    starts, ends = _ns(frame.source_start_at), _ns(frame.end_at)
    require((starts < ends).all(), 'Source interval has nonpositive length')
    ids = frame.episode_id.to_numpy(dtype='int64')
    days, weeks = _ns(frame.onset_day), _ns(frame.onset_week_start)
    limit = np.searchsorted(starts, ends, side='left')
    forward_counts = np.maximum(limit - np.arange(n) - 1, 0)
    total = int(forward_counts.sum())
    left_id = np.empty(total, dtype='int64'); right_id = np.empty(total, dtype='int64')
    seconds = np.empty(total); cross_day = np.empty(total, dtype=bool); cross_week = np.empty(total, dtype=bool)
    degree = np.zeros(n, dtype='int64')
    offset = 0
    for i in range(n):
        neighbors = np.arange(i+1, limit[i], dtype='int64')
        width = len(neighbors)
        if not width:
            continue
        section = slice(offset, offset+width)
        intersection = np.minimum(ends[i], ends[neighbors]) - starts[neighbors]
        require((intersection > 0).all(), 'Binary-search candidate lacks positive intersection')
        left_id[section] = np.minimum(ids[i], ids[neighbors])
        right_id[section] = np.maximum(ids[i], ids[neighbors])
        seconds[section] = intersection / 1_000_000_000
        cross_day[section] = days[i] != days[neighbors]
        cross_week[section] = weeks[i] != weeks[neighbors]
        degree[i] += width; degree[neighbors] += 1
        offset += width
    require(offset == total and degree.sum() == 2 * total, 'Pair-degree conservation failed')
    output['pairs'] = pd.DataFrame({'horizon_sec': np.full(total, horizon, dtype='int64'),
        'episode_id_left': left_id, 'episode_id_right': right_id, 'overlap_seconds': seconds,
        'cross_onset_day': cross_day, 'cross_onset_week': cross_week})

    labels = np.empty(n, dtype='int64'); sizes = np.empty(n, dtype='int64'); component_rows = []
    first = 0; rightmost = int(ends[0])
    for j in range(1, n+1):
        if j < n and starts[j] < rightmost:
            rightmost = max(rightmost, int(ends[j]))
            continue
        label, size = int(ids[first:j].min()), j-first
        labels[first:j] = label; sizes[first:j] = size
        component_rows.append((horizon, label, size, pd.Timestamp(int(starts[first]), unit='ns', tz='UTC'),
                               pd.Timestamp(rightmost, unit='ns', tz='UTC')))
        if j < n:
            first, rightmost = j, int(ends[j])
    output['components'] = pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)
    intervals = frame[['episode_id', 'nominal_start_at', 'source_start_at', 'end_at', 'onset_day', 'onset_week_start']].copy()
    intervals['horizon_sec'] = horizon
    intervals['degree'] = degree; intervals['component_id'] = labels; intervals['component_size'] = sizes
    output['intervals'] = intervals[list(INTERVAL_COLUMNS)]
    calendar_rows = []
    for label, column in [('day', 'onset_day'), ('week', 'onset_week_start')]:
        for stamp, count in sorted(Counter(frame[column]).items()):
            calendar_rows.append((horizon, label, stamp, count, count/n))
    output['calendar'] = pd.DataFrame(calendar_rows, columns=CALENDAR_COLUMNS)
    for name, keys in [('intervals', ['horizon_sec', 'episode_id']),
                       ('pairs', ['horizon_sec', 'episode_id_left', 'episode_id_right']),
                       ('components', ['horizon_sec', 'component_id']),
                       ('calendar', ['horizon_sec', 'block_type', 'block_start'])]:
        output[name] = output[name].sort_values(keys).reset_index(drop=True)
    return output


def reference_panel_dependence(inputs):
    for horizon in HORIZONS:
        yield horizon, reference_dependence_horizon(inputs['features'], inputs['samples'], horizon)


def reference_dep_summary(horizon, dependence):
    for frame in dependence.values():
        require(frame.horizon_sec.eq(horizon).all(), 'Mixed horizons in dependence summary')
    intervals, pairs, components = [dependence[key] for key in ['intervals', 'pairs', 'components']]
    degrees = intervals.degree.to_numpy(dtype='int64')
    return {'horizon_sec': int(horizon), 'cohort': 'main_eligible_continuation_later_reanalysis',
        'events': len(intervals), 'overlap_pairs': len(pairs),
        'events_with_overlap': int(np.count_nonzero(degrees)), 'maximum_degree': int(max(degrees, default=0)),
        'components': len(components), 'largest_component_events': int(max(components.events, default=0)),
        'cross_day_pairs': int(np.count_nonzero(pairs.cross_onset_day)),
        'cross_week_pairs': int(np.count_nonzero(pairs.cross_onset_week)),
        'source_start_before_analysis_start': int(np.count_nonzero(intervals.source_start_at.lt(ANALYSIS_START))),
        'inference_approved': False}


def reference_daily_calendar(inputs):
    features, samples = inputs['features'], inputs['samples']
    dates = pd.date_range(PANEL_START, PANEL_END, inclusive='left', freq='D')
    onset_day = (_ns(features.bin_start)-PANEL_START.value)//DAY_NS
    require(((onset_day >= 0) & (onset_day < len(dates))).all(), 'Feature outside panel calendar')
    all_counts = np.bincount(onset_day, minlength=len(dates))
    parts = []
    for horizon in HORIZONS:
        frame = samples.loc[samples.horizon_sec.eq(horizon)]
        day = (_ns(frame.bin_start)-PANEL_START.value)//DAY_NS
        require(((day >= 0) & (day < len(dates))).all(), 'Sample outside panel calendar')
        table = pd.DataFrame({'date': dates, 'analysis_segment': _segments(dates),
            'horizon_sec': horizon, 'all_onsets': all_counts})
        for component in COMPONENTS:
            selected = frame['main_eligible_'+component].to_numpy(dtype=bool)
            table['main_'+component] = np.bincount(day[selected], minlength=len(dates))
        parts.append(table)
    return pd.concat(parts, ignore_index=True).sort_values(['date', 'horizon_sec']).reset_index(drop=True)

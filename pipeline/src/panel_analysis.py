"""Extend the fixed 06c ledger and 07a inputs without response estimation.

Saved parent labels and measurements remain unchanged. The old development-kind
selector is adapted only in a validated temporary copy. Main computational rows
belong to the declared later chronological reanalysis, never an untouched holdout.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from src.analysis_ledger import (
    HORIZONS, assemble_ledger, dependence_tables, specification as ledger_specification,
)
from src.estimation_inputs import COMPONENTS, build_inputs
from src.panel_measurement import MEASUREMENT_COLUMNS, _validate_measurements


PANEL_START = pd.Timestamp('2024-04-22', tz='UTC')
ANALYSIS_START = pd.Timestamp('2024-05-06', tz='UTC')
PANEL_END = pd.Timestamp('2026-08-01', tz='UTC')
SEGMENTS = ('reused_development', 'later_reanalysis')


def analysis_segments(times):
    """Fixed onset-time partition; no outcome or source-quote date selects it."""
    index = pd.DatetimeIndex(times)
    if str(index.tz) != 'UTC' or index.hasnans:
        raise ValueError('Nonmissing UTC onset times required')
    if ((index.as_unit('ns').asi8 % 1_000_000_000 != 0).any()
            or (index < PANEL_START).any() or (index >= PANEL_END).any()):
        raise ValueError('Whole-second onset bins must lie in the fixed panel interval')
    return np.where(index < ANALYSIS_START, SEGMENTS[0], SEGMENTS[1])


def _check_segments(frame):
    if 'analysis_segment' not in frame:
        raise ValueError('Explicit analysis segment required')
    if not np.array_equal(frame.analysis_segment.to_numpy(), analysis_segments(frame.bin_start)):
        raise ValueError('Analysis segment contradicts the fixed onset-time partition')


def chronology_specification():
    return {
        'partition_time': 'bin_start',
        'reused_development_start': PANEL_START.isoformat(),
        'later_reanalysis_start': ANALYSIS_START.isoformat(),
        'panel_end_exclusive': PANEL_END.isoformat(),
        'main_analysis_segment': 'later_reanalysis',
        'later_data_previously_inspected': True,
        'untouched_holdout': False,
    }


def assemble_panel_ledger(anchors, measurements, direction, paths):
    """Retain every anchor and exactly the three previously declared horizons."""
    missing = set(MEASUREMENT_COLUMNS)-set(measurements)
    if missing:
        raise ValueError('Incomplete 07d measurement schema: '+str(sorted(missing)))
    _validate_measurements(measurements)
    analysis_segments(anchors.bin_start)
    if not paths.horizon_sec.isin(HORIZONS).all():
        raise ValueError('Read only the fixed 0/60/300-second response paths')
    if anchors.episode_id.duplicated().any():
        raise ValueError('Unique anchor identities required')
    linked = anchors.set_index('episode_id').state_sample_eligible.reindex(measurements.episode_id)
    if linked.isna().any() or not np.array_equal(linked.to_numpy(), measurements.state_sample_eligible.to_numpy()):
        raise ValueError('Inherited measurement state membership differs from anchors')
    adapted = measurements.copy()
    adapted['kind'] = 'development'
    result = assemble_ledger(anchors, adapted, direction, paths)
    for name in ['anchors', 'outcomes']:
        result[name]['analysis_segment'] = analysis_segments(result[name].bin_start)
    result['adapter'] = {
        'source_kind': 'chronological_onset', 'temporary_selector_kind': 'development',
        'purpose': 'Compatibility with the unchanged 06c selector; no chronological relabeling of saved parents',
        'measurement_rows': len(measurements), 'anchor_rows': len(anchors),
        'frequencies_seconds': [5, 15, 30],
        'saved_parent_labels_changed': False,
    }
    spec = ledger_specification()
    spec.update(version='07f.1', scope='full_panel_ledger_with_later_reanalysis_main_inputs',
                chronology=chronology_specification(), measurement_adapter=deepcopy(result['adapter']))
    spec['anchor_column_roles']['analysis_segment'] = 'fixed_chronological_sample_accounting_not_predictor'
    result['specification'] = spec
    return result


def build_panel_inputs(anchors, outcomes, scaling):
    """Reuse feature algebra by segment; retain all rows and only main matrices."""
    for frame in [anchors, outcomes]:
        _check_segments(frame)
    if pd.Timestamp(scaling['reference_cutoff']) != PANEL_START:
        raise ValueError('The frozen 07a reference cutoff must remain 2024-04-22 UTC')
    parts = {}
    frozen = deepcopy(scaling)
    for segment in SEGMENTS:
        a = anchors.loc[anchors.analysis_segment.eq(segment)].copy()
        o = outcomes.loc[outcomes.analysis_segment.eq(segment)].copy()
        parts[segment] = build_inputs(a, o, scaling)
    if scaling != frozen:
        raise AssertionError('Input construction changed frozen scaling')
    features = pd.concat([parts[name]['features'] for name in SEGMENTS], ignore_index=True)
    features = features.sort_values('episode_id').reset_index(drop=True)
    samples = pd.concat([parts[name]['samples'] for name in SEGMENTS], ignore_index=True)
    samples = samples.sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True)
    if features.episode_id.duplicated().any() or samples.duplicated(['episode_id', 'horizon_sec']).any():
        raise ValueError('Duplicated identities across chronological segments')
    for name in COMPONENTS:
        samples['main_eligible_'+name] = samples['estimation_eligible_'+name] & samples.analysis_segment.eq(SEGMENTS[1])
    main, development = (parts[name] for name in [SEGMENTS[1], SEGMENTS[0]])
    spec = deepcopy(main['specification'])
    spec.update(version='07f.1', scope='full_panel_rows_later_reanalysis_computational_designs',
                chronology=chronology_specification(),
                main_sample_rule='estimation_eligible_component and analysis_segment == later_reanalysis',
                frozen_scaling_reestimated=False,
                development_diagnostics_role='retained input audit; excluded from main computational matrices')
    statistics = {
        'full_panel_anchor_rows': len(features), 'full_panel_outcome_rows': len(samples),
        'main_design_rows': len(main['design_rows']),
        'segment_counts': {name: {
            'anchor_rows': len(parts[name]['features']), 'outcome_rows': len(parts[name]['samples']),
            'regressor_eligible': int(parts[name]['features'].regressor_eligible.sum()),
        } for name in SEGMENTS},
        'main_input_statistics': deepcopy(main['statistics']),
        'empirical_models_fitted': 0, 'production_inference_approved': False, 'downstream_ready': False,
    }
    return {'features': features, 'samples': samples,
            'design_rows': main['design_rows'], 'calendar_support': main['calendar_support'],
            'schemas': main['schemas'], 'statistics': statistics, 'specification': spec,
            'development_statistics': deepcopy(development['statistics'])}


def _main_tables(tables):
    features, samples = tables['features'], tables['samples']
    for frame in [features, samples]:
        _check_segments(frame)
    if features.episode_id.duplicated().any() or samples.duplicated(['episode_id', 'horizon_sec']).any():
        raise ValueError('Unique full-panel feature/outcome identities required')
    expected = [(i, h) for i in features.sort_values('episode_id').episode_id for h in HORIZONS]
    ordered = samples.sort_values(['episode_id', 'horizon_sec'])
    if list(zip(ordered.episode_id, ordered.horizon_sec)) != expected:
        raise ValueError('Every feature row requires exactly the three fixed outcomes')
    linked = features.set_index('episode_id').loc[samples.episode_id]
    if not np.array_equal(linked.bin_start.to_numpy(), samples.bin_start.to_numpy()):
        raise ValueError('Feature/sample onset identities disagree')
    for name in COMPONENTS:
        for prefix in ['estimation_eligible_', 'main_eligible_']:
            values = samples[prefix+name]
            if not pd.api.types.is_bool_dtype(values.dtype) or values.isna().any():
                raise ValueError('Main and estimation masks require nonmissing Boolean values')
        expected_mask = samples['estimation_eligible_'+name] & samples.analysis_segment.eq(SEGMENTS[1])
        if not samples['main_eligible_'+name].equals(expected_mask):
            raise ValueError('Main membership differs from the fixed chronology/component rule')
    return features, samples


def panel_dependence(tables):
    """Yield (horizon, dependence tables) for the main C sample, one at a time.

    All source quotes can precede the main onset interval. The earlier onset
    cohort is not reintroduced. Components remain descriptive overlap accounting,
    never an assertion of independent observations or independent clusters.
    """
    features, samples = _main_tables(tables)
    fields = ['episode_id', 'nominal_start_at', 'source_start_at', 'onset_day', 'onset_week_start']
    anchors = features[fields]
    for horizon in HORIZONS:
        selected = samples.loc[samples.horizon_sec.eq(horizon)].copy()
        selected['eligible_continuation'] = selected.main_eligible_continuation
        dependence = dependence_tables(anchors, selected)
        actual = dependence['intervals'].episode_id.sort_values().to_numpy()
        expected = selected.loc[selected.main_eligible_continuation, 'episode_id'].sort_values().to_numpy()
        if not np.array_equal(actual, expected):
            raise AssertionError('Main continuation dependence membership changed')
        yield horizon, dependence


def dependence_summary(horizon, dependence):
    intervals, pairs, components = (dependence[name] for name in ['intervals', 'pairs', 'components'])
    for frame in [intervals, pairs, components, dependence['calendar']]:
        if not frame.horizon_sec.eq(horizon).all():
            raise ValueError('Dependence summary must contain a single declared horizon')
    return {
        'horizon_sec': int(horizon), 'cohort': 'main_eligible_continuation_later_reanalysis',
        'events': len(intervals), 'overlap_pairs': len(pairs),
        'events_with_overlap': int(intervals.degree.gt(0).sum()),
        'maximum_degree': int(intervals.degree.max()) if len(intervals) else 0,
        'components': len(components),
        'largest_component_events': int(components.events.max()) if len(components) else 0,
        'cross_day_pairs': int(pairs.cross_onset_day.sum()),
        'cross_week_pairs': int(pairs.cross_onset_week.sum()),
        'source_start_before_analysis_start': int(intervals.source_start_at.lt(ANALYSIS_START).sum()),
        'inference_approved': False,
    }


def daily_calendar(tables):
    """All 831 UTC dates, including zero-onset dates; repeated for 0/60/300."""
    features, samples = _main_tables(tables)
    dates = pd.date_range(PANEL_START, PANEL_END, freq='D', inclusive='left')
    all_counts = features.groupby(features.bin_start.dt.normalize()).size()
    rows = []
    for horizon in HORIZONS:
        frame = pd.DataFrame({'date': dates})
        frame['analysis_segment'] = analysis_segments(frame.date)
        frame['horizon_sec'] = horizon
        frame['all_onsets'] = all_counts.reindex(dates, fill_value=0).to_numpy(dtype='int64')
        selected = samples.loc[samples.horizon_sec.eq(horizon)]
        for name in COMPONENTS:
            counts = selected.groupby(selected.bin_start.dt.normalize())['main_eligible_'+name].sum()
            frame['main_'+name] = counts.reindex(dates, fill_value=0).to_numpy(dtype='int64')
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(['date', 'horizon_sec']).reset_index(drop=True)

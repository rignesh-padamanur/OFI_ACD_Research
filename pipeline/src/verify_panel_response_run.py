"""Independent, bounded-memory audit of a complete stage-07e response run.

Every stored path is reconstructed from raw quotes first. Pooled summaries then
reuse those checked paths one horizon at a time; this avoids a second 40-million
row quote join or an in-memory concatenation of the full panel.
"""
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.verify_price_responses import (
    _registered_hashes, _compare, _json_compare, _check_identities, _ns,
)
from src.verify_panel_responses import (
    reference_panel_responses, reference_horizon_summary,
    reference_component_summary, reference_anchor_statistics,
)
from src.verify_chronological_panel_v2 import quote_window

STAGE = '07e_full_panel_continuous_price_responses'
SCOPE = 'full_cached_structural_price_responses_only'
SECOND = 1_000_000_000
DAY = 86400 * SECOND


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected, label):
    require(set(actual) == set(expected), label + ': columns differ')
    for name in expected:
        a, e = actual[name], expected[name]
        if pd.api.types.is_bool_dtype(e):
            require(pd.api.types.is_bool_dtype(a), label + ': Boolean field changed: ' + name)
        elif pd.api.types.is_integer_dtype(e):
            require(pd.api.types.is_integer_dtype(a) and not pd.api.types.is_bool_dtype(a),
                    label + ': integer field changed: ' + name)
        elif pd.api.types.is_float_dtype(e):
            require(pd.api.types.is_float_dtype(a), label + ': float field changed: ' + name)
    try:
        _compare(actual.reset_index(drop=True), expected.reset_index(drop=True))
    except AssertionError as error:
        raise AssertionError(label + ': ' + str(error)) from error


def registered_parents(run):
    """Audit immutable sources, full tests, snapshots and exact parent links."""
    m, checked = _registered_hashes(run)
    inputs = m['inputs']
    required = {'parent_manifest', 'parent_verification', 'parent_statistics', 'onsets',
        'cycle_schedule', 'clean_quotes', 'support_intervals', 'measurement_manifest',
        'measurement_verification', 'measurement_statistics', 'directions',
        'legacy_manifest', 'legacy_verification', 'legacy_statistics', 'legacy_anchors', 'legacy_paths'}
    require(set(inputs) == required, 'Missing or unexpected registered input')
    source_names = {'repair_panel_responses.py', 'reports/STAGE07E_PANEL_RESPONSE_PROTOCOL.md',
        'src/panel_responses.py', 'src/verify_panel_responses.py', 'src/verify_panel_response_run.py'}
    require(source_names.issubset(m.get('code_sha256', {})), 'Required executed source snapshot absent')
    previous = m.get('previous_files', {})
    old_pointers = [name for name in previous if name.startswith('data/active_') and name.endswith('.json')]
    require(len(old_pointers) == 21 and {'reports/pipeline_master_statistics.json',
        'reports/PIPELINE_REPAIR_LOG.md', 'REPAIR_WORKFLOW.md'}.issubset(previous),
        'The 21 prior pointers or reporting snapshots are absent')
    require(read(Path(m['snapshot_path'])/'preserved_files.json') == previous, 'Snapshot index differs')
    evidence = read(run/'test_evidence.json')
    require(evidence.get('status') == 'passed' and evidence.get('tests_run', 0) > 0
            and evidence.get('failures') == 0 and evidence.get('errors') == 0, 'Full tests did not pass')
    test_hashes = evidence.get('test_source_sha256', {})
    require(bool(test_hashes), 'Executed test sources missing')
    require(set(test_hashes) == {p.name for p in (run/'full_test_sources').glob('test_*.py')},
            'Full test-source inventory differs')
    for name, digest in test_hashes.items():
        actual = hashlib.sha256((run/'full_test_sources'/name).read_bytes()).hexdigest()
        require(actual == digest and m['code_sha256'].get('tests/'+name) == digest,
                'Executed test source differs: ' + name)
        checked += 1

    def parent_links(prefix, status, output_links):
        manifest = read(inputs[prefix+'_manifest']['path'])
        verification = read(inputs[prefix+'_verification']['path'])
        require(manifest['status'] == status and verification.get('status') == 'passed',
                prefix + ' parent is not verified')
        directory = Path(inputs[prefix+'_manifest']['path']).parent.resolve()
        require(manifest['run_id'] == directory.name, prefix + ' parent identity differs')
        for key, name in output_links:
            require(Path(inputs[key]['path']).resolve() == directory/name
                    and inputs[key]['sha256'] == manifest['outputs'][name],
                    prefix + ' output lineage differs: ' + key)
        return manifest

    parent = parent_links('parent', 'verified_full_chronological_baseline_panel', [
        ('parent_verification', 'verification.json'), ('parent_statistics', 'statistics.json'),
        ('onsets', 'onsets.parquet'), ('cycle_schedule', 'cycle_schedule.parquet')])
    require(parent['stage'] == '07c_full_chronological_baseline_panel'
            and parent['primary_policy'] == 'structural_only', 'Wrong chronological parent')
    measurement = parent_links('measurement', 'verified_full_panel_fixed_onset_measurement', [
        ('measurement_verification', 'verification.json'), ('measurement_statistics', 'statistics.json'),
        ('directions', 'direction.parquet')])
    require(measurement['stage'] == '07d_full_panel_fixed_onset_measurement'
            and measurement['parent_chronological_panel'] == parent['run_id']
            and measurement['inputs']['onsets'] == inputs['onsets'], 'Measurement parent differs')
    legacy = parent_links('legacy', 'verified_development_price_responses', [
        ('legacy_verification', 'verification.json'), ('legacy_statistics', 'statistics.json'),
        ('legacy_anchors', 'structural_only/anchors.parquet'), ('legacy_paths', 'structural_only/paths.parquet')])
    for name in ['clean_quotes', 'support_intervals']:
        require(inputs[name] == parent['inputs'][name] == measurement['inputs'][name]
                == legacy['inputs'][name], 'Parent raw/support lineage differs: ' + name)
    _json_compare(m, {'stage': STAGE, 'scope': SCOPE,
        'parent_chronological_panel': parent['run_id'], 'parent_measurement': measurement['run_id'],
        'primary_policy': 'structural_only', 'horizons_seconds': list(range(301)),
        'deployment_start': parent['first_cutoff'], 'deployment_end_exclusive': parent['deployment_end'],
        'price_mark_rule': 'strictly_prior_left_limit',
        'direction_rule': 'exact_sign_of_finite_nonzero_onset_quote_ofi',
        'path_storage': 'cycle_files_with_one_row_group_per_horizon',
        'production_inference_approved': False, 'downstream_ready': False})
    require(m['status'] in {'built_pending_independent_verification', 'verified_full_panel_price_responses'},
            'Response run is not ready for verification')
    return m, parent, measurement, checked


def reference_calendar(anchors, highlights, start, stop):
    dates = pd.date_range(start, stop, inclusive='left', freq='D')
    result = pd.DataFrame({'date': dates})
    anchor_day = (_ns(anchors.bin_start)-pd.Timestamp(start).value)//DAY
    require(((anchor_day >= 0) & (anchor_day < len(dates))).all(), 'Anchor outside calendar')
    count = lambda day, mask: np.bincount(day[np.asarray(mask, bool)], minlength=len(dates)).astype('int64')
    for name, mask in [('onsets', np.ones(len(anchors), bool)),
                       ('confirmed_onsets', anchors.onset_confirmed),
                       ('confirmed_directed_onsets', anchors.confirmed_direction_cohort),
                       ('state_directed_onsets', anchors.state_direction_cohort)]:
        result[name] = count(anchor_day, mask)
    for horizon in [60, 300]:
        frame = highlights[horizon]
        day = (_ns(frame.bin_start)-pd.Timestamp(start).value)//DAY
        for cohort in ['confirmed', 'state']:
            base = frame.onset_confirmed.to_numpy() & frame.direction_available.to_numpy()
            if cohort == 'state':
                base &= frame.state_sample_eligible.to_numpy()
            continuation = base & frame.continuation_quote_available.to_numpy()
            joint = base & frame.triplet_quote_available.to_numpy()
            result[f'{cohort}_continuation_{horizon}'] = count(day, continuation)
            result[f'{cohort}_joint_{horizon}'] = count(day, joint)
            if horizon == 300:
                specifications = [(f'{cohort}_mean_continuation_300', continuation, 'continuation_pips')]
                specifications += [(f'{cohort}_joint_mean_{component}_300', joint, component+'_pips')
                                   for component in ['initial', 'continuation', 'total']]
                for name, mask, column in specifications:
                    values = frame[column].to_numpy(dtype=float)
                    require(np.isfinite(values[mask]).all(), 'Nonfinite available calendar response')
                    totals = np.bincount(day[mask], weights=values[mask], minlength=len(dates))
                    counts = count(day, mask)
                    result[name] = np.divide(totals, counts, out=np.full(len(dates), np.nan), where=counts > 0)
    return result


def reference_boundary(onsets, schedule, support, stop):
    cycle = schedule[['cycle_id', 'parameter_cutoff', 'deployment_end']].rename(
        columns={'parameter_cutoff': 'cycle_start', 'deployment_end': 'cycle_end'})
    assignment = pd.merge_asof(onsets[['episode_id', 'bin_start', 'available_at', 'support_id']],
        cycle, left_on='bin_start', right_on='cycle_start', direction='backward', allow_exact_matches=True)
    require(assignment.cycle_id.notna().all() and assignment.bin_start.lt(assignment.cycle_end).all(),
            'Onset has no unique containing deployment')
    target = assignment.available_at + pd.Timedelta(seconds=300)
    next_detection = onsets.available_at.shift(-1)
    next_cycle = assignment.cycle_id.shift(-1)
    right = support.end_exclusive.iloc[onsets.support_id.to_numpy()].reset_index(drop=True)
    return pd.DataFrame({'episode_id': onsets.episode_id.to_numpy(),
        'cycle_id': assignment.cycle_id.to_numpy(dtype='int64'),
        'detected_at': onsets.available_at, 'cycle_end': assignment.cycle_end,
        'target_300': target,
        'crosses_internal_cutoff': target.gt(assignment.cycle_end) & assignment.cycle_end.lt(stop),
        'target_300_within_support': target.le(right), 'next_detection_at': next_detection,
        'next_detection_in_later_cycle_within300': next_cycle.gt(assignment.cycle_id) & next_detection.le(target)})


def reference_cycle_statistics(cycle, anchors, paths):
    after = paths.target_at.gt(cycle.deployment_end).to_numpy()
    available = paths.endpoint_status.eq('available').to_numpy()
    last = paths.episode_id.eq(anchors.episode_id.iloc[-1]).to_numpy() if len(anchors) else np.zeros(len(paths), bool)
    overlap = paths.later_observed_episode_by_horizon.fillna(False).to_numpy(dtype=bool)
    return {'cycle_id': int(cycle.cycle_id), 'onset_rows': len(anchors), 'path_rows': len(paths),
        'targets_after_cycle_end': int(np.count_nonzero(after)),
        'available_endpoints_after_cycle_end': int(np.count_nonzero(after & available)),
        'later_flags_recovered': int(np.count_nonzero(last & overlap))}


def reference_legacy(inputs, anchors, paths):
    old_a = pd.read_parquet(inputs['legacy_anchors']['path'])
    old = pd.read_parquet(inputs['legacy_paths']['path']).sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True)
    old_stop = pd.Timestamp(read(inputs['legacy_statistics']['path'])['development_end_exclusive'])
    selected_a = anchors.set_index('episode_id').reindex(old_a.episode_id).reset_index()
    require(selected_a.bin_start.notna().all(), 'Legacy onset absent from new anchors')
    common = [name for name in old_a if name in selected_a]
    compare(selected_a[common], old_a[common], 'Legacy anchors')
    keys = ['episode_id', 'horizon_sec']
    selected = paths.set_index(keys).reindex(pd.MultiIndex.from_frame(old[keys])).reset_index()
    require(selected.target_at.notna().all(), 'Legacy onset-horizon identity absent')
    admissible = old.target_at.le(old_stop)
    # whole-path masks intentionally change when the former future cutoff is removed.
    masks = ['common_path_available', 'common_path_age_le_15', 'common_path_age_le_30', 'common_path_age_le_60']
    require({name for name in old if name.startswith('common_path')} == set(masks), 'Legacy path-mask schema changed')
    own_columns = [name for name in old if name not in masks]
    compare(selected.loc[admissible, own_columns], old.loc[admissible, own_columns], 'Legacy pointwise path')
    newly_available = old.endpoint_status.eq('beyond_development_end') & selected.endpoint_status.eq('available')
    return {'legacy_onsets_checked': len(old_a), 'shared_anchor_columns_checked': len(common),
        'legacy_path_rows': len(old), 'pointwise_rows_within_old_cutoff_checked': int(np.count_nonzero(admissible)),
        'new_available_rows_beyond_old_cutoff': int(np.count_nonzero(newly_available)),
        'onsets_with_new_available_rows': len(set(selected.loc[newly_available, 'episode_id'])),
        'future_dependent_mask_changes': {name: int(np.count_nonzero(selected[name].to_numpy() != old[name].to_numpy())) for name in masks},
        'legacy_cutoff': old_stop.isoformat(), 'legacy_artifacts_preserved': True,
        'historical_pointwise_responses_reconciled': True}


def record_values(frame):
    def value(x):
        if pd.isna(x): return None
        return x.item() if isinstance(x, np.generic) else x
    return [{name: value(x) for name, x in row.items()} for row in frame.to_dict(orient='records')]


def verify_panel_response_run(run):
    run = Path(run)
    m, parent, measurement, hash_count = registered_parents(run)
    inputs = m['inputs']
    onsets = pd.read_parquet(inputs['onsets']['path'])
    schedule = pd.read_parquet(inputs['cycle_schedule']['path'])
    support = pd.read_parquet(inputs['support_intervals']['path'], filters=[('policy', '==', 'structural_only')]).reset_index(drop=True)
    start, stop = pd.Timestamp(m['deployment_start']), pd.Timestamp(m['deployment_end_exclusive'])
    require(start == start.normalize() and stop == stop.normalize() and stop > start, 'Whole UTC study dates required')
    require(str(start.tz) == 'UTC' and str(stop.tz) == 'UTC', 'UTC study dates required')
    np.testing.assert_array_equal(schedule.cycle_id.to_numpy(), np.arange(len(schedule)))
    require(len(schedule) > 0 and schedule.parameter_cutoff.iloc[0] == start
            and schedule.deployment_end.iloc[-1] == stop
            and schedule.parameter_cutoff.lt(schedule.deployment_end).all(), 'Invalid deployment schedule')
    np.testing.assert_array_equal(_ns(schedule.parameter_cutoff.iloc[1:]), _ns(schedule.deployment_end.iloc[:-1]))
    require(onsets.bin_start.ge(start).all() and onsets.bin_start.lt(stop).all(), 'Onset outside study')
    require(onsets.period.eq('development').all(), 'Inherited engine period label changed')
    require(len(onsets) == read(inputs['measurement_statistics']['path'])['onsets'], 'Measurement onset count differs')
    outputs = {'anchors.parquet', 'cycle_statistics.parquet', 'boundary_audit.parquet', 'horizon_summary.parquet',
        'component_summary.parquet', 'horizon_attrition.parquet', 'calendar.parquet',
        'legacy_reconciliation.json', 'statistics.json', 'test_evidence.json'}
    path_names = {f'paths/cycle_{int(c):03d}.parquet' for c in schedule.cycle_id}
    outputs |= path_names
    for cycle_id in schedule.cycle_id:
        outputs |= {f'cycles/{int(cycle_id):03d}/anchors.parquet', f'cycles/{int(cycle_id):03d}/statistics.json'}
    require(outputs.issubset(m['outputs']), 'Required output hash absent')
    require(set(m['outputs']) <= outputs | {'verification.json', 'review.md'}, 'Unexpected registered output')
    require({str(p.relative_to(run)) for p in (run/'paths').glob('*.parquet')} == path_names,
            'Path file inventory differs')

    expected_anchors, cycle_records = [], []
    for cycle in schedule.itertuples(index=False):
        selected = onsets.loc[onsets.bin_start.between(cycle.parameter_cutoff, cycle.deployment_end, inclusive='left')].reset_index(drop=True)
        quotes = quote_window(inputs['clean_quotes']['path'], cycle.parameter_cutoff,
                              min(stop, cycle.deployment_end + pd.Timedelta(seconds=300)))
        require(quotes.timestamp.is_monotonic_increasing and not quotes.timestamp.duplicated().any(), 'Nonunique raw quote clock')
        price = quotes[['bid', 'ask']].to_numpy(dtype=float)
        require(np.isfinite(price).all() and (price[:, 0] > 0).all() and (price[:, 1] > price[:, 0]).all(), 'Invalid raw quotes')
        expected = reference_panel_responses(selected, quotes, support, sample_end=stop, all_onsets=onsets)
        a = expected['anchors']
        p = expected['paths'].sort_values(['horizon_sec', 'episode_id']).reset_index(drop=True)
        path_file = run/f'paths/cycle_{cycle.cycle_id:03d}.parquet'
        actual = pd.read_parquet(path_file)
        compare(pd.read_parquet(run/f'cycles/{cycle.cycle_id:03d}/anchors.parquet'), a, 'Cycle anchors')
        compare(actual, p, 'Cycle paths '+str(cycle.cycle_id))
        require(len(actual) == 301 * len(selected) and not actual.duplicated(['episode_id', 'horizon_sec']).any(), 'Cycle onset-horizon conservation')
        _check_identities(actual)
        parquet = pq.ParquetFile(path_file)
        if len(selected):
            require(parquet.num_row_groups == 301 and all(parquet.metadata.row_group(i).num_rows == len(selected) for i in range(301)),
                    'Path row groups are not one full horizon each')
        count = reference_cycle_statistics(cycle, a, p)
        _json_compare(read(run/f'cycles/{cycle.cycle_id:03d}/statistics.json'), count, exact=True)
        cycle_records.append(count); expected_anchors.append(a)
        print(f'Independently reconstructed response cycle {cycle.cycle_id:03d}: {len(actual):,} paths', flush=True)
        del quotes, price, expected, p, actual, parquet
        gc.collect()
    anchors = pd.concat(expected_anchors, ignore_index=True)
    compare(pd.read_parquet(run/'anchors.parquet'), anchors, 'Combined anchors')
    compare(anchors[onsets.columns], onsets, 'Inherited onset panel')
    cycles = pd.DataFrame(cycle_records)
    compare(pd.read_parquet(run/'cycle_statistics.parquet'), cycles, 'Cycle statistics')
    directions = pd.read_parquet(inputs['directions']['path'])
    inherited_direction = directions[['episode_id', 'bin_start', 'onset_confirmed', 'inherited_net', 'inherited_direction']].copy()
    # Stage07d stores finite signs as integers; response directions allow NaN.
    inherited_direction['inherited_direction'] = inherited_direction.inherited_direction.astype(float)
    compare(anchors[['episode_id', 'bin_start', 'onset_confirmed', 'quote_ofi', 'direction']].rename(
        columns={'quote_ofi': 'inherited_net', 'direction': 'inherited_direction'}),
        inherited_direction, 'Stage07d direction preservation')

    source = ds.dataset(run/'paths', format='parquet')
    summaries, attrition, components, highlights, highlight_paths = [], [], [], {}, {}
    for horizon in range(301):
        paths = source.to_table(filter=ds.field('horizon_sec') == horizon).to_pandas().sort_values('episode_id').reset_index(drop=True)
        np.testing.assert_array_equal(paths.episode_id.to_numpy(), np.sort(onsets.episode_id.to_numpy()))
        summary, values = reference_horizon_summary(paths)
        summaries.append(summary)
        attrition.append({'horizon_sec': horizon,
            **{name: x for name, x in values.items() if name not in ['samples', 'endpoint_status_all_fragments']},
            **{'status_'+name: x for name, x in values['endpoint_status_all_fragments'].items()}})
        present = summary.observations.gt(0)
        np.testing.assert_allclose(summary.loc[present, 'mean_total_pips'],
            summary.loc[present, 'mean_initial_pips'] + summary.loc[present, 'mean_continuation_pips'], rtol=0, atol=1e-9)
        if horizon in [0, 60, 300]:
            highlights[str(horizon)] = values
            highlight_paths[horizon] = paths
            components.append(reference_component_summary(paths))
        if horizon % 60 == 0:
            print(f'Independently summarized checked full-panel horizon {horizon}', flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    component = pd.concat(components, ignore_index=True)
    compare(pd.read_parquet(run/'horizon_summary.parquet'), summary, 'Horizon summaries')
    compare(pd.read_parquet(run/'horizon_attrition.parquet'), pd.DataFrame(attrition), 'Horizon attrition')
    compare(pd.read_parquet(run/'component_summary.parquet'), component, 'Component summaries')
    calendar = reference_calendar(anchors, highlight_paths, start, stop)
    compare(pd.read_parquet(run/'calendar.parquet'), calendar, 'Full calendar')
    boundary = reference_boundary(onsets, schedule, support, stop)
    compare(pd.read_parquet(run/'boundary_audit.parquet'), boundary, 'Boundary audit')
    legacy = reference_legacy(inputs, anchors, pd.read_parquet(run/'paths/cycle_000.parquet'))
    _json_compare(read(run/'legacy_reconciliation.json'), legacy, exact=True)
    statistics = {'scope': SCOPE, 'primary_policy': 'structural_only',
        'deployment_start': start.isoformat(), 'deployment_end_exclusive': stop.isoformat(),
        **reference_anchor_statistics(anchors), 'path_rows': int(cycles.path_rows.sum()),
        'horizons_seconds': list(range(301)), 'primary_horizon_seconds': 300, 'secondary_horizon_seconds': 60,
        'horizons': highlights, 'component_summary': record_values(component), 'horizon_summary_rows': len(summary),
        'calendar': {'days': len(calendar), 'days_with_onsets': int(np.count_nonzero(calendar.onsets)),
            'totals': {name: int(calendar[name].sum()) for name in calendar if pd.api.types.is_integer_dtype(calendar[name])}},
        'boundary': {'crossing_onsets': int(np.count_nonzero(boundary.crosses_internal_cutoff)),
            'internal_boundaries_crossed': len(set(boundary.loc[boundary.crosses_internal_cutoff, 'cycle_id'])),
            'crossing_onsets_with_supported_300_target': int(np.count_nonzero(boundary.crosses_internal_cutoff & boundary.target_300_within_support)),
            'next_detections_in_later_cycle_within300': int(np.count_nonzero(boundary.next_detection_in_later_cycle_within300)),
            'available_endpoints_after_cycle_end': int(cycles.available_endpoints_after_cycle_end.sum()),
            'later_episode_flags_recovered_across_cycles': int(cycles.later_flags_recovered.sum())},
        'legacy_reconciliation': legacy, 'cycles': len(cycles), 'detector_fits': 0,
        'empirical_response_models_fitted': 0, 'baseline_references_preserved': True,
        'production_inference_approved': False, 'downstream_ready': False}
    _json_compare(read(run/'statistics.json'), statistics, exact=True)
    return {'status': 'passed', 'scope': SCOPE, 'hashes_verified': hash_count, 'prior_pointer_snapshots_verified': 21,
        'cycles_verified': len(cycles), 'anchor_rows_verified': len(anchors),
        'path_rows_verified': int(cycles.path_rows.sum()), 'price_mark_requests_verified': 303 * len(anchors),
        'horizon_summary_rows_verified': len(summary), 'horizon_attrition_rows_verified': len(attrition),
        'component_summary_rows_verified': len(component), 'calendar_days_verified': len(calendar),
        'boundary_rows_verified': len(boundary), 'legacy_pointwise_rows_verified': legacy['pointwise_rows_within_old_cutoff_checked'],
        'methods': 'Independent strict-backward raw-quote joins reconstruct every cycle anchor and path; strict-forward global onset joins reconstruct overlap. Exact parent/source/output/test/snapshot hashes and conserved identities are checked. Independently implemented pooled summaries reuse already raw-verified saved paths one horizon at a time; component samples, full calendar, boundaries, legacy pointwise reconciliation and every saved statistic are separately reconstructed. No production response module or runner is imported.',
        'production_inference_approved': False, 'downstream_ready': False}

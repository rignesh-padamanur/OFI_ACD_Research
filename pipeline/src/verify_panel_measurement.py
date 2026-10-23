"""Independent fixed-onset measurement verification for the extended panel.

No stage-07d production measurement functions are imported. Strict backward
joins reconstruct the full one-second boundary grid, with direct squared-return
sums at the three previously fixed frequencies. The verified detector is an
immutable input, not refitted or reselected by this stage.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.verify_chronological_panel_v2 import quote_window
from src.verify_state_measurement import (
    _compare, _json_compare, _ns, _quote_grid_batches, _quantile, _average_ranks,
    reference_direction, reference_direction_summary,
)

SECOND = 1_000_000_000
FREQUENCIES = (5, 15, 30)
CAPS = (15, 30, 60)
COHORTS = ('confirmed_onsets', 'state_eligible_onsets')
SAMPLES = ('all_available', 'common_age_15', 'common_age_30', 'common_age_60')
STATUSES = ('available', 'no_endpoint_support', 'window_crosses_support',
            'no_prior_initial_quote', 'initial_quote_outside_support')
STAGE = '07d_full_panel_fixed_onset_measurement'
SCOPE = 'full_cached_structural_fixed_onset_measurement_only'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, label):
    """Reject schema-family changes as well as incorrect values or missingness."""
    require(set(actual) == set(expected), label + ': columns differ')
    for name in expected:
        a, e = actual[name], expected[name]
        if pd.api.types.is_bool_dtype(e):
            require(pd.api.types.is_bool_dtype(a), label + ': Boolean field changed: ' + name)
        elif pd.api.types.is_integer_dtype(e):
            require(pd.api.types.is_integer_dtype(a) and not pd.api.types.is_bool_dtype(a),
                    label + ': integer field changed: ' + name)
        elif pd.api.types.is_float_dtype(e):
            require(pd.api.types.is_float_dtype(a), label + ': floating field changed: ' + name)
    try:
        _compare(actual.reset_index(drop=True), expected.reset_index(drop=True))
    except AssertionError as error:
        raise AssertionError(label + ': ' + str(error)) from error


def registered_hashes(run):
    """Verify current artifacts and selected verified parent links, without refits."""
    run = Path(run)
    manifest = read(run / 'manifest.json')
    cache, expected = {}, {}
    registrations = 0

    def check(path, digest):
        nonlocal registrations
        path = Path(path).resolve()
        name = str(path)
        require(name not in expected or expected[name] == digest,
                'Conflicting registered hash: ' + name)
        expected[name] = digest
        if name not in cache:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(block)
            cache[name] = h.hexdigest()
        require(cache[name] == digest, 'Changed registered artifact: ' + name)
        registrations += 1

    inputs = manifest['inputs']
    needed = {'parent_manifest', 'parent_verification', 'parent_statistics',
              'onsets', 'cycle_schedule', 'clean_quotes', 'support_intervals',
              'ofi_ticks', 'ofi_manifest', 'ofi_verification'}
    require(needed.issubset(inputs), 'Required input registration is absent')
    for record in inputs.values():
        check(record['path'], record['sha256'])
    for name, digest in manifest.get('outputs', {}).items():
        check(run / name, digest)
    for name, digest in manifest.get('code_sha256', {}).items():
        check(run / 'code' / name, digest)
    require(bool(manifest.get('code_sha256')), 'Executed sources were not preserved')
    for name, digest in manifest.get('previous_files', {}).items():
        check(Path(manifest['snapshot_path']) / name, digest)
    require(bool(manifest.get('previous_files')), 'Before-repair snapshot is absent')

    evidence = read(run / 'test_evidence.json')
    require(evidence.get('status') == 'passed' and evidence.get('failures', 0) == 0
            and evidence.get('errors', 0) == 0, 'Test evidence did not pass')
    for name, digest in evidence.get('test_source_sha256', {}).items():
        check(run / 'full_test_sources' / name, digest)
    require(bool(evidence.get('test_source_sha256')), 'Full test-source hashes are absent')

    parent = read(inputs['parent_manifest']['path'])
    verification = read(inputs['parent_verification']['path'])
    require(parent['stage'] == '07c_full_chronological_baseline_panel'
            and parent['status'] == 'verified_full_chronological_baseline_panel'
            and verification['status'] == 'passed', 'Chronological parent is not verified')
    require(parent['primary_policy'] == 'structural_only', 'Parent support changed')
    directory = Path(inputs['parent_manifest']['path']).parent.resolve()
    for name, filename in [('parent_verification', 'verification.json'),
                           ('parent_statistics', 'statistics.json'),
                           ('onsets', 'onsets.parquet'),
                           ('cycle_schedule', 'cycle_schedule.parquet')]:
        require(Path(inputs[name]['path']).resolve() == directory / filename
                and inputs[name]['sha256'] == parent['outputs'][filename],
                'Chronological parent output link differs: ' + name)
    for name in ('clean_quotes', 'support_intervals'):
        require(inputs[name] == parent['inputs'][name], 'Parent raw source differs: ' + name)

    imbalance = read(inputs['ofi_manifest']['path'])
    ofi_verification = read(inputs['ofi_verification']['path'])
    require(imbalance['status'] == 'verified_recorded_quote_proxy'
            and ofi_verification['status'] == 'passed', 'Imbalance parent is not verified')
    directory = Path(inputs['ofi_manifest']['path']).parent.resolve()
    for name, filename in [('ofi_ticks', 'quote_ofi_ticks.parquet')]:
        require(Path(inputs[name]['path']).resolve() == directory / filename
                and inputs[name]['sha256'] == imbalance['outputs'][filename],
                'Imbalance parent output link differs: ' + name)
    require(Path(inputs['ofi_verification']['path']).resolve() == directory / 'verification.json',
            'Imbalance verification belongs to another directory')
    for name, parent_name in [('clean_quotes', 'parent_clean'),
                              ('support_intervals', 'support')]:
        require(inputs[name] == imbalance['inputs'][parent_name],
                'Imbalance source/support differs: ' + name)
    return manifest, parent, {'hash_registrations_verified': registrations,
                             'unique_files_hashed': len(cache)}


def reference_windows(onsets, support, *, start, stop):
    """Conserve all parent onsets and independently validate support and timing."""
    a, b = _ns(support.start), _ns(support.end_exclusive)
    require(len(a) > 0 and np.all(a < b) and np.all(a[1:] >= b[:-1]),
            'Support must be ordered and disjoint')
    t, td = _ns(onsets.bin_start), _ns(onsets.available_at)
    ids = onsets.episode_id.to_numpy()
    sid = onsets.support_id.to_numpy()
    for name in ('episode_id', 'support_id'):
        require(pd.api.types.is_integer_dtype(onsets[name])
                and not pd.api.types.is_bool_dtype(onsets[name])
                and not onsets[name].isna().any(), 'Integer identities required')
    require(np.all(ids >= 0) and len(np.unique(ids)) == len(ids), 'Invalid episode identity')
    require(np.all(np.diff(t) > 0) and np.all(t % SECOND == 0), 'Invalid chronological onset bins')
    np.testing.assert_array_equal(td, t + SECOND, err_msg='Detection time')
    require(np.all(t >= pd.Timestamp(start).value) and np.all(td <= pd.Timestamp(stop).value),
            'Onset falls outside the registered panel')
    require(np.all((sid >= 0) & (sid < len(a))), 'Onset has invalid support identity')
    require(np.all(t >= a[sid]) and np.all(td <= b[sid]), 'Onset bin leaves registered support')
    for field in ('onset_confirmed', 'state_history_available', 'state_sample_eligible'):
        require(pd.api.types.is_bool_dtype(onsets[field]) and not onsets[field].isna().any(),
                'Complete Boolean parent flag required: ' + field)
    require(np.array_equal(onsets.state_sample_eligible,
                           onsets.onset_confirmed & onsets.state_history_available),
            'State eligibility contradicts confirmation/history')
    return pd.DataFrame({'window_id': ids.astype('int64'), 'kind': 'chronological_onset',
        'end_at': pd.to_datetime(t, utc=True), 'episode_id': ids.astype('int64'),
        'onset_confirmed': onsets.onset_confirmed.to_numpy(dtype=bool),
        'state_sample_eligible': onsets.state_sample_eligible.to_numpy(dtype=bool),
        'support_id': sid.astype('int64'),
        'inherited_state_available': onsets.state_history_available.to_numpy(dtype=bool),
        'reference_rv_bps': onsets.pre_rv_300s_5s_bps.to_numpy(dtype=float)})


def reference_measurements(windows, quotes, support, *, batch_size=613):
    """Use backward joins and nested grids; never use subtractive rolling sums."""
    a, b = _ns(support.start), _ns(support.end_exclusive)
    end = _ns(windows.end_at)
    support_id = windows.support_id.to_numpy(dtype='int64')
    pieces = []
    for lo, hi, stamps, logs, ages in _quote_grid_batches(windows.end_at, quotes, batch_size):
        sid, endpoint = support_id[lo:hi], end[lo:hi]
        first, last = a[np.maximum(sid, 0)], b[np.maximum(sid, 0)]
        absent = stamps[:, 0] == np.iinfo('int64').min
        status = np.select([sid < 0,
                            (endpoint - 300 * SECOND < first) | (endpoint > last),
                            absent, stamps[:, 0] < first], STATUSES[1:], default='available')
        good = status == 'available'
        targets = endpoint[:, None] + np.arange(-300, 1)[None, :] * SECOND
        require(np.all(stamps[good] >= first[good, None])
                and np.all(stamps[good] < targets[good]),
                'Used quote lies outside support or is not strictly prior')
        common_age = np.max(ages, axis=1)
        for frequency in FREQUENCIES:
            qgrid, lgrid = stamps[:, ::frequency], logs[:, ::frequency]
            returns = lgrid[:, 1:] - lgrid[:, :-1]
            variance = np.add.reduce(np.square(returns), axis=1) * 100_000_000.
            grid_age = np.max(ages[:, ::frequency], axis=1)
            out = windows.iloc[lo:hi][['window_id', 'kind', 'end_at', 'episode_id',
                                      'onset_confirmed', 'state_sample_eligible']].copy()
            out['frequency_sec'] = frequency
            out['n_returns'] = 300 // frequency
            out['status'] = status
            values = {'rv_bps2': variance, 'rv_bps': np.sqrt(variance),
                      'max_mark_age_sec': grid_age,
                      'common_max_mark_age_sec': common_age,
                      'repeat_fraction': np.count_nonzero(qgrid[:, 1:] == qgrid[:, :-1], axis=1) / (300 // frequency),
                      'zero_return_fraction': np.count_nonzero(returns == 0, axis=1) / (300 // frequency)}
            for name, values in values.items():
                out[name] = np.where(good, values, np.nan)
            out['first_quote_at'] = pd.to_datetime(np.where(good, stamps[:, 0], np.iinfo('int64').min), utc=True)
            out['last_quote_at'] = pd.to_datetime(np.where(good, stamps[:, -1], np.iinfo('int64').min), utc=True)
            for cap in CAPS:
                out['own_age_le_' + str(cap)] = good & (grid_age <= cap)
                out['common_age_le_' + str(cap)] = good & (common_age <= cap)
            pieces.append(out)
    require(bool(pieces), 'Reference measurement requires at least one registered onset')
    result = pd.concat(pieces, ignore_index=True).sort_values(['window_id', 'frequency_sec']).reset_index(drop=True)
    five = result.loc[result.frequency_sec.eq(5)].reset_index(drop=True)
    ordered = windows.sort_values('window_id').reset_index(drop=True)
    inherited = ordered.inherited_state_available.to_numpy(dtype=bool)
    require(five.loc[inherited, 'status'].eq('available').all(),
            'Inherited available variation cannot be reconstructed')
    np.testing.assert_allclose(five.loc[inherited, 'rv_bps'], ordered.loc[inherited, 'reference_rv_bps'],
                               rtol=1e-9, atol=1e-9, err_msg='Inherited five-second state')
    return result


def reference_direction_masks(onsets, ticks):
    result = reference_direction(onsets, ticks)
    for threshold in (1, 5, 10):
        result[f'ratio_gt_{threshold}pct'] = result.cancellation_ratio.gt(threshold / 100)
    return result


def selected_indices(group, cohort, sample):
    """Use explicit index sets, independent of production Boolean composition."""
    population = set(group.index[group.onset_confirmed if cohort == 'confirmed_onsets'
                                 else group.state_sample_eligible])
    available = set(group.index[group.status.eq('available')])
    selected = population & available
    if sample != 'all_available':
        cap = sample.rsplit('_', 1)[1]
        selected &= set(group.index[group['common_age_le_' + cap]])
        own = set(group.index[group['own_age_le_' + cap]])
    else:
        own = available
    return population, available, selected, own


def reference_summary_row(group, cohort, sample):
    population, available, selected, own = selected_indices(group, cohort, sample)
    chosen = group.loc[sorted(selected)]
    out = {'sample': sample, 'windows': len(group), 'cohort_windows': len(population),
           'available_windows': len(population & available), 'selected_windows': len(selected),
           'own_grid_age_pass_windows': len(population & own),
           'zero_variance_windows': int(np.count_nonzero(chosen.rv_bps2.to_numpy() == 0))}
    for name, column in [('mean_variance_bps2', 'rv_bps2'), ('mean_rv_bps', 'rv_bps'),
                         ('mean_repeat_fraction', 'repeat_fraction'),
                         ('mean_zero_return_fraction', 'zero_return_fraction')]:
        out[name] = float(np.mean(chosen[column].to_numpy())) if len(chosen) else np.nan
    out['median_rv_bps'] = _quantile(chosen.rv_bps, .5)
    out['median_max_mark_age_sec'] = _quantile(chosen.max_mark_age_sec, .5)
    out['p95_max_mark_age_sec'] = _quantile(chosen.max_mark_age_sec, .95)
    return out


def reference_summaries(measurements):
    frame = measurements.copy()
    frame['endpoint_date'] = pd.to_datetime(frame.end_at.dt.strftime('%Y-%m-%d'), utc=True)
    frame['endpoint_hour_utc'] = frame.end_at.dt.hour
    tables = {}
    for name, keys in [('summary', ['kind', 'frequency_sec']),
                       ('daily', ['kind', 'endpoint_date', 'frequency_sec']),
                       ('hourly', ['kind', 'endpoint_hour_utc', 'frequency_sec'])]:
        rows = []
        for key, group in frame.groupby(keys, sort=True):
            identity = dict(zip(keys, key))
            for cohort in COHORTS:
                for sample in SAMPLES:
                    rows.append({**identity, 'cohort': cohort,
                                 **reference_summary_row(group, cohort, sample)})
        tables[name] = pd.DataFrame(rows)
    rows = []
    for kind in sorted(frame.kind.unique()):
        group = frame.loc[frame.kind.eq(kind)]
        for cohort in COHORTS:
            for sample in SAMPLES:
                selected = group.loc[sorted(selected_indices(group, cohort, sample)[2])]
                ref = selected.loc[selected.frequency_sec.eq(5)].set_index('window_id')
                for frequency in FREQUENCIES:
                    candidate = selected.loc[selected.frequency_sec.eq(frequency)].set_index('window_id')
                    ids = sorted(set(ref.index) & set(candidate.index))
                    require(len(ids) == len(ref) == len(candidate),
                            'Frequency comparison did not conserve common window IDs')
                    x, y = candidate.loc[ids, 'rv_bps'].to_numpy(), ref.loc[ids, 'rv_bps'].to_numpy()
                    vx, vy = candidate.loc[ids, 'rv_bps2'].to_numpy(), ref.loc[ids, 'rv_bps2'].to_numpy()
                    rx, ry = _average_ranks(x), _average_ranks(y)
                    correlation = np.nan
                    if len(ids) >= 2:
                        rx, ry = rx - rx.mean(), ry - ry.mean()
                        denominator = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
                        if denominator > 0:
                            correlation = float(np.dot(rx, ry) / denominator)
                    ref_mean = float(np.mean(vy)) if len(ids) else np.nan
                    rows.append({'kind': kind, 'cohort': cohort, 'sample': sample,
                        'frequency_sec': frequency, 'reference_frequency_sec': 5,
                        'paired_windows': len(ids), 'zero_variance_windows': int(np.count_nonzero(vx == 0)),
                        'reference_zero_variance_windows': int(np.count_nonzero(vy == 0)),
                        'mean_variance_ratio_to_5s': float(np.mean(vx) / ref_mean) if ref_mean > 0 else np.nan,
                        'mean_absolute_rv_difference_bps': float(np.mean(np.abs(x - y))) if len(ids) else np.nan,
                        'spearman_rv_to_5s': correlation})
    tables['comparisons'] = pd.DataFrame(rows)
    return tables


def reference_calendar(onsets, measurements, direction, *, start, stop):
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    require(start == start.normalize() and stop == stop.normalize(), 'Whole UTC calendar dates required')
    count = (stop.value - start.value) // (86400 * SECOND)
    dates = pd.date_range(start, stop, freq='D', inclusive='left')
    output = pd.DataFrame({'date': dates})
    day = (_ns(onsets.bin_start) - start.value) // (86400 * SECOND)
    require(np.all((day >= 0) & (day < count)), 'Onset outside calendar')
    five = measurements.loc[measurements.frequency_sec.eq(5)].sort_values('episode_id').reset_index(drop=True)
    directions = direction.sort_values('episode_id').reset_index(drop=True)
    ordered = onsets.sort_values('episode_id').reset_index(drop=True)
    np.testing.assert_array_equal(ordered.episode_id, five.episode_id)
    np.testing.assert_array_equal(ordered.episode_id, directions.episode_id)
    day = (_ns(ordered.bin_start) - start.value) // (86400 * SECOND)
    eligible = ordered.state_sample_eligible.to_numpy(dtype=bool)
    directed = eligible & directions.inherited_direction.ne(0).to_numpy()
    masks = {'all_onsets': np.ones(len(ordered), dtype=bool),
             'confirmed_onsets': ordered.onset_confirmed.to_numpy(dtype=bool),
             'state_history_available': ordered.state_history_available.to_numpy(dtype=bool),
             'state_sample_eligible': eligible,
             'variation_available': five.status.eq('available').to_numpy()}
    masks.update({f'common_age_le_{cap}': eligible & five[f'common_age_le_{cap}'].to_numpy()
                  for cap in CAPS})
    masks['direction_nonzero'] = directed
    masks.update({f'direction_ratio_gt_{threshold}pct': directed & directions[f'ratio_gt_{threshold}pct'].to_numpy()
                  for threshold in (1, 5, 10)})
    for name, mask in masks.items():
        output[name] = np.bincount(day[mask], minlength=int(count)).astype('int64')
    return output


def records(frame):
    rows = []
    for record in frame.to_dict(orient='records'):
        out = {}
        for key, value in record.items():
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.isoformat()
            elif isinstance(value, np.generic):
                value = value.item()
            out[key] = value
        rows.append(out)
    return rows


def verify_statistics(actual, onsets, measurements, direction, tables, calendar, *, start, stop):
    """Audit every saved scalar and summary using independently rebuilt values."""
    ordered = onsets.sort_values('episode_id').reset_index(drop=True)
    five = measurements.loc[measurements.frequency_sec.eq(5)].sort_values('episode_id').reset_index(drop=True)
    np.testing.assert_array_equal(ordered.episode_id, five.episode_id)
    history = ordered.state_history_available.to_numpy(dtype=bool)
    eligible = ordered.state_sample_eligible.to_numpy(dtype=bool)
    available = five.status.eq('available').to_numpy()
    differences = np.abs(five.rv_bps.to_numpy()[history] - ordered.pre_rv_300s_5s_bps.to_numpy()[history])
    statuses = {}
    for frequency in FREQUENCIES:
        values = measurements.loc[measurements.frequency_sec.eq(frequency), 'status'].to_numpy()
        names, counts = np.unique(values, return_counts=True)
        statuses[str(frequency)] = {str(k): int(v) for k, v in zip(names, counts)}
    expected = {'scope': SCOPE, 'primary_policy': 'structural_only',
        'deployment_start': pd.Timestamp(start).isoformat(), 'deployment_end_exclusive': pd.Timestamp(stop).isoformat(),
        'onsets': len(ordered), 'confirmed_onsets': int(np.count_nonzero(ordered.onset_confirmed)),
        'uncertain_onsets': int(np.count_nonzero(~ordered.onset_confirmed)),
        'inherited_history_available': int(np.count_nonzero(history)), 'state_sample_eligible': int(np.count_nonzero(eligible)),
        'measurement_rows': len(measurements), 'frequencies_seconds': list(FREQUENCIES),
        'status_counts_by_frequency': statuses, 'raw_price_available_onsets': int(np.count_nonzero(available)),
        'raw_price_available_missing_state_history': int(np.count_nonzero(available & ~history)),
        'five_second_reconciliation': {'rows': int(np.count_nonzero(history)),
            'max_absolute_difference_bps': max(differences, default=0.), 'rtol': 1e-9, 'atol': 1e-9},
        'common_age_state_counts': {str(c): int(np.count_nonzero(eligible & five[f'common_age_le_{c}'])) for c in CAPS},
        'own5_age_state_counts': {str(c): int(np.count_nonzero(eligible & five[f'own_age_le_{c}'])) for c in CAPS},
        'summary': records(tables['summary']), 'comparisons': records(tables['comparisons']),
        'direction_summary': reference_direction_summary(direction),
        'direction_mask_state_counts': {str(c): int(np.sum(calendar[f'direction_ratio_gt_{c}pct'])) for c in (1, 5, 10)},
        'calendar': {'days': len(calendar), 'days_with_onsets': int(np.count_nonzero(calendar.all_onsets)),
            'days_with_state_eligible_onsets': int(np.count_nonzero(calendar.state_sample_eligible)),
            'totals': {k: int(np.sum(calendar[k])) for k in calendar if k != 'date'}},
        'baseline_references_preserved': True, 'detector_fits': 0, 'empirical_response_models_fitted': 0,
        'price_responses_computed': False, 'detector_support_sensitivities_extended': False,
        'production_inference_approved': False, 'downstream_ready': False}
    _json_compare(actual, expected, exact=True)


def verify_panel_measurement(run):
    """Reconstruct every current output from verified sources in bounded cycles."""
    run = Path(run)
    manifest, parent, counts = registered_hashes(run)
    _json_compare(manifest, {'stage': STAGE, 'scope': SCOPE,
        'primary_policy': 'structural_only', 'frequencies_seconds': list(FREQUENCIES),
        'age_caps_seconds': list(CAPS), 'window_seconds': 300,
        'direction_ratio_strict_thresholds': [0.01, 0.05, 0.10],
        'deployment_start': parent['first_cutoff'], 'deployment_end_exclusive': parent['deployment_end'],
        'production_inference_approved': False, 'downstream_ready': False})
    inputs = manifest['inputs']
    onsets = pd.read_parquet(inputs['onsets']['path'])
    support = pd.read_parquet(inputs['support_intervals']['path'],
                              filters=[('policy', '==', 'structural_only')]).reset_index(drop=True)
    schedule = pd.read_parquet(inputs['cycle_schedule']['path'])
    start, stop = pd.Timestamp(parent['first_cutoff']), pd.Timestamp(parent['deployment_end'])
    require(len(schedule) > 0 and schedule.parameter_cutoff.iloc[0] == start
            and schedule.deployment_end.iloc[-1] == stop,
            'Cycle schedule fails full-calendar boundaries')
    np.testing.assert_array_equal(_ns(schedule.deployment_end.iloc[:-1]),
                                  _ns(schedule.parameter_cutoff.iloc[1:]),
                                  err_msg='Cycle gaps or overlaps')
    require((schedule.deployment_end > schedule.parameter_cutoff).all(), 'Nonpositive cycle length')
    require(not schedule.cycle_id.duplicated().any(), 'Duplicate cycle identity')
    required_outputs = {'windows.parquet', 'measurements.parquet', 'direction.parquet', 'calendar.parquet',
                        'summary.parquet', 'daily.parquet', 'hourly.parquet', 'comparisons.parquet',
                        'statistics.json', 'test_evidence.json'}
    for cycle_id in schedule.cycle_id:
        required_outputs.update(f'cycles/{cycle_id:03d}/{name}' for name in
                                ('measurements.parquet', 'direction.parquet', 'statistics.json'))
    require(required_outputs.issubset(manifest['outputs']), 'Unregistered numerical output')
    windows = reference_windows(onsets, support, start=start, stop=stop)
    compare(pd.read_parquet(run / 'windows.parquet'), windows, 'Full onset windows')
    stored_measurements = pd.read_parquet(run / 'measurements.parquet')
    stored_direction = pd.read_parquet(run / 'direction.parquet')
    require(not stored_measurements.duplicated(['window_id', 'frequency_sec']).any(),
            'Duplicate measurement identities')
    require(len(stored_measurements) == len(onsets) * len(FREQUENCIES),
            'Three measurement rows per onset were not conserved')
    require(len(stored_direction) == len(onsets) and not stored_direction.episode_id.duplicated().any(),
            'Direction onset identities were not conserved')
    reference_parts, direction_parts, visited = [], [], []
    for cycle in schedule.itertuples(index=False):
        mask = onsets.bin_start.ge(cycle.parameter_cutoff) & onsets.bin_start.lt(cycle.deployment_end)
        selected = onsets.loc[mask].reset_index(drop=True)
        selected_windows = windows.loc[mask.to_numpy()].reset_index(drop=True)
        if not len(selected):
            directory = run / 'cycles' / f'{cycle.cycle_id:03d}'
            require(len(pd.read_parquet(directory / 'measurements.parquet')) == 0
                    and len(pd.read_parquet(directory / 'direction.parquet')) == 0,
                    'Empty cycle contains invented rows')
            _json_compare(read(directory / 'statistics.json'), {'cycle_id': int(cycle.cycle_id),
                          'onsets': 0, 'measurement_rows': 0, 'direction_rows': 0}, exact=True)
            print(json.dumps({'verify_measurement_cycle': int(cycle.cycle_id), 'onsets': 0}), flush=True)
            continue
        quotes = quote_window(inputs['clean_quotes']['path'],
                              pd.Timestamp(cycle.parameter_cutoff) - pd.Timedelta(seconds=300),
                              pd.Timestamp(cycle.deployment_end))
        expected = reference_measurements(selected_windows, quotes, support)
        del quotes
        ids = set(selected.episode_id)
        actual = stored_measurements.loc[stored_measurements.episode_id.isin(ids)].reset_index(drop=True)
        compare(actual, expected, f'Cycle {cycle.cycle_id} measurements')
        ticks = pd.read_parquet(inputs['ofi_ticks']['path'],
            columns=['timestamp', 'exclusion_code', 'quote_ofi'],
            filters=[('timestamp', '>=', cycle.parameter_cutoff),
                     ('timestamp', '<', cycle.deployment_end)])
        expected_direction = reference_direction_masks(selected, ticks)
        del ticks
        actual_direction = stored_direction.loc[stored_direction.episode_id.isin(ids)].reset_index(drop=True)
        compare(actual_direction, expected_direction, f'Cycle {cycle.cycle_id} direction')
        directory = run / 'cycles' / f'{cycle.cycle_id:03d}'
        compare(pd.read_parquet(directory / 'measurements.parquet'), expected, f'Cycle {cycle.cycle_id} saved measurements')
        compare(pd.read_parquet(directory / 'direction.parquet'), expected_direction, f'Cycle {cycle.cycle_id} saved direction')
        _json_compare(read(directory / 'statistics.json'), {'cycle_id': int(cycle.cycle_id),
                      'onsets': len(selected), 'measurement_rows': len(expected),
                      'direction_rows': len(expected_direction)}, exact=True)
        reference_parts.append(expected)
        direction_parts.append(expected_direction)
        visited.extend(selected.episode_id.to_list())
        print(json.dumps({'verify_measurement_cycle': int(cycle.cycle_id),
                          'onsets': len(selected), 'measurement_rows': len(expected)}), flush=True)
    np.testing.assert_array_equal(visited, onsets.episode_id, err_msg='Cycle population conservation')
    expected_measurements = pd.concat(reference_parts, ignore_index=True)
    expected_direction = pd.concat(direction_parts, ignore_index=True)
    compare(stored_measurements, expected_measurements, 'Combined measurements and order')
    compare(stored_direction, expected_direction, 'Combined direction and order')
    tables = reference_summaries(expected_measurements)
    for name, table in tables.items():
        compare(pd.read_parquet(run / (name + '.parquet')), table, name + ' table')
    calendar = reference_calendar(onsets, expected_measurements, expected_direction, start=start, stop=stop)
    compare(pd.read_parquet(run / 'calendar.parquet'), calendar, 'Full calendar including empty dates')
    # the root statistics are checked below against the same independently
    # reconstructed tables; saved measurements are never used as their oracle.
    statistics = read(run / 'statistics.json')
    verify_statistics(statistics, onsets, expected_measurements, expected_direction,
                      tables, calendar, start=start, stop=stop)
    return {'status': 'passed', **counts, 'cycles_verified': len(schedule),
        'onset_windows_verified': len(windows), 'measurement_rows_verified': len(expected_measurements),
        'direction_rows_verified': len(expected_direction), 'calendar_days_verified': len(calendar),
        'summary_rows_verified': len(tables['summary']), 'daily_rows_verified': len(tables['daily']),
        'hourly_rows_verified': len(tables['hourly']), 'comparison_rows_verified': len(tables['comparisons']),
        'scope': SCOPE, 'production_inference_approved': False,
        'methods': ['strict backward quote joins on every one-second boundary',
                    'direct three-frequency squared-return sums',
                    'calendar-bin regrouping and stable contribution sums',
                    'independent cohort sets, tied ranks and integer-day calendar counts',
                    'verified parent links and complete registered artifact hashes']}

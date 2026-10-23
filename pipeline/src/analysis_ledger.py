"""Stage 06c: fixed observational cohort and conservative dependence accounting.

No fitting, causal labels, volatility normalization, or independent-event inference.
The cohort flag uses detection-time direction; state values precede the onset.
"""
import numpy as np
import pandas as pd

HORIZONS = (0, 60, 300)
FREQUENCIES = (5, 15, 30)
CAPS = (15, 30, 60)
PRE_FIELDS = ('pre_rv_300s_5s_bps', 'pre_quote_ofi_300s', 'pre_spread_pips',
    'pre_displayed_size', 'pre_quote_age_sec', 'state_history_available',
    'state_sample_eligible', 'pre_mark_max_age_sec', 'pre_mark_repeat_fraction',
    'pre_recorded_updates_300s', 'pre_max_observed_silence_sec')
DETECTION_FIELDS = ('quote_ofi', 'distance', 'z_ofi', 'z_intensity', 'z_duration',
    'direction', 'direction_available', 'onset_confirmed')
ANCHOR_FIELDS = ('episode_id', 'bin_start', 'available_at', 'support_id',
    *PRE_FIELDS, *DETECTION_FIELDS, 'pre_mark_at', 'detection_mark_at')
PATH_FIELDS = ('episode_id', 'bin_start', 'detected_at', 'horizon_sec', 'target_at',
    'endpoint_status', 'endpoint_quote_at', 'endpoint_age_sec', 'pre_age_sec',
    'detection_age_sec', 'initial_pips', 'continuation_pips', 'total_pips',
    'continuation_quote_available', 'total_quote_available', 'triplet_quote_available',
    'triplet_age_le_15', 'triplet_age_le_30', 'triplet_age_le_60')


def specification():
    roles = {x: 'strictly_before_onset' for x in PRE_FIELDS}
    roles.update({x: 'detection_time' for x in DETECTION_FIELDS})
    roles.update({x: 'identity_or_calendar' for x in ('episode_id', 'bin_start',
        'available_at', 'support_id', 'onset_day', 'onset_week_start', 'onset_hour_utc',
        'onset_weekday_utc', 'onset_slot_15m_utc')})
    roles.update({x: 'source_accounting_not_predictor' for x in ('pre_mark_at',
        'detection_mark_at', 'nominal_start_at', 'source_start_at')})
    for freq in FREQUENCIES:
        for x in ('rv_bps', 'status', 'first_quote_at', 'last_quote_at'):
            roles[f'pre_{x}_{freq}s'] = 'strictly_before_onset'
    for cap in CAPS:
        roles[f'pre_common_age_le_{cap}'] = 'strictly_before_onset'
    roles['cancellation_ratio'] = 'detection_time'
    roles['pre_state_eligible'] = 'detection_time_cohort_flag_not_predictor'
    roles['state_sample_eligible'] = 'detection_time_cohort_flag_not_predictor'
    roles['state_history_available'] = 'historical_support_accounting_not_predictor'
    return {'version': '06c.1', 'policy': 'structural_only', 'window_seconds': 300,
        'working_state': 'pre_rv_bps_5s', 'units': 'bps; square root of summed squared log returns',
        'frequency_sensitivities_seconds': [15, 30], 'age_sensitivities_seconds': list(CAPS),
        'direction_ratio_strict_thresholds': [0.01, 0.05, 0.10],
        'primary_outcome': 'continuation_pips at horizon_sec=300',
        'secondary_horizon_seconds': 60, 'identity_horizon_seconds': 0,
        'estimand': 'conditional mean recorded continuation within selected directed onsets',
        'pre_state_eligible_role': 'cohort at detection, not a strictly pre-onset predictor',
        'response_age_rule': 'three marks; future endpoint quality is not ex-ante eligibility',
        'interval': '[earliest actual state/pre/detection quote, detection+horizon)',
        'week_convention': 'preceding Sunday 00:00 UTC; reporting label only',
        'inference_approved': False, 'latent_volatility_validated': False,
        'causal_identification': False, 'anchor_column_roles': roles}


def _require(frame, fields, label):
    missing = set(fields) - set(frame)
    if missing:
        raise ValueError(f'{label}: missing columns {sorted(missing)}')


def _keys(frame, keys, label):
    if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError(f'{label}: missing or duplicate keys')
    if 'episode_id' in keys:
        ids = frame.episode_id
        if (not pd.api.types.is_numeric_dtype(ids.dtype) or pd.api.types.is_bool_dtype(ids.dtype)
                or not np.isfinite(ids).all() or not ids.ge(0).all()
                or not ids.eq(np.floor(ids)).all() or not ids.le(np.iinfo(np.int64).max).all()):
            raise ValueError(f'{label}: nonnegative integer episode IDs required')


def _utc(values, label, missing=False):
    idx = pd.DatetimeIndex(values)
    if str(idx.tz) != 'UTC' or (idx.hasnans and not missing):
        raise ValueError(label + ': UTC timestamps required')


def _same(left, right, label):
    if len(left) != len(right) or not np.array_equal(np.asarray(left), np.asarray(right)):
        raise ValueError(label + ': inconsistent parent identity or timing')


def _bools(frame, fields, label):
    for name in fields:
        if frame[name].isna().any() or not pd.api.types.is_bool_dtype(frame[name].dtype):
            raise ValueError(f'{label}: {name} must contain nonmissing booleans')


def assemble_ledger(anchors, measurements, direction, paths):
    """Join verified parents without changing their values or full path source."""
    _require(anchors, (*ANCHOR_FIELDS, 'baseline_quote_available'), 'anchors')
    _require(measurements, ('kind', 'episode_id', 'frequency_sec', 'end_at',
        'onset_confirmed', 'rv_bps', 'status', 'first_quote_at', 'last_quote_at',
        *[f'common_age_le_{cap}' for cap in CAPS]), 'measurements')
    _require(direction, ('episode_id', 'bin_start', 'onset_confirmed',
        'inherited_direction', 'inherited_net', 'cancellation_ratio'), 'direction')
    _require(paths, (*PATH_FIELDS, 'direction', 'direction_available',
        'state_sample_eligible', 'onset_confirmed'), 'paths')
    _keys(anchors, ['episode_id'], 'anchors')
    _keys(direction, ['episode_id'], 'direction')
    _keys(paths, ['episode_id', 'horizon_sec'], 'paths')
    _bools(anchors, ['onset_confirmed', 'direction_available', 'state_history_available',
        'state_sample_eligible', 'baseline_quote_available'], 'anchors')
    _bools(direction, ['onset_confirmed'], 'direction')
    _bools(paths, ['direction_available', 'state_sample_eligible', 'onset_confirmed',
        'continuation_quote_available', 'total_quote_available', 'triplet_quote_available',
        *[f'triplet_age_le_{cap}' for cap in CAPS]], 'paths')
    a = anchors.loc[:, ANCHOR_FIELDS].sort_values('episode_id').reset_index(drop=True).copy()
    _utc(a.bin_start, 'onset'); _utc(a.available_at, 'detection')
    if not a.available_at.eq(a.bin_start + pd.Timedelta(seconds=1)).all():
        raise ValueError('Detection must be end of one-second onset bin')
    for col, bound in [('pre_mark_at', a.bin_start), ('detection_mark_at', a.available_at)]:
        _utc(a[col], col, missing=True)
        if not (a[col].isna() | a[col].lt(bound)).all():
            raise ValueError(col + ': not strictly prior to boundary')
    dev = measurements.loc[measurements.kind.eq('development')].copy()
    _keys(dev, ['episode_id', 'frequency_sec'], 'development measurements')
    _bools(dev, ['onset_confirmed', *[f'common_age_le_{cap}' for cap in CAPS]], 'measurements')
    for freq in FREQUENCIES:
        m = dev.loc[dev.frequency_sec.eq(freq)].sort_values('episode_id').reset_index(drop=True)
        _same(a.episode_id, m.episode_id, 'measurement IDs')
        _same(a.bin_start, m.end_at, 'measurement endpoints')
        _same(a.onset_confirmed, m.onset_confirmed, 'measurement confirmation')
        _utc(m.first_quote_at, 'first state quote', missing=True)
        _utc(m.last_quote_at, 'last state quote', missing=True)
        available = m.status.eq('available')
        if not ((m.first_quote_at.lt(a.bin_start-pd.Timedelta(seconds=300)) &
                 m.last_quote_at.lt(a.bin_start) & m.first_quote_at.le(m.last_quote_at)) | ~available).all():
            raise ValueError('Available state quotes violate strictly prior window boundaries')
        for field in ('rv_bps', 'status', 'first_quote_at', 'last_quote_at'):
            a[f'pre_{field}_{freq}s'] = m[field]
        if freq == 5:
            for cap in CAPS:
                a[f'pre_common_age_le_{cap}'] = m[f'common_age_le_{cap}'].astype(bool)
        else:
            for cap in CAPS:
                _same(a[f'pre_common_age_le_{cap}'], m[f'common_age_le_{cap}'], 'common-grid masks')
    d = direction.sort_values('episode_id').reset_index(drop=True)
    _same(a.episode_id, d.episode_id, 'direction IDs')
    _same(a.bin_start, d.bin_start, 'direction endpoints')
    _same(a.onset_confirmed, d.onset_confirmed, 'direction confirmation')
    # keep nonfinite directions ineligible; do not replace inherited signs
    if not np.allclose(a.direction, d.inherited_direction, rtol=0, atol=0, equal_nan=True):
        raise ValueError('Inherited direction changed')
    if not np.allclose(a.quote_ofi, d.inherited_net, rtol=0, atol=0, equal_nan=True):
        raise ValueError('Inherited onset imbalance changed')
    a['cancellation_ratio'] = d.cancellation_ratio
    if ((a.cancellation_ratio.dropna() < 0) | (a.cancellation_ratio.dropna() > 1)).any():
        raise ValueError('Cancellation ratio outside [0,1]')
    ref = a.pre_rv_bps_5s
    inherited = a.pre_rv_300s_5s_bps
    valid_inherited = a.state_sample_eligible & np.isfinite(inherited)
    if not np.allclose(ref[valid_inherited], inherited[valid_inherited], rtol=1e-12, atol=1e-9):
        raise ValueError('Working five-second reference differs from inherited state')
    valid_direction = np.isfinite(a.direction) & a.direction.ne(0) & a.direction_available
    if (np.isfinite(a.direction) & ~a.direction.isin([-1, 0, 1])).any():
        raise ValueError('Direction must be a sign')
    a['pre_state_eligible'] = (a.onset_confirmed & valid_direction & a.state_sample_eligible &
        a.pre_status_5s.eq('available') & np.isfinite(ref) & ref.ge(0) &
        np.isfinite(a.pre_quote_ofi_300s))
    a['onset_day'] = a.bin_start.dt.floor('D')
    a['onset_week_start'] = a.onset_day - pd.to_timedelta((a.bin_start.dt.dayofweek + 1) % 7, unit='D')
    a['onset_hour_utc'] = a.bin_start.dt.hour.astype('int64')
    a['onset_weekday_utc'] = a.bin_start.dt.dayofweek.astype('int64')
    a['onset_slot_15m_utc'] = (a.bin_start.dt.hour*4 + a.bin_start.dt.minute//15).astype('int64')
    a['nominal_start_at'] = a.bin_start - pd.Timedelta(seconds=300)
    # use the same oldest boundary for all grids; include every selected source
    source_cols = [f'pre_first_quote_at_{f}s' for f in FREQUENCIES] + ['pre_mark_at', 'detection_mark_at']
    a['source_start_at'] = a[source_cols].min(axis=1)
    if a.loc[a.pre_state_eligible, 'source_start_at'].isna().any():
        raise ValueError('Eligible state has no actual source start')
    p = paths.loc[paths.horizon_sec.isin(HORIZONS)].sort_values(['episode_id', 'horizon_sec']).reset_index(drop=True)
    expected = pd.MultiIndex.from_product([a.episode_id, HORIZONS])
    if list(zip(p.episode_id, p.horizon_sec)) != list(expected):
        raise ValueError('Exactly one path per anchor and selected horizon required')
    aa = a.set_index('episode_id').reindex(p.episode_id).reset_index(drop=True)
    for key, other in [('bin_start', 'bin_start'), ('detected_at', 'available_at'),
                       ('direction_available', 'direction_available'), ('state_sample_eligible', 'state_sample_eligible'),
                       ('onset_confirmed', 'onset_confirmed')]:
        _same(p[key], aa[other], 'path ' + key)
    if not np.allclose(p.direction, aa.direction, rtol=0, atol=0, equal_nan=True):
        raise ValueError('Path direction changed')
    _utc(p.target_at, 'target'); _utc(p.endpoint_quote_at, 'endpoint quote', missing=True)
    if not p.target_at.eq(p.detected_at + pd.to_timedelta(p.horizon_sec, unit='s')).all():
        raise ValueError('Path target mismatch')
    if not (p.endpoint_quote_at.isna() | p.endpoint_quote_at.lt(p.target_at)).all():
        raise ValueError('Endpoint quote is not strictly prior')
    o = p.loc[:, PATH_FIELDS].copy()
    o['pre_state_eligible'] = aa.pre_state_eligible
    baseline = anchors.set_index('episode_id').baseline_quote_available.reindex(o.episode_id).to_numpy(dtype=bool)
    o['initial_quote_available'] = baseline
    for component, flag in [('initial', 'initial_quote_available'),
                            ('continuation', 'continuation_quote_available'), ('total', 'total_quote_available')]:
        o[f'eligible_{component}'] = o.pre_state_eligible & o[flag] & np.isfinite(o[f'{component}_pips'])
    o['eligible_joint'] = o.eligible_initial & o.eligible_continuation & o.eligible_total
    for freq in (15, 30):
        x = aa[f'pre_rv_bps_{freq}s']
        o[f'eligible_frequency_{freq}'] = o.eligible_continuation & aa[f'pre_status_{freq}s'].eq('available') & np.isfinite(x) & x.ge(0)
    for cap in CAPS:
        o[f'eligible_pre_age_{cap}'] = o.eligible_continuation & aa[f'pre_common_age_le_{cap}']
        o[f'eligible_response_age_{cap}'] = o.eligible_continuation & o[f'triplet_age_le_{cap}']
        o[f'eligible_both_age_{cap}'] = o[f'eligible_pre_age_{cap}'] & o[f'eligible_response_age_{cap}']
    for pct in (1, 5, 10):
        o[f'eligible_direction_ratio_gt_{pct}pct'] = o.eligible_continuation & aa.cancellation_ratio.gt(pct/100)
    return {'anchors': a, 'outcomes': o}


INTERVAL_COLUMNS = ('episode_id', 'horizon_sec', 'nominal_start_at', 'source_start_at',
    'end_at', 'onset_day', 'onset_week_start', 'degree', 'component_id', 'component_size')
PAIR_COLUMNS = ('horizon_sec', 'episode_id_left', 'episode_id_right', 'overlap_seconds',
    'cross_onset_day', 'cross_onset_week')
COMPONENT_COLUMNS = ('horizon_sec', 'component_id', 'events', 'source_start_at', 'end_at')
CALENDAR_COLUMNS = ('horizon_sec', 'block_type', 'block_start', 'events', 'share')


def dependence_tables(anchors, outcomes):
    """Sweep half-open source intervals; no independence is inferred from components."""
    _keys(anchors, ['episode_id'], 'dependence anchors')
    _keys(outcomes, ['episode_id', 'horizon_sec'], 'dependence outcomes')
    _bools(outcomes, ['eligible_continuation'], 'dependence outcomes')
    all_i, all_p, all_c, all_b = [], [], [], []
    for h in HORIZONS:
        selected = outcomes.loc[outcomes.horizon_sec.eq(h) & outcomes.eligible_continuation]
        a = anchors.set_index('episode_id').loc[selected.episode_id].reset_index().sort_values('source_start_at')
        ends = selected.set_index('episode_id').target_at
        a['end_at'] = a.episode_id.map(ends)
        if a.source_start_at.isna().any() or not a.source_start_at.lt(a.end_at).all():
            raise ValueError('Invalid eligible source interval')
        ids = a.episode_id.astype(int).tolist()
        parent = {i: i for i in ids}; degree = dict.fromkeys(ids, 0)
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        active = []
        for row in a.itertuples(index=False):
            active = [old for old in active if old.end_at > row.source_start_at]
            for old in active:
                if row.end_at <= old.source_start_at:
                    continue
                left, right = sorted([int(old.episode_id), int(row.episode_id)])
                overlap = (min(row.end_at, old.end_at)-max(row.source_start_at, old.source_start_at)).value / 1e9
                all_p.append((h, left, right, overlap, old.onset_day != row.onset_day,
                              old.onset_week_start != row.onset_week_start))
                degree[left] += 1; degree[right] += 1
                rl, rr = root(left), root(right)
                parent[max(rl, rr)] = min(rl, rr)
            active.append(row)
        groups = {}
        for i in ids: groups.setdefault(root(i), []).append(i)
        indexed = a.set_index('episode_id')
        for comp, members in sorted(groups.items()):
            sub = indexed.loc[members]
            all_c.append((h, comp, len(members), sub.source_start_at.min(), sub.end_at.max()))
        for row in a.itertuples(index=False):
            comp = root(int(row.episode_id))
            all_i.append((int(row.episode_id), h, row.nominal_start_at, row.source_start_at,
                          row.end_at, row.onset_day, row.onset_week_start, degree[row.episode_id], comp, len(groups[comp])))
        for label, key in [('day', 'onset_day'), ('week', 'onset_week_start')]:
            for stamp, count in a.groupby(key).size().items():
                all_b.append((h, label, stamp, int(count), float(count/len(a))))
    result = {name: pd.DataFrame(rows, columns=cols) for name, rows, cols in [
        ('intervals', all_i, INTERVAL_COLUMNS), ('pairs', all_p, PAIR_COLUMNS),
        ('components', all_c, COMPONENT_COLUMNS), ('calendar', all_b, CALENDAR_COLUMNS)]}
    for name, fields in [('intervals', ['horizon_sec', 'episode_id']),
                         ('pairs', ['horizon_sec', 'episode_id_left', 'episode_id_right']),
                         ('components', ['horizon_sec', 'component_id']),
                         ('calendar', ['horizon_sec', 'block_type', 'block_start'])]:
        result[name] = result[name].sort_values(fields).reset_index(drop=True)
    return result


def _distribution(x):
    x = pd.Series(x, dtype=float); x = x[np.isfinite(x)]
    return {'n': len(x), 'zero': int(x.eq(0).sum()), 'mean': None if x.empty else float(x.mean()),
        **{name: None if x.empty else float(x.quantile(q)) for name, q in [('p10', .1), ('median', .5), ('p90', .9)]}}


def summarize_ledger(tables, dependence):
    a, o = tables['anchors'], tables['outcomes']
    masks = [name for name in o if name.startswith('eligible_')]
    seq = np.ones(len(a), dtype=bool); attrition = []
    for label, condition in [('all_inherited_onsets', seq.copy()), ('confirmed', a.onset_confirmed),
        ('finite_nonzero_direction', np.isfinite(a.direction) & a.direction.ne(0) & a.direction_available),
        ('inherited_state_sample_eligible', a.state_sample_eligible),
        ('available_finite_nonnegative_reference', a.pre_status_5s.eq('available') & np.isfinite(a.pre_rv_bps_5s) & a.pre_rv_bps_5s.ge(0)),
        ('finite_prior_signed_imbalance', np.isfinite(a.pre_quote_ofi_300s))]:
        before = int(seq.sum()); seq &= np.asarray(condition, dtype=bool)
        attrition.append({'step': label, 'remaining': int(seq.sum()), 'newly_excluded': before-int(seq.sum())})
    horizons = []
    for h in HORIZONS:
        g = o.loc[o.horizon_sec.eq(h)]; joint = g.loc[g.eligible_joint]
        pairs = dependence['pairs'].loc[dependence['pairs'].horizon_sec.eq(h)]
        ints = dependence['intervals'].loc[dependence['intervals'].horizon_sec.eq(h)]
        comps = dependence['components'].loc[dependence['components'].horizon_sec.eq(h)]
        blocks = dependence['calendar'].loc[dependence['calendar'].horizon_sec.eq(h)]
        horizons.append({'horizon_sec': h, 'inherited_onsets': len(g),
            'mask_counts': {key: int(g[key].sum()) for key in masks},
            'unavailable_continuation_after_pre_state': int((g.pre_state_eligible & ~g.eligible_continuation).sum()),
            'joint_component_means_pips': {c: None if joint.empty else float(joint[c+'_pips'].mean()) for c in ('initial', 'continuation', 'total')},
            'continuation_mean_own_sample_pips': None if not g.eligible_continuation.any() else float(g.loc[g.eligible_continuation, 'continuation_pips'].mean()),
            'dependence': {'events': len(ints), 'overlap_pairs': len(pairs),
                'events_with_overlap': int(ints.degree.gt(0).sum()), 'components': len(comps),
                'largest_component_events': 0 if comps.empty else int(comps.events.max()),
                'maximum_degree': 0 if ints.empty else int(ints.degree.max()),
                'source_extended_before_nominal': int(ints.source_start_at.lt(ints.nominal_start_at).sum()),
                'cross_day_pairs': int(pairs.cross_onset_day.sum()), 'cross_week_pairs': int(pairs.cross_onset_week.sum()),
                **{label: {'blocks': len(b), 'largest_share': None if b.empty else float(b.share.max()),
                          'inverse_hhi_descriptive': None if b.empty else float(1/(b.share**2).sum())}
                   for label in ('day', 'week') for b in [blocks.loc[blocks.block_type.eq(label)]]}}})
    eligible = a.loc[a.pre_state_eligible]
    return {'scope': 'reused_development_only', 'primary_policy': 'structural_only',
        'anchor_rows': len(a), 'outcome_rows': len(o), 'primary_horizon_seconds': 300,
        'pre_state_eligible': len(eligible), 'attrition_sequential': attrition,
        'state_distributions_pre_state_cohort': {key: _distribution(eligible[key]) for key in
            ('pre_rv_bps_5s', 'pre_rv_bps_15s', 'pre_rv_bps_30s', 'pre_quote_ofi_300s', 'pre_spread_pips', 'pre_displayed_size', 'cancellation_ratio')},
        'horizons': horizons, 'working_specification_frozen': True, 'production_inference_approved': False,
        'latent_volatility_validated': False, 'comparison_model_fitted': False, 'downstream_ready': False}

"""One frozen quote-pressure sequencing measurement; no response is read or fitted.

The estimator is adjacent sign agreement less its exact random-permutation
composition mean. That mean is a descriptive centering identity, not a null
hypothesis, p-value or claim that quotes are exchangeable.
"""
import math
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
WINDOW = 300
GRID = 5
N = WINDOW // GRID
ORIGIN_COLUMNS = ['episode_id', 'bin_start', 'forecast_at', 'origin_eligible',
                  'origin_exclusion_reason', 'direction', 'initial_pips', 'pre_mark_at']
PREDICTOR_COLUMNS = ['episode_id', 'bin_start', 'forecast_at', 'signed_pressure',
    'pressure_magnitude', 'variation_standardized', 'pre_spread_pips',
    'log_pre_displayed_size', 'onset_hour_utc', 'onset_weekday_utc']
ANCHOR_COLUMNS = ['episode_id', 'bin_start', 'support_id', 'state_history_available',
    'state_sample_eligible', 'pre_quote_ofi_300s', 'pre_rv_300s_5s_bps',
    'pre_recorded_updates_300s']
SPARSE_COLUMNS = ['bin_start', 'n_quotes', 'n_invalid_transitions', 'quote_ofi']
QUOTE_COLUMNS = ['timestamp', 'bid', 'ask']
NUMERIC_BASELINE = ['direction', 'initial_pips', 'aligned_pressure', 'pressure_magnitude',
    'gross_pressure_scaled', 'variation_standardized', 'pre_spread_pips',
    'log_pre_displayed_size', 'log1p_prior_quote_updates', 'aligned_sign_balance',
    'nonzero_fraction', 'composition_reference', 'aligned_recent_5s_pressure',
    'aligned_recent_30s_pressure', 'recent_5s_gross_scaled', 'recent_30s_gross_scaled',
    'aligned_prior_5s_price_pips', 'aligned_prior_30s_price_pips', 'aligned_prior_300s_price_pips']
CATEGORICAL_BASELINE = ['onset_hour_utc', 'onset_weekday_utc']
Q_COLUMNS = [f'q5_{k:02d}' for k in range(N)]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def ns(values):
    times = pd.DatetimeIndex(values)
    require(str(times.tz) == 'UTC' and not times.hasnans, 'Nonmissing UTC timestamps required')
    return times.as_unit('ns').asi8


def fields(frame, names):
    require(frame.columns.is_unique and set(names).issubset(frame.columns), 'Missing/duplicate required fields')


def support_arrays(support):
    fields(support, ['start', 'end_exclusive', 'segment_id'])
    if 'policy' in support:
        require(support.policy.eq('structural_only').all(), 'Only structural_only support allowed')
    a, b = ns(support.start), ns(support.end_exclusive)
    require(len(a) > 0 and (a < b).all() and (a[1:] >= b[:-1]).all(), 'Ordered disjoint support required')
    return a, b


def sequence_statistics(q5):
    """Finite rows of exactly 60 stored-flow sums; zeros are actual third signs."""
    q = np.asarray(q5, dtype=float)
    require(q.ndim == 2 and q.shape[1] == N and np.isfinite(q).all(), 'Finite 60-bin sequences required')
    z = np.sign(q).astype('int64')
    sign_sum, nonzero = z.sum(axis=1), (z != 0).sum(axis=1)
    adjacent_numerator = (z[:, :-1] * z[:, 1:]).sum(axis=1)
    raw = adjacent_numerator / (N - 1)
    composition = (sign_sum * sign_sum - nonzero) / (N * (N - 1))
    result = pd.DataFrame({'adjacent_agreement': raw, 'composition_reference': composition,
        'sequence_z': raw - composition, 'sign_sum': sign_sum, 'nonzero_subbins': nonzero,
        'nonzero_fraction': nonzero / N, 'all_zero_sequence': nonzero == 0,
        'constant_nonzero_sequence': np.abs(sign_sum) == N,
        'order_degenerate': (nonzero <= 1) | (np.abs(sign_sum) == N)})
    # retain these valid counterexamples in the sample
    require(result.loc[result.order_degenerate, 'sequence_z'].eq(0).all(), 'Degenerate sequence identity failed')
    return result


def measure_pressure(onsets, sparse, support):
    """Reconstruct only the 300 seconds preceding each onset.

    Absent sparse rows are zero only inside the same registered support. Any
    observed invalid transition or nonfinite active flow makes the window
    unavailable. Production callers supply every sparse row in the window.
    """
    fields(onsets, ['episode_id', 'bin_start', 'support_id'])
    fields(sparse, SPARSE_COLUMNS)
    a, b = support_arrays(support)
    t, st = ns(onsets.bin_start), ns(sparse.bin_start)
    require(pd.api.types.is_integer_dtype(onsets.support_id.dtype) and onsets.support_id.notna().all(), 'Integer support IDs required')
    require(pd.api.types.is_integer_dtype(onsets.episode_id.dtype) and onsets.episode_id.notna().all()
            and onsets.episode_id.is_unique, 'Unique integer event IDs required')
    sid = onsets.support_id.to_numpy(dtype='int64')
    require((sid >= 0).all() and (sid < len(a)).all(), 'Invalid support identity')
    require((t % SECOND == 0).all() and (np.diff(t) > 0).all(), 'Ordered whole-second onsets required')
    require(((a[sid] <= t) & (t + SECOND <= b[sid])).all(), 'Onset outside registered support')
    require((st % SECOND == 0).all() and (np.diff(st) > 0).all(), 'Ordered unique whole-second sparse bins required')
    require(all(pd.api.types.is_integer_dtype(sparse[c].dtype) and sparse[c].notna().all()
                for c in ['n_quotes', 'n_invalid_transitions']), 'Integer sparse counts required')
    count = sparse.n_quotes.to_numpy(dtype='int64')
    invalid = sparse.n_invalid_transitions.to_numpy(dtype='int64')
    flow = sparse.quote_ofi.to_numpy(dtype=float)
    require((count > 0).all() and (invalid >= 0).all() and (invalid <= count).all(), 'Invalid sparse counts')
    require(not np.isinf(flow).any(), 'Infinite observed flow')
    require(((invalid > 0) | np.isfinite(flow)).all(), 'Complete active flow is missing')
    targets = t[:, None] + np.arange(-WINDOW, 0, dtype='int64')[None, :] * SECOND
    pos = np.searchsorted(st, targets)
    safe = np.minimum(pos, max(len(st) - 1, 0))
    present = (pos < len(st)) & (st[safe] == targets) if len(st) else np.zeros(targets.shape, dtype=bool)
    seconds = np.zeros(targets.shape, dtype=float)
    counts, bad = np.zeros(targets.shape, dtype='int64'), np.zeros(targets.shape, dtype=bool)
    if len(st):
        seconds[present] = flow[safe[present]]
        counts[present] = count[safe[present]]
        bad[present] = (invalid[safe[present]] > 0) | ~np.isfinite(flow[safe[present]])
    supported = t - WINDOW * SECOND >= a[sid]
    status = np.where(supported, 'available', 'window_crosses_support').astype(object)
    status[supported & bad.any(axis=1)] = 'invalid_observed_transition'
    good = status == 'available'
    q5 = np.full((len(onsets), N), np.nan)
    numerical = []
    for row in range(len(onsets)):
        if not good[row]:
            numerical.append({'numpy_fsum_sign_differences': 0, 'bound_sensitive_nonzero_subbins': 0,
                'exact_zero_subbins_with_nonzero_gross': 0, 'max_sum_disagreement': np.nan,
                'min_abs_nonzero_subbin': np.nan, 'sum_1s_fsum': np.nan,
                'gross_1s_fsum': np.nan, 'sum_5s_fsum': np.nan, 'gross_5s_fsum': np.nan})
            continue
        values = seconds[row].reshape(N, GRID)
        stable = np.array([math.fsum(x) for x in values])
        gross = np.array([math.fsum(abs(float(v)) for v in x) for x in values])
        naive = np.sum(values, axis=1)
        epsilon = np.finfo(float).eps
        # diagnostic for this five-term aggregation only. It does not bound
        # inherited bin-construction error or introduce a zero tolerance.
        bound = (GRID * epsilon / (1 - GRID * epsilon)) * gross
        q5[row] = stable
        numerical.append({'numpy_fsum_sign_differences': int((np.sign(naive) != np.sign(stable)).sum()),
            'bound_sensitive_nonzero_subbins': int(((np.abs(stable) <= bound) & (stable != 0)).sum()),
            'exact_zero_subbins_with_nonzero_gross': int(((stable == 0) & (gross > 0)).sum()),
            'max_sum_disagreement': float(np.max(np.abs(naive - stable))),
            'min_abs_nonzero_subbin': float(np.min(np.abs(stable[stable != 0]))) if (stable != 0).any() else np.nan,
            'sum_1s_fsum': math.fsum(seconds[row]), 'gross_1s_fsum': math.fsum(abs(float(v)) for v in seconds[row]),
            'sum_5s_fsum': math.fsum(stable), 'gross_5s_fsum': math.fsum(abs(float(v)) for v in stable)})
    result = onsets[['episode_id']].copy().reset_index(drop=True)
    result['pressure_sequence_status'] = status
    result['pressure_sequence_available'] = good
    result['prior_quote_updates'] = np.where(good, counts.sum(axis=1), np.nan)
    result['supported_empty_seconds'] = np.where(good, (~present).sum(axis=1), np.nan)
    result['observed_invalid_seconds'] = bad.sum(axis=1)
    result = pd.concat([result, pd.DataFrame(numerical)], axis=1)
    measured = sequence_statistics(q5[good])
    for column in measured:
        if pd.api.types.is_bool_dtype(measured[column]):
            values = np.zeros(len(result), dtype=bool)
        else:
            values = np.full(len(result), np.nan)
        values[good] = measured[column]
        result[column] = values
    sequence = onsets[['episode_id']].reset_index(drop=True).copy()
    sequence[Q_COLUMNS] = q5
    return result, sequence


def measure_prior_prices(onsets, quotes, support):
    """Sixty returns from 61 strict-left marks, retaining all source timestamps."""
    fields(quotes, QUOTE_COLUMNS)
    a, b = support_arrays(support)
    t, qt = ns(onsets.bin_start), ns(quotes.timestamp)
    sid = onsets.support_id.to_numpy(dtype='int64')
    require((np.diff(qt) > 0).all(), 'Unique chronological clean quotes required')
    prices = quotes[['bid', 'ask']].to_numpy(dtype=float)
    require(np.isfinite(prices).all() and (prices[:, 0] > 0).all() and (prices[:, 1] > prices[:, 0]).all(), 'Invalid clean prices')
    target = t[:, None] + np.arange(-WINDOW, 1, GRID, dtype='int64')[None, :] * SECOND
    ix = np.searchsorted(qt, target, side='left') - 1
    stamp = qt[np.maximum(ix, 0)] if len(qt) else np.zeros(target.shape, dtype='int64')
    status = np.full(len(onsets), 'available', dtype=object)
    status[t - WINDOW * SECOND < a[sid]] = 'window_crosses_support'
    status[(status == 'available') & (ix[:, 0] < 0)] = 'no_prior_quote'
    status[(status == 'available') & (stamp[:, 0] < a[sid])] = 'prior_quote_outside_support'
    good = status == 'available'
    require(((stamp[good] < target[good]) & (stamp[good] >= a[sid[good], None])
             & (stamp[good] < b[sid[good], None])).all(), 'Price source timing/support failed')
    midpoint = (prices[:, 0] + prices[:, 1]) / 2
    marks = midpoint[np.maximum(ix, 0)] if len(qt) else np.ones(target.shape)
    returns = np.diff(np.log(marks), axis=1)
    rv = np.sqrt(np.sum(returns * returns, axis=1)) * 10000
    out = onsets[['episode_id']].reset_index(drop=True).copy()
    out['prior_price_status'] = status
    out['prior_price_available'] = good
    out['reconstructed_rv_5s_bps'] = np.where(good, rv, np.nan)
    for h in [5, 30, 300]:
        out[f'prior_{h}s_price_pips'] = np.where(good, (marks[:, -1] - marks[:, -1 - h // GRID]) * 10000, np.nan)
    out['prior_max_quote_age_seconds'] = np.where(good, ((target - stamp) / SECOND).max(axis=1), np.nan)
    out['repeated_mark_fraction'] = np.where(good, (ix[:, 1:] == ix[:, :-1]).mean(axis=1), np.nan)
    quote_count = np.searchsorted(qt, t, side='left') - np.searchsorted(qt, t - WINDOW * SECOND, side='left')
    out['raw_prior_quote_updates'] = np.where(good, quote_count, np.nan)
    out['first_prior_quote_at'] = pd.to_datetime(np.where(good, stamp[:, 0], np.iinfo('int64').min), utc=True)
    out['last_prior_quote_at'] = pd.to_datetime(np.where(good, stamp[:, -1], np.iinfo('int64').min), utc=True)
    wide = onsets[['episode_id']].reset_index(drop=True).copy()
    pieces = {}
    for k in range(N + 1):
        pieces[f'quote_at_{k:02d}'] = pd.to_datetime(np.where(good, stamp[:, k], np.iinfo('int64').min), utc=True)
        pieces[f'mid_{k:02d}'] = np.where(good, marks[:, k], np.nan)
    return out, pd.concat([wide, pd.DataFrame(pieces)], axis=1)


def build_features(origins, predictors, anchors, pressure, sequences, price, scaling):
    """Allowlist joins; neither future rows nor response values enter this function."""
    for frame, names in [(origins, ORIGIN_COLUMNS), (predictors, PREDICTOR_COLUMNS), (anchors, ANCHOR_COLUMNS)]:
        fields(frame, names)
    o = origins[ORIGIN_COLUMNS].reset_index(drop=True).copy()
    require(not o.episode_id.duplicated().any(), 'Unique origins required')
    frames = [predictors[PREDICTOR_COLUMNS], anchors[ANCHOR_COLUMNS], pressure, sequences, price]
    aligned = []
    for frame in frames:
        require(not frame.episode_id.duplicated().any() and set(frame.episode_id) == set(o.episode_id), 'Exact source origin conservation required')
        aligned.append(frame.set_index('episode_id').loc[o.episode_id].reset_index())
    p, a, pressure, sequences, price = aligned
    require(o.bin_start.equals(p.bin_start) and o.bin_start.equals(a.bin_start)
            and o.forecast_at.equals(p.forecast_at), 'Parent clocks disagree')
    require(o.forecast_at.eq(o.bin_start + pd.Timedelta(1, unit='s')).all(), 'First-alert clock required')
    require(o.origin_eligible.dtype == bool and o.origin_eligible.notna().all(), 'Boolean inherited eligibility required')
    d = o.direction.to_numpy(dtype=float)
    require((np.isnan(d) | np.isin(d, [-1, 0, 1])).all(), 'Invalid inherited direction')
    c_q = float(scaling['cQ'])
    require(np.isfinite(c_q) and c_q > 0 and float(scaling['sV']) > 0, 'Invalid frozen scales')
    require(o.bin_start.ge(pd.Timestamp(scaling['reference_cutoff'])).all(), 'Future scaling used')
    out = o.rename(columns={'origin_eligible': 'inherited_origin_eligible'})
    out['support_id'] = a.support_id
    for source in [pressure, price]:
        for col in source:
            if col != 'episode_id':
                require(col not in out, 'Duplicate measurement column ' + col)
                out[col] = source[col]
    q5 = sequences[Q_COLUMNS].to_numpy(dtype=float)
    out['aligned_pressure'] = d * p.signed_pressure
    for col in ['pressure_magnitude', 'variation_standardized', 'pre_spread_pips',
                'log_pre_displayed_size', *CATEGORICAL_BASELINE]:
        out[col] = p[col]
    out['gross_pressure_scaled'] = out.gross_5s_fsum / c_q
    out['log1p_prior_quote_updates'] = np.log1p(out.prior_quote_updates)
    out['aligned_sign_balance'] = d * out.sign_sum / N
    for h in [5, 30]:
        selected = q5[:, -h // GRID:]
        net = [math.fsum(row) if np.isfinite(row).all() else np.nan for row in selected]
        gross = [math.fsum(abs(float(v)) for v in row) if np.isfinite(row).all() else np.nan for row in selected]
        out[f'aligned_recent_{h}s_pressure'] = d * np.array(net) / c_q
        out[f'recent_{h}s_gross_scaled'] = np.array(gross) / c_q
    for h in [5, 30, 300]:
        out[f'aligned_prior_{h}s_price_pips'] = d * out[f'prior_{h}s_price_pips']
    out['inherited_pre_pressure'] = a.pre_quote_ofi_300s
    out['pre_pressure_reconstruction_difference'] = out.sum_1s_fsum - a.pre_quote_ofi_300s
    out['inherited_pre_rv'] = a.pre_rv_300s_5s_bps
    out['pre_rv_reconstruction_difference'] = out.reconstructed_rv_5s_bps - a.pre_rv_300s_5s_bps
    source_both = out.pressure_sequence_available & out.prior_price_available
    require(np.array_equal(out.loc[source_both, 'prior_quote_updates'], out.loc[source_both, 'raw_prior_quote_updates']),
            'Sparse quote counts differ from raw quote window')
    inherited = a.state_history_available.to_numpy(dtype=bool)
    require((~inherited | source_both).all(), 'Previously available history cannot be reconstructed')
    # existing upstream verification uses these absolute/relative precision
    # tolerances for independently reconstructed preceding flow and variation.
    require(np.allclose(out.loc[inherited, 'sum_1s_fsum'], a.loc[inherited, 'pre_quote_ofi_300s'],
                        rtol=1e-8, atol=1e-8), 'Inherited preceding pressure differs from sparse reconstruction')
    require(np.allclose(out.loc[inherited, 'reconstructed_rv_5s_bps'], a.loc[inherited, 'pre_rv_300s_5s_bps'],
                        rtol=1e-8, atol=1e-8), 'Inherited variation differs from raw quote reconstruction')
    require(np.array_equal(out.loc[inherited, 'prior_quote_updates'], a.loc[inherited, 'pre_recorded_updates_300s']),
            'Inherited preceding update count differs')
    mark_check = source_both & o.pre_mark_at.notna()
    require(out.loc[mark_check, 'last_prior_quote_at'].equals(o.loc[mark_check, 'pre_mark_at']), 'Last prior quote differs from origin mark')
    finite = np.isfinite(out[NUMERIC_BASELINE + CATEGORICAL_BASELINE + ['sequence_z']].to_numpy(dtype=float)).all(axis=1)
    out['measurement_available'] = source_both & finite
    out['combined_eligible'] = out.inherited_origin_eligible & out.measurement_available
    out['measurement_status'] = np.select([~out.pressure_sequence_available, ~out.prior_price_available, ~finite],
        [out.pressure_sequence_status, out.prior_price_status, 'nonfinite_baseline'], default='available')
    return out


def conditioning_diagnostics(frame):
    """SVD of predictor columns only; no target or projection/regression exists."""
    f = frame.loc[frame.combined_eligible]
    if len(f) < 2:
        return {'rows': len(f), 'status': 'insufficient_measurements'}
    raw = f[NUMERIC_BASELINE].astype(float).copy()
    for column, levels in [('onset_hour_utc', range(1, 24)), ('onset_weekday_utc', range(1, 7))]:
        for level in levels:
            raw[column + '_' + str(level)] = f[column].eq(level).astype(float)
    matrix = raw.to_numpy()
    sd = matrix.std(axis=0, ddof=0)
    active = (np.ptp(matrix, axis=0) > 0) & (sd > 0)
    standardized = (matrix[:, active] - matrix[:, active].mean(axis=0)) / sd[active]
    base = np.column_stack([np.ones(len(f)), standardized])
    z = f.sequence_z.to_numpy(dtype=float)
    zsd = float(z.std(ddof=0))
    augmented = np.column_stack([base, (z - z.mean()) / zsd]) if zsd > 0 else base
    def describe(x):
        singular = np.linalg.svd(x, compute_uv=False)
        tol = max(x.shape) * np.finfo(float).eps * singular[0]
        rank = int((singular > tol).sum())
        return {'columns': x.shape[1], 'rank': rank, 'rank_tolerance': float(tol),
                'largest_singular_value': float(singular[0]), 'smallest_singular_value': float(singular[-1]),
                'condition_number': float(singular[0] / singular[-1]) if singular[-1] > 0 else None}
    b, augmented_summary = describe(base), describe(augmented)
    correlations = []
    if zsd > 0:
        for column in raw.columns[active]:
            correlations.append({'column': column, 'correlation_with_sequence_z': float(np.corrcoef(raw[column], z)[0, 1])})
    return {'rows': len(f), 'status': 'measured', 'numeric_baseline': NUMERIC_BASELINE,
        'categorical_encoding': 'fixed hour1..23 and weekday1..6 indicators, zero reference; constant columns omitted',
        'constant_columns': raw.columns[~active].tolist(), 'baseline': b, 'augmented': augmented_summary,
        'augmented_rank_increment': augmented_summary['rank'] - b['rank'], 'sequence_standard_deviation': zsd,
        'pairwise_correlations': correlations,
        'maximum_absolute_pairwise_correlation': max((abs(x['correlation_with_sequence_z']) for x in correlations), default=None),
        'interpretation': 'Descriptive linear column support only; not conditional independence, causal identification or predictive evidence',
        'response_values_read': False, 'models_fitted': 0}

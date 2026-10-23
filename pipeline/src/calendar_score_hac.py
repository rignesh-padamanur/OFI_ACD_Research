"""Stage 07g: calendar-score arithmetic; no empirical response estimation runner.

Each Bartlett plug-in pilot follows Newey--West (1994); the project uses the
maximum bandwidth across the five original state scores for one common lag.
This fixed combination is not a multivariate optimality claim. It uses neither
prewhitening, score demeaning, nor a finite-sample degrees-of-freedom factor.
Source implementation used to resolve rounding/normalization conventions:
https://raw.githubusercontent.com/cran/sandwich/master/R/vcovHAC.R
"""
import numpy as np
import pandas as pd


PILOT_COLUMNS = (
    'variation_standardized', 'pressure_magnitude', 'aligned_prior_pressure',
    'variation_x_magnitude', 'variation_x_alignment',
)
EPS = np.finfo(float).eps
NUMERICAL_MULTIPLIER = 64.0


def _matrix(value, label, allow_empty=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 2:
        raise ValueError(label + ' must be a real numeric matrix')
    x = raw.astype(float)
    if not x.shape[1] or (not allow_empty and not x.shape[0]) or not np.isfinite(x).all():
        raise ValueError(label + ' must have finite entries and nonempty dimensions')
    return x


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label + ' must be an integer')
    return int(value)


def pilot_weights(columns):
    """Select five coordinate pilots; controls/direction/intercept zero.

    The selected coordinates are inspected separately, preventing cancellation
    across reinforcing/opposing pressure terms. Their maximum bandwidth is a
    project choice, not the sandwich default or optimal for every contrast.
"""
    names = list(columns)
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError('Unique string column names required')
    if not set(PILOT_COLUMNS).issubset(names):
        raise ValueError('All five frozen state columns required for the pilot')
    return np.array([float(name in PILOT_COLUMNS) for name in names])


def ols_scores(x, y):
    """Generic SVD OLS primitive, exercised on synthetic responses in stage07g.

Full column rank and positive residual information are required.  No
generalized inverse, outcome trimming, or numerical variance repair is used.
"""
    x = _matrix(x, 'Design')
    raw = np.asarray(y)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 1 or len(raw) != len(x):
        raise ValueError('Outcome must be a conformable real numeric vector')
    y = raw.astype(float)
    if not np.isfinite(y).all() or len(x) <= x.shape[1]:
        raise ValueError('Finite outcomes and more events than columns required')
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    tolerance = EPS * max(x.shape) * s[0]
    rank = int(np.count_nonzero(s > tolerance))
    if rank != x.shape[1]:
        raise ValueError('Rank-deficient design; no generalized inverse allowed')
    beta = vt.T @ ((u.T @ y) / s)
    fitted = x @ beta
    residuals = y - fitted
    if not all(np.isfinite(z).all() for z in [beta, fitted, residuals]):
        raise ValueError('Nonfinite OLS calculation')
    response_scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(fitted))))
    if response_scale == 0:
        raise ValueError('Degenerate or numerically exact fit; inference unavailable')
    scaled_residual_tolerance = NUMERICAL_MULTIPLIER * EPS * max(x.shape) * max(
        float(np.linalg.norm(y/response_scale)), float(np.linalg.norm(fitted/response_scale)))
    residual_tolerance = scaled_residual_tolerance * response_scale
    if np.linalg.norm(residuals/response_scale) <= scaled_residual_tolerance:
        raise ValueError('Degenerate or numerically exact fit; inference unavailable')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        inverse_factor = vt.T / s
        bread = inverse_factor @ inverse_factor.T
        scores = x * residuals[:, None]
    if not all(np.isfinite(z).all() for z in [beta, residuals, bread, scores]):
        raise ValueError('Nonfinite OLS calculation')
    return {'beta': beta, 'residuals': residuals, 'event_scores': scores,
            'bread': bread, 'rank': rank, 'rank_tolerance': float(tolerance),
            'residual_tolerance': float(residual_tolerance),
            'normal_equation_score_sum': scores.sum(axis=0),
            'events': len(x), 'columns': x.shape[1]}


def aggregate_daily_scores(event_scores, onset_times, start, end_exclusive):
    """Sum by actual UTC onset date, retaining every date in [start,end).

An empty date contributes zero observed selected-event score, not an imputed
return or a statement about feed completeness. Events are never daily-averaged.
"""
    scores = _matrix(event_scores, 'Event scores', allow_empty=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    if (str(start.tz) != 'UTC' or str(end.tz) != 'UTC' or pd.isna(start) or pd.isna(end)
            or start != start.normalize() or end != end.normalize() or start >= end):
        raise ValueError('Ordered UTC midnight calendar bounds required')
    dates = pd.date_range(start, end, freq='D', inclusive='left')
    times = pd.DatetimeIndex(onset_times)
    if str(times.tz) != 'UTC' or times.hasnans or len(times) != len(scores):
        raise ValueError('Nonmissing UTC onset timestamps matching scores required')
    if ((times < start) | (times >= end)).any():
        raise ValueError('Onset outside the declared calendar')
    indexes = dates.get_indexer(times.normalize())
    daily = np.zeros((len(dates), scores.shape[1]))
    np.add.at(daily, indexes, scores)
    counts = np.bincount(indexes, minlength=len(dates))
    if not np.isfinite(daily).all():
        raise ValueError('Nonfinite calendar sums')
    return {'daily_scores': daily, 'dates': dates, 'event_counts': counts}


def nw94_bandwidth(daily_scores, weights):
    """Uncentered, non-prewhitened NW1994 Bartlett plug-in bandwidth.

Pilot m=floor(4*(T/100)^(2/9)); gamma_j=T^-1 sum h_t h_(t-j),
s0=gamma0+2 sum gamma_j, s1=2 sum j gamma_j. The final continuous
b=1.1447*((s1/s0)^2*T)^(1/3) is distinct from pilot m. Return L=floor(b).
The HAC weights subsequently use 1-j/(L+1), not 1-j/b.
"""
    g = _matrix(daily_scores, 'Daily scores')
    raw = np.asarray(weights)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 1 or len(raw) != g.shape[1]:
        raise ValueError('Conformable real numeric pilot weights required')
    w = raw.astype(float)
    if not np.isfinite(w).all() or not np.any(w != 0):
        raise ValueError('Nonzero finite pilot weights required')
    t = len(g)
    if t < 3:
        raise ValueError('At least three calendar rows required for the pilot')
    m = int(np.floor(4 * (t / 100) ** (2 / 9)))
    if not 1 <= m < t:
        raise ValueError('Pilot lag outside observed calendar; no silent cap')
    with np.errstate(over='ignore', invalid='ignore'):
        h = g @ w
        # a common scale leaves both ratios unchanged
        # avoid overflow when squaring; keep the scale with the scaled moments
        scale = float(np.max(np.abs(h)))
        cancellation_scale = float(np.max(np.abs(g) @ np.abs(w)))
        projection_tolerance = NUMERICAL_MULTIPLIER * EPS * cancellation_scale
    if (not np.isfinite(h).all() or not np.isfinite([scale, projection_tolerance]).all()
            or scale <= projection_tolerance):
        raise ValueError('Degenerate or nonfinite scalar pilot projection')
    z = h / scale
    gamma = np.array([np.dot(z[j:], z[:t-j]) / t for j in range(m+1)])
    s0 = float(gamma[0] + 2 * gamma[1:].sum())
    s1 = float(2 * np.dot(np.arange(1, m+1), gamma[1:]))
    s0_scale = float(gamma[0] + 2 * np.abs(gamma[1:]).sum())
    s0_tolerance = NUMERICAL_MULTIPLIER * EPS * s0_scale
    if abs(s0) <= s0_tolerance:
        raise ValueError('Numerically zero pilot s0; no fallback bandwidth')
    with np.errstate(over='ignore', invalid='ignore'):
        b = float(1.1447 * np.cbrt((s1 / s0) ** 2 * t))
    if not np.isfinite(b):
        raise ValueError('Nonfinite automatic bandwidth')
    lag = int(np.floor(b))
    if lag >= t:
        raise ValueError('Automatic lag reaches calendar length; no silent cap')
    return {'calendar_days': t, 'pilot_lag': m, 'scalar_projection_scale': scale,
            'scalar_projection_tolerance': projection_tolerance,
            'scaled_autocovariances': gamma, 'scaled_s0': s0, 'scaled_s1': s1,
            'scaled_s0_tolerance': s0_tolerance,
            'negative_pilot_s0': bool(s0 < 0), 'bandwidth': b, 'lag': lag,
            'weights': w, 'prewhitening': False, 'daily_score_centering': False,
            'moment_denominator': t, 'lag_capped': False}


def bartlett_meat(daily_scores, lag):
    """Unnormalized calendar-score meat; L=0 is day-cluster meat, not HC0."""
    g = _matrix(daily_scores, 'Daily scores')
    lag = _integer(lag, 'Lag')
    if not 0 <= lag < len(g):
        raise ValueError('Lag must be within the observed calendar')
    with np.errstate(over='ignore', invalid='ignore'):
        result = g.T @ g
        for j in range(1, lag+1):
            cross = g[j:].T @ g[:-j]
            result += (1 - j / (lag+1)) * (cross + cross.T)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite Bartlett meat')
    return (result + result.T) / 2


def coordinate_bandwidth(daily_scores, weights):
    """One common lag from the largest of fixed coordinate NW1994 bandwidths.

The 0/1 weights select coordinates, not a sum of scores. Coordinate-wise sign
and nonzero rescaling do not alter each pilot. Cross-coordinate lag dependence
can nevertheless escape marginal pilots; this rule does not certify adequacy.
Every selected pilot must be computable; none is dropped on failure.
"""
    g = _matrix(daily_scores, 'Daily scores')
    raw = np.asarray(weights)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 1 or len(raw) != g.shape[1]:
        raise ValueError('Conformable real numeric coordinate selectors required')
    w = raw.astype(float)
    if not np.isfinite(w).all() or not np.isin(w, [0., 1.]).all() or not np.any(w == 1):
        raise ValueError('Nonempty zero/one coordinate selectors required')
    selected = np.flatnonzero(w == 1).tolist()
    pilots = [{'column_index': index, **nw94_bandwidth(g[:, [index]], [1.])} for index in selected]
    bandwidth = max(item['bandwidth'] for item in pilots)
    return {'calendar_days': len(g), 'pilot_lag': pilots[0]['pilot_lag'],
            'bandwidth': bandwidth, 'lag': int(np.floor(bandwidth)), 'weights': w,
            'selected_columns': selected, 'component_pilots': pilots,
            'selector': 'maximum_of_fixed_selected_coordinate_NW1994_bandwidths',
            'prewhitening': False, 'daily_score_centering': False,
            'lag_capped': False, 'moment_denominator': len(g)}


def _covariance(g, bread, lag):
    meat = bartlett_meat(g, lag)
    scale = float(np.max(np.abs(meat)))
    if scale == 0:
        raise ValueError('Zero calendar-score meat; inference unavailable')
    eigenvalues = np.linalg.eigvalsh(meat)
    tolerance = NUMERICAL_MULTIPLIER * EPS * max(meat.shape) * scale
    if eigenvalues[0] < -tolerance:
        raise ValueError('Numerically indefinite Bartlett meat')
    covariance = bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise ValueError('Nonfinite covariance')
    covariance_scale = float(np.max(np.abs(covariance)))
    if covariance_scale == 0:
        raise ValueError('Numerically zero covariance; inference unavailable')
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_tolerance = NUMERICAL_MULTIPLIER * EPS * max(covariance.shape) * covariance_scale
    if covariance_eigenvalues[0] < -covariance_tolerance:
        raise ValueError('Numerically indefinite covariance')
    return {'lag': lag, 'meat': meat, 'covariance': covariance,
            'meat_min_eigenvalue': float(eigenvalues[0]),
            'meat_psd_tolerance': tolerance, 'finite_sample_adjustment': False,
            'covariance_min_eigenvalue': float(covariance_eigenvalues[0]),
            'covariance_psd_tolerance': covariance_tolerance,
            'normalization': "inverse(X'X) @ unnormalized_daily_meat @ inverse(X'X)"}


def calendar_hac(daily_scores, bread, weights):
    """Arithmetic audit only; no claim of score-process or inferential validity.

One automatic estimate and one fixed longer-lag sensitivity max(1,2L).
An unobservable requested lag fails, without clipping or fallback. Bread is
the event-level inverse(X'X), not a regression on daily-average observations.
"""
    g = _matrix(daily_scores, 'Daily scores')
    a = _matrix(bread, 'Bread')
    if a.shape != (g.shape[1], g.shape[1]):
        raise ValueError('Bread must be square and match score columns')
    tolerance = NUMERICAL_MULTIPLIER * EPS * max(a.shape) * float(np.max(np.abs(a)))
    if np.max(np.abs(a-a.T)) > tolerance:
        raise ValueError('Symmetric bread required')
    try:
        np.linalg.cholesky((a+a.T)/2)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Positive definite full-rank bread required') from exc
    bw = coordinate_bandwidth(g, weights)
    longer = max(1, 2*bw['lag'])
    if longer >= len(g):
        raise ValueError('Longer lag reaches calendar length; no silent cap')
    return {'bandwidth': bw, 'automatic': _covariance(g, a, bw['lag']),
            'longer': _covariance(g, a, longer),
            'longer_lag_rule': 'max(1, 2 * automatic_lag)',
            'inference_approved': False, 'score_process_assumptions_verified': False}

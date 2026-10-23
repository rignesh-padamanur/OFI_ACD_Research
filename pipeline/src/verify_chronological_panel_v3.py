"""Bounded-memory independent verification of the chronological baseline panel.

The production panel builders are deliberately not imported. Independent
incoming-wait records, direct historical exposure contributions, SciPy filters,
raw covariance distances and candidate graphs supply the numerical references.
Version 3 adds a narrowly bounded failed-build reuse audit and inherited measurement checks.
"""
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.verify_chronological_acd import reference_sample, reference_predictions, check_diagnostics
from src.verify_event_audit import (reference_clock, reference_duration_bins, reference_detector,
                                    check_json_values)
from src.panel_reference_precision import reference_quote_diagnostics
from src.verify_event_episodes import reference_quote_quality, QUALITY_COLUMNS
from src.verify_chronological_transition import _frame, _partition_statistics

S = 1_000_000_000
DAY = 86400 * S
BATCH = 333_331
FEATURES = ['adjusted_abs_quote_ofi', 'adjusted_tick_count', 'negative_log_mean_residual']
METRICS = ['pre_spread_pips', 'end_spread_pips', 'spread_change_pips',
           'pre_displayed_size', 'end_displayed_size', 'log_size_ratio',
           'pre_quote_age_sec', 'end_quote_age_sec', 'pre_rv_300s_5s_bps', 'pre_quote_ofi_300s']
ISSUED = ['origin_quote_id', 'component_id', 'origin_at', 'scale_sec', 'psi', 'residual_parameter_cutoff']
PRECISION_SOURCE_SHA256 = 'a54717a628dd2464f14b3e68197dae535d90fb74e59cc7034bd8eede6496d7b0'
PRECISION_METRIC = 'pre_rv_300s_5s_bps'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def ns(values):
    index = pd.DatetimeIndex(values)
    require(str(index.tz) == 'UTC' and not index.hasnans, 'Complete UTC timestamps required')
    return index.as_unit('ns').asi8


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.array(array, dtype='<i8', order='C')
        np.testing.assert_array_equal(np.asarray(array), a)
        digest.update(np.array(a.shape, dtype='<i8').tobytes())
        digest.update(a.tobytes())
    return digest.hexdigest()


class FrameCursor:
    """Compare streams with independently chosen row-batch boundaries."""
    def __init__(self, path):
        self.iterator = iter(pq.ParquetFile(path).iter_batches(batch_size=BATCH))
        self.buffer = None
        self.offset = 0

    def take(self, count):
        pieces = []
        while count:
            if self.buffer is None or self.offset == len(self.buffer):
                self.buffer = next(self.iterator).to_pandas()
                self.offset = 0
            size = min(count, len(self.buffer)-self.offset)
            pieces.append(self.buffer.iloc[self.offset:self.offset+size])
            self.offset += size
            count -= size
        return pd.concat(pieces, ignore_index=True)

    def finished(self):
        require(self.buffer is None or self.offset == len(self.buffer), 'Extra serialized rows')
        require(next(self.iterator, None) is None, 'Extra serialized batches')


def incoming_spells(path, tails, stop):
    """Incoming gaps independently identify completed outgoing origin waits."""
    tails = tails.loc[tails.policy.eq('structural_only')].sort_values('last_quote_id')
    require(not tails.last_quote_id.duplicated().any(), 'Duplicate terminal identities')
    last_id, last_at, reached = -1, None, False
    cols = ['quote_id', 'available_at', 'audit_gap_ns', 'component_id_structural_only', 'completed_structural_only']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        f = batch.to_pandas()
        times = ns(f.available_at)
        k = int(np.searchsorted(times, stop))
        f = f.iloc[:min(k+1, len(f))]
        times = ns(f.available_at)
        ids = f.quote_id.to_numpy()
        np.testing.assert_array_equal(ids, np.arange(last_id+1, last_id+1+len(f)))
        if last_at is not None:
            require(times[0] > last_at, 'Unordered source boundary')
        require((np.diff(times) > 0).all(), 'Nonunique source timestamps')
        lo, hi = max(0, int(ids[0])-1), int(ids[-1])
        event = f.completed_structural_only.to_numpy(dtype=bool)
        a = times[event]-f.audit_gap_ns.to_numpy()[event]
        b = times[event]
        completed = pd.DataFrame({'origin_quote_id': ids[event]-1,
            'component_id': f.component_id_structural_only.to_numpy()[event],
            'origin_at': pd.to_datetime(a, utc=True), 'end_at': pd.to_datetime(np.minimum(b, stop), utc=True),
            'event_observed': b < stop, 'stop_code': np.where(b < stop, 0, 8).astype('uint8'),
            'exposure_ns': np.minimum(b, stop)-a})
        terminal = tails.loc[tails.last_quote_id.ge(lo) & tails.last_quote_id.lt(hi)]
        ta, tb = ns(terminal.origin), ns(terminal.stop)
        censored = pd.DataFrame({'origin_quote_id': terminal.last_quote_id.to_numpy(),
            'component_id': terminal.component_id.to_numpy(), 'origin_at': terminal.origin.to_numpy(),
            'end_at': pd.to_datetime(np.minimum(tb, stop), utc=True), 'event_observed': False,
            'stop_code': np.where(tb < stop, terminal.stop_code, 8).astype('uint8'),
            'exposure_ns': np.minimum(tb, stop)-ta})
        expected = pd.concat([completed, censored], ignore_index=True).sort_values('origin_quote_id').reset_index(drop=True)
        expected = expected.loc[expected.origin_at.lt(pd.Timestamp(stop, tz='UTC'))].reset_index(drop=True)
        np.testing.assert_array_equal(expected.origin_quote_id, np.arange(lo, hi))
        if len(expected):
            yield expected
        last_id, last_at = int(ids[-1]), int(times[-1])
        if last_at >= stop:
            reached = True
            break
    if not reached:
        terminal = tails.loc[tails.last_quote_id.eq(last_id)]
        require(len(terminal) == 1, 'Last recorded origin lacks terminal observation')
        a, b = ns(terminal.origin), ns(terminal.stop)
        yield pd.DataFrame({'origin_quote_id': terminal.last_quote_id.to_numpy(),
            'component_id': terminal.component_id.to_numpy(), 'origin_at': terminal.origin.to_numpy(),
            'end_at': pd.to_datetime(np.minimum(b, stop), utc=True), 'event_observed': False,
            'stop_code': np.where(b < stop, terminal.stop_code, 8).astype('uint8'),
            'exposure_ns': np.minimum(b, stop)-a})


def verify_observations(directory, inputs, origin, stop):
    directory = Path(directory)
    origin, stop = pd.Timestamp(origin).value, pd.Timestamp(stop).value
    ndays = (stop-origin)//DAY
    totals = np.zeros((ndays*96, 3), dtype='int64')
    historical_exposure = np.zeros(ndays*96, dtype='int64')
    historical_events = np.zeros(ndays*96, dtype='int64')
    corrections = {}
    raw = FrameCursor(directory/'raw_spells.parquet')
    tails = pd.read_parquet(inputs['terminal_spells']['path'])
    count, completed, exposure, positive = 0, 0, 0, 0
    for expected in incoming_spells(inputs['quote_durations']['path'], tails, stop):
        _frame(raw.take(len(expected)), expected, 'independent incoming source')
        a, b = ns(expected.origin_at), ns(expected.end_at)
        e = expected.event_observed.to_numpy()
        day = (a-origin)//DAY
        slot = (a % DAY)//(900*S)
        key = day*96+slot
        np.add.at(totals[:, 0], key, 1)
        np.add.at(totals[:, 1], key, e.astype('int64'))
        np.add.at(totals[:, 2], key, b-a)
        # direct contributions to each later same-weekday historical cutoff;
        # unlike production, this never fits seasons from corrected full totals.
        for lag in range(7, 57, 7):
            date = day+lag
            valid = date < ndays
            when = origin+date[valid]*DAY
            target = date[valid]*96+slot[valid]
            np.add.at(historical_exposure, target, np.minimum(b[valid], when)-a[valid])
            np.add.at(historical_events, target, (e[valid] & (b[valid] < when)).astype('int64'))
        for i in np.flatnonzero(b >= origin+(day+1)*DAY):
            for d in range(int(day[i])+1, min(ndays-1, int((b[i]-origin)//DAY))+1):
                ck = (d, int(day[i]), int(slot[i]))
                old = corrections.get(ck, (0, 0))
                corrections[ck] = (old[0]+int(b[i]-origin-d*DAY), old[1]+int(e[i]))
        count += len(expected)
        completed += int(e.sum())
        exposure += int((b-a).sum())
        positive += int(np.count_nonzero(b > a))
    raw.finished()
    table = pd.DataFrame({'origin_day': np.repeat(np.arange(ndays), 96), 'slot': np.tile(np.arange(96), ndays),
        'origins': totals[:, 0], 'completed_events': totals[:, 1], 'full_exposure_ns': totals[:, 2]})
    _frame(pd.read_parquet(directory/'origin_daily_sufficient_statistics.parquet'), table, 'source day-slot statistics')
    correction = pd.DataFrame([(*key, *value) for key, value in sorted(corrections.items())],
        columns=['cutoff_day', 'origin_day', 'slot', 'future_exposure_ns', 'future_completions'])
    _frame(pd.read_parquet(directory/'historical_cutoff_corrections.parquet'), correction, 'historical cutoffs')
    dates = np.repeat(np.arange(ndays), 96)
    prior = np.zeros((ndays, 96), dtype='int64')
    positive_days = totals[:, 2].reshape(ndays, 96) > 0
    for lag in range(7, 57, 7):
        prior[lag:] += positive_days[:-lag]
    scales = np.divide(historical_exposure/S, historical_events,
                       out=np.full(ndays*96, np.nan), where=historical_events > 0)
    status = np.full(ndays*96, 5, dtype='uint8')
    status[~np.isfinite(scales) | (scales <= 0)] = 4
    status[historical_events < 100] = 3
    status[historical_exposure < 1800*S] = 2
    status[prior.ravel() < 4] = 1
    status[dates < 56] = 0
    scales[status != 5] = np.nan
    seasons = pd.DataFrame({'effective_date': pd.to_datetime(origin+dates*DAY, utc=True),
        'training_start': pd.to_datetime(origin+(dates-56)*DAY, utc=True),
        'training_end_exclusive': pd.to_datetime(origin+dates*DAY, utc=True),
        'slot': np.tile(np.arange(96), ndays), 'prior_dates': prior.ravel(),
        'completed_events': historical_events, 'observed_exposure_ns': historical_exposure,
        'working_scale_sec': scales, 'status': status})
    _frame(pd.read_parquet(directory/'seasonal_parameters.parquet'), seasons, 'direct historical seasons')
    seasonal = FrameCursor(directory/'seasonal_spells.parquet')
    states = np.zeros(6, dtype='int64')
    ready_events = 0
    for batch in pq.ParquetFile(directory/'raw_spells.parquet').iter_batches(batch_size=BATCH):
        expected = batch.to_pandas()
        day = (ns(expected.origin_at)-origin)//DAY
        slot = (ns(expected.origin_at) % DAY)//(900*S)
        key = day*96+slot
        expected['seasonal_status'] = status[key]
        expected['scale_sec'] = scales[key]
        expected['normalized_exposure'] = expected.exposure_ns/S/expected.scale_sec
        expected['seasonal_cutoff'] = pd.to_datetime(origin+day*DAY, utc=True)
        _frame(seasonal.take(len(expected)), expected, 'normalized origin seasons')
        states += np.bincount(expected.seasonal_status.to_numpy(), minlength=6)
        ready_events += int((expected.seasonal_status.eq(5) & expected.event_observed).sum())
    seasonal.finished()
    return {'calendar_days': int(ndays), 'source_spell_rows': count, 'source_completed_events': completed,
        'source_exposure_ns': exposure, 'source_positive_exposure_rows': positive,
        'seasonal_parameter_rows': len(seasons), 'seasonal_ready_rows': int(states[5]),
        'seasonal_ready_completed_events': ready_events,
        'seasonal_status_counts': {str(i): int(n) for i, n in enumerate(states)}}


def quote_window(path, start, stop):
    """Linear metadata scan plus bounded row groups; retain the strict predecessor."""
    columns = ['timestamp', 'bid', 'ask', 'bid_volume', 'ask_volume']
    parquet = pq.ParquetFile(path)
    field = parquet.schema_arrow.get_field_index('timestamp')
    minima, maxima = [], []
    for i in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(i).column(field).statistics
        if stats is not None and stats.has_min_max:
            minima.append(pd.Timestamp(stats.min).value)
            maxima.append(pd.Timestamp(stats.max).value)
        else:
            times = ns(parquet.read_row_group(i, columns=['timestamp']).to_pandas().timestamp)
            minima.append(int(times[0])); maxima.append(int(times[-1]))
    first = int(np.searchsorted(maxima, start.value, side='left'))
    last = int(np.searchsorted(minima, stop.value, side='left'))
    groups = list(range(max(0, first-1), last))
    frame = parquet.read_row_groups(groups, columns=columns).to_pandas()
    time = ns(frame.timestamp)
    left = int(np.searchsorted(time, start.value, side='left'))
    right = int(np.searchsorted(time, stop.value, side='left'))
    return frame.iloc[max(0, left-1):right].reset_index(drop=True)


def feature_reference(raw, factors, duration, cutoff):
    clock = reference_clock(raw, factors, 'structural_only')
    times = ns(raw.bin_start)
    np.testing.assert_array_equal(np.diff(times), S)
    np.testing.assert_array_equal(ns(raw.available_at), times+S)
    d = pd.DataFrame({'bin_start': clock.bin_start}).merge(duration, how='left', on='bin_start',
                                                         sort=False, validate='one_to_one')
    count = raw.recorded_update_count.to_numpy(dtype='int64')
    completed = d.completed_residual_count.fillna(0).to_numpy(dtype='int64')
    require((completed >= 0).all() and (completed <= count).all(), 'Residual/quote count mismatch')
    ready = np.isfinite(clock[['adjusted_abs_quote_ofi', 'adjusted_tick_count']]).all(axis=1).to_numpy()
    status = clock.flow_status.to_numpy()
    reason = np.full(len(clock), 'eligible', dtype=object)
    reason[completed != count] = 'incomplete_duration'
    reason[count == 0] = 'quiet_no_duration'
    reason[~ready] = 'seasonal_unavailable'
    reason[~np.isin(status, [1, 2])] = 'invalid_flow'
    reason[status == 0] = 'unsupported'
    opportunities = pd.DataFrame({'bin_start': clock.bin_start, 'available_at': clock.available_at,
        'period': np.where(times < cutoff.value, 'calibration', 'development'),
        'recorded_update_count': count, 'support_id': clock.support_id, 'flow_status': status,
        'seasonal_ready': ready, 'completed_residual_count': completed,
        'eligible_without_duration': ready & np.isin(status, [1, 2]), 'reason': reason})
    take = reason == 'eligible'
    scored = clock.loc[take, ['bin_start', 'available_at', 'support_id', 'quote_ofi',
                               'adjusted_abs_quote_ofi', 'adjusted_tick_count']].reset_index(drop=True)
    scored['recorded_update_count'] = count[take]
    scored['completed_residual_count'] = completed[take]
    scored['mean_duration_residual'] = d.mean_duration_residual.to_numpy()[take]
    scored['negative_log_mean_residual'] = -np.log(scored.mean_duration_residual)
    scored['period'] = opportunities.period.to_numpy()[take]
    return clock, opportunities, scored


def verify_cycle(directory, observations, inputs, carry_in, support):
    directory, observations = Path(directory), Path(observations)
    parameters, statistics = read(directory/'parameters.json'), read(directory/'statistics.json')
    start, cutoff, stop = [pd.Timestamp(parameters[k]) for k in
                           ('training_start', 'parameter_cutoff', 'deployment_end')]
    require(cutoff-start == pd.Timedelta(days=56) and pd.Timedelta(0) < stop-cutoff <= pd.Timedelta(days=14),
            'Cycle calibration/deployment interval changed')
    full = pd.read_parquet(observations/'seasonal_spells.parquet',
                          filters=[('origin_at', '>=', start), ('origin_at', '<', stop)])
    training = reference_sample(full, start.value, cutoff.value)
    path = reference_sample(full, start.value, stop.value)
    require(statistics['training_spells'] == len(training) and
            statistics['training_events'] == int(training.event_observed.sum()), 'Training sample count differs')
    require(statistics['training_identity_sha256'] == fingerprint(training.origin_quote_id,
            training.exposure_ns, training.event_observed), 'Clipped training identity fingerprint differs')
    attempts = parameters['optimization_attempts']
    require(len(attempts) == 3, 'Three fixed optimizer starts required')
    for attempt, initial in zip(attempts, [(.05, .85, .7), (.15, .70, 1.), (.30, .50, .5)], strict=True):
        np.testing.assert_array_equal([attempt[k] for k in ('start_alpha', 'start_beta', 'start_shape')], initial)
    good = [x for x in attempts if x['success'] and np.isfinite(x['objective'])]
    require(bool(good), 'Unsuccessful fit must not be deployed')
    best = min(good, key=lambda x: x['objective'])
    keys = ['omega', 'alpha', 'beta', 'shape']
    p = np.asarray([parameters['acd_parameters'][k] for k in keys])
    np.testing.assert_array_equal(p, [best[k] for k in keys])
    require(np.isfinite(p).all() and p[0] > 0 and min(p[1:3]) >= 0 and p[1]+p[2] < .999 and .2 <= p[3] <= 5,
            'Inadmissible working duration parameters')
    predicted_training = reference_predictions(training, p, weibull=True)
    objective = float((predicted_training.negative_log_score_seconds-
                      training.event_observed*np.log(training.scale_sec)).mean())
    np.testing.assert_allclose([parameters['acd_fit_objective'], best['objective']], objective, rtol=1e-9, atol=1e-9)
    warning = bool(p[1]+p[2] > .999-1e-5 or p[3] < .2001 or p[3] > 4.999)
    require(parameters['numerical_constraint_warning'] == best['near_constraint'] == warning,
            'Constraint warning differs')
    prediction = reference_predictions(path, p, weibull=True)
    deployed = path.origin_at.ge(cutoff).to_numpy()
    expected_issued = path.loc[deployed, ISSUED[:4]].reset_index(drop=True)
    expected_issued['psi'] = prediction.psi.to_numpy()[deployed]
    expected_issued['residual_parameter_cutoff'] = cutoff
    issued = pd.read_parquet(directory/'issued_forecasts.parquet')
    _frame(issued, expected_issued, 'new origin forecasts')
    check_diagnostics(path.loc[deployed].reset_index(drop=True), prediction.loc[deployed].reset_index(drop=True),
                      statistics['duration_working_model_diagnostics'])
    actual_in = pd.read_parquet(directory/'carry_in.parquet')
    pd.testing.assert_frame_equal(actual_in.reset_index(drop=True), carry_in.reset_index(drop=True),
                                  check_exact=True, check_dtype=False, check_categorical=False)
    require(not carry_in.origin_quote_id.duplicated().any(), 'Duplicate carry identity')
    opened = pd.read_parquet(observations/'seasonal_spells.parquet',
                            filters=[('origin_at', '<', cutoff), ('end_at', '>=', cutoff)])
    opened = opened.loc[np.isfinite(opened.scale_sec) & opened.exposure_ns.gt(0)].reset_index(drop=True)
    require(set(opened.origin_quote_id) == set(carry_in.origin_quote_id), 'Missing or extraneous issued running wait')
    new = full.loc[full.origin_at.ge(cutoff) & np.isfinite(full.scale_sec) & full.exposure_ns.gt(0)].copy()
    actual = pd.concat([opened, new], ignore_index=True).sort_values('origin_quote_id').reset_index(drop=True)
    forecasts = pd.concat([carry_in, issued], ignore_index=True).sort_values('origin_quote_id').reset_index(drop=True)
    _frame(forecasts[ISSUED[:4]], actual[ISSUED[:4]], 'source/issued origin identity')
    require((forecasts.residual_parameter_cutoff <= forecasts.origin_at).all(), 'Origin forecast uses future parameters')
    require(np.isfinite(forecasts[['scale_sec', 'psi']]).all(axis=None) and
            (forecasts[['scale_sec', 'psi']] > 0).all(axis=None), 'Invalid issued mean/scale')
    observed = actual.event_observed & actual.end_at.ge(cutoff) & actual.end_at.lt(stop)
    ratio = actual.loc[observed, 'exposure_ns'].to_numpy()/S/forecasts.loc[observed, 'scale_sec']/forecasts.loc[observed, 'psi']
    values = pd.DataFrame({'endpoint': actual.loc[observed, 'end_at'].to_numpy(), 'residual': np.asarray(ratio)})
    bins = values.set_index('endpoint').resample('1s').residual.agg(['count', 'mean'])
    bins = bins.loc[bins['count'] > 0].rename(columns={'count': 'completed_residual_count',
        'mean': 'mean_duration_residual'}).rename_axis('bin_start').reset_index()
    _frame(pd.read_parquet(directory/'deployment_duration_bins.parquet'), bins, 'origin-vintage deployment residuals')
    outgoing = forecasts.loc[actual.end_at.ge(stop)].reset_index(drop=True)
    saved_out = pd.read_parquet(directory/'carry_out.parquet')
    pd.testing.assert_frame_equal(saved_out, outgoing, check_exact=True, check_dtype=False, check_categorical=False)
    carry_counts = {'issued_new': len(issued), 'carried_in': len(carry_in), 'carried_out': len(outgoing),
        'completed_residuals': int(observed.sum()),
        'completed_carried_residuals': int((observed & actual.origin_at.lt(cutoff)).sum()),
        'carried_origin_before_training_start': int(carry_in.origin_at.lt(start).sum()),
        'oldest_carry_vintage': carry_in.residual_parameter_cutoff.min().isoformat() if len(carry_in) else None}
    check_json_values(statistics['carry'], carry_counts)
    historical_bins = reference_duration_bins(path, prediction.psi)
    historical_bins = historical_bins.loc[historical_bins.bin_start.lt(cutoff)]
    duration = pd.concat([historical_bins, bins], ignore_index=True)
    counts = {'training_spells_verified': len(training), 'new_issued_forecasts_verified': len(issued),
              'carried_forecasts_verified': len(carry_in), 'completed_residuals_verified': int(observed.sum())}
    del full, training, path, prediction, predicted_training, actual, forecasts, opened, new
    gc.collect()
    raw = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    np.testing.assert_array_equal(ns(raw.bin_start), np.arange(start.value, stop.value, S))
    factors = pd.read_parquet(inputs['calibration_parameters']['path'], filters=[('effective_date', '>=', start),
        ('effective_date', '<', stop), ('policy', '==', 'structural_only')])
    clock, opportunities, scored = feature_reference(raw, factors, duration, cutoff)
    training_mask = scored.period.eq('calibration').to_numpy()
    calibration_features = scored.loc[training_mask, FEATURES].to_numpy()
    reference, scores = reference_detector(calibration_features, scored[FEATURES].to_numpy())
    fit = parameters['detector']
    for key in ('mean', 'sample_std', 'covariance_z', 'threshold', 'quantile'):
        np.testing.assert_allclose(fit[key], reference[key], rtol=1e-9, atol=1e-9, err_msg=key)
    eig = np.linalg.eigvalsh(reference['covariance_z'])
    require(eig[0] > 0 and eig[-1]/eig[0] <= 1e10, 'Invalid covariance')
    np.testing.assert_allclose(fit['condition_number'], eig[-1]/eig[0], rtol=1e-9)
    require(fit['calibration_rows'] == int(training_mask.sum()) and fit['quantile_method'] == 'linear',
            'Detector reference recipe differs')
    require(fit['training_threshold_ties'] == int(np.count_nonzero(scores[training_mask] == reference['threshold'])),
            'Threshold ties differ')
    scored['distance'] = scores
    scored['candidate_event'] = scores >= reference['threshold']
    calibration = _partition_statistics(opportunities.loc[opportunities.period.eq('calibration')],
                                        scored.loc[training_mask])
    calibration['eligible_bin_identity_sha256'] = fingerprint(ns(scored.loc[training_mask, 'bin_start']))
    check_json_values(statistics['calibration'], calibration)
    o = opportunities.loc[opportunities.period.eq('development')].reset_index(drop=True)
    s = scored.loc[~training_mask].reset_index(drop=True)
    history = clock.loc[clock.bin_start.ge(cutoff-pd.Timedelta(seconds=300))].reset_index(drop=True)
    quotes = quote_window(inputs['clean_quotes']['path'], cutoff-pd.Timedelta(seconds=300), stop)
    economic = reference_quote_diagnostics(history, quotes, support).set_index('bin_start')
    for name in METRICS:
        s[name] = economic[name].reindex(s.bin_start).to_numpy()
    z = (s[FEATURES].to_numpy()-reference['mean'])/reference['sample_std']
    for index, name in enumerate(['z_ofi', 'z_intensity', 'z_duration']):
        s[name] = z[:, index]
    _frame(pd.read_parquet(directory/'opportunities.parquet'), o, 'deployed opportunity clock')
    _frame(pd.read_parquet(directory/'scored.parquet'), s, 'deployed detector scores and prior measurements')
    check_json_values(statistics['deployment'], _partition_statistics(o, s))
    counts.update(deployment_clock_rows_verified=len(o), deployment_scored_rows_verified=len(s),
                  calibration_scored_rows_verified=int(training_mask.sum()))
    return saved_out, counts


class CandidateGraph:
    """Compressed global graph of candidates, separate from the streaming scanner."""
    def __init__(self, start):
        self.start = pd.Timestamp(start)
        self.seconds = 0
        self.barriers = 0
        self.previous_candidate = None
        self.previous_sid = None
        self.tail_states = np.empty(0, dtype=object)
        self.tail_sid = np.empty(0, dtype='int64')
        self.nodes, self.onsets = [], []
        self.pending_nodes = []
        self.state_counts = {k: 0 for k in ('trigger', 'clear_active', 'clear_quiet', 'unknown')}

    def add(self, directory, opportunities, scored, quotes, support):
        directory = Path(directory)
        o, s = opportunities.reset_index(drop=True), scored.reset_index(drop=True)
        t = ns(o.bin_start)
        require(len(o) > 0, 'Empty episode chunk')
        np.testing.assert_array_equal(t, self.start.value+(self.seconds+np.arange(len(o)))*S)
        np.testing.assert_array_equal(ns(o.available_at), t+S)
        eligible = o.reason.eq('eligible').to_numpy()
        np.testing.assert_array_equal(ns(s.bin_start), t[eligible])
        for key in ('support_id', 'recorded_update_count', 'completed_residual_count'):
            np.testing.assert_array_equal(s[key], o.loc[eligible, key])
        sid = o.support_id.to_numpy(dtype='int64')
        quiet = ((sid >= 0) & o.flow_status.eq(1).to_numpy() &
                 o.recorded_update_count.eq(0).to_numpy() & o.completed_residual_count.eq(0).to_numpy())
        state = np.full(len(o), 'unknown', dtype=object)
        state[quiet] = 'clear_quiet'
        state[eligible] = np.where(s.candidate_event, 'trigger', 'clear_active')
        reason = np.where(state == 'unknown', o.reason.astype(str), '')
        changed = np.r_[self.previous_sid is not None and sid[0] != self.previous_sid, sid[1:] != sid[:-1]]
        unknown = state == 'unknown'
        barrier = changed | unknown
        positions = np.flatnonzero(barrier)
        cumulative = self.barriers+np.cumsum(barrier, dtype='int64')
        candidates = np.flatnonzero(state == 'trigger')
        global_position = self.seconds+candidates
        labels = cumulative[candidates]
        new_component = np.ones(len(candidates), dtype=bool)
        if len(candidates) > 1:
            new_component[1:] = (np.diff(global_position) > 5) | (np.diff(labels) != 0)
        previous_id = -1
        if self.previous_candidate is not None:
            before_position, before_barrier, previous_id = self.previous_candidate
            if len(candidates):
                new_component[0] = global_position[0]-before_position > 5 or labels[0] != before_barrier
        identifiers = previous_id+np.cumsum(new_component.astype('int64'))
        assignments = np.full(len(o), -1, dtype='int64')
        assignments[candidates] = identifiers
        expected_clock = pd.DataFrame({'bin_start': o.bin_start, 'available_at': o.available_at,
            'support_id': sid, 'observation_state': state, 'barrier_reason': reason, 'episode_id': assignments})
        _frame(pd.read_parquet(directory/'episode_clock.parquet'), expected_clock, 'global candidate graph clock')
        membership = expected_clock.loc[state == 'trigger', ['bin_start', 'available_at', 'episode_id']].reset_index(drop=True)
        _frame(pd.read_parquet(directory/'candidate_membership.parquet'), membership, 'global candidate graph membership')
        # resolve candidate nodes whose five-second future crossed the previous
        # chunk. No market-state or detector computation reads this future.
        for node in self.pending_nodes:
            matches = positions[self.seconds+positions > node['position']]
            if len(matches):
                j = int(matches[0])
                if self.seconds+j <= node['position']+5:
                    node.update(next_barrier=self.seconds+j,
                                barrier_changed=bool(changed[j]), barrier_reason=str(o.reason.iloc[j]))
        self.pending_nodes = []
        for pos, global_pos, identifier in zip(candidates, global_position, identifiers, strict=True):
            k = int(np.searchsorted(positions, pos, side='right'))
            node = {'position': int(global_pos), 'episode_id': int(identifier), 'support_id': int(sid[pos]),
                    'next_barrier': None, 'barrier_changed': False, 'barrier_reason': ''}
            if k < len(positions) and positions[k] <= pos+5:
                j = int(positions[k])
                node.update(next_barrier=self.seconds+j, barrier_changed=bool(changed[j]),
                            barrier_reason=str(o.reason.iloc[j]))
            if pos+5 >= len(o):
                self.pending_nodes.append(node)
            self.nodes.append(node)
        score_rows = np.flatnonzero(s.candidate_event.to_numpy())
        onset_rows = score_rows[new_component]
        onsets = s.iloc[onset_rows].reset_index(drop=True).copy()
        onsets['episode_id'] = identifiers[new_component]
        context_states = np.r_[self.tail_states, state]
        context_sid = np.r_[self.tail_sid, sid]
        clear = np.isin(context_states, ['clear_active', 'clear_quiet'])
        preceding = []
        for pos in candidates[new_component]:
            absolute = len(self.tail_states)+int(pos)
            value = 0
            for lag in range(1, 6):
                j = absolute-lag
                if j < 0 or not clear[j] or context_sid[j] != sid[pos]:
                    break
                value += 1
            preceding.append(value)
        onsets['onset_confirmed'] = np.asarray(preceding) == 5
        onsets['preceding_clear_seconds'] = np.asarray(preceding, dtype='int64')
        onsets['state_history_available'] = np.isfinite(onsets[['pre_quote_ofi_300s', 'pre_rv_300s_5s_bps']]).all(axis=1)
        onsets['state_sample_eligible'] = onsets.onset_confirmed & onsets.state_history_available
        quality = reference_quote_quality(onsets, quotes, support)
        for key in QUALITY_COLUMNS:
            onsets[key] = quality[key].to_numpy()
        _frame(pd.read_parquet(directory/'onsets.parquet'), onsets, 'global onset identities and raw quote quality')
        self.onsets.append(onsets)
        self.barriers += int(barrier.sum())
        self.seconds += len(o)
        self.previous_sid = int(sid[-1])
        self.tail_states, self.tail_sid = context_states[-5:], context_sid[-5:]
        if len(candidates):
            self.previous_candidate = (int(global_position[-1]), int(labels[-1]), int(identifiers[-1]))
        for name in self.state_counts:
            self.state_counts[name] += int(np.count_nonzero(state == name))

    def checkpoint(self):
        clear = 0
        for state, sid in zip(self.tail_states[::-1], self.tail_sid[::-1], strict=True):
            if state not in ('clear_active', 'clear_quiet') or sid != self.previous_sid:
                break
            clear += 1
        opened = None
        if self.nodes:
            last = self.nodes[-1]
            if last['next_barrier'] is None and last['position']+5 >= self.seconds:
                members = [n for n in self.nodes if n['episode_id'] == last['episode_id']]
                first = members[0]
                onset = next(f.loc[f.episode_id.eq(last['episode_id'])].iloc[0]
                             for f in reversed(self.onsets) if f.episode_id.eq(last['episode_id']).any())
                opened = {'episode_id': last['episode_id'], 'support_id': first['support_id'],
                    'first_candidate_bin': self.start.value+first['position']*S,
                    'detected_at': self.start.value+(first['position']+1)*S,
                    'last_candidate_bin': self.start.value+last['position']*S,
                    'last_candidate_available_at': self.start.value+(last['position']+1)*S,
                    'candidate_bins': len(members), 'span_seconds': last['position']-first['position']+1,
                    'onset_confirmed': bool(onset.onset_confirmed),
                    'preceding_clear_seconds': int(onset.preceding_clear_seconds)}
        return {'version': 1, 'next_episode_id': self.nodes[-1]['episode_id']+1 if self.nodes else 0,
                'next_timestamp_ns': int(self.start.value+self.seconds*S),
                'previous_support_id': self.previous_sid, 'clear_run': clear,
                'open_episode': opened, 'finished': False}

    def finish(self):
        onsets = pd.concat(self.onsets, ignore_index=True)
        firsts = {int(row.episode_id): row for row in onsets.itertuples(index=False)}
        grouped = {}
        for node in self.nodes:
            grouped.setdefault(node['episode_id'], []).append(node)
        rows = []
        for identifier, members in sorted(grouped.items()):
            first, last = members[0], members[-1]
            onset = firsts[identifier]
            time = lambda pos: pd.Timestamp(self.start.value+pos*S, tz='UTC')
            closure, stop, why = pd.NaT, pd.NaT, ''
            if last['next_barrier'] is not None:
                stop = time(last['next_barrier'])
                if last['barrier_changed']:
                    termination, why = 'censored_support_change', 'support_id_change'
                else:
                    termination, why = 'censored_unknown', last['barrier_reason']
            elif last['position']+5 >= self.seconds:
                termination, stop, why = 'censored_sample_end', time(self.seconds), 'development_end'
            else:
                termination, closure = 'closed_no_trigger', time(last['position']+6)
            rows.append({'episode_id': identifier, 'support_id': first['support_id'],
                'first_candidate_bin': time(first['position']), 'detected_at': time(first['position']+1),
                'last_candidate_bin': time(last['position']), 'last_candidate_available_at': time(last['position']+1),
                'candidate_bins': len(members), 'span_seconds': last['position']-first['position']+1,
                'onset_confirmed': bool(onset.onset_confirmed), 'preceding_clear_seconds': int(onset.preceding_clear_seconds),
                'termination': termination, 'closure_known_at': closure, 'observation_stop_at': stop, 'censor_reason': why})
        episodes = pd.DataFrame(rows)
        for name in ('first_candidate_bin', 'detected_at', 'last_candidate_bin', 'last_candidate_available_at',
                     'closure_known_at', 'observation_stop_at'):
            episodes[name] = pd.to_datetime(episodes[name], utc=True)
        require(int(episodes.candidate_bins.sum()) == len(self.nodes), 'Global candidate conservation')
        return episodes, onsets


def registered_hashes(run):
    """Check registered file lineage without re-estimating prior model families."""
    cache, seen, expected, owners = {}, {}, {}, {}
    registrations = 0
    root_manifest = read(Path(run)/'manifest.json')
    repair = root_manifest.get('state_precision_repair')
    require(isinstance(repair, dict), 'Explicit state precision repair provenance required')
    source_record = root_manifest['inputs'].get('numerical_build_manifest')
    require(isinstance(source_record, dict) and set(source_record) == {'path', 'sha256'} and
            repair.get('source_manifest') == source_record, 'Precision repair source is not exactly registered')
    source_path = Path(source_record['path']).resolve()
    require(source_path.name == 'manifest.json' and source_path != (Path(run)/'manifest.json').resolve(),
            'Distinct immutable numerical-build manifest required')
    source_directory = source_path.parent
    require(source_record['sha256'] == PRECISION_SOURCE_SHA256 and
            source_directory.name == 'step07c_chronological_panel_20260930',
            'Precision reuse is limited to the preserved, specifically diagnosed failed build')
    require(repair.get('changed_metric') == 'pre_rv_300s_5s_bps' and
            repair.get('inherited_cycles_preserved') == [0, 1] and repair.get('model_refits') == 0 and
            repair.get('reference_tolerance') == {'rtol': 1e-9, 'atol': 1e-9},
            'Unapproved precision repair scope or numerical tolerance')

    def check(path, value):
        nonlocal registrations
        path = Path(path).resolve()
        digest = value['sha256'] if isinstance(value, dict) else value
        key = str(path)
        require(key not in expected or expected[key] == digest, 'Conflicting provenance: '+key)
        expected[key] = digest
        if key not in cache:
            h = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    h.update(block)
            cache[key] = h.hexdigest()
        require(cache[key] == digest, 'Changed registered file: '+key)
        registrations += 1

    def visit(directory, root=False):
        directory = Path(directory).resolve()
        if str(directory) in seen:
            return
        m = read(directory/'manifest.json')
        if m.get('stage') == '02a_whole_hour_coverage':
            require(m['status'] == 'coverage_audited_recovery_waiting_for_vendor_access' and
                    m['coverage_certified'] is False, 'Historical coverage scope changed')
            require(read(directory/'coverage_verification.json')['status'] == 'passed', 'Coverage context unverified')
            return
        seen[str(directory)] = m
        if not root:
            if directory == source_directory:
                require(m['status'] == 'failed_independent_verification' and
                        m.get('stage') == '07c_full_chronological_baseline_panel' and
                        m.get('scope') == 'full_cached_structural_baseline_detector_episode_states_only' and
                        m.get('production_inference_approved') is False and m.get('downstream_ready') is False,
                        'Only the exactly registered failed numerical build may be reused without prior verification')
                require(cache.get(str(source_path)) == source_record['sha256'],
                        'Unverified numerical source manifest hash was not checked')
            else:
                require(m['status'].startswith('verified') and read(directory/'verification.json')['status'] == 'passed',
                        'Unverified parent '+str(directory))
        parents = []
        for record in m.get('inputs', m.get('input_hashes', {})).values():
            check(record['path'], record['sha256'])
            if Path(record['path']).name == 'manifest.json':
                parents.append(Path(record['path']).parent)
        for record in m.get('input_files', []):
            check(record['path'], record['sha256'])
        for name, digest in m.get('outputs', m.get('output_sha256', {})).items():
            check(directory/name, digest)
            owners[str((directory/name).resolve())] = str(directory)
        for name, digest in m.get('code_sha256', m.get('code_files', {})).items():
            check(directory/'code'/name, digest)
        for name, digest in m.get('postfit_code_sha256', {}).items():
            check(directory/'postfit_code'/name, digest)
        for name, digest in m.get('ancestor_hashes', {}).items():
            check(name, digest)
            if Path(name).name == 'manifest.json':
                parents.append(Path(name).parent)
        for name, digest in m.get('previous_files', {}).items():
            check(Path(m['snapshot_path'])/name, digest)
        for parent in parents:
            visit(parent)
    visit(run, root=True)
    m = seen[str(Path(run).resolve())]
    for name in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals',
                 'quote_durations', 'terminal_spells', 'frozen_regression_scaling']:
        require(str(Path(m['inputs'][name]['path']).resolve()) in owners,
                'Unregistered upstream output '+name)
    transitions = [(Path(path), parent) for path, parent in seen.items()
                   if parent.get('stage') == '07b_first_chronological_transition']
    require(len(transitions) == 1, 'One verified transition ancestor required')
    previous, pm = transitions[0]
    for key in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals',
                'quote_durations', 'terminal_spells', 'frozen_regression_scaling']:
        actual, prior = m['inputs'][key], pm['inputs'][key]
        require(Path(actual['path']).resolve() == Path(prior['path']).resolve() and
                actual['sha256'] == prior['sha256'], 'Transition lineage mismatch: '+key)
    require(str(source_directory) in seen, 'Registered failed numerical source was not traversed')
    source_manifest = seen[str(source_directory)]
    for key in ['clock', 'calibration_parameters', 'clean_quotes', 'support_intervals',
                'quote_durations', 'terminal_spells', 'frozen_regression_scaling']:
        actual, prior = m['inputs'][key], source_manifest['inputs'][key]
        require(Path(actual['path']).resolve() == Path(prior['path']).resolve() and
                actual['sha256'] == prior['sha256'], 'Failed numerical source uses other ancestors: '+key)
    return m, previous, {'hash_registrations_verified': registrations, 'unique_files_hashed': len(cache),
                         'parent_runs_verified': len(seen)-2,
                         'failed_numerical_builds_hash_checked': 1}


def precision_frame_difference(old, new, name):
    """Allow one measurement to change; every other serialized value is exact."""
    require(list(old.columns) == list(new.columns) and PRECISION_METRIC in old,
            'Precision repair changed frame columns: '+name)
    pd.testing.assert_frame_equal(old.drop(columns=PRECISION_METRIC), new.drop(columns=PRECISION_METRIC),
                                  check_exact=True, check_dtype=False, check_categorical=False,
                                  obj='Unchanged fields of '+name)
    a, b = old[PRECISION_METRIC].to_numpy(), new[PRECISION_METRIC].to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg='Changed measurement missingness: '+name)
    require(np.logical_or(np.isfinite(a), np.isnan(a)).all() and
            np.logical_or(np.isfinite(b), np.isnan(b)).all(), 'Infinite measurement: '+name)
    finite = np.isfinite(a)
    require((a[finite] >= 0).all() and (b[finite] >= 0).all(), 'Negative variation: '+name)
    differences = np.abs(a[finite]-b[finite])
    outside = finite & ~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    return {'rows': len(a), 'finite_rows': int(finite.sum()),
        'exact_value_changes': int(np.count_nonzero(a[finite] != b[finite])),
        'existing_tolerance_exceedances': int(outside.sum()),
        'max_absolute_change_bps': float(differences.max(initial=0))}, outside


def verify_precision_reuse(run, manifest):
    """Bounded comparisons establish that no fits, classifications or counts changed.

    Registered hashes have already been checked for both physical builds. The
    whitelist below concerns differences between those builds; it never grants
    correctness to the changed measurement. Independent cycle reconstruction
    subsequently verifies every new measurement at the original tolerance.
    """
    run = Path(run)
    source = Path(manifest['inputs']['numerical_build_manifest']['path']).parent
    old_manifest = read(source/'manifest.json')
    old_outputs, new_outputs = old_manifest['outputs'], manifest['outputs']
    root_files = {'cycle_schedule.parquet', 'cycle_statistics.parquet', 'daily.parquet',
        'episodes.parquet', 'final_censored_episodes.parquet', 'final_episode_state.json',
        'kernel_build.json', 'monthly.parquet', 'onsets.parquet', 'progress.json',
        'statistics.json', 'test_evidence.json'}
    core = {name for name in old_outputs if not name.startswith('verification_attempts/')}
    require(all(name in root_files or Path(name).parts[0] in ('observations', 'cycles') for name in core),
            'Unexpected original artifact requires an explicit reuse policy')
    require(root_files <= core and core <= set(new_outputs), 'A source numerical artifact was omitted')
    cycle_ids = {int(Path(name).parts[1]) for name in core if Path(name).parts[0] == 'cycles'}
    require(cycle_ids == set(range(60)), 'Failed source must contain all 60 completed cycles')
    changed_frames = {'onsets.parquet'} | {
        f'cycles/{cycle:03d}/{name}.parquet' for cycle in range(2, 60) for name in ('scored', 'onsets')}
    changed_manifests = {f'cycles/{cycle:03d}/manifest.json' for cycle in range(2, 60)}
    require(changed_frames | changed_manifests <= core, 'Repair whitelist is missing from source artifacts')
    unchanged = core-changed_frames-changed_manifests
    for name in sorted(unchanged):
        require(old_outputs[name] == new_outputs[name], 'Unauthorized numerical artifact change: '+name)
    for cycle in range(2, 60):
        name = f'cycles/{cycle:03d}/manifest.json'
        old, new = read(source/name), read(run/name)
        require({k: v for k, v in old.items() if k != 'outputs'} ==
                {k: v for k, v in new.items() if k != 'outputs'}, 'Cycle metadata changed: '+name)
        require(set(old['outputs']) == set(new['outputs']), 'Cycle artifact set changed: '+name)
        for item, digest in old['outputs'].items():
            if item not in ('scored.parquet', 'onsets.parquet'):
                require(new['outputs'][item] == digest, 'Unauthorized cycle artifact change: '+name+'/'+item)
    audit = read(run/manifest['state_precision_repair']['audit'])
    check_json_values(audit, {'status': 'built_pending_independent_verification', 'metric': PRECISION_METRIC,
        'cycles_checked': 60, 'cycles_recomputed': 58, 'model_refits': 0, 'missingness_unchanged': True,
        'detector_and_episode_columns_unchanged': True})
    require(len(audit['cycles']) == 60 and [r['cycle_id'] for r in audit['cycles']] == list(range(60)),
            'Precision audit cycle order or coverage differs')
    changes = {}
    for name in sorted(changed_frames):
        old, new = pd.read_parquet(source/name), pd.read_parquet(run/name)
        delta, outside = precision_frame_difference(old, new, name)
        if name.startswith('cycles/'):
            cycle = int(Path(name).parts[1])
            record = audit['cycles'][cycle]
            require(record['inherited_preserved'] is False, 'Recomputed cycle marked inherited')
            if name.endswith('/scored.parquet'):
                check_json_values(record, delta)
                candidate = int(np.count_nonzero(outside & old.candidate_event.to_numpy()))
                require(record['candidate_tolerance_exceedances'] == candidate, 'Candidate repair count differs')
                changes.setdefault(cycle, {}).update(scored=delta, candidate=candidate)
            else:
                check_json_values(record['onsets'], delta)
                changes.setdefault(cycle, {})['onsets'] = delta
        del old, new
    for record in audit['cycles'][:2]:
        require(record['inherited_preserved'] is True and record['existing_tolerance_exceedances'] == 0 and
                record['candidate_tolerance_exceedances'] == 0 and
                record['onsets']['existing_tolerance_exceedances'] == 0, 'Inherited cycle required a numerical repair')
    check_json_values(audit, {
        'scored_tolerance_exceedances_corrected': sum(x['scored']['existing_tolerance_exceedances'] for x in changes.values()),
        'candidate_tolerance_exceedances_corrected': sum(x['candidate'] for x in changes.values()),
        'onset_tolerance_exceedances_corrected': sum(x['onsets']['existing_tolerance_exceedances'] for x in changes.values()),
        'max_absolute_scored_change_bps': max(x['scored']['max_absolute_change_bps'] for x in changes.values())})
    for key in ('test_evidence', 'reference_test_evidence'):
        name = manifest['state_precision_repair'][key]
        require(name in new_outputs, 'Precision test evidence not registered')
        evidence = read(run/name)
        require((evidence.get('passed') is True or evidence.get('status') == 'passed') and
                not evidence['failures'] and not evidence['errors'], 'Precision tests failed')
        for name, digest in evidence['source_sha256'].items():
            candidates = [run/'code'/name, source/'code'/name, source/'postfit_code'/name]
            require(any(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                        for path in candidates), 'Precision tested source lacks its saved matching version: '+name)
    return {'unchanged_numerical_artifacts_preserved': len(unchanged),
            'precision_frame_nonvariation_fields_verified': len(changed_frames),
            'precision_scored_rows_compared': sum(x['scored']['rows'] for x in changes.values()),
            'precision_scored_values_changed': sum(x['scored']['exact_value_changes'] for x in changes.values()),
            'precision_scored_tolerance_exceedances_corrected': audit['scored_tolerance_exceedances_corrected'],
            'precision_candidate_tolerance_exceedances_corrected': audit['candidate_tolerance_exceedances_corrected'],
            'precision_onset_tolerance_exceedances_corrected': audit['onset_tolerance_exceedances_corrected']}


def verify_inherited_measurements(scored, inputs, support, cutoff, stop):
    """Recheck the preserved pilot's V directly rather than assuming inheritance suffices."""
    start = cutoff-pd.Timedelta(seconds=300)
    raw = pd.read_parquet(inputs['clock']['path'], filters=[('bin_start', '>=', start), ('bin_start', '<', stop)])
    np.testing.assert_array_equal(ns(raw.bin_start), np.arange(start.value, stop.value, S))
    factors = pd.read_parquet(inputs['calibration_parameters']['path'],
        filters=[('effective_date', '>=', start.normalize()), ('effective_date', '<', stop),
                 ('policy', '==', 'structural_only')])
    clock = reference_clock(raw, factors, 'structural_only')
    quotes = quote_window(inputs['clean_quotes']['path'], start, stop)
    reference = reference_quote_diagnostics(clock, quotes, support).set_index('bin_start')
    expected = reference[PRECISION_METRIC].reindex(scored.bin_start)
    np.testing.assert_array_equal(scored[PRECISION_METRIC].isna(), expected.isna())
    np.testing.assert_allclose(scored[PRECISION_METRIC], expected, rtol=1e-9, atol=1e-9, equal_nan=True,
                               err_msg='Inherited pre-event variation must satisfy the direct reference')
    return len(scored)


def daily_reference(o, s, onsets, start, stop):
    dates = pd.date_range(start, stop, freq='D', inclusive='left')
    output = pd.DataFrame({'date': dates})
    reason_names = ['unsupported', 'invalid_flow', 'seasonal_unavailable', 'quiet_no_duration', 'incomplete_duration', 'eligible']
    for name in reason_names:
        counts = o.loc[o.reason.eq(name)].groupby(o.loc[o.reason.eq(name), 'bin_start'].dt.normalize()).size()
        output['reason_'+name] = [int(counts.get(date, 0)) for date in dates]
    eligible = o.reason.eq('eligible').to_numpy()
    state = np.full(len(o), 'unknown', dtype=object)
    quiet = o.support_id.ge(0) & o.flow_status.eq(1) & o.recorded_update_count.eq(0) & o.completed_residual_count.eq(0)
    state[quiet] = 'clear_quiet'
    state[eligible] = np.where(s.candidate_event, 'trigger', 'clear_active')
    normalized = o.bin_start.dt.normalize()
    for name in ('trigger', 'clear_active', 'clear_quiet', 'unknown'):
        counts = normalized.loc[state == name].value_counts()
        output['state_'+name] = [int(counts.get(date, 0)) for date in dates]
    for name, frame, mask in [('candidate_bins', s, s.candidate_event),
        ('episode_fragments', onsets, np.ones(len(onsets), dtype=bool)),
        ('confirmed_onsets', onsets, onsets.onset_confirmed),
        ('state_sample_eligible', onsets, onsets.state_sample_eligible)]:
        counts = frame.loc[mask, 'bin_start'].dt.normalize().value_counts()
        output[name] = [int(counts.get(date, 0)) for date in dates]
    output['calendar_seconds'] = output.filter(like='reason_').sum(axis=1)
    np.testing.assert_array_equal(output.calendar_seconds, 86400)
    return output


def verify_panel(run):
    """Verify the full structural baseline with bounded source/clock memory."""
    from src.verify_event_episodes import reference_policy_statistics
    run = Path(run)
    manifest, previous, counts = registered_hashes(run)
    counts.update(verify_precision_reuse(run, manifest))
    require(manifest['stage'] == '07c_full_chronological_baseline_panel' and
            manifest['scope'] == 'full_cached_structural_baseline_detector_episode_states_only' and
            manifest['primary_policy'] == 'structural_only' and
            manifest['production_inference_approved'] is False and manifest['downstream_ready'] is False,
            'Unexpected panel scope or inference promotion')
    origin = pd.Timestamp('2024-01-01', tz='UTC')
    first = pd.Timestamp('2024-04-22', tz='UTC')
    continuation = pd.Timestamp('2024-05-20', tz='UTC')
    stop = pd.Timestamp('2026-08-01', tz='UTC')
    for key, value in [('calendar_origin', origin), ('first_cutoff', first),
                       ('continuation_cutoff', continuation), ('deployment_end', stop)]:
        require(pd.Timestamp(manifest[key]) == value, 'Panel calendar changed: '+key)
    check_json_values(manifest['seasonal_configuration'], {'seasonal_days': 56, 'training_days': 56,
        'validation_days': int((stop-origin).days)-112, 'min_prior_dates': 4,
        'min_exposure_seconds': 1800, 'min_completions': 100})
    evidence = read(run/'test_evidence.json')
    require(evidence['status'] == 'passed' and evidence['errors'] == evidence['failures'] == 0, 'Tests did not pass')
    for name, expected in evidence['test_source_sha256'].items():
        require(hashlib.sha256((run/'full_test_sources'/name).read_bytes()).hexdigest() == expected,
                'Executed/saved test mismatch: '+name)
    cutoffs = pd.date_range(first, stop, freq='14D', inclusive='left')
    schedule = pd.DataFrame({'cycle_id': np.arange(len(cutoffs)),
        'training_start': cutoffs-pd.Timedelta(days=56), 'parameter_cutoff': cutoffs,
        'deployment_end': [min(c+pd.Timedelta(days=14), stop) for c in cutoffs]})
    _frame(pd.read_parquet(run/'cycle_schedule.parquet'), schedule, 'anchored 56/14 cycle schedule')
    inputs = manifest['inputs']
    observations = run/'observations'
    source = verify_observations(observations, inputs, origin, stop)
    source_saved = read(observations/'statistics.json')
    check_json_values(source_saved, source)
    require(source_saved['stream_batch_size'] == 500_000 and
            source_saved['source_batches'] == pq.ParquetFile(observations/'raw_spells.parquet').num_row_groups,
            'Source materialization batch accounting differs')
    counts.update(source_rows_verified=source['source_spell_rows'], seasonal_rows_verified=source['seasonal_parameter_rows'])
    support = pd.read_parquet(inputs['support_intervals']['path'])
    support = support.loc[support.policy.eq('structural_only')].reset_index(drop=True)
    # seed from the actual issued forecasts saved by the verified pilot, not a
    # fresh refilter under either old or new parameters.
    old_path = pd.read_parquet(previous/'model_path.parquet', filters=[('end_at', '>=', continuation)])
    old_forecasts = pd.read_parquet(previous/'forecasts.parquet').set_index('origin_quote_id').loc[old_path.origin_quote_id]
    require(old_forecasts.forecast_role.isin(['carried_origin_forecast', 'new_origin_forecast']).all(),
            'Unissued calibration forecast in panel carry seed')
    carry = old_path[ISSUED[:4]].reset_index(drop=True)
    carry['psi'] = old_forecasts.psi_for_residual.to_numpy()
    carry['residual_parameter_cutoff'] = old_forecasts.residual_parameter_cutoff.to_numpy()
    opened = pd.read_parquet(observations/'seasonal_spells.parquet',
        filters=[('origin_at', '<', continuation), ('end_at', '>=', continuation)])
    opened = opened.loc[np.isfinite(opened.scale_sec) & opened.exposure_ns.gt(0)].reset_index(drop=True)
    _frame(carry[ISSUED[:4]], opened[ISSUED[:4]], 'prefix forecast seed source identity')
    pd.testing.assert_frame_equal(pd.read_parquet(run/'cycles/001/carry_out.parquet'), carry,
                                  check_exact=True, check_dtype=False, check_categorical=False)
    del old_path, old_forecasts, opened
    graph = CandidateGraph(first)
    daily_parts, cycle_rows = [], []
    count_keys = ['training_spells_verified', 'new_issued_forecasts_verified', 'carried_forecasts_verified',
                  'completed_residuals_verified', 'deployment_clock_rows_verified',
                  'deployment_scored_rows_verified', 'calibration_scored_rows_verified']
    counts.update({key: 0 for key in count_keys})
    counts['inherited_scored_variation_rows_verified'] = 0
    for row in schedule.itertuples(index=False):
        directory = run/'cycles'/f'{row.cycle_id:03d}'
        cm = read(directory/'manifest.json')
        inherited = row.cycle_id < 2
        check_json_values(cm, {'cycle_id': int(row.cycle_id), 'training_start': row.training_start.isoformat(),
            'parameter_cutoff': row.parameter_cutoff.isoformat(), 'deployment_end': row.deployment_end.isoformat(),
            'inherited': inherited, 'status': 'built_pending_independent_verification'})
        for name, expected in cm['outputs'].items():
            require(manifest['outputs']['cycles/'+f'{row.cycle_id:03d}'+'/'+name] == expected,
                    'Cycle/root output registration differs')
        o = pd.read_parquet(directory/'opportunities.parquet')
        s = pd.read_parquet(directory/'scored.parquet')
        if inherited:
            filters = [('bin_start', '>=', row.parameter_cutoff), ('bin_start', '<', row.deployment_end)]
            for actual, filename in [(o, 'combined_opportunities.parquet'), (s, 'combined_scored.parquet')]:
                old = pd.read_parquet(previous/filename, filters=filters)
                pd.testing.assert_frame_equal(actual, old, check_exact=True, check_dtype=False, check_categorical=False)
            counts['inherited_scored_variation_rows_verified'] += verify_inherited_measurements(
                s, inputs, support, row.parameter_cutoff, row.deployment_end)
        else:
            parameters = read(directory/'parameters.json')
            for key in ('training_start', 'parameter_cutoff', 'deployment_end'):
                require(parameters[key] == cm[key], 'Cycle parameter timing mismatch')
            require(parameters['optimization_attempts'] == read(directory/'optimization_attempts.json'), 'Attempt logs differ')
            carry, checked = verify_cycle(directory, observations, inputs, carry, support)
            for key in count_keys:
                counts[key] += checked[key]
        quotes = quote_window(inputs['clean_quotes']['path'], row.parameter_cutoff-pd.Timedelta(seconds=300), row.deployment_end)
        graph.add(directory, o, s, quotes, support)
        checkpoint = graph.checkpoint()
        check_json_values(read(directory/'episode_state.json'), checkpoint)
        onsets = graph.onsets[-1]
        daily = daily_reference(o, s, onsets, row.parameter_cutoff, row.deployment_end)
        _frame(pd.read_parquet(directory/'daily.parquet'), daily, 'cycle calendar accounting')
        daily_parts.append(daily)
        stats = read(directory/'statistics.json')
        deployment = _partition_statistics(o, s)
        check_json_values(stats, {'deployment': deployment, 'onset_fragments': len(onsets),
            'confirmed_onsets': int(onsets.onset_confirmed.sum()),
            'state_eligible_onsets': int(onsets.state_sample_eligible.sum())})
        cycle_rows.append({'cycle_id': int(row.cycle_id), 'parameter_cutoff': row.parameter_cutoff,
            'deployment_end': row.deployment_end, 'inherited': inherited,
            **{k: v for k, v in deployment.items() if k != 'reason_counts'},
            'onset_fragments': len(onsets), 'confirmed_onsets': int(onsets.onset_confirmed.sum()),
            'state_eligible_onsets': int(onsets.state_sample_eligible.sum()),
            'duration_diagnostic_flags': '|'.join(stats.get('duration_working_model_diagnostics', {}).get('diagnostic_flags', []))})
        print(f'Independent cycle {row.cycle_id:03d} passed: {len(s):,} scored bins, {len(onsets):,} onsets', flush=True)
        del o, s, quotes, onsets, daily
        gc.collect()
    episodes, onsets = graph.finish()
    _frame(pd.read_parquet(run/'episodes.parquet'), episodes, 'full graph episodes')
    _frame(pd.read_parquet(run/'onsets.parquet'), onsets, 'full graph onsets')
    for row in schedule.itertuples(index=False):
        directory = run/'cycles'/f'{row.cycle_id:03d}'
        selected = ((episodes.termination.eq('closed_no_trigger') & episodes.closure_known_at.gt(row.parameter_cutoff)
                     & episodes.closure_known_at.le(row.deployment_end)) |
                    (episodes.termination.isin(['censored_unknown', 'censored_support_change'])
                     & episodes.observation_stop_at.ge(row.parameter_cutoff) & episodes.observation_stop_at.lt(row.deployment_end)))
        closed = episodes.loc[selected].reset_index(drop=True)
        _frame(pd.read_parquet(directory/'closed_episodes.parquet'), closed, 'cycle finalized episode emission')
        require(read(directory/'statistics.json')['finalized_episodes'] == len(closed), 'Finalized episode count differs')
    final = episodes.loc[episodes.termination.eq('censored_sample_end')].reset_index(drop=True)
    _frame(pd.read_parquet(run/'final_censored_episodes.parquet'), final, 'actual-end censored episodes')
    final_checkpoint = graph.checkpoint()
    final_checkpoint.update(open_episode=None, finished=True)
    check_json_values(read(run/'final_episode_state.json'), final_checkpoint)
    daily = pd.concat(daily_parts, ignore_index=True)
    _frame(pd.read_parquet(run/'daily.parquet'), daily, 'full daily accounting')
    month_keys = daily.date.dt.strftime('%Y-%m')
    monthly = pd.DataFrame([{'month': month, **{key: int(part[key].sum()) for key in daily.columns if key != 'date'}}
                           for month, part in daily.groupby(month_keys, sort=True)])
    _frame(pd.read_parquet(run/'monthly.parquet'), monthly, 'full monthly accounting')
    _frame(pd.read_parquet(run/'cycle_statistics.parquet'), pd.DataFrame(cycle_rows), 'full cycle accounting')
    old_onsets, old_episodes = pd.read_parquet(previous/'onsets.parquet'), pd.read_parquet(previous/'episodes.parquet')
    prefix = onsets.loc[onsets.bin_start.lt(continuation)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, old_onsets, check_exact=True, check_dtype=False, check_categorical=False)
    previously_closed = old_episodes.loc[old_episodes.termination.ne('censored_sample_end')].reset_index(drop=True)
    pd.testing.assert_frame_equal(episodes.set_index('episode_id').loc[previously_closed.episode_id].reset_index(),
                                  previously_closed, check_exact=True, check_dtype=False, check_categorical=False)
    # independent legacy summary needs a clock only for its state counters; use
    # an empty reference and replace those exact counters from the compressed graph.
    epstats = reference_policy_statistics(pd.DataFrame({'observation_state': pd.Series(dtype=object)}), episodes, onsets)
    epstats.update(clock_seconds=graph.seconds, observation_states=graph.state_counts, candidate_bins=len(graph.nodes))
    reasons = {name.removeprefix('reason_'): int(daily[name].sum()) for name in daily if name.startswith('reason_')}
    expected = {'scope': manifest['scope'], 'primary_policy': 'structural_only', 'source': source_saved,
        'cycles': len(schedule), 'inherited_cycles': 2, 'new_fitted_cycles': len(schedule)-2,
        'final_cycle_days': int((stop-schedule.parameter_cutoff.iloc[-1]).days),
        'deployment': {'calendar_seconds': graph.seconds, 'eligible_bins': reasons['eligible'],
            'candidate_bins': len(graph.nodes), 'candidate_share_of_eligible': len(graph.nodes)/reasons['eligible'],
            'reason_counts': reasons}, 'combined_episodes': epstats,
        'previous_onsets_preserved': len(prefix), 'previous_closed_episodes_preserved': len(previously_closed),
        'new_onset_fragments': len(onsets)-len(prefix), 'fitted_duration_specifications': 1,
        'empirical_response_models_fitted': 0, 'full_panel_extension_performed': True,
        'fixed_sensitivities_extended': False, 'extended_responses_computed': False,
        'production_inference_approved': False, 'downstream_ready': False}
    first_times = ns(episodes.first_candidate_bin)
    last_times = ns(episodes.last_candidate_bin)
    boundaries = ns(schedule.parameter_cutoff.iloc[1:])
    # count an episode once even if its candidate chain spans several cutoffs.
    crossing = np.searchsorted(boundaries, last_times, side='right') > np.searchsorted(boundaries, first_times, side='right')
    expected['episodes_with_candidates_across_cutoffs'] = int(crossing.sum())
    fitted_stats = [read(run/'cycles'/f'{i:03d}'/'statistics.json') for i in range(2, len(schedule))]
    fitted_parameters = [read(run/'cycles'/f'{i:03d}'/'parameters.json') for i in range(2, len(schedule))]
    flags = sorted({flag for item in fitted_stats for flag in item['duration_working_model_diagnostics']['diagnostic_flags']})
    expected['new_cycle_diagnostics'] = {
        'constraint_warning_cycles': sum(bool(item['numerical_constraint_warning']) for item in fitted_parameters),
        'threshold_min': min(item['detector']['threshold'] for item in fitted_parameters),
        'threshold_max': max(item['detector']['threshold'] for item in fitted_parameters),
        'diagnostic_flag_cycles': {flag: sum(flag in item['duration_working_model_diagnostics']['diagnostic_flags']
                                           for item in fitted_stats) for flag in flags},
        'carried_forecast_transitions': sum(item['carry']['carried_in'] for item in fitted_stats),
        'completed_carried_residuals': sum(item['carry']['completed_carried_residuals'] for item in fitted_stats),
        'carry_origins_before_training_start': sum(item['carry']['carried_origin_before_training_start'] for item in fitted_stats)}
    check_json_values(read(run/'statistics.json'), expected)
    counts.update(cycles_verified=len(schedule), newly_fitted_cycles_verified=len(schedule)-2,
        full_calendar_seconds_verified=graph.seconds, candidate_rows_verified=len(graph.nodes),
        episode_rows_verified=len(episodes), onset_rows_verified=len(onsets),
        daily_rows_verified=len(daily), monthly_rows_verified=len(monthly),
        inherited_onsets_preserved=len(prefix), inherited_terminated_episodes_preserved=len(previously_closed))
    return {'status': 'passed', **counts,
        'scope': 'Complete cached structural-only detector/episode/state panel; fixed sensitivities, responses and inference remain pending.',
        'methods': ['Exactly registered failed-build reuse audited separately; all unchanged numerical artifacts and non-variation fields preserved, inherited variation independently checked',
            'Streaming independent incoming waits and terminal records; direct prior-spell same-weekday exposure contributions',
            'Every 56/14 cycle reconstructed with clipped training, SciPy likelihood/filter, origin-vintage ledger and raw covariance/q99',
            'Strict preceding quote marks and states, including the predecessor across computational read boundaries',
            'Global candidate graph compressed from bounded clock chunks; independent carry checkpoints, onset quality, closure and calendar ledgers',
            'Registered input/output/source/test hashes, exact inherited values and complete denominator/candidate conservation'],
        'production_inference_approved': False}

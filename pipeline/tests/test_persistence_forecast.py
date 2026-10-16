import json
import unittest

import numpy as np
import pandas as pd

from src.persistence_forecast import fit_forecasts, evaluate_forecasts, METHODS, CONTINUOUS


START = pd.Timestamp('2024-05-06T00:00:00+00:00')
END = pd.Timestamp('2024-05-20T00:00:00+00:00')


def fixture():
    rng = np.random.default_rng(921)
    clock = pd.date_range(pd.Timestamp('2024-04-22T00:00:00+00:00'), END, freq='2h', inclusive='left')
    clock = clock[clock.weekday != 5]
    n = len(clock)
    signed = rng.normal(size=n); magnitude = abs(signed); variation = rng.normal(size=n)
    predictors = pd.DataFrame({'episode_id': np.arange(n), 'bin_start': clock,
        'forecast_at': clock + pd.Timedelta(1, unit='s'), 'signed_pressure': signed,
        'pressure_magnitude': magnitude, 'variation_standardized': variation,
        'variation_x_magnitude': variation * magnitude, 'pre_spread_pips': rng.uniform(.1, .8, n),
        'log_pre_displayed_size': rng.normal(size=n), 'onset_hour_utc': clock.hour,
        'onset_weekday_utc': clock.weekday})
    origins = predictors[['episode_id', 'bin_start', 'forecast_at']].copy()
    origins['origin_eligible'] = True
    origins['initial_pips'] = np.resize([0., -.5, .4], n)
    labels = origins[['episode_id', 'forecast_at']].copy()
    labels['horizon_sec'] = 300
    labels['label_ready_at'] = labels.forecast_at + pd.Timedelta(300, unit='s')
    labels['label_available'] = True; labels['label_status'] = 'available'
    labels['continuation_pips'] = .2 + .1 * signed + .05 * variation + .08 * magnitude * variation + rng.normal(0, .02, n)
    result = {'origins': origins, 'predictors': predictors, 'labels': labels}
    refresh(result)
    return result


def refresh(data):
    o, labels = data['origins'], data['labels']
    lookup = labels.set_index('episode_id')
    ready = o.forecast_at + pd.Timedelta(300, unit='s')
    available = o.episode_id.map(lookup.label_available)
    train = o.forecast_at.lt(START) & o.origin_eligible & ready.le(START) & available
    prediction = o.forecast_at.ge(START) & o.forecast_at.lt(END)
    members = []
    for role, mask in [('training', train), ('prediction', prediction)]:
        part = o.loc[mask, ['episode_id', 'forecast_at', 'origin_eligible']].copy()
        part['fold_id'] = 1; part['role'] = role
        part['label_ready_at'] = part.forecast_at + pd.Timedelta(300, unit='s')
        part['label_available'] = part.episode_id.map(lookup.label_available)
        members.append(part)
    data['membership'] = pd.concat(members, ignore_index=True)
    data['folds'] = pd.DataFrame([{'fold_id': 1, 'fit_cutoff': START, 'prediction_start': START,
        'prediction_end_exclusive': END, 'training_origins': int(train.sum()),
        'prediction_origins': int(prediction.sum()), 'prediction_origin_eligible': int((prediction & o.origin_eligible).sum())}])


def change_time(data, event, onset):
    for name in ['origins', 'predictors']:
        row = data[name].episode_id.eq(event)
        data[name].loc[row, 'bin_start'] = onset
        data[name].loc[row, 'forecast_at'] = onset + pd.Timedelta(1, unit='s')
    row = data['predictors'].episode_id.eq(event)
    data['predictors'].loc[row, 'onset_hour_utc'] = onset.hour
    data['predictors'].loc[row, 'onset_weekday_utc'] = onset.weekday()
    row = data['labels'].episode_id.eq(event)
    data['labels'].loc[row, 'forecast_at'] = onset + pd.Timedelta(1, unit='s')
    data['labels'].loc[row, 'label_ready_at'] = onset + pd.Timedelta(301, unit='s')
    refresh(data)


class PersistenceForecastTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def fit(self):
        return fit_forecasts(**self.data)

    def evaluate(self, fitted=None):
        return evaluate_forecasts((fitted or self.fit())['forecasts'], self.data['labels'], start=START, end=END)

    def first_prediction(self):
        return int(self.data['origins'].loc[self.data['origins'].forecast_at.ge(START), 'episode_id'].iloc[0])

    def test_ordinary_fit_and_raw_units_reconcile_independent_lstsq(self):
        actual = self.fit()
        self.assertEqual(actual['statistics']['ols_fits'], 2)
        self.assertEqual(actual['statistics']['training_mean_calibrations'], 1)
        self.assertEqual(actual['statistics']['empirical_models_fitted'], 3)
        ids = self.data['membership'].query('role == "training"').episode_id
        x = self.data['predictors'].set_index('episode_id').loc[ids]
        y = self.data['labels'].set_index('episode_id').loc[ids].continuation_pips.to_numpy()
        for method in ['additive', 'interaction']:
            saved = actual['coefficients'].loc[actual['coefficients'].method.eq(method)]
            columns = []
            for term in saved.term:
                if term == 'intercept': columns.append(np.ones(len(x)))
                elif term.startswith('hour_'): columns.append(x.onset_hour_utc.eq(int(term[5:])).to_numpy(float))
                elif term.startswith('weekday_'): columns.append(x.onset_weekday_utc.eq(int(term[8:])).to_numpy(float))
                else: columns.append(x[term].to_numpy())
            reference = np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0]
            np.testing.assert_allclose(saved.coefficient, reference, rtol=1e-10, atol=1e-12)
        json.dumps(actual['statistics'], allow_nan=False)

    def test_future_targets_and_availability_cannot_change_forecasts(self):
        expected = self.fit()
        later = self.data['labels'].forecast_at.ge(START)
        self.data['labels'].loc[later, 'continuation_pips'] = np.nan
        self.data['labels'].loc[later, 'label_available'] = False
        pd.testing.assert_frame_equal(expected['forecasts'], self.fit()['forecasts'])

    def test_cutoff_equality_mature_but_one_nanosecond_late_rejected(self):
        event = int(self.data['membership'].query('role == "training"').episode_id.iloc[-1])
        change_time(self.data, event, START - pd.Timedelta(301, unit='s'))
        self.fit()
        self.data['folds'].loc[0, 'fit_cutoff'] -= pd.Timedelta(1, unit='ns')
        self.data['folds'].loc[0, 'prediction_start'] -= pd.Timedelta(1, unit='ns')
        with self.assertRaisesRegex(ValueError, 'not matured'):
            self.fit()

    def test_corrupt_ready_time_cannot_admit_future_label(self):
        event = int(self.data['membership'].query('role == "training"').episode_id.iloc[-1])
        self.data['membership'].loc[self.data['membership'].episode_id.eq(event), 'label_ready_at'] -= pd.Timedelta(1, unit='s')
        with self.assertRaisesRegex(ValueError, 'maturity mismatch'):
            self.fit()

    def test_unseen_hour_both_ols_use_exact_training_mean(self):
        event = self.first_prediction()
        change_time(self.data, event, START + pd.Timedelta(1, unit='h'))
        rows = self.fit()['forecasts'].query('episode_id == @event').set_index('method')
        for method in ['additive', 'interaction']:
            self.assertEqual(rows.loc[method, 'predicted_C'], rows.loc['training_mean', 'predicted_C'])
            self.assertEqual(rows.loc[method, 'forecast_status'], 'unseen_calendar_mean')
            self.assertTrue(rows.loc[method, 'unseen_hour'])
            self.assertFalse(rows.loc[method, 'unseen_weekday'])
        self.assertFalse(rows.loc['zero', 'unseen_hour'])

    def test_unseen_weekday_fallback_but_unseen_combination_is_estimable(self):
        prior = self.data['labels'].forecast_at.lt(START)
        sunday = self.data['labels'].forecast_at.dt.weekday.eq(6)
        mask = prior & sunday
        self.data['labels'].loc[mask, ['label_available', 'label_status', 'continuation_pips']] = [False, 'unavailable', np.nan]
        refresh(self.data)
        forecasts = self.fit()['forecasts']
        future_sunday = forecasts.forecast_at.dt.weekday.eq(6) & forecasts.method.eq('interaction')
        self.assertTrue(forecasts.loc[future_sunday, 'unseen_weekday'].all())
        self.data = fixture()
        mask = self.data['labels'].forecast_at.lt(START) & self.data['labels'].forecast_at.dt.weekday.eq(0) & self.data['labels'].forecast_at.dt.hour.eq(2)
        self.data['labels'].loc[mask, ['label_available', 'label_status', 'continuation_pips']] = [False, 'unavailable', np.nan]
        refresh(self.data)
        forecasts = self.fit()['forecasts']
        test = forecasts.forecast_at.dt.weekday.eq(0) & forecasts.forecast_at.dt.hour.eq(2) & forecasts.method.eq('interaction')
        self.assertTrue(forecasts.loc[test, 'forecast_status'].eq('estimated').all())

    def test_missing_outcome_retains_issued_forecasts_and_missing_loss(self):
        event = self.first_prediction()
        self.data['labels'] = self.data['labels'].loc[~self.data['labels'].episode_id.eq(event)]
        fitted = self.fit(); evaluation = self.evaluate(fitted)
        rows = evaluation['scored_events'].query('episode_id == @event')
        self.assertTrue(rows.predicted_C.notna().all())
        self.assertTrue(rows.squared_error.isna().all())
        self.assertTrue(rows.label_status.eq('missing_record').all())
        self.assertEqual(evaluation['statistics']['issued_unavailable'], 1)

    def test_ineligible_and_zero_initial_movement_preserved(self):
        event = self.first_prediction()
        self.data['origins'].loc[self.data['origins'].episode_id.eq(event), 'origin_eligible'] = False
        refresh(self.data)
        actual = self.fit()['forecasts']
        self.assertTrue(actual.query('episode_id == @event').predicted_C.isna().all())
        zero = actual.origin_eligible & actual.initial_pips.eq(0)
        np.testing.assert_array_equal(actual.loc[zero, 'predicted_C'], actual.loc[zero, 'predicted_I'])
        self.assertTrue((actual.initial_pips < 0).any())

    def test_feature_contamination_rejected(self):
        self.data['predictors']['future_C'] = 0.
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            self.fit()

    def test_rank_failure_cannot_drop_continuous_column(self):
        self.data['predictors']['signed_pressure'] = self.data['predictors'].pressure_magnitude
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            self.fit()

    def test_constant_variation_cannot_silently_drop_interaction(self):
        self.data['predictors']['variation_standardized'] = 1.
        self.data['predictors']['variation_x_magnitude'] = self.data['predictors'].pressure_magnitude
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            self.fit()

    def test_empty_training_fails_closed(self):
        self.data['membership'] = self.data['membership'].query('role != "training"')
        self.data['folds'].loc[0, 'training_origins'] = 0
        with self.assertRaisesRegex(ValueError, 'Training membership'):
            self.fit()

    def test_reference_level_absent_fails(self):
        mask = self.data['labels'].forecast_at.lt(START) & self.data['labels'].forecast_at.dt.hour.eq(0)
        self.data['labels'].loc[mask, ['label_available', 'label_status', 'continuation_pips']] = [False, 'unavailable', np.nan]
        refresh(self.data)
        with self.assertRaisesRegex(ValueError, 'reference hour'):
            self.fit()

    def test_calendar_timestamp_disagreement_and_saturday_fail(self):
        self.data['predictors'].loc[0, 'onset_hour_utc'] = 1
        with self.assertRaisesRegex(ValueError, 'calendar disagrees'):
            self.fit()
        self.data = fixture()
        change_time(self.data, self.first_prediction(), START + pd.Timedelta(5, unit='D'))
        with self.assertRaisesRegex(ValueError, 'Saturday'):
            self.fit()

    def test_zero_target_has_zero_losses_and_undefined_ratios(self):
        self.data['labels']['continuation_pips'] = 0.
        actual = self.evaluate()
        self.assertTrue(actual['metrics'].mse.eq(0).all())
        self.assertTrue(actual['comparisons'].relative_mse_reduction.isna().all())
        self.assertTrue(actual['comparisons'].relative_mse_status.eq('zero_benchmark_mse').all())
        self.assertTrue(actual['serial_dependence'].normalized_autocorrelation.isna().all())
        self.assertTrue(all(x['zero_energy'] for x in actual['statistics']['comparison_diagnostics'].values()))
        json.dumps(actual['statistics'], allow_nan=False)

    def test_constant_nonzero_target_is_valid(self):
        self.data['labels']['continuation_pips'] = 3.
        actual = self.fit()['forecasts']
        np.testing.assert_allclose(actual.loc[actual.method.ne('zero'), 'predicted_C'], 3., atol=1e-12, rtol=0)

    def test_event_weighted_losses_calendar_and_stable_differences(self):
        first_day = self.data['labels'].forecast_at.ge(START) & self.data['labels'].forecast_at.lt(START + pd.Timedelta(1, unit='D'))
        remove = self.data['labels'].loc[first_day, 'episode_id'].iloc[:9]
        self.data['labels'] = self.data['labels'].loc[~self.data['labels'].episode_id.isin(remove)]
        actual = self.evaluate()
        scored = actual['scored_events']
        for row in actual['metrics'].itertuples(index=False):
            values = scored.loc[scored.method.eq(row.method) & scored.scored, 'squared_error']
            self.assertAlmostEqual(row.mse, values.mean(), places=13)
            self.assertAlmostEqual(actual['daily_losses']['sse_' + row.method].sum(), row.sse, places=10)
        daily = actual['daily_losses']
        zero_metric = actual['metrics'].loc[actual['metrics'].method.eq('zero'), 'mse'].iloc[0]
        self.assertNotAlmostEqual(zero_metric, (daily.sse_zero / daily.n_scored.replace(0, np.nan)).mean(), places=8)
        self.assertEqual(len(actual['daily_losses']), 14)
        self.assertTrue(actual['daily_losses'].loc[actual['daily_losses'].date.dt.weekday.eq(5), 'n_scored'].eq(0).all())
        self.assertEqual(len(actual['serial_dependence']), 123)
        losses = scored.pivot(index='episode_id', columns='method', values='squared_error')
        for row in actual['comparisons'].itertuples(index=False):
            self.assertAlmostEqual(row.mean_loss_difference, (losses.interaction - losses[row.benchmark]).mean(), places=12)
            psi = actual['daily_losses']['psi_' + row.contrast]
            self.assertAlmostEqual(psi.sum(), 0., places=10)

    def test_corrupt_unavailable_finite_target_rejected_only_at_evaluation(self):
        event = self.first_prediction()
        self.data['labels'].loc[self.data['labels'].episode_id.eq(event), 'label_available'] = False
        fitted = self.fit()
        with self.assertRaisesRegex(ValueError, 'availability must equal'):
            self.evaluate(fitted)

    def test_corrupt_evaluation_prediction_identity_rejected(self):
        fitted = self.fit()
        fitted['forecasts'].loc[0, 'predicted_I'] += 1
        with self.assertRaisesRegex(ValueError, 'J/C/I identity'):
            self.evaluate(fitted)

    def test_serialized_frames_round_trip(self):
        import tempfile
        from pathlib import Path
        actual = self.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forecasts.parquet'
            actual['forecasts'].to_parquet(path, index=False)
            restored = pd.read_parquet(path)
            pd.testing.assert_frame_equal(restored, actual['forecasts'])
            reference = self.evaluate(actual)
            recovered = evaluate_forecasts(restored, self.data['labels'], start=START, end=END)
            pd.testing.assert_frame_equal(reference['metrics'], recovered['metrics'])


if __name__ == '__main__':
    unittest.main()

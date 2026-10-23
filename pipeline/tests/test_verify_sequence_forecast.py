"""Adversarial independent validation of the fixed chronological comparison."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.verify_sequence_forecast import verify_forecasts, verify_evaluation, verify_source_projection


START = pd.Timestamp('2024-05-06', tz='UTC')
END = pd.Timestamp('2024-06-03', tz='UTC')


def fixture():
    rng = np.random.default_rng(918401)
    times = []
    for date in pd.date_range('2024-04-22', '2024-05-24', tz='UTC'):
        if date.dayofweek >= 5:
            continue
        for hour in [0, 1, 2]:
            if date < START and date.dayofweek == 0 and hour == 1:
                continue  # an unseen combination, not an unseen marginal level.
            for minute in range(2 if date < START else 1):
                times.append(date + pd.Timedelta(hour, unit='h') + pd.Timedelta(minute, unit='m'))
    times += [START + pd.Timedelta(13, unit='h'),
              pd.Timestamp('2024-05-12T01:00:00Z'),
              START - pd.Timedelta(1, unit='m')]
    ts = pd.DatetimeIndex(sorted(times)) + pd.Timedelta(1, unit='s')
    n = len(ts)
    pressure = rng.normal(size=n)
    variation = rng.normal(size=n)
    spread = .1+rng.uniform(size=n)
    size = 4+rng.normal(size=n)
    j = rng.choice([-1., 0., 1.], n)
    eligible = np.ones(n, dtype=bool)
    prediction_ix = np.flatnonzero(ts >= START)
    eligible[prediction_ix[5]] = False
    origins = pd.DataFrame(dict(episode_id=np.arange(n), bin_start=ts-pd.Timedelta(1, unit='s'),
        forecast_at=ts, origin_eligible=eligible, initial_pips=j))
    direction = rng.choice([-1., 1.], n)
    origins['direction'] = direction
    q5 = rng.normal(size=(n, 60))
    q5[rng.uniform(size=q5.shape) < .25] = 0.
    signs = np.sign(q5)
    sign_sum = signs.sum(axis=1)
    nonzero = (signs != 0).sum(axis=1)
    raw = (signs[:, :-1]*signs[:, 1:]).sum(axis=1)/59
    reference = (sign_sum**2-nonzero)/(60*59)
    pressure = q5.sum(axis=1)
    predictors = origins[['episode_id', 'bin_start', 'forecast_at']].copy()
    predictors['direction'] = direction
    predictors['initial_pips'] = j
    predictors['aligned_pressure'] = direction*pressure
    predictors['pressure_magnitude'] = abs(pressure)
    predictors['gross_pressure_scaled'] = abs(q5).sum(axis=1)
    predictors['variation_standardized'] = variation
    predictors['pre_spread_pips'] = spread
    predictors['log_pre_displayed_size'] = size
    predictors['log1p_prior_quote_updates'] = np.log1p(rng.integers(61, 1000, n))
    predictors['aligned_sign_balance'] = direction*sign_sum/60
    predictors['nonzero_fraction'] = nonzero/60
    predictors['composition_reference'] = reference
    predictors['aligned_recent_5s_pressure'] = direction*q5[:, -1]
    predictors['aligned_recent_30s_pressure'] = direction*q5[:, -6:].sum(axis=1)
    predictors['recent_5s_gross_scaled'] = abs(q5[:, -1])
    predictors['recent_30s_gross_scaled'] = abs(q5[:, -6:]).sum(axis=1)
    for window in (5, 30, 300):
        predictors[f'aligned_prior_{window}s_price_pips'] = rng.normal(size=n)
    predictors['sequence_z'] = raw-reference
    predictors['onset_hour_utc'] = predictors.bin_start.dt.hour
    predictors['onset_weekday_utc'] = predictors.bin_start.dt.dayofweek
    available = eligible.copy()
    available[prediction_ix[-1]] = False
    y = .2*pressure-.1*abs(pressure)+.2*variation+.5*variation*abs(pressure)+rng.normal(0, .2, n)
    y[~available] = np.nan
    labels = origins[['episode_id', 'forecast_at', 'origin_eligible']].copy()
    labels['horizon_sec'] = 300
    labels['target_at'] = ts+pd.Timedelta(300, unit='s')
    labels['label_ready_at'] = labels.target_at
    labels['label_available'] = available
    labels['evaluation_eligible'] = eligible & available
    labels['initial_pips'] = j
    labels['continuation_pips'] = y
    labels['total_pips'] = j+y
    labels['endpoint_status'] = np.where(available, 'available', 'outside_onset_support')
    labels['label_status'] = labels.endpoint_status
    folds = pd.DataFrame(dict(fold_id=[1, 2],
        fit_cutoff=[START, START+pd.Timedelta(14, unit='D')],
        prediction_start=[START, START+pd.Timedelta(14, unit='D')],
        prediction_end_exclusive=[START+pd.Timedelta(14, unit='D'), END]))
    members = []
    for fold in folds.itertuples(index=False):
        training = eligible & (ts < fold.fit_cutoff) & available & labels.label_ready_at.le(fold.fit_cutoff)
        prediction = (ts >= fold.fit_cutoff) & (ts < fold.prediction_end_exclusive)
        for role, mask in [('training', training), ('prediction', prediction)]:
            for event in np.flatnonzero(mask):
                members.append(dict(fold_id=fold.fold_id, role=role, episode_id=int(event),
                    forecast_at=ts[event], origin_eligible=bool(eligible[event]),
                    horizon_sec=300, label_ready_at=labels.label_ready_at.iloc[event],
                    label_available=bool(available[event]),
                    label_mature_at_refit=bool(labels.label_ready_at.iloc[event] <= fold.fit_cutoff),
                    evaluation_eligible=bool(role == 'prediction' and eligible[event] and available[event])))
    membership = pd.DataFrame(members)
    folds['training_origins'] = [int(((membership.fold_id == i) & (membership.role == 'training')).sum()) for i in folds.fold_id]
    folds['prediction_origins'] = [int(((membership.fold_id == i) & (membership.role == 'prediction')).sum()) for i in folds.fold_id]
    folds['prediction_origin_eligible'] = [int(((membership.fold_id == i) & (membership.role == 'prediction') & membership.origin_eligible).sum()) for i in folds.fold_id]
    labels = labels[['episode_id', 'horizon_sec', 'forecast_at', 'label_ready_at',
                     'label_available', 'label_status', 'continuation_pips']]
    membership = membership[['fold_id', 'role', 'episode_id', 'forecast_at',
                             'origin_eligible', 'label_ready_at', 'label_available']]
    return dict(origins=origins, predictors=predictors, labels=labels,
                folds=folds, membership=membership)


class VerifySequenceForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # producer calls create candidate artifacts only; the verifier does not
        # import it or use its matrix builder, solver or loss aggregation.
        from src.sequence_forecast import fit_forecasts, evaluate_forecasts
        cls.parent = fixture()
        cls.fitted = fit_forecasts(**cls.parent)
        cls.evaluated = evaluate_forecasts(cls.fitted['forecasts'], cls.parent['labels'], start=START, end=END)

    def verify_fit(self, parent=None, fitted=None):
        return verify_forecasts(self.parent if parent is None else parent,
                                self.fitted if fitted is None else fitted)

    def verify_loss(self, evaluated=None, forecasts=None, labels=None):
        return verify_evaluation(self.fitted['forecasts'] if forecasts is None else forecasts,
            self.parent['labels'] if labels is None else labels,
            self.evaluated if evaluated is None else evaluated, start=START, end=END)

    def test_independent_QR_and_all_loss_tables_reconstruct(self):
        self.assertEqual(self.verify_fit()['ols_fits_verified'], 4)
        report = self.verify_loss()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['calendar_days_verified'], 28)
        self.assertEqual(report['issued_unavailable_verified'], 1)

    def test_future_value_and_availability_mutations_leave_first_fold_unchanged(self):
        from src.sequence_forecast import fit_forecasts
        parent = deepcopy(self.parent)
        parent['folds'] = parent['folds'].iloc[:1].copy()
        parent['membership'] = parent['membership'].loc[parent['membership'].fold_id.eq(1)].copy()
        actual = fit_forecasts(**parent)
        future = parent['labels'].label_ready_at.gt(START)
        # includes an origin before START whose 300-second label is not mature.
        self.assertTrue((future & parent['labels'].forecast_at.lt(START)).any())
        changed = deepcopy(parent)
        changed['labels'].loc[future & changed['labels'].label_available, 'continuation_pips'] += 100000.
        value_fit = fit_forecasts(**changed)
        pd.testing.assert_frame_equal(actual['forecasts'], value_fit['forecasts'])
        pd.testing.assert_frame_equal(actual['coefficients'], value_fit['coefficients'])
        self.assertEqual(verify_forecasts(changed, actual)['status'], 'passed')
        changed['labels'].loc[future, 'continuation_pips'] = np.nan
        changed['labels'].loc[future, 'label_available'] = False
        missing_fit = fit_forecasts(**changed)
        pd.testing.assert_frame_equal(actual['forecasts'], missing_fit['forecasts'])
        self.assertEqual(verify_forecasts(changed, actual)['status'], 'passed')

    def test_omitted_matured_training_origin_is_not_hidden_by_matching_count(self):
        from src.sequence_forecast import fit_forecasts
        parent = deepcopy(self.parent)
        row = parent['membership'].index[(parent['membership'].fold_id == 1) &
                                         (parent['membership'].role == 'training')][0]
        parent['membership'] = parent['membership'].drop(row)
        parent['folds'].loc[parent['folds'].fold_id.eq(1), 'training_origins'] -= 1
        candidate = fit_forecasts(**parent)
        with self.assertRaises(AssertionError):
            self.verify_fit(parent, candidate)

    def test_leaked_output_and_predictor_columns_rejected(self):
        for field in ['continuation_pips', 'label_available', 'endpoint_age_sec']:
            candidate = deepcopy(self.fitted)
            candidate['forecasts'][field] = 0.
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.verify_fit(fitted=candidate)
        parent = deepcopy(self.parent)
        parent['predictors']['label_available'] = True
        with self.assertRaises(AssertionError):
            self.verify_fit(parent=parent)

    def test_coefficients_predictions_and_reported_fit_statistics_checked(self):
        for table, column in [('coefficients', 'coefficient'), ('coefficients', 'equilibration'),
                              ('fit_diagnostics', 'training_mean'), ('forecasts', 'predicted_C')]:
            candidate = deepcopy(self.fitted)
            candidate[table].loc[0, column] += .01
            with self.subTest(table=table, column=column), self.assertRaises(AssertionError):
                self.verify_fit(fitted=candidate)
        candidate = deepcopy(self.fitted)
        candidate['statistics']['eligible_origins'] -= 1
        with self.assertRaises(AssertionError):
            self.verify_fit(fitted=candidate)

    def test_missing_future_origin_forecast_cannot_be_removed(self):
        ids = self.parent['labels'].loc[self.parent['origins'].origin_eligible &
                                       ~self.parent['labels'].label_available, 'episode_id']
        candidate = deepcopy(self.fitted)
        candidate['forecasts'] = candidate['forecasts'].loc[~candidate['forecasts'].episode_id.isin(ids)]
        with self.assertRaises(AssertionError):
            self.verify_fit(fitted=candidate)

    def test_unseen_marginal_fallback_and_unseen_combination_are_distinct(self):
        forecast = self.fitted['forecasts']
        fallback = forecast.loc[forecast.forecast_status.eq('unseen_calendar_mean')]
        self.assertGreater(len(fallback), 0)
        means = forecast.loc[forecast.method.eq('training_mean')].set_index('episode_id').predicted_C
        np.testing.assert_array_equal(fallback.predicted_C, fallback.episode_id.map(means))
        ordinary_time = START+pd.Timedelta(1, unit='h')+pd.Timedelta(1, unit='s')
        combination = forecast.loc[forecast.forecast_at.eq(ordinary_time) & forecast.method.isin(['baseline', 'sequence'])]
        self.assertEqual(len(combination), 2)
        self.assertTrue(combination.forecast_status.eq('estimated').all())
        bad = deepcopy(self.fitted)
        ix = combination.index[0]
        bad['forecasts'].loc[ix, 'forecast_status'] = 'unseen_calendar_mean'
        with self.assertRaises(AssertionError):
            self.verify_fit(fitted=bad)

    def test_tampered_sequence_and_alert_information_rejected(self):
        for field in ['sequence_z', 'initial_pips', 'direction']:
            parent = deepcopy(self.parent)
            ix = parent['origins'].index[parent['origins'].origin_eligible][0]
            parent['predictors'].loc[ix, field] += .05
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.verify_fit(parent=parent)

    def test_frozen_source_projection_and_centering_identity_checked(self):
        source = self.parent['predictors'].copy()
        source['inherited_origin_eligible'] = self.parent['origins'].origin_eligible
        source['combined_eligible'] = source.inherited_origin_eligible
        source['adjacent_agreement'] = source.sequence_z+source.composition_reference
        # reconstruct adjacency then centered score may round differently at the last bit;
        # source uses the measured centered values, so freeze its exact identity here.
        parent = deepcopy(self.parent)
        source['sequence_z'] = source.adjacent_agreement-source.composition_reference
        parent['predictors']['sequence_z'] = source.sequence_z
        self.assertEqual(verify_source_projection(parent, source)['status'], 'passed')
        parent['predictors'].loc[0, 'sequence_z'] += .001
        with self.assertRaises(AssertionError):
            verify_source_projection(parent, source)
        parent = deepcopy(self.parent)
        parent['predictors']['sequence_z'] = source.sequence_z
        bad = source.copy()
        bad.loc[0, 'adjacent_agreement'] += .001
        with self.assertRaises(AssertionError):
            verify_source_projection(parent, bad)
        bad = source.copy()
        bad.loc[0, 'combined_eligible'] = not bad.loc[0, 'combined_eligible']
        with self.assertRaises(AssertionError):
            verify_source_projection(parent, bad)

    def test_rank_deficiency_cannot_be_silently_repaired(self):
        parent = deepcopy(self.parent)
        parent['predictors']['log_pre_displayed_size'] = parent['predictors'].pre_spread_pips
        with self.assertRaises(AssertionError):
            self.verify_fit(parent=parent)

    def test_wrong_label_maturity_and_source_calendar_fail(self):
        parent = deepcopy(self.parent)
        parent['labels'].loc[0, 'label_ready_at'] = parent['labels'].forecast_at.iloc[0]
        with self.assertRaises(AssertionError):
            self.verify_fit(parent=parent)
        parent = deepcopy(self.parent)
        parent['predictors'].loc[0, 'onset_hour_utc'] = 23
        with self.assertRaises(AssertionError):
            self.verify_fit(parent=parent)

    def test_event_weighted_loss_not_mean_of_daily_means(self):
        evaluated = deepcopy(self.evaluated)
        daily = evaluated['daily_losses']
        key = evaluated['metrics'].index[evaluated['metrics'].method.eq('sequence')][0]
        wrongly_weighted = (daily.loc[daily.n_scored.gt(0), 'sse_sequence'] /
                            daily.loc[daily.n_scored.gt(0), 'n_scored']).mean()
        self.assertNotAlmostEqual(wrongly_weighted, evaluated['metrics'].loc[key, 'mse'], places=8)
        evaluated['metrics'].loc[key, 'mse'] = wrongly_weighted
        with self.assertRaises(AssertionError):
            self.verify_loss(evaluated)

    def test_calendar_empty_days_and_count_weighted_centering_checked(self):
        evaluated = deepcopy(self.evaluated)
        empty = evaluated['daily_losses'].index[evaluated['daily_losses'].n_scored.eq(0)]
        self.assertGreater(len(empty), 0)
        evaluated['daily_losses'] = evaluated['daily_losses'].drop(empty[0])
        with self.assertRaises(AssertionError):
            self.verify_loss(evaluated)
        evaluated = deepcopy(self.evaluated)
        name = 'sequence_minus_baseline'
        daily = evaluated['daily_losses']
        daily['psi_'+name] = daily['sum_delta_'+name]-daily['sum_delta_'+name].mean()
        with self.assertRaises(AssertionError):
            self.verify_loss(evaluated)

    def test_loss_delta_and_serial_rows_are_verified(self):
        for table, column in [('scored_events', 'squared_error'),
                              ('comparisons', 'mean_loss_difference'), ('comparisons', 'rmse_difference'),
                              ('serial_dependence', 'cross_product'),
                              ('quarter_losses', 'sse_sequence')]:
            candidate = deepcopy(self.evaluated)
            ix = candidate[table].index[candidate[table][column].notna()][0]
            candidate[table].loc[ix, column] += .01
            with self.subTest(table=table), self.assertRaises(AssertionError):
                self.verify_loss(candidate)

    def test_zero_denominators_and_exact_constant_target(self):
        from src.sequence_forecast import fit_forecasts, evaluate_forecasts
        parent = deepcopy(self.parent)
        parent['labels'].loc[parent['labels'].label_available, 'continuation_pips'] = 0.
        fitted = fit_forecasts(**parent)
        self.assertEqual(self.verify_fit(parent, fitted)['status'], 'passed')
        evaluated = evaluate_forecasts(fitted['forecasts'], parent['labels'], start=START, end=END)
        self.assertTrue(evaluated['comparisons'].relative_mse_reduction.isna().all())
        self.assertTrue(evaluated['serial_dependence'].normalized_autocorrelation.isna().all())
        self.assertEqual(verify_evaluation(fitted['forecasts'], parent['labels'], evaluated, start=START, end=END)['status'], 'passed')
        evaluated['serial_dependence'].loc[0, 'normalized_autocorrelation'] = 0.
        with self.assertRaises(AssertionError):
            verify_evaluation(fitted['forecasts'], parent['labels'], evaluated, start=START, end=END)

    def test_missing_target_not_filled_with_zero_loss(self):
        evaluated = deepcopy(self.evaluated)
        unscored = ~evaluated['scored_events'].scored
        evaluated['scored_events'].loc[unscored, 'squared_error'] = 0.
        with self.assertRaises(AssertionError):
            self.verify_loss(evaluated)

    def test_statistics_concentration_counts_and_no_inference_flags_checked(self):
        for kind in ['count', 'concentration', 'claim']:
            evaluated = deepcopy(self.evaluated)
            if kind == 'count':
                evaluated['statistics']['issued_unavailable'] += 1
            elif kind == 'claim':
                evaluated['statistics']['inference_computed'] = True
            else:
                evaluated['statistics']['comparison_diagnostics']['sequence_minus_baseline']['maximum_daily_energy_share'] += .01
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                self.verify_loss(evaluated)

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            fitted, evaluated = deepcopy(self.fitted), deepcopy(self.evaluated)
            for label, bundle in [('fit', fitted), ('loss', evaluated)]:
                for name, value in list(bundle.items()):
                    if isinstance(value, pd.DataFrame):
                        path = Path(directory)/(label+'_'+name+'.parquet')
                        value.to_parquet(path, index=False)
                        bundle[name] = pd.read_parquet(path)
            self.assertEqual(self.verify_fit(fitted=fitted)['status'], 'passed')
            self.assertEqual(self.verify_loss(evaluated, fitted['forecasts'])['status'], 'passed')


if __name__ == '__main__':
    unittest.main()

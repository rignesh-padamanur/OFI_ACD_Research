import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.timing_forecast import (seasonal_bins, predictor_frame, select_training,
    quartile_groups, daily_hac, estimates, BASE, EXTRA)


class TimingForecastTests(unittest.TestCase):
    def raw(self):
        s = pd.Timestamp('2024-07-01', tz='UTC')
        return pd.DataFrame({'origin_at': [s-pd.Timedelta(seconds=2.8), s+pd.Timedelta(seconds=.2)],
            'end_at': [s+pd.Timedelta(seconds=.2), s+pd.Timedelta(seconds=.6)],
            'event_observed': [True, True], 'scale_sec': [2., 4.],
            'seasonal_cutoff': [s-pd.Timedelta(days=1)]*2,
            'exposure_ns': [3_000_000_000, 400_000_000]})

    def test_leading_gap_is_included_and_mean_precedes_log(self):
        bins = seasonal_bins(self.raw())
        self.assertEqual(bins.iloc[0].n, 2)
        self.assertAlmostEqual(bins.iloc[0].total, 3.4)
        self.assertAlmostEqual(bins.iloc[0].seasonal_mean, .8)
        self.assertNotAlmostEqual(-np.log(.8), -(np.log(1.5)+np.log(.1))/2)

    def test_completion_at_right_boundary_belongs_to_next_bin(self):
        raw = self.raw()
        raw.loc[1, 'end_at'] = raw.loc[1, 'end_at'].floor('s')+pd.Timedelta(seconds=1)
        raw.loc[1, 'exposure_ns'] = 800_000_000
        self.assertEqual(len(seasonal_bins(raw)), 2)

    def test_future_seasonal_vintage_is_rejected(self):
        raw = self.raw()
        raw.loc[0, 'seasonal_cutoff'] = raw.loc[0, 'origin_at']+pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):
            seasonal_bins(raw)

    def test_missing_scale_does_not_become_a_complete_mean(self):
        raw = self.raw()
        raw.loc[1, 'scale_sec'] = np.nan
        bins = seasonal_bins(raw)
        self.assertEqual(bins.iloc[0].n_scale, 1)
        self.assertEqual(bins.iloc[0].n, 2)

    def test_unfinished_wait_is_rejected(self):
        raw = self.raw()
        raw.loc[1, 'event_observed'] = False
        with self.assertRaises(ValueError):
            seasonal_bins(raw)

    def frame_fixture(self):
        s = pd.Timestamp('2024-07-01', tz='UTC')
        scored = pd.DataFrame({'bin_start': [s], 'available_at': [s+pd.Timedelta(seconds=1)],
            'support_id': [0], 'recorded_update_count': [2], 'completed_residual_count': [2],
            'pre_quote_age_sec': [2.8], 'end_quote_age_sec': [.4], EXTRA: [.7]})
        ledger = pd.DataFrame({name: [1.] for name in BASE[:13]})
        ledger['bin_start'], ledger['support_id'], ledger['recorded_update_count'] = [s], [0], [2]
        ledger['quote_age'], ledger[EXTRA] = [.4], [.7]
        return scored, ledger

    def test_future_outcomes_and_detector_flags_cannot_enter_predictors(self):
        scored, ledger = self.frame_fixture()
        ledger['y300'], ledger['candidate_full'] = [1.], [True]
        first = predictor_frame(scored, ledger, seasonal_bins(self.raw()))
        ledger['y300'], ledger['candidate_full'] = [999999.], [False]
        second = predictor_frame(scored, ledger, seasonal_bins(self.raw()))
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn('y300', first.columns)

    def test_partial_seasonal_mean_cannot_enter_common_population(self):
        scored, ledger = self.frame_fixture()
        raw = self.raw()
        raw.loc[1, 'scale_sec'] = np.nan
        with self.assertRaises(ValueError):
            predictor_frame(scored, ledger, seasonal_bins(raw))

    def test_label_maturity_boundary_and_future_mutation(self):
        cutoff = pd.Timestamp('2024-07-01', tz='UTC')
        f = pd.DataFrame({'bin_start': [cutoff-pd.Timedelta(seconds=301), cutoff-pd.Timedelta(seconds=300)],
            'forecast_ready': [True]*2, 'outcome_ready': [True]*2, 'y300': [1., 2.]})
        with patch('src.timing_forecast.hash_training_sample', return_value=np.ones(2, bool)):
            np.testing.assert_array_equal(select_training(f, cutoff), [True, False])
            f.loc[1, 'y300'] = 1000000.
            np.testing.assert_array_equal(select_training(f, cutoff), [True, False])

    def test_old_and_unavailable_training_are_excluded(self):
        cutoff = pd.Timestamp('2024-07-01', tz='UTC')
        f = pd.DataFrame({'bin_start': [cutoff-pd.Timedelta(days=57), cutoff-pd.Timedelta(days=1)],
            'forecast_ready': [True, False], 'outcome_ready': [True, True], 'y300': [1., 1.]})
        with patch('src.timing_forecast.hash_training_sample', return_value=np.ones(2, bool)):
            self.assertFalse(select_training(f, cutoff).any())

    def test_quartile_ties_are_not_forced_into_equal_cells(self):
        np.testing.assert_array_equal(quartile_groups([.1, .2, .3], [.2, .2, .2]), [1, 4, 4])

    def test_hac_respects_calendar_zeros(self):
        x = np.array([1., 0., -1.])
        self.assertAlmostEqual(daily_hac(x, 1), np.sqrt(2))
        self.assertNotAlmostEqual(daily_hac(x, 1), daily_hac(x[[0, 2]], 1))

    def test_relative_interval_uses_its_own_influence(self):
        daily = pd.DataFrame({'n': [2, 3, 0], 'sum_lb': [4., 9., 0.],
            'sum_lt': [2., 6., 0.], 'sum_d': [2., 3., 0.], 'sum_carry': [6., 10., 0.]})
        r = estimates(daily)
        self.assertAlmostEqual(r['delta_pips_squared'], 1)
        self.assertAlmostEqual(r['relative_mse_reduction'], 5/13)
        self.assertAlmostEqual(r['nominal_intervals'][0]['nominal_se'], 0)
        self.assertGreater(r['nominal_intervals'][0]['relative_nominal_se'], 0)


if __name__ == '__main__':
    unittest.main()

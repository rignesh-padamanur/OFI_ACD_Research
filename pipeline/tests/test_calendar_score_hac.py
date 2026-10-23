"""Failure-oriented arithmetic checks for calendar-score inference preparation."""
import unittest
import numpy as np
import pandas as pd

from src.calendar_score_hac import (
    PILOT_COLUMNS, pilot_weights, ols_scores, aggregate_daily_scores,
    nw94_bandwidth, bartlett_meat, calendar_hac, coordinate_bandwidth,
)


def fixture():
    rng = np.random.default_rng(707)
    n = 1200
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 5))])
    y = x @ np.array([.2, .5, -.4, .1, .2, -.3]) + rng.normal(size=n)
    times = pd.Timestamp('2024-05-06', tz='UTC') + pd.to_timedelta(np.arange(n)//3, unit='D')
    start = pd.Timestamp('2024-05-06', tz='UTC')
    end = start + pd.Timedelta(days=410)
    weights = pilot_weights(['intercept', *PILOT_COLUMNS])
    return x, y, times, start, end, weights


class CalendarScoreHACChecks(unittest.TestCase):
    def test_pilot_has_only_five_original_states_even_with_curvature_controls(self):
        columns = ['intercept', 'direction', *PILOT_COLUMNS, 'pre_spread_pips',
                   'log_pre_displayed_size', 'hour_01', 'magnitude_squared', 'variation_squared']
        expected = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(pilot_weights(columns), expected)
        np.testing.assert_array_equal(pilot_weights(columns[::-1]), expected[::-1])

    def test_pilot_rejects_missing_or_duplicate_scientific_names(self):
        for names in [list(PILOT_COLUMNS[:-1]), [*PILOT_COLUMNS, PILOT_COLUMNS[0]],
                      [*PILOT_COLUMNS, 1]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                pilot_weights(names)

    def test_ols_satisfies_event_normal_equations_and_unscaled_bread(self):
        x, y, *_ = fixture()
        result = ols_scores(x, y)
        np.testing.assert_allclose(x.T @ result['residuals'], 0, atol=2e-11)
        np.testing.assert_allclose(result['bread'] @ (x.T @ x), np.eye(x.shape[1]), atol=1e-12)
        np.testing.assert_allclose(result['beta'], np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-13)
        np.testing.assert_array_equal(result['event_scores'], x*result['residuals'][:, None])

    def test_exact_fit_and_zero_outcome_are_not_reported_as_certain_estimates(self):
        x, *_ = fixture()
        for y in [x @ np.arange(x.shape[1]), np.zeros(len(x))]:
            with self.assertRaisesRegex(ValueError, 'exact fit'):
                ols_scores(x, y)

    def test_ols_rejects_rank_deficiency_bad_values_and_insufficient_rows(self):
        x, y, *_ = fixture()
        for bad_x, bad_y in [(np.column_stack([x, x[:, 1]]), y),
                              (x[:3], y[:3]), (x, y[:, None]),
                              (np.full_like(x, np.nan), y), (x.astype(bool), y)]:
            with self.subTest(shape=bad_x.shape), self.assertRaises(ValueError):
                ols_scores(bad_x, bad_y)

    def test_calendar_retains_empty_dates_and_uses_utc_onset_date(self):
        start = pd.Timestamp('2024-05-06', tz='UTC')
        times = pd.DatetimeIndex([start+pd.Timedelta(seconds=86399), start+pd.Timedelta(days=2)])
        out = aggregate_daily_scores([[1., 2.], [3., 4.]], times, start, start+pd.Timedelta(days=4))
        np.testing.assert_array_equal(out['daily_scores'], [[1, 2], [0, 0], [3, 4], [0, 0]])
        np.testing.assert_array_equal(out['event_counts'], [1, 0, 1, 0])
        self.assertEqual(len(out['dates']), 4)

    def test_daily_sums_are_invariant_to_event_order(self):
        x, y, times, start, end, _ = fixture()
        s = ols_scores(x, y)['event_scores']
        forward = aggregate_daily_scores(s, times, start, end)
        backward = aggregate_daily_scores(s[::-1], times[::-1], start, end)
        np.testing.assert_allclose(forward['daily_scores'], backward['daily_scores'], atol=1e-14)
        np.testing.assert_array_equal(forward['event_counts'], backward['event_counts'])

    def test_calendar_rejects_naive_outside_and_nonmidnight_inputs(self):
        start = pd.Timestamp('2024-05-06', tz='UTC'); end = start+pd.Timedelta(days=3)
        for times, a, b in [([start.tz_localize(None)], start, end), ([end], start, end),
                            ([start], start+pd.Timedelta(seconds=1), end),
                            ([start], start.tz_localize(None), end), ([pd.NaT], start, end)]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                aggregate_daily_scores([[1., 2.]], times, a, b)

    def test_no_events_preserve_the_calendar_but_cannot_supply_inference(self):
        start = pd.Timestamp('2024-05-06', tz='UTC')
        daily = aggregate_daily_scores(np.empty((0, 2)), pd.DatetimeIndex([], tz='UTC'),
                                       start, start+pd.Timedelta(days=10))
        self.assertEqual(daily['daily_scores'].shape, (10, 2))
        self.assertEqual(daily['event_counts'].sum(), 0)
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            calendar_hac(daily['daily_scores'], np.eye(2), [1., 1.])

    def test_bartlett_equals_pair_weight_matrix_with_correct_l_plus_one(self):
        g = np.array([[1., 2.], [-2., .4], [3., -1.], [0., .7]])
        for lag in [0, 1, 3]:
            distance = np.abs(np.arange(len(g))[:, None]-np.arange(len(g))[None, :])
            weights = np.maximum(0., 1-distance/(lag+1))
            np.testing.assert_allclose(bartlett_meat(g, lag), g.T @ weights @ g, atol=1e-14)

    def test_lag_zero_is_day_cluster_not_independent_event_hc0(self):
        scores = np.array([[1., 2.], [3., 4.]])
        daily = scores.sum(axis=0)[None, :]
        np.testing.assert_array_equal(bartlett_meat(daily, 0), daily.T @ daily)
        self.assertFalse(np.allclose(bartlett_meat(daily, 0), scores.T @ scores))

    def test_dropping_empty_dates_changes_real_time_covariance(self):
        full = np.array([[1.], [0.], [2.], [0.]])
        compressed = full[[0, 2]]
        self.assertNotEqual(bartlett_meat(full, 1)[0, 0], bartlett_meat(compressed, 1)[0, 0])

    def test_bandwidth_uses_pilot_moments_and_final_floor_separately(self):
        t = 100
        h = np.cos(2*np.pi*np.arange(t)/10)
        out = nw94_bandwidth(h[:, None], [1.])
        gamma = [sum(h[j:]*h[:t-j])/t for j in range(5)]
        s0 = gamma[0]+2*sum(gamma[1:]); s1 = 2*sum(j*gamma[j] for j in range(1, 5))
        expected = 1.1447*((s1/s0)**2*t)**(1/3)
        self.assertEqual(out['pilot_lag'], 4)
        self.assertEqual(out['lag'], int(np.floor(expected)))
        self.assertNotEqual(out['pilot_lag'], out['lag'])
        self.assertAlmostEqual(out['bandwidth'], expected, places=12)
        self.assertFalse(out['daily_score_centering']); self.assertFalse(out['prewhitening'])

    def test_scalar_pilot_common_rescaling_preserves_bandwidth_and_avoids_overflow(self):
        h = np.cos(2*np.pi*np.arange(100)/10)[:, None]
        original = nw94_bandwidth(h, [1.])
        for scale in [1e200, 1e-200, -7.]:
            out = nw94_bandwidth(scale*h, [1.])
            self.assertEqual(out['lag'], original['lag'])
            self.assertAlmostEqual(out['bandwidth'], original['bandwidth'], places=11)

    def test_zero_projection_and_zero_spectral_denominator_fail_without_fallback(self):
        h = np.cos(np.arange(100))
        for g, w in [(np.column_stack([h, -h]), [1., 1.]),
                     (np.column_stack([h, -h*(1+1e-15)]), [1., 1.]),
                     (np.r_[1., -1., np.zeros(98)][:, None], [1.])]:
            with self.assertRaises(ValueError):
                nw94_bandwidth(g, w)

    def test_negative_pilot_s0_is_explicit_and_uses_published_squared_ratio(self):
        h = np.cos(2*np.pi*np.arange(100)/5)[:, None]
        out = nw94_bandwidth(h, [1.])
        self.assertTrue(out['negative_pilot_s0'])
        self.assertGreater(out['lag'], 0)

    def test_unobservable_lags_are_not_capped_and_invalid_weights_fail(self):
        h = np.cos(2*np.pi*np.arange(100)/9)[:, None]
        with self.assertRaisesRegex(ValueError, 'no silent cap'):
            nw94_bandwidth(h, [1.])
        h = np.cos(2*np.pi*np.arange(100)/3)[:, None]
        with self.assertRaisesRegex(ValueError, 'Longer lag'):
            calendar_hac(h, [[1.]], [1.])
        for lag in [-1, 100, 1., True]:
            with self.assertRaises(ValueError):
                bartlett_meat(h, lag)
        for weights in [[0.], [np.nan], [1., 1.]]:
            with self.assertRaises(ValueError):
                nw94_bandwidth(h, weights)

    def test_hac_keeps_full_score_matrix_and_declared_longer_lag(self):
        x, y, times, start, end, weights = fixture()
        fit = ols_scores(x, y)
        daily = aggregate_daily_scores(fit['event_scores'], times, start, end)['daily_scores']
        out = calendar_hac(daily, fit['bread'], weights)
        self.assertEqual(out['longer']['lag'], max(1, 2*out['automatic']['lag']))
        self.assertEqual(out['automatic']['covariance'].shape, (6, 6))
        self.assertFalse(out['inference_approved'])
        self.assertGreater(np.linalg.eigvalsh(out['automatic']['covariance']).min(), 0)
        changed = daily.copy(); changed[:, 0] *= 100
        self.assertAlmostEqual(coordinate_bandwidth(changed, weights)['bandwidth'],
                               out['bandwidth']['bandwidth'])
        self.assertFalse(np.allclose(bartlett_meat(changed, out['automatic']['lag']),
                                    out['automatic']['meat']))

    def test_duplicate_events_do_not_create_false_precision(self):
        x, y, times, start, end, weights = fixture()
        original = ols_scores(x, y)
        repeated = ols_scores(np.repeat(x, 2, axis=0), np.repeat(y, 2))
        daily = aggregate_daily_scores(original['event_scores'], times, start, end)['daily_scores']
        repeat_daily = aggregate_daily_scores(repeated['event_scores'], times.repeat(2), start, end)['daily_scores']
        np.testing.assert_allclose(original['beta'], repeated['beta'], atol=1e-13)
        np.testing.assert_allclose(repeated['bread'], original['bread']/2, atol=1e-14)
        np.testing.assert_allclose(repeat_daily, 2*daily, atol=1e-12)
        before = calendar_hac(daily, original['bread'], weights)
        after = calendar_hac(repeat_daily, repeated['bread'], weights)
        for key in ['automatic', 'longer']:
            np.testing.assert_allclose(before[key]['covariance'], after[key]['covariance'], rtol=1e-10, atol=1e-13)

    def test_outcome_units_scale_covariance_quadratically(self):
        x, y, times, start, end, weights = fixture()
        fit = ols_scores(x, y)
        daily = aggregate_daily_scores(fit['event_scores'], times, start, end)['daily_scores']
        base = calendar_hac(daily, fit['bread'], weights)
        for scale in [1e90, 1e-90, -3.]:
            transformed = ols_scores(x, y*scale)
            transformed_daily = aggregate_daily_scores(transformed['event_scores'], times, start, end)['daily_scores']
            actual = calendar_hac(transformed_daily, transformed['bread'], weights)
            np.testing.assert_allclose(actual['automatic']['covariance']/scale**2,
                                       base['automatic']['covariance'], rtol=1e-10, atol=1e-13)

    def test_unrepresentable_covariances_fail_instead_of_reporting_zero_precision(self):
        x, y, times, start, end, weights = fixture()
        for scale in [1e200, 1e-200]:
            fit = ols_scores(x, y*scale)
            daily = aggregate_daily_scores(fit['event_scores'], times, start, end)['daily_scores']
            with self.assertRaises(ValueError):
                calendar_hac(daily, fit['bread'], weights)

    def test_coordinate_pilots_do_not_cancel_opposing_branch_pressure_scores(self):
        h = np.cos(2*np.pi*np.arange(100)/10)
        variation_score = np.zeros(100); variation_score[[20, 70]] = [1., -1.]
        g = np.column_stack([variation_score, h, -h, .3*h, -.3*h])
        selected = coordinate_bandwidth(g, np.ones(5))
        self.assertEqual(len(selected['component_pilots']), 5)
        self.assertEqual(selected['lag'], nw94_bandwidth(h[:, None], [1.])['lag'])
        summed = nw94_bandwidth(g, np.ones(5))
        self.assertEqual(summed['lag'], 0)
        self.assertGreater(selected['lag'], summed['lag'])

    def test_coordinate_rule_is_sign_and_units_invariant_with_full_covariance_transform(self):
        x, y, times, start, end, weights = fixture()
        fit = ols_scores(x, y)
        g = aggregate_daily_scores(fit['event_scores'], times, start, end)['daily_scores']
        factors = np.array([1., -1e6, 1e-6, -3., 7., -.2])
        before = calendar_hac(g, fit['bread'], weights)
        # if X*=X D, scores*=scores D and bread*=D^-1 bread D^-1.
        transformed_bread = fit['bread']/factors[:, None]/factors[None, :]
        after = calendar_hac(g*factors, transformed_bread, weights)
        self.assertEqual(before['automatic']['lag'], after['automatic']['lag'])
        self.assertAlmostEqual(before['bandwidth']['bandwidth'], after['bandwidth']['bandwidth'], places=11)
        for key in ['automatic', 'longer']:
            np.testing.assert_allclose(after[key]['covariance']*factors[:, None]*factors[None, :],
                                       before[key]['covariance'], rtol=1e-9, atol=1e-13)

    def test_coordinate_selector_rejects_invalid_or_degenerate_selected_columns(self):
        h = np.cos(2*np.pi*np.arange(100)/10)
        g = np.column_stack([h, np.zeros_like(h)])
        for weights in [[1., 1.], [.5, 0.], [0., 0.], [-1., 0.]]:
            with self.assertRaises(ValueError):
                coordinate_bandwidth(g, weights)
        self.assertEqual(coordinate_bandwidth(g, [1., 0.])['selected_columns'], [0])


if __name__ == '__main__':
    unittest.main()

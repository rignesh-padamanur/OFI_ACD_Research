"""Independent scalar-oracle tests for the postfit panel quote-state verifier."""
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import io
import json
import numpy as np
import pandas as pd

from src.panel_reference_precision import direct_square_window_sums, reference_quote_diagnostics
from src.verify_event_audit import reference_quote_diagnostics as original_reference

START = pd.Timestamp('2024-01-08', tz='UTC')
SECOND = pd.Timedelta(seconds=1)


def fixture(n=720, log_prices=None):
    time = pd.date_range(START, periods=n, freq='s')
    clock = pd.DataFrame({'bin_start': time, 'available_at': time+SECOND,
        'support_id': 0, 'flow_status': 2, 'quote_ofi': (np.arange(n)%7-3).astype(float)})
    if log_prices is None:
        log_prices = 0.05+np.sin(np.arange(n+2)/13)*1e-4
    mid = np.exp(np.asarray(log_prices))
    quotes = pd.DataFrame({'timestamp': pd.date_range(START-SECOND, periods=len(mid), freq='s'),
        'bid': mid-0.00001, 'ask': mid+0.00001, 'bid_volume': 2., 'ask_volume': 3.})
    support = pd.DataFrame({'start': [START-SECOND], 'end_exclusive': [START+pd.Timedelta(seconds=n+10)]})
    return clock, quotes, support


def scalar_variation(clock, quotes, support, row):
    if row < 300:
        return math.nan
    sid = int(clock.support_id.iloc[row])
    if sid < 0 or sid != int(clock.support_id.iloc[row-300]):
        return math.nan
    history = clock.iloc[row-300:row]
    if not history.flow_status.isin([1, 2]).all() or not np.isfinite(history.quote_ofi).all():
        return math.nan
    time = clock.bin_start.iloc[row]
    interval = support.iloc[sid]
    marks = []
    for offset in range(-300, 1, 5):
        mark = time+pd.Timedelta(seconds=offset)
        earlier = quotes.loc[quotes.timestamp.lt(mark)]
        if not len(earlier):
            return math.nan
        quote = earlier.iloc[-1]
        if not interval.start <= quote.timestamp < interval.end_exclusive:
            return math.nan
        marks.append(float(np.log((float(quote.bid)+float(quote.ask))/2)))
    terms = [(right-left)*(right-left) for left, right in zip(marks[:-1], marks[1:])]
    return math.sqrt(math.fsum(terms))*10000


class DirectSquareSumTests(unittest.TestCase):
    def test_tiny_terms_after_large_history_do_not_disappear(self):
        values = np.r_[np.ones(100), np.full(120, 1e-32)]
        actual = direct_square_window_sums(values)
        for i in [59, 99, 100, 158, 159, 219]:
            self.assertAlmostEqual(actual[i], math.fsum(values[i-59:i+1]), delta=1e-45 if i >=159 else 1e-12)
        self.assertGreater(actual[-1], 0)

    def test_long_flat_history_is_exactly_zero(self):
        values = np.r_[np.logspace(-20, 1, 100), np.zeros(20_000)]
        actual = direct_square_window_sums(values)
        np.testing.assert_array_equal(actual[159:], 0.)

    def test_missing_and_infinite_terms_require_sixty_finite_terms(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            values = np.ones(200); values[70] = invalid
            actual = direct_square_window_sums(values)
            self.assertTrue(np.isnan(actual[:59]).all())
            self.assertTrue(np.isnan(actual[70:130]).all())
            self.assertEqual(actual[69], 60.)
            self.assertEqual(actual[130], 60.)

    def test_nonnegative_terms_never_need_a_variance_floor(self):
        values = np.square(np.sin(np.arange(1000)))*1e-30
        actual = direct_square_window_sums(values)
        self.assertTrue((actual[np.isfinite(actual)] >= 0).all())
        for i in range(59, len(values), 37):
            self.assertAlmostEqual(actual[i], math.fsum(values[i-59:i+1]), delta=1e-42)

    def test_empty_and_short_sequences_remain_unavailable(self):
        self.assertEqual(len(direct_square_window_sums([])), 0)
        self.assertTrue(np.isnan(direct_square_window_sums(np.ones(59))).all())

    def test_invalid_inputs_fail(self):
        for values, window in [([-1.], 60), ([[1.]], 60), ([1.], 0), ([1.], True), ([1.], 1.5)]:
            with self.subTest(values=values, window=window), self.assertRaises(ValueError):
                direct_square_window_sums(values, window)


class QuoteStatePrecisionTests(unittest.TestCase):
    def assert_scalar(self, clock, quotes, support, rows):
        output = reference_quote_diagnostics(clock, quotes, support)
        for row in rows:
            expected = scalar_variation(clock, quotes, support, row)
            actual = output.pre_rv_300s_5s_bps.iloc[row]
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual), row)
            else:
                self.assertAlmostEqual(actual, expected, delta=1e-12, msg=str(row))
        return output

    def test_all_five_grid_phases_match_sixty_scalar_returns(self):
        f = fixture()
        self.assert_scalar(*f, rows=list(range(300, 310))+list(range(650, 655)))

    def test_alternating_tiny_log_returns_after_large_history(self):
        log = np.r_[np.linspace(.01, .3, 100), .3+((np.arange(622)%2)*1e-13)]
        self.assert_scalar(*fixture(log_prices=log), rows=list(range(500, 505))+[650, 700])

    def test_flat_quotes_after_large_moves_match_scalar_zero(self):
        log = np.r_[np.linspace(.01, .3, 100), np.full(622, .3)]
        output = self.assert_scalar(*fixture(log_prices=log), rows=list(range(500, 505))+[700])
        np.testing.assert_array_equal(output.pre_rv_300s_5s_bps.iloc[500:505], 0.)

    def test_missing_initial_predecessor_does_not_become_zero(self):
        clock, quotes, support = fixture()
        quotes = quotes.iloc[1:].reset_index(drop=True)
        output = self.assert_scalar(clock, quotes, support, [299, 300, 301, 305])
        self.assertTrue(np.isnan(output.pre_rv_300s_5s_bps.iloc[300]))
        self.assertTrue(np.isfinite(output.pre_rv_300s_5s_bps.iloc[301]))

    def test_missing_flow_breaks_history_until_it_leaves_window(self):
        clock, quotes, support = fixture()
        clock.loc[100, 'quote_ofi'] = np.nan
        output = self.assert_scalar(clock, quotes, support, [300, 399, 400, 401])
        self.assertTrue(output.pre_rv_300s_5s_bps.iloc[300:401].isna().all())

    def test_observation_support_change_is_a_real_history_barrier(self):
        clock, quotes, _ = fixture()
        clock.loc[240:, 'support_id'] = 1
        support = pd.DataFrame({'start': [START-SECOND, START+240*SECOND],
            'end_exclusive': [START+240*SECOND, START+730*SECOND]})
        output = self.assert_scalar(clock, quotes, support, [300, 539, 540, 541, 545, 700])
        self.assertTrue(np.isnan(output.pre_rv_300s_5s_bps.iloc[540]))
        self.assertTrue(np.isfinite(output.pre_rv_300s_5s_bps.iloc[541]))

    def test_quote_exactly_at_state_time_is_excluded(self):
        clock, quotes, support = fixture()
        before = reference_quote_diagnostics(clock, quotes, support)
        quotes.loc[quotes.timestamp.eq(START+300*SECOND), ['bid','ask']] *= 1.2
        after = self.assert_scalar(clock, quotes, support, [300, 301])
        self.assertEqual(before.pre_rv_300s_5s_bps.iloc[300], after.pre_rv_300s_5s_bps.iloc[300])
        self.assertNotEqual(before.pre_rv_300s_5s_bps.iloc[301], after.pre_rv_300s_5s_bps.iloc[301])

    def test_repeated_marks_are_valid_under_existing_structural_policy(self):
        clock, quotes, support = fixture()
        quotes = quotes.loc[~quotes.timestamp.between(START+50*SECOND, START+200*SECOND)].reset_index(drop=True)
        output = self.assert_scalar(clock, quotes, support, [300, 301, 350, 500])
        self.assertTrue(np.isfinite(output.pre_rv_300s_5s_bps.iloc[300]))

    def test_unrelated_measurement_fields_are_unchanged(self):
        f = fixture()
        old, new = original_reference(*f), reference_quote_diagnostics(*f)
        for column in old:
            if column != 'pre_rv_300s_5s_bps':
                pd.testing.assert_series_equal(old[column], new[column], check_exact=True)


class ReadOnlyVerificationCommandTests(unittest.TestCase):
    def test_verify_only_reports_result_without_build_tests_or_writes(self):
        import repair_chronological_panel as cli
        import src.verify_chronological_panel_v2 as verifier
        output = io.StringIO()
        with (patch('sys.argv', ['repair_chronological_panel.py', '--verify-only', 'fixture']),
              patch.object(verifier, 'verify_panel', return_value={'status': 'passed', 'checked': 1}) as verify,
              patch.object(cli, 'atomic_json') as write,
              patch.object(cli, 'build') as build,
              patch.object(cli, 'run_tests') as tests,
              redirect_stdout(output)):
            cli.main()
        verify.assert_called_once_with(cli.ROOT/'data/signals/fixture')
        write.assert_not_called(); build.assert_not_called(); tests.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), {'status': 'passed', 'checked': 1})

    def test_failed_verify_only_raises_without_build_tests_or_writes(self):
        import repair_chronological_panel as cli
        import src.verify_chronological_panel_v2 as verifier
        output = io.StringIO()
        with (patch('sys.argv', ['repair_chronological_panel.py', '--verify-only', 'fixture']),
              patch.object(verifier, 'verify_panel', return_value={'status': 'failed'}) as verify,
              patch.object(cli, 'atomic_json') as write,
              patch.object(cli, 'build') as build,
              patch.object(cli, 'run_tests') as tests,
              redirect_stdout(output)):
            with self.assertRaisesRegex(RuntimeError, 'did not pass'):
                cli.main()
        verify.assert_called_once_with(cli.ROOT/'data/signals/fixture')
        write.assert_not_called(); build.assert_not_called(); tests.assert_not_called()
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()

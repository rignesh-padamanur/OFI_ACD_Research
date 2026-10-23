"""Separate numerical-repair evidence; the earlier panel test snapshot is intact."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.event_audit import quote_diagnostics as historical_quote_diagnostics
from src.panel_reference_precision import (
    direct_square_window_sums as reference_sums,
    reference_quote_diagnostics,
)
from src.panel_state_precision import direct_square_window_sums, quote_diagnostics


ORIGIN = pd.Timestamp('2025-12-29 00:00:00', tz='UTC')
SECOND = 1_000_000_000


def fixture(log_levels, *, flow=None):
    levels = np.asarray(log_levels, dtype=float)
    times = pd.date_range(ORIGIN, periods=len(levels), freq='s')
    qt = pd.date_range(ORIGIN-pd.Timedelta(seconds=1), periods=len(levels)+2, freq='s')
    mid = np.exp(np.r_[levels, levels[-1], levels[-1]])
    quotes = pd.DataFrame({'timestamp': qt, 'bid': mid-.000001, 'ask': mid+.000001,
                           'bid_volume': 2.+np.arange(len(qt)) % 7,
                           'ask_volume': 3.+np.arange(len(qt)) % 11})
    clock = pd.DataFrame({'bin_start': times, 'support_id': 0,
                          'flow_status': 2,
                          'quote_ofi': np.sin(np.arange(len(times))*.13) if flow is None else flow})
    support = pd.DataFrame({'start': [qt[0]],
                            'end_exclusive': [qt[-1]+pd.Timedelta(seconds=1)]})
    return clock, quotes, support


def tiny_after_large_fixture():
    rng = np.random.default_rng(3)
    tail = np.cumsum(rng.choice([0., 0., 0., 2.22e-16, -2.22e-16], size=700))
    levels = np.r_[rng.normal(0., .01, 700), tail]
    return fixture(levels)


def scalar_variation(clock, quotes, support, position):
    """Rebuild the 61 raw marks and use scalar math.fsum on 60 squared returns."""
    if position < 300:
        return np.nan
    t = clock.bin_start.iloc[position]
    sid = int(clock.support_id.iloc[position])
    if sid < 0 or sid != int(clock.support_id.iloc[position-300]):
        return np.nan
    prior_flow = clock.quote_ofi.iloc[position-300:position].to_numpy()
    if not np.isfinite(prior_flow).all():
        return np.nan
    interval = support.iloc[sid]
    logs = []
    for offset in range(-300, 1, 5):
        mark = t+pd.Timedelta(seconds=offset)
        past = quotes.loc[quotes.timestamp.lt(mark)]
        if past.empty:
            return np.nan
        quote = past.iloc[-1]
        if quote.timestamp < interval.start or quote.timestamp >= interval.end_exclusive:
            return np.nan
        midpoint = (float(quote.bid)+float(quote.ask))/2.
        logs.append(float(np.log(midpoint)))
    squares = [(logs[j+1]-logs[j])**2 for j in range(60)]
    if not all(math.isfinite(value) for value in squares):
        return np.nan
    return math.sqrt(math.fsum(squares))*10000


class PanelStatePrecisionTests(unittest.TestCase):
    def assert_independent(self, clock, quotes, support, positions):
        actual = quote_diagnostics(clock, quotes, support, 'structural_only')
        reference = reference_quote_diagnostics(clock, quotes, support)
        for name in actual.columns:
            if name == 'bin_start':
                pd.testing.assert_series_equal(actual[name], reference[name], check_names=False)
            else:
                np.testing.assert_allclose(actual[name], reference[name], rtol=2e-14, atol=1e-14,
                                           equal_nan=True, err_msg=name)
        for position in positions:
            expected = scalar_variation(clock, quotes, support, position)
            np.testing.assert_allclose(actual.pre_rv_300s_5s_bps.iloc[position], expected,
                                       rtol=2e-15, atol=1e-25, equal_nan=True,
                                       err_msg='scalar window at '+str(position))
        return actual

    def test_tiny_returns_after_large_moves_match_scalar_without_a_floor(self):
        clock, quotes, support = tiny_after_large_fixture()
        actual = self.assert_independent(clock, quotes, support,
                                         list(range(1005, 1015))+[1101, 1193, 1399])
        values = actual.pre_rv_300s_5s_bps.iloc[1010:]
        self.assertTrue(values.gt(0.).all())
        self.assertLess(float(values.max()), 1e-9)
        historical = historical_quote_diagnostics(clock, quotes, support, 'structural_only')
        other = [name for name in actual if name != 'pre_rv_300s_5s_bps']
        pd.testing.assert_frame_equal(actual[other], historical[other], check_exact=True)

    def test_fully_flat_windows_are_exactly_zero_after_large_moves_leave(self):
        rng = np.random.default_rng(41)
        clock, quotes, support = fixture(np.r_[rng.normal(0., .03, 700), np.zeros(700)])
        actual = self.assert_independent(clock, quotes, support, [1000, 1001, 1002, 1003, 1004, 1399])
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[1000:].eq(0.).all())

    def test_all_five_grid_phases_and_exact_first_available_row(self):
        clock, quotes, support = fixture(np.sin(np.arange(905)*.173)*.001)
        actual = self.assert_independent(clock, quotes, support, list(range(295, 310))+list(range(900, 905)))
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[:300].isna().all())
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[300:].notna().all())

    def test_support_gap_and_new_support_require_full_new_history(self):
        clock, quotes, _ = fixture(np.sin(np.arange(1200)*.173)*.001)
        clock.loc[450:459, 'support_id'] = -1
        clock.loc[450:459, 'flow_status'] = 0
        clock.loc[450:459, 'quote_ofi'] = np.nan
        clock.loc[460:, 'support_id'] = 1
        support = pd.DataFrame({'start': [quotes.timestamp.iloc[0], ORIGIN+pd.Timedelta(seconds=460)],
                               'end_exclusive': [ORIGIN+pd.Timedelta(seconds=450),
                                                 quotes.timestamp.iloc[-1]+pd.Timedelta(seconds=1)]})
        actual = self.assert_independent(clock, quotes, support, [449, 450, 459, 460, 759, 760, 761, 762, 1199])
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[450:761].isna().all())
        self.assertTrue(actual.pre_rv_300s_5s_bps.iloc[761:].notna().all())

    def test_nan_flow_and_nan_quote_mark_preserve_unavailability(self):
        clock, quotes, support = fixture(np.sin(np.arange(1200)*.173)*.001)
        clock.loc[450, 'quote_ofi'] = np.nan
        clock.loc[450, 'flow_status'] = 3
        quotes.loc[551, ['bid', 'ask']] = np.nan
        actual = self.assert_independent(clock, quotes, support,
                                         [449, 450, 451, 549, 550, 551, 749, 750, 751, 850, 851, 1199])
        self.assertTrue(actual.pre_quote_ofi_300s.iloc[451:751].isna().all())

    def test_quotes_at_marks_are_excluded_and_future_quotes_cannot_change_states(self):
        clock, quotes, support = fixture(np.sin(np.arange(900)*.1)*.001)
        baseline = quote_diagnostics(clock, quotes, support, 'structural_only')
        changed = quotes.copy()
        at = ORIGIN+pd.Timedelta(seconds=600)
        changed.loc[changed.timestamp.ge(at), ['bid', 'ask']] *= 1.25
        result = quote_diagnostics(clock, changed, support, 'structural_only')
        prior_columns = ['pre_rv_300s_5s_bps', 'pre_quote_ofi_300s', 'pre_spread_pips',
                         'pre_displayed_size', 'pre_quote_age_sec']
        pd.testing.assert_frame_equal(baseline.loc[:600, prior_columns], result.loc[:600, prior_columns],
                                      check_exact=True)
        self.assertNotEqual(baseline.pre_rv_300s_5s_bps.iloc[601], result.pre_rv_300s_5s_bps.iloc[601])
        self.assertNotEqual(baseline.end_spread_pips.iloc[600], result.end_spread_pips.iloc[600])
        self.assert_independent(clock, changed, support, [599, 600, 601, 602, 603, 604, 605])

    def test_all_nonvariation_metrics_are_identical_to_historical_function(self):
        rng = np.random.default_rng(53)
        clock, quotes, support = fixture(np.cumsum(rng.normal(0., 1e-5, 1100)))
        quotes.loc[20:25, ['bid_volume', 'ask_volume']] = 0.
        actual = quote_diagnostics(clock, quotes, support, 'structural_only')
        historical = historical_quote_diagnostics(clock, quotes, support, 'structural_only')
        columns = [name for name in actual if name != 'pre_rv_300s_5s_bps']
        pd.testing.assert_frame_equal(actual[columns], historical[columns], check_exact=True)

    def test_no_quote_before_first_mark_and_short_clocks_remain_missing(self):
        for length in [1, 4, 5, 59, 60, 299, 300, 301, 302]:
            with self.subTest(length=length):
                clock, quotes, support = fixture(np.zeros(length))
                quotes = quotes.iloc[1:].reset_index(drop=True)
                support.loc[0, 'start'] = ORIGIN
                actual = self.assert_independent(clock, quotes, support, [length-1])
                if length <= 301:
                    self.assertTrue(actual.pre_rv_300s_5s_bps.isna().all())
                else:
                    self.assertEqual(actual.pre_rv_300s_5s_bps.iloc[301], 0.)

    def test_window_sum_has_no_memory_of_expired_large_terms(self):
        rng = np.random.default_rng(87)
        squares = np.r_[rng.uniform(.1, 100., 200), rng.uniform(0., 1e-32, 200)]
        actual = direct_square_window_sums(squares)
        reference = reference_sums(squares)
        np.testing.assert_allclose(actual, reference, rtol=3e-15, atol=0., equal_nan=True)
        for position in [59, 199, 200, 258, 259, 260, 399]:
            expected = math.fsum(squares[position-59:position+1])
            self.assertAlmostEqual(actual[position]/expected, 1., places=14)
        self.assertLess(float(actual[259:].max()), 1e-29)
        truncated = direct_square_window_sums(squares[200:])
        np.testing.assert_array_equal(actual[259:], truncated[59:])

    def test_finite_counts_reject_each_missing_term_without_zero_imputation(self):
        for missing in [np.nan, np.inf, -np.inf]:
            values = np.ones(140)
            values[65] = missing
            actual = direct_square_window_sums(values)
            expected = reference_sums(values)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.isnan(actual[:59]).all())
            self.assertTrue(np.isnan(actual[65:125]).all())
            self.assertEqual(actual[125], 60.)

    def test_window_input_validation_and_no_mutation(self):
        original = np.r_[np.zeros(60), np.ones(60)]
        saved = original.copy()
        self.assertEqual(direct_square_window_sums(original).shape, original.shape)
        np.testing.assert_array_equal(original, saved)
        self.assertEqual(len(direct_square_window_sums([])), 0)
        self.assertTrue(np.isnan(direct_square_window_sums([1., 2.])).all())
        for value, window in [([-1.], 60), ([[1.]], 60), ([1.], 0), ([1.], True), ([1.], 2.5)]:
            with self.subTest(value=value, window=window), self.assertRaises(ValueError):
                direct_square_window_sums(value, window)


if __name__ == '__main__':
    begun = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PanelStatePrecisionTests)
    names = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    clock, quotes, support = tiny_after_large_fixture()
    old = historical_quote_diagnostics(clock, quotes, support, 'structural_only')
    new = quote_diagnostics(clock, quotes, support, 'structural_only')
    reference = reference_quote_diagnostics(clock, quotes, support)
    paths = ['src/panel_state_precision.py', 'src/panel_reference_precision.py', 'src/event_audit.py',
             'reports/tests/test_panel_state_precision.py']
    evidence = {
        'stage': '07c_state_precision_numerical_repair',
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tests_run': result.testsRun, 'passed': result.wasSuccessful(),
        'failures': [{'test': test.id(), 'traceback': detail} for test, detail in result.failures],
        'errors': [{'test': test.id(), 'traceback': detail} for test, detail in result.errors],
        'skipped': [{'test': test.id(), 'reason': detail} for test, detail in result.skipped],
        'test_names': names, 'elapsed_seconds': time.monotonic()-begun,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        'tiny_after_large_fixture': {
            'rows': len(clock),
            'historical_max_bps_in_tiny_return_region': float(old.pre_rv_300s_5s_bps.iloc[1010:].max()),
            'direct_max_bps_in_tiny_return_region': float(new.pre_rv_300s_5s_bps.iloc[1010:].max()),
            'max_absolute_direct_reference_difference_bps': float(np.nanmax(np.abs(
                new.pre_rv_300s_5s_bps-reference.pre_rv_300s_5s_bps))),
        },
        'scope': 'Standalone numerical tests; does not replace the earlier 573-test panel evidence or verify the rebuilt full panel.',
    }
    (ROOT/'reports/tests/stage07c_state_precision_tests.json').write_text(json.dumps(evidence, indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)

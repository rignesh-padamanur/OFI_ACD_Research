"""Independent mean-uncertainty reconstruction and corruption detection."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.joint_uncertainty import build_joint_uncertainty
from src.verify_joint_uncertainty import verify_joint_uncertainty

START = pd.Timestamp('2024-05-06', tz='UTC')
END = START + pd.Timedelta(days=96)


def aggregate(membership):
    dates = pd.date_range(START, END, inclusive='left', freq='D')
    joint = membership.loc[membership.joint_included]
    common = set(joint.loc[joint.horizon_sec.eq(60), 'episode_id']) & set(joint.loc[joint.horizon_sec.eq(300), 'episode_id'])
    rows = []
    for cohort in ('horizon_joint', 'common_horizons_joint'):
        for h in (60, 300):
            chosen = joint.loc[joint.horizon_sec.eq(h)]
            if cohort == 'common_horizons_joint':
                chosen = chosen.loc[chosen.episode_id.isin(common)]
            grouped = dict(tuple(chosen.groupby(chosen.bin_start.dt.normalize())))
            for date in dates:
                part = grouped.get(date, chosen.iloc[:0])
                row = dict(cohort=cohort, horizon_sec=h, date=date, n=len(part))
                for component, field in [('J', 'initial_pips'), ('C', 'continuation_pips'), ('I', 'total_pips')]:
                    row['sum_' + component] = math.fsum(part[field])
                rows.append(row)
    return pd.DataFrame(rows)


def fixture():
    rng = np.random.default_rng(76128)
    event, records, state = 0, [], 0.
    for day, date in enumerate(pd.date_range(START, END, inclusive='left', freq='D')):
        state = .35 * state + rng.normal()
        if day % 8 == 0:
            continue
        for within in range(2 + day % 5):
            j = 1. + .3 * state + rng.normal(scale=.4)
            for horizon in (60, 300):
                c = -.2 * j + .1 * state + rng.normal(scale=.5 if horizon == 60 else .9)
                records.append(dict(episode_id=event, horizon_sec=horizon,
                    bin_start=date + pd.Timedelta(seconds=within),
                    joint_included=not (horizon == 300 and event % 13 == 0),
                    initial_pips=j, continuation_pips=c, total_pips=j + c))
            event += 1
    membership = pd.DataFrame(records)
    return aggregate(membership), membership


class VerifyJointUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily, cls.membership = fixture()
        cls.actual = build_joint_uncertainty(cls.daily, START, END)

    def verify(self, actual=None, daily=None, membership=None):
        return verify_joint_uncertainty(self.actual if actual is None else actual,
            self.daily if daily is None else daily,
            self.membership if membership is None else membership, START, END)

    def test_independent_reconstruction_and_serialized_round_trip(self):
        result = self.verify()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['target_count'], 12)
        self.assertEqual(result['interval_rows_verified'], 24)
        self.assertEqual(result['covariance_rows_verified'], 288)
        self.assertEqual(result['score_rows_verified'], 96 * 12)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            saved = {}
            for name, value in self.actual.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(base / (name + '.parquet'), index=False)
                    saved[name] = pd.read_parquet(base / (name + '.parquet'))
                else:
                    saved[name] = json.loads(json.dumps(value, allow_nan=False))
            self.assertEqual(self.verify(saved)['status'], 'passed')

    def test_interval_mean_and_standard_error_corruption_rejected(self):
        for field in ['mean', 'se', 'ci95_low', 'ci95_high']:
            bad = deepcopy(self.actual)
            bad['intervals'].loc[0, field] += .001
            with self.assertRaises(AssertionError):
                self.verify(bad)

    def test_covariance_cross_horizon_entry_corruption_rejected(self):
        bad = deepcopy(self.actual)
        indexes = bad['covariances'].index[bad['covariances'].target_i.ne(bad['covariances'].target_j)]
        bad['covariances'].loc[indexes[0], 'value'] += .001
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_scores_and_energy_shares_cannot_be_replaced(self):
        for field in ['score', 'influence', 'energy_share']:
            bad = deepcopy(self.actual)
            bad['scores'].loc[0, field] += .001
            with self.assertRaises(AssertionError):
                self.verify(bad)

    def test_lag_product_and_common_lag_corruption_rejected(self):
        bad = deepcopy(self.actual)
        bad['lag_products'].loc[0, 'normalized_product'] += .01
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.actual)
        bad['intervals'].loc[0, 'lag'] += 1
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_raw_event_regrouping_detects_daily_count_or_sum_corruption(self):
        for field in ['n', 'sum_J', 'sum_C', 'sum_I']:
            daily = self.daily.copy(deep=True)
            daily.loc[1, field] += 1
            with self.assertRaises(AssertionError):
                self.verify(daily=daily)

    def test_common_membership_selection_reconstructed_from_event_ids(self):
        membership = self.membership.copy(deep=True)
        index = membership.index[~membership.joint_included][0]
        membership.loc[index, 'joint_included'] = True
        with self.assertRaises(AssertionError):
            self.verify(membership=membership)

    def test_zero_date_removal_and_duplicate_source_keys_rejected(self):
        with self.assertRaises(AssertionError):
            self.verify(daily=self.daily.iloc[1:])
        bad = pd.concat([self.membership, self.membership.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            self.verify(membership=bad)

    def test_output_duplicate_keys_and_extra_columns_rejected(self):
        bad = deepcopy(self.actual)
        bad['intervals'] = pd.concat([bad['intervals'], bad['intervals'].iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.actual)
        bad['covariances']['unregistered'] = 0
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_event_weighted_mean_and_identical_common_j_are_preserved(self):
        rows = self.actual['intervals']
        common_j = rows.query("cohort == 'common_horizons_joint' and component == 'J'")
        for _, part in common_j.groupby('method'):
            self.assertEqual(part[['n', 'lag', 'mean', 'se', 'ci95_low', 'ci95_high']].drop_duplicates().shape[0], 1)
        daily = self.daily.query("cohort == 'horizon_joint' and horizon_sec == 60 and n > 0")
        event_mean = math.fsum(daily.sum_J) / int(daily.n.sum())
        daily_mean = (daily.sum_J / daily.n).mean()
        self.assertGreater(abs(event_mean - daily_mean), 1e-4)
        value = rows.query("cohort == 'horizon_joint' and horizon_sec == 60 and component == 'J'")['mean']
        np.testing.assert_array_equal(value, [event_mean, event_mean])

    def test_membership_identity_or_nonfinite_selected_response_rejected(self):
        for value in [999., np.inf]:
            bad = self.membership.copy(deep=True)
            index = bad.index[bad.joint_included][0]
            bad.loc[index, 'total_pips'] = value
            with self.assertRaises(AssertionError):
                self.verify(membership=bad)

    def test_pilot_statistics_cannot_be_changed(self):
        for field in ['bandwidth', 'scaled_s0', 'scaled_s1']:
            bad = deepcopy(self.actual)
            bad['statistics']['bandwidth']['component_pilots'][0][field] += .01
            with self.assertRaises(AssertionError):
                self.verify(bad)
        bad = deepcopy(self.actual)
        bad['statistics']['bandwidth']['selected_columns'] = [0]
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_provisional_flags_and_group_sizes_cannot_be_changed(self):
        for field in ['inference_approved', 'confidence_interval_coverage_validated', 'covariance_inverted']:
            bad = deepcopy(self.actual)
            bad['statistics'][field] = True
            with self.assertRaises(AssertionError):
                self.verify(bad)
        bad = deepcopy(self.actual)
        bad['statistics']['groups']['horizon_joint:60']['n'] += 1
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_concentration_tail_and_covariance_diagnostics_cannot_be_changed(self):
        bad = deepcopy(self.actual)
        bad['statistics']['influence_summary'][0]['maximum_day_energy_share'] += .01
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.actual)
        bad['statistics']['score_lag_tail_summary']['automatic']['max_abs_product'] += .01
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.actual)
        bad['statistics']['covariance_diagnostics']['automatic']['covariance_min_eigenvalue'] = .01
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_singular_accounting_and_unequal_cohort_counts(self):
        groups = self.actual['statistics']['groups']
        self.assertGreater(groups['horizon_joint:60']['n'], groups['horizon_joint:300']['n'])
        self.assertEqual(groups['common_horizons_joint:60']['n'], groups['common_horizons_joint:300']['n'])
        for _, rows in self.actual['covariances'].groupby('method'):
            covariance = rows.pivot(index='target_i', columns='target_j', values='value')
            self.assertLess(np.linalg.matrix_rank(covariance), 12)
            for cohort in ('horizon_joint', 'common_horizons_joint'):
                for horizon in (60, 300):
                    stem = f'{cohort}:{horizon}:'
                    np.testing.assert_allclose(covariance.loc[stem + 'I'],
                        covariance.loc[stem + 'J'] + covariance.loc[stem + 'C'], rtol=1e-12, atol=1e-14)

    def test_degenerate_coordinate_does_not_silently_leave_pilot(self):
        membership = self.membership.copy(deep=True)
        membership['initial_pips'] = 1.
        membership['total_pips'] = 1. + membership.continuation_pips
        with self.assertRaisesRegex(AssertionError, 'Degenerate scalar'):
            self.verify(daily=aggregate(membership), membership=membership)


if __name__ == '__main__':
    unittest.main()

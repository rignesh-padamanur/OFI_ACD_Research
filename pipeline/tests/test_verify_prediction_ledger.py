"""Adversarial verification of issuance, censoring and chronological refits."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.verify_prediction_ledger import verify_prediction_ledger


START = pd.Timestamp('2024-05-06', tz='UTC')
END = pd.Timestamp('2024-06-03', tz='UTC')


def fixture():
    times = pd.to_datetime([
        '2024-05-05T23:54:59Z', '2024-05-05T23:55:00Z',
        '2024-05-05T23:55:01Z', '2024-05-06T00:00:00Z',
        '2024-05-06T01:00:00Z', '2024-05-06T02:00:00Z',
        '2024-05-06T03:00:00Z', '2024-05-19T23:59:00Z',
        '2024-06-02T23:59:30Z', '2024-05-06T04:00:00Z'], utc=True)
    # keep IDs chronological without depending on file row order.
    times = times.sort_values()
    n = len(times)
    q = np.ones(n)
    zero_direction = int(np.flatnonzero(times == START + pd.Timedelta(1, unit='h'))[0])
    zero_initial = int(np.flatnonzero(times == START + pd.Timedelta(2, unit='h'))[0])
    negative_initial = int(np.flatnonzero(times == START + pd.Timedelta(3, unit='h'))[0])
    missing_endpoint = int(np.flatnonzero(times == START + pd.Timedelta(4, unit='h'))[0])
    q[zero_direction] = 0.
    pre_mid = np.full(n, 1.1)
    detection_mid = pre_mid + .0001
    detection_mid[zero_initial] = pre_mid[zero_initial]
    detection_mid[negative_initial] = pre_mid[negative_initial] - .0001
    raw_j = (detection_mid - pre_mid) * 10000
    d = np.sign(q)
    j = np.where(d != 0, d * raw_j, np.nan)
    a = pd.DataFrame(dict(episode_id=np.arange(n), bin_start=times-pd.Timedelta(1, unit='s'),
        available_at=times, support_id=np.zeros(n, dtype=int), quote_ofi=q, direction=d,
        direction_available=d != 0, pre_mid=pre_mid, detection_mid=detection_mid,
        initial_pips=j, raw_initial_pips=raw_j,
        pre_mark_at=times-pd.Timedelta(2, unit='s'),
        detection_mark_at=times-pd.Timedelta(100, unit='ms'),
        baseline_quote_available=np.ones(n, dtype=bool),
        onset_confirmed=np.ones(n, dtype=bool),
        state_history_available=np.ones(n, dtype=bool),
        state_sample_eligible=np.ones(n, dtype=bool),
        pre_mark_status=['available']*n, detection_mark_status=['available']*n,
        pre_quote_ofi_300s=np.arange(n, dtype=float)+1,
        pre_rv_300s_5s_bps=np.ones(n), pre_spread_pips=np.full(n, .2),
        pre_displayed_size=np.full(n, 100.), pre_quote_age_sec=np.ones(n),
        distance=np.full(n, 4.), z_ofi=np.full(n, 2.), z_intensity=np.ones(n),
        z_duration=np.ones(n), preceding_clear_seconds=np.full(n, 5)))
    f = a.copy()
    f['pre_rv_bps_5s'] = 1.
    f['pre_status_5s'] = 'available'
    f['pre_first_quote_at_5s'] = f.bin_start - pd.Timedelta(301, unit='s')
    f['pre_last_quote_at_5s'] = f.bin_start - pd.Timedelta(1, unit='s')
    f['pre_state_eligible'] = d != 0
    f['regressor_eligible'] = d != 0
    f['onset_hour_utc'] = f.bin_start.dt.hour
    f['onset_weekday_utc'] = f.bin_start.dt.dayofweek
    f['analysis_segment'] = np.where(f.bin_start < START, 'reused_development', 'later_reanalysis')
    scaling = dict(cQ=2., cV=1., sV=.5,
        reference_start='2024-02-26T00:00:00+00:00',
        reference_cutoff='2024-04-22T00:00:00+00:00')
    rows = []
    for row in a.itertuples(index=False):
        for h in (0, 60, 300):
            target = row.available_at + pd.Timedelta(h, unit='s')
            status = ('beyond_sample_end' if target > END else
                      'outside_onset_support' if row.episode_id == missing_endpoint and h else
                      'available')
            available = status == 'available'
            c = (.25 if h else 0.) if available and row.direction_available else np.nan
            rows.append(dict(episode_id=row.episode_id, bin_start=row.bin_start,
                detected_at=row.available_at, horizon_sec=h, target_at=target,
                endpoint_status=status,
                # deliberately stale: the timestamp cannot certify label maturity.
                endpoint_quote_at=row.available_at-pd.Timedelta(2, unit='s') if available else pd.NaT,
                endpoint_age_sec=float(h+2) if available else np.nan,
                initial_pips=row.initial_pips, continuation_pips=c,
                total_pips=row.initial_pips+c,
                initial_quote_available=True, continuation_quote_available=available,
                total_quote_available=available, triplet_quote_available=available,
                main_eligible_joint=available and row.direction_available))
    outcomes = pd.DataFrame(rows)
    outcomes['endpoint_quote_at'] = pd.to_datetime(outcomes.endpoint_quote_at, utc=True)
    cutoffs = pd.to_datetime(['2024-04-22', '2024-05-06', '2024-05-20'], utc=True)
    schedule = pd.DataFrame(dict(cycle_id=[0, 1, 2],
        training_start=cutoffs-pd.Timedelta(56, unit='D'), parameter_cutoff=cutoffs,
        deployment_end=pd.to_datetime(['2024-05-06', '2024-05-20', '2024-06-03'], utc=True)))
    return a, f, scaling, outcomes, schedule


def produced(a, f, scaling, outcomes, schedule):
    # producer is used only to create the candidate artifacts under test.
    from src.prediction_ledger import build_origins, attach_labels, build_folds
    first = build_origins(a, f, scaling)
    labels = attach_labels(first['origins'], outcomes, sample_end=END)
    last = build_folds(first['origins'], labels, schedule,
                       evaluation_start=START, sample_end=END)
    return {'origins': first['origins'], 'predictors': first['predictors'],
            'origin_statistics': first['statistics'], 'labels': labels,
            'folds': last['folds'], 'membership': last['membership'],
            'fold_statistics': last['statistics']}


class VerifyPredictionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.parents = fixture()
        self.artifacts = produced(*self.parents)

    def verify(self, artifacts=None, parents=None):
        return verify_prediction_ledger(*(self.parents if parents is None else parents),
            self.artifacts if artifacts is None else artifacts,
            evaluation_start=START, sample_end=END)

    def test_independent_reconstruction_and_maturity_boundary(self):
        result = self.verify()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['origins_verified'], 10)
        self.assertEqual(result['labels_verified'], 20)
        first = self.artifacts['membership'].query('fold_id == 1 and role == "training"')
        self.assertEqual(set(first.episode_id), {0, 1})
        self.assertEqual(first.iloc[1].label_ready_at, START)
        self.assertFalse(result['forecast_performance_validated'])

    def test_source_order_does_not_define_identity(self):
        a, f, sc, o, schedule = self.parents
        changed = (a.sample(frac=1, random_state=1), f.sample(frac=1, random_state=2),
                   sc, o.sample(frac=1, random_state=3), schedule.iloc[::-1])
        self.assertEqual(self.verify(parents=changed)['status'], 'passed')

    def test_duplicate_source_keys_and_missing_origins_fail(self):
        a, f, sc, o, s = self.parents
        cases = [(pd.concat([a, a.iloc[:1]]), f, sc, o, s),
                 (a, f.iloc[:-1], sc, o, s),
                 (a, f, sc, pd.concat([o, o.iloc[:1]]), s)]
        for parents in cases:
            with self.subTest(rows=len(parents[0])), self.assertRaises(AssertionError):
                self.verify(parents=parents)

    def test_predictor_leak_and_value_corruption_fail(self):
        for column in ['direction', 'initial_pips', 'endpoint_age_sec', 'main_eligible_joint']:
            bad = deepcopy(self.artifacts)
            bad['predictors'][column] = 0.
            with self.subTest(column=column), self.assertRaises(AssertionError):
                self.verify(bad)
        bad = deepcopy(self.artifacts)
        bad['predictors'].loc[0, 'signed_pressure'] += .1
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_missing_unavailable_label_or_prediction_row_fails(self):
        unavailable = self.artifacts['labels'].query('origin_eligible and not label_available')
        event = int(unavailable.episode_id.iloc[0])
        bad = deepcopy(self.artifacts)
        bad['labels'] = bad['labels'].drop(unavailable.index[0])
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.artifacts)
        mask = bad['membership'].role.eq('prediction') & bad['membership'].episode_id.eq(event)
        bad['membership'] = bad['membership'].loc[~mask]
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_missing_source_future_record_still_has_full_label_grid(self):
        a, f, sc, o, schedule = self.parents
        changed = (a, f, sc, o.loc[~(o.episode_id.eq(3) & o.horizon_sec.eq(300))], schedule)
        artifacts = produced(*changed)
        self.assertEqual(self.verify(artifacts, changed)['labels_verified'], 20)
        label = artifacts['labels'].query('episode_id == 3 and horizon_sec == 300').iloc[0]
        self.assertEqual(label.label_status, 'missing_record')
        self.assertTrue(label.origin_eligible)
        self.assertFalse(label.label_available)

    def test_stale_mark_cannot_mature_label_early(self):
        bad = deepcopy(self.artifacts)
        bad['labels']['label_ready_at'] = bad['labels'].endpoint_quote_at
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_premature_primary_training_and_secondary_substitution_fail(self):
        bad = deepcopy(self.artifacts)
        template = bad['membership'].query('fold_id == 1 and role == "training"').iloc[-1].copy()
        label = bad['labels'].query('episode_id == 2 and horizon_sec == 300').iloc[0]
        for key in ['episode_id', 'forecast_at', 'label_ready_at']:
            template[key] = label[key]
        bad['membership'] = pd.concat([bad['membership'], template.to_frame().T], ignore_index=True)
        # preserve dtypes so the failure concerns membership, not serialization.
        for key in ['episode_id', 'fold_id', 'horizon_sec']:
            bad['membership'][key] = bad['membership'][key].astype('int64')
        with self.assertRaises(AssertionError):
            self.verify(bad)
        bad = deepcopy(self.artifacts)
        bad['membership'].loc[0, 'horizon_sec'] = 60
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_zero_J_and_negative_J_remain_eligible(self):
        origins = self.artifacts['origins']
        for sign in ['zero', 'negative']:
            selected = origins.loc[origins.initial_sign.eq(sign)]
            self.assertTrue(len(selected) and selected.origin_eligible.all())
        self.assertFalse(origins.loc[origins.direction_status.eq('zero'), 'origin_eligible'].any())
        bad = deepcopy(self.artifacts)
        bad['origins'].loc[bad['origins'].initial_sign.eq('zero'), 'origin_eligible'] = False
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_available_J_C_requires_finite_total(self):
        a, f, sc, outcomes, s = self.parents
        outcomes = outcomes.copy()
        outcomes.loc[(outcomes.episode_id == 0) & (outcomes.horizon_sec == 300), 'total_pips'] = np.nan
        with self.assertRaises(AssertionError):
            self.verify(parents=(a, f, sc, outcomes, s))

    def test_future_accounting_flags_cannot_change_eligibility(self):
        a, f, sc, o, s = deepcopy(self.parents)
        a['closure_known_at'] = END + pd.Timedelta(1, unit='D')
        f['main_eligible_joint'] = False
        f['regressor_eligible'] = False
        o['main_eligible_joint'] = False
        self.assertEqual(self.verify(parents=(a, f, sc, o, s))['status'], 'passed')

    def test_wrong_endcap_and_aggregate_counts_fail(self):
        a, f, sc, o, s = deepcopy(self.parents)
        o.loc[o.target_at.gt(END), 'endpoint_status'] = 'outside_onset_support'
        with self.assertRaises(AssertionError):
            self.verify(parents=(a, f, sc, o, s))
        bad = deepcopy(self.artifacts)
        bad['fold_statistics']['prediction_origins'] -= 1
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_missing_future_response_cannot_be_zero_or_mean_filled(self):
        bad = deepcopy(self.artifacts)
        missing = ~bad['labels'].label_available
        bad['labels'].loc[missing, 'continuation_pips'] = 0.
        with self.assertRaises(AssertionError):
            self.verify(bad)

    def test_reported_waterfall_signs_and_origin_counts_are_verified(self):
        for name in ['direction_counts', 'eligible_initial_sign_counts', 'exclusion_reasons']:
            bad = deepcopy(self.artifacts)
            key = next(iter(bad['origin_statistics'][name]))
            bad['origin_statistics'][name][key] += 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.verify(bad)
        bad = deepcopy(self.artifacts)
        bad['origin_statistics']['waterfall'][0]['remaining'] -= 1
        with self.assertRaises(AssertionError):
            self.verify(bad)
        for name in ['warmup_origins', 'origins_at_sample_end_not_in_evaluation_window']:
            bad = deepcopy(self.artifacts)
            bad['fold_statistics'][name] += 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.verify(bad)

    def test_serialized_tables_preserve_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = deepcopy(self.artifacts)
            for name in ['origins', 'predictors', 'labels', 'folds', 'membership']:
                path = Path(directory) / (name + '.parquet')
                artifacts[name].to_parquet(path, index=False)
                artifacts[name] = pd.read_parquet(path)
            self.assertEqual(self.verify(artifacts)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()

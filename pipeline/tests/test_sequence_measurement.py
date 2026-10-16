"""Focused synthetic source/timing/cancellation checks; no empirical outcomes."""
import math
import unittest
import numpy as np
import pandas as pd
from src.sequence_measurement import (sequence_statistics, measure_pressure,
    measure_prior_prices, build_features, Q_COLUMNS, ORIGIN_COLUMNS,
    PREDICTOR_COLUMNS, ANCHOR_COLUMNS, NUMERIC_BASELINE, conditioning_diagnostics)

T = pd.Timestamp('2025-01-02T12:00:03Z')


def fixture():
    support = pd.DataFrame({'start': [T-pd.Timedelta(1,unit='h')],
        'end_exclusive': [T+pd.Timedelta(1,unit='h')], 'segment_id': [0], 'policy': ['structural_only']})
    onsets = pd.DataFrame({'episode_id': [0], 'bin_start': [T], 'support_id': [0]})
    sparse = pd.DataFrame({'bin_start': pd.date_range(T-pd.Timedelta(300,unit='s'), periods=301, freq='s'),
        'n_quotes': 1, 'n_invalid_transitions': 0, 'quote_ofi': 1.})
    quotes = pd.DataFrame({'timestamp': pd.date_range(T-pd.Timedelta(301,unit='s'), periods=303, freq='s'),
        'bid': 1.1 + np.arange(303)*1e-5, 'ask': 1.10002 + np.arange(303)*1e-5})
    return onsets, sparse, quotes, support


class SequenceAlgebraTests(unittest.TestCase):
    def test_constant_and_zero_compositions(self):
        q = np.array([[0.]*60, [1.]*60, [-3.]*60, [2.]+[0.]*59])
        r = sequence_statistics(q)
        np.testing.assert_array_equal(r.sequence_z, 0)
        self.assertTrue(r.order_degenerate.all())
        self.assertTrue(r.all_zero_sequence.iloc[0])
        self.assertFalse(r.all_zero_sequence.iloc[1])

    def test_reflection_preserves_sequence_score(self):
        q = np.tile(np.array([1., 1., -2., 0., -1.]), 12)[None, :]
        a, b = sequence_statistics(q), sequence_statistics(-q)
        self.assertEqual(a.sequence_z.iloc[0], b.sequence_z.iloc[0])
        self.assertEqual(a.sign_sum.iloc[0], -b.sign_sum.iloc[0])

    def test_permutation_reference_by_all_ordered_pairs(self):
        z = np.tile([1, -1, 0, 1, 1], 12)
        exact = math.fsum(z[i]*z[j] for i in range(60) for j in range(60) if i != j)/(60*59)
        r = sequence_statistics(z[None, :])
        self.assertEqual(r.composition_reference.iloc[0], exact)
        self.assertAlmostEqual(r.sequence_z.iloc[0], r.adjacent_agreement.iloc[0]-exact)

    def test_equal_aggregates_and_recent_six_different_order(self):
        early_a = np.r_[np.ones(27), -np.ones(27)]
        early_b = np.tile([1., -1.], 27)
        tail = np.array([1., -1., 1., -1., 1., -1.])
        q = np.vstack([np.r_[early_a, tail], np.r_[early_b, tail]])
        self.assertEqual(q[0].sum(), q[1].sum())
        self.assertEqual(abs(q[0]).sum(), abs(q[1]).sum())
        np.testing.assert_array_equal(q[0, -6:], q[1, -6:])
        r = sequence_statistics(q)
        self.assertEqual(r.composition_reference.iloc[0], r.composition_reference.iloc[1])
        self.assertNotEqual(r.sequence_z.iloc[0], r.sequence_z.iloc[1])

    def test_requires_exact_finite_width(self):
        for bad in [np.ones((1,59)), np.full((1,60), np.nan)]:
            with self.assertRaises(ValueError): sequence_statistics(bad)


class SourceTimingTests(unittest.TestCase):
    def test_onset_excluded_and_event_relative_phase(self):
        o, s, _, u = fixture()
        s.loc[s.bin_start.eq(T), 'quote_ofi'] = 1e9
        p, q = measure_pressure(o, s, u)
        np.testing.assert_array_equal(q[Q_COLUMNS], np.full((1,60),5.))
        self.assertEqual(p.sum_1s_fsum.iloc[0], 300.)
        self.assertEqual(p.prior_quote_updates.iloc[0], 300)
        self.assertEqual(o.bin_start.dt.second.iloc[0],3)

    def test_supported_empty_seconds_are_zero(self):
        o, s, _, u = fixture()
        s = s.iloc[0:0]
        p, q = measure_pressure(o, s, u)
        self.assertTrue(p.pressure_sequence_available.iloc[0])
        self.assertEqual(p.supported_empty_seconds.iloc[0],300)
        self.assertTrue(p.all_zero_sequence.iloc[0])
        np.testing.assert_array_equal(q[Q_COLUMNS],0)

    def test_invalid_active_bin_is_unavailable(self):
        o, s, _, u = fixture()
        s.loc[1, ['n_invalid_transitions', 'quote_ofi']] = [1, np.nan]
        p, q = measure_pressure(o,s,u)
        self.assertEqual(p.pressure_sequence_status.iloc[0],'invalid_observed_transition')
        self.assertTrue(q[Q_COLUMNS].isna().all(axis=None))

    def test_window_crossing_support_not_zero_filled(self):
        o,s,_,u=fixture();u.loc[0,'start']=T-pd.Timedelta(299,unit='s')
        p,q=measure_pressure(o,s,u)
        self.assertEqual(p.pressure_sequence_status.iloc[0],'window_crosses_support')
        self.assertTrue(q[Q_COLUMNS].isna().all(axis=None))

    def test_exact_quote_boundaries_excluded(self):
        o,_,quotes,u=fixture()
        p, marks=measure_prior_prices(o,quotes,u)
        self.assertEqual(marks.quote_at_60.iloc[0], T-pd.Timedelta(1,unit='s'))
        self.assertEqual(marks.quote_at_00.iloc[0], T-pd.Timedelta(301,unit='s'))
        self.assertAlmostEqual(p.prior_5s_price_pips.iloc[0],.5)
        quotes.loc[quotes.timestamp.ge(T),['bid','ask']] += 100
        p2,marks2=measure_prior_prices(o,quotes,u)
        pd.testing.assert_frame_equal(p,p2);pd.testing.assert_frame_equal(marks,marks2)

    def test_strict_predecessor_outside_support_is_unavailable(self):
        o,_,quotes,u=fixture();u.loc[0,'start']=T-pd.Timedelta(300,unit='s')
        p,_=measure_prior_prices(o,quotes,u)
        self.assertEqual(p.prior_price_status.iloc[0],'prior_quote_outside_support')

    def test_no_quotes_not_available(self):
        o,_,quotes,u=fixture()
        p,_=measure_prior_prices(o,quotes.iloc[0:0],u)
        self.assertEqual(p.prior_price_status.iloc[0],'no_prior_quote')

    def test_fsum_cancellation_preserves_nonzero_without_threshold(self):
        o,s,_,u=fixture()
        s.loc[:4,'quote_ofi']=[1e16,1.,-1e16,0.,0.]
        p,q=measure_pressure(o,s,u)
        self.assertEqual(q.q5_00.iloc[0],1.)
        self.assertEqual(p.numpy_fsum_sign_differences.iloc[0],1)
        self.assertEqual(p.bound_sensitive_nonzero_subbins.iloc[0],1)

    def test_duplicate_sparse_rows_rejected(self):
        o,s,_,u=fixture()
        with self.assertRaises(ValueError): measure_pressure(o,pd.concat([s,s.iloc[:1]]),u)


class FeatureContractTests(unittest.TestCase):
    def assembled(self):
        o,s,quotes,u=fixture();pressure,seq=measure_pressure(o,s,u);price,_=measure_prior_prices(o,quotes,u)
        origins=pd.DataFrame({'episode_id':[0],'bin_start':[T],'forecast_at':[T+pd.Timedelta(1,unit='s')],
            'origin_eligible':[True],'origin_exclusion_reason':['eligible'],'direction':[1.],'initial_pips':[.1],
            'pre_mark_at':[T-pd.Timedelta(1,unit='s')]})
        pred=pd.DataFrame({'episode_id':[0],'bin_start':[T],'forecast_at':origins.forecast_at,
            'signed_pressure':[3.],'pressure_magnitude':[3.],'variation_standardized':[0.],
            'pre_spread_pips':[.2],'log_pre_displayed_size':[1.],'onset_hour_utc':[12],'onset_weekday_utc':[3]})
        anchors=o.copy();anchors['state_history_available']=True;anchors['state_sample_eligible']=True
        anchors['pre_quote_ofi_300s']=300.;anchors['pre_rv_300s_5s_bps']=price.reconstructed_rv_5s_bps
        anchors['pre_recorded_updates_300s']=300
        scaling={'cQ':100.,'cV':0.,'sV':1.,'reference_cutoff':'2024-01-01T00:00:00Z'}
        return origins,pred,anchors,pressure,seq,price,scaling

    def test_future_columns_ignored_and_origins_preserved(self):
        args=self.assembled();a=build_features(*args)
        args[0]['continuation_pips']=1e100;args[1]['label_available']=False;args[2]['total_pips']=-1e100
        b=build_features(*args)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(a.combined_eligible.iloc[0]);self.assertEqual(a.sequence_z.iloc[0],0)
        self.assertIn('composition_reference',NUMERIC_BASELINE)
        self.assertNotIn('continuation_pips',a.columns)

    def test_corrupted_identity_fails(self):
        args=self.assembled();args[1].loc[0,'episode_id']=3
        with self.assertRaises(ValueError):build_features(*args)

    def test_inherited_ineligible_preserved_not_reselected(self):
        args=self.assembled();args[0].loc[0,'origin_eligible']=False
        a=build_features(*args)
        self.assertTrue(a.measurement_available.iloc[0]);self.assertFalse(a.combined_eligible.iloc[0])

    def test_inherited_pressure_contradiction_fails(self):
        args=self.assembled();args[2].loc[0,'pre_quote_ofi_300s']=100
        with self.assertRaises(ValueError):build_features(*args)

    def test_rank_audit_reads_predictors_only(self):
        args=self.assembled();f=build_features(*args)
        f=pd.concat([f]*80,ignore_index=True)
        f['initial_pips']=np.arange(80,dtype=float)
        f['sequence_z']=np.tile([1.,-1.],40)
        result=conditioning_diagnostics(f)
        self.assertFalse(result['response_values_read']);self.assertEqual(result['augmented_rank_increment'],1)


if __name__=='__main__':unittest.main()

"""Origin-vintage and history counterexamples for repeated deployment cycles."""
import unittest
import numpy as np
import pandas as pd
from src.chronological_panel import (ISSUED_COLUMNS, deployment_durations,
    issued_from_predictions, integer_fingerprint)
from src.event_audit import quote_diagnostics


C = pd.Timestamp('2024-05-20', tz='UTC')


def raw(origin, end, *, qid=1, scale=2., event=True, component=0):
    return pd.DataFrame({'origin_quote_id': [qid], 'component_id': [component],
        'origin_at': [origin], 'end_at': [end], 'event_observed': [event],
        'scale_sec': [scale], 'exposure_ns': [(end-origin).value]})


def issued(frame, *, vintage, psi=3.):
    f = frame[['origin_quote_id', 'component_id', 'origin_at', 'scale_sec']].copy()
    f['psi'] = psi; f['residual_parameter_cutoff'] = vintage
    return f[ISSUED_COLUMNS]


class PanelCarryTests(unittest.TestCase):
    def setUp(self):
        self.open = raw(C-pd.Timedelta(days=1), C+pd.Timedelta(days=60))
        self.carry = issued(self.open, vintage=C-pd.Timedelta(days=14))
        self.empty_raw = self.open.iloc[:0].copy(); self.empty = self.carry.iloc[:0].copy()

    def deploy(self, cutoff=C, stop=None, carry=None, source=None):
        return deployment_durations(self.empty, self.empty_raw, self.carry if carry is None else carry,
            self.open if source is None else source, cutoff=cutoff, stop=cutoff+pd.Timedelta(days=14) if stop is None else stop)

    def test_true_vintage_and_psi_survive_three_boundaries(self):
        carry = self.carry
        for n in range(3):
            bins, carry, counts = self.deploy(C+pd.Timedelta(days=14*n), carry=carry)
            self.assertEqual(len(bins), 0)
            pd.testing.assert_frame_equal(carry, self.carry)
            self.assertEqual(counts['carried_in'], 1)

    def test_carry_origin_before_current_training_start_is_not_lost(self):
        cutoff = C+pd.Timedelta(days=56)
        bins, out, count = self.deploy(cutoff)
        self.assertEqual(count['carried_origin_before_training_start'], 1)
        self.assertEqual(count['completed_carried_residuals'], 1)
        self.assertEqual(len(out), 0)
        expected = (self.open.end_at.iloc[0]-self.open.origin_at.iloc[0]).total_seconds()/2/3
        self.assertAlmostEqual(bins.mean_duration_residual.iloc[0], expected)

    def test_exact_cutoff_completion_uses_old_denominator(self):
        frame = raw(C-pd.Timedelta(seconds=6), C)
        bins, out, count = self.deploy(source=frame, carry=issued(frame,vintage=C-pd.Timedelta(days=14)))
        self.assertEqual(bins.bin_start.iloc[0], C)
        self.assertEqual(bins.mean_duration_residual.iloc[0], 1.)
        self.assertEqual(count['completed_carried_residuals'], 1)
        self.assertEqual(len(out), 0)

    def test_exact_end_completion_is_carried_then_observed_once(self):
        end = C+pd.Timedelta(days=14); source = raw(C-pd.Timedelta(seconds=6), end)
        i = issued(source,vintage=C-pd.Timedelta(days=14))
        bins, out, count = self.deploy(source=source, carry=i)
        self.assertEqual(len(bins),0); self.assertEqual(len(out),1)
        bins, out, count = self.deploy(end, source=source, carry=out)
        self.assertEqual(bins.completed_residual_count.sum(),1);self.assertEqual(len(out),0)

    def test_new_origin_at_cutoff_uses_its_own_forecast(self):
        old = raw(C-pd.Timedelta(seconds=6),C)
        new = raw(C,C+pd.Timedelta(seconds=1),qid=2,scale=1.)
        bins,out,c = deployment_durations(issued(new,vintage=C,psi=1.),new,
            issued(old,vintage=C-pd.Timedelta(days=14),psi=3.),old,cutoff=C,stop=C+pd.Timedelta(days=14))
        self.assertEqual(bins.completed_residual_count.sum(),2)
        np.testing.assert_array_equal(bins.mean_duration_residual,[1.,1.])

    def test_censored_terminal_never_becomes_completed_residual(self):
        frame=raw(C-pd.Timedelta(seconds=6),C+pd.Timedelta(seconds=1),event=False)
        bins,out,count=self.deploy(source=frame,carry=issued(frame,vintage=C-pd.Timedelta(days=14)))
        self.assertEqual(len(bins),0);self.assertEqual(len(out),0);self.assertEqual(count['completed_residuals'],0)

    def test_missing_required_carry_rejected(self):
        with self.assertRaises(ValueError):self.deploy(carry=self.empty)

    def test_duplicate_carry_rejected(self):
        with self.assertRaises(ValueError):self.deploy(carry=pd.concat([self.carry,self.carry]))

    def test_extraneous_carry_rejected(self):
        with self.assertRaises(ValueError):self.deploy(source=self.empty_raw)

    def test_origin_scale_and_component_cannot_change(self):
        for name,value in [('scale_sec',99.),('component_id',4),('origin_at',C-pd.Timedelta(days=2))]:
            f=self.carry.copy();f[name]=value
            with self.subTest(field=name),self.assertRaises(ValueError):self.deploy(carry=f)

    def test_parameter_vintage_after_origin_rejected(self):
        f=self.carry.copy();f['residual_parameter_cutoff']=C
        with self.assertRaises(ValueError):self.deploy(carry=f)

    def test_nonpositive_and_nonfinite_issued_forecast_rejected(self):
        for v in [0.,-1.,np.nan,np.inf]:
            f=self.carry.copy();f['psi']=v
            with self.subTest(value=v),self.assertRaises(ValueError):self.deploy(carry=f)

    def test_future_endpoint_change_does_not_revise_issued_state(self):
        f=self.open.copy();f['end_at']+=pd.Timedelta(days=100);f['exposure_ns']+=pd.Timedelta(days=100).value
        _,a,_=self.deploy();_,b,_=self.deploy(source=f)
        pd.testing.assert_frame_equal(a,b)

    def test_new_forecast_selection_does_not_replace_old_filter(self):
        p=pd.concat([raw(C-pd.Timedelta(seconds=1),C),raw(C,C+pd.Timedelta(seconds=1),qid=2)],ignore_index=True)
        prediction=pd.DataFrame({'origin_quote_id':[1,2],'psi':[11.,12.]})
        result=issued_from_predictions(p,prediction,cutoff=C)
        self.assertEqual(result.origin_quote_id.to_list(),[2]);self.assertEqual(result.psi.iloc[0],12.)
        self.assertEqual(prediction.psi.to_list(),[11.,12.])

    def test_prediction_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError):issued_from_predictions(self.open,pd.DataFrame({'origin_quote_id':[3],'psi':[2.]}),cutoff=C)

    def test_integer_fingerprint_is_order_and_boundary_sensitive(self):
        self.assertEqual(integer_fingerprint([1,2]),integer_fingerprint(np.array([1,2],dtype='int32')))
        self.assertNotEqual(integer_fingerprint([1,2]),integer_fingerprint([2,1]))
        self.assertNotEqual(integer_fingerprint([1],[2]),integer_fingerprint([1,2]))
        with self.assertRaises(ValueError):integer_fingerprint([1.5])

    def test_prediction_index_labels_do_not_reassign_forecast(self):
        path = pd.DataFrame({'origin_quote_id': [1, 2], 'component_id': [0, 0],
            'origin_at': [C-pd.Timedelta(seconds=1), C], 'scale_sec': [1., 1.]}, index=[1, 0])
        prediction = pd.DataFrame({'origin_quote_id': [1, 2], 'psi': [11., 12.]})
        result = issued_from_predictions(path, prediction, cutoff=C)
        self.assertEqual(result.origin_quote_id.tolist(), [2])
        self.assertEqual(result.psi.tolist(), [12.])

    def test_three_hundred_clock_seconds_preserve_deployment_states(self):
        times=pd.date_range(C-pd.Timedelta(seconds=700),periods=1001,freq='s')
        clock=pd.DataFrame({'bin_start':times,'support_id':0,'quote_ofi':np.sin(np.arange(len(times))*.2)})
        qt=pd.date_range(times[0]-pd.Timedelta(seconds=1000),times[-1]+pd.Timedelta(seconds=1),freq='2s')
        mid=1.+np.arange(len(qt))*.000001
        quotes=pd.DataFrame({'timestamp':qt,'bid':mid-.00001,'ask':mid+.00001,'bid_volume':1.,'ask_volume':1.})
        support=pd.DataFrame({'start':[qt[0]],'end_exclusive':[qt[-1]+pd.Timedelta(seconds=2)]})
        whole=quote_diagnostics(clock,quotes,support,'structural_only')
        piece=quote_diagnostics(clock.loc[clock.bin_start.ge(C-pd.Timedelta(seconds=300))].reset_index(drop=True),quotes,support,'structural_only')
        cols=['pre_quote_ofi_300s','pre_rv_300s_5s_bps']
        np.testing.assert_allclose(whole.loc[whole.bin_start.ge(C),cols],piece.loc[piece.bin_start.ge(C),cols],rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()

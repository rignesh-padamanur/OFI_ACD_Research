import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
from scipy.special import gamma
from scipy.stats import weibull_min
from src.duration_calibration import (DurationAuditSpec,SECOND,DAY,ns,sufficient_statistics,fit_scales,apply_scales,model_sample,source_spells)
from src.chronological_acd import arrays,evaluate,transform,transformed_objective,fit_acd,predict,diagnostics
from src.verify_chronological_acd import reference_spells,reference_scales,reference_sample,reference_predictions,check_diagnostics


def model_frame(y,event=None,reset=None):
    y=np.asarray(y,dtype=float);n=len(y)
    return pd.DataFrame({'origin_quote_id':np.arange(n),'normalized_exposure':y,'scale_sec':np.ones(n),
        'event_observed':np.ones(n,dtype=bool) if event is None else event,
        'recurrence_reset':np.r_[True,np.zeros(n-1,dtype=bool)] if reset is None else reset})


def calendar_frame(spec):
    origin=pd.Timestamp('2024-01-01',tz='UTC').value;day=np.repeat(np.arange(spec.total_days),4)
    a=origin+day*DAY+np.tile([1,10,20,30],spec.total_days)*SECOND
    end=a+np.tile([1,2,3,4],spec.total_days)*SECOND
    return origin,pd.DataFrame({'origin_quote_id':np.arange(len(a)),'component_id':np.arange(len(a)),
        'origin_at':pd.to_datetime(a,utc=True),'end_at':pd.to_datetime(end,utc=True),'event_observed':True,
        'stop_code':np.zeros(len(a),dtype='uint8'),'exposure_ns':end-a})


class SeasonalDurationTests(unittest.TestCase):
    def setUp(self):
        self.spec=DurationAuditSpec(14,7,7,1,1,1);self.origin,self.frame=calendar_frame(self.spec)

    def test_invalid_reference_specifications(self):
        for kwargs in [{'seasonal_days':0},{'validation_days':True},{'min_prior_dates':9}]:
            with self.assertRaises(ValueError):DurationAuditSpec(**kwargs)

    def test_origin_scale_uses_only_prior_matching_dates(self):
        table,corr=sufficient_statistics(self.frame,self.origin,self.spec.total_days)
        fitted,scales,status=fit_scales(table,corr,self.origin,self.spec)
        self.assertTrue(np.isnan(scales[:14]).all());self.assertEqual(scales[14,0],2.5)
        self.assertEqual(fitted.query('effective_date == @fitted.effective_date.iloc[14*96] and slot == 0').completed_events.iloc[0],8)

    def test_future_changes_do_not_revise_earlier_scales(self):
        table,corr=sufficient_statistics(self.frame,self.origin,self.spec.total_days)
        before,_,_=fit_scales(table,corr,self.origin,self.spec)
        f=self.frame.copy();future=ns(f.origin_at)>=self.origin+20*DAY
        f.loc[future,'end_at']+=pd.Timedelta(seconds=100);f.loc[future,'exposure_ns']+=100*SECOND
        t,c=sufficient_statistics(f,self.origin,self.spec.total_days);after,_,_=fit_scales(t,c,self.origin,self.spec)
        pd.testing.assert_frame_equal(before.iloc[:21*96],after.iloc[:21*96])

    def test_wait_running_at_fit_cutoff_is_censored_not_completed(self):
        f=self.frame.iloc[:1].copy();f.loc[0,'end_at']=pd.Timestamp(self.origin+15*DAY,tz='UTC')
        f['exposure_ns']=ns(f.end_at)-ns(f.origin_at)
        table,corr=sufficient_statistics(f,self.origin,self.spec.total_days)
        fitted,_,_=fit_scales(table,corr,self.origin,self.spec)
        row=fitted.iloc[14*96]
        self.assertEqual(row.completed_events,0)
        self.assertEqual(row.observed_exposure_ns,14*DAY-SECOND)

    def test_completion_exactly_at_cutoff_is_not_training_event(self):
        f=self.frame.iloc[:1].copy();f.loc[0,'end_at']=pd.Timestamp(self.origin+14*DAY,tz='UTC')
        f['exposure_ns']=ns(f.end_at)-ns(f.origin_at)
        table,corr=sufficient_statistics(f,self.origin,self.spec.total_days);fitted,_,_=fit_scales(table,corr,self.origin,self.spec)
        self.assertEqual(fitted.iloc[14*96].completed_events,0)

    def test_scale_is_frozen_at_origin_across_day_boundary(self):
        f=self.frame.iloc[:1].copy();a=self.origin+14*DAY-SECOND;b=a+3*SECOND
        f['origin_at']=pd.to_datetime([a],utc=True);f['end_at']=pd.to_datetime([b],utc=True);f['exposure_ns']=3*SECOND
        scales=np.ones((28,96))*2;scales[14]=100;statuses=np.ones((28,96),dtype='uint8')*5
        out=apply_scales(f,scales,statuses,self.origin)
        self.assertEqual(out.scale_sec.iloc[0],2);self.assertEqual(out.normalized_exposure.iloc[0],1.5)

    def test_exposure_scale_recovers_censored_exponential_reference(self):
        rng=np.random.default_rng(440);latent=rng.exponential(5,200000);x=np.minimum(latent,3);events=latent<=3
        self.assertAlmostEqual(x.sum()/events.sum(),5,delta=.06)
        self.assertLess(x[events].mean(),2)

    def test_training_cutoff_does_not_use_future_duration(self):
        f=self.frame.iloc[:2].copy();a=ns(f.origin_at)
        f.loc[1,'end_at']=pd.Timestamp(a[1]+100*SECOND,tz='UTC');f['scale_sec']=2.
        f['exposure_ns']=ns(f.end_at)-a
        out=model_sample(f,start_ns=self.origin,stop_ns=self.origin+50*SECOND)
        self.assertFalse(out.event_observed.iloc[-1]);self.assertEqual(out.normalized_exposure.iloc[-1],20)

    def test_missing_seasonal_factor_breaks_recursion(self):
        f=self.frame.iloc[:4].copy();a=self.origin+np.arange(4)*SECOND
        f['component_id']=0;f['origin_at']=pd.to_datetime(a,utc=True);f['end_at']=pd.to_datetime(a+SECOND,utc=True)
        f['exposure_ns']=SECOND;f['scale_sec']=[1,np.nan,1,1]
        out=model_sample(f,start_ns=self.origin,stop_ns=self.origin+10*SECOND)
        np.testing.assert_array_equal(out.recurrence_reset,[True,True,False])

    def test_future_completion_perturbation_keeps_training_sample(self):
        f=self.frame.iloc[:4].copy();f['scale_sec']=2.
        f.loc[3,'end_at']=pd.Timestamp(self.origin+100*SECOND,tz='UTC');f['exposure_ns']=ns(f.end_at)-ns(f.origin_at)
        before=model_sample(f,start_ns=self.origin,stop_ns=self.origin+50*SECOND)
        f.loc[3,'end_at']+=pd.Timedelta(seconds=200);f.loc[3,'exposure_ns']+=200*SECOND
        after=model_sample(f,start_ns=self.origin,stop_ns=self.origin+50*SECOND)
        pd.testing.assert_frame_equal(before,after)

    def test_independent_scale_reference(self):
        f=self.frame.copy();f.loc[[0,4,8],'event_observed']=False
        table,corr=sufficient_statistics(f,self.origin,self.spec.total_days);fitted,_,_=fit_scales(table,corr,self.origin,self.spec)
        from dataclasses import asdict
        reference=reference_scales(f,self.origin,asdict(self.spec))
        pd.testing.assert_frame_equal(fitted,reference)

    def test_tail_sensitivity_does_not_use_future_stop_type(self):
        f=self.frame.iloc[:1].copy();f['scale_sec']=2.
        stop=self.origin+50*SECOND
        for endpoint in [stop,stop+SECOND,stop+200*SECOND]:
            f['end_at']=pd.to_datetime([endpoint],utc=True)
            f['exposure_ns']=endpoint-ns(f.origin_at)
            expected=model_sample(f,start_ns=self.origin,stop_ns=stop,omit_structural_tails=True)
            for code in [2,3,6]:
                g=f.copy();g['event_observed']=False;g['stop_code']=code
                actual=model_sample(g,start_ns=self.origin,stop_ns=stop,omit_structural_tails=True)
                pd.testing.assert_frame_equal(actual,expected)
                pd.testing.assert_frame_equal(actual,reference_sample(g,self.origin,stop,True))
                self.assertEqual(len(actual),1);self.assertEqual(actual.stop_code.iloc[0],8)

    def test_tail_sensitivity_omits_only_past_structural_stops(self):
        f=self.frame.iloc[:3].copy();f['scale_sec']=1.
        f['event_observed']=False;f['stop_code']=[2,1,3]
        actual=model_sample(f,start_ns=self.origin,stop_ns=self.origin+50*SECOND,omit_structural_tails=True)
        self.assertEqual(actual.origin_quote_id.tolist(),[1])
        self.assertTrue(actual.recurrence_reset.iloc[0])
        pd.testing.assert_frame_equal(actual,reference_sample(f,self.origin,self.origin+50*SECOND,True))

    def test_source_and_independent_incoming_spells_agree(self):
        from tests.test_duration_observation import build_fixture
        with tempfile.TemporaryDirectory() as d:
            build_fixture(d);path=Path(d);tails=pd.read_parquet(path/'terminal_spells.parquet')
            for policy in ['structural_only','fresh_15s','fresh_30s','fresh_60s']:
                stop=self.origin+170*SECOND
                actual=source_spells(path/'quote_durations.parquet',tails,policy=policy,stop_ns=stop)
                expected=reference_spells(path/'quote_durations.parquet',tails,policy,stop)
                pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
                self.assertEqual(len(actual),6);self.assertEqual(actual.exposure_ns.iloc[-1],4*SECOND)
                self.assertFalse(actual.event_observed.iloc[-1]);self.assertEqual(actual.stop_code.iloc[-1],8)

    def test_seasonal_and_training_reference_sample(self):
        table,corr=sufficient_statistics(self.frame,self.origin,self.spec.total_days)
        _,scales,status=fit_scales(table,corr,self.origin,self.spec)
        frame=apply_scales(self.frame,scales,status,self.origin)
        a=self.origin+14*DAY;b=self.origin+21*DAY
        pd.testing.assert_frame_equal(model_sample(frame,start_ns=a,stop_ns=b),reference_sample(frame,a,b))


class ACDLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.frame=model_frame([.1,2.,1.,.4,3.,.7],[True,True,False,True,True,False],[True,False,False,True,False,False])

    def test_compiled_recursion_matches_manual_and_scipy_likelihood(self):
        params=[.2,.12,.7,.65];p=np.empty(6);y=self.frame.normalized_exposure.to_numpy()
        for i in range(6):p[i]=params[0]/(1-sum(params[1:3])) if self.frame.recurrence_reset.iloc[i] else params[0]+params[1]*y[i-1]+params[2]*p[i-1]
        scale=p/gamma(1+1/params[3]);event=self.frame.event_observed.to_numpy()
        expected=np.where(event,-weibull_min.logpdf(y,params[3],scale=scale),-weibull_min.logsf(y,params[3],scale=scale))
        nll,g,psi,h=evaluate(arrays(self.frame),params,weibull=True,predictions=True)
        np.testing.assert_allclose(psi,p,rtol=1e-13);self.assertAlmostEqual(nll,expected.mean(),places=12)
        np.testing.assert_allclose(h,-weibull_min.logsf(y,params[3],scale=scale),rtol=1e-13)

    def test_analytic_gradient_matches_finite_difference(self):
        for weibull in [False,True]:
            for unit in [False,True]:
                theta=np.array([-2.,.3,1.5]+([-.4] if weibull else []));data=arrays(self.frame)
                value,grad=transformed_objective(theta,data,weibull=weibull,unit_init=unit)
                numerical=approx_derivative(lambda z:transformed_objective(z,data,weibull=weibull,unit_init=unit)[0],theta).ravel()
                np.testing.assert_allclose(grad,numerical,rtol=1e-6,atol=1e-7)

    def test_weibull_shape_one_matches_exponential(self):
        p=[.2,.1,.7,1.];a=evaluate(arrays(self.frame),p,predictions=True);b=evaluate(arrays(self.frame),p,weibull=True,predictions=True)
        self.assertAlmostEqual(a[0],b[0],places=13);np.testing.assert_allclose(a[1][:3],b[1][:3],atol=1e-13)

    def test_mean_parameterization_of_weibull(self):
        p=np.array([.2,1.,5.]);k=.6
        np.testing.assert_allclose(weibull_min.mean(k,scale=p/gamma(1+1/k)),p,rtol=1e-13)

    def test_incomplete_exposure_contributes_survival_not_density(self):
        frame=model_frame([3.],[False],[True]);nll,_,_,_=evaluate(arrays(frame),[2.,0.,0.,1.])
        self.assertEqual(nll,1.5)

    def test_zero_censored_exposure_has_zero_likelihood(self):
        frame=model_frame([0.],[False],[True]);nll,g,_,_=evaluate(arrays(frame),[.2,.1,.7,.6],weibull=True)
        self.assertEqual(nll,0);np.testing.assert_array_equal(g,np.zeros(4))

    def test_invalid_continuation_after_censor_rejected(self):
        with self.assertRaises(ValueError):arrays(model_frame([1.,2.],[False,True],[True,False]))

    def test_invalid_parameters_rejected(self):
        for p in [[0,.1,.2,1.],[1,.6,.5,1.],[1,-.1,.2,1.],[1,.1,.2,0.]]:
            with self.assertRaises(ValueError):evaluate(arrays(self.frame),p)

    def test_future_duration_does_not_change_earlier_forecast(self):
        f=self.frame.copy();p=[.2,.1,.7,1.];a=predict(f,p);f.loc[4,'normalized_exposure']=99;b=predict(f,p)
        np.testing.assert_allclose(a.psi.iloc[:5],b.psi.iloc[:5])

    def test_reset_removes_previous_component_state(self):
        f=self.frame.copy();p=[.2,.1,.7,1.];a=predict(f,p);f.loc[:2,'normalized_exposure']*=500;b=predict(f,p)
        np.testing.assert_allclose(a.psi.iloc[3:],b.psi.iloc[3:])

    def test_raw_seconds_density_includes_jacobian(self):
        p=[.2,.1,.7,1.];a=predict(self.frame,p);f=self.frame.copy();f.scale_sec*=3;b=predict(f,p)
        np.testing.assert_allclose(b.negative_log_score_seconds-a.negative_log_score_seconds,self.frame.event_observed*np.log(3))

    def test_transform_constraints(self):
        for theta in [[-2.,12.,12.,-.5],[0.,-12.,-12.,1.]]:
            w,a,b,k=transform(theta,True);self.assertTrue(w>0 and a>=0 and b>=0 and a+b<1 and k>0)

    def test_censor_aware_hazard_calibration_known_exponential(self):
        rng=np.random.default_rng(508);x=rng.exponential(1,150000);cap=.5;event=x<=cap
        f=model_frame(np.minimum(x,cap),event,np.ones(len(x),dtype=bool));pred=predict(f,[1.,0.,0.,1.]);d=diagnostics(f,pred)
        self.assertAlmostEqual(d['event_hazard_ratio'],1,delta=.015)
        # Completed-only means are selected by censoring; do not compare them to one.
        self.assertLess(f.loc[f.event_observed,'normalized_exposure'].mean(),.3)

    def test_fit_recovers_simulated_censored_weibull_acd(self):
        rng=np.random.default_rng(7003);n=35000;k=.8;p=.2/(1-.12-.68);y=[];events=[];reset=[];new=True
        for i in range(n):
            if new:p=1.
            x=p*rng.weibull(k)/gamma(1+1/k);cap=3.;e=x<=cap
            y.append(min(x,cap));events.append(e);reset.append(new)
            if e:p=.2+.12*x+.68*p
            new=not e
        f=model_frame(y,events,reset);fit,attempts=fit_acd(f,weibull=True)
        self.assertIsNotNone(fit);self.assertEqual(len(attempts),3)
        self.assertAlmostEqual(fit['shape'],k,delta=.04)
        self.assertAlmostEqual(fit['alpha'],.12,delta=.04);self.assertAlmostEqual(fit['beta'],.68,delta=.09)

    def test_independent_reference_predictions_and_diagnostics(self):
        p=[.2,.1,.7,.6];actual=predict(self.frame,p,weibull=True)
        expected=reference_predictions(self.frame,p,weibull=True)
        pd.testing.assert_frame_equal(actual,expected,rtol=1e-12,atol=1e-12)
        check_diagnostics(self.frame,expected,diagnostics(self.frame,actual))

    def test_diagnostic_verifier_rejects_wrong_score(self):
        p=[.2,.1,.7,1.];pred=predict(self.frame,p);saved=diagnostics(self.frame,pred)
        saved['event_hazard_ratio']+=.1
        with self.assertRaises(AssertionError):check_diagnostics(self.frame,pred,saved)


if __name__=='__main__':unittest.main()

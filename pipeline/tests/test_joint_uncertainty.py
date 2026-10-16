import unittest
import numpy as np
import pandas as pd
from src.joint_uncertainty import build_joint_uncertainty


def fixture():
    rng=np.random.default_rng(741)
    dates=pd.date_range('2020-01-01',periods=90,tz='UTC')
    n=rng.integers(2,20,len(dates)); n[[2,8,15,23]]=0
    initial=rng.normal(.5,.15,len(dates))*n
    rows=[]
    for cohort in ['horizon_joint','common_horizons_joint']:
        for h in [60,300]:
            c=rng.normal(0,1,len(dates))*n
            for k,date in enumerate(dates):
                rows.append({'cohort':cohort,'horizon_sec':h,'date':date,'n':n[k],
                             'sum_J':initial[k],'sum_C':c[k],'sum_I':initial[k]+c[k]})
    return pd.DataFrame(rows),dates[0],dates[-1]+pd.Timedelta(days=1)


class TestJointUncertainty(unittest.TestCase):
    def test_event_weighted_and_zero_calendar(self):
        d,start,end=fixture();a=build_joint_uncertainty(d,start,end)
        self.assertEqual(len(a['intervals']),24);self.assertEqual(len(a['covariances']),288)
        self.assertEqual(len(a['scores']),90*12)
        g=d.loc[d.cohort.eq('horizon_joint')&d.horizon_sec.eq(60)]
        mu=g.sum_J.sum()/g.n.sum()
        self.assertNotAlmostEqual(mu,(g.sum_J/g.n).mean(),places=6)
        s=a['scores'].loc[a['scores'].target.eq('horizon_joint:60:J')]
        np.testing.assert_allclose(s.score,g.sum_J-g.n*mu,atol=1e-12)
        self.assertTrue(s.loc[s.n.eq(0),['score','influence','energy_share']].eq(0).all().all())

    def test_identical_targets_and_covariance_identity(self):
        d,start,end=fixture();a=build_joint_uncertainty(d,start,end)
        for method in ['automatic','longer']:
            f=a['intervals'].query('method==@method')
            j=f.loc[f.component.eq('J')]
            self.assertEqual(j.se.nunique(),1)
            c=a['covariances'].query('method==@method').pivot(index='target_i',columns='target_j',values='value')
            for prefix in ['horizon_joint:60','horizon_joint:300','common_horizons_joint:60','common_horizons_joint:300']:
                np.testing.assert_allclose(c.loc[prefix+':I'],c.loc[prefix+':J']+c.loc[prefix+':C'],atol=1e-12)
            self.assertLess(np.linalg.matrix_rank(c),12)
        self.assertFalse(a['statistics']['inference_approved'])

    def test_invariant_to_row_order(self):
        d,s,e=fixture();a=build_joint_uncertainty(d,s,e);b=build_joint_uncertainty(d.sample(frac=1,random_state=2),s,e)
        for k in ['intervals','covariances','scores','lag_products']:
            pd.testing.assert_frame_equal(a[k],b[k],check_exact=True)

    def test_common_scaling(self):
        d,s,e=fixture();a=build_joint_uncertainty(d,s,e)
        for scale in [10.,1e100]:
            with self.subTest(scale=scale):
                scaled=d.copy();scaled[['sum_J','sum_C','sum_I']]*=scale
                b=build_joint_uncertainty(scaled,s,e)
                self.assertEqual(a['statistics']['automatic_lag'],b['statistics']['automatic_lag'])
                np.testing.assert_allclose(b['intervals'].se,scale*a['intervals'].se)
                np.testing.assert_allclose(b['covariances'].value,scale**2*a['covariances'].value)
                self.assertTrue(np.isfinite(b['lag_products'].normalized_product).all())

    def test_bad_inputs(self):
        for variant in ['duplicate','missing_day','bad_sum','empty_sum','fraction','negative','nan','infinity','unknown','no_timezone']:
            with self.subTest(variant=variant):
                d,s,e=fixture()
                if variant=='duplicate':d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
                if variant=='missing_day':d=d.iloc[1:]
                if variant=='bad_sum':d.loc[0,'sum_I']+=1
                if variant=='empty_sum':d.loc[2,['sum_J','sum_I']]=1
                if variant=='fraction':d['n']=d.n.astype(float)+.1
                if variant=='negative':d.loc[0,'n']=-1
                if variant=='nan':d.loc[0,'sum_J']=np.nan
                if variant=='infinity':d.loc[0,'sum_C']=np.inf
                if variant=='unknown':d.loc[0,'cohort']='new'
                if variant=='no_timezone':s=s.tz_localize(None)
                with self.assertRaises(ValueError):build_joint_uncertainty(d,s,e)

    def test_degenerate_target_stops(self):
        d,s,e=fixture();d['sum_J']=0.;d['sum_I']=d.sum_C
        with self.assertRaisesRegex(ValueError,'Degenerate'):build_joint_uncertainty(d,s,e)

    def test_calendar_compression_is_rejected(self):
        d,s,e=fixture();d=d.loc[d.n.gt(0)]
        with self.assertRaises(ValueError):build_joint_uncertainty(d,s,e)


if __name__=='__main__':unittest.main()

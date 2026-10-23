import json
from pathlib import Path
import tempfile
import unittest
import runpy
import numpy as np
import pandas as pd
from src.data_preparation import atomic_json, sha256
from src.duration_observation import (SECOND, POLICIES, prepare_components, duration_chunk,
    materialize_duration_policy, terminal_spells, build_duration_artifacts)
from src.observation_eligibility import support_intervals
from src.verify_duration_observation import verify_duration_run
from src.pipeline_state import load_duration_observations


def fixture():
    origin = pd.Timestamp('2024-01-01', tz='UTC').value
    t = origin+np.array([0, 15*SECOND, 45*SECOND, 105*SECOND, 165*SECOND+1, 166*SECOND, 180*SECOND, 200*SECOND], dtype='int64')
    segments = pd.DataFrame({'segment_id':[0,1,2], 'quote_start':[0,5,7], 'quote_stop_exclusive':[5,7,8],
        'start':pd.to_datetime(t[[0,5,7]], utc=True), 'last_quote':pd.to_datetime(t[[4,6,7]], utc=True),
        'end_exclusive':pd.to_datetime([t[4]+1_000_000, t[6]+SECOND, t[-1]+1], utc=True)})
    support = support_intervals(t, segments)
    return t, segments, support


def build_fixture(directory, batch_size=3):
    directory = Path(directory); directory.mkdir(exist_ok=True)
    t, segments, support = fixture()
    source = pd.DataFrame({'timestamp':pd.to_datetime(t, utc=True)})
    for file,frame in [('clean.parquet',source),('raw.parquet',source),('segments.parquet',segments),('support.parquet',support)]:
        frame.to_parquet(directory/file, index=False)
    stats = build_duration_artifacts(directory/'clean.parquet', directory/'raw.parquet', support, segments, directory, batch_size=batch_size)
    atomic_json(directory/'statistics.json', stats)
    manifest = {'inputs':{k:{'path':str(directory/(k+'.parquet')), 'sha256':sha256(directory/(k+'.parquet'))} for k in ['clean','raw','segments','support']},
        'outputs':{file:sha256(directory/file) for file in ['quote_durations.parquet','terminal_spells.parquet','monthly_statistics.parquet','statistics.json']}}
    atomic_json(directory/'manifest.json', manifest)
    return stats


class DurationContractTests(unittest.TestCase):
    def setUp(self):
        self.t,self.segments,self.support = fixture()
        self.components = prepare_components(self.support,self.segments,len(self.t))
        self.frame = duration_chunk(pd.to_datetime(self.t,utc=True),0,self.components)

    def test_exact_integer_gaps_and_initial_sentinel(self):
        np.testing.assert_array_equal(self.frame.audit_gap_ns, np.r_[-1,np.diff(self.t)])
        self.assertFalse(self.frame.completed_structural_only.iloc[0])

    def test_exact_cap_equality_is_completed(self):
        self.assertTrue(self.frame.completed_fresh_15s.iloc[1])
        self.assertTrue(self.frame.completed_fresh_30s.iloc[2])
        self.assertTrue(self.frame.completed_fresh_60s.iloc[3])
        self.assertFalse(self.frame.completed_fresh_60s.iloc[4])

    def test_structural_keeps_long_wait_but_resets_short_boundary(self):
        self.assertTrue(self.frame.completed_structural_only.iloc[4])
        self.assertFalse(self.frame.completed_structural_only.iloc[5])
        self.assertLess(self.frame.audit_gap_ns.iloc[5],SECOND)

    def test_reset_marks_first_completed_not_first_quote(self):
        np.testing.assert_array_equal(np.flatnonzero(self.frame.recurrence_reset_structural_only), [1,6])
        self.assertFalse(self.frame.recurrence_reset_structural_only.iloc[7])

    def test_model_facing_missing_values_not_audit_gaps(self):
        out = materialize_duration_policy(self.frame,policy='structural_only')
        self.assertTrue(out.loc[[0,5,7],'duration_start'].isna().all())
        self.assertTrue(out.loc[[0,5,7],'duration_ns'].isna().all())
        self.assertTrue(out.loc[[0,5,7],'duration_sec'].isna().all())
        self.assertEqual(out.duration_start.iloc[6],pd.Timestamp(self.t[5],tz='UTC'))
        self.assertEqual(out.available_at.iloc[6],pd.Timestamp(self.t[6],tz='UTC'))

    def test_chunk_invariance(self):
        for size in [1,2,3,5,8]:
            chunks = [duration_chunk(pd.to_datetime(self.t[i:i+size],utc=True),i,self.components,None if i==0 else int(self.t[i-1])) for i in range(0,len(self.t),size)]
            pd.testing.assert_frame_equal(self.frame,pd.concat(chunks,ignore_index=True))

    def test_future_timestamp_does_not_change_prior_durations_given_audit_mask(self):
        changed=self.t.copy();changed[-1]+=SECOND
        out=duration_chunk(pd.to_datetime(changed,utc=True),0,self.components)
        pd.testing.assert_frame_equal(self.frame.iloc[:-1],out.iloc[:-1])

    def test_duplicate_time_rejected(self):
        t=self.t.copy();t[1]=t[0]
        with self.assertRaises(ValueError):duration_chunk(pd.to_datetime(t,utc=True),0,self.components)

    def test_missing_timestamp_rejected(self):
        t=self.t.copy();t[0]=pd.NaT.value
        with self.assertRaises(ValueError):duration_chunk(pd.to_datetime(t,utc=True),0,self.components)

    def test_backwards_time_rejected(self):
        with self.assertRaises(ValueError):duration_chunk(pd.to_datetime(self.t[::-1],utc=True),0,self.components)

    def test_absent_chunk_predecessor_rejected(self):
        with self.assertRaises(ValueError):duration_chunk(pd.to_datetime(self.t[2:],utc=True),2,self.components)

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):materialize_duration_policy(self.frame,policy='default')

    def test_partition_hole_rejected(self):
        s=self.support.copy();i=s.index[s.policy.eq('fresh_15s')][1];s.loc[i,'quote_start']+=1
        with self.assertRaises(ValueError):prepare_components(s,self.segments,len(self.t))

    def test_missing_policy_rejected(self):
        with self.assertRaises(ValueError):prepare_components(self.support[self.support.policy!='fresh_60s'],self.segments,len(self.t))

    def test_cap_epsilon_removed_and_sample_zero(self):
        tails=terminal_spells(self.components,self.segments,np.arange(len(self.t)),self.t,int(self.t[-1]))
        fresh=tails[tails.policy.eq('fresh_15s')]
        self.assertEqual(fresh.exposure_ns.iloc[0],15*SECOND)
        self.assertEqual(fresh.stop_code.iloc[0],1)
        for p in POLICIES:
            row=tails[tails.policy.eq(p)].iloc[-1]
            self.assertEqual(row.exposure_ns,0);self.assertFalse(row.positive_exposure);self.assertEqual(row.stop_code,4)

    def test_subsecond_structural_stop_preserved(self):
        tails=terminal_spells(self.components,self.segments,np.arange(len(self.t)),self.t,int(self.t[-1]))
        structural=tails[tails.policy.eq('structural_only')]
        self.assertEqual(structural.exposure_ns.iloc[0],1_000_000)
        self.assertEqual(structural.stop_code.iloc[0],2)

    def test_single_quote_component_no_invented_duration(self):
        tails=terminal_spells(self.components,self.segments,np.arange(len(self.t)),self.t,int(self.t[-1]))
        self.assertTrue((tails[tails.first_quote_id.eq(7)].completed_spells==0).all())

    def test_tied_structural_and_cap_stop_keeps_both_reasons(self):
        segments=self.segments.copy();segments.loc[0,'end_exclusive']=pd.Timestamp(self.t[4]+15*SECOND,tz='UTC')
        t=self.t.copy();t[5:]+=20*SECOND
        segments.loc[1,'start']=pd.Timestamp(t[5],tz='UTC');segments.loc[1,'last_quote']=pd.Timestamp(t[6],tz='UTC')
        segments.loc[1,'end_exclusive']=pd.Timestamp(t[6]+SECOND,tz='UTC')
        segments.loc[2,['start','last_quote']]=pd.Timestamp(t[7],tz='UTC');segments.loc[2,'end_exclusive']=pd.Timestamp(t[7]+1,tz='UTC')
        support=support_intervals(t,segments);components=prepare_components(support,segments,len(t))
        tails=terminal_spells(components,segments,np.arange(len(t)),t,int(t[-1]))
        row=tails[tails.policy.eq('fresh_15s') & tails.last_quote_id.eq(4)].iloc[0]
        self.assertEqual(row.stop_code,3);self.assertEqual(row.exposure_ns,15*SECOND)

    def test_legacy_apis_refuse_unverified_acd(self):
        from src.eacd import prepare_acd_data,load_eacd_params,compute_eacd_residuals
        from src.weibull import load_weibull_params,compute_weibull_residuals
        for func,args in [(prepare_acd_data,(pd.DataFrame(),False)),(load_eacd_params,(Path('.'),)),
            (compute_eacd_residuals,(pd.DataFrame(),{})),(load_weibull_params,(Path('.'),)),(compute_weibull_residuals,(pd.DataFrame(),{}))]:
            with self.assertRaisesRegex(RuntimeError,'stage 04d'):func(*args)

    def test_legacy_estimation_scripts_stop_before_data_io(self):
        root=Path(__file__).resolve().parents[1]
        for name in ['src/re_estimate_acd.py','src/run_calibration_audit.py']:
            with self.assertRaisesRegex(RuntimeError,'stage 04d'):runpy.run_path(str(root/name),run_name='__main__')


class DurationArtifactTests(unittest.TestCase):
    def registered_fixture(self, root):
        root=Path(root);run=root/'data/signals/fixture';run.mkdir(parents=True)
        build_fixture(run);m=json.loads((run/'manifest.json').read_text())
        m.update(status='verified_duration_observation_only',parent_data_run='quotes',parent_eligibility_run='support')
        atomic_json(run/'manifest.json',m);atomic_json(run/'verification.json',verify_duration_run(run))
        atomic_json(root/'data/active_run.json',{'data_run_id':'quotes'})
        atomic_json(root/'data/active_eligibility.json',{'eligibility_run_id':'support'})
        atomic_json(root/'data/active_duration_observation.json',{'duration_run_id':'fixture','parent_data_run':'quotes',
            'parent_eligibility_run':'support','status':m['status']})
        return run

    def test_loader_returns_completed_rows_and_terminal_records_together(self):
        with tempfile.TemporaryDirectory() as d:
            self.registered_fixture(d);iterator,tails=load_duration_observations(d,policy='fresh_15s',batch_size=2)
            rows=pd.concat(iterator,ignore_index=True)
            self.assertEqual(len(rows),8);self.assertEqual(len(tails),6)
            self.assertTrue(rows.loc[~rows.completed,'duration_sec'].isna().all())
            self.assertTrue(tails.policy.eq('fresh_15s').all())

    def test_loader_rejects_changed_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            self.registered_fixture(d);atomic_json(Path(d)/'data/active_run.json',{'data_run_id':'different'})
            with self.assertRaisesRegex(RuntimeError,'lineage'):load_duration_observations(d,policy='fresh_15s')

    def test_loader_rejects_changed_terminal_file(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.registered_fixture(d)
            with (run/'terminal_spells.parquet').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError,'artifact changed'):load_duration_observations(d,policy='fresh_15s')

    def test_full_fixture_independent_verification(self):
        with tempfile.TemporaryDirectory() as d:
            stats=build_fixture(d);v=verify_duration_run(d)
            self.assertEqual(v['status'],'passed');self.assertEqual(v['quote_rows_checked'],8)
            self.assertEqual(stats['policies']['structural_only']['completed'],5)
            self.assertEqual(stats['policies']['structural_only']['completed_over_60s'],1)
            self.assertEqual(stats['policies']['structural_only']['legacy_kept_but_ineligible'],2)

    def test_artifact_batch_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';sa=build_fixture(a,1);sb=build_fixture(b,5)
            for file in ['quote_durations.parquet','terminal_spells.parquet','monthly_statistics.parquet']:
                pd.testing.assert_frame_equal(pd.read_parquet(a/file),pd.read_parquet(b/file))
            self.assertEqual(sa['policies']['structural_only']['completed_exposure_ns'],sb['policies']['structural_only']['completed_exposure_ns'])

    def test_verifier_detects_wrong_reset_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            build_fixture(d);p=Path(d);f=p/'quote_durations.parquet';df=pd.read_parquet(f)
            df.loc[5,'recurrence_reset_structural_only']=True;df.to_parquet(f,index=False)
            m=json.loads((p/'manifest.json').read_text());m['outputs'][f.name]=sha256(f);atomic_json(p/'manifest.json',m)
            with self.assertRaises(AssertionError):verify_duration_run(p)

    def test_verifier_detects_artificial_terminal_nanosecond(self):
        with tempfile.TemporaryDirectory() as d:
            build_fixture(d);p=Path(d);f=p/'terminal_spells.parquet';df=pd.read_parquet(f)
            df.loc[2,'exposure_ns']+=1;df.to_parquet(f,index=False)
            m=json.loads((p/'manifest.json').read_text());m['outputs'][f.name]=sha256(f);atomic_json(p/'manifest.json',m)
            with self.assertRaises(AssertionError):verify_duration_run(p)

    def test_loader_validates_arguments_before_io(self):
        for policy,size in [('default',5),('structural_only',0),('fresh_15s',True)]:
            with self.assertRaises(ValueError):load_duration_observations('/missing',policy=policy,batch_size=size)


if __name__=='__main__':unittest.main()

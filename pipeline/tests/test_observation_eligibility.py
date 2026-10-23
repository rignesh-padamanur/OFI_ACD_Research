import unittest
import numpy as np
import pandas as pd
from src.observation_eligibility import (SECOND, HOUR, POLICIES, make_hours, quote_structure,
    build_segments, support_intervals, evaluate_windows, grid_ranges, count_ranges_by_hour,
    transition_statistics, utc_ns)
from src.acquisition import nominal_weekly_open


def fixture(seconds, keep=None, start='2024-01-08T00:00:00Z', end_hours=4):
    origin = pd.Timestamp(start).value
    raw = origin + np.array(seconds, dtype='int64')
    ids = np.arange(len(raw)) if keep is None else np.array(keep, dtype='int64')
    ts = raw[ids]
    ix = pd.date_range(pd.Timestamp(start), periods=end_hours, freq='h')
    hours = pd.DataFrame({'hour': ix, 'rows': np.zeros(end_hours, dtype='int64'),
        'nominal_weekly_open': nominal_weekly_open(ix)})
    for h in range(end_hours): hours.loc[h, 'rows'] = int(((raw >= origin+h*HOUR)&(raw < origin+(h+1)*HOUR)).sum())
    hours = make_hours(hours)
    removed = set((ids[:-1][np.diff(ids)>1]+1).tolist())
    if ids[-1] < len(raw)-1: removed.add(int(ids[-1]+1))
    bounds = {i: int(raw[i]) for i in removed}
    seg = build_segments(ts, ids, hours, bounds, int(raw[-1]+1))
    return ts, ids, hours, seg, support_intervals(ts, seg)


class ObservationContract(unittest.TestCase):
    def test_long_silence_retained_in_structural_support(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 10, 100])*SECOND)
        self.assertEqual(len(seg), 1)
        self.assertEqual(len(support[support.policy=='fresh_60s']), 2)
        counts, gaps = transition_statistics(ts, seg)
        self.assertEqual(counts['retained_quotes'], 3)
        self.assertEqual(counts['within_segment_gaps_gt_60s'], 1)
        self.assertFalse(gaps.crosses_structural_boundary.iloc[0])

    def test_absent_hour_breaks_transition(self):
        ts, ids, hours, seg, support = fixture(np.array([1, 3599, 7201])*SECOND)
        self.assertEqual(len(seg), 2)
        self.assertIn('absent_nominal_open_hour', seg.start_reason.iloc[1])
        self.assertEqual(seg.end_exclusive.iloc[0].value, pd.Timestamp('2024-01-08T01:00Z').value)
        self.assertEqual(seg.absent_open_hours_before.iloc[1], 1)

    def test_hour_boundary_without_missing_hour_does_not_reset(self):
        *_, seg, support = fixture(np.array([3599, 3601])*SECOND)
        self.assertEqual(len(seg), 1)

    def test_removed_record_boundary_starts_when_record_occurs(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 5, 10, 15])*SECOND, [0, 2, 3])
        self.assertEqual(len(seg), 2)
        self.assertEqual(seg.end_exclusive.iloc[0].value, ts[0]+5*SECOND)
        self.assertEqual(seg.removed_raw_records_before.iloc[1], 1)
        part = support[support.policy=='fresh_15s']
        anchors=pd.to_datetime(ts[0]+np.array([4,5,9,10])*SECOND,utc=True)
        self.assertEqual(evaluate_windows(part,anchors,0).pre_context_eligible.tolist(),[True,False,False,True])

    def test_trailing_removed_record_caps_support(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 5, 10])*SECOND, [0,1])
        self.assertEqual(seg.end_exclusive.iloc[-1].value, ts[0]+10*SECOND)

    def test_initial_removed_records_are_counted(self):
        *_, seg, support = fixture(np.array([0, 5, 10])*SECOND, [1,2])
        self.assertEqual(seg.removed_raw_records_before.iloc[0],1)
        self.assertIn('removed_raw_record',seg.start_reason.iloc[0])

    def test_sample_end_never_extrapolated(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 5])*SECOND)
        self.assertTrue((support.end_exclusive.astype('int64') <= ts[-1]+1).all())

    def test_freshness_exact_boundary_is_inclusive(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 100])*SECOND)
        part=support[support.policy=='fresh_15s']
        anchors=pd.to_datetime([ts[0]+15*SECOND, ts[0]+15*SECOND+1],utc=True)
        self.assertEqual(evaluate_windows(part,anchors,0).pre_context_eligible.tolist(),[True,False])

    def test_gap_equal_threshold_does_not_reset(self):
        *_, support = fixture(np.array([0, 15, 30])*SECOND)
        self.assertEqual(len(support[support.policy=='fresh_15s']),1)

    def test_gap_one_nanosecond_above_threshold_resets(self):
        *_, support = fixture([0, 15*SECOND+1])
        self.assertEqual(len(support[support.policy=='fresh_15s']),2)

    def test_both_endpoints_fresh_is_insufficient(self):
        ts, ids, hours, seg, support = fixture(np.array([0, 100, 101])*SECOND)
        part=support[support.policy=='fresh_15s']
        answer=evaluate_windows(part,pd.to_datetime([ts[0]],utc=True),0,[100])
        self.assertTrue(answer.pre_context_eligible.iloc[0])
        self.assertFalse(answer.outcome_available_100s.iloc[0])

    def test_future_availability_never_changes_pre_flag(self):
        ts, ids, hours, seg, support = fixture(np.arange(21)*SECOND)
        part=support[support.policy=='fresh_15s']
        answer=evaluate_windows(part,pd.to_datetime([ts[15]],utc=True),5,[1,300])
        self.assertTrue(answer.pre_context_eligible.iloc[0])
        self.assertTrue(answer.outcome_available_1s.iloc[0])
        self.assertFalse(answer.outcome_available_300s.iloc[0])

    def test_prefix_invariance_after_future_gap_and_future_removed_record(self):
        ts, _, _, _, old=fixture(np.arange(21)*SECOND)
        _, _, _, _, new=fixture(np.r_[np.arange(21),25,40,100]*SECOND, list(range(21))+[22,23])
        anchors=pd.to_datetime(ts,utc=True)
        for policy in POLICIES:
            a=evaluate_windows(old[old.policy==policy],anchors,5,[1])
            b=evaluate_windows(new[new.policy==policy],anchors,5,[1])
            np.testing.assert_array_equal(a.pre_context_eligible,b.pre_context_eligible)

    def test_clock_count_matches_brute_force_every_second(self):
        ts, ids, hours, seg, support=fixture(np.array([0,1,10,50,3599,3601,7201,7250])*SECOND+123)
        edges=np.r_[utc_ns(hours.hour)//SECOND,utc_ns(hours.hour)[-1]//SECOND+3600]
        anchors=pd.to_datetime(np.arange(edges[0],edges[-1])*SECOND,utc=True)
        for policy in POLICIES:
            part=support[support.policy==policy]
            for lookback in [0,1,60,300]:
                for horizon in [0,1,5,60,300]:
                    lo,hi=grid_ranges(part,lookback,horizon)
                    counts=count_ranges_by_hour(lo,hi,edges)
                    a=utc_ns(part.start);b=utc_ns(part.end_exclusive)
                    oracle=np.zeros(len(anchors),dtype=bool)
                    for left,right in zip(a,b):
                        oracle |= (anchors.asi8-lookback*SECOND>=left)&(anchors.asi8+horizon*SECOND<right)
                    np.testing.assert_array_equal(counts,oracle.reshape(-1,3600).sum(axis=1))

    def test_calendar_closed_period_is_separate_from_unresolved_absence(self):
        # Friday 20:59, Sunday 22:00 UTC in winter; winter Friday 21:00 missing too.
        ts, ids, hours, seg, support=fixture(np.array([59*60,50*3600])*SECOND,
            start='2024-01-05T20:00:00Z',end_hours=51)
        self.assertEqual(seg.absent_open_hours_before.iloc[1],1)
        self.assertEqual(seg.weekly_closed_hours_before.iloc[1],48)
        self.assertIn('weekly_closure',seg.start_reason.iloc[1])

    def test_negative_window_rejected(self):
        ts, _, _, _, support=fixture([0, SECOND])
        with self.assertRaises(ValueError): evaluate_windows(support[support.policy=='fresh_15s'],pd.to_datetime(ts,utc=True),-1)

    def test_mixed_policies_rejected(self):
        ts, _, _, _, support=fixture([0, SECOND])
        with self.assertRaises(ValueError): evaluate_windows(support,pd.to_datetime(ts,utc=True),0)

    def test_naive_time_rejected(self):
        with self.assertRaises(ValueError): utc_ns(pd.to_datetime(['2024-01-01']))

    def test_calendar_corruption_rejected(self):
        _, _, hours, _, _=fixture([0, SECOND]); hours.loc[0,'nominal_weekly_open']=False
        with self.assertRaises(ValueError): make_hours(hours)

    def test_missing_hour_in_ledger_rejected(self):
        _, _, hours, _, _=fixture([0, SECOND])
        with self.assertRaises(ValueError): make_hours(hours.drop(index=1))

    def test_noncovered_quote_rejected(self):
        ts, ids, hours, _, _=fixture([0, SECOND]);hours.loc[0,'observation_block_id']=-1
        with self.assertRaises(ValueError): quote_structure(ts,ids,hours)

    def test_single_quote_has_one_clock_anchor_only_if_on_grid(self):
        for offset,expected in [(0,1),(1,0)]:
            *_,support=fixture([offset])
            for policy in POLICIES:
                lo,hi=grid_ranges(support[support.policy==policy],0,0)
                self.assertEqual(int((hi-lo).sum()),expected)

    def test_two_independent_failures_have_both_reasons(self):
        *_, seg, support=fixture(np.array([1,3601,7201])*SECOND,[0,2])
        # raw middle hour has a removed quote, so the structural cut is solely cleaning.
        self.assertEqual(seg.start_reason.iloc[1],'removed_raw_record')
        self.assertEqual(seg.quote_count.sum(),2)


if __name__=='__main__': unittest.main()

class SerializedObservationContract(unittest.TestCase):
    def setUp(self):
        import tempfile, json, contextlib, io
        from pathlib import Path
        from src.data_preparation import atomic_json, sha256
        from repair_observation_eligibility import write_artifacts
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.parent=self.root/'parent';self.parent.mkdir()
        self.run=self.root/'run';self.run.mkdir()
        origin=pd.Timestamp('2024-01-08T00:00Z').value
        raw_ts=origin+np.array([1,2,3,4,50,51,3599,7201,7202])*SECOND
        ids=np.array([0,1,3,4,5,6,7,8])
        raw=pd.DataFrame({'timestamp':pd.to_datetime(raw_ts,utc=True)})
        clean=pd.DataFrame({'timestamp':pd.to_datetime(raw_ts[ids],utc=True),
            'source_file':'eurusd_2024_01.parquet','source_row':ids})
        raw.to_parquet(self.parent/'raw_quotes.parquet',index=False,row_group_size=2)
        clean.to_parquet(self.parent/'clean_quotes.parquet',index=False,row_group_size=2)
        atomic_json(self.parent/'manifest.json',{'input_files':[{'name':'eurusd_2024_01.parquet','rows':len(raw)}]})
        ix=pd.date_range('2024-01-08',periods=4,freq='h',tz='UTC')
        grouped=raw.groupby(raw.timestamp.dt.floor('h')).timestamp.agg(['size','min','max']).reindex(ix)
        coverage=pd.DataFrame({'hour':ix,'rows':grouped['size'].fillna(0).astype('int64').to_numpy(),
            'first_timestamp':grouped['min'].to_numpy(),'last_timestamp':grouped['max'].to_numpy(),
            'nominal_weekly_open':True})
        coverage.to_parquet(self.root/'coverage.parquet',index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.stats,names=write_artifacts(self.parent,self.root/'coverage.parquet',self.run)
        self.manifest={'parent_directory':str(self.parent),'input_hashes':{},'code_sha256':{},
            'output_sha256':{name:sha256(self.run/name) for name in names}}
        atomic_json(self.run/'manifest.json',self.manifest)

    def refresh_hash(self,name):
        from src.data_preparation import atomic_json,sha256
        self.manifest['output_sha256'][name]=sha256(self.run/name)
        atomic_json(self.run/'manifest.json',self.manifest)

    def test_serialized_outputs_all_verified(self):
        from src.verify_observation_eligibility import verify_eligibility
        result=verify_eligibility(self.run)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['retained_quotes_checked'],8)
        self.assertEqual(result['raw_timestamps_checked'],9)

    def test_wrong_segment_boundary_detected_even_with_refreshed_hash(self):
        from src.verify_observation_eligibility import verify_eligibility
        name='segments.parquet';frame=pd.read_parquet(self.run/name)
        frame.loc[0,'end_exclusive']+=pd.Timedelta(nanoseconds=1)
        frame.to_parquet(self.run/name,index=False);self.refresh_hash(name)
        with self.assertRaises(AssertionError):verify_eligibility(self.run)

    def test_wrong_window_count_detected_even_with_refreshed_hash(self):
        from src.verify_observation_eligibility import verify_eligibility
        name='window_attrition_by_hour.parquet';frame=pd.read_parquet(self.run/name)
        frame.loc[0,'eligible_clock_anchors']+=1
        frame.to_parquet(self.run/name,index=False);self.refresh_hash(name)
        with self.assertRaises(AssertionError):verify_eligibility(self.run)

    def test_wrong_freshness_endpoint_detected_even_with_refreshed_hash(self):
        from src.verify_observation_eligibility import verify_eligibility
        name='support_intervals.parquet';frame=pd.read_parquet(self.run/name)
        i=frame.index[frame.policy.eq('fresh_15s')][0]
        frame.loc[i,'end_exclusive']-=pd.Timedelta(nanoseconds=1)
        frame.to_parquet(self.run/name,index=False);self.refresh_hash(name)
        with self.assertRaises(AssertionError):verify_eligibility(self.run)

    def test_changed_raw_provenance_detected(self):
        from src.verify_observation_eligibility import verify_eligibility
        path=self.parent/'raw_quotes.parquet';frame=pd.read_parquet(path)
        frame.loc[0,'timestamp']+=pd.Timedelta(nanoseconds=1)
        frame.to_parquet(path,index=False)
        with self.assertRaises(AssertionError):verify_eligibility(self.run)

    def test_corrupted_quote_transition_statistic_detected(self):
        import json
        from src.data_preparation import atomic_json
        from src.verify_observation_eligibility import verify_eligibility
        name='statistics.json';stats=json.loads((self.run/name).read_text())
        stats['quote_transitions']['structurally_eligible_transitions']+=1
        atomic_json(self.run/name,stats);self.refresh_hash(name)
        with self.assertRaises(AssertionError):verify_eligibility(self.run)

    def test_loader_rejects_mismatched_data_lineage(self):
        from src.pipeline_state import load_observation_support
        from src.data_preparation import atomic_json
        atomic_json(self.root/'data/active_run.json',{'data_run_id':'new_parent'})
        atomic_json(self.root/'data/active_eligibility.json',{'parent_data_run':'old_parent'})
        with self.assertRaises(RuntimeError):load_observation_support(self.root,policy='fresh_15s')

    def test_implicit_policy_not_allowed(self):
        from src.pipeline_state import load_observation_support
        with self.assertRaises(ValueError):load_observation_support(self.root,policy=None)

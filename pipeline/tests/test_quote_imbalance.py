import unittest
import numpy as np
import pandas as pd
from src.ofi import compute_quote_imbalance,compute_ofi,prepare_support,aggregate_observed_bins,BinStream,POLICIES

BASE=pd.Timestamp('2024-01-07T22:00:00Z')


def quotes(states, nanos=None, segments=None):
    n=len(states)
    return pd.DataFrame({'quote_id':np.arange(n,dtype='int64'),
        'timestamp':pd.to_datetime(BASE.value+(np.arange(n)*100_000_000 if nanos is None else np.array(nanos)),utc=True),
        'segment_id':np.zeros(n,dtype='int64') if segments is None else np.array(segments,dtype='int64'),
        'bid':[r[0] for r in states],'ask':[r[1] for r in states],
        'bid_volume':[r[2] for r in states],'ask_volume':[r[3] for r in states]})


def support(start=BASE,end=BASE+pd.Timedelta(seconds=100)):
    return prepare_support(pd.DataFrame([{'policy':p,'start':start,'end_exclusive':end} for p in POLICIES]))


class QuoteImbalanceContract(unittest.TestCase):
    def test_all_nine_price_move_combinations(self):
        for bid_change in (-1,0,1):
            for ask_change in (-1,0,1):
                first=[100,110,10,20];second=[100+bid_change,110+ask_change,13,17]
                expected_bid={-1:-10,0:3,1:13}[bid_change]
                expected_ask={-1:-17,0:3,1:20}[ask_change]
                out=compute_quote_imbalance(quotes([first,second]))
                self.assertEqual(out.bid_pressure.iloc[1],expected_bid)
                self.assertEqual(out.ask_pressure.iloc[1],expected_ask)
                self.assertEqual(out.quote_ofi.iloc[1],expected_bid+expected_ask)

    def test_parallel_up_has_positive_flow(self):
        out=compute_quote_imbalance(quotes([[100,110,10,20],[101,111,13,17]]))
        self.assertEqual(out.quote_ofi.iloc[1],33)

    def test_parallel_down_has_negative_flow(self):
        out=compute_quote_imbalance(quotes([[100,110,10,20],[99,109,13,17]]))
        self.assertEqual(out.quote_ofi.iloc[1],-27)

    def test_bid_size_increase_and_ask_size_decrease_positive(self):
        out=compute_quote_imbalance(quotes([[100,110,10,20],[100,110,13,17]]))
        self.assertEqual(out.quote_ofi.iloc[1],6)

    def test_symmetric_narrowing_balanced_sizes_has_zero_net(self):
        out=compute_quote_imbalance(quotes([[100,110,10,10],[101,109,10,10]]))
        self.assertEqual(out.quote_ofi.iloc[1],0)

    def test_fixed_size_spread_roundtrip_cancels(self):
        out=compute_quote_imbalance(quotes([[100,110,10,20],[101,109,10,20],[100,110,10,20]]))
        self.assertEqual(out.quote_ofi.sum(),0)

    def test_inserted_unchanged_state_does_not_change_total(self):
        a=[100,110,10,20];b=[101,111,13,17]
        self.assertEqual(compute_quote_imbalance(quotes([a,b])).quote_ofi.sum(),
            compute_quote_imbalance(quotes([a,a,b,b])).quote_ofi.sum())

    def test_fixed_price_size_changes_telescope(self):
        a=[100,110,10,20];b=[100,110,13,17];c=[100,110,7,29]
        self.assertEqual(compute_quote_imbalance(quotes([a,c])).quote_ofi.sum(),
            compute_quote_imbalance(quotes([a,b,c])).quote_ofi.sum())

    def test_coarsening_is_not_claimed_invariant(self):
        a=[100,110,10,20];b=[101,110,10,20];c=[102,110,10,20]
        self.assertNotEqual(compute_quote_imbalance(quotes([a,c])).quote_ofi.sum(),
            compute_quote_imbalance(quotes([a,b,c])).quote_ofi.sum())

    def test_size_unit_scaling_is_linear(self):
        q=quotes([[100,110,10,20],[101,111,13,17]])
        scaled=q.copy();scaled[['bid_volume','ask_volume']]*=1_000_000
        np.testing.assert_allclose(compute_quote_imbalance(scaled).quote_ofi,
            compute_quote_imbalance(q).quote_ofi*1_000_000,equal_nan=True)

    def test_no_spread_division_or_absolute_price_dependence(self):
        q=quotes([[100,110,10,20],[100,110,13,17]])
        other=q.copy();other['ask']+=100
        np.testing.assert_allclose(compute_quote_imbalance(q).quote_ofi,compute_quote_imbalance(other).quote_ofi,equal_nan=True)

    def test_initial_and_segment_starts_are_nan(self):
        q=quotes([[100,110,10,20]]*4,segments=[0,0,1,1])
        out=compute_quote_imbalance(q)
        self.assertEqual(out.exclusion_code.tolist(),[1,0,2,0])
        self.assertEqual(out.quote_ofi.isna().tolist(),[True,False,True,False])

    def test_invalid_size_poison_both_adjacent_transitions_without_skipping(self):
        for size in (np.nan,np.inf,-1):
            q=quotes([[100,110,10,20],[100,110,size,20],[100,110,11,20],[100,110,12,20]])
            out=compute_quote_imbalance(q)
            self.assertEqual(out.exclusion_code.tolist(),[1,16,32,0])
            self.assertEqual(out.quote_ofi.isna().tolist(),[True,True,True,False])
            self.assertEqual(out.quote_ofi.iloc[3],1)

    def test_invalid_price_poison_both_adjacent_transitions(self):
        q=quotes([[100,110,10,20],[100,99,10,20],[100,110,10,20]])
        self.assertEqual(compute_quote_imbalance(q).exclusion_code.tolist(),[1,4,8])

    def test_zero_sizes_retained_without_division(self):
        out=compute_quote_imbalance(quotes([[100,110,0,0],[101,111,0,0]]))
        self.assertEqual(out.exclusion_code.tolist(),[1,0]);self.assertEqual(out.quote_ofi.iloc[1],0)

    def test_tied_and_decreasing_time_rejected(self):
        for ns in ([0,0],[1,0]):
            with self.assertRaises(ValueError):compute_quote_imbalance(quotes([[100,110,10,20]]*2,ns))

    def test_nonconsecutive_quote_ids_rejected(self):
        q=quotes([[100,110,10,20]]*2);q.loc[1,'quote_id']=3
        with self.assertRaises(ValueError):compute_quote_imbalance(q)

    def test_naive_time_rejected(self):
        q=quotes([[100,110,10,20]]*2);q['timestamp']=q.timestamp.dt.tz_localize(None)
        with self.assertRaises(ValueError):compute_quote_imbalance(q)

    def test_tick_chunk_invariance_and_prefix_causality(self):
        q=quotes([[100,110,10,20],[101,111,11,20],[102,112,12,20],[102,112,13,18]])
        full=compute_quote_imbalance(q)
        pieces=[compute_quote_imbalance(q.iloc[:2]),compute_quote_imbalance(q.iloc[2:],previous=q.iloc[1].to_dict())]
        pd.testing.assert_frame_equal(full,pd.concat(pieces,ignore_index=True))
        pd.testing.assert_frame_equal(full.iloc[:2],compute_quote_imbalance(q.iloc[:2]))

    def test_long_gap_kept_and_sunday_kept(self):
        q=quotes([[100,110,10,20],[101,111,13,17]],nanos=[0,61_000_000_000])
        out=compute_quote_imbalance(q);self.assertEqual(len(out),2)
        self.assertEqual(out.delta_sec.iloc[1],61);self.assertEqual(out.quote_ofi.iloc[1],33)

    def test_legacy_function_refuses_old_definition(self):
        with self.assertRaises(RuntimeError):compute_ofi(pd.DataFrame())


class QuoteBinContract(unittest.TestCase):
    def ticks(self):
        return compute_quote_imbalance(quotes([[100,110,10,20],[101,111,13,17],[102,112,11,19],[101,111,14,16]],
            nanos=[0,999_999_999,1_000_000_000,3_000_000_000]))

    def test_exact_second_belongs_to_new_bin_and_end_label(self):
        bins=aggregate_observed_bins(self.ticks(),support())
        self.assertEqual(bins.n_quotes.tolist(),[2,1,1])
        self.assertEqual(bins.bin_start.tolist(),[BASE,BASE+pd.Timedelta(seconds=1),BASE+pd.Timedelta(seconds=3)])
        self.assertTrue((bins.available_at==bins.bin_start+pd.Timedelta(seconds=1)).all())

    def test_partial_invalid_bin_not_silent_partial_sum(self):
        bins=aggregate_observed_bins(self.ticks(),support())
        self.assertEqual(bins.observed_subtotal.iloc[0],33)
        self.assertTrue(pd.isna(bins.quote_ofi.iloc[0]));self.assertEqual(bins.n_invalid_transitions.iloc[0],1)
        self.assertFalse(bins.usable_structural_only.iloc[0])

    def test_no_quote_bin_is_absent_not_zero(self):
        bins=aggregate_observed_bins(self.ticks(),support())
        self.assertNotIn(BASE+pd.Timedelta(seconds=2),bins.bin_start.tolist())

    def test_no_valid_transition_subtotal_is_nan(self):
        ticks=self.ticks().iloc[:1];bins=aggregate_observed_bins(ticks,support())
        self.assertTrue(pd.isna(bins.observed_subtotal.iloc[0]))

    def test_zero_net_valid_flow_is_zero_not_missing(self):
        q=quotes([[100,110,10,20]]*2,nanos=[0,1_000_000_000])
        bins=aggregate_observed_bins(compute_quote_imbalance(q),support())
        self.assertEqual(bins.quote_ofi.iloc[1],0);self.assertTrue(bins.usable_fresh_15s.iloc[1])

    def test_partial_clock_support_rejects_bin(self):
        bins=aggregate_observed_bins(self.ticks(),support(end=BASE+pd.Timedelta(seconds=1,nanoseconds=1)))
        self.assertFalse(bins.usable_structural_only.iloc[1])

    def test_long_incoming_gap_rejected_even_if_current_bin_supported(self):
        q=quotes([[100,110,10,20],[101,111,13,17]],nanos=[0,61_000_000_000])
        bins=aggregate_observed_bins(compute_quote_imbalance(q),support())
        self.assertTrue(bins.full_bin_fresh_60s.iloc[1]);self.assertFalse(bins.usable_fresh_60s.iloc[1])
        self.assertTrue(bins.usable_structural_only.iloc[1])

    def test_chunk_sizes_do_not_change_bins(self):
        ticks=self.ticks();expected=aggregate_observed_bins(ticks,support())
        for size in (1,2,3,4):
            stream=BinStream(support());parts=[]
            for i in range(0,len(ticks),size):
                p=stream.push(ticks.iloc[i:i+size]);
                if p is not None:parts.append(p)
            p=stream.finish()
            if p is not None:parts.append(p)
            pd.testing.assert_frame_equal(expected,pd.concat(parts,ignore_index=True))

    def test_unfinished_second_is_not_emitted(self):
        stream=BinStream(support());ticks=self.ticks()
        self.assertIsNone(stream.push(ticks.iloc[:1]))
        self.assertIsNone(stream.push(ticks.iloc[1:2]))
        result=stream.push(ticks.iloc[2:3]);self.assertEqual(len(result),1)
        self.assertEqual(result.available_at.iloc[0],BASE+pd.Timedelta(seconds=1))



class SerializedQuoteContract(unittest.TestCase):
    def setUp(self):
        import tempfile,contextlib,io
        from pathlib import Path
        from src.data_preparation import atomic_json,sha256
        from repair_quote_imbalance import build_artifacts
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/'run';self.run.mkdir();self.elig=self.root/'elig';self.elig.mkdir()
        q=quotes([[100,110,10,20],[101,111,13,17],[102,112,11,19],[101,111,14,16],
            [100,110,np.nan,15],[101,111,12,18],[102,112,11,17],[103,113,9,19]],
            nanos=[0,999_999_999,1_000_000_000,3_000_000_000,3_500_000_000,4_000_000_000,5_000_000_000,61_000_000_000],
            segments=[0,0,0,1,1,1,1,1])
        self.parent=self.root/'clean.parquet';q[['timestamp','bid','ask','bid_volume','ask_volume']].to_parquet(self.parent,index=False)
        pd.DataFrame({'segment_id':[0,1],'quote_start':[0,3],'quote_stop_exclusive':[3,8]}).to_parquet(self.elig/'segments.parquet',index=False)
        rows=[]
        for p in POLICIES:
            rows += [{'policy':p,'start':BASE,'end_exclusive':pd.Timestamp(BASE.value+2_000_000_000,tz='UTC')},
                {'policy':p,'start':pd.Timestamp(BASE.value+3_000_000_000,tz='UTC'),'end_exclusive':pd.Timestamp(BASE.value+62_000_000_000,tz='UTC')}]
        pd.DataFrame(rows).to_parquet(self.elig/'support_intervals.parquet',index=False)
        with contextlib.redirect_stdout(io.StringIO()):build_artifacts(self.parent,self.elig,self.run,batch_size=2)
        self.manifest={'parent_clean_path':str(self.parent),'eligibility_directory':str(self.elig),
            'inputs':{},'outputs':{p.name:sha256(p) for p in self.run.iterdir()}}
        atomic_json(self.run/'manifest.json',self.manifest)

    def mutate(self,name,operation):
        from src.data_preparation import atomic_json,sha256
        frame=pd.read_parquet(self.run/name);operation(frame);frame.to_parquet(self.run/name,index=False)
        self.manifest['outputs'][name]=sha256(self.run/name);atomic_json(self.run/'manifest.json',self.manifest)

    def test_all_serialized_contributions_bins_and_statistics_verified(self):
        from src.verify_quote_imbalance import verify_quote_run
        result=verify_quote_run(self.run)
        self.assertEqual(result['quote_rows_checked'],8);self.assertEqual(result['status'],'passed')

    def test_tick_corruption_detected_with_updated_hash(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_ticks.parquet',lambda x:x.__setitem__('quote_ofi',x.quote_ofi+1))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_bid_ask_components_cannot_be_swapped(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_ticks.parquet',lambda x:x.__setitem__('bid_pressure',x.ask_pressure))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_bin_sum_corruption_detected_with_updated_hash(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_1s.parquet',lambda x:x.__setitem__('quote_ofi',x.quote_ofi+1))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_early_availability_detected_with_updated_hash(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_1s.parquet',lambda x:x.__setitem__('available_at',x.bin_start))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_partial_sum_promoted_to_valid_signal_detected(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_1s.parquet',lambda x:x.__setitem__('quote_ofi',x.observed_subtotal))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_false_freshness_flag_detected(self):
        from src.verify_quote_imbalance import verify_quote_run
        self.mutate('quote_ofi_1s.parquet',lambda x:x.__setitem__('usable_fresh_15s',True))
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_count_corruption_detected_with_updated_hash(self):
        import json
        from src.data_preparation import atomic_json,sha256
        from src.verify_quote_imbalance import verify_quote_run
        p=self.run/'statistics.json';s=json.loads(p.read_text());s['ticks']['positive']+=1;atomic_json(p,s)
        self.manifest['outputs'][p.name]=sha256(p);atomic_json(self.run/'manifest.json',self.manifest)
        with self.assertRaises(AssertionError):verify_quote_run(self.run)

    def test_loader_requires_explicit_policy(self):
        from src.pipeline_state import iter_quote_imbalance
        with self.assertRaises(ValueError):next(iter_quote_imbalance(self.root,policy=None))

    def test_loader_rejects_mismatched_eligibility(self):
        from src.pipeline_state import iter_quote_imbalance
        from src.data_preparation import atomic_json
        atomic_json(self.root/'data/active_run.json',{'data_run_id':'data'})
        atomic_json(self.root/'data/active_eligibility.json',{'eligibility_run_id':'new'})
        atomic_json(self.root/'data/active_quote_imbalance.json',{'parent_data_run':'data','parent_eligibility_run':'old'})
        with self.assertRaises(RuntimeError):next(iter_quote_imbalance(self.root,policy='fresh_15s'))


if __name__=='__main__':unittest.main()

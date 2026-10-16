"""Economic-data contract examples and deliberate corruption tests (unittest)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from collections import Counter
from src.data_preparation import clean_quotes, load_source, write_panel, exact_median
from src.verify_data_stage import verify_run


def quotes(rows):
    frame=pd.DataFrame(rows,columns=['timestamp','bid','ask','bid_volume','ask_volume'])
    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,format='mixed')
    for c in ['bid','ask','bid_volume','ask_volume']:frame[c]=frame[c].astype(float)
    frame['source_file']='eurusd_2024_01.parquet'
    frame['source_row']=np.arange(len(frame),dtype=np.int64)
    return frame


class CleaningContract(unittest.TestCase):
    def test_size_only_update_survives(self):
        q=quotes([('2024-01-01',1,1.0002,3,4),('2024-01-01',1,1.0002,7,4)])
        c,r,_=clean_quotes(q)
        self.assertEqual(c.bid_volume.tolist(),[3,7]);self.assertTrue(c.timestamp_tied.all())
        self.assertEqual(r['duplicate_full_snapshot'],0)

    def test_full_duplicate_removed_only_once(self):
        row=('2024-01-01',1,1.0002,3,4)
        c,r,_=clean_quotes(quotes([row,row]))
        self.assertEqual(len(c),1);self.assertEqual(r['duplicate_full_snapshot'],1)
        self.assertFalse(c.timestamp_tied.any())

    def test_same_state_at_later_time_is_retained(self):
        c,_,_=clean_quotes(quotes([('2024-01-01',1,1.0002,3,4),('2024-01-01 00:00:01',1,1.0002,3,4)]))
        self.assertEqual(len(c),2);self.assertEqual(c.delta_sec.iloc[1],1)

    def test_price_filters_are_disjoint(self):
        q=quotes([('2024-01-01',1,np.inf,3,4),('2024-01-02',0,1,3,4),
                  ('2024-01-03',2,1,3,4),('2024-01-04',1,1,3,4),('2024-01-05',1,1.0002,3,4)])
        c,r,_=clean_quotes(q)
        self.assertEqual(len(c),1)
        self.assertEqual(r,{'nonfinite_price':1,'nonpositive_price':1,'crossed_quote':1,'zero_spread':1,'duplicate_full_snapshot':0})

    def test_invalid_sizes_flagged_prices_retained(self):
        q=quotes([('2024-01-01',1,1.0002,np.nan,4),('2024-01-02',1,1.0002,3,-1),
                  ('2024-01-03',1,1.0002,np.inf,4),('2024-01-04',1,1.0002,0,4)])
        c,_,_=clean_quotes(q)
        self.assertEqual(c.quote_size_valid.tolist(),[False,False,False,True])
        self.assertTrue(np.isnan(c.bid_volume.iloc[0]));self.assertEqual(c.ask_volume.iloc[1],-1)

    def test_ties_preserve_source_order_and_flag_all(self):
        q=quotes([('2024-01-01',1,1.0002,3,4),('2024-01-01',1.0001,1.0003,8,9)])
        c,_,_=clean_quotes(q.iloc[::-1])
        self.assertEqual(c.source_row.tolist(),[0,1]);self.assertTrue(c.timestamp_tied.all())
        self.assertEqual(c.delta_sec.iloc[1],0)

    def test_cross_month_duration(self):
        c,_,_=clean_quotes(quotes([('2024-02-01 00:00:00.250',1,1.0002,3,4)]),pd.Timestamp('2024-01-31 23:59:59.750',tz='UTC'))
        self.assertEqual(c.delta_sec.iloc[0],.5)

    def test_cross_month_millisecond_rounding_matches_within_month(self):
        boundary=pd.Timestamp('2024-01-31 23:59:59.128',tz='UTC')
        q=quotes([('2024-02-01 00:00:01',1,1.0002,3,4)])
        c,_,_=clean_quotes(q,boundary)
        self.assertEqual(c.delta_sec.iloc[0],1.872)

    def test_cross_month_nanosecond_is_not_truncated(self):
        boundary=pd.Timestamp('2024-01-31 23:59:59.999999999',tz='UTC')
        c,_,_=clean_quotes(quotes([('2024-02-01',1,1.0002,3,4)]),boundary)
        self.assertEqual(c.delta_sec.iloc[0],1e-9)

    def test_long_gap_and_sunday_are_retained(self):
        c,_,_=clean_quotes(quotes([('2024-01-07 22:00:00',1,1.0002,3,4)]),pd.Timestamp('2024-01-05 21:00:00',tz='UTC'))
        self.assertEqual(len(c),1);self.assertEqual(c.delta_sec.iloc[0],176400)

    def test_first_duration_undefined(self):
        c,_,_=clean_quotes(quotes([('2024-01-01',1,1.0002,3,4)]))
        self.assertTrue(pd.isna(c.delta_sec.iloc[0]))

    def test_naive_timestamp_rejected(self):
        q=quotes([('2024-01-01',1,1.0002,3,4)]);q.timestamp=q.timestamp.dt.tz_localize(None)
        with self.assertRaises(ValueError):clean_quotes(q)

    def test_overlapping_partitions_rejected(self):
        q=quotes([('2024-01-01',1,1.0002,3,4)])
        with self.assertRaises(ValueError):clean_quotes(q,q.timestamp.iloc[0])

    def test_empty_input_preserves_boundary(self):
        q=quotes([('2024-01-01',1,1.0002,3,4)]).iloc[:0]
        boundary=pd.Timestamp('2023-12-31',tz='UTC');c,r,last=clean_quotes(q,boundary)
        self.assertTrue(c.empty);self.assertEqual(last,boundary);self.assertEqual(sum(r.values()),0)

    def test_even_and_odd_exact_medians(self):
        self.assertEqual(exact_median(Counter({1:1,3:1})),2)
        self.assertEqual(exact_median(Counter({1:2,8:1})),1)
        self.assertIsNone(exact_median(Counter()))


class SerializedVerification(unittest.TestCase):
    def fixture(self,root,all_invalid_first=False,nanosecond_boundary=False):
        src=root/'sources';src.mkdir()
        january_time='2024-01-31 23:59:59.999999999' if nanosecond_boundary else '2024-01-31 23:59:59.750'
        february_time='2024-02-01 00:00:00' if nanosecond_boundary else '2024-02-01 00:00:00.250'
        january=quotes([(january_time,1,1 if all_invalid_first else 1.0002,3,4)])
        february=quotes([(february_time,1,1.0002,5,6)])
        for month,q in [('01',january),('02',february)]:
            q.drop(columns=['source_file','source_row']).to_parquet(src/f'eurusd_2024_{month}.parquet',index=False)
        return write_panel(root,src,'fixture','2024-01','2024-02')

    def test_full_serialized_source_verification(self):
        with TemporaryDirectory() as d:
            run=self.fixture(Path(d));v=verify_run(run)
            self.assertEqual(v['statistics']['clean_rows'],2)
            self.assertEqual(v['statistics']['month_boundaries_checked'],1)

    def test_empty_first_clean_month_supported(self):
        with TemporaryDirectory() as d:
            run=self.fixture(Path(d),True);v=verify_run(run)
            self.assertEqual(v['statistics']['clean_rows'],1)
            self.assertEqual(v['statistics']['missing_duration_rows'],1)

    def test_serialized_nanosecond_boundary_is_verified(self):
        with TemporaryDirectory() as d:
            run=self.fixture(Path(d),nanosecond_boundary=True);v=verify_run(run)
            self.assertEqual(v['statistics']['median_delta_sec'],1e-9)

    def test_corrupted_size_is_detected(self):
        with TemporaryDirectory() as d:
            run=self.fixture(Path(d));p=run/'clean_quotes.parquet'
            c=pd.read_parquet(p);c.loc[0,'bid_volume']=999;c.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_run(run)

    def test_corrupted_boundary_duration_is_detected(self):
        with TemporaryDirectory() as d:
            run=self.fixture(Path(d));p=run/'clean_quotes.parquet'
            c=pd.read_parquet(p);c.loc[1,'delta_sec']=0;c.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_run(run)

    def test_source_mutation_is_detected(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);p=root/'sources/eurusd_2024_01.parquet'
            c=pd.read_parquet(p);c.loc[0,'ask_volume']=900;c.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_run(run)

    def test_run_id_cannot_overwrite_existing_run(self):
        with TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaises(FileExistsError):write_panel(root,root/'sources','fixture','2024-01','2024-02')

    def test_named_month_mismatch_fails(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'eurusd_2024_02.parquet'
            quotes([('2024-01-01',1,1.0002,3,4)]).to_parquet(p,index=False)
            with self.assertRaises(ValueError):load_source(p)

    def test_missing_month_fails_before_run_created(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(FileNotFoundError):write_panel(root,root,'missing','2024-01','2024-02')
            self.assertFalse((root/'data/runs/missing').exists())


if __name__=='__main__':unittest.main()

"""Calendar, wire-format, failure classification, and coverage counterexamples."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import lzma
import struct
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import requests

from src.acquisition import nominal_weekly_open, hour_url, decode_hour, fetch_hour, audit_months
from src.data_preparation import atomic_json, sha256, promote_statistics
from src.verify_acquisition import reference_decode, verify_recovery, verify_coverage_audit


def payload(rows):
    return lzma.compress(b''.join(struct.pack('>IIIff',*r) for r in rows),format=lzma.FORMAT_ALONE)


class CalendarContract(unittest.TestCase):
    def test_winter_friday_last_hour_is_open(self):
        self.assertEqual(nominal_weekly_open(pd.to_datetime(['2024-01-05 21:00Z','2024-01-05 22:00Z'])).tolist(),[True,False])

    def test_summer_friday_close(self):
        self.assertEqual(nominal_weekly_open(pd.to_datetime(['2024-07-05 20:00Z','2024-07-05 21:00Z'])).tolist(),[True,False])

    def test_us_spring_transition_not_european_transition(self):
        self.assertEqual(nominal_weekly_open(pd.to_datetime(['2024-03-03 21:00Z','2024-03-03 22:00Z','2024-03-10 21:00Z'])).tolist(),[False,True,True])

    def test_us_fall_transition(self):
        self.assertEqual(nominal_weekly_open(pd.to_datetime(['2024-10-27 21:00Z','2024-11-03 21:00Z','2024-11-03 22:00Z'])).tolist(),[True,False,True])

    def test_holiday_not_automatically_proven_closed(self):
        self.assertTrue(nominal_weekly_open(pd.to_datetime(['2024-12-25 12:00Z']))[0])

    def test_naive_calendar_rejected(self):
        with self.assertRaises(ValueError):nominal_weekly_open(pd.to_datetime(['2024-01-01']))

    def test_url_zero_based_month_and_hour(self):
        self.assertTrue(hour_url('2024-01-05 21:00Z').endswith('/2024/00/05/21h_ticks.bi5'))

    def test_non_hour_boundary_rejected(self):
        with self.assertRaises(ValueError):hour_url('2024-01-05 21:01Z')


class BinaryContract(unittest.TestCase):
    def test_known_bytes_time_scaling_and_size(self):
        c=decode_hour(payload([(82,109395,109393,1.8,1.89)]),'2024-01-05 21:00Z')
        self.assertEqual(c.timestamp.iloc[0],pd.Timestamp('2024-01-05 21:00:00.082Z'))
        self.assertEqual(c.ask.iloc[0],1.09395);self.assertEqual(c.bid.iloc[0],1.09393)
        self.assertEqual(c.ask_volume.iloc[0],float(np.float32(1.8)))

    def test_size_only_same_timestamp_update_and_duplicate_are_retained(self):
        c=decode_hour(payload([(0,100002,100000,3,4),(0,100002,100000,7,4),(0,100002,100000,7,4)]),'2024-01-01 00:00Z')
        self.assertEqual(c.ask_volume.tolist(),[3,7,7]);self.assertEqual(c.vendor_record.tolist(),[0,1,2])

    def test_invalid_prices_and_sizes_remain_auditable(self):
        c=decode_hour(payload([(0,0,0,-1,float('nan'))]),'2024-01-01 00:00Z')
        self.assertEqual(len(c),1);self.assertEqual(c.ask.iloc[0],0);self.assertTrue(pd.isna(c.bid_volume.iloc[0]))

    def test_invalid_alignment_rejected(self):
        with self.assertRaises(ValueError):decode_hour(lzma.compress(b'a'),'2024-01-01 00:00Z')

    def test_offset_at_next_hour_rejected(self):
        with self.assertRaises(ValueError):decode_hour(payload([(3600000,1,1,1,1)]),'2024-01-01 00:00Z')

    def test_decreasing_vendor_order_rejected(self):
        with self.assertRaises(ValueError):decode_hour(payload([(2,1,1,1,1),(1,1,1,1,1)]),'2024-01-01 00:00Z')

    def test_empty_compressed_stream(self):
        self.assertTrue(decode_hour(payload([]),'2024-01-01 00:00Z').empty)


class RequestContract(unittest.TestCase):
    def get(self,session,cache,max_attempts=3):
        return fetch_hour('2024-01-05 21:00Z',cache,session=session,max_attempts=max_attempts,sleep=lambda _:None)

    def test_empty_404_access_denied_and_error_are_distinct(self):
        for code,body,status in [(200,b'','empty_200'),(200,payload([]),'empty_decoded'),(404,b'not found','not_found_404'),(403,b'denied','access_denied'),(429,b'limit','rate_limited'),(503,b'error','http_error'),(200,b'corrupt','invalid_payload')]:
            with self.subTest(code=code,status=status),TemporaryDirectory() as d:
                session=Mock();session.get.return_value=Mock(status_code=code,content=body)
                record=self.get(session,Path(d),1)
                self.assertEqual(record['status'],status)
                self.assertNotIn('closed',record['status'])

    def test_transient_error_then_success_preserves_attempts(self):
        with TemporaryDirectory() as d:
            session=Mock();session.get.side_effect=[requests.Timeout('timeout'),Mock(status_code=200,content=payload([(0,2,1,3,4)]))]
            r=self.get(session,Path(d));self.assertEqual(r['status'],'ok');self.assertEqual(len(r['attempts']),2)

    def test_winter_friday_is_actually_requested(self):
        with TemporaryDirectory() as d:
            session=Mock();session.get.return_value=Mock(status_code=200,content=payload([(0,2,1,3,4)]))
            self.get(session,Path(d));self.assertEqual(session.get.call_count,1)

    def test_saved_response_reused_but_corruption_rejected(self):
        with TemporaryDirectory() as d:
            cache=Path(d);session=Mock();session.get.return_value=Mock(status_code=200,content=payload([(0,2,1,3,4)]))
            r=self.get(session,cache);self.get(session,cache);self.assertEqual(session.get.call_count,1)
            (cache/r['response_file']).write_bytes(b'bad')
            with self.assertRaises(ValueError):self.get(session,cache)

    def test_failed_request_can_be_retried_without_erasing_history(self):
        with TemporaryDirectory() as d:
            session=Mock();session.get.side_effect=requests.Timeout('timeout');self.get(session,Path(d),1)
            session.get.side_effect=None;session.get.return_value=Mock(status_code=200,content=payload([(0,2,1,3,4)]))
            r=self.get(session,Path(d));self.assertEqual(len(r['attempts']),2);self.assertEqual(r['status'],'ok')

    def test_rate_limit_stops_without_retry_and_preserves_guidance(self):
        with TemporaryDirectory() as d:
            session=Mock();session.get.return_value=Mock(status_code=429,content=b'rate limited',headers={'Retry-After':'120'})
            r=self.get(session,Path(d));self.assertEqual(session.get.call_count,1);self.assertEqual(r['retry_after'],'120')


class CoverageContract(unittest.TestCase):
    def test_entirely_empty_month_has_explicit_absent_hours(self):
        with TemporaryDirectory() as d:
            src=Path(d)
            pd.DataFrame({'timestamp':pd.Series([],dtype='datetime64[ns, UTC]')}).to_parquet(src/'eurusd_2024_01.parquet',index=False)
            ledger,gaps,stats,_=audit_months(src,'2024-01-01 00:00Z','2024-01-01 03:00Z')
            self.assertEqual(stats['absent_nominal_open_hours'],3);self.assertTrue(gaps.empty)

    def test_requested_bounds_absent_hour_and_partial_hour(self):
        with TemporaryDirectory() as d:
            src=Path(d)
            pd.DataFrame({'timestamp':pd.to_datetime(['2024-01-01 01:02Z','2024-01-01 01:05Z'])}).to_parquet(src/'eurusd_2024_01.parquet',index=False)
            ledger,gaps,stats,_=audit_months(src,'2024-01-01 00:00Z','2024-01-01 03:00Z')
            self.assertEqual(stats['absent_nominal_open_hours'],2)
            self.assertEqual(ledger.rows.tolist(),[0,2,0]);self.assertEqual(len(gaps),1)
            self.assertEqual(ledger.first_quote_delay_sec.iloc[1],120)
            self.assertEqual(ledger.last_quote_to_hour_end_sec.iloc[1],3300)
            self.assertEqual(ledger.max_internal_gap_sec.iloc[1],180)

    def test_month_boundary_gap_is_not_lost(self):
        with TemporaryDirectory() as d:
            src=Path(d)
            for month,ts in [('01','2024-01-31 23:59:00Z'),('02','2024-02-01 00:01:00Z')]:
                pd.DataFrame({'timestamp':pd.to_datetime([ts])}).to_parquet(src/f'eurusd_2024_{month}.parquet',index=False)
            _,gaps,_,_=audit_months(src,'2024-01-31 23:00Z','2024-02-01 01:00Z')
            self.assertEqual(gaps.delta_sec.tolist(),[120])

    def test_observation_outside_weekly_schedule_is_retained_and_flagged(self):
        with TemporaryDirectory() as d:
            src=Path(d)
            pd.DataFrame({'timestamp':pd.to_datetime(['2024-01-06 12:01Z'])}).to_parquet(src/'eurusd_2024_01.parquet',index=False)
            ledger,_,stats,_=audit_months(src,'2024-01-06 12:00Z','2024-01-06 13:00Z')
            self.assertEqual(stats['observed_weekly_closed_hours'],1);self.assertEqual(ledger.rows.iloc[0],1)


class RecoveryVerification(unittest.TestCase):
    def fixture(self,root):
        original=root/'original';original.mkdir();acq=root/'acquisition';acq.mkdir();(acq/'monthly').mkdir()
        first='2024-01-01T00:00:00+00:00';second='2024-01-01T01:00:00+00:00'
        bodies=[payload([(82,109395,109393,1.8,1.89)]),payload([(7,109400,109398,2,3),(7,109400,109398,4,3)])]
        old=reference_decode(bodies[0],first);old.to_parquet(original/'eurusd_2024_01.parquet',index=False)
        ledger,gaps,_,fingerprints=audit_months(original,first,'2024-01-01T02:00:00+00:00');ledger.to_parquet(acq/'coverage_before.parquet',index=False)
        gaps.to_parquet(acq/'gaps_before.parquet',index=False)
        results=[];targets=[]
        for hour,purpose,body in [(first,'decoder_control_existing_hour',bodies[0]),(second,'absent_nominal_open_hour',bodies[1])]:
            session=Mock();session.get.return_value=Mock(status_code=200,content=body)
            receipt=fetch_hour(hour,acq/'receipts',session=session,sleep=lambda _:None)
            results.append({**receipt,'purpose':purpose});targets.append({'hour':hour,'purpose':purpose})
        atomic_json(acq/'acquisition_results.json',{'results':results});atomic_json(acq/'targets.json',{'targets':targets})
        atomic_json(acq/'manifest.json',{'start_inclusive':first,'end_exclusive':'2024-01-01T02:00:00+00:00','input_files':fingerprints,'source_dir':str(original)})
        combined=pd.concat([old,decode_hour(bodies[1],second).drop(columns='vendor_record')],ignore_index=True)
        combined.to_parquet(acq/'monthly/eurusd_2024_01.parquet',index=False)
        return acq

    def test_independent_decoder_and_serialized_merge(self):
        with TemporaryDirectory() as d:
            r=verify_recovery(self.fixture(Path(d)))
            self.assertEqual(r['original_rows_preserved_exactly'],1);self.assertEqual(r['recovered_raw_rows'],2)
            self.assertTrue(r['control_comparisons'][0]['exact_full_record_match'])

    def test_independent_hourly_audit_verification(self):
        with TemporaryDirectory() as d:
            r=verify_coverage_audit(self.fixture(Path(d)))
            self.assertEqual(r['raw_timestamps_checked'],1);self.assertEqual(r['hourly_rows_checked'],2)

    def test_corrupted_hour_edge_diagnostic_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));p=acq/'coverage_before.parquet';ledger=pd.read_parquet(p)
            ledger.loc[0,'first_quote_delay_sec']=0;ledger.to_parquet(p,index=False)
            with self.assertRaisesRegex(AssertionError,'first_quote_delay'):verify_coverage_audit(acq)

    def test_lost_same_price_size_update_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));p=acq/'monthly/eurusd_2024_01.parquet'
            frame=pd.read_parquet(p).iloc[:-1];frame.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_recovery(acq)

    def test_original_observation_mutation_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));p=acq/'monthly/eurusd_2024_01.parquet'
            frame=pd.read_parquet(p);frame.loc[0,'ask_volume']=999;frame.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_recovery(acq)

    def test_missing_acquisition_outcome_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));p=acq/'acquisition_results.json';r=json.loads(p.read_text());r['results'].pop();atomic_json(p,r)
            with self.assertRaises(AssertionError):verify_recovery(acq)

    def test_wire_response_corruption_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));next((acq/'receipts').glob('*.response')).write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):verify_recovery(acq)

    def test_weekly_calendar_corruption_detected(self):
        with TemporaryDirectory() as d:
            acq=self.fixture(Path(d));p=acq/'coverage_before.parquet';ledger=pd.read_parquet(p);ledger.loc[0,'nominal_weekly_open']=False;ledger.to_parquet(p,index=False)
            with self.assertRaises(AssertionError):verify_recovery(acq)

    def test_stage_one_cannot_replace_a_later_data_stage(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=root/'data/runs/fake';run.mkdir(parents=True)
            atomic_json(run/'manifest.json',{'status':'verified'})
            atomic_json(root/'reports/pipeline_master_statistics.json',{'current_data_stage':{'stage':'02a_whole_hour_coverage'}})
            with self.assertRaisesRegex(ValueError,'cannot replace'):promote_statistics(root,run,{'status':'passed'}, {})


if __name__=='__main__':unittest.main()

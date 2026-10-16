from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_preparation import atomic_json,sha256
from src.deseasonal import (CalibrationSpec,POLICY_NAMES,CHANNELS,fit_day,daily_moments,normalize_day,
    normalize_policy,fit_table,moments_table,table_to_fits,compute_diurnal_pattern,deseasonalize_series)
from src.event_detection import build_event_detection_pipeline
from src.verify_seasonal_calibration import reference_daily,reference_fit,reference_output,check_frame,verify_calibration_run

BASE=pd.Timestamp('2024-01-01T00:00:00Z');S=1_000_000_000


def frame(day=0,seconds=(0,1,2,3),counts=(1,3,0,0),flows=(2.,-6.,np.nan,np.nan),codes=(2,2,1,1)):
    t=BASE.value+day*86400*S+np.asarray(seconds)*S
    out=pd.DataFrame({'bin_start':pd.to_datetime(t,utc=True),'available_at':pd.to_datetime(t+S,utc=True),
                      'recorded_update_count':counts,'observed_quote_ofi':flows})
    for p in POLICY_NAMES:
        out['support_id_'+p]=np.where(np.asarray(codes)>0,0,-1).astype('int32')
        out['flow_status_'+p]=np.array(codes,dtype='uint8')
    return out


def history(n=15):
    h=np.zeros((n,4,2,96,3))
    for d in range(n):h[d]=daily_moments(frame(day=d))
    return h


class PastDayCalibrationContract(unittest.TestCase):
    def setUp(self):self.spec=CalibrationSpec(7,1,2)

    def test_hand_calculated_cell_mean_sd_and_global_factor(self):
        h=np.zeros((15,4,2,96,3));h[0,:,:,0]=[2,2,2];h[7,:,:,0]=[2,4,2]
        h[0,:,:,1]=[2,10,2];h[7,:,:,1]=[2,8,2]
        f=fit_day(h,14,CalibrationSpec(14,2,2))
        self.assertEqual(f['n'][0,0,0],4);self.assertEqual(f['mean'][0,0,0],3)
        self.assertAlmostEqual(f['sample_std'][0,0,0],np.sqrt(8/3))
        self.assertEqual(f['global_mean'][0,0,0],6);self.assertEqual(f['phi'][0,0,0],.5)

    def test_current_and_future_values_cannot_change_any_parameter(self):
        h=history();a=fit_day(h,7,self.spec);h[7:]=np.random.default_rng(6).uniform(1,10000,h[7:].shape)
        b=fit_day(h,7,self.spec)
        for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True)

    def test_expired_old_day_cannot_change_current_fit(self):
        h=history(22);a=fit_day(h,21,self.spec);h[:14]*=1000;b=fit_day(h,21,self.spec)
        for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True)

    def test_lower_window_boundary_included(self):
        h=history();a=fit_day(h,7,self.spec);h[0,:,:,0,1]*=2;b=fit_day(h,7,self.spec)
        self.assertNotEqual(a['mean'][0,0,0],b['mean'][0,0,0])

    def test_wrong_weekday_does_not_enter_cell_moments(self):
        h=history();a=fit_day(h,7,self.spec);h[1,:,:,0,1]*=100;b=fit_day(h,7,self.spec)
        np.testing.assert_allclose(a['mean'],b['mean'],equal_nan=True)
        np.testing.assert_allclose(a['sample_std'],b['sample_std'],equal_nan=True)
        self.assertNotEqual(a['global_mean'][0,0,0],b['global_mean'][0,0,0])

    def test_full_window_warmup_not_backfilled(self):
        self.assertTrue((fit_day(history(),6,self.spec)['status']==0).all())
        self.assertEqual(fit_day(history(),7,self.spec)['status'][0,0,0],5)

    def test_distinct_dates_not_just_many_ticks_required(self):
        h=history(15);h[7]=0
        f=fit_day(h,14,CalibrationSpec(14,2,2));self.assertEqual(f['status'][0,0,0],1)

    def test_minimum_seconds_exact_boundary(self):
        h=history();self.assertEqual(fit_day(h,7,CalibrationSpec(7,1,4))['status'][0,0,0],5)
        self.assertEqual(fit_day(h,7,CalibrationSpec(7,1,5))['status'][0,0,0],2)

    def test_all_zero_history_is_unavailable_not_denominator_one(self):
        h=history();h[...,1:]=0
        self.assertEqual(fit_day(h,7,self.spec)['status'][0,0,0],3)
        self.assertTrue(np.isnan(fit_day(h,7,self.spec)['phi'][0,0,0]))

    def test_constant_positive_history_has_unavailable_scale(self):
        h=history();h[:,:,:,0,1]=2;h[...,2]=0
        f=fit_day(h,7,self.spec);self.assertEqual(f['status'][0,0,0],4)
        self.assertTrue(normalize_day(frame(day=7),f).iloc[:,2:].isna().all().all())

    def test_supported_zero_enters_calibration_denominator(self):
        d=daily_moments(frame())
        self.assertEqual(d[0,0,0,0],4);self.assertEqual(d[0,0,0,1],1)
        self.assertEqual(d[0,1,0,1],2)

    def test_unsupported_audit_zeros_are_not_observations(self):
        f=frame(codes=(2,2,0,0));d=daily_moments(f)
        self.assertEqual(d[0,0,0,0],2);self.assertEqual(d[0,0,0,1],2)

    def test_invalid_flow_keeps_count_channel_independent(self):
        f=frame(counts=(1,3,2,0),flows=(2.,-6.,np.nan,np.nan),codes=(2,2,3,1));d=daily_moments(f)
        self.assertEqual(d[0,0,0,0],4);self.assertEqual(d[0,1,0,0],3)

    def test_quiet_adjusted_zero_has_negative_z(self):
        f=fit_day(history(),7,self.spec);out=normalize_policy(frame(day=7),f,policy='structural_only')
        self.assertEqual(out.adjusted_tick_count.iloc[2],0);self.assertEqual(out.adjusted_quote_ofi.iloc[2],0)
        self.assertLess(out.z_tick_count.iloc[2],0)

    def test_missing_input_never_becomes_calibrated_zero(self):
        f=fit_day(history(),7,self.spec);out=normalize_policy(frame(day=7,codes=(2,2,0,0)),f,policy='structural_only')
        self.assertTrue(out.z_tick_count.iloc[2:].isna().all())
        self.assertTrue(out.adjusted_quote_ofi.iloc[2:].isna().all())

    def test_imbalance_direction_preserved_by_positive_factor(self):
        f=fit_day(history(),7,self.spec);out=normalize_policy(frame(day=7),f,policy='structural_only')
        self.assertGreater(out.adjusted_quote_ofi.iloc[0],0);self.assertLess(out.adjusted_quote_ofi.iloc[1],0)

    def test_quote_size_units_do_not_change_z_or_phi(self):
        h=history();f=fit_day(h,7,self.spec);scaled=h.copy();scaled[:,:,1,:,1]*=1e6;scaled[:,:,1,:,2]*=1e12
        fs=fit_day(scaled,7,self.spec);raw=frame(day=7);rscaled=raw.copy();rscaled['observed_quote_ofi']*=1e6
        a=normalize_day(raw,f);b=normalize_day(rscaled,fs)
        for p in POLICY_NAMES:np.testing.assert_allclose(a['z_abs_quote_ofi_'+p],b['z_abs_quote_ofi_'+p],equal_nan=True)
        np.testing.assert_allclose(f['phi'],fs['phi'],equal_nan=True)

    def test_later_same_day_quote_does_not_change_earlier_scores(self):
        f=fit_day(history(),7,self.spec);raw=frame(day=7);a=normalize_day(raw,f)
        raw.loc[3,'recorded_update_count']=1000;raw.loc[3,'observed_quote_ofi']=1e6
        for p in POLICY_NAMES:raw.loc[3,'flow_status_'+p]=2
        b=normalize_day(raw,f);pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])

    def test_15_minute_boundary_uses_next_slot(self):
        h=history();h[:,:,:,1]=h[:,:,:,0];h[:,:,:,1,1]*=2
        fit=fit_day(h,7,self.spec)
        x=frame(day=7,seconds=(899,900),counts=(1,1),flows=(2.,2.),codes=(2,2))
        out=normalize_day(x,fit)
        self.assertNotEqual(out.z_tick_count_structural_only.iloc[0],out.z_tick_count_structural_only.iloc[1])

    def test_available_at_not_backdated(self):
        x=frame(day=7);out=normalize_day(x,fit_day(history(),7,self.spec))
        pd.testing.assert_series_equal(x.available_at,out.available_at)
        x.loc[0,'available_at']-=pd.Timedelta(nanoseconds=1)
        with self.assertRaises(ValueError):daily_moments(x)

    def test_midnight_observation_belongs_to_new_day(self):
        x=frame(seconds=(86399,86400),counts=(1,3),flows=(2.,6.),codes=(2,2))
        with self.assertRaises(ValueError):daily_moments(x)

    def test_calendar_cells_use_utc_even_if_display_timezone_differs(self):
        x=frame(seconds=(0,1,86399),counts=(1,3,0),flows=(2.,6.,np.nan),codes=(2,2,1))
        expected=daily_moments(x)
        for name in ['bin_start','available_at']:x[name]=x[name].dt.tz_convert('America/New_York')
        np.testing.assert_allclose(daily_moments(x),expected)

    def test_naive_availability_rejected(self):
        x=frame();x['available_at']=x.available_at.dt.tz_localize(None)
        with self.assertRaises(ValueError):daily_moments(x)

    def test_future_extended_history_has_identical_prefix_parameters(self):
        h=history(22)
        for i in range(8,15):
            a=fit_day(h,i,self.spec);b=fit_day(h[:i],i,self.spec)
            for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True)

    def test_no_pooled_fallback_for_unobserved_cell(self):
        f=fit_day(history(),7,self.spec);self.assertEqual(f['status'][0,0,1],1)
        self.assertTrue(np.isnan(f['phi'][0,0,1]))

    def test_policy_specific_calibration_does_not_mix_support(self):
        h=history();h[:,1,:,:,0]=0;h[:,1,:,:,1:]=0;f=fit_day(h,7,self.spec)
        self.assertEqual(f['status'][0,0,0],5);self.assertEqual(f['status'][1,0,0],1)

    def test_independent_reference_for_every_parameter(self):
        h=history(22)
        for i in [0,6,7,14,21]:
            a=fit_day(h,i,self.spec);b=reference_fit(list(h),i,self.spec)
            for k in a:np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12,equal_nan=True)

    def test_stable_variance_for_large_offset(self):
        h=history();h[:,:,:,0]=[4,1e12,4]
        f=fit_day(h,7,self.spec);self.assertAlmostEqual(f['sample_std'][0,0,0],np.sqrt(4/3))
        np.testing.assert_allclose(f['sample_std'],reference_fit(list(h),7,self.spec)['sample_std'],equal_nan=True)

    def test_no_normality_or_clipping_imposed(self):
        fit=fit_day(history(),7,self.spec);x=frame(day=7,seconds=(0,),counts=(10000,),flows=(10000.,),codes=(2,))
        self.assertGreater(normalize_day(x,fit).z_tick_count_structural_only.iloc[0],1000)

    def test_parameter_round_trip_and_missing_cell_rejection(self):
        f=fit_day(history(),7,self.spec);t=fit_table(BASE+pd.Timedelta(days=7),f,self.spec)
        restored=table_to_fits(t)[BASE.value+7*86400*S]
        for k in f:np.testing.assert_allclose(f[k],restored[k],equal_nan=True)
        with self.assertRaises(ValueError):table_to_fits(t.iloc[:-1])

    def test_invalid_reference_settings_rejected(self):
        for args in [(6,1,2),(7,2,2),(7,1,1),(7.,1,2)]:
            with self.assertRaises(ValueError):CalibrationSpec(*args)

    def test_legacy_future_fitting_paths_refuse_execution(self):
        for fn in [compute_diurnal_pattern,deseasonalize_series]:
            with self.assertRaises(RuntimeError):fn(pd.Series([1.,2.]))
        with self.assertRaises(RuntimeError):build_event_detection_pipeline(None,None,None)


class CalibrationArtifactContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name);cls.spec=CalibrationSpec(7,1,2)
        path=cls.path;writer=None
        try:
            for d in range(8):
                sec=np.arange(86400);count=np.where(sec%900<4,(sec%4)+1,0)
                flow=np.where(count>0,count*2.*np.where(sec%2,1,-1),np.nan)
                code=np.where(count>0,2,1)
                f=frame(d,sec,count,flow,code)
                # The synthetic feed has a fully unsupported Sunday, never silent-zero filling.
                if d==6:
                    for p in POLICY_NAMES:f['flow_status_'+p]=np.uint8(0);f['support_id_'+p]=np.int32(-1)
                table=pa.Table.from_pandas(f,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(path/'clock.parquet',table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:writer.close()
        from repair_seasonal_calibration import build_artifacts
        cls.statistics=build_artifacts(path/'clock.parquet',path,cls.spec)
        files=['standardized_clock_1s.parquet','daily_sufficient_statistics.parquet','calibration_parameters.parquet',
               'monthly_statistics.parquet','weekday_slot_diagnostics.parquet','statistics.json']
        cls.manifest={'configuration':asdict(cls.spec),'inputs':{'clock':{'path':str(path/'clock.parquet'),'sha256':sha256(path/'clock.parquet')}},
                      'outputs':{f:sha256(path/f) for f in files}}
        atomic_json(path/'manifest.json',cls.manifest)
        cls.original={f:(path/f).read_bytes() for f in files}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full_serialized_independent_verification(self):
        result=verify_calibration_run(self.path)
        self.assertEqual(result['clock_rows_checked'],8*86400)
        self.assertEqual(self.statistics['channels']['structural_only']['tick_count']['counts']['calibrated_seconds'],86400)

    def corrupt(self,file,change):
        path=self.path;manifest=json.loads(json.dumps(self.manifest))
        try:
            if file.endswith('.json'):
                value=json.loads((path/file).read_text());change(value);atomic_json(path/file,value)
            else:
                value=pd.read_parquet(path/file);change(value);value.to_parquet(path/file,index=False)
            manifest['outputs'][file]=sha256(path/file);atomic_json(path/'manifest.json',manifest)
            with self.assertRaises(AssertionError):verify_calibration_run(path)
        finally:
            (path/file).write_bytes(self.original[file]);atomic_json(path/'manifest.json',self.manifest)

    def test_normalized_value_corruption_detected_after_hash_refresh(self):
        self.corrupt('standardized_clock_1s.parquet',lambda x:x.__setitem__('z_tick_count_structural_only',np.zeros(len(x))))

    def test_parameter_cutoff_corruption_detected_after_hash_refresh(self):
        self.corrupt('calibration_parameters.parquet',lambda x:x.__setitem__('training_end_exclusive',x.training_end_exclusive+pd.Timedelta(seconds=1)))

    def test_daily_mean_corruption_detected_after_hash_refresh(self):
        self.corrupt('daily_sufficient_statistics.parquet',lambda x:x.__setitem__('mean',x['mean']+1))

    def test_aggregate_statistic_corruption_detected_after_hash_refresh(self):
        self.corrupt('statistics.json',lambda x:x.__setitem__('clock_rows',1))

    def test_numeric_parameter_corruption_detected_after_hash_refresh(self):
        self.corrupt('calibration_parameters.parquet',lambda x:x.__setitem__('mean',x['mean']+1))

    def test_loader_checks_lineage_and_retains_full_clock(self):
        import shutil
        from src.pipeline_state import iter_calibrated_clock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'data/signals/calibration';run.mkdir(parents=True)
            for file in ['calibration_parameters.parquet','standardized_clock_1s.parquet']:
                shutil.copy2(self.path/file,run/file)
            parents={'parent_data_run':'data','parent_eligibility_run':'elig','parent_signal_run':'signal','parent_clock_run':'clock'}
            m={**self.manifest,**parents,'status':'verified_past_day_reference_calibration'}
            atomic_json(run/'manifest.json',m);atomic_json(run/'verification.json',{'status':'passed'})
            atomic_json(root/'data/active_calibration.json',{'calibration_run_id':'calibration',**parents,'configuration':asdict(self.spec)})
            for file,key,value in [('active_run.json','data_run_id','data'),('active_eligibility.json','eligibility_run_id','elig'),
                ('active_quote_imbalance.json','signal_run_id','signal'),('active_signal_clock.json','clock_run_id','clock')]:
                atomic_json(root/'data'/file,{key:value})
            frames=list(iter_calibrated_clock(root,policy='structural_only'))
            self.assertEqual(sum(map(len,frames)),8*86400)
            self.assertTrue(frames[0].z_tick_count.isna().all())
            self.assertTrue(frames[-1].z_tick_count.notna().all())
            self.assertTrue((frames[-1].calibration_cutoff<=frames[-1].bin_start).all())
            atomic_json(root/'data/active_signal_clock.json',{'clock_run_id':'different'})
            with self.assertRaises(RuntimeError):next(iter_calibrated_clock(root,policy='structural_only'))
            with self.assertRaises(ValueError):next(iter_calibrated_clock(root,policy=None))


if __name__=='__main__':unittest.main()

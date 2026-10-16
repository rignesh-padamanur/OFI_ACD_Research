import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from src.verify_chronological_panel_v3 import verify_precision_reuse,precision_frame_difference,registered_hashes

RUN=Path('/tmp/panel-v3-test-new');SOURCE=Path('/tmp/panel-v3-test-source')
ROOT_FILES=['cycle_schedule.parquet','cycle_statistics.parquet','daily.parquet','episodes.parquet','final_censored_episodes.parquet','final_episode_state.json','kernel_build.json','monthly.parquet','onsets.parquet','progress.json','statistics.json','test_evidence.json']
class PrecisionReuseGuards(unittest.TestCase):
 def setUp(self):
  self.old={'outputs':{name:'unchanged' for name in ROOT_FILES}}
  for i in range(60):
   for name in ['manifest.json','scored.parquet','onsets.parquet','parameters.json']:
    self.old['outputs'][f'cycles/{i:03d}/{name}']='unchanged'
  self.new={'outputs':dict(self.old['outputs']),'inputs':{'numerical_build_manifest':{'path':str(SOURCE/'manifest.json')}},'state_precision_repair':{'audit':'state_precision_audit.json','test_evidence':'test1.json','reference_test_evidence':'test2.json'}}
  self.new['outputs'].update({'test1.json':'test1','test2.json':'test2'})
  self.frames={};self.docs={SOURCE/'manifest.json':self.old}
  old=pd.DataFrame({'bin_start':pd.date_range('2024-01-01',periods=3,freq='s',tz='UTC'),'candidate_event':[False,True,False],'other':[1.,2.,3.],'pre_rv_300s_5s_bps':[np.nan,1.,2.]})
  new=old.copy();new.loc[1,'pre_rv_300s_5s_bps']=1.000001
  delta={'rows':3,'finite_rows':2,'exact_value_changes':1,'existing_tolerance_exceedances':1,'max_absolute_change_bps':1.000001-1.}
  rows=[]
  for i in range(60):
   prefix=f'cycles/{i:03d}'
   cm={'cycle_id':i,'status':'built_pending_independent_verification','outputs':{'scored.parquet':'unchanged','onsets.parquet':'unchanged','parameters.json':'unchanged'}}
   self.docs[SOURCE/prefix/'manifest.json']=cm
   self.docs[RUN/prefix/'manifest.json']=copy.deepcopy(cm)
   rows.append({'cycle_id':i,'inherited_preserved':i<2,'existing_tolerance_exceedances':0 if i<2 else 1,'candidate_tolerance_exceedances':0 if i<2 else 1,'onsets':{'existing_tolerance_exceedances':0} if i<2 else dict(delta)})
   if i>=2:
    rows[-1].update(delta)
    for name in ['scored.parquet','onsets.parquet']:
     self.new['outputs'][prefix+'/'+name]='changed'
     self.docs[RUN/prefix/'manifest.json']['outputs'][name]='changed'
     self.frames[SOURCE/prefix/name]=old.copy();self.frames[RUN/prefix/name]=new.copy()
    self.new['outputs'][prefix+'/manifest.json']='changed'
  self.frames[SOURCE/'onsets.parquet']=old.copy();self.frames[RUN/'onsets.parquet']=new.copy();self.new['outputs']['onsets.parquet']='changed'
  self.docs[RUN/'state_precision_audit.json']={'status':'built_pending_independent_verification','metric':'pre_rv_300s_5s_bps','cycles_checked':60,'cycles_recomputed':58,'model_refits':0,'missingness_unchanged':True,'detector_and_episode_columns_unchanged':True,'cycles':rows,'scored_tolerance_exceedances_corrected':58,'candidate_tolerance_exceedances_corrected':58,'onset_tolerance_exceedances_corrected':58,'max_absolute_scored_change_bps':1.000001-1.}
  for name in ['test1.json','test2.json']:self.docs[RUN/name]={'status':'passed','failures':0,'errors':0,'source_sha256':{}}
 def verify(self):
  with patch('src.verify_chronological_panel_v3.read',side_effect=lambda p:copy.deepcopy(self.docs[Path(p)])),patch('src.verify_chronological_panel_v3.pd.read_parquet',side_effect=lambda p:self.frames[Path(p)].copy()):return verify_precision_reuse(RUN,self.new)
 def rejects(self):
  with self.assertRaises(AssertionError):self.verify()
 def test_allowed_single_metric_repair_and_exact_audit_pass(self):
  r=self.verify();self.assertEqual(r['precision_frame_nonvariation_fields_verified'],117);self.assertEqual(r['precision_scored_values_changed'],58);self.assertEqual(r['precision_scored_rows_compared'],174)
 def test_nonvariation_value_change_rejected(self):
  self.frames[RUN/'cycles/002/scored.parquet'].loc[0,'other']=999;self.rejects()
 def test_candidate_change_rejected(self):
  self.frames[RUN/'cycles/002/scored.parquet'].loc[0,'candidate_event']=True;self.rejects()
 def test_missingness_change_rejected(self):
  self.frames[RUN/'cycles/002/scored.parquet'].loc[0,'pre_rv_300s_5s_bps']=0.;self.rejects()
 def test_negative_variation_rejected(self):
  self.frames[RUN/'cycles/002/scored.parquet'].loc[1,'pre_rv_300s_5s_bps']=-1.;self.rejects()
 def test_infinite_variation_rejected(self):
  self.frames[RUN/'cycles/002/scored.parquet'].loc[1,'pre_rv_300s_5s_bps']=float('inf');self.rejects()
 def test_root_statistics_change_rejected(self):
  self.new['outputs']['statistics.json']='changed';self.rejects()
 def test_inherited_metric_artifact_change_rejected(self):
  self.new['outputs']['cycles/001/scored.parquet']='changed';self.rejects()
 def test_model_parameter_change_rejected(self):
  self.new['outputs']['cycles/043/parameters.json']='changed';self.rejects()
 def test_cycle_metadata_change_rejected(self):
  self.docs[RUN/'cycles/043/manifest.json']['cycle_id']=42;self.rejects()
 def test_manifest_extra_artifact_rejected(self):
  self.docs[RUN/'cycles/043/manifest.json']['outputs']['another_model.json']='new';self.rejects()
 def test_missing_numerical_artifact_rejected(self):
  del self.new['outputs']['daily.parquet'];self.rejects()
 def test_unsupported_original_artifact_rejected(self):
  self.old['outputs']['unknown_model.parquet']='x';self.new['outputs']['unknown_model.parquet']='x';self.rejects()
 def test_false_correction_count_rejected(self):
  self.docs[RUN/'state_precision_audit.json']['candidate_tolerance_exceedances_corrected']=57;self.rejects()
 def test_unregistered_failed_source_hash_rejected_before_io(self):
  manifest={'inputs':{'numerical_build_manifest':{'path':'/tmp/step07c_chronological_panel_20260930/manifest.json','sha256':'another'}},'state_precision_repair':{}}
  manifest['state_precision_repair']['source_manifest']=manifest['inputs']['numerical_build_manifest']
  with patch('src.verify_chronological_panel_v3.read',return_value=manifest):
   with self.assertRaisesRegex(AssertionError,'preserved, specifically diagnosed'):registered_hashes(RUN)

if __name__=='__main__':unittest.main(verbosity=2)

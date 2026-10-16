"""Build and verify stage-01 quote data without overwriting legacy artifacts.

Example: python prepare_master_dataset.py --run-id step01_quote_sizes_20260927
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import unittest
from src.data_preparation import write_panel, promote_statistics, atomic_json
from src.verify_data_stage import verify_run

ROOT = Path(__file__).resolve().parent


def run_contract_tests():
    path=ROOT/'tests/test_data_preparation.py'
    spec=importlib.util.spec_from_file_location('data_stage_contract_tests',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    evidence={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'test_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output':stream.getvalue()}
    atomic_json(ROOT/'reports/tests/stage01_tests.json',evidence)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('Stage-01 contract tests failed; no data build or statistics promotion')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='data_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    evidence=run_contract_tests()
    if args.test_only:return
    active=ROOT/'data/active_run.json'
    if active.exists() and json.loads(active.read_text())['stage']!='01_quote_size_preservation':
        parser.error('A later data stage is active; use repair_acquisition.py instead of rebuilding legacy inputs')
    run=write_panel(ROOT,ROOT/'data/raw/dukascopy',args.run_id)
    atomic_json(run/'test_evidence.json',evidence)
    verification=verify_run(run)
    promote_statistics(ROOT,run,verification,evidence)
    print(json.dumps({'run':str(run),'statistics':verification['statistics'],
        'coverage_certified':False,'downstream_ready':False},indent=2),flush=True)


if __name__=='__main__':main()

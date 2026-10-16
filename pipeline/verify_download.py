"""Verify an active repaired data run against every registered source record.

This does not certify that all vendor hours or updates have been downloaded.
Use --run-id to select a run, or omit it for the registered active data run.
"""
from pathlib import Path
import argparse
import json
import re
from src.verify_data_stage import verify_run

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args=parser.parse_args()
    name=args.run_id
    if name is None:
        active=ROOT/'data/active_run.json'
        if not active.exists():parser.error('No active repaired run; run prepare_master_dataset.py first')
        name=json.loads(active.read_text())['data_run_id']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',name):parser.error('Invalid run id')
    print(json.dumps(verify_run(ROOT/'data/runs'/name),indent=2))


if __name__=='__main__':main()

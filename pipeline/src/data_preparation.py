"""Stage 01: preserve stored quote snapshots and rebuild a traceable clean panel.

This stage makes no claim of vendor completeness or validity of downstream OFI.
Run directories are immutable once created; legacy research outputs are untouched.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

QUOTE_COLUMNS = ['timestamp', 'ask', 'bid', 'ask_volume', 'bid_volume']
RAW_COLUMNS = QUOTE_COLUMNS + ['source_file', 'source_row']
CLEAN_COLUMNS = RAW_COLUMNS + ['spread', 'mid', 'spread_pips', 'delta_sec',
                               'quote_size_valid', 'timestamp_tied']
RAW_SCHEMA = pa.schema([('timestamp', pa.timestamp('ns', tz='UTC'))]
    + [(c, pa.float64()) for c in QUOTE_COLUMNS[1:]]
    + [('source_file',pa.string()),('source_row',pa.int64())])
CLEAN_SCHEMA = pa.schema(list(RAW_SCHEMA)
    + [(c,pa.float64()) for c in ['spread','mid','spread_pips','delta_sec']]
    + [('quote_size_valid',pa.bool_()),('timestamp_tied',pa.bool_())])
LIMITATIONS = [
    'Existing monthly files are cached observations, not a certified complete vendor feed.',
    'The legacy downloader may have removed size-only updates sharing a timestamp and prices; those records cannot be recovered locally.',
    'Missing hours, holidays, DST, and within-hour coverage require stage 02.',
    'Source-row order within a tied timestamp is not independently established market-event order.',
    'Quote-size units/update semantics and downstream OFI identification require separate verification.',
    'Long gaps are retained; no stale-quote filling, duration truncation, or session filtering is performed here.',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_source(path: Path) -> pd.DataFrame:
    match = re.fullmatch(r'eurusd_(\d{4})_(\d{2})\.parquet', path.name)
    if not match:
        raise ValueError(f'Unexpected source filename: {path.name}')
    frame = pd.read_parquet(path, columns=QUOTE_COLUMNS)
    if not isinstance(frame.timestamp.dtype, pd.DatetimeTZDtype):
        raise ValueError(f'{path.name}: timestamps must explicitly carry a timezone')
    frame['timestamp'] = frame.timestamp.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    if frame.timestamp.isna().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError(f'{path.name}: missing or decreasing timestamps')
    year, month = map(int, match.groups())
    if not ((frame.timestamp.dt.year == year) & (frame.timestamp.dt.month == month)).all():
        raise ValueError(f'{path.name}: records outside the named UTC month')
    for col in QUOTE_COLUMNS[1:]:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise TypeError(f'{path.name}: {col} must be numeric, not silently coerced')
        frame[col] = frame[col].astype('float64')
    frame['source_file'] = path.name
    frame['source_row'] = np.arange(len(frame), dtype=np.int64)
    return frame[RAW_COLUMNS]


def clean_quotes(raw: pd.DataFrame, previous_timestamp=None):
    """Return cleaned records, disjoint removal counts, and last retained time.

    Distinct quote sizes and tied timestamps survive. Invalid sizes are flagged.
    ``previous_timestamp`` belongs to the last retained record in the prior month.
    """
    if set(RAW_COLUMNS) - set(raw):
        raise ValueError('Missing raw quote/provenance columns')
    if not isinstance(raw.timestamp.dtype, pd.DatetimeTZDtype) or raw.timestamp.isna().any():
        raise ValueError('Explicit valid timezone-aware timestamps are required')
    frame = raw[RAW_COLUMNS].copy()
    frame['timestamp'] = frame.timestamp.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    frame = frame.sort_values(['timestamp', 'source_row'], kind='stable')
    reasons = {}

    def remove(name, mask):
        nonlocal frame
        reasons[name] = int(mask.sum())
        frame = frame.loc[~mask].copy()

    remove('nonfinite_price', ~np.isfinite(frame[['ask', 'bid']]).all(axis=1))
    remove('nonpositive_price', (frame.ask <= 0) | (frame.bid <= 0))
    remove('crossed_quote', frame.ask < frame.bid)
    remove('zero_spread', frame.ask == frame.bid)
    remove('duplicate_full_snapshot', frame.duplicated(QUOTE_COLUMNS, keep='first'))
    if len(frame) and previous_timestamp is not None and frame.timestamp.iloc[0] <= previous_timestamp:
        raise ValueError('Partitions overlap or are not in chronological order')
    frame['spread'] = frame.ask - frame.bid
    frame['mid'] = (frame.ask + frame.bid) / 2
    frame['spread_pips'] = frame.spread * 10000
    frame['delta_sec'] = frame.timestamp.diff().dt.total_seconds()
    if len(frame) and previous_timestamp is not None:
        # scalar Timedelta.total_seconds() discards sub-microsecond precision
        # and can round differently from the vectorized within-month operation.
        elapsed_ns = frame.timestamp.iloc[0].value - previous_timestamp.value
        frame.loc[frame.index[0], 'delta_sec'] = elapsed_ns / 1_000_000_000
    frame['quote_size_valid'] = (np.isfinite(frame[['ask_volume', 'bid_volume']]).all(axis=1)
                                 & (frame.ask_volume >= 0) & (frame.bid_volume >= 0))
    frame['timestamp_tied'] = frame.timestamp.duplicated(keep=False)
    last = frame.timestamp.iloc[-1] if len(frame) else previous_timestamp
    return frame[CLEAN_COLUMNS].reset_index(drop=True), reasons, last


def write_panel(root: Path, source_dir: Path, run_id: str,
                start_month='2024-01', end_month='2026-07') -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('run_id must be a simple identifier of at most 80 characters')
    periods = pd.period_range(start_month, end_month, freq='M')
    if not len(periods):
        raise ValueError('Empty requested month range')
    files = [source_dir / f'eurusd_{p.year}_{p.month:02d}.parquet' for p in periods]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing requested source months: {missing}')
    run = root / 'data/runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema_version': 2, 'run_id': run_id, 'stage': '01_quote_size_preservation',
        'status': 'building', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'requested_start_month': start_month, 'requested_end_month': end_month,
        'source_scope': 'existing_cached_monthly_files', 'coverage_status': 'not_certified',
        'limitations': LIMITATIONS, 'input_files': [], 'monthly_cleaning': [],
        'environment': {'python': platform.python_version(), **{
            p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow']}},
        'implementation_sha256': sha256(Path(__file__)),
    }
    code_root = Path(__file__).resolve().parents[1]
    manifest['code_files'] = {}
    for name in ['src/data_preparation.py', 'src/verify_data_stage.py',
                 'prepare_master_dataset.py', 'verify_download.py',
                 'tests/test_data_preparation.py']:
        target = run / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code_root / name, target)
        manifest['code_files'][name] = sha256(target)
    atomic_json(run / 'manifest.json', manifest)
    previous = None
    writers = {}
    try:
        for path in files:
            fingerprint = sha256(path)
            raw = load_source(path)
            clean, removed, previous = clean_quotes(raw, previous)
            manifest['input_files'].append({'path': str(path.resolve()), 'name': path.name,
                'sha256': fingerprint, 'bytes': path.stat().st_size, 'rows': len(raw)})
            manifest['monthly_cleaning'].append({'source_file': path.name, 'raw_rows': len(raw),
                'clean_rows': len(clean), 'removed': removed})
            for name, frame in [('raw_quotes.parquet', raw), ('clean_quotes.parquet', clean)]:
                schema = RAW_SCHEMA if name == 'raw_quotes.parquet' else CLEAN_SCHEMA
                table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(run / name, table.schema, compression='zstd')
                writers[name].write_table(table, row_group_size=250000)
            print(f'{path.name}: {len(raw):,} raw -> {len(clean):,} clean', flush=True)
        for writer in writers.values():
            writer.close()
        writers = {}
        if previous is None:
            raise ValueError('No retained observations')
        manifest['status'] = 'built_awaiting_independent_verification'
        atomic_json(run / 'manifest.json', manifest)
    except Exception as error:
        for writer in writers.values():
            writer.close()
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        atomic_json(run / 'manifest.json', manifest)
        raise
    return run


def promote_statistics(root: Path, run: Path, verification: dict, test_evidence: dict):
    """Register a verified data stage without representing later stages as repaired."""
    if verification.get('status') != 'passed':
        raise ValueError('Cannot publish statistics from an unverified run')
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'verified':
        raise ValueError('Run manifest has not been verified')
    master_path = root / 'reports/pipeline_master_statistics.json'
    prior = json.loads(master_path.read_text()) if master_path.exists() else {}
    current_stage = prior.get('current_data_stage', {}).get('stage')
    if current_stage not in [None, '01_quote_size_preservation']:
        raise ValueError('Stage 01 cannot replace a later repaired data stage; use its stage-specific entry point')
    if prior.get('schema_version') == 2:
        master = prior
    else:
        historical = root / 'reports/repair_history/2026-09-27_pre_repair/reports/pipeline_master_statistics.json'
        if not historical.exists() or (master_path.exists() and historical.read_bytes() != master_path.read_bytes()):
            raise ValueError('Preserve the exact legacy statistics before the first promotion')
        master = {'schema_version': 2, 'legacy_statistics': {'status': 'exploratory_invalid_for_repaired_pipeline',
            'path': str(historical.relative_to(root)), 'sha256': sha256(historical)}, 'stage_history': []}
    if 'statistics' not in master['legacy_statistics']:
        historical = root / master['legacy_statistics']['path']
        if sha256(historical) != master['legacy_statistics']['sha256']:
            raise ValueError('Legacy statistics archive changed')
        # the old file contains nonstandard NaN JSON. Preserve its exact bytes
        # in the archive, and represent undefined numbers as null here.
        master['legacy_statistics']['statistics'] = json.loads(
            historical.read_text(), parse_constant=lambda _: None)
        master['legacy_statistics']['encoding_note'] = (
            'Nonfinite legacy JSON constants are null here; exact original bytes remain archived.')
    for previous in master['stage_history']:
        if previous['stage'] == '01_quote_size_preservation' and previous['status'] == 'verified_on_cached_inputs':
            previous['status'] = 'superseded'
            previous['superseded_by'] = run.name
    entry = {'stage': '01_quote_size_preservation', 'status': 'verified_on_cached_inputs',
        'run_id': run.name, 'manifest': str((run / 'manifest.json').relative_to(root)),
        'verification': str((run / 'verification.json').relative_to(root)),
        'statistics': verification['statistics'], 'test_evidence': test_evidence,
        'limitations': LIMITATIONS, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    master['stage_history'].append(entry)
    master['current_data_stage'] = entry
    master['pipeline_status'] = 'repair_in_progress_not_paper_ready'
    master['downstream_status'] = {s:'pending_rebuild_do_not_use_legacy_estimates' for s in
        ['coverage_recovery','ofi','seasonality_acd','event_detection','persistence','regressions','robustness','figures_tables']}
    master['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(master_path, master)
    atomic_json(root / 'data/active_run.json', {'schema_version':2,'data_run_id':run.name,
        'stage':'01_quote_size_preservation','coverage_certified':False,'downstream_ready':False})


def exact_median(hist: Counter):
    total = sum(hist.values())
    if not total:
        return None
    positions = [(total - 1)//2, total//2]
    cumulative = 0
    found = []
    for value, count in sorted(hist.items()):
        for position in positions:
            if cumulative <= position < cumulative + count:
                found.append(float(value))
        cumulative += count
    return sum(found)/2

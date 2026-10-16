"""Prevent the legacy notebooks from overwriting outputs during staged repair."""
from pathlib import Path
import json


def require_legacy_pipeline(root):
    root=Path(root)
    lock=root/'reports/PIPELINE_REPAIR_LOG.md'
    if lock.exists():
        raise RuntimeError(
            'This is a legacy notebook/script whose outputs are not valid for the repaired pipeline. '
            'Use prepare_master_dataset.py for the verified data stage and consult '
            'reports/PIPELINE_REPAIR_LOG.md before running downstream stages. '
            'The original source is preserved under reports/repair_history/.')


def load_observation_support(root, *, policy):
    """Load an explicit diagnostic policy tied to the current verified quote data.

    No policy is selected implicitly. Interval ends and outcome availability
    are audit information, not ex-ante predictors. Legacy analyses stay blocked.
    """
    import pandas as pd
    from src.data_preparation import sha256
    from src.observation_eligibility import POLICIES
    if policy not in POLICIES:
        raise ValueError('Choose an explicit registered observation-support policy')
    root = Path(root)
    active = json.loads((root/'data/active_run.json').read_text())
    pointer = json.loads((root/'data/active_eligibility.json').read_text())
    if active['data_run_id'] != pointer['parent_data_run']:
        raise RuntimeError('Eligibility belongs to a different data run; rebuild it')
    run = root/'data/eligibility'/pointer['eligibility_run_id']
    manifest = json.loads((run/'manifest.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    if manifest['status'] != 'verified_observation_support_only' or verification['status'] != 'passed':
        raise RuntimeError('Unverified observation support')
    if manifest['parent_data_run'] != active['data_run_id']:
        raise RuntimeError('Eligibility manifest and active data differ')
    parent = root/'data/runs'/active['data_run_id']/'clean_quotes.parquet'
    if sha256(parent) != pointer['parent_clean_sha256']:
        raise RuntimeError('Registered quote data have changed')
    path = run/'support_intervals.parquet'
    if sha256(path) != manifest['output_sha256'][path.name]:
        raise RuntimeError('Registered observation support has changed')
    frame = pd.read_parquet(path)
    return frame.loc[frame.policy.eq(policy)].reset_index(drop=True)


def iter_quote_imbalance(root, *, policy, batch_size=500_000):
    """Yield registered completed-bin signals under an explicit sampling policy.

    Unusable signals stay NaN, with audit counts retained. No empty bins are
    generated. Use available_at for signal timing; audit flags are not predictors.
    """
    import pyarrow.parquet as pq
    from src.data_preparation import sha256
    from src.ofi import POLICIES
    if policy not in POLICIES or batch_size < 1:
        raise ValueError('An explicit policy and positive batch size are required')
    root = Path(root)
    data = json.loads((root/'data/active_run.json').read_text())
    eligibility = json.loads((root/'data/active_eligibility.json').read_text())
    pointer = json.loads((root/'data/active_quote_imbalance.json').read_text())
    if pointer['parent_data_run'] != data['data_run_id'] or pointer['parent_eligibility_run'] != eligibility['eligibility_run_id']:
        raise RuntimeError('Quote imbalance belongs to different data or eligibility')
    run = root/'data/signals'/pointer['signal_run_id']
    manifest = json.loads((run/'manifest.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    if manifest['status'] != 'verified_recorded_quote_proxy' or verification['status'] != 'passed':
        raise RuntimeError('Quote imbalance is not verified')
    if manifest['parent_data_run'] != data['data_run_id'] or manifest['parent_eligibility_run'] != eligibility['eligibility_run_id']:
        raise RuntimeError('Signal manifest and active lineage differ')
    for name, item in manifest['inputs'].items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise RuntimeError('Signal input changed: '+name)
    path = run/'quote_ofi_1s.parquet'
    if sha256(path) != manifest['outputs'][path.name]:
        raise RuntimeError('Signal output changed')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        frame['quote_ofi'] = frame.quote_ofi.where(frame[f'usable_{policy}'])
        frame['abs_quote_ofi'] = frame.quote_ofi.abs()
        yield frame


def iter_signal_clock(root, *, policy, batch_size=216_000):
    """Yield aligned, masked count/imbalance columns; keep all missing clock rows.

    available_at is the completed-bin timestamp. support_id/flow_status are
    retrospective audit metadata, not predictors. Never lag across missing bins
    or join different support IDs. No default observation policy is selected.
    """
    import pyarrow.parquet as pq
    from src.data_preparation import sha256
    from src.intensity import materialize_clock_policy
    from src.observation_eligibility import POLICIES
    if policy not in POLICIES or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('An explicit policy and positive integer batch size are required')
    root = Path(root)
    pointer = json.loads((root/'data/active_signal_clock.json').read_text())
    expected = {
        'parent_data_run': ('active_run.json', 'data_run_id'),
        'parent_eligibility_run': ('active_eligibility.json', 'eligibility_run_id'),
        'parent_signal_run': ('active_quote_imbalance.json', 'signal_run_id'),
    }
    run = root/'data/signals'/pointer['clock_run_id']
    manifest = json.loads((run/'manifest.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    if manifest['status'] != 'verified_supported_signal_clock' or verification['status'] != 'passed':
        raise RuntimeError('Signal clock is not verified')
    for field, (file, key) in expected.items():
        active = json.loads((root/'data'/file).read_text())
        if pointer[field] != active[key] or manifest[field] != active[key]:
            raise RuntimeError('Signal clock belongs to different active inputs: '+field)
    for name, record in manifest['inputs'].items():
        if sha256(Path(record['path'])) != record['sha256']:
            raise RuntimeError('Clock ancestor changed: '+name)
    path = run/'signal_clock_1s.parquet'
    if sha256(path) != manifest['outputs'][path.name]:
        raise RuntimeError('Clock output changed')
    columns = ['bin_start', 'available_at', 'recorded_update_count', 'observed_quote_ofi',
               f'support_id_{policy}', f'flow_status_{policy}']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield materialize_clock_policy(batch.to_pandas(), policy=policy)


def iter_calibrated_clock(root, *, policy):
    """Registered past-day calibrated counts/quote pressure, retaining unavailable rows.

    Calibration is conditional on the audited support convention. Scores describe
    completed bins; they are not automatically pre-onset market-state features.
    The calibration history is not a supervised train/test split.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from src.data_preparation import sha256
    from src.deseasonal import POLICY_NAMES, normalize_policy, table_to_fits
    if policy not in POLICY_NAMES:
        raise ValueError('An explicit registered support policy is required')
    root=Path(root);pointer=json.loads((root/'data/active_calibration.json').read_text())
    run=root/'data/signals'/pointer['calibration_run_id'];m=json.loads((run/'manifest.json').read_text())
    if m['status']!='verified_past_day_reference_calibration' or json.loads((run/'verification.json').read_text())['status']!='passed':
        raise RuntimeError('Unverified past-day calibration')
    for field,file,key in [('parent_data_run','active_run.json','data_run_id'),
        ('parent_eligibility_run','active_eligibility.json','eligibility_run_id'),
        ('parent_signal_run','active_quote_imbalance.json','signal_run_id'),
        ('parent_clock_run','active_signal_clock.json','clock_run_id')]:
        value=json.loads((root/'data'/file).read_text())[key]
        if pointer[field]!=value or m[field]!=value:raise RuntimeError('Calibration lineage changed: '+field)
    if pointer['configuration']!=m['configuration']:raise RuntimeError('Calibration settings disagree')
    for name,record in m['inputs'].items():
        if sha256(Path(record['path']))!=record['sha256']:raise RuntimeError('Calibration ancestor changed: '+name)
    for file in ['calibration_parameters.parquet','standardized_clock_1s.parquet']:
        if sha256(run/file)!=m['outputs'][file]:raise RuntimeError('Calibration output changed: '+file)
    fits=table_to_fits(pd.read_parquet(run/'calibration_parameters.parquet'))
    parent=pq.ParquetFile(m['inputs']['clock']['path'])
    normalized=pq.ParquetFile(run/'standardized_clock_1s.parquet')
    if parent.metadata.num_rows!=normalized.metadata.num_rows:raise RuntimeError('Normalized clock length differs')
    columns=['bin_start','available_at',f'z_tick_count_{policy}',f'z_abs_quote_ofi_{policy}']
    source=parent.iter_batches(batch_size=86400);saved=normalized.iter_batches(batch_size=86400,columns=columns)
    for raw_batch,z_batch in zip(source,saved,strict=True):
        frame=raw_batch.to_pandas();z=z_batch.to_pandas();day=pd.Timestamp(frame.bin_start.iloc[0]).normalize()
        np.testing.assert_array_equal(frame.bin_start.to_numpy(),z.bin_start.to_numpy())
        out=normalize_policy(frame,fits[day.value],policy=policy)
        for v in ['tick_count','abs_quote_ofi']:
            np.testing.assert_allclose(out['z_'+v],z[f'z_{v}_{policy}'],rtol=1e-10,atol=1e-9,equal_nan=True)
            out['z_'+v]=z[f'z_{v}_{policy}'].to_numpy()
        out['calibration_cutoff']=day
        yield out


def load_duration_observations(root, *, policy, batch_size=500_000):
    """Return (completed-row iterator, terminal-spell table) after provenance checks.

    All quote rows remain in the iterator, including missing incoming durations
    at component starts. The terminal table is required for a later observation-
    mechanism/likelihood decision; positive exposure alone is not eligibility.
    Audit IDs/stops are not Ft0 covariates. No fitted ACD model is returned.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from src.data_preparation import sha256
    from src.duration_observation import materialize_duration_policy, POLICIES
    if policy not in POLICIES or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('Explicit registered policy and positive integer batch size required')
    root = Path(root); pointer = json.loads((root/'data/active_duration_observation.json').read_text())
    run = root/'data/signals'/pointer['duration_run_id']; m = json.loads((run/'manifest.json').read_text())
    if pointer['status'] != 'verified_duration_observation_only' or m['status'] != pointer['status'] or json.loads((run/'verification.json').read_text())['status'] != 'passed':
        raise RuntimeError('Unverified duration observation ledger')
    for field,file,key in [('parent_data_run','active_run.json','data_run_id'),('parent_eligibility_run','active_eligibility.json','eligibility_run_id')]:
        value = json.loads((root/'data'/file).read_text())[key]
        if pointer[field] != value or m[field] != value: raise RuntimeError('Duration lineage changed: '+field)
    for name,item in m['inputs'].items():
        if sha256(Path(item['path'])) != item['sha256']: raise RuntimeError('Duration input changed: '+name)
    for file,expected in m['outputs'].items():
        if sha256(run/file) != expected: raise RuntimeError('Duration artifact changed: '+file)
    terminals = pd.read_parquet(run/'terminal_spells.parquet')
    terminals = terminals.loc[terminals.policy.eq(policy)].reset_index(drop=True)
    def completed_rows():
        columns = ['quote_id','available_at','audit_gap_ns', 'component_id_'+policy,'completed_'+policy,'recurrence_reset_'+policy]
        for batch in pq.ParquetFile(run/'quote_durations.parquet').iter_batches(batch_size=batch_size, columns=columns):
            yield materialize_duration_policy(batch.to_pandas(), policy=policy)
    return completed_rows(), terminals

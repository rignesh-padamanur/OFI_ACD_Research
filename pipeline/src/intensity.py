"""Recorded quote-update activity on an explicitly supported signal clock."""
import numpy as np
import pandas as pd
from src.observation_eligibility import POLICIES


def compute_tick_intensity(*args, **kwargs):
    raise RuntimeError('Legacy resampling/Sunday exclusion/full-sample adjustment is disabled. '
                       'Use src.pipeline_state.iter_signal_clock(root, policy=...) instead.')


def materialize_clock_policy(frame, *, policy):
    """Create model columns; stored audit counts alone must not enter estimation.

    Support IDs/status are retrospective sample-quality metadata, not predictors.
    Later rolling operations must require contiguous valid bins in one support ID.
    No durations, prices, seasonal factors or normalized signals are constructed.
    """
    if policy not in POLICIES:
        raise ValueError('An explicit registered support policy is required')
    code = frame[f'flow_status_{policy}'].to_numpy()
    sid = frame[f'support_id_{policy}'].to_numpy()
    count = frame.recorded_update_count.to_numpy()
    if not np.isin(code, [0, 1, 2, 3, 4]).all() or not np.array_equal(code == 0, sid < 0):
        raise ValueError('Inconsistent support/status encoding')
    if ((code == 1) & (count != 0)).any() or ((code >= 2) & (count <= 0)).any():
        raise ValueError('Inconsistent empty/active encoding')
    if ((code == 2) & ~np.isfinite(frame.observed_quote_ofi)).any():
        raise ValueError('Valid active imbalance must be finite')
    return pd.DataFrame({
        'bin_start': frame.bin_start, 'available_at': frame.available_at,
        'tick_count': np.where(sid >= 0, count, np.nan),
        'quote_ofi': np.where(code == 1, 0., np.where(code == 2, frame.observed_quote_ofi, np.nan)),
        'support_id': sid, 'flow_status': code,
        'joint_usable': (code == 1) | (code == 2),
    }).assign(abs_quote_ofi=lambda x: x.quote_ofi.abs())

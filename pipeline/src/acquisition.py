"""Auditable EURUSD hourly acquisition. A calendar label never proves feed coverage."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import lzma
import time

import numpy as np
import pandas as pd
import requests

from src.data_preparation import QUOTE_COLUMNS, atomic_json, sha256

FEED_ROOT = 'https://datafeed.dukascopy.com/datafeed/EURUSD'
RECORD = np.dtype([('ms', '>u4'), ('ask', '>u4'), ('bid', '>u4'),
                   ('ask_volume', '>f4'), ('bid_volume', '>f4')])


def utc_hour(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None or t != t.floor('h'):
        raise ValueError('An explicit timezone and exact hourly boundary are required')
    return t.tz_convert('UTC')


def nominal_weekly_open(hours):
    """Vendor weekly schedule, BEFORE exceptional/holiday closure reconciliation."""
    ix = pd.DatetimeIndex(hours)
    if ix.tz is None:
        raise ValueError('Timezone-aware calendar required')
    ny = ix.tz_convert('America/New_York')
    return (ny.dayofweek < 4) | ((ny.dayofweek == 4) & (ny.hour < 17)) | ((ny.dayofweek == 6) & (ny.hour >= 17))


def hour_url(hour):
    t = utc_hour(hour)
    return f'{FEED_ROOT}/{t.year}/{t.month-1:02d}/{t.day:02d}/{t.hour:02d}h_ticks.bi5'


def decode_hour(payload: bytes, hour) -> pd.DataFrame:
    """Keep every binary record, its order, prices, sizes, and millisecond timestamp.

    This decoder is for the HOURLY endpoint. The current vendor daily-bucket
    documentation has a different timestamp origin and cannot be substituted.
    Invalid price/size states are retained for later explicit eligibility rules.
    """
    t = utc_hour(hour)
    raw = lzma.decompress(payload)
    if len(raw) % 20:
        raise ValueError('BI5 payload does not contain whole 20-byte records')
    records = np.frombuffer(raw, dtype=RECORD)
    offsets = records['ms'].astype(np.int64)
    if (offsets >= 3_600_000).any():
        raise ValueError('Timestamp offset outside requested hour')
    if (np.diff(offsets) < 0).any():
        raise ValueError('Decreasing source timestamps; do not silently reorder vendor records')
    frame = pd.DataFrame({'timestamp': pd.to_datetime(t.value + offsets * 1_000_000, utc=True),
        'ask': records['ask'].astype(float) / 100_000,
        'bid': records['bid'].astype(float) / 100_000,
        'ask_volume': records['ask_volume'].astype(float),
        'bid_volume': records['bid_volume'].astype(float),
        'vendor_record': np.arange(len(records), dtype=np.int64)})
    return frame


def fetch_hour(hour, cache: Path, session=None, max_attempts=3, sleep=time.sleep):
    """Persist every response and outcome; never interpret 404/empty as a closure.

    A previous successful/empty/404 response is reused only with its fingerprint.
    A failed request remains retryable on a later invocation. HTTP access denials
    are returned immediately, without changing hosts or disguising the client.
    """
    hour = utc_hour(hour)
    if not 1 <= max_attempts <= 10: raise ValueError('max_attempts must be between 1 and 10')
    cache.mkdir(parents=True, exist_ok=True)
    key = hour.strftime('%Y%m%dT%H'); record_path = cache / f'{key}.json'
    if record_path.exists():
        prior = json.loads(record_path.read_text())
        if prior['status'] in ['ok', 'empty_200', 'empty_decoded', 'not_found_404']:
            if sha256(cache / prior['response_file']) != prior['response_sha256']:
                raise ValueError('Cached response fingerprint changed')
            return prior
    else:
        prior = {'hour':hour.isoformat(), 'url':hour_url(hour), 'attempts':[]}
    own_session = session is None
    session = session or requests.Session()
    try:
        for attempt in range(max_attempts):
            item = {'requested_at_utc':datetime.now(timezone.utc).isoformat()}
            try:
                response = session.get(prior['url'], timeout=(30, 30))
                item['http_status'] = response.status_code
                if isinstance(response.headers,dict) or hasattr(response.headers,'items'):
                    retry_after=response.headers.get('Retry-After')
                    if isinstance(retry_after,str):item['retry_after']=retry_after
                body = response.content
                name = f'{key}_attempt{len(prior["attempts"])+1}.response'
                (cache / name).write_bytes(body)
                item.update(response_file=name, response_bytes=len(body),
                            response_sha256=hashlib.sha256(body).hexdigest())
                if response.status_code == 200:
                    if not body:
                        item.update(status='empty_200', rows=0)
                    else:
                        try:
                            frame = decode_hour(body, hour)
                            item.update(status='ok' if len(frame) else 'empty_decoded', rows=len(frame))
                        except (lzma.LZMAError, EOFError, ValueError) as error:
                            item.update(status='invalid_payload', error=str(error))
                elif response.status_code == 404:
                    item.update(status='not_found_404', rows=0)
                elif response.status_code in (401, 403):
                    item['status'] = 'access_denied'
                elif response.status_code == 429:
                    item['status'] = 'rate_limited'
                else:
                    item['status'] = 'http_error'
            except requests.RequestException as error:
                item.update(status='network_error', error=f'{type(error).__name__}: {error}')
            prior['attempts'].append(item)
            for field in ['response_file','response_bytes','response_sha256','http_status','rows','error','retry_after']:
                prior.pop(field,None)
            prior.update({k:v for k,v in item.items() if k != 'requested_at_utc'})
            atomic_json(record_path,prior)
            if item['status'] in ['ok','empty_200','empty_decoded','not_found_404','access_denied','rate_limited']:
                break
            if attempt+1 < max_attempts:
                sleep(min(2**attempt, 8))
    finally:
        if own_session: session.close()
    return prior


def audit_months(source_dir: Path, start, end):
    """Inspect all requested hours and quote gaps, without inferring outage causes."""
    start, end = utc_hour(start), utc_hour(end)
    if end <= start: raise ValueError('End must follow start (exclusive end)')
    grid = pd.date_range(start,end,freq='h',inclusive='left')
    months = pd.period_range(start.tz_localize(None), (end-pd.Timedelta(nanoseconds=1)).tz_localize(None),freq='M')
    parts=[]; gaps=[]; fingerprints=[]; prev=None
    for month in months:
        path = source_dir/f'eurusd_{month.year}_{month.month:02d}.parquet'
        ts = pd.read_parquet(path,columns=['timestamp']).timestamp
        if not isinstance(ts.dtype,pd.DatetimeTZDtype) or ts.isna().any(): raise ValueError('Invalid source timestamps')
        ts=ts.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
        if not ts.is_monotonic_increasing:raise ValueError('Unsorted source month')
        if not ((ts.dt.year==month.year)&(ts.dt.month==month.month)).all():raise ValueError('Source outside named month')
        fingerprints.append({'name':path.name,'sha256':sha256(path),'rows':len(ts)})
        ts=ts[(ts>=start)&(ts<end)].reset_index(drop=True)
        if not len(ts):continue
        hours=ts.dt.floor('h')
        differences=ts.diff().dt.total_seconds()
        if prev is not None:differences.iloc[0]=(ts.iloc[0].value-prev.value)/1e9
        previous=ts.shift();previous.iloc[0]=prev if prev is not None else pd.NaT
        long=differences>60
        gaps.append(pd.DataFrame({'previous_timestamp':previous[long], 'timestamp':ts[long], 'delta_sec':differences[long]}))
        within=differences.where(hours.eq(hours.shift()))
        table=pd.DataFrame({'hour':hours,'timestamp':ts,'internal_gap':within}).groupby('hour').agg(
            rows=('timestamp','size'),first_timestamp=('timestamp','min'),last_timestamp=('timestamp','max'),
            max_internal_gap_sec=('internal_gap','max'))
        parts.append(table);prev=ts.iloc[-1]
    observed=pd.concat(parts) if parts else pd.DataFrame(columns=['rows','first_timestamp','last_timestamp','max_internal_gap_sec'])
    if observed.index.has_duplicates:raise ValueError('Overlapping input months')
    ledger=observed.reindex(grid);ledger.index.name='hour'
    for name in ['first_timestamp','last_timestamp']:
        ledger[name]=pd.to_datetime(ledger[name],utc=True)
    ledger['rows']=ledger.rows.fillna(0).astype('int64')
    ledger['nominal_weekly_open']=nominal_weekly_open(grid)
    ledger['calendar_status']=np.where(ledger.nominal_weekly_open,'nominal_open_holiday_unreconciled','weekly_closed')
    ledger['observation_status']=np.where(ledger.rows>0,'observed_not_completeness_certified','absent')
    ledger['first_quote_delay_sec']=(ledger.first_timestamp-ledger.index.to_series()).dt.total_seconds()
    ledger['last_quote_to_hour_end_sec']=(ledger.index.to_series()+pd.Timedelta(hours=1)-ledger.last_timestamp).dt.total_seconds()
    gap_frame=pd.concat(gaps,ignore_index=True) if gaps else pd.DataFrame(columns=['previous_timestamp','timestamp','delta_sec'])
    for name in ['previous_timestamp','timestamp']:gap_frame[name]=pd.to_datetime(gap_frame[name],utc=True)
    gap_frame['delta_sec']=gap_frame.delta_sec.astype(float)
    missing=ledger.nominal_weekly_open & ledger.rows.eq(0)
    winter_friday=missing & (grid.dayofweek==4) & (grid.hour==21)
    statistics={'requested_calendar_hours':len(grid),'raw_rows':int(ledger.rows.sum()),
        'observed_hours':int((ledger.rows>0).sum()),'nominal_weekly_open_hours':int(ledger.nominal_weekly_open.sum()),
        'absent_nominal_open_hours':int(missing.sum()),'absent_winter_friday_21utc_hours':int(winter_friday.sum()),
        'observed_weekly_closed_hours':int(((ledger.rows>0)&~ledger.nominal_weekly_open).sum()),
        'raw_gaps_gt_60s':len(gap_frame),'observed_hours_with_internal_gap_gt_60s':int((ledger.max_internal_gap_sec>60).sum()),
        'observed_hours_with_first_quote_delay_gt_60s':int((ledger.first_quote_delay_sec>60).sum()),
        'observed_hours_with_last_quote_silence_gt_60s':int((ledger.last_quote_to_hour_end_sec>60).sum())}
    return ledger.reset_index(),gap_frame,statistics,fingerprints

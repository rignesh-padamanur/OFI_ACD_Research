"""Independent serialized-output audit; does not call the eligibility algorithms."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import sha256

S = 1_000_000_000
H = 3600*S


def ns(column):
    return pd.DatetimeIndex(column).as_unit('ns').asi8


def reference_hour_counts(starts, ends, lookback, horizon, origin, n):
    left = np.floor_divide(starts-1, S)+1+lookback
    right = np.floor_divide(ends-1, S)-horizon
    keep = left <= right; left = left[keep]; right = right[keep]
    first = (left-origin)//3600; last = (right-origin)//3600
    result = np.zeros(n, dtype='int64'); diff = np.zeros(n+1, dtype='int64')
    same = first == last
    np.add.at(result, first[same], right[same]-left[same]+1)
    cross = ~same
    np.add.at(result, first[cross], origin+(first[cross]+1)*3600-left[cross])
    np.add.at(result, last[cross], right[cross]-(origin+last[cross]*3600)+1)
    np.add.at(diff, first[cross]+1, 3600); np.add.at(diff, last[cross], -3600)
    return result+np.cumsum(diff[:-1])


def verify_eligibility(run):
    run = Path(run); manifest = json.loads((run/'manifest.json').read_text())
    for name, item in manifest['input_hashes'].items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise AssertionError(f'Changed input {name}')
    for name, fingerprint in manifest['output_sha256'].items():
        if sha256(run/name) != fingerprint:
            raise AssertionError(f'Changed output {name}')
    for name, fingerprint in manifest['code_sha256'].items():
        if sha256(run/'code'/name) != fingerprint:
            raise AssertionError(f'Changed source snapshot {name}')
    parent = Path(manifest['parent_directory'])
    pm = json.loads((parent/'manifest.json').read_text())
    hour = pd.read_parquet(run/'hourly_eligibility.parquet')
    origin = int(ns(hour.hour)[0]); hns = ns(hour.hour)
    np.testing.assert_array_equal(np.diff(hns), np.full(len(hour)-1,H))
    ny = ZoneInfo('America/New_York')
    calendar=[]
    for t in hour.hour:
        local=t.to_pydatetime().astimezone(ny); day=local.weekday()
        calendar.append(day < 4 or (day==4 and local.hour<17) or (day==6 and local.hour>=17))
    np.testing.assert_array_equal(calendar,hour.nominal_weekly_open)
    allowed=hour.rows.to_numpy()>0
    assert not (allowed & ~np.array(calendar)).any()
    np.testing.assert_array_equal(allowed,hour.structurally_observed_hour)
    block=[]; last_good=False; bid=-1
    for good in allowed:
        if good and not last_good: bid+=1
        block.append(bid if good else -1);last_good=good
    np.testing.assert_array_equal(block,hour.observation_block_id)
    reasons=np.where(~np.array(calendar),'weekly_closed',np.where(allowed,
        'observed_not_completeness_certified','absent_nominal_open_unresolved'))
    np.testing.assert_array_equal(reasons,hour.eligibility_reason)

    offsets={}; total=0
    for item in pm['input_files']:
        offsets[item['name']]=(total,item['rows']); total+=item['rows']
    ts_parts=[]; ordinal_parts=[]
    for batch in pq.ParquetFile(parent/'clean_quotes.parquet').iter_batches(
            batch_size=700_000,columns=['timestamp','source_file','source_row']):
        frame=batch.to_pandas(); ordinal=np.empty(len(frame),dtype='int64')
        for name,group in frame.groupby('source_file',sort=False):
            base,count=offsets[name]
            assert group.source_row.between(0,count-1).all()
            ordinal[group.index.to_numpy()]=group.source_row.to_numpy()+base
        ts_parts.append(ns(frame.timestamp));ordinal_parts.append(ordinal)
    ts=np.concatenate(ts_parts); rid=np.concatenate(ordinal_parts)
    del ts_parts,ordinal_parts
    assert len(ts)>0 and (np.diff(ts)>0).all() and (np.diff(rid)>0).all()
    clean_hours=(ts-origin)//H
    assert allowed[clean_hours].all()
    blocked_prefix=np.r_[0,np.cumsum(~allowed)]
    cross=blocked_prefix[clean_hours[1:]]-blocked_prefix[clean_hours[:-1]+1] > 0
    removed=np.diff(rid)>1
    breaks=np.flatnonzero(cross | removed)+1
    expected_starts=np.r_[0,breaks]; expected_stops=np.r_[breaks,len(ts)]
    seg=pd.read_parquet(run/'segments.parquet')
    np.testing.assert_array_equal(seg.segment_id,np.arange(len(expected_starts)))
    np.testing.assert_array_equal(seg.quote_start,expected_starts)
    np.testing.assert_array_equal(seg.quote_stop_exclusive,expected_stops)
    np.testing.assert_array_equal(seg.quote_count,expected_stops-expected_starts)
    np.testing.assert_array_equal(ns(seg.start),ts[expected_starts])
    np.testing.assert_array_equal(ns(seg.last_quote),ts[expected_stops-1])

    requests=set((rid[:-1][removed]+1).tolist())
    if rid[-1] < total-1: requests.add(int(rid[-1]+1))
    wanted=np.array(sorted(requests),dtype='int64'); found={}; raw_counts=np.zeros(len(hour),dtype='int64')
    raw_first=np.full(len(hour),np.iinfo('int64').max,dtype='int64')
    raw_last=np.full(len(hour),np.iinfo('int64').min,dtype='int64')
    offset=0; previous=None
    for batch in pq.ParquetFile(parent/'raw_quotes.parquet').iter_batches(batch_size=800_000,columns=['timestamp']):
        raw=ns(batch.column(0).to_pandas()); assert (np.diff(raw)>=0).all()
        if previous is not None: assert raw[0]>=previous
        lo,hi=np.searchsorted(rid,[offset,offset+len(raw)])
        np.testing.assert_array_equal(raw[rid[lo:hi]-offset],ts[lo:hi])
        h=(raw-origin)//H
        raw_counts+=np.bincount(h,minlength=len(hour))
        first=np.r_[0,np.flatnonzero(np.diff(h))+1];last=np.r_[first[1:]-1,len(raw)-1]
        raw_first[h[first]]=np.minimum(raw_first[h[first]],raw[first])
        raw_last[h[last]]=np.maximum(raw_last[h[last]],raw[last])
        ids=wanted[(wanted>=offset)&(wanted<offset+len(raw))]
        found.update(zip(ids.tolist(),raw[ids-offset].tolist()))
        offset+=len(raw);previous=int(raw[-1])
    assert offset==total and len(found)==len(wanted)
    np.testing.assert_array_equal(raw_counts,hour.rows)
    np.testing.assert_array_equal(raw_first[allowed],ns(hour.loc[allowed,'first_timestamp']))
    np.testing.assert_array_equal(raw_last[allowed],ns(hour.loc[allowed,'last_timestamp']))
    for i,row in seg.iterrows():
        a,b=int(row.quote_start),int(row.quote_stop_exclusive)
        last_h=int(clean_hours[b-1]); next_h=last_h+1
        while next_h<len(hour) and allowed[next_h]: next_h+=1
        end=min(origin+next_h*H, previous+1, found.get(int(rid[b-1]+1),previous+1))
        assert row.end_exclusive.value==end
        removed_count=int(rid[a]-(rid[a-1]+1)) if a else int(rid[a])
        left_h=int(clean_hours[a-1]+1) if a else 0
        between=hour.iloc[left_h:int(clean_hours[a])]
        n_absent=int((between.nominal_weekly_open & between.rows.eq(0)).sum())
        n_closed=int((~between.nominal_weekly_open).sum())
        assert (row.removed_raw_records_before,row.absent_open_hours_before,row.weekly_closed_hours_before)==(removed_count,n_absent,n_closed)
        labels=[]
        if a==0: labels.append('sample_start')
        if a and n_closed: labels.append('weekly_closure')
        if a and n_absent: labels.append('absent_nominal_open_hour')
        if removed_count: labels.append('removed_raw_record')
        assert row.start_reason=='|'.join(labels)

    support=pd.read_parquet(run/'support_intervals.parquet')
    policies={'structural_only':None,'fresh_15s':15,'fresh_30s':30,'fresh_60s':60}
    assert set(support.policy)==set(policies)
    delta=np.diff(ts)
    for policy,fresh in policies.items():
        part=support.loc[support.policy.eq(policy)]
        cuts=np.zeros(len(ts),dtype=bool);cuts[expected_starts]=True
        if fresh is not None: cuts[1:] |= delta>fresh*S
        starts=np.flatnonzero(cuts);stops=np.r_[starts[1:],len(ts)]
        np.testing.assert_array_equal(part.quote_start,starts)
        np.testing.assert_array_equal(part.quote_stop_exclusive,stops)
        np.testing.assert_array_equal(ns(part.start),ts[starts])
        sid=np.searchsorted(expected_stops,starts,side='right')
        np.testing.assert_array_equal(part.segment_id,sid)
        ends=ns(seg.end_exclusive)[sid]
        if fresh is not None: ends=np.minimum(ends,ts[stops-1]+fresh*S+1)
        np.testing.assert_array_equal(ns(part.end_exclusive),ends)
        assert (ns(part.start)<ends).all() and (ns(part.start)[1:]>=ends[:-1]).all()
    stats=json.loads((run/'statistics.json').read_text())
    eligible=np.ones(len(delta),dtype=bool);eligible[breaks-1]=False
    expected={'retained_quotes':len(ts),'potential_adjacent_quote_transitions':len(delta),
        'structural_segments':len(seg),'structurally_eligible_transitions':int(eligible.sum()),
        'structurally_excluded_transitions':len(breaks),'initial_quotes_without_predecessor':1}
    for fresh in (15,30,60):
        expected[f'within_segment_gaps_gt_{fresh}s']=int(((delta>fresh*S)&eligible).sum())
        expected[f'transitions_with_structural_support_and_gap_le_{fresh}s']=int(((delta<=fresh*S)&eligible).sum())
    assert stats['quote_transitions']==expected
    gaps=pd.read_parquet(run/'long_interquote_gaps.parquet'); ix=np.flatnonzero(delta>15*S)
    np.testing.assert_array_equal(gaps.previous_quote_id,ix)
    np.testing.assert_array_equal(gaps.quote_id,ix+1)
    np.testing.assert_array_equal(ns(gaps.previous_timestamp),ts[ix])
    np.testing.assert_array_equal(ns(gaps.timestamp),ts[ix+1])
    np.testing.assert_array_equal(gaps.delta_sec,delta[ix]/S)
    np.testing.assert_array_equal(gaps.crosses_structural_boundary,~eligible[ix])
    del ts,rid,delta,eligible

    windows=pd.read_parquet(run/'window_attrition_by_hour.parquet')
    assert len(windows)==len(hour)*4*3*6
    summaries=[]
    for policy in policies:
        part=support.loc[support.policy.eq(policy)]
        for lookback in (0,60,300):
            past=reference_hour_counts(ns(part.start),ns(part.end_exclusive),lookback,0,origin//S,len(hour))
            for horizon in (0,1,5,30,60,300):
                counts=reference_hour_counts(ns(part.start),ns(part.end_exclusive),lookback,horizon,origin//S,len(hour))
                rows=windows.loc[(windows.policy==policy)&(windows.lookback_seconds==lookback)&(windows.horizon_seconds==horizon)]
                np.testing.assert_array_equal(ns(rows.hour),hns)
                np.testing.assert_array_equal(rows.eligible_clock_anchors,counts)
                assert (counts>=0).all() and (counts<=3600).all() and (counts<=past).all()
                assert not counts[~allowed].any()
                summaries.append({'policy':policy,'lookback_seconds':lookback,'horizon_seconds':horizon,
                    'pre_context_clock_anchors':int(past.sum()),'joint_clock_anchors':int(counts.sum()),
                    'future_unavailable_among_pre_eligible':int((past-counts).sum())})
    assert stats['window_opportunities']==summaries
    assert stats['coverage']['raw_rows']==offset
    assert stats['coverage']['nominal_open_hours']==sum(calendar)
    assert stats['coverage']['observed_hours']==int(allowed.sum())
    assert stats['coverage']['absent_nominal_open_hours']==int((~allowed & np.array(calendar)).sum())
    composition=windows.assign(month=windows.hour.dt.strftime('%Y-%m'),
        utc_six_hour_band=(windows.hour.dt.hour//6)*6).groupby(
        ['policy','lookback_seconds','horizon_seconds','month','utc_six_hour_band'],as_index=False).eligible_clock_anchors.sum()
    pd.testing.assert_frame_equal(composition,pd.read_parquet(run/'window_attrition_by_month_band.parquet'))
    return {'status':'passed','retained_quotes_checked':expected['retained_quotes'],
        'raw_timestamps_checked':offset,'structural_segments_checked':len(seg),
        'support_intervals_checked':len(support),'hourly_window_counts_checked':len(windows),
        'past_and_future_window_scenarios':len(summaries),
        'calendar_checked_independently':True,'all_clean_time_provenance_pairs_checked_against_raw':True,
        'coverage_certified':False,'scope':'Observation-support arithmetic and cached-data provenance; no event/model validation'}

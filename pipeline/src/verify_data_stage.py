"""Independent full-record verification of serialized stage-01 artifacts.

Does not call the cleaner or source loader: membership is derived independently
from the original source tuples and checked through row provenance.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data_preparation import atomic_json, sha256, exact_median


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def verify_run(run: Path) -> dict:
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest['status'] in ['built_awaiting_independent_verification','verified'], 'Run did not finish building')
    stats = dict(raw_rows=0, clean_rows=0, invalid_size_rows=0, tied_timestamp_rows=0,
                 zero_duration_rows=0, gap_gt_60s_rows=0, sunday_rows=0,
                 missing_duration_rows=0, month_boundaries_checked=0)
    source_cols = ['timestamp','ask','bid','ask_volume','bid_volume']
    extra_cols = ['source_file','source_row','spread','mid','spread_pips','delta_sec','quote_size_valid','timestamp_tied']
    require(set(pq.ParquetFile(run/'clean_quotes.parquet').schema_arrow.names)==set(source_cols+extra_cols), 'Clean schema differs')
    require(set(pq.ParquetFile(run/'raw_quotes.parquet').schema_arrow.names)==set(source_cols+extra_cols[:2]), 'Raw schema differs')
    prev = None; first = None; dates=set(); hours=set(); spread_hist=Counter(); duration_hist=Counter()
    removed_totals=Counter(); size_distinct={'ask_volume':set(),'bid_volume':set()}
    try:
        for i,item in enumerate(manifest['input_files']):
            path=Path(item['path'])
            require(sha256(path)==item['sha256'], f'Source changed: {path}')
            source=pd.read_parquet(path,columns=source_cols)
            source.timestamp=source.timestamp.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
            for col in source_cols[1:]:source[col]=source[col].astype('float64')
            raw=pd.read_parquet(run/'raw_quotes.parquet',filters=[('source_file','==',item['name'])])
            clean=pd.read_parquet(run/'clean_quotes.parquet',filters=[('source_file','==',item['name'])])
            require(len(raw)==len(source)==item['rows'], f'Raw row count: {path.name}')
            require(np.array_equal(raw.source_row,np.arange(len(source))), 'Raw row provenance')
            pd.testing.assert_frame_equal(raw[source_cols].reset_index(drop=True), source.reset_index(drop=True),check_exact=True)

            # independently enumerate the admissible source tuples. Valid tied
            # timestamps and size-only changes are distinct observations.
            admissible=(np.isfinite(source.ask)&np.isfinite(source.bid)&(source.ask>source.bid)&(source.bid>0))
            selected=source.loc[admissible]
            selected=selected.loc[~selected.duplicated(source_cols,keep='first')]
            rows=selected.index.to_numpy()
            require(np.array_equal(clean.source_row,rows),f'Clean membership/order: {path.name}')
            pd.testing.assert_frame_equal(clean[source_cols].reset_index(drop=True), selected.reset_index(drop=True),check_exact=True)
            require(clean.timestamp.is_monotonic_increasing,'Clean chronological order')
            expected_spread=clean.ask.to_numpy()-clean.bid.to_numpy()
            require(np.array_equal(clean.spread,expected_spread),'Spread identity')
            require(np.array_equal(clean.mid,(clean.ask.to_numpy()+clean.bid.to_numpy())/2),'Midpoint identity')
            require(np.array_equal(clean.spread_pips,expected_spread*10000),'Pip conversion')
            ts=clean.timestamp
            # use integer timestamp differences, independently of the cleaner's
            # pandas duration operation. Subtract before conversion to float.
            time_ns=ts.array.asi8
            expected_delta=np.full(len(ts),np.nan,dtype=float)
            expected_delta[1:]=np.diff(time_ns)/1_000_000_000
            if len(clean) and prev is not None:
                require(ts.iloc[0]>prev,'Nonchronological month boundary')
                expected_delta[0]=(int(time_ns[0])-prev.value)/1_000_000_000
                stats['month_boundaries_checked']+=1
            require(np.array_equal(clean.delta_sec,expected_delta,equal_nan=True),'Duration identity including boundary')
            good_size=(np.isfinite(clean.ask_volume)&np.isfinite(clean.bid_volume)&(clean.ask_volume>=0)&(clean.bid_volume>=0))
            require(np.array_equal(clean.quote_size_valid,good_size),'Size validity flag')
            require(np.array_equal(clean.timestamp_tied,ts.duplicated(keep=False)),'Tie flag')
            log=manifest['monthly_cleaning'][i]
            require(log['raw_rows']==len(source) and log['clean_rows']==len(clean),'Cleaning log counts')
            finite=np.isfinite(source.ask)&np.isfinite(source.bid)
            positive=finite&(source.ask>0)&(source.bid>0)
            expected_removals={
                'nonfinite_price':int((~finite).sum()),
                'nonpositive_price':int((finite&~positive).sum()),
                'crossed_quote':int((positive&(source.ask<source.bid)).sum()),
                'zero_spread':int((positive&(source.ask==source.bid)).sum()),
                'duplicate_full_snapshot':int(admissible.sum()-len(selected)),
            }
            require(log['removed']==expected_removals,'Independently computed removal reasons')
            require(sum(log['removed'].values())==len(source)-len(clean),'Disjoint removal conservation')
            removed_totals.update(log['removed'])
            stats['raw_rows']+=len(raw);stats['clean_rows']+=len(clean)
            stats['invalid_size_rows']+=int((~good_size).sum())
            stats['tied_timestamp_rows']+=int(clean.timestamp_tied.sum())
            stats['zero_duration_rows']+=int((clean.delta_sec==0).sum())
            stats['gap_gt_60s_rows']+=int((clean.delta_sec>60).sum())
            stats['missing_duration_rows']+=int(clean.delta_sec.isna().sum())
            stats['sunday_rows']+=int((ts.dt.dayofweek==6).sum())
            dates.update(ts.dt.date.unique());hours.update(ts.dt.floor('h').unique())
            spread_hist.update(clean.spread_pips.value_counts().to_dict())
            duration_hist.update(clean.delta_sec.dropna().value_counts().to_dict())
            for col in size_distinct:size_distinct[col].update(clean.loc[good_size,col].unique())
            if len(clean):
                if first is None:first=ts.iloc[0]
                prev=ts.iloc[-1]
            print(f'Verified every stored and retained row: {path.name}',flush=True)
        require(stats['raw_rows']==pq.ParquetFile(run/'raw_quotes.parquet').metadata.num_rows,'Extra unverified raw rows')
        require(stats['clean_rows']==pq.ParquetFile(run/'clean_quotes.parquet').metadata.num_rows,'Extra unverified clean rows')
        require(stats['clean_rows']>0 and stats['missing_duration_rows']==1,'Exactly one initial missing duration')
        require(stats['raw_rows']-stats['clean_rows']==sum(removed_totals.values()),'Total row conservation')
        stats.update(first_timestamp=str(first),last_timestamp=str(prev),removed_by_reason=dict(removed_totals),
            utc_dates_with_quotes=len(dates),utc_hours_with_quotes=len(hours),
            mean_spread_pips=sum(v*n for v,n in spread_hist.items())/stats['clean_rows'],
            median_spread_pips=exact_median(spread_hist),
            mean_delta_sec=(sum(v*n for v,n in duration_hist.items())/sum(duration_hist.values())
                            if duration_hist else None),
            median_delta_sec=exact_median(duration_hist),
            distinct_valid_sizes={k:len(v) for k,v in size_distinct.items()})
        result={'status':'passed','verification_scope':'all source/raw/clean records; cached-input fidelity, not vendor completeness',
            'verifier_sha256':sha256(Path(__file__)),'statistics':stats,
            'outputs':{n:{'sha256':sha256(run/n),'bytes':(run/n).stat().st_size} for n in ['raw_quotes.parquet','clean_quotes.parquet']}}
        atomic_json(run/'verification.json',result)
        manifest['status']='verified';manifest['outputs']=result['outputs']
        atomic_json(run/'manifest.json',manifest)
        return result
    except Exception as error:
        atomic_json(run/'verification.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        manifest['status']='verification_failed'
        atomic_json(run/'manifest.json',manifest)
        raise

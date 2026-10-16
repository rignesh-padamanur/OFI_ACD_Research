"""Recorded best-quote imbalance; not identified transaction flow.

Cont, Kukanov & Stoikov, https://arxiv.org/abs/1011.6402, section 2.1.
Sizes stay in their stored units. No spread scaling, filling, or calendar filter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SECOND = 1_000_000_000
POLICIES = {'structural_only': None, 'fresh_15s': 15, 'fresh_30s': 30, 'fresh_60s': 60}
INPUT = ['quote_id','timestamp','segment_id','bid','ask','bid_volume','ask_volume']
NO_PREDECESSOR, STRUCTURAL_RESET = 1, 2
CURRENT_PRICE_INVALID, PREVIOUS_PRICE_INVALID = 4, 8
CURRENT_SIZE_INVALID, PREVIOUS_SIZE_INVALID = 16, 32


def compute_ofi(*args, **kwargs):
    raise RuntimeError('Legacy price-sign/spread OFI is invalid. Use compute_quote_imbalance with explicit quote IDs and structural segments, or repair_quote_imbalance.py.')


def compute_quote_imbalance(frame, *, previous=None):
    """One output per input quote; an invalid endpoint invalidates its transition.

    ``previous`` is the immediately preceding cached quote, even if invalid.
    No row is skipped. Structural starts reset the difference to unavailable.
    Tied timestamps fail explicitly; recorded source order is not market order.
    """
    if set(INPUT)-set(frame): raise ValueError('Missing quote prices, sizes, time or structural provenance')
    if frame.empty: raise ValueError('Empty quote chunk')
    current=frame[INPUT].reset_index(drop=True)
    joined=pd.concat([pd.DataFrame([previous])[INPUT],current],ignore_index=True) if previous is not None else current
    ix=pd.DatetimeIndex(joined.timestamp)
    if ix.tz is None or ix.hasnans: raise ValueError('Explicit nonmissing timezone-aware timestamps required')
    ts=ix.tz_convert('UTC').as_unit('ns').asi8
    if (np.diff(ts)<=0).any(): raise ValueError('Tied or decreasing timestamps require a separate ordering policy')
    ids=joined.quote_id.to_numpy()
    sid=joined.segment_id.to_numpy()
    if not pd.api.types.is_integer_dtype(joined.quote_id.dtype) or (ids<0).any() or (np.diff(ids)!=1).any():
        raise ValueError('Quote IDs must be consecutive nonnegative integers')
    if not pd.api.types.is_integer_dtype(joined.segment_id.dtype) or (sid<0).any() or (np.diff(sid)<0).any():
        raise ValueError('Segment IDs must be nonnegative ordered integers')
    n=len(joined)
    price=joined[['bid','ask']].to_numpy(dtype=float)
    size=joined[['bid_volume','ask_volume']].to_numpy(dtype=float)
    price_valid=np.isfinite(price).all(axis=1)&(price[:,0]>0)&(price[:,1]>price[:,0])
    size_valid=np.isfinite(size).all(axis=1)&(size>=0).all(axis=1)
    code=np.zeros(n,dtype='uint8');code[0]=NO_PREDECESSOR
    code[1:] |= np.where(sid[1:]!=sid[:-1],STRUCTURAL_RESET,0).astype('uint8')
    code[~price_valid] |= CURRENT_PRICE_INVALID
    code[1:] |= np.where(~price_valid[:-1],PREVIOUS_PRICE_INVALID,0).astype('uint8')
    code[~size_valid] |= CURRENT_SIZE_INVALID
    code[1:] |= np.where(~size_valid[:-1],PREVIOUS_SIZE_INVALID,0).astype('uint8')
    bid_move=np.zeros(n,dtype='int8');ask_move=bid_move.copy()
    price_pairs=price_valid[1:]&price_valid[:-1]
    bid_move[1:]=np.where(price_pairs,np.sign(price[1:,0]-price[:-1,0]),0).astype('int8')
    ask_move[1:]=np.where(price_pairs,np.sign(price[1:,1]-price[:-1,1]),0).astype('int8')
    bid_pressure=np.full(n,np.nan);ask_pressure=bid_pressure.copy()
    with np.errstate(invalid='ignore'):
        bid_pressure[1:]=np.where(bid_move[1:]>0,size[1:,0],
            np.where(bid_move[1:]<0,-size[:-1,0],size[1:,0]-size[:-1,0]))
        ask_pressure[1:]=np.where(ask_move[1:]>0,size[:-1,1],
            np.where(ask_move[1:]<0,-size[1:,1],size[:-1,1]-size[1:,1]))
    bid_pressure[code!=0]=np.nan;ask_pressure[code!=0]=np.nan
    delta=np.r_[np.nan,np.diff(ts)/SECOND]
    result=pd.DataFrame({'quote_id':ids.astype('int64'),'timestamp':pd.to_datetime(ts,utc=True),
        'segment_id':sid.astype('int64'),'delta_sec':delta,'exclusion_code':code,
        'bid_move':bid_move,'ask_move':ask_move,'bid_pressure':bid_pressure,
        'ask_pressure':ask_pressure,'quote_ofi':bid_pressure+ask_pressure})
    return result.iloc[1 if previous is not None else 0:].reset_index(drop=True)


def prepare_support(support):
    if set(support.policy.unique())!=set(POLICIES): raise ValueError('All four diagnostic policies are required')
    result={}
    for name in POLICIES:
        part=support.loc[support.policy.eq(name)]
        a=pd.DatetimeIndex(part.start).as_unit('ns').asi8
        b=pd.DatetimeIndex(part.end_exclusive).as_unit('ns').asi8
        if not len(a) or (a>=b).any() or (a[1:]<b[:-1]).any(): raise ValueError('Invalid support intervals')
        result[name]=(a,b)
    return result


def add_bin_support(bins, support):
    """Retrospective audit flags, never explanatory market-state variables."""
    left=pd.DatetimeIndex(bins.bin_start).as_unit('ns').asi8;right=left+SECOND
    complete=bins.n_invalid_transitions.eq(0).to_numpy()
    for policy,freshness in POLICIES.items():
        a,b=support[policy];idx=np.searchsorted(a,left,side='right')-1;j=np.maximum(idx,0)
        covered=(idx>=0)&(b[j]>=right)
        bins[f'full_bin_{policy}']=covered
        gap_ok=np.ones(len(bins),dtype=bool) if freshness is None else bins.max_incoming_gap_sec.le(freshness).to_numpy()
        bins[f'usable_{policy}']=covered&complete&gap_ok
    return bins


def aggregate_observed_bins(ticks, support):
    """Only quote-containing [s,s+1s) bins; no silent empty-bin zero filling.

    quote_ofi is NaN if any included transition is excluded. observed_subtotal
    remains available for diagnostics when at least one transition is valid.
    All bin-derived signals are available at s+1s, never at s.
    """
    group_key=ticks.timestamp.dt.floor('s')
    work=ticks.assign(valid=ticks.exclusion_code.eq(0).astype('int64'))
    grouped=work.groupby(group_key,sort=True)
    bins=grouped.agg(n_quotes=('quote_id','size'),n_valid_transitions=('valid','sum'),
        max_incoming_gap_sec=('delta_sec','max'),segment_first=('segment_id','first'),segment_last=('segment_id','last'))
    bins['observed_subtotal']=grouped.quote_ofi.sum(min_count=1)
    bins['n_invalid_transitions']=bins.n_quotes-bins.n_valid_transitions
    bins['quote_ofi']=bins.observed_subtotal.where(bins.n_invalid_transitions.eq(0))
    bins['segment_id']=np.where(bins.segment_first==bins.segment_last,bins.segment_first,-1)
    bins=bins.drop(columns=['segment_first','segment_last']).rename_axis('bin_start').reset_index()
    bins.insert(1,'available_at',pd.to_datetime(pd.DatetimeIndex(bins.bin_start).as_unit('ns').asi8+SECOND,utc=True))
    return add_bin_support(bins,support)


class BinStream:
    """Hold the unfinished last second across arbitrary input chunk boundaries."""
    def __init__(self,support): self.support=support;self.pending=None

    def push(self,ticks,*,final=False):
        if self.pending is not None: ticks=pd.concat([self.pending,ticks],ignore_index=True)
        if ticks.empty: return None
        if final: ready=ticks;self.pending=None
        else:
            last=ticks.timestamp.iloc[-1].floor('s');cut=int(ticks.timestamp.searchsorted(last,side='left'))
            ready=ticks.iloc[:cut];self.pending=ticks.iloc[cut:].copy()
        return aggregate_observed_bins(ready,self.support) if len(ready) else None

    def finish(self):
        if self.pending is None:return None
        ready=self.pending;self.pending=None
        return aggregate_observed_bins(ready,self.support)

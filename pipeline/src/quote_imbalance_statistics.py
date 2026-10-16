"""Descriptive statistics only; none are used to fit or normalize signals."""
from collections import Counter
import math
import numpy as np
import pandas as pd
from src.ofi import POLICIES


class Moments:
    def __init__(self):self.n=0;self.mean=0.;self.m2=0.;self.minimum=None;self.maximum=None
    def add(self,values):
        x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
        if not len(x):return
        n=len(x);mean=float(x.mean());m2=float(np.square(x-mean).sum())
        total=self.n+n;delta=mean-self.mean
        self.m2+=m2+delta*delta*self.n*n/total;self.mean+=delta*n/total;self.n=total
        low=float(x.min());high=float(x.max())
        self.minimum=low if self.minimum is None else min(self.minimum,low)
        self.maximum=high if self.maximum is None else max(self.maximum,high)
    def result(self):return {'count':self.n,'mean':self.mean if self.n else None,
        'sample_std':math.sqrt(self.m2/(self.n-1)) if self.n>1 else None,'minimum':self.minimum,'maximum':self.maximum}


class SignalStatistics:
    def __init__(self):
        self.ticks=Counter();self.codes=Counter();self.cases=Counter();self.comparison=Counter()
        self.bins=Counter();self.months={};self.tick_moments={p:Moments() for p in POLICIES};self.bin_moments={p:Moments() for p in POLICIES}
    def add_ticks(self,frame):
        valid=frame.exclusion_code.eq(0).to_numpy();x=frame.quote_ofi.to_numpy()
        self.ticks.update({'quotes':len(frame),'eligible_transitions':int(valid.sum()),'excluded_transitions':int((~valid).sum()),
            'sunday_quotes':int((frame.timestamp.dt.dayofweek==6).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum()),'positive':int((x>0).sum())})
        for k,n in frame.exclusion_code.value_counts().items():self.codes[str(k)]+=int(n)
        bid=frame.bid_move.to_numpy()[valid];ask=frame.ask_move.to_numpy()[valid];flow=x[valid];legacy=np.sign(bid-ask)
        for b in (-1,0,1):
            for a in (-1,0,1):self.cases[f'bid_{b}_ask_{a}']+=int(((bid==b)&(ask==a)).sum())
        self.comparison.update({'both_zero':int(((legacy==0)&(flow==0)).sum()),
            'legacy_zero_corrected_nonzero':int(((legacy==0)&(flow!=0)).sum()),
            'legacy_nonzero_corrected_zero':int(((legacy!=0)&(flow==0)).sum()),
            'both_nonzero_same_sign':int(((legacy!=0)&(flow!=0)&(legacy==np.sign(flow))).sum()),
            'both_nonzero_opposite_sign':int(((legacy!=0)&(flow!=0)&(legacy!=np.sign(flow))).sum()),
            'unchanged_prices_nonzero_size_flow':int(((bid==0)&(ask==0)&(flow!=0)).sum())})
        for p,fresh in POLICIES.items():
            mask=valid if fresh is None else valid&frame.delta_sec.le(fresh).to_numpy()
            self.tick_moments[p].add(x[mask])
        months=pd.DatetimeIndex(frame.timestamp).tz_localize(None).to_numpy().astype('datetime64[M]')
        for m in np.unique(months):
            mask=months==m; key=str(m);row=self.months.setdefault(key,Counter())
            row.update({'quotes':int(mask.sum()),'eligible_transitions':int((valid&mask).sum()),
                'excluded_transitions':int((~valid&mask).sum()),'positive_tick_ofi':int(((x>0)&mask).sum()),
                'negative_tick_ofi':int(((x<0)&mask).sum()),'zero_tick_ofi':int(((x==0)&mask).sum())})
    def add_bins(self,frame):
        self.bins.update({'quote_containing_bins':len(frame),'quotes_accounted_for':int(frame.n_quotes.sum()),
            'bins_with_excluded_transitions':int(frame.n_invalid_transitions.gt(0).sum()),
            'bins_with_no_valid_transition':int(frame.n_valid_transitions.eq(0).sum()),
            'bins_with_multiple_structural_segments':int(frame.segment_id.eq(-1).sum()),
            'zero_complete_observed_sums':int(frame.quote_ofi.eq(0).sum())})
        months=pd.DatetimeIndex(frame.bin_start).tz_localize(None).to_numpy().astype('datetime64[M]')
        for p in POLICIES:
            usable=frame[f'usable_{p}'].to_numpy()
            self.bins[f'full_bin_{p}']+=int(frame[f'full_bin_{p}'].sum())
            self.bins[f'usable_{p}']+=int(usable.sum())
            self.bin_moments[p].add(frame.loc[usable,'quote_ofi'])
            for m in np.unique(months):
                self.months.setdefault(str(m),Counter())[f'usable_bins_{p}']+=int((usable&(months==m)).sum())
        for m in np.unique(months):self.months.setdefault(str(m),Counter())['quote_containing_bins']+=int((months==m).sum())
    def result(self):return {'ticks':dict(self.ticks),'transition_exclusion_codes':dict(self.codes),
        'price_move_cases':dict(self.cases),'legacy_sign_comparison_on_common_transitions':dict(self.comparison),
        'tick_ofi_descriptive_by_gap_policy':{p:v.result() for p,v in self.tick_moments.items()},
        'bins':dict(self.bins),'bin_ofi_descriptive_by_policy':{p:v.result() for p,v in self.bin_moments.items()},
        'monthly':{k:dict(v) for k,v in sorted(self.months.items())},
        'units':'stored_quote_size_units','primary_freshness_policy_selected':False,
        'event_detection_recomputed':False,'persistence_or_regression_recomputed':False,
        'coverage_certified':False,'new_quotes_acquired':0}

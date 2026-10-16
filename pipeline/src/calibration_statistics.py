"""Descriptive diagnostics for sequential calibration, never used to tune it."""
from collections import Counter
import numpy as np
import pandas as pd
from src.deseasonal import POLICY_NAMES, CHANNELS, STATUS, slots
from src.quote_imbalance_statistics import Moments


class CalibrationStatistics:
    def __init__(self):
        self.days = 0; self.rows = 0; self.counters = {}; self.moments = {}; self.months = {}
        self.cells = np.zeros((4,2,7,96,3), dtype=float)
        self.fit_status = {p: {v: Counter() for v in CHANNELS} for p in POLICY_NAMES}
        self.first_ready = {p: {v: None for v in CHANNELS} for p in POLICY_NAMES}
        for p in POLICY_NAMES:
            for v in CHANNELS:
                self.counters[p,v] = Counter(); self.moments[p,v] = Moments()

    def add(self, day, frame, normalized, fit):
        self.days += 1; self.rows += len(frame); cell = slots(frame)
        month = day.strftime('%Y-%m'); weekday = day.dayofweek
        for pi,p in enumerate(POLICY_NAMES):
            code = frame[f'flow_status_{p}'].to_numpy()
            for vi,v in enumerate(CHANNELS):
                eligible = code != 0 if vi == 0 else (code == 1) | (code == 2)
                fit_code = fit['status'][pi,vi,cell]
                z = normalized[f'z_{v}_{p}'].to_numpy(); valid = np.isfinite(z)
                if not np.array_equal(valid, eligible & (fit_code==5)):
                    raise AssertionError('Output eligibility differs from input and fitted status')
                row = {'clock_seconds': len(frame), 'input_eligible_seconds': int(eligible.sum()),
                       'unavailable_input_seconds': int((~eligible).sum()), 'calibrated_seconds': int(valid.sum())}
                for number,label in STATUS.items():
                    row['eligible_'+label] = int((eligible & (fit_code==number)).sum())
                    self.fit_status[p][v][label] += int((fit['status'][pi,vi]==number).sum())
                self.counters[p,v].update(row); self.moments[p,v].add(z[valid])
                key = (month,p,v)
                if key not in self.months:
                    self.months[key] = (Counter(), Moments())
                self.months[key][0].update(row); self.months[key][1].add(z[valid])
                if valid.any() and self.first_ready[p][v] is None:
                    self.first_ready[p][v] = frame.available_at.iloc[int(np.flatnonzero(valid)[0])].isoformat()
                s = cell[valid]; x = z[valid]
                self.cells[pi,vi,weekday,:,0] += np.bincount(s,minlength=96)
                self.cells[pi,vi,weekday,:,1] += np.bincount(s,weights=x,minlength=96)
                self.cells[pi,vi,weekday,:,2] += np.bincount(s,weights=x*x,minlength=96)

    def tables(self):
        monthly = pd.DataFrame([{'month':m,'policy':p,'channel':v,**dict(c),'z_mean':mom.result()['mean'],
                                'z_sample_std':mom.result()['sample_std']} for (m,p,v),(c,mom) in sorted(self.months.items())])
        rows = []
        for pi,p in enumerate(POLICY_NAMES):
            for vi,v in enumerate(CHANNELS):
                for w in range(7):
                    for cell in range(96):
                        n,s,q = self.cells[pi,vi,w,cell]
                        rows.append({'policy':p,'channel':v,'weekday':w,'slot':cell,'n':int(n),
                            'z_mean':s/n if n else np.nan,
                            'z_sample_std':np.sqrt(max(0.,q-s*s/n)/(n-1)) if n>1 else np.nan})
        return monthly, pd.DataFrame(rows)

    def result(self):
        return {'clock_rows': self.rows, 'utc_days': self.days,
            'channels': {p:{v:{'counts':dict(self.counters[p,v]),'z_descriptive':self.moments[p,v].result(),
                               'first_calibrated_available_at': self.first_ready[p][v],
                               'parameter_cell_status_counts':dict(self.fit_status[p][v])} for v in CHANNELS} for p in POLICY_NAMES},
            'primary_freshness_policy':None,'coverage_certified':False,'downstream_ready':False,
            'acd_or_events_reestimated':False,'new_quotes_acquired':0}

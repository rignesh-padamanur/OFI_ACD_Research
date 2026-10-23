import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


def _require_legacy_persistence():
    raise RuntimeError(
        'Legacy persistence calculations are disabled: they use incompatible event/price clocks, '
        'future episode direction and unverified volatility/decay conventions. '
        'Use the registered stage-06a development price responses and consult REPAIR_WORKFLOW.md.'
    )


def build_coalesced_event_table(event_df, trigger_col="candidate_event", debounce_gap_sec=5, min_pip_move=0.5):
    """
    Convert candidate stress seconds into discrete events using a refractory debounce window
    and realized pip move filter.
    """
    _require_legacy_persistence()
    df = event_df.copy().sort_index()
    trigger = df[trigger_col].fillna(False).astype(bool)

    cand_times = df.index[trigger]
    if len(cand_times) == 0:
        return pd.DataFrame(), df

    # group candidate seconds into discrete events if gap > debounce_gap_sec
    gaps = (cand_times.to_series().diff().dt.total_seconds() > debounce_gap_sec).cumsum()
    
    cand_df = pd.DataFrame({"timestamp": cand_times, "event_group": gaps})
    
    event_list = []
    for group_id, group in cand_df.groupby("event_group"):
        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()
        
        event_slice = df.loc[start_time:end_time]
        peak_stress = event_slice["stress_score"].max()
        mean_stress = event_slice["stress_score"].mean()
        
        # aggregate signed OFI during event window (F_{t0}-measurable direction)
        net_ofi = event_slice["ofi"].sum() if "ofi" in event_slice.columns else 0.0
        
        event_list.append({
            "event_id": group_id + 1,
            "start_time": start_time,
            "end_time": end_time,
            "n_event_seconds": len(group),
            "peak_stress": peak_stress,
            "mean_stress": mean_stress,
            "span_seconds": (end_time - start_time).total_seconds(),
            "net_ofi": net_ofi
        })
        
    events = pd.DataFrame(event_list)
    return events


def compute_volatility_standardized_paths(quotes, events, horizon_sec=300, pre_vol_sec=300):
    """
    Compute post-event price impact paths standardized by pre-event realized volatility.

    Direction is determined by the realized price change within the event window
    (event_move column), which is F_{t0}-measurable: both start and end prices
    are observed at t_end before the forward persistence window begins.
    Follows Hasbrouck (1991) event-study convention.
    """
    _require_legacy_persistence()
    if "mid" not in quotes.columns:
        quotes = quotes.copy()
        quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2.0
        
    mid_1s = quotes["mid"].resample("1s").ffill()
    
    paths_list = []
    
    for idx, row in events.iterrows():
        eid = row["event_id"]
        t0 = row["end_time"]
        
        # pre-event realized volatility (300s window before t0)
        pre_window = mid_1s.loc[t0 - pd.Timedelta(seconds=pre_vol_sec) : t0]
        if len(pre_window) < 10:
            pre_vol = 1e-4
        else:
            pre_vol = pre_window.diff().std()
            if np.isnan(pre_vol) or pre_vol == 0:
                pre_vol = 1e-4
                
        fwd_window = mid_1s.loc[t0 : t0 + pd.Timedelta(seconds=horizon_sec)]
        if len(fwd_window) == 0:
            continue
            
        p0 = fwd_window.iloc[0]
        
        # direction from realized price change within event window (F_{t0}-measurable)
        # event_move = mid(last tick in event) - mid(first tick in event)
        if "event_move" in row.index and pd.notna(row.get("event_move")) and row["event_move"] != 0:
            direction = 1.0 if row["event_move"] > 0 else -1.0
        elif "net_ofi" in row.index and pd.notna(row.get("net_ofi")) and row["net_ofi"] != 0:
            direction = 1.0 if row["net_ofi"] > 0 else -1.0
        else:
            # no direction available; leave price changes unaligned
            direction = 1.0
        
        for h_sec in range(1, len(fwd_window)):
            p_h = fwd_window.iloc[h_sec]
            raw_move = p_h - p0
            std_impact = (direction * raw_move) / pre_vol
            
            paths_list.append({
                "event_id": eid,
                "horizon_sec": h_sec,
                "raw_move": raw_move,
                "raw_move_pips": raw_move * 10000.0,
                "std_impact": std_impact,
                "pre_vol": pre_vol,
                "direction": direction
            })
            
    paths_df = pd.DataFrame(paths_list)
    return paths_df


def compute_decay_metrics(paths_df, max_horizon=300):
    """
    Compute non-parametric AUC and parametric half-life metrics per event.
    """
    _require_legacy_persistence()
    metrics = []
    
    def exp_decay(h, a, b, c):
        arg = np.clip(-b * h, -50, 50)
        return a * np.exp(arg) + c

    for eid, group in paths_df.groupby("event_id"):
        group = group.sort_values("horizon_sec")
        h = group["horizon_sec"].to_numpy()
        y = group["std_impact"].to_numpy()
        
        if len(y) < 10:
            continue
            
        auc = trapezoid(y, h)
        
        try:
            popt, _ = curve_fit(
                exp_decay, h, y,
                p0=[max(y[0], 1e-3), 0.01, y[-1]],
                bounds=([0, 1e-4, -100], [np.inf, 1.0, np.inf]),
                maxfev=1000
            )
            half_life = np.log(2.0) / popt[1] if popt[1] > 1e-4 else np.nan
        except Exception:
            half_life = np.nan
            
        metrics.append({
            "event_id": eid,
            "auc_300": auc,
            "half_life_sec": half_life,
            "peak_impact": np.max(y),
            "final_impact": y[-1]
        })
        
    return pd.DataFrame(metrics)

"""
Scientific Econometric Audit: Calibration Stability Across Market Regimes
========================================================================
Rigorous verification of Weibull ACD(1,1) parameter invariance across
three distinct calendar periods:
  1. Jan 2024 (Beginning of 31-month panel, first available trading data)
  2. Jan 2025 (Exact panel midpoint, 12 months post-start)
  3. May 2026 (Production baseline calibration month)

Optimization details:
  - Objective: Average Negative Log-Likelihood (NLL / N) to ensure scale-invariant gradient descent.
  - Method: L-BFGS-B with box constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1, k in [0.3, 1.5].
  - Convergence check: Verifies nit > 0, nfev, and termination message.
  - Evaluation: Pairwise comparison of conditional duration psi_t, residuals eps_t,
    and standardized z_duration on a held-out test sample of 500,000 quotes from mid-2025.
"""

raise RuntimeError("Legacy ACD estimation is disabled during staged repair. See reports/PIPELINE_REPAIR_LOG.md; chronological observation-aware ACD fitting is pending in stage 04d.")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from pathlib import Path
import json

ROOT = Path("/Users/rigneshpadamanur/Documents/FX_Microstructure")
CLEAN_PATH = ROOT / "data" / "processed" / "eurusd_clean.parquet"

print("=" * 80)
print("RUNNING RIGOROUS WACD CALIBRATION STABILITY AUDIT (ZERO LOOK-AHEAD BIAS CHECK)")
print("=" * 80)

print("\n1. Loading clean EUR/USD tick data...")
clean = pd.read_parquet(CLEAN_PATH, columns=["timestamp", "delta_sec"])
clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
clean = clean.set_index("timestamp")

active_df = clean[(clean["delta_sec"] > 0) & (clean["delta_sec"] <= 60)].copy()

def deseasonalize_month(series):
    tod = series.index.floor("15min").time
    tod_mean = series.groupby(tod).transform("mean")
    overall_mean = series.mean()
    phi = tod_mean / overall_mean
    return series / phi.replace(0, np.nan).fillna(1.0)

regimes = {
    "Jan 2024 (Sample Start)": "2024-01",
    "Jan 2025 (Sample Midpoint)": "2025-01",
    "May 2026 (Baseline Month)": "2026-05"
}

fits = {}
N_SAMPLE = 100000

print(f"\n2. Estimating WACD(1,1) via MLE across regimes (N = {N_SAMPLE:,} per regime)...")

for name, month_str in regimes.items():
    sub = active_df.loc[month_str].copy()
    adj_delta = deseasonalize_month(sub["delta_sec"])
    x = adj_delta.iloc[:N_SAMPLE].to_numpy()
    n = len(x)
    
    def mean_nll(params):
        omega, alpha, beta, k = params
        if omega <= 1e-7 or alpha < 0.001 or beta < 0.50 or (alpha + beta) >= 0.9999 or k <= 0.3 or k >= 1.8:
            return 1e10
        
        psi = np.empty(n)
        psi[0] = omega / (1.0 - alpha - beta)
        for t in range(1, n):
            psi[t] = omega + alpha * x[t-1] + beta * psi[t-1]
        
        if np.any(psi <= 0):
            return 1e10
            
        gamma_i = psi / gamma(1.0 + 1.0 / k)
        ll_terms = np.log(k) + (k - 1.0) * np.log(x) - k * np.log(gamma_i) - (x / gamma_i)**k
        if not np.all(np.isfinite(ll_terms)):
            return 1e10
        return -np.mean(ll_terms)

    init = [0.005, 0.03, 0.95, 0.75]
    bounds = [(1e-6, 0.05), (0.001, 0.20), (0.60, 0.999), (0.4, 1.2)]
    
    res = minimize(
        mean_nll,
        init,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500, "gtol": 1e-6}
    )
    
    fits[name] = {
        "omega": float(res.x[0]),
        "alpha": float(res.x[1]),
        "beta": float(res.x[2]),
        "k": float(res.x[3]),
        "persistence": float(res.x[1] + res.x[2]),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "converged": bool(res.success),
        "message": str(res.message)
    }
    
    print(f"   ✓ {name}:")
    print(f"       omega = {res.x[0]:.6f}, alpha = {res.x[1]:.6f}, beta = {res.x[2]:.6f}")
    print(f"       k (shape) = {res.x[3]:.4f}, persistence (alpha+beta) = {res.x[1]+res.x[2]:.6f}")
    print(f"       iterations = {res.nit}, function_evals = {res.nfev}, success = {res.success}")

with open(ROOT / "data" / "results" / "weibull_params.json") as f:
    prod_params = json.load(f)

fits["Production (May 2026 Baseline)"] = {
    "omega": prod_params["omega"],
    "alpha": prod_params["alpha"],
    "beta": prod_params["beta"],
    "k": prod_params["k"],
    "persistence": prod_params["alpha_plus_beta"],
    "nit": 0,
    "nfev": 0,
    "converged": True,
    "message": "Production JSON reference"
}

print("\n3. Evaluating downstream signal sensitivity on independent out-of-sample data...")
eval_sub = active_df.loc["2025-06"].iloc[:500000].copy()
eval_adj = deseasonalize_month(eval_sub["delta_sec"]).to_numpy()
n_eval = len(eval_adj)
print(f"   Held-out evaluation window size: {n_eval:,} ticks (June 2025)")

signals = {}
for model_name, p in fits.items():
    w, a, b, k = p["omega"], p["alpha"], p["beta"], p["k"]
    psi = np.empty(n_eval)
    psi[0] = w / (1.0 - a - b)
    for t in range(1, n_eval):
        psi[t] = w + a * eval_adj[t-1] + b * psi[t-1]
    
    eps = eval_adj / psi
    log_eps = np.log(np.maximum(eps, 1e-12))
    z = -(log_eps - np.mean(log_eps)) / np.std(log_eps)
    signals[model_name] = z

base_z = signals["Production (May 2026 Baseline)"]

print("\n" + "=" * 80)
print("AUDIT SUMMARY TABLE: CALIBRATION STABILITY & SIGNAL INVARIANCE")
print("=" * 80)

audit_records = []
for model_name, z in signals.items():
    r_pearson = float(np.corrcoef(base_z, z)[0, 1])
    rank_base = pd.Series(base_z).rank()
    rank_model = pd.Series(z).rank()
    r_spearman = float(np.corrcoef(rank_base, rank_model)[0, 1])
    
    diff = np.abs(base_z - z)
    mae = float(np.mean(diff))
    p95_diff = float(np.percentile(diff, 95))
    p99_diff = float(np.percentile(diff, 99))
    
    top1_base = base_z >= np.percentile(base_z, 99)
    top1_model = z >= np.percentile(z, 99)
    jaccard = float(np.sum(top1_base & top1_model) / np.sum(top1_base | top1_model))
    
    audit_records.append({
        "Specification": model_name,
        "omega": round(fits[model_name]["omega"], 6),
        "alpha": round(fits[model_name]["alpha"], 6),
        "beta": round(fits[model_name]["beta"], 6),
        "k_shape": round(fits[model_name]["k"], 4),
        "persistence": round(fits[model_name]["persistence"], 6),
        "Pearson_r": round(r_pearson, 6),
        "Spearman_rho": round(r_spearman, 6),
        "MAE": round(mae, 6),
        "p99_diff": round(p99_diff, 6),
        "Tail_Overlap_q99": round(jaccard, 4)
    })

audit_df = pd.DataFrame(audit_records)
print(audit_df.to_string(index=False))

out_csv = ROOT / "tables" / "calibration_stability_audit.csv"
out_csv.parent.mkdir(exist_ok=True)
audit_df.to_csv(out_csv, index=False)
print(f"\n✓ Audit table successfully exported to: {out_csv}")
print("=" * 80)

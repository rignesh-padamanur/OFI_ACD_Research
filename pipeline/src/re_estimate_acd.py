"""
re_estimate_acd.py: Re-estimate EACD(1,1) and WACD(1,1) on Deseasonalized Durations
=====================================================================================
FAST version: Uses warm-started optimization from old parameters rather than 
exhaustive grid search, which makes the estimation tractable in pure Python.
"""

raise RuntimeError("Legacy ACD estimation is disabled during staged repair. See reports/PIPELINE_REPAIR_LOG.md; chronological observation-aware ACD fitting is pending in stage 04d.")

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CLEAN_PATH   = PROJECT_ROOT / "data" / "processed" / "eurusd_clean.parquet"
ACD_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "acd_input.parquet"
EACD_JSON    = PROJECT_ROOT / "data" / "results" / "eacd_params.json"
WACD_JSON    = PROJECT_ROOT / "data" / "results" / "weibull_params.json"

print("=" * 80)
print("RE-ESTIMATING ACD MODELS ON DESEASONALIZED DURATIONS")
print("=" * 80)

print("\n1. Loading clean tick data and filtering to May 2026...")
df = pd.read_parquet(CLEAN_PATH, columns=["timestamp", "bid", "ask", "spread",
                                           "mid", "spread_pips", "delta_sec"])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

may2026 = df.loc["2026-05"].copy()
print(f"   May 2026 raw ticks: {len(may2026):,}")
del df

mask = (may2026["delta_sec"].notna()) & (may2026["delta_sec"] > 0) & (may2026["delta_sec"] <= 60)
df_acd = may2026.loc[mask].copy()
print(f"   Active durations (0 < δ ≤ 60s): {len(df_acd):,}")
del may2026

print("\n2. Computing 15-minute diurnal spline Φ_δ(t) and deseasonalizing...")

from src.deseasonal import deseasonalize_series
df_acd["adj_delta_sec"], df_acd["phi_duration"] = deseasonalize_series(df_acd["delta_sec"])

assert np.isclose(df_acd["phi_duration"].mean(), 1.0, atol=1e-6), \
    f"Phi mean should be ~1.0, got {df_acd['phi_duration'].mean():.6f}"
assert (df_acd["adj_delta_sec"] > 0).all(), "All deseasonalized durations must be positive"
assert df_acd.index.is_monotonic_increasing, "Index must be monotonically increasing"

phi_range = df_acd["phi_duration"]
print(f"   Φ_δ range: [{phi_range.min():.4f}, {phi_range.max():.4f}] (mean = {phi_range.mean():.6f})")
print(f"   raw δ mean:   {df_acd['delta_sec'].mean():.6f}s")
print(f"   adj δ̃ mean:   {df_acd['adj_delta_sec'].mean():.6f}s")
print(f"   adj δ̃ median: {df_acd['adj_delta_sec'].median():.6f}s")

N_cal = len(df_acd)
x = df_acd["adj_delta_sec"].to_numpy()

def forward_filter(omega, alpha, beta, x):
    """ACD(1,1) conditional expectation recursion."""
    n = len(x)
    psi = np.empty(n)
    psi[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        psi[t] = omega + alpha * x[t-1] + beta * psi[t-1]
    return psi

print(f"\n3. Estimating EACD(1,1) on N = {N_cal:,} deseasonalized durations...")
print("   (Warm-started from old parameters + perturbations)")

def eacd_nll(params):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e20
    psi = forward_filter(omega, alpha, beta, x)
    if np.any(psi <= 0):
        return 1e20
    nll = np.sum(np.log(psi) + x / psi)
    return nll if np.isfinite(nll) else 1e20

eacd_inits = [
    [0.003688, 0.027007, 0.965855],
    [0.005, 0.03, 0.95],
    [0.002, 0.02, 0.97],
    [0.01, 0.05, 0.90],
    [0.001, 0.015, 0.975],
    [0.008, 0.04, 0.93],
]

eacd_bounds = [(1e-7, 0.5), (1e-5, 0.5), (0.5, 0.99999)]

best_eacd = None
best_eacd_nll = 1e20

for init in eacd_inits:
    if init[1] + init[2] >= 0.999:
        continue
    res = minimize(eacd_nll, x0=init, method="L-BFGS-B", bounds=eacd_bounds,
                   options={"maxiter": 2000, "ftol": 1e-14})
    if res.success and res.fun < best_eacd_nll:
        best_eacd_nll = res.fun
        best_eacd = res
    print(f"   Init {init} → NLL = {res.fun:,.0f} ({'✓' if res.success else '✗'})")

eacd_omega, eacd_alpha, eacd_beta = best_eacd.x
eacd_persistence = eacd_alpha + eacd_beta

assert eacd_persistence < 1.0, f"EACD not stationary: α+β = {eacd_persistence:.6f}"
assert eacd_omega > 0

print(f"\n   BEST EACD: ω = {eacd_omega:.6f}, α = {eacd_alpha:.6f}, β = {eacd_beta:.6f}")
print(f"   α + β = {eacd_persistence:.6f}, NLL = {best_eacd_nll:,.0f}, ln L = {-best_eacd_nll:,.0f}")

print(f"\n4. Estimating Weibull ACD(1,1) on N = {N_cal:,} deseasonalized durations...")
print("   (Warm-started from old parameters + perturbations)")

def wacd_nll(params):
    omega, alpha, beta, k = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or k <= 0.2 or k >= 3.0:
        return 1e20
    psi = forward_filter(omega, alpha, beta, x)
    if np.any(psi <= 0):
        return 1e20
    gamma_i = psi / gamma_func(1.0 + 1.0 / k)
    if np.any(gamma_i <= 0):
        return 1e20
    ll = np.log(k) + (k - 1.0) * np.log(x) - k * np.log(gamma_i) - (x / gamma_i)**k
    nll = -np.sum(ll)
    return nll if np.isfinite(nll) else 1e20

wacd_inits = [
    [0.001575, 0.022548, 0.96502, 0.749015],
    [0.002, 0.025, 0.965, 0.75],
    [0.003, 0.03, 0.95, 0.80],
    [0.001, 0.015, 0.975, 0.70],
    [0.005, 0.04, 0.93, 0.85],
    [0.001, 0.02, 0.97, 0.65],
]

wacd_bounds = [(1e-7, 0.5), (1e-5, 0.5), (0.5, 0.99999), (0.3, 2.0)]

best_wacd = None
best_wacd_nll = 1e20

for init in wacd_inits:
    if init[1] + init[2] >= 0.999:
        continue
    res = minimize(wacd_nll, x0=init, method="L-BFGS-B", bounds=wacd_bounds,
                   options={"maxiter": 2000, "ftol": 1e-14})
    if res.success and res.fun < best_wacd_nll:
        best_wacd_nll = res.fun
        best_wacd = res
    print(f"   Init {init} → NLL = {res.fun:,.0f} ({'✓' if res.success else '✗'})")

wacd_omega, wacd_alpha, wacd_beta, wacd_k = best_wacd.x
wacd_persistence = wacd_alpha + wacd_beta

assert wacd_persistence < 1.0, f"WACD not stationary: α+β = {wacd_persistence:.6f}"
assert wacd_omega > 0
assert 0.3 < wacd_k < 2.0

print(f"\n   BEST WACD: ω = {wacd_omega:.6f}, α = {wacd_alpha:.6f}, β = {wacd_beta:.6f}, k = {wacd_k:.6f}")
print(f"   α + β = {wacd_persistence:.6f}, NLL = {best_wacd_nll:,.0f}, ln L = {-best_wacd_nll:,.0f}")

print(f"\n5. Likelihood Ratio Test: H₀: k = 1 (EACD) vs H₁: k ≠ 1 (WACD)")
delta_lnL = best_eacd_nll - best_wacd_nll
LR = 2.0 * delta_lnL
print(f"   Δ ln L = {delta_lnL:,.0f}")
print(f"   LR = 2 × Δ ln L = {LR:,.0f} ~ χ²(1)")
print(f"   {'REJECT H₀' if LR > 3.841 else 'FAIL TO REJECT'} at α = 0.05")

print(f"\n6. Computing Hessian-based standard errors at WACD MLE...")

from scipy.optimize import approx_fprime

eps_fd = 1e-5
n_params = 4
hessian = np.zeros((n_params, n_params))
grad0 = approx_fprime(best_wacd.x, wacd_nll, eps_fd)

for j in range(n_params):
    x_plus = best_wacd.x.copy()
    x_plus[j] += eps_fd
    grad_plus = approx_fprime(x_plus, wacd_nll, eps_fd)
    hessian[j, :] = (grad_plus - grad0) / eps_fd

hessian = 0.5 * (hessian + hessian.T)

try:
    cov_matrix = np.linalg.inv(hessian)
    se = np.sqrt(np.abs(np.diag(cov_matrix)))
    se_valid = np.all(np.isfinite(se)) and np.all(se > 0)
except np.linalg.LinAlgError:
    se = np.full(n_params, np.nan)
    se_valid = False

param_names = ["ω", "α", "β", "k"]
null_vals = [0.0, 0.0, 0.0, 1.0]
for name, val, s, null in zip(param_names, best_wacd.x, se, null_vals):
    t_stat = (val - null) / s if se_valid and s > 0 else np.nan
    print(f"   {name}: estimate = {val:.6f}, SE = {s:.6f}, t({name}={null}) = {t_stat:.2f}")

print(f"\n7. Residual diagnostics...")

psi = forward_filter(wacd_omega, wacd_alpha, wacd_beta, x)
eps_standard = x / psi

print(f"   Standard residual (ε = δ̃/ψ) mean: {eps_standard.mean():.6f}")
print(f"   Standard residual median:          {np.median(eps_standard):.6f}")
print(f"   Standard residual std:             {eps_standard.std():.6f}")
print(f"   Filtered ψ mean:                   {psi.mean():.6f}s")
print(f"   Filtered ψ median:                 {np.median(psi):.6f}s")

import math
half_life = math.log(2) / math.log(1.0 / wacd_beta)
print(f"   Half-life: {half_life:.1f} quote updates ({half_life * np.median(x):.2f}s at median duration)")

print(f"\n   Computing Ljung-Box Q(10) on N = {N_cal:,}...")
demeaned = eps_standard - eps_standard.mean()
var_eps = np.sum(demeaned**2) / N_cal
rho = np.zeros(10)
for lag in range(1, 11):
    rho[lag-1] = np.sum(demeaned[:N_cal-lag] * demeaned[lag:]) / (N_cal * var_eps)
Q10_residual = N_cal * (N_cal + 2) * np.sum(rho**2 / (N_cal - np.arange(1, 11)))

x_raw = df_acd["delta_sec"].to_numpy()
demeaned_raw = x_raw - x_raw.mean()
var_raw = np.sum(demeaned_raw**2) / N_cal
rho_raw = np.zeros(10)
for lag in range(1, 11):
    rho_raw[lag-1] = np.sum(demeaned_raw[:N_cal-lag] * demeaned_raw[lag:]) / (N_cal * var_raw)
Q10_raw = N_cal * (N_cal + 2) * np.sum(rho_raw**2 / (N_cal - np.arange(1, 11)))

demeaned_adj = x - x.mean()
var_adj = np.sum(demeaned_adj**2) / N_cal
rho_adj = np.zeros(10)
for lag in range(1, 11):
    rho_adj[lag-1] = np.sum(demeaned_adj[:N_cal-lag] * demeaned_adj[lag:]) / (N_cal * var_adj)
Q10_adj = N_cal * (N_cal + 2) * np.sum(rho_adj**2 / (N_cal - np.arange(1, 11)))

reduction_raw = (1.0 - Q10_residual / Q10_raw) * 100.0
reduction_adj = (1.0 - Q10_residual / Q10_adj) * 100.0

print(f"   Raw durations Q(10) = {Q10_raw:,.1f}")
print(f"   Deseasonalized durations Q(10) = {Q10_adj:,.1f}")
print(f"   WACD residuals Q(10) = {Q10_residual:,.2f}")
print(f"   Reduction from raw: {reduction_raw:.2f}%")
print(f"   Reduction from deseasonalized: {reduction_adj:.2f}%")

print(f"\n8. Saving updated parameter files...")

eacd_params = {
    "model": "EACD",
    "distribution": "Exponential",
    "sample": "EURUSD May 2026 (deseasonalized)",
    "optimizer": "L-BFGS-B",
    "omega": round(float(eacd_omega), 6),
    "alpha": round(float(eacd_alpha), 6),
    "beta": round(float(eacd_beta), 6),
    "alpha_plus_beta": round(float(eacd_persistence), 6),
    "nll": int(round(best_eacd_nll)),
    "N_cal": N_cal,
    "deseasonalized": True
}

wacd_params = {
    "model": "Weibull ACD",
    "distribution": "Weibull",
    "optimizer": "L-BFGS-B",
    "omega": round(float(wacd_omega), 6),
    "alpha": round(float(wacd_alpha), 6),
    "beta": round(float(wacd_beta), 6),
    "k": round(float(wacd_k), 6),
    "alpha_plus_beta": round(float(wacd_persistence), 6),
    "nll": int(round(best_wacd_nll)),
    "N_cal": N_cal,
    "deseasonalized": True,
    "LR_test": round(float(LR), 0),
    "se_k": round(float(se[3]), 6) if se_valid else None,
    "t_stat_k": round(float((wacd_k - 1.0) / se[3]), 2) if se_valid and se[3] > 0 else None,
    "Q10_raw": round(float(Q10_raw), 1),
    "Q10_deseasonalized": round(float(Q10_adj), 1),
    "Q10_residual": round(float(Q10_residual), 2),
    "residual_mean": round(float(eps_standard.mean()), 6),
    "psi_filtered_mean": round(float(psi.mean()), 6),
    "psi_filtered_median": round(float(np.median(psi)), 6),
    "half_life_quotes": round(float(half_life), 1)
}

with open(EACD_JSON, "w") as f:
    json.dump(eacd_params, f, indent=2)
print(f"   ✓ Saved {EACD_JSON}")

with open(WACD_JSON, "w") as f:
    json.dump(wacd_params, f, indent=2)
print(f"   ✓ Saved {WACD_JSON}")

print(f"\n9. Saving updated acd_input.parquet with adj_delta_sec column...")
df_acd.to_parquet(ACD_OUT_PATH)
print(f"   ✓ Saved {ACD_OUT_PATH}")
print(f"   Columns: {df_acd.columns.tolist()}")
print(f"   N = {len(df_acd):,}")
assert "adj_delta_sec" in df_acd.columns
assert "phi_duration" in df_acd.columns

print(f"\n10. Old vs. New parameter comparison:")
print(f"{'':5s} {'Parameter':10s} {'Old':>12s} {'New':>12s} {'Δ%':>10s}")
print(f"{'':5s} {'─'*10:10s} {'─'*12:>12s} {'─'*12:>12s} {'─'*10:>10s}")

old_wacd = {"omega": 0.001575, "alpha": 0.022548, "beta": 0.96502, "k": 0.749015}
for p in ["omega", "alpha", "beta", "k"]:
    old_val = old_wacd[p]
    new_val = wacd_params[p]
    pct = (new_val - old_val) / old_val * 100 if old_val != 0 else 0
    print(f"{'':5s} {p:10s} {old_val:12.6f} {new_val:12.6f} {pct:+9.2f}%")

old_p = 0.987568
new_p = wacd_persistence
print(f"{'':5s} {'α+β':10s} {old_p:12.6f} {new_p:12.6f} {(new_p-old_p)/old_p*100:+9.2f}%")

print("\n" + "=" * 80)
print("ESTIMATION COMPLETE")
print("=" * 80)
print(f"Calibration: May 2026, N = {N_cal:,}, DESEASONALIZED durations")
print(f"EACD:  ω={eacd_omega:.6f}, α={eacd_alpha:.6f}, β={eacd_beta:.6f}")
print(f"WACD:  ω={wacd_omega:.6f}, α={wacd_alpha:.6f}, β={wacd_beta:.6f}, k={wacd_k:.6f}")
print(f"LR = {LR:,.0f}, k SE = {se[3]:.6f}, t = {(wacd_k-1)/se[3]:.2f}" if se_valid else "")
print(f"Residual ε mean = {eps_standard.mean():.6f}, Q(10)_raw = {Q10_raw:,.1f}, Q(10)_res = {Q10_residual:,.2f}")
print("=" * 80)

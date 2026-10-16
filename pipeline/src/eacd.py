import pandas as pd
#from scipy.optimize import minimize
import numpy as np
import json
from pathlib import Path

from src.deseasonal import deseasonalize_series

def prepare_acd_data(df, deseasonalize=True):
    """
    Prepare quote duration data for ACD estimation.
    Matches the preprocessing performed in the research notebook.
    """
    raise RuntimeError("Legacy ACD preparation/parameters/recursion are invalid for the repaired pipeline. "
                       "Use src.pipeline_state.load_duration_observations with an explicit policy; "
                       "chronological censoring-aware ACD estimation remains pending (stage 04d).")

    df_acd = df[
        (df["delta_sec"].notna()) &
        (df["delta_sec"] > 0) &
        (df["delta_sec"] <= 60)
    ].copy()

    if deseasonalize:
        df_acd["adj_delta_sec"], df_acd["phi_duration"] = deseasonalize_series(df_acd["delta_sec"])

    return df_acd

'''def fit_eacd(df_acd):

    x = df_acd["delta_sec"].to_numpy()

    def acd_nll(params):

        omega, alpha, beta = params

        if (
            omega <= 0
            or alpha < 0
            or beta < 0
            or alpha + beta >= 0.999
        ):
            return 1e20

        n = len(x)

        psi = np.zeros(n)
        psi[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            psi[t] = (
                omega
                + alpha * x[t - 1]
                + beta * psi[t - 1]
            )

        if np.any(psi <= 0):
            return 1e20

        return (
            np.log(psi)
            + x / psi
        ).sum()

    bounds = [
        (1e-6, None),
        (0, 0.999),
        (0, 0.999),
    ]

    omegas = [0.005, 0.01, 0.03, 0.05]
    alphas = [0.05, 0.10, 0.15, 0.20]
    betas = [0.70, 0.80, 0.90]

    results = []

    for w in omegas:
        for a in alphas:
            for b in betas:

                if a + b >= 0.99:
                    continue

                res = minimize(
                    acd_nll,
                    x0=[w, a, b],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": 1000,
                        "ftol": 1e-12,
                    },
                )

                results.append(
                    {
                        "start": [w, a, b],
                        "omega": res.x[0],
                        "alpha": res.x[1],
                        "beta": res.x[2],
                        "ab": res.x[1] + res.x[2],
                        "nll": res.fun,
                        "success": res.success,
                        "iterations": res.nit,
                        "message": res.message,
                    }
                )

    results_df = (
        pd.DataFrame(results)
        .sort_values("nll")
        .reset_index(drop=True)
    )

    best = results_df.iloc[0]

    omega = best["omega"]
    alpha = best["alpha"]
    beta = best["beta"]

    mu = np.empty(len(x))
    mu[0] = np.mean(x)

    for i in range(1, len(x)):
        mu[i] = (
            omega
            + alpha * x[i - 1]
            + beta * mu[i - 1]
        )

    residual = x / mu

    out = df_acd.copy()

    out["mu_eacd"] = mu
    out["residual_eacd"] = residual

    params = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
    }
'''
def load_eacd_params(project_root):

    raise RuntimeError("Legacy ACD preparation/parameters/recursion are invalid for the repaired pipeline. "
                       "Use src.pipeline_state.load_duration_observations with an explicit policy; "
                       "chronological censoring-aware ACD estimation remains pending (stage 04d).")
    with open(project_root / "data" / "results" / "eacd_params.json", "r") as f:
        return json.load(f)
    
def compute_eacd_residuals(df_acd, params):
    """
    Compute EACD(1,1) conditional duration and standard residuals.

    Uses deseasonalized durations (adj_delta_sec) if available, otherwise
    falls back to raw durations (delta_sec).

    The standard ACD residual is ε_i = δ̃_i / ψ_i, which has E[ε] = 1
    under correct model specification.
    """
    raise RuntimeError("Legacy ACD preparation/parameters/recursion are invalid for the repaired pipeline. "
                       "Use src.pipeline_state.load_duration_observations with an explicit policy; "
                       "chronological censoring-aware ACD estimation remains pending (stage 04d).")
    dur_col = "adj_delta_sec" if "adj_delta_sec" in df_acd.columns else "delta_sec"
    x = df_acd[dur_col].to_numpy()

    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]

    assert omega > 0, f"omega must be positive, got {omega}"
    assert alpha + beta < 1.0, f"alpha+beta must be < 1 for stationarity, got {alpha+beta}"

    n = len(x)
    mu = np.empty(n)
    mu[0] = omega / (1 - alpha - beta)

    for i in range(1, n):
        mu[i] = (
            omega
            + alpha * x[i - 1]
            + beta * mu[i - 1]
        )

    residual = x / mu

    out = df_acd.copy()
    out["mu_eacd"] = mu
    out["residual_eacd"] = residual

    return out
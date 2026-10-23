import json
import numpy as np
from scipy.special import gamma


def load_weibull_params(project_root):

    raise RuntimeError("Legacy ACD preparation/parameters/recursion are invalid for the repaired pipeline. "
                       "Use src.pipeline_state.load_duration_observations with an explicit policy; "
                       "chronological censoring-aware ACD estimation remains pending (stage 04d).")
    with open(
        project_root / "data" / "results" / "weibull_params.json",
        "r"
    ) as f:
        return json.load(f)


def compute_weibull_residuals(df_acd, params):
    """
    Compute Weibull ACD(1,1) conditional duration and standard residuals.

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
    k = params["k"]

    assert omega > 0, f"omega must be positive, got {omega}"
    assert alpha + beta < 1.0, f"alpha+beta must be < 1 for stationarity, got {alpha+beta}"

    n = len(x)
    psi = np.zeros(n)
    psi[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        psi[t] = (
            omega
            + alpha * x[t - 1]
            + beta * psi[t - 1]
        )

    # standard unit-mean ACD residual: ε_i = δ̃_i / ψ_i
    # E[ε_i] = 1 under correct specification (Engle & Russell, 1998)
    eps = x / psi

    out = df_acd.copy()
    out["psi_weibull"] = psi
    out["residual_weibull"] = eps

    return out
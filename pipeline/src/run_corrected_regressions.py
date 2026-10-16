"""
================================================================================
CORRECTED NOTEBOOK 5: PROBIT WITH SESSION FE & HALF-LIFE WINSORIZATION
================================================================================
Implements three critical econometric fixes for JFE/JFQA peer-review readiness:

  Fix 1 (OLS → Probit):
    Replace the linear OLS regression (AUC_300 ~ pre_vol_pips + stress)
    with a Binary Probit model predicting is_persistent = I(AUC_300 > 0).
    Reports average marginal effects (dF/dx), not raw Probit coefficients.

  Fix 2 (Half-Life Winsorization):
    Report median τ_1/2 = 60.06s as primary metric. Winsorize at 300-second
    observational boundary. Disclose that mean (1,521s) is contaminated by
    parametric fitting artifacts where λ̂ → 0 ⟹ τ_1/2 → ∞.

  Fix 3 (Session Fixed Effects):
    Add session dummy variables (Asian, London, NY vs. Off-Peak baseline)
    to control for omitted variable bias from session heterogeneity.
    London median AUC = -5.70 vs. NY median AUC = +99.24 — these are
    not draws from the same DGP.

Reads only final_event_database.parquet (4.3 MB, 29,145 events).
No 55.7M tick loading required. Execution time: <5 seconds.

Output Files:
  - data/results/probit_regression_table.csv
  - data/results/probit_marginal_effects.csv
  - data/results/session_analysis_corrected.csv
================================================================================
"""

# This file is preserved as historical source.  Neither direct execution nor
# import may expose the superseded event labels and episode-peak regressions.
raise RuntimeError(
    "Legacy regression pipeline is blocked: this module uses superseded "
    "persistence labels and episode-peak features. Use the verified stage-07a "
    "regression-input artifacts; empirical estimation is not yet implemented "
    "or approved. Historical results are preserved and have not been changed."
)

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FDB_PATH = PROJECT_ROOT / "data" / "results" / "final_event_database.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# ============================================================
# SECTION A: LOAD UNIFIED DATASET
# ============================================================
def load_unified_dataset():
    """
    Load the pre-computed final_event_database.parquet which already
    contains all F_{t0}-measurable features merged for 29,145 events.
    Add session dummy variables for fixed effects.
    """
    print("=" * 80)
    print("SECTION A: Loading Unified Dataset")
    print("=" * 80)

    assert FDB_PATH.exists(), f"Missing: {FDB_PATH}"
    df = pd.read_parquet(FDB_PATH)

    print(f"  Loaded: {FDB_PATH.name}")
    print(f"  Shape:  {df.shape[0]:,} events × {df.shape[1]} columns")

    # ── Session Dummies (Fix 3) ──
    # Base category: Off-Peak (17:00–24:00 UTC)
    hour = df["start_time"].dt.hour
    df["session_asian"] = ((hour >= 0) & (hour < 7)).astype(int)
    df["session_london"] = ((hour >= 7) & (hour < 12)).astype(int)
    df["session_ny"] = ((hour >= 12) & (hour < 17)).astype(int)

    # Sort chronologically (should already be sorted, but enforce)
    df = df.sort_values("start_time").reset_index(drop=True)

    # ── Validate critical columns ──
    critical_cols = [
        "peak_stress", "z_ofi", "z_intensity", "z_duration",
        "pre_vol_pips", "auc_300", "is_persistent",
    ]
    nan_counts = df[critical_cols].isna().sum()
    has_nans = nan_counts.sum() > 0

    if has_nans:
        print(f"\n  ⚠ WARNING: NaN detected in critical columns:")
        for col, cnt in nan_counts.items():
            if cnt > 0:
                print(f"    {col}: {cnt:,} NaN values")
        n_before = len(df)
        df = df.dropna(subset=critical_cols)
        n_after = len(df)
        print(f"    Dropped {n_before - n_after:,} rows → {n_after:,} valid events")
    else:
        print(f"  ✓ Zero NaN in critical columns")

    # ── Assertions ──
    assert len(df) > 1000, f"Too few events: {len(df)}"
    assert df["is_persistent"].isin([0, 1]).all(), "is_persistent not binary"
    assert df["start_time"].is_monotonic_increasing, "Not chronologically sorted"

    print(f"\n  Total events:         {len(df):,}")
    print(f"  Chronological range:  {df['start_time'].min()} → {df['start_time'].max()}")
    print(f"  Persistent (AUC>0):   {df['is_persistent'].sum():,} "
          f"({df['is_persistent'].mean():.1%})")
    print(f"  Reversal (AUC≤0):     {(df['is_persistent'] == 0).sum():,} "
          f"({(df['is_persistent'] == 0).mean():.1%})")

    print(f"\n  Session Breakdown:")
    print(f"    Asian (00-07 UTC):    {df['session_asian'].sum():,} events")
    print(f"    London (07-12 UTC):   {df['session_london'].sum():,} events")
    print(f"    NY (12-17 UTC):       {df['session_ny'].sum():,} events")
    n_offpeak = len(df) - df[["session_asian", "session_london", "session_ny"]].any(axis=1).sum()
    print(f"    Off-Peak (17-24 UTC): {n_offpeak:,} events (base category)")

    return df


# ============================================================
# SECTION B: CORRECTED PROBIT WITH SESSION FIXED EFFECTS
# ============================================================
def run_probit_with_session_fe(df):
    """
    Headline regression (Fix 1 + Fix 3):

      P(Persistent_k = 1 | F_{t0}) = Φ(β₀ + β₁·σ_pre + β₂·peak_stress
                                        + β₃·z_ofi + β₄·z_intensity + β₅·z_duration
                                        + γ₁·Asian + γ₂·London + γ₃·NY)

    Reports:
      - Probit coefficients with MLE standard errors
      - Average Marginal Effects (dF/dx) — the publication-ready quantities
      - McFadden's Pseudo-R²
      - Out-of-sample ROC-AUC on 30% chronological test set
      - Wald test for joint significance of session dummies
    """
    print("\n" + "=" * 80)
    print("SECTION B: PROBIT MODEL WITH SESSION FIXED EFFECTS")
    print("  Dependent Variable: is_persistent = I(AUC_300 > 0)")
    print("  Note: start_spread_pips excluded (insignificant in pilot, p=0.431;")
    print("        collinear with pre_vol_pips; requires 797 MB tick file to compute)")
    print("=" * 80)

    # ── Feature sets ──
    features_no_fe = ["pre_vol_pips", "peak_stress",
                      "z_ofi", "z_intensity", "z_duration"]
    features_full = features_no_fe + ["session_asian", "session_london", "session_ny"]

    # ── Chronological 70/30 split ──
    n_train = int(len(df) * 0.70)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    y_train = train_df["is_persistent"]
    y_test = test_df["is_persistent"]

    assert train_df["start_time"].max() <= test_df["start_time"].min(), \
        "Temporal leakage: train overlaps test"

    print(f"\n  Train: N = {len(train_df):,}  |  Test: N = {len(test_df):,}")
    print(f"  Train persistence rate: {y_train.mean():.4f}")
    print(f"  Test persistence rate:  {y_test.mean():.4f}")

    # ── Model A: Probit WITHOUT session FE (restricted) ──
    X_train_r = sm.add_constant(train_df[features_no_fe])
    X_test_r = sm.add_constant(test_df[features_no_fe])
    probit_r = Probit(y_train, X_train_r).fit(disp=False)
    assert probit_r.mle_retvals["converged"], "Restricted Probit did not converge"

    # ── Model B: Probit WITH session FE (full) ──
    X_train_f = sm.add_constant(train_df[features_full])
    X_test_f = sm.add_constant(test_df[features_full])
    probit_f = Probit(y_train, X_train_f).fit(disp=False)
    assert probit_f.mle_retvals["converged"], "Full Probit did not converge"

    # ── Average Marginal Effects ──
    margeff = probit_f.get_margeff()

    # ── Out-of-sample evaluation ──
    p_r_test = probit_r.predict(X_test_r)
    p_f_test = probit_f.predict(X_test_f)

    auc_r = roc_auc_score(y_test, p_r_test)
    auc_f = roc_auc_score(y_test, p_f_test)
    brier_r = brier_score_loss(y_test, p_r_test)
    brier_f = brier_score_loss(y_test, p_f_test)

    # ── TABLE 1: Probit Coefficients ──
    col_names = ["const"] + features_full
    print(f"\n{'─' * 90}")
    print(f"TABLE 1: PROBIT COEFFICIENT ESTIMATES (Full Model with Session FE)")
    print(f"{'─' * 90}")
    print(f"{'Variable':<20} {'Coef':>12} {'Std.Err':>12} {'z-stat':>10} "
          f"{'P>|z|':>14} {'Sig':>5}")
    print(f"{'─' * 90}")

    for i, name in enumerate(col_names):
        coef = probit_f.params[i]
        se = probit_f.bse[i]
        z = probit_f.tvalues[i]
        p = probit_f.pvalues[i]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"{name:<20} {coef:>12.6f} {se:>12.6f} {z:>10.4f} {p:>14.6e} {sig:>5}")

    print(f"{'─' * 90}")
    print(f"  Log-Likelihood:      {probit_f.llf:.4f}")
    print(f"  McFadden Pseudo-R²:  {probit_f.prsquared:.6f}")
    print(f"  AIC:                 {probit_f.aic:.4f}")
    print(f"  BIC:                 {probit_f.bic:.4f}")
    print(f"  N (train):           {len(train_df):,}")

    # ── TABLE 2: Average Marginal Effects ──
    print(f"\n{'─' * 90}")
    print(f"TABLE 2: AVERAGE MARGINAL EFFECTS (dF/dx)")
    print(f"  Interpretation: 1-unit increase in X → dF/dx change in P(persistent)")
    print(f"{'─' * 90}")
    print(margeff.summary())

    # ── TABLE 3: Out-of-Sample Performance ──
    print(f"\n{'─' * 90}")
    print(f"TABLE 3: OUT-OF-SAMPLE PERFORMANCE (Test N = {len(test_df):,})")
    print(f"{'─' * 90}")
    print(f"{'Model':<45} {'Test AUC':>10} {'Brier':>10}")
    print(f"{'─' * 70}")
    print(f"{'Probit (No Session FE)':<45} {auc_r:>10.4f} {brier_r:>10.4f}")
    print(f"{'Probit (With Session FE)':<45} {auc_f:>10.4f} {brier_f:>10.4f}")
    print(f"{'─' * 70}")

    delta_auc = auc_f - auc_r
    print(f"  ΔAUC from Session FE: {delta_auc:+.4f}")

    # ── Wald test: joint significance of session dummies ──
    # Use restriction matrix: test that coefficients on the last 3 variables = 0
    n_params = len(probit_f.params)
    R = np.zeros((3, n_params))
    R[0, n_params - 3] = 1  # session_asian
    R[1, n_params - 2] = 1  # session_london
    R[2, n_params - 1] = 1  # session_ny
    r = np.zeros(3)

    wald_result = probit_f.wald_test((R, r))
    wald_chi2 = float(np.asarray(wald_result.statistic).flat[0])
    wald_p = float(np.asarray(wald_result.pvalue).flat[0])

    print(f"\n  Wald Test (H₀: γ_Asian = γ_London = γ_NY = 0):")
    print(f"    χ²(3) = {wald_chi2:.4f},  p = {wald_p:.6e}")
    if wald_p < 0.05:
        print(f"    → REJECT H₀: Session FE are jointly significant at 5%")
    else:
        print(f"    → FAIL TO REJECT H₀: Session FE are NOT jointly significant")

    # ── Likelihood Ratio test: restricted vs. full ──
    lr_stat = 2 * (probit_f.llf - probit_r.llf)
    from scipy.stats import chi2 as chi2_dist
    lr_p = chi2_dist.sf(lr_stat, df=3)
    print(f"\n  Likelihood Ratio Test (restricted vs. full):")
    print(f"    LR stat = {lr_stat:.4f},  p = {lr_p:.6e}")

    # ── Save results ──
    reg_data = []
    for i, name in enumerate(col_names):
        reg_data.append({
            "Feature": name,
            "Coefficient": probit_f.params[i],
            "Std_Error": probit_f.bse[i],
            "z_Statistic": probit_f.tvalues[i],
            "p_Value": probit_f.pvalues[i],
            "Significance": ("***" if probit_f.pvalues[i] < 0.001
                             else ("**" if probit_f.pvalues[i] < 0.01
                                   else ("*" if probit_f.pvalues[i] < 0.05 else ""))),
        })
    reg_df = pd.DataFrame(reg_data)
    reg_path = RESULTS_DIR / "probit_regression_table.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"\n  ✓ Saved: {reg_path}")

    me_df = margeff.summary_frame()
    me_path = RESULTS_DIR / "probit_marginal_effects.csv"
    me_df.to_csv(me_path)
    print(f"  ✓ Saved: {me_path}")

    return probit_f, train_df, test_df


# ============================================================
# SECTION C: HALF-LIFE DIAGNOSTICS (Fix 2)
# ============================================================
def report_halflife_diagnostics(df):
    """
    Report half-life statistics with proper winsorization disclosure.
    """
    print("\n" + "=" * 80)
    print("SECTION C: HALF-LIFE DIAGNOSTICS (Fix 2)")
    print("=" * 80)

    raw_hl = df["half_life_sec"].dropna()
    raw_hl = raw_hl[raw_hl > 0]
    capped_hl = df["half_life_capped"].dropna()

    print(f"\n  Raw Half-Life (τ_1/2) — BEFORE winsorization:")
    print(f"    N:      {len(raw_hl):,}")
    print(f"    Mean:   {raw_hl.mean():,.2f} seconds  ← CONTAMINATED")
    print(f"    Median: {raw_hl.median():.2f} seconds  ← REPORT THIS")
    print(f"    Std:    {raw_hl.std():,.2f}")
    print(f"    Max:    {raw_hl.max():,.2f}")
    print(f"    Mean/Median Ratio: {raw_hl.mean() / raw_hl.median():.1f}×")

    n_extreme = (raw_hl > 300).sum()
    pct_extreme = n_extreme / len(raw_hl) * 100
    print(f"\n    Events with τ_1/2 > 300s: {n_extreme:,} ({pct_extreme:.1f}%)")
    print(f"    → Parametric artifacts where λ̂ → 0 ⟹ τ_1/2 → ∞")

    print(f"\n  Winsorized Half-Life (capped at 300s):")
    print(f"    Mean:   {capped_hl.mean():.2f} seconds")
    print(f"    Median: {capped_hl.median():.2f} seconds")
    print(f"    Std:    {capped_hl.std():.2f}")

    # Non-parametric AUC_300 as primary metric
    auc = df["auc_300"]
    print(f"\n  Primary Invariance Metric: Non-Parametric AUC_300")
    print(f"    Mean:   {auc.mean():,.2f} σ·s")
    print(f"    Median: {auc.median():,.2f} σ·s")

    print(f"\n  PAPER LANGUAGE:")
    print(f'    "We report median half-lives (τ_1/2 = {capped_hl.median():.1f}s)')
    print(f'     and non-parametric AUC_300 as our primary persistence metrics,')
    print(f'     winsorizing τ_1/2 at the 300-second observational boundary.')
    print(f'     The sample mean ({raw_hl.mean():,.0f}s) is dominated by')
    print(f'     {n_extreme:,} events ({pct_extreme:.1f}%) where the exponential')
    print(f'     decay rate estimate λ̂ → 0, producing unbounded half-lives."')


# ============================================================
# SECTION D: SESSION-LEVEL PERSISTENCE HETEROGENEITY
# ============================================================
def report_session_analysis(df):
    """
    Document session heterogeneity to justify session FE inclusion.
    """
    print("\n" + "=" * 80)
    print("SECTION D: SESSION-LEVEL PERSISTENCE HETEROGENEITY")
    print("=" * 80)

    hour = df["start_time"].dt.hour
    df_s = df.copy()
    df_s["session"] = "4_Off-Peak (17-24)"
    df_s.loc[(hour >= 0) & (hour < 7), "session"] = "1_Asian (00-07)"
    df_s.loc[(hour >= 7) & (hour < 12), "session"] = "2_London (07-12)"
    df_s.loc[(hour >= 12) & (hour < 17), "session"] = "3_NY (12-17)"

    stats = df_s.groupby("session").agg(
        n_events=("event_id", "count"),
        pct_persistent=("is_persistent", "mean"),
        median_auc=("auc_300", "median"),
        mean_auc=("auc_300", "mean"),
        median_half_life=("half_life_capped", "median"),
        median_peak=("peak_impact", "median"),
        median_final=("final_impact", "median"),
    ).reset_index()
    stats["pct_persistent"] *= 100

    print(f"\n{'Session':<25} {'N':>7} {'%Pers':>8} {'Med AUC':>10} "
          f"{'Med τ½':>8} {'Med Final':>10}")
    print(f"{'─' * 75}")
    for _, r in stats.iterrows():
        print(f"{r['session']:<25} {r['n_events']:>7,} {r['pct_persistent']:>7.1f}% "
              f"{r['median_auc']:>10.2f} {r['median_half_life']:>8.2f} "
              f"{r['median_final']:>10.2f}")
    print(f"{'─' * 75}")

    lon = stats.loc[stats["session"].str.contains("London"), "median_auc"].values[0]
    ny = stats.loc[stats["session"].str.contains("NY"), "median_auc"].values[0]
    print(f"\n  KEY FINDING:")
    print(f"    London Median AUC = {lon:.2f} (NET REVERSAL)")
    print(f"    NY Median AUC     = {ny:.2f} (STRONG PERSISTENCE)")
    print(f"    Gap: {abs(ny - lon):.2f} σ·s — pooling without session FE")
    print(f"    introduces classical omitted variable bias.")
    print(f"\n    Economic interpretation:")
    print(f"    • London AM: inter-dealer inventory squaring → mean-reverting")
    print(f"    • NY PM: directional US macro flow absorption → permanent")

    # Save
    out_path = RESULTS_DIR / "session_analysis_corrected.csv"
    stats.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")


# ============================================================
# SECTION E: PERSISTENCE SCORE — CORRECTED FRAMING
# ============================================================
def report_persistence_score(df):
    """
    S_persistence = Φ((AUC_300 − μ_train) / σ_train) × 100

    This is an EX-POST standardization index, NOT a predictive model.
    The high Spearman correlation with final_impact is TAUTOLOGICAL
    (both measure the same 300-second price trajectory).
    """
    print("\n" + "=" * 80)
    print("SECTION E: PERSISTENCE SCORE (EX-POST INDEX — NOT PREDICTIVE)")
    print("=" * 80)

    n_train = int(len(df) * 0.70)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]

    mu = train["auc_300"].mean()
    sigma = train["auc_300"].std()

    z = (df["auc_300"] - mu) / sigma
    S = norm.cdf(z) * 100

    print(f"\n  Calibration (training-set only, N={len(train):,}):")
    print(f"    μ_AUC = {mu:.2f}")
    print(f"    σ_AUC = {sigma:.2f}")

    print(f"\n  S_persistence Distribution:")
    print(f"    Mean:  {S.mean():.2f}")
    print(f"    Std:   {S.std():.2f}")
    print(f"    Range: [{S.min():.2f}, {S.max():.2f}]")

    rho_full, _ = spearmanr(S, df["final_impact"])
    z_test = (test["auc_300"] - mu) / sigma
    S_test = norm.cdf(z_test) * 100
    rho_oos, _ = spearmanr(S_test, test["final_impact"])

    print(f"\n  Spearman ρ(S_persistence, final_impact):")
    print(f"    Full sample:    ρ = {rho_full:.4f}")
    print(f"    Out-of-sample:  ρ = {rho_oos:.4f}")

    print(f"\n  ⚠ CRITICAL NOTE FOR PAPER:")
    print(f"    DO NOT claim ρ = {rho_oos:.2f} as 'predictive accuracy.'")
    print(f"    S_persistence is a monotonic transform of AUC_300, which is")
    print(f"    itself an integral over the price path that final_impact")
    print(f"    is the endpoint of. High ρ is an algebraic tautology.")
    print(f"    Frame as: 'normalized [0,100] percentile index for realized")
    print(f"    shock permanence, analogous to a Richter scale classification.'")

    return S


# ============================================================
# SECTION F: VALIDATION SUITE
# ============================================================
def run_validation_suite(df, probit_model):
    """
    Automated validation checks per JFE/JFQA peer-review standards.
    """
    print("\n" + "=" * 80)
    print("SECTION F: VALIDATION SUITE")
    print("=" * 80)

    n_pass = 0
    n_total = 0

    def check(name, condition):
        nonlocal n_pass, n_total
        n_total += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        if condition:
            n_pass += 1
        print(f"  {status}: {name}")

    check(f"Total events > 1000 (actual: {len(df):,})", len(df) > 1000)
    check("Zero NaN in is_persistent", df["is_persistent"].notna().all())
    check("Zero NaN in auc_300", df["auc_300"].notna().all())
    check("Binary outcome in {0, 1}", df["is_persistent"].isin([0, 1]).all())
    check("Probit model converged", probit_model.mle_retvals["converged"])
    check(f"Pseudo-R² in [0,1] (actual: {probit_model.prsquared:.6f})",
          0 <= probit_model.prsquared <= 1)

    n_train = int(len(df) * 0.70)
    check("Train end < Test start (no temporal leakage)",
          df.iloc[n_train - 1]["start_time"] <= df.iloc[n_train]["start_time"])

    check("All capped half-lives ≤ 300s",
          (df["half_life_capped"] <= 300).all())

    if "S_persistence" in df.columns:
        check("S_persistence in [0, 100]",
              (df["S_persistence"] >= 0).all() and (df["S_persistence"] <= 100).all())

    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("  ⚠ SOME CHECKS FAILED — review before submission")
    else:
        print("  ✓ All validation checks passed")


# ============================================================
# MAIN
# ============================================================
def main():
    print("╔" + "═" * 78 + "╗")
    print("║  CORRECTED NOTEBOOK 5: PROBIT + SESSION FE + HALF-LIFE WINSORIZATION      ║")
    print("║  Fixes: (1) OLS→Probit  (2) τ½ outliers  (3) Session fixed effects        ║")
    print("║  Dataset: final_event_database.parquet (29,145 events, 4.3 MB)             ║")
    print("╚" + "═" * 78 + "╝\n")

    df = load_unified_dataset()
    probit_model, train_df, test_df = run_probit_with_session_fe(df)
    report_halflife_diagnostics(df)
    report_session_analysis(df)
    S = report_persistence_score(df)
    run_validation_suite(df, probit_model)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE — All corrected results saved to data/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()

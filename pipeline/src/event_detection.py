import numpy as np
import pandas as pd


def build_event_detection_pipeline(
    trading_ofi,
    trading_intensity,
    df_acd,
    method="mahalanobis",
    quantile_threshold=0.99,
    acd_model="weibull",
    train_frac=0.70,
):
    """Disabled legacy detector, retained below solely for implementation history.

    Applying a training-partition fit back onto earlier observations in that same
    partition is not past-only calibration. Scoring/covariance/thresholds and
    observable episode timing must be rebuilt after the calibration and duration
    stages; this API refuses execution until that separate repair is completed.
    """

    raise RuntimeError(
        'Legacy detector is disabled: applying training-fit scales/thresholds to earlier '
        'training observations is not past-only. Rebuild scoring and onset timing '
        'after the registered calibration and duration repairs.'
    )

    # 1-second ACD summaries
    res_col = f"residual_{acd_model.lower()}"
    mu_col = "psi_weibull" if acd_model.lower() == "weibull" else "mu_eacd"

    if res_col not in df_acd.columns:
        res_col = "residual_weibull" if "residual_weibull" in df_acd.columns else "residual_eacd"

    acd_1s = (
        df_acd
        .resample("1s")
        .agg({
            res_col: "mean"
        })
    )


    intensity_col = "adj_tick_count" if "adj_tick_count" in trading_intensity.columns else "tick_count"

    event_df = (
        trading_ofi[["ofi", "abs_ofi"]]
        .join(trading_intensity[[intensity_col]])
        .join(acd_1s)
    )

    event_df = event_df.dropna()

    # chronological train/test split

    N = len(event_df)
    n_train = int(N * train_frac)

    assert n_train > 0, f"Training set is empty (N={N}, train_frac={train_frac})"
    assert n_train < N, f"No test data (N={N}, train_frac={train_frac})"

    train_end_date = event_df.index[n_train - 1]

    train = event_df.iloc[:n_train]

    # fit scaling on the training partition

    # z_ofi uses absolute OFI
    ofi_train_mean = train["abs_ofi"].mean()
    ofi_train_std = train["abs_ofi"].std()
    assert ofi_train_std > 0, "abs_ofi has zero variance in training set"

    event_df["z_ofi"] = (
        event_df["abs_ofi"] - ofi_train_mean
    ) / ofi_train_std

    # z_intensity uses quote counts
    int_train_mean = train[intensity_col].mean()
    int_train_std = train[intensity_col].std()
    assert int_train_std > 0, "Intensity has zero variance in training set"

    event_df["z_intensity"] = (
        event_df[intensity_col] - int_train_mean
    ) / int_train_std

    # z_duration reverses the sign of the standardized log residual
    log_duration_full = np.log(
        event_df[res_col].replace(0, np.nan).fillna(1.0)
    )

    log_duration_train = log_duration_full.iloc[:n_train]
    log_dur_train_mean = log_duration_train.mean()
    log_dur_train_std = log_duration_train.std()
    assert log_dur_train_std > 0, "log_duration has zero variance in training set"

    # sign flip: smaller log residuals give larger z_duration
    event_df["z_duration"] = (
        log_dur_train_mean - log_duration_full
    ) / log_dur_train_std

    # fit covariance on the same training partition

    Z_full = event_df[["z_ofi", "z_intensity", "z_duration"]].to_numpy()
    Z_train = Z_full[:n_train, :]

    if method == "mahalanobis":
        cov_train = np.cov(Z_train, rowvar=False)

        # require positive covariance eigenvalues
        eigvals = np.linalg.eigvalsh(cov_train)
        assert np.all(eigvals > 0), (
            f"Train covariance is not positive definite. "
            f"Min eigenvalue: {eigvals.min():.6e}"
        )

        inv_cov = np.linalg.pinv(cov_train)
        stress = np.sqrt(np.sum(np.dot(Z_full, inv_cov) * Z_full, axis=1))
        event_df["stress_score"] = stress

    elif method == "pca":
        cov_train = np.cov(Z_train, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_train)
        pc1_vec = eigvecs[:, -1]
        event_df["stress_score"] = Z_full @ pc1_vec

    else:
        event_df["stress_score"] = (
            event_df["z_ofi"]
            + event_df["z_intensity"]
            + event_df["z_duration"]
        )

    # estimate the threshold from training scores only

    train_stress = event_df["stress_score"].iloc[:n_train]
    threshold = train_stress.quantile(quantile_threshold)

    event_df["candidate_event"] = (
        event_df["stress_score"] >= threshold
    )

    candidate_events = event_df[
        event_df["candidate_event"]
    ].copy()

    print(f"  Train/test split: {train_frac:.0%} / {1-train_frac:.0%}")
    print(f"  Train: N_train = {n_train:,} (ends {train_end_date})")
    print(f"  Test:  N_test  = {N - n_train:,}")
    print(f"  Train z_ofi   mean/std: {ofi_train_mean:.4f} / {ofi_train_std:.4f}")
    print(f"  Train z_int   mean/std: {int_train_mean:.4f} / {int_train_std:.4f}")
    print(f"  Train z_dur   mean/std: {log_dur_train_mean:.6f} / {log_dur_train_std:.6f}")
    print(f"  Train cov diagonal: {np.diag(cov_train) if method == 'mahalanobis' else 'N/A'}")
    print(f"  Threshold (q{quantile_threshold}): {threshold:.4f}")
    print(f"  Candidates (train): {train_stress.ge(threshold).sum():,}")
    print(f"  Candidates (total): {candidate_events.shape[0]:,}")

    return event_df, candidate_events, threshold
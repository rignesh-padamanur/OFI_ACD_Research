> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

**Stage 04b: prior-day seasonal calibration — step04b_seasonal_calibration_20260927**

The full-series seasonal estimator has been replaced by a sequential prior-day estimator, executed on the registered clock, and independently verified. This is **Notebook 2: Signal Construction**, after the clock/missingness repair and before duration modelling, multivariate scoring and event detection. The notebook guards remain while those later dependencies are rebuilt.

**Defect and correction**

The old seasonal function grouped the complete series by time of day, allowing future values to determine earlier seasonal factors. The old detector fitted normalization on its first 70% partition and then applied that fit backwards throughout the same partition. A chronological partition does not make those earlier normalized observations past-only. Both legacy entry points now refuse execution. The detector body is retained as historical implementation; its covariance, threshold and onset logic have not been rebuilt.

The replacement fits a baseline before each application day, applies it to that day, and only then adds that day to the historical sufficient statistics. This follows the prior-only evaluation principle described by [Hyndman and Athanasopoulos](https://otexts.com/fpp3/tscv.html). [Andersen and Bollerslev (1997)](https://public.econ.duke.edu/~boller/Published_Papers/joef_97.pdf) document the importance of intraday periodicity for return-volatility analysis; that paper does not validate our quote-count/quote-pressure estimator or prescribe the settings below.

**Explicit reference specification**

- Historical window: the preceding **56 completed calendar days**, `[d - 56 days, d)`, refreshed once per UTC day.
- Local cell: **UTC weekday × 15-minute interval**. Monday is not pooled into Sunday; there is no cross-cell fallback.
- Warm-up: a complete 56-day window. Local history additionally requires at least **4 distinct prior matching dates** and **1,800 eligible seconds**.
- Weights: one per eligible second, including supported zeros. Counts use count support, and absolute imbalance uses flow support. Their denominators are retained separately; later joint models must use their common valid sample.
- The settings were fixed for this repair before executing its results. They are configurable reference choices, not optimized settings, a study preregistration, or a selected primary paper specification. None was selected from persistence outcomes.
- The four observation-support scenarios remain separate. There is still no primary freshness policy.

UTC cells provide a reproducible clock but do not fully align changing London/New York sessions, DST transitions or holidays. Newly active cells can therefore remain uncalibrated until enough same-weekday history exists. Missing baselines are reported, not interpolated or backfilled. Subsequent specification validation must assess these effects; this repair does not claim exact removal of every seasonal pattern.

**Numerical definition and timing**

For either nonnegative channel x (retained quote count or absolute net quote imbalance), let μ_cell and σ_cell be the prior matching-cell mean and sample SD, and μ_global the mean over all eligible seconds of that same channel/policy in the same prior window. Then:

```text
phi = mu_cell / mu_global
adjusted_x = x / phi
z_x = (x - mu_cell) / sigma_cell
```

The z-score also equals `(adjusted_x - mu_global) / (sigma_cell / phi)`. It uses a cell-specific scale. Signed quote imbalance is divided by the positive absolute-imbalance factor without centering, preserving its sign. The signed mean is never used as a denominator. Z-scores of absolute imbalance measure relative magnitude, not direction.

Nonpositive/nonfinite means or scales, insufficient history, warm-up, and unavailable inputs remain missing with separate reasons. There is no scale-of-one substitution, winsorization, clipping or Gaussian probability conversion. Daily centered sums of squares are pooled to avoid cancellation; sample SD uses n−1. Those observation counts are not treated as independent observations for inference.

Every output keeps `[bin_start, bin_start + 1 second)` and `available_at = bin_start + 1 second`. Calibration cutoff is no later than bin start. A completed-bin z-score nevertheless uses that bin’s observations and is not available at its start. Later event-onset/conditioning logic must respect this distinction. Historical coverage masks remain retrospective sample-quality conventions; causal arithmetic conditional on those masks does not establish an online outage detector.

**Executed sample and attrition**

The run preserves all **81,475,200 calendar rows** across **943 UTC days**. It adds calibration availability; it does not change retained quotes, the parent clock or structural support. The first structural-support count signal is available at **2024-02-26T00:00:01+00:00**.

| Support scenario | Channel | Eligible input seconds | Calibrated seconds | Warm-up exclusions | Too few dates | Too few seconds |
|---|---|---:|---:|---:|---:|---:|
| structural_only | tick_count | 57,787,099 | 54,396,323 | 3,347,737 | 43,039 | 0 |
| structural_only | abs_quote_ofi | 57,787,029 | 54,396,253 | 3,347,737 | 43,039 | 0 |
| fresh_15s | tick_count | 53,900,964 | 50,713,190 | 3,162,710 | 24,369 | 695 |
| fresh_15s | abs_quote_ofi | 53,900,627 | 50,712,863 | 3,162,701 | 24,368 | 695 |
| fresh_30s | tick_count | 56,521,587 | 53,199,150 | 3,289,817 | 32,620 | 0 |
| fresh_30s | abs_quote_ofi | 56,521,444 | 53,199,010 | 3,289,814 | 32,620 | 0 |
| fresh_60s | tick_count | 57,507,325 | 54,131,802 | 3,335,713 | 39,810 | 0 |
| fresh_60s | abs_quote_ofi | 57,507,243 | 54,131,720 | 3,335,713 | 39,810 | 0 |

Across these policy/channel counts, 0 eligible observations are additionally excluded for nonpositive/unavailable means and 0 for nonpositive/unavailable scales. These reasons are tested even when absent in this dataset. Unsupported inputs remain missing separately and are not included in the eligible-input denominator. These are channel/scenario diagnostics, not counts of distinct events.

**Out-of-calibration-period diagnostics**

| Support scenario | Channel | Mean z | Sample SD z | Minimum z | Maximum z |
|---|---|---:|---:|---:|---:|
| structural_only | tick_count | 0.013957 | 1.072589 | -1.682109 | 36.065756 |
| structural_only | abs_quote_ofi | -0.003045 | 1.031631 | -1.031566 | 100.549136 |
| fresh_15s | tick_count | 0.015154 | 1.072205 | -1.682109 | 32.063985 |
| fresh_15s | abs_quote_ofi | -0.003014 | 1.029445 | -1.031566 | 97.372924 |
| fresh_30s | tick_count | 0.014284 | 1.072078 | -1.682109 | 34.124272 |
| fresh_30s | abs_quote_ofi | -0.003061 | 1.030551 | -1.031566 | 99.166371 |
| fresh_60s | tick_count | 0.014002 | 1.072331 | -1.682109 | 35.739480 |
| fresh_60s | abs_quote_ofi | -0.003059 | 1.031221 | -1.031566 | 100.317122 |

These are forward applications relative to each fit’s training window, not an untouched research holdout. Rolling unsupervised updates use previously observed covariates, including past shocks. A later frozen or expanding supervised evaluation must state its separate training, validation and test protocol.

Near-zero aggregate means do not establish Gaussian scores or removal of seasonality. Large positive deviations are retained; they must not be converted into Gaussian tail probabilities or described as fitted event thresholds. No score covariance, quantile threshold or shock definition has been estimated here.

The saved weekday/slot diagnostics expose remaining mean and scale variation. The following descriptive RMS is the observation-weighted dispersion of cell mean z around the overall mean, using cells with at least 1,800 calibrated seconds. It is not a significance test, and dependent seconds are not treated as independent replications.

| Scenario | Channel | Diagnostic cells | Weighted RMS of remaining cell means | Lowest cell mean | Highest cell mean |
|---|---|---:|---:|---:|---:|
| structural_only | tick_count | 480 | 0.015807 | -0.021731 | 0.090633 |
| structural_only | abs_quote_ofi | 480 | 0.011137 | -0.029129 | 0.090342 |
| fresh_15s | tick_count | 480 | 0.017052 | -0.021629 | 0.123678 |
| fresh_15s | abs_quote_ofi | 480 | 0.011911 | -0.029124 | 0.125717 |
| fresh_30s | tick_count | 480 | 0.016203 | -0.021730 | 0.102601 |
| fresh_30s | abs_quote_ofi | 480 | 0.011419 | -0.029131 | 0.104935 |
| fresh_60s | tick_count | 480 | 0.015892 | -0.021731 | 0.091690 |
| fresh_60s | abs_quote_ofi | 480 | 0.011189 | -0.029129 | 0.092599 |

**Verification**

All **203 tests passed**. Independent verification checked every one of the **81,475,200 normalized clock rows**, **724,224 daily sufficient-statistic rows**, and **724,224 day/cell/channel parameter rows**. This covers **651,801,600 standardized channel entries**, including missingness. Counts, moments, first-availability timestamps, monthly tables, and all weekday/slot diagnostics were reconciled.

Production builds daily moments using NumPy bin counting and centered sums of squares. The verifier reconstructs them using pandas grouped counts, means and variances. It computes pooled between-date variation through pairwise mean differences, separately from the production centered-pooling formula. Source data are read in different chunk sizes (86,400 versus 400,000), and every parameter is fitted before processing its application day in each implementation.

Targeted tests cover current/future-value poisoning, prefix extension, historical-window expiry and inclusive lower bounds, same-weekday matching, warm-up, minimum dates/seconds, zero mean/variance, separate count/flow support, retained quiet zeros, UTC/15-minute/midnight boundaries, non-UTC display timezones, large-offset numerical stability, sign preservation, quote-size unit invariance, no clipping, loader lineage, and corrupted values/parameters/cutoffs/statistics even after refreshing hashes.

**Downstream implications and remaining work**

This resolves future-value contamination in the two reference seasonal channels. It does not repair durations/ACD, validate a primary seasonal specification, or establish liquidity-shock identification. Primary support/window choices and sensitivity to alternative windows, time bases and holiday treatment remain to be settled before paper estimation. The z-scores are relative deviations; simultaneous high count/imbalance is not by itself an identified liquidity shock or an exogenous information event.

Next bounded work is duration observation and chronological ACD calibration, followed by multivariate scoring/threshold calibration and observable event onset. The old detector is disabled; it must not be combined with these scores. Old event candidates, persistence labels, regressions and classifiers remain ineligible. The pipeline is still not ready for paper estimation.

**Artifacts and reproduction**

- Registered run: `data/signals/step04b_seasonal_calibration_20260927/`, selected by `data/active_calibration.json`.
- `standardized_clock_1s.parquet`: full calendar clock with count and absolute-imbalance z-scores for all four support scenarios; unavailable rows are retained.
- `daily_sufficient_statistics.parquet`: eligible counts, means and centered sums of squares for each day/channel/policy/slot.
- `calibration_parameters.parquet`: effective day, historical bounds, sample/date counts, mean, SD, global mean, factor and status for every cell.
- `statistics.json`, `monthly_statistics.parquet`, `weekday_slot_diagnostics.parquet`, `methodology.json`, tests, code snapshots and independent verification provide the audit trail.
- `src.pipeline_state.iter_calibrated_clock(root, policy=...)` validates lineage/hashes and returns raw, multiplicatively adjusted and standardized channels with availability, calibration status, support IDs and missing clock rows. Use common valid observations for joint models and preserve support boundaries in subsequent rolling features.
- `reports/pipeline_master_statistics.json` and `reports/PIPELINE_REPAIR_LOG.md` record the new checkpoint while preserving prior verified and quarantined legacy statistics.
- No purchase, new acquisition, or manuscript edit occurred.

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_seasonal_calibration.py --test-only
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_seasonal_calibration.py --verify-only step04b_seasonal_calibration_20260927
```
New builds require a new unique `--run-id`; reference settings can be changed explicitly through CLI flags and are recorded in the manifest.

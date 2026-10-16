> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Incremental timing normalization: verified forecast comparison

8 October 2026. New run: `step03_timing_forecast_20261008`. Exploratory chronological evaluation of the previously unexecuted comparison, with the specification frozen before fitting. Earlier detector, matched-event and price-forecast studies were not rerun.

**Answer:** adding the working ACD-normalized timing feature to the specified raw/seasonal-timing and quote-condition benchmark did not improve realized forecasts of the following five-minute average quoted spread. It increased mean squared error by **0.267%**. The nominal conditional 95% interval for relative MSE improvement is **−0.614% to +0.080%**. This interval includes zero: neither improvement nor deterioration is statistically established under this uncertainty calculation. The result is a quantified comparison of two forecasting procedures, not evidence that durations lack economic information.

## What changed, and why this avoids a circular test

The overlap audit (full research archive: `reports/RESEARCH_LOOP_AUDIT_AND_NEXT_STEP_2026_10_08.md`; not bundled) identified one remaining contrast: isolate dynamic timing normalization after giving the benchmark raw timing, seasonal timing, both boundary quote ages, update counts and observable quote conditions. Previous tests compared detector selection, matched responses, or price forecasts. Here both forecasts predict the same liquidity target at all common active observation seconds. No alert rule selects a favorable outcome population.

The [Step 02 measurement audit](STEP02_TIMING_MEASUREMENT_AUDIT_2026_10_08.json) established that mean raw incoming gap is `(1 + pre_age − end_age)/count` for the eligible one-second bins. Its explicit transformation remains in the benchmark so that the learner need not rediscover this identity. Seasonal-only timing was newly materialized and independently checked for all 59 archived cycles, covering 20,189,813 predictor rows. This is distinct from repeating the earlier three-cycle physical-source audit.

The extension adds only `negative_log_mean_residual`, the existing ACD-normalized timing summary. It transforms recorded history using issued expectations; it is not an independent new observation. Its effective expectation component also weights within-bin completed gaps. Consequently, this comparison measures incremental representation value for the selected learner and summaries. It cannot isolate a causal effect of quote arrivals or a pure state known before the observation bin opens.

## Fixed design and accounting

The [frozen protocol](STEP03_TIMING_FORECAST_PROTOCOL_2026_10_08.md) specifies the same gradient-boosted squared-error learner and settings for both models, identical training origins and targets, and no parameter search. At each of 55 evaluation cutoffs, each model uses the preceding 56 days of archived predictors and matured labels, with the fixed timestamp-only approximately 5% training sample. No ACD model was re-estimated for this comparison. There are 55 pairs of fitted models, rather than 55 alternative specifications.

Predictions are issued at `t0 = bin_start + 1 second`. The inherited target is the mean of strict-left quoted spreads at `t0+1,…,t0+300`, in pips, within the archived structural support. It measures quoted spread on this recorded feed, not transaction costs, depth, fill probability or marketwide liquidity. The repaired quote-flow proxy is not signed executed order flow.

Predictor/label archives begin on 6 May 2024. A full 56-day training history first exists at cycle 005, so evaluation runs from **1 July 2024 through 31 July 2026**, as fixed before fitting. Cycles 001–004 serve startup training. This population differs from the earlier 59-cycle detector comparison.

| Accounting item | Origins |
|---|---:|
| Scored origins in evaluation cycles | 19,065,217 |
| Missing complete predictors | 5,336 |
| Forecasts issued | 19,059,881 |
| Issued forecasts without an available future label | 17,041 |
| Common paired outcomes evaluated | **19,042,840** |

Forecasts without a later label are retained; future quote ages, detector flags and target magnitude do not determine issuance. Evaluation includes active eligible seconds, not every calendar second or zero-update waiting interval.

## Primary result and its precision

| Forecast | MSE, pips² | RMSE, pips |
|---|---:|---:|
| Raw/seasonal timing and quote-condition benchmark | 0.042948734 | 0.207240763 |
| Same benchmark plus ACD-normalized timing | 0.043063423 | 0.207517283 |

The paired difference is `loss_B − loss_T`, so positive favors added timing. Its mean is **−0.000114689 pips²**; the nominal 56-day HAC interval is **[−0.000278693, +0.000049315] pips²**. The RMSE increase is approximately 0.000277 pips.

Uncertainty uses daily ratio influences across all **761 UTC calendar days**, including zero-observation days. Relative improvement has its own ratio influence, accounting for the benchmark-loss denominator. The two declared lag sensitivities agree with the primary conclusion:

| Calendar-day HAC lag | Nominal 95% interval for relative MSE improvement |
|---|---:|
| 56, primary | [−0.614%, +0.080%] |
| 28 | [−0.616%, +0.081%] |
| 112 | [−0.608%, +0.074%] |

These intervals condition on the issued forecasts. They do not certify repeated-estimation coverage for the entire rolling model pipeline. The cache has already informed earlier project decisions; chronological fitting does not make it an untouched confirmation sample. The millions of overlapping outcomes are not millions of independent replications.

The point estimate and interval give a useful scale for this comparison: realized gains are absent and the nominal upper improvement bound is small. No operational materiality threshold was justified, however, so this does **not** certify economic negligibility, equivalence or viability.

As a prespecified diagnostic, the benchmark's MSE is 16.23% below carrying the current spread forward unchanged. That establishes realized improvement for that whole procedure over that particular simple forecast. It neither assigns the gain to an individual feature nor changes the adverse primary comparison.

## All declared patterns

!All fixed forecast diagnostics (full research archive: `output/figures/timing_forecast_fixed_patterns.png`; not bundled)

Positive reductions favor added timing. The bars are descriptive estimates without subgroup inference. Spread and timing categories use each cycle's prior-training quartile thresholds. Ties remain, so groups need not contain 25% of evaluated observations, and pooled group labels do not denote one common absolute threshold. The final calendar quarter contains July only. No category was selected because it performed favorably.

| Dimension | Category | Outcomes | Mean target, pips | Benchmark MSE | Added-timing MSE | MSE reduction |
|---|---|---:|---:|---:|---:|---:|
| Spread category | 1 | 2,538,115 | 0.326264 | 0.004212 | 0.004211 | +0.019% |
| Spread category | 2 | 3,486,743 | 0.365877 | 0.003701 | 0.003694 | +0.170% |
| Spread category | 3 | 5,060,569 | 0.338927 | 0.004395 | 0.004389 | +0.153% |
| Spread category | 4 | 7,957,413 | 0.502486 | 0.097020 | 0.097302 | -0.290% |
| UTC hour | 00–05 | 3,738,088 | 0.409317 | 0.008541 | 0.008604 | -0.739% |
| UTC hour | 06–11 | 5,548,160 | 0.355186 | 0.007009 | 0.007072 | -0.898% |
| UTC hour | 12–17 | 6,747,027 | 0.341884 | 0.003648 | 0.003711 | -1.725% |
| UTC hour | 18–23 | 3,009,565 | 0.667896 | 0.240047 | 0.240437 | -0.163% |
| Updates | 1 | 7,416,749 | 0.416584 | 0.053884 | 0.053977 | -0.173% |
| Updates | 2–4 | 9,436,148 | 0.398672 | 0.033429 | 0.033561 | -0.395% |
| Updates | 5+ | 2,189,943 | 0.441035 | 0.046935 | 0.047048 | -0.242% |
| Timing category | 1 | 4,775,025 | 0.402668 | 0.037441 | 0.037585 | -0.385% |
| Timing category | 2 | 4,788,010 | 0.397841 | 0.031287 | 0.031359 | -0.228% |
| Timing category | 3 | 4,763,161 | 0.404823 | 0.037171 | 0.037287 | -0.313% |
| Timing category | 4 | 4,716,644 | 0.437094 | 0.066197 | 0.066324 | -0.192% |
| Quarter | 2024Q3 | 2,219,772 | 0.251211 | 0.012172 | 0.012142 | +0.246% |
| Quarter | 2024Q4 | 2,402,729 | 0.289169 | 0.040518 | 0.040523 | -0.013% |
| Quarter | 2025Q1 | 2,562,104 | 0.484279 | 0.034852 | 0.034765 | +0.249% |
| Quarter | 2025Q2 | 2,960,452 | 0.608263 | 0.062028 | 0.062563 | -0.861% |
| Quarter | 2025Q3 | 1,998,986 | 0.451622 | 0.041477 | 0.041541 | -0.154% |
| Quarter | 2025Q4 | 1,867,692 | 0.430768 | 0.060409 | 0.060326 | +0.137% |
| Quarter | 2026Q1 | 2,237,819 | 0.356919 | 0.030444 | 0.030467 | -0.074% |
| Quarter | 2026Q2 | 2,123,318 | 0.366423 | 0.059731 | 0.060123 | -0.656% |
| Quarter | 2026Q3 (July) | 669,968 | 0.357418 | 0.044592 | 0.044627 | -0.078% |

The [saved pattern table](../data/signals/step03_timing_forecast_20261008/patterns.parquet) additionally records both prediction means and sufficient statistics. [Cell accounting](../data/signals/step03_timing_forecast_20261008/pattern_cell_coverage.parquet) retains all 825 planned per-cycle categories, including 16 with no evaluated observations. This reporting completion changed no fitted model, sample or primary estimate.

### Interpretation of the loss concentration

**Small gains in lower-spread categories are overwhelmed where errors concentrate.** The highest historical spread category contains 41.8% of evaluated origins but **94.4% of the benchmark's total squared error**. Added timing improves the other three category point estimates slightly, with a combined summed loss reduction of approximately 57.97 pips², but worsens the highest category by approximately 2,241.97 pips². Its relative deterioration is 0.290%. This identifies where the pooled comparison is decided. It does not show that wide spreads cause normalization to fail; squared loss can concentrate mechanically in difficult observations, and model approximation and feed support can also matter.

**Calendar context matters more than a uniform “busy versus quiet” story would suggest.** The 18:00–23:59 UTC block contains 15.8% of origins and **88.3% of benchmark squared error**. Added timing nevertheless worsens all four UTC-block point estimates. It also worsens all three update-count groups and all four normalized-timing categories. Thus the declared bins do not reveal a general incremental gain restricted to fast, slow, busy or less busy active observations. A six-hour block cannot identify a rollover, closure or fixing mechanism.

**Performance varies across periods and is influenced by particular dates.** Three of the nine reported calendar quarters have positive point estimates, while six are adverse. The largest adverse day, 18 April 2025, supplies 55.5% of the *net summed deterioration*. This is a signed contribution to the total, not a share of sampling variance. Removing any single calendar day, without refitting, leaves the paired mean adverse: the range is approximately −0.000122446 to −0.000051204 pips². The outcome therefore is not reversed by one exceptional date, although date concentration limits how broadly it should be interpreted.

The ten largest absolute daily contributions are retained below, including favorable dates.

| Date, UTC | Outcomes | Summed paired loss reduction, pips² |
|---|---:|---:|
| 2025-04-18 | 45,220 | -1,211.248 |
| 2026-04-29 | 36,929 | -549.827 |
| 2025-04-06 | 6,946 | -408.657 |
| 2026-06-19 | 24,862 | -236.680 |
| 2025-07-04 | 26,795 | -192.465 |
| 2026-06-24 | 33,765 | +143.580 |
| 2026-04-03 | 13,454 | -140.849 |
| 2025-04-09 | 71,721 | +137.855 |
| 2026-04-23 | 40,865 | -119.693 |
| 2025-04-11 | 65,266 | +104.064 |

There is a plausible calendar clue, not an identified explanation. Both 18 April 2025 and 3 April 2026 were Good Friday, as listed by the [official public-holiday calendar](https://www.stirling.gov.uk/about-the-council/public-holidays/). This calendar lookup occurred after the forecast results and is post hoc. On the 2025 date, mean observed future spread was 1.380 pips versus forecasts of 1.779 and 1.802: both models overpredicted, and adding normalized timing moved farther upward. On the 2026 date, the corresponding means were 0.705, 0.576 and 0.561: both underpredicted, and added timing moved farther downward. These opposite directions argue against describing the observed failures as one common underprediction mechanism. They are consistent with date-dependent calibration problems, but aggregated biases do not decompose squared loss or establish a holiday effect. Dates are retained and no holiday-specific model was fitted.

## What this can add to the paper

A defensible sentence is:

> In an exploratory chronological evaluation of recorded EUR/USD quotes, adding working ACD-normalized timing to a fixed benchmark containing raw and seasonal timing and observable quote conditions increased five-minute average-spread forecast MSE by 0.27%. Nominal conditional uncertainty allowed approximately 0.08% relative improvement at its upper bound. Forecast loss and the adverse incremental result were concentrated in observations with higher historically defined spreads and varied across calendar periods.

The modest potential insight is a **boundary on the usefulness of this normalization in this specified application**, together with transparent evidence about where its comparison is decided. It qualifies an interpretation that a richer timing representation necessarily improves liquidity forecasts. It does not establish a new timing–liquidity relationship, a universal failure of ACD, or an economically successful detector. Nor is this forecast test a causal validation of the detector or its Mahalanobis weighting.

The existing literature audit already identified strong precedents: [Dionne and Zhou](https://www.researchgate.net/publication/337577627_The_dynamics_of_ex-ante_weighted_spread_an_empirical_analysis) combine duration models, spread dynamics and other market conditions, while [Tse and Dong](https://www.sciencedirect.com/science/article/pii/S0927539814000425) assess duration adjustments for a different economic estimate. Their designs and outcomes differ from this pair; broad multivariate duration analysis and normalization comparisons are established. An exact precedent was not established by the inspected sources, and that is not proof of priority. The result is new to this project, with possible application-specific value; a field-first or journal-readiness claim remains unsupported.

Careful drafting can now state an actual quantitative answer rather than promising to find one. This result belongs alongside the retained finite-horizon adjustment evidence and unsuccessful fixed benchmarks. It does not, by itself, justify rebuilding the paper around a claimed forecasting advance. The next writing step is to integrate the measured boundary and its limitations. Stronger confirmation would require genuinely unexamined observations or a comparable independent feed; the current comparison needed no additional data. No new predictor, horizon, subgroup exclusion or learner search follows automatically from these diagnostics.

## Verification and reproduction

Twelve unit tests passed before the core code/protocol freeze. The [feature receipt](../data/signals/step03_timing_forecast_20261008/feature_verification.json) independently reconstructs all 20,189,813 seasonal predictor rows; maximum absolute seasonal-mean discrepancy is 1.42e−14. The [results receipt](../data/signals/step03_timing_forecast_20261008/verification.json) verifies every paired outcome, 3,837,748 training-origin instances, 110,000 saved-model prediction checks, daily and subgroup aggregates, and all three HAC calculations using an independent matrix calculation. All 39 protected earlier scientific files remained unchanged. Numerical verification does not certify economic identification or statistical coverage.

The frozen [manifest](../data/signals/step03_timing_forecast_20261008/manifest.json), [summary](../data/signals/step03_timing_forecast_20261008/summary.json), copied sources under the run's `code/` directory, and per-cycle models/predictions retain the result. The separate [reporting receipt](../data/signals/step03_timing_forecast_20261008/reporting_receipt.json) discloses completion of empty descriptive cells. The normalizer remains a working model with previously recorded calibration flags.

To reproduce into a **fresh** run, from this repository and the recorded runtime, execute the phases in order. The new run identifier below is illustrative, not a run already executed:

```sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/miniconda3/envs/fxresearch/bin/python -m unittest discover -s tests -p test_timing_forecast.py
/opt/miniconda3/envs/fxresearch/bin/python run_timing_forecast.py features --run-id step03_timing_forecast_replication
/opt/miniconda3/envs/fxresearch/bin/python verify_timing_forecast.py features --run-id step03_timing_forecast_replication
/opt/miniconda3/envs/fxresearch/bin/python run_timing_forecast.py fit --run-id step03_timing_forecast_replication
/opt/miniconda3/envs/fxresearch/bin/python verify_timing_forecast.py results --run-id step03_timing_forecast_replication
/opt/miniconda3/envs/fxresearch/bin/python reports/complete_timing_forecast_reporting.py --run-id step03_timing_forecast_replication
```

The supplementary report tables above are rendered directly from verified saved summaries and aggregates. They do not involve new fits or changes to the original losses.

> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# One forecast comparison: incremental ACD normalization

8 October 2026. Frozen before new future-outcome contrasts or forecast fits. The question and outcome were chosen after inspecting earlier work on this cache, so this is an exploratory chronological comparison, not an untouched holdout or external preregistration.

## Distinction from completed work

Reuse the passed Step 02 measurement audit and the verified Step 11 feature/label archive. Do not replay the detector comparisons, matched-event comparisons, state/sequence forecast extensions, or Step 02 numerical audit. The new computation reconstructs seasonal-only timing on the archived scored population, then compares a fixed pair of future-spread forecasts at common active observation seconds. Earlier studies compared detector selection or predicted price continuation; this test isolates the incremental representation from ACD normalization. The overlap audit already establishes strong literature antecedents. No claim of being first follows from this new specification.

## Population, clock and startup

Use scored deployment seconds from the existing chronological panel. Predict at `t0 = bin_start + 1 second`. The target is the inherited `y300`: the mean of strict-left quoted spreads at `t0+1,...,t0+300`, in pips, within the original structural support. Reuse its verified labels and availability flags unchanged. Do not use future quote ages, alerts, direction, target magnitude or a desired result to select origins.

Both archived deployment predictors and future-spread labels are available from 6 May 2024. A full 56-day training history first exists at the 1 July 2024 cutoff (cycle 005). Evaluate cycles 005–059 through 31 July 2026; use cycles 001–004 only for startup training, with subsequent 56-day rolling archives. Do not retroactively refilter historical observations with newly fitted ACD parameters. Normalized predictors use their originally issued states.

Training origins must satisfy `bin_start >= cutoff-56 days`, `bin_start+301 seconds <= cutoff`, complete predictors, available labels and the fixed timestamp-only SplitMix64 modulo-20 sample. This reuses the approximately 5% sampling rule, independent of outcomes. At least 1,000 training rows are required; failure stops the run. Forecast at every complete-feature deployment origin, preserving predictions whose future outcomes are unavailable. Compare losses on the common evaluable origins and account for every exclusion.

## Measurements and redundancy

Raw mean gap is the verified identity `(1+pre_age-end_age)/count`. Include its negative logarithm explicitly in the benchmark, despite its algebraic redundancy with count/ages, so a finite learner need not rediscover that transformation. Reconstruct the seasonal-only feature as `-log(mean(gap/origin-issued scale))`, on the same completed incoming gaps; average before logarithms. Check all completed counts, exposure, positive finite scales and seasonal cutoffs no later than origins. Independent source reconstruction verifies the new seasonal means and predictor consistency before economic fitting.

The extension adds only the existing `negative_log_mean_residual`. Any benefit is predictive representation value conditional on the selected summaries and learner. The normalizer is a working model with known calibration flags, not a calibrated shock probability. Its effective expectation component also uses within-bin gap weights, so it cannot be interpreted purely as a state known before the bin opens.

## Exact model pair

Both models use HistGradientBoostingRegressor: squared_error, learning_rate 0.05, max_iter 100, max_leaf_nodes 15, min_samples_leaf 100, l2_regularization 1, max_bins 63, early_stopping False, random_state 20261004. These are the previously fixed reference-model settings. Fit exactly two models per evaluation cutoff. No model/parameter alternatives, feature selection, early stopping, horizon search or retry to improve the result.

Benchmark columns: current_spread, pre_spread, history_spread, variation, log_count, log_history_count, quote_age, history_age, hour, weekday, quote_ofi, adjusted_abs_quote_ofi, adjusted_tick_count, pre_quote_age_sec, raw_speed, seasonal_speed. The first ten retain the Step 11 definitions; quote_age is the end-boundary age, and pre_quote_age_sec supplies the other boundary. raw_speed is minus log mean raw gap. seasonal_speed is the seasonal-only summary above.

The extension contains those same 16 columns plus negative_log_mean_residual. Use the identical training origins, target values and refit schedule. No normalization parameter is re-estimated by this study. A current-spread carry forecast is retained solely as a diagnostic, without fitting a third model.

## Primary statistic and uncertainty

For every shared evaluable origin, save predictions, target, loss_B, loss_T and their difference `d = loss_B-loss_T`. Primary estimate is mean(d), in squared pips; positive favors the extension. Report both MSEs, RMSEs and realized relative MSE reduction. There is no justified operational materiality margin, so do not certify economic negligibility, equivalence or viability.

Aggregate sufficient statistics to every UTC calendar day from 1 July 2024 through 31 July 2026, including days with zero scored observations. For the paired mean, use daily ratio influence `(sum_d_day-Delta*n_day)/total_n`. Apply Bartlett HAC with 56 calendar-day lags; 28 and 112 are fixed sensitivity choices. Report nominal normal 95% intervals, without selecting a favorable lag. For relative loss reduction, use its own daily ratio influence rather than dividing an interval by an uncertain denominator. These intervals condition on the issued forecasts and do not establish repeated-estimation coverage of the whole model pipeline.

## Fixed descriptive patterns

Save all of the following, regardless of sign: calendar-quarter loss differences; UTC six-hour blocks; update-count groups 1, 2–4 and 5+; current-spread quartiles; and normalized-timing quartiles. Quartile cutoffs come only from that cycle's training predictors; retain ties and empty cells. Report counts, target/prediction means and both losses. These are descriptive diagnostics, not additional discovered hypotheses or independent replications.

Also report the ten largest absolute daily summed loss differences and the range of leave-one-calendar-day-out means, without changing the primary sample or refitting. This assesses whether a few dates dominate the realized comparison. Wider-spread groups can dominate squared loss mechanically; concentration does not identify a causal economic event. Singleton bins help distinguish origin-expectation representation from within-bin updating, but subgroup performance is not proof of either mechanism.

## Verification, preservation and stopping rule

Freeze code, protocol, runtime, source hashes and protected scientific state in a new run. Measurement tests must pass, all new predictor cycles must be built, and an independent measurement receipt must pass before fitting. Verify label maturity and training identities, saved model predictions, every paired outcome identity, daily sufficient statistics, HAC intervals and descriptive aggregates. Preserve earlier files and active pointers. Record implementation repairs; do not alter the scientific specification after losses become available.

Run one pair and report positive, negative or inconclusive evidence with its actual scope. Descriptive patterns may motivate future work but do not authorize fitting alternative subgroups/models in this run. New field-level novelty, causation, executed-cost savings, marketwide liquidity and submission readiness remain unestablished by construction.

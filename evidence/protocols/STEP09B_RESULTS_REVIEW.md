> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 09b: chronological persistence-forecast results

Completed 3 October 2026. Verified run: `step09b_persistence_forecast_20261003`. The frozen pressure-by-variation forecasting rule did **not** improve realized five-minute continuation prediction. This is a negative result for this specified algorithm, not proof that every preceding condition lacks predictive information.

## What was tested

At the first recorded alert, predict the subsequent direction-oriented continuation C300, in pips, using strictly preceding states. The known initial displacement J translates this to total displacement: predicted I300 = J + predicted C300. Both targets have the same forecast error. No ratio, binary informational label or arbitrary 0-100 score is introduced. Event detection remains unchanged.

Two fixed OLS equations share signed preceding pressure, pressure magnitude, preceding price variation, spread, log displayed size and hour/weekday effects. The interaction equation adds only variation times pressure magnitude. A zero-continuation reference and expanding historical-mean reference provide context. The interaction-versus-additive raw squared-loss difference is primary. Fifty-nine scheduled updates entail 118 OLS fits and 59 historical-mean calibrations; these are repeated fitting operations for four declared methods, not 177 competing model specifications.

Each training label must have matured by that cutoff. Predictions are saved, hashed and independently verified before the separate outcome evaluation begins. All eligible origins receive predictions, including those whose future labels turn out to be unavailable. Both state equations use the identical declared historical-mean fallback for an unseen marginal hour or weekday. This occurs for two origins in each equation and creates zero interaction-versus-additive loss difference for those origins.

## Results on exactly the same 127,013 evaluable origins

Evaluation covers forecast origins from 6 May 2024 through 31 July 2026. MSE pools individual squared errors; RMSE is its square root. The zero rule means no *further* movement after the initial observed displacement, not zero total event impact.

| Fixed method | MSE, pips squared | RMSE, pips |
|---|---:|---:|
| Zero continuation | 10.5249179415 | 3.2442129926 |
| Historical training mean | 10.5250685343 | 3.2442362020 |
| Additive preceding-state equation | 10.5472785768 | 3.2476573983 |
| Same equation plus pressure-variation interaction | 10.5487942172 | 3.2478907336 |

The interaction increases MSE relative to the additive equation by **0.001515640454 pips squared, or 0.014370%**. It increases MSE relative to the historical mean by **0.225421%**, and relative to zero continuation by **0.226855%**. The two state equations therefore supply no observed improvement over these simple references in this replay. No significance, population equivalence or economic profitability statement follows from these differences.

The saved quarter table changes sign for the primary comparison: the interaction has lower MSE in four of ten calendar-quarter segments and higher MSE in six. The first and last segments are partial quarters. The interaction has higher MSE than zero and the historical mean in all ten segments. These are summaries of the same fixed predictions, not independent replications or a basis for selecting subperiods.

## Coverage and concentration

| Quantity | Verified value |
|---|---:|
| Later recorded origins retained | 133,918 |
| Eligible origins receiving all four forecasts | 127,058 |
| Ineligible origins, explicit abstentions | 6,860 |
| Eligible origins with measured C300 | 127,013 |
| Eligible origins with unavailable C300 | 45 |
| Forecast-table rows, including abstentions | 535,672 |
| Numeric forecasts across all four methods | 508,232 |
| Complete calendar dates / dates with scored origins | 817 / 701 |
| Mean fallbacks in each state equation | 2 |

The maximum daily absolute primary-loss-difference share is 10.7744%, on 2 April 2025. Its denominator is the sum of absolute daily loss differences. That date also supplies 38.9895% of the sum of squared centered daily primary-loss differences. This second percentage measures concentration of a diagnostic, not explained variance or share of total forecast error. On that date the interaction actually improves on the additive equation; it must not be described as a date causing the overall deterioration. All observations and dates remain, with the same event-weighted estimand.

The daily ledger, 59 refit-period summaries, ten quarter summaries and fixed 0-40 lag diagnostics are saved. Empty dates carry zero counts and sums, not imputed outcomes or assurances that the feed was complete. Serial and concentration diagnostics do not establish a valid sampling distribution. This stage reports no p-values or confidence intervals. Nested recursively estimated models require an appropriate forecast-inference argument; ordinary normal inference is not assumed.

## What is verified and what is not

The parent recheck reconstructs stage 09a and rehashes 164 ancestor files, including all 31 cached monthly inputs; all match their recorded versions. This verifies fidelity to the intended existing cache, not vendor completeness. No upstream detector or raw-data acquisition was rerun.

All 36 focused synthetic tests pass. Independent code reconstructs exact matured training memberships, all 118 OLS fits through QR rather than the producer's SVD, 4,061 coefficients, all 535,672 forecast records, individual losses and every reported summary. Saved Parquet and JSON artifacts pass the same checks after reopening. Source, protocol, parent inputs, test receipts and output hashes are bound in the manifest; forecasts were saved and verified before scoring. Every scheduled design was full rank under the declared numerical rule. The largest equilibrated condition number was 19.565363; full rank is an arithmetic check, not a guarantee of statistical stability.

The independent checks support correctness within the explicit contract. They do not prove that no methodological error could remain. The cache has already been inspected, observation support is inherited, zero processing latency is assumed, and missing future outcomes cannot be scored. There is no untouched confirmatory holdout, established causal identification, informational/inventory decomposition, permanent-impact conclusion, two-regime validation or established novelty.

## Research decision and next bounded task

Notebook 5's prediction objective is now implemented and tested. A continuous forecast can be useful as a score only if its predictive content is supported; relabeling this output as a persistence score cannot turn the observed loss comparison into success. Earlier average total displacement can coexist with this negative result: I = J + C can remain positive because J is already known, while C is hard to forecast from the declared preceding conditions.

Preserve the negative result. Do not tune thresholds, horizons, history features, model classes or calendar splits to replace it. The next bounded task is **one evidence-and-claims reconciliation**: connect the existing joint-response, conditional-association and chronological-forecast tables to the original research questions; distinguish what is measured, associated, predicted and identified; and state which contribution the existing evidence can actually support. This requires no new empirical fit or data purchase. A stronger predictive or structural claim would require a separately justified design and validation, not a new label for the current estimates. JFE-level novelty and paper readiness remain unestablished.

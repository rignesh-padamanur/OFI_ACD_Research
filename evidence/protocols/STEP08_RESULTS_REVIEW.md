> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step 08: what the preferred research pathway establishes

3 October 2026. Verified exploratory run: `step08_spread_history_20261003_v2`. This is one prespecified comparison on an already inspected cache. It is not an untouched prediction test, a new detector, or a correction to the old regression estimates.

## Decision

**Stop the two-endpoint spread-history extension as a proposed headline explanation of price persistence.** It did not establish incremental price-response information in the declared comparison. Preserve its measured spread responses and the complete exploratory result. Do not replace it with another history variable, horizon, threshold or interaction selected after these results.

This repairs the research direction: explanatory variables are strictly before onset, the initial price adjustment remains visible, and later spread adjustment is an outcome. It does not establish a novel economic contribution. The present data support studying finite-horizon recorded quote responses conditional on detected events; they do not identify informational trading or dealer inventories.

## What changed and what remained fixed

Added exact bid/ask marks at onset minus 300 seconds, onset, detection, and detection plus 60/300 seconds. Every mark uses the last quote strictly before its target and the original structural support. Physical source identities, timestamps, ages and missingness are saved. Integer price ticks define spread differences and signs. The detector, q99 threshold, onset IDs, direction rule, parent price responses and inherited cohorts are unchanged. No future spread or price sign selects observations.

Measurement covers 135,724 onsets and 678,620 requested marks over 60 cycles. Of these, 678,263 marks are available; the missing marks remain in the full ledger. All 556,173 distinct available physical quotes match exact cached monthly source rows across 28 files. All 270,836 prior-state boundary marks reconcile. The already eligible 127,013 primary and 127,052 secondary observations lose **zero** additional rows. This does not erase earlier sample exclusions or establish completeness of the vendor feed.

Among the primary observations, initial spreads widened for 48,239 (37.98%), stayed unchanged for 33,169 (26.11%), and narrowed for 45,605 (35.91%). The q99 event rule detects unusual combinations of quote variables; it is not a spread-widening classifier. Non-widening does not prove absence of liquidity stress.

Exact source identity is not proof of freshness. In the primary sample, the 99th percentile ages are 60.52 seconds at the earlier history boundary and 54.15 seconds before onset. The longest pre-onset mark age is 1,060.516 seconds. Only 123,944 of 127,013 observations have all four required marks no older than 60 seconds. These are descriptions, not a newly selected filter.

## The frozen comparison

Let H = current pre-onset spread minus the spread at the preceding 300-second boundary, in pips. Positive H means the two endpoints widened; it does not describe the intervening trajectory. At fixed current spread, H is simply the negative of the earlier spread plus a constant.

Both models contain the same strictly prior signed quote pressure, pressure magnitude, price variation, magnitude-variation interaction, current spread, displayed size, prior raw price change, three declared squared terms, and inherited hour/weekday indicators. The benchmark has 39 columns including its intercept. The expanded model adds only H, for 40 columns. Both use the same rows within each horizon. Future direction, detection-time alignment and initial displacement J are excluded as predictors.

The primary target is total direction-aligned displacement I at five minutes. Initial displacement J and post-detection movement C explain the identity I = J + C. Three spread-change outcomes have the corresponding additive identity. These are dependent accounting coordinates, not six independent confirmations. Outcomes use the inherited direction learned at detection, so this is neither an advance price-direction forecast nor a warning of event arrival.

The protocol was frozen before the first history-response fit. Both fixed 20- and 40-calendar-day Bartlett HAC calculations include all 817 calendar dates, with 701 occupied and 116 zero-event dates. Intervals are pointwise nominal 95% normal intervals; their coverage is unvalidated. No p-values, tuning, iid-event inference or model selection were introduced. Twelve equations per horizon are the two nested designs for six accounting outcomes, not twelve competing pathways.

## Results: primary 300-second horizon

Slopes below are outcome pips per pip of H, conditional on the fixed benchmark. They are not causal effects or demonstrated feasible one-pip interventions.

| Outcome | H slope | Nominal 95% interval, 20 days | Nominal 95% interval, 40 days |
|---|---:|---:|---:|
| Initial price displacement J | +0.000686 | [-0.005151, +0.006523] | [-0.005209, +0.006581] |
| Subsequent price movement C | +0.003891 | [-0.014677, +0.022460] | [-0.013575, +0.021358] |
| Total price displacement I | +0.004577 | [-0.012887, +0.022041] | [-0.011567, +0.020721] |
| Initial spread change | -0.075012 | [-0.089123, -0.060900] | [-0.089530, -0.060494] |
| Post-detection spread change | -0.097585 | [-0.121325, -0.073846] | [-0.123773, -0.071398] |
| Total spread change | -0.172597 | [-0.199219, -0.145975] | [-0.202932, -0.142262] |

For I, adding H increases in-sample R-squared by only 0.0000012155, from 0.00378695 to 0.00378816, and reduces RMSE from 3.26613855 to 3.26613656 pips. This is descriptive in-sample fit, not validated predictive performance. J and C provide no resolved incremental price channel in these provisional intervals. An interval crossing zero is not proof of a zero effect or equivalence.

The secondary 60-second I slope is +0.010173, with intervals [-0.005447, +0.025794] and [-0.004281, +0.024628]. Its total-spread slope is -0.133456, with intervals [-0.157426, -0.109485] and [-0.160833, -0.106079]. This does not supply a reason to substitute 60 seconds for the primary horizon. All secondary coordinates and benchmark coefficients remain saved, including the 39 additional inherited observations.

Mean price coordinates are J = +0.380258, C = +0.014752 and I = +0.395010 pips. Mean spread changes are +0.016301 initially, -0.036315 subsequently and -0.020014 overall. These describe different transformations of the same quotes. They do not establish eventwise recovery, permanent impact, or an independent spread-based mechanism.

## Support, uncertainty and counterexamples

1. **Ordinary spread persistence can generate the result.** Suppose future spread is a times current spread plus b times earlier spread plus noise, with b positive, and prices unrelated. Substituting H gives future-minus-current spread = (a+b-1) times current spread minus b times H plus noise. A negative H slope follows without informational price discovery or inventory rebalancing. Because current spread is controlled, its shared subtraction alone does not mechanically force an H effect.
2. **The support is not an intervention experiment.** H has median zero and interquartile range [-0.1, +0.1] pips, but tails extend from -16.7 to +20.2. Positive earlier spreads imply H < current spread. On the verified 0.1-pip spread grid, at current spread 0.1 pip, positive H is impossible; at 0.3 pip the observed maximum is 0.2. A universal one-pip widening contrast extrapolates in common states. Around 82.03% of centered H variation remains after benchmark residualization, so exact collinearity is not the explanation for the negligible incremental fit.
3. **Many events do not guarantee reliable inference.** For I300, 36.79% of summed squared daily H influence lies on 22 April 2025 and 69.42% in 2025 Q2. For I60 the corresponding maximum day accounts for 66.69%. These are concentration diagnostics, not shares of the coefficient, HAC variance, or effective sample size. No influential date is deleted. Dependence and coverage remain unresolved.
4. **An inconclusive pooled association is not a universal impossibility theorem.** Different state-specific relationships might cancel; low-order controls might miss nonlinearities; stale quotes, common news and event selection might confound associations. Those possibilities limit the conclusion but do not justify searching this reused cache until a preferred result appears.
5. **Different-looking price and spread intervals are not a tested difference between mechanisms.** The outcomes have different meanings and noise scales, share bid/ask observations, and were not subjected to a formal mechanism contrast. The result is a recorded spread association and unresolved incremental price evidence in this specification.

## Actionable repair to the paper

Use the operational RQ: **Which conditions observable before a detected quote-stress event are associated with its initial and subsequent recorded price adjustment over the next one to five minutes?** Keep the original pressure, prior variation and finite-horizon persistence question. Treat informative trading versus inventory adjustment as possible interpretations that these data cannot distinguish, not identified labels. Use “quoted” or “quote-derived” consistently; best-quote size changes are not signed executed trading flow.

Keep the chain short: verified quote data -> fixed detector and episodes -> strictly prior state -> joint J/C/I outcomes -> declared state comparison -> dependence and claim limits. Spread responses remain a supporting description. Retire H as the headline price mechanism; leave the legacy persistence score out of informational classification. Do not call the result a discovery of two regimes or permanent impact.

The next bounded task is a **final evidence and claims table**, using saved results: for each original RQ, identify the observable estimand, prior-versus-detection timing, exact saved table, effect size and uncertainty, closest literature, and remaining identification gap. Separate the new strictly prior benchmark from older detection-time alignment equations; they answer different questions. This synthesis requires no new history candidates or detector reruns. It must decide whether the original pressure/variation evidence already supports a distinctive, precisely bounded empirical claim. A forecast claim would require a separately frozen chronological evaluation; the inspected cache cannot become an untouched holdout. A warning of event arrival would additionally require eligible non-event opportunities. Neither extension is implied by this result.

The earlier literature review already identifies overlapping work on pre-jump FX conditions and price/spread resilience; see `RESEARCH_PATHWAYS_2026_10_03.md` for sources and the limits of source access. This task does not establish priority or novelty. **The honest current conclusion is a coherent, auditable empirical project with a constrained contribution question, not a settled novel informational-liquidity finding or a paper-ready JFE result.**

## Verification and reproducibility

Forty-three focused tests passed, covering measurement, analysis, independent verification and numerical cancellation. All saved inputs reconcile to original states, memberships and quote marks. All nine analysis tables plus statistics at both horizons pass independent QR/FWL/covariance reconstruction. Source comparisons and integer spread accounting pass. See the saved run, test evidence and final integrity record; prior tests are not represented as rerun.

The original attempt is preserved as `step08_spread_history_20261003` with analysis_failed status. Its fixed-tolerance verifier rejected cancellation-sensitive score aggregation and nested SSE subtraction. The repair binds aggregation to independently checked event residuals using a computed floating-point summation bound, and nested metadata to independently checked saved diagnostics. Independent QR coefficient, residual, covariance and influence checks remain. No model, sample or producer arithmetic changed. Version 2 reuses measurement files byte-for-byte and saves the repaired verifier before re-estimating the identical declared equations. The audit records reconstruction/threading qualifications.

Executed entry points: `repair_spread_history.py` (measurement, reuse-measurement, analysis) and `reports/step08_saved_verification.py` (bind-inputs, raw-source, reload-analysis, with explicit --run-id). This is a versioned research extension corresponding to Notebook 2 prior-state measurement, Notebooks 3/4 joint outcomes and Notebook 5 comparison; the legacy notebooks and estimates are preserved rather than silently overwritten.

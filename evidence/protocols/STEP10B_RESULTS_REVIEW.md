> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step10b result: the sequence extension did not improve realized forecasting

3 October2026. Completed run: `step10b_sequence_forecast_20261003`. The user authorized continuing the contribution work after10a. One predeclared exploratory comparison was executed; there was no feature, horizon or model-family search. manuscript.tex was excluded.

## Decision and connection to the research question

**The frozen sequence extension does not support the proposed headline predictive contribution.** Its realized event-weighted MSE is 0.00271415% higher than the declared benchmark and 0.57145204% higher than no further movement. Under the rule fixed before estimation, stop this candidate as a supported headline extension. Do not tune another sequence metric, interaction, threshold or horizon to reverse this decision.

The question was whether preceding quote-pressure ordering helps forecast five-minute price adjustment after the first alert, beyond the exact stage10a aggregate-state and recent-dynamics benchmark. This is closely connected to the preceding-condition RQ. It is a conditional forecast at detection, not a warning before event arrival or a test identifying private information versus dealer inventory. The new comparison establishes neither a successful previously undocumented relationship nor broad absence of predictability.

The earlier gate correctly prohibited automatic fitting. This task distinguished a credible question from an already established finding: one may test a credible candidate without guaranteeing novelty. Its negative result now supplies a concrete stopping decision. The previous unsuccessful pressure-by-variation forecast and stopped spread-history extension remain visible and unchanged.

## What changed, and what was held fixed

Stage10a's conventional centered adjacent-sign statistic S, exact19 numeric benchmark variables, calendar terms, detector, eligibility, five-minute target, support and sampling rules were unchanged. The two fitted equations differ by S alone. The benchmark includes direction d and initial movement J known at the first alert, prior net/gross pressure, quoted conditions, activity, sign composition and fixed recent pressure/price summaries. These recent summaries are themselves order-dependent; this is not a comparison to a wholly order-free information set.

Relative to09b, this is a different, richer benchmark including alert-time information. Comparing a new fitted equation with an old one would not isolate S. The primary comparison therefore uses the two new equations on identical origins, matured training labels and refits. All-one-sign and all-zero sequences can both have S=0; S represents arrangement conditional on composition, not total pressure persistence.

There were four declared forecasting methods: zero, historical matured-label mean, baseline and baseline plus S. Across 59 fixed refits, the two equations required 118 OLS fits and 59 mean calibrations. These are scheduled updates, not177 candidate models. Both equations used the same training-only numerical scaling and calendar fallback. No alternative classifier, persistence score or economic mechanism was fitted.

## Verified primary results

All methods below use the same 127,013 available outcomes. MSE is in pips squared; RMSE is in pips. Lower values mean better realized forecasting.

| Fixed forecast | MSE | RMSE |
|---|---:|---:|
| No further movement | 10.524917941471 | 3.244212992618 |
| Historical matured-label mean | 10.525068534285 | 3.244236201987 |
| Frozen first-alert benchmark | 10.584775513686 | 3.253425197186 |
| Same benchmark plus S | 10.585062800226 | 3.253469348284 |

The primary sequence-minus-baseline mean loss difference is **+0.000287286540 pip squared**; summed loss difference is +36.489125276929. The RMSE increase is +0.000044151098 pips. The small unfavorable difference is not a significance, equivalence or materiality test. No p-value or confidence interval was computed. The zero and mean reference statistics exactly match the earlier09b saved results, confirming unchanged response/reference accounting.

Neither fitted equation beats the no-further-movement reference. We therefore cannot portray a gain over a weak comparator as practical success. The primary failure does not identify its cause: estimation error, model limitations, selected events, feed measurement or an absent effect are possible, unseparated explanations.

## Population and chronology

The ledger retains 133,918 later recorded origins: 127,058 eligible and 6,860 origin-ineligible. Every eligible origin receives all four forecasts; 45 subsequently lack an evaluable C300. Those missing outcomes are excluded equally from scoring, never from issuance. The two fitted rules share exactly 2 training-mean fallbacks for previously unseen calendar levels. No new attrition rule was used.

There are 817 calendar dates, including 701 with evaluable events. Training labels must have matured at each cutoff; prediction outcomes were attached only after forecasts were serialized and independently verified. Forecasts were saved at 2026-10-03T14:24:57.930302+00:00; evaluation began at 2026-10-03T14:25:51.575327+00:00. The run finished at 2026-10-03T14:26:43.143839+00:00.

The cache had already been inspected in earlier research. Chronological ordering protects each replayed forecast from future-label inputs; it does not retrospectively create an untouched confirmatory test. Zero processing latency remains an idealized assumption, and these are recorded midquotes rather than executable returns.

## All calendar-quarter diagnostics

Positive mean loss differences favor the baseline; negative values favor S. The first and last quarters are partial sample periods. All quarters are reported, none promoted as a replacement target. Differences across quarters are descriptive, not significance tests or identified regimes.

| Origin quarter | Scored events | Sequence-minus-baseline MSE |
|---|---:|---:|
| 2024Q2 | 7,116 | +0.001054349 |
| 2024Q3 | 17,227 | -0.000519376 |
| 2024Q4 | 17,148 | +0.001330066 |
| 2025Q1 | 11,280 | +0.000017588 |
| 2025Q2 | 14,363 | -0.002755896 |
| 2025Q3 | 13,904 | +0.000248029 |
| 2025Q4 | 10,883 | +0.001594661 |
| 2026Q1 | 18,729 | +0.000265746 |
| 2026Q2 | 11,081 | +0.002908617 |
| 2026Q3 | 5,282 | -0.000662739 |

For the primary contrast, the largest absolute daily difference accounts for 3.461387% of the sum of absolute daily differences. The largest squared centered daily contribution accounts for 19.193090% of their sum of squares. Both maxima occur on 2025-04-11. These are different concentration denominators, not percentages of explained variance. The full daily, fold, quarter and fixed0-40-lag descriptions remain saved. No influential date was removed or selected for refitting.

## What verification establishes

All43 focused tests passed. Independent code, importing no producer fitting/design helpers, reconstructed complete matured training memberships, all118 OLS fits through QR, 5,713 coefficients and all535,672 forecast rows. It checked exact projection from the frozen10a feature table, source hashes, eligibility, first-alert values, common fallback, event-weighted losses, missing labels, calendar aggregation, stable loss subtraction and saved-output reloads. The maximum equilibrated condition number was 76.745299035; every declared fit satisfied its frozen full-rank rule.

The verifier also checked the algebraic reparameterization: since E is already a benchmark column and S=A-E, replacing the S column with A while subtracting its coefficient from E's coefficient leaves predictions unchanged. No alternate adjacency model was fitted. Centering is a measurement/interpretation choice, not a novel OLS information set.

The separate result notebook `notebooks/repaired/5B_Sequence_Forecast.ipynb` executed all three code cells, checked saved hashes and displayed actual results without another scientific fit. Original Notebook5 and the repaired09b notebook were preserved. Nonfatal sandbox hardware-query/IPython-directory warnings did not affect successful execution. Verification is an internal independent implementation, not external referee certification or proof that every economic assumption is true.

## Literature and contribution boundary

The targeted free-access check obtained full methods for Hashimoto et al.'s FX run study and an additional Ito-Hashimoto forecasting predecessor. The latter already studies five-minute returns conditional on earlier flow/returns and hour effects. Consequently, neither FX history dependence nor this horizon/control idea is new. Exact feed/alert/feature differences remain specifications, not proof of an important new relationship. See STEP10B_LITERATURE_ADDENDUM.md for primary links, inspected versions and remaining access limits.

Even a favorable result would not have established general information loss: a function of ordinary states can improve a linear model merely by adding omitted curvature. Selected-event conditioning, quote repricing, incomplete size-only updates and stale carried quotes also remain economic measurement problems. Neither this negative result nor existing positive average displacement resolves information versus inventory or permanent impact.

**The contribution gap is not patched.** This task completes and closes the declared sequence test. The defensible deliverable remains a bounded working-paper account of finite-horizon adjustment and unsuccessful declared forecasts, with all negative findings retained. Do not launch another candidate search as a routine repair. A stronger paper requires a separately defensible source of economic identification/measurement or an independently validated relationship, neither supplied by renaming S or rewriting the contribution paragraph.

Full outputs: data/signals/step10b_sequence_forecast_20261003. Central record: reports/pipeline_master_statistics.json. Protocol and prefit hashes: STEP10B_FORECAST_PROTOCOL.md and STEP10B_FROZEN_DESIGN.json. The living methodology guide advances to v1.27; final preservation/document evidence will be recorded in step10b_final_integrity.json.

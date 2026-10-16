> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# What the completed evidence supports

Review date: 3 October 2026. Existing results through stage 09b; no new empirical model, test, detector run or sample selection. This review ignores `manuscript.tex`. Its numerical source map is `STEP09C_EVIDENCE_AUDIT.json`; the independent economic critique is `STEP09C_INDEPENDENT_REVIEW.md`.

## Decision

The project now supports a reproducible empirical account of **finite-horizon quote-price adjustment and the performance of one prespecified first-alert forecast**. Each event's initial displacement is observed by its detection time. Its sample mean is positive, and the average total displacement remains positive at five minutes. The declared pressure-variation forecast does not improve realized prediction of the movement that follows detection.

This is more than an untested descriptive narrative: the predictive extension was implemented and evaluated, and its negative result belongs in the evidence. It is not an established discovery of informational liquidity interactions, two structural regimes or a novel successful persistence score. The strongest honest contribution candidate is a carefully delimited empirical benchmark in this recorded EUR/USD event population. Its substantive novelty and suitability for JFE remain unestablished.

**The next action is to write within these limits, not keep adding models until one produces a favorable result.** If the paper must deliver the original structural/private-information claim, the present results are insufficient. That would require a separately justified identification and validation design, not another routine pipeline repair. No such expansion is authorized or implemented by this review.

## The original research questions and their actual answers

| Original question | What is observed and estimated | Supported answer | Claim that remains unsupported |
|---|---|---|---|
| RQ1: Does the impact persist permanently or dissipate? | Initial movement J, subsequent C(h), and total I(h)=J+C(h), using recorded midquotes and the event's detected direction; primary h=300 seconds, secondary h=60. | Positive average total displacement remains at 300 seconds in the selected sample. Individual endpoints differ substantially. | Permanent impact, an eventual long-run limit, monotonic recovery, or private information versus dealer inventory. |
| RQ2: Which ex-ante states determine the permanent/transient regime? | Saved conditional state regressions, plus one first-alert forecast of C300 using strictly preceding pressure, variation, spread, displayed size and calendar variables. | The earlier associations remain conditional and uncertain. The fixed interaction forecast gives no realized MSE improvement over its additive benchmark or simple references. | A structural regime switch, advance warning of event arrival, a calibrated probability of informational trading, or universal absence of predictability. |
| RQ3: Can liquidity stress be identified invariantly to intraday patterns, clustering and quote oscillation? | Past-calibrated seasonal features, explicit timestamp/duration/support treatment, observable episodes, and declared sampling/freshness/direction comparisons. | These define and audit a reproducible quote-derived event measure within stated observation conventions. | Complete vendor coverage, invariance to every observation convention, latent efficient-price identification, or an exogenous liquidity-supply shock. |

The word *shock* needs an operational qualification throughout: these are threshold-detected quote disturbances. Rarity is not causal identification. The full-sample spread accounting supplies a concrete check: of 127,013 primary events, spreads initially widen at 48,239, remain unchanged at 33,169, and narrow at 45,605. Lack of widening does not prove absence of stress; it shows that the detector and spread-widening definitions are different.

## A short quantitative account

The following primary means use exactly the same 127,013 recorded events. All quantities are direction-aligned pips; J is known by the first alert.

| Component | Saved sample mean | Saved automatic nominal 95% interval |
|---|---:|---|
| J: movement during detection | +0.380258 | [+0.341466, +0.419049] |
| C300: movement after detection | +0.014752 | [-0.004937, +0.034442] |
| I300: total movement | +0.395010 | [+0.347638, +0.442381] |

Source: stage 07n `intervals.parquet`, `cohort=horizon_joint`, `horizon_sec=300`, `method=automatic`, with one row for each component. The longer-lag intervals remain available and must not be suppressed. These are provisional pointwise calculations under dependence/stability/normal-approximation assumptions, not validated coverage. The sample means themselves are descriptive facts about the saved cohort. An interval containing zero is not an equivalence result.

At five minutes, 58,637 observations end farther in the direction of their initial nonzero movement and 47,909 end beyond the pre-onset price in the opposite direction. The denominator for both counts is all 127,013 joint observations. This heterogeneity cannot be collapsed into two identified economic mechanisms. J is negative for 9,267 and zero for 5,442 observations, so the sign of C alone is not a universal continuation/reversal label.

The original conditional-association baseline gives a primary joint interaction nominal p-value of 0.387286; its declared curvature counterpart gives 0.989369. The secondary 60-second baseline value is 0.029691, changing to 0.082375 with curvature. These are the saved, provisional calculations, not newly run tests. They neither establish zero state effects nor justify promoting the shorter horizon. The full coefficients, branch derivatives and both covariance choices remain part of the record.

For the chronological forecast, all four methods use the same 127,013 eventual evaluable outcomes. The issuance ledger also retains 45 eligible origins whose future outcome cannot be evaluated.

| Fixed C300 forecast | Saved MSE, pips squared |
|---|---:|
| No further movement | 10.5249179415 |
| Historical matured-label mean | 10.5250685343 |
| Additive preceding-state equation | 10.5472785768 |
| Same equation plus variation x pressure magnitude | 10.5487942172 |

The interaction is 0.01437% worse than the additive rule and 0.22685% worse than zero continuation in realized MSE. No forecast p-value, equivalence test or population superiority claim was computed. The inspected cache is not an untouched confirmatory sample. Results across quarters vary; there is no selection of favorable dates or new model. The prior spread-history extension likewise remains stopped as a proposed headline price-persistence explanation; its recorded spread associations are supporting evidence, not a substitute mechanism.

## Why these findings do not contradict each other

1. **A retained initial move does not require predictable continuation.** If J=1 and C=0, I=1. The initial move remains even though the best forecast of further movement could be zero. The actual positive mean I and unsuccessful state forecast therefore answer different questions. Do not describe this example as a fitted model of the market.
2. **Finite-horizon retention does not establish permanence.** Two price processes can be identical through five minutes and diverge afterward: one retains the move and the other later reverses. These outcomes cannot be distinguished by a 300-second endpoint.
3. **Endpoint signs do not identify the mechanism.** Public news, information, inventory, changes in risk compensation and quote measurement can produce overlapping recorded paths. Bid/ask and displayed sizes alone do not supply the excluded source of variation needed to choose a causal explanation.
4. **A pooled negative result does not rule out every conditional pattern.** Opposite relationships may cancel, and an estimated model can forecast poorly because of estimation error or misspecification. These possibilities limit the claim; they are not evidence that an untested alternative will succeed.
5. **Changing a measurement and changing a sample are distinct.** The saved 5/15/30-second variation comparison keeps observations fixed. Freshness and direction restrictions change which observations remain. Their coefficient differences are not automatically a test of measurement invariance, and future-endpoint freshness screens are not prior predictors.

## What changed from the original pipeline

The substantive interest in preceding conditions and persistence is retained. The operational methodology changed materially where the original interpretation could not be sustained; this should be acknowledged explicitly.

| Component | Original ambition or implementation | Current defensible role |
|---|---|---|
| Imbalance | Often described as order flow | Recorded best-quote price/size imbalance; not signed executed trading flow. |
| Mahalanobis q99 | A severe liquidity-shock label | Historical tail event in a specified quote-feature vector; neither an informational label nor a 1% false-positive probability. |
| Decision clock | Mixed onset, episode-end and peak-based quantities | First completed onset second; preceding predictors end strictly before onset. Episode-end/peak quantities cannot enter this forecast. |
| Persistence | Future-path/AUC score, half-life and informative/transient interpretation | Joint continuous J/C/I and endpoint descriptions over fixed horizons, with explicit missingness. |
| Notebook 5 score | A normal-CDF transform of realized future-path AUC | Preserved historical description. A training-only scale does not make the event's own future AUC observable at detection. |
| Earlier repaired regressions | Conditional pressure/alignment-by-variation relationships | Detection-time conditional associations; some regressors use the detected direction. They are not strictly pre-onset prediction results. |
| Repaired Notebook 5 forecast | Pending prediction objective | Two fixed preceding-state equations plus zero/mean references, matured-label chronology and common-event loss accounting; no realized interaction gain. |
| Paper claim | Permanent information discovery versus temporary inventory, dual regimes | Finite-horizon observed adjustment and a bounded forecast assessment; the original mechanisms and permanent/transient regimes remain unidentified. |

The old notebook is preserved and guarded. Its predictive sections are not described as merely descriptive: the problem is their incompatible information clocks and the unsupported interpretation of future-path labels. The repaired forecast is a new, explicitly timed comparison, not numerical reproduction of the legacy score or the earlier detection-alignment regression.

## Literature and contribution boundary

The broad connection between flow, volatility and FX reversal is already present in the literature. Krohn, Mueller and Whelan study fixing-related reversals using transaction imbalance and prior-day FX variation, with institutional timing evidence. The current all-day quote-event sample and short horizon are different, but those differences alone do not establish economic novelty. This comparison is not a replication or refutation of their finding. [Published paper, Section VI.C](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13306).

Best-quote imbalance already has a price-impact literature, and EBS work studies exchange-rate/order-flow relations and intraday variation using transactions. A quote-derived proxy should not inherit the interpretation or validation of those measurements. [Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402); [Berger and coauthors, September 2006 revision](https://www.federalreserve.gov/pubs/ifdp/2005/830/revision/ifdp830.htm).

The review is focused, not an exhaustive priority search. The existing evidence supplies neither a successful new forecasting result nor a demonstrated rejection of a well-identified economic mechanism. A carefully reported negative benchmark can still be useful research, but this review cannot certify that its scale or originality meets JFE's editorial threshold. More verification improves credibility; it does not manufacture novelty.

## Concrete paper deliverable and stopping decision

A defensible working title is **Five-Minute Price Adjustment and Forecastability after Quote-Derived Stress Events in EUR/USD**. This is a recommendation for a working-paper framing, not a claim of journal readiness.

The results section can be organized around six existing evidence blocks, without new estimations:

1. **Data and event construction:** source scope, coverage, exclusions, exact detector clock and economic spread diagnostics. Do not pool development-only detector checks with the later main sample.
2. **Joint price response:** primary J/C/I means and endpoint accounting; secondary 60-second and matched-cohort results remain subordinate. Preserve the distinction between direction of recorded imbalance and direction of J.
3. **Conditional state associations:** the fixed 07h baseline, declared curvature comparison and both provisional covariance choices. Explain detection-time alignment rather than labeling these equations ex-ante forecasts.
4. **Declared sensitivities:** frequency, freshness and cancellation results with their actual sample changes and inferential limits. Do not claim every sensitivity is identical or choose a favorable screen.
5. **Chronological prediction:** all four frozen forecasts, origin-to-outcome coverage, primary paired loss and calendar diagnostics. The lack of realized improvement is a principal result, not an appendix omission.
6. **Interpretation and limits:** explain the failure of the informational/dual-regime claim, the stopped spread-history extension, dependence, inspected-data history and external validity. The full spread-history result can sit in an appendix.

Suggested core contribution sentence:

> In a recorded EUR/USD quote-event sample, we distinguish displacement already observed at detection from subsequent adjustment and test whether preceding quote pressure and price variation improve a fixed first-alert forecast. The sample retains a positive average displacement at five minutes, but the declared interaction does not improve realized forecast accuracy over additive states or simple references. These results delimit what this quote-derived event measure and information set establish; they do not identify informational trading, inventory adjustment or permanent impact.

**Readiness verdict:** the executed stages and saved numerical outputs can support this bounded working-paper draft. The stronger original economic identification, inferential coverage and JFE-level novelty remain unresolved. A clean empirical computation is necessary, but it is not sufficient for those claims. This completes the evidence-reconciliation task; no further routine model search is proposed.

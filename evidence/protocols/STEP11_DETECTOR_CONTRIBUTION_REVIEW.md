> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Does the joint detector add economic value?

4 October 2026. Executed exploratory comparison on the previously inspected EUR/USD quote cache. Run: `step11_detector_contribution_20261004`.

**Finding: these tests do not establish a robust incremental economic advantage for the full three-feature Mahalanobis detector.** Its primary adjusted-spread differences are small positive estimates with nominal intervals spanning zero. Some additional detections have wider subsequent spreads, but the adjusted disagreement-group differences are imprecise. Results vary across calendar periods and quote-age restrictions. This is insufficient evidence for making the economic superiority of the OFI/activity/duration combination the paper's central contribution.

This conclusion concerns the declared five-minute quoted-spread measurement test. It does not prove equivalence of detectors, absence of every possible use, or absence of price-response relationships. Independent validation and an operationally justified materiality threshold have not been supplied.

## What was executed

The [fixed protocol](STEP11_DETECTOR_CONTRIBUTION_PROTOCOL.md) implements the research design (full research archive: `paper/DETECTOR_CONTRIBUTION_TEST_PLAN_2026_10_04.md`; not bundled). The full detector and eight benchmarks were reconstructed with historical calibration, without optimizing a new duration model or choosing thresholds from economic outcomes.

- Replayed all 60 calibration periods and reproduced the original full detector's scores within numerical tolerance, its candidate flags exactly, and its confirmed and uncertain onset identities exactly.
- Evaluated 20,189,813 common eligible active seconds from 6 May 2024 through 31 July 2026, over 817 calendar days. The preceding deployment period initializes the episode histories.
- Measured outcomes and produced past-trained reference forecasts at 20,166,282 common evaluable origins. A reference model was fitted at each of 59 evaluation cutoffs, using only completed historical outcomes and a deterministic approximately 5% sample of calibration origins.
- Compared mean quoted spread over the next 300 seconds, with an ordinary-condition adjustment for current spread, spread history, preceding variation, activity, quote age, and calendar conditions. The model specification was fixed before these economic comparisons.
- Preserved every issued alert and its eligibility reasons. The full rule issued 133,905 confirmed alerts, with 133,563 evaluable in the primary comparison; 297 lacked reference features and 45 lacked the future outcome. Thirteen additional uncertain onset fragments are retained outside the primary alert comparison.

The quoted-spread target does not require a nonzero OFI direction. This population is therefore broader than the earlier 127,013-event directional price-forecast cohort. It must not be described as the same sample.

## Primary result

The table reports **full detector minus benchmark**, in pips of average future quoted spread after subtracting the same past-trained reference forecast. A positive difference indicates higher adjusted subsequent spreads at the full detector's alerts. It is an observational selection contrast, not a causal effect or a trading return.

| Benchmark | Full minus benchmark | Nominal simultaneous 95% interval |
|---|---:|---:|
| OFI and activity; duration omitted | +0.001827 | [−0.000861, +0.004515] |
| Diagonal distance; correlations ignored | +0.003524 | [−0.000529, +0.007577] |
| Any individual standardized feature unusually large | +0.001261 | [−0.001907, +0.004429] |

These are the fixed 56-calendar-day Bartlett HAC intervals, with Bonferroni adjustment across the three primary comparisons. All three intervals also span zero under the fixed 28- and 112-day sensitivity choices. They are conditional, nominal intervals: their coverage for repeated estimation of the entire detector/reference-model pipeline is not validated. Nonrejection does not establish equivalence or economic negligibility.

!Primary detector comparison (full research archive: `output/figures/detector_contribution_primary.png`; not bundled)

The full detector's raw mean future spread is 0.903578 pips, and its adjusted mean is 0.017773 pips. The two-feature detector's corresponding means are 0.885363 and 0.015946. The large raw spread level at detected events is therefore not itself evidence of an incremental contribution.

## All benchmarks and monitoring burden

| Detector | Confirmed alerts issued | Evaluable alerts | Mean future spread, pips | Mean adjusted future spread, pips |
|---|---:|---:|---:|---:|
| Full Mahalanobis | 133,905 | 133,563 | 0.903578 | 0.017773 |
| OFI + activity | 133,448 | 133,128 | 0.885363 | 0.015946 |
| Diagonal distance | 135,804 | 135,475 | 0.867753 | 0.014249 |
| Any unusual feature | 134,708 | 134,389 | 0.876353 | 0.016512 |
| OFI only | 156,008 | 155,700 | 0.751195 | 0.014259 |
| Activity only | 121,075 | 120,742 | 0.867189 | 0.009723 |
| Duration only | 193,340 | 193,032 | 0.563254 | 0.004071 |
| Current spread | 73,116 | 71,563 | 2.964124 | 0.305606 |
| Preceding variation | 2,362 | 2,356 | 0.986545 | 0.169107 |

Historical q99 cutoffs do not produce equal realized onset counts. Persistent exceedances can generate many candidate seconds but few new alerts, particularly for preceding variation. The table is not an equal-alert-budget utility comparison.

The full detector does not dominate obvious economic monitors on this target. However, the especially large spread-monitor residual also exposes substantial conditional underprediction in extreme spread states. Do not treat the reference model as a correctly specified conditional expectation, or convert these residual means into an execution-policy ranking. On all common evaluable origins the reference model's MSE is 0.041761 pips squared, compared with 0.049565 for using current spread unchanged; its mean residual is +0.000240 pips. Good aggregate calibration does not establish adequate tail calibration.

The fixed current-spread-decile/hour stratification check gives primary differences of +0.002012, +0.006228, and +0.001612 pips, respectively. Its three 56-day nominal family intervals also span zero. No alternative adjustment model was selected to improve the result.

## Does it find economically different observations that simpler rules miss?

There is descriptive evidence worth preserving, but it does not yet establish the proposed contribution.

On common candidate seconds, the full and two-feature detectors have 93.495% intersection/union overlap. The full rule adds 7,135 seconds and loses 7,038 relative to the two-feature rule. Full versus diagonal overlap is 73.750%; full versus the marginal alarm is 84.609%. Covariance weighting changes selection more substantially than adding duration.

At exact onset timestamps, the full detector has 6,892 alerts absent from the two-feature rule, while the two-feature rule has 6,435 absent from the full rule. Among those evaluable for the spread comparison:

| Exact-onset disagreement group | Evaluable alerts | Mean future spread | Mean adjusted future spread |
|---|---:|---:|---:|
| Full only | 6,863 | 1.121767 pips | 0.048662 pips |
| Two-feature only | 6,428 | 0.759299 pips | 0.012911 pips |

The adjusted difference is +0.035751 pips. A supplementary 56-day nominal interval is [−0.023791, +0.095293], using a six-comparison Bonferroni family across the three declared benchmarks and candidate/onset groupings. All six adjusted disagreement-group intervals span zero. This supplementary uncertainty calculation was completed after viewing the primary results, on the already-declared groups and outcome, and must remain labeled exploratory. It is not a replacement primary test.

Some apparent disagreements are timing changes. Matching onsets one-to-one within five seconds and the same support interval raises full/two-feature matches from 127,013 exact matches to 128,632, leaving 5,273 full alerts and 4,816 two-feature alerts unmatched. The economic group table above uses exact timestamps, not these unmatched subsets. Neither matching nor proximity establishes a common underlying economic cause.

## Stability and quote age

The quarterly adjusted differences change sign. Full minus two-feature is negative in 2024Q2, 2025Q2, and 2026Q1; full minus diagonal is negative in 2024Q2, 2025Q2, and 2025Q3. The first and last quarters are partial. Calendar influence is concentrated: 24 December 2024 accounts for approximately 15.1% of the summed squared daily influence for the duration contrast and 23.0% for the diagonal contrast. These are concentration diagnostics, not variance shares or causal explanations. No dates were removed.

!Quarterly detector comparison (full research archive: `output/figures/detector_contribution_quarters.png`; not bundled)

The inherited-style fixed quote-age screens also change the result. Full minus two-feature adjusted differences are approximately −0.001403 pips at 15 seconds, −0.000516 at 30 seconds, and +0.001161 at 60 seconds. Full minus diagonal and full minus marginal are likewise negative at the 15- and 30-second screens, and positive at 60 seconds. These screens restrict the population using recorded quote ages over the declared history/current/future marks; they are not live alert rules or evidence that one restriction is the correct sample. They weaken a robustness claim for the positive unrestricted differences.

The full detector's restricted samples contain 23,896, 60,756, and 114,702 evaluable alerts at the respective caps, versus 133,563 unrestricted alerts. The strictest screens therefore change the sample substantially.

The saved summaries also retain the declared 60-second spread and price-movement diagnostics. None is promoted to a new primary hypothesis after the five-minute spread comparison.

## Verification and reproducibility

All 37 targeted and directly related detector/episode tests passed. They cover strict timing, future-data invariance of features, support boundaries, causal episode construction, chunk invariance, separate thresholds, and ratio-covariance arithmetic.

The separate verifier passed 1,202,000 sampled source quote-mark checks across every evaluation cycle, checked saved-model predictions on fixed samples, verified label maturity, reconciled all 20,189,813 deployment rows, and reproduced the complete daily adjusted ratios and all three HAC calculations using a separate matrix-weight calculation. It verified 180 parent-cycle input fingerprints and the additional historical inputs. This is computational verification; it is not independent market validation or proof of statistical coverage.

Sources and principal artifacts:

- Execution script (full research archive: `run_detector_contribution.py`; not bundled), measurement and statistics module (full research archive: `src/detector_contribution.py`; not bundled), separate verifier (full research archive: `verify_detector_contribution.py`; not bundled).
- [Run manifest](../data/signals/step11_detector_contribution_20261004/manifest.json), [verification](../data/signals/step11_detector_contribution_20261004/verification.json), [runtime versions](../data/signals/step11_detector_contribution_20261004/environment.json).
- [All economic means](../data/signals/step11_detector_contribution_20261004/economic_summary.csv), [primary and benchmark contrasts](../data/signals/step11_detector_contribution_20261004/contrasts.csv), [coverage](../data/signals/step11_detector_contribution_20261004/coverage.csv).
- [Disagreement groups](../data/signals/step11_detector_contribution_20261004/group_comparison.csv), [supplementary disagreement uncertainty](../data/signals/step11_detector_contribution_20261004/disagreement_precision.csv), [timing comparison](../data/signals/step11_detector_contribution_20261004/timing_comparison.csv).
- [Quarterly summaries](../data/signals/step11_detector_contribution_20261004/quarterly_summary.csv), [quote-age sensitivity](../data/signals/step11_detector_contribution_20261004/age_sensitivity.csv), all-benchmark figure (full research archive: `output/figures/detector_contribution_contrasts.png`; not bundled).

The run directory contains each cycle's common-origin ledger, scores, flags, training features, completed training labels, reference model, predictions, outcomes, and source snapshots. No new external data were acquired. No empirical pipeline result was overwritten.

Reproduce in the recorded research environment with a new run ID; use the same run ID to resume an interrupted phase:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/miniconda3/envs/fxresearch/bin/python run_detector_contribution.py build --run-id NEW_RUN_ID
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/miniconda3/envs/fxresearch/bin/python run_detector_contribution.py analyze --run-id NEW_RUN_ID
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/miniconda3/envs/fxresearch/bin/python verify_detector_contribution.py --run-id NEW_RUN_ID
```

The plot and supplementary precision scripts currently target the recorded run ID explicitly. Frozen model settings, timestamp sampling, and environment versions are part of the reproduction record.

## Consequence for the paper

Describe the joint detector as a measurement choice and report this benchmark assessment honestly. The available evidence does not justify a central claim that combining OFI, activity, and ACD-normalized duration through Mahalanobis distance delivers a robust economic improvement.

A defensible sentence is:

> In an exploratory chronological comparison of EUR/USD quote alerts, the full joint detector exhibits small positive adjusted future-spread differences relative to simpler multivariate rules, but these differences are not resolved by the declared uncertainty analysis and vary across calendar periods and quote-age restrictions.

The contribution now has to rest on the importance of the economic question and the credibility of the combined price-adjustment and benchmarking evidence. This run does not certify a major field contribution or novelty. Further confirmation would require genuinely unexamined data and a justified economic use case; searching another outcome until a favorable result appears would not supply either.

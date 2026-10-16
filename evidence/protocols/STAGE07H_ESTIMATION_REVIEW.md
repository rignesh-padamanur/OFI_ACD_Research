> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 07h: fixed continuation estimates and provisional uncertainty

This Notebook 5 step fits the previously declared continuation equation and its one curvature diagnostic at C(300), then repeats them at secondary C(60). It preserves the verified samples, frozen scales, detector and covariates. The estimates describe conditional associations among selected recorded-quote events in an inspected cache.

Every coefficient and the two saved within-branch pressure derivatives are reported under the automatic calendar-score bandwidth and the fixed longer lag. The main hypothesis remains baseline C(300), automatic lag, with both interaction coefficients jointly zero. Curvature, C(60) and the longer lag are declared comparisons, not a search for favorable significance.

## Saved results

The primary baseline C(300) joint interaction calculation gives nominal p=0.3873; the declared curvature calculation gives p=0.9894. Both five-minute branch-slope intervals include zero under both equations and both declared lags. These are provisional conditional calculations. Intervals including zero and a nonrejection do not establish zero effects, equivalence or permanence.

At secondary C(60), the nominal baseline interaction p=0.02969 becomes p=0.08237 in the fixed curvature diagnostic. The nominal assessment depends on the declared functional form. The secondary finding does not replace the primary horizon, and neither equation is selected by its probability.

| Sample / equation | Events | Calendar days | Automatic lag | Longer lag |
|---|---:|---:|---:|---:|
| c_300/baseline | 127,013 | 817 | 7 | 14 |
| c_300/curvature | 127,013 | 817 | 8 | 16 |
| c_60/baseline | 127,052 | 817 | 8 | 16 |
| c_60/curvature | 127,052 | 817 | 10 | 20 |

| Sample / equation | Branch | Covariance | Slope | Conditional 95% interval |
|---|---|---|---:|---|
| c_300/baseline | reinforcing | automatic | -0.00766308 | [-0.134327, 0.119001] |
| c_300/baseline | opposing | automatic | 0.0686292 | [-0.093165, 0.230423] |
| c_300/baseline | reinforcing | longer | -0.00766308 | [-0.131814, 0.116487] |
| c_300/baseline | opposing | longer | 0.0686292 | [-0.0978131, 0.235072] |
| c_300/curvature | reinforcing | automatic | 0.0157469 | [-0.0940415, 0.125535] |
| c_300/curvature | opposing | automatic | 0.0804485 | [-0.114337, 0.275234] |
| c_300/curvature | reinforcing | longer | 0.0157469 | [-0.0906081, 0.122102] |
| c_300/curvature | opposing | longer | 0.0804485 | [-0.121139, 0.282036] |
| c_60/baseline | reinforcing | automatic | 0.0678696 | [0.0211596, 0.11458] |
| c_60/baseline | opposing | automatic | 0.00367795 | [-0.052187, 0.0595429] |
| c_60/baseline | reinforcing | longer | 0.0678696 | [0.0207368, 0.115002] |
| c_60/baseline | opposing | longer | 0.00367795 | [-0.050895, 0.0582509] |
| c_60/curvature | reinforcing | automatic | 0.0433598 | [-0.00788908, 0.0946086] |
| c_60/curvature | opposing | automatic | -0.0321092 | [-0.0827779, 0.0185595] |
| c_60/curvature | reinforcing | longer | 0.0433598 | [-0.0120919, 0.0988115] |
| c_60/curvature | opposing | longer | -0.0321092 | [-0.0846988, 0.0204804] |

| Sample / equation | Covariance | Joint interaction Wald | Conditional chi-square(2) p |
|---|---|---:|---:|
| c_300/baseline | automatic | 1.89719 | 0.387286 |
| c_300/baseline | longer | 2.1327 | 0.344263 |
| c_300/curvature | automatic | 0.0213764 | 0.989369 |
| c_300/curvature | longer | 0.0202744 | 0.989914 |
| c_60/baseline | automatic | 7.03384 | 0.0296908 |
| c_60/baseline | longer | 6.37724 | 0.0412288 |
| c_60/curvature | automatic | 4.99295 | 0.0823748 |
| c_60/curvature | longer | 4.19371 | 0.122842 |

Every coefficient, covariance entry and daily/event record remains in the named run; the master statistics retain all coefficient, slope and joint-test tables. The tables above do not select a preferred specification or infer a structural regime.

## Meaning of the comparisons and uncertainty

Pressure slopes move along the feasible A=s*M branch at fixed variation, direction and controls. Each branch average uses its own observed state distribution, so their difference is not a controlled comparison at common states. Zero-pressure events remain in the regression but receive no arbitrary branch derivative. Slopes are local fitted derivatives, not predictions of a finite one-unit intervention.

Normal 95% intervals are pointwise conditional asymptotic calculations. The two-term Wald probability uses chi-square(2); no finite-sample coverage or simultaneous family coverage is established. Both full covariance matrices and the component bandwidth pilots are saved. A longer lag need not increase uncertainty. No model, coordinate or observation is removed because its result is inconvenient.

## Realized score diagnostics

Complete UTC daily scores preserve empty dates. Ordered score-pair lag products and the five substantive coefficient plus two branch influence sequences are saved over the fixed diagnostic span. Daily energy concentration and calendar-quarter sums and shares describe the realized fit. They are not independence, stationarity or adequate bandwidth certificates; zero energy is explicitly undefined. No correlation threshold is used to tune the estimator.

The primary baseline C(300) fit shows the following concentration in the seven prespecified daily influence targets. Energy is the squared daily contribution, normalized by its sum across all dates.

| Primary C300 influence target | Largest-energy date (UTC) | Share of squared daily influence | Largest absolute lag product beyond longer lag |
|---|---|---:|---:|
| coefficient:variation_standardized | 2025-04-02 | 58.51% | 0.1099 |
| coefficient:pressure_magnitude | 2024-11-06 | 15.16% | 0.1116 |
| coefficient:aligned_prior_pressure | 2026-03-23 | 22.15% | 0.2434 |
| coefficient:variation_x_magnitude | 2025-04-02 | 38.42% | 0.1680 |
| coefficient:variation_x_alignment | 2026-03-23 | 55.41% | 0.2441 |
| branch:reinforcing | 2026-03-23 | 19.81% | 0.2479 |
| branch:opposing | 2024-11-06 | 26.91% | 0.1650 |

The largest inspected ordered score-pair product beyond both primary bandwidths (7 and 14 days) has absolute normalized value 0.7031 at lag 16, for `hour_09` against `hour_03`. It is a maximum over many inspected products, involving calendar control scores. It is not a formal serial-dependence test or proof that a particular bandwidth is inadequate. Two isolated, otherwise unrelated high-score dates can produce a large lagged product.

A date's squared-influence share is not its share of the coefficient estimate or of the full HAC variance, which also contains weighted cross-date products. The concentration and remaining target lag products are material finite-sample concerns; they do not by themselves identify erroneous quotes or justify deleting observations.

The next bounded audit traces these quantities to dates, individual event contributions and their original quote/state/response records. It preserves all existing results and does not expand a lag until significance or apparent stability is achieved. A verified measurement error would justify repairing the input and recomputing affected descendants; valid extreme observations would remain part of the declared sample.

Dependence beyond the inspected lag span, cancellation within quarters, delayed cross-score dependence and overlapping rolling calibration histories remain relevant counterexamples. Concentration is not an effective number of independent days. Observed intervals and diagnostic agreement cannot establish causal information or inventory mechanisms, permanent impact or two structural regimes.

## Verification and completion boundary

All 764 tests passed. Independent verification reconstructs all four fixed models, response/design identities, estimates, residuals, scores, covariance, intervals, joint tests and diagnostic arrays. This establishes consistency of the implemented calculation, not empirical coverage or paper readiness. All twenty-four prior pointers and earlier master stages remain preserved.

Next: Trace the source provenance of the dominant dates, their event influence contributions and the largest lagged score products. Preserve all observations, four equations and declared bandwidths. Repair measured inputs only if this trace demonstrates an error, then rerun affected downstream calculations. The fixed measurement sensitivities and joint J/C/I interpretation follow this bounded audit; no outlier deletion or model/bandwidth search.

Verified run: `step07h_continuation_estimation_20261001`.

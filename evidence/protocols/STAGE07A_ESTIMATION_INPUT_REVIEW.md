> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 07a: verified development regression inputs

Notebook 5 preparation using the existing Notebook 2 states and Notebook 3/4 price responses. The detector, onsets and outcomes are unchanged. No response model, coefficient, p-value or confidence interval was fitted.

## Economic purpose and earlier scaling

RQ1 keeps the saved J/C/I displacement accounting. RQ2 now has explicit numerical inputs for pressure magnitude, agreement with the detected direction and their relationships with prior variation. These are transformations of existing states, not new raw economic variables. Detection-time alignment is not labeled strictly pre-onset. The recorded-data and finite-horizon interpretation remains necessary.

The fixed reference contains 1,281,932 of 1,282,116 detector-eligible calibration bins from February 26-April 21, 2024. 184 lack complete finite reference states. Selection does not use q99 membership, distance or future returns. Zero pressure remains allowed (95 reference rows). RMS(Q) preserves the economically meaningful zero; variation uses its reference mean and population standard deviation.

| Scale | Value |
|---|---:|
| cQ, RMS prior imbalance | 134.541089272434 |
| cV, mean prior variation (bps) | 1.962402411841 |
| sV, population SD of prior variation (bps) | 1.541555169861 |

Finite positive affine scaling preserves the specified regression space because intercept, main effects and interactions remain included. Outcomes are not divided by variation or initial displacement.

## Sample and control checks

All 1,806 anchors and 5,418 parent outcome rows are retained. Finite positive prior spread and displayed size are required for the chosen controls; no imputation, size floor or outcome-based exclusion is used. Baseline and curvature numerical validity are separate.

| Sequential condition | Remaining | Newly excluded |
|---|---:|---:|
| pre_state_eligible | 1,783 | 23 |
| valid_spread | 1,783 | 0 |
| valid_size | 1,783 | 0 |
| baseline_finite | 1,783 | 0 |

| Horizon | C sample | J sample | I sample | Joint |
|---:|---:|---:|---:|---:|
| 0 | 1,783 | 1,783 | 1,783 | 1,783 |
| 60 | 1,782 | 1,783 | 1,782 | 1,782 |
| 300 | 1,782 | 1,783 | 1,782 | 1,782 |

The continuation mask does not require the initial-price mark. Component comparisons use the joint intersection. No future outcome or freshness flag enters the feature matrix. Existing sensitivity flags remain available but are not additional baseline exclusions.

## Numerical design, not response estimation

| Sample | Rows | Baseline columns / rank | Curvature columns / rank | Baseline max leverage |
|---|---:|---:|---:|---:|
| c_60 | 1,782 | 35 / 35 | 37 / 37 | 1.000000 |
| joint_60 | 1,782 | 35 / 35 | 37 / 37 | 1.000000 |
| c_300 | 1,782 | 35 / 35 | 37 / 37 | 1.000000 |
| joint_300 | 1,782 | 35 / 35 | 37 / 37 | 1.000000 |

Canonical calendar columns are preserved. The current 300-second sample has unsupported columns: hour_13, hour_14, weekday_5. They are explicitly unestimated, not estimated zero effects. Only all-zero calendar indicators are omitted from the computational column list. Missing reference categories or remaining rank deficiencies flag a design as unusable. No scientific regressor is silently removed.

Hour 15 has one observation in this development cohort; its dummy can produce unit leverage. Full column rank therefore does not establish reliable uncertainty or sufficient support. Leverage and the 12-date concentration are limitations to retain when extending the sample, not reasons for outcome-driven trimming.

The baseline and fixed curvature columns are saved, but the curvature comparison itself has not been fitted. Sample-specific schemas, singular values, rank tolerances, condition numbers and leverage summaries make the matrix reproducible. No inferential approval is granted.

## Legacy protection, verification and next work

The standalone src/run_corrected_regressions.py is now blocked immediately after its docstring, before imports and legacy database reads. Importing it is also blocked, so its old fitting functions cannot bypass the guard. The exact former source and all legacy results are preserved.

478 tests passed. Independent verification reconstructed the reference scaling, derived features, component-specific masks, design rows, calendar support and summary diagnostics from the parent artifacts. Source/input/output fingerprints were checked. Tests concern the declared implementation and specific failure cases; they do not prove the regression is economically correct.

Next: Freeze and implement the chronological calibration/carry/boundary contract on later cached data, then verify the extended detector, episode, state and response inputs. Specify and validate the calendar-dependence inference implementation before fitting the fixed response equation. Keep the declared baseline, one curvature diagnostic and existing fixed measurement sensitivities; no classifier, regime-threshold or duration-model search. Later cache remains previously inspected.

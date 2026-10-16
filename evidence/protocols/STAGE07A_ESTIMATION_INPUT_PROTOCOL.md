> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 07a: regression-input contract

Recorded before computing the new input statistics on 30 September 2026. Scope: Notebook 5 input preparation from the verified stage-06c development ledger, plus blocking the legacy standalone regression route. No response fitting, confidence intervals, threshold search or chronological extension. Ignore manuscript.tex. Preserve all existing onsets, raw states, outcomes and active pointers.

## Link to the research questions

RQ1 uses existing initial J, subsequent C and total I=J+C. RQ2 will compare C across preceding pressure and variation, conditional on the detected direction. This stage makes the exact comparison inputs and sample exclusions reproducible. RQ3 retains the already fixed measurement sensitivities; it does not add a new measurement search. Primary horizon is 300 seconds, secondary 60; h=0 remains accounting only.

## Earlier scaling reference

Use structural-only scored bins from the verified stage-05a detector audit labeled calibration, with bin starts in [2024-02-26 00:00 UTC, 2024-04-22 00:00 UTC) and completed-bin availability no later than the latter cutoff. These are detector-eligible calibration bins, not all calendar seconds and not only detected events. Retain the same rows having finite prior signed quote imbalance Q and finite nonnegative five-second-grid prior variation V. Zero Q and zero V are legitimate. No event label, score, response, spread/size control or development observation selects these reference rows.

Set cQ=sqrt(mean(Q squared)), cV=mean(V), sV=sqrt(mean((V-cV) squared)) with population denominator n. All constants come from the same reference rows. Preserve zero in signed pressure by not subtracting its mean. Reject missing/zero/nonfinite scales; no arbitrary fallback. Save the selected reference values/times, reference attrition and exact constants. Their role is units and numerical scaling, not a new economic estimator or causal normalization. Applying invertible fixed scaling does not expand the regression space. Never divide a response by these states. The reference is frozen for a later extension unless an explicitly revised protocol records otherwise.

## Design and timing

Baseline columns are intercept, detected direction d, v=(V-cV)/sV, M=abs(Q)/cQ, A=d*Q/cQ, v*M, v*A, prior spread in pips, log of prior positive displayed size, additive hour and weekday indicators. The sole curvature diagnostic appends M squared and v squared. No fitted coefficients or empirical response comparison in this stage.

Primitive Q,V,spread,size precede onset. M,v,log size and v*M use those prior states; A,v*A and d are known at the completed onset-bin detection time. Sample flags incorporating confirmed onset/direction are detection-time accounting, not strictly pre-onset predictors. Outcome eligibility and future freshness flags are never predictors. Original direction-zero rows remain in the table and are not imputed to positive direction.

Keep canonical hour indicators 01-23 (hour00 reference) and weekday indicators 1-6 (Monday0 reference). Preserve calendar fields and all canonical columns. Within each actual sample, omit only all-zero calendar dummy columns from the computational column list and mark their categories unsupported/not estimated. Never silently treat an absent category as an estimated zero effect. Record a missing reference level and remaining rank deficiency as unusable design flags, without deleting scientific terms, changing sample membership or applying a hidden generalized inverse. Do not infer unavailable raw data from absent selected-event categories.

## Sample rules and outputs

Retain every anchor and all parent outcome rows at 0/60/300. In addition to parent pre_state_eligible, require finite strictly positive prior spread and displayed size, consistent with the upstream quote cleaner, and finite constructed baseline numerical columns. Nonpositive/nonfinite spread and nonpositive/nonfinite size are separate reason flags; no zero filling, arbitrary size floor, winsorization or return-sign selection. Save both nonexclusive failure flags and a fixed sequential exclusion reason (parent state, spread, size, transformed finite). Transformation overflow is flagged, never silently repaired. Curvature finiteness is reported separately and never silently changes the baseline sample; an overflow prevents that diagnostic on the declared sample.

Intersect regressor eligibility independently with the parent C, J and I eligibility masks. A C sample must not require J. The joint mask additionally requires all three components. Keep all existing sensitivity flags as parent accounting. Sensitivity estimation is future work; these flags are not extra baseline exclusions.

Save feature rows, all horizon-specific sample membership rows, and design-only numerical rows for C and joint samples at 60/300. Each design has a saved column schema for baseline and curvature, full-rank/reference/finite flags, SVD rank tolerance, singular values and condition number. Save calendar support, state distributions and leverage diagnostics; leverage is descriptive and never a selection filter. Empty/rank-deficient designs remain auditable and are not inference-ready.

## Verification, legacy guard and limits

Meaningful tests target mirrored signs, zero/negative values, reference timing and sample isolation, outcome-independent features, component-specific masks, empty cohorts, unsupported categories, rank deficiency and tampered lineage. Independent code reconstructs scaling, derived rows, masks, encodings and design accounting from immutable parent inputs. Confirm J+C=I on joint rows without fitting any component. Snapshot source, parameters, inputs, outputs and test evidence; register only after checks pass. Preserve every prior active pointer and master research record.

Block import/execution of src/run_corrected_regressions.py before it reads the old database or can expose its obsolete estimation functions. Archive its exact previous source. A subprocess guard check must confirm the intended error and no legacy result changes. Do not remove its historical outputs.

Completion means verified development regression inputs and a closed legacy route. It does not establish model adequacy, a valid HAC bandwidth, full-panel chronology, causal identification or paper readiness. Next: specify and implement the chronological calibration/carry/boundary contract and the dependence-aware inference procedure before empirical estimation.

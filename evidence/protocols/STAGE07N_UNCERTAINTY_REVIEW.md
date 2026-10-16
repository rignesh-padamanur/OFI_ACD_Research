> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage07n: uncertainty around fixed recorded-price summaries

Run `step07n_joint_uncertainty_20261002`. 955 tests passed; the new mean outputs were independently reconstructed from inherited event records. Covariance and concentration records for all four saved regression fits were rechecked without refitting. Zero new models and zero new hypothesis tests.

This belongs to the Notebook3/4 response summaries and Notebook5 uncertainty/reporting layer. It preserves the detector, event samples, states and equations.

J is detection-second movement; C is subsequent movement; I=J+C is total displacement. All are pips aligned to recorded quote direction. Event-weighted means remain unchanged up to floating-point summation order.

All 817 UTC dates are retained. Each daily score is the component sum minus that day's event count times the overall event mean. One common lag of 20 calendar days is selected by the frozen twelve-coordinate rule; the declared longer comparison uses 40 days. Both are reported without selection by significance.

| Horizon | Component | Events | Mean pips | Automatic nominal95% interval | Longer nominal95% interval |
|---|---|---:|---:|---|---|
| 300s | J | 127,013 | +0.380258 | [+0.341466, +0.419049] | [+0.332622, +0.427893] |
| 300s | C | 127,013 | +0.014752 | [-0.004937, +0.034442] | [-0.004179, +0.033684] |
| 300s | I | 127,013 | +0.395010 | [+0.347638, +0.442381] | [+0.337562, +0.452457] |
| 60s | J | 127,052 | +0.380295 | [+0.341517, +0.419074] | [+0.332673, +0.427918] |
| 60s | C | 127,052 | -0.003153 | [-0.013754, +0.007448] | [-0.013329, +0.007023] |
| 60s | I | 127,052 | +0.377142 | [+0.339739, +0.414546] | [+0.330936, +0.423348] |

The common60/300 cohort is separately retained in the saved intervals and master ledger. Its identical J observations receive identical uncertainty at both horizons. The full covariance includes all twelve targets and respects I=J+C; singularity is expected and no inverse is taken. These are pointwise intervals, not a simultaneous confidence region. Overlap of intervals is not a test of a difference.

The finite saved-sample means are known. The intervals approximate uncertainty for the event-weighted mean of a hypothetical repeated selected-event process, subject to stability, moments, weak dependence and a normal approximation. Empty dates are zero selected-event scores, not imputed returns or proof of feed coverage.

| Primary mean target | Largest daily squared influence | Share of squared daily influences |
|---|---|---:|
| J | 2025-04-09 | 9.63% |
| C | 2025-04-02 | 24.56% |
| I | 2025-04-02 | 22.86% |

These squared-influence shares describe concentration; they are not shares of the full HAC variance or counts of independent dates. Saved cross-products through the fixed diagnostic range do not prove or disprove stability. Longer lags need not produce wider intervals. No observation is removed.

Existing state-regression uncertainty remains provisional. In the primary baseline the largest daily squared-influence shares remain 58.51% for the variation coefficient and 55.41% for variation times alignment. Source consistency does not establish a reliable normal approximation. All four original automatic/longer results and the existing observed-state branch comparisons remain unchanged. Their branch weights remain treated as fixed, not estimated population-distribution quantities.

A past-only rolling detector can transmit estimation variation into the resulting score process. HAC could in principle capture its serial covariance under appropriate conditions; those conditions and bandwidth adequacy are not established here. No full-pipeline coverage simulation, ideal-detector inference, fresh holdout or external-feed validation is claimed.

The uncertainty assessment is now implemented and verified within its stated scope. It does not establish permanent impact, causal shocks, information/inventory separation or two structural regimes. Do not turn a small or imprecise average C into an equivalence claim, or a positive average I into a statement that every event persists.

The first scalar bandwidth formula follows Newey and West (1994); coordinate-maximum aggregation is this project's declared extension, not a multivariate optimality or conservative-bound claim. References and counterexamples are in STAGE07N_UNCERTAINTY_PROTOCOL.md.

Next: Complete the fixed reporting and reproducibility closeout: map each RQ to its final table, reconcile original versus repaired methodology and notebook entry points, and record claim limits. Do not start another model or sensitivity search. Interval coverage, causality and JFE readiness remain unestablished.

The pipeline is not yet certified paper-ready.

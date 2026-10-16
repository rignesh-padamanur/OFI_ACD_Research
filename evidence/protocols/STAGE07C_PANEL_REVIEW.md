> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 07c: full cached structural-baseline detector and episode panel

This repair covers Notebook 2 detection and Notebook 3 episode/state inputs. The structural-only baseline now spans April 22, 2024-July 31, 2026. Price responses, the enlarged analysis ledger, regression matrices and inference have not been computed by this repair.

## Economic purpose and unchanged method

The research question still concerns finite-horizon price persistence conditional on preceding recorded imbalance and price variation. This stage supplies a longer, chronologically constructed event/state sample for that comparison. It adds no economic predictor or alternative detector.

The original three-feature Mahalanobis/linear-q99 detector uses 56 preceding UTC calibration days followed by 14 deployment days. There are 60 cycles: 2 inherited verified cycles and 58 later fitted windows of the same Weibull ACD specification. The final deployment contains 5 days. Three fixed numerical starts estimate one specified model; they are not a model-family search. Daily seasonal timing rules and the stage-07a regression scales remain unchanged. The final precision-rebuilt run reuses all 58 later parameter fits from the preserved earlier attempt; the precision repair performs zero model refits.

Rolling q99 identifies unusualness relative to recent recorded conditions, not a constant absolute liquidity-severity threshold or a probability of an exogenous liquidity shock. The cached later data were inspected in earlier work; this is chronological reanalysis, not an untouched holdout.

## Executed statistics

| Quantity | Value |
|---|---:|
| Source quote spells | 55,708,318 |
| Full deployment calendar seconds | 71,798,400 |
| Eligible deployment seconds | 20,494,799 |
| Candidate seconds | 213,325 |
| Candidate share of eligible seconds | 1.040874% |
| Episode fragments | 135,724 |
| Confirmed onsets | 135,711 |
| Uncertain onsets retained | 13 |
| State-eligible onsets | 135,414 |
| Preserved prior onset/state rows | 3,011 |
| Preserved previously terminated episodes | 3,011 |
| New onset fragments after the inherited prefix | 132,713 |

| Opportunity reason | Seconds |
|---|---:|
| unsupported | 20,811,409 |
| invalid_flow | 70 |
| seasonal_unavailable | 28,653 |
| quiet_no_duration | 30,463,461 |
| incomplete_duration | 8 |
| eligible | 20,494,799 |

These disjoint counts cover every calendar second in the deployment interval. Unknown or unsupported observations are not filled with neutral signals. Candidate frequency can differ from 1% in deployment because q99 was estimated on preceding calibration observations.

## Boundary handling and preservation

A running eligible wait keeps its actually issued conditional mean, origin seasonal scale, global quote identity and original parameter cutoff. Carry can survive several deployment windows or predate the current training window. New-origin forecasts use a separate history refilter under new parameters. A completion exactly at the cutoff is excluded from that historical training fit and included in the new deployment completion bin. The initial April 22 convention remains as previously audited; its parameters are not relabeled as known before the initial cutoff.

Episode state survives computational chunks. Only unknown observation intervals, physical support changes, the existing five-clear-second rule or the actual panel end can close/censor an episode. Global onset IDs and inherited score/onset values are preserved; future closure and membership may extend an episode that was unfinished at the old endpoint. Preceding states use 300 seconds of clock history and the nearest strictly preceding quote, even if it predates a read chunk.

## Numerical precision repair and preserved failed attempts

Verification first stopped at cycle 017 when the earlier independent reference accumulated a tiny negative squared-return sum and produced a missing square root. Direct nonnegative summation and scalar math.fsum checks supported a repair to that reference. The continued verification then stopped at cycle 043: the production rolling calculation retained a positive rounding remnant after larger returns had left the window. The failing scored row was neither a detector candidate nor an episode onset. The earlier attempts retain their failure records; a reference-only correction was insufficient to certify the full production panel.

The rebuilt run directly sums the 60 squared five-second returns in each preceding 300-second window. It requires all 60 terms to be finite and keeps the same quote marks, support, imbalance history and economic definition. The independent reference uses a separate direct accumulation method. Neither calculation adds a variance floor or relaxes the 1e-9 absolute/relative comparison tolerance. Only pre_rv_300s_5s_bps is regenerated in the later scored/onset artifacts; the first two inherited cycles stay exact and are independently checked against the reference.

17 historical reference/verification checks and 11 production state-precision checks passed in addition to the 573-test base suite. These are distinct saved test runs, not an assertion that the original suite was rerun after every numerical change. All 58 later model fits are reused; this correction performs zero model refits. Full reconstruction and the explicit reuse audit verify preserved detector features, scores, candidates, episode eligibility, other state fields and summary statistics. Earlier numerical files and source snapshots remain in the failed run; the final run has current, explicit source snapshots.

| Precision-rebuild audit | Value |
|---|---:|
| Later cycles recomputed | 58 |
| Scored rows with old values outside unchanged tolerance | 1 |
| Candidate rows with old values outside unchanged tolerance | 0 |
| Onset rows with old values outside unchanged tolerance | 0 |
| Maximum absolute scored variation change (bps) | 5.14468468325e-08 |
| Model refits in this precision rebuild | 0 |

The frozen protocol's shorthand “feature fingerprints” means the exact ordered eligible-second identities. These are integer identity hashes, not hashes of floating-point feature values. The independent verifier separately reconstructs the numerical features, covariance, q99 and deployment scores. This clarification does not change any calculation.

## Working-model diagnostics

Across the 58 later fitted cycles, q99 distance thresholds range from 5.093555963748 to 5.523993571262. The two inherited cycles retain their own previously verified parameters and warnings.

| New-cycle diagnostic | Count |
|---|---:|
| Cycles with at least one duration diagnostic flag | 58 |
| Cycles passing the declared duration screening tolerances | 0 |
| Cycles with a numerical constraint warning | 1 |
| Unsuccessful fixed optimizer starts | 0 |
| Cycles flagged: hazard_band_calibration | 58 |

Duration calibration or residual-dependence failures remain visible. The duration residual is a working detector feature, not an asserted calibrated probability law. Even a cycle passing these descriptive tolerances does not validate the law or establish economic identification. A failed numerical cycle would have stopped registration; diagnostic warnings do not trigger model searches or outcome-based exclusions. The optimizer does not certify a global optimum.

## Verification and remaining work

573 tests passed. Independent verification checked 60 cycles, 213,325 candidate rows and 135,724 onset rows. It reconstructs the full source/seasonal ledger, each new cycle's historical clipping, conditional-mean filtering and likelihood objective, raw-covariance/q99 scores, origin-vintage carry, strict quote marks and pre-states, candidate connectivity, closure and daily/monthly accounting. Registered input/output/source/test hashes and exact inherited fields are checked.

The already declared freshness/sampling sensitivities still need their longer-panel extension. Existing response artifacts censored at old computational boundaries cannot be appended and called continuous outcomes: rebuild J/C/I using actual quote/support restrictions across those boundaries. Then rebuild the ledger and matrices, assess support and validate dependence-aware uncertainty before any response regression.

Passing these checks establishes implementation of the stated baseline contract. It does not establish permanent efficient-price impact, information versus inventory mechanisms, two regimes, causal shock identification or JFE readiness. Recorded-feed and observation limitations remain.

Next: Extend the already declared fixed measurement sensitivities and rebuild J/C/I from continuous quote/support data across internal calibration boundaries. Then rebuild the analysis ledger and regression inputs using the frozen stage-07a scaling. Verify economic/calendar support and implement the dependence-aware uncertainty method before fitting the fixed response equation. No new detector, duration-model, classifier or regime-threshold search.

Verified run: `step07c_chronological_panel_20261001_v2`.

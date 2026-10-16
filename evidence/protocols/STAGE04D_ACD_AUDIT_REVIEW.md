> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage 04d: first chronological duration-model audit

**Implementation: independently verified. Production ACD scores: not approved. Full-panel rolling estimation: not performed.**

This is Notebook 2 (Signal Construction). It completes the first model-validation gate within stage 04d. The research architecture is retained: quote-arrival timing contributes to a later liquidity-stress measure; ACD parameter persistence is not price-impact permanence. No shock outcomes or price-response results were used to choose these reference settings.

## What was repaired

The new path fits from an earlier period and evaluates a later period. It uses both completed waits and positive unfinished exposure, resets explicitly across observation/factor gaps, normalizes a wait using information available at its origin, and uses a mean-correct Weibull parameterization. It saves every optimizer attempt. No legacy full-sample factors, later-period parameter JSON, or unsegmented residual recursion is reused.

The seasonal working scale is observed exposure divided by completed events within prior same-weekday 15-minute origin cells. Its exponential working-likelihood interpretation is explicit; this ratio is not an assumption-free estimate of latent market waiting times. A duration running at a historical cutoff is clipped there, and its future completion is excluded. The factor remains fixed when the wait crosses a time cell or midnight.

## Fixed reference design

- Calendar origin: 2024-01-01 UTC. Initial seasonal history: 56 calendar days.
- ACD training origins: [2024-02-26T00:00:00+00:00, 2024-04-22T00:00:00+00:00).
- Validation origins: [2024-04-22T00:00:00+00:00, 2024-05-06T00:00:00+00:00).
- Seasonal requirements: four prior matching dates, 1,800 exposure seconds and 100 completed waits within a full 56-day history. Unavailable factors stay unavailable.
- Three fixed starts per dynamic fit; exponential and Weibull ACD(1,1), plus an iid exponential reference. Alpha and beta are nonnegative with sum below 0.999; Weibull shape lies in [0.2, 5]. These are reference constraints, not estimated facts.
- All four existing support policies are alternatives. No primary freshness policy is selected. Structural-only Weibull also has two isolated sensitivity fits: omit structural terminal contributions in training, or initialize conditional mean at one.
- This early validation window is model-development data, not the final paper test sample. Later observations do not fit these models. Repeated full-panel estimation was deferred pending this gate to avoid spending compute propagating a misspecified reference.

## Executed sample and attrition

The audited prefix contains 7,135,589 distinct quote origins, represented separately under each of four policies. These are alternative observation ledgers, not independent samples.

| Policy | Prefix spells | Completed events in prefix | Seasonal-ready spells | Training spells | Training completed events | Validation spells | Validation completed events |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural_only | 7,135,589 | 7,135,570 | 3,545,786 | 2,849,553 | 2,849,544 | 696,233 | 696,230 |
| fresh_15s | 7,135,589 | 7,090,258 | 3,545,786 | 2,849,553 | 2,826,759 | 696,233 | 689,883 |
| fresh_30s | 7,135,589 | 7,125,432 | 3,545,786 | 2,849,553 | 2,844,459 | 696,233 | 694,745 |
| fresh_60s | 7,135,589 | 7,134,097 | 3,545,786 | 2,849,553 | 2,848,803 | 696,233 | 696,034 |

Exact seasonal unavailability by reason, nanosecond exposure totals and all likelihood sample sizes are in the saved statistics. A component first quote is used as the origin of a subsequent wait; no incoming wait before that quote is fabricated. Zero terminal exposure is bookkeeping and does not enter the likelihood.

## Model results

14 reference/sensitivity models were fitted. 0 passed the predeclared diagnostic screen. Of the eight main dynamic ACD fits, 8 had validation diagnostic flags. Numerical convergence and model adequacy are separate outcomes.

| Policy | Model | Shape | Alpha + beta | Events / hazard exposure | Martingale lag-1 correlation | Mean negative log score (seconds) | Diagnostic flags |
|---|---|---:|---:|---:|---:|---:|---|
| structural_only | iid_exponential | 1.000000 | 0.000000 | 1.100695 | 0.126565 | 1.043184 | total_hazard_calibration, martingale_lag1_dependence, hazard_band_calibration |
| structural_only | eacd | 1.000000 | 0.998994 | 0.986494 | 0.009799 | 0.848455 | hazard_band_calibration |
| structural_only | weibull_acd | 0.743088 | 0.998489 | 0.996550 | 0.014381 | 0.764006 | hazard_band_calibration |
| structural_only | weibull_omit_structural_tail | 0.743090 | 0.998488 | 0.996543 | 0.014383 | 0.764006 | hazard_band_calibration |
| structural_only | weibull_unit_initialization | 0.743054 | 0.998512 | 0.997193 | 0.014295 | 0.763997 | hazard_band_calibration |
| fresh_15s | iid_exponential | 1.000000 | 0.000000 | 1.099017 | 0.118390 | 0.987502 | martingale_lag1_dependence, hazard_band_calibration |
| fresh_15s | eacd | 1.000000 | 0.998994 | 0.980512 | 0.009559 | 0.813641 | hazard_band_calibration |
| fresh_15s | weibull_acd | 0.747641 | 0.998994 | 0.986518 | 0.012011 | 0.734227 | hazard_band_calibration |
| fresh_30s | iid_exponential | 1.000000 | 0.000000 | 1.100226 | 0.118695 | 1.027608 | total_hazard_calibration, martingale_lag1_dependence, hazard_band_calibration |
| fresh_30s | eacd | 1.000000 | 0.998994 | 0.996449 | 0.008289 | 0.840881 | hazard_band_calibration |
| fresh_30s | weibull_acd | 0.742969 | 0.998994 | 1.004678 | 0.011453 | 0.756736 | hazard_band_calibration |
| fresh_60s | iid_exponential | 1.000000 | 0.000000 | 1.100082 | 0.124238 | 1.040815 | total_hazard_calibration, martingale_lag1_dependence, hazard_band_calibration |
| fresh_60s | eacd | 1.000000 | 0.998994 | 0.999363 | 0.008767 | 0.848182 | hazard_band_calibration |
| fresh_60s | weibull_acd | 0.742006 | 0.998994 | 1.009514 | 0.012838 | 0.763287 | hazard_band_calibration |

Compare scores between models within the same policy and validation sample. Different policies change the event/censoring mechanism, so their scores are not a common cross-policy ranking. A lower score does not guarantee calibrated tails. No model was selected merely because it beat the iid reference.

Constraint warnings: structural_only/eacd, fresh_15s/eacd, fresh_15s/weibull_acd, fresh_30s/eacd, fresh_30s/weibull_acd, fresh_60s/eacd, fresh_60s/weibull_acd. Proximity to the imposed persistence ceiling is a numerical/specification warning, not evidence of permanent price impact.

For the structural-only Weibull reference, validation hazard-band calibration is:

| Hazard band | Observed events | Model hazard exposure | Events / exposure |
|---|---:|---:|---:|
| [0.000000, 0.010050) | 0 | 6997.375 | 0.000000 |
| [0.010050, 0.051293) | 2,401 | 28690.295 | 0.083687 |
| [0.051293, 0.105361) | 22,931 | 37014.706 | 0.619511 |
| [0.105361, 0.693147) | 358,139 | 272582.338 | 1.313875 |
| [0.693147, 2.302585) | 238,375 | 255997.120 | 0.931163 |
| [2.302585, 4.605170) | 61,878 | 79245.077 | 0.780843 |
| [4.605170, infinity) | 12,506 | 18113.314 | 0.690431 |

Under the working observation model, event counts and integrated hazard exposure should agree in these bands. The diagnostic includes censored exposures; it does not incorrectly demand a uniform PIT from completed waits selected by censoring. Screen tolerances were fixed before execution: aggregate event/exposure ratio [0.9,1.1], absolute lag-1 martingale correlation at most 0.05, and band ratios [0.8,1.2] where exposure is at least 100. They are descriptive screening tolerances, not formal hypothesis-test critical values.

## Observation assumptions and remaining limits

The likelihood conditions on recorded-stream support and treats its stopping mechanism as ignorable for a reference calculation. A deterministic freshness cap supplies a known elapsed-time observation rule. Structural interruptions and discarded invalid quotes are not established independent censoring. Historical missing updates and reception latency remain unresolved. The fit therefore cannot be interpreted as an identified model of all spot-FX order arrivals.
Omitting structural terminal contributions is a sensitivity exercise, not a correction for informative censoring. The common seasonal scale is retained in that isolated likelihood comparison. Unit initialization tests a particular alternative; it does not reconstruct the latent state after missing observations. Joint seasonal/dynamic estimation, other initialization conventions, UTC/DST/holiday behavior and stability across later periods still require validation.
The current continuous exponential/Weibull reference can be incompatible with recorded timestamp granularity, vendor batching, remaining temporal patterns or a more complicated duration distribution. The present diagnostic failures do not uniquely identify which mechanism causes the discrepancy. No cause is asserted from the observed minimum duration alone.

The smallest completed structural-support wait in this audited prefix is 50 ms; the full previously verified duration ledger has a minimum of 4 ms. These are observed minima, not proven sampling thresholds. Audit timestamp spacing and preprocessing conventions across dates before imposing a shifted, rounded, discrete or other duration distribution.

## Verification evidence

All 260 tests passed. They include analytic-versus-finite-difference gradients, mean-normalized Weibull expectations, exponential nesting at shape one, survival-only censor contributions, state resets, absence of future information, exact training-cutoff equality, source-to-spell construction and parameter recovery from simulated censored Weibull ACD data.
The separate verifier checked 28,542,356 policy-spell rows, 48,384 seasonal parameter cells, 9,747,262 validation prediction rows and all 14 fitted model likelihoods. It reconstructed outgoing spells from incoming duration records and terminal rows, refitted seasonal cells directly from past clipped spells, and used SciPy linear filtering plus distribution logpdf/logsf instead of the compiled recurrence. Aggregate and daily diagnostics were reconstructed independently.
Source hashes, the compiler/version/options and kernel hash, all test sources, all optimizer attempts and all output fingerprints are saved. The compiled kernel only accelerates the explicit scalar recurrence; synthetic and full-data comparisons verify its numerical behavior. No additional software package or data subscription was purchased.

## Downstream decision

**Do not use these reference fits to generate calibrated shock probabilities or persistence conclusions.** The next bounded correction should investigate the observed distributional calibration failures, including timestamp/clustering behavior and seasonal/initialization sensitivity, on this designated development sample. It should compare justified alternatives using the same observation and chronology rules. Full-panel rolling calibration follows only after a defensible specification passes a broader validation plan; the final evaluation period must remain distinct from model development.
The pipeline remains unready for paper estimation. Shock onset, ambient-volatility state construction, price-response horizons, overlapping-event inference and robustness remain downstream.

## Saved evidence and sources

Registered audit: `data/signals/step04d_chronological_acd_audit_20260928/`; pointer `data/active_duration_model_audit.json`; central key `current_duration_model_audit` in `reports/pipeline_master_statistics.json`.
Each policy directory contains the seasonal spell ledger, daily sufficient statistics, cutoff corrections, day/slot parameters, validation observations, model predictions and daily validation statistics. Root statistics contain all fit attempts, parameters and aggregate diagnostics. Production readiness and approval of ACD shock scores remain false.

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_chronological_acd.py --verify-only step04d_chronological_acd_audit_20260928
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_chronological_acd.py --test-only
```

The [Engle–Russell ACD paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6809) motivates conditional-duration dynamics. [Prigent, Renault and Scaillet, section 2.1](https://eprints.lse.ac.uk/119095/1/dp364.pdf) gives the exponential/Weibull mean and hazard parameterizations. [Statsmodels duration documentation](https://www.statsmodels.org/stable/duration.html) distinguishes completed and censored observations; its [survival estimator notes](https://www.statsmodels.org/dev/generated/statsmodels.duration.survfunc.SurvfuncRight.html) make the role of censoring assumptions explicit. These sources do not validate this data’s observation mechanism or select our numerical reference settings.

All six earlier active pointers, all prior verified and archived exploratory statistics, and 36 registered ancestor output fingerprints are unchanged. No manuscript, old parameter JSON, legacy processed data or figure was overwritten.

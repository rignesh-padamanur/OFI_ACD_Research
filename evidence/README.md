# Saved evidence

These selected records support the paper and permit recomputation of its reported response summaries, forecast losses and figures. They are not a complete raw-data or model-fitting archive. Run `python reproduce.py results` from the repository root to check the included arithmetic.

## Reading map

| Question | Saved records under `data/signals/` |
|---|---|
| Initial, subsequent and total displacement; common cohort; quote-age restrictions | `step07m_joint_response_20261002/` contains membership, complete response summaries, endpoint classifications and daily accounting. |
| Conditional uncertainty | `step07n_joint_uncertainty_20261002/` contains daily scores, all 24 mean intervals, all cross-covariances, alternative lag 40 and related diagnostics. |
| Forecast timing and sample accounting | `step09a_prediction_ledger_20261003/` contains fold and issuance summaries. Full training membership and input rows remain in the research archive. |
| Pressure–variation forecast comparison | `step09b_persistence_forecast_20261003/` contains all saved scored-event records, predictions and loss tables, including unavailable labels and all reference methods. |
| Ordering measurement and its forecast comparison | `step10a_sequence_measurement_20261003/` contains coverage and validation records; `step10b_sequence_forecast_20261003/` contains all saved scored-event records and loss tables. |
| Separate quoted-spread forecast exercise | `step03_timing_forecast_20261008/` contains complete daily and declared subgroup aggregates, all planned category-cell coverage, all three uncertainty lags and verification receipts. |
| Detector comparison | `step11_detector_contribution_20261004/` contains the complete declared comparisons, age screens, calendar and overlap results, and sufficient-statistic summaries. |
| Measurement and failed calibration diagnostics | Initial acquisition, observation support, quote imbalance, seasonal normalization, ACD validation, frozen scaling and chronological panel records are retained under their original run names. The ACD records retain every originally reported freshness policy. |

The earlier conditional-association and spread-history summary records remain accessible because the historical reviews discuss their unfavorable or unresolved results. They do not enlarge the current paper's principal forecast claim.

## Provenance and limits

[manifest.json](manifest.json) lists every copied numerical record and protocol with original and release SHA-256 hashes. Numerical files, JSON specifications, original manifests and original verification receipts preserve their source bytes. Markdown protocols have only a historical notice and navigation-link conversions; both versions' hashes are recorded.

Historical JSON manifests contain original paths and upstream hashes. Those paths describe the archived run; they are not instructions to run code from another machine's filesystem. A `passed` field in a retained receipt records the original audit. It does not imply that its full original verifier can run against this smaller selection. The new result verifier checks saved-result arithmetic directly using the bundled files and writes a fresh, separate receipt in `build/`.

The compact repository omits raw observations, intermediate signals, full model-fitting inputs, saved timing-model objects and cycle-level spread predictions, complete training membership, the 301-second response-path archive, and frozen copies of every historical source. Rebuilding the full pipeline requires the original archive or regenerated inputs and a separately verified reconstruction. No public archive location is invented by this release. See the repository's reproduction instructions for the available checks and the restoration boundary.

The old archive's `forecasts.parquet` files are omitted because the retained `scored_events.parquet` contains the same prediction columns plus outcome availability and losses. The new verifier recomputes squared losses from those predictions and observed outcomes. This does not independently re-estimate their coefficients or establish when the original fits were run.

The evidence uses recorded quote prices and vendor-reported size fields. It does not establish complete market coverage, causal mechanisms, permanent price effects or universal absence of predictability. All negative comparisons and declared diagnostic strata remain represented.

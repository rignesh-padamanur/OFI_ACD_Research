> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

**Stage 02b: observation eligibility and attrition — step02b_observation_20260927**

Status: executed and independently verified on cached data. No new quotes were acquired; no retained quotes, prices, sizes, legacy estimates, or the active data pointer were changed. This stage registers observation support separately. Coverage completeness and the economic pipeline remain unverified.

**What was corrected**

Known absent hours and nominal weekly closures now delimit structural segments. Consecutive clean quotes are also separated when an intervening cached raw record was removed during stage-01 cleaning. Support stops at the first removed raw timestamp, so an invalid state does not retroactively remove the preceding valid quote. The current parent removed 1,589 zero-spread records; this layer does not delete them again or infer new outages.

Unexplained within-session silence remains in the source data and is separately reported. Freshness limits of 15, 30 and 60 seconds are diagnostic scenarios. Structural-only support has no freshness limit and is a comparison benchmark, not an endorsed primary sample. No threshold was selected using persistence, regression or classification results.

**Quote and transition accounting**

| Quantity | Count |
|---|---:|
| Retained quotes, unchanged | 55,708,318 |
| Adjacent quote pairs before segmentation | 55,708,317 |
| Structural segments | 208 |
| Pairs excluded across structural boundaries | 207 |
| Pairs remaining within structural segments | 55,708,110 |

There is also one initial quote without a predecessor. Excluding a transition does not remove either endpoint quote. Segment starts remain valid current observations, but their stateful features require new historical context.

| Segment-start reason (combined reasons counted once) | Starts |
|---|---:|
| weekly_closure | 88 |
| removed_raw_record | 69 |
| weekly_closure + absent_nominal_open_hour | 46 |
| absent_nominal_open_hour | 4 |
| sample_start | 1 |

| Maximum interquote gap scenario | Within-segment gaps exceeding limit | Remaining adjacent pairs under this scenario |
|---|---:|---:|
| 15 seconds | 327,906 | 55,380,204 |
| 30 seconds | 71,687 | 55,636,423 |
| 60 seconds | 9,958 | 55,698,152 |

**Clock-window availability**

The denominator below is 57,794,400 integer-second clock anchors in 16,054 observed nominally open UTC hours. Before observation restrictions, the nominal weekly calendar contains 16,198 hours; 144 are absent, with holiday/outage causes unresolved. No percentage below is an event-retention rate.

The 300-second historical context is a diagnostic. It is not a chosen ACD warm-up or final volatility window. The future 300-second window must remain continuously supported; checking its two endpoints alone is insufficient.

| Support scenario | Current-time support | 300s history available | 300s history + 300s future available | Joint / observed-hour anchors | Future unavailable among history-eligible |
|---|---:|---:|---:|---:|---:|
| structural_only | 57,787,100 | 57,742,984 | 57,700,984 | 99.838% | 42,000 (0.073%) |
| fresh_15s | 54,228,878 | 26,454,408 | 20,438,400 | 35.364% | 6,016,008 (22.741%) |
| fresh_30s | 56,593,282 | 45,624,961 | 41,344,137 | 71.537% | 4,280,824 (9.383%) |
| fresh_60s | 57,517,286 | 55,139,540 | 53,587,129 | 92.720% | 1,552,411 (2.815%) |

These are one-second opportunities for a future analysis, not shocks. Eligibility depends on the policy; enough clock coverage does not establish enough independent events or adequate power. A zero-horizon entry describes observation support, not an estimated impact.

**Time-of-day composition**

Percent of observed-hour anchors supporting both the diagnostic 300-second history and 300-second future. Bands use UTC and do not stand for London, New York or Tokyo sessions.

| Scenario | 00:00–06:00 UTC | 06:00–12:00 UTC | 12:00–18:00 UTC | 18:00–24:00 UTC |
|---|---:|---:|---:|---:|
| structural_only | 99.994% | 99.996% | 100.000% | 99.358% |
| fresh_15s | 14.038% | 47.869% | 68.161% | 11.186% |
| fresh_30s | 58.068% | 92.629% | 95.284% | 39.853% |
| fresh_60s | 94.819% | 99.612% | 99.586% | 76.684% |

Complete hourly and monthly-by-band counts are saved for all 72 combinations: four support scenarios, three historical contexts (0/60/300 seconds), and six future horizons (0/1/5/30/60/300 seconds).

**Timing and downstream contract**

- All timestamps are UTC nanoseconds. Intervals include their start and exclude their end. Quote age equal to a freshness limit is allowed; hard boundaries stay exclusive. Support never extends beyond the last cached raw timestamp.
- Structural segments partition every retained quote by its zero-based row position in the fingerprinted parent panel. Month/file boundaries alone do not reset a segment.
- `pre_context_eligible` measures support over the historical window; future-window availability is returned separately. The hourly presence audit is retrospective, so clock-grid masks are sample-quality diagnostics, not an online data-health detector or automatically Ft0-measurable predictors. Detect shocks using contemporaneously observed information, then attach audit eligibility and outcome availability. Interval ends, availability flags and eventual segment duration must not become predictors.
- Continuous support is required across each full window. Quote values have not been forward-filled, interpolated, or generated by this stage. Any later as-of quote lookup must obey the selected policy and structural boundaries.
- Duration and OFI calculations must suppress differences across structural starts. Durations inside segments remain untruncated. Proper treatment of uncertain silence in a duration model is still pending.
- Rebuilt modules must explicitly select an observation policy through `load_observation_support`; there is no silent default. Legacy notebooks remain blocked. Event-dependent exclusions and pre-event sample-composition comparisons must be reported once a valid detector exists.

**Verification and limits**

89 tests passed. The independent verifier checked all 55,709,907 cached raw timestamps, every one of 55,708,318 clean timestamp/provenance pairs, 208 structural segments, 410,383 support intervals, and 1,629,504 hourly opportunity counts. It uses a separate calendar implementation and a separate count-by-bin algorithm. Corruption tests alter numerical outputs while refreshing their hashes, ensuring verification is not only a fingerprint check.

Controlled cases cover missing hours, weekly closures, cleaning omissions, cross-hour continuity, exact freshness thresholds, nanosecond edges, absent future outcomes, interrupted paths with fresh endpoints, and invariance of history eligibility at previously observed quote timestamps to appended future records. This test does not establish online observability of whole-hour absence labels.

This establishes the implemented observation-support contract on the cached sample. It cannot certify unseen quote updates, resolve historical size-only losses or holiday causes, identify informed trading/dealer inventory, select a primary freshness rule, or establish paper readiness. No OFI, shock labels, persistence estimates, regressions, or classifier results were recomputed.

**Artifacts and reproducibility**

- Parent: `data/runs/step01_quote_sizes_20260927_v2/`; unchanged.
- Registered eligibility: `data/eligibility/step02b_observation_20260927/` with input/output hashes, executed source snapshots, test evidence, numerical statistics, and independent verification.
- `segments.parquet`: compact quote-index partition and reasons for resets.
- `support_intervals.parquet`: structural and freshness-specific interval support.
- `long_interquote_gaps.parquet`: retained clean interquote gaps above 15 seconds, identifying structural crossings.
- `window_attrition_by_hour.parquet` and `window_attrition_by_month_band.parquet`: all diagnostic window counts.
- `reports/pipeline_master_statistics.json`: new `current_eligibility_stage`; existing data and acquisition history remain separate.

Reproduce tests: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_observation_eligibility.py --test-only`.
Reverify: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_observation_eligibility.py --verify-only step02b_observation_20260927`.
Regenerate this report: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python reports/stage02b_report.py step02b_observation_20260927`.

Next bounded repair: validate the imbalance measure using the preserved quote sizes and these boundaries. Retain the existing research architecture provisionally; do not substitute a different primary shock definition on the basis of this coverage exercise alone.

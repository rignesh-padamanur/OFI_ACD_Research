> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

**Stage 03: recorded best-quote imbalance — step03_quote_imbalance_20260927**

The incorrect price-sign/spread statistic has been replaced by a price-and-size construction, executed on the full cached sample, and independently verified. Quote and eligibility inputs remain unchanged. This verifies a recorded-quote proxy; it does not identify transaction flow or make the economic pipeline ready for paper estimation.

**Definition and economic interpretation**

For consecutive admissible observations, let b and a denote best bid and ask prices and qb and qa their displayed sizes. The tick contribution is:

```text
e[n] = 1(b[n] >= b[n-1]) qb[n] - 1(b[n] <= b[n-1]) qb[n-1]
     - 1(a[n] <= a[n-1]) qa[n] + 1(a[n] >= a[n-1]) qa[n-1]
```

This follows the best-quote construction in [Cont, Kukanov and Stoikov, section 2.1](https://arxiv.org/abs/1011.6402). Positive contributions represent upward-directed quote pressure under this convention. Sizes at unchanged prices enter as differences; rising bid and ask prices no longer mechanically cancel. Applying the expression here produces a proxy from this particular FX feed, not a reconstruction of all underlying orders.

The [vendor API](https://www.dukascopy.com/client/javadoc/com/dukascopy/api/ITick.html) defines these sizes as quantities available at the best quotes. Its [BI5 documentation](https://www.dukascopy.com/wiki/en/development/data-export/) describes size units as millions of base currency. The archived downloader passes the two binary float fields through unchanged. Outputs use **stored size units**, with no conversion, spread division, depth normalization or seasonal standardization. Historical completeness and discarded same-timestamp size updates remain unverified.

**Executed accounting**

| Quantity | Count |
|---|---:|
| Quote rows preserved in tick output | 55,708,318 |
| Valid within-segment transitions | 55,708,110 |
| Unavailable contributions, retained as missing | 208 |
| Sunday quote rows retained | 614,902 |
| Quote-containing one-second buckets | 23,253,415 |
| Buckets containing an excluded transition | 208 |
| Buckets with no valid transition | 180 |
| Buckets crossing structural segments | 0 |

Exclusion codes are a bit mask: 1=no predecessor, 2=structural reset, 4/8=invalid current/previous price, and 16/32=invalid current/previous size. Actual counts: 0: 55,708,110, 1: 1, 2: 207. Invalid endpoints are not skipped to join nonadjacent quotes. Zero sizes are permitted without dividing by them; tied/decreasing timestamps fail explicitly.

**What the old formula missed**

The following comparison recomputes the old unscaled price-sign expression on the same valid transitions. Its sign is also the sign of the old spread-scaled expression for positive spreads. It is not a comparison with old event labels or fitted results.

| Relationship on common valid transitions | Count | Share |
|---|---:|---:|
| Both zero | 103,426 | 0.186% |
| Old formula zero; corrected proxy nonzero | 18,255,096 | 32.769% |
| Old formula nonzero; corrected proxy zero | 2,871,414 | 5.154% |
| Both nonzero, same sign | 17,513,374 | 31.438% |
| Both nonzero, opposite signs | 16,964,800 | 30.453% |

Additionally, 458,735 transitions have unchanged bid and ask prices but nonzero size-based contributions. This subset overlaps the old-zero/corrected-nonzero row. Parallel upward quote moves: 8,903,116; parallel downward moves: 8,893,890. Both categories were zero under the old price-sign difference.

These are measurement diagnostics. Since current price changes enter the formula, contemporaneous direction agreement cannot establish information, causality or forecasting ability.

| Corrected tick contribution | Count | Share of valid transitions |
|---|---:|---:|
| negative | 26,327,798 | 47.260% |
| zero | 2,974,840 | 5.340% |
| positive | 26,405,472 | 47.400% |

**Policy diagnostics and moments**

No primary freshness policy has been selected. Tick-level policy counts apply the indicated maximum predecessor gap within structural segments. A usable bucket additionally requires full continuous bin support and zero excluded constituent transitions. Structural-only has no freshness cutoff and is a diagnostic benchmark.

| Scenario | Tick contributions under gap rule | Usable quote-containing buckets | Mean tick proxy | Tick sample SD | Mean bucket proxy | Bucket sample SD |
|---|---:|---:|---:|---:|---:|---:|
| structural_only | 55,708,110 | 23,253,206 | 0.021028989 | 3.6248937 | 0.050380042 | 6.5048152 |
| fresh_15s | 55,380,204 | 22,925,300 | 0.021098766 | 3.6261272 | 0.050658737 | 6.5173677 |
| fresh_30s | 55,636,423 | 23,181,519 | 0.021035278 | 3.6253837 | 0.050348837 | 6.508514 |
| fresh_60s | 55,698,152 | 23,243,248 | 0.021044023 | 3.6249975 | 0.05040638 | 6.505525 |

All moments are descriptive in stored size units, not calibration parameters. They were not used to select a threshold or fit any model. Policy-dependent bucket counts cannot be compared directly to stage-02b clock-window counts: this table contains only seconds with recorded quotes and concerns a one-second signal, not a 300-second history/future requirement.

**Timing and missingness contract**

- Each tick contribution uses only the current and immediately preceding admissible quote. Structural starts have missing contributions, not invented zeros.
- Buckets cover `[bin_start, bin_start + 1 second)`. The aggregate is available at the right boundary, recorded in `available_at`. An update exactly on that boundary belongs to the next bucket. Recorded event time is used; unobserved network/receive latency is not estimated.
- `observed_subtotal` sums valid recorded contributions for diagnostics. `quote_ofi` is missing whenever any constituent transition is excluded. Consumers must also apply the selected `usable_<policy>` flag; a partial observation interval is not a complete valid signal.
- Seconds with no recorded quotes are absent from the sparse output. The next clock-grid stage must distinguish admissible no-update seconds (zero recorded contributions under the chosen support contract) from unsupported periods (missing). Neither blanket zero filling nor silently treating quote-containing seconds as the full estimation sample is admissible. This stage has not selected that final grid policy.
- Full-bin support and usability flags are retrospective audit labels. They are not ex-ante market-state predictors. Signal availability is kept separate from the historical coverage audit.
- Stateful normalization and event detection must respect segment boundaries and the completed-bin clock. Strictly pre-shock conditioning must use information available before the defined onset. Future episode closure must not be backdated.

**Verification evidence**

All 129 tests passed. The independent verifier checked all 55,708,318 saved quote contributions and 23,253,415 saved bins. Production uses branch cases and pandas grouping; verification uses indicator arithmetic and NumPy reduce-at aggregation. Different processing chunk sizes test continuity across arbitrary file-processing boundaries. Counts, monthly tables, sign comparisons and moments were independently reconciled.

Controlled checks include all nine bid/ask move combinations, equal-price size changes, symmetric spread narrowing, fixed-size spread round trips, zero and invalid sizes, structural resets, exact nanosecond/second boundaries, Sunday preservation, partial bins, long incoming gaps and unchanged prefixes when later quotes are added. Corruption tests update fingerprints after altering numeric values or availability flags, confirming detection beyond hash mismatches.

Inserting an unchanged state leaves raw accumulated flow unchanged; splitting fixed-price size changes telescopes. Arbitrary coarsening does not: a test deliberately demonstrates different totals after deleting an intermediate price update. Therefore this repair does not claim full invariance to tick clustering, quote oscillations, missing updates, or every closed price-size path.

**Downstream consequences and remaining work**

The new measure changes signal units, signs and magnitudes. Existing seasonal factors, OFI means/standard deviations, stress-score covariance matrices, thresholds, candidate events, labels and model estimates are not reusable. They remain marked pending; this repair has not recomputed or endorsed any persistence/regime result. The legacy `compute_ofi` call now raises an explicit error instead of silently returning the invalid definition.

The next repair is past-only seasonal adjustment, duration treatment and signal normalization under an explicit observation policy. The final window/freshness policy and economic treatment of uncertain silence still need to be resolved before fitting the event pipeline. A correct arithmetic proxy alone cannot distinguish informed trading from inventory adjustment or establish permanent price discovery.

**Saved artifacts**

- `data/signals/step03_quote_imbalance_20260927/quote_ofi_ticks.parquet`: one row per parent quote, with bid/ask components, exclusion code and raw proxy.
- `data/signals/step03_quote_imbalance_20260927/quote_ofi_1s.parquet`: sparse completed-bucket aggregates, counts, support and usability flags.
- The same run contains `statistics.json`, `monthly_statistics.parquet`, `methodology.json`, input/output fingerprints, executed code snapshots, tests and independent verification.
- `data/active_quote_imbalance.json` registers this signal run separately from the unchanged data and observation-support pointers.
- `src.pipeline_state.iter_quote_imbalance(root, policy=...)` checks registered lineage and hashes, masks unusable sums and yields completed-bin signals. It requires an explicit policy and never manufactures empty-bin zeros.
- `reports/pipeline_master_statistics.json` records the new stage while preserving prior verified stages and quarantined legacy statistics.

Tests: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_quote_imbalance.py --test-only`.
Reverify: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python repair_quote_imbalance.py --verify-only step03_quote_imbalance_20260927`.
Regenerate this review: `PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/fxresearch/bin/python reports/stage03_report.py step03_quote_imbalance_20260927`.

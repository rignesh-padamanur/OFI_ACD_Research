# Compact inputs for the two main empirical figures

These files contain the observed values needed by `reporting/plot_price_results.py`.
They reproduce the historical figures without downloading or loading the full
saved response-path archive. They do not add events, refit models, change sampling,
or supply new uncertainty estimates.

| File | Contents |
|---|---|
| `mean_response_path.csv` | The three direction-aligned mean components, in pips, at each of the 301 integer horizons from 0 to 300 seconds. Every row uses the same 127,013 events; `n_events` records that denominator. |
| `continuation_distribution.npz` | All 127,013 five-minute continuation values, in pips, with their event IDs in ascending order. Load with `numpy.load(..., allow_pickle=False)`. No tail observations are removed. |
| `interaction_quarter_losses.csv` | All ten saved quarterly rows for the interaction forecast, including its additive, zero-continuation and training-mean comparisons. |
| `sequence_quarter_losses.csv` | All ten saved quarterly rows for the ordering forecast, including its richer baseline, zero-continuation and training-mean comparisons. |
| `export_receipt.json` | SHA256 and size of each original source and exported data file, cohort selection, numerical checks and summary values. |
| `figure_validation.json` | Exact pixel agreement with the original two PNG figures and identical PNG/PDF hashes after regeneration in a relocated copy. |

The export retains the exact binary continuation values. CSV floats use 17
significant digits and are read with pandas `float_precision="round_trip"`;
export validation confirms exact equality to every original exported value.
The exporter also checks all 38,230,913 event/horizon pairs, `I = J + C`, fixed
initial displacement, endpoint equality to the membership table, and agreement
with the archived mean and quartile summaries. This is a reconstruction check
against saved outputs, not an independent validation of the original market feed.

The path uses strict-left recorded quote marks; it does not require a fresh quote
at every second. Panel B's main view displays -10 to 10 pips, but the ECDF is
computed from the entire sample and its inset shows the full observed range.
Quarterly differences equal extension MSE minus the stated benchmark MSE, in
pips squared. Negative values favor the extension. Pooled differences use event
weights, not an unweighted mean of quarters. The first and last quarters are
partial periods (6 May--30 June 2024 and 1--31 July 2026). The figures imply neither independent quarterly observations
nor statistical precision beyond the underlying historical comparisons.

To verify the compact inputs without changing files, run from the release root:

```sh
python reporting/plot_price_results.py --check-only
```

To regenerate both figures into ignored `build/figures/` (preserving the published PDFs):

```sh
python reporting/plot_price_results.py
```

Use `--output-dir /path/to/output` to select another generated-output directory.
Run without Python's `-O` or `-OO` options: those modes disable verification
assertions and are rejected by both scripts.

The optional exporter needs the intact historical archive, including its 60
`step07e` path fragments and the `step07m`, `step09b`, and `step10b` sources named
in the receipt. It never modifies that archive:

```sh
python reporting/export_price_plot_data.py --archive-root /path/to/research-archive
```

The original reference implementation is `reports/plot_section4_results.py` in
the research archive. Its SHA256 is recorded for provenance. Neither a successful
compact-input check nor matching figure files constitutes a complete pipeline rerun.

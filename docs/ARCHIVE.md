# Historical inputs outside Git

The repository includes enough saved evidence to run its documented result checks and figures. It does not include the full quote and intermediate-run archive. There is currently no public archive URL or DOI, and downloading a fresh vendor feed is not guaranteed to recover the exact historical data vintage. The original research workspace is preserved by the author.

## Verify an existing archive

Use an intact copy of the original research archive, with its original relative paths. The following command only reads files:

```sh
python tools/archive.py /path/to/original/FX_Microstructure
```

`provenance/archive_inputs.json` lists the named original sources used for the release's code, evidence and compact plot export, with byte counts and SHA256 hashes. This check covers that named input set; it is not a complete raw-data rebuild or a guarantee that every transitive historical dependency is present.

To rebuild the compact price inputs from that archive, run:

```sh
python reporting/export_price_plot_data.py --archive-root /path/to/original/FX_Microstructure --output-dir build/reexported_price_inputs
```

The exporter requires the original event-membership and summary tables, both quarterly loss tables, and all 60 cycle path tables. Its explicit archive argument avoids assuming the author's machine path. Re-export into a separate output directory when comparing versions; preserve the shipped input hashes.

## Required upstream classes

Full historical reconstruction requires more than the compact result tables:

- Cached monthly quote observations, acquisition coverage and structural-support masks.
- Selected quote-measurement, seasonal calibration, duration, chronological-panel and response runs.
- The earlier `step07c_chronological_panel_20260930` run used by the corrected-panel builder and verifier, including its diagnosed failed state.
- Origin, predictor, label and fold records for the price forecasts; the sequence-measurement archive; saved training/fit records and predictions.
- Step 11 features and labels, the repaired panel and Step 12 preservation records used by the separate timing/spread comparison.
- Frozen `code/`, `postfit_code/`, `full_test_sources/`, manifests, registration ledgers, protocols and verification receipts at their registered paths.

Original manifests sometimes contain absolute paths and hashes of earlier documentation. Those records are retained unchanged in `evidence/` for provenance; the portable reporting commands never open those absolute locations. Do not rewrite historical hashes merely to make a relocated builder pass. A full raw reconstruction needs an explicitly mapped archive and the matching dated environment/registration state. That workflow has not been converted into a portable one-command rebuild in this release.

## Keeping the public checkout small

Raw observations, large intermediate panels/models, failed-run history and frozen archive copies remain outside ordinary Git. Original notebooks, superseded manuscripts, unused figures, working renders and caches are also excluded. Selected verification and unfavorable results remain included. A file's exclusion from Git is not a recommendation to delete its local or archival copy.

The publication repository has a separate Git history based on the existing remote licence commit. It does not include the research branch's oversized data commit. Publishing this repository does not require rewriting or deleting the original research history.

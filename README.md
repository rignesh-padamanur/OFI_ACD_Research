# EUR/USD quote adjustment and forecast evaluation

This repository accompanies **Price Adjustment at Detection: A Historical Evaluation of EUR/USD Quote Events**. It contains the current paper, the evidence underlying its reported comparisons, reproducible figures, and the scientific code and selected tests. Start with the [paper](paper/main.pdf), then use the [evidence guide](evidence/README.md) to inspect the results.

The question is how much of the five-minute displacement is already present when an unusual quote event is detected, and whether two specified summaries of preceding conditions improve forecasts of the remaining movement. On 127,013 common events, average displacement is about 0.380 pips at detection and 0.395 pips five minutes later. The pressure–variation interaction and pressure-ordering extension increase realized forecast MSE by approximately 0.0144% and 0.0027%, respectively. The separate quoted-spread application also retains its unfavorable pooled comparison. These are historical, sample-specific measurements and forecast comparisons; they do not identify a causal mechanism or establish universal unpredictability.

## Reproduce the saved evidence

Use Python 3.12 and a C compiler for the scientific unit tests. The release was tested on macOS arm64; platform details and limits are in the [reproduction guide](docs/REPRODUCTION.md).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py all
```

The last command checks the release inventory, reconstructs reported results from saved observations and losses, regenerates the three figures, runs the selected computational tests, and checks file integrity again. It does not download quote records or refit the historical models. Check receipts and regenerated figures go into ignored `build/`; the published figure files remain unchanged.

Individual commands are available:

```sh
python reproduce.py check      # Standard library only: hashes, inventory and links
python reproduce.py results    # Independent arithmetic from saved evidence
python reproduce.py figures    # Three data-driven figures
python reproduce.py tests      # Scientific unit and guard tests
```

## Find what you need

| Location | Purpose |
|---|---|
| [paper/](paper/main.tex) | Current manuscript source, inline bibliography and PDF |
| [evidence/](evidence/README.md) | Selected saved results, protocols, diagnostics and original verification receipts |
| [reporting/](reporting/verify_saved_results.py) | Portable result reconstruction and plotting |
| [data/plot_inputs/price/](data/plot_inputs/price/README.md) | Compact, hash-bound inputs for the two main figures |
| [pipeline/](docs/PIPELINE.md) | Scientific implementations, historical production entry points and selected tests |
| [provenance/](provenance/release_manifest.json) | Release inventory, source mappings and verification records |

The root entry point is `reproduce.py`; readers do not need to navigate the historical runners to reproduce the displayed evidence. All reported quarters, unfavorable forecasts, relevant measurement restrictions and declared supporting checks remain accessible.

## Reproduction boundary

The compact repository is self-contained for its saved-evidence checks and figures. Full reconstruction from raw quotes additionally requires the historical observation, intermediate-run and frozen-source archive. Some corrected stages consume earlier failed runs and registration records. Those files are deliberately outside ordinary Git, and the historical runners are not advertised as a fresh one-command raw-data rebuild.

[Archive instructions](docs/ARCHIVE.md) explain the required records, original-source hash verification and the distinction between original frozen code and the portable release copy. No public archive download location is currently established. This limitation concerns upstream reconstruction; the included results and figures can be checked without that archive.

The software retains the existing [GPLv3 licence](LICENSE). Vendor observation data are not bundled as raw downloads; the software licence does not establish vendor-data redistribution terms.

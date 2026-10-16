# Reproducing the publication release

## Environment and commands

Run the commands in the root [README](../README.md). `requirements.txt` pins the eight direct scientific/reporting packages and their active transitive dependencies from the tested Python 3.12 environment. It is intentionally smaller than the author's complete research environment.

`python reproduce.py check` needs only Python's standard library. It validates every shipped file against `provenance/release_manifest.json`, detects unexpected non-build files, checks documentation links and rejects files at or above the regular GitHub 100 MiB boundary. The manifest excludes itself to avoid a circular hash.

`python reproduce.py results` reconstructs descriptive response statistics and quote-age restrictions from saved event membership; response covariance arithmetic from saved scores; and pooled, daily, fold and quarterly price losses from saved event-level predictions and outcomes. It also checks the timing/spread aggregates and declared diagnostics. The receipt is written to `build/results_verification.json`. Original run verification receipts are historical evidence, not substitutes for these fresh checks.

`python reproduce.py figures` regenerates the two main price figures and the appendix spread-concentration figure under `build/figures/`. The price export retains the complete common-cohort distribution and 301 mean-path points. The export receipt binds these inputs to all 64 original source files. No smoothing, horizon selection or model refitting is introduced. PDF timestamps are fixed or suppressed; regeneration was byte-identical in the tested environment. Rendering can differ across operating systems or font stacks, so the portable command checks numerical inputs without requiring a new platform's PDF bytes to match. The published PDFs remain unchanged. PNG previews are generated locally but are not tracked.

`python reproduce.py tests` runs the selected scientific unit tests and the panel-precision/guard tests under `pipeline/`. These include synthetic computations and integrity rejection cases. They do not establish market-data completeness, statistical coverage, causal identification or a full historical refit. A failing check raises a nonzero exit status; do not replace the manifest to conceal a failure.

## Compiler and platform support

The ACD implementation includes `pipeline/src/acd_kernel.c`. The release copy builds a shared library using the local C compiler: a dynamic library on macOS and a shared position-independent library on Linux. `CC` can specify the compiler command. Windows requires a Linux environment such as WSL for this implementation. Install compiler tools before running the scientific tests.

The portable build change is isolated to the copied implementation. Original model code and frozen source hashes remain in the research archive. Linux command construction is tested, but actual Linux numerical execution is not certified by the macOS verification receipt. No platform result should be inferred from an unexecuted CI configuration.

## Paper compilation

The included PDF is the verified October 2026 manuscript. The source uses a release-relative figure search path and clarifies that the normal critical value 1.96 is approximate. Numerical estimates and the empirical scope are unchanged. From `paper/`, compile `main.tex` with Tectonic or a LaTeX installation containing the packages declared in its preamble. The bibliography is inline; no external `.bib` file is required. Build logs and auxiliary TeX files are ignored. TeX installation is optional for reproducing the numerical evidence and figures.

## What is independently checked

Release hashes establish file identity. Result reconstruction establishes agreement between saved observations/predictions and the reported calculations. Unit tests check specified computational behavior. Figure checks establish agreement with the archived plotting inputs and reference images. These are distinct from raw-feed validation and from statistical or economic claims.

The [release verification record](../provenance/verification.json) records the actual commands, environment and outcomes of the clean-copy check. The [source mapping](../provenance/source_mapping.json) identifies copied, transformed and newly added files. Instructions for upstream records appear in [the archive guide](ARCHIVE.md).

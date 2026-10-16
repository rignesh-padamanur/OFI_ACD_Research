> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Timing measurement audit: fixed scope

8 October 2026. This protocol is saved before the numerical audit. It follows the question brief in `paper/STEP01_TIMING_LIQUIDITY_QUESTION_2026_10_08.md` and the user's request to check repetition before pursuing another finding. Two example scored rows and source schemas have already been inspected. This is a retrospective measurement audit of an inspected cache, not a preregistered economic test.

## One claim

For a duration-eligible one-second bin containing every completed incoming recorded quote gap, mean raw duration is determined by update count and the quote ages at the bin's two strict-left boundaries. ACD normalization can add a representation of past timing expectations; it does not supply an independent additional raw-duration observation.

For the bin `[s,s+1)`, let `n` be its recorded update count, `a_pre` the age of the last recorded quote strictly before `s`, and `a_end` the age of the last quote strictly before `s+1`. All quantities are in seconds. With an available predecessor in the same structural component, the completed gaps telescope:

`A = sum(gap_j) = 1 + a_pre - a_end`, and `mean_raw_gap = A/n`.

The current duration feature is `z = -log(mean(gap_j/(scale_j*psi_j)))`. The seasonal scale and ACD expected normalized duration were issued at each gap's origin. Define the duration-weighted harmonic expectation `H = A/sum(gap_j/(scale_j*psi_j))`. Then:

`z = log(n) - log(A) + log(H)`.

This is an algebraic identity under the stated measurement convention, not a novel economic theorem. The empirical audit tests whether the convention and recorded values actually satisfy it. A nonlinear deterministic relationship does not mean a finite learner will use the inputs equally well.

## Fixed population and calculations

1. Read predictor columns only from all 59 scored deployment cycles `001` through `059` of `step07c_chronological_panel_20261001_v2`. Report counts, count equality, finite boundary ages, positive exposure, the residual/log identity, and the exact information clock. Do not silently drop failures. If any row fails these conditions, retain its count and halt the intended full-population assertion.
2. Report Pearson correlations and variance/covariance accounting for `log(n)`, raw speed `log(n)-log(A)`, the normalized feature `z`, and its expectation component `log(H)`. These are descriptive predictor summaries. They neither estimate a future relationship nor establish economic value. Component variances are not shares of explained economic variation.
3. Independently reconstruct **every scored bin** in fixed cycles `002`, `030`, and `059`: the first newly fitted deployment, the central numbered deployment, and the final deployment. Read the structural raw-spell ledger, issued origin forecasts plus carried forecasts, and physical cleaned quote timestamps. Reconstruct incoming gap sums and counts, normalized residual means, seasonal-only means, both boundary ages, and the identity above. Check support IDs against the inherited structural intervals and check forecast vintage at origin. Do not change cycles after seeing results.
4. Raw-spell counts, quote counts, support membership, timestamp boundaries, and forecast identities must match exactly. Floating values must agree within relative tolerance `1e-10` and absolute tolerance `1e-10` for seconds/residuals/log values. Record maximum discrepancies as well as failures. Reject the claimed source reconstruction if a declared check fails.
5. Save a separate JSON receipt with source fingerprints and runtime versions. Preserve the earlier question brief, its receipt, scientific result files, and active pointers. This audit does not promote a production pipeline change.

## Exclusions and stopping rule

Read no future prices/spreads, economic outcomes, forecast training labels, or model performance files during computation. Fit no new model and choose no new outcome, horizon, feature threshold, or subgroup. Exclude neither holidays nor large gaps based on the summaries. Scored eligibility already excludes incomplete incoming-gap coverage; the identity is not asserted for all calendar seconds or unsupported bins.

A verified identity warrants an implementation-specific interpretation: raw mean timing duplicates count and boundary-age information; the candidate added representation is the normalization state. It does not establish that ACD has no value, that recorded quotes represent all market updates, that the feature predicts future costs, or that this is field-new. If the source audit fails, stop and report the failure before proposing economic fitting.

## Consequence for the next research decision

Combine this measurement result with a literature and saved-analysis overlap table. Proceed to an economic comparison only if it supplies a specific unanswered contrast and a defensible reason that a reader would care. Any subsequent comparison needs its own frozen protocol and must remain exploratory on this already inspected sample. Another fit is not automatically authorized by completion of this audit.

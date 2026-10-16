> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step10a completed: sequence design, measurement and contribution decision

3 October 2026. Run: `step10a_sequence_measurement_20261003`. This completes the authorized steps 1-2. The protocol was frozen before production measurement. No new response model, forecast comparison, outcome sort or research-hypothesis test was run. The original detector, response definitions, notebooks, scientific estimates and active pointers are preserved. manuscript.tex was excluded.

## Decision

**The recorded sequence is reproducibly measurable, retains the existing eligible population and adds a linearly nonredundant column to the declared benchmark. The substantial-novelty gate is not passed.** Consequently, the result is a completed design/measurement deliverable, not a newly discovered predictive relationship or an automatic recommendation to fit another response model.

The targeted literature review examined seven primary studies, with full-method access for three and explicitly limited access for four. History-dependent impact, adaptive liquidity, FX runs and sequence/liquidity relationships have close antecedents. The precise first-alert specification was not located, but that does not establish that it is previously undocumented. A conventional sequence statistic on this feed does not by itself supply a material economic contribution. See `STEP10A_LITERATURE_GATE.md` for the exact comparisons and access limits.

## What was frozen and measured

At each recorded onset s, the first alert is s+1 second. Divide the strictly preceding 300 seconds into sixty event-relative five-second intervals. Sum the five stored one-second quote-pressure values accurately within each interval, then retain their signs, including zero. The feature S is adjacent sign agreement minus the exact average adjacency over permutations with those same sign counts. No permutations are simulated and no exchangeability hypothesis is tested.

This subtraction distinguishes arrangement from the mechanical contribution of sign composition. The benchmark explicitly includes the subtracted composition reference as well as sign balance, nonzero-bin fraction, aggregate pressure, quoted conditions, recent 5/30-second pressure, prior 5/30/300-second price changes and calendar indicators. Detected direction and initial movement J are known at first alert and included. Their inclusion is a disclosed change from stage 09b's strictly-prior predictor set; no warning before event arrival is claimed.

The exact definition, allowlisted measurements and hypothetical C300 squared-error comparison are in `STEP10A_SEQUENCE_PROTOCOL.md` and `STEP10A_SEQUENCE_SPEC.json`. The definition was chosen for measurement clarity and compositional control, not its relationship to later prices. It is a conventional serial-dependence statistic. No alternative sequence measure or sampling grid was fitted or screened.

## Coverage and attrition

All figures below come from the versioned run's `statistics.json`; the new measurement was never conditioned on a future label.

| Population | Recorded origins | Previously eligible | Sequence available | Complete benchmark/sequence | Combined eligible |
|---|---:|---:|---:|---:|---:|
| All inherited origins |135,724|128,841|135,418|128,845|128,841|
| Initial training period |1,806|1,783|1,806|1,783|1,783|
| Later deployment origins |133,918|127,058|133,612|127,062|127,058|

There are **zero additional exclusions among the 128,841 previously eligible origins**. All135,724 origins remain in the saved table. Both pressure and price-history availability fail for 306 windows crossing inherited support. A further 6,573 origins lack a complete benchmark. Four origins have complete measurements but fail the inherited onset-confirmation requirement; they remain ineligible. The task does not reconsider or silently relax those earlier rules.

The later 127,058 figure is the issuance population, not the 127,013 eventual scored outcomes in stage 09b. No future-label availability was loaded or used to reproduce a scored sample. This preserves eligible origins even when a future response was unavailable in the earlier evaluation.

## Measurement and design diagnostics

The following statistics refer to the 128,841 inherited-eligible origins unless stated otherwise.

| Check or descriptive quantity | Result |
|---|---:|
| S mean |−0.008107022991|
| S standard deviation, denominator n |0.087820633687|
| S minimum / maximum |−0.681355932203 / +0.456497175141|
| All-zero preceding-pressure sequences |10|
| Order-degenerate compositions |21|
| Ordinary versus accurate five-term sum sign disagreements |0|
| Nonzero five-second sums within the local arithmetic diagnostic bound |0|
| Exact-zero five-second net sums with nonzero one-second gross pressure |164,766|
| Maximum absolute inherited pressure discrepancy |0|
| Maximum absolute inherited variation discrepancy, bps |9.140327e−13|
| Standardized baseline rank / columns |48 /48|
| Standardized augmented rank / columns |49 /49|
| Augmented matrix condition number |11.7772053|
| Largest absolute pairwise correlation between S and an active benchmark column |0.09453597|

The exact-zero cancellation count concerns origin-window subbins; overlapping histories can reuse a physical interval. It is not a count of unique market events. Such zero net pressure does not mean no quote activity. Degenerate compositions remain in the sample; S=0 describes the lack of an ordering contrast at that composition, not an absence of directional pressure.

Both initial-training and later-origin matrix audits add one rank with S:46 to 47 and48 to 49, respectively. Constant calendar indicators are omitted for this diagnostic only: Saturday is absent overall; hours 13/14 are also absent in the initial eligible training sample. Any later forecasting implementation would have to handle unseen calendar categories explicitly. No model was fitted here. Full-sample means/standard deviations used in matrix conditioning are diagnostic quantities, not newly authorized forecast preprocessing.

The rank increment excludes exact linear duplication in this specified matrix. Small pairwise correlations do not establish conditional independence, adequate economic controls, predictive value or causal identification. They cannot satisfy the novelty gate.

The inherited quote-mark convention remains consequential. Among eligible origins, the median of each window's maximum quote age is 24.186 seconds, its 99th percentile is 131.327 seconds, and its maximum is1,116.327seconds. No age threshold was introduced. Correct strict-left lookup does not certify fresh or executable quotes. The local arithmetic bound also excludes errors already present in one-second pressure construction and unrecorded vendor updates.

## What was independently checked

Nineteen focused tests passed, covering composition centering, sign reflection, equal-total/different-order histories, event-relative alignment, onset exclusion, supported empty seconds, invalid transitions, support boundaries, strict-left marks, cancellation, source identities, preserved eligibility and forbidden future columns.

A separately implemented verifier then passed **91 checks**. It reconstructed pressure from the inherited complete structural clock rather than the producer's sparse reader, and reconstructed historical quotes through an independent sequential matcher. It checked all 135,724 origin identities, all serialized sequence and price-mark slots, including missing values, all measured feature components, source hashes and eligibility. The available values comprise **8,125,080 five-second pressure bins and 8,260,498 historical price marks**. Saved pressure-bin values and source quote timestamps/midpoints agree with those reconstructions. Existing pressure and variation also reconcile within the frozen 1e−8 absolute/relative tolerance; that tolerance is not used to replace signs or choose observations.

The producer's serialized reload also reproduced saved descriptive statistics. The source-column allowlists exclude future responses and future-availability fields. The independent review is a separate implementation and critique, not a blind external referee certification. Its limitations and counterexamples are preserved in `STEP10A_INDEPENDENT_REVIEW.md`.

## Economic meaning and next decision

The available evidence permits the statement: **a single declared sequence feature can be measured on the same eligible recorded-event population and is not an exact linear restatement of the specified benchmark.** It does not permit a claim that the feature predicts persistence, that sequence structure identifies informational trading, or that it reveals dealer inventory absorption.

Public repricing, ordinary dynamics, event selection, shared quote error and vendor transformations remain competing explanations for any later association. In particular, the original downloader may have removed size-only updates sharing timestamps and prices. Their market ordering and size semantics cannot be recovered from this audit. A measurement pass describes the cache faithfully; it does not resolve those economic measurement limits.

**Do not automatically proceed to step 3.** The strong contribution case remains insufficient after the closest-literature comparison. A later exploratory forecast comparison could only be presented as such, with a separately justified reason it would add substantive knowledge. It would not be a routine repair, guaranteed discovery or confirmation on untouched data. No such response estimation was performed in this task.

The central statistics record registers this as a methodology/measurement review and preserves every prior scientific entry. The living methodology guide is updated to v1.26. Final source, ledger, archive and PDF checks are recorded in `step10a_final_integrity.json`.

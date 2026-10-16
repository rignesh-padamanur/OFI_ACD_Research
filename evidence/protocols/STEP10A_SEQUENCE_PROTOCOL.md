> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step 10a: one preceding-sequence measurement and literature gate

Protocol status: frozen before production measurement

Date: 3 October 2026. Authorization covers steps 1-2: contribution/literature assessment, a single specified feature, measurement, coverage and design diagnostics. It does not cover response estimation. This prospective contract is recorded before the stage10a production measurement; previously inspected data and earlier unsuccessful extensions remain disclosed. No new data are purchased. manuscript.tex is excluded.

## Question, claim and stopping decision

The proposed economic question is whether ordering in preceding recorded quote pressure adds information about five-minute price adjustment beyond aggregate state, recent dynamics and information already available at first alert. It is a conditional prediction question, not an identified private-information or dealer-inventory mechanism. The exact specification is a candidate, not evidence that the relationship exists.

The separate literature gate does not establish a substantial novel contribution. History-dependent impact, adaptive liquidity, FX runs and sequence/liquidity relations have close antecedents. Not locating this precise combination of feed, alert and horizon is not a priority claim. The authorized measurement audit will still be completed, but its success will not authorize a response-model run or override a failed novelty gate.

## Preserved clock and population

Preserve every stage09a recorded origin. Onset s is a whole-second boundary; the first alert is t0=s+1 second. The sequence uses exactly [s-300 seconds,s), partitioned relative to s into sixty consecutive five-second intervals. This matches the inherited state window and variation sampling interval; neither is optimized here. No eventual episode end/peak enters this construction.

Use inherited structural observation support. Every second of the sequence must lie in the onset's registered interval. An observed invalid transition makes pressure measurement unavailable. A second with no recorded update inside this support contributes zero recorded pressure, following the existing clock convention; unsupported silence is unavailable. This describes the recorded cache and does not certify a complete market feed. Do not drop zero bins or join signs across gaps.

Origin eligibility remains the stage09a past/first-alert rule. Measure available components for all origins and retain separate inherited eligibility, sequence availability, price-history availability and combined usability flags. Future outcome availability is never consulted. Report all-origin, eligible-origin, training-origin (before 6 May 2024) and later-origin counts. No new active data pointer or replacement of prior scientific estimates is permitted.

## One frozen sequence measure

For k=1,...,60, Q_k is the accurately accumulated sum of the five stored one-second recorded quote-OFI values in the kth preceding interval. Use math.fsum on those five binary floating-point inputs. Set z_k=sign(Q_k), including z_k=0. Define

- raw adjacency A = sum(k=1,...,59) z_k z_(k+1) / 59;
- composition reference E = [(sum z_k)^2 - sum z_k^2] / (60*59);
- the sole proposed extension S = A - E.

E is exactly the arithmetic mean adjacency over permutations of the same multiset of signs. It is not an economic null model, an exchangeability assumption, a simulated permutation p-value or a fitted adjustment. The centering prevents raw adjacency from merely restating sign prevalence. The benchmark includes E explicitly, because a linear sign-balance control alone would omit its quadratic dependence.

All-positive, all-negative and all-zero sequences give S=0: there is no ordering variation conditional on that composition. Retain these observations and save a composition-degeneracy flag; do not interpret zero as absence of persistent pressure. A conservative algebraic range is [-2,2]. All parameters are fixed; no alternative autocorrelation lag, sampling frequency, activity-time clock or normalized ratio will be searched.

Five-second gross pressure means sum |Q_k|, not tick-level gross order flow. The five-second grid intentionally discards within-bin order. This measure is a conventional serial-dependence summary, not claimed mathematical innovation. No one-sided economic sign is imposed: anticipatory liquidity could offset predictable pressure, while incomplete adjustment could accompany continuation.

## Frozen benchmark measurements

Let d be the detected onset quote-imbalance direction and J the recorded initial aligned price movement. Both are known at t0; they are not strictly pre-onset features. Keep the inherited pressure and variation scaling constants, estimated before the original deployment period, without re-estimation.

The benchmark measurement set consists of:

1. d and J; d-aligned preceding net pressure/cQ and preceding pressure magnitude/cQ.
2. Five-second gross pressure/cQ; standardized preceding variation; pre-onset spread in pips; log pre-onset displayed size; log(1+preceding recorded quote updates).
3. d-aligned sign balance/60, nonzero-sign fraction and the exact composition reference E.
4. d-aligned net pressure and gross five-second pressure over the final 5 and 30 seconds, in cQ units. Thirty seconds is a fixed compact tenth-window benchmark, not an estimated optimum.
5. d-aligned preceding midpoint changes over 5, 30 and 300 seconds in pips.
6. UTC onset hour and weekday, with fixed treatment indicators for a design-rank audit.

Every midpoint is the last recorded quote strictly before the requested boundary. The earliest mark must remain inside inherited support; no new age screen is introduced. Save quote ages/timestamps and reconcile preceding 300-second net pressure and five-second variation with the inherited states. Undefined d or missing J is retained in the origin ledger and prevents a complete first-alert benchmark where appropriate.

The hypothetical augmented model adds S alone to exactly this benchmark. C300 remains the eventual primary response, with squared prediction loss, inherited first-alert origin accounting, chronological matured-label training and fixed refit schedule. No C300, I300, future returns, forecast losses or future-availability columns are loaded for this gate. This contract freezes the question and measurements; estimation and inference are not implemented. The existing secondary horizon is not an alternative selected on these diagnostics.

This benchmark is a declared compact representation of ordinary dynamics, not every possible dynamic model. Comparing before-onset features alone and forecasting with all known first-alert information are different estimands. The present proposal is the latter and discloses that change from stage09b.

## Numerical, coverage and distinctness checks

- Reconstruct sequence windows from the registered sparse one-second source and support. An independent verifier will reconstruct against the inherited materialized structural clock, preserving exact boundaries.
- Save all sixty Q_k values per available origin, sign counts, A/E/S, zero/degenerate flags, recorded activity and local aggregation diagnostics.
- Compare ordinary five-term sums with math.fsum; record sign disagreements and near-cancellation counts using an arithmetic-error bound as a diagnostic only. No numerical/economic epsilon will replace signs or select observations. The local bound does not cover errors already inherited in stored one-second OFI or missing vendor records.
- Reconcile aggregate pressure with inherited values under a declared scale-aware numerical tolerance, retaining measured discrepancies. Record exact strict-left price source times and agreement with existing variation and pre-onset quotes.
- Report coverage/attrition reasons and feature distributions overall and by month, without conditioning on future responses. Do not silently discard origins unavailable to the extension.
- Report predictor-only correlations, degeneracy and standardized design-matrix rank/conditioning, including the incremental rank from S. No response regression, coefficient, p-value, loss comparison or hypothesis test is computed. Full rank only excludes exact linear redundancy in this matrix; it does not establish independent information, sufficient common support, forecast skill or economic novelty.

Meaningful synthetic cases cover onset/endpoint exclusion, event-relative bin phase, support boundaries, invalid transitions, supported empty bins, zero/constant signs, sign reflection, permutation-composition algebra, equal totals with different ordering, floating cancellation and corrupted joins. Independently verify serialized outputs and source/protocol hashes.

## Acceptance and limits

Measurement passes only if identities/timing/support and serialized reconstruction agree, discrepancies are within the frozen numerical tolerances, every origin is accounted for and measurement is reproducible. A zero/constant or rank-redundant S fails the proposed incremental design. Do not tune a coverage threshold after observing attrition; report whether any further estimation population would differ and whether the missingness is explainable under the existing convention.

Passing means a reproducible recorded-quote measure is available. It does not certify true-market ordering, complete size updates, an informative-trading label or sufficient external validity. Literature/novelty remains a separate gate. No automatic response fitting follows, even if measurement passes. If novelty is still unsupported, the final recommendation must say so plainly.

Deliverables: this frozen protocol and JSON specification, literature comparison, measurement feature/sequence tables and descriptive statistics, focused tests, independent source verification, a concise decision report, central review/statistics registration, and an updated living methodology PDF. Earlier results, notebooks, source code and all existing active pointers remain preserved.

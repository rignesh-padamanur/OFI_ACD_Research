> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step 10b: one exploratory sequence forecast comparison

3 October 2026. Prospective execution and interpretation contract. The user has now requested continuing the contribution work after completed stage 10a. This authorizes one bounded test of the frozen candidate; it does not establish novelty or authorize a search for a favorable feature. Ignore manuscript.tex. Preserve all preceding scientific results, protocols, notebooks and active pointers.

## Why this test, and what it can answer

At a recorded EUR/USD first alert, does adding the single preceding sequence statistic S improve chronological five-minute continuation forecasts over the exact stage10a benchmark of aggregate states, recent dynamics and known alert information?

This asks whether one compact recorded-history representation helps one declared forecasting algorithm. Aggregate net/gross pressure need not determine serial arrangement. Recent pressure and price changes are included to confront a simple recency/repricing explanation. Those recent summaries themselves contain ordering information; the baseline is not wholly order-free. A favorable result would be an algorithm-specific empirical candidate, not proof that every aggregation loses information or that this relation is previously undocumented.

The prior gate correctly prevented automatic estimation after measurement. It must not be interpreted as requiring a demonstrated novel finding before any test of a credible question. The present new authorization and explicit limited question permit this single exploratory comparison. Substantial economic novelty remains an unresolved reporting criterion. Neither the prior unsuccessful interaction forecast nor the stopped spread-history extension is removed from the record.

The original substantive interest in conditions preceding price persistence is retained, but the strongest original information-versus-inventory and permanent-impact claims remain unidentified. S uses only observations strictly before onset s. Direction d and initial movement J are observed at first alert t0=s+1 second and are included in both equations, as frozen in10a. This is prediction conditional on an alert, not advance warning of event arrival.

## Fixed data, features and response

Use verified stage10a features and the stage09a origins, labels, folds and membership, binding their manifest and output hashes. Reuse the existing detector, support policy, all recorded origins, reference scales and 59 expanding refit windows. Combined10a eligibility must exactly equal inherited09a eligibility; fail rather than change the sample if it does not. Keep all ineligible origins with explicit abstentions and all eligible origins even when future labels are missing.

The response is the saved direction-aligned C300 in pips. J is known at issuance, so the implied total forecast is J+C-hat; its squared error equals C forecast squared error. Zero/negative J remain. No alternative horizon, persistence label, probability score, event threshold, sampling interval, quote-age cap, winsorization, response screen or sequence interaction is introduced.

The baseline has an intercept and these exact19 numeric inputs:

1. direction; initial_pips; aligned_pressure; pressure_magnitude;
2. gross_pressure_scaled; variation_standardized; pre_spread_pips;
3. log_pre_displayed_size; log1p_prior_quote_updates; aligned_sign_balance;
4. nonzero_fraction; composition_reference;
5. aligned_recent_5s_pressure; aligned_recent_30s_pressure;
6. recent_5s_gross_scaled; recent_30s_gross_scaled;
7. aligned_prior_5s_price_pips; aligned_prior_30s_price_pips; aligned_prior_300s_price_pips.

Add the same fixed UTC hour and weekday indicators as09b: hour00 and Monday references; hour01-23 and Tuesday-Friday/Sunday indicators. The sequence equation adds only saved sequence_z=S. S=A-E retains the exact frozen10a definition. No cross-sectional standardization from10a diagnostics enters fitting. State scaling remains the inherited pre-deployment scaling.

Report four predeclared methods: zero continuation; expanding matured-label mean; baseline OLS; baseline OLS plus S. The sole primary contrast is sequence minus baseline squared loss. Zero and mean comparisons are context, not alternative primary tests or a search for a winner. The fixed schedule entails118 OLS fits and59 mean calibrations, not118 candidate models.

## Timing, numerical identification and serialization

At cutoff T use only eligible prior origins with available C300 whose target time t0+300 seconds is at or before T. Check membership against source identities, times, eligibility and selected training labels. Preserve every forecast origin in the inherited deployment interval. Do not condition forecast issuance on its eventual response availability. A target revealed after a cutoff must not change that cutoff's forecast. The idealized replay assumes zero processing latency and does not certify executable trading.

Use09b's numerical rule: unpenalized OLS via training-only column-equilibrated SVD; require finite values, n greater than the active column count, and full rank under machine epsilon times max(n,p) times the largest singular value. Do not drop continuous columns, regularize or clip forecasts after seeing a failure. Any remaining numerical failure stops the attempt with evidence retained. Verify fitting independently using QR, not a call to the producer's fitting or design functions.

Calendar indicators absent in training are omitted only from that training equation. Require hour00 and Monday references. If a prediction hour or weekday has never occurred in training, both fitted methods issue the same matured training-mean fallback. Retain those origins, report the count, and assign them zero paired difference. Saturday is outside inherited support; it is not silently assigned a reference coefficient. Save training support, omitted calendar terms, coefficients, scales, ranks and conditioning.

Freeze protocol, producer, independent verifier and tests before empirical fitting. Save and hash forecasts, coefficients and fit diagnostics before attaching evaluation losses. Independently reconstruct forecasts before evaluation. Evaluate saved forecasts, independently reconstruct event and calendar loss accounting, reload all saved outputs, and register only a verified run. Preserve any failed attempt instead of overwriting its evidence. Do not alter old active pointers or the original Notebook5. Save a separate executed stage10b result notebook linked to the repaired forecasting workflow.

## Loss accounting and limits on inference

Use the same eligible origins with available C300 for all four methods. Event-weighted MSE is total squared error divided by the event count; report RMSE in pips. Missing outcomes contribute neither fabricated zero errors nor imputed labels. Report eligibility, issuance, unavailable outcomes and fallback counts separately.

For baseline prediction b, augmented prediction f and realized y, compute delta=(y-f)^2-(y-b)^2 using the stable identity (b-f)(2y-f-b). Reconcile it to direct squared losses within numerical roundoff. Negative mean delta favors the sequence equation. Report its absolute value, relative MSE reduction where the comparator MSE is positive, and the corresponding RMSE difference. A zero denominator remains undefined.

Preserve09b's full UTC origin-date calendar, including no-event dates with zero counts/sums. Save every fold and calendar quarter; do not select favorable periods. Store daily D=sum(delta), n_scored, and psi=D-n_scored*pooled_mean_delta. Report fixed lags0-40 of daily cross-products/autocorrelation and the largest absolute-D and squared-psi shares with their denominators. These are concentration/dependence diagnostics, not independent observations or validated inferential coverage. The primary statistic remains event-weighted.

Do not report coefficient p-values, normal confidence bands, a significance-selected winner, profitability or population forecast superiority. Nested recursively fitted comparisons require an inference design justified for their dependence; this task does not supply one. Prior outcomes and failures have already been inspected, so a new freeze does not create an untouched holdout. Literature differences or a lower realized MSE cannot remove that limitation.

## Counterexamples that bind the interpretation

- If X is symmetric, S=X^2-E[X^2], and C=X^2+noise, a linear X benchmark can improve with S even though S contains no information beyond X. This hypothetical example shows why rank, low pairwise correlations and a positive forecast increment do not prove general conditional information loss. No new nonlinear estimator is fitted to overcome this example in this task.
- Because E is already a baseline column and S=A-E, the spans of [baseline,S] and [baseline,A] are identical. Verify the coefficient reparameterization as arithmetic, without fitting an alternate candidate. Centering aids interpretation; it cannot create a new OLS information set.
- All-one-sign and all-zero sequences have S=0; S measures arrangement conditional on composition, not total directional persistence. An additive S coefficient can also average away opposing effects. A failed comparison does not rule out every history-dependent process.
- Conditioning on an extreme joint detector can associate otherwise unrelated variables. Forecast evidence applies to this selected recorded-alert population; it does not identify a causal sequence effect or the wider market.
- Public repricing, adaptive liquidity, shared quote error and omitted size-only updates can mimic sequence effects. Detection-price error enters J and C with opposite signs. Neither a coefficient nor retained endpoint identifies private information or inventory.
- A gain over a fitted baseline that still loses to zero continuation is weak evidence of useful prediction. A tiny gain or one concentrated in a few dates cannot simply be described as a material stable contribution. Do not choose a numerical materiality threshold after seeing the result.

## Prospective decision rule and deliverable

This one comparison ends regardless of its sign. If augmented realized MSE is not lower than baseline, stop S as a supported headline forecasting extension. If it is lower, report the exact magnitude and simple-reference/context diagnostics as exploratory evidence only. Do not automatically authorize another model, S interaction, threshold, horizon or targeted date exclusion. Favorable magnitude would justify a separate economic validation discussion, not a claim that novelty, mechanism or JFE readiness is settled.

The contribution work is complete for this bounded task when the source-bound result, independent verification, public research record, central statistics entry, separate executable result notebook and updated methodology guide are saved. The conclusion may be that the current data and this candidate do not support the desired contribution; it must not be rewritten to guarantee discovery.

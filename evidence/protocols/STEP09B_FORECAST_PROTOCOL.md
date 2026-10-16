> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Step 09b: one chronological persistence-forecast comparison

Date: 3 October 2026. Frozen before any response fit in this stage. Parent: step09a_prediction_ledger_20261003. This contract implements the next task authorized by the user: test the predictive purpose of Notebook 5 using the existing measurements, without another detector, target or model-family search.

## Research question and scope

At the first recorded event alert, does allowing the effect of preceding quote-pressure magnitude to vary with preceding price variation improve prediction beyond an additive preceding-state forecast? The target is C300 in pips. Known J supplies the implied total-displacement forecast I-hat = J + C-hat; its squared prediction error is identical to that for C. Zero and negative J remain. Event direction orients the outcome but is excluded from the preceding predictor vector.

The question is conditional on detection, not advance prediction of whether an event will occur. It concerns recorded quotes in the existing EUR/USD cache, not identified information or inventory shocks. The inspected cache supports an exploratory chronological replay, not an untouched confirmatory test. Sixty-second labels remain saved secondary descriptions; this stage does not fit or select a second horizon.

## Fixed forecasting methods

Four methods are reported on the same prediction origins and eventual evaluable IDs:

1. **Zero:** predict C300=0, or I300=J. No response parameter is estimated.
2. **Training mean:** predict the expanding mean of matured, available training C300 labels at that cutoff.
3. **Additive:** ordinary least squares with an intercept; signed prior pressure Q/cQ, its magnitude M=|Q|/cQ, standardized prior variation v, preceding spread and log preceding displayed size; and the fixed calendar coding below.
4. **Interaction:** exactly the additive equation plus the single product vM. Its raw squared-loss difference relative to the additive method is the primary comparison. Comparisons to the training mean and zero are context, not alternative targets from which a winner is selected.

No outcome-based screening, response trimming, sign restriction, clipping, winsorization, ridge penalty, hyperparameter search, classifier, persistence probability, new nonlinear transformation or additional history variable is introduced. The reference scales remain those frozen before 22 April 2024. A vM coefficient is not by itself evidence of structural regimes. Correlated states could allow vM to approximate omitted ordinary curvature; any gain is relative to this declared additive benchmark.

There are four fixed forecasting methods, not a competition among newly selected model families. Across 59 scheduled cutoffs, the two equations require 118 OLS refits and the mean benchmark requires 59 mean calibrations; the zero reference has no fitted parameter. The statistics distinguish these repeated fitting operations from the number of declared methods.

## Calendar identification and numerical rule

Use onset UTC hour indicators 01-23 with hour 00 as reference, and weekday indicators Tuesday-Friday and Sunday with Monday as reference. Saturday is outside the inherited supported calendar; it is not silently assigned the reference effect. Inputs must agree with the stored onset clock. The hour and weekday main effects are additive: a previously unseen combination of two individually observed levels is not an unseen level.

An input-only audit before fitting found no observations at hours 13/14 in the first training sample. Therefore use training-only active calendar indicators: omit indicators that are identically zero in that training sample and save their identities. Require the hour 00 and Monday references in training. Do not remove continuous columns. For a prediction origin whose hour OR weekday has never occurred in that training sample, both state methods use exactly the same training-mean fallback, with an explicit reason. These origins are retained in all performance denominators; their primary state-method loss difference is zero. Their count and coverage must be reported. This is an explicit forecasting algorithm, not identification of an absent category's coefficient.

Fit by an equilibrated singular-value decomposition, scaling each column using its training values only and mapping coefficients back to the original units. Require finite entries, n greater than the active column count, and full rank under machine-epsilon times max(n,p) times the largest singular value of the scaled design. Save the column scales, rank, singular-value diagnostics, training size, calendar support and coefficients. No covariance inversion is needed. Exact fits and constant targets are legitimate forecasting cases; they do not establish inference. A remaining rank failure, missing reference, unsupported weekday or nonfinite prediction stops the run with a diagnostic; there is no response-driven repair or numerical clipping. Independent verification reconstructs least squares through QR.

## Information boundaries and output order

Keep the parent's 59 refit windows and existing origin eligibility. Training at T contains only the parent's eligible earlier origins whose C300 label is available and whose target time t0+300 seconds is at or before T. Recheck these conditions against the labels; do not trust membership alone. Positive processing delay remains outside the inherited idealized zero-latency replay. Training histories expand; prediction origins occupy [T, deployment_end).

Every later recorded origin appears for every method. Origin-ineligible records contain an explicit abstention and no numeric prediction. Every eligible origin gets a forecast even if its future outcome is unavailable. Join fields, J, direction and support diagnostics must never enter the numerical model. Only the explicit stage09a predictor fields are accepted, with the interaction column used in its one declared equation. The producer reads response values only for the specified matured training IDs. It must not inspect evaluation losses when constructing coefficients or predictions.

Save, hash and independently verify the forecast table and fitted parameters before calling the separate evaluation function. Forecast records contain no realized future C/I, endpoint availability, label status or future-age feature. J is permitted only to translate the C forecast to an I forecast. Missing future outcomes stay missing; they are never mean-filled or replaced by zero loss.

## Performance and calendar diagnostics

The primary empirical quantity is event-weighted mean squared error (MSE), with RMSE reported in pips. For every method use the same eligible origin/available C300 pairs. Pool sums of squared errors across all events, divided by the total event count; do not average daily or refit-window MSEs with equal weights. Report event coverage and all forecast/fallback counts. The missing-label subset remains a limitation on what can be evaluated.

Let a_i and f_i be additive and interaction predictions, and y_i the observed C300. The primary event loss difference is delta_i=(y_i-f_i)^2-(y_i-a_i)^2. Negative values favor the interaction. Compute the equivalent expression (a_i-f_i)(2y_i-f_i-a_i) for stable subtraction and independently reconcile it to squared losses with a scale-aware roundoff bound. Report raw mean loss differences and relative reduction 1-MSE_interaction/MSE_benchmark when the benchmark MSE is positive; a zero denominator remains undefined. No gain is presumed in advance.

Aggregate by UTC **forecast-origin date**, retaining every date from 6 May 2024 through 31 July 2026, including dates without evaluable events. Store each method's daily count and summed loss; primary daily D_t=sum(delta_i); and centered event-weighted contributions psi_t=D_t-n_t*mean(delta). Empty dates contribute zero counts and sums, not invented observations or proof of complete observation. Report fixed 40-lag cross-products of psi and their normalized autocorrelations where defined, fold and quarter loss summaries, the largest absolute daily D share (denominator sum_t|D_t|), and the largest psi-squared share (denominator sum_t psi_t^2). Undefined zero-denominator diagnostics remain explicitly undefined. These are descriptions of temporal dependence and concentration, not independent-cluster counts or percentages of explained variance.

Do not attach conventional normal p-values, 95% confidence bands, or a claim of validated forecast superiority. Recursive estimation and nesting complicate loss-comparison null distributions; ordinary raw-MSPE normal inference is not automatically justified. Clark and West (2007), *Journal of Econometrics* 138:291-311, explains the extra parameter-estimation noise in nested comparisons ([publisher](https://doi.org/10.1016/j.jeconom.2006.05.023); [original Federal Reserve paper](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf)). No adjusted test is introduced to obtain a preferred conclusion. Subsequent inferential work, if justified, needs assumptions matched to this irregular event sample and inspected-data history.

## Verification, stopping and reporting

Before fitting, preserve the current master, notebook and PDF release. Reverify parent tables and lineage. Freeze this protocol, producer, independent verifier, runner and synthetic tests with hashes. Tests must cover label-maturity boundaries, changes to future labels leaving earlier forecasts unchanged, missing future outcomes retaining forecasts, unseen marginal categories versus unseen combinations, zero/negative J, constant targets, rank failures, exact shared fallback, unequal calendar event counts, zero-loss denominators, tampered outputs and serialized round trips.

Then fit only these declared methods, save predictions before evaluation, independently reconstruct fits and predictions, attach evaluation outcomes, independently verify losses/calendar sums, reopen saved artifacts, and only then register the result. Preserve a failed full attempt and its exact sources if a check rejects it. No earlier detector, measurement or model result is overwritten. Save a public decision record, updated central statistics, executed repaired Notebook 5, accessible methodology PDF and actual limitations.

If the interaction fails to lower realized MSE, report that this particular algorithm does not support the proposed predictive contribution. This does not prove all predictability is absent or identify why it failed. If it lowers MSE, report the observed magnitude, coverage and temporal concentration without claiming statistical significance, structural mechanism, profitability or established novelty. Both paths end this bounded comparison; neither authorizes a replacement-candidate search.

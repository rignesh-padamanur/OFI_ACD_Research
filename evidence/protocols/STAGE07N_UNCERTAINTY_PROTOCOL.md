> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage07n: uncertainty for the fixed recorded-price means

Declared 2 October 2026 before new empirical interval calculation. This completes
the bounded uncertainty assessment in the narrowed scope contract. It adds no
regression, detector, selection threshold, endpoint-category test or subgroup model.
Ignore manuscript.tex. Notebook3/4 response summaries feed Notebook5 reporting.

## Target and unchanged observations

Use the verified stage07m overall J/C/I observations at 300 seconds (primary) and
60 seconds (secondary), for both the horizon-specific joint sample and their
common-ID intersection. No outcome trimming or additional missingness filter.
J is the detection-second displacement, C the subsequent movement, and I=J+C,
all in recorded pips aligned to the inherited detection-time quote direction.

The finite saved-sample means are known. Intervals instead approximate uncertainty
about the event-weighted means of a hypothetical repeated selected-event process:
mu_k=E[S_d,k]/E[n_d], under suitable stability, moments, weak dependence and a
central limit approximation. They are not uncertainty conditional on every saved
outcome. The common-horizon target conditions on availability at both horizons.

## Calculation fixed before inspecting new results

For each of the four cohort/horizon groups, N=sum_d n_d, mu_hat=sum_d S_d/N,
and daily score G_d,k=S_d,k-n_d*mu_hat_k. Use all 817 UTC dates in
[2024-05-06,2026-08-01). Empty dates have zero selected-event counts/sums/scores;
they are not imputed zero returns or evidence of complete feed coverage.
Use the ratio-score centering by n_d, not equal weights on occupied daily means.

Stack the twelve target coordinates in order: cohort horizon_joint then
common_horizons_joint; within cohort 60 then300; within horizon J,C,I. Let
B=diag(1/N_k). Apply the already verified calendar_hac implementation to (G,B)
with all twelve coordinates selected. Each scalar NW1994 pilot has
m=floor(4*(T/100)^(2/9)), gamma_j=sum G_d G_(d-j)/T,
s0=gamma0+2 sum gamma_j, s1=2 sum j*gamma_j,
b=1.1447*((s1/s0)^2*T)^(1/3). One common L=floor(max b) across these fixed
coordinates and one longer sensitivity max(1,2L). This extends the project's
coordinate-maximum rule, not its claim to multivariate optimality. A shared lag
ensures identical targets receive identical uncertainty. It is not guaranteed
to be conservative. Preserve negative pilot s0 flags; no lag cap or fallback.

For each lag, M=G'G+sum_(j=1..L)(1-j/(L+1))*(G_j'G_0+G_0'G_j),
V=B*M*B. Save the full 12x12 covariance, including cross-component and
cross-horizon terms. No extra N/T/degree-of-freedom adjustment, daily demeaning,
prewhitening, covariance clipping or variance repair. Report each coordinate's
mean +/- Phi^(-1)(.975)*sqrt(V_kk) as a pointwise nominal95% normal interval.
No new p-values or joint/horizon-difference tests. Interval overlap is not a test.

Singularity is expected from I=J+C and repeated identical targets. Do not invert
this covariance or require positive definiteness. Require numerical positive
semidefiniteness, positive finite marginal variances, accounting identities and
equal covariances for identical score columns. A degenerate pilot or invalid
variance stops this run explicitly; it is not silently omitted or replaced.

Save full-calendar daily influences G_d,k/N_k and their squared shares, plus
normalized cross-products for lags0..min(T-1,max(70,longerL)), the inherited
diagnostic range. Report maximum squared daily shares and maximum absolute
products beyond each requested lag. These are descriptive diagnostics, not
formal stationarity tests, variance shares, independent-day counts, or a cutoff
that approves/rejects inference. No lag re-selection after these diagnostics.

## Existing regression assessment

Preserve all four stage07h fits and their saved automatic/longer calculations.
Recheck their saved covariance arithmetic from daily scores/bread and their
concentration summaries from saved daily influences; report the five original
state coefficients and two existing branch averages with their intervals and
diagnostics. No regression refit or revised bandwidth. The branch averages
condition on the saved observed-state contrast weights; population-average
weight uncertainty is not supplied by those intervals.

## Counterexamples and limits

- Unequal event counts make an equal-day average target different from this mean.
- Empty calendar dates cannot be compressed: doing so changes lag distances.
- I=J+C requires its covariance term; adding variances alone is wrong.
- Identical J at both horizons must not acquire different uncertainty because
  a different C coordinate chooses a different local bandwidth.
- A maximum scalar pilot can miss long or cancelling lag dependence. Including
  I captures some symmetric J/C cross-lag dependence, not all dependence.
- Longer lags need not widen intervals. Normal approximations can be unreliable
  under dominant dates even when arithmetic/source reconstruction is correct.
- A finite-memory past-only detector can in principle transmit its estimation
  variation into the observed score process. Adequacy of this HAC approximation
  for the adaptive process is unestablished; there is no inference for a latent
  ideal detector or externally complete market. There is no full-pipeline
  resampling/coverage study or untouched holdout.
- Positive average I and small C do not demonstrate every event persists,
  infinite permanence, causality or private information versus inventory.

## Verification and downstream consequence

Before registration: freeze source/protocol hashes, run the full synthetic suite,
independently reconstruct grouping from stage07m event membership, independently
recalculate the pilot and Bartlett covariance, reload saved tables and independently
verify every new mean-output table and statistic. Separately recheck the already
verified regression covariances/concentration from frozen source outputs and check
the serialized assessment/interval copy against those source-bound results.
This step does not independently re-estimate the regressions. Preserve all earlier
source artifacts and active pointers.
Update master statistics, repair log/checkpoint and living methodology PDF.
This closes the planned arithmetic/assessment task, not the unresolved economic
identification or interval-coverage assumptions. Reporting/reproducibility
closeout follows; do not initiate another model/sensitivity search.

References: Newey and West (1994), [publisher record](https://doi.org/10.2307/2297912)
and [author paper](https://users.ssc.wisc.edu/~kdwest/publications/1990/Automatic%20Lag%20Selection%20in%20Covariance%20Matrix%20Estimation.pdf);
the maintained [sandwich implementation](https://raw.githubusercontent.com/cran/sandwich/master/R/vcovHAC.R)
fixes this project's pilot and rounding conventions. The maximum over coordinates
is a project choice. [Cameron and Miller (2015)](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf)
supports the distinction between sandwich arithmetic and its asymptotic conditions;
independent-day clustering is not assumed here.

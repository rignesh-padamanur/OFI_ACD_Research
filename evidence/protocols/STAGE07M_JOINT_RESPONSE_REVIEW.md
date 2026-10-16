> Historical protocol/review preserved for provenance. Dated conclusions reflect the record at that time. Navigation links were made portable for this release; paths marked “full research archive” are not bundled. Original and release hashes are in the evidence manifest.

# Stage07m: joint price-response interpretation

Run `step07m_joint_response_20261002`. This is descriptive accounting across Notebooks3/4 feeding Notebook5, using the original main continuation cohorts. 932 tests passed; independent reconstruction verified every accounting table. Zero new regression models were fitted.

J is price movement during the detection second, C is subsequent movement, and I=J+C is total movement from just before onset. All are recorded pips aligned to the inherited quote-imbalance direction. J is not necessarily positive and is not the whole eventual episode impact.

| Horizon | Original C sample | Joint J/C/I sample | Missing joint | Mean J | Mean C | Mean I |
|---|---:|---:|---:|---:|---:|---:|
| 300s | 127,013 | 127,013 | 0 | +0.380258 | +0.014752 | +0.395010 |
| 60s | 127,052 | 127,052 | 0 | +0.380295 | -0.003153 | +0.377142 |

These means use identical observations within each row. Medians need not add. Counts, medians and quartiles by positive/zero/negative J are saved, without dividing by J or variation.

At300s, initial J is positive for 112,304 events, zero for 5,442, and negative for 9,267. Endpoint descriptions below use all 127,013 jointly observed events as the displayed denominator.

| Five-minute endpoint description | Events | Share of joint sample |
|---|---:|---:|
| Same as detection price (nonzero J) | 2,099 | 1.65% |
| Farther in initial direction | 58,637 | 46.17% |
| Partial retracement | 10,816 | 8.52% |
| At pre-onset price | 2,110 | 1.66% |
| Beyond pre-onset in opposite direction | 47,909 | 37.72% |
| No initial move; positive aligned endpoint | 2,671 | 2.10% |
| No initial move; negative aligned endpoint | 2,617 | 2.06% |
| No initial move; unchanged endpoint | 154 | 0.12% |

Negative C alone does not imply retracement: J=-1,C=-0.5 gives I=-1.5, extending the initial move. J=1,C=-0.2 gives partial retracement; J=0 has no initial displacement to retain. Exact observed signs and strict endpoint ordering define the descriptions. Tiny nonzero moves remain mathematically nonzero and do not establish economically material displacement.

The common60/300 cohort has 127,013 events; 39 joint events are60-only and 0 are300-only. Both horizon means are also saved on the intersection, together with all64 endpoint-transition cells and explicit total/origin denominators. This cohort conditions on later availability; the transition table is descriptive, not a forecast or regime model.

All 817 calendar dates remain in each cohort/horizon daily table. Empty dates have zero counts/sums and missing means. Saved event counts and sums prepare event-weighted uncertainty; equally averaging daily means would change the target.

Source accounting reconciles the registered anchors/paths and original stage07h response keys. It rechecks direct saved price arithmetic, not raw ticks or an independent external feed. All29 previous active pointers and the existing fitted equations remain preserved.

Interpretation boundary: endpoint position cannot show monotonic decay, first crossing, permanence, or private information versus inventory. A positive mean total displacement can coexist with many opposite-signed individual paths. J-sign conditioning is on a realized detection-time outcome. No confidence intervals, significance tests, causal identification or new state association is claimed in this step. Joint uncertainty is still pending and must respect the singular J/C/I accounting identity.

Next: Assess dependence and joint uncertainty for the fixed response summaries and existing state equation; then complete reporting/reproducibility closeout. No detector, model or threshold selection. Causal identification and paper readiness are not implied by completing these steps.

The pipeline is not yet paper-ready.

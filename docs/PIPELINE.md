# Scientific code and stage dependencies

The `pipeline/` directory preserves the relative layout of the selected scientific code. Names beginning with `repair_` are current corrected implementations, not automatically obsolete files. Most readers should run the root `reproduce.py` commands; the following runners expose the historical construction for inspection.

| Stage | Main entry points under `pipeline/` | Role |
|---|---|---|
| Observations and support | `prepare_master_dataset.py`, `repair_acquisition.py`, `repair_observation_eligibility.py` | Acquisition, admissibility and supported intervals |
| Quote measurements | `repair_quote_imbalance.py`, `repair_signal_clock.py`, `repair_seasonal_calibration.py`, `repair_duration_observation.py` | Quote-size imbalance, clock and historical timing normalization |
| Events and panel | `repair_event_audit.py`, `repair_chronological_transition.py`, `repair_chronological_panel.py`, `repair_panel_state_precision.py` | Event construction, deployment chronology and corrected panel |
| Responses | `repair_panel_measurement.py`, `repair_panel_responses.py`, `repair_panel_analysis.py`, `repair_joint_response.py`, `repair_joint_uncertainty.py` | Measurements, joint samples and response uncertainty |
| Price forecasts | `repair_prediction_ledger.py`, `repair_persistence_forecast.py`, `repair_sequence_measurement.py`, `repair_sequence_forecast.py` | Origin accounting, matured training labels and specified comparisons |
| Supporting application | `run_detector_contribution.py`, `run_timing_forecast.py` and their verifier scripts | Existing detector checks and separate spread forecast |

These are historical registration-aware builders. Some validate specific parent hashes or an earlier master-ledger state. Running every file in lexical order against a modern directory is not a supported rebuild procedure. Full replay uses the appropriate dated source snapshot and state, as explained in [the archive guide](ARCHIVE.md).

The source selection includes imported scientific modules, independent verifiers, relevant tests, and several otherwise disabled legacy scripts that retained tests load to check their rejection behavior. The release includes the legacy guard marker at `pipeline/reports/PIPELINE_REPAIR_LOG.md`; removing it could accidentally permit superseded APIs. No notebook is a producer dependency of this selected script pipeline.

The code edits made for publication concern compiler portability, release tooling and comment cleanup. They do not change event thresholds, predictors, fitted models, horizons or saved empirical results. Original-versus-release hashes and transformation reasons are recorded in [the source mapping](../provenance/source_mapping.json).

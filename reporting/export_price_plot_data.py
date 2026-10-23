"""Export compact, lossless inputs for the paper's two main empirical figures.

This optional archive-facing command requires the intact historical data archive.
It does not refit a model or alter event membership. The default plot command
uses the already-exported compact inputs and does not need the full archive.

Example:
    python reporting/export_price_plot_data.py --archive-root /path/to/research-archive

The common-cohort and quarter checks preserve reports/plot_section4_results.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

if not __debug__:
    raise RuntimeError("Verification requires assertions: run Python without -O or -OO.")

N = 127_013
VALUE_COLUMNS = ["initial_pips", "continuation_pips", "total_pips"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ROOT = None
PATHS = []
SOURCES = {}


def read_responses():
    assert len(PATHS) == 60
    membership = pd.read_parquet(SOURCES["membership"], columns=[
        "episode_id", "horizon_sec", "joint_included", *VALUE_COLUMNS])
    cohort = membership.loc[(membership.horizon_sec == 300) & membership.joint_included]
    cohort = cohort.sort_values("episode_id").set_index("episode_id")
    assert len(cohort) == N and cohort.index.is_unique
    ids = cohort.index
    endpoints = {}
    for horizon in (60, 300):
        endpoint = membership.loc[membership.horizon_sec == horizon].set_index("episode_id")
        assert endpoint.index.is_unique
        endpoints[horizon] = endpoint.loc[ids, VALUE_COLUMNS].to_numpy()

    seen = np.zeros(N * 301, dtype=bool)
    sums = np.zeros((301, 3), dtype=float)
    counts = np.zeros(301, dtype=np.int64)
    for number, path in enumerate(PATHS, 1):
        frame = pd.read_parquet(path, columns=[
            "episode_id", "horizon_sec", "triplet_quote_available", *VALUE_COLUMNS])
        frame = frame.loc[frame.episode_id.isin(ids)]
        positions = ids.get_indexer(frame.episode_id)
        horizons = frame.horizon_sec.to_numpy(dtype=np.int64)
        values = frame[VALUE_COLUMNS].to_numpy()
        assert (positions >= 0).all() and ((horizons >= 0) & (horizons <= 300)).all()
        assert frame.triplet_quote_available.all() and np.isfinite(values).all()
        keys = positions * 301 + horizons
        assert len(np.unique(keys)) == len(keys) and not seen[keys].any()
        seen[keys] = True
        np.testing.assert_allclose(values[:, 2], values[:, 0] + values[:, 1], atol=1e-9, rtol=0)
        np.testing.assert_allclose(values[:, 0], cohort.initial_pips.to_numpy()[positions], atol=1e-12, rtol=0)
        counts += np.bincount(horizons, minlength=301)
        for column in range(3):
            sums[:, column] += np.bincount(horizons, weights=values[:, column], minlength=301)
        for horizon in (60, 300):
            selected = horizons == horizon
            np.testing.assert_allclose(values[selected], endpoints[horizon][positions[selected]], atol=0, rtol=0)
        if number % 15 == 0:
            print(f"Checked {number}/60 path fragments", flush=True)
    assert seen.all() and (counts == N).all()
    means = sums / counts[:, None]
    np.testing.assert_allclose(means[:, 2], means[:, 0] + means[:, 1], atol=1e-11, rtol=0)
    np.testing.assert_allclose(means[0, 1], 0, atol=0, rtol=0)
    np.testing.assert_allclose(means[0, 2], cohort.initial_pips.mean(), atol=1e-11, rtol=0)

    summary = pd.read_parquet(SOURCES["summary"])
    summary = summary.loc[(summary.horizon_sec == 300) & (summary.cohort == "horizon_joint")
                          & (summary.initial_group == "all")].set_index("component").loc[["J", "C", "I"]]
    assert (summary.n == N).all()
    np.testing.assert_allclose(means[300], summary["mean"], atol=1e-11, rtol=0)
    np.testing.assert_allclose(cohort[VALUE_COLUMNS].quantile([.25, .5, .75]).to_numpy().T,
                               summary[["p25", "median", "p75"]], atol=1e-11, rtol=0)
    return cohort, means


def read_quarters():
    result = []
    expected = pd.period_range("2024Q2", "2026Q3", freq="Q").astype(str).tolist()
    for source, extension, benchmark in [("interaction", "interaction", "additive"),
                                          ("sequence", "sequence", "baseline")]:
        q = pd.read_parquet(SOURCES[source]).sort_values("quarter").reset_index(drop=True)
        assert q.quarter.is_unique and q.quarter.tolist() == expected
        assert q.n_scored.sum() == N and (q.n_scored > 0).all()
        column = f"mean_delta_{extension}_minus_{benchmark}"
        np.testing.assert_allclose(q[column], q[f"mse_{extension}"] - q[f"mse_{benchmark}"], atol=1e-12, rtol=0)
        pooled = np.average(q[column], weights=q.n_scored)
        np.testing.assert_allclose(pooled, (q[f"sse_{extension}"].sum() - q[f"sse_{benchmark}"].sum()) / N,
                                   atol=1e-12, rtol=0)
        result.append((q, column, pooled))
    np.testing.assert_array_equal(result[0][0].n_scored, result[1][0].n_scored)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/plot_inputs/price")
    args = parser.parse_args()
    global ROOT, PATHS, SOURCES
    ROOT = args.archive_root.resolve()
    PATHS = sorted((ROOT / "data/signals/step07e_panel_responses_20261001/paths").glob("cycle_*.parquet"))
    SOURCES = {
        "membership": ROOT / "data/signals/step07m_joint_response_20261002/membership.parquet",
        "summary": ROOT / "data/signals/step07m_joint_response_20261002/summary.parquet",
        "interaction": ROOT / "data/signals/step09b_persistence_forecast_20261003/quarter_losses.parquet",
        "sequence": ROOT / "data/signals/step10b_sequence_forecast_20261003/quarter_losses.parquet",
        **{f"path_{i:03d}": path for i, path in enumerate(PATHS)},
    }
    before = {key: sha256(path) for key, path in SOURCES.items()}
    cohort, means = read_responses()
    quarters = read_quarters()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    path_frame = pd.DataFrame(means, columns=VALUE_COLUMNS)
    path_frame.insert(0, "n_events", N)
    path_frame.insert(0, "horizon_sec", np.arange(301))
    path_frame.to_csv(out / "mean_response_path.csv", index=False, float_format="%.17g")
    np.savez_compressed(out / "continuation_distribution.npz",
                        episode_id=cohort.index.to_numpy(dtype=np.int64),
                        continuation_pips=cohort.continuation_pips.to_numpy())
    for key, (q, _, _) in zip(("interaction", "sequence"), quarters):
        q.to_csv(out / f"{key}_quarter_losses.csv", index=False, float_format="%.17g")
    roundtrip = pd.read_csv(out / "mean_response_path.csv", float_precision="round_trip")
    np.testing.assert_array_equal(roundtrip[VALUE_COLUMNS].to_numpy(), means)
    with np.load(out / "continuation_distribution.npz", allow_pickle=False) as distribution:
        np.testing.assert_array_equal(distribution["episode_id"], cohort.index.to_numpy())
        np.testing.assert_array_equal(distribution["continuation_pips"], cohort.continuation_pips.to_numpy())
    for key, (q, _, _) in zip(("interaction", "sequence"), quarters):
        pd.testing.assert_frame_equal(pd.read_csv(out / f"{key}_quarter_losses.csv", float_precision="round_trip"), q, check_exact=True)
    after = {key: sha256(path) for key, path in SOURCES.items()}
    assert before == after, "An archive source changed during export"
    filenames = ["mean_response_path.csv", "continuation_distribution.npz", "interaction_quarter_losses.csv", "sequence_quarter_losses.csv"]
    values = cohort.continuation_pips.to_numpy()
    receipt = {
        "schema_version": 1,
        "purpose": "Lossless compact inputs for historical main-paper figures; no model refitting or sample changes",
        "exporter": "reporting/export_price_plot_data.py",
        "exporter_sha256": sha256(Path(__file__)),
        "reference_script": "reports/plot_section4_results.py",
        "reference_script_sha256": sha256(ROOT / "reports/plot_section4_results.py"),
        "cohort": {"n_events": N, "selection": "horizon_sec == 300 and joint_included in membership.parquet",
                   "ordering": "episode_id ascending", "horizons": [0, 300], "n_horizons": 301,
                   "observations_across_horizons": N * 301, "path_fragments": len(PATHS),
                   "recorded_quote_convention": "strict-left; fresh quote at each second is not required"},
        "original_sources": {key: {"path": str(path.relative_to(ROOT)), "sha256": before[key], "bytes": path.stat().st_size}
                             for key, path in SOURCES.items()},
        "exports": {name: {"sha256": sha256(out/name), "bytes": (out/name).stat().st_size} for name in filenames},
        "checks": {"all_event_horizon_pairs_present_exactly_once": True,
                   "identities_J_plus_C_equals_I": True, "endpoints_equal_membership": True,
                   "means_and_quartiles_equal_archived_summary": True,
                   "all_exported_values_roundtrip_exactly": True, "original_source_hashes_unchanged": True,
                   "all_ten_calendar_quarters_retained": True,
                   "matched_quarter_counts_between_models": True,
                   "pooled_delta_equals_event_weighted_quarter_delta_and_total_SSE_delta": True},
        "numerics": {"mean_initial_pips": float(cohort.initial_pips.mean()),
                     "mean_path_I_0_60_300": means[[0, 60, 300], 2].tolist(),
                     "mean_path_I_min_max": [float(means[:,2].min()), float(means[:,2].max())],
                     "continuation_min_max": [float(values.min()), float(values.max())],
                     "continuation_quartiles": np.quantile(values, [.25,.5,.75]).tolist(),
                     "central_view_share": float(((values >= -10) & (values <= 10)).mean()),
                     "pooled_interaction_minus_additive": float(quarters[0][2]),
                     "pooled_sequence_minus_baseline": float(quarters[1][2])},
        "limits": ["These are exported observed data, not new estimates of sampling uncertainty.",
                   "ECDF plotting retains every observation and shows full support in an inset.",
                   "Pooled points use event weights; quarter comparisons are not independent trials."]
    }
    (out / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"events": N, "horizons": 301, "input_sources": len(SOURCES),
                      "exported_bytes": sum((out / name).stat().st_size for name in filenames),
                      "numerics": receipt["numerics"]}, indent=2))


if __name__ == "__main__":
    main()

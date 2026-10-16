"""Report the fixed spread-forecast comparison from archived sufficient statistics.

This script writes only two figures and a reporting receipt. It does not fit a
model, select observations, or change a scientific archive. Panel A identifies
where benchmark squared error is concentrated; panel B shows the incremental
MSE difference within those same, prior-training-defined spread categories.
Run from the release root with: python reporting/plot_spread_concentration.py.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Verification must run without Python optimization (-O/-OO or PYTHONOPTIMIZE).")


import argparse
import hashlib
import tempfile
import json
import os
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fx_spread_concentration_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "evidence/data/signals/step03_timing_forecast_20261008"
OUT = ROOT / "figures"
SOURCES = {name: RUN / name for name in (
    "patterns.parquet", "daily.parquet", "summary.json", "manifest.json",
    "pattern_cell_coverage.parquet", "verification.json", "feature_verification.json")}
BLUE, RUST, INK = "#1E5673", "#A66042", "#252525"


def hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in SOURCES.items()}


def close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol)


def verify():
    summary = json.loads(SOURCES["summary.json"].read_text())
    patterns = pd.read_parquet(SOURCES["patterns.parquet"])
    daily = pd.read_parquet(SOURCES["daily.parquet"])
    cells = pd.read_parquet(SOURCES["pattern_cell_coverage.parquet"])
    assert len(daily) == 761 and daily.date.is_unique
    assert daily.date.tolist() == list(pd.date_range("2024-07-01", "2026-07-31", tz="UTC"))
    n = int(daily.n.sum())
    assert n == summary["n"] == 19_042_840
    for source in ("verification.json", "feature_verification.json"):
        assert json.loads(SOURCES[source].read_text())["status"] == "passed"
    sums = daily[["sum_lb", "sum_lt", "sum_d", "sum_carry"]].sum()
    close(sums.sum_d, sums.sum_lb - sums.sum_lt, atol=1e-7)
    close(sums.sum_lb / n, summary["mse_b"])
    close(sums.sum_lt / n, summary["mse_t"])
    close(sums.sum_carry / n, summary["mse_current_spread_carry"])
    expected_groups = {
        "count_group": {"1", "2", "3"},
        "quarter": set(pd.period_range("2024Q3", "2026Q3", freq="Q").astype(str)),
        "spread_group": {"1", "2", "3", "4"},
        "timing_group": {"1", "2", "3", "4"},
        "utc_block": {"0", "1", "2", "3"},
    }
    expected_patterns = {(dimension, group) for dimension, groups in expected_groups.items() for group in groups}
    assert len(patterns) == len(expected_patterns)
    assert set(zip(patterns.dimension, patterns.group)) == expected_patterns
    for dimension, group in patterns.groupby("dimension"):
        assert int(group.n.sum()) == n, dimension
        close(group[sums.index].sum(), sums, atol=1e-7)
        close(group.mse_b, group.sum_lb / group.n)
        close(group.mse_t, group.sum_lt / group.n)
        close(group.relative_reduction, group.sum_d / group.sum_lb)
    # Recompute the declared daily ratio influences and Bartlett covariance,
    # independently of src/timing_forecast.py, in the paper's increase convention.
    excess = -daily.sum_d.to_numpy()
    delta, relative = excess.sum() / n, excess.sum() / sums.sum_lb
    influence = (excess - delta * daily.n.to_numpy()) / n
    ratio_influence = (excess - relative * daily.sum_lb.to_numpy()) / sums.sum_lb
    close(-delta, summary["delta_pips_squared"], atol=1e-15)
    close(-relative, summary["relative_mse_reduction"], atol=1e-15)
    assert [item["lag_days"] for item in summary["nominal_intervals"]] == [56, 28, 112]
    z = NormalDist().inv_cdf(.975)
    intervals = []
    for item in summary["nominal_intervals"]:
        lag = item["lag_days"]
        dist = np.abs(np.arange(len(daily))[:, None] - np.arange(len(daily))[None, :])
        kernel = np.maximum(1 - dist / (lag + 1), 0)
        se = float(np.sqrt(influence @ kernel @ influence))
        rse = float(np.sqrt(ratio_influence @ kernel @ ratio_influence))
        close(se, item["nominal_se"], atol=1e-15)
        close(rse, item["relative_nominal_se"], atol=1e-15)
        close(-delta, item["delta_pips_squared"], atol=1e-15)
        close(-relative, item["relative_mse_reduction"], atol=1e-15)
        close([-delta-z*se, -delta+z*se],
              [item["nominal_low"], item["nominal_high"]], atol=1e-15)
        close([-relative-z*rse, -relative+z*rse],
              [item["relative_nominal_low"], item["relative_nominal_high"]], atol=1e-15)
        intervals.append({"lag_days": lag, "relative_increase_pct": relative * 100,
            "nominal_95_low_pct": (relative-z*rse) * 100,
            "nominal_95_high_pct": (relative+z*rse) * 100})
    leave_out = (excess.sum() - excess) / (n - daily.n.to_numpy())
    close(leave_out.min(), -summary["leave_one_day_out_delta_max"])
    close(leave_out.max(), -summary["leave_one_day_out_delta_min"])
    top = daily.iloc[np.argsort(-np.abs(daily.sum_d.to_numpy()), kind="stable")[:10]]
    assert top.date.dt.strftime("%Y-%m-%d").tolist() == [
        row["date"][:10] for row in summary["top_absolute_daily_loss_differences"]]
    expected_cells = {(cycle, dimension, int(group)) for cycle in range(5, 60)
                      for dimension, groups in expected_groups.items() if dimension != "quarter"
                      for group in groups}
    assert len(cells) == len(expected_cells) == 825
    assert set(zip(cells.cycle, cells.dimension, cells.group)) == expected_cells
    assert cells.empty_evaluation_cell.eq(cells.evaluated.eq(0)).all()
    assert int(cells.empty_evaluation_cell.sum()) == 16
    assert cells.evaluated.ge(0).all() and cells.issued.ge(cells.evaluated).all()
    spread = patterns.loc[patterns.dimension == "spread_group"].sort_values("group").copy()
    assert spread.group.tolist() == ["1", "2", "3", "4"]
    spread["origin_share_pct"] = 100 * spread.n / n
    spread["benchmark_sse_share_pct"] = 100 * spread.sum_lb / sums.sum_lb
    spread["relative_mse_increase_pct"] = -100 * spread.relative_reduction
    close(spread.origin_share_pct.sum(), 100)
    close(spread.benchmark_sse_share_pct.sum(), 100)
    close(np.sum(spread.benchmark_sse_share_pct / 100 * spread.relative_mse_increase_pct),
          relative * 100)
    checks = {
        "n": n, "calendar_days": len(daily), "all_pattern_dimensions_reconcile": True,
        "mse_b": float(sums.sum_lb / n), "mse_t": float(sums.sum_lt / n),
        "delta_t_minus_b_pips_squared": float(delta),
        "relative_mse_increase_pct": float(relative * 100),
        "nominal_conditional_intervals": intervals, "normal_critical_value": z,
        "leave_one_day_out_increase_pips_squared": [float(leave_out.min()), float(leave_out.max())],
        "largest_adverse_day": top.iloc[0].date.strftime("%Y-%m-%d"),
        "largest_day_share_net_deterioration_pct": float(100 * -top.iloc[0].sum_d / excess.sum()),
        "lower_three_groups_summed_loss_reduction": float(spread.iloc[:3].sum_d.sum()),
        "highest_group_summed_loss_increase": float(-spread.iloc[3].sum_d),
        "declared_cycle_cells": len(cells), "empty_cycle_cells_retained": int(cells.empty_evaluation_cell.sum()),
        "spread_groups": spread[["group", "n", "origin_share_pct", "benchmark_sse_share_pct",
                                 "mse_b", "mse_t", "relative_mse_increase_pct"]].to_dict("records")}
    return spread, checks


def draw(spread, source_hashes):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": INK, "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), gridspec_kw={"width_ratios": [1.08, 1]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.86, wspace=.30)
    x = np.arange(4)
    left, right = axes
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ACB1B5")
        ax.tick_params(color="#ACB1B5", labelcolor=INK)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
        ax.set_xticks(x, ["1", "2", "3", "4"])
        ax.set_xlabel("Current-spread category (low to high)", labelpad=9)
    width = .34
    a = left.bar(x-width/2, spread.origin_share_pct, width, color=BLUE, label="Origins")
    b = left.bar(x+width/2, spread.benchmark_sse_share_pct, width, color=RUST, label="Benchmark squared error")
    left.bar_label(a, fmt="%.1f", padding=3, fontsize=8)
    left.bar_label(b, fmt="%.1f", padding=3, fontsize=8)
    left.set_ylim(0, 105)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_ylabel("Share of total (%)")
    left.set_title("A. Where benchmark error concentrates", loc="left", pad=14, fontsize=10)
    left.legend(frameon=False, fontsize=8.5, loc="upper left")
    values = spread.relative_mse_increase_pct.to_numpy()
    right.axhline(0, color=INK, linewidth=.9)
    right.vlines(x, 0, values, color=BLUE, linewidth=1.2)
    right.scatter(x, values, color=BLUE, s=34, zorder=3)
    for xi, value in zip(x, values):
        right.annotate(f"{value:+.3f}", (xi, value), xytext=(0, 8 if value >= 0 else -14),
                       textcoords="offset points", ha="center", fontsize=9)
    right.set_xlim(-.5, 3.5)
    right.set_ylim(-.23, .35)
    right.set_yticks([-.2, -.1, 0, .1, .2, .3])
    right.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    right.set_ylabel("Relative MSE change (%)")
    right.set_title("B. Added timing minus benchmark", loc="left", pad=14, fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert (box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1
                    and box.y1 <= fig.bbox.height + 1), f"Clipped text: {item.get_text()}"
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = "; ".join(f"{name} SHA256={digest}" for name, digest in source_hashes.items())
    fig.savefig(OUT / "appendix_spread_concentration.png", dpi=300, facecolor="white",
                metadata={"Title": "Spread forecast error concentration", "Description": provenance})
    fig.savefig(OUT / "appendix_spread_concentration.pdf", facecolor="white",
                metadata={"Title": "Spread forecast error concentration", "Subject": provenance,
                          "Creator": "reporting/plot_spread_concentration.py",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    before = hashes()
    spread, checks = verify()
    draw(spread, before)
    assert before == hashes(), "A source changed during reporting"
    receipt = {"status": "passed", "scope": "reporting_only_no_refits_or_archive_changes",
               "sign_convention": "extension_minus_benchmark; negative favors added timing",
               "group_definition": "Each cycle's prior-training current-spread quartiles; ties assigned upward; no fixed absolute pooled boundaries",
               "checks": checks, "source_sha256": before,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "output_sha256": {suffix: hashlib.sha256((OUT / f"appendix_spread_concentration.{suffix}").read_bytes()).hexdigest()
                                 for suffix in ("png", "pdf")}}
    (OUT / "appendix_spread_concentration_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "passed", "n": checks["n"], "relative_mse_increase_pct": checks["relative_mse_increase_pct"],
                      "outputs": [str(OUT / f"appendix_spread_concentration.{x}") for x in ("png", "pdf")]}, indent=2))


if __name__ == "__main__":
    main()

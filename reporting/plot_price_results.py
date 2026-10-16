"""Reproduce the two main empirical figures from compact, verified saved inputs.

Run from any working directory; input and output paths are relative to this file.
No full archive, network, model fitting, or inferred standard errors are needed.
The original paths contain the same 127,013 events at every horizon (0--300 s).
The continuation ECDF includes all observations and a full-support inset.
Quarter plots preserve chronological order, matched events and event-weighted pools.
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

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fx_publication_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/plot_inputs/price"
OUT = ROOT / "build/figures"
SOURCES = {
    "mean_path": DATA / "mean_response_path.csv",
    "distribution": DATA / "continuation_distribution.npz",
    "interaction": DATA / "interaction_quarter_losses.csv",
    "sequence": DATA / "sequence_quarter_losses.csv",
    "receipt": DATA / "export_receipt.json",
}
BLUE = "#1E5673"
RUST = "#A66042"
GREY = "#6C7378"
INK = "#252525"


def hashes():
    return {key: sha256(path) for key, path in SOURCES.items()}


def read_responses():
    receipt = json.loads(SOURCES["receipt"].read_text())
    assert receipt["cohort"]["n_events"] == N
    for name, expected in receipt["exports"].items():
        assert sha256(DATA / name) == expected["sha256"], f"Export hash mismatch: {name}"
    frame = pd.read_csv(SOURCES["mean_path"], float_precision="round_trip")
    np.testing.assert_array_equal(frame.horizon_sec, np.arange(301))
    assert (frame.n_events == N).all()
    means = frame[VALUE_COLUMNS].to_numpy()
    assert np.isfinite(means).all()
    np.testing.assert_allclose(means[:,2], means[:,0] + means[:,1], atol=1e-11, rtol=0)
    assert means[0,1] == 0
    with np.load(SOURCES["distribution"], allow_pickle=False) as distribution:
        ids = distribution["episode_id"]
        values = distribution["continuation_pips"]
        assert len(ids) == N and np.all(np.diff(ids) > 0)
        assert values.shape == (N,) and np.isfinite(values).all()
    mean_initial = receipt["numerics"]["mean_initial_pips"]
    np.testing.assert_allclose(means[:,0], mean_initial, atol=1e-11, rtol=0)
    np.testing.assert_allclose(means[300,1], values.mean(), atol=1e-11, rtol=0)
    np.testing.assert_array_equal(means[[0,60,300],2], receipt["numerics"]["mean_path_I_0_60_300"])
    return values, mean_initial, means


def read_quarters():
    result = []
    expected = pd.period_range("2024Q2", "2026Q3", freq="Q").astype(str).tolist()
    for source, extension, benchmark in [("interaction", "interaction", "additive"),
                                          ("sequence", "sequence", "baseline")]:
        q = pd.read_csv(SOURCES[source], float_precision="round_trip").sort_values("quarter").reset_index(drop=True)
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


def clean(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ACB1B5")
    ax.tick_params(color="#ACB1B5", labelcolor=INK)
    if grid:
        ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, stem, title, keys, source_hashes):
    provenance = "; ".join(f"{SOURCES[k].relative_to(ROOT)} SHA256={source_hashes[k]}" for k in keys)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if not (box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1):
                raise ValueError(f"Clipped figure text: {item.get_text()!r}")
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white",
                metadata={"Title": title, "Description": provenance,
                          "Software": "Matplotlib; observed data, no smoothing or model fitting"})
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white",
                metadata={"Title": title, "Subject": provenance, "Creator": "reporting/plot_price_results.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def adjustment_figure(continuation, mean_initial, means, source_hashes):
    assert ((means[:, 2] >= 0) & (means[:, 2] <= .45)).all(), "Mean path exceeds plotted limits"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65))
    fig.subplots_adjust(left=.075, right=.98, bottom=.19, top=.91, wspace=.29)
    for ax in axes:
        clean(ax)
    ax = axes[0]
    ax.plot(np.arange(301), means[:, 2], color=BLUE, lw=1.5, label=r"$I(h)$")
    ax.axhline(mean_initial, color=GREY, lw=1.15, ls="--", label=r"$J$")
    ax.set(xlim=(0, 300), ylim=(0, .45), xlabel="Seconds after detection", ylabel="Mean displacement (pips)")
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_yticks([0, .1, .2, .3, .4])
    ax.set_title("A", loc="left", fontsize=11, pad=9)
    ax.legend(frameon=False, loc="lower right", handlelength=2.1, borderaxespad=.9)

    values = np.sort(continuation)
    probability = np.arange(1, N + 1) / N
    ax = axes[1]
    ax.step(values, probability, where="post", color=BLUE, lw=1.35)
    ax.axvline(0, color=GREY, lw=.75, ls=":")
    ax.set(xlim=(-10, 10), ylim=(0, 1), xlabel=r"$C(300)$ (pips)", ylabel="Cumulative probability")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_title("B", loc="left", fontsize=11, pad=9)
    inset = ax.inset_axes([.62, .16, .34, .35])
    clean(inset, grid=False)
    inset.step(values, probability, where="post", color=BLUE, lw=.85)
    inset.axvline(0, color=GREY, lw=.6, ls=":")
    inset.set_xlim(values[0] - 3, values[-1] + 3)
    inset.set_ylim(0, 1)
    inset.set_xticks([-100, 0, 100])
    inset.set_yticks([0, 1])
    inset.tick_params(labelsize=7, length=2, pad=2)
    inset.set_title("Full range", fontsize=8, pad=3)
    save(fig, "section4_adjustment", "Observed mean price path and five-minute continuation distribution",
         ["mean_path", "distribution", "receipt"], source_hashes)
    print("C(300) central-view share:", ((values >= -10) & (values <= 10)).mean(), flush=True)
    print("Mean I(0), I(60), I(300):", means[[0, 60, 300], 2].tolist(), flush=True)
    print("Mean path min/max:", means[:, 2].min(), means[:, 2].max(), flush=True)


def forecast_figure(quarters, source_hashes):
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.13, top=.94, hspace=.27)
    positions = np.arange(10)
    titles = ["A  Pressure-variation interaction", "B  Pressure ordering"]
    for ax, (q, column, pooled), title, color in zip(axes, quarters, titles, [BLUE, RUST]):
        clean(ax)
        delta = q[column].to_numpy()
        ax.axhline(0, color=GREY, lw=.95)
        ax.axvline(9.7, color="#B9BEC1", lw=.8, ls=":")
        ax.vlines(positions, 0, delta, color=color, lw=1.1)
        ax.scatter(positions, delta, color=color, s=22, zorder=3)
        ax.scatter([10.6], [pooled], color=color, marker="D", s=29, zorder=3)
        ax.set_xlim(-.5, 11.15)
        ax.set_ylim(-.0045, .008)
        ax.set_yticks([-.004, 0, .004, .008])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_ylabel(r"MSE difference (pips$^2$)")
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    labels = [f"{q[:4]}\n{q[4:]}" for q in quarters[0][0].quarter] + ["Pooled"]
    axes[1].set_xticks([*positions, 10.6], labels, fontsize=8.5)
    save(fig, "section4_forecast_quarters", "Quarterly paired forecast loss differences",
         ["interaction", "sequence"], source_hashes)
    print("Pooled MSE differences:", [float(item[2]) for item in quarters], flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Generated figures directory (default: build/figures)")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and numerical identities without writing figures")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.labelcolor": INK, "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    before = hashes()
    continuation, mean_initial, means = read_responses()
    quarters = read_quarters()
    receipt = json.loads(SOURCES["receipt"].read_text())
    np.testing.assert_array_equal([q[2] for q in quarters],
        [receipt["numerics"]["pooled_interaction_minus_additive"],
         receipt["numerics"]["pooled_sequence_minus_baseline"]])
    if not args.check_only:
        OUT.mkdir(parents=True, exist_ok=True)
        adjustment_figure(continuation, mean_initial, means, before)
        forecast_figure(quarters, before)
    assert hashes() == before, "An exported input changed during plotting"
    print(f"Verified {N:,} events, 301 common-cohort horizons and 10 matched calendar quarters.")
    if not args.check_only:
        for stem in ("section4_adjustment", "section4_forecast_quarters"):
            for suffix in ("png", "pdf"):
                path = OUT / f"{stem}.{suffix}"
                print(path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

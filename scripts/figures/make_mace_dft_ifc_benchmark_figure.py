#!/usr/bin/env python3
"""Create the PRB Appendix figure for the SnSe MLIP--DFT IFC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"
DEFAULT_ANALYSIS = (
    ROOT
    / "outputs"
    / "analysis"
    / "dft_mlip_ifc_validation"
)

DFT_COLOR = "#252525"
MACE_COLOR = "#2F6788"
REGISTRY_COLORS = {
    "minimum": "#2F6788",
    "median": "#C18A2D",
    "maximum": "#B64E3B",
}
SPECIES_COLORS = {"Sn": "#6677A8", "Se": "#D5A13C"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.transparent": False,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def metric_text(summary: dict[str, Any], key: str) -> str:
    metric = summary[key]
    return (
        rf"$r={metric['pearson_r']:.3f}$" + "\n"
        rf"$s_0={metric['origin_slope']:.3f}$" + "\n"
        rf"NRMSE$={metric['normalized_rmse']:.3f}$"
    )


def plot_hessian_parity(
    ax: plt.Axes, rows: list[dict[str, str]], summary: dict[str, Any]
) -> None:
    all_values: list[float] = []
    for registry in ("minimum", "median", "maximum"):
        subset = [row for row in rows if row["registry"] == registry]
        x = np.asarray([float(row["dft_hessian_ev_per_angstrom2"]) for row in subset])
        y = np.asarray([float(row["mace_hessian_ev_per_angstrom2"]) for row in subset])
        all_values.extend(x.tolist())
        all_values.extend(y.tolist())
        ax.scatter(
            x,
            y,
            s=7,
            alpha=0.48,
            linewidths=0,
            color=REGISTRY_COLORS[registry],
            label=registry.capitalize(),
        )
    limit = 1.05 * max(abs(min(all_values)), abs(max(all_values)))
    ax.plot([-limit, limit], [-limit, limit], color="#777777", lw=0.8, zorder=0)
    ax.axhline(0, color="#D4D4D4", lw=0.5, zorder=0)
    ax.axvline(0, color="#D4D4D4", lw=0.5, zorder=0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"DFT $\Phi$ ($\mathrm{eV}\,\AA^{-2}$)")
    ax.set_ylabel(r"MACE $\Phi$ ($\mathrm{eV}\,\AA^{-2}$)")
    ax.set_title("Harmonic curvature across registries")
    ax.legend(loc="lower right", handletextpad=0.25, borderaxespad=0.3)
    ax.text(
        0.04,
        0.96,
        metric_text(summary, "one_by_one_raw_hessian"),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.3},
    )


def plot_gamma_modes(
    ax: plt.Axes, rows: list[dict[str, str]], summary: dict[str, Any]
) -> None:
    stable = [row for row in rows if row["stable_dft_optical"].lower() == "true"]
    stable.sort(key=lambda row: float(row["dft_frequency_thz"]))
    mode = np.arange(1, len(stable) + 1)
    dft = np.asarray([float(row["dft_frequency_thz"]) for row in stable])
    mace = np.asarray([float(row["mace_frequency_thz"]) for row in stable])
    low_frequency = dft < 2.0
    low_mode_count = int(np.count_nonzero(low_frequency))
    low_frequency_mae = float(np.mean(np.abs(mace[low_frequency] - dft[low_frequency])))
    ax.axvspan(
        0.5,
        low_mode_count + 0.5,
        color="#E6EFF3",
        alpha=0.9,
        lw=0,
        zorder=-1,
    )
    for x, y0, y1 in zip(mode, dft, mace):
        ax.plot([x, x], [y0, y1], color="#C8CDD1", lw=0.7, zorder=0)
    ax.plot(mode, dft, color=DFT_COLOR, lw=1.0, marker="o", ms=2.8, label="DFT")
    ax.plot(
        mode,
        mace,
        color=MACE_COLOR,
        lw=1.0,
        ls="--",
        marker="s",
        ms=2.6,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label="MACE",
    )
    ax.set_xlim(0.3, len(stable) + 0.7)
    ax.set_xlabel("Matched stable optical mode")
    ax.set_ylabel(r"$\Gamma$ frequency (THz)")
    ax.set_title(r"Minimum-registry $\Gamma$ spectrum")
    ax.legend(loc="upper left")
    gamma = summary["gamma"]
    ax.text(
        0.985,
        0.035,
        rf"MAE$_{{<2\,\mathrm{{THz}}}}={low_frequency_mae:.3f}$ THz" + "\n"
        rf"MAE$_{{\mathrm{{all}}}}={gamma['stable_optical_frequency_mae_thz']:.3f}$ THz" + "\n"
        rf"max-$\nu$ err. $={100*gamma['maximum_frequency_relative_error']:.1f}\%$" + "\n"
        rf"median overlap $={gamma['median_mass_weighted_matched_mode_overlap']:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6},
    )


def plot_spatial_decay(
    ax: plt.Axes, rows: list[dict[str, str]], summary: dict[str, Any]
) -> None:
    positive: list[float] = []
    for species in ("Sn", "Se"):
        subset = [row for row in rows if row["displaced_species"] == species]
        subset.sort(key=lambda row: float(row["periodic_distance_angstrom"]))
        distance = np.asarray([float(row["periodic_distance_angstrom"]) for row in subset])
        dft = np.asarray(
            [float(row["dft_centered_response_magnitude_ev_per_angstrom"]) for row in subset]
        )
        mace = np.asarray(
            [float(row["mace_centered_response_magnitude_ev_per_angstrom"]) for row in subset]
        )
        positive.extend(dft[dft > 0].tolist())
        positive.extend(mace[mace > 0].tolist())
        color = SPECIES_COLORS[species]
        ax.plot(distance, dft, color=color, lw=0.8, alpha=0.70)
        ax.scatter(distance, dft, s=11, color=color, linewidths=0, alpha=0.78)
        ax.plot(distance, mace, color=color, lw=0.8, ls="--", alpha=0.85)
        ax.scatter(
            distance,
            mace,
            s=12,
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
        )
    ax.set_yscale("log")
    ax.set_ylim(max(min(positive) * 0.55, 1.0e-8), max(positive) * 1.8)
    ax.set_xlabel(r"Periodic distance ($\AA$)")
    ax.set_ylabel(r"Centered $|\Delta\mathbf{F}|$ ($\mathrm{eV}\,\AA^{-1}$)")
    ax.set_title(r"Minimum-registry $2\times2$ response")
    handles = [
        Line2D([], [], color=SPECIES_COLORS["Sn"], lw=1.2, label="Sn"),
        Line2D([], [], color=SPECIES_COLORS["Se"], lw=1.2, label="Se"),
        Line2D([], [], color="#555555", marker="o", ms=3, lw=0.9, label="DFT"),
        Line2D(
            [],
            [],
            color="#555555",
            marker="o",
            markerfacecolor="white",
            ms=3,
            lw=0.9,
            ls="--",
            label="MACE",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        fontsize=5.7,
        columnspacing=0.55,
        handlelength=1.7,
        handletextpad=0.25,
        borderaxespad=0.25,
    )
    ax.text(
        0.035,
        0.035,
        rf"response NRMSE $={summary['two_by_two_force_response']['normalized_rmse']:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
    )


def save_figure(fig: plt.Figure, stem: str, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
    }.items():
        path = figure_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        if suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DATA)
    args = parser.parse_args()

    summary = json.loads(
        (args.analysis_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    hessian = read_rows(args.analysis_dir / "one_by_one_hessian_components.csv")
    gamma = read_rows(args.analysis_dir / "minimum_gamma_matched_modes.csv")
    spatial = read_rows(args.analysis_dir / "two_by_two_spatial_response_decay.csv")

    fig = plt.figure(figsize=(7.01, 2.65), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.065,
        right=0.995,
        bottom=0.18,
        top=0.89,
        wspace=0.36,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    plot_hessian_parity(axes[0], hessian, summary)
    plot_gamma_modes(axes[1], gamma, summary)
    plot_spatial_decay(axes[2], spatial, summary)
    for ax, label in zip(axes, ("a", "b", "c")):
        ax.set_box_aspect(1.0)
        panel_label(ax, label)
        ax.tick_params(direction="out")
    save_figure(fig, "FigA2_mace_dft_ifc_benchmark", args.figure_dir)

    args.source_data_dir.mkdir(parents=True, exist_ok=True)
    for source, target in (
        ("one_by_one_hessian_components.csv", "FigA2_hessian_components.csv"),
        ("minimum_gamma_matched_modes.csv", "FigA2_gamma_modes.csv"),
        ("two_by_two_spatial_response_decay.csv", "FigA2_spatial_decay.csv"),
    ):
        shutil.copy2(args.analysis_dir / source, args.source_data_dir / target)
    for obsolete in (
        "FigA2_comparison_metrics.csv",
        "FigA2_primary_gates.csv",
        "FigA2_reference_force_offsets.csv",
    ):
        (args.source_data_dir / obsolete).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "figure": str(args.figure_dir / "FigA2_mace_dft_ifc_benchmark.pdf"),
                "source_tables": 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create Appendix Fig. A3 for the corrected 7.61-degree shallow mode."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORRECTED = ROOT / "work"
ANALYSIS = CORRECTED / "analysis" / "soft_modes" / "7p61deg"
SPECIAL_STUDY = (
    CORRECTED
    / "analysis"
    / "7p61_special_study"
)
TIGHT_PROBE = (
    SPECIAL_STUDY
    / "residual_force_closure"
    / "tight_relax_probe_summary.json"
)
ENDPOINT_STATS = (
    SPECIAL_STUDY
    / "endpoint_full_phonon"
    / "plus_endpoint"
    / "full_phonon_stats.json"
)
PHONONS = CORRECTED / "phonons" / "7p61deg"
FIGURES = ROOT / "outputs" / "figures"
SOURCE_DATA = ROOT / "outputs" / "source_data"

COLORS = {
    "twist": "#B64E3B",
    "control_lower": "#2F718E",
    "control_upper": "#C38B34",
    "sn": "#6677A8",
    "se": "#D5A13C",
}

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
    }.items():
        path = FIGURES / f"FigA3_soft_mode_and_displacement_convergence{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        if suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    plt.close(fig)


def main() -> None:
    global ANALYSIS, TIGHT_PROBE, ENDPOINT_STATS, PHONONS, FIGURES, SOURCE_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--tight-probe", type=Path, default=TIGHT_PROBE)
    parser.add_argument("--endpoint-stats", type=Path, default=ENDPOINT_STATS)
    parser.add_argument("--phonon-root", type=Path, default=PHONONS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DATA)
    args = parser.parse_args()
    ANALYSIS = args.analysis_dir.resolve()
    TIGHT_PROBE = args.tight_probe.resolve()
    ENDPOINT_STATS = args.endpoint_stats.resolve()
    PHONONS = args.phonon_root.resolve()
    FIGURES = args.figure_dir.resolve()
    SOURCE_DATA = args.source_data_dir.resolve()
    payload = json.loads(
        (ANALYSIS / "s_point_soft_mode_summary.json").read_text(encoding="ascii")
    )
    summaries = {row["kind"]: row for row in payload["summaries"]}
    atom_rows = read_csv(ANALYSIS / "twist_selected_mode_atoms.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.01, 4.20), layout="constrained")
    ax_low, ax_mode, ax_disp, ax_energy = axes.ravel()

    labels = {
        "twist": "Twisted",
        "control_lower": "Lower control",
        "control_upper": "Upper control",
    }
    low_rows: list[dict[str, Any]] = []
    for kind in ("twist", "control_lower", "control_upper"):
        values = np.asarray(summaries[kind]["lowest_frequencies_thz"], dtype=float)
        ranks = np.arange(1, len(values) + 1)
        ax_low.plot(
            ranks,
            values,
            marker="o",
            ms=2.4,
            lw=1.0,
            color=COLORS[kind],
            label=labels[kind],
        )
        low_rows.extend(
            {
                "case": kind,
                "mode_rank_at_s": int(rank),
                "frequency_thz": float(frequency),
            }
            for rank, frequency in zip(ranks, values)
        )
    ax_low.axhline(0.0, color="#333333", lw=0.7)
    ax_low.set_xlabel(r"Mode rank at $S$")
    ax_low.set_ylabel("Frequency (THz)")
    ax_low.set_title(r"Low-frequency modes at $S$")
    ax_low.legend(loc="upper left")
    ax_low.grid(axis="y", color="#E6E8EA", lw=0.5)

    positions = np.asarray(
        [[float(row["x_angstrom"]), float(row["y_angstrom"])] for row in atom_rows]
    )
    vectors = np.asarray(
        [[float(row["ux_normalized"]), float(row["uy_normalized"])] for row in atom_rows]
    )
    amplitudes = np.asarray(
        [float(row["physical_amplitude_weight"]) for row in atom_rows]
    )
    symbols = np.asarray([row["symbol"] for row in atom_rows])
    for symbol, color in (("Sn", COLORS["sn"]), ("Se", COLORS["se"])):
        selected_symbol = symbols == symbol
        ax_mode.scatter(
            positions[selected_symbol, 0],
            positions[selected_symbol, 1],
            s=4.0,
            color=color,
            alpha=0.25,
            linewidths=0,
        )
    selected = amplitudes >= np.quantile(amplitudes, 0.70)
    ax_mode.quiver(
        positions[selected, 0],
        positions[selected, 1],
        vectors[selected, 0],
        vectors[selected, 1],
        amplitudes[selected],
        cmap="magma",
        angles="xy",
        scale_units="xy",
        scale=0.18,
        width=0.0026,
    )
    ax_mode.set_aspect("equal")
    ax_mode.set_xlabel(r"$x$ ($\AA$)")
    ax_mode.set_ylabel(r"$y$ ($\AA$)")
    ax_mode.set_title("Shallow-mode displacement field")

    sensitivity_rows: list[dict[str, Any]] = []
    band_data: dict[str, dict[str, np.ndarray]] = {}
    for tag, dirname in (
        ("0.005", "fmax0p001_disp0p005_plusminus"),
        ("0.010", "fmax0p001_disp0p01_plusminus"),
    ):
        with np.load(PHONONS / dirname / "full_phonon_band.npz") as data:
            band_data[tag] = {key: np.asarray(data[key]) for key in data.files}
    reference = band_data["0.010"]
    ticks = np.asarray(reference["ticks"], dtype=float)
    x = np.asarray(reference["x"], dtype=float)
    mask = (x >= ticks[1] - 1.0e-12) & (x <= ticks[3] + 1.0e-12)
    for tag, style, width, zorder, marker in (
        ("0.010", "-", 1.45, 2, None),
        ("0.005", "--", 1.05, 3, "o"),
    ):
        frequencies = np.asarray(band_data[tag]["frequencies"], dtype=float)
        lowest = np.min(frequencies, axis=1)
        ax_disp.plot(
            x[mask],
            lowest[mask],
            ls=style,
            color=COLORS["twist"],
            lw=width,
            zorder=zorder,
            marker=marker,
            markevery=3,
            ms=2.4,
            markerfacecolor="white",
            markeredgewidth=0.6,
            label=rf"$\Delta={tag}$ $\AA$",
        )
        sensitivity_rows.extend(
            {
                "displacement_angstrom": tag,
                "q_path_distance": float(distance),
                "lowest_frequency_thz": float(frequency),
            }
            for distance, frequency in zip(x[mask], lowest[mask])
        )
    rms_difference = float(
        np.sqrt(
            np.mean(
                (
                    band_data["0.005"]["frequencies"]
                    - band_data["0.010"]["frequencies"]
                )
                ** 2
            )
        )
    )
    ax_disp.axhline(0.0, color="#333333", lw=0.7)
    ax_disp.axvline(ticks[2], color="#9D7A35", lw=0.7, ls=":")
    ax_disp.set_xticks(ticks[1:4], ["X", "S", "Y"])
    ax_disp.set_ylabel("Lowest frequency (THz)")
    ax_disp.set_title("Finite-displacement convergence")
    ax_disp.legend(loc="lower left")
    ax_disp.text(
        0.985,
        0.045,
        "spectrum RMS\n" + r"$\Delta\nu=9.2\times10^{-5}$ THz",
        transform=ax_disp.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
    )
    ax_disp.grid(axis="y", color="#E6E8EA", lw=0.5)

    twist = summaries["twist"]
    character_labels = ["In plane", "Out of plane", "Lower layer", "Upper layer"]
    character_values = [
        float(twist["in_plane_displacement_fraction"]),
        float(twist["out_of_plane_displacement_fraction"]),
        float(twist["lower_layer_displacement_fraction"]),
        float(twist["upper_layer_displacement_fraction"]),
    ]
    tight = json.loads(TIGHT_PROBE.read_text(encoding="ascii"))
    endpoint = json.loads(ENDPOINT_STATS.read_text(encoding="ascii"))
    probes = sorted(
        tight["centered_probes"], key=lambda row: float(row["amplitude_angstrom"])
    )
    probe_rows: list[dict[str, Any]] = []
    for probe in probes:
        amplitude = float(probe["amplitude_angstrom"])
        center = float(probe["center_energy_ev"])
        curvature = float(probe["energy_curvature_ev_per_angstrom2_per_moire_cell"])
        force_curvature = float(probe["force_projected_curvature_ev_per_angstrom2"])
        for signed_amplitude, energy in (
            (-amplitude, float(probe["minus_energy_ev"])),
            (0.0, center),
            (amplitude, float(probe["plus_energy_ev"])),
        ):
            probe_rows.append(
                {
                    "probe_extent_angstrom": amplitude,
                    "amplitude_angstrom": signed_amplitude,
                    "delta_energy_mev_per_moire_cell": 250.0 * (energy - center),
                    "energy_curvature_ev_per_angstrom2_per_moire_cell": curvature,
                    "force_projected_curvature_ev_per_angstrom2": force_curvature,
                }
            )
    amplitudes_scan = np.asarray(
        [float(row["amplitude_angstrom"]) for row in probe_rows], dtype=float
    )
    energies_mev = np.asarray(
        [float(row["delta_energy_mev_per_moire_cell"]) for row in probe_rows],
        dtype=float,
    )
    curvature_values = np.asarray(
        [float(row["energy_curvature_ev_per_angstrom2_per_moire_cell"]) for row in probes]
    )
    representative_curvature = float(np.mean(curvature_values))
    dense_amplitude = np.linspace(-0.0105, 0.0105, 301)
    dense_energy_mev = 500.0 * representative_curvature * dense_amplitude**2
    ax_energy.plot(
        dense_amplitude,
        dense_energy_mev,
        color=COLORS["twist"],
        lw=1.2,
    )
    ax_energy.scatter(
        amplitudes_scan,
        energies_mev,
        s=18,
        color=COLORS["twist"],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    ax_energy.axhline(0.0, color="#333333", lw=0.7)
    ax_energy.axvline(0.0, color="#A9ADB1", lw=0.6)
    ax_energy.set_xlabel(r"Frozen-mode amplitude ($\AA$)")
    ax_energy.set_ylabel(
        r"$\Delta E$ (meV per moir$\acute{\mathrm{e}}$ cell)"
    )
    ax_energy.set_title("Tight-relaxation closure")
    ax_energy.text(
        0.50,
        0.93,
        r"Positive curvature: $12.516$ to $12.518$ eV $\AA^{-2}$"
        + "\n"
        + rf"endpoint $\nu_{{\min}}={float(endpoint['min_frequency_thz']):.4f}$ THz",
        transform=ax_energy.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.9},
    )
    ax_energy.grid(axis="y", color="#E6E8EA", lw=0.5)

    for ax, label in zip(axes.ravel(), ("a", "b", "c", "d")):
        panel_label(ax, label)
        ax.tick_params(direction="out")
    save_figure(fig)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    write_csv(SOURCE_DATA / "FigA3_low_modes_at_s.csv", low_rows)
    write_csv(SOURCE_DATA / "FigA3_displacement_sensitivity.csv", sensitivity_rows)
    write_csv(
        SOURCE_DATA / "FigA3_mode_character.csv",
        [
            {"component": label, "fraction": value}
            for label, value in zip(character_labels, character_values)
        ],
    )
    write_csv(
        SOURCE_DATA / "FigA3_frozen_mode_energy.csv",
        probe_rows,
    )
    shutil.copy2(
        ANALYSIS / "twist_selected_mode_atoms.csv",
        SOURCE_DATA / "FigA3_soft_mode_atoms.csv",
    )
    print(
        json.dumps(
            {
                "figure": str(
                    FIGURES / "FigA3_soft_mode_and_displacement_convergence.pdf"
                ),
                "spectrum_rms_difference_thz": rms_difference,
                "source_tables": 5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

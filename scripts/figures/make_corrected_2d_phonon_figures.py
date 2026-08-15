#!/usr/bin/env python3
"""Build manuscript figures from the reviewer-audited analytical velocities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "work"
ANALYSIS_DIR = (
    DATA_ROOT
    / "analysis"
    / "review_analytical_velocity"
    / "final_aggregate"
)
FIGURE_DIR = ROOT / "outputs" / "figures"
SOURCE_DIR = ROOT / "outputs" / "source_data"
PHONON_DIRNAME = "fmax0p001_disp0p01_plusminus"

NEUTRAL = "#333333"
CONTROL = "#777C82"
SIGNAL = "#2F6F8F"
ACCENT = "#B64E3B"
PALE = "#E8ECEF"
ANGLE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "snse_angles", ["#B64E3B", "#C98942", "#718C78", "#2F6F8F"]
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DIR)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty source table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def angle_color(angle: float, minimum: float, maximum: float) -> Any:
    span = max(maximum - minimum, 1.0e-12)
    return ANGLE_CMAP((angle - minimum) / span)


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[
        values[0] - (middle[0] - values[0]),
        middle,
        values[-1] + (values[-1] - middle[-1]),
    ]


def number(value: str | None) -> float:
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


def load_state(
    analysis_dir: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    summary_path = analysis_dir / "review_analytical_velocity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if int(summary["case_count"]) != 21 or int(summary["angle_count"]) != 7:
        raise RuntimeError("review aggregation is incomplete")
    thermal = read_csv(analysis_dir / "review_analytical_velocity_angle_summary.csv")
    spectral = read_csv(analysis_dir / "review_analytical_velocity_spectral_bins.csv")
    stability = read_csv(analysis_dir / "review_q_grid_stability.csv")
    asr = read_csv(analysis_dir / "review_asr_sensitivity.csv")
    path_density = read_csv(analysis_dir / "review_path_density_sensitivity.csv")
    return thermal, spectral, stability, asr, path_density


def representative_band_paths() -> dict[str, Path]:
    base = DATA_ROOT / "matched_controls_registry_min" / "phonons"
    return {
        "twist": DATA_ROOT / "phonons" / "8p77deg" / PHONON_DIRNAME / "full_phonon_band.npz",
        "control_lower": base / "8p77deg_lower_registry_min" / PHONON_DIRNAME / "full_phonon_band.npz",
        "control_upper": base / "8p77deg_upper_registry_min" / PHONON_DIRNAME / "full_phonon_band.npz",
    }


def branch_density(path: Path, frequency_edges: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    frequencies = np.asarray(payload["frequencies"], dtype=float)
    density = np.zeros((len(frequency_edges) - 1, frequencies.shape[0]), dtype=float)
    for index, values in enumerate(frequencies):
        histogram, _ = np.histogram(values, bins=frequency_edges)
        density[:, index] = histogram / (frequencies.shape[1] * np.diff(frequency_edges))
    return payload, density


def make_figure_2(thermal: list[dict[str, str]]) -> None:
    rows = [
        row
        for row in thermal
        if number(row["temperature_k"]) == 300.0
        and math.isclose(number(row["cutoff_thz"]), 0.05)
    ]
    rows.sort(key=lambda row: number(row["angle_deg"]))
    if len(rows) != 7:
        raise RuntimeError(f"expected seven 300-K angle rows, found {len(rows)}")

    paths = representative_band_paths()
    frequency_edges = np.linspace(0.0, 2.0, 101)
    twist_band, twist_density = branch_density(paths["twist"], frequency_edges)
    lower_band, lower_density = branch_density(paths["control_lower"], frequency_edges)
    upper_band, upper_density = branch_density(paths["control_upper"], frequency_edges)
    for candidate in (lower_band, upper_band):
        if not np.allclose(candidate["x"], twist_band["x"], atol=1.0e-12):
            raise RuntimeError("representative band paths do not share a q path")
    control_density = 0.5 * (lower_density + upper_density)
    frequency_mid = 0.5 * (frequency_edges[:-1] + frequency_edges[1:])
    vmax = float(np.percentile(np.r_[twist_density.ravel(), control_density.ravel()], 99.0))

    fig = plt.figure(figsize=(3.38, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.0, 0.045),
        height_ratios=(0.78, 0.78, 1.18),
        hspace=0.07,
        wspace=0.04,
    )
    ax_control = fig.add_subplot(grid[0, 0])
    ax_twist = fig.add_subplot(grid[1, 0], sharex=ax_control, sharey=ax_control)
    cax = fig.add_subplot(grid[0:2, 1])
    ax_angle = fig.add_subplot(grid[2, :])

    mesh = None
    for axis, density, title in (
        (ax_control, control_density, "Matched-control mean"),
        (ax_twist, twist_density, r"Twisted $8.77^\circ$ cell"),
    ):
        mesh = axis.pcolormesh(
            coordinate_edges(np.asarray(twist_band["x"], dtype=float)),
            frequency_edges,
            density,
            shading="flat",
            cmap="magma_r",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        for tick in np.asarray(twist_band["ticks"], dtype=float):
            axis.axvline(tick, color="white", lw=0.38, alpha=0.55)
        axis.text(
            0.985,
            0.92,
            title,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            color=NEUTRAL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )
        axis.set_ylim(0.0, 2.0)
        axis.set_ylabel("Frequency (THz)")
    ax_control.tick_params(labelbottom=False)
    ax_twist.set_xticks(
        np.asarray(twist_band["ticks"], dtype=float),
        [r"$\Gamma$", "X", "S", "Y", r"$\Gamma$"],
    )
    ax_twist.set_xlabel("Wave vector")
    assert mesh is not None
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Normalized branch density", fontsize=6.7)
    cbar.ax.tick_params(labelsize=6.0)
    panel_label(ax_control, "a", x=-0.14, y=1.04)

    angles = np.asarray([number(row["angle_deg"]) for row in rows])
    ratios = np.asarray([number(row["p_twist_over_control_mean"]) for row in rows])
    lower = np.asarray([number(row["p_ratio_control_range_min"]) for row in rows])
    upper = np.asarray([number(row["p_ratio_control_range_max"]) for row in rows])
    special = np.asarray([row["label"] == "7p61deg" for row in rows])
    ax_angle.axvspan(3.10, 3.90, color=SIGNAL, alpha=0.08, lw=0, zorder=0)
    ax_angle.plot(angles, ratios, color="#A9ADB1", lw=0.9, zorder=1)
    ax_angle.errorbar(
        angles[~special],
        ratios[~special],
        yerr=np.vstack(
            (
                ratios[~special] - lower[~special],
                upper[~special] - ratios[~special],
            )
        ),
        fmt="o",
        color=SIGNAL,
        ecolor=SIGNAL,
        elinewidth=0.8,
        capsize=2.0,
        ms=4.2,
        label="Six-structure equilibrium trend",
        zorder=3,
    )
    ax_angle.errorbar(
        angles[special],
        ratios[special],
        yerr=np.vstack(
            (
                ratios[special] - lower[special],
                upper[special] - ratios[special],
            )
        ),
        fmt="D",
        mfc="white",
        mec=ACCENT,
        mew=1.0,
        ecolor=ACCENT,
        elinewidth=0.8,
        capsize=2.0,
        ms=4.4,
        label=r"$7.61^\circ$ relaxation-sensitive reference",
        zorder=4,
    )
    margin = max(0.008, 0.12 * float(np.ptp(ratios)))
    ax_angle.set(
        xlabel=r"Twist angle $\theta$ (deg)",
        ylabel=r"$\mathcal{P}^{\rm tw}/\mathcal{P}^{\rm ctl}$",
        ylim=(max(0.0, float(lower.min()) - margin), float(upper.max()) + margin),
        title=r"Analytical $\langle v^2\rangle_C$ at 300 K",
    )
    ax_angle.legend(loc="upper left", fontsize=6.2, handlelength=1.3)
    ax_angle.grid(axis="y", color="#E7E8EA", lw=0.45, zorder=0)
    panel_label(ax_angle, "b", x=-0.14, y=1.04)

    density_source: list[dict[str, Any]] = []
    for q_index, q_distance in enumerate(np.asarray(twist_band["x"], dtype=float)):
        for f_index, frequency in enumerate(frequency_mid):
            density_source.append(
                {
                    "q_path_distance": float(q_distance),
                    "frequency_thz": float(frequency),
                    "twist_branch_density": float(twist_density[f_index, q_index]),
                    "control_mean_branch_density": float(control_density[f_index, q_index]),
                }
            )
    write_csv(SOURCE_DIR / "Fig2_representative_dispersion_density.csv", density_source)
    write_csv(SOURCE_DIR / "Fig2_angle_response.csv", rows)
    save_figure(fig, FIGURE_DIR / "Fig2_phonons_and_velocity_scale")


def group_spectral(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: number(row["bin_mid_thz"]))
    return grouped


def make_figure_3(thermal: list[dict[str, str]], spectral: list[dict[str, str]]) -> None:
    rows = [
        row
        for row in thermal
        if number(row["temperature_k"]) == 300.0
        and math.isclose(number(row["cutoff_thz"]), 0.05)
    ]
    rows.sort(key=lambda row: number(row["angle_deg"]))
    angles = np.asarray([number(row["angle_deg"]) for row in rows])
    amin, amax = float(angles.min()), float(angles.max())
    labels = [row["label"] for row in rows]
    label_to_angle = {row["label"]: number(row["angle_deg"]) for row in rows}
    grouped = group_spectral(spectral)

    fig = plt.figure(figsize=(7.05, 2.18), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.10, 1.30, 1.10), wspace=0.17)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    control_profiles = []
    frequency = None
    bin_width = None
    for label in labels:
        values = grouped[label]
        frequency = np.asarray([number(row["bin_mid_thz"]) for row in values])
        bin_width = np.asarray(
            [number(row["bin_high_thz"]) - number(row["bin_low_thz"]) for row in values]
        )
        twist_share = np.asarray([number(row["twist_cv2_share"]) for row in values]) / bin_width
        control_share = 0.5 * (
            np.asarray([number(row["control_lower_cv2_share"]) for row in values])
            + np.asarray([number(row["control_upper_cv2_share"]) for row in values])
        ) / bin_width
        control_profiles.append(control_share)
        ax_a.plot(
            frequency,
            twist_share,
            color=(ACCENT if label == "7p61deg" else angle_color(label_to_angle[label], amin, amax)),
            lw=(0.9 if label == "7p61deg" else 1.1),
            ls=((0, (2, 1.5)) if label == "7p61deg" else "-"),
            alpha=(0.75 if label == "7p61deg" else 1.0),
        )
    assert frequency is not None and bin_width is not None
    control_array = np.asarray(control_profiles)
    ax_a.fill_between(
        frequency,
        np.nanmin(control_array, axis=0),
        np.nanmax(control_array, axis=0),
        color="#D9DDE0",
        alpha=0.95,
        label="Matched controls",
    )
    ax_a.plot(
        frequency,
        np.nanmean(control_array, axis=0),
        color=NEUTRAL,
        lw=1.0,
        label="Control mean",
    )
    ax_a.axvline(1.5, color="#9A7134", lw=0.75, ls=(0, (3, 2)))
    ax_a.set(
        xlim=(0.0, 5.7),
        xlabel="Frequency (THz)",
        ylabel=r"Normalized $C v^2$ density (THz$^{-1}$)",
        title=r"Spectral $C v^2$ distribution",
    )

    velocity_ratio = np.asarray(
        [[number(row["velocity_ratio"]) for row in grouped[label]] for label in labels]
    )
    finite = velocity_ratio[np.isfinite(velocity_ratio)]
    vmin = max(0.0, 0.05 * math.floor(float(finite.min()) / 0.05))
    vmax = min(1.0, 0.05 * math.ceil(float(finite.max()) / 0.05))
    positions = np.arange(len(angles), dtype=float)
    mesh = ax_b.pcolormesh(
        coordinate_edges(frequency),
        coordinate_edges(positions),
        velocity_ratio,
        shading="flat",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax_b.axvline(1.5, color="white", lw=0.75, ls=(0, (3, 2)), alpha=0.9)
    ax_b.set(
        xlim=(0.0, 5.7),
        xlabel="Frequency (THz)",
        ylabel=r"Twist angle $\theta$ (deg)",
        yticks=positions,
        yticklabels=[
            (f"{angle:.2f}*" if label == "7p61deg" else f"{angle:.2f}")
            for angle, label in zip(angles, labels)
        ],
        title="Frequency-resolved velocity ratio",
    )
    cbar = fig.colorbar(mesh, ax=ax_b, pad=0.02, fraction=0.052)
    cbar.set_label(r"$v_{\rm rms}^{\rm tw}/v_{\rm rms}^{\rm ctl}$")
    cbar.set_ticks(np.linspace(vmin, vmax, 5))

    representative = ("3p18deg", "4p78deg", "8p77deg")
    for label in representative:
        values = grouped[label]
        angle = label_to_angle[label]
        color = angle_color(angle, amin, amax)
        x = np.asarray([number(row["bin_high_thz"]) for row in values])
        twist = np.cumsum([number(row["twist_heat_capacity_share"]) for row in values])
        control = np.cumsum(
            [
                0.5
                * (
                    number(row["control_lower_heat_capacity_share"])
                    + number(row["control_upper_heat_capacity_share"])
                )
                for row in values
            ]
        )
        ax_c.plot(x, twist, color=color, lw=1.2)
        ax_c.plot(x, control, color=color, lw=0.9, ls=(0, (3, 2)))
    ax_c.axvline(1.5, color="#9A7134", lw=0.75, ls=(0, (3, 2)))
    ax_c.set(
        xlim=(0.0, 5.7),
        ylim=(0.0, 1.02),
        xlabel="Frequency (THz)",
        ylabel="Cumulative heat-capacity share",
        title="Thermal mode population",
    )
    ax_c.legend(
        handles=(
            Line2D([0], [0], color=CONTROL, lw=1.1, label="Twisted"),
            Line2D([0], [0], color=CONTROL, lw=0.9, ls=(0, (3, 2)), label="Control"),
        ),
        fontsize=5.9,
        loc="lower right",
        handlelength=1.7,
    )

    angle_handles = [
        Line2D(
            [0],
            [0],
            color=(ACCENT if label == "7p61deg" else angle_color(angle, amin, amax)),
            lw=(0.9 if label == "7p61deg" else 1.2),
            ls=((0, (2, 1.5)) if label == "7p61deg" else "-"),
            label=(
                rf"${angle:.2f}^\circ$ relaxation-sensitive ref."
                if label == "7p61deg"
                else rf"${angle:.2f}^\circ$"
            ),
        )
        for angle, label in sorted(zip(angles, labels))
    ]
    fig.legend(
        handles=[
            Patch(facecolor="#D9DDE0", edgecolor="none", label="Control range"),
            Line2D([0], [0], color=NEUTRAL, lw=1.0, label="Control mean"),
            *angle_handles,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=9,
        fontsize=5.35,
        handlelength=1.45,
        columnspacing=0.7,
    )

    for label, axis in zip("abc", (ax_a, ax_b, ax_c)):
        panel_label(axis, label, x=-0.13, y=1.06)
        axis.grid(axis="y", color="#E7E8EA", lw=0.45, zorder=0)

    write_csv(SOURCE_DIR / "Fig3_frequency_resolved.csv", spectral)
    save_figure(fig, FIGURE_DIR / "Fig3_frequency_resolved_origin")


def main() -> int:
    global DATA_ROOT, FIGURE_DIR, SOURCE_DIR
    args = parse_args()
    DATA_ROOT = args.data_root.resolve()
    FIGURE_DIR = args.figure_dir.resolve()
    SOURCE_DIR = args.source_data_dir.resolve()
    thermal, spectral, stability, asr, path_density = load_state(
        args.analysis_dir.resolve()
    )
    make_figure_2(thermal)
    make_figure_3(thermal, spectral)
    write_csv(SOURCE_DIR / "TableA4_low_frequency_stability.csv", stability)
    write_csv(SOURCE_DIR / "TableA5_asr_sensitivity.csv", asr)
    write_csv(SOURCE_DIR / "TableA6_path_density_sensitivity.csv", path_density)
    print(
        json.dumps(
            {
                "figure_2": str(FIGURE_DIR / "Fig2_phonons_and_velocity_scale.pdf"),
                "figure_3": str(FIGURE_DIR / "Fig3_frequency_resolved_origin.pdf"),
                "stability_table": str(SOURCE_DIR / "TableA4_low_frequency_stability.csv"),
                "asr_table": str(SOURCE_DIR / "TableA5_asr_sensitivity.csv"),
                "path_density_table": str(
                    SOURCE_DIR / "TableA6_path_density_sensitivity.csv"
                ),
                "angle_count": len({row["label"] for row in thermal}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

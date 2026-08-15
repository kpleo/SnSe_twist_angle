#!/usr/bin/env python3
"""Quantify the small-angle structural and phonon crossover in bilayer SnSe.

The analysis uses the six equilibrium structures, retains the original
7.61-degree structure as a separately marked relaxation-sensitive reference,
and reads the curated source data behind Figs. 2 and 3.  It measures relaxation
amplitudes, compares low-order relaxation textures in reduced moire coordinates,
and tests whether the frequency-resolved velocity-renormalization profiles
converge at small angles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from scipy.optimize import curve_fit


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "work"
DATA_ROOTS = (DATA_ROOT,)
EVIDENCE_DIR = REPO / "outputs" / "analysis" / "small_angle_crossover"
FIGURE_DIR = REPO / "outputs" / "figures"
SOURCE_DIR = REPO / "outputs" / "source_data"

EXPECTED_LABELS = (
    "8p77deg",
    "7p61deg",
    "6p02deg",
    "4p78deg",
    "3p82deg",
    "3p58deg",
    "3p18deg",
)
SMALL_ANGLE_LABELS = ("3p82deg", "3p58deg", "3p18deg")
SPECIAL_LABEL = "7p61deg"
EQUILIBRIUM_LABELS = tuple(
    label for label in EXPECTED_LABELS if label != SPECIAL_LABEL
)

NEUTRAL = "#383B3E"
CONTROL = "#73777B"
SIGNAL = "#2F6F8F"
ACCENT = "#B64E3B"
GOLD = "#C1873B"
GREEN = "#5F8069"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in DATA_ROOTS:
        manifest_path = root / "manifests" / "initial_structures.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload["structures"]:
            row = dict(item)
            row["data_root"] = str(root)
            row["manifest"] = str(manifest_path)
            rows.append(row)
    rows_by_label = {str(row["label"]): row for row in rows}
    if tuple(rows_by_label) != EXPECTED_LABELS:
        if set(rows_by_label) != set(EXPECTED_LABELS):
            raise ValueError(
                f"Expected labels {EXPECTED_LABELS}, found {tuple(rows_by_label)}"
            )
    return [rows_by_label[label] for label in EXPECTED_LABELS]


def periodic_design(xy: np.ndarray, order: int) -> np.ndarray:
    terms = [np.ones(len(xy), dtype=float)]
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            if m == 0 and n == 0:
                continue
            if m < 0 or (m == 0 and n < 0):
                continue
            phase = 2.0 * np.pi * (m * xy[:, 0] + n * xy[:, 1])
            terms.extend((np.cos(phase), np.sin(phase)))
    return np.column_stack(terms)


def low_order_texture(
    fractional_xy: np.ndarray,
    displacement_xy: np.ndarray,
    lower_count: int,
    order: int = 1,
    grid_size: int = 48,
) -> np.ndarray:
    axis = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    grid_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_design = periodic_design(grid_xy, order)
    fields: list[np.ndarray] = []
    for indices in (
        np.arange(lower_count),
        np.arange(lower_count, len(fractional_xy)),
    ):
        layer_xy = fractional_xy[indices]
        layer_u = displacement_xy[indices].copy()
        layer_u -= np.mean(layer_u, axis=0)
        coefficients, _, _, _ = np.linalg.lstsq(
            periodic_design(layer_xy, order), layer_u, rcond=None
        )
        fields.append(grid_design @ coefficients)
    return np.concatenate(fields, axis=0).ravel()


def structure_metrics(row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    root = Path(row["data_root"])
    label = str(row["label"])
    initial_path = root / "structures" / "initial" / label / "initial.extxyz"
    relaxed_path = (
        root
        / "relaxations"
        / label
        / "confirmed_fmax0p001"
        / "relaxed.extxyz"
    )
    initial = read(initial_path)
    relaxed = read(relaxed_path)
    if len(initial) != int(row["atom_count"]) or len(relaxed) != len(initial):
        raise ValueError(f"Atom-count mismatch for {label}")
    if initial.get_chemical_symbols() != relaxed.get_chemical_symbols():
        raise ValueError(f"Atom ordering changed for {label}")
    if not np.allclose(initial.cell.array, relaxed.cell.array, atol=1.0e-8):
        raise ValueError(f"Cell changed during fixed-cell relaxation for {label}")

    delta_fractional = (
        relaxed.get_scaled_positions(wrap=False)
        - initial.get_scaled_positions(wrap=False)
    )
    delta_fractional -= np.rint(delta_fractional)
    displacement = delta_fractional @ initial.cell.array
    displacement -= np.mean(displacement, axis=0)
    xy_magnitude = np.linalg.norm(displacement[:, :2], axis=1)

    lower_count = int(row["lower_atom_count"])
    layer_slices = (slice(0, lower_count), slice(lower_count, len(initial)))
    nonaffine = displacement.copy()
    layer_means = []
    for layer_slice in layer_slices:
        layer_mean = np.mean(displacement[layer_slice], axis=0)
        layer_means.append(layer_mean)
        nonaffine[layer_slice] -= layer_mean
    relative_shift = layer_means[1] - layer_means[0]

    initial_fractional = initial.get_scaled_positions(wrap=True)
    texture = low_order_texture(
        initial_fractional[:, :2], displacement[:, :2], lower_count
    )
    texture_xy = texture.reshape(-1, 2)
    texture_rms = float(np.sqrt(np.mean(np.sum(texture_xy**2, axis=1))))

    relaxed_z = relaxed.positions[:, 2]
    lower_center = float(np.mean(relaxed_z[:lower_count]))
    upper_center = float(np.mean(relaxed_z[lower_count:]))
    return (
        {
            "label": label,
            "angle_deg": float(row["angle_deg"]),
            "atom_count": len(initial),
            "rms_inplane_displacement_angstrom": float(
                np.sqrt(np.mean(xy_magnitude**2))
            ),
            "max_inplane_displacement_angstrom": float(np.max(xy_magnitude)),
            "rms_out_of_plane_displacement_angstrom": float(
                np.sqrt(np.mean(displacement[:, 2] ** 2))
            ),
            "rms_inplane_nonaffine_displacement_angstrom": float(
                np.sqrt(np.mean(np.sum(nonaffine[:, :2] ** 2, axis=1)))
            ),
            "rms_out_of_plane_corrugation_change_angstrom": float(
                np.sqrt(np.mean(nonaffine[:, 2] ** 2))
            ),
            "relative_layer_shift_xy_angstrom": float(
                np.linalg.norm(relative_shift[:2])
            ),
            "low_order_texture_rms_angstrom": texture_rms,
            "relaxed_layer_center_separation_angstrom": upper_center - lower_center,
            "initial_structure_sha256": sha256_file(initial_path),
            "relaxed_structure_sha256": sha256_file(relaxed_path),
        },
        texture,
    )


def saturation_model(
    angle_deg: np.ndarray,
    asymptote: float,
    half_angle_deg: float,
    exponent: float,
) -> np.ndarray:
    return asymptote / (1.0 + (angle_deg / half_angle_deg) ** exponent)


def fit_relaxation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fit_rows = [row for row in rows if row["label"] != SPECIAL_LABEL]
    angle = np.asarray([float(row["angle_deg"]) for row in fit_rows])
    displacement = np.asarray(
        [float(row["rms_inplane_displacement_angstrom"]) for row in fit_rows]
    )
    parameters, covariance = curve_fit(
        saturation_model,
        angle,
        displacement,
        p0=(0.60, 7.5, 2.0),
        bounds=((0.45, 0.1, 0.1), (2.0, 30.0, 8.0)),
        maxfev=100000,
    )
    predicted = saturation_model(angle, *parameters)
    leave_one_out = []
    for omitted in range(len(angle)):
        keep = np.arange(len(angle)) != omitted
        values, _ = curve_fit(
            saturation_model,
            angle[keep],
            displacement[keep],
            p0=parameters,
            bounds=((0.45, 0.1, 0.1), (2.0, 30.0, 8.0)),
            maxfev=100000,
        )
        leave_one_out.append(values)
    loo = np.asarray(leave_one_out)
    standard_error = np.sqrt(np.diag(covariance))
    return {
        "model": "u_rms(theta)=u_inf/[1+(theta/theta_half)^m]",
        "included_labels": [str(row["label"]) for row in fit_rows],
        "excluded_labels": [SPECIAL_LABEL],
        "asymptote_angstrom": float(parameters[0]),
        "half_saturation_angle_deg": float(parameters[1]),
        "exponent": float(parameters[2]),
        "formal_standard_errors": {
            "asymptote_angstrom": float(standard_error[0]),
            "half_saturation_angle_deg": float(standard_error[1]),
            "exponent": float(standard_error[2]),
        },
        "leave_one_out_ranges": {
            "asymptote_angstrom": [float(np.min(loo[:, 0])), float(np.max(loo[:, 0]))],
            "half_saturation_angle_deg": [
                float(np.min(loo[:, 1])),
                float(np.max(loo[:, 1])),
            ],
            "exponent": [float(np.min(loo[:, 2])), float(np.max(loo[:, 2]))],
        },
        "rms_residual_angstrom": float(
            np.sqrt(np.mean((predicted - displacement) ** 2))
        ),
    }


def adjacent_structure_similarity(
    rows: list[dict[str, Any]], textures: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    result = []
    equilibrium_rows = [row for row in rows if row["label"] != SPECIAL_LABEL]
    for higher, lower in zip(equilibrium_rows[:-1], equilibrium_rows[1:]):
        high_texture = textures[str(higher["label"])]
        low_texture = textures[str(lower["label"])]
        correlation = float(np.corrcoef(high_texture, low_texture)[0, 1])
        amplitude_ratio = float(
            np.sqrt(np.dot(low_texture, low_texture) / np.dot(high_texture, high_texture))
        )
        result.append(
            {
                "higher_angle_label": higher["label"],
                "higher_angle_deg": float(higher["angle_deg"]),
                "lower_angle_label": lower["label"],
                "lower_angle_deg": float(lower["angle_deg"]),
                "angle_midpoint_deg": 0.5
                * (float(higher["angle_deg"]) + float(lower["angle_deg"])),
                "low_order_texture_pearson_r": correlation,
                "low_order_texture_rms_amplitude_ratio": amplitude_ratio,
            }
        )
    return result


def spectral_profiles() -> tuple[dict[str, dict[float, float]], list[dict[str, Any]]]:
    source = read_csv(SOURCE_DIR / "Fig3_frequency_resolved.csv")
    profiles: dict[str, dict[float, float]] = {}
    for label in EXPECTED_LABELS:
        selected = [row for row in source if row["label"] == label]
        profiles[label] = {
            float(row["bin_mid_thz"]): float(row["velocity_ratio"])
            for row in selected
            if row["velocity_ratio"] not in ("", "None")
        }

    comparisons = []
    manifest_rows = load_manifest_rows()
    by_label = {str(row["label"]): row for row in manifest_rows}
    for higher_label, lower_label in zip(
        EQUILIBRIUM_LABELS[:-1], EQUILIBRIUM_LABELS[1:]
    ):
        common = sorted(set(profiles[higher_label]) & set(profiles[lower_label]))
        values = [
            (
                frequency,
                profiles[higher_label][frequency],
                profiles[lower_label][frequency],
            )
            for frequency in common
            if frequency <= 5.625
            and np.isfinite(profiles[higher_label][frequency])
            and np.isfinite(profiles[lower_label][frequency])
        ]
        high = np.asarray([value[1] for value in values])
        low = np.asarray([value[2] for value in values])
        difference = low - high
        higher_angle = float(by_label[higher_label]["angle_deg"])
        lower_angle = float(by_label[lower_label]["angle_deg"])
        comparisons.append(
            {
                "higher_angle_label": higher_label,
                "higher_angle_deg": higher_angle,
                "lower_angle_label": lower_label,
                "lower_angle_deg": lower_angle,
                "angle_midpoint_deg": 0.5 * (higher_angle + lower_angle),
                "n_frequency_bins": len(values),
                "velocity_ratio_profile_pearson_r": float(
                    np.corrcoef(high, low)[0, 1]
                ),
                "velocity_ratio_profile_mae": float(np.mean(np.abs(difference))),
                "velocity_ratio_profile_rmse": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "velocity_ratio_profile_max_abs_difference": float(
                    np.max(np.abs(difference))
                ),
            }
        )
    return profiles, comparisons


def thermal_metrics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(SOURCE_DIR / "Fig2_angle_response.csv")
    rows = [
        row
        for row in source
        if float(row["temperature_k"]) == 300.0
        and np.isclose(float(row["cutoff_thz"]), 0.05)
    ]
    rows.sort(key=lambda row: float(row["angle_deg"]), reverse=True)
    if tuple(row["label"] for row in rows) != EXPECTED_LABELS:
        raise ValueError("The 300-K Fig. 2 source data do not contain the expected series")
    compact = [
        {
            "label": row["label"],
            "angle_deg": float(row["angle_deg"]),
            "twist_cv2_mean_proxy": float(row["twist_p_m2_per_s2"]),
            "control_mean_cv2_mean_proxy": float(row["control_mean_p_m2_per_s2"]),
            "cv2_twist_over_control_mean": float(row["p_twist_over_control_mean"]),
            "twist_vrms_proxy_m_per_s": float(row["twist_vrms_m_per_s"]),
            "control_mean_vrms_proxy_m_per_s": float(
                row["control_mean_vrms_m_per_s"]
            ),
            "vrms_twist_over_control_mean": float(row["vrms_twist_over_control_mean"]),
        }
        for row in rows
    ]
    small = [row for row in compact if row["label"] in SMALL_ANGLE_LABELS]
    cv2_ratio = np.asarray([row["cv2_twist_over_control_mean"] for row in small])
    vrms_ratio = np.asarray([row["vrms_twist_over_control_mean"] for row in small])
    plateau = {
        "labels": list(SMALL_ANGLE_LABELS),
        "angle_range_deg": [
            float(min(row["angle_deg"] for row in small)),
            float(max(row["angle_deg"] for row in small)),
        ],
        "cv2_ratio_mean": float(np.mean(cv2_ratio)),
        "cv2_ratio_sample_sd": float(np.std(cv2_ratio, ddof=1)),
        "cv2_ratio_range": [float(np.min(cv2_ratio)), float(np.max(cv2_ratio))],
        "cv2_ratio_relative_range": float(np.ptp(cv2_ratio) / np.mean(cv2_ratio)),
        "vrms_ratio_mean": float(np.mean(vrms_ratio)),
        "vrms_ratio_sample_sd": float(np.std(vrms_ratio, ddof=1)),
        "vrms_ratio_range": [float(np.min(vrms_ratio)), float(np.max(vrms_ratio))],
        "vrms_ratio_relative_range": float(np.ptp(vrms_ratio) / np.mean(vrms_ratio)),
    }
    return compact, plateau


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def make_figure(
    structure_rows: list[dict[str, Any]],
    fit: dict[str, Any],
    profiles: dict[str, dict[float, float]],
    structure_similarity: list[dict[str, Any]],
    spectral_similarity: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), constrained_layout=True)
    fit_rows = [row for row in structure_rows if row["label"] != SPECIAL_LABEL]
    special_row = next(row for row in structure_rows if row["label"] == SPECIAL_LABEL)
    angle = np.asarray([row["angle_deg"] for row in fit_rows])
    displacement = np.asarray([row["rms_inplane_displacement_angstrom"] for row in fit_rows])
    fit_angle = np.linspace(float(np.min(angle)), float(np.max(angle)), 300)
    fit_value = saturation_model(
        fit_angle,
        fit["asymptote_angstrom"],
        fit["half_saturation_angle_deg"],
        fit["exponent"],
    )
    axes[0].axvspan(3.10, 3.90, color=SIGNAL, alpha=0.08, lw=0)
    axes[0].plot(fit_angle, fit_value, color=CONTROL, lw=1.1, ls=(0, (4, 2)))
    axes[0].plot(
        angle,
        displacement,
        "o-",
        color=ACCENT,
        lw=1.25,
        ms=4.0,
        label="Six equilibrium structures",
    )
    axes[0].plot(
        [special_row["angle_deg"]],
        [special_row["rms_inplane_displacement_angstrom"]],
        marker="D",
        ls="none",
        ms=4.3,
        mfc="white",
        mec=ACCENT,
        mew=1.0,
        label=r"$7.61^\circ$ excluded from fit",
    )
    axes[0].axhline(
        fit["asymptote_angstrom"], color=NEUTRAL, lw=0.65, ls=(0, (2, 2))
    )
    axes[0].text(
        8.55,
        fit["asymptote_angstrom"] + 0.012,
        r"$u_\infty$",
        ha="right",
        va="bottom",
        color=NEUTRAL,
    )
    axes[0].set(
        xlabel=r"Twist angle $\theta$ (deg)",
        ylabel=r"RMS in-plane relaxation ($\mathrm{\AA}$)",
        title="Approach to structural saturation",
        xlim=(3.0, 9.0),
        ylim=(0.20, 0.64),
    )
    axes[0].legend(fontsize=5.8, loc="lower left", handlelength=1.4)

    profile_colors = {
        "4p78deg": CONTROL,
        "3p82deg": SIGNAL,
        "3p58deg": GREEN,
        "3p18deg": ACCENT,
    }
    profile_styles = {"4p78deg": (0, (4, 2))}
    angle_lookup = {row["label"]: row["angle_deg"] for row in structure_rows}
    for label in ("4p78deg", "3p82deg", "3p58deg", "3p18deg"):
        frequencies = np.asarray(sorted(profiles[label]))
        values = np.asarray([profiles[label][value] for value in frequencies])
        axes[1].plot(
            frequencies,
            values,
            color=profile_colors[label],
            lw=1.15,
            ls=profile_styles.get(label, "-"),
            label=rf"${angle_lookup[label]:.2f}^\circ$",
        )
    axes[1].set(
        xlabel="Frequency (THz)",
        ylabel=r"$v_{\mathrm{rms}}^{\mathrm{tw}}/v_{\mathrm{rms}}^{\mathrm{ctl}}$",
        title="Converging spectral renormalization",
        xlim=(0.0, 5.75),
        ylim=(0.0, 1.0),
    )
    axes[1].legend(fontsize=6.1, ncol=2, columnspacing=0.8, handlelength=1.8)

    structure_by_pair = {
        (row["higher_angle_label"], row["lower_angle_label"]): row
        for row in structure_similarity
    }
    spectral_by_pair = {
        (row["higher_angle_label"], row["lower_angle_label"]): row
        for row in spectral_similarity
    }
    pairs = list(structure_by_pair)
    higher_angle = np.asarray(
        [structure_by_pair[pair]["higher_angle_deg"] for pair in pairs]
    )
    texture_r = np.asarray(
        [structure_by_pair[pair]["low_order_texture_pearson_r"] for pair in pairs]
    )
    spectrum_r = np.asarray(
        [spectral_by_pair[pair]["velocity_ratio_profile_pearson_r"] for pair in pairs]
    )
    axes[2].axvspan(3.10, 3.90, color=SIGNAL, alpha=0.08, lw=0)
    axes[2].plot(
        higher_angle,
        texture_r,
        "o-",
        color=GOLD,
        lw=1.2,
        ms=3.8,
        label="Relaxation texture",
    )
    axes[2].plot(
        higher_angle,
        spectrum_r,
        "s-",
        color=SIGNAL,
        lw=1.2,
        ms=3.5,
        label="Velocity spectrum",
    )
    axes[2].set(
        xlabel="Higher angle in adjacent pair (deg)",
        ylabel="Pearson correlation",
        title="Self-similarity across angles",
        xlim=(3.0, 9.0),
        ylim=(0.80, 1.003),
    )
    axes[2].legend(fontsize=6.1, loc="lower left")

    for label, axis in zip("abc", axes):
        panel_label(axis, label)
        axis.grid(axis="y", color="#E7E8EA", lw=0.45, zorder=0)

    for destination in (EVIDENCE_DIR / "small_angle_crossover", FIGURE_DIR / "FigA4_small_angle_crossover"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(destination.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global DATA_ROOT, DATA_ROOTS, EVIDENCE_DIR, FIGURE_DIR, SOURCE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--additional-root", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    DATA_ROOTS = (
        (DATA_ROOT, args.additional_root.resolve())
        if args.additional_root is not None
        else (DATA_ROOT,)
    )
    EVIDENCE_DIR = args.evidence_dir.resolve()
    FIGURE_DIR = args.figure_dir.resolve()
    SOURCE_DIR = args.source_data_dir.resolve()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest_rows()
    structure_rows: list[dict[str, Any]] = []
    textures: dict[str, np.ndarray] = {}
    for row in manifest_rows:
        metrics, texture = structure_metrics(row)
        structure_rows.append(metrics)
        textures[str(row["label"])] = texture

    fit = fit_relaxation(structure_rows)
    structure_similarity = adjacent_structure_similarity(structure_rows, textures)
    profiles, spectral_similarity = spectral_profiles()
    thermal_rows, plateau = thermal_metrics()

    merged_similarity = []
    spectral_by_pair = {
        (row["higher_angle_label"], row["lower_angle_label"]): row
        for row in spectral_similarity
    }
    for row in structure_similarity:
        pair = (row["higher_angle_label"], row["lower_angle_label"])
        merged = dict(row)
        merged.update(
            {
                key: value
                for key, value in spectral_by_pair[pair].items()
                if key not in merged
            }
        )
        merged_similarity.append(merged)

    summary = {
        "description": (
            "Structural-relaxation and frequency-resolved tests of the corrected "
            "six-structure equilibrium crossover in vacuum-separated bilayer SnSe; "
            "the original 7.61-degree structure is retained only as a marked reference."
        ),
        "accepted_interpretation": (
            "The data support a finite-width crossover to a self-similar small-angle "
            "moire-phonon regime, not a singular zero-velocity angle."
        ),
        "transport_response_crossover_bracket_deg": [3.82, 4.78],
        "continuum_relaxation_parameter": (
            "eta_n=|G_n|^2 |V_n|/[C_ijkl^rel q_ni e_nj q_nk e_nl], "
            "with q_n approximately theta z-cross-G_n and eta_n proportional to theta^-2"
        ),
        "small_angle_plateau_300k": plateau,
        "relaxation_fit": fit,
        "structures": structure_rows,
        "adjacent_angle_similarity": merged_similarity,
        "source_files": {
            "fig2_angle_response": str(SOURCE_DIR / "Fig2_angle_response.csv"),
            "fig2_angle_response_sha256": sha256_file(
                SOURCE_DIR / "Fig2_angle_response.csv"
            ),
            "fig3_frequency_resolved": str(
                SOURCE_DIR / "Fig3_frequency_resolved.csv"
            ),
            "fig3_frequency_resolved_sha256": sha256_file(
                SOURCE_DIR / "Fig3_frequency_resolved.csv"
            ),
        },
        "scope": [
            "The velocity quantity is a matched-control-normalized harmonic band-path descriptor.",
            "The fitted half-saturation angle is a structural fit parameter, not a magic angle.",
            "A full thermal-conductivity claim requires Brillouin-zone integration and anharmonic lifetimes.",
        ],
    }

    write_csv(EVIDENCE_DIR / "relaxation_scaling.csv", structure_rows)
    write_csv(EVIDENCE_DIR / "adjacent_angle_self_similarity.csv", merged_similarity)
    write_csv(EVIDENCE_DIR / "thermal_plateau_300k.csv", thermal_rows)
    write_csv(SOURCE_DIR / "FigA4_relaxation_scaling.csv", structure_rows)
    write_csv(SOURCE_DIR / "FigA4_self_similarity.csv", merged_similarity)
    (EVIDENCE_DIR / "small_angle_crossover_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(
        structure_rows,
        fit,
        profiles,
        structure_similarity,
        spectral_similarity,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

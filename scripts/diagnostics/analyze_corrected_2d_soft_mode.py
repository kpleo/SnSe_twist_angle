#!/usr/bin/env python3
"""Characterize a finite-q soft phonon in the corrected SnSe bilayer series."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from phonopy import load


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "work"
PHONON_DIRNAME = "fmax0p001_disp0p01_plusminus"


def parse_qpoint(text: str) -> np.ndarray:
    values = [float(value) for value in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("q point must look like 0.5,0.5,0")
    return np.asarray(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--label", default="7p61deg")
    parser.add_argument("--qpoint", type=parse_qpoint, default=np.array([0.5, 0.5, 0.0]))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode-index", type=int)
    parser.add_argument("--low-mode-count", type=int, default=24)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths_for_case(root: Path, label: str, kind: str) -> tuple[Path, Path, Path]:
    if kind == "twist":
        structure = root / "relaxations" / label / "confirmed_fmax0p001" / "relaxed.extxyz"
        phonon_dir = root / "phonons" / label / PHONON_DIRNAME
    else:
        suffix = "lower_registry_min" if kind == "control_lower" else "upper_registry_min"
        control_label = f"{label}_{suffix}"
        base = root / "matched_controls_registry_min"
        structure = base / "relaxations" / control_label / "confirmed_fmax0p001" / "relaxed.extxyz"
        phonon_dir = base / "phonons" / control_label / PHONON_DIRNAME
    return structure, phonon_dir / "phonopy_full.yaml", phonon_dir / "full_phonon_band.npz"


def participation_ratio(atom_amplitude: np.ndarray) -> float:
    numerator = float(np.sum(atom_amplitude) ** 2)
    denominator = float(len(atom_amplitude) * np.sum(atom_amplitude**2))
    return numerator / max(denominator, 1e-300)


def best_real_representation(vector: np.ndarray) -> np.ndarray:
    phase = -0.5 * np.angle(np.sum(vector**2))
    real_vector = np.real(vector * np.exp(1j * phase))
    if np.linalg.norm(real_vector) < 1e-12:
        real_vector = np.imag(vector)
    return real_vector


def analyze_case(
    root: Path,
    label: str,
    kind: str,
    qpoint: np.ndarray,
    requested_mode: int | None,
    low_mode_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    structure_path, yaml_path, band_path = paths_for_case(root, label, kind)
    for path in (structure_path, yaml_path, band_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    atoms = read(structure_path)
    phonon = load(str(yaml_path))
    phonon.run_qpoints([qpoint], with_eigenvectors=True)
    qpoint_data = phonon.get_qpoints_dict()
    frequencies = np.asarray(qpoint_data["frequencies"][0], dtype=float)
    eigenvector_columns = np.asarray(qpoint_data["eigenvectors"][0])
    if eigenvector_columns.shape != (3 * len(atoms), 3 * len(atoms)):
        raise ValueError(f"unexpected eigenvector shape {eigenvector_columns.shape}")
    modes = eigenvector_columns.T
    column_orthogonality_error = float(
        np.max(np.abs(eigenvector_columns.conj().T @ eigenvector_columns - np.eye(3 * len(atoms))))
    )

    mode_index = int(np.argmin(frequencies) if requested_mode is None else requested_mode)
    if not 0 <= mode_index < len(frequencies):
        raise ValueError(f"mode index {mode_index} is outside [0, {len(frequencies)})")
    mass_weighted_mode = modes[mode_index].reshape((len(atoms), 3))
    masses = atoms.get_masses()
    physical_mode = mass_weighted_mode / np.sqrt(masses[:, None])
    physical_norm = float(np.sqrt(np.sum(np.abs(physical_mode) ** 2)))
    physical_mode /= physical_norm
    mass_atom_amplitude = np.sum(np.abs(mass_weighted_mode) ** 2, axis=1)
    physical_atom_amplitude = np.sum(np.abs(physical_mode) ** 2, axis=1)
    symbols = np.asarray(atoms.get_chemical_symbols())
    layer_ids = np.asarray(atoms.arrays.get("layer_id", np.full(len(atoms), -1)), dtype=int)

    real_mode = best_real_representation(physical_mode)
    real_norms = np.linalg.norm(real_mode, axis=1)
    real_scale = float(np.max(real_norms))
    if real_scale > 0:
        real_mode = real_mode / real_scale

    atom_rows: list[dict[str, Any]] = []
    for index, (position, displacement) in enumerate(zip(atoms.positions, real_mode)):
        atom_rows.append(
            {
                "atom_index_zero_based": index,
                "symbol": str(symbols[index]),
                "layer_id": int(layer_ids[index]),
                "x_angstrom": float(position[0]),
                "y_angstrom": float(position[1]),
                "z_angstrom": float(position[2]),
                "ux_normalized": float(displacement[0]),
                "uy_normalized": float(displacement[1]),
                "uz_normalized": float(displacement[2]),
                "physical_amplitude_weight": float(physical_atom_amplitude[index]),
            }
        )

    amplitude_total = float(np.sum(physical_atom_amplitude))
    z_weight = float(np.sum(np.abs(physical_mode[:, 2]) ** 2))
    summary: dict[str, Any] = {
        "kind": kind,
        "label": label,
        "structure": str(structure_path),
        "structure_sha256": sha256_file(structure_path),
        "phonopy_yaml": str(yaml_path),
        "phonopy_yaml_sha256": sha256_file(yaml_path),
        "band_npz": str(band_path),
        "band_npz_sha256": sha256_file(band_path),
        "atom_count": len(atoms),
        "qpoint_reduced": qpoint.tolist(),
        "selected_mode_index_zero_based": mode_index,
        "selected_frequency_thz": float(frequencies[mode_index]),
        "negative_mode_count_lt_minus_0p05_at_q": int(np.sum(frequencies < -0.05)),
        "lowest_frequencies_thz": np.sort(frequencies)[:low_mode_count].tolist(),
        "mass_weighted_participation_ratio": participation_ratio(mass_atom_amplitude),
        "physical_displacement_participation_ratio": participation_ratio(physical_atom_amplitude),
        "out_of_plane_displacement_fraction": z_weight / amplitude_total,
        "in_plane_displacement_fraction": 1.0 - z_weight / amplitude_total,
        "sn_displacement_fraction": float(np.sum(physical_atom_amplitude[symbols == "Sn"]) / amplitude_total),
        "se_displacement_fraction": float(np.sum(physical_atom_amplitude[symbols == "Se"]) / amplitude_total),
        "eigenvector_column_orthogonality_max_error": column_orthogonality_error,
    }
    if set(layer_ids) == {0, 1}:
        lower = float(np.sum(physical_atom_amplitude[layer_ids == 0]) / amplitude_total)
        upper = float(np.sum(physical_atom_amplitude[layer_ids == 1]) / amplitude_total)
        summary.update(
            {
                "lower_layer_displacement_fraction": lower,
                "upper_layer_displacement_fraction": upper,
                "layer_amplitude_imbalance": abs(lower - upper),
            }
        )

    band = np.load(band_path)
    band_frequencies = np.asarray(band["frequencies"], dtype=float)
    if np.allclose(qpoint, [0.5, 0.5, 0.0]):
        s_rows = [35, 36]
        summary["band_path_s_frequency_thz"] = float(
            np.mean([band_frequencies[row, mode_index] for row in s_rows])
        )
        summary["qpoint_band_frequency_abs_difference_thz"] = abs(
            summary["band_path_s_frequency_thz"] - summary["selected_frequency_thz"]
        )
    return summary, atom_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    summaries: list[dict[str, Any]], atom_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    colors = {"twist": "#a33c36", "control_lower": "#4d7895", "control_upper": "#9b7441"}
    labels = {"twist": "twisted", "control_lower": "lower control", "control_upper": "upper control"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)

    for summary in summaries:
        values = np.asarray(summary["lowest_frequencies_thz"])
        axes[0].plot(
            np.arange(len(values)),
            values,
            marker="o",
            ms=2.7,
            lw=1.0,
            color=colors[summary["kind"]],
            label=labels[summary["kind"]],
        )
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set(xlabel="mode rank at S", ylabel="frequency (THz)")
    axes[0].legend(frameon=False, fontsize=7)

    positions = np.asarray([[row["x_angstrom"], row["y_angstrom"]] for row in atom_rows])
    vectors = np.asarray([[row["ux_normalized"], row["uy_normalized"]] for row in atom_rows])
    amplitudes = np.asarray([row["physical_amplitude_weight"] for row in atom_rows])
    layers = np.asarray([row["layer_id"] for row in atom_rows])
    cutoff = np.quantile(amplitudes, 0.70)
    axes[1].scatter(
        positions[:, 0],
        positions[:, 1],
        s=3.0,
        c=np.where(layers == 0, "#4d7895", "#d3913b"),
        alpha=0.28,
        linewidths=0,
    )
    selected = amplitudes >= cutoff
    axes[1].quiver(
        positions[selected, 0],
        positions[selected, 1],
        vectors[selected, 0],
        vectors[selected, 1],
        amplitudes[selected],
        cmap="magma",
        angles="xy",
        scale_units="xy",
        scale=0.18,
        width=0.0024,
    )
    axes[1].set_aspect("equal")
    axes[1].set(xlabel="x (Angstrom)", ylabel="y (Angstrom)")

    twist = next(summary for summary in summaries if summary["kind"] == "twist")
    metric_labels = ["in plane", "out of plane", "lower layer", "upper layer"]
    metric_values = [
        twist["in_plane_displacement_fraction"],
        twist["out_of_plane_displacement_fraction"],
        twist["lower_layer_displacement_fraction"],
        twist["upper_layer_displacement_fraction"],
    ]
    axes[2].bar(
        np.arange(4),
        metric_values,
        color=["#4d7895", "#d3913b", "#4d7895", "#d3913b"],
        width=0.72,
    )
    axes[2].set_xticks(np.arange(4), metric_labels, rotation=28, ha="right")
    axes[2].set_ylabel("displacement fraction")
    axes[2].set_ylim(0, 1)
    axes[2].text(
        0.04,
        0.95,
        f"PR = {twist['physical_displacement_participation_ratio']:.3f}",
        transform=axes[2].transAxes,
        va="top",
    )
    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.03, f"({chr(97 + index)})", transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", color="0.91", lw=0.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{summaries[0]['label']}_s_point_soft_mode.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "analysis" / "soft_modes" / args.label
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    twist_atoms: list[dict[str, Any]] = []
    for kind in ("twist", "control_lower", "control_upper"):
        summary, atoms = analyze_case(
            root,
            args.label,
            kind,
            args.qpoint,
            args.mode_index if kind == "twist" else None,
            args.low_mode_count,
        )
        summaries.append(summary)
        if kind == "twist":
            twist_atoms = atoms
    write_csv(output_dir / "s_point_mode_summary.csv", summaries)
    write_csv(output_dir / "twist_selected_mode_atoms.csv", twist_atoms)
    payload = {
        "schema_version": 1,
        "description": "Finite-q mode character for a corrected twisted SnSe bilayer and its registry-minimum controls.",
        "qpoint_reduced": args.qpoint.tolist(),
        "eigenvector_convention": "Phonopy eigenvectors are columns; physical displacements are obtained by dividing by square-root atomic mass.",
        "summaries": summaries,
    }
    (output_dir / "s_point_soft_mode_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    plot_results(summaries, twist_atoms, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit a fixed-cell relaxation of a corrected 2D twisted SnSe bilayer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--relaxed", type=Path, required=True)
    parser.add_argument("--relax-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_periodic_distance(atoms) -> float:
    first, second, distances = neighbor_list("ijd", atoms, 4.0)
    mask = first != second
    if not np.any(mask):
        raise ValueError("no periodic neighbor distances found")
    return float(np.min(distances[mask]))


def minimum_interlayer_distance(atoms, layer_ids: np.ndarray) -> float:
    first, second, distances = neighbor_list("ijd", atoms, 6.0)
    mask = layer_ids[first] != layer_ids[second]
    if not np.any(mask):
        raise ValueError("no interlayer neighbors found within 6 Angstrom")
    return float(np.min(distances[mask]))


def sublattice_corrugation(atoms, layer_ids: np.ndarray, primitive: np.ndarray) -> dict:
    rows = []
    for layer in (0, 1):
        for primitive_index in sorted(set(int(value) for value in primitive)):
            values = atoms.positions[
                (layer_ids == layer) & (primitive == primitive_index), 2
            ]
            rows.append(
                {
                    "layer_id": layer,
                    "primitive_index": primitive_index,
                    "atom_count": int(values.size),
                    "z_mean_angstrom": float(np.mean(values)),
                    "z_peak_to_peak_angstrom": float(np.ptp(values)),
                    "z_rms_about_mean_angstrom": float(
                        np.sqrt(np.mean((values - np.mean(values)) ** 2))
                    ),
                }
            )
    return {
        "groups": rows,
        "maximum_peak_to_peak_angstrom": max(
            row["z_peak_to_peak_angstrom"] for row in rows
        ),
        "maximum_rms_angstrom": max(row["z_rms_about_mean_angstrom"] for row in rows),
    }


def main() -> int:
    args = parse_args()
    initial_path = args.initial.resolve()
    relaxed_path = args.relaxed.resolve()
    stats_path = args.relax_stats.resolve()
    output_path = args.output.resolve()
    initial = read(initial_path)
    relaxed = read(relaxed_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if "layer_id" not in initial.arrays or "primitive_index" not in initial.arrays:
        raise SystemExit("initial structure lacks layer provenance arrays")
    layer_ids = np.asarray(initial.arrays["layer_id"], dtype=int)
    primitive = np.asarray(initial.arrays["primitive_index"], dtype=int)
    lower = np.flatnonzero(layer_ids == 0)
    upper = np.flatnonzero(layer_ids == 1)
    if len(lower) == 0 or len(upper) == 0 or set(layer_ids) != {0, 1}:
        raise SystemExit("invalid initial layer partition")
    relaxed_layer_ids = np.asarray(relaxed.arrays.get("layer_id", []), dtype=int)
    relaxed_primitive = np.asarray(
        relaxed.arrays.get("primitive_index", []), dtype=int
    )

    cell_delta = np.asarray(relaxed.cell.array) - np.asarray(initial.cell.array)
    displacements = relaxed.positions - initial.positions
    fractional = np.linalg.solve(initial.cell.array.T, displacements.T).T
    fractional[:, :2] -= np.round(fractional[:, :2])
    displacements = fractional @ initial.cell.array
    z_lower = relaxed.positions[lower, 2]
    z_upper = relaxed.positions[upper, 2]
    direct_gap = float(np.min(z_upper) - np.max(z_lower))
    slab_span = float(np.max(relaxed.positions[:, 2]) - np.min(relaxed.positions[:, 2]))
    vacuum = float(relaxed.cell.lengths()[2] - slab_span)
    center_separation = float(np.mean(z_upper) - np.mean(z_lower))
    corrugation = sublattice_corrugation(relaxed, layer_ids, primitive)

    diagnostics = {
        "atom_count": len(relaxed),
        "lower_atom_count": len(lower),
        "upper_atom_count": len(upper),
        "cell_max_abs_change_angstrom": float(np.max(np.abs(cell_delta))),
        "rms_atomic_displacement_angstrom": float(
            np.sqrt(np.mean(np.sum(displacements**2, axis=1)))
        ),
        "maximum_atomic_displacement_angstrom": float(
            np.max(np.linalg.norm(displacements, axis=1))
        ),
        "rms_z_displacement_angstrom": float(
            np.sqrt(np.mean(displacements[:, 2] ** 2))
        ),
        "layer_center_separation_angstrom": center_separation,
        "direct_interlayer_z_gap_angstrom": direct_gap,
        "minimum_interlayer_distance_angstrom": minimum_interlayer_distance(
            relaxed, layer_ids
        ),
        "minimum_periodic_distance_angstrom": minimum_periodic_distance(relaxed),
        "periodic_vacuum_gap_angstrom": vacuum,
        "corrugation": corrugation,
    }
    target = float(stats["fmax_target_ev_per_ang"])
    gates = {
        "relaxation_converged": stats.get("converged") is True,
        "final_force_within_target": float(stats["final_max_force_ev_per_ang"])
        <= 1.01 * target,
        "energy_not_increased": float(stats["final_energy_ev"])
        <= float(stats["initial_energy_ev"]) + 1.0e-8,
        "atom_count_preserved": len(initial) == len(relaxed),
        "symbols_preserved": initial.get_chemical_symbols()
        == relaxed.get_chemical_symbols(),
        "cell_preserved": diagnostics["cell_max_abs_change_angstrom"] <= 1.0e-10,
        "layer_provenance_preserved": (
            np.array_equal(relaxed_layer_ids, layer_ids)
            and np.array_equal(relaxed_primitive, primitive)
        ),
        "layers_do_not_cross": direct_gap >= 2.0,
        "minimum_interlayer_distance_ge_2p30": diagnostics[
            "minimum_interlayer_distance_angstrom"
        ]
        >= 2.30,
        "minimum_periodic_distance_ge_2p30": diagnostics[
            "minimum_periodic_distance_angstrom"
        ]
        >= 2.30,
        "layer_center_separation_physical": 5.0 <= center_separation <= 6.8,
        "vacuum_gap_ge_23": vacuum >= 23.0,
        "sublattice_corrugation_le_1p5": corrugation[
            "maximum_peak_to_peak_angstrom"
        ]
        <= 1.5,
    }
    gates["all_pass"] = all(bool(value) for value in gates.values())
    result = {
        "schema_version": 1,
        "initial": str(initial_path),
        "initial_sha256": sha256_file(initial_path),
        "relaxed": str(relaxed_path),
        "relaxed_sha256": sha256_file(relaxed_path),
        "relax_stats": str(stats_path),
        "relax_stats_sha256": sha256_file(stats_path),
        "relaxation": stats,
        "diagnostics": diagnostics,
        "gates": gates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gates["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

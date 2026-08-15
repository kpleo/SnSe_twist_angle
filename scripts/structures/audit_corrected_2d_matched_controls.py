#!/usr/bin/env python3
"""Independently audit corrected SnSe registry-matched control structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_distance(atoms, layer_ids: np.ndarray | None = None) -> float:
    first, second, distances = neighbor_list("ijd", atoms, 6.5)
    valid = first != second
    if layer_ids is not None:
        valid &= layer_ids[first] != layer_ids[second]
    if not np.any(valid):
        raise RuntimeError("no valid periodic neighbors found")
    return float(np.min(distances[valid]))


def audit_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    structure_path = root / str(row["extxyz"])
    source_path = Path(str(row["source_relaxed"]))
    atoms = read(structure_path)
    source = read(source_path)
    atom_count = len(atoms)
    half = atom_count // 2
    layer_ids = np.asarray(atoms.arrays.get("layer_id", []), dtype=int)
    primitive = np.asarray(atoms.arrays.get("primitive_index", []), dtype=int)
    source_indices = np.asarray(
        atoms.arrays.get("source_twist_atom_index", []), dtype=int
    )
    control_source = np.asarray(
        atoms.arrays.get("control_source_layer", []), dtype=int
    )

    required_array_lengths_pass = all(
        len(values) == atom_count
        for values in (layer_ids, primitive, source_indices, control_source)
    )
    source_hash_pass = (
        source_path.is_file()
        and sha256_file(source_path) == row["source_relaxed_sha256"]
    )
    extxyz_hash_pass = (
        structure_path.is_file()
        and sha256_file(structure_path) == row["extxyz_sha256"]
    )
    cell_difference = float(
        np.max(np.abs(np.asarray(atoms.cell) - np.asarray(source.cell)))
    )

    paired_species_pass = False
    paired_provenance_pass = False
    registry_max_residual = float("inf")
    if atom_count % 2 == 0 and required_array_lengths_pass:
        symbols = np.asarray(atoms.get_chemical_symbols())
        paired_species_pass = bool(np.array_equal(symbols[:half], symbols[half:]))
        paired_provenance_pass = bool(
            np.array_equal(source_indices[:half], source_indices[half:])
            and np.array_equal(primitive[:half], primitive[half:])
        )
        expected_shift = np.asarray(
            row["registry_shift_cartesian_angstrom"], dtype=float
        )
        source_layer_ids = np.asarray(source.arrays["layer_id"], dtype=int)
        expected_shift[2] += float(
            np.mean(source.positions[source_layer_ids == 1, 2])
            - np.mean(source.positions[source_layer_ids == 0, 2])
        )
        residual = atoms.positions[half:] - atoms.positions[:half] - expected_shift
        fractional = np.linalg.solve(np.asarray(atoms.cell).T, residual.T).T
        fractional -= np.round(fractional)
        registry_max_residual = float(
            np.max(np.linalg.norm(fractional @ np.asarray(atoms.cell), axis=1))
        )

    lower = layer_ids == 0 if required_array_lengths_pass else np.zeros(atom_count, bool)
    upper = layer_ids == 1 if required_array_lengths_pass else np.zeros(atom_count, bool)
    source_layer_ids = np.asarray(source.arrays.get("layer_id", []), dtype=int)
    source_centers = [
        float(np.mean(source.positions[source_layer_ids == layer, 2]))
        for layer in (0, 1)
    ]
    control_centers = [
        float(np.mean(atoms.positions[mask, 2])) if np.any(mask) else float("nan")
        for mask in (lower, upper)
    ]
    layer_center_residual = float(
        np.max(np.abs(np.asarray(control_centers) - np.asarray(source_centers)))
    )
    slab_span = float(np.ptp(atoms.positions[:, 2]))
    diagnostics = {
        "atom_count": atom_count,
        "lower_atom_count": int(np.sum(lower)),
        "upper_atom_count": int(np.sum(upper)),
        "cell_max_abs_difference_from_source_angstrom": cell_difference,
        "layer_center_separation_angstrom": control_centers[1]
        - control_centers[0],
        "layer_center_max_abs_residual_from_source_angstrom": layer_center_residual,
        "direct_interlayer_z_gap_angstrom": float(
            np.min(atoms.positions[upper, 2]) - np.max(atoms.positions[lower, 2])
        ),
        "minimum_periodic_distance_angstrom": minimum_distance(atoms),
        "minimum_interlayer_distance_angstrom": minimum_distance(atoms, layer_ids),
        "periodic_vacuum_gap_angstrom": float(atoms.cell.lengths()[2] - slab_span),
        "registry_pair_max_residual_angstrom": registry_max_residual,
    }
    gates = {
        "structure_hash_pass": extxyz_hash_pass,
        "source_hash_pass": source_hash_pass,
        "atom_count_pass": atom_count == int(row["atom_count"]),
        "required_provenance_arrays_pass": required_array_lengths_pass,
        "equal_layer_counts_pass": diagnostics["lower_atom_count"]
        == diagnostics["upper_atom_count"]
        == half,
        "paired_species_pass": paired_species_pass,
        "paired_provenance_pass": paired_provenance_pass,
        "control_source_layer_pass": bool(
            required_array_lengths_pass
            and np.all(control_source == int(row["source_layer_id"]))
        ),
        "cell_matches_source_pass": cell_difference <= 1.0e-10,
        "layer_centers_match_source_pass": layer_center_residual <= 1.0e-8,
        "registry_translation_pass": registry_max_residual <= 1.0e-8,
        "layers_do_not_cross_pass": diagnostics[
            "direct_interlayer_z_gap_angstrom"
        ]
        >= 2.0,
        "minimum_periodic_distance_pass": diagnostics[
            "minimum_periodic_distance_angstrom"
        ]
        >= 2.30,
        "minimum_interlayer_distance_pass": diagnostics[
            "minimum_interlayer_distance_angstrom"
        ]
        >= 2.30,
        "vacuum_gap_pass": diagnostics["periodic_vacuum_gap_angstrom"] >= 23.0,
    }
    gates["all_pass"] = all(bool(value) for value in gates.values())
    return {
        "label": str(row["label"]),
        "matched_twist_label": str(row["matched_twist_label"]),
        "source_layer": str(row["source_layer"]),
        "structure": str(structure_path.resolve()),
        "source": str(source_path.resolve()),
        "diagnostics": diagnostics,
        "gates": gates,
    }


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = root / "manifests" / "initial_structures.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [audit_row(root, row) for row in manifest["structures"]]
    labels = [row["label"] for row in rows]
    labels_unique = len(labels) == len(set(labels))
    all_pass = bool(labels_unique and rows and all(row["gates"]["all_pass"] for row in rows))
    payload = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "structure_count": len(rows),
        "labels_unique_pass": labels_unique,
        "all_pass": all_pass,
        "structures": rows,
    }
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else root / "manifests" / "initial_structure_audit.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else root / "manifests" / "initial_structure_audit.csv"
    )
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    flat_rows = [
        {
            "label": row["label"],
            "matched_twist_label": row["matched_twist_label"],
            "source_layer": row["source_layer"],
            **row["diagnostics"],
            "all_pass": row["gates"]["all_pass"],
        }
        for row in rows
    ]
    with output_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

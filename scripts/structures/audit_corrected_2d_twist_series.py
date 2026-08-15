#!/usr/bin/env python3
"""Independently audit the corrected 2D SnSe commensurate twist structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_ROOT = REPO_DIR / "work"
OLD_REFERENCE_STRUCTURES: dict[str, Path] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minimum-vacuum", type=float, default=24.99)
    parser.add_argument("--topology-tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_layers_largest_z_gap(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(atoms.positions[:, 2])
    gaps = np.diff(atoms.positions[order, 2])
    if len(gaps) == 0:
        raise ValueError("Cannot split a structure with fewer than two atoms")
    split = int(np.argmax(gaps)) + 1
    return order[:split], order[split:]


def pair_metrics(atoms: Atoms) -> dict[str, float]:
    i, j, distances = neighbor_list("ijd", atoms, cutoff=6.0, self_interaction=False)
    unique = i < j
    i = i[unique]
    j = j[unique]
    distances = distances[unique]
    layers = atoms.get_array("layer_id")
    same = layers[i] == layers[j]
    different = ~same
    return {
        "minimum_pair_distance_angstrom": float(np.min(distances)),
        "minimum_intralayer_distance_angstrom": float(np.min(distances[same])),
        "minimum_interlayer_distance_angstrom": float(np.min(distances[different])),
    }


def operation_matrices() -> dict[str, np.ndarray]:
    return {
        "identity": np.eye(2),
        "x_reflection": np.diag([-1.0, 1.0]),
        "y_reflection": np.diag([1.0, -1.0]),
        "c2_z": -np.eye(2),
    }


def topology_fit(generated: Atoms, reference: Atoms) -> dict[str, Any]:
    if len(generated) != len(reference):
        raise ValueError("Topology comparison requires equal atom counts")
    generated_symbols = np.asarray(generated.get_chemical_symbols())
    reference_symbols = np.asarray(reference.get_chemical_symbols())
    if sorted(generated_symbols) != sorted(reference_symbols):
        raise ValueError("Topology comparison requires equal compositions")
    generated_fractional = generated.get_scaled_positions(wrap=True)[:, :2]
    reference_fractional = reference.get_scaled_positions(wrap=True)[:, :2]
    reference_lengths = np.asarray(reference.cell.lengths()[:2], dtype=float)
    species = sorted(set(generated_symbols))
    anchor = 0
    anchor_targets = np.flatnonzero(reference_symbols == generated_symbols[anchor])
    best: dict[str, Any] | None = None

    for operation, matrix in operation_matrices().items():
        transformed = (generated_fractional @ matrix.T) % 1.0
        species_data: dict[str, dict[str, Any]] = {}
        for symbol in species:
            reference_indices = np.flatnonzero(reference_symbols == symbol)
            points = reference_fractional[reference_indices]
            tiled = np.vstack(
                [points + [dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            )
            species_data[symbol] = {
                "reference_indices": reference_indices,
                "tree": cKDTree(tiled * reference_lengths),
                "points_per_tile": len(points),
            }

        for target in anchor_targets:
            shift = reference_fractional[target] - transformed[anchor]
            squared_distances: list[float] = []
            maximum = 0.0
            one_to_one = True
            for symbol in species:
                generated_indices = np.flatnonzero(generated_symbols == symbol)
                query = ((transformed[generated_indices] + shift) % 1.0) * reference_lengths
                data = species_data[symbol]
                distances, tiled_indices = data["tree"].query(query, k=1)
                local_indices = tiled_indices % int(data["points_per_tile"])
                if len(np.unique(local_indices)) != len(local_indices):
                    one_to_one = False
                    break
                squared_distances.extend(np.square(distances).tolist())
                maximum = max(maximum, float(np.max(distances)))
            if not one_to_one:
                continue
            rms = float(math.sqrt(np.mean(squared_distances)))
            key = (rms, maximum, operation)
            if best is None or key < best["_key"]:
                best = {
                    "_key": key,
                    "operation": operation,
                    "fractional_translation": (shift % 1.0).tolist(),
                    "rms_inplane_angstrom": rms,
                    "maximum_inplane_residual_angstrom": maximum,
                    "one_to_one_mapping": True,
                }
    if best is None:
        raise RuntimeError("No one-to-one periodic topology mapping was found")
    best.pop("_key")
    return best


def audit_hashes(root: Path) -> list[dict[str, Any]]:
    ledger = root / "manifests" / "initial_structure_sha256.csv"
    rows: list[dict[str, Any]] = []
    with ledger.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            path = root / record["path"]
            actual = sha256_file(path) if path.is_file() else None
            rows.append(
                {
                    "path": record["path"],
                    "expected_sha256": record["sha256"],
                    "actual_sha256": actual,
                    "pass": actual == record["sha256"],
                }
            )
    return rows


def audit_structure(
    root: Path,
    record: dict[str, Any],
    minimum_vacuum: float,
    topology_tolerance: float,
) -> dict[str, Any]:
    path = root / record["extxyz"]
    atoms = read(path)
    failures: list[str] = []
    expected_atoms = 4 * (
        int(record["lower_primitive_cells"]) + int(record["upper_primitive_cells"])
    )
    if len(atoms) != expected_atoms:
        failures.append(f"atom_count={len(atoms)} expected={expected_atoms}")
    if not bool(np.all(atoms.pbc)):
        failures.append("pbc is not True,True,True")
    cell = np.asarray(atoms.cell.array, dtype=float)
    off_diagonal = cell.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if float(np.max(np.abs(off_diagonal))) > 1.0e-8:
        failures.append("cell is not axis-aligned rectangular")
    if "layer_id" not in atoms.arrays or "primitive_index" not in atoms.arrays:
        failures.append("layer_id or primitive_index array is missing")
        return {"label": record["label"], "path": str(path), "pass": False, "failures": failures}

    layer_ids = atoms.get_array("layer_id")
    lower = np.flatnonzero(layer_ids == 0)
    upper = np.flatnonzero(layer_ids == 1)
    if len(lower) != int(record["lower_atom_count"]) or len(upper) != int(record["upper_atom_count"]):
        failures.append("explicit layer counts disagree with manifest")
    for indices, cells, name in (
        (lower, int(record["lower_primitive_cells"]), "lower"),
        (upper, int(record["upper_primitive_cells"]), "upper"),
    ):
        symbols = atoms[indices].get_chemical_symbols()
        if symbols.count("Sn") != 2 * cells or symbols.count("Se") != 2 * cells:
            failures.append(f"{name} composition is not (Sn2Se2)x{cells}")

    lower_center = float(np.mean(atoms.positions[lower, 2]))
    upper_center = float(np.mean(atoms.positions[upper, 2]))
    separation = upper_center - lower_center
    direct_gap = float(np.min(atoms.positions[upper, 2]) - np.max(atoms.positions[lower, 2]))
    vacuum = float(atoms.cell.lengths()[2] - np.ptp(atoms.positions[:, 2]))
    pairs = pair_metrics(atoms)
    if abs(separation - float(record["initial_layer_center_separation_angstrom"])) > 1.0e-7:
        failures.append("layer-center separation disagrees with manifest")
    if vacuum < minimum_vacuum:
        failures.append(f"periodic vacuum {vacuum:.6f} A is below {minimum_vacuum:.6f} A")
    if not 2.4 < direct_gap < 3.5:
        failures.append(f"direct interlayer z gap {direct_gap:.6f} A is outside the audit window")
    if pairs["minimum_intralayer_distance_angstrom"] < 2.5:
        failures.append("minimum intralayer distance is below 2.5 A")
    if pairs["minimum_interlayer_distance_angstrom"] < 2.5:
        failures.append("minimum interlayer distance is below 2.5 A")

    expected_angle = math.degrees(
        math.atan(math.sqrt(int(record["s"]) * int(record["t"]) / (int(record["p"]) * int(record["q"]))))
    )
    if abs(expected_angle - float(record["angle_deg"])) > 1.0e-10:
        failures.append("angle does not follow the commensurate integer relation")
    lower_det = int(round(abs(np.linalg.det(np.asarray(record["lower_matrix"], dtype=int)))))
    upper_det = int(round(abs(np.linalg.det(np.asarray(record["upper_matrix"], dtype=int)))))
    if lower_det != int(record["lower_primitive_cells"]) or upper_det != int(record["upper_primitive_cells"]):
        failures.append("matrix determinants disagree with primitive-cell counts")

    topology: dict[str, Any] | None = None
    if record["label"] in OLD_REFERENCE_STRUCTURES:
        reference_path = OLD_REFERENCE_STRUCTURES[record["label"]]
        reference = read(reference_path)
        reference.pbc = True
        ref_lower, ref_upper = split_layers_largest_z_gap(reference)
        topology = {
            "reference_path": str(reference_path.resolve()),
            "lower": topology_fit(atoms[lower], reference[ref_lower]),
            "upper": topology_fit(atoms[upper], reference[ref_upper]),
        }
        for name in ("lower", "upper"):
            if topology[name]["maximum_inplane_residual_angstrom"] > topology_tolerance:
                failures.append(
                    f"{name} topology residual {topology[name]['maximum_inplane_residual_angstrom']:.6g} A "
                    f"exceeds {topology_tolerance:.6g} A"
                )

    return {
        "label": record["label"],
        "path": str(path),
        "angle_deg": float(record["angle_deg"]),
        "atom_count": len(atoms),
        "lower_atom_count": len(lower),
        "upper_atom_count": len(upper),
        "cell_lengths_angstrom": atoms.cell.lengths().tolist(),
        "layer_center_separation_angstrom": separation,
        "direct_interlayer_z_gap_angstrom": direct_gap,
        "periodic_vacuum_gap_angstrom": vacuum,
        **pairs,
        "legacy_inplane_topology_check": topology,
        "pass": not failures,
        "failures": failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifests" / "initial_structures.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["structures"]
    audits = [
        audit_structure(root, record, float(args.minimum_vacuum), float(args.topology_tolerance))
        for record in records
    ]
    hash_audits = audit_hashes(root)
    passed = all(row["pass"] for row in audits) and all(row["pass"] for row in hash_audits)
    report = {
        "root": str(root),
        "structure_count": len(audits),
        "all_structures_pass": all(row["pass"] for row in audits),
        "all_recorded_hashes_pass": all(row["pass"] for row in hash_audits),
        "overall_pass": passed,
        "structures": audits,
        "hashes": hash_audits,
    }
    out_json = root / "manifests" / "initial_structure_audit.json"
    out_csv = root / "manifests" / "initial_structure_audit.csv"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(out_csv, audits)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

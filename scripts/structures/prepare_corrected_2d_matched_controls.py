#!/usr/bin/env python3
"""Build periodic untwisted controls matched to corrected SnSe twist cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work"),
    )
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--controls-subdir",
        default="matched_controls_registry_min",
        help="Output directory below --root.",
    )
    parser.add_argument(
        "--registry-u",
        type=float,
        default=1.0 / 3.0,
        help="Upper-layer translation along the source monolayer a vector.",
    )
    parser.add_argument(
        "--registry-v",
        type=float,
        default=0.0,
        help="Upper-layer translation along the source monolayer b vector.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_periodic_distance(atoms: Atoms) -> float:
    first, second, distances = neighbor_list("ijd", atoms, 4.0)
    valid = first != second
    if not np.any(valid):
        raise RuntimeError("no periodic neighbor distances found")
    return float(np.min(distances[valid]))


def minimum_interlayer_distance(atoms: Atoms, layer_ids: np.ndarray) -> float:
    first, second, distances = neighbor_list("ijd", atoms, 6.5)
    valid = layer_ids[first] != layer_ids[second]
    if not np.any(valid):
        raise RuntimeError("no interlayer neighbors found")
    return float(np.min(distances[valid]))


def source_primitive_vectors(
    source_row: dict[str, object], source_layer: int
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(source_row["common_cell_lengths_angstrom"][:2], dtype=float)
    p = int(source_row["p"])
    q = int(source_row["q"])
    s = int(source_row["s"])
    t = int(source_row["t"])
    if source_layer == 0:
        # The common-cell construction reflects y after making a p-by-q cell.
        a_vector = np.array([lengths[0] / p, 0.0, 0.0])
        b_vector = np.array([0.0, -lengths[1] / q, 0.0])
    elif source_layer == 1:
        determinant = p * q + s * t
        # Rows of inv([[p,s],[-t,q]]) mapped to the common rectangular cell,
        # followed by the same y reflection used for the twist structures.
        a_vector = np.array(
            [q * lengths[0] / determinant, s * lengths[1] / determinant, 0.0]
        )
        b_vector = np.array(
            [t * lengths[0] / determinant, -p * lengths[1] / determinant, 0.0]
        )
    else:
        raise ValueError("source_layer must be 0 or 1")
    return a_vector, b_vector


def duplicate_source_layer(
    atoms: Atoms,
    source_layer: int,
    registry_shift: np.ndarray,
) -> Atoms:
    if source_layer not in (0, 1):
        raise ValueError("source_layer must be 0 or 1")
    layer_ids = np.asarray(atoms.arrays["layer_id"], dtype=int)
    primitive = np.asarray(atoms.arrays["primitive_index"], dtype=int)
    source_indices = np.flatnonzero(layer_ids == source_layer)
    if source_indices.size == 0:
        raise RuntimeError(f"source layer {source_layer} is empty")

    lower_center = float(np.mean(atoms.positions[layer_ids == 0, 2]))
    upper_center = float(np.mean(atoms.positions[layer_ids == 1, 2]))
    source_center = lower_center if source_layer == 0 else upper_center
    source_positions = atoms.positions[source_indices].copy()
    source_symbols = [atoms.symbols[index] for index in source_indices]

    lower_positions = source_positions.copy()
    upper_positions = source_positions.copy()
    lower_positions[:, 2] += lower_center - source_center
    upper_positions[:, 2] += upper_center - source_center
    upper_positions += registry_shift
    positions = np.vstack([lower_positions, upper_positions])
    symbols = source_symbols + source_symbols
    control = Atoms(
        symbols=symbols,
        positions=positions,
        cell=atoms.cell.array.copy(),
        pbc=True,
    )
    n_source = len(source_indices)
    control.arrays["layer_id"] = np.concatenate(
        [np.zeros(n_source, dtype=int), np.ones(n_source, dtype=int)]
    )
    source_primitive = primitive[source_indices]
    control.arrays["primitive_index"] = np.concatenate(
        [source_primitive, source_primitive]
    )
    control.arrays["source_twist_atom_index"] = np.concatenate(
        [source_indices, source_indices]
    )
    control.arrays["control_source_layer"] = np.full(
        len(control), source_layer, dtype=int
    )
    control.info.update(
        {
            "control_kind": "periodic_untwisted_registry_minimum",
            "source_layer": source_layer,
            "registry_shift_cartesian_angstrom": registry_shift.tolist(),
        }
    )
    control.wrap()
    return control


def summarize_geometry(atoms: Atoms) -> dict[str, object]:
    layer_ids = np.asarray(atoms.arrays["layer_id"], dtype=int)
    lower_z = atoms.positions[layer_ids == 0, 2]
    upper_z = atoms.positions[layer_ids == 1, 2]
    slab_span = float(np.ptp(atoms.positions[:, 2]))
    diagnostics = {
        "atom_count": len(atoms),
        "lower_atom_count": int(np.sum(layer_ids == 0)),
        "upper_atom_count": int(np.sum(layer_ids == 1)),
        "layer_center_separation_angstrom": float(
            np.mean(upper_z) - np.mean(lower_z)
        ),
        "direct_interlayer_z_gap_angstrom": float(
            np.min(upper_z) - np.max(lower_z)
        ),
        "minimum_periodic_distance_angstrom": minimum_periodic_distance(atoms),
        "minimum_interlayer_distance_angstrom": minimum_interlayer_distance(
            atoms, layer_ids
        ),
        "slab_span_angstrom": slab_span,
        "periodic_vacuum_gap_angstrom": float(atoms.cell.lengths()[2] - slab_span),
    }
    gates = {
        "equal_layer_counts": diagnostics["lower_atom_count"]
        == diagnostics["upper_atom_count"],
        "layers_do_not_cross": diagnostics["direct_interlayer_z_gap_angstrom"]
        >= 2.0,
        "minimum_periodic_distance_ge_2p30": diagnostics[
            "minimum_periodic_distance_angstrom"
        ]
        >= 2.30,
        "minimum_interlayer_distance_ge_2p30": diagnostics[
            "minimum_interlayer_distance_angstrom"
        ]
        >= 2.30,
        "physical_layer_center_separation": 5.0
        <= diagnostics["layer_center_separation_angstrom"]
        <= 6.8,
        "vacuum_gap_ge_23": diagnostics["periodic_vacuum_gap_angstrom"] >= 23.0,
    }
    gates["all_pass"] = all(bool(value) for value in gates.values())
    return {"diagnostics": diagnostics, "gates": gates}


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    source_manifest_path = root / "manifests" / "initial_structures.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = set(args.label)
    if not args.controls_subdir or Path(args.controls_subdir).is_absolute():
        raise ValueError("--controls-subdir must be a nonempty relative path")
    controls_root = root / args.controls_subdir
    rows: list[dict[str, object]] = []
    unavailable: list[str] = []

    for source_row in source_manifest["structures"]:
        source_label = str(source_row["label"])
        if selected and source_label not in selected:
            continue
        source_path = (
            root
            / "relaxations"
            / source_label
            / "confirmed_fmax0p001"
            / "relaxed.extxyz"
        )
        audit_path = source_path.parent / "relaxation_audit.json"
        refinement_path = source_path.parent / "refinement_convergence.json"
        if not source_path.is_file() or not audit_path.is_file() or not refinement_path.is_file():
            unavailable.append(source_label)
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        if not audit["gates"]["all_pass"] or not refinement["gates"]["all_pass"]:
            raise RuntimeError(f"source relaxation gate failed for {source_label}")
        source_atoms = read(source_path)
        if "layer_id" not in source_atoms.arrays or "primitive_index" not in source_atoms.arrays:
            raise RuntimeError(f"missing provenance arrays in {source_path}")

        for source_layer, source_name in ((0, "lower"), (1, "upper")):
            label = f"{source_label}_{source_name}_registry_min"
            a_vector, b_vector = source_primitive_vectors(source_row, source_layer)
            registry_shift = args.registry_u * a_vector + args.registry_v * b_vector
            control = duplicate_source_layer(
                source_atoms,
                source_layer,
                registry_shift,
            )
            geometry = summarize_geometry(control)
            output_dir = controls_root / "structures" / "initial" / label
            output_dir.mkdir(parents=True, exist_ok=True)
            extxyz = output_dir / "initial.extxyz"
            cif = output_dir / "initial.cif"
            poscar = output_dir / "POSCAR"
            write(extxyz, control)
            write(cif, control)
            write(poscar, control, format="vasp", direct=True, sort=False, vasp5=True)
            row = {
                "label": label,
                "angle_deg": 0.0,
                "matched_twist_label": source_label,
                "matched_twist_angle_deg": float(source_row["angle_deg"]),
                "source_layer": source_name,
                "source_layer_id": source_layer,
                "source_relaxed": str(source_path),
                "source_relaxed_sha256": sha256_file(source_path),
                "registry_translation_fraction_a": float(args.registry_u),
                "registry_translation_fraction_b": float(args.registry_v),
                "source_primitive_a_vector_angstrom": a_vector.tolist(),
                "source_primitive_b_vector_angstrom": b_vector.tolist(),
                "registry_shift_cartesian_angstrom": registry_shift.tolist(),
                "atom_count": len(control),
                "formula": control.get_chemical_formula(),
                "extxyz": str(extxyz.relative_to(controls_root)),
                "extxyz_sha256": sha256_file(extxyz),
                "cif": str(cif.relative_to(controls_root)),
                "poscar": str(poscar.relative_to(controls_root)),
                **geometry,
            }
            rows.append(row)

    if selected:
        known = {str(row["label"]) for row in source_manifest["structures"]}
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown source labels: {sorted(unknown)}")

    manifest = {
        "schema_version": 2,
        "protocol": {
            "purpose": (
                "Stable periodic untwisted controls matched to corrected SnSe twist cells"
            ),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "construction": (
                "Duplicate each relaxed source layer into both layer-center planes; "
                "translate the upper copy to the DFT registry minimum; preserve the "
                "twist-cell lattice and periodic vacuum; relax at fixed cell."
            ),
            "registry_translation_fraction_a": float(args.registry_u),
            "registry_translation_fraction_b": float(args.registry_v),
            "registry_evidence": (
                "PBE-D3(BJ) 3x3 registry atlas minimum registry_ix01_iy00_n3x3"
            ),
            "baseline_definition": (
                "For each twist angle, average the lower-source and upper-source "
                "untwisted observables."
            ),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "unavailable_source_labels": unavailable,
        "structures": rows,
    }
    manifest_path = controls_root / "manifests" / "initial_structures.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    all_pass = bool(rows) and all(row["gates"]["all_pass"] for row in rows)
    result = {
        "manifest": str(manifest_path),
        "structure_count": len(rows),
        "unavailable_source_labels": unavailable,
        "all_generated_geometry_gates_pass": all_pass,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

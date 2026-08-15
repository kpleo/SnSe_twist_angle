#!/usr/bin/env python3
"""Export compact Phonopy inputs for the corrected SnSe review analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from phonopy import Phonopy
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
PHONON_DIRNAME = "fmax0p001_disp0p01_plusminus"
DEFAULT_ROOT = REPO / "work"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--additional-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_paths(root: Path, label: str) -> dict[str, Path]:
    controls = root / "matched_controls_registry_min" / "phonons"
    return {
        "twist": root / "phonons" / label / PHONON_DIRNAME,
        "control_lower": controls
        / f"{label}_lower_registry_min"
        / PHONON_DIRNAME,
        "control_upper": controls
        / f"{label}_upper_registry_min"
        / PHONON_DIRNAME,
    }


def structure_rows(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (root / "manifests" / "initial_structures.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        {
            "root": root,
            "label": str(row["label"]),
            "angle_deg": float(row["angle_deg"]),
            "atom_count": int(row["atom_count"]),
        }
        for row in manifest["structures"]
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_case(
    source_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    replace: bool,
) -> dict[str, Any]:
    case_dir = output_dir / metadata["label"] / metadata["kind"]
    structure_yaml = case_dir / "phonopy_structure.yaml"
    force_constants = case_dir / "force_constants.hdf5"
    case_json = case_dir / "case.json"
    if case_dir.exists() and not replace:
        if not all(
            path.is_file() for path in (structure_yaml, force_constants, case_json)
        ):
            raise RuntimeError(f"Incomplete existing export: {case_dir}")
        return json.loads(case_json.read_text(encoding="ascii"))

    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = source_dir / "displacement_manifest.json"
    force_differences_path = source_dir / "forces_minus_base.npy"
    base_supercell_path = source_dir / "base_supercell.extxyz"
    band_path = source_dir / "full_phonon_band.npz"
    for source in (
        manifest_path,
        force_differences_path,
        base_supercell_path,
        band_path,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    atoms = read(base_supercell_path)
    atom_count = len(atoms)
    if atom_count != int(manifest["atom_count"]):
        raise RuntimeError(f"Base-cell atom count disagrees with {manifest_path}")
    force_differences = np.load(force_differences_path, mmap_mode="r")
    if force_differences.shape != (
        int(manifest["displacement_count"]),
        atom_count,
        3,
    ):
        raise RuntimeError(
            f"Unexpected force-array shape {force_differences.shape} in {source_dir}"
        )
    pairs: dict[tuple[int, int], dict[int, tuple[int, float]]] = {}
    for row in manifest["displacements"]:
        displacement = np.asarray(row["displacement_angstrom"], dtype=float)
        nonzero = np.flatnonzero(np.abs(displacement) > 1e-14)
        if nonzero.size != 1:
            raise RuntimeError(f"Non-Cartesian displacement in {manifest_path}")
        axis = int(nonzero[0])
        value = float(displacement[axis])
        sign = 1 if value > 0 else -1
        key = (int(row["atom_index"]), axis)
        pairs.setdefault(key, {})[sign] = (int(row["index"]), value)
    if len(pairs) != 3 * atom_count:
        raise RuntimeError(f"Incomplete Cartesian displacement set in {manifest_path}")
    raw_force_constants = np.empty((atom_count, atom_count, 3, 3), dtype=float)
    for (displaced_atom, axis), signs in pairs.items():
        if set(signs) != {-1, 1}:
            raise RuntimeError(
                f"Incomplete centered pair for atom {displaced_atom}, axis {axis}"
            )
        plus_index, plus_value = signs[1]
        minus_index, minus_value = signs[-1]
        denominator = plus_value - minus_value
        raw_force_constants[:, displaced_atom, :, axis] = -(
            force_differences[plus_index] - force_differences[minus_index]
        ) / denominator

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
        symprec=float(manifest["symprec"]),
        log_level=0,
    )
    phonon.force_constants = raw_force_constants
    raw_max_row_sum = float(
        np.max(np.abs(np.asarray(phonon.force_constants).sum(axis=1)))
    )
    raw_force_constants_path = None
    if metadata["label"] == "8p77deg":
        raw_force_constants_path = case_dir / "raw_force_constants.hdf5"
        write_force_constants_to_hdf5(
            phonon.force_constants,
            filename=str(raw_force_constants_path),
            physical_unit="eV/angstrom^2",
        )
    phonon.symmetrize_force_constants(level=1, show_drift=False)
    asr_max_row_sum = float(
        np.max(np.abs(np.asarray(phonon.force_constants).sum(axis=1)))
    )
    phonon.save(
        structure_yaml,
        settings={
            "force_sets": False,
            "displacements": False,
            "force_constants": False,
        },
    )
    write_force_constants_to_hdf5(
        phonon.force_constants,
        filename=str(force_constants),
        physical_unit="eV/angstrom^2",
    )
    if (
        metadata["expected_atom_count"] is not None
        and atom_count != int(metadata["expected_atom_count"])
    ):
        raise RuntimeError(
            f"Atom-count mismatch for {metadata['label']} {metadata['kind']}: "
            f"{atom_count} != {metadata['expected_atom_count']}"
        )
    payload = {
        "schema_version": 1,
        **metadata,
        "atom_count": atom_count,
        "force_constants_shape": list(phonon.force_constants.shape),
        "source_directory": str(source_dir),
        "source_displacement_manifest": str(manifest_path),
        "source_displacement_manifest_sha256": sha256_file(manifest_path),
        "source_force_differences": str(force_differences_path),
        "source_force_differences_sha256": sha256_file(force_differences_path),
        "source_base_supercell": str(base_supercell_path),
        "source_base_supercell_sha256": sha256_file(base_supercell_path),
        "source_band": str(band_path),
        "source_band_sha256": sha256_file(band_path),
        "phonopy_structure": "phonopy_structure.yaml",
        "phonopy_structure_sha256": sha256_file(structure_yaml),
        "force_constants": "force_constants.hdf5",
        "force_constants_sha256": sha256_file(force_constants),
        "raw_force_constants": (
            raw_force_constants_path.name if raw_force_constants_path else None
        ),
        "raw_force_constants_sha256": (
            sha256_file(raw_force_constants_path)
            if raw_force_constants_path
            else None
        ),
        "raw_max_translational_row_sum_ev_per_angstrom2": raw_max_row_sum,
        "asr_max_translational_row_sum_ev_per_angstrom2": asr_max_row_sum,
        "production_ifc_processing": (
            "Phonopy translational and permutation symmetrization, level 1"
        ),
    }
    case_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return payload


def main() -> int:
    args = parse_args()
    roots = [args.root.expanduser().resolve()]
    if args.additional_root is not None:
        additional = args.additional_root.expanduser().resolve()
        if additional != roots[0]:
            roots.append(additional)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.label)
    structures = [row for root in roots for row in structure_rows(root)]
    if requested:
        structures = [row for row in structures if row["label"] in requested]
        found = {row["label"] for row in structures}
        if requested - found:
            raise ValueError(f"Unknown labels: {sorted(requested - found)}")
    structures.sort(key=lambda row: row["angle_deg"], reverse=True)

    rows: list[dict[str, Any]] = []
    for structure in structures:
        paths = case_paths(structure["root"], structure["label"])
        for kind, source_dir in paths.items():
            payload = export_case(
                source_dir,
                output_dir,
                {
                    "label": structure["label"],
                    "kind": kind,
                    "angle_deg": structure["angle_deg"],
                    "expected_atom_count": structure["atom_count"]
                    if kind == "twist"
                    else None,
                },
                args.replace,
            )
            if kind != "twist" and payload["expected_atom_count"] is None:
                payload["expected_atom_count"] = payload["atom_count"]
                (output_dir / structure["label"] / kind / "case.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="ascii",
                )
            rows.append(
                {
                    "task_index": len(rows),
                    "label": payload["label"],
                    "kind": payload["kind"],
                    "angle_deg": payload["angle_deg"],
                    "atom_count": payload["atom_count"],
                    "case_dir": str(
                        Path(payload["label"]) / payload["kind"]
                    ),
                    "case_json_sha256": sha256_file(
                        output_dir
                        / payload["label"]
                        / payload["kind"]
                        / "case.json"
                    ),
                }
            )
            print(
                f"exported {payload['label']} {payload['kind']} "
                f"({payload['atom_count']} atoms)",
                flush=True,
            )

    write_csv(output_dir / "case_manifest.csv", rows)
    (output_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": (
                    "Compact force-constant inputs for analytical group-velocity "
                    "and Brillouin-zone stability checks."
                ),
                "cases": rows,
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({"case_count": len(rows), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

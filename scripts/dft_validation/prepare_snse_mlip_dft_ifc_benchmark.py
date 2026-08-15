#!/usr/bin/env python3
"""Prepare direct DFT-versus-MACE force-constant benchmark tasks for SnSe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_OUTPUT = REPO_DIR / "work" / "dft_mlip_ifc_benchmark_v3"
DEFAULT_REGISTRY_ROOT = REPO_DIR / "work" / "dft_registry_phase_b_results"
DEFAULT_MAPPING_TABLE = REPO_DIR / "inputs" / "registry" / "registry_atom_mapping.csv"
DISPLACEMENT_ANGSTROM = 0.01
ENCUT_EV = 600
EDIFF_EV = 1.0e-9
SUPPORTED_EDIFF_EV = (1.0e-9, 1.0e-8)
MINIMUM_DISTANCE_ANGSTROM = 1.5
MINIMUM_VACUUM_ANGSTROM = 30.0
POTCAR_LABELS = ["Sn_d", "Se"]
DIRECTIONS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


REGISTRIES = [
    {
        "label": "minimum",
        "case": "registry_ix00_iy00_n3x3",
        "coverage": "complete_24_by_24_hessian",
    },
    {
        "label": "median",
        "case": "registry_ix00_iy02_n3x3",
        "coverage": "selected_sn_se_xyz_hessian_columns",
    },
    {
        "label": "maximum",
        "case": "registry_ix01_iy01_n3x3",
        "coverage": "selected_sn_se_xyz_hessian_columns",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def geometry_sha256(atoms: Atoms) -> str:
    payload = {
        "cell": np.round(atoms.cell.array, 12).tolist(),
        "scaled_positions": np.round(
            atoms.get_scaled_positions(wrap=True), 12
        ).tolist(),
        "symbols": atoms.get_chemical_symbols(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def minimum_periodic_distance(atoms: Atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    mask = ~np.eye(len(atoms), dtype=bool)
    return float(np.min(distances[mask]))


def periodic_vacuum(atoms: Atoms) -> tuple[float, float, float]:
    cell = atoms.cell.array
    normal = np.cross(cell[0], cell[1])
    area = float(np.linalg.norm(normal))
    if area <= 0.0:
        raise ValueError("Degenerate in-plane cell")
    normal /= area
    height = abs(float(np.dot(cell[2], normal)))
    projections = np.mod(np.dot(atoms.positions, normal), height)
    projections.sort()
    gaps = np.diff(np.concatenate([projections, projections[:1] + height]))
    vacuum = float(np.max(gaps))
    return vacuum, height - vacuum, height


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_poscar(path: Path) -> None:
    path.write_text(
        "\n".join(
            line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )


def write_vasp(path: Path, atoms: Atoms) -> None:
    atoms = atoms.copy()
    atoms.set_constraint()
    write(path, atoms, format="vasp", direct=True, sort=False, vasp5=True)
    normalize_poscar(path)


def kpoints_text(label: str, mesh: tuple[int, int, int]) -> str:
    return f"""Automatic mesh for {label}
0
Gamma
{mesh[0]} {mesh[1]} {mesh[2]}
0 0 0
"""


def incar_text(
    label: str, write_wavecar: bool, use_wavecar: bool, ediff_ev: float
) -> str:
    return f"""SYSTEM = SnSe DFT-MLIP IFC benchmark {label}
PREC = Accurate
ENCUT = {ENCUT_EV}
EDIFF = {ediff_ev:.0E}
ALGO = Normal
NELM = 300
NELMIN = 6
ISMEAR = 0
SIGMA = 0.05
ISPIN = 1
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
GGA = PE
IVDW = 12
ISYM = 0
IBRION = -1
NSW = 0
ISTART = {1 if use_wavecar else 0}
LWAVE = {'.TRUE.' if write_wavecar else '.FALSE.'}
LCHARG = .FALSE.
"""


def ensure_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_registry(
    record: dict[str, Any], registry_root: Path, mapping_table: Path
) -> dict[str, Any]:
    source_dir = (registry_root / str(record["case"])).resolve()
    contcar = source_dir / "CONTCAR"
    if not contcar.is_file() or contcar.stat().st_size == 0:
        raise FileNotFoundError(contcar)
    mapping = [
        {
            **row,
            "is_inplane_anchor": str(
                int(row["poscar_atom_index_1based"]) in (1, 3)
            ).lower(),
        }
        for row in read_csv(mapping_table)
        if row["case"] == record["case"]
    ]
    atoms = read(contcar, format="vasp")
    atoms.pbc = True
    atoms.set_constraint()
    if len(atoms) != 8 or Counter(atoms.get_chemical_symbols()) != {"Sn": 4, "Se": 4}:
        raise ValueError(f"Unexpected source composition: {record['case']}")
    if len(mapping) != 8:
        raise ValueError(f"Incomplete atom mapping: {record['case']}")
    return {
        **record,
        "source_dir": source_dir,
        "atoms": atoms,
        "mapping": mapping,
        "relative_energy_mev_per_bilayer": None,
        "source_job_id": None,
        "source_contcar": contcar,
        "source_summary": None,
        "source_mapping": mapping_table,
        "source_contcar_sha256": sha256_file(contcar),
        "source_summary_sha256": None,
        "source_mapping_sha256": sha256_file(mapping_table),
    }


def build_supercell(
    primitive: Atoms, primitive_mapping: list[dict[str, str]], repeats: tuple[int, int, int]
) -> tuple[Atoms, list[dict[str, Any]]]:
    if repeats[2] != 1:
        raise ValueError("This slab benchmark only supports c repeat 1")
    mapping_by_index = {
        int(row["poscar_atom_index_1based"]) - 1: row for row in primitive_mapping
    }
    scaled = primitive.get_scaled_positions(wrap=True)
    symbols: list[str] = []
    positions: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    for species in ("Sn", "Se"):
        primitive_indices = [
            index
            for index, symbol in enumerate(primitive.get_chemical_symbols())
            if symbol == species
        ]
        for primitive_index in primitive_indices:
            for replica_a in range(repeats[0]):
                for replica_b in range(repeats[1]):
                    new_index = len(symbols)
                    position = [
                        (float(scaled[primitive_index, 0]) + replica_a) / repeats[0],
                        (float(scaled[primitive_index, 1]) + replica_b) / repeats[1],
                        float(scaled[primitive_index, 2]),
                    ]
                    symbols.append(species)
                    positions.append(position)
                    source = mapping_by_index[primitive_index]
                    rows.append(
                        {
                            "supercell_atom_index_1based": new_index + 1,
                            "primitive_atom_index_1based": primitive_index + 1,
                            "species": species,
                            "layer": source["layer"],
                            "replica_a": replica_a,
                            "replica_b": replica_b,
                            "replica_c": 0,
                            "fractional_a": position[0],
                            "fractional_b": position[1],
                            "fractional_c": position[2],
                        }
                    )
    cell = primitive.cell.array.copy()
    cell[0] *= repeats[0]
    cell[1] *= repeats[1]
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    return atoms, rows


def selected_source_indices(registry: dict[str, Any]) -> list[int]:
    mapping = registry["mapping"]
    selected = [1, 4]
    expected = [("Sn", "false"), ("Se", "false")]
    for index, (species, anchor) in zip(selected, expected):
        row = mapping[index]
        if row["species"] != species or row["is_inplane_anchor"].lower() != anchor:
            raise ValueError(f"Unexpected representative atom in {registry['case']}")
    return selected


def task_metadata(
    *,
    group: str,
    task_index: int,
    task_name: str,
    task_dir: Path,
    registry: dict[str, Any],
    atoms: Atoms,
    supercell: tuple[int, int, int],
    kmesh: tuple[int, int, int],
    reference_task_index: int,
    reference_task_name: str,
    displaced_atom_index: int | None,
    displaced_primitive_atom_index: int | None,
    direction_label: str | None,
    sign: int,
    coverage: str,
    ediff_ev: float,
    reference_restart: bool,
) -> dict[str, Any]:
    vacuum, slab_span, height = periodic_vacuum(atoms)
    minimum_distance = minimum_periodic_distance(atoms)
    if minimum_distance < MINIMUM_DISTANCE_ANGSTROM:
        raise ValueError(f"Minimum distance failed for {task_name}")
    if vacuum < MINIMUM_VACUUM_ANGSTROM:
        raise ValueError(f"Vacuum failed for {task_name}")
    symbols = atoms.get_chemical_symbols()
    composition = dict(sorted(Counter(symbols).items()))
    is_reference = displaced_atom_index is None
    direction = DIRECTIONS[direction_label].tolist() if direction_label else None
    return {
        "schema_version": 2,
        "benchmark": "SnSe direct DFT-versus-MACE harmonic IFC benchmark",
        "group": group,
        "task_index": task_index,
        "task_name": task_name,
        "coverage": coverage,
        "registry_label": registry["label"],
        "registry_case": registry["case"],
        "registry_relative_energy_mev_per_bilayer": registry[
            "relative_energy_mev_per_bilayer"
        ],
        "source_job_id": registry["source_job_id"],
        "source_contcar": str(registry["source_contcar"]),
        "source_contcar_sha256": registry["source_contcar_sha256"],
        "source_summary": registry["source_summary"],
        "source_summary_sha256": registry["source_summary_sha256"],
        "source_mapping": str(registry["source_mapping"]),
        "source_mapping_sha256": registry["source_mapping_sha256"],
        "source_quality_gate_pass": True,
        "functional": "PBE-D3(BJ)",
        "potcar_labels": POTCAR_LABELS,
        "encut_ev": ENCUT_EV,
        "ediff_ev": ediff_ev,
        "protocol_revision": (
            "ediff_1e-8_reference_restart"
            if reference_restart
            else "original_ediff_1e-9"
        ),
        "kmesh": list(kmesh),
        "isym": 0,
        "static_calculation": True,
        "selective_dynamics_removed_for_static_force_readout": True,
        "supercell": list(supercell),
        "atom_count": len(atoms),
        "composition": composition,
        "is_reference": is_reference,
        "write_wavecar_for_displacement_seed": is_reference,
        "use_reference_wavecar_seed": not is_reference,
        "use_previous_reference_wavecar_seed": is_reference and reference_restart,
        "reference_task_index": reference_task_index,
        "reference_task_name": reference_task_name,
        "displacement_angstrom": 0.0 if is_reference else DISPLACEMENT_ANGSTROM,
        "displacement_sign": sign,
        "displacement_direction_label": direction_label,
        "displacement_cartesian_direction": direction,
        "displaced_atom_index_1based": (
            None if displaced_atom_index is None else displaced_atom_index + 1
        ),
        "displaced_primitive_atom_index_1based": (
            None
            if displaced_primitive_atom_index is None
            else displaced_primitive_atom_index + 1
        ),
        "displaced_species": (
            None if displaced_atom_index is None else symbols[displaced_atom_index]
        ),
        "minimum_periodic_distance_angstrom": minimum_distance,
        "periodic_vacuum_gap_angstrom": vacuum,
        "slab_span_angstrom": slab_span,
        "periodic_height_angstrom": height,
        "geometry_sha256": geometry_sha256(atoms),
        "poscar_sha256": sha256_file(task_dir / "POSCAR"),
        "incar_sha256": sha256_file(task_dir / "INCAR"),
        "kpoints_sha256": sha256_file(task_dir / "KPOINTS"),
    }


def add_task(
    *,
    group_dir: Path,
    tasks: list[dict[str, Any]],
    registry: dict[str, Any],
    base_atoms: Atoms,
    supercell: tuple[int, int, int],
    kmesh: tuple[int, int, int],
    reference_task_index: int,
    reference_task_name: str,
    displaced_atom_index: int | None,
    displaced_primitive_atom_index: int | None,
    direction_label: str | None,
    sign: int,
    coverage: str,
    ediff_ev: float,
    reference_restart: bool,
) -> dict[str, Any]:
    task_index = len(tasks)
    if displaced_atom_index is None:
        task_name = reference_task_name
    else:
        species = base_atoms[displaced_atom_index].symbol.lower()
        suffix = "p" if sign > 0 else "m"
        group_tag = {"one_by_one": "1x1", "two_by_two": "2x2"}[group_dir.name]
        task_name = (
            f"{registry['label']}_{group_tag}_a{displaced_atom_index + 1:02d}_"
            f"{species}_{direction_label}_{suffix}"
        )
    task_dir = group_dir / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    atoms = base_atoms.copy()
    atoms.set_constraint()
    if displaced_atom_index is not None:
        atoms.positions[displaced_atom_index] += (
            sign * DISPLACEMENT_ANGSTROM * DIRECTIONS[direction_label]
        )
    is_reference = displaced_atom_index is None
    (task_dir / "INCAR").write_text(
        incar_text(
            task_name,
            write_wavecar=is_reference,
            use_wavecar=not is_reference or reference_restart,
            ediff_ev=ediff_ev,
        ),
        encoding="utf-8",
    )
    (task_dir / "KPOINTS").write_text(
        kpoints_text(task_name, kmesh), encoding="utf-8"
    )
    write_vasp(task_dir / "POSCAR", atoms)
    metadata = task_metadata(
        group=group_dir.name,
        task_index=task_index,
        task_name=task_name,
        task_dir=task_dir,
        registry=registry,
        atoms=atoms,
        supercell=supercell,
        kmesh=kmesh,
        reference_task_index=reference_task_index,
        reference_task_name=reference_task_name,
        displaced_atom_index=displaced_atom_index,
        displaced_primitive_atom_index=displaced_primitive_atom_index,
        direction_label=direction_label,
        sign=sign,
        coverage=coverage,
        ediff_ev=ediff_ev,
        reference_restart=reference_restart,
    )
    (task_dir / "task_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["task_metadata_sha256"] = sha256_file(task_dir / "task_metadata.json")
    tasks.append(metadata)
    return metadata


def write_group_files(group_dir: Path, tasks: list[dict[str, Any]]) -> None:
    references = [task for task in tasks if task["is_reference"]]
    production = [task for task in tasks if not task["is_reference"]]
    (group_dir / "task_list.txt").write_text(
        "\n".join(task["task_name"] for task in tasks) + "\n", encoding="utf-8"
    )
    (group_dir / "smoke_task_indices.txt").write_text(
        "\n".join(str(task["task_index"]) for task in references) + "\n",
        encoding="utf-8",
    )
    (group_dir / "production_task_indices.txt").write_text(
        "\n".join(str(task["task_index"]) for task in production) + "\n",
        encoding="utf-8",
    )
    write_csv(
        group_dir / "task_manifest.csv",
        tasks,
        [
            "task_index",
            "task_name",
            "registry_label",
            "registry_case",
            "coverage",
            "atom_count",
            "is_reference",
            "reference_task_index",
            "reference_task_name",
            "displaced_atom_index_1based",
            "displaced_primitive_atom_index_1based",
            "displaced_species",
            "displacement_direction_label",
            "displacement_sign",
            "displacement_angstrom",
            "geometry_sha256",
            "poscar_sha256",
            "incar_sha256",
            "kpoints_sha256",
            "task_metadata_sha256",
        ],
    )


def build(
    output: Path,
    force: bool,
    *,
    registry_root: Path,
    mapping_table: Path,
    ediff_ev: float = EDIFF_EV,
    reference_restart: bool = False,
) -> dict[str, Any]:
    if not any(np.isclose(ediff_ev, value, rtol=0.0, atol=1.0e-15) for value in SUPPORTED_EDIFF_EV):
        raise ValueError(f"Unsupported EDIFF: {ediff_ev}")
    if reference_restart and not np.isclose(ediff_ev, 1.0e-8, rtol=0.0, atol=1.0e-15):
        raise ValueError("Reference restart is registered only for the EDIFF=1e-8 revision")
    output = output.resolve()
    ensure_output(output, force)
    registry_root = registry_root.resolve()
    mapping_table = mapping_table.resolve()
    if not mapping_table.is_file():
        raise FileNotFoundError(mapping_table)
    registries = [
        load_registry(record, registry_root, mapping_table) for record in REGISTRIES
    ]
    registry_by_label = {registry["label"]: registry for registry in registries}

    one_dir = output / "one_by_one"
    one_dir.mkdir()
    one_tasks: list[dict[str, Any]] = []
    for registry in registries:
        base = registry["atoms"].copy()
        base.set_constraint()
        reference_index = len(one_tasks)
        reference_name = f"{registry['label']}_1x1_reference"
        add_task(
            group_dir=one_dir,
            tasks=one_tasks,
            registry=registry,
            base_atoms=base,
            supercell=(1, 1, 1),
            kmesh=(16, 16, 1),
            reference_task_index=reference_index,
            reference_task_name=reference_name,
            displaced_atom_index=None,
            displaced_primitive_atom_index=None,
            direction_label=None,
            sign=0,
            coverage=registry["coverage"],
            ediff_ev=ediff_ev,
            reference_restart=reference_restart,
        )
        atom_indices = (
            list(range(len(base)))
            if registry["label"] == "minimum"
            else selected_source_indices(registry)
        )
        for atom_index in atom_indices:
            for direction_label in DIRECTIONS:
                for sign in (-1, 1):
                    add_task(
                        group_dir=one_dir,
                        tasks=one_tasks,
                        registry=registry,
                        base_atoms=base,
                        supercell=(1, 1, 1),
                        kmesh=(16, 16, 1),
                        reference_task_index=reference_index,
                        reference_task_name=reference_name,
                        displaced_atom_index=atom_index,
                        displaced_primitive_atom_index=atom_index,
                        direction_label=direction_label,
                        sign=sign,
                        coverage=registry["coverage"],
                        ediff_ev=ediff_ev,
                        reference_restart=reference_restart,
                    )
    if len(one_tasks) != 75:
        raise ValueError(f"Expected 75 one-by-one tasks, found {len(one_tasks)}")
    write_group_files(one_dir, one_tasks)

    two_dir = output / "two_by_two"
    two_dir.mkdir()
    two_tasks: list[dict[str, Any]] = []
    minimum = registry_by_label["minimum"]
    supercell_atoms, supercell_mapping = build_supercell(
        minimum["atoms"], minimum["mapping"], (2, 2, 1)
    )
    write_csv(
        two_dir / "supercell_atom_mapping.csv",
        supercell_mapping,
        list(supercell_mapping[0]),
    )
    reference_name = "minimum_2x2_reference"
    add_task(
        group_dir=two_dir,
        tasks=two_tasks,
        registry=minimum,
        base_atoms=supercell_atoms,
        supercell=(2, 2, 1),
        kmesh=(8, 8, 1),
        reference_task_index=0,
        reference_task_name=reference_name,
        displaced_atom_index=None,
        displaced_primitive_atom_index=None,
        direction_label=None,
        sign=0,
        coverage="minimum_registry_2x2_spatial_response",
        ediff_ev=ediff_ev,
        reference_restart=reference_restart,
    )
    for primitive_index in selected_source_indices(minimum):
        matches = [
            int(row["supercell_atom_index_1based"]) - 1
            for row in supercell_mapping
            if int(row["primitive_atom_index_1based"]) - 1 == primitive_index
            and int(row["replica_a"]) == 0
            and int(row["replica_b"]) == 0
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot locate 2x2 representative for atom {primitive_index + 1}")
        atom_index = matches[0]
        for sign in (-1, 1):
            add_task(
                group_dir=two_dir,
                tasks=two_tasks,
                registry=minimum,
                base_atoms=supercell_atoms,
                supercell=(2, 2, 1),
                kmesh=(8, 8, 1),
                reference_task_index=0,
                reference_task_name=reference_name,
                displaced_atom_index=atom_index,
                displaced_primitive_atom_index=primitive_index,
                direction_label="x",
                sign=sign,
                coverage="minimum_registry_2x2_spatial_response",
                ediff_ev=ediff_ev,
                reference_restart=reference_restart,
            )
    if len(two_tasks) != 5:
        raise ValueError(f"Expected 5 two-by-two tasks, found {len(two_tasks)}")
    write_group_files(two_dir, two_tasks)

    manifest = {
        "schema_version": 2,
        "description": "Direct DFT-versus-MACE harmonic IFC benchmark for bilayer SnSe",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "registry_result_root": str(registry_root),
        "method": {
            "dft": "PBE-D3(BJ)",
            "potcar_labels": POTCAR_LABELS,
            "encut_ev": ENCUT_EV,
            "ediff_ev": ediff_ev,
            "protocol_revision": (
                "ediff_1e-8_reference_restart"
                if reference_restart
                else "original_ediff_1e-9"
            ),
            "reference_restart": reference_restart,
            "displacement_angstrom": DISPLACEMENT_ANGSTROM,
            "finite_difference": "centered plus/minus",
            "one_by_one_kmesh": [16, 16, 1],
            "two_by_two_kmesh": [8, 8, 1],
            "mlip": "MACE-MPA-0 medium, float64, dispersion=False",
        },
        "coverage": {
            "minimum_registry": "complete 24x24 primitive-cell Hessian",
            "median_registry": "Sn and Se xyz Hessian columns",
            "maximum_registry": "Sn and Se xyz Hessian columns",
            "finite_size": "minimum-registry 2x2 Sn and Se x columns",
        },
        "registry_selection": [
            {
                "label": registry["label"],
                "case": registry["case"],
                "coverage": registry["coverage"],
                "source_contcar": str(registry["source_contcar"]),
                "source_contcar_sha256": registry["source_contcar_sha256"],
            }
            for registry in registries
        ],
        "groups": {
            "one_by_one": {
                "task_count": len(one_tasks),
                "smoke_indices": [
                    task["task_index"] for task in one_tasks if task["is_reference"]
                ],
                "production_indices": [
                    task["task_index"] for task in one_tasks if not task["is_reference"]
                ],
            },
            "two_by_two": {
                "task_count": len(two_tasks),
                "smoke_indices": [0],
                "production_indices": [1, 2, 3, 4],
            },
        },
        "total_task_count": len(one_tasks) + len(two_tasks),
        "release_sequence": [
            "submit exactly the three 1x1 references and one 2x2 reference",
            "require all four references to pass and retain nonempty WAVECAR seeds",
            "submit each remaining displacement index exactly once with concurrency four",
            "sync only lightweight force tables, summaries, OSZICAR, and log tails",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hash_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            hash_rows.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_csv(
        output / "sha256_manifest.csv",
        hash_rows,
        ["relative_path", "sha256", "size_bytes"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=DEFAULT_REGISTRY_ROOT,
        help="Completed strict-registry result root containing the three selected case directories",
    )
    parser.add_argument(
        "--mapping-table",
        type=Path,
        default=DEFAULT_MAPPING_TABLE,
        help="Audited atom mapping for the nine compact registry inputs",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ediff", type=float, default=EDIFF_EV)
    parser.add_argument("--reference-restart", action="store_true")
    args = parser.parse_args()
    manifest = build(
        args.output,
        args.force,
        registry_root=args.registry_root,
        mapping_table=args.mapping_table,
        ediff_ev=args.ediff,
        reference_restart=args.reference_restart,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

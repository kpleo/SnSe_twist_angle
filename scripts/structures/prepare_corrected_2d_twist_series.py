#!/usr/bin/env python3
"""Build the corrected two-dimensional commensurate SnSe twist series.

The two monolayer bases are read independently from the audited lower and
upper primitive structures.  Each commensurate cell uses symmetric logarithmic
area accommodation, an area-preserving aspect correction, a genuine periodic
vacuum gap, and a DFT-informed initial layer-center separation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write
from ase.neighborlist import neighbor_list


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_DIR = PACKAGE_DIR.parent

DEFAULT_LOWER_PRIMITIVE = (
    REPO_DIR / "inputs" / "source_layers" / "POSCAR_LOWER_PRIMITIVE"
)
DEFAULT_UPPER_PRIMITIVE = (
    REPO_DIR / "inputs" / "source_layers" / "POSCAR_UPPER_PRIMITIVE"
)
DEFAULT_OUTPUT = REPO_DIR / "work"
DEFAULT_VACUUM_ANGSTROM = 25.0
DEFAULT_LAYER_SEPARATION_ANGSTROM = 5.9127456535574385


@dataclass(frozen=True)
class TwistSpec:
    label: str
    p: int
    q: int
    s: int = 1
    t: int = 1


TWIST_SERIES = (
    TwistSpec("8p77deg", 7, 6),
    TwistSpec("7p61deg", 8, 7),
    TwistSpec("6p02deg", 10, 9),
    TwistSpec("4p78deg", 13, 11),
    TwistSpec("3p82deg", 16, 14),
    TwistSpec("3p58deg", 17, 15),
    TwistSpec("3p18deg", 19, 17),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-primitive", type=Path, default=DEFAULT_LOWER_PRIMITIVE)
    parser.add_argument("--upper-primitive", type=Path, default=DEFAULT_UPPER_PRIMITIVE)
    parser.add_argument(
        "--dft-registry-table",
        type=Path,
        default=None,
        help="Optional nine-row relaxed-registry table used to recompute the mean separation",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="LABEL:P:Q[:S:T]",
        help=(
            "Generate an explicit commensurate cell. Omit S:T for the established "
            "s=t=1 family; repeat for a custom series"
        ),
    )
    parser.add_argument(
        "--layer-separation",
        type=float,
        default=None,
        help=(
            "Initial layer-center separation in angstrom; default: published "
            f"nine-registry mean ({DEFAULT_LAYER_SEPARATION_ANGSTROM:.15g} A)"
        ),
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=DEFAULT_VACUUM_ANGSTROM,
        help="Empty periodic gap along z in angstrom",
    )
    parser.add_argument(
        "--replace-structures",
        action="store_true",
        help="Replace only structures/initial and initial-structure manifests",
    )
    return parser.parse_args()


def parse_twist_specs(values: list[str]) -> tuple[TwistSpec, ...]:
    if not values:
        return TWIST_SERIES
    specs: list[TwistSpec] = []
    labels: set[str] = set()
    for value in values:
        fields = value.split(":")
        if len(fields) not in (3, 5):
            raise ValueError(
                f"Invalid --spec {value!r}; expected LABEL:P:Q or LABEL:P:Q:S:T"
            )
        label, p_text, q_text = fields[:3]
        p, q = int(p_text), int(q_text)
        s, t = (1, 1) if len(fields) == 3 else (int(fields[3]), int(fields[4]))
        if not label or label in labels:
            raise ValueError(f"Duplicate or empty twist label: {label!r}")
        if p < 1 or q < 1:
            raise ValueError(f"Custom twist spec requires p,q >= 1: {value!r}")
        if s < 1 or t < 1:
            raise ValueError(f"Custom twist spec requires s,t >= 1: {value!r}")
        labels.add(label)
        specs.append(TwistSpec(label, p, q, s, t))
    return tuple(specs)


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.lower_primitive, args.upper_primitive):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dft_registry_table is not None and not args.dft_registry_table.is_file():
        raise FileNotFoundError(args.dft_registry_table)
    if args.vacuum < 22.0:
        raise ValueError(
            "The corrected MACE+D3 protocol requires at least 22 A of vacuum "
            "to exceed the 21.167 A dispersion cutoff"
        )
    if args.layer_separation is not None and not 5.0 <= args.layer_separation <= 7.0:
        raise ValueError("Layer-center separation must lie between 5 and 7 A")


def dft_registry_separation(table: Path) -> tuple[float, dict[str, float]]:
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["layer_center_separation_angstrom"]) for row in rows]
    if len(values) != 9 or not all(5.0 < value < 7.0 for value in values):
        raise ValueError("Expected nine valid DFT registry separations")
    return float(np.mean(values)), {
        "count": len(values),
        "mean_angstrom": float(np.mean(values)),
        "median_angstrom": float(np.median(values)),
        "minimum_angstrom": float(np.min(values)),
        "maximum_angstrom": float(np.max(values)),
        "standard_deviation_angstrom": float(np.std(values)),
    }


def primitive_area(atoms: Atoms) -> float:
    return float(np.linalg.norm(np.cross(atoms.cell.array[0], atoms.cell.array[1])))


def primitive_aspect(atoms: Atoms) -> float:
    lengths = atoms.cell.lengths()
    return float(lengths[1] / lengths[0])


def centered_primitive(atoms: Atoms, a: float, b: float) -> Atoms:
    if len(atoms) != 4 or sorted(atoms.get_chemical_symbols()) != ["Se", "Se", "Sn", "Sn"]:
        raise ValueError("Each primitive must contain four atoms with composition Sn2Se2")
    fractional = atoms.get_scaled_positions(wrap=True)
    centered_z = atoms.positions[:, 2] - float(np.mean(atoms.positions[:, 2]))
    out = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=np.column_stack((fractional[:, 0] * a, fractional[:, 1] * b, centered_z)),
        cell=[[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, 30.0]],
        pbc=True,
    )
    out.set_array("primitive_index", np.arange(len(out), dtype=int))
    return out


def rotate_upper_to_rectangular(upper: Atoms, angle_rad: float) -> None:
    cosine = math.cos(-angle_rad)
    sine = math.sin(-angle_rad)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    upper.positions = upper.positions @ rotation.T
    upper.set_cell(upper.cell.array @ rotation.T, scale_atoms=False)


def rectangularize_layer(atoms: Atoms, target_lengths: np.ndarray) -> None:
    cell = np.asarray(atoms.cell.array, dtype=float)
    dot = float(np.dot(cell[0, :2], cell[1, :2]))
    scale = float(np.linalg.norm(cell[0, :2]) * np.linalg.norm(cell[1, :2]))
    if abs(dot) > 1.0e-9 * scale:
        raise RuntimeError(f"Supercell is not rectangular: normalized dot={dot / scale:.3e}")
    fractional = atoms.get_scaled_positions(wrap=True)
    z = atoms.positions[:, 2].copy()
    atoms.set_cell(
        [[target_lengths[0], 0.0, 0.0], [0.0, target_lengths[1], 0.0], [0.0, 0.0, 30.0]],
        scale_atoms=False,
    )
    atoms.positions = np.column_stack(
        (fractional[:, 0] * target_lengths[0], fractional[:, 1] * target_lengths[1], z)
    )


def reflect_y_to_source_chirality(atoms: Atoms) -> None:
    fractional = atoms.get_scaled_positions(wrap=True)
    fractional[:, 1] = (-fractional[:, 1]) % 1.0
    z = atoms.positions[:, 2].copy()
    atoms.positions = fractional @ atoms.cell.array
    atoms.positions[:, 2] = z


def center_anchor(atoms: Atoms, primitive_index: int = 0) -> dict[str, Any]:
    fractional = atoms.get_scaled_positions(wrap=True)
    primitive_ids = atoms.get_array("primitive_index")
    candidates = np.flatnonzero(primitive_ids == primitive_index)
    if len(candidates) == 0:
        raise RuntimeError("No primitive anchor candidates were retained by make_supercell")
    wrapped = fractional[candidates, :2] - np.rint(fractional[candidates, :2])
    anchor = int(candidates[np.argmin(np.sum(wrapped**2, axis=1))])
    shift = np.array([0.5, 0.5]) - fractional[anchor, :2]
    fractional[:, :2] = (fractional[:, :2] + shift) % 1.0
    z = atoms.positions[:, 2].copy()
    atoms.positions = fractional @ atoms.cell.array
    atoms.positions[:, 2] = z
    return {
        "primitive_index": primitive_index,
        "supercell_atom_index": anchor,
        "fractional_shift": shift.tolist(),
        "final_fractional_xy": atoms.get_scaled_positions(wrap=True)[anchor, :2].tolist(),
    }


def pair_metrics(atoms: Atoms) -> dict[str, float]:
    i, j, distances = neighbor_list("ijd", atoms, cutoff=6.0, self_interaction=False)
    unique = i < j
    i = i[unique]
    j = j[unique]
    distances = distances[unique]
    layer = atoms.get_array("layer_id")
    same = layer[i] == layer[j]
    different = ~same
    if not np.any(same) or not np.any(different):
        raise RuntimeError("Neighbor audit did not recover intra- and interlayer pairs")
    return {
        "minimum_pair_distance_angstrom": float(np.min(distances)),
        "minimum_intralayer_distance_angstrom": float(np.min(distances[same])),
        "minimum_interlayer_distance_angstrom": float(np.min(distances[different])),
    }


def build_twist(
    lower_primitive: Atoms,
    upper_primitive: Atoms,
    spec: TwistSpec,
    reference_area: float,
    native_aspect: float,
    layer_separation: float,
    vacuum: float,
) -> tuple[Atoms, dict[str, Any]]:
    p, q, s, t = spec.p, spec.q, spec.s, spec.t
    lower_cells = p * q
    upper_cells = lower_cells + s * t
    # Orthogonality of (p*a, s*b) and (-t*a, q*b) requires
    # (b/a)^2 = p*t/(s*q). This reduces to sqrt(p/q) for s=t=1.
    required_aspect = math.sqrt(p * t / (s * q))
    primitive_a = math.sqrt(reference_area / required_aspect)
    primitive_b = math.sqrt(reference_area * required_aspect)
    angle_rad = math.atan(math.sqrt(s * t / lower_cells))

    lower_basis = centered_primitive(lower_primitive, primitive_a, primitive_b)
    upper_basis = centered_primitive(upper_primitive, primitive_a, primitive_b)
    lower_matrix = np.array([[p, 0, 0], [0, q, 0], [0, 0, 1]], dtype=int)
    upper_matrix = np.array([[p, s, 0], [-t, q, 0], [0, 0, 1]], dtype=int)
    lower = make_supercell(lower_basis, lower_matrix, wrap=False)
    upper = make_supercell(upper_basis, upper_matrix, wrap=False)
    rotate_upper_to_rectangular(upper, angle_rad)

    lower_lengths = np.linalg.norm(lower.cell.array[:2, :2], axis=1)
    upper_lengths = np.linalg.norm(upper.cell.array[:2, :2], axis=1)
    target_lengths = np.sqrt(lower_lengths * upper_lengths)
    rectangularize_layer(lower, target_lengths)
    rectangularize_layer(upper, target_lengths)
    reflect_y_to_source_chirality(lower)
    reflect_y_to_source_chirality(upper)
    lower_anchor = center_anchor(lower)
    upper_anchor = center_anchor(upper)

    lower_relative_z = lower.positions[:, 2] - float(np.mean(lower.positions[:, 2]))
    upper_relative_z = upper.positions[:, 2] - float(np.mean(upper.positions[:, 2]))
    lower_z = lower_relative_z - 0.5 * layer_separation
    upper_z = upper_relative_z + 0.5 * layer_separation
    z_min = float(min(np.min(lower_z), np.min(upper_z)))
    z_max = float(max(np.max(lower_z), np.max(upper_z)))
    slab_span = z_max - z_min
    c_length = slab_span + vacuum
    z_shift = 0.5 * (c_length - z_min - z_max)

    lower_positions = lower.positions.copy()
    upper_positions = upper.positions.copy()
    lower_positions[:, 2] = lower_z + z_shift
    upper_positions[:, 2] = upper_z + z_shift
    atoms = Atoms(
        symbols=lower.get_chemical_symbols() + upper.get_chemical_symbols(),
        positions=np.vstack((lower_positions, upper_positions)),
        cell=[
            [target_lengths[0], 0.0, 0.0],
            [0.0, target_lengths[1], 0.0],
            [0.0, 0.0, c_length],
        ],
        pbc=True,
    )
    atoms.set_array("layer_id", np.array([0] * len(lower) + [1] * len(upper), dtype=int))
    atoms.set_array(
        "primitive_index",
        np.concatenate((lower.get_array("primitive_index"), upper.get_array("primitive_index"))),
    )
    atoms.wrap()

    layer_ids = atoms.get_array("layer_id")
    lower_indices = np.flatnonzero(layer_ids == 0)
    upper_indices = np.flatnonzero(layer_ids == 1)
    actual_separation = float(
        np.mean(atoms.positions[upper_indices, 2]) - np.mean(atoms.positions[lower_indices, 2])
    )
    direct_gap = float(
        np.min(atoms.positions[upper_indices, 2]) - np.max(atoms.positions[lower_indices, 2])
    )
    periodic_vacuum = float(c_length - np.ptp(atoms.positions[:, 2]))
    area_delta = 0.25 * math.log(upper_cells / lower_cells)
    aspect_delta = 0.5 * math.log(required_aspect / native_aspect)
    metrics: dict[str, Any] = {
        "label": spec.label,
        "angle_deg": math.degrees(angle_rad),
        "p": p,
        "q": q,
        "s": s,
        "t": t,
        "lower_matrix": lower_matrix[:2, :2].tolist(),
        "upper_matrix": upper_matrix[:2, :2].tolist(),
        "lower_primitive_cells": lower_cells,
        "upper_primitive_cells": upper_cells,
        "lower_atom_count": len(lower),
        "upper_atom_count": len(upper),
        "atom_count": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "reference_primitive_area_angstrom2": reference_area,
        "native_b_over_a": native_aspect,
        "required_b_over_a": required_aspect,
        "commensurate_primitive_a_angstrom": primitive_a,
        "commensurate_primitive_b_angstrom": primitive_b,
        "symmetric_area_log_strain": area_delta,
        "symmetric_area_linear_strain_percent": 100.0 * (math.exp(abs(area_delta)) - 1.0),
        "aspect_log_strain": aspect_delta,
        "aspect_linear_strain_percent": 100.0 * (math.exp(abs(aspect_delta)) - 1.0),
        "raw_lower_cell_lengths_angstrom": lower_lengths.tolist(),
        "raw_upper_cell_lengths_angstrom": upper_lengths.tolist(),
        "common_cell_lengths_angstrom": [float(target_lengths[0]), float(target_lengths[1]), c_length],
        "initial_layer_center_separation_angstrom": actual_separation,
        "direct_interlayer_z_gap_angstrom": direct_gap,
        "slab_span_angstrom": slab_span,
        "periodic_vacuum_gap_angstrom": periodic_vacuum,
        "lower_layer_thickness_angstrom": float(np.ptp(atoms.positions[lower_indices, 2])),
        "upper_layer_thickness_angstrom": float(np.ptp(atoms.positions[upper_indices, 2])),
        "lower_anchor": lower_anchor,
        "upper_anchor": upper_anchor,
        "registry_convention": "primitive-index-0 Sn anchors coincident at fractional (0.5,0.5)",
        "source_chirality_operation": "y reflection after commensurate construction",
        **pair_metrics(atoms),
    }
    return atoms, metrics


def validate_structure(atoms: Atoms, metrics: dict[str, Any], requested_vacuum: float) -> None:
    if not bool(np.all(atoms.pbc)):
        raise RuntimeError("All three periodic flags must be true; z isolation is supplied by vacuum")
    expected_atoms = 4 * (
        int(metrics["lower_primitive_cells"]) + int(metrics["upper_primitive_cells"])
    )
    if len(atoms) != expected_atoms:
        raise RuntimeError(f"Expected {expected_atoms} atoms, found {len(atoms)}")
    layer = atoms.get_array("layer_id")
    for layer_id, cells in (
        (0, int(metrics["lower_primitive_cells"])),
        (1, int(metrics["upper_primitive_cells"])),
    ):
        subset = atoms[np.flatnonzero(layer == layer_id)]
        symbols = subset.get_chemical_symbols()
        if symbols.count("Sn") != 2 * cells or symbols.count("Se") != 2 * cells:
            raise RuntimeError(f"Layer {layer_id} composition is not (Sn2Se2)x{cells}")
    if metrics["periodic_vacuum_gap_angstrom"] + 1.0e-8 < requested_vacuum:
        raise RuntimeError("Periodic vacuum is below the requested value")
    if metrics["minimum_intralayer_distance_angstrom"] < 2.5:
        raise RuntimeError("Unphysical intralayer contact detected")
    if metrics["direct_interlayer_z_gap_angstrom"] < 2.4:
        raise RuntimeError("Unphysical direct interlayer z gap detected")
    if metrics["minimum_interlayer_distance_angstrom"] < 2.5:
        raise RuntimeError("Unphysical three-dimensional interlayer contact detected")
    if metrics["aspect_linear_strain_percent"] > 1.0:
        raise RuntimeError("Aspect correction exceeds the one-percent preregistered limit")


def write_poscar(path: Path, atoms: Atoms) -> None:
    write(path, atoms, format="vasp", direct=True, sort=False)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def prepare_output_dirs(root: Path, replace: bool) -> tuple[Path, Path]:
    structures = root / "structures" / "initial"
    manifests = root / "manifests"
    if structures.exists() and any(structures.iterdir()):
        if not replace:
            raise FileExistsError(
                f"Initial structures already exist in {structures}; use --replace-structures explicitly"
            )
        shutil.rmtree(structures)
        for name in (
            "initial_structures.csv",
            "initial_structures.json",
            "initial_structure_sha256.csv",
            "initial_structure_audit.csv",
            "initial_structure_audit.json",
        ):
            path = manifests / name
            if path.exists():
                path.unlink()
    structures.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    for name in ("relaxations", "phonons", "analysis", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return structures, manifests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    twist_series = parse_twist_specs(args.spec)
    root = args.output_dir.resolve()
    structures_dir, manifests_dir = prepare_output_dirs(root, args.replace_structures)
    lower_path = args.lower_primitive.resolve()
    upper_path = args.upper_primitive.resolve()
    lower_primitive = read(lower_path)
    upper_primitive = read(upper_path)

    registry_table = (
        args.dft_registry_table.resolve()
        if args.dft_registry_table is not None
        else None
    )
    if registry_table is None:
        dft_stats = None
        layer_separation = (
            DEFAULT_LAYER_SEPARATION_ANGSTROM
            if args.layer_separation is None
            else float(args.layer_separation)
        )
    else:
        dft_mean, dft_stats = dft_registry_separation(registry_table)
        layer_separation = (
            dft_mean if args.layer_separation is None else float(args.layer_separation)
        )
    lower_area = primitive_area(lower_primitive)
    upper_area = primitive_area(upper_primitive)
    reference_area = math.sqrt(lower_area * upper_area)
    native_aspect = math.sqrt(
        primitive_aspect(lower_primitive) * primitive_aspect(upper_primitive)
    )

    records: list[dict[str, Any]] = []
    hash_records: list[dict[str, str]] = []
    for spec in twist_series:
        atoms, metrics = build_twist(
            lower_primitive,
            upper_primitive,
            spec,
            reference_area,
            native_aspect,
            layer_separation,
            float(args.vacuum),
        )
        validate_structure(atoms, metrics, float(args.vacuum))
        case_dir = structures_dir / spec.label
        case_dir.mkdir()
        extxyz = case_dir / "initial.extxyz"
        cif = case_dir / "initial.cif"
        poscar = case_dir / "POSCAR"
        write(extxyz, atoms)
        write(cif, atoms)
        write_poscar(poscar, atoms)
        metrics_path = case_dir / "geometry.json"
        metrics.update(
            {
                "extxyz": str(extxyz.relative_to(root)),
                "extxyz_sha256": sha256_file(extxyz),
                "cif": str(cif.relative_to(root)),
                "cif_sha256": sha256_file(cif),
                "poscar": str(poscar.relative_to(root)),
                "poscar_sha256": sha256_file(poscar),
            }
        )
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        for artifact in (extxyz, cif, poscar, metrics_path):
            hash_records.append(
                {
                    "path": str(artifact.relative_to(root)),
                    "sha256": sha256_file(artifact),
                }
            )
        records.append(metrics)

    protocol = {
        "purpose": "Corrected isolated-bilayer SnSe twist-angle recalculation",
        "twist_series": [asdict(spec) for spec in twist_series],
        "construction": {
            "commensurate_family": "M_lower=[[p,0],[0,q]], M_upper=[[p,s],[-t,q]]",
            "area_convention": "equal-and-opposite logarithmic area accommodation",
            "aspect_convention": "area-preserving correction to exact rectangular closure",
            "registry_convention": "primitive-index-0 Sn anchors coincident at cell center",
            "periodicity": "3D periodic cell with an explicit empty z gap",
            "requested_periodic_vacuum_angstrom": float(args.vacuum),
            "initial_layer_center_separation_angstrom": layer_separation,
            "initial_separation_source": (
                "optional nine-row DFT registry table"
                if registry_table is not None and args.layer_separation is None
                else (
                    "published mean of nine DFT-relaxed registries"
                    if args.layer_separation is None
                    else "user supplied"
                )
            ),
        },
        "dft_registry_separation_statistics": dft_stats,
        "primitive_reference": {
            "lower_path": str(lower_path),
            "upper_path": str(upper_path),
            "lower_sha256": sha256_file(lower_path),
            "upper_sha256": sha256_file(upper_path),
            "lower_area_angstrom2": lower_area,
            "upper_area_angstrom2": upper_area,
            "geometric_mean_area_angstrom2": reference_area,
            "geometric_mean_b_over_a": native_aspect,
        },
        "dft_registry_table": (
            {
                "path": str(registry_table),
                "sha256": sha256_file(registry_table),
            }
            if registry_table is not None
            else None
        ),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (root / "README.md").write_text(
        "# Corrected 2D SnSe twist recalculation\n\n"
        "This directory is isolated from the superseded finite-c structures. "
        "All structures have an explicit periodic vacuum gap and use the independently "
        "audited lower and upper SnSe primitive motifs. The initial structures must pass "
        "the independent geometry audit before any relaxation or phonon production run.\n",
        encoding="utf-8",
    )
    (manifests_dir / "initial_structures.json").write_text(
        json.dumps({"protocol": protocol, "structures": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_records = [
        {
            key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in records
    ]
    write_csv(manifests_dir / "initial_structures.csv", flat_records)
    write_csv(manifests_dir / "initial_structure_sha256.csv", hash_records)
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "angle_count": len(records),
                "angles_deg": [row["angle_deg"] for row in records],
                "atom_counts": [row["atom_count"] for row in records],
                "layer_center_separation_angstrom": layer_separation,
                "periodic_vacuum_gap_angstrom": float(args.vacuum),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

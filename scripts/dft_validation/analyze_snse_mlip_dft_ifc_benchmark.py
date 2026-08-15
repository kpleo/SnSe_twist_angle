#!/usr/bin/env python3
"""Analyze the preregistered SnSe DFT-versus-MACE IFC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_DIR = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_DIR / "work" / "dft_mlip_ifc_benchmark_v3"
VALIDATION_ROOT = REPO_DIR / "work" / "dft_mlip_ifc_validation"
MACE_ROOT = VALIDATION_ROOT / "mace_results"
AXES = {"x": 0, "y": 1, "z": 2}
MASSES_AMU = {"Sn": 118.710, "Se": 78.971}
EV_PER_ANGSTROM2_TO_NEWTON_PER_METER = 16.02176634
AMU_TO_KG = 1.66053906660e-27


@dataclass(frozen=True)
class Task:
    group: str
    name: str
    metadata: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def repository_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    resolved = path.resolve()
    repository = REPO_DIR.resolve()
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def load_tasks(group: str) -> list[Task]:
    names = [
        line.strip()
        for line in (INPUT_ROOT / group / "task_list.txt").read_text().splitlines()
        if line.strip()
    ]
    tasks = []
    for name in names:
        metadata = read_json(INPUT_ROOT / group / "tasks" / name / "task_metadata.json")
        if metadata["task_name"] != name or metadata["group"] != group:
            raise ValueError(f"Task metadata mismatch: {name}")
        tasks.append(Task(group=group, name=name, metadata=metadata))
    return tasks


def read_forces(path: Path, expected_atoms: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_atoms:
        raise ValueError(f"Expected {expected_atoms} force rows in {path}, found {len(rows)}")
    rows.sort(key=lambda row: int(row["atom_index_1based"]))
    return np.asarray(
        [
            [
                float(row["fx_ev_per_angstrom"]),
                float(row["fy_ev_per_angstrom"]),
                float(row["fz_ev_per_angstrom"]),
            ]
            for row in rows
        ],
        dtype=float,
    )


def parse_poscar(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError("Only positive POSCAR scale factors are supported")
    cell = np.asarray(
        [[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)]
    ) * scale
    species = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError(f"Unexpected POSCAR species/count lines: {path}")
    cursor = 7
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    mode = lines[cursor].strip().lower()
    cursor += 1
    atom_count = sum(counts)
    raw = np.asarray(
        [
            [float(value) for value in lines[index].split()[:3]]
            for index in range(cursor, cursor + atom_count)
        ],
        dtype=float,
    )
    positions = raw @ cell if mode.startswith("d") else raw * scale
    symbols = [symbol for symbol, count in zip(species, counts) for _ in range(count)]
    return cell, positions, symbols


def minimum_image_distances(
    cell: np.ndarray, positions: np.ndarray, center_index: int
) -> np.ndarray:
    differences = positions - positions[center_index]
    fractional = np.linalg.solve(cell.T, differences.T).T
    fractional -= np.round(fractional)
    cartesian = fractional @ cell
    return np.linalg.norm(cartesian, axis=1)


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    x = np.asarray(reference, dtype=float).ravel()
    y = np.asarray(prediction, dtype=float).ravel()
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Metric vectors must have the same nontrivial shape")
    denominator = float(np.dot(x, x))
    rms_reference = float(np.sqrt(np.mean(x * x)))
    return {
        "component_count": int(x.size),
        "dft_rms": rms_reference,
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "origin_slope": float(np.dot(x, y) / denominator),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "normalized_rmse": float(
            np.sqrt(np.mean((y - x) ** 2)) / rms_reference
        ),
    }


def task_result_dir(
    task: Task,
    dft_root: Path,
    one_smoke_job: str,
    two_smoke_job: str,
    one_production_jobs: list[str],
    two_production_jobs: list[str],
) -> Path:
    one = task.group == "one_by_one"
    if task.metadata["is_reference"]:
        stage = "smoke"
        job_ids = [one_smoke_job if one else two_smoke_job]
    else:
        stage = "production"
        job_ids = one_production_jobs if one else two_production_jobs
    group_tag = "one_by_one" if one else "two_by_two"
    candidates = [
        dft_root / stage / f"{group_tag}_job{job_id}" / task.name
        for job_id in job_ids
    ]
    present = [path for path in candidates if (path / "force_summary.json").is_file()]
    if len(present) != 1:
        raise ValueError(
            f"Expected exactly one DFT result for {task.name}; found {present}"
        )
    return present[0]


def load_method_forces(
    tasks: list[Task],
    dft_root: Path,
    one_smoke_job: str,
    two_smoke_job: str,
    one_production_jobs: list[str],
    two_production_jobs: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    dft: dict[str, np.ndarray] = {}
    mace: dict[str, np.ndarray] = {}
    task_gate_rows: list[dict[str, Any]] = []
    for task in tasks:
        expected_atoms = int(task.metadata["atom_count"])
        dft_dir = task_result_dir(
            task,
            dft_root,
            one_smoke_job,
            two_smoke_job,
            one_production_jobs,
            two_production_jobs,
        )
        dft_summary = read_json(dft_dir / "force_summary.json")
        mace_summary = read_json(MACE_ROOT / task.group / task.name / "mace_summary.json")
        geometry_hash_match = bool(
            dft_summary.get("geometry_sha256") == task.metadata["geometry_sha256"]
            and mace_summary.get("geometry_sha256") == task.metadata["geometry_sha256"]
        )
        poscar_hash_match = bool(
            dft_summary.get("poscar_sha256") == task.metadata["poscar_sha256"]
            and mace_summary.get("poscar_sha256") == task.metadata["poscar_sha256"]
        )
        passed = bool(
            dft_summary.get("quality_gate_pass") is True
            and mace_summary.get("quality_gate_pass") is True
            and dft_summary.get("task_name") == task.name
            and mace_summary.get("task_name") == task.name
            and geometry_hash_match
            and poscar_hash_match
        )
        task_gate_rows.append(
            {
                "group": task.group,
                "task_index": task.metadata["task_index"],
                "task_name": task.name,
                "is_reference": task.metadata["is_reference"],
                "dft_scheduler_job_id": dft_dir.parent.name.removeprefix(
                    f"{'one_by_one' if task.group == 'one_by_one' else 'two_by_two'}_job"
                ),
                "dft_quality_gate_pass": dft_summary.get("quality_gate_pass"),
                "mace_quality_gate_pass": mace_summary.get("quality_gate_pass"),
                "dft_mace_geometry_hash_match": geometry_hash_match,
                "dft_mace_poscar_hash_match": poscar_hash_match,
                "joint_quality_gate_pass": passed,
                "dft_electronic_iteration_rows": dft_summary.get(
                    "electronic_iteration_row_count"
                ),
                "dft_force_block_count": dft_summary.get("force_block_count"),
            }
        )
        if not passed:
            raise ValueError(f"Task gate failed before analysis: {task.name}")
        dft[task.name] = read_forces(dft_dir / "forces.csv", expected_atoms)
        mace[task.name] = read_forces(
            MACE_ROOT / task.group / task.name / "forces.csv", expected_atoms
        )
    return dft, mace, task_gate_rows


def pair_key(metadata: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(metadata["registry_label"]),
        int(metadata["displaced_atom_index_1based"]),
        str(metadata["displacement_direction_label"]),
    )


def build_pairs(tasks: Iterable[Task]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[int, Task]] = defaultdict(dict)
    for task in tasks:
        if not task.metadata["is_reference"]:
            grouped[pair_key(task.metadata)][int(task.metadata["displacement_sign"])] = task
    pairs: list[dict[str, Any]] = []
    for key, signs in sorted(grouped.items()):
        if set(signs) != {-1, 1}:
            raise ValueError(f"Incomplete centered pair: {key}")
        metadata = signs[1].metadata
        pairs.append(
            {
                "registry_label": key[0],
                "displaced_atom_index_1based": key[1],
                "direction": key[2],
                "species": metadata["displaced_species"],
                "reference_name": metadata["reference_task_name"],
                "delta": float(metadata["displacement_angstrom"]),
                "plus": signs[1],
                "minus": signs[-1],
            }
        )
    return pairs


def response_and_hessian_rows(
    group: str,
    pairs: list[dict[str, Any]],
    dft: dict[str, np.ndarray],
    mace: dict[str, np.ndarray],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
]:
    response_rows: list[dict[str, Any]] = []
    hessian_rows: list[dict[str, Any]] = []
    even_rows: list[dict[str, Any]] = []
    columns: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for pair in pairs:
        reference = pair["reference_name"]
        plus_name = pair["plus"].name
        minus_name = pair["minus"].name
        dft_zero, dft_plus, dft_minus = dft[reference], dft[plus_name], dft[minus_name]
        mace_zero, mace_plus, mace_minus = (
            mace[reference],
            mace[plus_name],
            mace[minus_name],
        )
        delta = pair["delta"]
        dft_hessian = -(dft_plus - dft_minus) / (2.0 * delta)
        mace_hessian = -(mace_plus - mace_minus) / (2.0 * delta)
        key = (
            pair["registry_label"],
            pair["displaced_atom_index_1based"],
            pair["direction"],
        )
        columns[key] = (dft_hessian.ravel(), mace_hessian.ravel())
        for method, zero, plus, minus in (
            ("dft", dft_zero, dft_plus, dft_minus),
            ("mace", mace_zero, mace_plus, mace_minus),
        ):
            numerator = float(np.linalg.norm(plus + minus - 2.0 * zero))
            denominator = float(np.linalg.norm(plus - minus))
            even_rows.append(
                {
                    "group": group,
                    "registry": pair["registry_label"],
                    "displaced_atom_index_1based": pair[
                        "displaced_atom_index_1based"
                    ],
                    "displaced_species": pair["species"],
                    "direction": pair["direction"],
                    "method": method,
                    "even_residual_norm": numerator,
                    "antisymmetric_response_norm": denominator,
                    "even_to_antisymmetric_ratio": numerator / denominator,
                }
            )
        for sign, dft_displaced, mace_displaced in (
            (-1, dft_minus, mace_minus),
            (1, dft_plus, mace_plus),
        ):
            dft_response = dft_displaced - dft_zero
            mace_response = mace_displaced - mace_zero
            for atom_index in range(dft_response.shape[0]):
                for component, component_label in enumerate("xyz"):
                    response_rows.append(
                        {
                            "group": group,
                            "registry": pair["registry_label"],
                            "displaced_atom_index_1based": pair[
                                "displaced_atom_index_1based"
                            ],
                            "displaced_species": pair["species"],
                            "displacement_direction": pair["direction"],
                            "displacement_sign": sign,
                            "response_atom_index_1based": atom_index + 1,
                            "force_component": component_label,
                            "dft_response_ev_per_angstrom": dft_response[
                                atom_index, component
                            ],
                            "mace_response_ev_per_angstrom": mace_response[
                                atom_index, component
                            ],
                        }
                    )
        for atom_index in range(dft_hessian.shape[0]):
            for component, component_label in enumerate("xyz"):
                hessian_rows.append(
                    {
                        "group": group,
                        "registry": pair["registry_label"],
                        "displaced_atom_index_1based": pair[
                            "displaced_atom_index_1based"
                        ],
                        "displaced_species": pair["species"],
                        "displacement_direction": pair["direction"],
                        "response_atom_index_1based": atom_index + 1,
                        "force_component": component_label,
                        "dft_hessian_ev_per_angstrom2": dft_hessian[
                            atom_index, component
                        ],
                        "mace_hessian_ev_per_angstrom2": mace_hessian[
                            atom_index, component
                        ],
                    }
                )
    return response_rows, hessian_rows, even_rows, columns


def values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def slice_metrics(hessian_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "registry": lambda row: row["registry"],
        "species": lambda row: row["displaced_species"],
        "direction": lambda row: row["displacement_direction"],
        "combined": lambda row: (
            f"{row['registry']}|{row['displaced_species']}|"
            f"{row['displacement_direction']}"
        ),
    }
    output: list[dict[str, Any]] = []
    for slice_type, key_function in definitions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in hessian_rows:
            grouped[str(key_function(row))].append(row)
        for label, subset in sorted(grouped.items()):
            dft_values = values(subset, "dft_hessian_ev_per_angstrom2")
            mace_values = values(subset, "mace_hessian_ev_per_angstrom2")
            output.append(
                {
                    "slice_type": slice_type,
                    "slice_label": label,
                    "dft_response_norm": float(np.linalg.norm(dft_values)),
                    **metrics(dft_values, mace_values),
                }
            )
    combined_norms = [
        row["dft_response_norm"] for row in output if row["slice_type"] == "combined"
    ]
    median_norm = float(np.median(combined_norms))
    for row in output:
        row["strong_slice"] = bool(
            row["slice_type"] == "combined"
            and row["dft_response_norm"] > median_norm
        )
        row["strong_slice_gate_pass"] = bool(
            not row["strong_slice"] or row["normalized_rmse"] <= 0.50
        )
    return output


def assemble_minimum_hessian(
    columns: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (3 * atom_count, 3 * atom_count)
    dft_hessian = np.full(shape, np.nan)
    mace_hessian = np.full(shape, np.nan)
    for atom_index in range(1, atom_count + 1):
        for direction, axis in AXES.items():
            key = ("minimum", atom_index, direction)
            if key not in columns:
                raise ValueError(f"Missing minimum-registry Hessian column {key}")
            column = 3 * (atom_index - 1) + axis
            dft_hessian[:, column], mace_hessian[:, column] = columns[key]
    if not np.isfinite(dft_hessian).all() or not np.isfinite(mace_hessian).all():
        raise ValueError("Incomplete minimum-registry Hessian")
    return dft_hessian, mace_hessian


def translation_basis(atom_count: int) -> np.ndarray:
    basis = np.zeros((3 * atom_count, 3), dtype=float)
    for atom_index in range(atom_count):
        basis[3 * atom_index : 3 * atom_index + 3, :] = np.eye(3)
    return basis


def hessian_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    norm = float(np.linalg.norm(matrix))
    atom_count = matrix.shape[0] // 3
    basis = translation_basis(atom_count)
    return {
        "frobenius_norm_ev_per_angstrom2": norm,
        "raw_reciprocity_defect": float(np.linalg.norm(matrix - matrix.T) / norm),
        "raw_acoustic_sum_residual": float(np.linalg.norm(matrix @ basis) / norm),
    }


def project_hessian(matrix: np.ndarray) -> np.ndarray:
    atom_count = matrix.shape[0] // 3
    basis = translation_basis(atom_count)
    projector = np.eye(3 * atom_count) - basis @ np.linalg.inv(basis.T @ basis) @ basis.T
    symmetric = 0.5 * (matrix + matrix.T)
    return projector @ symmetric @ projector


def signed_frequencies_thz(eigenvalues: np.ndarray) -> np.ndarray:
    omega = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    return omega / (2.0 * math.pi * 1.0e12)


def gamma_modes(
    hessian: np.ndarray, symbols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = np.repeat([MASSES_AMU[symbol] for symbol in symbols], 3)
    prefactor = EV_PER_ANGSTROM2_TO_NEWTON_PER_METER / AMU_TO_KG
    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(masses))
    dynamical = prefactor * inverse_sqrt_mass @ hessian @ inverse_sqrt_mass
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
    frequencies = signed_frequencies_thz(eigenvalues)
    acoustic = np.argsort(np.abs(frequencies))[:3]
    return frequencies, eigenvectors, acoustic


def matched_gamma_rows(
    dft_hessian: np.ndarray, mace_hessian: np.ndarray, symbols: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dft_frequency, dft_vectors, dft_acoustic = gamma_modes(dft_hessian, symbols)
    mace_frequency, mace_vectors, mace_acoustic = gamma_modes(mace_hessian, symbols)
    dft_optical = [index for index in range(len(dft_frequency)) if index not in dft_acoustic]
    mace_optical = [index for index in range(len(mace_frequency)) if index not in mace_acoustic]
    overlaps = np.clip(
        np.abs(dft_vectors[:, dft_optical].T @ mace_vectors[:, mace_optical]),
        0.0,
        1.0,
    )
    dft_match, mace_match = linear_sum_assignment(-overlaps)
    rows: list[dict[str, Any]] = []
    for rank, (dft_local, mace_local) in enumerate(zip(dft_match, mace_match), start=1):
        dft_index = dft_optical[dft_local]
        mace_index = mace_optical[mace_local]
        rows.append(
            {
                "matched_mode_rank": rank,
                "dft_mode_index_1based": dft_index + 1,
                "mace_mode_index_1based": mace_index + 1,
                "dft_frequency_thz": dft_frequency[dft_index],
                "mace_frequency_thz": mace_frequency[mace_index],
                "absolute_frequency_error_thz": abs(
                    mace_frequency[mace_index] - dft_frequency[dft_index]
                ),
                "mass_weighted_mode_overlap": overlaps[dft_local, mace_local],
                "stable_dft_optical": bool(dft_frequency[dft_index] > 0.05),
            }
        )
    rows.sort(key=lambda row: row["dft_frequency_thz"])
    stable = [row for row in rows if row["stable_dft_optical"]]
    if not stable:
        raise ValueError("No stable DFT optical modes found")
    dft_stable = np.asarray([row["dft_frequency_thz"] for row in stable])
    mace_stable = np.asarray([row["mace_frequency_thz"] for row in stable])
    summary = {
        "dft_projected_acoustic_frequencies_thz": [
            float(dft_frequency[index]) for index in dft_acoustic
        ],
        "mace_projected_acoustic_frequencies_thz": [
            float(mace_frequency[index]) for index in mace_acoustic
        ],
        "stable_optical_mode_count": len(stable),
        "stable_optical_frequency_mae_thz": float(
            np.mean(np.abs(mace_stable - dft_stable))
        ),
        "maximum_frequency_relative_error": float(
            abs(np.max(mace_stable) - np.max(dft_stable)) / np.max(dft_stable)
        ),
        "stable_optical_frequency_origin_slope": float(
            np.dot(dft_stable, mace_stable) / np.dot(dft_stable, dft_stable)
        ),
        "median_mass_weighted_matched_mode_overlap": float(
            np.median([row["mass_weighted_mode_overlap"] for row in stable])
        ),
        "minimum_mass_weighted_matched_mode_overlap": float(
            np.min([row["mass_weighted_mode_overlap"] for row in stable])
        ),
        "stable_modes_below_0p90_overlap": int(
            sum(row["mass_weighted_mode_overlap"] < 0.90 for row in stable)
        ),
    }
    return rows, summary


def spatial_decay_rows(
    pairs: list[dict[str, Any]],
    dft: dict[str, np.ndarray],
    mace: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    reference_task = pairs[0]["plus"].metadata["reference_task_name"]
    poscar = INPUT_ROOT / "two_by_two" / "tasks" / reference_task / "POSCAR"
    cell, positions, symbols = parse_poscar(poscar)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        displaced_index = pair["displaced_atom_index_1based"] - 1
        distances = minimum_image_distances(cell, positions, displaced_index)
        dft_odd = 0.5 * (dft[pair["plus"].name] - dft[pair["minus"].name])
        mace_odd = 0.5 * (mace[pair["plus"].name] - mace[pair["minus"].name])
        for atom_index, (distance, symbol) in enumerate(zip(distances, symbols)):
            rows.append(
                {
                    "displaced_species": pair["species"],
                    "displaced_atom_index_1based": displaced_index + 1,
                    "response_atom_index_1based": atom_index + 1,
                    "response_species": symbol,
                    "periodic_distance_angstrom": distance,
                    "dft_centered_response_magnitude_ev_per_angstrom": float(
                        np.linalg.norm(dft_odd[atom_index])
                    ),
                    "mace_centered_response_magnitude_ev_per_angstrom": float(
                        np.linalg.norm(mace_odd[atom_index])
                    ),
                }
            )
    return rows


def reference_force_rows(
    tasks: list[Task], dft: dict[str, np.ndarray], mace: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not task.metadata["is_reference"]:
            continue
        comparison = metrics(dft[task.name], mace[task.name])
        rows.append(
            {
                "group": task.group,
                "registry": task.metadata["registry_label"],
                "task_name": task.name,
                **comparison,
                "dft_max_raw_force_ev_per_angstrom": float(
                    np.max(np.linalg.norm(dft[task.name], axis=1))
                ),
                "mace_max_raw_force_ev_per_angstrom": float(
                    np.max(np.linalg.norm(mace[task.name], axis=1))
                ),
            }
        )
    return rows


def gate_result(value: float, comparator: str, threshold: Any) -> dict[str, Any]:
    if comparator == "ge":
        passed = value >= float(threshold)
    elif comparator == "le":
        passed = value <= float(threshold)
    elif comparator == "range":
        lower, upper = threshold
        passed = float(lower) <= value <= float(upper)
    else:
        raise ValueError(comparator)
    return {"value": value, "comparator": comparator, "threshold": threshold, "pass": passed}


def main() -> None:
    global INPUT_ROOT, MACE_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-version", choices=["v2", "v3"], default="v3")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--one-smoke-job-id", required=True)
    parser.add_argument("--two-smoke-job-id", required=True)
    parser.add_argument(
        "--one-production-job-id", action="append", required=True,
        help="Repeat for scheduler-split one-by-one production waves.",
    )
    parser.add_argument(
        "--two-production-job-id", action="append", required=True,
        help="Repeat for scheduler-split two-by-two production waves.",
    )
    parser.add_argument("--dft-root", type=Path)
    parser.add_argument("--mace-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    INPUT_ROOT = args.input_root.resolve()
    validation_root = VALIDATION_ROOT
    dft_root = args.dft_root or validation_root / "dft_results"
    MACE_ROOT = args.mace_root or (
        validation_root / "mace_results"
    )
    output_root = args.out or validation_root / "analysis"

    one_tasks = load_tasks("one_by_one")
    two_tasks = load_tasks("two_by_two")
    all_tasks = [*one_tasks, *two_tasks]
    dft, mace, task_gate_rows = load_method_forces(
        all_tasks,
        dft_root,
        args.one_smoke_job_id,
        args.two_smoke_job_id,
        args.one_production_job_id,
        args.two_production_job_id,
    )
    one_pairs = build_pairs(one_tasks)
    two_pairs = build_pairs(two_tasks)
    one_response, one_hessian, one_even, one_columns = response_and_hessian_rows(
        "one_by_one", one_pairs, dft, mace
    )
    two_response, two_hessian, two_even, _ = response_and_hessian_rows(
        "two_by_two", two_pairs, dft, mace
    )

    one_response_metrics = metrics(
        values(one_response, "dft_response_ev_per_angstrom"),
        values(one_response, "mace_response_ev_per_angstrom"),
    )
    one_hessian_metrics = metrics(
        values(one_hessian, "dft_hessian_ev_per_angstrom2"),
        values(one_hessian, "mace_hessian_ev_per_angstrom2"),
    )
    two_response_metrics = metrics(
        values(two_response, "dft_response_ev_per_angstrom"),
        values(two_response, "mace_response_ev_per_angstrom"),
    )
    slices = slice_metrics(one_hessian)

    dft_raw_hessian, mace_raw_hessian = assemble_minimum_hessian(one_columns, 8)
    dft_projected = project_hessian(dft_raw_hessian)
    mace_projected = project_hessian(mace_raw_hessian)
    _, _, symbols = parse_poscar(
        INPUT_ROOT / "one_by_one" / "tasks" / "minimum_1x1_reference" / "POSCAR"
    )
    gamma_rows, gamma_summary = matched_gamma_rows(
        dft_projected, mace_projected, symbols
    )
    for method, raw in (("dft", dft_raw_hessian), ("mace", mace_raw_hessian)):
        raw_frequency, _, raw_acoustic = gamma_modes(0.5 * (raw + raw.T), symbols)
        gamma_summary[f"{method}_raw_symmetrized_acoustic_frequencies_thz"] = [
            float(raw_frequency[index]) for index in raw_acoustic
        ]
    hessian_diagnostic_rows = []
    for method, raw, projected in (
        ("DFT", dft_raw_hessian, dft_projected),
        ("MACE", mace_raw_hessian, mace_projected),
    ):
        raw_metrics = hessian_diagnostics(raw)
        projected_metrics = hessian_diagnostics(projected)
        hessian_diagnostic_rows.append(
            {
                "method": method,
                **raw_metrics,
                "projected_reciprocity_defect": projected_metrics[
                    "raw_reciprocity_defect"
                ],
                "projected_acoustic_sum_residual": projected_metrics[
                    "raw_acoustic_sum_residual"
                ],
            }
        )

    all_even = [*one_even, *two_even]
    max_dft_even = max(
        row["even_to_antisymmetric_ratio"]
        for row in all_even
        if row["method"] == "dft"
    )
    strong_slices = [row for row in slices if row["strong_slice"]]
    max_strong_slice_nrmse = max(row["normalized_rmse"] for row in strong_slices)
    gates = {
        "dft_even_residual": gate_result(max_dft_even, "le", 0.10),
        "force_response_pearson": gate_result(
            one_response_metrics["pearson_r"], "ge", 0.90
        ),
        "force_response_origin_slope": gate_result(
            one_response_metrics["origin_slope"], "range", [0.80, 1.20]
        ),
        "force_response_normalized_rmse": gate_result(
            one_response_metrics["normalized_rmse"], "le", 0.30
        ),
        "hessian_pearson": gate_result(
            one_hessian_metrics["pearson_r"], "ge", 0.90
        ),
        "hessian_origin_slope": gate_result(
            one_hessian_metrics["origin_slope"], "range", [0.75, 1.25]
        ),
        "hessian_normalized_rmse": gate_result(
            one_hessian_metrics["normalized_rmse"], "le", 0.30
        ),
        "strong_slice_normalized_rmse": gate_result(
            max_strong_slice_nrmse, "le", 0.50
        ),
        "stable_optical_frequency_mae": gate_result(
            gamma_summary["stable_optical_frequency_mae_thz"], "le", 0.30
        ),
        "maximum_frequency_relative_error": gate_result(
            gamma_summary["maximum_frequency_relative_error"], "le", 0.10
        ),
        "median_mode_overlap": gate_result(
            gamma_summary["median_mass_weighted_matched_mode_overlap"], "ge", 0.90
        ),
        "two_by_two_response_normalized_rmse": gate_result(
            two_response_metrics["normalized_rmse"], "le", 0.30
        ),
    }
    failed_gates = [name for name, gate in gates.items() if not gate["pass"]]
    if not failed_gates:
        verdict = "pass"
    else:
        preservation_gates = {
            "dft_even_residual",
            "force_response_pearson",
            "hessian_pearson",
            "strong_slice_normalized_rmse",
            "median_mode_overlap",
            "two_by_two_response_normalized_rmse",
        }
        verdict = (
            "qualified"
            if all(gates[name]["pass"] for name in preservation_gates)
            else "fail"
        )

    spatial_rows = spatial_decay_rows(two_pairs, dft, mace)
    reference_rows = reference_force_rows(all_tasks, dft, mace)
    reference_summary = {
        "case_count": len(reference_rows),
        "minimum_pearson_r": min(row["pearson_r"] for row in reference_rows),
        "maximum_normalized_rmse": max(
            row["normalized_rmse"] for row in reference_rows
        ),
        "primary_gate": False,
        "interpretation": (
            "Raw static-force offsets differ on the DFT-relaxed reference "
            "structures. The IFC comparison uses preregistered reference-subtracted "
            "centered responses, so these offsets cancel but remain disclosed."
        ),
    }
    gate_rows = []
    for name, gate in gates.items():
        threshold = gate["threshold"]
        gate_rows.append(
            {
                "gate": name,
                "value": gate["value"],
                "comparator": gate["comparator"],
                "threshold_lower": (
                    threshold[0] if gate["comparator"] == "range" else ""
                ),
                "threshold_upper": (
                    threshold[1]
                    if gate["comparator"] == "range"
                    else threshold
                ),
                "pass": gate["pass"],
            }
        )
    summary = {
        "schema_version": 3,
        "verdict": verdict,
        "all_primary_gates_pass": not failed_gates,
        "failed_gates": failed_gates,
        "jobs": {
            "one_smoke": args.one_smoke_job_id,
            "two_smoke": args.two_smoke_job_id,
            "one_production": args.one_production_job_id,
            "two_production": args.two_production_job_id,
        },
        "protocol_version": args.protocol_version,
        "mace_result_root": repository_path(MACE_ROOT),
        "mace_geometry_provenance": {
            "source_protocol": "v2",
            "reason_for_reuse": (
                "Protocol v3 changed only the DFT electronic tolerance and "
                "reference-restart procedure; all benchmark geometries are unchanged."
            ),
            "geometry_hash_match_count": sum(
                row["dft_mace_geometry_hash_match"] for row in task_gate_rows
            ),
            "poscar_hash_match_count": sum(
                row["dft_mace_poscar_hash_match"] for row in task_gate_rows
            ),
            "expected_match_count": len(all_tasks),
        },
        "task_count": len(all_tasks),
        "centered_pair_count": len(one_pairs) + len(two_pairs),
        "one_by_one_force_response": one_response_metrics,
        "one_by_one_raw_hessian": one_hessian_metrics,
        "two_by_two_force_response": two_response_metrics,
        "gamma": gamma_summary,
        "reference_raw_force_offsets": reference_summary,
        "maximum_dft_even_to_antisymmetric_ratio": max_dft_even,
        "maximum_strong_slice_normalized_rmse": max_strong_slice_nrmse,
        "gates": gates,
        "interpretation": {
            "pass": "The preregistered centered MACE harmonic responses pass direct DFT validation across representative local SnSe registries and the selected 2x2 spatial check.",
            "qualified": "Correlation and mode identity are retained, but a measured scale or frequency bias must bound quantitative claims.",
            "fail": "The present MLIP phonons cannot stand alone as quantitative evidence without a validated replacement or fine-tuning.",
        }[verdict],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "benchmark_summary.json", summary)
    write_csv(output_root / "benchmark_primary_gates.csv", gate_rows)
    write_csv(output_root / "task_quality_gates.csv", task_gate_rows)
    write_csv(output_root / "one_by_one_force_response_components.csv", one_response)
    write_csv(output_root / "one_by_one_hessian_components.csv", one_hessian)
    write_csv(output_root / "two_by_two_force_response_components.csv", two_response)
    write_csv(output_root / "two_by_two_hessian_components.csv", two_hessian)
    write_csv(output_root / "centered_pair_even_residuals.csv", all_even)
    write_csv(output_root / "hessian_slice_metrics.csv", slices)
    write_csv(output_root / "minimum_hessian_diagnostics.csv", hessian_diagnostic_rows)
    write_csv(output_root / "minimum_gamma_matched_modes.csv", gamma_rows)
    write_csv(output_root / "two_by_two_spatial_response_decay.csv", spatial_rows)
    write_csv(output_root / "reference_raw_force_comparison.csv", reference_rows)
    np.savez_compressed(
        output_root / "minimum_hessians.npz",
        dft_raw=dft_raw_hessian,
        mace_raw=mace_raw_hessian,
        dft_projected=dft_projected,
        mace_projected=mace_projected,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

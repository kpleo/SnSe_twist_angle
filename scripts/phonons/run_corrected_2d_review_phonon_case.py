#!/usr/bin/env python3
"""Run one analytical-velocity and stability case for the SnSe revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import phonopy
from phonopy import load


THZ_ANGSTROM_TO_M_PER_S = 100.0
KB_EV_PER_K = 8.617333262145e-5
H_EV_THz = 4.135667696e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band-points", type=int, default=9)
    parser.add_argument("--temperatures", default="100,300,500,700")
    parser.add_argument("--cutoffs", default="0.02,0.05,0.10")
    parser.add_argument("--q-grid", type=int, default=0)
    parser.add_argument(
        "--use-raw-force-constants",
        action="store_true",
        help="Use the preserved pre-ASR force constants when available.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heat_capacity_over_kb(frequency_thz: np.ndarray, temperature_k: float) -> np.ndarray:
    x = H_EV_THz * frequency_thz / (KB_EV_PER_K * temperature_k)
    values = np.ones_like(x)
    finite = x > 1e-7
    exp_x = np.exp(np.clip(x[finite], None, 700.0))
    values[finite] = x[finite] ** 2 * exp_x / (exp_x - 1.0) ** 2
    return values


def path_midpoints(points_per_segment: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if points_per_segment < 3:
        raise ValueError("--band-points must be at least 3")
    vertices = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    fractions = (np.arange(points_per_segment, dtype=float) + 0.5) / points_per_segment
    for start, end in zip(vertices[:-1], vertices[1:]):
        direction = end - start
        for fraction in fractions:
            rows.append((start + fraction * direction, direction))
    return rows


def thermal_metrics(
    frequency: np.ndarray,
    velocity: np.ndarray,
    temperature_k: float,
    cutoff_thz: float,
) -> dict[str, float | int]:
    valid = (
        np.isfinite(frequency)
        & np.isfinite(velocity)
        & (frequency > cutoff_thz)
    )
    f = frequency[valid]
    v = velocity[valid]
    heat_capacity = heat_capacity_over_kb(f, temperature_k)
    weighted_v2 = heat_capacity * v**2
    c_sum = float(np.sum(heat_capacity))
    cv2_sum = float(np.sum(weighted_v2))
    return {
        "n_samples": int(f.size),
        "heat_capacity_sum_over_kb": c_sum,
        "cv2_sum_proxy": cv2_sum,
        "cv2_mean_proxy": cv2_sum / c_sum,
        "cv_weighted_rms_velocity_m_per_s": float(np.sqrt(cv2_sum / c_sum)),
        "fraction_cv2_below_1p5_thz": float(
            np.sum(weighted_v2[f < 1.5]) / cv2_sum
        ),
        "fraction_heat_capacity_below_1p5_thz": float(
            np.sum(heat_capacity[f < 1.5]) / c_sum
        ),
    }


def spectral_bins(
    frequency: np.ndarray,
    velocity: np.ndarray,
    temperature_k: float,
    cutoff_thz: float,
) -> list[dict[str, float | int]]:
    valid = (
        np.isfinite(frequency)
        & np.isfinite(velocity)
        & (frequency > cutoff_thz)
    )
    f = frequency[valid]
    v = velocity[valid]
    heat_capacity = heat_capacity_over_kb(f, temperature_k)
    weighted_v2 = heat_capacity * v**2
    c_total = float(np.sum(heat_capacity))
    cv2_total = float(np.sum(weighted_v2))
    edges = np.arange(0.0, 6.0001, 0.1)
    rows: list[dict[str, float | int]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (f >= low) & (f < high)
        c_bin = float(np.sum(heat_capacity[mask]))
        cv2_bin = float(np.sum(weighted_v2[mask]))
        rows.append(
            {
                "low_thz": float(low),
                "high_thz": float(high),
                "sample_count": int(np.sum(mask)),
                "heat_capacity_share": c_bin / c_total,
                "cv2_share": cv2_bin / cv2_total,
                "rms_velocity_m_per_s": (
                    float(np.sqrt(cv2_bin / c_bin)) if c_bin > 0 else None
                ),
            }
        )
    return rows


def analytical_path_samples(
    phonon_object, points_per_segment: int
) -> tuple[np.ndarray, np.ndarray]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(
        np.asarray(phonon_object.unitcell.cell)
    ).T
    frequencies: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    rows = path_midpoints(points_per_segment)
    for index, (qpoint, fractional_direction) in enumerate(rows, start=1):
        cartesian_direction = fractional_direction @ reciprocal
        cartesian_direction /= np.linalg.norm(cartesian_direction)
        phonon_object.run_qpoints([qpoint], with_group_velocities=True)
        data = phonon_object.get_qpoints_dict()
        group_velocities = data["group_velocities"]
        if group_velocities is None:
            raise RuntimeError("Phonopy did not return group velocities")
        frequencies.append(np.asarray(data["frequencies"][0], dtype=float).copy())
        velocities.append(
            (
                np.asarray(group_velocities[0], dtype=float)
                @ cartesian_direction
                * THZ_ANGSTROM_TO_M_PER_S
            ).copy()
        )
        print(f"path q point {index}/{len(rows)} complete", flush=True)
    return np.concatenate(frequencies), np.concatenate(velocities)


def q_grid_summary(phonon_object, size: int) -> dict[str, Any] | None:
    if size <= 0:
        return None
    minima: list[float] = []
    negative_counts: list[int] = []
    qpoints: list[list[float]] = []
    for i in range(size):
        for j in range(size):
            qpoint = [i / size, j / size, 0.0]
            phonon_object.run_qpoints([qpoint])
            frequencies = np.asarray(
                phonon_object.get_qpoints_dict()["frequencies"][0], dtype=float
            )
            minima.append(float(np.min(frequencies)))
            negative_counts.append(int(np.sum(frequencies < -0.05)))
            qpoints.append(qpoint)
            print(
                f"q-grid point {len(qpoints)}/{size * size} complete",
                flush=True,
            )
    minimum_index = int(np.argmin(minima))
    return {
        "grid": [size, size, 1],
        "qpoint_count": len(qpoints),
        "minimum_frequency_thz": minima[minimum_index],
        "minimum_qpoint": qpoints[minimum_index],
        "qpoints_with_frequency_below_minus_0p05_thz": int(
            np.sum(np.asarray(negative_counts) > 0)
        ),
        "total_modes_below_minus_0p05_thz": int(np.sum(negative_counts)),
    }


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(output)
    metadata_path = case_dir / "case.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    structure_yaml = case_dir / metadata["phonopy_structure"]
    force_constants_key = (
        "raw_force_constants"
        if args.use_raw_force_constants
        else "force_constants"
    )
    force_constants_hash_key = f"{force_constants_key}_sha256"
    if force_constants_key not in metadata:
        raise RuntimeError(
            f"{force_constants_key} is not available for {case_dir}"
        )
    force_constants = case_dir / metadata[force_constants_key]
    if sha256_file(structure_yaml) != metadata["phonopy_structure_sha256"]:
        raise RuntimeError("Phonopy structure hash mismatch")
    if sha256_file(force_constants) != metadata[force_constants_hash_key]:
        raise RuntimeError("Force-constant hash mismatch")

    temperatures = [float(value) for value in args.temperatures.split(",")]
    cutoffs = [float(value) for value in args.cutoffs.split(",")]
    start = time.time()
    phonon_object = load(
        phonopy_yaml=str(structure_yaml),
        force_constants_filename=str(force_constants),
        symmetrize_fc=False,
    )
    if len(phonon_object.unitcell) != metadata["atom_count"]:
        raise RuntimeError("Loaded atom count does not match case metadata")

    phonon_object.run_qpoints([[0.0, 0.0, 0.0]])
    gamma = np.sort(
        np.asarray(
            phonon_object.get_qpoints_dict()["frequencies"][0], dtype=float
        )
    )
    frequency, velocity = analytical_path_samples(
        phonon_object, args.band_points
    )
    metrics = {
        f"{temperature:g}": {
            f"{cutoff:.2f}": thermal_metrics(
                frequency, velocity, temperature, cutoff
            )
            for cutoff in cutoffs
        }
        for temperature in temperatures
    }
    valid_velocity = np.abs(velocity[np.isfinite(velocity)])
    result = {
        "schema_version": 1,
        "case": metadata,
        "method": (
            "Phonopy dynamical-matrix derivative group velocities projected "
            "onto Gamma-X-S-Y-Gamma segment directions"
        ),
        "force_constant_variant": (
            "raw_pre_asr"
            if args.use_raw_force_constants
            else "production_symmetrized_asr"
        ),
        "path_sampling": (
            "equal-weight segment-midpoint quadrature; high-symmetry vertices "
            "are excluded from the velocity average"
        ),
        "band_points_per_segment": args.band_points,
        "path_qpoint_count": 4 * args.band_points,
        "temperatures_k": temperatures,
        "positive_frequency_cutoffs_thz": cutoffs,
        "gamma_six_lowest_frequencies_thz": gamma[:6].tolist(),
        "path_minimum_frequency_thz": float(np.min(frequency)),
        "path_modes_below_minus_0p05_thz": int(np.sum(frequency < -0.05)),
        "absolute_velocity_quantiles_m_per_s": {
            "q50": float(np.quantile(valid_velocity, 0.50)),
            "q90": float(np.quantile(valid_velocity, 0.90)),
            "q99": float(np.quantile(valid_velocity, 0.99)),
            "maximum": float(np.max(valid_velocity)),
        },
        "metrics": metrics,
        "spectral_bins_300k_cutoff_0p05": spectral_bins(
            frequency, velocity, 300.0, 0.05
        ),
        "q_grid_stability": q_grid_summary(phonon_object, args.q_grid),
        "runtime_seconds": time.time() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "phonopy": phonopy.__version__,
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_ID",
                "SLURMD_NODENAME",
                "SLURM_CPUS_PER_TASK",
            )
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"output": str(output), "runtime_seconds": result["runtime_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

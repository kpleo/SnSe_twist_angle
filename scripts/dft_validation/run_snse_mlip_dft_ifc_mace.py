#!/usr/bin/env python3
"""Evaluate pinned MACE-MPA-0 forces on every DFT IFC benchmark geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_INPUT = REPO_DIR / "work" / "dft_mlip_ifc_benchmark_v3"
DEFAULT_OUTPUT = REPO_DIR / "work" / "dft_mlip_ifc_validation" / "mace_results"
DEFAULT_MODEL = Path.home() / ".cache" / "mace" / "macempa0mediummodel"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def task_dirs(input_root: Path) -> list[Path]:
    paths: list[Path] = []
    for group in ("one_by_one", "two_by_two"):
        group_root = input_root / group
        names = [
            line.strip()
            for line in (group_root / "task_list.txt").read_text().splitlines()
            if line.strip()
        ]
        for name in names:
            path = group_root / "tasks" / name
            if not path.is_dir():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def evaluate(
    input_root: Path,
    output_root: Path,
    model_path: Path,
    device: str,
    torch_threads: int,
    model_label: str,
    force: bool,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(model_path)
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ensure_output(output_root, force)

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    calculator = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype="float64",
    )
    model_sha256 = sha256_file(model_path)
    combined: list[dict[str, Any]] = []
    started = time.time()
    for number, task_dir in enumerate(task_dirs(input_root), start=1):
        metadata_path = task_dir / "task_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(task_dir / "POSCAR") != metadata["poscar_sha256"]:
            raise ValueError(f"POSCAR hash mismatch: {task_dir}")
        atoms = read(task_dir / "POSCAR", format="vasp")
        atoms.pbc = True
        atoms.set_constraint()
        atoms.calc = calculator
        task_started = time.time()
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        elapsed = time.time() - task_started
        if forces.shape != (int(metadata["atom_count"]), 3):
            raise ValueError(f"Unexpected force shape for {metadata['task_name']}")
        if not np.all(np.isfinite(forces)) or not np.isfinite(energy):
            raise ValueError(f"Nonfinite MACE output for {metadata['task_name']}")

        task_out = output_root / metadata["group"] / metadata["task_name"]
        task_out.mkdir(parents=True, exist_ok=False)
        force_rows = [
            {
                "atom_index_1based": atom_index + 1,
                "fx_ev_per_angstrom": float(vector[0]),
                "fy_ev_per_angstrom": float(vector[1]),
                "fz_ev_per_angstrom": float(vector[2]),
            }
            for atom_index, vector in enumerate(forces)
        ]
        forces_path = task_out / "forces.csv"
        write_csv(
            forces_path,
            force_rows,
            [
                "atom_index_1based",
                "fx_ev_per_angstrom",
                "fy_ev_per_angstrom",
                "fz_ev_per_angstrom",
            ],
        )
        norms = np.linalg.norm(forces, axis=1)
        net_force = np.sum(forces, axis=0)
        summary = {
            **metadata,
            "model": model_label,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "device": device,
            "default_dtype": "float64",
            "dispersion": False,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "energy_ev": energy,
            "force_atom_count": int(forces.shape[0]),
            "force_component_count": int(forces.size),
            "raw_force_max_ev_per_angstrom": float(np.max(norms)),
            "raw_force_rms_ev_per_angstrom": float(
                np.sqrt(np.mean(np.square(norms)))
            ),
            "net_force_vector_ev_per_angstrom": net_force.tolist(),
            "net_force_norm_ev_per_angstrom": float(np.linalg.norm(net_force)),
            "forces_csv_sha256": sha256_file(forces_path),
            "elapsed_seconds": elapsed,
            "quality_gate_pass": True,
        }
        summary_path = task_out / "mace_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        combined.append(
            {
                "group": metadata["group"],
                "task_index": metadata["task_index"],
                "task_name": metadata["task_name"],
                "registry_label": metadata["registry_label"],
                "coverage": metadata["coverage"],
                "atom_count": metadata["atom_count"],
                "is_reference": metadata["is_reference"],
                "displaced_atom_index_1based": metadata[
                    "displaced_atom_index_1based"
                ],
                "displacement_direction_label": metadata[
                    "displacement_direction_label"
                ],
                "displacement_sign": metadata["displacement_sign"],
                "energy_ev": energy,
                "raw_force_max_ev_per_angstrom": summary[
                    "raw_force_max_ev_per_angstrom"
                ],
                "raw_force_rms_ev_per_angstrom": summary[
                    "raw_force_rms_ev_per_angstrom"
                ],
                "net_force_norm_ev_per_angstrom": summary[
                    "net_force_norm_ev_per_angstrom"
                ],
                "elapsed_seconds": elapsed,
                "forces_csv_sha256": summary["forces_csv_sha256"],
            }
        )
        print(
            f"[{number:02d}/{manifest['total_task_count']}] "
            f"{metadata['group']}/{metadata['task_name']} {elapsed:.3f} s",
            flush=True,
        )

    combined.sort(key=lambda row: (row["group"], int(row["task_index"])))
    combined_path = output_root / "mace_task_summary.csv"
    write_csv(combined_path, combined, list(combined[0]))
    run_summary = {
        "schema_version": 2,
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "task_count": len(combined),
        "expected_task_count": manifest["total_task_count"],
        "all_tasks_pass": len(combined) == manifest["total_task_count"],
        "model": model_label,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "device": device,
        "default_dtype": "float64",
        "dispersion": False,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.time() - started,
        "task_summary_csv": str(combined_path),
        "task_summary_csv_sha256": sha256_file(combined_path),
    }
    (output_root / "mace_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--model-label", default="MACE-MPA-0 medium")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = evaluate(
        args.input_root,
        args.output_root,
        args.model_path,
        args.device,
        args.torch_threads,
        args.model_label,
        args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

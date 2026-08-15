#!/usr/bin/env python3
"""Probe and relax the 7.61-degree finite-q soft mode with MACE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpoint", default="0.5,0.5,0")
    parser.add_argument(
        "--scan-amplitudes",
        default="-0.12,-0.08,-0.05,-0.03,-0.02,-0.01,-0.005,0,0.005,0.01,0.02,0.03,0.05,0.08,0.12",
    )
    parser.add_argument("--relax-amplitude", type=float, default=0.05)
    parser.add_argument("--curvature-probe", type=float, default=0.005)
    parser.add_argument("--fmax", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=48)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def build_mode_supercell(phonon_object, qpoint: np.ndarray):
    from ase import Atoms
    from phonopy.phonon.modulation import Modulation

    unit = phonon_object.unitcell
    atom_count = len(unit)
    phonon_object.run_qpoints([qpoint], with_eigenvectors=True)
    data = phonon_object.get_qpoints_dict()
    frequencies = np.asarray(data["frequencies"][0], dtype=float)
    eigenvectors = np.asarray(data["eigenvectors"][0])
    if eigenvectors.shape != (3 * atom_count, 3 * atom_count):
        raise RuntimeError(f"unexpected eigenvector shape {eigenvectors.shape}")

    mode_index = int(np.argmin(frequencies))
    modulation = Modulation(
        phonon_object.dynamical_matrix,
        dimension=[2, 2, 1],
        phonon_modes=[(qpoint, mode_index, 1.0, 0.0)],
    )
    modulation.run()
    modulations, phonopy_supercell = modulation.get_modulations_and_supercell()
    if float(np.max(np.abs(np.imag(modulations[0])))) > 1e-10:
        raise RuntimeError("commensurate S-point modulation is not real")
    pattern = np.real(modulations[0])
    maximum = float(np.max(np.linalg.norm(pattern, axis=1)))
    if maximum <= 0:
        raise RuntimeError("selected soft-mode displacement is zero")
    pattern /= maximum

    supercell = Atoms(
        symbols=phonopy_supercell.symbols,
        positions=np.asarray(phonopy_supercell.positions, dtype=float),
        cell=np.asarray(phonopy_supercell.cell, dtype=float),
        pbc=True,
    )
    return (
        supercell,
        pattern,
        np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int),
        mode_index,
        float(frequencies[mode_index]),
    )


def evaluated_copy(atoms, calculator, pattern: np.ndarray, amplitude: float):
    displaced = atoms.copy()
    displaced.positions += amplitude * pattern
    displaced.calc = calculator
    energy = float(displaced.get_potential_energy())
    forces = np.asarray(displaced.get_forces(), dtype=float)
    return displaced, energy, forces


def max_force(forces: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(forces, axis=1)))


def projected_force(forces: np.ndarray, pattern: np.ndarray) -> float:
    return float(np.sum(forces * pattern))


def curvature_at(
    atoms,
    calculator,
    pattern: np.ndarray,
    displacement: float,
    reference_energy: float | None = None,
) -> dict[str, float]:
    center = atoms.copy()
    center.calc = calculator
    e0 = float(center.get_potential_energy()) if reference_energy is None else reference_energy
    _, e_minus, _ = evaluated_copy(atoms, calculator, pattern, -displacement)
    _, e_plus, _ = evaluated_copy(atoms, calculator, pattern, displacement)
    return {
        "probe_amplitude_angstrom": displacement,
        "minus_energy_ev": e_minus,
        "center_energy_ev": e0,
        "plus_energy_ev": e_plus,
        "curvature_ev_per_angstrom2": (e_plus - 2.0 * e0 + e_minus) / displacement**2,
        "odd_energy_difference_ev": e_plus - e_minus,
    }


def relax_seed(
    base,
    calculator,
    pattern: np.ndarray,
    amplitude: float,
    output_dir: Path,
    fmax: float,
    max_steps: int,
    curvature_probe: float,
) -> dict[str, Any]:
    from ase.geometry import find_mic
    from ase.io import write
    from ase.optimize import FIRE

    atoms, initial_energy, initial_forces = evaluated_copy(
        base, calculator, pattern, amplitude
    )
    initial_positions = atoms.positions.copy()
    optimizer = FIRE(
        atoms,
        logfile=str(output_dir / "relax.log"),
        trajectory=str(output_dir / "relax.traj"),
    )
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    final_energy = float(atoms.get_potential_energy())
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    displacement, _ = find_mic(atoms.positions - base.positions, base.cell, pbc=True)
    projection = float(np.sum(displacement * pattern) / np.sum(pattern**2))
    residual = displacement - projection * pattern
    write(output_dir / "relaxed.extxyz", atoms)
    return {
        "seed_amplitude_angstrom": amplitude,
        "converged": converged,
        "optimizer": "ASE FIRE",
        "steps": int(optimizer.nsteps),
        "fmax_target_ev_per_angstrom": fmax,
        "initial_energy_ev": initial_energy,
        "final_energy_ev": final_energy,
        "energy_change_ev": final_energy - initial_energy,
        "initial_max_force_ev_per_angstrom": max_force(initial_forces),
        "final_max_force_ev_per_angstrom": max_force(final_forces),
        "initial_to_final_rms_displacement_angstrom": float(
            np.sqrt(np.mean(np.sum((atoms.positions - initial_positions) ** 2, axis=1)))
        ),
        "final_projection_on_initial_mode_angstrom": projection,
        "final_residual_rms_displacement_angstrom": float(
            np.sqrt(np.mean(np.sum(residual**2, axis=1)))
        ),
        "same_pattern_curvature": curvature_at(
            atoms,
            calculator,
            pattern,
            curvature_probe,
            reference_energy=final_energy,
        ),
        "relaxed_structure": str(output_dir / "relaxed.extxyz"),
        "relaxed_structure_sha256": sha256_file(output_dir / "relaxed.extxyz"),
    }


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    import ase
    import mace
    import phonopy
    import torch
    from mace.calculators import MACECalculator
    from phonopy import load

    torch.set_num_threads(args.threads)
    case_dir = args.case_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    if sha256_file(model_path) != args.expected_model_sha256:
        raise RuntimeError("model hash mismatch")
    metadata_path = case_dir / "case.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    structure_yaml = case_dir / metadata["phonopy_structure"]
    force_constants = case_dir / metadata["force_constants"]
    if sha256_file(structure_yaml) != metadata["phonopy_structure_sha256"]:
        raise RuntimeError("Phonopy structure hash mismatch")
    if sha256_file(force_constants) != metadata["force_constants_sha256"]:
        raise RuntimeError("force-constant hash mismatch")

    phonon_object = load(
        phonopy_yaml=str(structure_yaml),
        force_constants_filename=str(force_constants),
        symmetrize_fc=False,
    )
    qpoint = np.asarray(parse_floats(args.qpoint), dtype=float)
    if qpoint.shape != (3,):
        raise ValueError("--qpoint must contain three values")
    base, pattern, translations, mode_index, mode_frequency = build_mode_supercell(
        phonon_object, qpoint
    )
    if len(base) != 4 * metadata["atom_count"]:
        raise RuntimeError("unexpected frozen-mode supercell atom count")

    calculator = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        default_dtype="float64",
    )
    amplitudes = sorted(set(parse_floats(args.scan_amplitudes)))
    scan_rows: list[dict[str, float]] = []
    for amplitude in amplitudes:
        _, energy, forces = evaluated_copy(base, calculator, pattern, amplitude)
        scan_rows.append(
            {
                "amplitude_angstrom": amplitude,
                "energy_ev": energy,
                "max_force_ev_per_angstrom": max_force(forces),
                "projected_force_ev_per_angstrom": projected_force(forces, pattern),
            }
        )

    reference = next(row["energy_ev"] for row in scan_rows if row["amplitude_angstrom"] == 0)
    for row in scan_rows:
        row["delta_energy_ev_per_moire_cell"] = (row["energy_ev"] - reference) / 4.0
    fit_matrix = np.asarray(
        [[row["amplitude_angstrom"] ** power for power in (2, 4, 6)] for row in scan_rows]
    )
    fit_target = np.asarray([row["delta_energy_ev_per_moire_cell"] for row in scan_rows])
    coefficients, _, _, _ = np.linalg.lstsq(fit_matrix, fit_target, rcond=None)

    relaxations: dict[str, Any] = {}
    for sign, name in ((1.0, "plus"), (-1.0, "minus")):
        branch_dir = output_dir / name
        branch_dir.mkdir()
        relaxations[name] = relax_seed(
            base,
            calculator,
            pattern,
            sign * args.relax_amplitude,
            branch_dir,
            args.fmax,
            args.max_steps,
            args.curvature_probe,
        )
        relaxations[name]["final_delta_energy_ev_per_moire_cell"] = (
            relaxations[name]["final_energy_ev"] - reference
        ) / 4.0

    payload = {
        "schema_version": 1,
        "description": "Frozen S-point soft-mode energy scan and fixed-cell relaxation.",
        "case": metadata,
        "case_metadata_sha256": sha256_file(metadata_path),
        "qpoint_reduced": qpoint.tolist(),
        "selected_mode_index_zero_based": mode_index,
        "selected_frequency_thz": mode_frequency,
        "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        "supercell_atom_count": len(base),
        "mode_normalization": "maximum per-atom displacement norm equals one",
        "mode_translation_phase_counts": {
            "positive": int(np.sum(np.linalg.norm(pattern, axis=1) > 0)),
            "translation_rows": np.unique(translations, axis=0).tolist(),
        },
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "software": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "ase": ase.__version__,
            "mace": mace.__version__,
            "phonopy": phonopy.__version__,
            "torch": torch.__version__,
        },
        "scan": scan_rows,
        "even_polynomial_fit_delta_energy_per_moire_cell": {
            "powers": [2, 4, 6],
            "coefficients_ev_per_angstrom_power": coefficients.tolist(),
            "harmonic_curvature_ev_per_angstrom2": float(2.0 * coefficients[0]),
            "rms_residual_ev_per_moire_cell": float(
                np.sqrt(np.mean((fit_matrix @ coefficients - fit_target) ** 2))
            ),
        },
        "base_same_pattern_curvature": curvature_at(
            base, calculator, pattern, args.curvature_probe, reference_energy=reference
        ),
        "relaxations": relaxations,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURMD_NODENAME",
                "SLURM_CPUS_PER_TASK",
            )
        },
    }
    summary_path = output_dir / "frozen_soft_mode_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

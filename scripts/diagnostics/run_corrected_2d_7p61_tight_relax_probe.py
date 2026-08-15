#!/usr/bin/env python3
"""Tighten one accepted structure and remeasure its lowest accepted S mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--expected-structure-sha256", required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="7.61-degree")
    parser.add_argument("--fmax", type=float, default=1.0e-5)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--qpoint", default="0.5,0.5,0")
    parser.add_argument("--probe-amplitudes", default="0.0025,0.005,0.01")
    parser.add_argument("--threads", type=int, default=28)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vector(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=float)
    if values.shape != (3,):
        raise ValueError("Expected exactly three comma-separated values")
    return values


def max_force(forces: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(forces, axis=1)))


def geometry(atoms) -> dict[str, Any]:
    return {
        "atom_count": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "cell_angstrom": np.asarray(atoms.cell, dtype=float).tolist(),
        "cell_volume_angstrom3": float(atoms.get_volume()),
        "pbc": [bool(value) for value in atoms.pbc],
    }


def build_accepted_mode(phonon, qpoint: np.ndarray):
    from ase import Atoms
    from phonopy.phonon.modulation import Modulation

    atom_count = len(phonon.unitcell)
    phonon.run_qpoints([qpoint], with_eigenvectors=True)
    qpoint_data = phonon.get_qpoints_dict()
    frequencies = np.asarray(qpoint_data["frequencies"][0], dtype=float)
    eigenvectors = np.asarray(qpoint_data["eigenvectors"][0])
    if eigenvectors.shape != (3 * atom_count, 3 * atom_count):
        raise RuntimeError(f"Unexpected eigenvector shape {eigenvectors.shape}")
    mode_index = int(np.argmin(frequencies))
    modulation = Modulation(
        phonon.dynamical_matrix,
        dimension=[2, 2, 1],
        phonon_modes=[(qpoint, mode_index, 1.0, 0.0)],
    )
    modulation.run()
    modulations, phonopy_supercell = modulation.get_modulations_and_supercell()
    imaginary_maximum = float(np.max(np.abs(np.imag(modulations[0]))))
    if imaginary_maximum > 1.0e-10:
        raise RuntimeError(
            f"The commensurate S-point modulation is not real: {imaginary_maximum}"
        )
    pattern = np.real(modulations[0])
    normalization = float(np.max(np.linalg.norm(pattern, axis=1)))
    if normalization <= 0:
        raise RuntimeError("The selected mode has a zero displacement pattern")
    pattern /= normalization
    base = Atoms(
        symbols=phonopy_supercell.symbols,
        positions=np.asarray(phonopy_supercell.positions, dtype=float),
        cell=np.asarray(phonopy_supercell.cell, dtype=float),
        pbc=True,
    )
    return base, pattern, mode_index, float(frequencies[mode_index])


def build_relaxed_supercell(relaxed, reference_supercell):
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    unitcell = PhonopyAtoms(
        symbols=relaxed.get_chemical_symbols(),
        cell=np.asarray(relaxed.cell, dtype=float),
        scaled_positions=relaxed.get_scaled_positions(wrap=True),
        masses=relaxed.get_masses(),
    )
    phonopy_supercell = get_supercell(
        unitcell,
        np.diag([2, 2, 1]),
        is_old_style=True,
    )
    supercell = Atoms(
        symbols=phonopy_supercell.symbols,
        positions=np.asarray(phonopy_supercell.positions, dtype=float),
        cell=np.asarray(phonopy_supercell.cell, dtype=float),
        pbc=True,
    )
    if supercell.get_chemical_symbols() != reference_supercell.get_chemical_symbols():
        raise RuntimeError("Relaxed and reference 2x2 supercells have different ordering")
    if not np.allclose(supercell.cell, reference_supercell.cell, atol=1.0e-8, rtol=0):
        raise RuntimeError("The tight relaxation changed the fixed simulation cell")
    return supercell


def evaluate(atoms, calculator, pattern: np.ndarray, amplitude: float):
    displaced = atoms.copy()
    displaced.positions += amplitude * pattern
    displaced.calc = calculator
    energy = float(displaced.get_potential_energy())
    forces = np.asarray(displaced.get_forces(), dtype=float)
    return energy, forces


def centered_probe(atoms, calculator, pattern: np.ndarray, amplitude: float):
    center_energy, center_forces = evaluate(atoms, calculator, pattern, 0.0)
    minus_energy, minus_forces = evaluate(atoms, calculator, pattern, -amplitude)
    plus_energy, plus_forces = evaluate(atoms, calculator, pattern, amplitude)
    curvature = (plus_energy - 2.0 * center_energy + minus_energy) / amplitude**2
    force_curvature = -float(
        np.sum((plus_forces - minus_forces) * pattern)
        / (2.0 * amplitude * np.sum(pattern**2))
    )
    return {
        "amplitude_angstrom": amplitude,
        "minus_energy_ev": minus_energy,
        "center_energy_ev": center_energy,
        "plus_energy_ev": plus_energy,
        "energy_curvature_ev_per_angstrom2_supercell": curvature,
        "energy_curvature_ev_per_angstrom2_per_moire_cell": curvature / 4.0,
        "force_projected_curvature_ev_per_angstrom2": force_curvature,
        "odd_energy_ev": plus_energy - minus_energy,
        "center_max_force_ev_per_angstrom": max_force(center_forces),
        "minus_max_force_ev_per_angstrom": max_force(minus_forces),
        "plus_max_force_ev_per_angstrom": max_force(plus_forces),
    }


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    import ase
    import mace
    import phonopy
    import torch
    from ase.geometry import find_mic
    from ase.io import read, write
    from ase.optimize import FIRE
    from mace.calculators import MACECalculator
    from phonopy import load

    torch.set_num_threads(args.threads)
    structure = args.structure.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    if sha256_file(structure) != args.expected_structure_sha256:
        raise RuntimeError("Input structure hash mismatch")
    if sha256_file(model_path) != args.expected_model_sha256:
        raise RuntimeError("Model hash mismatch")
    case_json = case_dir / "case.json"
    metadata = json.loads(case_json.read_text(encoding="ascii"))
    structure_yaml = case_dir / metadata["phonopy_structure"]
    force_constants = case_dir / metadata["force_constants"]
    if sha256_file(structure_yaml) != metadata["phonopy_structure_sha256"]:
        raise RuntimeError("Phonopy structure hash mismatch")
    if sha256_file(force_constants) != metadata["force_constants_sha256"]:
        raise RuntimeError("Force-constant hash mismatch")

    phonon = load(
        phonopy_yaml=str(structure_yaml),
        force_constants_filename=str(force_constants),
        symmetrize_fc=False,
    )
    atoms = read(structure)
    atoms.pbc = True
    if len(atoms) != int(metadata["atom_count"]):
        raise RuntimeError("Input atom count does not match the accepted IFC")
    if atoms.get_chemical_symbols() != list(phonon.unitcell.symbols):
        raise RuntimeError("Input atom ordering does not match the accepted IFC")
    if not np.allclose(atoms.cell, phonon.unitcell.cell, atol=1.0e-8, rtol=0):
        raise RuntimeError("Input cell does not match the accepted IFC")

    calculator = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        default_dtype="float64",
    )
    atoms.calc = calculator
    initial_positions = atoms.positions.copy()
    initial_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=float)
    optimizer = FIRE(atoms, logfile=str(output_dir / "tight_relax.log"))
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))
    final_energy = float(atoms.get_potential_energy())
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    if not converged or max_force(final_forces) > args.fmax * (1.0 + 1.0e-6):
        raise RuntimeError("Tight 1x1 relaxation did not satisfy the force gate")

    relaxed_path = output_dir / "tight_relaxed_1x1.extxyz"
    write(relaxed_path, atoms, format="extxyz")
    qpoint = parse_vector(args.qpoint)
    reference_supercell, pattern, mode_index, frequency = build_accepted_mode(
        phonon, qpoint
    )
    relaxed_supercell = build_relaxed_supercell(atoms, reference_supercell)
    probes = [
        centered_probe(relaxed_supercell, calculator, pattern, amplitude)
        for amplitude in sorted(set(float(x) for x in args.probe_amplitudes.split(",")))
    ]
    displacement, _ = find_mic(
        atoms.positions - initial_positions,
        atoms.cell,
        pbc=True,
    )
    repeated_final_forces = np.asarray(atoms.get_forces(), dtype=float)

    payload = {
        "schema_version": 1,
        "description": (
            "Fixed-cell 1x1 tightening followed by a centered frozen-mode test "
            f"along the independently accepted {args.label} S-point eigenvector."
        ),
        "decision_scope": (
            "Tests whether the reported S-point negative curvature is removed by "
            "tightening the residual-force threshold; it is not a new full IFC."
        ),
        "inputs": {
            "structure": str(structure),
            "structure_sha256": sha256_file(structure),
            "case_json": str(case_json),
            "case_json_sha256": sha256_file(case_json),
            "phonopy_structure_sha256": sha256_file(structure_yaml),
            "force_constants_sha256": sha256_file(force_constants),
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
        },
        "tight_relaxation": {
            "optimizer": "ASE FIRE",
            "fixed_cell": True,
            "converged": converged,
            "steps": int(optimizer.nsteps),
            "fmax_target_ev_per_angstrom": args.fmax,
            "initial_energy_ev": initial_energy,
            "final_energy_ev": final_energy,
            "energy_change_ev": final_energy - initial_energy,
            "initial_max_force_ev_per_angstrom": max_force(initial_forces),
            "final_max_force_ev_per_angstrom": max_force(final_forces),
            "zero_step_repeat_max_force_ev_per_angstrom": max_force(
                repeated_final_forces
            ),
            "rms_displacement_angstrom": float(
                np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
            ),
            "maximum_displacement_angstrom": float(
                np.max(np.linalg.norm(displacement, axis=1))
            ),
            "initial_geometry": geometry(read(structure)),
            "final_geometry": geometry(atoms),
            "relaxed_structure": str(relaxed_path),
            "relaxed_structure_sha256": sha256_file(relaxed_path),
        },
        "accepted_mode": {
            "qpoint_reduced": qpoint.tolist(),
            "mode_index_zero_based": mode_index,
            "accepted_ifc_frequency_thz": frequency,
            "normalization": "maximum per-atom displacement norm equals one",
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            "supercell_atom_count": len(relaxed_supercell),
        },
        "centered_probes": probes,
        "gates": {
            "tight_relaxation_pass": bool(
                converged and max_force(final_forces) <= args.fmax * (1.0 + 1.0e-6)
            ),
            "geometry_preserved": bool(
                np.allclose(read(structure).cell, atoms.cell, atol=1.0e-8, rtol=0)
                and len(atoms) == len(read(structure))
                and atoms.get_chemical_symbols() == read(structure).get_chemical_symbols()
            ),
            "all_probe_values_finite": bool(
                all(
                    np.isfinite(value)
                    for row in probes
                    for value in row.values()
                )
            ),
        },
        "software": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "ase": ase.__version__,
            "mace": mace.__version__,
            "phonopy": phonopy.__version__,
            "torch": torch.__version__,
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURMD_NODENAME",
                "SLURM_CPUS_PER_TASK",
            )
        },
    }
    summary = output_dir / "tight_relax_probe_summary.json"
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

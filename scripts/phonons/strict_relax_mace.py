#!/usr/bin/env python3
"""Strict fixed-cell MACE relaxation for the twisted SnSe structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE


def max_force(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max())


def foundation_calculator(device: str):
    from mace.calculators import mace_mp

    return mace_mp(
        model="medium-mpa-0",
        device=device,
        default_dtype="float64",
        dispersion=False,
    )


def relax_structure(atoms, calc, out_dir: Path, fmax: float, max_steps: int):
    atoms = atoms.copy()
    atoms.calc = calc
    initial_energy = float(atoms.get_potential_energy())
    initial_forces = atoms.get_forces()
    start = time.time()
    optimizer = FIRE(
        atoms,
        logfile=str(out_dir / "relax.log"),
        trajectory=str(out_dir / "relax.traj"),
    )
    converged = optimizer.run(fmax=fmax, steps=max_steps)
    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    stats = {
        "converged": bool(converged),
        "optimizer": "FIRE",
        "fmax_target_ev_per_ang": fmax,
        "max_steps": max_steps,
        "n_steps": int(optimizer.nsteps),
        "wall_time_s": time.time() - start,
        "initial_energy_ev": initial_energy,
        "final_energy_ev": final_energy,
        "initial_max_force_ev_per_ang": max_force(initial_forces),
        "final_max_force_ev_per_ang": max_force(final_forces),
        "final_mean_force_ev_per_ang": float(
            np.linalg.norm(final_forces, axis=1).mean()
        ),
    }
    write(out_dir / "relaxed.cif", atoms)
    write(out_dir / "relaxed.extxyz", atoms)
    (out_dir / "relax_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return atoms, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-label", default="MACE-MPA-0 medium-mpa-0")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(in_path)
    atoms.pbc = True
    if args.model_path is None:
        calc = foundation_calculator(args.device)
        model_path = None
        model_sha256 = None
    else:
        from mace.calculators import MACECalculator

        model_path = args.model_path.expanduser().resolve()
        calc = MACECalculator(
            model_paths=str(model_path),
            device=args.device,
            default_dtype="float64",
        )
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    relaxed, stats = relax_structure(
        atoms,
        calc,
        out_dir,
        args.fmax,
        args.max_steps,
    )

    meta = {
        "source_structure": str(in_path),
        "output_dir": str(out_dir),
        "model": args.model_label,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": model_sha256,
        "relaxation": "fixed cell, atomic coordinates only",
        "fmax_target_ev_per_ang": args.fmax,
        "max_steps": args.max_steps,
        "relax_stats": stats,
        "n_atoms": len(relaxed),
        "formula": relaxed.get_chemical_formula(),
        "cell": relaxed.cell.array.tolist(),
    }
    (out_dir / "strict_relax_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full finite-displacement phonon calculation for a relaxed moire supercell.

This is deliberately restartable: every displaced-supercell force array is saved
as an individual ``.npy`` file before force constants and plots are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def build_foundation_calculator(device: str):
    from mace.calculators import mace_mp

    return mace_mp(
        model="medium-mpa-0",
        device=device,
        default_dtype="float64",
        dispersion=False,
    )


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def atoms_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(wrap=True),
        masses=atoms.get_masses(),
    )


def phonopy_to_atoms(cell: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
        masses=cell.masses,
    )


def q_path(npoints: int) -> tuple[list[np.ndarray], list[str], list[bool]]:
    qs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    paths = [np.linspace(qs[i], qs[i + 1], npoints) for i in range(len(qs) - 1)]
    return paths, ["G", "X", "S", "Y", "G"], [True, True, True, False]


def band_distances(paths: list[np.ndarray], cell: np.ndarray) -> tuple[np.ndarray, list[float]]:
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    xvals: list[float] = []
    ticks = [0.0]
    cursor = 0.0
    prev_cart: np.ndarray | None = None
    for path in paths:
        for i, q in enumerate(path):
            q_cart = q @ reciprocal
            if prev_cart is not None and not (i == 0 and len(xvals) > 0):
                cursor += float(np.linalg.norm(q_cart - prev_cart))
            xvals.append(cursor)
            prev_cart = q_cart
        ticks.append(cursor)
    return np.array(xvals), ticks


def max_force(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max())


def parse_supercell(text: str) -> np.ndarray:
    parts = [int(part) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("--supercell must look like 1,1,1 or 4x4x2")
    return np.diag(parts)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    data = vars(args).copy()
    if isinstance(data.get("supercell"), np.ndarray):
        data["supercell"] = data["supercell"].tolist()
    if isinstance(data.get("model_path"), Path):
        data["model_path"] = str(data["model_path"].resolve())
    return data


def compute_forces(
    atoms: Atoms,
    phonon: Phonopy,
    calc,
    force_dir: Path,
    base_forces: np.ndarray,
    start: int,
    stop: int | None,
) -> np.ndarray:
    force_dir.mkdir(parents=True, exist_ok=True)
    displaced = phonon.supercells_with_displacements
    if displaced is None:
        raise RuntimeError("No displaced supercells generated")
    n_total = len(displaced)
    stop = n_total if stop is None else min(stop, n_total)
    t0 = time.time()
    durations = []

    for idx in range(start, stop):
        out = force_dir / f"force_{idx:05d}.npy"
        if out.exists():
            continue
        disp_atoms = phonopy_to_atoms(displaced[idx])
        disp_atoms.calc = calc
        if len(disp_atoms) != len(base_forces):
            raise RuntimeError(
                f"Base-force shape mismatch: displaced supercell has {len(disp_atoms)} atoms, "
                f"base forces have {len(base_forces)} atoms"
            )
        f0 = time.time()
        forces = disp_atoms.get_forces() - base_forces
        durations.append(time.time() - f0)
        np.save(out, forces)
        done = len(list(force_dir.glob("force_*.npy")))
        mean_dt = float(np.mean(durations)) if durations else float("nan")
        elapsed = time.time() - t0
        print(
            f"force {idx + 1}/{n_total} saved; completed={done}/{n_total}; "
            f"last={durations[-1]:.2f}s mean_new={mean_dt:.2f}s elapsed={elapsed:.1f}s",
            flush=True,
        )

    missing = [idx for idx in range(n_total) if not (force_dir / f"force_{idx:05d}.npy").exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} force files; first missing index {missing[0]}")

    return np.array([np.load(force_dir / f"force_{idx:05d}.npy") for idx in range(n_total)])


def plot_band(npz_path: Path, out_png: Path, title: str) -> None:
    data = np.load(npz_path)
    x = data["x"]
    freqs = data["frequencies"]
    ticks = data["ticks"]
    labels = [str(s) for s in data["labels"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for branch in range(freqs.shape[1]):
        ax.plot(x, freqs[:, branch], color="#234f9f", lw=0.18, alpha=0.22)
    ax.axhline(0, color="black", lw=0.8)
    for tick in ticks:
        ax.axvline(float(tick), color="0.83", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("frequency (THz)")
    ax.set_xlabel("q path")
    ax.set_title(title)
    ymin = max(-3.0, float(np.percentile(freqs, 0.1)) - 0.3)
    ymax = float(np.percentile(freqs, 99.9)) + 0.5
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color="0.9", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    force_dir = out_dir / "forces"
    atoms = read(Path(args.structure).resolve())
    atoms.pbc = True

    meta: dict[str, Any] = {
        "model": args.model,
        "structure": str(Path(args.structure).resolve()),
        "out_dir": str(out_dir),
        "python": sys.executable,
        "command": [sys.executable, *sys.argv],
        "package_versions": {
            pkg: package_version(pkg)
            for pkg in [
                "ase",
                "mace-torch",
                "numpy",
                "phonopy",
                "torch",
            ]
        },
        "parameters": jsonable_args(args),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "cell": atoms.cell.array.tolist(),
        "supercell_matrix": args.supercell.tolist(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Loading {args.model} calculator on {args.device}", flush=True)
    if args.model == "mace-checkpoint":
        if args.model_path is None:
            raise ValueError("--model-path is required for mace-checkpoint")
        from mace.calculators import MACECalculator

        model_path = args.model_path.expanduser().resolve()
        calc = MACECalculator(
            model_paths=str(model_path),
            device=args.device,
            default_dtype="float64",
        )
        meta["model_path"] = str(model_path)
        meta["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    else:
        calc = build_foundation_calculator(args.device)

    phonon = Phonopy(
        atoms_to_phonopy(atoms),
        supercell_matrix=args.supercell,
        primitive_matrix=np.eye(3),
        symprec=args.symprec,
        is_symmetry=not args.disable_symmetry,
    )
    base_supercell = phonopy_to_atoms(phonon.supercell)
    base_supercell.calc = calc
    base_forces = base_supercell.get_forces()
    np.save(out_dir / "base_forces.npy", base_forces)
    print(
        f"Unit cell atoms={len(atoms)}; supercell atoms={len(base_supercell)}; "
        f"supercell={args.supercell.tolist()}",
        flush=True,
    )

    phonon.generate_displacements(
        distance=args.displacement,
        is_plusminus=args.plusminus,
        is_diagonal=True,
    )
    displaced = phonon.supercells_with_displacements
    if displaced is None:
        raise RuntimeError("No displaced supercells generated")
    if (
        args.expected_displacement_count is not None
        and len(displaced) != args.expected_displacement_count
    ):
        raise RuntimeError(
            "Displacement-count mismatch: "
            f"expected {args.expected_displacement_count}, generated {len(displaced)}"
        )
    print(f"Generated {len(displaced)} finite displacements", flush=True)

    force_arr = compute_forces(
        atoms,
        phonon,
        calc,
        force_dir,
        base_forces,
        start=args.start,
        stop=args.stop,
    )
    np.save(out_dir / "forces_minus_base.npy", force_arr)

    print("Producing force constants with phonopy traditional finite differences", flush=True)
    phonon.produce_force_constants(
        forces=force_arr,
        calculate_full_force_constants=True,
        fc_calculator="traditional",
        show_drift=True,
    )
    phonon.symmetrize_force_constants(level=1, show_drift=True)
    phonon.save(out_dir / "phonopy_full.yaml", settings={"force_constants": True})

    paths, labels, connections = q_path(args.band_points)
    phonon.run_band_structure(paths, path_connections=connections, labels=labels)
    band = phonon.get_band_structure_dict()
    frequencies = np.concatenate([np.asarray(x) for x in band["frequencies"]], axis=0)
    xvals, ticks = band_distances(paths, atoms.cell.array)

    gamma = frequencies[0]
    stats = {
        "model": args.model,
        "n_displacements": len(displaced),
        "displacement_ang": args.displacement,
        "plusminus": args.plusminus,
        "band_points_per_segment": args.band_points,
        "base_max_force_ev_per_ang": max_force(base_forces),
        "n_unitcell_atoms": len(atoms),
        "n_supercell_atoms": len(base_supercell),
        "supercell_matrix": args.supercell.tolist(),
        "min_frequency_thz": float(frequencies.min()),
        "p01_frequency_thz": float(np.percentile(frequencies, 1)),
        "p05_frequency_thz": float(np.percentile(frequencies, 5)),
        "median_frequency_thz": float(np.median(frequencies)),
        "max_frequency_thz": float(frequencies.max()),
        "negative_mode_count_lt_minus_0p05": int(np.sum(frequencies < -0.05)),
        "negative_mode_fraction_lt_minus_0p05": float(np.mean(frequencies < -0.05)),
        "gamma_acoustic_abs_thz": np.sort(np.abs(gamma))[:3].tolist(),
        "gamma_acoustic_max_abs_thz": float(np.sort(np.abs(gamma))[:3].max()),
    }

    np.savez_compressed(
        out_dir / "full_phonon_band.npz",
        x=xvals,
        ticks=np.array(ticks),
        labels=np.array(labels),
        frequencies=frequencies,
    )
    (out_dir / "full_phonon_stats.json").write_text(json.dumps(stats, indent=2))
    plot_band(
        out_dir / "full_phonon_band.npz",
        out_dir / "full_phonon_band.png",
        f"{args.model} full finite-displacement phonon band",
    )
    print(json.dumps(stats, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["mace-mpa-0", "mace-checkpoint"],
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--supercell", type=parse_supercell, default=np.eye(3, dtype=int))
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--band-points", type=int, default=18)
    parser.add_argument("--plusminus", action="store_true")
    parser.add_argument("--disable-symmetry", action="store_true")
    parser.add_argument("--expected-displacement-count", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

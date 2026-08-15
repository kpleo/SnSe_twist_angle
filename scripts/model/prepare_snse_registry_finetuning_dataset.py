#!/usr/bin/env python3
"""Prepare leakage-resistant SnSe registry data for MACE fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


TRAIN_REGISTRIES = (
    "ix00iy00",
    "ix00iy01",
    "ix01iy01",
    "ix02iy01",
    "ix01iy02",
    "ix02iy02",
)
VALID_REGISTRIES = ("ix01iy00", "ix02iy00")
TEST_REGISTRIES = ("ix00iy02",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work"),
    )
    parser.add_argument("--phase-a-stride", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path) -> list[Atoms]:
    frames = read(path, index=":")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames in {path}")
    for frame in frames:
        if len(frame) != 8 or not frame.pbc.all():
            raise ValueError(f"invalid frame in {path}")
        if "REF_energy" not in frame.info or "REF_forces" not in frame.arrays:
            raise ValueError(f"missing labels in {path}")
        if frame.arrays["REF_forces"].shape != (8, 3):
            raise ValueError(f"invalid force shape in {path}")
    return frames


def prepare_frame(
    source: Atoms,
    registry: str,
    phase: str,
    role: str,
    energy_shift_ev: float,
) -> Atoms:
    frame = source.copy()
    frame.info["REF_energy"] = float(source.info["REF_energy"]) + energy_shift_ev
    frame.info["registry"] = registry
    frame.info["data_phase"] = phase
    frame.info["data_role"] = role
    frame.info["energy_alignment_ev"] = energy_shift_ev
    frame.info["config_type"] = f"snse_registry_{phase}"
    frame.arrays["REF_forces"] = np.asarray(
        source.arrays["REF_forces"], dtype=float
    ).copy()
    return frame


def select_phase_a(frames: list[Atoms], stride: int) -> list[Atoms]:
    if stride < 1:
        raise ValueError("phase-a stride must be positive")
    indices = list(range(0, max(1, len(frames) - 1), stride))
    return [frames[index] for index in indices]


def dataset_summary(frames: list[Atoms]) -> dict[str, object]:
    energies = np.asarray([float(frame.info["REF_energy"]) for frame in frames])
    force_norms = np.concatenate(
        [np.linalg.norm(frame.arrays["REF_forces"], axis=1) for frame in frames]
    )
    phase_counts: dict[str, int] = {}
    registry_counts: dict[str, int] = {}
    for frame in frames:
        phase = str(frame.info["data_phase"])
        registry = str(frame.info["registry"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        registry_counts[registry] = registry_counts.get(registry, 0) + 1
    return {
        "frame_count": len(frames),
        "phase_counts": dict(sorted(phase_counts.items())),
        "registry_counts": dict(sorted(registry_counts.items())),
        "energy_min_ev": float(energies.min()),
        "energy_max_ev": float(energies.max()),
        "force_norm_rms_ev_per_a": float(np.sqrt(np.mean(force_norms**2))),
        "force_norm_max_ev_per_a": float(force_norms.max()),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    source_root = root / "training_data"
    output = source_root / "finetuning_split"
    output.mkdir(parents=True, exist_ok=True)

    registry_sets = {
        "train": TRAIN_REGISTRIES,
        "valid": VALID_REGISTRIES,
        "test": TEST_REGISTRIES,
    }
    flat = [registry for values in registry_sets.values() for registry in values]
    expected = [f"ix{ix:02d}iy{iy:02d}" for ix in range(3) for iy in range(3)]
    if len(flat) != len(set(flat)) or sorted(flat) != expected:
        raise SystemExit("registry split is not a disjoint partition of the 3x3 grid")

    raw: dict[str, dict[str, list[Atoms]]] = {"phase_a": {}, "phase_b": {}}
    for phase in raw:
        for registry in expected:
            raw[phase][registry] = load_registry(
                source_root / phase / f"{phase}_{registry}.extxyz"
            )

    shifts: dict[str, float] = {}
    for registry in expected:
        phase_a_final = float(raw["phase_a"][registry][-1].info["REF_energy"])
        phase_b_initial = float(raw["phase_b"][registry][0].info["REF_energy"])
        shifts[registry] = phase_b_initial - phase_a_final

    datasets: dict[str, list[Atoms]] = {"train": [], "valid": [], "test": []}
    all_finetune: list[Atoms] = []
    strict_finals: list[Atoms] = []

    role_by_registry = {
        registry: role for role, registries in registry_sets.items() for registry in registries
    }
    for registry in expected:
        role = role_by_registry[registry]
        phase_a_frames = select_phase_a(
            raw["phase_a"][registry], args.phase_a_stride
        )
        phase_b_frames = raw["phase_b"][registry]
        prepared_a = [
            prepare_frame(frame, registry, "phase_a", role, shifts[registry])
            for frame in phase_a_frames
        ]
        prepared_b = [
            prepare_frame(frame, registry, "phase_b", role, 0.0)
            for frame in phase_b_frames
        ]
        datasets[role].extend(prepared_a + prepared_b)
        all_finetune.extend(prepared_a + prepared_b)
        strict_finals.append(prepared_b[-1])

    output_files: dict[str, dict[str, object]] = {}
    for name, frames in {
        **datasets,
        "all_finetune": all_finetune,
        "strict_finals": strict_finals,
    }.items():
        path = output / f"{name}.extxyz"
        write(path, frames, format="extxyz")
        reread = read(path, index=":")
        if len(reread) != len(frames):
            raise RuntimeError(f"round-trip frame mismatch for {path}")
        output_files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            **dataset_summary(frames),
        }

    manifest = {
        "schema_version": 1,
        "purpose": "registry-grouped MACE fine-tuning and independent evaluation",
        "phase_a_stride": args.phase_a_stride,
        "phase_a_final_frame_excluded": True,
        "phase_a_energy_alignment": (
            "Per-registry additive shift matching the Phase-A final energy to "
            "the Phase-B initial energy; forces are unchanged."
        ),
        "registry_split": registry_sets,
        "energy_alignment_ev": shifts,
        "files": output_files,
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

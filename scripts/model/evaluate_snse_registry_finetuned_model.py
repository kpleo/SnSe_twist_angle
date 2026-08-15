#!/usr/bin/env python3
"""Evaluate a fine-tuned SnSe registry model on grouped DFT trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-prefix", default="MACE")
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    if reference.size < 2 or np.std(reference) == 0.0 or np.std(prediction) == 0.0:
        return float("nan")
    return float(pearsonr(reference, prediction).statistic)


def subset_metrics(frames: list[Atoms], prefix: str) -> dict[str, float | int]:
    ref_energy = np.asarray([float(frame.info["REF_energy"]) for frame in frames])
    pred_energy = np.asarray(
        [float(frame.info[f"{prefix}_energy"]) for frame in frames]
    )
    energy_error = pred_energy - ref_energy
    centered_energy_error = energy_error - np.mean(energy_error)
    ref_forces = np.concatenate(
        [np.asarray(frame.arrays["REF_forces"]).reshape(-1) for frame in frames]
    )
    pred_forces = np.concatenate(
        [
            np.asarray(frame.arrays[f"{prefix}_forces"]).reshape(-1)
            for frame in frames
        ]
    )
    force_error = pred_forces - ref_forces
    denominator = float(np.dot(ref_forces, ref_forces))
    return {
        "frame_count": len(frames),
        "energy_bias_mev_per_atom": float(1000.0 * np.mean(energy_error) / 8.0),
        "energy_rmse_mev_per_atom": float(
            1000.0 * np.sqrt(np.mean(energy_error**2)) / 8.0
        ),
        "energy_mae_mev_per_atom": float(
            1000.0 * np.mean(np.abs(energy_error)) / 8.0
        ),
        "energy_offset_corrected_rmse_mev_per_atom": float(
            1000.0 * np.sqrt(np.mean(centered_energy_error**2)) / 8.0
        ),
        "force_component_rmse_ev_per_a": float(np.sqrt(np.mean(force_error**2))),
        "force_component_mae_ev_per_a": float(np.mean(np.abs(force_error))),
        "force_component_pearson_r": correlation(ref_forces, pred_forces),
        "force_component_origin_slope": float(
            np.dot(ref_forces, pred_forces) / denominator
        ),
        "force_component_normalized_rmse": float(
            np.sqrt(np.mean(force_error**2))
            / np.sqrt(np.mean(ref_forces**2))
        ),
    }


def strict_final_frames(frames: list[Atoms]) -> list[Atoms]:
    registries = sorted({str(frame.info["registry"]) for frame in frames})
    selected = []
    for registry in registries:
        candidates = [
            frame
            for frame in frames
            if str(frame.info["registry"]) == registry
            and str(frame.info["data_phase"]) == "phase_b"
        ]
        if not candidates:
            raise ValueError(f"no Phase-B frames for {registry}")
        selected.append(
            max(candidates, key=lambda frame: int(frame.info["source_step"]))
        )
    if len(selected) != 9:
        raise ValueError(f"expected nine strict final frames, found {len(selected)}")
    return selected


def stacking_metrics(
    frames: list[Atoms], prefix: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    registries = [str(frame.info["registry"]) for frame in frames]
    reference = np.asarray([float(frame.info["REF_energy"]) for frame in frames])
    prediction = np.asarray(
        [float(frame.info[f"{prefix}_energy"]) for frame in frames]
    )
    reference -= reference.min()
    prediction -= prediction.min()
    error = prediction - reference
    denominator = float(np.dot(reference, reference))
    reference_minimum = registries[int(np.argmin(reference))]
    predicted_minimum = registries[int(np.argmin(prediction))]
    rows = [
        {
            "registry": registry,
            "reference_relative_energy_mev_per_bilayer": float(1000.0 * ref),
            "predicted_relative_energy_mev_per_bilayer": float(1000.0 * pred),
            "error_mev_per_bilayer": float(1000.0 * delta),
        }
        for registry, ref, pred, delta in zip(
            registries, reference, prediction, error
        )
    ]
    summary = {
        "reference_minimum_registry": reference_minimum,
        "predicted_minimum_registry": predicted_minimum,
        "minimum_registry_matches": predicted_minimum == reference_minimum,
        "pearson_r": correlation(reference, prediction),
        "spearman_r": float(spearmanr(reference, prediction).statistic),
        "origin_slope": float(np.dot(reference, prediction) / denominator),
        "rmse_mev_per_bilayer": float(1000.0 * np.sqrt(np.mean(error**2))),
        "mae_mev_per_bilayer": float(1000.0 * np.mean(np.abs(error))),
        "max_abs_error_mev_per_bilayer": float(1000.0 * np.max(np.abs(error))),
    }
    return summary, rows


def main() -> int:
    args = parse_args()
    predictions = args.predictions.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = read(predictions, index=":")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("prediction file contains no frames")

    subsets: dict[str, dict[str, float | int]] = {}
    for role in ("train", "valid", "test"):
        role_frames = [
            frame for frame in frames if str(frame.info["data_role"]) == role
        ]
        subsets[role] = subset_metrics(role_frames, args.prediction_prefix)
        for phase in ("phase_a", "phase_b"):
            phase_frames = [
                frame
                for frame in role_frames
                if str(frame.info["data_phase"]) == phase
            ]
            subsets[f"{role}_{phase}"] = subset_metrics(
                phase_frames, args.prediction_prefix
            )

    stacking, rows = stacking_metrics(
        strict_final_frames(frames), args.prediction_prefix
    )
    gates = {
        "validation_energy_rmse_le_5_mev_per_atom": (
            subsets["valid"]["energy_rmse_mev_per_atom"] <= 5.0
        ),
        "test_energy_rmse_le_5_mev_per_atom": (
            subsets["test"]["energy_rmse_mev_per_atom"] <= 5.0
        ),
        "validation_force_rmse_le_0p030_ev_per_a": (
            subsets["valid"]["force_component_rmse_ev_per_a"] <= 0.030
        ),
        "test_force_rmse_le_0p040_ev_per_a": (
            subsets["test"]["force_component_rmse_ev_per_a"] <= 0.040
        ),
        "strict_final_minimum_registry_matches": stacking[
            "minimum_registry_matches"
        ],
        "strict_final_pearson_ge_0p95": stacking["pearson_r"] >= 0.95,
        "strict_final_spearman_ge_0p90": stacking["spearman_r"] >= 0.90,
        "strict_final_slope_between_0p8_and_1p2": (
            0.8 <= stacking["origin_slope"] <= 1.2
        ),
        "strict_final_rmse_le_20_mev_per_bilayer": (
            stacking["rmse_mev_per_bilayer"] <= 20.0
        ),
    }
    gates["all_pass"] = all(bool(value) for value in gates.values())

    csv_path = output / "strict_final_stacking.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": 1,
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "prediction_prefix": args.prediction_prefix,
        "model": str(args.model.resolve()) if args.model else None,
        "model_sha256": sha256_file(args.model.resolve()) if args.model else None,
        "subsets": subsets,
        "strict_final_stacking": stacking,
        "gates": gates,
    }
    result_path = output / "model_evaluation.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gates["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

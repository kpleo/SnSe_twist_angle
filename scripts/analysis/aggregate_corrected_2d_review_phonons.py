#!/usr/bin/env python3
"""Audit and aggregate the reviewer-response analytical phonon calculations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ANGLE_CASES = (
    ("8p77deg", 8.771750236347492),
    ("7p61deg", 7.611378517094657),
    ("6p02deg", 6.017284864032033),
    ("4p78deg", 4.780191847199163),
    ("3p82deg", 3.822553729274344),
    ("3p58deg", 3.583321698471974),
    ("3p18deg", 3.1847385367204093),
)
KINDS = ("twist", "control_lower", "control_upper")
ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-asr-dir", type=Path)
    parser.add_argument("--path18-dir", type=Path)
    parser.add_argument(
        "--production-job-id",
        help="Optional Slurm array job id required for the 21 production JSON files",
    )
    parser.add_argument(
        "--asr-job-id",
        help="Optional Slurm array job id required for the raw-ASR triplet",
    )
    parser.add_argument(
        "--path18-job-id",
        help="Optional Slurm array job id required for the 18-point triplets",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_case(
    payload: dict[str, Any],
    label: str,
    kind: str,
    expected_angle: float,
    *,
    expected_job_id: str | None,
    expected_task_index: int,
    expected_band_points: int,
    expected_path_qpoints: int,
    require_q_grid: bool,
    require_asr: bool,
    expected_temperatures: tuple[str, ...],
    expected_cutoffs: tuple[str, ...],
) -> None:
    case = payload["case"]
    if case["label"] != label or case["kind"] != kind:
        raise RuntimeError(f"case mismatch for {label}/{kind}: {case}")
    if abs(float(case["angle_deg"]) - expected_angle) > 1e-8:
        raise RuntimeError(f"angle mismatch for {label}/{kind}")
    if not payload["method"].startswith(
        "Phonopy dynamical-matrix derivative group velocities"
    ):
        raise RuntimeError(f"unexpected velocity method for {label}/{kind}")
    if int(payload["band_points_per_segment"]) != expected_band_points:
        raise RuntimeError(f"unexpected path sampling for {label}/{kind}")
    if expected_job_id is not None:
        if payload["slurm"].get("SLURM_ARRAY_JOB_ID") != expected_job_id:
            raise RuntimeError(f"unexpected Slurm provenance for {label}/{kind}")
        if int(payload["slurm"]["SLURM_ARRAY_TASK_ID"]) != expected_task_index:
            raise RuntimeError(f"unexpected task provenance for {label}/{kind}")
    if (
        require_asr
        and float(case["asr_max_translational_row_sum_ev_per_angstrom2"])
        > 1e-10
    ):
        raise RuntimeError(f"ASR gate failed for {label}/{kind}")
    if int(payload["path_qpoint_count"]) != expected_path_qpoints:
        raise RuntimeError(f"path count mismatch for {label}/{kind}")
    if not math.isfinite(float(payload["path_minimum_frequency_thz"])):
        raise RuntimeError(f"non-finite path minimum for {label}/{kind}")
    for temperature in expected_temperatures:
        for cutoff in expected_cutoffs:
            metric = payload["metrics"][temperature][cutoff]
            for key in ("cv2_mean_proxy", "cv_weighted_rms_velocity_m_per_s"):
                if not math.isfinite(float(metric[key])):
                    raise RuntimeError(
                        f"non-finite {key} for {label}/{kind}/{temperature}/{cutoff}"
                    )
    if require_q_grid:
        q_grid = payload["q_grid_stability"]
        if q_grid["grid"] != [4, 4, 1] or int(q_grid["qpoint_count"]) != 16:
            raise RuntimeError(f"q-grid mismatch for {label}/{kind}")
        if not math.isfinite(float(q_grid["minimum_frequency_thz"])):
            raise RuntimeError(f"non-finite q-grid minimum for {label}/{kind}")


def aggregate_angle_rows(cases: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, angle in ANGLE_CASES:
        twist = cases[(label, "twist")]
        lower = cases[(label, "control_lower")]
        upper = cases[(label, "control_upper")]
        for temperature in (100, 300, 500, 700):
            for cutoff in (0.02, 0.05, 0.10):
                key_t = str(temperature)
                key_c = f"{cutoff:.2f}"
                tm = twist["metrics"][key_t][key_c]
                lm = lower["metrics"][key_t][key_c]
                um = upper["metrics"][key_t][key_c]
                control_p = 0.5 * (
                    float(lm["cv2_mean_proxy"]) + float(um["cv2_mean_proxy"])
                )
                control_v = 0.5 * (
                    float(lm["cv_weighted_rms_velocity_m_per_s"])
                    + float(um["cv_weighted_rms_velocity_m_per_s"])
                )
                p_ratio_candidates = (
                    float(tm["cv2_mean_proxy"]) / float(lm["cv2_mean_proxy"]),
                    float(tm["cv2_mean_proxy"]) / float(um["cv2_mean_proxy"]),
                )
                v_ratio_candidates = (
                    float(tm["cv_weighted_rms_velocity_m_per_s"])
                    / float(lm["cv_weighted_rms_velocity_m_per_s"]),
                    float(tm["cv_weighted_rms_velocity_m_per_s"])
                    / float(um["cv_weighted_rms_velocity_m_per_s"]),
                )
                rows.append(
                    {
                        "label": label,
                        "angle_deg": angle,
                        "temperature_k": temperature,
                        "cutoff_thz": cutoff,
                        "twist_p_m2_per_s2": tm["cv2_mean_proxy"],
                        "control_lower_p_m2_per_s2": lm["cv2_mean_proxy"],
                        "control_upper_p_m2_per_s2": um["cv2_mean_proxy"],
                        "control_mean_p_m2_per_s2": control_p,
                        "p_twist_over_control_mean": float(tm["cv2_mean_proxy"])
                        / control_p,
                        "p_ratio_control_range_min": min(p_ratio_candidates),
                        "p_ratio_control_range_max": max(p_ratio_candidates),
                        "twist_vrms_m_per_s": tm[
                            "cv_weighted_rms_velocity_m_per_s"
                        ],
                        "control_lower_vrms_m_per_s": lm[
                            "cv_weighted_rms_velocity_m_per_s"
                        ],
                        "control_upper_vrms_m_per_s": um[
                            "cv_weighted_rms_velocity_m_per_s"
                        ],
                        "control_mean_vrms_m_per_s": control_v,
                        "vrms_twist_over_control_mean": float(
                            tm["cv_weighted_rms_velocity_m_per_s"]
                        )
                        / control_v,
                        "vrms_ratio_control_range_min": min(v_ratio_candidates),
                        "vrms_ratio_control_range_max": max(v_ratio_candidates),
                        "twist_heat_capacity_fraction_below_1p5_thz": tm[
                            "fraction_heat_capacity_below_1p5_thz"
                        ],
                        "control_lower_heat_capacity_fraction_below_1p5_thz": lm[
                            "fraction_heat_capacity_below_1p5_thz"
                        ],
                        "control_upper_heat_capacity_fraction_below_1p5_thz": um[
                            "fraction_heat_capacity_below_1p5_thz"
                        ],
                        "twist_cv2_fraction_below_1p5_thz": tm[
                            "fraction_cv2_below_1p5_thz"
                        ],
                        "control_lower_cv2_fraction_below_1p5_thz": lm[
                            "fraction_cv2_below_1p5_thz"
                        ],
                        "control_upper_cv2_fraction_below_1p5_thz": um[
                            "fraction_cv2_below_1p5_thz"
                        ],
                    }
                )
    return rows


def aggregate_spectral_rows(
    cases: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, angle in ANGLE_CASES:
        values = {
            kind: cases[(label, kind)]["spectral_bins_300k_cutoff_0p05"]
            for kind in KINDS
        }
        lengths = {len(value) for value in values.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"spectral bin mismatch for {label}")
        for index, twist in enumerate(values["twist"]):
            lower = values["control_lower"][index]
            upper = values["control_upper"][index]
            for key in ("low_thz", "high_thz"):
                if not (
                    float(twist[key]) == float(lower[key]) == float(upper[key])
                ):
                    raise RuntimeError(f"spectral edge mismatch for {label}")
            lower_v = lower["rms_velocity_m_per_s"]
            upper_v = upper["rms_velocity_m_per_s"]
            control_v = (
                None
                if lower_v is None or upper_v is None
                else 0.5 * (float(lower_v) + float(upper_v))
            )
            twist_v = twist["rms_velocity_m_per_s"]
            rows.append(
                {
                    "label": label,
                    "angle_deg": angle,
                    "bin_low_thz": twist["low_thz"],
                    "bin_high_thz": twist["high_thz"],
                    "bin_mid_thz": 0.5
                    * (float(twist["low_thz"]) + float(twist["high_thz"])),
                    "twist_rms_velocity_m_per_s": twist_v,
                    "control_lower_rms_velocity_m_per_s": lower_v,
                    "control_upper_rms_velocity_m_per_s": upper_v,
                    "control_mean_rms_velocity_m_per_s": control_v,
                    "velocity_ratio": (
                        None
                        if twist_v is None or control_v in (None, 0.0)
                        else float(twist_v) / control_v
                    ),
                    "twist_cv2_share": twist["cv2_share"],
                    "control_lower_cv2_share": lower["cv2_share"],
                    "control_upper_cv2_share": upper["cv2_share"],
                    "twist_heat_capacity_share": twist["heat_capacity_share"],
                    "control_lower_heat_capacity_share": lower[
                        "heat_capacity_share"
                    ],
                    "control_upper_heat_capacity_share": upper[
                        "heat_capacity_share"
                    ],
                }
            )
    return rows


def descriptor(payload: dict[str, Any]) -> tuple[float, float]:
    metric = payload["metrics"]["300"]["0.05"]
    return (
        float(metric["cv2_mean_proxy"]),
        float(metric["cv_weighted_rms_velocity_m_per_s"]),
    )


def aggregate_asr_rows(
    cases: dict[tuple[str, str], dict[str, Any]], raw_dir: Path, job_id: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for kind in KINDS:
        raw_path = raw_dir / f"8p77deg_{kind}_raw.json"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw = read_json(raw_path)
        audit_case(
            raw,
            "8p77deg",
            kind,
            dict(ANGLE_CASES)["8p77deg"],
            expected_job_id=job_id,
            expected_task_index=KINDS.index(kind),
            expected_band_points=9,
            expected_path_qpoints=36,
            require_q_grid=False,
            require_asr=False,
            expected_temperatures=("300",),
            expected_cutoffs=("0.05",),
        )
        production = cases[("8p77deg", kind)]
        raw_cv2, raw_vrms = descriptor(raw)
        asr_cv2, asr_vrms = descriptor(production)
        rows.append(
            {
                "kind": kind,
                "raw_cv2_m2_per_s2": raw_cv2,
                "asr_cv2_m2_per_s2": asr_cv2,
                "cv2_change_percent": 100.0 * (asr_cv2 / raw_cv2 - 1.0),
                "raw_vrms_m_per_s": raw_vrms,
                "asr_vrms_m_per_s": asr_vrms,
                "vrms_change_percent": 100.0 * (asr_vrms / raw_vrms - 1.0),
                "raw_path_minimum_frequency_thz": raw[
                    "path_minimum_frequency_thz"
                ],
                "asr_path_minimum_frequency_thz": production[
                    "path_minimum_frequency_thz"
                ],
            }
        )
        sources.append(
            {
                "path": provenance_path(raw_path),
                "sha256": sha256_file(raw_path),
                "kind": kind,
            }
        )
    return rows, sources


def triplet_ratios(
    triplet: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    twist_cv2, twist_vrms = descriptor(triplet["twist"])
    lower_cv2, lower_vrms = descriptor(triplet["control_lower"])
    upper_cv2, upper_vrms = descriptor(triplet["control_upper"])
    return (
        twist_cv2 / (0.5 * (lower_cv2 + upper_cv2)),
        twist_vrms / (0.5 * (lower_vrms + upper_vrms)),
    )


def aggregate_path_density_rows(
    cases: dict[tuple[str, str], dict[str, Any]],
    path18_dir: Path,
    job_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    angle_lookup = dict(ANGLE_CASES)
    for label in ("8p77deg", "4p78deg"):
        path9 = {kind: cases[(label, kind)] for kind in KINDS}
        path18: dict[str, dict[str, Any]] = {}
        for kind in KINDS:
            path = path18_dir / f"{label}_{kind}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = read_json(path)
            task_base = 0 if label == "8p77deg" else 9
            audit_case(
                payload,
                label,
                kind,
                angle_lookup[label],
                expected_job_id=job_id,
                expected_task_index=task_base + KINDS.index(kind),
                expected_band_points=18,
                expected_path_qpoints=72,
                require_q_grid=False,
                require_asr=True,
                expected_temperatures=("300",),
                expected_cutoffs=("0.05",),
            )
            path18[kind] = payload
            sources.append(
                {
                    "path": provenance_path(path),
                    "sha256": sha256_file(path),
                    "label": label,
                    "kind": kind,
                }
            )
        cv2_9, vrms_9 = triplet_ratios(path9)
        cv2_18, vrms_18 = triplet_ratios(path18)
        rows.append(
            {
                "label": label,
                "angle_deg": angle_lookup[label],
                "path9_cv2_ratio": cv2_9,
                "path18_cv2_ratio": cv2_18,
                "cv2_ratio_change_percent": 100.0 * (cv2_18 / cv2_9 - 1.0),
                "path9_vrms_ratio": vrms_9,
                "path18_vrms_ratio": vrms_18,
                "vrms_ratio_change_percent": 100.0 * (vrms_18 / vrms_9 - 1.0),
            }
        )
    return rows, sources


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[tuple[str, str], dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    for angle_index, (label, angle) in enumerate(ANGLE_CASES):
        for kind in KINDS:
            path = input_dir / f"{label}_{kind}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = read_json(path)
            audit_case(
                payload,
                label,
                kind,
                angle,
                expected_job_id=args.production_job_id,
                expected_task_index=3 * angle_index + KINDS.index(kind),
                expected_band_points=9,
                expected_path_qpoints=36,
                require_q_grid=kind == "twist",
                require_asr=True,
                expected_temperatures=("100", "300", "500", "700"),
                expected_cutoffs=("0.02", "0.05", "0.10"),
            )
            cases[(label, kind)] = payload
            source_files.append(
                {
                    "path": provenance_path(path),
                    "sha256": sha256_file(path),
                    "label": label,
                    "kind": kind,
                }
            )

    angle_rows = aggregate_angle_rows(cases)
    spectral_rows = aggregate_spectral_rows(cases)
    stability_rows = []
    for label, angle in ANGLE_CASES:
        twist = cases[(label, "twist")]
        qgrid = twist["q_grid_stability"]
        stability_rows.append(
            {
                "label": label,
                "angle_deg": angle,
                "path_minimum_frequency_thz": twist[
                    "path_minimum_frequency_thz"
                ],
                "path_modes_below_minus_0p05_thz": twist[
                    "path_modes_below_minus_0p05_thz"
                ],
                "q_grid_minimum_frequency_thz": qgrid[
                    "minimum_frequency_thz"
                ],
                "q_grid_minimum_qpoint": json.dumps(
                    qgrid["minimum_qpoint"], separators=(",", ":")
                ),
                "q_grid_modes_below_minus_0p05_thz": qgrid[
                    "total_modes_below_minus_0p05_thz"
                ],
                "gamma_six_lowest_frequencies_thz": json.dumps(
                    twist["gamma_six_lowest_frequencies_thz"],
                    separators=(",", ":"),
                ),
                "raw_asr_row_sum_ev_per_angstrom2": twist["case"][
                    "raw_max_translational_row_sum_ev_per_angstrom2"
                ],
                "production_asr_row_sum_ev_per_angstrom2": twist["case"][
                    "asr_max_translational_row_sum_ev_per_angstrom2"
                ],
            }
        )

    write_csv(output_dir / "review_analytical_velocity_angle_summary.csv", angle_rows)
    write_csv(output_dir / "review_analytical_velocity_spectral_bins.csv", spectral_rows)
    write_csv(output_dir / "review_q_grid_stability.csv", stability_rows)

    raw_asr: list[dict[str, Any]] = []
    if args.raw_asr_dir:
        raw_dir = args.raw_asr_dir.resolve()
        asr_rows, raw_asr = aggregate_asr_rows(cases, raw_dir, args.asr_job_id)
        write_csv(output_dir / "review_asr_sensitivity.csv", asr_rows)

    path_density_sources: list[dict[str, Any]] = []
    if args.path18_dir:
        path18_dir = args.path18_dir.resolve()
        path_density_rows, path_density_sources = aggregate_path_density_rows(
            cases, path18_dir, args.path18_job_id
        )
        write_csv(
            output_dir / "review_path_density_sensitivity.csv", path_density_rows
        )

    summary = {
        "schema_version": 1,
        "description": "Analytical group-velocity and low-frequency reviewer-response evidence.",
        "production_job_id": args.production_job_id,
        "asr_job_id": args.asr_job_id,
        "path18_job_id": args.path18_job_id,
        "case_count": len(cases),
        "angle_count": len(ANGLE_CASES),
        "source_files": source_files,
        "raw_asr_sensitivity": raw_asr,
        "path_density_sources": path_density_sources,
        "output_files": {
            "angle_summary": "review_analytical_velocity_angle_summary.csv",
            "spectral_bins": "review_analytical_velocity_spectral_bins.csv",
            "q_grid_stability": "review_q_grid_stability.csv",
            "asr_sensitivity": (
                "review_asr_sensitivity.csv" if args.raw_asr_dir else None
            ),
            "path_density_sensitivity": (
                "review_path_density_sensitivity.csv" if args.path18_dir else None
            ),
        },
    }
    (output_dir / "review_analytical_velocity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

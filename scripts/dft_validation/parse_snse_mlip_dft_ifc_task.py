#!/usr/bin/env python3
"""Validate inputs and parse one SnSe DFT-MLIP IFC benchmark VASP task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


SUPPORTED_EDIFF_EV = (1.0e-9, 1.0e-8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def vector_norm(vector: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def load_metadata(task_index: int, task_name: str, group: str) -> dict[str, Any]:
    path = Path("task_metadata.json")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("Unexpected task schema")
    if metadata.get("task_index") != task_index:
        raise ValueError("Task index mismatch")
    if metadata.get("task_name") != task_name:
        raise ValueError("Task name mismatch")
    if metadata.get("group") != group:
        raise ValueError("Task group mismatch")
    return metadata


def preflight(metadata: dict[str, Any]) -> dict[str, Any]:
    expected_kmesh = [16, 16, 1] if metadata["group"] == "one_by_one" else [8, 8, 1]
    expected_atoms = 8 if metadata["group"] == "one_by_one" else 32
    ediff_ev = float(metadata.get("ediff_ev", 0.0))
    ediff_supported = any(
        math.isclose(ediff_ev, value, rel_tol=0.0, abs_tol=1.0e-15)
        for value in SUPPORTED_EDIFF_EV
    )
    protocol_revision = metadata.get("protocol_revision", "original_ediff_1e-9")
    protocol_registered = bool(
        (
            math.isclose(ediff_ev, 1.0e-9, rel_tol=0.0, abs_tol=1.0e-15)
            and protocol_revision == "original_ediff_1e-9"
        )
        or (
            math.isclose(ediff_ev, 1.0e-8, rel_tol=0.0, abs_tol=1.0e-15)
            and protocol_revision == "ediff_1e-8_reference_restart"
        )
        or (
            math.isclose(ediff_ev, 1.0e-8, rel_tol=0.0, abs_tol=1.0e-15)
            and protocol_revision == "ediff_1e-8_fresh_scf"
            and metadata.get("use_reference_wavecar_seed") is False
            and metadata.get("use_previous_reference_wavecar_seed") is False
            and metadata.get("write_wavecar_for_displacement_seed") is False
        )
    )
    expected_istart = int(
        metadata.get("use_reference_wavecar_seed") is True
        or metadata.get("use_previous_reference_wavecar_seed") is True
    )
    incar_match = re.search(
        r"(?im)^\s*ISTART\s*=\s*([0-9]+)\s*$",
        Path("INCAR").read_text(encoding="utf-8"),
    )
    checks = {
        "functional": metadata.get("functional") == "PBE-D3(BJ)",
        "potcar_labels": metadata.get("potcar_labels") == ["Sn_d", "Se"],
        "encut": metadata.get("encut_ev") == 600,
        "ediff_supported": ediff_supported,
        "protocol_registered": protocol_registered,
        "istart": bool(incar_match and int(incar_match.group(1)) == expected_istart),
        "kmesh": metadata.get("kmesh") == expected_kmesh,
        "isym": metadata.get("isym") == 0,
        "static": metadata.get("static_calculation") is True,
        "atom_count": metadata.get("atom_count") == expected_atoms,
        "minimum_distance": float(
            metadata.get("minimum_periodic_distance_angstrom", 0.0)
        )
        >= 1.5,
        "vacuum": float(metadata.get("periodic_vacuum_gap_angstrom", 0.0)) >= 30.0,
        "poscar_hash": sha256_file(Path("POSCAR")) == metadata.get("poscar_sha256"),
        "incar_hash": sha256_file(Path("INCAR")) == metadata.get("incar_sha256"),
        "kpoints_hash": sha256_file(Path("KPOINTS"))
        == metadata.get("kpoints_sha256"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Input preflight failed: {failed}")
    return {"quality_gate_pass": True, "checks": checks}


def parse_force_blocks(lines: list[str], expected_atoms: int) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    for line_index, line in enumerate(lines):
        if "TOTAL-FORCE (eV/Angst)" not in line:
            continue
        rows: list[list[float]] = []
        for candidate in lines[line_index + 2 : line_index + 2 + expected_atoms]:
            fields = candidate.split()
            if len(fields) < 6:
                break
            try:
                rows.append([float(value) for value in fields[:6]])
            except ValueError:
                break
        if len(rows) == expected_atoms:
            blocks.append(rows)
    return blocks


def postprocess(
    metadata: dict[str, Any], vasp_exit: int, sync_dir: Path
) -> dict[str, Any]:
    outcar = Path("OUTCAR")
    oszicar = Path("OSZICAR")
    text = outcar.read_text(errors="replace") if outcar.is_file() else ""
    lines = text.splitlines()
    expected_atoms = int(metadata["atom_count"])
    force_blocks = parse_force_blocks(lines, expected_atoms)
    final_rows = force_blocks[-1] if force_blocks else []
    force_vectors = [row[3:6] for row in final_rows]
    force_norms = [vector_norm(vector) for vector in force_vectors]
    net_force = (
        [sum(vector[axis] for vector in force_vectors) for axis in range(3)]
        if force_vectors
        else None
    )
    energy_matches = re.findall(
        r"free\s+energy\s+TOTEN\s+=\s+([-+0-9.Ee]+)", text
    )
    error_markers = {
        "VERY BAD NEWS": text.count("VERY BAD NEWS"),
        "EXPLICIT ERROR": len(re.findall(r"(?im)^\s*ERROR(?:\s|:)", text)),
        "ZBRENT": text.count("ZBRENT"),
        "BRMIX": text.count("BRMIX"),
    }
    image_markers = {"IMAGES": text.count("IMAGES")}
    oszicar_text = oszicar.read_text(errors="replace") if oszicar.is_file() else ""
    electronic_rows = len(re.findall(r"(?m)^\s*(?:DAV|RMM):", oszicar_text))
    convergence_markers = text.count("aborting loop because EDIFF is reached")
    wavecar = Path("WAVECAR")
    wavecar_size = wavecar.stat().st_size if wavecar.is_file() else 0
    reference_wavecar_required = bool(
        metadata["is_reference"]
        and metadata.get("write_wavecar_for_displacement_seed") is True
    )

    forces_path = Path("forces.csv")
    with forces_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "atom_index_1based",
                "x_angstrom",
                "y_angstrom",
                "z_angstrom",
                "fx_ev_per_angstrom",
                "fy_ev_per_angstrom",
                "fz_ev_per_angstrom",
            ]
        )
        for atom_index, row in enumerate(final_rows, start=1):
            writer.writerow([atom_index, *row])

    summary = {
        **metadata,
        "vasp_exit_code": vasp_exit,
        "has_outcar": outcar.is_file(),
        "has_oszicar": oszicar.is_file(),
        "completed": "General timing and accounting informations for this job" in text,
        "electronic_convergence_marker_count": convergence_markers,
        "electronic_iteration_row_count": electronic_rows,
        "force_block_count": len(force_blocks),
        "final_force_atom_count": len(force_vectors),
        "final_raw_max_force_ev_per_angstrom": (
            max(force_norms) if force_norms else None
        ),
        "final_raw_rms_force_ev_per_angstrom": (
            math.sqrt(sum(value * value for value in force_norms) / len(force_norms))
            if force_norms
            else None
        ),
        "final_net_force_vector_ev_per_angstrom": net_force,
        "final_net_force_norm_ev_per_angstrom": (
            vector_norm(net_force) if net_force else None
        ),
        "final_toten_ev": float(energy_matches[-1]) if energy_matches else None,
        "forces_csv_sha256": sha256_file(forces_path),
        "reference_wavecar_size_bytes": wavecar_size if metadata["is_reference"] else None,
        "reference_wavecar_seed_present": (
            wavecar_size > 0 if metadata["is_reference"] else None
        ),
        "reference_wavecar_required": reference_wavecar_required,
        "input_poscar_hash_preserved": sha256_file(Path("POSCAR"))
        == metadata["poscar_sha256"],
        "error_markers": error_markers,
        "error_marker_count": sum(error_markers.values()),
        "image_markers": image_markers,
        "image_mode_marker": any(image_markers.values()),
    }
    summary["quality_gate_pass"] = bool(
        summary["vasp_exit_code"] == 0
        and summary["has_outcar"]
        and summary["has_oszicar"]
        and summary["completed"]
        and summary["electronic_convergence_marker_count"] >= 1
        and summary["force_block_count"] >= 1
        and summary["final_force_atom_count"] == expected_atoms
        and summary["input_poscar_hash_preserved"]
        and summary["error_marker_count"] == 0
        and not summary["image_mode_marker"]
        and (
            not reference_wavecar_required
            or summary["reference_wavecar_seed_present"] is True
        )
    )
    summary_path = Path("force_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    diagnostic_lines = [
        f"task={metadata['task_name']}",
        f"quality_gate_pass={str(summary['quality_gate_pass']).lower()}",
        f"vasp_exit_code={vasp_exit}",
        f"completed={str(summary['completed']).lower()}",
        f"electronic_convergence_markers={convergence_markers}",
        f"electronic_iteration_rows={electronic_rows}",
        f"force_blocks={len(force_blocks)}",
        f"force_rows={len(force_vectors)}",
        f"error_marker_count={summary['error_marker_count']}",
        f"image_mode_marker={str(summary['image_mode_marker']).lower()}",
        f"reference_wavecar_size_bytes={wavecar_size if metadata['is_reference'] else 'not_applicable'}",
    ]
    Path("diagnostic_excerpt.txt").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )
    for source_name, target_name in (
        ("vasp_stdout.log", "vasp_stdout_tail.log"),
        ("vasp_stderr.log", "vasp_stderr_tail.log"),
    ):
        source = Path(source_name)
        tail = source.read_text(errors="replace").splitlines()[-120:] if source.is_file() else []
        Path(target_name).write_text("\n".join(tail) + ("\n" if tail else ""))

    sync_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "force_summary.json",
        "forces.csv",
        "task_metadata.json",
        "run_metadata.txt",
        "diagnostic_excerpt.txt",
        "vasp_stdout_tail.log",
        "vasp_stderr_tail.log",
        "OSZICAR",
    ):
        source = Path(name)
        if source.is_file():
            shutil.copy2(source, sync_dir / name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "post"])
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--group", choices=["one_by_one", "two_by_two"], required=True)
    parser.add_argument("--vasp-exit", type=int)
    parser.add_argument("--sync-dir", type=Path)
    args = parser.parse_args()
    metadata = load_metadata(args.task_index, args.task_name, args.group)
    if args.mode == "preflight":
        result = preflight(metadata)
    else:
        if args.vasp_exit is None or args.sync_dir is None:
            parser.error("post mode requires --vasp-exit and --sync-dir")
        result = postprocess(metadata, args.vasp_exit, args.sync_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

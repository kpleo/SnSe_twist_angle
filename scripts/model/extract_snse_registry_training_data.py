#!/usr/bin/env python3
"""Extract lightweight SnSe registry trajectories from remote VASP XML files.

The script is intentionally dependency-free so it can run beside the remote
VASP calculations.  It exports only structures, total energies, forces, and
compact provenance metadata; the VASP XML and licensed/heavy files stay on the
cluster.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Set, Tuple


REGISTRY_RE = re.compile(r"registry_ix(?P<ix>\d+)_iy(?P<iy>\d+)_n3x3")
JOB_RE = re.compile(r"job(?P<job_id>\d+)")


class Frame(NamedTuple):
    step: int
    energy_ev: float
    lattice: Tuple[Tuple[float, float, float], ...]
    positions: Tuple[Tuple[float, float, float], ...]
    forces: Tuple[Tuple[float, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--job-id",
        action="append",
        required=True,
        help="Allowed production job ID; repeat for multiple jobs.",
    )
    parser.add_argument("--expected-registries", type=int, default=9)
    parser.add_argument("--expected-atoms", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_vector(node: ET.Element) -> Tuple[float, float, float]:
    values = tuple(float(item) for item in (node.text or "").split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid vector: {node.text!r}")
    return values


def matmul_fractional(
    fractional: Sequence[float], lattice: Sequence[Sequence[float]]
) -> Tuple[float, float, float]:
    return tuple(
        sum(fractional[row] * lattice[row][column] for row in range(3))
        for column in range(3)
    )


def determinant(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def find_species(root: ET.Element, expected_atoms: int) -> Tuple[str, ...]:
    rows = root.findall("./atominfo/array[@name='atoms']/set/rc")
    species = tuple((row.findtext("c") or "").strip() for row in rows)
    if len(species) != expected_atoms or any(not item for item in species):
        raise ValueError(
            f"expected {expected_atoms} species rows, found {len(species)}"
        )
    if sorted(species) != ["Se"] * (expected_atoms // 2) + ["Sn"] * (
        expected_atoms // 2
    ):
        raise ValueError(f"unexpected composition: {species}")
    return species


def parse_frames(root: ET.Element, expected_atoms: int) -> List[Frame]:
    frames = []
    for step, calculation in enumerate(root.findall("./calculation")):
        energy_node = calculation.find("./energy/i[@name='e_fr_energy']")
        structure = calculation.find("./structure")
        basis = (
            structure.findall("./crystal/varray[@name='basis']/v")
            if structure is not None
            else []
        )
        fractional_nodes = (
            structure.findall("./varray[@name='positions']/v")
            if structure is not None
            else []
        )
        force_nodes = calculation.findall("./varray[@name='forces']/v")
        if energy_node is None or energy_node.text is None:
            raise ValueError(f"missing free energy at ionic step {step}")
        energy = float(energy_node.text)
        lattice = tuple(parse_vector(node) for node in basis)
        fractional = tuple(parse_vector(node) for node in fractional_nodes)
        forces = tuple(parse_vector(node) for node in force_nodes)
        if len(lattice) != 3 or determinant(lattice) <= 0.0:
            raise ValueError(f"invalid lattice at ionic step {step}")
        if len(fractional) != expected_atoms or len(forces) != expected_atoms:
            raise ValueError(
                f"step {step}: positions={len(fractional)}, forces={len(forces)}"
            )
        if not math.isfinite(energy):
            raise ValueError(f"non-finite energy at ionic step {step}")
        positions = tuple(matmul_fractional(row, lattice) for row in fractional)
        frames.append(
            Frame(
                step=step,
                energy_ev=energy,
                lattice=lattice,
                positions=positions,
                forces=forces,
            )
        )
    if not frames:
        raise ValueError("no complete ionic calculations found")
    return frames


def flatten(values: Iterable[Sequence[float]]) -> str:
    return " ".join(f"{number:.16g}" for row in values for number in row)


def frame_text(
    frame: Frame,
    species: Sequence[str],
    phase: str,
    registry: str,
    job_id: str,
    source_sha256: str,
) -> str:
    config_type = f"snse_{phase}_{registry}"
    header = (
        f'Lattice="{flatten(frame.lattice)}" '
        "Properties=species:S:1:pos:R:3:REF_forces:R:3 "
        f"REF_energy={frame.energy_ev:.16g} "
        'pbc="T T T" '
        f"config_type={config_type} source_phase={phase} "
        f"source_registry={registry} source_job_id={job_id} "
        f"source_step={frame.step} source_vasprun_sha256={source_sha256}"
    )
    rows = [str(len(species)), header]
    for symbol, position, force in zip(species, frame.positions, frame.forces):
        values = " ".join(f"{value:.16g}" for value in (*position, *force))
        rows.append(f"{symbol} {values}")
    return "\n".join(rows) + "\n"


def discover_sources(input_root: Path, allowed_jobs: Set[str]) -> List[Path]:
    selected = []
    for path in sorted(input_root.glob("job*/registry*/vasprun.xml")):
        job_match = JOB_RE.fullmatch(path.parents[1].name)
        if job_match and job_match.group("job_id") in allowed_jobs:
            selected.append(path)
    return selected


def main() -> int:
    args = parse_args()
    allowed_jobs = set(args.job_id)
    sources = discover_sources(args.input_root.resolve(), allowed_jobs)
    if not sources:
        raise SystemExit("no matching vasprun.xml files found")

    by_registry = {}  # type: Dict[str, Path]
    for source in sources:
        match = REGISTRY_RE.search(source.parent.name)
        if match is None:
            raise SystemExit(f"cannot identify registry from {source}")
        registry = f"ix{match.group('ix')}iy{match.group('iy')}"
        if registry in by_registry:
            raise SystemExit(
                f"duplicate selected source for {registry}: {by_registry[registry]} and {source}"
            )
        by_registry[registry] = source
    if len(by_registry) != args.expected_registries:
        raise SystemExit(
            f"expected {args.expected_registries} registries, found {len(by_registry)}"
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    combined_parts = []  # type: List[str]
    final_parts = []  # type: List[str]
    rows = []  # type: List[Dict[str, object]]

    for registry, source in sorted(by_registry.items()):
        job_match = JOB_RE.fullmatch(source.parents[1].name)
        assert job_match is not None
        job_id = job_match.group("job_id")
        source_hash = sha256_file(source)
        root = ET.parse(source).getroot()
        species = find_species(root, args.expected_atoms)
        frames = parse_frames(root, args.expected_atoms)
        texts = [
            frame_text(
                frame,
                species,
                args.phase,
                registry,
                job_id,
                source_hash,
            )
            for frame in frames
        ]
        trajectory_path = output_root / f"{args.phase}_{registry}.extxyz"
        trajectory_path.write_text("".join(texts), encoding="ascii")
        combined_parts.extend(texts)
        final_parts.append(texts[-1])
        force_norms = [
            math.sqrt(sum(component * component for component in force))
            for frame in frames
            for force in frame.forces
        ]
        rows.append(
            {
                "phase": args.phase,
                "registry": registry,
                "job_id": job_id,
                "atom_count": len(species),
                "frame_count": len(frames),
                "initial_energy_ev": frames[0].energy_ev,
                "final_energy_ev": frames[-1].energy_ev,
                "energy_drop_ev": frames[-1].energy_ev - frames[0].energy_ev,
                "force_norm_max_ev_per_a": max(force_norms),
                "trajectory_file": trajectory_path.name,
                "trajectory_sha256": sha256_file(trajectory_path),
                "source_vasprun_sha256": source_hash,
            }
        )

    combined_path = output_root / f"{args.phase}_all_trajectories.extxyz"
    final_path = output_root / f"{args.phase}_final_structures.extxyz"
    combined_path.write_text("".join(combined_parts), encoding="ascii")
    final_path.write_text("".join(final_parts), encoding="ascii")

    csv_path = output_root / f"{args.phase}_trajectory_manifest.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "input_root": str(args.input_root.resolve()),
        "allowed_job_ids": sorted(allowed_jobs),
        "expected_atoms_per_frame": args.expected_atoms,
        "registry_count": len(rows),
        "total_frame_count": sum(int(row["frame_count"]) for row in rows),
        "combined_file": combined_path.name,
        "combined_sha256": sha256_file(combined_path),
        "final_file": final_path.name,
        "final_sha256": sha256_file(final_path),
        "registries": rows,
    }
    manifest_path = output_root / f"{args.phase}_trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

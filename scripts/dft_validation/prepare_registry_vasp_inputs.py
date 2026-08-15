#!/usr/bin/env python3
"""Stage the nine strict SnSe registry VASP inputs without licensed POTCAR data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "inputs" / "registry"
DEFAULT_CONFIG = REPO / "configs" / "dft_registry_relaxation"
DEFAULT_OUTPUT = REPO / "work" / "dft_registry_strict_inputs"
CASES = tuple(
    f"registry_ix{ix:02d}_iy{iy:02d}_n3x3"
    for ix in range(3)
    for iy in range(3)
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    config_dir = args.config_dir.resolve()
    output_root = args.output_root.resolve()
    incar = config_dir / "INCAR"
    kpoints = config_dir / "KPOINTS"
    for required in (incar, kpoints):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, case in enumerate(CASES):
        source_poscar = source_root / case / "POSCAR"
        if not source_poscar.is_file():
            raise FileNotFoundError(source_poscar)
        case_dir = output_root / case
        case_dir.mkdir()
        for source in (source_poscar, incar, kpoints):
            shutil.copy2(source, case_dir / source.name)
        rows.append(
            {
                "task_index": index,
                "case": case,
                "poscar_sha256": sha256_file(case_dir / "POSCAR"),
                "incar_sha256": sha256_file(case_dir / "INCAR"),
                "kpoints_sha256": sha256_file(case_dir / "KPOINTS"),
                "potcar_included": False,
            }
        )

    with (output_root / "manifest.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_count": len(rows),
                "functional": "PBE-D3(BJ)",
                "potcar_labels": ["Sn_d", "Se"],
                "potcar_included": False,
                "cases": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({"output_root": str(output_root), "task_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reject files that do not belong in the lightweight code release."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2 * 1024 * 1024
FORBIDDEN_NAMES = {
    "POTCAR",
    "OUTCAR",
    "WAVECAR",
    "CHGCAR",
    "CHG",
    "vasprun.xml",
    "XDATCAR",
    "OSZICAR",
    "EIGENVAL",
    "DOSCAR",
    "PROCAR",
}
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".model",
    ".npy",
    ".npz",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".svg",
    ".tex",
    ".tif",
    ".tiff",
    ".traj",
}
FORBIDDEN_TOP_LEVEL = {
    "manuscript",
    "outputs",
    "paper_prb",
    "raw",
    "results",
    "source_data",
    "work",
}
FORBIDDEN_TEXT = ("/Users/", "/data/home/", "/online1/")


def main() -> int:
    problems: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in FORBIDDEN_TOP_LEVEL:
            problems.append(f"forbidden top-level path: {relative}")
        if path.name in FORBIDDEN_NAMES:
            problems.append(f"forbidden calculation file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden binary/manuscript suffix: {relative}")
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"file exceeds 2 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if path.resolve() != Path(__file__).resolve():
            for marker in FORBIDDEN_TEXT:
                if marker in text:
                    problems.append(
                        f"personal/cluster absolute path {marker!r}: {relative}"
                    )

    if problems:
        print("Release audit failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Release audit passed: code, compact inputs, and documentation only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

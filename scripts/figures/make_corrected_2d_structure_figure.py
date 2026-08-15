#!/usr/bin/env python3
"""Build the seven-angle bilayer structure plate for the PRB paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "work"
DATA_ROOTS = (DATA_ROOT,)
FIGURE_DIR = ROOT / "outputs" / "figures"
SOURCE_DIR = ROOT / "outputs" / "source_data"

SN = "#687AA8"
SE = "#D4A13A"
BOND_LOWER = "#A9ADB4"
BOND_UPPER = "#595E66"
SIGNAL = "#326C8E"
ACCENT = "#B64E3B"
NEUTRAL = "#333333"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.025, y: float = 1.015) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )


def load_cases() -> list[dict[str, Any]]:
    rows = []
    for data_root in DATA_ROOTS:
        payload = json.loads(
            (data_root / "manifests" / "initial_structures.json").read_text(
                encoding="utf-8"
            )
        )
        for row in payload["structures"]:
            label = str(row["label"])
            path = (
                data_root
                / "relaxations"
                / label
                / "confirmed_fmax0p001"
                / "relaxed.extxyz"
            )
            audit_path = path.parent / "relaxation_audit.json"
            audit = json.loads(audit_path.read_text(encoding="ascii"))
            atoms = read(path)
            z = np.asarray(atoms.positions[:, 2], dtype=float)
            midpoint = 0.5 * (float(z.min()) + float(z.max()))
            lower = z < midpoint
            upper = ~lower
            rows.append(
                {
                    "label": label,
                    "angle_deg": float(row["angle_deg"]),
                    "atom_count": len(atoms),
                    "atoms": atoms,
                    "structure": path,
                    "layer_center_separation_angstrom": float(
                        np.mean(z[upper]) - np.mean(z[lower])
                    ),
                    "periodic_vacuum_gap_angstrom": float(
                        atoms.cell.lengths()[2] - (z.max() - z.min())
                    ),
                    "direct_interlayer_z_gap_angstrom": float(
                        audit["diagnostics"]["direct_interlayer_z_gap_angstrom"]
                    ),
                    "minimum_interlayer_distance_angstrom": float(
                        audit["diagnostics"]["minimum_interlayer_distance_angstrom"]
                    ),
                }
            )
    return sorted(rows, key=lambda item: item["angle_deg"], reverse=True)


def layer_masks(atoms) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    midpoint = 0.5 * (float(z.min()) + float(z.max()))
    lower = z < midpoint
    return lower, ~lower


def atom_colors(atoms) -> np.ndarray:
    return np.asarray([SN if symbol == "Sn" else SE for symbol in atoms.symbols])


def intralayer_segments(atoms, projection: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray]:
    lower, _ = layer_masks(atoms)
    positions = np.asarray(atoms.positions, dtype=float)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    i_list, j_list, shifts = neighbor_list("ijS", atoms, 3.05)
    segments: list[np.ndarray] = []
    layers: list[int] = []
    for i, j, shift in zip(i_list, j_list, shifts):
        if int(i) >= int(j) or lower[i] != lower[j]:
            continue
        shifted_fraction = scaled[j] + shift
        if np.any(shifted_fraction[:2] < -1e-9) or np.any(
            shifted_fraction[:2] > 1.0 + 1e-9
        ):
            continue
        p0 = positions[i]
        p1 = positions[j] + shift @ cell
        if np.linalg.norm(p1 - p0) > 3.05:
            continue
        segments.append(
            np.asarray(
                [
                    [p0[projection[0]], p0[projection[1]]],
                    [p1[projection[0]], p1[projection[1]]],
                ]
            )
        )
        layers.append(0 if lower[i] else 1)
    return segments, np.asarray(layers, dtype=int)


def draw_top_view(ax: plt.Axes, atoms, angle_deg: float, atom_count: int) -> None:
    atoms = atoms.copy()
    atoms.wrap()
    positions = np.asarray(atoms.positions, dtype=float)
    lower, upper = layer_masks(atoms)
    colors = atom_colors(atoms)
    segments, layers = intralayer_segments(atoms, (0, 1))
    for layer, color, alpha, width, zorder in (
        (0, BOND_LOWER, 0.22, 0.20, 1),
        (1, BOND_UPPER, 0.38, 0.24, 3),
    ):
        subset = [segment for segment, value in zip(segments, layers) if value == layer]
        if subset:
            ax.add_collection(
                LineCollection(subset, colors=color, alpha=alpha, linewidths=width, zorder=zorder)
            )

    marker_size = float(np.clip(8.8 * (340.0 / atom_count) ** 0.52, 2.8, 8.8))
    for mask, alpha, edge, zorder in (
        (lower, 0.48, 0.0, 2),
        (upper, 0.94, 0.12, 4),
    ):
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=marker_size,
            c=colors[mask],
            alpha=alpha,
            linewidths=edge,
            edgecolors="white" if edge else "none",
            zorder=zorder,
        )

    a = atoms.cell.array[0, :2]
    b = atoms.cell.array[1, :2]
    boundary = np.asarray([[0.0, 0.0], a, a + b, b, [0.0, 0.0]])
    ax.plot(boundary[:, 0], boundary[:, 1], color=NEUTRAL, lw=0.62, zorder=6)
    margin = 0.025 * max(np.linalg.norm(a), np.linalg.norm(b))
    ax.set_xlim(float(boundary[:, 0].min() - margin), float(boundary[:, 0].max() + margin))
    ax.set_ylim(float(boundary[:, 1].min() - margin), float(boundary[:, 1].max() + margin))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        rf"${angle_deg:.2f}^{{\circ}}$  |  {atom_count} atoms",
        fontsize=6.7,
        fontweight="medium",
        pad=2.0,
    )
    scale = 10.0 if atom_count < 1000 else 20.0
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.070 * (ymax - ymin)
    ax.plot(
        [x0, x0 + scale],
        [y0, y0],
        color=NEUTRAL,
        lw=1.05,
        solid_capstyle="butt",
        zorder=10,
    )
    ax.text(
        x0 + 0.5 * scale,
        y0 + 0.022 * (ymax - ymin),
        f"{scale:.0f} Å",
        ha="center",
        va="bottom",
        fontsize=5.4,
    )


def draw_side_view(ax: plt.Axes, row: dict[str, Any]) -> None:
    atoms = row["atoms"].copy()
    atoms.wrap()
    lower, upper = layer_masks(atoms)

    distances = atoms.get_all_distances(mic=True)
    interlayer = np.where(lower[:, None] & upper[None, :], distances, np.inf)
    pair_i, pair_j = np.unravel_index(np.argmin(interlayer), interlayer.shape)

    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    scaled[:, :2] = (scaled[:, :2] + (0.5 - scaled[pair_i, :2])) % 1.0
    atoms.set_scaled_positions(scaled)
    atoms.wrap()
    positions = np.asarray(atoms.positions, dtype=float)
    colors = atom_colors(atoms)
    center_x = 0.5 * atoms.cell.lengths()[0]
    center_y = 0.5 * atoms.cell.lengths()[1]
    half_width = 7.7
    half_strip = 2.35
    use = (np.abs(positions[:, 0] - center_x) < half_width) & (
        np.abs(positions[:, 1] - center_y) < half_strip
    )

    i_list, j_list, shifts = neighbor_list("ijS", atoms, 3.05)
    side_segments = []
    side_layers = []
    cell = np.asarray(atoms.cell.array, dtype=float)
    for i, j, shift in zip(i_list, j_list, shifts):
        if int(i) >= int(j) or lower[i] != lower[j] or not (use[i] and use[j]):
            continue
        p0 = positions[i]
        p1 = positions[j] + shift @ cell
        if np.linalg.norm(p1 - p0) > 3.05:
            continue
        if abs(p1[0] - center_x) >= half_width or abs(p1[1] - center_y) >= half_strip:
            continue
        side_segments.append(np.asarray([[p0[0], p0[2]], [p1[0], p1[2]]]))
        side_layers.append(0 if lower[i] else 1)
    for layer, color, alpha, width, zorder in (
        (0, BOND_LOWER, 0.55, 0.65, 1),
        (1, BOND_UPPER, 0.72, 0.70, 2),
    ):
        subset = [
            segment
            for segment, value in zip(side_segments, side_layers)
            if value == layer
        ]
        if subset:
            ax.add_collection(
                LineCollection(
                    subset,
                    colors=color,
                    alpha=alpha,
                    linewidths=width,
                    zorder=zorder,
                )
            )

    for mask, alpha, zorder in ((use & lower, 0.76, 3), (use & upper, 0.98, 4)):
        ax.scatter(
            positions[mask, 0],
            positions[mask, 2],
            s=21,
            c=colors[mask],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.25,
            zorder=zorder,
        )

    z_lower = float(np.mean(positions[lower, 2]))
    z_upper = float(np.mean(positions[upper, 2]))
    z_min = float(positions[:, 2].min())
    z_max = float(positions[:, 2].max())
    ax.axhline(z_lower, color=SIGNAL, ls=(0, (3, 2)), lw=0.60, alpha=0.80)
    ax.axhline(z_upper, color=SIGNAL, ls=(0, (3, 2)), lw=0.60, alpha=0.80)

    pair_x = positions[[pair_i, pair_j], 0]
    pair_z = positions[[pair_i, pair_j], 2]
    ax.plot(pair_x, pair_z, color=ACCENT, ls=(0, (2, 1.5)), lw=0.95, zorder=5)
    ax.text(
        float(np.mean(pair_x)) - 0.15,
        float(np.mean(pair_z)),
        rf"$d_{{\rm min}}={row['minimum_interlayer_distance_angstrom']:.2f}$ Å",
        ha="right",
        va="center",
        color=ACCENT,
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.6},
        zorder=6,
    )

    arrow_x = center_x + 6.2
    cap = 0.18
    ax.plot([arrow_x, arrow_x], [z_lower, z_upper], color=SIGNAL, lw=0.85)
    ax.plot([arrow_x - cap, arrow_x + cap], [z_lower, z_lower], color=ACCENT, lw=0.8)
    ax.plot([arrow_x - cap, arrow_x + cap], [z_upper, z_upper], color=ACCENT, lw=0.8)
    ax.text(
        arrow_x - 0.30,
        0.5 * (z_lower + z_upper),
        rf"$d_{{\rm cent}}={row['layer_center_separation_angstrom']:.2f}$ Å",
        ha="right",
        va="center",
        color=SIGNAL,
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.6},
    )
    ax.set_xlim(center_x - 8.0, center_x + 8.0)
    ax.set_ylim(z_min - 0.85, z_max + 0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        rf"Side view  |  ${row['angle_deg']:.2f}^\circ$",
        fontsize=6.7,
        fontweight="medium",
        pad=2.0,
    )
    origin_x = center_x - 7.1
    origin_z = z_min - 0.47
    ax.annotate(
        "",
        xy=(origin_x + 1.65, origin_z),
        xytext=(origin_x, origin_z),
        arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": NEUTRAL},
    )
    ax.annotate(
        "",
        xy=(origin_x, origin_z + 0.95),
        xytext=(origin_x, origin_z),
        arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": NEUTRAL},
    )
    ax.text(origin_x + 1.78, origin_z, "$x$", ha="left", va="center", fontsize=5.8)
    ax.text(
        origin_x - 0.18,
        origin_z + 0.98,
        "$z$",
        ha="right",
        va="center",
        fontsize=5.8,
    )
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_source_data(rows: list[dict[str, Any]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_DIR / "Fig1_structure_summary.csv"
    with summary_path.open("w", newline="", encoding="ascii") as handle:
        fields = [
            "label",
            "angle_deg",
            "atom_count",
            "layer_center_separation_angstrom",
            "direct_interlayer_z_gap_angstrom",
            "minimum_interlayer_distance_angstrom",
            "periodic_vacuum_gap_angstrom",
            "structure",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fields}
            output["structure"] = str(Path(row["structure"]).relative_to(ROOT))
            writer.writerow(output)

    coordinate_path = SOURCE_DIR / "Fig1_structure_coordinates.csv"
    with coordinate_path.open("w", newline="", encoding="ascii") as handle:
        fields = ["label", "atom_index", "element", "x_angstrom", "y_angstrom", "z_angstrom"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            atoms = row["atoms"]
            for index, (symbol, position) in enumerate(zip(atoms.symbols, atoms.positions)):
                writer.writerow(
                    {
                        "label": row["label"],
                        "atom_index": index,
                        "element": symbol,
                        "x_angstrom": position[0],
                        "y_angstrom": position[1],
                        "z_angstrom": position[2],
                    }
                )


def main() -> None:
    global DATA_ROOT, DATA_ROOTS, FIGURE_DIR, SOURCE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--additional-root", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    DATA_ROOTS = (
        (DATA_ROOT, args.additional_root.resolve())
        if args.additional_root is not None
        else (DATA_ROOT,)
    )
    FIGURE_DIR = args.figure_dir.resolve()
    SOURCE_DIR = args.source_data_dir.resolve()
    rows = load_cases()
    representative = min(rows, key=lambda row: abs(row["angle_deg"] - 7.61))

    fig = plt.figure(figsize=(7.10, 3.72))
    grid = fig.add_gridspec(
        2,
        4,
        wspace=0.08,
        hspace=0.22,
        left=0.025,
        right=0.995,
        top=0.94,
        bottom=0.085,
    )
    axes = [fig.add_subplot(grid[index // 4, index % 4]) for index in range(8)]
    for ax, row in zip(axes[:7], rows):
        draw_top_view(ax, row["atoms"], row["angle_deg"], row["atom_count"])
    draw_side_view(axes[7], representative)

    angles = np.asarray([row["angle_deg"] for row in rows], dtype=float)
    atom_counts = np.asarray([row["atom_count"] for row in rows], dtype=float)
    separations = np.asarray(
        [row["layer_center_separation_angstrom"] for row in rows], dtype=float
    )
    for label, ax in zip("abcdefgh", axes):
        panel_label(ax, label)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SN, markersize=5, label="Sn"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SE, markersize=5, label="Se"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.50, 0.003),
        handletextpad=0.35,
        columnspacing=0.9,
        fontsize=6.2,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "Fig1_twisted_structures"
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        output_path = stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, **kwargs)
        if suffix == "svg":
            svg = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)
    write_source_data(rows)
    print(
        json.dumps(
            {
                "figure": str(stem),
                "angles_deg": angles.tolist(),
                "atom_counts": atom_counts.astype(int).tolist(),
                "layer_center_separations_angstrom": separations.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

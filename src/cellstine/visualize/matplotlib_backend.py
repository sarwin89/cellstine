"""Matplotlib-first static visualizations for CELLSTINE structures and searches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io.models import StructureRecord


@dataclass(frozen=True)
class MatplotlibRun:
    """Metadata returned by static visualization writers."""

    output_path: Path
    item_count: int
    visualization_type: str


_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


_COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Matplotlib is required for the default static visualizations. "
            'Install it with `pip install -e ".[viz]"`, or pass `--plotly` for the optional HTML viewer.'
        ) from exc


def _expanded_species(record: StructureRecord) -> list[str]:
    expanded: list[str] = []
    for symbol, count in zip(record.species, record.counts):
        expanded.extend([str(symbol)] * int(count))
    if len(expanded) < record.natoms:
        expanded.extend(["X"] * (record.natoms - len(expanded)))
    return expanded[: record.natoms]


def _element_symbol(label: str) -> str:
    letters = "".join(character for character in str(label) if character.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[:2].capitalize() in _COVALENT_RADII_ANGSTROM:
        return letters[:2].capitalize()
    return letters[:1].upper()


def _atomic_radius(label: str) -> float:
    return float(_COVALENT_RADII_ANGSTROM.get(_element_symbol(label), 1.0))


def _marker_size(label: str, *, projection: str) -> float:
    radius = _atomic_radius(label)
    scale = 26.0 if projection == "2d" else 18.0
    return float(np.clip((radius * scale) ** 2, 170.0 if projection == "2d" else 90.0, 1800.0 if projection == "2d" else 900.0))


def _cell_corners(lattice: np.ndarray) -> dict[str, np.ndarray]:
    origin = np.zeros(3, dtype=float)
    a_vec = np.asarray(lattice[0], dtype=float)
    b_vec = np.asarray(lattice[1], dtype=float)
    c_vec = np.asarray(lattice[2], dtype=float)
    return {
        "000": origin,
        "100": a_vec,
        "010": b_vec,
        "001": c_vec,
        "110": a_vec + b_vec,
        "101": a_vec + c_vec,
        "011": b_vec + c_vec,
        "111": a_vec + b_vec + c_vec,
    }


def _cell_edges() -> list[tuple[str, str]]:
    return [
        ("000", "100"),
        ("000", "010"),
        ("000", "001"),
        ("100", "110"),
        ("100", "101"),
        ("010", "110"),
        ("010", "011"),
        ("001", "101"),
        ("001", "011"),
        ("110", "111"),
        ("101", "111"),
        ("011", "111"),
    ]


def _set_equal_2d(ax, x_values: np.ndarray, y_values: np.ndarray) -> None:
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    if finite_x.size == 0 or finite_y.size == 0:
        return
    x_min, x_max = float(finite_x.min()), float(finite_x.max())
    y_min, y_max = float(finite_y.min()), float(finite_y.max())
    span = max(x_max - x_min, y_max - y_min, 1.0)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    pad = 0.08 * span
    ax.set_xlim(x_mid - 0.5 * span - pad, x_mid + 0.5 * span + pad)
    ax.set_ylim(y_mid - 0.5 * span - pad, y_mid + 0.5 * span + pad)
    ax.set_aspect("equal", adjustable="box")


def _set_equal_3d(ax, positions: np.ndarray, corners: dict[str, np.ndarray]) -> None:
    all_points = np.vstack([positions, *corners.values()]) if positions.size else np.vstack(list(corners.values()))
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    radius = float(spans.max()) * 0.55
    center = 0.5 * (mins + maxs)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_projected_cell(ax, lattice: np.ndarray, axes: tuple[int, int], *, label: str | None = None) -> None:
    corners = _cell_corners(lattice)
    first = True
    for start, end in _cell_edges():
        p0 = corners[start]
        p1 = corners[end]
        ax.plot(
            [p0[axes[0]], p1[axes[0]]],
            [p0[axes[1]], p1[axes[1]]],
            color="#222222",
            linewidth=1.1,
            alpha=0.8,
            label=label if first else None,
        )
        first = False


def _draw_3d_cell(ax, lattice: np.ndarray) -> None:
    corners = _cell_corners(lattice)
    for start, end in _cell_edges():
        p0 = corners[start]
        p1 = corners[end]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#222222",
            linewidth=1.0,
            alpha=0.75,
        )


def plot_structure_multiview(
    record: StructureRecord,
    *,
    output_path: str | Path,
    title: str | None = None,
    show: bool = False,
) -> MatplotlibRun:
    """Write a labelled four-panel structure view: xy, xz, yz, and 3D."""

    plt = _pyplot()
    positions = np.asarray(record.positions_cartesian, dtype=float)
    species = np.asarray(_expanded_species(record), dtype=object)
    unique_species = list(dict.fromkeys(species.tolist()))
    colors = {symbol: _PALETTE[index % len(_PALETTE)] for index, symbol in enumerate(unique_species)}

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    fig.suptitle(title or record.comment or "CELLSTINE structure view", fontsize=15, fontweight="bold")
    axes_2d = [
        (fig.add_subplot(2, 2, 1), (0, 1), "Top view: x-y", "x (Angstrom)", "y (Angstrom)"),
        (fig.add_subplot(2, 2, 2), (0, 2), "Side view: x-z", "x (Angstrom)", "z (Angstrom)"),
        (fig.add_subplot(2, 2, 3), (1, 2), "Side view: y-z", "y (Angstrom)", "z (Angstrom)"),
    ]

    for ax, projection, panel_title, x_label, y_label in axes_2d:
        for symbol in unique_species:
            mask = species == symbol
            ax.scatter(
                positions[mask, projection[0]],
                positions[mask, projection[1]],
                s=_marker_size(str(symbol), projection="2d"),
                alpha=0.9,
                label=f"{symbol} (r={_atomic_radius(str(symbol)):.2f} A)",
                color=colors[str(symbol)],
                edgecolor="white",
                linewidth=0.55,
            )
        _draw_projected_cell(ax, record.lattice, projection, label="unit cell")
        _set_equal_2d(ax, positions[:, projection[0]], positions[:, projection[1]])
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linewidth=0.4, alpha=0.35)

    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    for symbol in unique_species:
        mask = species == symbol
        ax_3d.scatter(
            positions[mask, 0],
            positions[mask, 1],
            positions[mask, 2],
            s=_marker_size(str(symbol), projection="3d"),
            alpha=0.9,
            label=f"{symbol} (r={_atomic_radius(str(symbol)):.2f} A)",
            color=colors[str(symbol)],
            edgecolor="white",
            linewidth=0.45,
        )
    _draw_3d_cell(ax_3d, record.lattice)
    _set_equal_3d(ax_3d, positions, _cell_corners(record.lattice))
    ax_3d.set_title("3D overview")
    ax_3d.set_xlabel("x (Angstrom)")
    ax_3d.set_ylabel("y (Angstrom)")
    ax_3d.set_zlabel("z (Angstrom)")
    ax_3d.view_init(elev=22, azim=-55)

    handles, labels = axes_2d[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=True)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:  # pragma: no cover - interactive user convenience
        plt.show()
    plt.close(fig)
    return MatplotlibRun(output_path=output, item_count=record.natoms, visualization_type="structure_multiview")


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _read_dat_summary(path: Path) -> tuple[str, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped.startswith("|") or stripped.lower().startswith("| idx"):
                continue
            parts = [part.strip() for part in stripped.split("|") if part.strip()]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "index": _safe_int(parts[0]),
                    "angle": _safe_float(parts[1]),
                    "strain": _safe_float(parts[2]),
                    "atoms": _safe_int(parts[5]),
                }
            )
    return "bilayer", rows


def _read_json_summary(path: Path) -> tuple[str, list[dict[str, float | int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidates", []))
    rows: list[dict[str, float | int]] = []
    if candidates and "upper_layers" in candidates[0]:
        for fallback_index, candidate in enumerate(candidates, start=1):
            layers = list(candidate.get("upper_layers", []))
            angles = [_safe_float(layer.get("angle_deg")) for layer in layers]
            rows.append(
                {
                    "index": _safe_int(candidate.get("index"), fallback_index),
                    "angle": float(np.nanmean(angles)) if angles else np.nan,
                    "strain": _safe_float(candidate.get("strain_max", candidate.get("strain_mean"))),
                    "atoms": _safe_int(candidate.get("total_atoms")),
                }
            )
        return "nlayer", rows

    if candidates and "angle_middle_deg" in candidates[0]:
        for fallback_index, candidate in enumerate(candidates, start=1):
            angles = [_safe_float(candidate.get("angle_middle_deg")), _safe_float(candidate.get("angle_top_deg"))]
            rows.append(
                {
                    "index": _safe_int(candidate.get("index"), fallback_index),
                    "angle": float(np.nanmean(angles)),
                    "strain": _safe_float(candidate.get("strain_max", candidate.get("strain_mean"))),
                    "atoms": _safe_int(candidate.get("total_atoms")),
                }
            )
        return "trilayer", rows

    for fallback_index, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "index": _safe_int(candidate.get("index"), fallback_index),
                "angle": _safe_float(candidate.get("angle_deg")),
                "strain": _safe_float(candidate.get("strain_avg")),
                "atoms": _safe_int(candidate.get("total_atoms")),
            }
        )
    return "bilayer", rows


def _read_moire_summary(results_file: str | Path, indices: Sequence[int] | None = None) -> tuple[str, list[dict[str, float | int]]]:
    path = Path(results_file).resolve()
    if path.suffix.lower() == ".json":
        results_type, rows = _read_json_summary(path)
    else:
        results_type, rows = _read_dat_summary(path)
    if indices is not None:
        wanted = {int(index) for index in indices}
        rows = [row for row in rows if int(row["index"]) in wanted]
    if not rows:
        raise ValueError("no candidates were selected for visualization")
    return results_type, rows


def plot_moire_summary(
    results_file: str | Path,
    *,
    indices: Sequence[int] | None = None,
    output_path: str | Path,
    title: str | None = None,
    show: bool = False,
) -> MatplotlibRun:
    """Write a static summary plot for a bilayer or N-layer moire search."""

    plt = _pyplot()
    results_type, rows = _read_moire_summary(results_file, indices)
    indexes = np.asarray([row["index"] for row in rows], dtype=float)
    angles = np.asarray([row["angle"] for row in rows], dtype=float)
    strain_percent = 100.0 * np.asarray([row["strain"] for row in rows], dtype=float)
    atoms = np.asarray([row["atoms"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    resolved_title = title or f"CELLSTINE {results_type} commensuration summary"
    fig.suptitle(resolved_title, fontsize=15, fontweight="bold")

    scatter = axes[0, 0].scatter(angles, strain_percent, c=atoms, cmap="viridis", s=48, edgecolor="white", linewidth=0.4)
    axes[0, 0].set_title("Candidate strain by twist angle")
    axes[0, 0].set_xlabel("twist angle (degrees)")
    axes[0, 0].set_ylabel("strain or mismatch (%)")
    axes[0, 0].grid(True, linewidth=0.4, alpha=0.35)
    colorbar = fig.colorbar(scatter, ax=axes[0, 0])
    colorbar.set_label("total atoms")

    axes[0, 1].scatter(angles, atoms, color="#2a9d8f", s=44, edgecolor="white", linewidth=0.4, label="candidate")
    axes[0, 1].set_title("Cell size by twist angle")
    axes[0, 1].set_xlabel("twist angle (degrees)")
    axes[0, 1].set_ylabel("total atoms")
    axes[0, 1].grid(True, linewidth=0.4, alpha=0.35)
    axes[0, 1].legend()

    order = np.argsort(indexes)
    axes[1, 0].plot(indexes[order], strain_percent[order], marker="o", color="#e76f51", label="strain")
    axes[1, 0].set_title("Search ranking")
    axes[1, 0].set_xlabel("candidate index")
    axes[1, 0].set_ylabel("strain or mismatch (%)")
    axes[1, 0].grid(True, linewidth=0.4, alpha=0.35)
    axes[1, 0].legend()

    bins = min(max(len(rows), 1), 18)
    axes[1, 1].hist(angles[np.isfinite(angles)], bins=bins, color="#457b9d", edgecolor="white", alpha=0.9, label="angles")
    axes[1, 1].set_title("Twist-angle distribution")
    axes[1, 1].set_xlabel("twist angle (degrees)")
    axes[1, 1].set_ylabel("candidate count")
    axes[1, 1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[1, 1].legend()

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:  # pragma: no cover - interactive user convenience
        plt.show()
    plt.close(fig)
    return MatplotlibRun(output_path=output, item_count=len(rows), visualization_type=f"{results_type}_summary")

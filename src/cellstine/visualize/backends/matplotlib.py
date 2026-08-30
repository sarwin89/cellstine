"""Matplotlib-first static visualizations for CELLSTINE structures and searches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...core.directions import ViewDirection
from ...core.species import expand_species
from ...io.models import StructureRecord
from ...moire.search.results import read_results


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


def species_depth_order(
    species: np.ndarray,
    positions: np.ndarray,
    depth_axis: int,
    unique_species: Sequence[str],
) -> list[str]:
    """Order the species so the ones further from the observer are drawn first.

    A panel drops one coordinate, and two atoms that differ only along it land on
    the same point of the picture, so one hides the other
    (``Cellstine.planarProj_eq_iff_smul`` in
    ``aristotle-lean-reference/RequestProject/ViewProjection.lean``).  Which one is visible is decided by
    the drawing order.  The markers of one species are identical, so the order
    *inside* a species cannot change the picture, and it is enough to order the
    species themselves; ordering them by how close they come to the observer
    resolves every overlap correctly whenever the species do not interleave in
    depth -- an adsorbate on a substrate, or the two sides of an interface
    (``Cellstine.drawn_later_of_separated``).  Panels are read from the positive
    end of the axis they drop, so a larger dropped coordinate is nearer.

    Species that are not present keep their place, and ties are broken by the
    order the species appear in the file, so the picture is deterministic.
    """

    def _depth(symbol: str) -> float:
        mask = species == symbol
        if not np.any(mask):
            return float("-inf")
        return float(np.max(positions[mask, depth_axis]))

    order = {symbol: index for index, symbol in enumerate(unique_species)}
    return sorted(unique_species, key=lambda symbol: (_depth(symbol), order[symbol]))


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
    direction: ViewDirection | None = None,
) -> MatplotlibRun:
    """Write a labelled four-panel structure view: two side views, a plan view, and 3D.

    Without a direction the panels are the Cartesian ``x-y``, ``x-z`` and
    ``y-z`` projections of the file.  With one, the structure is turned so that
    the direction of observation points out of the plan view: the coordinates
    are taken in a right-handed frame whose third axis is that direction, which
    is a rotation and therefore moves no atom relative to another.
    """

    plt = _pyplot()
    positions = np.asarray(record.positions_cartesian, dtype=float)
    lattice = np.asarray(record.lattice, dtype=float)
    if direction is None:
        panel_names = (
            ("Top view: x-y", "x (Angstrom)", "y (Angstrom)"),
            ("Side view: x-z", "x (Angstrom)", "z (Angstrom)"),
            ("Side view: y-z", "y (Angstrom)", "z (Angstrom)"),
        )
        axis_labels = ("x (Angstrom)", "y (Angstrom)", "z (Angstrom)")
    else:
        frame = direction.frame(lattice)
        positions = positions @ frame.T
        lattice = lattice @ frame.T
        panel_names = (
            (f"Looking along the {direction.label}", "u (Angstrom)", "v (Angstrom)"),
            ("Side view: u-h", "u (Angstrom)", "height along the direction (Angstrom)"),
            ("Side view: v-h", "v (Angstrom)", "height along the direction (Angstrom)"),
        )
        axis_labels = ("u (Angstrom)", "v (Angstrom)", "height (Angstrom)")

    species = np.asarray(expand_species(record.species, record.counts, natoms=record.natoms), dtype=object)
    unique_species = list(dict.fromkeys(species.tolist()))
    colors = {symbol: _PALETTE[index % len(_PALETTE)] for index, symbol in enumerate(unique_species)}

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    heading = title or record.comment or "CELLSTINE structure view"
    if direction is not None:
        heading = f"{heading}  |  observed along the {direction.label}"
    fig.suptitle(heading, fontsize=15, fontweight="bold")
    axes_2d = [
        (fig.add_subplot(2, 2, 1), (0, 1), *panel_names[0]),
        (fig.add_subplot(2, 2, 2), (0, 2), *panel_names[1]),
        (fig.add_subplot(2, 2, 3), (1, 2), *panel_names[2]),
    ]

    handles_by_species: dict[str, object] = {}
    for ax, projection, panel_title, x_label, y_label in axes_2d:
        depth_axis = 3 - projection[0] - projection[1]
        drawn = species_depth_order(species, positions, depth_axis, unique_species)
        for depth_index, symbol in enumerate(drawn):
            mask = species == symbol
            handle = ax.scatter(
                positions[mask, projection[0]],
                positions[mask, projection[1]],
                s=_marker_size(str(symbol), projection="2d"),
                alpha=0.9,
                label=f"{symbol} (r={_atomic_radius(str(symbol)):.2f} A)",
                color=colors[str(symbol)],
                edgecolor="white",
                linewidth=0.55,
                zorder=3 + depth_index,
            )
            handles_by_species.setdefault(str(symbol), handle)
        _draw_projected_cell(ax, lattice, projection, label="unit cell")
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
    _draw_3d_cell(ax_3d, lattice)
    _set_equal_3d(ax_3d, positions, _cell_corners(lattice))
    ax_3d.set_title("3D overview")
    ax_3d.set_xlabel(axis_labels[0])
    ax_3d.set_ylabel(axis_labels[1])
    ax_3d.set_zlabel(axis_labels[2])
    ax_3d.view_init(elev=22, azim=-55)

    # The panels draw the species in depth order, which differs from panel to
    # panel; the legend keeps the order of the file so it reads the same way.
    handles = [handles_by_species[str(symbol)] for symbol in unique_species]
    labels = [f"{symbol} (r={_atomic_radius(str(symbol)):.2f} A)" for symbol in unique_species]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=True)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:  # pragma: no cover - interactive user convenience
        plt.show()
    plt.close(fig)
    return MatplotlibRun(output_path=output, item_count=record.natoms, visualization_type="structure_multiview")


def _read_moire_summary(
    results_file: str | Path, indices: Sequence[int] | None = None
) -> tuple[str, list[dict[str, object]], dict[str, Any]]:
    """Return plotting rows copied from validated native Gram JSON v2."""

    payload = read_results(results_file)
    rows = [
        {
            "index": candidate["index"],
            "angle_deg": candidate["angle_deg"],
            "relative_principal_strain": candidate["strain"],
            "top_layer_strain": candidate["top_layer_strain"],
            "bottom_layer_strain": candidate["bottom_layer_strain"],
            "moire_a": candidate["moire_a"],
            "moire_b": candidate["moire_b"],
            "moire_gamma_deg": candidate["moire_gamma_deg"],
            "coincidence_index": candidate["coincidence_index"],
            "top_atom_count": candidate["top_atom_count"],
            "bottom_atom_count": candidate["bottom_atom_count"],
            "atom_count": candidate["atom_count"],
            "rank": candidate["rank"],
            "pareto_optimal": candidate["pareto_optimal"],
            "loewner_certified": candidate["loewner_certified"],
            "loewner_borderline": candidate["loewner_borderline"],
            "top_matrix": candidate["top_matrix"],
            "bottom_matrix": candidate["bottom_matrix"],
            "shared_lattice": candidate["shared_lattice"],
        }
        for candidate in payload["candidates"]
    ]
    if indices is not None:
        wanted = {int(index) for index in indices}
        rows = [row for row in rows if int(row["index"]) in wanted]
    if not rows:
        raise ValueError("no candidates were selected for visualization")
    return "bilayer", rows, payload


def plot_moire_summary(
    results_file: str | Path,
    *,
    indices: Sequence[int] | None = None,
    output_path: str | Path,
    title: str | None = None,
    show: bool = False,
) -> MatplotlibRun:
    """Write a static summary plot for validated native Gram JSON v1."""

    plt = _pyplot()
    results_type, rows, payload = _read_moire_summary(results_file, indices)
    indexes = np.asarray([row["index"] for row in rows], dtype=float)
    ranks = np.asarray([row["rank"] for row in rows], dtype=float)
    angles = np.asarray([row["angle_deg"] for row in rows], dtype=float)
    relative_strain_percent = 100.0 * np.asarray(
        [max(abs(float(value)) for value in row["relative_principal_strain"]) for row in rows],
        dtype=float,
    )
    top_atoms = np.asarray([row["top_atom_count"] for row in rows], dtype=float)
    bottom_atoms = np.asarray([row["bottom_atom_count"] for row in rows], dtype=float)
    atoms = np.asarray([row["atom_count"] for row in rows], dtype=float)
    pareto = np.asarray([row["pareto_optimal"] for row in rows], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    metadata = payload["metadata"]
    search = payload["search"]
    resolved_title = title or (
        f"CELLSTINE {results_type} Gram commensuration summary "
        f"({metadata['engine']}, max length {float(search['max_length']):g} Angstrom)"
    )
    fig.suptitle(resolved_title, fontsize=15, fontweight="bold")

    scatter = axes[0, 0].scatter(angles, relative_strain_percent, c=atoms, cmap="viridis", s=48, edgecolor="white", linewidth=0.4)
    axes[0, 0].set_title("Relative principal strain by twist angle")
    axes[0, 0].set_xlabel("twist angle (degrees)")
    axes[0, 0].set_ylabel("max |relative principal strain| (%)")
    axes[0, 0].grid(True, linewidth=0.4, alpha=0.35)
    colorbar = fig.colorbar(scatter, ax=axes[0, 0])
    colorbar.set_label("total atoms")

    axes[0, 1].bar(indexes, bottom_atoms, color="#264653", label="bottom atoms")
    axes[0, 1].bar(indexes, top_atoms, bottom=bottom_atoms, color="#e76f51", label="top atoms")
    axes[0, 1].set_title("Candidate atom counts")
    axes[0, 1].set_xlabel("candidate index")
    axes[0, 1].set_ylabel("atoms")
    axes[0, 1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[0, 1].legend()

    for mask, color, label in (
        (pareto, "#2a9d8f", "Pareto optimal"),
        (~pareto, "#8d99ae", "non-Pareto"),
    ):
        if np.any(mask):
            axes[1, 0].scatter(
                ranks[mask],
                relative_strain_percent[mask],
                c=color,
                s=52,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
    axes[1, 0].axhline(
        100.0 * (float(search["top_strain"]) + float(search["bottom_strain"])),
        color="#e76f51",
        linestyle="--",
        linewidth=1.0,
        label="combined strain budget",
    )
    axes[1, 0].set_title("Rank and Pareto status")
    axes[1, 0].set_xlabel("rank")
    axes[1, 0].set_ylabel("max |relative principal strain| (%)")
    axes[1, 0].grid(True, linewidth=0.4, alpha=0.35)
    axes[1, 0].legend(fontsize=8)

    first = rows[0]
    certification = (
        "borderline"
        if first["loewner_borderline"]
        else "certified"
        if first["loewner_certified"]
        else "uncertified"
    )
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Selected candidate provenance")
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"candidate {int(first['index'])}, rank {int(first['rank'])}, "
                f"Pareto={bool(first['pareto_optimal'])}",
                f"Loewner certification: {certification}",
                f"top matrix: {first['top_matrix']}",
                f"bottom matrix: {first['bottom_matrix']}",
                f"shared lattice: {first['shared_lattice']}",
                f"symmetric used={metadata['symmetric_used']}; "
                f"fallback={metadata['symmetric_fallback'] or 'none'}",
            ]
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        wrap=True,
    )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:  # pragma: no cover - interactive user convenience
        plt.show()
    plt.close(fig)
    return MatplotlibRun(output_path=output, item_count=len(rows), visualization_type=f"{results_type}_summary")

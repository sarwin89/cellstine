"""Human-readable summaries of a defect analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .layers import member_layer_ids
from .records import DefectAnalysis
from .supercell import SupercellChoice


def _void_text(site: Any) -> str:
    """Return the empty sphere of an insertion site as ``radius kind (contacts)``.

    A ``maximum`` shrinks along every direction out of the site; a ``saddle``
    still grows along some of them, and is held only by the atoms on its sphere.
    """

    radius = getattr(site, "void_radius", None)
    if radius is None:
        return "-"
    kind = getattr(site, "void_kind", None) or "maximum"
    contacts = getattr(site, "void_coordination", None)
    suffix = "" if not contacts else f"({int(contacts)})"
    return f"{float(radius):.2f} {kind[:3]}{suffix}"


def _direction_label(analysis: DefectAnalysis) -> str:
    direction = analysis.view_direction or {}
    return str(direction.get("label") or "a-b surface normal")


class DefectReportingMixin:
    """Text reports for the defect workflow."""

    @staticmethod
    def format_layer_census(analysis: DefectAnalysis) -> str:
        """Return the per-plane census of atoms and of inequivalent sites."""

        layers = [layer for layer in analysis.layers if layer.get("species_counts")]
        if not layers:
            return ""
        direction = analysis.view_direction or {}
        spacing = direction.get("spacing")
        miller = direction.get("miller")
        heading = (
            f"Atomic planes seen along the {_direction_label(analysis)}"
            " (height along that direction, atoms, and how many inequivalent sites each plane holds)"
        )
        if miller and spacing:
            heading += (
                f"\nThe planes of the ({int(miller[0])} {int(miller[1])} {int(miller[2])}) family"
                f" are {float(spacing):.4f} A apart."
            )
        lines = [
            heading,
            " layer  height (A)  atoms  composition                inequivalent sites",
            "-" * 92,
        ]
        for layer in layers:
            composition = " ".join(
                f"{species}{count}"
                for species, count in dict(layer["species_counts"]).items()
            )
            distinct = " ".join(
                f"{species}:{count}"
                for species, count in dict(layer.get("inequivalent_sites", {})).items()
            )
            lines.append(
                f" {int(layer['layer_id']):5d}  {float(layer['projection']):10.4f}"
                f"  {int(layer['atom_count']):5d}  {composition:<25s}  {distinct}"
            )
        return "\n".join(lines)

    @staticmethod
    def format_analysis(analysis: DefectAnalysis, *, limit: int = 30) -> str:
        """Return a compact table of discovered defect sites.

        The ``planes`` column lists every atomic plane in which the orbit of a
        site has a member: those are the planes ``--layers`` can put that defect
        in, and a site that shows more than one of them is a defect the plane
        selection genuinely multiplies.
        """

        rows = list(analysis.sites)
        shown = rows[: max(0, int(limit))]
        if not shown:
            return "No defect sites were detected."
        lines = [
            f"Defect sites for {Path(analysis.structure_path).name} ({analysis.structure_kind}, backend={analysis.backend})",
            " site_id                 kind          species  layer  mult  wyckoff  empty sphere  direct (u, v, w)              planes            represented atoms",
            "-" * 152,
        ]
        for site in shown:
            direct = tuple(float(value) for value in site.direct)
            represented = ",".join(str(value) for value in site.equivalent_indices) if site.equivalent_indices else "-"
            planes = ",".join(str(value) for value in member_layer_ids(site)) or "-"
            lines.append(
                f" {site.site_id:<23s} {site.site_kind:<13s} {(site.species or '-'):>7s} "
                f"{str(site.layer_id or '-'):>6s} {int(site.multiplicity):5d} "
                f"{(site.wyckoff or '-'):>8s}  {_void_text(site):<12s}  "
                f"({direct[0]:7.4f}, {direct[1]:7.4f}, {direct[2]:7.4f})    {planes:<16s}  {represented}"
            )
        if len(rows) > len(shown):
            lines.append(f"... {len(rows) - len(shown)} more site(s) not shown.")
        census = DefectReportingMixin.format_layer_census(analysis)
        if census:
            lines.append("")
            lines.append(census)
        if analysis.notes:
            lines.append("")
            lines.append("Notes:")
            for note in analysis.notes:
                lines.append(f"- {note}")
        return "\n".join(lines)


def format_supercell_choice(
    choice: SupercellChoice, table: Sequence[dict[str, Any]] | None = None
) -> str:
    """Return the chosen supercell, and optionally the sizes around it, as text.

    The rows of the matrix are the new lattice vectors written in the host
    basis, which is the form a supercell is usually quoted in, and the reach
    column of the table is what Minkowski's theorem allows a cell of that size:
    a row that already sits at its own limit cannot be improved by reshaping,
    only by growing.
    """

    rows = [
        "Supercell chosen for the distance between the defect and its images",
        f"  host cells          {int(choice.cells)}",
        f"  image separation    {float(choice.image_distance):.3f} A ({choice.periodicity})",
        f"  best possible here  {float(choice.upper_bound):.3f} A",
    ]
    if choice.diagonal_distance is not None:
        rows.append(f"  best plain repeat   {float(choice.diagonal_distance):.3f} A")
    rows.append("  matrix              " + _matrix_text(choice.matrix))
    if table:
        rows.extend(
            [
                "",
                " cells  separation (A)  best possible (A)  matrix",
                "-" * 78,
            ]
        )
        for entry in table:
            marker = "*" if entry.get("improves") else " "
            rows.append(
                f"{marker}{int(entry['cells']):5d}  {float(entry['image_distance']):14.3f}"
                f"  {float(entry['best_possible_distance']):17.3f}  {_matrix_text(entry['matrix'])}"
            )
        rows.append("* marks a size that beats every smaller one.")
    return "\n".join(rows)


def _matrix_text(matrix: Sequence[Sequence[int]]) -> str:
    return " ".join(
        "[" + " ".join(f"{int(value):d}" for value in row) + "]" for row in matrix
    )

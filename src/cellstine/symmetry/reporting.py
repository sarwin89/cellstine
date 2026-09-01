"""Human-readable reporting helpers for symmetry workflow results."""

from __future__ import annotations

from .models import SymmetryAnalysis


def format_symmetry_analysis(analysis: SymmetryAnalysis) -> str:
    """Return the compact CLI preview for a symmetry analysis."""

    lines = [f"Symmetry analysis ({analysis.backend})"]
    if analysis.space_group_symbol:
        lines.append(f"Space group: {analysis.space_group_symbol} ({analysis.space_group_number})")
    if analysis.point_group:
        lines.append(f"Point group: {analysis.point_group}")
    if analysis.crystal_system:
        lines.append(f"Crystal system: {analysis.crystal_system}")
    if analysis.lattice_point_group:
        lines.append(f"Lattice point group: {analysis.lattice_point_group}")
    lines.append(f"Atoms: {analysis.atom_count}")
    lines.append(f"Operations: {analysis.operation_count}")
    if analysis.centering_translation_count and analysis.centering_translation_count > 1:
        lines.append(f"Centering translations: {analysis.centering_translation_count}")
    if analysis.equivalent_groups:
        lines.append("Equivalent atom groups:")
        for group in analysis.equivalent_groups[:20]:
            represented = ",".join(str(index) for index in group.equivalent_indices)
            wyckoff = group.wyckoff or "-"
            lines.append(f"  {group.group_id} {group.species} mult={group.multiplicity} wyckoff={wyckoff} atoms={represented}")
        if len(analysis.equivalent_groups) > 20:
            lines.append(f"  ... {len(analysis.equivalent_groups) - 20} more group(s)")
    if analysis.notes:
        lines.append("Notes:")
        for note in analysis.notes[:4]:
            lines.append(f"- {note}")
    return "\n".join(lines)

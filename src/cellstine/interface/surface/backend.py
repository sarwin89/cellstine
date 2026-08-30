"""Surface-slab builder and adsorption-site analysis for substrate POSCAR inputs.

The heavy lifting lives in three layered modules: :mod:`surface_cell` builds the
primitive surface cell for a Miller plane, :mod:`surface_supercell` turns it into
a slab, and :mod:`surface_sites` provides the adsorption-site geometry.  This
module keeps the site report, the site selection helpers, and the two top-level
entry points, and re-exports the rest -- including the stoichiometry and
termination checks of :mod:`termination` -- so that ``surface.backend`` remains
the single public face of the package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ...core.layers import LAYER_TOLERANCE
from ...core.planar_voids import find_planar_voids
from ...core.provenance import stage_comment
from ...io import native as io_mod
from .surface_cell import (
    _build_native_primitive_surface_cell,
    _centering_type,
    _deduplicate_scalar_levels,
    _group_surface_atoms_by_species,
    _integer_plane_form,
    _primitive_surface_vectors,
    _primitive_surface_vectors_from_lattice,
    _primitive_translation_lattice,
    _reciprocal_normal,
    _select_surface_pair,
    _structure_from_inplane_transform,
    _structure_from_transform,
    _surface_coordinate_frame,
    _surface_vector_search_limit,
    _translation_maps_structure,
)
from .surface_sites import (
    _anchor_image_distance_matrix,
    _classify_hollow,
    _cluster_projection_levels,
    _deduplicate_uv_points,
    _expanded_periodic_arrays,
    _find_bridge_sites,
    _inplane_cartesian_from_uv,
    _is_close_packed_stacking,
    _nearest_neighbor_distance,
    _projection_layer_points,
    _site_from_uv,
    _subsurface_depth_below,
    _top_layer_coordination_counts,
    _uv_to_cartesian,
)
from .sequence import shortest_repeating_prefix, stacking_sequence
from .termination import TerminationReport, formula_unit, layer_species, termination_report
from .stacking import surface_normal
from .surface_supercell import (
    _resolve_inplane_repeats,
    analyse_primitive_surface,
    apply_inplane_supercell_matrix,
    build_surface_structure,
    repeat_structure_inplane,
)
from .surface_types import (
    DEFAULT_OUTPUT_DIR,
    SITE_TYPE_ALIASES,
    AdsorptionSite,
    PrimitiveSurfaceAnalysis,
    SiteAnalysisRun,
    SurfaceRun,
    SurfaceStructureBuild,
)

__all__ = [
    "AdsorptionSite",
    "TerminationReport",
    "formula_unit",
    "layer_species",
    "termination_report",
    "DEFAULT_OUTPUT_DIR",
    "PrimitiveSurfaceAnalysis",
    "SITE_TYPE_ALIASES",
    "SiteAnalysisRun",
    "SurfaceRun",
    "SurfaceStructureBuild",
    "analyse_primitive_surface",
    "apply_inplane_supercell_matrix",
    "build_surface",
    "build_surface_structure",
    "canonical_site_type",
    "find_adsorption_sites",
    "repeat_structure_inplane",
    "shortest_repeating_prefix",
    "stacking_sequence",
    "surface_normal",
    "select_adsorption_site",
    "sorted_sites_for_type",
    "write_site_report_json",
    "write_surface_poscar",
    "_anchor_image_distance_matrix",
    "_build_native_primitive_surface_cell",
    "_centering_type",
    "_classify_hollow",
    "_cluster_projection_levels",
    "_deduplicate_scalar_levels",
    "_deduplicate_uv_points",
    "_expanded_periodic_arrays",
    "_find_bridge_sites",
    "_group_surface_atoms_by_species",
    "_inplane_cartesian_from_uv",
    "_integer_plane_form",
    "_is_close_packed_stacking",
    "_nearest_neighbor_distance",
    "_primitive_surface_vectors",
    "_primitive_surface_vectors_from_lattice",
    "_primitive_translation_lattice",
    "_projection_layer_points",
    "_reciprocal_normal",
    "_resolve_inplane_repeats",
    "_select_surface_pair",
    "_site_from_uv",
    "_site_report_to_dict",
    "_structure_from_inplane_transform",
    "_structure_from_transform",
    "_subsurface_depth_below",
    "_surface_coordinate_frame",
    "_surface_vector_search_limit",
    "_top_layer_coordination_counts",
    "_translation_maps_structure",
    "_uv_to_cartesian",
]


def _site_report_to_dict(run: SiteAnalysisRun) -> dict[str, object]:
    sites_by_type: Dict[str, list[dict[str, object]]] = {}
    for site in run.sites:
        entry: dict[str, object] = {
            "direct": [float(value) for value in site.direct],
            "cartesian": [float(value) for value in site.cartesian],
        }
        if site.coordination is not None:
            entry["coordination"] = int(site.coordination)
        if site.void_radius is not None:
            entry["empty_circle_radius_angstrom"] = float(site.void_radius)
        if site.subsurface_depth is not None:
            entry["subsurface_depth"] = int(site.subsurface_depth)
        sites_by_type.setdefault(site.site_type, []).append(entry)
    return {
        "source_poscar": run.source_poscar,
        "surface_side": run.surface_side,
        "top_layer_atom_count": int(run.top_layer_atom_count),
        "detected_layer_count": int(run.detected_layer_count),
        "nearest_neighbor_distance_angstrom": float(run.nearest_neighbor_distance),
        "neighbor_cutoff_angstrom": float(run.neighbor_cutoff),
        "average_top_layer_coordination": float(run.average_top_layer_coordination),
        "site_counts": {str(key): int(value) for key, value in run.site_counts.items()},
        "sites": sites_by_type,
    }


def write_site_report_json(path: str, run: SiteAnalysisRun) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_site_report_to_dict(run), handle, indent=2)
        handle.write("\n")
    return output_path


def canonical_site_type(site_type: str) -> str:
    key = str(site_type).strip().lower()
    if key not in SITE_TYPE_ALIASES:
        allowed = ", ".join(sorted(SITE_TYPE_ALIASES))
        raise ValueError(f"unsupported site type {site_type!r}; choose one of: {allowed}")
    return SITE_TYPE_ALIASES[key]


def sorted_sites_for_type(run: SiteAnalysisRun, site_type: str) -> list[AdsorptionSite]:
    canonical = canonical_site_type(site_type)
    filtered = [site for site in run.sites if site.site_type == canonical]
    filtered.sort(key=lambda item: (round(float(item.direct[0]), 12), round(float(item.direct[1]), 12), round(float(item.direct[2]), 12)))
    return filtered


def select_adsorption_site(run: SiteAnalysisRun, site_type: str, site_index: int = 1) -> AdsorptionSite:
    index = int(site_index)
    if index < 1:
        raise ValueError("site_index is 1-based and must be at least 1")
    filtered = sorted_sites_for_type(run, site_type)
    if not filtered:
        available = ", ".join(sorted(run.site_counts)) if run.site_counts else "none"
        raise ValueError(f"no adsorption sites of type {canonical_site_type(site_type)!r} were found; available types: {available}")
    if index > len(filtered):
        raise ValueError(
            f"site_index={index} is out of range for site type {canonical_site_type(site_type)!r}; "
            f"there are only {len(filtered)} matching sites in this cell"
        )
    return filtered[index - 1]


def find_adsorption_sites(
    structure_or_path: io_mod.PoscarData | str,
    *,
    surface_side: str = "top",
    layer_tolerance: float = LAYER_TOLERANCE,
    neighbour_tolerance: float = 0.15,
    hollow_match_tolerance: float | None = None,
    output_path: str | None = None,
    source_poscar: str | None = None,
) -> SiteAnalysisRun:
    """Enumerate the adsorption sites exposed by one side of a slab.

    Top sites are the surface atoms, bridge sites the midpoints of the
    nearest-neighbour bonds within the surface layer, and hollows the local
    maxima of the in-plane distance to the surface atoms -- the vertices of
    their two-dimensional Voronoi diagram.  That definition finds the hollow of
    any surface, including a honeycomb layer, whose hexagon centre is not the
    centroid of any triangle of mutual neighbours.

    Every hollow is reported with the radius of the largest circle that fits in
    it, with the number of surface atoms that touch that circle, and with the
    depth of the first subsurface layer holding an atom directly below it.
    """

    if surface_side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'")

    if isinstance(structure_or_path, str):
        resolved_source = str(Path(structure_or_path).resolve())
        structure = io_mod.read_poscar(structure_or_path)
    else:
        resolved_source = str(Path(source_poscar).resolve()) if source_poscar is not None else None
        structure = structure_or_path

    lattice = np.asarray(structure.lattice, dtype=float)
    normal = surface_normal(lattice)
    projections = np.asarray(structure.positions_cartesian, dtype=float) @ normal
    detected_layers = _projection_layer_points(structure.positions_direct, projections, surface_side, float(layer_tolerance))
    if not detected_layers:
        raise ValueError("could not detect any surface layers in the slab")

    top_projection, top_points_uv = detected_layers[0]
    if top_points_uv.size == 0:
        raise ValueError("surface layer detection returned no atoms in the outermost layer")

    nearest_neighbour = _nearest_neighbor_distance(top_points_uv, lattice)
    neighbour_cutoff = nearest_neighbour * (1.0 + float(neighbour_tolerance))
    match_tolerance = float(hollow_match_tolerance) if hollow_match_tolerance is not None else max(1e-4, 0.2 * nearest_neighbour)

    coordination_counts = _top_layer_coordination_counts(top_points_uv, lattice, neighbour_cutoff)
    top_sites_uv = _deduplicate_uv_points([np.array(point, dtype=float) for point in top_points_uv], lattice)
    bridge_sites_uv = _find_bridge_sites(top_points_uv, lattice, neighbour_cutoff)
    hollows = find_planar_voids(lattice[:2], top_points_uv)

    lower_layers = detected_layers[1:]
    close_packed = _is_close_packed_stacking(detected_layers)
    sites: list[AdsorptionSite] = []
    for uv in top_sites_uv:
        sites.append(_site_from_uv("top", uv, lattice, top_projection, normal))
    for uv in bridge_sites_uv:
        sites.append(_site_from_uv("bridge", uv, lattice, top_projection, normal))
    for hollow in hollows:
        hollow_uv = np.asarray(hollow.uv, dtype=float)
        depth = _subsurface_depth_below(hollow_uv, lattice, lower_layers, match_tolerance)
        sites.append(
            _site_from_uv(
                _classify_hollow(hollow, depth, close_packed),
                hollow_uv,
                lattice,
                top_projection,
                normal,
                coordination=hollow.coordination,
                void_radius=hollow.radius,
                subsurface_depth=depth,
            )
        )

    site_counts: Dict[str, int] = {}
    for site in sites:
        site_counts[site.site_type] = site_counts.get(site.site_type, 0) + 1

    run = SiteAnalysisRun(
        output_path=None,
        source_poscar=resolved_source,
        surface_side=str(surface_side),
        top_layer_atom_count=int(top_points_uv.shape[0]),
        detected_layer_count=int(len(detected_layers)),
        nearest_neighbor_distance=float(nearest_neighbour),
        neighbor_cutoff=float(neighbour_cutoff),
        average_top_layer_coordination=float(np.mean(coordination_counts)) if coordination_counts else 0.0,
        site_counts=site_counts,
        sites=sites,
    )

    if output_path is not None:
        written_path = write_site_report_json(output_path, run)
        run.output_path = written_path
    return run


def write_surface_poscar(output_path: str, structure: io_mod.PoscarData, miller: Sequence[int]) -> Path:
    """Write a slab POSCAR, creating the destination directory if need be."""

    resolved = Path(output_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    io_mod.write_poscar(
        str(resolved),
        structure.lattice,
        structure.positions_direct,
        structure.counts,
        structure.species,
        comment=stage_comment(
            "surface",
            f"Miller ({int(miller[0])} {int(miller[1])} {int(miller[2])})",
        ),
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=structure.selective_flags,
    )
    return resolved


def build_surface(
    bulk_poscar: str,
    *,
    miller: tuple[int, int, int],
    layers: int,
    vacuum: float,
    repeat_a: int = 1,
    repeat_b: int = 1,
    min_length_a: float | None = None,
    min_length_b: float | None = None,
    supercell_matrix: Sequence[int] | None = None,
    output_path: str | None = None,
    sites_output_path: str | None = None,
    analyse_sites: bool = False,
    site_surface_side: str = "top",
) -> SurfaceRun:
    build = build_surface_structure(
        bulk_poscar,
        miller=miller,
        layers=int(layers),
        vacuum=float(vacuum),
        repeat_a=int(repeat_a),
        repeat_b=int(repeat_b),
        min_length_a=min_length_a,
        min_length_b=min_length_b,
        supercell_matrix=supercell_matrix,
    )
    surfaced = build.structure
    resolved_repeat_a = build.repeat_a
    resolved_repeat_b = build.repeat_b
    applied_matrix = build.supercell_matrix

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            DEFAULT_OUTPUT_DIR
            / f"surface_{Path(bulk_poscar).stem}_{int(miller[0])}{int(miller[1])}{int(miller[2])}_layers{int(layers)}.vasp"
        )
    write_surface_poscar(output_path, surfaced, miller)

    site_report_path: Path | None = None
    site_counts: Dict[str, int] | None = None
    if analyse_sites or sites_output_path is not None:
        if sites_output_path is None:
            site_file = Path(output_path).with_suffix("")
            sites_output_path = str(site_file) + "_sites.json"
        site_run = find_adsorption_sites(
            surfaced,
            surface_side=site_surface_side,
            output_path=sites_output_path,
            source_poscar=output_path,
        )
        site_report_path = site_run.output_path
        site_counts = dict(site_run.site_counts)

    return SurfaceRun(
        output_path=Path(output_path).resolve(),
        miller=(int(miller[0]), int(miller[1]), int(miller[2])),
        layers=int(layers),
        vacuum=float(vacuum),
        total_atoms=surfaced.natoms,
        repeat_a=int(resolved_repeat_a),
        repeat_b=int(resolved_repeat_b),
        supercell_matrix=applied_matrix,
        site_output_path=site_report_path,
        site_counts=site_counts,
    )

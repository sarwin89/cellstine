"""Surface-slab builder and adsorption-site analysis for substrate POSCAR inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import io as io_mod

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class AdsorptionSite:
    site_type: str
    direct: tuple[float, float, float]
    cartesian: tuple[float, float, float]


@dataclass
class SiteAnalysisRun:
    output_path: Path | None
    source_poscar: str | None
    surface_side: str
    top_layer_atom_count: int
    detected_layer_count: int
    nearest_neighbor_distance: float
    neighbor_cutoff: float
    average_top_layer_coordination: float
    site_counts: Dict[str, int]
    sites: List[AdsorptionSite]


@dataclass
class SurfaceRun:
    output_path: Path
    miller: tuple[int, int, int]
    layers: int
    vacuum: float
    total_atoms: int
    repeat_a: int
    repeat_b: int
    supercell_matrix: tuple[int, int, int, int] | None
    site_output_path: Path | None
    site_counts: Dict[str, int] | None


SITE_TYPE_ALIASES = {
    "top": "top",
    "bridge": "bridge",
    "hcp": "hcp_hollow",
    "hcp_hollow": "hcp_hollow",
    "fcc": "fcc_hollow",
    "fcc_hollow": "fcc_hollow",
    "hollow": "hollow",
    "fourfold": "fourfold_hollow",
    "fourfold_hollow": "fourfold_hollow",
}


def _reduce_integer_vector(values: Sequence[int]) -> tuple[int, int, int]:
    entries = [int(value) for value in values]
    divisor = 0
    for entry in entries:
        divisor = math.gcd(divisor, abs(entry))
    divisor = max(divisor, 1)
    return tuple(int(entry // divisor) for entry in entries)


def _is_orthogonal_lattice(lattice: np.ndarray, tolerance: float = 1e-6) -> bool:
    vectors = np.asarray(lattice, dtype=float)
    return (
        abs(float(np.dot(vectors[0], vectors[1]))) <= tolerance
        and abs(float(np.dot(vectors[0], vectors[2]))) <= tolerance
        and abs(float(np.dot(vectors[1], vectors[2]))) <= tolerance
    )


def _choose_in_plane_vectors(miller: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    h, k, l = (int(value) for value in miller)
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices cannot all be zero")

    normal = np.array([h, k, l], dtype=int)
    axis_candidates = [
        np.array([1, 0, 0], dtype=int),
        np.array([0, 1, 0], dtype=int),
        np.array([0, 0, 1], dtype=int),
    ]
    first = None
    for axis in axis_candidates:
        candidate = np.cross(normal, axis)
        if np.any(candidate != 0):
            first = candidate
            break
    if first is None:
        raise ValueError(f"could not build an in-plane basis for Miller indices {miller}")

    second = np.cross(normal, first)
    first_reduced = _reduce_integer_vector(first.tolist())
    second_reduced = _reduce_integer_vector(second.tolist())
    normal_reduced = _reduce_integer_vector(normal.tolist())

    transform = np.array([first_reduced, second_reduced, normal_reduced], dtype=int)
    determinant = int(round(np.linalg.det(transform)))
    if determinant == 0:
        raise ValueError(f"surface transform for Miller indices {miller} is singular")
    if determinant < 0:
        transform[[0, 1]] = transform[[1, 0]]
        first_reduced, second_reduced = second_reduced, first_reduced
    return first_reduced, second_reduced, normal_reduced


def _structure_from_transform(structure: io_mod.PoscarData, transform: np.ndarray, tolerance: float = 1e-8) -> io_mod.PoscarData:
    lattice_old = np.asarray(structure.lattice, dtype=float)
    lattice_new = transform @ lattice_old
    inverse_new = np.linalg.inv(lattice_new)

    search_pad = max(2, int(np.max(np.abs(transform))) + 1)
    collected_positions: List[np.ndarray] = []
    collected_flags: List[Tuple[str, str, str]] | None = [] if structure.selective_flags is not None else None
    expanded_flags = structure.selective_flags or []

    for atom_index, base_direct in enumerate(np.asarray(structure.positions_direct, dtype=float)):
        for shift_a in range(-search_pad, search_pad + 1):
            for shift_b in range(-search_pad, search_pad + 1):
                for shift_c in range(-search_pad, search_pad + 1):
                    image_direct = np.array(
                        [base_direct[0] + shift_a, base_direct[1] + shift_b, base_direct[2] + shift_c],
                        dtype=float,
                    )
                    image_cart = io_mod.direct_to_cartesian(image_direct.reshape(1, 3), lattice_old)[0]
                    new_direct = image_cart @ inverse_new
                    if not np.all((-tolerance <= new_direct) & (new_direct <= 1.0 + tolerance)):
                        continue
                    wrapped = np.mod(new_direct, 1.0)
                    duplicate = False
                    for previous in collected_positions:
                        difference = wrapped - previous
                        if np.all(np.abs(difference - np.round(difference)) <= tolerance):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    collected_positions.append(wrapped)
                    if collected_flags is not None:
                        collected_flags.append(tuple(expanded_flags[atom_index]))

    if not collected_positions:
        raise ValueError("surface transform did not capture any atoms; try a simpler Miller index")

    positions_direct = np.array(collected_positions, dtype=float)
    positions_cartesian = io_mod.direct_to_cartesian(positions_direct, lattice_new)
    multiplicity = int(round(abs(np.linalg.det(transform))))
    counts = [int(count) * multiplicity for count in structure.counts]
    return io_mod.PoscarData(
        comment=f"{structure.comment} | oriented surface cell",
        lattice=lattice_new,
        species=list(structure.species),
        counts=counts,
        positions_direct=positions_direct,
        positions_cartesian=positions_cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=collected_flags,
    )


def _repeat_structure_inplane(structure: io_mod.PoscarData, repeat_a: int, repeat_b: int) -> io_mod.PoscarData:
    repeat_a = int(repeat_a)
    repeat_b = int(repeat_b)
    if repeat_a < 1 or repeat_b < 1:
        raise ValueError("in-plane repeats must be at least 1")
    if repeat_a == 1 and repeat_b == 1:
        return structure

    lattice = np.array(structure.lattice, dtype=float, copy=True)
    lattice[0] *= float(repeat_a)
    lattice[1] *= float(repeat_b)

    direct_blocks = []
    flags_out: List[Tuple[str, str, str]] | None = [] if structure.selective_flags is not None else None
    for i_repeat in range(repeat_a):
        for j_repeat in range(repeat_b):
            shifted = np.array(structure.positions_direct, dtype=float, copy=True)
            shifted[:, 0] = (shifted[:, 0] + float(i_repeat)) / float(repeat_a)
            shifted[:, 1] = (shifted[:, 1] + float(j_repeat)) / float(repeat_b)
            direct_blocks.append(shifted)
            if flags_out is not None:
                flags_out.extend(tuple(flags) for flags in structure.selective_flags or [])

    positions_direct = np.vstack(direct_blocks)
    return io_mod.PoscarData(
        comment=f"{structure.comment} | in-plane repeat {repeat_a}x{repeat_b}",
        lattice=lattice,
        species=list(structure.species),
        counts=[int(count) * repeat_a * repeat_b for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=io_mod.direct_to_cartesian(positions_direct, lattice),
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=flags_out,
    )


def _apply_inplane_supercell_matrix(
    structure: io_mod.PoscarData,
    supercell_matrix: Sequence[int] | None,
) -> tuple[io_mod.PoscarData, tuple[int, int, int, int] | None]:
    if supercell_matrix is None:
        return structure, None

    entries = [int(value) for value in supercell_matrix]
    if len(entries) != 4:
        raise ValueError("supercell_matrix must contain exactly four integers")

    matrix = np.array([[entries[0], entries[1], 0], [entries[2], entries[3], 0], [0, 0, 1]], dtype=int)
    determinant = int(round(np.linalg.det(matrix)))
    if determinant == 0:
        raise ValueError("in-plane supercell matrix must have a non-zero determinant")
    if determinant < 0:
        matrix[[0, 1]] = matrix[[1, 0]]

    transformed = _structure_from_transform(structure, matrix)
    applied = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )
    return transformed, applied


def _resolve_inplane_repeats(
    structure: io_mod.PoscarData,
    repeat_a: int,
    repeat_b: int,
    min_length_a: float | None,
    min_length_b: float | None,
) -> tuple[int, int]:
    resolved_a = max(1, int(repeat_a))
    resolved_b = max(1, int(repeat_b))

    length_a = float(np.linalg.norm(structure.lattice[0]))
    length_b = float(np.linalg.norm(structure.lattice[1]))
    if min_length_a is not None and min_length_a > 0.0 and length_a > 1e-12:
        resolved_a = max(resolved_a, int(math.ceil(float(min_length_a) / length_a)))
    if min_length_b is not None and min_length_b > 0.0 and length_b > 1e-12:
        resolved_b = max(resolved_b, int(math.ceil(float(min_length_b) / length_b)))
    return resolved_a, resolved_b


def _add_vacuum_along_c(structure: io_mod.PoscarData, vacuum: float, padding: float = 0.5) -> io_mod.PoscarData:
    vacuum = float(vacuum)
    if vacuum < 0.0:
        raise ValueError("vacuum must be non-negative")

    lattice = np.array(structure.lattice, dtype=float, copy=True)
    c_vector = lattice[2]
    c_length = float(np.linalg.norm(c_vector))
    if c_length <= 1e-12:
        raise ValueError("surface cell has a zero-length c vector")
    c_unit = c_vector / c_length

    cartesian = np.array(structure.positions_cartesian, dtype=float, copy=True)
    projections = cartesian @ c_unit
    cartesian += (padding - float(projections.min())) * c_unit

    lattice[2] = c_unit * (c_length + vacuum)
    positions_direct = io_mod.cartesian_to_direct(cartesian, lattice)
    return io_mod.PoscarData(
        comment=f"{structure.comment} | vacuum {vacuum:.3f} A",
        lattice=lattice,
        species=list(structure.species),
        counts=[int(count) for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=None if structure.selective_flags is None else [tuple(flags) for flags in structure.selective_flags],
    )


def _surface_normal(lattice: np.ndarray) -> np.ndarray:
    normal = np.cross(np.asarray(lattice, dtype=float)[0], np.asarray(lattice, dtype=float)[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("surface lattice has a zero in-plane area")
    return normal / norm


def _shift_boundary_into_interlayer_gap(structure: io_mod.PoscarData) -> io_mod.PoscarData:
    lattice = np.asarray(structure.lattice, dtype=float)
    normal = _surface_normal(lattice)
    c_length = float(np.linalg.norm(lattice[2]))
    if c_length <= 1e-12:
        return structure

    projections = np.mod(np.asarray(structure.positions_cartesian, dtype=float) @ normal, c_length)
    if projections.size == 0:
        return structure

    ordered = np.sort(projections)
    cyclic = np.concatenate([ordered, ordered[:1] + c_length])
    gaps = np.diff(cyclic)
    gap_index = int(np.argmax(gaps))
    cut = float((ordered[gap_index] + 0.5 * gaps[gap_index]) % c_length)
    shifted_cartesian = np.asarray(structure.positions_cartesian, dtype=float) - cut * normal
    shifted_direct = io_mod.cartesian_to_direct(shifted_cartesian, lattice)
    shifted_direct = io_mod.wrap_direct(shifted_direct)
    return io_mod.PoscarData(
        comment=f"{structure.comment} | shifted to keep full surface planes at the boundaries",
        lattice=np.array(structure.lattice, dtype=float, copy=True),
        species=list(structure.species),
        counts=[int(count) for count in structure.counts],
        positions_direct=shifted_direct,
        positions_cartesian=io_mod.direct_to_cartesian(shifted_direct, lattice),
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=None if structure.selective_flags is None else [tuple(flags) for flags in structure.selective_flags],
    )


def _cluster_projection_levels(values: np.ndarray, tolerance: float) -> list[tuple[float, list[int]]]:
    order = np.argsort(values)
    groups: list[list[int]] = []
    centers: list[float] = []
    for index in order.tolist():
        value = float(values[index])
        if not groups:
            groups.append([index])
            centers.append(value)
            continue
        if abs(value - centers[-1]) <= tolerance:
            groups[-1].append(index)
            centers[-1] = float(np.mean(values[groups[-1]]))
        else:
            groups.append([index])
            centers.append(value)
    return [(float(centers[idx]), groups[idx]) for idx in range(len(groups))]


def _inplane_cartesian_from_uv(uv: Sequence[float], lattice: np.ndarray) -> np.ndarray:
    basis = np.asarray(lattice, dtype=float)[:2]
    return float(uv[0]) * basis[0] + float(uv[1]) * basis[1]


def _minimum_image_inplane_delta(uv_a: Sequence[float], uv_b: Sequence[float]) -> np.ndarray:
    delta = np.array([float(uv_a[0]) - float(uv_b[0]), float(uv_a[1]) - float(uv_b[1])], dtype=float)
    return delta - np.round(delta)


def _minimum_image_inplane_distance(uv_a: Sequence[float], uv_b: Sequence[float], lattice: np.ndarray) -> float:
    delta = _minimum_image_inplane_delta(uv_a, uv_b)
    return float(np.linalg.norm(_inplane_cartesian_from_uv(delta, lattice)))


def _deduplicate_uv_points(points_uv: Sequence[np.ndarray], lattice: np.ndarray, tolerance: float = 1e-4) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points_uv:
        wrapped = np.mod(np.asarray(point, dtype=float), 1.0)
        if any(_minimum_image_inplane_distance(wrapped, existing, lattice) <= tolerance for existing in unique):
            continue
        unique.append(wrapped)
    return unique


def _expanded_periodic_points(points_uv: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    expanded: list[tuple[int, int, int, np.ndarray]] = []
    for base_index, uv in enumerate(np.asarray(points_uv, dtype=float)):
        for shift_u in (-1, 0, 1):
            for shift_v in (-1, 0, 1):
                expanded.append((base_index, shift_u, shift_v, uv + np.array([shift_u, shift_v], dtype=float)))
    return expanded


def _nearest_neighbor_distance(points_uv: np.ndarray, lattice: np.ndarray) -> float:
    expanded = _expanded_periodic_points(points_uv)
    best = math.inf
    for anchor_index, anchor_uv in enumerate(np.asarray(points_uv, dtype=float)):
        for base_index, shift_u, shift_v, shifted_uv in expanded:
            if base_index == anchor_index and shift_u == 0 and shift_v == 0:
                continue
            distance = float(np.linalg.norm(_inplane_cartesian_from_uv(shifted_uv - anchor_uv, lattice)))
            if distance <= 1e-8:
                continue
            best = min(best, distance)
    if not math.isfinite(best):
        raise ValueError("could not determine an in-plane nearest-neighbour distance from the top surface atoms")
    return best


def _top_layer_coordination_counts(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[int]:
    expanded = _expanded_periodic_points(points_uv)
    counts: list[int] = []
    for anchor_index, anchor_uv in enumerate(np.asarray(points_uv, dtype=float)):
        count = 0
        for base_index, shift_u, shift_v, shifted_uv in expanded:
            if base_index == anchor_index and shift_u == 0 and shift_v == 0:
                continue
            distance = float(np.linalg.norm(_inplane_cartesian_from_uv(shifted_uv - anchor_uv, lattice)))
            if 1e-8 < distance <= neighbour_cutoff + 1e-12:
                count += 1
        counts.append(count)
    return counts


def _find_bridge_sites(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    expanded = _expanded_periodic_points(points_uv)
    bridge_points: list[np.ndarray] = []
    for anchor_index, anchor_uv in enumerate(np.asarray(points_uv, dtype=float)):
        for base_index, shift_u, shift_v, shifted_uv in expanded:
            if base_index == anchor_index and shift_u == 0 and shift_v == 0:
                continue
            distance = float(np.linalg.norm(_inplane_cartesian_from_uv(shifted_uv - anchor_uv, lattice)))
            if 1e-8 < distance <= neighbour_cutoff + 1e-12:
                bridge_points.append(np.mod(0.5 * (anchor_uv + shifted_uv), 1.0))
    return _deduplicate_uv_points(bridge_points, lattice)


def _find_triangular_hollows(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    expanded = _expanded_periodic_points(points_uv)
    hollows: list[np.ndarray] = []
    for anchor_uv in np.asarray(points_uv, dtype=float):
        neighbours = [
            shifted_uv
            for _, shift_u, shift_v, shifted_uv in expanded
            if not (shift_u == 0 and shift_v == 0 and np.allclose(shifted_uv, anchor_uv))
            and 1e-8 < float(np.linalg.norm(_inplane_cartesian_from_uv(shifted_uv - anchor_uv, lattice))) <= neighbour_cutoff + 1e-12
        ]
        for first_index in range(len(neighbours)):
            for second_index in range(first_index + 1, len(neighbours)):
                point_b = neighbours[first_index]
                point_c = neighbours[second_index]
                edge_bc = float(np.linalg.norm(_inplane_cartesian_from_uv(point_c - point_b, lattice)))
                if edge_bc > neighbour_cutoff + 1e-12:
                    continue
                vector_ab = _inplane_cartesian_from_uv(point_b - anchor_uv, lattice)
                vector_ac = _inplane_cartesian_from_uv(point_c - anchor_uv, lattice)
                area = abs(float(np.cross(vector_ab, vector_ac)[2])) if vector_ab.shape[0] == 3 else abs(float(vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0]))
                if area <= 1e-8:
                    continue
                hollows.append(np.mod((anchor_uv + point_b + point_c) / 3.0, 1.0))
    return _deduplicate_uv_points(hollows, lattice)


def _find_fourfold_hollows(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    expanded = _expanded_periodic_points(points_uv)
    hollows: list[np.ndarray] = []
    angle_window = (70.0, 110.0)
    for anchor_uv in np.asarray(points_uv, dtype=float):
        neighbours: list[np.ndarray] = []
        for _, shift_u, shift_v, shifted_uv in expanded:
            if shift_u == 0 and shift_v == 0 and np.allclose(shifted_uv, anchor_uv):
                continue
            displacement = shifted_uv - anchor_uv
            distance = float(np.linalg.norm(_inplane_cartesian_from_uv(displacement, lattice)))
            if 1e-8 < distance <= neighbour_cutoff + 1e-12:
                neighbours.append(displacement)

        for first_index in range(len(neighbours)):
            for second_index in range(first_index + 1, len(neighbours)):
                disp_a = neighbours[first_index]
                disp_b = neighbours[second_index]
                vec_a = _inplane_cartesian_from_uv(disp_a, lattice)
                vec_b = _inplane_cartesian_from_uv(disp_b, lattice)
                denominator = max(float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b)), 1e-12)
                cosine = np.clip(float(np.dot(vec_a, vec_b) / denominator), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cosine)))
                if not (angle_window[0] <= angle_deg <= angle_window[1]):
                    continue

                corner_uv = anchor_uv + disp_a + disp_b
                has_fourth_corner = any(
                    float(np.linalg.norm(_inplane_cartesian_from_uv(candidate_uv - corner_uv, lattice))) <= 1e-6
                    for _, _, _, candidate_uv in expanded
                )
                if not has_fourth_corner:
                    continue
                hollows.append(np.mod(anchor_uv + 0.5 * (disp_a + disp_b), 1.0))
    return _deduplicate_uv_points(hollows, lattice)


def _projection_layer_points(
    positions_direct: np.ndarray,
    projections: np.ndarray,
    side: str,
    tolerance: float,
    max_layers: int = 3,
) -> list[tuple[float, np.ndarray]]:
    groups = _cluster_projection_levels(projections, tolerance)
    ordered = sorted(groups, key=lambda item: item[0], reverse=(side == "top"))
    layers: list[tuple[float, np.ndarray]] = []
    for center, indices in ordered[:max_layers]:
        layers.append((center, np.mod(np.asarray(positions_direct[indices, :2], dtype=float), 1.0)))
    return layers


def _match_subsurface_hollow(
    hollow_uv: np.ndarray,
    lattice: np.ndarray,
    lower_layers: Sequence[tuple[float, np.ndarray]],
    match_tolerance: float,
) -> str:
    if len(lower_layers) >= 1 and lower_layers[0][1].size > 0:
        distances_first = [
            _minimum_image_inplane_distance(hollow_uv, candidate_uv, lattice)
            for candidate_uv in lower_layers[0][1]
        ]
        if distances_first and min(distances_first) <= match_tolerance:
            return "hcp_hollow"

    if len(lower_layers) >= 2 and lower_layers[1][1].size > 0:
        distances_second = [
            _minimum_image_inplane_distance(hollow_uv, candidate_uv, lattice)
            for candidate_uv in lower_layers[1][1]
        ]
        if distances_second and min(distances_second) <= match_tolerance:
            return "fcc_hollow"

    return "hollow"


def _site_from_uv(site_type: str, uv: np.ndarray, lattice: np.ndarray, plane_projection: float, normal: np.ndarray) -> AdsorptionSite:
    cartesian = _inplane_cartesian_from_uv(uv, lattice) + float(plane_projection) * np.asarray(normal, dtype=float)
    direct = io_mod.wrap_direct(io_mod.cartesian_to_direct(cartesian.reshape(1, 3), lattice))[0]
    return AdsorptionSite(
        site_type=str(site_type),
        direct=(float(direct[0]), float(direct[1]), float(direct[2])),
        cartesian=(float(cartesian[0]), float(cartesian[1]), float(cartesian[2])),
    )


def _site_report_to_dict(run: SiteAnalysisRun) -> dict[str, object]:
    sites_by_type: Dict[str, list[dict[str, object]]] = {}
    for site in run.sites:
        sites_by_type.setdefault(site.site_type, []).append(
            {
                "direct": [float(value) for value in site.direct],
                "cartesian": [float(value) for value in site.cartesian],
            }
        )
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
    layer_tolerance: float = 0.35,
    neighbour_tolerance: float = 0.15,
    hollow_match_tolerance: float | None = None,
    output_path: str | None = None,
) -> SiteAnalysisRun:
    if surface_side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'")

    if isinstance(structure_or_path, str):
        source_poscar = str(Path(structure_or_path).resolve())
        structure = io_mod.read_poscar(structure_or_path)
    else:
        source_poscar = None
        structure = structure_or_path

    lattice = np.asarray(structure.lattice, dtype=float)
    normal = _surface_normal(lattice)
    projections = np.asarray(structure.positions_cartesian, dtype=float) @ normal
    detected_layers = _projection_layer_points(structure.positions_direct, projections, surface_side, float(layer_tolerance), max_layers=3)
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
    triangular_hollows_uv = _find_triangular_hollows(top_points_uv, lattice, neighbour_cutoff)
    fourfold_hollows_uv = _find_fourfold_hollows(top_points_uv, lattice, neighbour_cutoff)

    lower_layers = detected_layers[1:]
    sites: list[AdsorptionSite] = []
    for uv in top_sites_uv:
        sites.append(_site_from_uv("top", uv, lattice, top_projection, normal))
    for uv in bridge_sites_uv:
        sites.append(_site_from_uv("bridge", uv, lattice, top_projection, normal))
    for uv in triangular_hollows_uv:
        site_type = _match_subsurface_hollow(uv, lattice, lower_layers, match_tolerance)
        sites.append(_site_from_uv(site_type, uv, lattice, top_projection, normal))
    for uv in fourfold_hollows_uv:
        sites.append(_site_from_uv("fourfold_hollow", uv, lattice, top_projection, normal))

    site_counts: Dict[str, int] = {}
    for site in sites:
        site_counts[site.site_type] = site_counts.get(site.site_type, 0) + 1

    run = SiteAnalysisRun(
        output_path=None,
        source_poscar=source_poscar,
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
    structure = io_mod.read_poscar(bulk_poscar)
    if not _is_orthogonal_lattice(structure.lattice):
        raise ValueError(
            "the current surface builder expects a conventional orthogonal bulk cell "
            "(for example cubic, tetragonal, or orthorhombic)."
        )

    in_plane_a, in_plane_b, normal = _choose_in_plane_vectors(miller)
    oriented = _structure_from_transform(structure, np.array([in_plane_a, in_plane_b, normal], dtype=int))
    oriented = _shift_boundary_into_interlayer_gap(oriented)
    layered = io_mod.repeat_structure_along_c(oriented, int(layers))

    resolved_repeat_a, resolved_repeat_b = _resolve_inplane_repeats(
        oriented,
        int(repeat_a),
        int(repeat_b),
        min_length_a,
        min_length_b,
    )
    repeated = _repeat_structure_inplane(layered, resolved_repeat_a, resolved_repeat_b)
    scaled, applied_matrix = _apply_inplane_supercell_matrix(repeated, supercell_matrix)
    surfaced = _add_vacuum_along_c(scaled, float(vacuum))

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            DEFAULT_OUTPUT_DIR
            / f"surface_{Path(bulk_poscar).stem}_{int(miller[0])}{int(miller[1])}{int(miller[2])}_layers{int(layers)}.vasp"
        )
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        surfaced.lattice,
        surfaced.positions_direct,
        surfaced.counts,
        surfaced.species,
        comment=(
            "Generated by CELLSTINE surface stage | "
            f"Miller ({int(miller[0])} {int(miller[1])} {int(miller[2])}) | Made by Sarwin Chandran"
        ),
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=surfaced.selective_flags,
    )

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

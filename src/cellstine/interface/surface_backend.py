"""Surface-slab builder and adsorption-site analysis for substrate POSCAR inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..io import native as io_mod

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


@dataclass(frozen=True)
class PrimitiveSurfaceAnalysis:
    miller: tuple[int, int, int]
    centering: str
    probe_layers: int
    atoms_per_layer: tuple[int, ...]
    stacking_sequence: str
    stacking_period: str
    inplane_angle_deg: float
    lattice: np.ndarray


@dataclass(frozen=True)
class SurfaceStructureBuild:
    structure: io_mod.PoscarData
    repeat_a: int
    repeat_b: int
    supercell_matrix: tuple[int, int, int, int] | None


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


def _expanded_species(structure: io_mod.PoscarData) -> list[str]:
    expanded: list[str] = []
    for symbol, count in zip(structure.species, structure.counts):
        expanded.extend([str(symbol)] * int(count))
    if len(expanded) < structure.natoms:
        expanded.extend(["X"] * (structure.natoms - len(expanded)))
    return expanded[: structure.natoms]


def _translation_maps_structure(structure: io_mod.PoscarData, translation: Sequence[float], tolerance: float = 1e-5) -> bool:
    positions = np.mod(np.asarray(structure.positions_direct, dtype=float), 1.0)
    if positions.shape[0] == 0:
        return True
    species = _expanded_species(structure)
    translation_array = np.asarray(translation, dtype=float)
    shifted = np.mod(positions + translation_array, 1.0)
    # Pairwise minimum-image fractional differences between every shifted atom
    # and every original atom, all at once instead of an O(n^2) Python loop.
    diff = shifted[:, None, :] - positions[None, :, :]
    diff -= np.round(diff)
    coincident = np.all(np.abs(diff) <= tolerance, axis=2)
    codes = np.unique(np.asarray(species), return_inverse=True)[1]
    same_species = codes[:, None] == codes[None, :]
    matched = np.any(coincident & same_species, axis=1)
    return bool(np.all(matched))


def _centering_type(structure: io_mod.PoscarData) -> str:
    candidates = {
        "A": np.array([0.0, 0.5, 0.5], dtype=float),
        "B": np.array([0.5, 0.0, 0.5], dtype=float),
        "C": np.array([0.5, 0.5, 0.0], dtype=float),
        "I": np.array([0.5, 0.5, 0.5], dtype=float),
    }
    valid = {key for key, value in candidates.items() if _translation_maps_structure(structure, value)}
    if {"A", "B", "C"}.issubset(valid):
        return "F"
    if "I" in valid:
        return "I"
    for key in ("A", "B", "C"):
        if key in valid:
            return key
    return "P"


def _primitive_translation_lattice(structure: io_mod.PoscarData) -> tuple[np.ndarray, str]:
    centering = _centering_type(structure)
    matrices = {
        "P": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "F": np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        ),
        "I": np.array(
            [
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            dtype=float,
        ),
        "A": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, -0.5, 0.5],
            ],
            dtype=float,
        ),
        "B": np.array(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [-0.5, 0.0, 0.5],
            ],
            dtype=float,
        ),
        "C": np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=float,
        ),
    }
    return matrices[centering] @ np.asarray(structure.lattice, dtype=float), centering


def _reciprocal_normal(lattice: np.ndarray, miller: tuple[int, int, int]) -> np.ndarray:
    h, k, l = (int(value) for value in miller)
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices cannot all be zero")
    reciprocal_rows = np.linalg.inv(np.asarray(lattice, dtype=float)).T
    normal = float(h) * reciprocal_rows[0] + float(k) * reciprocal_rows[1] + float(l) * reciprocal_rows[2]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("could not determine a surface normal for the requested Miller indices")
    return normal / norm


def _primitive_surface_vectors_from_lattice(
    primitive_lattice: np.ndarray,
    normal: np.ndarray,
    *,
    search: int = 4,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_lattice = np.asarray(primitive_lattice, dtype=float)
    normal = np.asarray(normal, dtype=float)
    # Vectorised enumeration of all integer combinations in the search box.
    rng = np.arange(-int(search), int(search) + 1)
    grid = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    # Reproduce ``i*a + j*b + k*c`` term-by-term so the floating point result is
    # bit-identical to the original scalar accumulation.
    vectors = (
        grid[:, 0:1] * primitive_lattice[0]
        + grid[:, 1:2] * primitive_lattice[1]
        + grid[:, 2:3] * primitive_lattice[2]
    )
    lengths_all = np.linalg.norm(vectors, axis=1)
    nonzero = ~np.all(grid == 0.0, axis=1)
    in_plane = np.abs(vectors @ normal) <= tolerance
    keep = nonzero & in_plane & (lengths_all > tolerance)
    candidate_vectors = vectors[keep]
    if candidate_vectors.shape[0] < 2:
        raise ValueError("could not find primitive in-plane surface vectors")

    candidate_lengths = lengths_all[keep]
    rounded = np.round(candidate_vectors, 12)
    sort_order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0], candidate_lengths))
    candidate_vectors = candidate_vectors[sort_order]
    candidate_lengths = candidate_lengths[sort_order]

    # Vectorised pairwise scan over all i < j pairs (row-major triangular order
    # matching the original nested loop, so tie-breaking is identical).
    idx_i, idx_j = np.triu_indices(candidate_vectors.shape[0], k=1)
    first_vectors = candidate_vectors[idx_i]
    second_vectors = candidate_vectors[idx_j]
    cross = np.cross(first_vectors, second_vectors)
    oriented_area = cross @ normal
    area = np.abs(oriented_area)
    valid = area > tolerance
    if not np.any(valid):
        raise ValueError("could not find a non-singular primitive surface cell")
    valid_positions = np.nonzero(valid)[0]
    first_lengths = candidate_lengths[idx_i][valid]
    second_lengths = candidate_lengths[idx_j][valid]
    max_lengths = np.maximum(first_lengths, second_lengths)
    sum_lengths = first_lengths + second_lengths
    valid_area = area[valid]
    # Stable lexicographic selection: primary max length, then sum length, then
    # area; ties keep the earliest pair (stable sort over triangular order).
    winner_local = np.lexsort((valid_area, sum_lengths, max_lengths))[0]
    winner = int(valid_positions[winner_local])
    surface_a = np.array(first_vectors[winner], dtype=float, copy=True)
    surface_b = np.array(second_vectors[winner], dtype=float, copy=True)
    if float(oriented_area[winner]) < 0.0:
        surface_a, surface_b = surface_b, surface_a
    cosine = np.clip(
        float(np.dot(surface_a, surface_b) / max(float(np.linalg.norm(surface_a) * np.linalg.norm(surface_b)), 1e-12)),
        -1.0,
        1.0,
    )
    angle_deg = float(np.degrees(np.arccos(cosine)))
    if angle_deg <= 60.0 + 1e-8:
        surface_b = surface_b - surface_a
    return surface_a, surface_b


def _surface_vector_search_limit(miller: Sequence[int]) -> int:
    return max(5, 2 * max(abs(int(value)) for value in miller) + 3)


def _surface_coordinate_frame(surface_a: np.ndarray, surface_b: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.asarray(surface_a, dtype=float) / float(np.linalg.norm(surface_a))
    y_axis = np.asarray(surface_b, dtype=float) - float(np.dot(surface_b, x_axis)) * x_axis
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-12:
        raise ValueError("surface in-plane vectors are linearly dependent")
    y_axis /= y_norm
    if float(np.dot(np.cross(x_axis, y_axis), normal)) < 0.0:
        y_axis *= -1.0
    basis_2d = np.array(
        [
            [float(np.dot(surface_a, x_axis)), float(np.dot(surface_b, x_axis))],
            [float(np.dot(surface_a, y_axis)), float(np.dot(surface_b, y_axis))],
        ],
        dtype=float,
    )
    return x_axis, y_axis, basis_2d


def _deduplicate_scalar_levels(values: Sequence[float], tolerance: float) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return []
    arr.sort()
    tol = float(tolerance)
    # Greedy clustering identical to scanning the sorted values and keeping a
    # value whenever it is more than ``tolerance`` away from the last kept one.
    # ``searchsorted`` jumps directly to the next kept value, so the loop runs
    # once per surviving level instead of once per (highly degenerate) sample.
    levels: list[float] = [float(arr[0])]
    last = levels[0]
    size = arr.size
    while True:
        nxt = int(np.searchsorted(arr, last + tol, side="right"))
        if nxt >= size:
            break
        last = float(arr[nxt])
        levels.append(last)
    return levels


def _same_surface_uv(uv_a: np.ndarray, uv_b: np.ndarray, lattice: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(uv_a, dtype=float) - np.asarray(uv_b, dtype=float)
    delta -= np.round(delta)
    cartesian = delta[0] * np.asarray(lattice, dtype=float)[0] + delta[1] * np.asarray(lattice, dtype=float)[1]
    return float(np.linalg.norm(cartesian)) <= float(tolerance)


def _group_surface_atoms_by_species(
    atoms: Sequence[tuple[str, np.ndarray, tuple[str, str, str] | None]],
    species_order: Sequence[str],
) -> tuple[np.ndarray, list[int], list[str], list[tuple[str, str, str]] | None]:
    grouped: dict[str, list[np.ndarray]] = {str(symbol): [] for symbol in species_order}
    grouped_flags: dict[str, list[tuple[str, str, str] | None]] = {str(symbol): [] for symbol in species_order}
    order = [str(symbol) for symbol in species_order]
    for symbol, direct, flags in atoms:
        if symbol not in grouped:
            grouped[symbol] = []
            grouped_flags[symbol] = []
            order.append(symbol)
        grouped[symbol].append(np.asarray(direct, dtype=float))
        grouped_flags[symbol].append(flags)

    positions = []
    counts = []
    species = []
    flags_out: list[tuple[str, str, str]] = []
    has_flags = any(flags is not None for _, _, flags in atoms)
    for symbol in order:
        if not grouped[symbol]:
            continue
        species.append(symbol)
        counts.append(len(grouped[symbol]))
        positions.extend(grouped[symbol])
        if has_flags:
            for flags in grouped_flags[symbol]:
                flags_out.append(tuple(flags or ("T", "T", "T")))
    return np.asarray(positions, dtype=float), counts, species, flags_out if has_flags else None


def _integer_shift_grid_3d(limit: int) -> np.ndarray:
    values = np.arange(-int(limit), int(limit) + 1, dtype=float)
    return np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3)


def _integer_shift_grid_2d(limit: int) -> np.ndarray:
    values = np.arange(-int(limit), int(limit) + 1, dtype=float)
    return np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1).reshape(-1, 2)


def _build_native_primitive_surface_cell(
    structure: io_mod.PoscarData,
    miller: tuple[int, int, int],
    *,
    layers: int,
    vacuum: float,
    search_padding: int = 5,
    tolerance: float = 1e-5,
) -> io_mod.PoscarData:
    layers = int(layers)
    if layers < 1:
        raise ValueError("layers must be at least 1")
    if float(vacuum) < 0.0:
        raise ValueError("vacuum must be non-negative")

    bulk_lattice = np.asarray(structure.lattice, dtype=float)
    primitive_lattice, centering = _primitive_translation_lattice(structure)
    normal = _reciprocal_normal(bulk_lattice, miller)
    surface_a, surface_b = _primitive_surface_vectors_from_lattice(
        primitive_lattice,
        normal,
        search=_surface_vector_search_limit(miller),
        tolerance=1e-7,
    )
    if float(np.dot(np.cross(surface_a, surface_b), normal)) < 0.0:
        surface_a, surface_b = surface_b, surface_a
    x_axis, y_axis, basis_2d = _surface_coordinate_frame(surface_a, surface_b, normal)

    species_expanded = _expanded_species(structure)
    flags_expanded = structure.selective_flags or [None] * structure.natoms
    base_direct = np.asarray(structure.positions_direct, dtype=float)
    base_cartesian = io_mod.direct_to_cartesian(base_direct, bulk_lattice)
    base_projections = base_cartesian @ normal
    base_projected_2d = np.column_stack((base_cartesian @ x_axis, base_cartesian @ y_axis))

    image_limit = max(layers + search_padding, 6)
    shifts = _integer_shift_grid_3d(image_limit)
    shift_cartesian = shifts @ bulk_lattice
    shift_projections = shift_cartesian @ normal
    shift_projected_2d = np.column_stack((shift_cartesian @ x_axis, shift_cartesian @ y_axis))
    basis_inverse_transposed = np.linalg.inv(basis_2d).T

    all_projections = (base_projections[:, None] + shift_projections[None, :]).reshape(-1)
    all_levels = _deduplicate_scalar_levels(all_projections, tolerance)
    start_candidates = [index for index, level in enumerate(all_levels) if level >= -tolerance]
    if not start_candidates:
        raise ValueError("could not locate a starting atomic layer for the requested surface")
    start_index = start_candidates[0]
    selected_levels = all_levels[start_index : start_index + layers]
    if len(selected_levels) < layers:
        raise ValueError("could not generate enough atomic layers; try a smaller layer count or check the bulk cell")

    slab_thickness = float(selected_levels[-1] - selected_levels[0]) if len(selected_levels) > 1 else 0.0
    c_length = slab_thickness + float(vacuum)
    if c_length <= 1e-12:
        raise ValueError("surface c axis would have zero length")
    surface_lattice = np.vstack([surface_a, surface_b, normal * c_length])
    if len(selected_levels) > 1:
        interlayer_spacing = float(selected_levels[1] - selected_levels[0])
    elif start_index + 1 < len(all_levels):
        interlayer_spacing = float(all_levels[start_index + 1] - selected_levels[0])
    else:
        interlayer_spacing = 0.0
    lower_vacuum = min(max(interlayer_spacing, 0.0), float(vacuum))

    atoms: list[tuple[str, np.ndarray, tuple[str, str, str] | None]] = []
    uv_tolerance = max(1e-4, float(np.linalg.norm(surface_a) + np.linalg.norm(surface_b)) * 1e-7)
    surface_basis_2d = np.asarray(surface_lattice, dtype=float)[:2]
    for layer_index, level in enumerate(selected_levels):
        layer_unique: list[tuple[str, np.ndarray, tuple[str, str, str] | None]] = []
        # Per-species accumulator of already-accepted uv points for this layer,
        # so the duplicate test against existing points is a single vectorised
        # min-image distance computation rather than a Python inner loop.
        existing_by_species: dict[str, list[np.ndarray]] = {}
        for atom_index in range(structure.natoms):
            matching_shifts = np.abs(base_projections[atom_index] + shift_projections - float(level)) <= tolerance
            if not np.any(matching_shifts):
                continue
            projected_2d = base_projected_2d[atom_index] + shift_projected_2d[matching_shifts]
            uv_values = projected_2d @ basis_inverse_transposed
            uv_values = np.mod(uv_values, 1.0)
            uv_values[np.isclose(uv_values, 1.0, atol=tolerance)] = 0.0
            uv_values[np.isclose(uv_values, 0.0, atol=tolerance)] = 0.0
            symbol = str(species_expanded[atom_index])
            accepted = existing_by_species.setdefault(symbol, [])
            flags_value = None if flags_expanded[atom_index] is None else tuple(flags_expanded[atom_index])
            # The periodic images of a single atom that land on this level all map
            # to the same in-plane fractional point, so collapse exact (to ~1e-9)
            # duplicates first (preserving first-occurrence order). This is far
            # finer than ``uv_tolerance`` and cannot merge genuinely distinct
            # sites, but removes the hundreds of identical images that otherwise
            # drive the dedup loop.
            if uv_values.shape[0] > 1:
                _, first_occurrence = np.unique(np.round(uv_values, 9), axis=0, return_index=True)
                uv_values = uv_values[np.sort(first_occurrence)]
            for uv in uv_values:
                if accepted:
                    existing_array = np.asarray(accepted, dtype=float)
                    delta = uv[None, :] - existing_array
                    delta -= np.round(delta)
                    cartesian = delta @ surface_basis_2d
                    distances = np.sqrt(np.einsum("ij,ij->i", cartesian, cartesian))
                    if np.any(distances <= uv_tolerance):
                        continue
                uv_array = np.asarray(uv, dtype=float)
                accepted.append(uv_array)
                layer_unique.append((symbol, uv_array, flags_value))
        if not layer_unique:
            raise ValueError(f"surface layer {layer_index + 1} did not contain any atoms")
        w_value = (lower_vacuum + (float(level) - float(selected_levels[0]))) / c_length
        for species, uv, flags in layer_unique:
            atoms.append((species, np.array([float(uv[0]), float(uv[1]), float(w_value)], dtype=float), flags))

    positions_direct, counts, species, flags = _group_surface_atoms_by_species(atoms, structure.species or sorted(set(species_expanded)))
    return io_mod.PoscarData(
        comment=(
            f"{structure.comment} | primitive {centering}-lattice surface cell "
            f"({int(miller[0])} {int(miller[1])} {int(miller[2])})"
        ),
        lattice=surface_lattice,
        species=species,
        counts=counts,
        positions_direct=positions_direct,
        positions_cartesian=io_mod.direct_to_cartesian(positions_direct, surface_lattice),
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=flags,
    )


def _structure_from_transform(structure: io_mod.PoscarData, transform: np.ndarray, tolerance: float = 1e-8) -> io_mod.PoscarData:
    if np.allclose(transform[2], np.array([0, 0, 1], dtype=float), atol=tolerance) and np.allclose(
        transform[:2, 2],
        np.zeros(2, dtype=float),
        atol=tolerance,
    ):
        return _structure_from_inplane_transform(structure, transform[:2, :2], tolerance=tolerance)

    lattice_old = np.asarray(structure.lattice, dtype=float)
    lattice_new = transform @ lattice_old
    inverse_new = np.linalg.inv(lattice_new)

    search_pad = max(2, int(np.max(np.abs(transform))) + 1)
    shifts_3d = _integer_shift_grid_3d(search_pad)
    collected_positions: List[np.ndarray] = []
    collected_array = np.empty((0, 3), dtype=float)
    collected_flags: List[Tuple[str, str, str]] | None = [] if structure.selective_flags is not None else None
    expanded_flags = structure.selective_flags or []

    for atom_index, base_direct in enumerate(np.asarray(structure.positions_direct, dtype=float)):
        images = base_direct + shifts_3d
        new_directs = io_mod.direct_to_cartesian(images, lattice_old) @ inverse_new
        inside = np.all((-tolerance <= new_directs) & (new_directs <= 1.0 + tolerance), axis=1)
        wrapped_all = np.mod(new_directs[inside], 1.0)
        if wrapped_all.shape[0] == 0:
            continue
        # Collapse exact (to ~1e-9) duplicate images first, preserving order.
        if wrapped_all.shape[0] > 1:
            _, first_occurrence = np.unique(np.round(wrapped_all, 9), axis=0, return_index=True)
            wrapped_all = wrapped_all[np.sort(first_occurrence)]
        flag_value = tuple(expanded_flags[atom_index]) if collected_flags is not None else None
        for wrapped in wrapped_all:
            if collected_array.shape[0]:
                difference = wrapped[None, :] - collected_array
                difference -= np.round(difference)
                if np.any(np.all(np.abs(difference) <= tolerance, axis=1)):
                    continue
            collected_positions.append(wrapped)
            collected_array = np.vstack((collected_array, wrapped[None, :]))
            if collected_flags is not None:
                collected_flags.append(flag_value)

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


def _structure_from_inplane_transform(
    structure: io_mod.PoscarData,
    matrix_2d: np.ndarray,
    tolerance: float = 1e-8,
) -> io_mod.PoscarData:
    matrix = np.asarray(matrix_2d, dtype=float)
    determinant = int(round(abs(np.linalg.det(matrix))))
    if determinant == 0:
        raise ValueError("in-plane supercell matrix must have a non-zero determinant")

    lattice_old = np.asarray(structure.lattice, dtype=float)
    transform_3d = np.array(
        [
            [matrix[0, 0], matrix[0, 1], 0.0],
            [matrix[1, 0], matrix[1, 1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    lattice_new = transform_3d @ lattice_old
    inverse_2d = np.linalg.inv(matrix)

    search_pad = max(2, int(np.max(np.abs(matrix))) + determinant + 1)
    shifts_2d = _integer_shift_grid_2d(search_pad)
    species_expanded = _expanded_species(structure)
    flags_expanded = structure.selective_flags or [None] * structure.natoms
    collected_atoms: list[tuple[str, np.ndarray, tuple[str, str, str] | None]] = []
    # Per-species accumulator of accepted fractional positions so each new image
    # is tested against the existing ones with a single vectorised min-image
    # comparison rather than a Python scan over all collected atoms.
    existing_by_species: dict[str, list[np.ndarray]] = {}

    for atom_index, base_direct in enumerate(np.asarray(structure.positions_direct, dtype=float)):
        shifted_2d = base_direct[:2] + shifts_2d
        new_2d = shifted_2d @ inverse_2d
        inside = np.all((-tolerance <= new_2d) & (new_2d <= 1.0 + tolerance), axis=1)
        candidates_2d = new_2d[inside]
        if candidates_2d.shape[0] == 0:
            continue
        wrapped_all = np.empty((candidates_2d.shape[0], 3), dtype=float)
        wrapped_all[:, 0] = candidates_2d[:, 0]
        wrapped_all[:, 1] = candidates_2d[:, 1]
        wrapped_all[:, 2] = base_direct[2]
        wrapped_all = np.mod(wrapped_all, 1.0)
        wrapped_all[np.isclose(wrapped_all, 1.0, atol=tolerance)] = 0.0
        wrapped_all[np.isclose(wrapped_all, 0.0, atol=tolerance)] = 0.0
        # Collapse exact (to ~1e-9) duplicate images first, preserving first
        # occurrence: far finer than ``tolerance`` so it cannot merge distinct
        # sites, but removes the many identical images from the shift grid.
        if wrapped_all.shape[0] > 1:
            _, first_occurrence = np.unique(np.round(wrapped_all, 9), axis=0, return_index=True)
            wrapped_all = wrapped_all[np.sort(first_occurrence)]
        symbol = str(species_expanded[atom_index])
        accepted = existing_by_species.setdefault(symbol, [])
        flags_value = None if flags_expanded[atom_index] is None else tuple(flags_expanded[atom_index])
        for wrapped in wrapped_all:
            if accepted:
                existing_array = np.asarray(accepted, dtype=float)
                difference = wrapped[None, :] - existing_array
                difference -= np.round(difference)
                if np.any(np.all(np.abs(difference) <= tolerance, axis=1)):
                    continue
            accepted.append(wrapped)
            collected_atoms.append((symbol, wrapped, flags_value))

    expected_atoms = determinant * structure.natoms
    if len(collected_atoms) != expected_atoms:
        raise ValueError(
            f"in-plane surface transform captured {len(collected_atoms)} atoms, expected {expected_atoms}; "
            "try a simpler matrix or check the input cell"
        )

    positions_direct, counts, species, flags = _group_surface_atoms_by_species(
        collected_atoms,
        structure.species or sorted(set(species_expanded)),
    )
    return io_mod.PoscarData(
        comment=f"{structure.comment} | in-plane transformed surface cell",
        lattice=lattice_new,
        species=species,
        counts=counts,
        positions_direct=positions_direct,
        positions_cartesian=io_mod.direct_to_cartesian(positions_direct, lattice_new),
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=flags,
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


def _surface_normal(lattice: np.ndarray) -> np.ndarray:
    normal = np.cross(np.asarray(lattice, dtype=float)[0], np.asarray(lattice, dtype=float)[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("surface lattice has a zero in-plane area")
    return normal / norm


def _stacking_sequence_for_structure(structure: io_mod.PoscarData, z_tolerance: float = 0.35, xy_tolerance: float = 1e-3) -> tuple[str, tuple[int, ...]]:
    direct = np.asarray(structure.positions_direct, dtype=float)
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    if direct.size == 0:
        return "", tuple()
    normal = _surface_normal(structure.lattice)
    projections = cartesian @ normal
    order = np.argsort(projections)
    groups: list[list[int]] = []
    current = [int(order[0])]
    last_projection = float(projections[order[0]])
    for atom_index in order[1:]:
        projection = float(projections[atom_index])
        if abs(projection - last_projection) <= float(z_tolerance):
            current.append(int(atom_index))
        else:
            groups.append(current)
            current = [int(atom_index)]
        last_projection = projection
    groups.append(current)

    signature_to_letter: dict[tuple[tuple[float, float], ...], str] = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sequence = []
    for group in groups:
        points = np.mod(direct[np.asarray(group, dtype=int), :2], 1.0)
        points[np.isclose(points, 1.0, atol=xy_tolerance)] = 0.0
        points[np.isclose(points, 0.0, atol=xy_tolerance)] = 0.0
        signature = tuple(
            sorted(
                (
                    round(float(point[0]) / xy_tolerance) * xy_tolerance,
                    round(float(point[1]) / xy_tolerance) * xy_tolerance,
                )
                for point in points
            )
        )
        if signature not in signature_to_letter:
            signature_to_letter[signature] = letters[len(signature_to_letter) % len(letters)]
        sequence.append(signature_to_letter[signature])
    return "".join(sequence), tuple(len(group) for group in groups)


def _shortest_repeating_prefix(sequence: str) -> str:
    if not sequence:
        return ""
    for size in range(1, len(sequence) + 1):
        prefix = sequence[:size]
        repeats = (prefix * ((len(sequence) // size) + 1))[: len(sequence)]
        if repeats == sequence:
            return prefix
    return sequence


def analyse_primitive_surface(
    bulk_poscar: str,
    *,
    miller: tuple[int, int, int],
    probe_layers: int = 8,
) -> PrimitiveSurfaceAnalysis:
    structure = io_mod.read_poscar(bulk_poscar)
    primitive_lattice, centering = _primitive_translation_lattice(structure)
    normal = _reciprocal_normal(np.asarray(structure.lattice, dtype=float), miller)
    surface_a, surface_b = _primitive_surface_vectors_from_lattice(
        primitive_lattice,
        normal,
        search=_surface_vector_search_limit(miller),
    )
    probe = _build_native_primitive_surface_cell(
        structure,
        miller,
        layers=max(1, int(probe_layers)),
        vacuum=0.0,
    )
    sequence, atoms_per_layer = _stacking_sequence_for_structure(probe)
    cosine = np.clip(
        float(np.dot(surface_a, surface_b) / max(float(np.linalg.norm(surface_a) * np.linalg.norm(surface_b)), 1e-12)),
        -1.0,
        1.0,
    )
    return PrimitiveSurfaceAnalysis(
        miller=(int(miller[0]), int(miller[1]), int(miller[2])),
        centering=centering,
        probe_layers=max(1, int(probe_layers)),
        atoms_per_layer=atoms_per_layer,
        stacking_sequence=sequence,
        stacking_period=_shortest_repeating_prefix(sequence),
        inplane_angle_deg=float(np.degrees(np.arccos(cosine))),
        lattice=np.array(probe.lattice, dtype=float, copy=True),
    )


def build_surface_structure(
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
) -> SurfaceStructureBuild:
    structure = io_mod.read_poscar(bulk_poscar)
    primitive_surface = _build_native_primitive_surface_cell(
        structure,
        miller,
        layers=int(layers),
        vacuum=float(vacuum),
    )
    resolved_repeat_a, resolved_repeat_b = _resolve_inplane_repeats(
        primitive_surface,
        int(repeat_a),
        int(repeat_b),
        min_length_a,
        min_length_b,
    )
    repeated = _repeat_structure_inplane(primitive_surface, resolved_repeat_a, resolved_repeat_b)
    surfaced, applied_matrix = _apply_inplane_supercell_matrix(repeated, supercell_matrix)
    return SurfaceStructureBuild(
        structure=surfaced,
        repeat_a=int(resolved_repeat_a),
        repeat_b=int(resolved_repeat_b),
        supercell_matrix=applied_matrix,
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


def _deduplicate_uv_points(points_uv: Sequence[np.ndarray], lattice: np.ndarray, tolerance: float = 1e-4) -> list[np.ndarray]:
    points = [np.mod(np.asarray(point, dtype=float), 1.0) for point in points_uv]
    if not points:
        return []
    basis_2d = np.asarray(lattice, dtype=float)[:2]
    stacked = np.asarray(points, dtype=float)
    # Collapse exact (to ~1e-9) duplicate points first, preserving first
    # occurrence. This is far finer than ``tolerance`` so it cannot merge
    # genuinely distinct sites, but the candidate lists are dominated by exact
    # repeats (the same site found from many anchors), so it removes the bulk of
    # the work before the greedy minimum-image pass.
    if stacked.shape[0] > 1:
        _, first_occurrence = np.unique(np.round(stacked, 9), axis=0, return_index=True)
        order = np.sort(first_occurrence)
        candidates = stacked[order]
    else:
        candidates = stacked
    kept: list[np.ndarray] = []
    kept_array = np.empty((0, 2), dtype=float)
    for point in candidates:
        if kept_array.shape[0]:
            delta = point[None, :] - kept_array
            delta -= np.round(delta)
            cartesian = _uv_to_cartesian(delta, basis_2d)
            distances = np.linalg.norm(cartesian, axis=1)
            if np.any(distances <= tolerance):
                continue
        kept.append(point)
        kept_array = np.vstack((kept_array, point[None, :]))
    return kept


_PERIODIC_SHIFTS_2D = np.array(
    [[shift_u, shift_v] for shift_u in (-1, 0, 1) for shift_v in (-1, 0, 1)], dtype=float
)


def _expanded_periodic_arrays(points_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, expanded)`` where ``expanded`` holds the nine periodic
    images (shifts of -1/0/1 along each in-plane axis) of every point in
    base-major / shift order.
    """

    points = np.asarray(points_uv, dtype=float)
    expanded = (points[:, None, :] + _PERIODIC_SHIFTS_2D[None, :, :]).reshape(-1, 2)
    return points, expanded


def _uv_to_cartesian(uv_array: np.ndarray, basis_2d: np.ndarray) -> np.ndarray:
    """Batched ``uv -> cartesian`` matching ``_inplane_cartesian_from_uv`` exactly
    (``uv[0] * a + uv[1] * b`` with the same elementwise floating-point ops, so
    threshold comparisons are bit-identical to the original scalar code).
    """

    uv_array = np.asarray(uv_array, dtype=float)
    return uv_array[..., 0:1] * basis_2d[0] + uv_array[..., 1:2] * basis_2d[1]


def _anchor_image_distance_matrix(points: np.ndarray, expanded: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """``(n_points, 9 * n_points)`` Cartesian distances from each point to every
    periodic image, matching ``norm(_inplane_cartesian_from_uv(image - point))``.
    """

    basis_2d = np.asarray(lattice, dtype=float)[:2]
    displacement = expanded[None, :, :] - points[:, None, :]
    cartesian = _uv_to_cartesian(displacement, basis_2d)
    return np.linalg.norm(cartesian, axis=2)


def _nearest_neighbor_distance(points_uv: np.ndarray, lattice: np.ndarray) -> float:
    points, expanded = _expanded_periodic_arrays(points_uv)
    if points.shape[0] == 0:
        raise ValueError("could not determine an in-plane nearest-neighbour distance from the top surface atoms")
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    valid = distances > 1e-8
    if not np.any(valid):
        raise ValueError("could not determine an in-plane nearest-neighbour distance from the top surface atoms")
    return float(distances[valid].min())


def _top_layer_coordination_counts(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[int]:
    points, expanded = _expanded_periodic_arrays(points_uv)
    if points.shape[0] == 0:
        return []
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    return [int(value) for value in within.sum(axis=1)]


def _find_bridge_sites(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    points, expanded = _expanded_periodic_arrays(points_uv)
    if points.shape[0] == 0:
        return []
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    anchor_idx, image_idx = np.nonzero(within)
    if anchor_idx.size == 0:
        return []
    midpoints = np.mod(0.5 * (points[anchor_idx] + expanded[image_idx]), 1.0)
    return _deduplicate_uv_points(list(midpoints), lattice)


def _find_triangular_hollows(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    points, expanded = _expanded_periodic_arrays(points_uv)
    if points.shape[0] == 0:
        return []
    basis_2d = np.asarray(lattice, dtype=float)[:2]
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    hollows: list[np.ndarray] = []
    for anchor_index in range(points.shape[0]):
        anchor_uv = points[anchor_index]
        neighbours = expanded[within[anchor_index]]
        count = neighbours.shape[0]
        if count < 2:
            continue
        first_sel, second_sel = np.triu_indices(count, k=1)
        point_b = neighbours[first_sel]
        point_c = neighbours[second_sel]
        edge_bc = np.linalg.norm(_uv_to_cartesian(point_c - point_b, basis_2d), axis=1)
        vector_ab = _uv_to_cartesian(point_b - anchor_uv, basis_2d)
        vector_ac = _uv_to_cartesian(point_c - anchor_uv, basis_2d)
        area = np.abs(np.cross(vector_ab, vector_ac)[:, 2])
        keep = (edge_bc <= neighbour_cutoff + 1e-12) & (area > 1e-8)
        if not np.any(keep):
            continue
        centroids = np.mod((anchor_uv + point_b[keep] + point_c[keep]) / 3.0, 1.0)
        hollows.extend(list(centroids))
    return _deduplicate_uv_points(hollows, lattice)


def _find_fourfold_hollows(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    points, expanded = _expanded_periodic_arrays(points_uv)
    if points.shape[0] == 0:
        return []
    basis_2d = np.asarray(lattice, dtype=float)[:2]
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    angle_window = (70.0, 110.0)
    hollows: list[np.ndarray] = []
    for anchor_index in range(points.shape[0]):
        anchor_uv = points[anchor_index]
        displacements = expanded[within[anchor_index]] - anchor_uv
        count = displacements.shape[0]
        if count < 2:
            continue
        first_sel, second_sel = np.triu_indices(count, k=1)
        disp_a = displacements[first_sel]
        disp_b = displacements[second_sel]
        vec_a = _uv_to_cartesian(disp_a, basis_2d)
        vec_b = _uv_to_cartesian(disp_b, basis_2d)
        norm_a = np.linalg.norm(vec_a, axis=1)
        norm_b = np.linalg.norm(vec_b, axis=1)
        denominator = np.maximum(norm_a * norm_b, 1e-12)
        cosine = np.clip(np.einsum("ij,ij->i", vec_a, vec_b) / denominator, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine))
        angle_ok = (angle_deg >= angle_window[0]) & (angle_deg <= angle_window[1])
        candidate_positions = np.nonzero(angle_ok)[0]
        if candidate_positions.size == 0:
            continue
        corners = anchor_uv + disp_a[candidate_positions] + disp_b[candidate_positions]
        corner_diff = expanded[None, :, :] - corners[:, None, :]
        corner_cart = _uv_to_cartesian(corner_diff, basis_2d)
        nearest_corner = np.linalg.norm(corner_cart, axis=2).min(axis=1)
        has_fourth = nearest_corner <= 1e-6
        kept_positions = candidate_positions[has_fourth]
        if kept_positions.size == 0:
            continue
        hollow_points = np.mod(
            anchor_uv + 0.5 * (disp_a[kept_positions] + disp_b[kept_positions]), 1.0
        )
        hollows.extend(list(hollow_points))
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
    basis_2d = np.asarray(lattice, dtype=float)[:2]

    def _min_distance(layer_points: np.ndarray) -> float:
        delta = np.asarray(hollow_uv, dtype=float)[None, :] - np.asarray(layer_points, dtype=float)
        delta -= np.round(delta)
        cartesian = _uv_to_cartesian(delta, basis_2d)
        return float(np.linalg.norm(cartesian, axis=1).min())

    if len(lower_layers) >= 1 and lower_layers[0][1].size > 0:
        if _min_distance(lower_layers[0][1]) <= match_tolerance:
            return "hcp_hollow"

    if len(lower_layers) >= 2 and lower_layers[1][1].size > 0:
        if _min_distance(lower_layers[1][1]) <= match_tolerance:
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

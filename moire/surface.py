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


def _periodic_direct_distance(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    delta = np.asarray(a_values, dtype=float) - np.asarray(b_values, dtype=float)
    return delta - np.round(delta)


def _translation_maps_structure(structure: io_mod.PoscarData, translation: Sequence[float], tolerance: float = 1e-5) -> bool:
    positions = np.mod(np.asarray(structure.positions_direct, dtype=float), 1.0)
    species = _expanded_species(structure)
    translation_array = np.asarray(translation, dtype=float)
    for atom_index, position in enumerate(positions):
        shifted = np.mod(position + translation_array, 1.0)
        matched = False
        for candidate_index, candidate in enumerate(positions):
            if species[candidate_index] != species[atom_index]:
                continue
            if np.all(np.abs(_periodic_direct_distance(shifted, candidate)) <= tolerance):
                matched = True
                break
        if not matched:
            return False
    return True


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


def _enumerate_lattice_vectors(primitive_lattice: np.ndarray, search: int) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    vectors: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for i_value in range(-search, search + 1):
        for j_value in range(-search, search + 1):
            for k_value in range(-search, search + 1):
                coeffs = (int(i_value), int(j_value), int(k_value))
                if coeffs == (0, 0, 0):
                    continue
                vector = i_value * primitive_lattice[0] + j_value * primitive_lattice[1] + k_value * primitive_lattice[2]
                vectors.append((coeffs, vector))
    return vectors


def _primitive_surface_vectors_from_lattice(
    primitive_lattice: np.ndarray,
    normal: np.ndarray,
    *,
    search: int = 4,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for _, vector in _enumerate_lattice_vectors(primitive_lattice, search):
        if abs(float(np.dot(vector, normal))) > tolerance:
            continue
        length = float(np.linalg.norm(vector))
        if length <= tolerance:
            continue
        candidates.append(vector)
    if len(candidates) < 2:
        raise ValueError("could not find primitive in-plane surface vectors")

    candidates.sort(key=lambda item: (float(np.linalg.norm(item)), tuple(round(float(value), 12) for value in item.tolist())))
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            cross = np.cross(first, second)
            oriented_area = float(np.dot(cross, normal))
            area = abs(oriented_area)
            if area <= tolerance:
                continue
            first_length = float(np.linalg.norm(first))
            second_length = float(np.linalg.norm(second))
            score = (max(first_length, second_length), first_length + second_length, area)
            if best is None or score < best[:3]:
                first_out = np.array(first, dtype=float, copy=True)
                second_out = np.array(second, dtype=float, copy=True)
                if oriented_area < 0.0:
                    first_out, second_out = second_out, first_out
                best = (score[0], score[1], score[2], first_out, second_out)
    if best is None:
        raise ValueError("could not find a non-singular primitive surface cell")
    surface_a = best[3]
    surface_b = best[4]
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
    levels: list[float] = []
    for value in sorted(float(item) for item in values):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
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
    for layer_index, level in enumerate(selected_levels):
        layer_unique: list[tuple[str, np.ndarray, tuple[str, str, str] | None]] = []
        for atom_index in range(structure.natoms):
            matching_shifts = np.abs(base_projections[atom_index] + shift_projections - float(level)) <= tolerance
            if not np.any(matching_shifts):
                continue
            projected_2d = base_projected_2d[atom_index] + shift_projected_2d[matching_shifts]
            uv_values = projected_2d @ basis_inverse_transposed
            uv_values = np.mod(uv_values, 1.0)
            uv_values[np.isclose(uv_values, 1.0, atol=tolerance)] = 0.0
            uv_values[np.isclose(uv_values, 0.0, atol=tolerance)] = 0.0
            for uv in uv_values:
                duplicate = False
                for existing_species, existing_uv, _ in layer_unique:
                    if existing_species != species_expanded[atom_index]:
                        continue
                    if _same_surface_uv(uv, existing_uv, surface_lattice, uv_tolerance):
                        duplicate = True
                        break
                if duplicate:
                    continue
                layer_unique.append(
                    (
                        str(species_expanded[atom_index]),
                        np.asarray(uv, dtype=float),
                        None if flags_expanded[atom_index] is None else tuple(flags_expanded[atom_index]),
                    )
                )
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

    for atom_index, base_direct in enumerate(np.asarray(structure.positions_direct, dtype=float)):
        shifted_2d = base_direct[:2] + shifts_2d
        new_2d = shifted_2d @ inverse_2d
        inside = np.all((-tolerance <= new_2d) & (new_2d <= 1.0 + tolerance), axis=1)
        for candidate_2d in new_2d[inside]:
            wrapped = np.array([candidate_2d[0], candidate_2d[1], base_direct[2]], dtype=float)
            wrapped = np.mod(wrapped, 1.0)
            wrapped[np.isclose(wrapped, 1.0, atol=tolerance)] = 0.0
            wrapped[np.isclose(wrapped, 0.0, atol=tolerance)] = 0.0
            duplicate = False
            for existing_species, existing_direct, _ in collected_atoms:
                if existing_species != species_expanded[atom_index]:
                    continue
                difference = wrapped - existing_direct
                if np.all(np.abs(difference - np.round(difference)) <= tolerance):
                    duplicate = True
                    break
            if duplicate:
                continue
            collected_atoms.append(
                (
                    str(species_expanded[atom_index]),
                    wrapped,
                    None if flags_expanded[atom_index] is None else tuple(flags_expanded[atom_index]),
                )
            )

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

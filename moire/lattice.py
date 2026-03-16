"""Geometric utilities used by the moire finder and generator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VectorMatch:
    """One nearly coincident in-plane vector pair."""

    layer1_coeffs: Tuple[int, int]
    layer2_coeffs: Tuple[int, int]
    layer1_vector: Tuple[float, float]
    layer2_vector: Tuple[float, float]
    absolute_error: float
    relative_error: float
    relative_length_mismatch: float


@dataclass(frozen=True)
class SupercellCandidate:
    """A commensurate supercell candidate for one twist angle."""

    angle_deg: float
    strain_avg: float
    strain_layer1: float
    strain_layer2: float
    ratio1: int
    ratio2: int
    total_atoms: int
    layer1_vector1: Tuple[int, int]
    layer1_vector2: Tuple[int, int]
    layer2_vector1: Tuple[int, int]
    layer2_vector2: Tuple[int, int]
    eps1: float
    eps2: float
    vector_product: float
    area1: float
    area2: float


def in_plane_lengths_and_angle(lattice: np.ndarray) -> Tuple[float, float, float]:
    """Return (|a|, |b|, gamma_deg) for the in-plane primitive vectors."""

    basis = np.asarray(lattice, dtype=float)[:2, :2]
    vector_a = basis[0]
    vector_b = basis[1]
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    denominator = max(length_a * length_b, 1e-12)
    cosine = np.clip(float(np.dot(vector_a, vector_b) / denominator), -1.0, 1.0)
    gamma_deg = float(np.degrees(np.arccos(cosine)))
    return length_a, length_b, gamma_deg


def infer_rotational_symmetry_angle(lattice: np.ndarray, tolerance: float = 1e-2) -> int:
    """Infer the smallest rotational periodicity angle for a 2D lattice.

    Heuristic classes:
    - hexagonal: a=b and gamma~60 or 120 -> 60 deg
    - square: a=b and gamma~90 -> 90 deg
    - other Bravais classes -> 180 deg
    """

    length_a, length_b, gamma_deg = in_plane_lengths_and_angle(lattice)
    relative_length_delta = abs(length_a - length_b) / max((length_a + length_b) * 0.5, 1e-12)
    equal_lengths = relative_length_delta <= tolerance
    if equal_lengths and (abs(gamma_deg - 60.0) <= 3.0 or abs(gamma_deg - 120.0) <= 3.0):
        return 60
    if equal_lengths and abs(gamma_deg - 90.0) <= 3.0:
        return 90
    return 180


def combined_symmetry_limit(lattice1: np.ndarray, lattice2: np.ndarray) -> Tuple[int, int, int]:
    """Return (sym1, sym2, lcm(sym1, sym2)) for two lattices."""

    symmetry_1 = infer_rotational_symmetry_angle(lattice1)
    symmetry_2 = infer_rotational_symmetry_angle(lattice2)
    return symmetry_1, symmetry_2, int(math.lcm(symmetry_1, symmetry_2))


def rotate_vector(vector: Sequence[float], theta_rad: float) -> np.ndarray:
    """Rotate a 2D vector by theta_rad around the z-axis."""

    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    x_value, y_value = float(vector[0]), float(vector[1])
    return np.array(
        [
            cos_theta * x_value - sin_theta * y_value,
            sin_theta * x_value + cos_theta * y_value,
        ],
        dtype=float,
    )


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """Return the 3D rotation matrix for an in-plane rotation."""

    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotate_lattice(lattice: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a lattice in the x-y plane."""

    return np.asarray(lattice, dtype=float) @ rotation_matrix_z(angle_deg).T


def rotate_cartesian_positions(positions: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate Cartesian coordinates in the x-y plane."""

    return np.asarray(positions, dtype=float) @ rotation_matrix_z(angle_deg).T


def unit_area(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Absolute 2D cross product."""

    return abs(float(v1[0]) * float(v2[1]) - float(v1[1]) * float(v2[0]))


cross_2d = unit_area


def _metric_tensor(a_vec: Sequence[float], b_vec: Sequence[float], c_vec: Sequence[float]) -> np.ndarray:
    a = np.array([float(a_vec[0]), float(a_vec[1]), float(a_vec[2])], dtype=float)
    b = np.array([float(b_vec[0]), float(b_vec[1]), float(b_vec[2])], dtype=float)
    c = np.array([float(c_vec[0]), float(c_vec[1]), 1.0], dtype=float)
    return np.array(
        [
            [np.dot(a, a), np.dot(a, b), np.dot(a, c)],
            [np.dot(b, a), np.dot(b, b), np.dot(b, c)],
            [np.dot(c, a), np.dot(c, b), np.dot(c, c)],
        ],
        dtype=float,
    )


def calculate_strain(a1: Sequence[float], b1: Sequence[float], c1: Sequence[float], a2: Sequence[float], b2: Sequence[float], c2: Sequence[float]) -> float:
    """Legacy-compatible deformation strain metric from CellMatch."""

    metric_tensor1 = _metric_tensor(a1, b1, c1)
    metric_tensor2 = _metric_tensor(a2, b2, c2)

    rt1 = np.linalg.cholesky(metric_tensor1).T
    rt2 = np.linalg.cholesky(metric_tensor2).T

    e_tensor = rt2 @ np.linalg.inv(rt1) - np.eye(3)
    strain_tensor = 0.5 * (e_tensor + e_tensor.T + e_tensor @ e_tensor.T)
    eigenvalues = np.linalg.eigvals(strain_tensor)
    return float(math.sqrt(float(np.sum(np.real(eigenvalues) ** 2))) / 3.0)


def enumerate_integer_coefficients(nindex: int) -> np.ndarray:
    """Return all integer coefficient pairs except the origin."""

    values = np.arange(-nindex, nindex + 1, dtype=int)
    grid_x, grid_y = np.meshgrid(values, values, indexing="ij")
    coeffs = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    return coeffs[np.any(coeffs != 0, axis=1)]


def enumerate_in_plane_vectors(lattice: np.ndarray, nindex: int) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate all in-plane lattice vectors a*v1 + b*v2."""

    coeffs = enumerate_integer_coefficients(nindex)
    basis = np.asarray(lattice, dtype=float)[:2, :2]
    vectors = coeffs @ basis
    return coeffs, vectors


def find_coincident_vector_pairs(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    nindex: int,
    tolerance: float,
    *,
    strain_tolerance: float | None = None,
) -> List[VectorMatch]:
    """Find nearly coincident vectors between two in-plane lattices."""

    coeffs1, vectors1 = enumerate_in_plane_vectors(lattice1, nindex)
    coeffs2, vectors2 = enumerate_in_plane_vectors(lattice2, nindex)

    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)
    deltas = vectors1[:, None, :] - vectors2[None, :, :]
    absolute_errors = np.linalg.norm(deltas, axis=2)
    relative_errors = absolute_errors / np.maximum(norms1[:, None] + norms2[None, :], 1e-12)
    length_mismatch = np.abs(norms1[:, None] - norms2[None, :]) / np.maximum((norms1[:, None] + norms2[None, :]) * 0.5, 1e-12)

    valid_mask = relative_errors <= tolerance
    if strain_tolerance is not None:
        valid_mask &= length_mismatch <= strain_tolerance

    match_rows, match_cols = np.nonzero(valid_mask)
    matches: List[VectorMatch] = []
    for row_index, col_index in zip(match_rows.tolist(), match_cols.tolist()):
        matches.append(
            VectorMatch(
                layer1_coeffs=(int(coeffs1[row_index, 0]), int(coeffs1[row_index, 1])),
                layer2_coeffs=(int(coeffs2[col_index, 0]), int(coeffs2[col_index, 1])),
                layer1_vector=(float(vectors1[row_index, 0]), float(vectors1[row_index, 1])),
                layer2_vector=(float(vectors2[col_index, 0]), float(vectors2[col_index, 1])),
                absolute_error=float(absolute_errors[row_index, col_index]),
                relative_error=float(relative_errors[row_index, col_index]),
                relative_length_mismatch=float(length_mismatch[row_index, col_index]),
            )
        )

    matches.sort(key=lambda item: (item.relative_error, item.relative_length_mismatch, item.absolute_error))
    return matches


def _vector_pair_strains(v1: np.ndarray, v2: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float]:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    norm_g1 = np.linalg.norm(g1)
    norm_g2 = np.linalg.norm(g2)

    strain1 = 0.0
    if norm_v1 > 0.0 and norm_v2 > 0.0:
        strain1 = math.sqrt(((norm_g1 / norm_v1 - 1.0) ** 2 + (norm_g2 / norm_v2 - 1.0) ** 2) / 2.0)

    strain2 = 0.0
    if norm_g1 > 0.0 and norm_g2 > 0.0:
        strain2 = math.sqrt(((norm_v1 / norm_g1 - 1.0) ** 2 + (norm_v2 / norm_g2 - 1.0) ** 2) / 2.0)

    return float(strain1), float(strain2)


def build_supercell_candidates(
    matches: Sequence[VectorMatch],
    rotated_lattice1: np.ndarray,
    lattice2: np.ndarray,
    atom_count1: int,
    atom_count2: int,
    candidate_tolerance: float,
    angle_deg: float,
) -> List[SupercellCandidate]:
    """Build commensurate supercell candidates from matched vectors."""

    base_area1 = unit_area(rotated_lattice1[0, :2], rotated_lattice1[1, :2])
    base_area2 = unit_area(lattice2[0, :2], lattice2[1, :2])
    if base_area1 == 0.0 or base_area2 == 0.0:
        return []

    usable = [match for match in matches if match.relative_error <= candidate_tolerance]
    if len(usable) < 2:
        return []

    candidates: List[SupercellCandidate] = []
    for first_index, first_match in enumerate(usable):
        for second_match in usable[first_index:]:
            v1 = np.array(first_match.layer1_vector, dtype=float)
            v2 = np.array(second_match.layer1_vector, dtype=float)
            g1 = np.array(first_match.layer2_vector, dtype=float)
            g2 = np.array(second_match.layer2_vector, dtype=float)

            area1_signed = float(v1[0] * v2[1] - v1[1] * v2[0])
            area2_signed = float(g1[0] * g2[1] - g1[1] * g2[0])
            if math.isclose(area1_signed, 0.0, abs_tol=1e-12) or math.isclose(area2_signed, 0.0, abs_tol=1e-12):
                continue

            ratio1 = int(round(abs(area1_signed / base_area1)))
            ratio2 = int(round(abs(area2_signed / base_area2)))
            if ratio1 <= 0 or ratio2 <= 0:
                continue

            strain_avg = calculate_strain(
                [v1[0], v1[1], 0.0],
                [v2[0], v2[1], 0.0],
                [0.0, 0.0, 1.0],
                [g1[0], g1[1], 0.0],
                [g2[0], g2[1], 0.0],
                [0.0, 0.0, 1.0],
            )
            strain1, strain2 = _vector_pair_strains(v1, v2, g1, g2)
            candidates.append(
                SupercellCandidate(
                    angle_deg=float(angle_deg),
                    strain_avg=float(strain_avg),
                    strain_layer1=float(strain1),
                    strain_layer2=float(strain2),
                    ratio1=ratio1,
                    ratio2=ratio2,
                    total_atoms=int(round(atom_count1 * ratio1 + atom_count2 * ratio2)),
                    layer1_vector1=first_match.layer1_coeffs,
                    layer1_vector2=second_match.layer1_coeffs,
                    layer2_vector1=first_match.layer2_coeffs,
                    layer2_vector2=second_match.layer2_coeffs,
                    eps1=float(first_match.relative_error),
                    eps2=float(second_match.relative_error),
                    vector_product=float(np.linalg.norm(v1) * np.linalg.norm(v2)),
                    area1=abs(area1_signed),
                    area2=abs(area2_signed),
                )
            )

    candidates.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.vector_product, item.angle_deg))
    return candidates


def deduplicate_candidates(
    candidates: Sequence[SupercellCandidate],
    strain_tolerance: float = 1e-4,
    ratio_tolerance: float = 1e-5,
    angle_tolerance: float = 1e-9,
) -> List[SupercellCandidate]:
    """Collapse near-identical candidates while keeping the shortest cell."""

    ordered = sorted(candidates, key=lambda item: (item.angle_deg, item.strain_avg, item.vector_product, item.total_atoms))
    unique: List[SupercellCandidate] = []

    for index, candidate in enumerate(ordered):
        ratio = candidate.ratio1 / float(candidate.ratio2)
        duplicate = False
        for kept in unique:
            kept_ratio = kept.ratio1 / float(kept.ratio2)
            if (
                abs(candidate.angle_deg - kept.angle_deg) <= angle_tolerance
                and abs(candidate.strain_avg - kept.strain_avg) < strain_tolerance
                and abs(ratio - kept_ratio) < ratio_tolerance
            ):
                duplicate = True
                break
        if duplicate:
            continue

        best = candidate
        scan_index = index + 1
        while scan_index < len(ordered):
            other = ordered[scan_index]
            if other.angle_deg > candidate.angle_deg + angle_tolerance:
                break
            other_ratio = other.ratio1 / float(other.ratio2)
            if (
                abs(other.angle_deg - candidate.angle_deg) <= angle_tolerance
                and abs(other.strain_avg - candidate.strain_avg) < strain_tolerance
                and abs(other_ratio - ratio) < ratio_tolerance
            ):
                current_key = (best.vector_product, best.total_atoms, best.eps1 + best.eps2)
                other_key = (other.vector_product, other.total_atoms, other.eps1 + other.eps2)
                if other_key < current_key:
                    best = other
            scan_index += 1
        unique.append(best)

    unique.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.angle_deg, item.vector_product))
    return unique

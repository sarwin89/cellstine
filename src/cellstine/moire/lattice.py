"""Geometric utilities used by the moire finder and generator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# Hard physical ceiling on the relative length / strain mismatch a commensurate
# cell may carry.  Real crystals do not sustain more than a few percent strain in
# any direction, so any candidate beyond this is unphysical.  Enforcing it also
# keeps the norm-matching band (and hence the candidate count) tight, which
# speeds up the search.  Tolerances supplied by callers are clamped to this.
MAX_PHYSICAL_MISMATCH: float = 0.05

# Sanity ceiling on the symmetric ``strain_avg`` of a reported cell.  A genuine
# commensurate supercell carries at most a few-percent strain; values far above
# this only ever arise from the strain metric inverting an ill-conditioned
# (near-collinear "sliver") basis, which is a numerical artifact rather than a
# real cell.  The ceiling is set very loose (well above any physical strain) so
# it removes only that obvious garbage and never a genuine candidate.
MAX_PHYSICAL_STRAIN: float = 1.0


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
    length_a, length_b, gamma_deg = in_plane_lengths_and_angle(lattice)
    relative_length_delta = abs(length_a - length_b) / max((length_a + length_b) * 0.5, 1e-12)
    equal_lengths = relative_length_delta <= tolerance
    if equal_lengths and (abs(gamma_deg - 60.0) <= 3.0 or abs(gamma_deg - 120.0) <= 3.0):
        return 60
    if equal_lengths and abs(gamma_deg - 90.0) <= 3.0:
        return 90
    return 180


def combined_symmetry_limit(lattice1: np.ndarray, lattice2: np.ndarray) -> Tuple[int, int, int]:
    symmetry_1 = infer_rotational_symmetry_angle(lattice1)
    symmetry_2 = infer_rotational_symmetry_angle(lattice2)
    return symmetry_1, symmetry_2, int(math.lcm(symmetry_1, symmetry_2))


def rotate_vector(vector: Sequence[float], theta_rad: float) -> np.ndarray:
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


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=float,
    )


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ],
        dtype=float,
    )


def yaw_pitch_roll_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    r_z = rotation_matrix_z(yaw_deg)
    r_y = rotation_matrix_y(pitch_deg)
    r_x = rotation_matrix_x(roll_deg)
    return r_x @ r_y @ r_z


def rotate_lattice(lattice: np.ndarray, angle_deg: float) -> np.ndarray:
    return np.asarray(lattice, dtype=float) @ rotation_matrix_z(angle_deg).T


def rotate_cartesian_positions(positions: np.ndarray, angle_deg: float) -> np.ndarray:
    return np.asarray(positions, dtype=float) @ rotation_matrix_z(angle_deg).T


def unit_area(v1: Sequence[float], v2: Sequence[float]) -> float:
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
    metric_tensor1 = _metric_tensor(a1, b1, c1)
    metric_tensor2 = _metric_tensor(a2, b2, c2)

    rt1 = np.linalg.cholesky(metric_tensor1).T
    rt2 = np.linalg.cholesky(metric_tensor2).T

    e_tensor = rt2 @ np.linalg.inv(rt1) - np.eye(3)
    strain_tensor = 0.5 * (e_tensor + e_tensor.T + e_tensor @ e_tensor.T)
    # ``strain_tensor`` is symmetric, so the sum of squared eigenvalues equals
    # its squared Frobenius norm (sum of squared entries).  Using that identity
    # avoids a full eigendecomposition entirely -- it is faster and numerically
    # more accurate than reducing the complex output of ``np.linalg.eigvals``.
    return float(math.sqrt(float(np.sum(strain_tensor * strain_tensor))) / 3.0)


def calculate_strain_batch(layer1_vectors: np.ndarray, layer2_vectors: np.ndarray) -> np.ndarray:
    # Stack the two lattice vectors of each cell as the columns of a 2x2 basis.
    # ``layer_vectors[:, k, :]`` is vector ``k``; transposing the last two axes
    # places component ``c`` of vector ``k`` at ``[:, c, k]`` (a cheap view).
    basis1 = np.swapaxes(layer1_vectors, 1, 2)
    basis2 = np.swapaxes(layer2_vectors, 1, 2)

    inv_basis1 = np.linalg.inv(basis1)
    identity = np.broadcast_to(np.eye(2, dtype=float), basis1.shape)
    e_tensor = basis2 @ inv_basis1 - identity
    strain_tensor = 0.5 * (e_tensor + np.swapaxes(e_tensor, 1, 2) + e_tensor @ np.swapaxes(e_tensor, 1, 2))
    # Each ``strain_tensor`` slice is symmetric, so sum of squared eigenvalues =
    # squared Frobenius norm.  This replaces the per-cell complex eigensolve with
    # a single reduction over the last two axes (~2x faster, more accurate).
    return np.sqrt(np.sum(strain_tensor * strain_tensor, axis=(1, 2))) / 3.0


def enumerate_integer_coefficients(nindex: int) -> np.ndarray:
    values = np.arange(-nindex, nindex + 1, dtype=int)
    grid_x, grid_y = np.meshgrid(values, values, indexing="ij")
    coeffs = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    return coeffs[np.any(coeffs != 0, axis=1)]


def enumerate_in_plane_vectors(lattice: np.ndarray, nindex: int) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = enumerate_integer_coefficients(nindex)
    basis = np.asarray(lattice, dtype=float)[:2, :2]
    vectors = coeffs @ basis
    return coeffs, vectors


def norm_match_candidate_pairs(
    norms1: np.ndarray,
    norms2: np.ndarray,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return index pairs ``(rows, cols)`` whose norms are nearly equal.

    This is the central pruning primitive for commensuration: two lattice
    vectors can only coincide (under any rotation) if their lengths match, and
    length is rotation invariant.  Instead of forming the dense ``N1 x N2``
    mismatch matrix, the candidate list is produced in ``O(N log N + K)`` time
    by sorting ``norms2`` once and binary-searching the admissible band for
    every entry of ``norms1`` (``K`` is the number of surviving pairs).

    The returned set is a *superset* of the pairs satisfying either
    ``|n1 - n2| <= abs_tol`` or ``|n1 - n2| <= rel_tol * (n1 + n2) / 2``; the
    caller is expected to re-apply the exact predicate it needs.  The band
    ``abs_tol + 2 * rel_tol * n1`` is a safe envelope for any ``rel_tol <= 1``.
    """
    norms1 = np.asarray(norms1, dtype=float)
    norms2 = np.asarray(norms2, dtype=float)
    size1 = int(norms1.shape[0])
    size2 = int(norms2.shape[0])
    empty = np.empty(0, dtype=np.intp)
    if size1 == 0 or size2 == 0:
        return empty, empty

    order2 = np.argsort(norms2, kind="stable")
    sorted2 = norms2[order2]

    band = float(abs_tol) + 2.0 * float(rel_tol) * norms1
    lower = np.searchsorted(sorted2, norms1 - band, side="left")
    upper = np.searchsorted(sorted2, norms1 + band, side="right")
    counts = upper - lower
    total = int(counts.sum())
    if total == 0:
        return empty, empty

    rows = np.repeat(np.arange(size1, dtype=np.intp), counts)
    segment_start = np.repeat(lower, counts)
    within = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(counts) - counts, counts)
    cols = order2[segment_start + within]

    # Emit pairs in row-major (row, col) order so that downstream stable sorts
    # break ties exactly as the original dense ``np.nonzero(mask)`` scan did.
    reorder = np.lexsort((cols, rows))
    return rows[reorder], cols[reorder]


def find_coincident_vector_pairs(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    nindex: int | tuple[int, int],
    tolerance: float,
    *,
    strain_tolerance: float | None = None,
    precomputed_candidates: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    precomputed_enum: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> List[VectorMatch]:
    if strain_tolerance is not None:
        strain_tolerance = min(float(strain_tolerance), MAX_PHYSICAL_MISMATCH)
    if isinstance(nindex, tuple):
        n1, n2 = nindex
    else:
        n1 = n2 = int(nindex)
    if precomputed_enum is not None:
        # The integer coefficient grids and the (unrotated) layer-2 vectors are
        # the same for every twist angle, so the caller can share them across
        # the whole angle sweep.  Only the layer-1 vectors depend on the angle
        # (via the rotated basis), and they are recovered exactly here.
        coeffs1, coeffs2, vectors2 = precomputed_enum
        vectors1 = coeffs1 @ np.asarray(lattice1, dtype=float)[:2, :2]
    else:
        coeffs1, vectors1 = enumerate_in_plane_vectors(lattice1, n1)
        coeffs2, vectors2 = enumerate_in_plane_vectors(lattice2, n2)

    if precomputed_candidates is None:
        norms1 = np.linalg.norm(vectors1, axis=1)
        norms2 = np.linalg.norm(vectors2, axis=1)
        # A coincident pair (relative_error <= tolerance) necessarily satisfies
        # |n1 - n2| / (n1 + n2) <= tolerance, i.e. a relative length mismatch of
        # at most 2 * tolerance; restrict the search to that band first.
        match_rows, match_cols = norm_match_candidate_pairs(
            norms1, norms2, abs_tol=0.0, rel_tol=2.0 * float(tolerance)
        )
        precomputed_length_mismatch = None
    else:
        match_rows, match_cols, norms1, norms2, lm_values = precomputed_candidates
        precomputed_length_mismatch = np.asarray(lm_values, dtype=float)

    if len(match_rows) == 0:
        return []

    # Evaluate the exact predicate only on the candidate pairs.
    v1_cand = vectors1[match_rows]
    v2_cand = vectors2[match_cols]
    norms1_cand = norms1[match_rows]
    norms2_cand = norms2[match_cols]

    deltas = v1_cand - v2_cand
    absolute_errors = np.linalg.norm(deltas, axis=1)
    sum_norms = norms1_cand + norms2_cand
    relative_errors = absolute_errors / np.maximum(sum_norms, 1e-12)
    if precomputed_length_mismatch is not None and precomputed_length_mismatch.shape[0] == match_rows.shape[0]:
        length_mismatch_cand = precomputed_length_mismatch
    else:
        length_mismatch_cand = np.abs(norms1_cand - norms2_cand) / np.maximum(sum_norms * 0.5, 1e-12)

    valid_mask = relative_errors <= tolerance
    if strain_tolerance is not None:
        valid_mask &= length_mismatch_cand <= strain_tolerance

    valid_indices = np.nonzero(valid_mask)[0]
    matches: List[VectorMatch] = []
    for idx in valid_indices.tolist():
        r = int(match_rows[idx])
        c = int(match_cols[idx])
        matches.append(
            VectorMatch(
                layer1_coeffs=(int(coeffs1[r, 0]), int(coeffs1[r, 1])),
                layer2_coeffs=(int(coeffs2[c, 0]), int(coeffs2[c, 1])),
                layer1_vector=(float(vectors1[r, 0]), float(vectors1[r, 1])),
                layer2_vector=(float(vectors2[c, 0]), float(vectors2[c, 1])),
                absolute_error=float(absolute_errors[idx]),
                relative_error=float(relative_errors[idx]),
                relative_length_mismatch=float(length_mismatch_cand[idx]),
            )
        )

    matches.sort(key=lambda item: (item.relative_error, item.relative_length_mismatch, item.absolute_error))
    return matches


def _extended_gcd_columns(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised extended Euclid: return ``(g, s, t)`` with ``s*first + t*second == g``
    and ``g >= 0``, elementwise over the integer arrays ``first`` and ``second``.
    """
    old_r = first.astype(np.int64).copy()
    r = second.astype(np.int64).copy()
    old_s = np.ones_like(old_r)
    s = np.zeros_like(old_r)
    old_t = np.zeros_like(old_r)
    t = np.ones_like(old_r)
    # Euclid converges in O(log) steps; the integer indices here are bounded by
    # ``nindex`` so a handful of iterations suffice, but loop until the whole
    # remainder array is zero to stay correct for any input range.
    while np.any(r != 0):
        active = r != 0
        quotient = np.zeros_like(old_r)
        quotient[active] = old_r[active] // r[active]
        old_r, r = (np.where(active, r, old_r), np.where(active, old_r - quotient * r, r))
        old_s, s = (np.where(active, s, old_s), np.where(active, old_s - quotient * s, s))
        old_t, t = (np.where(active, t, old_t), np.where(active, old_t - quotient * t, t))
    negative = old_r < 0
    g = np.where(negative, -old_r, old_r)
    s = np.where(negative, -old_s, old_s)
    t = np.where(negative, -old_t, old_t)
    return g, s, t


def _row_hnf_2x2(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-style Hermite Normal Form of the integer matrices ``[[a, b], [c, d]]``.

    Returns ``(h11, h12, h22, u00, u01, u10, u11)`` where ``H = [[h11, h12], [0, h22]]``
    is the unique canonical basis of the lattice spanned by the *rows* of the input
    (``h11 > 0``, ``h22 > 0``, ``0 <= h12 < h22``) and ``U = [[u00, u01], [u10, u11]]``
    is the unimodular matrix with ``U @ M == H``.  Two integer matrices share the
    same ``H`` exactly when they span the same row lattice, i.e. differ by a left
    unimodular factor; this is the canonical key used to collapse the many integer
    bases that describe one and the same supercell.
    """
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    c = c.astype(np.int64)
    d = d.astype(np.int64)
    g, s, t = _extended_gcd_columns(a, c)
    # The input is non-singular, so the first column is never all-zero and g > 0.
    h12 = s * b + t * d
    h22 = (b * c - a * d) // g
    u00 = s.copy()
    u01 = t.copy()
    u10 = c // g
    u11 = -(a // g)
    negative = h22 < 0
    h22 = np.where(negative, -h22, h22)
    u10 = np.where(negative, -u10, u10)
    u11 = np.where(negative, -u11, u11)
    reduction = np.floor_divide(h12, h22)
    h12 = h12 - reduction * h22
    u00 = u00 - reduction * u10
    u01 = u01 - reduction * u11
    return g, h12, h22, u00, u01, u10, u11


def _canonical_group_ids(
    layer1_coeffs: np.ndarray,
    layer2_coeffs: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
) -> np.ndarray:
    """Integer group id per candidate pair that is constant exactly on the pairs
    describing the same physical supercell.

    The layer-1 integer basis ``[[i1], [i2]]`` is reduced to its row-Hermite
    Normal Form ``H = U @ M1`` and the *same* unimodular ``U`` is applied to the
    layer-2 basis; the concatenated ``(H, U @ M2)`` is the canonical key.  Two
    pairs share a key iff they span the same pair of (coincident) sublattices,
    i.e. they are the same supercell expressed in a different integer basis --
    which leaves the twist angle, strain, area ratio and atom count unchanged.
    """
    m1_a = layer1_coeffs[first_indices, 0]
    m1_b = layer1_coeffs[first_indices, 1]
    m1_c = layer1_coeffs[second_indices, 0]
    m1_d = layer1_coeffs[second_indices, 1]
    h11, h12, h22, u00, u01, u10, u11 = _row_hnf_2x2(m1_a, m1_b, m1_c, m1_d)
    n1_a = layer2_coeffs[first_indices, 0]
    n1_b = layer2_coeffs[first_indices, 1]
    n2_a = layer2_coeffs[second_indices, 0]
    n2_b = layer2_coeffs[second_indices, 1]
    keys = np.stack(
        (
            h11,
            h12,
            h22,
            u00 * n1_a + u01 * n2_a,
            u00 * n1_b + u01 * n2_b,
            u10 * n1_a + u11 * n2_a,
            u10 * n1_b + u11 * n2_b,
        ),
        axis=1,
    )
    _, group_id = np.unique(keys, axis=0, return_inverse=True)
    return np.asarray(group_id).reshape(-1)


def _area_ratio_filter(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    layer1_vectors: np.ndarray,
    layer2_vectors: np.ndarray,
    base_area1: float,
    base_area2: float,
    atom_count1: int,
    atom_count2: int,
    min_atoms: int | None,
    max_atoms: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the zero-area and out-of-atom-window pairs from ``(first, second)``.

    Mirrors the filtering done by the full-enumeration path exactly, returning the
    surviving index pair arrays (which span a non-degenerate cell whose atom count
    lies inside ``[min_atoms, max_atoms]``).
    """
    v1 = layer1_vectors[first_indices]
    v2 = layer1_vectors[second_indices]
    g1 = layer2_vectors[first_indices]
    g2 = layer2_vectors[second_indices]
    area1_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    area2_signed = g1[:, 0] * g2[:, 1] - g1[:, 1] * g2[:, 0]
    valid = (~np.isclose(area1_signed, 0.0, atol=1e-12)) & (~np.isclose(area2_signed, 0.0, atol=1e-12))
    first_indices = first_indices[valid]
    second_indices = second_indices[valid]
    if first_indices.size == 0:
        return first_indices, second_indices
    area1_signed = area1_signed[valid]
    area2_signed = area2_signed[valid]
    ratio1 = np.rint(np.abs(area1_signed / base_area1)).astype(int)
    ratio2 = np.rint(np.abs(area2_signed / base_area2)).astype(int)
    keep = (ratio1 > 0) & (ratio2 > 0)
    if min_atoms is not None or max_atoms is not None:
        total_atoms_all = atom_count1 * ratio1 + atom_count2 * ratio2
        if min_atoms is not None:
            keep &= total_atoms_all >= int(min_atoms)
        if max_atoms is not None:
            keep &= total_atoms_all <= int(max_atoms)
    return first_indices[keep], second_indices[keep]


def _tied_min_keep(
    group_id: np.ndarray, vp: np.ndarray, atoms: np.ndarray, eps: np.ndarray
) -> np.ndarray:
    """Indices of the members tying for the smallest ``(vp, atoms, eps)`` per group."""
    order = np.lexsort((eps, atoms, vp, group_id))
    sorted_groups = group_id[order]
    first_in_group = np.empty(sorted_groups.shape[0], dtype=bool)
    first_in_group[0] = True
    np.not_equal(sorted_groups[1:], sorted_groups[:-1], out=first_in_group[1:])
    group_first = np.maximum.accumulate(
        np.where(first_in_group, np.arange(sorted_groups.shape[0]), 0)
    )
    sorted_vp = vp[order]
    sorted_atoms = atoms[order]
    sorted_eps = eps[order]
    tied = (
        (sorted_vp == sorted_vp[group_first])
        & (sorted_atoms == sorted_atoms[group_first])
        & (sorted_eps == sorted_eps[group_first])
    )
    return np.sort(order[tied])


def _filter_and_reduce_pair_block(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    layer1_vectors: np.ndarray,
    layer2_vectors: np.ndarray,
    layer1_coeffs: np.ndarray,
    layer2_coeffs: np.ndarray,
    relative_errors: np.ndarray,
    base_area1: float,
    base_area2: float,
    atom_count1: int,
    atom_count2: int,
    min_atoms: int | None,
    max_atoms: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area/atom-filter a block of candidate pairs, then collapse each canonical
    sublattice group to the members tying for the deduplication key."""
    first_indices, second_indices = _area_ratio_filter(
        first_indices, second_indices, layer1_vectors, layer2_vectors,
        base_area1, base_area2, atom_count1, atom_count2, min_atoms, max_atoms,
    )
    if first_indices.size <= 1:
        return first_indices, second_indices
    v1 = layer1_vectors[first_indices]
    v2 = layer1_vectors[second_indices]
    g1 = layer2_vectors[first_indices]
    g2 = layer2_vectors[second_indices]
    vp = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    eps = relative_errors[first_indices] + relative_errors[second_indices]
    ratio1 = np.rint(np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / base_area1).astype(int)
    ratio2 = np.rint(np.abs(g1[:, 0] * g2[:, 1] - g1[:, 1] * g2[:, 0]) / base_area2).astype(int)
    atoms = atom_count1 * ratio1 + atom_count2 * ratio2
    group_id = _canonical_group_ids(layer1_coeffs, layer2_coeffs, first_indices, second_indices)
    keep = _tied_min_keep(group_id, vp, atoms, eps)
    return first_indices[keep], second_indices[keep]


def _all_pair_survivors(
    layer1_vectors: np.ndarray,
    layer2_vectors: np.ndarray,
    base_area1: float,
    base_area2: float,
    atom_count1: int,
    atom_count2: int,
    min_atoms: int | None,
    max_atoms: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full strict-upper-triangle enumeration with area/atom filtering (no
    canonical collapse) -- the exact survivor set the original code produced."""
    n = layer1_vectors.shape[0]
    first_indices, second_indices = np.triu_indices(n, k=1)
    if first_indices.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    return _area_ratio_filter(
        first_indices, second_indices, layer1_vectors, layer2_vectors,
        base_area1, base_area2, atom_count1, atom_count2, min_atoms, max_atoms,
    )


def _canonical_pair_survivors(
    layer1_vectors: np.ndarray,
    layer2_vectors: np.ndarray,
    layer1_coeffs: np.ndarray,
    layer2_coeffs: np.ndarray,
    relative_errors: np.ndarray,
    base_area1: float,
    base_area2: float,
    atom_count1: int,
    atom_count2: int,
    min_atoms: int | None,
    max_atoms: int | None,
    max_pairs_per_chunk: int = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical-sublattice survivors, computed in row-chunks to bound memory.

    The strict-upper-triangle pairing is generated and reduced one block of rows
    at a time, so the O(M^2) temporaries never materialise in full.  Each block
    is collapsed to its per-group deduplication-key ties; the surviving blocks are
    concatenated and reduced once more to merge groups that straddle chunk
    boundaries.  The result is exactly the set the single-shot reduction produces.
    """
    n = layer1_vectors.shape[0]
    empty = np.empty(0, dtype=np.intp)
    if n < 2:
        return empty, empty
    rows_per_chunk = max(1, int(max_pairs_per_chunk // n))
    first_parts: List[np.ndarray] = []
    second_parts: List[np.ndarray] = []
    for start in range(0, n - 1, rows_per_chunk):
        end = min(start + rows_per_chunk, n - 1)
        rows = np.arange(start, end, dtype=np.intp)
        counts = (n - 1) - rows
        total = int(counts.sum())
        if total == 0:
            continue
        block_first = np.repeat(rows, counts)
        segment_start = np.repeat(np.cumsum(counts) - counts, counts)
        block_second = (np.arange(total, dtype=np.intp) - segment_start) + np.repeat(rows + 1, counts)
        block_first, block_second = _filter_and_reduce_pair_block(
            block_first, block_second, layer1_vectors, layer2_vectors,
            layer1_coeffs, layer2_coeffs, relative_errors,
            base_area1, base_area2, atom_count1, atom_count2, min_atoms, max_atoms,
        )
        if block_first.size:
            first_parts.append(block_first)
            second_parts.append(block_second)
    if not first_parts:
        return empty, empty
    if len(first_parts) == 1:
        # A single chunk holds the whole upper triangle, so no canonical group can
        # straddle a chunk boundary; the extra merge reduction below would just
        # repeat the (expensive) Hermite grouping for no effect.  This is the
        # common case once ``max_pair_matches`` bounds the per-angle vector count.
        return first_parts[0], second_parts[0]
    first_indices = np.concatenate(first_parts)
    second_indices = np.concatenate(second_parts)
    # Merge groups split across chunk boundaries with one more reduction.
    return _filter_and_reduce_pair_block(
        first_indices, second_indices, layer1_vectors, layer2_vectors,
        layer1_coeffs, layer2_coeffs, relative_errors,
        base_area1, base_area2, atom_count1, atom_count2, min_atoms, max_atoms,
    )


def _strain_ratio_dedup_keep(
    strain_avg: np.ndarray,
    ratio1: np.ndarray,
    ratio2: np.ndarray,
    total_atoms: np.ndarray,
    vector_product: np.ndarray,
    eps_sum: np.ndarray,
    strain_tol: float,
    ratio_tol: float,
) -> np.ndarray:
    """Per-angle duplicate cull, vectorised.

    Two cells at one twist angle are duplicates when they share (within tolerance)
    both the symmetric strain and the area ratio ``ratio1 / ratio2``; the
    representative kept is the one with the smallest ``(vector_product,
    total_atoms, eps_sum)``.  This is the per-angle equivalent of
    ``deduplicate_candidates`` and -- crucially -- it is what suppresses the
    spurious thin "sliver" cells: a sliver and the genuine compact supercell that
    spans the same superlattice share a strain and ratio, and the genuine cell
    (shorter vectors) has the smaller vector product, so it wins the tie.
    """
    n = strain_avg.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.intp)
    ratio = ratio1.astype(float) / np.maximum(ratio2.astype(float), 1.0)
    strain_key = np.rint(strain_avg / max(strain_tol, 1e-300)).astype(np.int64)
    ratio_key = np.rint(ratio / max(ratio_tol, 1e-300)).astype(np.int64)
    order = np.lexsort((eps_sum, total_atoms, vector_product, ratio_key, strain_key))
    sk = strain_key[order]
    rk = ratio_key[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.logical_or(sk[1:] != sk[:-1], rk[1:] != rk[:-1], out=first[1:])
    return np.sort(order[first])


def _pareto_frontier_keep(
    total_atoms: np.ndarray,
    strain_avg: np.ndarray,
    vector_product: np.ndarray,
    eps_sum: np.ndarray,
    strain_epsilon: float = 1e-9,
) -> np.ndarray:
    """Indices of the per-angle ``(total_atoms, strain_avg)`` Pareto frontier.

    A cell is kept only when no smaller-or-equal-atom cell already achieves a
    strictly lower strain, i.e. exactly the set ``pareto_cull`` retains for a
    single twist angle.  Ties on ``(total_atoms, strain_avg)`` are resolved in
    favour of the smallest ``(vector_product, eps_sum)`` representative, matching
    the deduplication key used downstream.  Computing this frontier here -- in
    numpy, before any Python ``SupercellCandidate`` objects are built -- collapses
    the per-angle candidate set (often thousands at degenerate angles) to the
    handful of non-dominated cells, which is what makes the search cheap.
    """
    order = np.lexsort((eps_sum, vector_product, strain_avg, total_atoms))
    sorted_strain = strain_avg[order]
    # Running minimum strain over the strictly-earlier (fewer/equal atom) cells.
    running_min_before = np.empty(sorted_strain.shape[0], dtype=float)
    running_min_before[0] = math.inf
    if sorted_strain.shape[0] > 1:
        np.minimum.accumulate(sorted_strain[:-1], out=running_min_before[1:])
    keep_mask = sorted_strain < running_min_before - strain_epsilon
    return np.sort(order[keep_mask])


def build_supercell_candidates(
    matches: Sequence[VectorMatch],
    rotated_lattice1: np.ndarray,
    lattice2: np.ndarray,
    atom_count1: int,
    atom_count2: int,
    candidate_tolerance: float,
    angle_deg: float,
    max_pair_matches: int | None = None,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    canonicalize: bool = False,
    frontier_only: bool = False,
    unique_strain_tol: float = 1e-4,
    unique_ratio_tol: float = 1e-5,
) -> List[SupercellCandidate]:
    base_area1 = unit_area(rotated_lattice1[0, :2], rotated_lattice1[1, :2])
    base_area2 = unit_area(lattice2[0, :2], lattice2[1, :2])
    if base_area1 == 0.0 or base_area2 == 0.0:
        return []

    usable = [match for match in matches if match.relative_error <= candidate_tolerance]
    if len(usable) < 2:
        return []

    # Supercell pairing is O(M^2) in the number of coincident vectors.  At
    # highly degenerate angles M can reach several hundred, so ``max_pair_matches``
    # offers an opt-in speed knob that keeps only the shortest coincident
    # vectors.  It is ``None`` by default (exact, full enumeration); set it to a
    # finite value when scanning for small cells (modest ``max_atoms``), where
    # the dropped long vectors cannot contribute surviving candidates.
    if max_pair_matches is not None and len(usable) > max_pair_matches:
        usable = sorted(
            usable,
            key=lambda match: match.layer1_vector[0] ** 2 + match.layer1_vector[1] ** 2,
        )[:max_pair_matches]

    layer1_coeffs = np.array([match.layer1_coeffs for match in usable], dtype=int)
    layer2_coeffs = np.array([match.layer2_coeffs for match in usable], dtype=int)
    layer1_vectors = np.array([match.layer1_vector for match in usable], dtype=float)
    layer2_vectors = np.array([match.layer2_vector for match in usable], dtype=float)
    relative_errors = np.array([match.relative_error for match in usable], dtype=float)

    # Enumerate the candidate vector pairs (strict upper triangle -- a pair of
    # identical vectors spans no area), keeping only the non-degenerate cells
    # inside the atom window.  When ``canonicalize`` is set, the pairing is done
    # in row-chunks and each canonical sublattice is collapsed to its
    # deduplication-key ties as it is generated, so the O(M^2) temporaries that
    # blow up at perfectly aligned (degenerate) twist angles -- 0/30/60 deg of a
    # homobilayer and the like -- never materialise in full.  Every integer basis
    # dropped here spans a superlattice already represented by a kept basis, so
    # the downstream deduplication returns exactly the same result without ever
    # building (or scanning) the millions of redundant duplicates.
    if canonicalize:
        first_indices, second_indices = _canonical_pair_survivors(
            layer1_vectors, layer2_vectors, layer1_coeffs, layer2_coeffs,
            relative_errors, base_area1, base_area2, atom_count1, atom_count2,
            min_atoms, max_atoms,
        )
    else:
        first_indices, second_indices = _all_pair_survivors(
            layer1_vectors, layer2_vectors, base_area1, base_area2,
            atom_count1, atom_count2, min_atoms, max_atoms,
        )
    if first_indices.size == 0:
        return []

    c1v1 = layer1_coeffs[first_indices]
    c1v2 = layer1_coeffs[second_indices]
    c2v1 = layer2_coeffs[first_indices]
    c2v2 = layer2_coeffs[second_indices]
    eps1 = relative_errors[first_indices]
    eps2 = relative_errors[second_indices]

    # Use the stored (rotated) match vectors for the geometry so the strain is
    # computed on exactly the same floating-point inputs as the full path -- the
    # ``coeffs @ basis`` route differs in the last bits, which the strain metric
    # amplifies wildly on ill-conditioned cells.  The integer coeffs above are
    # only carried through for reporting.
    v1 = layer1_vectors[first_indices]
    v2 = layer1_vectors[second_indices]
    g1 = layer2_vectors[first_indices]
    g2 = layer2_vectors[second_indices]

    area1_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    area2_signed = g1[:, 0] * g2[:, 1] - g1[:, 1] * g2[:, 0]
    ratio1 = np.rint(np.abs(area1_signed / base_area1)).astype(int)
    ratio2 = np.rint(np.abs(area2_signed / base_area2)).astype(int)

    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    norm_g1 = np.linalg.norm(g1, axis=1)
    norm_g2 = np.linalg.norm(g2, axis=1)

    strain1 = np.zeros_like(norm_v1)
    nonzero_layer1 = (norm_v1 > 0.0) & (norm_v2 > 0.0)
    strain1[nonzero_layer1] = np.sqrt(
        ((norm_g1[nonzero_layer1] / norm_v1[nonzero_layer1] - 1.0) ** 2
        + (norm_g2[nonzero_layer1] / norm_v2[nonzero_layer1] - 1.0) ** 2)
        / 2.0
    )

    strain2 = np.zeros_like(norm_g1)
    nonzero_layer2 = (norm_g1 > 0.0) & (norm_g2 > 0.0)
    strain2[nonzero_layer2] = np.sqrt(
        ((norm_v1[nonzero_layer2] / norm_g1[nonzero_layer2] - 1.0) ** 2
        + (norm_v2[nonzero_layer2] / norm_g2[nonzero_layer2] - 1.0) ** 2)
        / 2.0
    )

    layer1_pair_vectors = np.stack((v1, v2), axis=1)
    layer2_pair_vectors = np.stack((g1, g2), axis=1)
    strain_avg = calculate_strain_batch(layer1_pair_vectors, layer2_pair_vectors)
    total_atoms = atom_count1 * ratio1 + atom_count2 * ratio2
    vector_product = norm_v1 * norm_v2

    def _apply_keep(keep: np.ndarray) -> None:
        nonlocal c1v1, c1v2, c2v1, c2v2
        nonlocal ratio1, ratio2, strain1, strain2
        nonlocal strain_avg, total_atoms, vector_product, eps1, eps2
        nonlocal area1_signed, area2_signed
        c1v1 = c1v1[keep]; c1v2 = c1v2[keep]
        c2v1 = c2v1[keep]; c2v2 = c2v2[keep]
        ratio1 = ratio1[keep]; ratio2 = ratio2[keep]
        strain1 = strain1[keep]; strain2 = strain2[keep]
        strain_avg = strain_avg[keep]; total_atoms = total_atoms[keep]
        vector_product = vector_product[keep]
        eps1 = eps1[keep]; eps2 = eps2[keep]
        area1_signed = area1_signed[keep]; area2_signed = area2_signed[keep]

    if frontier_only and c1v1.shape[0] >= 1:
        # Drop unphysical garbage-strain cells (ill-conditioned sliver bases) so
        # they cannot occupy the small-atom end of the Pareto frontier; their
        # huge strain is a numerical artifact, not a real candidate.
        physical = strain_avg <= MAX_PHYSICAL_STRAIN
        if not np.all(physical):
            _apply_keep(np.nonzero(physical)[0])
        if c1v1.shape[0] == 0:
            return []
    if frontier_only and c1v1.shape[0] > 1:
        # Reproduce the original cull, but per angle and in numpy, before any
        # Python candidate objects exist (this is the dominant speed-up):
        #   1. drop (strain, ratio) duplicates, keeping the most compact (lowest
        #      vector-product) representative -- suppresses the spurious slivers;
        #   2. keep only the (atoms, strain) Pareto frontier -- removes the
        #      "same angle, larger-and-more-strained" redundancy.
        eps_sum = eps1 + eps2
        _apply_keep(
            _strain_ratio_dedup_keep(
                strain_avg, ratio1, ratio2, total_atoms, vector_product, eps_sum,
                unique_strain_tol, unique_ratio_tol,
            )
        )
        if c1v1.shape[0] > 1:
            _apply_keep(_pareto_frontier_keep(total_atoms, strain_avg, vector_product, eps1 + eps2))
    elif canonicalize and first_indices.shape[0] > 1:
        # Canonical-sublattice collapse, stage 2 (final strain tie-break).  Stage
        # 1 already reduced each canonical sublattice group to the integer bases
        # that tie for the smallest ``(vector_product, total_atoms, eps1 + eps2)``
        # deduplication key; those survivors are mutual duplicates, and
        # ``deduplicate_candidates`` would keep the smallest-``strain_avg`` one.
        group_id = _canonical_group_ids(layer1_coeffs, layer2_coeffs, first_indices, second_indices)
        order = np.lexsort((strain_avg, group_id))
        sorted_groups = group_id[order]
        first_in_group = np.empty(sorted_groups.shape[0], dtype=bool)
        first_in_group[0] = True
        np.not_equal(sorted_groups[1:], sorted_groups[:-1], out=first_in_group[1:])
        _apply_keep(np.sort(order[first_in_group]))

    candidates = [
        SupercellCandidate(
            angle_deg=float(angle_deg),
            strain_avg=float(strain_avg[index]),
            strain_layer1=float(strain1[index]),
            strain_layer2=float(strain2[index]),
            ratio1=int(ratio1[index]),
            ratio2=int(ratio2[index]),
            total_atoms=int(total_atoms[index]),
            layer1_vector1=(int(c1v1[index, 0]), int(c1v1[index, 1])),
            layer1_vector2=(int(c1v2[index, 0]), int(c1v2[index, 1])),
            layer2_vector1=(int(c2v1[index, 0]), int(c2v1[index, 1])),
            layer2_vector2=(int(c2v2[index, 0]), int(c2v2[index, 1])),
            eps1=float(eps1[index]),
            eps2=float(eps2[index]),
            vector_product=float(vector_product[index]),
            area1=float(abs(area1_signed[index])),
            area2=float(abs(area2_signed[index])),
        )
        for index in range(c1v1.shape[0])
    ]

    candidates.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.vector_product, item.angle_deg))
    return candidates


def deduplicate_candidates(
    candidates: Sequence[SupercellCandidate],
    strain_tolerance: float = 1e-4,
    ratio_tolerance: float = 1e-5,
    angle_tolerance: float = 1e-9,
) -> List[SupercellCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.angle_deg, item.strain_avg, item.vector_product, item.total_atoms))
    unique: List[SupercellCandidate] = []

    # The duplicate predicate only ever holds for candidates whose angles differ
    # by at most ``angle_tolerance``; commensurate twist angles are far better
    # separated than that.  Partitioning the sorted list into contiguous
    # equal-angle blocks therefore reproduces the original cross-comparison
    # exactly while turning the worst-case O(U^2) scan into the sum of much
    # smaller per-angle O(g^2) scans.
    total = len(ordered)
    block_start = 0
    while block_start < total:
        block_end = block_start + 1
        block_angle = ordered[block_start].angle_deg
        while block_end < total and ordered[block_end].angle_deg <= block_angle + angle_tolerance:
            block_end += 1
        block = ordered[block_start:block_end]
        block_start = block_end

        block_unique: List[SupercellCandidate] = []
        block_size = len(block)
        for index in range(block_size):
            candidate = block[index]
            ratio = candidate.ratio1 / float(candidate.ratio2)
            duplicate = False
            for kept in block_unique:
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
            for scan_index in range(index + 1, block_size):
                other = block[scan_index]
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
            block_unique.append(best)
        unique.extend(block_unique)

    unique.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.angle_deg, item.vector_product))
    return unique


def _reflection_matrix_2d(axis_angle_rad: float) -> np.ndarray:
    """Cartesian 2x2 reflection across the line through the origin at the given angle."""
    cos2 = math.cos(2.0 * axis_angle_rad)
    sin2 = math.sin(2.0 * axis_angle_rad)
    return np.array([[cos2, sin2], [sin2, -cos2]], dtype=float)


def _is_integer_lattice_symmetry(basis_2d: np.ndarray, operation: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ``operation`` (a 2x2 Cartesian matrix) maps the lattice onto itself
    with an integer change-of-basis."""
    transform = basis_2d @ operation @ np.linalg.inv(basis_2d)
    rounded = np.rint(transform)
    if np.max(np.abs(transform - rounded)) > tol:
        return False
    scale = max(1.0, float(np.max(np.abs(basis_2d))))
    return np.max(np.abs(rounded @ basis_2d - basis_2d @ operation)) <= tol * scale


def fold_symmetry_operation(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, int] | None:
    """Detect a mirror symmetry that folds the twist-angle search in half.

    Many bilayers (e.g. MoS2/MoS2 and other hexagonal or square systems) possess
    a mirror line that relates the twist angle ``theta`` to ``sym - theta`` where
    ``sym`` is the combined rotational-symmetry period.  When such a reflection is
    a common symmetry of *both* lattices, the search only needs to cover the
    irreducible wedge ``[0, sym/2]``; every commensurate cell in ``(sym/2, sym]``
    is the mirror image of one in the wedge (the classic "search to 30 deg, then
    use signs to reach 60 and 120" trick for hexagonal crystals).

    The reflection across the axis at ``sym/2`` degrees maps ``theta -> sym -
    theta`` directly, so it is the operation returned here.  Returns ``(M, sym)``
    where ``M`` is the 2x2 Cartesian reflection, or ``None`` when no such common
    mirror exists (e.g. oblique lattices, or misaligned heterostructures).
    """
    basis1 = np.asarray(lattice1, dtype=float)[:2, :2]
    basis2 = np.asarray(lattice2, dtype=float)[:2, :2]
    _, _, sym = combined_symmetry_limit(lattice1, lattice2)

    def _axis_angle(basis: np.ndarray, coeffs: Tuple[int, int]) -> float:
        vector = coeffs[0] * basis[0] + coeffs[1] * basis[1]
        return math.atan2(float(vector[1]), float(vector[0]))

    # The reflection axis must sit at sym/2 measured from a mirror line of the
    # lattice; try the absolute sym/2 axis first, then sym/2 offsets from the
    # principal lattice directions to tolerate rotated input cells.
    half = math.radians(sym / 2.0)
    candidate_axes = [half]
    for basis in (basis2, basis1):
        for coeffs in ((1, 0), (0, 1)):
            candidate_axes.append(_axis_angle(basis, coeffs) + half)
            candidate_axes.append(_axis_angle(basis, coeffs))

    for axis_angle in candidate_axes:
        operation = _reflection_matrix_2d(axis_angle)
        if _is_integer_lattice_symmetry(basis1, operation, tol) and _is_integer_lattice_symmetry(
            basis2, operation, tol
        ):
            # Confirm it actually realises theta -> sym - theta (axis at sym/2).
            twist_probe = 17.0
            reflected = _reflection_matrix_2d(axis_angle)
            mapped = (2.0 * math.degrees(axis_angle) - twist_probe) % sym
            if abs(((mapped - (sym - twist_probe)) % sym)) < 1e-6 or abs(
                ((mapped - (sym - twist_probe)) % sym) - sym
            ) < 1e-6:
                return reflected, int(sym)
    return None


def mirror_supercell_candidate(
    candidate: SupercellCandidate,
    operation: np.ndarray,
    sym: int,
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    tol: float = 1e-4,
) -> SupercellCandidate | None:
    """Reflect a wedge supercell candidate to its mirror partner at ``sym - angle``.

    The reflection is an isometry, so all scalar quantities (strain, ratios, atom
    count, areas, errors) are preserved; only the twist angle and the integer
    index matrices change.  The new integer coefficients are recovered by solving
    the reflected Cartesian vectors against the target (``sym - angle``) basis and
    rounding, returning ``None`` if any coefficient fails to land on an integer
    (which would signal that the reflection is not a valid symmetry here).
    """
    basis1 = np.asarray(lattice1, dtype=float)[:2, :2]
    basis2 = np.asarray(lattice2, dtype=float)[:2, :2]
    operation = np.asarray(operation, dtype=float)

    angle = float(candidate.angle_deg)
    mirror_angle = float(sym) - angle

    rotated_basis1 = basis1 @ rotation_matrix_z(angle)[:2, :2].T
    target_basis1 = basis1 @ rotation_matrix_z(mirror_angle)[:2, :2].T
    inv_target1 = np.linalg.inv(target_basis1)
    inv_basis2 = np.linalg.inv(basis2)

    def _reflect_layer1(coeffs: Tuple[int, int]) -> Tuple[int, int] | None:
        vector = coeffs[0] * rotated_basis1[0] + coeffs[1] * rotated_basis1[1]
        reflected = vector @ operation
        raw = reflected @ inv_target1
        rounded = np.rint(raw)
        if np.max(np.abs(raw - rounded)) > tol:
            return None
        return (int(rounded[0]), int(rounded[1]))

    def _reflect_layer2(coeffs: Tuple[int, int]) -> Tuple[int, int] | None:
        vector = coeffs[0] * basis2[0] + coeffs[1] * basis2[1]
        reflected = vector @ operation
        raw = reflected @ inv_basis2
        rounded = np.rint(raw)
        if np.max(np.abs(raw - rounded)) > tol:
            return None
        return (int(rounded[0]), int(rounded[1]))

    new_l1_v1 = _reflect_layer1(candidate.layer1_vector1)
    new_l1_v2 = _reflect_layer1(candidate.layer1_vector2)
    new_l2_v1 = _reflect_layer2(candidate.layer2_vector1)
    new_l2_v2 = _reflect_layer2(candidate.layer2_vector2)
    if None in (new_l1_v1, new_l1_v2, new_l2_v1, new_l2_v2):
        return None

    import dataclasses

    return dataclasses.replace(
        candidate,
        angle_deg=mirror_angle,
        layer1_vector1=new_l1_v1,
        layer1_vector2=new_l1_v2,
        layer2_vector1=new_l2_v1,
        layer2_vector2=new_l2_v2,
    )

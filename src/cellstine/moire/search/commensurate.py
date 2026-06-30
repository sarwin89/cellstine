"""Analytic single-pass commensurate-supercell engine with redundancy culling.

Mathematical background
=======================
A twisted bilayer is commensurate when a lattice vector of the (rotated) top
layer coincides with a lattice vector of the bottom layer.  Write a top-layer
vector as ``u = m a1 + n a2`` (integers ``m, n``) and a bottom-layer vector as
``w = p b1 + q b2`` (integers ``p, q``).  Length is rotation invariant, so a
coincidence can only happen when ``|u| ~= |w|``.  *Given* such an equal-length
pair, the twist angle that makes them parallel is fixed analytically::

    theta = angle(w) - angle(u) = atan2(u x w, u . w)

i.e. the twist angle is an **output** of each integer vector pair, never a value
that has to be swept.  Two non-collinear coincidences that share the same
``theta`` span a commensurate supercell.

The engine therefore:

1. enumerates the in-plane integer vectors of both layers once;
2. keeps only equal-length pairs (a rotation-invariant prune, done by sorting
   the norms and binary-searching the admissible band -- never an N1 x N2
   matrix);
3. computes each surviving pair's twist ``theta`` and sorts the pairs by it;
4. for every candidate twist angle, gathers the pairs in the narrow angular
   window around it with a binary search (a contiguous slice -- no per-angle
   rescan of the whole pair set) and builds the supercells from them.

This replaces the old "shortlist angles, then re-rotate and re-scan *all*
length-matched pairs for every angle" loop, whose cost was ``O(angles x pairs)``
with the per-angle slice cost ``O(pairs + sum of slice sizes)``.

Redundancy culling
==================
At a fixed commensurate angle the search can return many supercells that are all
worse than one another: a larger cell that also carries more strain is never
useful, because the smaller / less strained cell already describes the same
moire.  ``pareto_cull`` keeps, per angle, only the cells on the
``(total_atoms, strain)`` Pareto frontier and drops every dominated cell.

``reduce_candidate`` additionally runs a Lagrange-Gauss reduction on each
reported integer basis so the cell vectors are the shortest / most orthogonal
representatives of the same superlattice (this turns skewed bases such as
``(-16, 17), (-15, 16)`` into clean ones).
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Sequence, Tuple

import numpy as np

from . import lattice as lat


def angle_window_deg(tol: float, merge_tol: float = 0.0) -> float:
    """Half-width (deg) of the angular window that can hold a coincidence.

    For length-matched vectors the relative error of a pair seen at a twist
    ``theta`` is at least ``|sin((alpha - theta) / 2)|``, so a pair with
    relative error ``<= tol`` necessarily satisfies
    ``|alpha - theta| <= 2 * asin(tol)``.  A safety factor and the angle merge
    tolerance are added so no admissible pair is ever missed by the slice.
    """
    clamped = min(max(float(tol), 0.0), 1.0)
    return math.degrees(2.0 * math.asin(clamped)) * 2.0 + float(merge_tol) + 1e-6


def precompute_pairs(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    nindex1: int,
    nindex2: int,
    tolerance: float,
    vector_strain_tol: float | None,
) -> dict:
    """Enumerate vectors, keep equal-length pairs, tag each with its twist angle.

    Returns a dict of arrays sorted by twist angle ``alpha`` (degrees, folded
    into ``[0, sym)``), ready for per-angle slicing.
    """
    basis1 = np.asarray(lattice1, dtype=float)[:2, :2]
    basis2 = np.asarray(lattice2, dtype=float)[:2, :2]
    coeffs1, vectors1 = lat.enumerate_in_plane_vectors(lattice1, nindex1)
    coeffs2, vectors2 = lat.enumerate_in_plane_vectors(lattice2, nindex2)
    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)

    lm_max = 2.0 * float(tolerance)
    if vector_strain_tol is not None:
        lm_max = min(lm_max, float(vector_strain_tol))
    lm_max = min(lm_max, lat.MAX_PHYSICAL_MISMATCH)

    _, _, sym = lat.combined_symmetry_limit(lattice1, lattice2)
    empty_i = np.empty(0, dtype=np.intp)
    empty_f = np.empty(0, dtype=float)
    base = {
        "coeffs1": coeffs1,
        "coeffs2": coeffs2,
        "vectors1": vectors1,
        "vectors2": vectors2,
        "norms1": norms1,
        "norms2": norms2,
        "sym": int(sym),
        "lm_max": float(lm_max),
    }
    if norms1.size == 0 or norms2.size == 0:
        base.update(rows=empty_i, cols=empty_i, lm=empty_f, alpha=empty_f)
        return base

    rows, cols = lat.norm_match_candidate_pairs(norms1, norms2, abs_tol=0.0, rel_tol=lm_max)
    if rows.size == 0:
        base.update(rows=empty_i, cols=empty_i, lm=empty_f, alpha=empty_f)
        return base
    half_sum = (norms1[rows] + norms2[cols]) * 0.5
    lm = np.abs(norms1[rows] - norms2[cols]) / np.maximum(half_sum, 1e-12)
    keep = lm <= lm_max
    rows, cols, lm = rows[keep], cols[keep], lm[keep]
    if rows.size == 0:
        base.update(rows=empty_i, cols=empty_i, lm=empty_f, alpha=empty_f)
        return base

    u = vectors1[rows]
    w = vectors2[cols]
    cross = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    dot = u[:, 0] * w[:, 0] + u[:, 1] * w[:, 1]
    alpha = np.mod(np.degrees(np.arctan2(cross, dot)), float(sym))
    order = np.argsort(alpha, kind="stable")
    base.update(rows=rows[order], cols=cols[order], lm=lm[order], alpha=alpha[order])
    return base


def _alpha_slice(alpha_sorted: np.ndarray, theta: float, window: float, sym: float) -> np.ndarray:
    """Indices of pairs whose twist angle lies within ``window`` of ``theta``
    (modulo ``sym``)."""
    lo = np.searchsorted(alpha_sorted, theta - window, side="left")
    hi = np.searchsorted(alpha_sorted, theta + window, side="right")
    parts = [np.arange(lo, hi, dtype=np.intp)]
    if theta - window < 0.0:  # wrap from the top of the period
        wlo = np.searchsorted(alpha_sorted, sym + (theta - window), side="left")
        parts.append(np.arange(wlo, alpha_sorted.shape[0], dtype=np.intp))
    if theta + window > sym:  # wrap to the bottom of the period
        whi = np.searchsorted(alpha_sorted, (theta + window) - sym, side="right")
        parts.append(np.arange(0, whi, dtype=np.intp))
    if len(parts) == 1:
        return parts[0]
    return np.unique(np.concatenate(parts))


def matches_at_angle(
    pre: dict,
    theta: float,
    tolerance: float,
    window: float,
    max_pair_matches: int | None,
) -> List[lat.VectorMatch]:
    """Build the coincident-vector matches at twist ``theta`` from the
    precomputed, angle-sorted pair set (a local slice, no global rescan)."""
    alpha = pre["alpha"]
    if alpha.size == 0:
        return []
    sym = float(pre["sym"])
    sel = _alpha_slice(alpha, float(theta), float(window), sym)
    if sel.size < 2:
        return []
    # signed angular distance to theta, folded into (-sym/2, sym/2]
    d = (alpha[sel] - float(theta) + sym * 0.5) % sym - sym * 0.5
    # If the caller has already requested a finite pair budget, avoid spending
    # vectorized trig and allocation work on every pair in a very dense angular
    # slice.  The closest-alpha pairs are the only ones that can survive with low
    # geometric error, and the later exact/short-vector cap still chooses the
    # final representatives.  Passing max_pair_matches<=0 disables this path via
    # the caller converting it to None.
    if max_pair_matches is not None:
        pre_cap = max(2048, int(max_pair_matches) * 16)
        if sel.size > pre_cap:
            nearest = np.argpartition(np.abs(d), pre_cap - 1)[:pre_cap]
            sel = sel[nearest]
            d = d[nearest]
    rows = pre["rows"][sel]
    cols = pre["cols"][sel]
    lm = pre["lm"][sel]
    L1 = pre["norms1"][rows]
    L2 = pre["norms2"][cols]
    err = np.sqrt(np.maximum(L1 * L1 + L2 * L2 - 2.0 * L1 * L2 * np.cos(np.radians(d)), 0.0))
    rel = err / np.maximum(L1 + L2, 1e-12)
    good = rel <= float(tolerance)
    rows, cols, lm = rows[good], cols[good], lm[good]
    err, rel = err[good], rel[good]
    if rows.size < 2:
        return []
    # Bound the per-angle pairing so highly symmetric (degenerate) angles, where
    # hundreds of vectors fall in the window, do not blow up the O(M^2) pairing.
    # The cap keeps two pools and unions them:
    #   * the shortest layer-1 vectors overall (captures the small cells), and
    #   * the shortest among the *near-exact* coincidences (captures the
    #     zero-/low-strain primitive cell even when its basis vectors are long).
    # Prioritising the exact coincidences is what keeps a long-period but
    # strain-free supercell (e.g. a small-twist commensuration) from being lost
    # behind a crowd of short, strained near-coincidences.
    if max_pair_matches is not None and rows.size > int(max_pair_matches):
        k = int(max_pair_matches)
        lengths = L1[good]
        short = np.argpartition(lengths, k)[:k]
        exact_eps = max(1e-6, 0.05 * float(tolerance))
        exact_idx = np.nonzero(rel < exact_eps)[0]
        if exact_idx.size > k:
            exact_idx = exact_idx[np.argpartition(lengths[exact_idx], k)[:k]]
        keep = np.union1d(short, exact_idx)
        rows, cols, lm, err, rel = rows[keep], cols[keep], lm[keep], err[keep], rel[keep]

    rot = lat.rotation_matrix_z(float(theta))[:2, :2]
    v1_rot = pre["vectors1"][rows] @ rot.T
    v2 = pre["vectors2"][cols]
    coeffs1 = pre["coeffs1"]
    coeffs2 = pre["coeffs2"]
    matches: List[lat.VectorMatch] = []
    for k in range(rows.shape[0]):
        r = int(rows[k])
        c = int(cols[k])
        matches.append(
            lat.VectorMatch(
                layer1_coeffs=(int(coeffs1[r, 0]), int(coeffs1[r, 1])),
                layer2_coeffs=(int(coeffs2[c, 0]), int(coeffs2[c, 1])),
                layer1_vector=(float(v1_rot[k, 0]), float(v1_rot[k, 1])),
                layer2_vector=(float(v2[k, 0]), float(v2[k, 1])),
                absolute_error=float(err[k]),
                relative_error=float(rel[k]),
                relative_length_mismatch=float(lm[k]),
            )
        )
    matches.sort(key=lambda item: (item.relative_error, item.relative_length_mismatch, item.absolute_error))
    return matches


# ---------------------------------------------------------------------------
# Redundancy culling
# ---------------------------------------------------------------------------

def _gauss_reduce_indices(
    matrix1: np.ndarray, matrix2: np.ndarray, basis1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Lagrange-Gauss reduce the supercell.

    ``matrix1`` / ``matrix2`` are the 2x2 integer index matrices whose rows are
    the two cell vectors expressed in ``basis1`` / ``basis2``.  The rows are
    reduced (by unimodular row operations) so the layer-1 Cartesian cell vectors
    are the shortest / most orthogonal pair; the *same* row operations are
    applied to ``matrix2`` so both layers keep describing the same supercell.
    """
    m1 = np.array(matrix1, dtype=np.int64)
    m2 = np.array(matrix2, dtype=np.int64)
    b1 = np.asarray(basis1, dtype=float)[:2, :2]
    for _ in range(128):
        vectors = m1 @ b1
        norm0 = float(vectors[0] @ vectors[0])
        norm1 = float(vectors[1] @ vectors[1])
        if norm0 > norm1:
            m1 = m1[[1, 0]]
            m2 = m2[[1, 0]]
            norm0, norm1 = norm1, norm0
            vectors = m1 @ b1
        if norm0 < 1e-18:
            break
        mu = int(round(float(vectors[0] @ vectors[1]) / norm0))
        if mu == 0:
            break
        m1[1] -= mu * m1[0]
        m2[1] -= mu * m2[0]
    # Canonical row signs: a row and its negative span the same lattice vector
    # line.  Flip each paired row so the first nonzero integer in the reported
    # layer-1/layer-2 coefficients is positive; this avoids equivalent all-
    # negative matrices without changing the supercell.
    for row_index in range(2):
        combined = (
            int(m1[row_index, 0]),
            int(m1[row_index, 1]),
            int(m2[row_index, 0]),
            int(m2[row_index, 1]),
        )
        first_nonzero = next((value for value in combined if value != 0), 0)
        if first_nonzero < 0:
            m1[row_index] *= -1
            m2[row_index] *= -1
    return m1, m2


def reduce_candidate(candidate: lat.SupercellCandidate, lattice1: np.ndarray, angle_deg: float) -> lat.SupercellCandidate:
    """Return ``candidate`` with both integer bases Lagrange-Gauss reduced.

    Strain, ratios, atom count, areas and angle are invariant under the
    unimodular row operations used, so only the reported index matrices change.
    """
    rotated_basis1 = lat.rotate_lattice(lattice1, float(angle_deg))
    m1 = np.array([candidate.layer1_vector1, candidate.layer1_vector2], dtype=np.int64)
    m2 = np.array([candidate.layer2_vector1, candidate.layer2_vector2], dtype=np.int64)
    r1, r2 = _gauss_reduce_indices(m1, m2, rotated_basis1)
    return dataclasses.replace(
        candidate,
        layer1_vector1=(int(r1[0, 0]), int(r1[0, 1])),
        layer1_vector2=(int(r1[1, 0]), int(r1[1, 1])),
        layer2_vector1=(int(r2[0, 0]), int(r2[0, 1])),
        layer2_vector2=(int(r2[1, 0]), int(r2[1, 1])),
    )


def reduce_candidate_checked(
    candidate: lat.SupercellCandidate,
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    angle_deg: float,
    tolerance: float,
) -> lat.SupercellCandidate | None:
    """Reduce a candidate basis while preserving the represented supercell.

    Earlier versions tried to validate the reduced basis row-by-row and flipped
    bottom rows independently when that looked closer.  That is too strict for
    legitimate symmetry-folded moire cells and can also manufacture impossible
    reported cells by changing the handedness of only one row.  The safe check is
    instead basis-level: after shared unimodular reduction, recompute the compact
    cell strain and keep the candidate only if the reduced basis still describes
    the same low-strain two-dimensional cell.
    """

    reduced = reduce_candidate(candidate, lattice1, angle_deg)
    rotated_basis1 = lat.rotate_lattice(lattice1, float(angle_deg))[:2, :2]
    basis2 = np.asarray(lattice2, dtype=float)[:2, :2]
    m1 = np.array([reduced.layer1_vector1, reduced.layer1_vector2], dtype=np.int64)
    m2 = np.array([reduced.layer2_vector1, reduced.layer2_vector2], dtype=np.int64)
    det1 = int(m1[0, 0] * m1[1, 1] - m1[0, 1] * m1[1, 0])
    det2 = int(m2[0, 0] * m2[1, 1] - m2[0, 1] * m2[1, 0])
    if det1 * det2 <= 0:
        return None
    v1 = m1 @ rotated_basis1
    v2 = m2 @ basis2

    reduced_strain = float(
        lat.calculate_strain_batch(
            np.stack((v1,), axis=0),
            np.stack((v2,), axis=0),
        )[0]
    )
    if not math.isfinite(reduced_strain):
        return None
    if reduced_strain > float(candidate.strain_avg) + max(float(tolerance), 1e-12):
        return None
    return reduced


def pareto_cull(
    candidates: Sequence[lat.SupercellCandidate],
    *,
    angle_tolerance: float = 1e-3,
    strain_epsilon: float = 1e-9,
) -> List[lat.SupercellCandidate]:
    """Keep, per twist angle, only the ``(total_atoms, strain_avg)`` Pareto
    frontier; drop every cell that another cell at the same angle beats on both
    size and strain.

    This removes the "same angle, again and again, with almost identical strain"
    redundancy: a bigger, more strained cell at a twist angle that already has a
    smaller / less strained cell is dominated and dropped.  Genuinely different
    trade-offs (a small but slightly strained cell vs a larger zero-strain cell)
    are *both* on the frontier and both kept.
    """
    if not candidates:
        return list(candidates)
    ordered = sorted(
        candidates,
        key=lambda c: (round(c.angle_deg, 6), c.total_atoms, c.strain_avg, c.vector_product),
    )
    kept: List[lat.SupercellCandidate] = []
    n = len(ordered)
    i = 0
    while i < n:
        angle = ordered[i].angle_deg
        j = i + 1
        while j < n and abs(ordered[j].angle_deg - angle) <= angle_tolerance:
            j += 1
        block = sorted(ordered[i:j], key=lambda c: (c.total_atoms, c.strain_avg, c.vector_product))
        i = j
        best_strain = math.inf
        for cand in block:
            if cand.strain_avg < best_strain - strain_epsilon:
                kept.append(cand)
                best_strain = cand.strain_avg
    kept.sort(key=lambda c: (c.strain_avg, c.total_atoms, c.angle_deg, c.vector_product))
    return kept

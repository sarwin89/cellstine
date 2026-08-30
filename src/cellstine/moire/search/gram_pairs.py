"""Joining the two layer families and reading off their invariants.

The Loewner acceptance test, the bucketed join over the two basis tables, the
principal stretches and twist angles of an accepted pair, the Pareto front, and
the canonical keys that identify pairs describing the same bilayer.
"""

from __future__ import annotations

import math

import numpy as np

from .gram_config import SearchConfig, _JOIN_CHUNK, _REL, _SLACK
from .gram_lattice import (
    _BottomIndex,
    _Table,
    _expand,
    _hermite_normal_form,
    _lexicographic_min_pair,
)

def _loewner_mask(p11, p12, p22, q11, q12, q22, lower, upper, tolerance=0.0):
    """Four-inequality Löwner acceptance for scalar or array Gram triples."""
    a11, a12, a22 = q11 - lower * p11, q12 - lower * p12, q22 - lower * p22
    b11, b12, b22 = upper * p11 - q11, upper * p12 - q12, upper * p22 - q22
    return (
        (a11 + a22 >= -tolerance)
        & (a11 * a22 - a12 * a12 >= -tolerance)
        & (b11 + b22 >= -tolerance)
        & (b11 * b22 - b12 * b12 >= -tolerance)
    )


def _join_candidates(top: _Table, index: _BottomIndex, lower: float, upper: float):
    """Four-probe bucket join with bucket-local off-diagonal windows."""
    output_top, output_bottom = [], []
    width = index.width
    bucket_top11 = np.floor((np.log(top.g11) + np.log(lower)) / width).astype(np.int64)
    bucket_top22 = np.floor((np.log(top.g22) + np.log(lower)) / width).astype(np.int64)
    for start in range(0, len(top), _JOIN_CHUNK):
        stop = min(start + _JOIN_CHUNK, len(top))
        selection = slice(start, stop)
        p11, p12, p22 = top.g11[selection], top.g12[selection], top.g22[selection]
        top_indices = np.arange(start, stop)
        for offset11 in (0, 1):
            for offset22 in (0, 1):
                bucket11 = bucket_top11[selection] + offset11
                bucket22 = bucket_top22[selection] + offset22
                valid = (
                    (bucket11 >= index.min11)
                    & (bucket11 <= index.max11)
                    & (bucket22 >= index.min22)
                    & (bucket22 <= index.max22)
                )
                q11_low = np.maximum(lower * p11, np.exp(width * bucket11))
                q11_high = np.minimum(upper * p11, np.exp(width * (bucket11 + 1)))
                q22_low = np.maximum(lower * p22, np.exp(width * bucket22))
                q22_high = np.minimum(upper * p22, np.exp(width * (bucket22 + 1)))
                low_radius = np.sqrt(
                    np.maximum(q11_high - lower * p11, 0.0)
                    * np.maximum(q22_high - lower * p22, 0.0)
                )
                high_radius = np.sqrt(
                    np.maximum(upper * p11 - q11_low, 0.0)
                    * np.maximum(upper * p22 - q22_low, 0.0)
                )
                padding = _SLACK * (1.0 + np.abs(p12))
                low12 = np.maximum(
                    lower * p12 - low_radius, upper * p12 - high_radius
                ) - padding
                high12 = np.minimum(
                    lower * p12 + low_radius, upper * p12 + high_radius
                ) + padding
                valid &= (
                    (q11_low <= q11_high)
                    & (q22_low <= q22_high)
                    & (low12 <= high12)
                )
                base = index.composite(bucket11, bucket22) * index.scale
                low = np.searchsorted(index.key, base + low12, side="left")
                high = np.searchsorted(index.key, base + high12, side="right")
                high = np.where(valid, high, low)
                top_rows, bottom_rows = _expand(low, high, top_indices)
                if top_rows.size == 0:
                    continue
                top11, bottom11 = top.g11[top_rows], index.g11[bottom_rows]
                keep = (bottom11 >= lower * top11 * (1.0 - _SLACK)) & (
                    bottom11 <= upper * top11 * (1.0 + _SLACK)
                )
                if not np.any(keep):
                    continue
                top_rows, bottom_rows = top_rows[keep], bottom_rows[keep]
                top11, bottom11 = top11[keep], bottom11[keep]
                top22, bottom22 = top.g22[top_rows], index.g22[bottom_rows]
                keep = (bottom22 >= lower * top22 * (1.0 - _SLACK)) & (
                    bottom22 <= upper * top22 * (1.0 + _SLACK)
                )
                if not np.any(keep):
                    continue
                top_rows, bottom_rows = top_rows[keep], bottom_rows[keep]
                top11, bottom11 = top11[keep], bottom11[keep]
                top22, bottom22 = top22[keep], bottom22[keep]
                top12, bottom12 = top.g12[top_rows], index.g12[bottom_rows]
                tolerance = 1e-9 * top11 * top22
                keep = _loewner_mask(
                    top11,
                    top12,
                    top22,
                    bottom11,
                    bottom12,
                    bottom22,
                    lower,
                    upper,
                    tolerance,
                )
                if np.any(keep):
                    output_top.append(top_rows[keep])
                    output_bottom.append(bottom_rows[keep])
    if not output_top:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(output_top), np.concatenate(output_bottom)


def _stretches_from_gram(p11, p12, p22, q11, q12, q22):
    """Principal stretches and relative logarithmic strains from two Gram forms.

    With ``T = p22 q11 - 2 p12 q12 + p11 q22`` the two principal stretches obey

    ``lambda1 + lambda2 = sqrt(T / det P + 2 sqrt(det Q / det P))`` and
    ``lambda1 - lambda2 = sqrt(T / det P - 2 sqrt(det Q / det P))``.

    The difference is the delicate one: for an almost unstrained match both terms
    are close to one another, so forming it as ``sqrt(sum^2 - 4 product)`` loses
    half of the available digits and turns an exactly commensurate cell into a
    spurious strain of order ``1e-8``.  Writing the same quantity as
    ``(T^2 - 4 det P det Q) / (det P (T + 2 sqrt(det P det Q)))`` moves the
    cancellation into the numerator, where it is still square-root sized: for an
    isotropic match ``T^2`` and ``4 det P det Q`` agree to the last digit, so the
    subtraction returns rounding noise and the square root turns an error of
    ``eps`` into an anisotropy of ``sqrt(eps)``.

    The numerator is therefore never formed as that difference.  With
    ``N = adj(P) Q`` one has ``tr N = T`` and ``det N = det P det Q``, and the
    discriminant of a two by two matrix obeys
    ``(tr N)^2 - 4 det N = (n11 - n22)^2 + 4 n12 n21``, which here reads

    ``(p22 q11 - p11 q22)^2 + 4 (p22 q12 - p12 q22) (p11 q12 - p12 q11)``.

    Every bracket is a cross difference that vanishes *identically* when ``Q`` is
    a multiple of ``P``, that is exactly when the match is isotropic, so each is
    accurate to ``eps`` times the size of its terms rather than to ``eps`` times
    ``T^2``.  The reported anisotropy of an isotropic match is then of order
    ``eps`` instead of ``sqrt(eps)``.
    """

    precise = np.longdouble
    p11, p12, p22 = (np.asarray(value, dtype=precise) for value in (p11, p12, p22))
    q11, q12, q22 = (np.asarray(value, dtype=precise) for value in (q11, q12, q22))
    determinant_p = p11 * p22 - p12 * p12
    determinant_q = q11 * q22 - q12 * q12
    combined = p22 * q11 - 2.0 * p12 * q12 + p11 * q22
    root_of_products = np.sqrt(np.maximum(determinant_p * determinant_q, precise(0.0)))
    product = root_of_products / determinant_p
    stretch_sum = np.sqrt(
        np.maximum(combined / determinant_p + 2.0 * product, precise(0.0))
    )
    discriminant = (p22 * q11 - p11 * q22) ** 2 + 4.0 * (p22 * q12 - p12 * q22) * (
        p11 * q12 - p12 * q11
    )
    difference_squared = discriminant / (
        determinant_p * (combined + 2.0 * root_of_products)
    )
    difference = np.sqrt(np.maximum(difference_squared, precise(0.0)))
    first = np.asarray(0.5 * (stretch_sum + difference), dtype=float)
    second = np.asarray(0.5 * (stretch_sum - difference), dtype=float)
    return first, second, np.log(first), np.log(second)


def _twist_angles(
    top_basis: np.ndarray,
    bottom_basis: np.ndarray,
    top_first: np.ndarray,
    top_second: np.ndarray,
    bottom_first: np.ndarray,
    bottom_second: np.ndarray,
) -> np.ndarray:
    """Polar-rotation angles from the integer adjugate expression, without forming F."""
    adjugate_top = np.array(
        [
            [top_basis[1, 1], -top_basis[0, 1]],
            [-top_basis[1, 0], top_basis[0, 0]],
        ]
    )
    trace_coefficients = adjugate_top @ bottom_basis
    skew_coefficients = (
        adjugate_top @ np.array([[0.0, 1.0], [-1.0, 0.0]]) @ bottom_basis
    )
    k11 = bottom_first[:, 0] * top_second[:, 1] - bottom_second[:, 0] * top_first[:, 1]
    k12 = bottom_second[:, 0] * top_first[:, 0] - bottom_first[:, 0] * top_second[:, 0]
    k21 = bottom_first[:, 1] * top_second[:, 1] - bottom_second[:, 1] * top_first[:, 1]
    k22 = bottom_second[:, 1] * top_first[:, 0] - bottom_first[:, 1] * top_second[:, 0]
    trace = (
        trace_coefficients[0, 0] * k11
        + trace_coefficients[0, 1] * k21
        + trace_coefficients[1, 0] * k12
        + trace_coefficients[1, 1] * k22
    )
    skew = (
        skew_coefficients[0, 0] * k11
        + skew_coefficients[0, 1] * k21
        + skew_coefficients[1, 0] * k12
        + skew_coefficients[1, 1] * k22
    )
    return np.arctan2(skew, trace)


def _pareto_front(first_cost: np.ndarray, second_cost: np.ndarray) -> np.ndarray:
    """Deterministic indices of strict Pareto records in increasing first-cost order.

    The sweep sorts by ``(first cost, second cost, index)`` and keeps a candidate
    when its second cost is strictly below the running minimum of the second
    costs already seen.  ``RequestProject/ParetoFront.lean`` proves that this is
    exactly the Pareto front: a candidate is kept iff nothing scanned before it
    is at least as good in both costs (``Cellstine.Pareto.isRecord_iff_not_dominated``),
    a kept candidate is undominated outright
    (``Cellstine.Pareto.eq_costs_of_isRecord_of_le``), every candidate is matched
    or beaten in both costs by a kept one (``Cellstine.Pareto.exists_isRecord_le``),
    and the front is a strict staircase of distinct first costs
    (``Cellstine.Pareto.second_lt_of_isRecord_of_first_lt``).
    """
    order = np.lexsort((np.arange(len(first_cost)), second_cost, first_cost))
    sorted_second = second_cost[order]
    previous_best = np.minimum.accumulate(np.concatenate([[np.inf], sorted_second[:-1]]))
    return order[sorted_second < previous_best]


def _canonical_pair_keys(top_matrices: np.ndarray, bottom_matrices: np.ndarray) -> np.ndarray:
    """Exact class keys modulo the common right action by unimodular matrices.

    The top matrix is sent to its column HNF and the identical right transform is applied
    to the bottom matrix.  This is an integer-only canonicalization, unlike rounded
    floating-point geometry keys.
    """
    count = len(top_matrices)
    if count == 0:
        return np.zeros((0, 8), dtype=np.int64)
    top_first, top_second = top_matrices[:, :, 0], top_matrices[:, :, 1]
    triples = _hermite_normal_form(top_first, top_second)
    h11, h12, h22 = triples[:, 0], triples[:, 1], triples[:, 2]
    m00, m01 = top_matrices[:, 0, 0], top_matrices[:, 0, 1]
    m10, m11 = top_matrices[:, 1, 0], top_matrices[:, 1, 1]
    determinant = m00 * m11 - m01 * m10
    # ``adj(M) @ H`` written out.  ``H`` is upper triangular, so the products
    # below are the whole matmul; the generic integer ``@`` on a stack of 2x2
    # blocks has no BLAS path and costs several times as much.
    n00 = m11 * h11
    n01 = m11 * h12 - m01 * h22
    n10 = -m10 * h11
    n11 = -m10 * h12 + m00 * h22
    t00, t01 = n00 // determinant, n01 // determinant
    t10, t11 = n10 // determinant, n11 // determinant
    # Cheaper than a second integer division: the quotient is exact exactly when
    # multiplying it back reproduces the numerator.
    if not (
        np.array_equal(t00 * determinant, n00)
        and np.array_equal(t01 * determinant, n01)
        and np.array_equal(t10 * determinant, n10)
        and np.array_equal(t11 * determinant, n11)
    ):
        raise ArithmeticError("column-HNF transform was not unimodular integral")
    b00, b01 = bottom_matrices[:, 0, 0], bottom_matrices[:, 0, 1]
    b10, b11 = bottom_matrices[:, 1, 0], bottom_matrices[:, 1, 1]
    keys = np.empty((count, 8), dtype=np.int64)
    keys[:, 0] = h11
    keys[:, 1] = h12
    keys[:, 2] = 0
    keys[:, 3] = h22
    keys[:, 4] = b00 * t00 + b01 * t10
    keys[:, 5] = b00 * t01 + b01 * t11
    keys[:, 6] = b10 * t00 + b11 * t10
    keys[:, 7] = b10 * t01 + b11 * t11
    return keys


def _symmetry_combinations(
    top_group: np.ndarray, bottom_group: np.ndarray
) -> list[tuple[int, int]]:
    """Return the layer-symmetry pairs that map an enumerated pair to another one.

    Acting with ``G_t`` on the top supercell and ``G_b`` on the bottom one leaves
    the bilayer congruent, because each ``G`` is a symmetry of its own layer.  The
    two determinants must agree in sign: a lone reflection would turn the
    right-handed pairs produced by the enumeration into left-handed ones.
    """

    top = np.asarray(top_group, dtype=np.int64)
    bottom = np.asarray(bottom_group, dtype=np.int64)
    top_sign = np.sign(top[:, 0, 0] * top[:, 1, 1] - top[:, 0, 1] * top[:, 1, 0])
    bottom_sign = np.sign(
        bottom[:, 0, 0] * bottom[:, 1, 1] - bottom[:, 0, 1] * bottom[:, 1, 0]
    )
    return [
        (int(left), int(right))
        for left in range(len(top))
        for right in range(len(bottom))
        if top_sign[left] == bottom_sign[right]
    ]


def _reduced_symmetry_combinations(
    top_group: np.ndarray, bottom_group: np.ndarray
) -> list[tuple[int, int]]:
    """Drop the symmetry pairs that cannot change a class key.

    ``(-G_t, -G_b)`` always produces the very same key as ``(G_t, G_b)``: the
    column Hermite form depends only on the column lattice of the top matrix,
    which ``-1`` leaves alone, and the two signs cancel in the transported
    bottom matrix ``N M^{-1} H``.  Removing one of each such pair halves the work
    of :func:`_pair_orbit_keys` for a centrosymmetric layer without changing a
    single key.  Proved as ``Cellstine.classKey_neg`` in
    ``RequestProject/PairClassKey.lean``.
    """

    combinations = _symmetry_combinations(top_group, bottom_group)
    kept: list[tuple[int, int]] = []
    seen: set[bytes] = set()
    for left, right in combinations:
        pair = np.concatenate(
            [top_group[left].reshape(4), bottom_group[right].reshape(4)]
        ).astype(np.int64)
        signature = pair.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        seen.add((-pair).tobytes())
        kept.append((left, right))
    return kept


def _pair_orbit_keys(
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    top_group: np.ndarray,
    bottom_group: np.ndarray,
) -> np.ndarray:
    """Return exact class keys modulo layer symmetries *and* cell relabelling.

    Two enumerated pairs describe the same bilayer exactly when they are related
    by ``(M, N) -> (G_t M K, G_b N K)`` with layer symmetries ``G_t``, ``G_b`` and
    a common unimodular ``K``.  :func:`_canonical_pair_keys` removes only the
    ``K`` freedom, which still reports one copy of the same stacking per symmetry
    image; minimising that key over the symmetry pairs removes the rest.
    """

    count = len(top_matrices)
    if count == 0:
        return np.zeros((0, 8), dtype=np.int64)
    top = np.asarray(top_group, dtype=np.int64)
    bottom = np.asarray(bottom_group, dtype=np.int64)
    combinations = _reduced_symmetry_combinations(top, bottom)
    if len(combinations) <= 1:
        return _canonical_pair_keys(top_matrices, bottom_matrices)
    best: np.ndarray | None = None
    for left, right in combinations:
        # Folded in one at a time: holding all |G_t| |G_b| key tables at once is
        # pure memory traffic and nothing else.
        candidate = _canonical_pair_keys(
            top[left] @ top_matrices,
            bottom[right] @ bottom_matrices,
        )
        best = candidate if best is None else _lexicographic_min_pair(best, candidate)
    assert best is not None
    return best


def _shape_mask(table: _Table, config: SearchConfig) -> np.ndarray:
    keep = table.g22 <= (
        config.max_aspect_ratio * config.max_aspect_ratio * table.g11 * (1.0 + _REL)
    )
    cosine = table.g12 / np.sqrt(table.g11 * table.g22)
    lower_cosine = math.cos(math.radians(config.max_cell_angle_deg))
    upper_cosine = math.cos(math.radians(config.min_cell_angle_deg))
    keep &= (cosine >= lower_cosine - _REL) & (cosine <= upper_cosine + _REL)
    return keep

"""Turning accepted pairs into reportable moire cells.

Matrix powers of symmetric positive-definite Gram forms, the affine maps that
carry each layer onto the shared cell, the coincidence index, folding the twist
angle into the fundamental range, and the final assembly of a
:class:`~cellstine.moire.search.gram_config.SearchResult`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...core.symmetry2d import cartesian_mirror_angles, cartesian_rotation_angles
from .gram_config import SearchConfig, SearchResult, _CERTIFICATION_MARGIN, _REL, _TWO_PI
from .gram_lattice import _first_per_key
from .gram_pairs import (
    _canonical_pair_keys,
    _loewner_mask,
    _pair_orbit_keys,
    _pareto_front,
    _stretches_from_gram,
)

#: Slack on a requested twist window, in radians (about 6e-6 degrees).
#: A commensurate angle is an exact arccosine of a rational, and asking for a
#: window that ends on one must not miss it by a rounding of the bound itself.
_ANGLE_WINDOW_TOLERANCE = 1e-7


def _divided_difference_power(
    first: np.ndarray, second: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """Return ``(a**p - b**p) / (a - b)`` accurately for any two positive stretches.

    Evaluating the quotient directly cancels catastrophically once the two
    principal stretches nearly coincide, which is precisely the case for an
    almost isotropic match: a relative gap of ``1e-10`` already costs six
    digits, so the layer affines then disagree with the shared lattice at the
    micro-angstrom level.  Guarding the quotient with a hard threshold only
    moves the problem to the values just above it.

    Writing ``a = m (1 + d)`` and ``b = m (1 - d)`` with ``m`` the mean and ``d``
    the relative half-difference gives

    ``(a**p - b**p) / (a - b) = m**(p-1) * exp(p (L+ + L-) / 2) * sinh(w) / d``

    with ``L± = log1p(±d)`` and ``w = p (L+ - L-) / 2``.  Every factor is then
    evaluated where it is well conditioned: ``log1p`` near ``d = 0``, ``sinh(w)/w``
    by its series for small ``w``, and ``w / d`` by the series of
    ``p (log1p(d) - log1p(-d)) / (2 d)``, which tends to ``p``.
    """

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    p = np.asarray(power, dtype=float)
    mean = 0.5 * (a + b)
    safe_mean = np.where(mean > 0.0, mean, 1.0)
    half_difference = np.clip(0.5 * (a - b) / safe_mean, -1.0 + 1e-15, 1.0 - 1e-15)
    log_plus = np.log1p(half_difference)
    log_minus = np.log1p(-half_difference)
    argument = 0.5 * p * (log_plus - log_minus)
    centre = np.exp(0.5 * p * (log_plus + log_minus))
    small_argument = np.abs(argument) < 1e-8
    sinh_ratio = np.where(
        small_argument,
        1.0 + argument * argument / 6.0,
        np.sinh(np.where(small_argument, 1.0, argument)) / np.where(small_argument, 1.0, argument),
    )
    small_difference = np.abs(half_difference) < 1e-8
    slope = np.where(
        small_difference,
        p * (1.0 + half_difference * half_difference / 3.0),
        argument / np.where(small_difference, 1.0, half_difference),
    )
    return np.power(safe_mean, p - 1.0) * centre * sinh_ratio * slope


def _matrix_power_spd(
    stretch: np.ndarray, first: np.ndarray, second: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """Vectorised analytic power of a symmetric positive-definite 2x2 matrix.

    For a 2x2 matrix with eigenvalues ``a`` and ``b`` the Cayley-Hamilton form of
    ``S**p`` is ``alpha S + beta I`` with ``alpha`` the divided difference of
    ``x**p`` and ``beta = -a b`` times the divided difference of ``x**(p-1)``.
    """

    alpha = _divided_difference_power(first, second, power)
    beta = -first * second * _divided_difference_power(first, second, power - 1.0)
    return _cayley_hamilton(stretch, alpha, beta)


def _cayley_hamilton(stretch: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Assemble ``alpha S + beta I`` for a stack of 2x2 blocks."""

    result = alpha[:, None, None] * stretch
    result[:, 0, 0] += beta
    result[:, 1, 1] += beta
    return result


def _consecutive_matrix_powers_spd(
    stretch: np.ndarray, first: np.ndarray, second: np.ndarray, power: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``S**power`` and ``S**(power - 1)`` in one pass.

    Two consecutive powers need the divided differences at ``power``,
    ``power - 1`` and ``power - 2``; evaluating them independently would repeat
    the one at ``power - 1``, which is the single most expensive scalar kernel in
    the whole report.
    """

    upper = _divided_difference_power(first, second, power)
    middle = _divided_difference_power(first, second, power - 1.0)
    lower = _divided_difference_power(first, second, power - 1.0 - 1.0)
    product = first * second
    return (
        _cayley_hamilton(stretch, upper, -product * middle),
        _cayley_hamilton(stretch, middle, -product * lower),
    )


def _affine_geometry(
    config: SearchConfig,
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    twist: np.ndarray,
    first_stretch: np.ndarray,
    second_stretch: np.ndarray,
    sharing: np.ndarray,
):
    count = len(top_matrices)
    if count == 0:
        empty = np.zeros((0, 2, 2))
        return empty, empty, empty
    top_cells = config.top_basis @ top_matrices
    bottom_cells = config.bottom_basis @ bottom_matrices
    # Solving is both cheaper and better conditioned than forming the inverse:
    # D T = B with columns, i.e. T^T D^T = B^T.
    deformation = np.swapaxes(
        np.linalg.solve(np.swapaxes(top_cells, 1, 2), np.swapaxes(bottom_cells, 1, 2)), 1, 2
    )
    cosine, sine = np.cos(twist), np.sin(twist)
    rotation = np.empty((count, 2, 2))
    rotation[:, 0, 0], rotation[:, 0, 1] = cosine, -sine
    rotation[:, 1, 0], rotation[:, 1, 1] = sine, cosine
    stretch = np.swapaxes(rotation, 1, 2) @ deformation
    top_power, bottom_power = _consecutive_matrix_powers_spd(
        stretch, first_stretch, second_stretch, sharing
    )
    top_affine = rotation @ top_power
    bottom_affine = rotation @ bottom_power @ np.swapaxes(rotation, 1, 2)
    shared = top_affine @ top_cells
    return top_affine, bottom_affine, shared


def coincidence_index(top_matrices: np.ndarray, bottom_matrices: np.ndarray) -> np.ndarray:
    """Return how many primitive moire cells each reported supercell contains.

    In supercell coordinates a point ``x`` belongs to the top lattice exactly when
    ``M x`` is integral and to the bottom lattice exactly when ``N x`` is integral,
    so the coincidence lattice is ``L = M^-1 Z^2 cap N^-1 Z^2``.  Its dual is the
    lattice spanned by the columns of ``[M.T | N.T]``, whose determinant -- the gcd
    of the six 2x2 minors -- is the index ``[L : Z^2]``.  A value of one means the
    reported cell *is* the primitive commensurate cell; a value ``d > 1`` means the
    same bilayer is described ``d`` times more coarsely than necessary.
    """

    top = np.asarray(top_matrices, dtype=np.int64)
    bottom = np.asarray(bottom_matrices, dtype=np.int64)
    if len(top) == 0:
        return np.zeros(0, dtype=np.int64)
    columns = np.concatenate(
        [np.swapaxes(top, 1, 2), np.swapaxes(bottom, 1, 2)], axis=2
    )
    minors = []
    for first in range(4):
        for second in range(first + 1, 4):
            minors.append(
                columns[:, 0, first] * columns[:, 1, second]
                - columns[:, 1, first] * columns[:, 0, second]
            )
    stacked = np.abs(np.stack(minors, axis=1))
    result = stacked[:, 0]
    for column in range(1, stacked.shape[1]):
        result = np.gcd(result, stacked[:, column])
    return result.astype(np.int64)


def _reduce_common_basis(gram11, gram12, gram22, determinant_sign):
    """Vectorised Lagrange--Gauss reduction of a family of 2x2 Gram forms.

    Returns the integer transforms ``K`` such that the basis ``B @ K`` is reduced
    (``|2 g12| <= g11 <= g22``) and right handed.  ``K`` acts on both layers at
    once, so it is a pure relabelling of the shared moire cell.

    On the boundary ``2 |g12| = g11`` -- every hexagonal moire cell, which is
    most of the interesting ones -- both signs of ``g12`` satisfy the reduction
    condition, one describing the cell at sixty degrees and the other the same
    cell at a hundred and twenty, and which one the rounds stop at is decided by
    the last bit of ``g12``.  Without a rule the reported ``moire_gamma_deg`` of
    one and the same cell flips between the two whenever an upstream sum is
    reassociated.  The obtuse one is chosen, matching the ``(60, 120]`` range the
    rest of CELLSTINE reports and :func:`core.reduction.plane_reduce`; the step
    that gets there is the shear ``b -> b - a``, which
    ``Cellstine.gaussStep_boundary_involutive`` shows cycles on the boundary and
    which, unlike negating ``b``, leaves the handedness fixed above intact.
    """

    count = len(gram11)
    transform = np.zeros((count, 2, 2), dtype=np.int64)
    transform[:, 0, 0] = 1
    transform[:, 1, 1] = 1
    if count == 0:
        return transform
    g11 = np.array(gram11, dtype=float, copy=True)
    g12 = np.array(gram12, dtype=float, copy=True)
    g22 = np.array(gram22, dtype=float, copy=True)
    for _ in range(64):
        swap = g11 > g22 * (1.0 + _REL)
        if np.any(swap):
            g11[swap], g22[swap] = g22[swap], g11[swap]
            columns = transform[swap][:, :, ::-1]
            transform[swap] = columns
            g12[swap] = g12[swap]
        multiplier = np.where(
            2.0 * np.abs(g12) <= g11 * (1.0 + _REL),
            0,
            np.round(np.divide(g12, np.maximum(g11, np.finfo(float).tiny))),
        ).astype(np.int64)
        if not np.any(multiplier):
            break
        g22 = g22 - 2.0 * multiplier * g12 + (multiplier**2) * g11
        g12 = g12 - multiplier * g11
        transform[:, :, 1] -= multiplier[:, None] * transform[:, :, 0]
    else:  # pragma: no cover - defensive, reduction converges in a few steps
        raise ArithmeticError("common Lagrange--Gauss reduction did not converge")
    transform_determinant = (
        transform[:, 0, 0] * transform[:, 1, 1] - transform[:, 0, 1] * transform[:, 1, 0]
    )
    flip = transform_determinant * np.asarray(determinant_sign, dtype=np.int64) < 0
    transform[flip, :, 1] *= -1
    g12 = np.where(flip, -g12, g12)
    obtuse = 2.0 * g12 >= g11 * (1.0 - _REL)
    if np.any(obtuse):
        transform[obtuse, :, 1] -= transform[obtuse, :, 0]
    return transform


def _matmul2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise product of two stacks of 2x2 integer blocks, written out.

    ``numpy``'s ``@`` on a stack of small integer matrices has no BLAS path and
    pays the generic matmul machinery per block; the eight products written out
    are several times faster on the millions of blocks a wide search produces.
    Used for integer operands only, so that no floating-point summation order is
    changed anywhere in the reported geometry.
    """

    a00, a01 = left[:, 0, 0], left[:, 0, 1]
    a10, a11 = left[:, 1, 0], left[:, 1, 1]
    b00, b01 = right[:, 0, 0], right[:, 0, 1]
    b10, b11 = right[:, 1, 0], right[:, 1, 1]
    result = np.empty(
        (len(left), 2, 2), dtype=np.result_type(left.dtype, right.dtype)
    )
    result[:, 0, 0] = a00 * b00 + a01 * b10
    result[:, 0, 1] = a00 * b01 + a01 * b11
    result[:, 1, 0] = a10 * b00 + a11 * b10
    result[:, 1, 1] = a10 * b01 + a11 * b11
    return result


def _apply_gram_transform(gram: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Return the Gram triples of ``B @ K`` from the triples of ``B``."""

    k11 = transform[:, 0, 0].astype(float)
    k21 = transform[:, 1, 0].astype(float)
    k12 = transform[:, 0, 1].astype(float)
    k22 = transform[:, 1, 1].astype(float)
    g11, g12, g22 = gram[:, 0], gram[:, 1], gram[:, 2]
    new11 = k11 * k11 * g11 + 2.0 * k11 * k21 * g12 + k21 * k21 * g22
    new12 = k11 * k12 * g11 + (k11 * k22 + k21 * k12) * g12 + k21 * k22 * g22
    new22 = k12 * k12 * g11 + 2.0 * k12 * k22 * g12 + k22 * k22 * g22
    return np.stack([new11, new12, new22], axis=1)


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return values - _TWO_PI * np.round(values / _TWO_PI)


def _canonical_half_turn(values: np.ndarray) -> np.ndarray:
    """Report a half turn as ``+pi`` rather than ``-pi``.

    ``arctan2`` returns the half turn with the sign of its first argument, so a
    cross product that lands on ``-0.0`` -- which is exactly what an antiparallel
    pair of supercell vectors gives -- reports ``-180`` degrees where the very
    same structure found along another route reports ``+180``.  The two are the
    same rotation, so the positive representative is used for both.
    """

    angles = np.asarray(values, dtype=float)
    if not angles.size:
        return angles
    return np.where(np.abs(angles + np.pi) <= _ANGLE_WINDOW_TOLERANCE, np.pi, angles)


def _canonical_twist(
    config: SearchConfig,
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    twist: np.ndarray,
):
    """Fold the twist angle into its fundamental range and move the matrices along.

    Acting with a layer symmetry ``G_t`` on the top supercell rotates the top layer
    by the Cartesian angle ``phi_t`` of ``G_t``, so the twist becomes
    ``theta - phi_t``; a bottom symmetry adds ``+phi_b``.  A pair of reflections
    (one per layer) is orientation preserving as a whole and sends
    ``theta -> -theta + psi_b - psi_t``.  The representative with the smallest
    magnitude is reported, and the matrices are transformed with it so that the
    reported integer cells really do generate the reported angle.
    """

    count = len(top_matrices)
    if count == 0:
        return top_matrices, bottom_matrices, twist
    top_group = np.asarray(config.top_group, dtype=np.int64)
    bottom_group = np.asarray(config.bottom_group, dtype=np.int64)
    top_determinant = (
        top_group[:, 0, 0] * top_group[:, 1, 1] - top_group[:, 0, 1] * top_group[:, 1, 0]
    )
    bottom_determinant = (
        bottom_group[:, 0, 0] * bottom_group[:, 1, 1]
        - bottom_group[:, 0, 1] * bottom_group[:, 1, 0]
    )
    identity = np.eye(2, dtype=np.int64)

    def _identity_first(indices: np.ndarray, group: np.ndarray) -> np.ndarray:
        """List the identity first so untransformed matrices win every tie."""

        order = sorted(
            indices.tolist(), key=lambda index: 0 if np.array_equal(group[index], identity) else 1
        )
        return np.asarray(order, dtype=np.int64)

    top_proper = _identity_first(np.nonzero(top_determinant > 0)[0], top_group)
    bottom_proper = _identity_first(np.nonzero(bottom_determinant > 0)[0], bottom_group)
    top_improper = np.nonzero(top_determinant < 0)[0]
    bottom_improper = np.nonzero(bottom_determinant < 0)[0]
    top_angles = cartesian_rotation_angles(config.top_basis, top_group[top_proper])
    bottom_angles = cartesian_rotation_angles(config.bottom_basis, bottom_group[bottom_proper])
    top_axes = cartesian_mirror_angles(config.top_basis, top_group[top_improper])
    bottom_axes = cartesian_mirror_angles(config.bottom_basis, bottom_group[bottom_improper])

    combinations: list[tuple[int, int, float, float]] = []
    for left, phi_top in zip(top_proper, top_angles):
        for right, phi_bottom in zip(bottom_proper, bottom_angles):
            combinations.append((int(left), int(right), 1.0, float(phi_bottom - phi_top)))
    for left, psi_top in zip(top_improper, top_axes):
        for right, psi_bottom in zip(bottom_improper, bottom_axes):
            combinations.append((int(left), int(right), -1.0, float(psi_bottom - psi_top)))

    best_angle = None
    best_score = None
    best_choice = None
    for index, (_, _, sign, offset) in enumerate(combinations):
        candidate = _wrap_angle(sign * twist + offset)
        score = np.abs(candidate) - 1e-12 * np.sign(candidate)
        if best_score is None:
            best_angle = candidate
            best_score = score
            best_choice = np.zeros(count, dtype=np.int64)
            continue
        better = score < best_score
        best_angle = np.where(better, candidate, best_angle)
        best_score = np.where(better, score, best_score)
        best_choice = np.where(better, index, best_choice)

    new_top = np.array(top_matrices, dtype=np.int64, copy=True)
    new_bottom = np.array(bottom_matrices, dtype=np.int64, copy=True)
    for index, (left, right, _, _) in enumerate(combinations):
        selection = best_choice == index
        if not np.any(selection):
            continue
        new_top[selection] = top_group[left] @ new_top[selection]
        new_bottom[selection] = bottom_group[right] @ new_bottom[selection]
    return new_top, new_bottom, best_angle


def _finalize(
    config: SearchConfig,
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    top_gram: np.ndarray,
    bottom_gram: np.ndarray,
    top_multiplicity: np.ndarray,
    bottom_multiplicity: np.ndarray,
    twist: np.ndarray,
    stats: dict[str, Any],
    length_scale: float,
) -> SearchResult:
    """Turn raw accepted pairs into canonical, reportable candidates."""

    finalize_started = time.perf_counter()
    stats = dict(stats)

    # 1. Drop pairs that merely repeat a smaller commensurate cell, and
    # 2. report each bilayer once by folding the pair by the two layer symmetries.
    #
    # Both stages only *select* rows, so their selections are composed and the
    # payload is gathered a single time.  On a wide search the payload runs to
    # hundreds of megabytes and the gathers, not the arithmetic, dominate; a
    # stage that turns out to drop nothing now costs no copy at all.
    total = int(len(top_matrices))
    indices = coincidence_index(top_matrices, bottom_matrices)
    stats["n_before_primitive"] = total
    selection: np.ndarray | None = None
    if config.primitive_only and total:
        keep = np.flatnonzero(indices == 1)
        if len(keep) != total:
            selection = keep
    remaining = total if selection is None else int(len(selection))
    stats["n_imprimitive_dropped"] = total - remaining

    if config.fold_symmetry and remaining:
        folded_top = top_matrices if selection is None else top_matrices[selection]
        folded_bottom = (
            bottom_matrices if selection is None else bottom_matrices[selection]
        )
        unique = _first_per_key(
            _pair_orbit_keys(
                folded_top, folded_bottom, config.top_group, config.bottom_group
            )
        )
        if len(unique) != remaining:
            selection = unique if selection is None else selection[unique]
            remaining = int(len(unique))
    if selection is not None:
        top_matrices = top_matrices[selection]
        bottom_matrices = bottom_matrices[selection]
        top_gram, bottom_gram = top_gram[selection], bottom_gram[selection]
        top_multiplicity, bottom_multiplicity = (
            top_multiplicity[selection],
            bottom_multiplicity[selection],
        )
        twist, indices = twist[selection], indices[selection]
    raw_twist = np.array(twist, dtype=float, copy=True)
    stats["n_after_symmetry_folding"] = remaining

    # 3. Fold the twist angle into its fundamental range, carrying the matrices.
    if config.fold_symmetry:
        top_matrices, bottom_matrices, twist = _canonical_twist(
            config, top_matrices, bottom_matrices, twist
        )
    twist = _canonical_half_turn(twist)
    raw_twist = _canonical_half_turn(raw_twist)
    stats["n_after_angle_folding"] = int(len(top_matrices))

    # 3b. Keep only the twists the caller asked for, read on the folded angle.
    window = config.twist_window_radians
    if window is not None and len(top_matrices):
        lower_angle, upper_angle = window
        magnitude = np.abs(twist)
        inside = (magnitude >= lower_angle - _ANGLE_WINDOW_TOLERANCE) & (
            magnitude <= upper_angle + _ANGLE_WINDOW_TOLERANCE
        )
        top_matrices, bottom_matrices = top_matrices[inside], bottom_matrices[inside]
        top_gram, bottom_gram = top_gram[inside], bottom_gram[inside]
        top_multiplicity, bottom_multiplicity = (
            top_multiplicity[inside],
            bottom_multiplicity[inside],
        )
        twist, raw_twist, indices = twist[inside], raw_twist[inside], indices[inside]
    stats["n_after_angle_window"] = int(len(top_matrices))

    # 4. Strains, certification and atom counts are basis independent.
    first_stretch, second_stretch, first_strain, second_strain = _stretches_from_gram(
        top_gram[:, 0],
        top_gram[:, 1],
        top_gram[:, 2],
        bottom_gram[:, 0],
        bottom_gram[:, 1],
        bottom_gram[:, 2],
    )
    principal_strains = np.stack([first_strain, second_strain], axis=1)
    sharing = np.full(len(top_matrices), config._sharing)
    lower, upper = config._band
    certified = _loewner_mask(
        top_gram[:, 0],
        top_gram[:, 1],
        top_gram[:, 2],
        bottom_gram[:, 0],
        bottom_gram[:, 1],
        bottom_gram[:, 2],
        lower * (1.0 + _CERTIFICATION_MARGIN),
        upper * (1.0 - _CERTIFICATION_MARGIN),
    )
    top_atom_counts = top_multiplicity.astype(np.int64) * config.top_atoms
    bottom_atom_counts = bottom_multiplicity.astype(np.int64) * config.bottom_atoms
    atom_counts = top_atom_counts + bottom_atom_counts

    # 5. Deterministic ranking: smallest cell first, then least strain.
    strain_cost = (
        np.max(np.abs(principal_strains), axis=1)
        if len(principal_strains)
        else np.zeros(0)
    )
    pareto = np.zeros(len(top_matrices), dtype=bool)
    pareto[_pareto_front(atom_counts.astype(float), strain_cost)] = True
    keys = _canonical_pair_keys(top_matrices, bottom_matrices)
    if len(top_matrices):
        # Column 2 of a class key is the lower-left entry of an upper triangular
        # Hermite form and is identically zero, so it can never break a tie.
        ordering_keys = [
            keys[:, column]
            for column in range(keys.shape[1] - 1, -1, -1)
            if column != 2
        ]
        order = np.lexsort(
            tuple(ordering_keys + [np.abs(twist), strain_cost, atom_counts])
        )
    else:
        order = np.zeros(0, dtype=np.int64)
    top_matrices, bottom_matrices = top_matrices[order], bottom_matrices[order]
    top_gram, bottom_gram = top_gram[order], bottom_gram[order]
    twist, raw_twist, indices = twist[order], raw_twist[order], indices[order]
    principal_strains = principal_strains[order]
    first_stretch, second_stretch = first_stretch[order], second_stretch[order]
    top_atom_counts, bottom_atom_counts = top_atom_counts[order], bottom_atom_counts[order]
    # ``sharing`` is one constant repeated, and the total is one add: neither is
    # worth a random gather over millions of rows.
    atom_counts = top_atom_counts + bottom_atom_counts
    certified, pareto, keys = certified[order], pareto[order], keys[order]

    # 6. Reduce and orient the shared moire cell.
    top_affine, bottom_affine, shared = _affine_geometry(
        config,
        top_matrices,
        bottom_matrices,
        twist,
        first_stretch,
        second_stretch,
        sharing,
    )
    if len(top_matrices):
        s00, s01 = shared[:, 0, 0], shared[:, 0, 1]
        s10, s11 = shared[:, 1, 0], shared[:, 1, 1]
        # Only three entries of the shared Gram matrix and the sign of the
        # determinant are wanted, and both are one line for a 2x2 block.
        shared_g11 = s00 * s00 + s10 * s10
        shared_g12 = s00 * s01 + s10 * s11
        shared_g22 = s01 * s01 + s11 * s11
        determinant_sign = np.sign(s00 * s11 - s01 * s10).astype(np.int64)
        transform = _reduce_common_basis(
            shared_g11, shared_g12, shared_g22, determinant_sign
        )
        top_matrices = _matmul2(top_matrices, transform)
        bottom_matrices = _matmul2(bottom_matrices, transform)
        top_gram = _apply_gram_transform(top_gram, transform)
        bottom_gram = _apply_gram_transform(bottom_gram, transform)
        # ``keys`` needs no recomputation here: it is by construction invariant
        # under the common right action by a unimodular matrix, which is exactly
        # what ``transform`` is.  The column HNF of ``M K`` is that of ``M``, and
        # the companion ``(M K)^-1 H = K^-1 M^-1 H`` cancels the ``K`` carried by
        # the bottom matrix.  Recomputing cost a second pass of extended Euclid
        # over every candidate.
        shared = shared @ transform.astype(float)
        rotation_angle = np.arctan2(shared[:, 1, 0], shared[:, 0, 0])
        cosine, sine = np.cos(rotation_angle), np.sin(rotation_angle)
        frame = np.empty((len(shared), 2, 2))
        frame[:, 0, 0], frame[:, 0, 1] = cosine, sine
        frame[:, 1, 0], frame[:, 1, 1] = -sine, cosine
        shared = frame @ shared
        top_affine = frame @ top_affine
        bottom_affine = frame @ bottom_affine

    physical_top_gram = (top_gram * length_scale) * length_scale
    physical_bottom_gram = (bottom_gram * length_scale) * length_scale
    finalize_elapsed = time.perf_counter() - finalize_started
    stats["t_finalize"] = finalize_elapsed
    stats["t_finish"] = stats.get("t_finish", 0.0) + finalize_elapsed
    stats["t_total"] = stats.get("t_total", 0.0) + finalize_elapsed
    stats["n_accepted"] = int(len(top_matrices))
    stats["n_pareto"] = int(pareto.sum())
    stats["n_borderline"] = int((~certified).sum())
    stats["internal_length_scale"] = length_scale
    stats["angle_period_deg"] = float(np.degrees(config.angle_period_radians))
    stats["primitive_only"] = bool(config.primitive_only)
    return SearchResult(
        top_matrices=top_matrices,
        bottom_matrices=bottom_matrices,
        top_gram=physical_top_gram,
        bottom_gram=physical_bottom_gram,
        twist_radians=twist,
        twist_degrees=np.degrees(twist),
        principal_strains=principal_strains,
        sharing_fraction=sharing,
        top_atom_counts=top_atom_counts,
        bottom_atom_counts=bottom_atom_counts,
        atom_counts=atom_counts,
        loewner_certified=certified.astype(bool),
        loewner_borderline=(~certified).astype(bool),
        top_affine=top_affine,
        bottom_affine=bottom_affine,
        shared_lattice=shared,
        canonical_keys=keys,
        pareto_optimal=pareto,
        rank=np.arange(1, len(top_matrices) + 1, dtype=np.int64),
        stats=stats,
        raw_twist_radians=raw_twist,
        coincidence_indices=indices,
    )

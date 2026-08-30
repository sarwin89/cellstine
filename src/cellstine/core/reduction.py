"""Lattice-basis validation and reduction.

Everything here works with *row* bases -- ``basis[i]`` is the Cartesian vector of
the ``i``-th basis vector -- and every reduction returns the pair
``(reduced, transform)`` with ``reduced == transform @ basis`` and ``transform``
an integer matrix of determinant ``+-1``, so the reduced rows span exactly the
same lattice.  Reduction is what keeps the periodic searches in
:mod:`cellstine.core.geometry` small: in a Delaunay-reduced basis a shortest
translation has coefficients in ``{-1, 0, 1}``, and in a Lagrange--Gauss reduced
plane basis the first row is already a shortest in-plane vector.

The Niggli reduction is also the canonical cell a crystal is reported in, so it
is used by the symmetry stage as well as by the searches.

The statements behind the reductions are in ``RequestProject/NiggliCell.lean``
and ``RequestProject/LagrangeGauss.lean``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "as_lattice",
    "wrap_to_cell",
    "wrap_fractional",
    "axis_spacings",
    "reciprocal_norms",
    "niggli_reduce",
    "delaunay_reduce",
    "integer_lattice_basis",
    "rational_lattice_basis",
    "plane_form_kernel_basis",
    "gauss_reduction_multiplier",
    "plane_reduce",
    "plane_reciprocal_norms",
]


# ---------------------------------------------------------------------------
# validation and small helpers
# ---------------------------------------------------------------------------


def as_lattice(lattice: np.ndarray, name: str = "lattice") -> np.ndarray:
    """Return ``lattice`` as a validated ``(3, 3)`` array of row vectors."""

    array = np.asarray(lattice, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 row-vector lattice")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if abs(float(np.linalg.det(array))) <= 1e-12:
        raise ValueError(f"{name} vectors must be linearly independent")
    return array


def wrap_to_cell(values: np.ndarray) -> np.ndarray:
    """Return the representative of ``values`` in ``[0, 1)`` per component."""

    return np.mod(np.asarray(values, dtype=float), 1.0)


def wrap_fractional(values: np.ndarray) -> np.ndarray:
    """Return the representative of ``values`` in ``[-1/2, 1/2]`` per component.

    This is the classical "minimum image convention".  It is exact only for a
    cell whose basis is orthogonal; use :func:`minimum_image_displacements` when
    the actual shortest image is wanted.
    """

    array = np.asarray(values, dtype=float)
    return array - np.rint(array)


def reciprocal_norms(lattice: np.ndarray) -> np.ndarray:
    """Return ``‖b_i‖`` for the reciprocal rows ``b_i`` of ``inv(lattice).T``."""

    return np.linalg.norm(np.linalg.inv(as_lattice(lattice)).T, axis=1)


def axis_spacings(lattice: np.ndarray) -> np.ndarray:
    """Return the perpendicular spacing of the lattice planes normal to each axis."""

    return 1.0 / reciprocal_norms(lattice)




# ---------------------------------------------------------------------------
# three-dimensional cell reduction
# ---------------------------------------------------------------------------


def niggli_reduce(lattice: np.ndarray, eps: float | None = None, max_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Niggli-reduced cell of ``lattice`` and its integer transform.

    The returned pair ``(reduced, transform)`` satisfies
    ``reduced == transform @ lattice`` with ``transform`` unimodular, so the two
    cells describe the same lattice.  The algorithm is the Krivy-Gruber
    reduction in the numerically stable form of Grosse-Kunstleve, Sauter and
    Adams (2004); ``eps`` defaults to a relative tolerance derived from the cell
    size.
    """

    array = as_lattice(lattice)
    tolerance = float(eps) if eps is not None else 1e-5 * float(abs(np.linalg.det(array))) ** (1.0 / 3.0)
    transform = np.eye(3, dtype=np.int64)

    def metric() -> Tuple[float, float, float, float, float, float]:
        current = transform.astype(float) @ array
        gram = current @ current.T
        return (
            float(gram[0, 0]),
            float(gram[1, 1]),
            float(gram[2, 2]),
            2.0 * float(gram[1, 2]),
            2.0 * float(gram[0, 2]),
            2.0 * float(gram[0, 1]),
        )

    def apply(operation: Sequence[Sequence[int]]) -> None:
        nonlocal transform
        transform = np.asarray(operation, dtype=np.int64) @ transform

    for _ in range(int(max_steps)):
        a_value, b_value, c_value, xi, eta, zeta = metric()

        if a_value > b_value + tolerance or (
            abs(a_value - b_value) <= tolerance and abs(xi) > abs(eta) + tolerance
        ):
            apply([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            continue

        if b_value > c_value + tolerance or (
            abs(b_value - c_value) <= tolerance and abs(eta) > abs(zeta) + tolerance
        ):
            apply([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            continue

        sign_xi = 0 if abs(xi) <= tolerance else (1 if xi > 0 else -1)
        sign_eta = 0 if abs(eta) <= tolerance else (1 if eta > 0 else -1)
        sign_zeta = 0 if abs(zeta) <= tolerance else (1 if zeta > 0 else -1)

        if sign_xi * sign_eta * sign_zeta == 1:
            flip = [1 if value >= 0 else -1 for value in (sign_xi, sign_eta, sign_zeta)]
            if flip != [1, 1, 1]:
                apply(np.diag(flip))
                continue
        else:
            flip = [1, 1, 1]
            zero_axis = None
            for axis, sign in enumerate((sign_xi, sign_eta, sign_zeta)):
                if sign == 1:
                    flip[axis] = -1
                elif sign == 0:
                    zero_axis = axis
            if flip[0] * flip[1] * flip[2] < 0:
                if zero_axis is None:
                    raise RuntimeError("Niggli reduction reached an inconsistent sign state")
                flip[zero_axis] = -1
            if flip != [1, 1, 1]:
                apply(np.diag(flip))
                continue

        if (
            abs(xi) > b_value + tolerance
            or (abs(xi - b_value) <= tolerance and 2.0 * eta < zeta - tolerance)
            or (abs(xi + b_value) <= tolerance and zeta < -tolerance)
        ):
            step = 1 if xi > 0 else -1
            apply([[1, 0, 0], [0, 1, 0], [0, -step, 1]])
            continue

        if (
            abs(eta) > a_value + tolerance
            or (abs(eta - a_value) <= tolerance and 2.0 * xi < zeta - tolerance)
            or (abs(eta + a_value) <= tolerance and zeta < -tolerance)
        ):
            step = 1 if eta > 0 else -1
            apply([[1, 0, 0], [0, 1, 0], [-step, 0, 1]])
            continue

        if (
            abs(zeta) > a_value + tolerance
            or (abs(zeta - a_value) <= tolerance and 2.0 * xi < eta - tolerance)
            or (abs(zeta + a_value) <= tolerance and eta < -tolerance)
        ):
            step = 1 if zeta > 0 else -1
            apply([[1, 0, 0], [-step, 1, 0], [0, 0, 1]])
            continue

        total = a_value + b_value + xi + eta + zeta
        if total < -tolerance or (
            abs(total) <= tolerance and 2.0 * (a_value + eta) + zeta > tolerance
        ):
            apply([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
            continue

        break
    else:  # pragma: no cover - defensive, the reduction terminates in practice
        raise RuntimeError("Niggli reduction did not converge")

    reduced = transform.astype(float) @ array
    if float(np.linalg.det(reduced)) < 0.0:
        transform = -transform
        reduced = -reduced
    return reduced, transform


def delaunay_reduce(lattice: np.ndarray, eps: float | None = None, max_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Delaunay-reduced cell of ``lattice`` and its integer transform.

    The four vectors ``a``, ``b``, ``c`` and ``d = -(a + b + c)`` are made
    pairwise obtuse, and the reduced cell is the shortest linearly independent
    triple among the seven vectors ``{a, b, c, d, a+b, b+c, a+c}``.  As for
    :func:`niggli_reduce`, ``reduced == transform @ lattice``.
    """

    array = as_lattice(lattice)
    tolerance = float(eps) if eps is not None else 1e-5 * float(abs(np.linalg.det(array))) ** (2.0 / 3.0)
    basis = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]],
        dtype=np.int64,
    )

    for _ in range(int(max_steps)):
        vectors = basis.astype(float) @ array
        products = vectors @ vectors.T
        np.fill_diagonal(products, -np.inf)
        first, second = np.unravel_index(int(np.argmax(products)), products.shape)
        if float(products[first, second]) <= tolerance:
            break
        others = [index for index in range(4) if index not in (first, second)]
        for index in others:
            basis[index] = basis[index] + basis[first]
        basis[first] = -basis[first]
    else:  # pragma: no cover - defensive
        raise RuntimeError("Delaunay reduction did not converge")

    candidates = [basis[0], basis[1], basis[2], basis[3], basis[0] + basis[1], basis[1] + basis[2], basis[0] + basis[2]]
    lengths = [float(np.linalg.norm(candidate.astype(float) @ array)) for candidate in candidates]
    order = np.argsort(lengths)
    chosen: List[np.ndarray] = []
    for position in order:
        candidate = candidates[int(position)]
        trial = chosen + [candidate]
        matrix = np.asarray(trial, dtype=float)
        if len(trial) < 3:
            if np.linalg.matrix_rank(matrix) == len(trial):
                chosen.append(candidate)
        elif abs(float(np.linalg.det(matrix))) > 1e-8:
            chosen.append(candidate)
            break
    if len(chosen) != 3:  # pragma: no cover - defensive
        raise RuntimeError("Delaunay reduction failed to find an independent triple")

    transform = np.asarray(chosen, dtype=np.int64)
    if int(round(float(np.linalg.det(transform.astype(float))))) < 0:
        transform = np.array([transform[0], transform[1], -transform[2]], dtype=np.int64)
    reduced = transform.astype(float) @ array
    return reduced, transform


def integer_lattice_basis(generators: np.ndarray) -> np.ndarray:
    """Return a basis of the lattice spanned by integer row vectors.

    ``generators`` is a ``(k, 3)`` array of integers of rank three.  The three
    returned rows generate exactly the same subgroup of ``Z^3``: the routine is
    the row-style Hermite normal form, upper triangular with positive diagonal
    and every entry above a pivot reduced below it, so it is exact integer
    arithmetic and the answer never depends on the order the generators come
    in.  Picking a basis out of the generating set itself, as a search over
    triples does, is not always possible: ``(2, 0)``, ``(0, 3)`` and ``(1, 1)``
    generate the whole of ``Z^2`` while no two of them do, so a normal form is
    used instead.
    """

    rows = np.asarray(generators, dtype=np.int64).reshape(-1, 3).copy()
    if len(rows) < 3:
        raise ValueError("three or more integer generators are required")
    pivot = 0
    for column in range(3):
        while True:
            nonzero = [index for index in range(pivot, len(rows)) if rows[index, column] != 0]
            if len(nonzero) <= 1:
                break
            nonzero.sort(key=lambda index: abs(int(rows[index, column])))
            head = nonzero[0]
            for index in nonzero[1:]:
                rows[index] -= (int(rows[index, column]) // int(rows[head, column])) * rows[head]
        nonzero = [index for index in range(pivot, len(rows)) if rows[index, column] != 0]
        if not nonzero:
            raise ValueError("the generators do not span a three-dimensional lattice")
        head = nonzero[0]
        rows[[pivot, head]] = rows[[head, pivot]]
        if rows[pivot, column] < 0:
            rows[pivot] = -rows[pivot]
        for above in range(pivot):
            rows[above] -= (int(rows[above, column]) // int(rows[pivot, column])) * rows[pivot]
        pivot += 1
    return np.asarray(rows[:3], dtype=np.int64)


def rational_lattice_basis(generators: np.ndarray, denominator: int) -> np.ndarray:
    """Return a basis of a lattice given by generators with a common denominator.

    Every generator is a rational vector whose product with ``denominator`` is
    an integer vector, which is the situation for the centring translations of
    a cell: they form a group of order ``m`` above the unit translations, so
    ``m t`` is integral for each of them.  The rows returned are the exact
    basis in the same fractional coordinates.
    """

    scale = int(denominator)
    if scale <= 0:
        raise ValueError("the common denominator must be positive")
    scaled = np.asarray(generators, dtype=float).reshape(-1, 3) * float(scale)
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-6):
        raise ValueError("the generators are not multiples of 1 / denominator")
    basis = integer_lattice_basis(rounded.astype(np.int64))
    return basis.astype(float) / float(scale)



def _extended_gcd(first: int, second: int) -> Tuple[int, int, int]:
    """Return ``(g, p, q)`` with ``g = gcd(first, second) >= 0`` and ``p a + q b = g``."""

    old_r, r = int(first), int(second)
    old_p, p = 1, 0
    old_q, q = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_p, p = p, old_p - quotient * p
        old_q, q = q, old_q - quotient * q
    if old_r < 0:
        old_r, old_p, old_q = -old_r, -old_p, -old_q
    return old_r, old_p, old_q


def plane_form_kernel_basis(form: np.ndarray) -> np.ndarray:
    """Return an exact integer basis of the vectors annihilated by ``form``.

    ``form`` is a nonzero integer triple ``f``; the returned ``(2, 3)`` integer
    array spans exactly ``{m in Z^3 : m . f = 0}``, the lattice points of the
    plane the form defines.  The construction is one extended Euclid step, so it
    is exact and costs nothing, where enumerating a box of integer coefficients
    and hoping the plane vectors fall inside it is neither: a skew cell puts the
    short in-plane vectors at coefficients far outside any fixed box.

    Writing ``g = gcd(f0, f1) = p f0 + q f1`` and ``d = gcd(g, f2)``, the rows
    are ``(f1/g, -f0/g, 0)`` and ``(-p f2/d, -q f2/d, g/d)``.  Both are
    annihilated by ``f`` by construction, and every annihilated vector is an
    integer combination of them --- see ``Cellstine.Plane.exists_kernel_coords``
    in ``RequestProject/PlaneSublattice.lean``.
    """

    values = np.asarray(form, dtype=np.int64).reshape(3)
    f0, f1, f2 = (int(value) for value in values)
    if f0 == 0 and f1 == 0 and f2 == 0:
        raise ValueError("the plane form must be nonzero")
    g, p, q = _extended_gcd(f0, f1)
    if g == 0:
        # The form is (0, 0, f2): the plane lattice is spanned by the first two axes.
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
    d, _, _ = _extended_gcd(g, f2)
    return np.array(
        [
            [f1 // g, -f0 // g, 0],
            [-p * (f2 // d), -q * (f2 // d), g // d],
        ],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# plane reduction
# ---------------------------------------------------------------------------


def gauss_reduction_multiplier(dot_product: float, first_norm: float) -> int:
    """Return the shear of one Lagrange--Gauss round, with a reduced boundary.

    A round of the reduction replaces ``r1`` by ``r1 - m r0`` for the nearest
    integer ``m`` to ``(r0 . r1) / |r0|^2``, and stops once ``m`` is zero, which
    is the reduction condition ``2 |r0 . r1| <= |r0|^2``.  The boundary of that
    condition, ``2 |r0 . r1| = |r0|^2``, is reached by every 120 degree cell --
    a hexagonal surface, say -- and such a basis *is* reduced.  In floating
    point the ratio there lands a few ulps beyond ``+-1/2``, so the exact rule
    would take a step that only flips the sign of the dot product and never
    shortens anything: the rounds then oscillate for ever.  A ratio within a few
    hundred ulps of the boundary is therefore reported as reduced.

    ``RequestProject/GaussStep.lean`` backs all of it: the boundary step is the
    cycle (``Cellstine.gaussStep_boundary_involutive``), a boundary basis is
    already reduced (``Cellstine.isReducedGram_of_boundary``), any other basis
    is strictly shortened by its round (``Cellstine.gaussStep_lt_of_not_reduced``,
    so the loop only ever stops early at the boundary), and a basis accepted
    with a slack ``t`` still bounds every nonzero lattice vector by ``A - t``
    (``Cellstine.first_minimum_slack``).
    """

    scale = max(abs(dot_product), abs(first_norm), float(np.finfo(float).tiny))
    tolerance = 256.0 * float(np.finfo(float).eps) * scale
    if 2.0 * abs(dot_product) <= first_norm + tolerance:
        return 0
    return int(np.rint(dot_product / first_norm))


def plane_reduce(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lagrange--Gauss reduce a two-vector basis, in any embedding dimension.

    Returns ``(reduced, transform)`` with ``reduced = transform @ basis`` and
    ``transform`` an integer matrix of determinant +-1, so the two bases span the
    same lattice.  The reduced basis has ``|r0| <= |r1|`` and
    ``2 |r0 . r1| <= |r0|^2`` (up to the rounding width of
    :func:`gauss_reduction_multiplier`), which makes ``r0`` a shortest vector of
    the lattice and keeps every enumeration over the plane small.

    On the boundary ``2 |r0 . r1| = |r0|^2`` -- every hexagonal lattice, and so
    every hexagonal surface cell and every moire cell of a hexagonal pair -- the
    two signs of ``r1`` are *both* reduced, one describing the cell at sixty
    degrees and the other the same cell at a hundred and twenty.  Which of them
    the rounds happen to stop at is decided by the last bit of a dot product, so
    without a rule the reported cell angle of one and the same lattice flips
    between 60 and 120 degrees when an upstream sum is reassociated.  The obtuse
    one is chosen here, which is both the crystallographic convention and the
    range ``(60, 120]`` that the rest of CELLSTINE reports
    (``Cellstine.two_abs_gramInner_le_of_minima``,
    ``Cellstine.shear_gram_of_hexagonal``); away from the boundary nothing is
    changed, so an acute cell that is genuinely reduced is left alone.

    The step that crosses the boundary is ``r1 -> r1 - r0``, the very shear that
    ``Cellstine.gaussStep_boundary_involutive`` shows cycles there: it sends
    ``r0 . r1`` to ``r0 . r1 - |r0|^2`` and leaves ``|r1|`` alone, so the cell is
    the same one seen from its obtuse side, and its determinant -- the handedness
    of the pair -- is untouched, which negating ``r1`` would not be.
    """

    vectors = np.asarray(basis, dtype=float).reshape(2, -1).copy()
    transform = np.eye(2, dtype=np.int64)
    for _ in range(64):
        first = float(vectors[0] @ vectors[0])
        second = float(vectors[1] @ vectors[1])
        if first > second:
            vectors = vectors[::-1].copy()
            transform = transform[::-1].copy()
            first = second
        if first <= 0.0:
            raise ValueError("a plane basis needs two nonzero vectors")
        dot_product = float(vectors[0] @ vectors[1])
        multiplier = gauss_reduction_multiplier(dot_product, first)
        if multiplier == 0:
            slack = 256.0 * float(np.finfo(float).eps) * max(
                abs(dot_product), abs(first), float(np.finfo(float).tiny)
            )
            if 2.0 * dot_product >= first - slack:
                vectors[1] -= vectors[0]
                transform[1] -= transform[0]
            return vectors, transform
        vectors[1] -= multiplier * vectors[0]
        transform[1] -= multiplier * transform[0]
    raise ArithmeticError("Lagrange--Gauss reduction of the plane basis did not converge")


def plane_reciprocal_norms(basis: np.ndarray) -> np.ndarray:
    """Return ``1 / d_i`` for a plane basis, ``d_i`` the spacing of its rows of points.

    The reciprocal vectors of a two-dimensional lattice embedded in space are
    the columns of the pseudo-inverse of the basis, since that is the unique
    matrix with ``basis @ pinv(basis) = I`` whose columns lie in the plane.
    """

    array = np.asarray(basis, dtype=float).reshape(2, -1)
    return np.linalg.norm(np.linalg.pinv(array), axis=0)


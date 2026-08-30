"""Planar (2D) point-group utilities shared by the moire and interface stages.

Everything here works with *column* in-plane bases: ``basis`` is the 2x2 matrix
whose columns are the Cartesian in-plane lattice vectors, so a lattice point with
integer coefficients ``x`` sits at ``basis @ x``.  A symmetry operation is an
integer matrix ``G`` with ``basis @ G == R @ basis`` for an orthogonal ``R``;
equivalently ``G.T @ metric @ G == metric`` with ``metric = basis.T @ basis``.

Two groups are distinguished:

``lattice_point_group``
    all integer operations of the bare 2D lattice;
``layer_point_group``
    the subgroup that also maps a decorated layer (atom positions and species)
    onto itself, allowing an arbitrary in-plane translation and keeping every
    Cartesian ``z`` coordinate fixed.

The decorated group is what makes twist angles physical: two twisted-bilayer
stackings are equivalent only under operations that leave each *layer* -- not
merely each lattice -- invariant.  Using the bare lattice group for a layer such
as monolayer hBN or MoS2 (three-fold, not six-fold) silently merges genuinely
different stackings.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_SYMMETRY_TOLERANCE",
    "column_basis_from_lattice",
    "close_group",
    "lattice_point_group",
    "layer_point_group",
    "proper_subgroup",
    "group_has_mirror",
    "rotation_order",
    "cartesian_rotation_angles",
    "cartesian_mirror_angles",
    "equivalence_period_radians",
    "symmetrised_basis",
    "idealised_layer_lattice",
]

#: Relative metric tolerance used when detecting planar point groups.
#:
#: The value plays the role of spglib's ``symprec``: it is the fractional
#: deviation of the Gram matrix that still counts as a symmetry.  It must be
#: loose enough to survive the finite precision of a real POSCAR -- a cell
#: printed with six decimal places already deviates by ~1e-7 relative, which a
#: machine-epsilon tolerance rejects.
from .constants import DEFAULT_SYMMETRY_TOLERANCE  # re-exported for callers
from .idealisation import close_matrix_group, symmetrise_basis


def column_basis_from_lattice(lattice: np.ndarray, *, name: str = "lattice") -> np.ndarray:
    """Return the in-plane Cartesian column basis of a row-vector 3x3 lattice."""

    array = np.asarray(lattice, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 row-vector lattice")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    scale = max(float(np.max(np.abs(array[:2]))), 1.0)
    if float(np.max(np.abs(array[:2, 2]))) > 1e-9 * scale:
        raise ValueError(f"{name} a/b vectors must be planar in Cartesian xy")
    basis = np.array(array[:2, :2].T, dtype=float)
    if abs(float(np.linalg.det(basis))) <= 1e-12 * scale * scale:
        raise ValueError(f"{name} in-plane vectors must be linearly independent")
    return basis


def _short_vectors(metric: np.ndarray, radius_squared: float) -> np.ndarray:
    g11, g12, g22 = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])
    determinant = g11 * g22 - g12 * g12
    if determinant <= 0.0:
        raise ValueError("metric must be positive definite")
    m_max = int(math.floor(math.sqrt(max(radius_squared, 0.0) * g22 / determinant))) + 1
    n_max = int(math.floor(math.sqrt(max(radius_squared, 0.0) * g11 / determinant))) + 1
    m, n = np.meshgrid(
        np.arange(-m_max, m_max + 1, dtype=np.int64),
        np.arange(-n_max, n_max + 1, dtype=np.int64),
        indexing="ij",
    )
    m, n = m.ravel(), n.ravel()
    squared = g11 * m * m + 2.0 * g12 * m * n + g22 * n * n
    keep = (squared <= radius_squared) & ~((m == 0) & (n == 0))
    return np.stack([m[keep], n[keep]], axis=1)


def lattice_point_group(
    basis: np.ndarray, tolerance: float = DEFAULT_SYMMETRY_TOLERANCE
) -> np.ndarray:
    """Return every integer operation of the 2D lattice as an ``(k, 2, 2)`` array.

    ``tolerance`` is relative and applies to the *metric*: a candidate operation
    is accepted when it reproduces every metric entry to within
    ``tolerance * max(g11, g22)``.  A relative length error ``d`` in the input
    cell perturbs the metric by about ``2 d``, so the default
    ``DEFAULT_SYMMETRY_TOLERANCE`` accepts cells that are correct to roughly five
    significant digits -- the accuracy of a POSCAR written with the customary six
    decimal places, or of a relaxed DFT cell.  Machine-epsilon tolerances silently
    reject those cells and cost the search its symmetry folding.

    Two facts about the search are proved in
    ``aristotle-lean-reference/RequestProject/PlanarPointGroup.lean``: the three scalar equations tested
    here are exactly ``G.T @ metric @ G == metric``
    (``Cellstine.gram_preserving_iff_columns``), and the integer box
    :func:`_short_vectors` sweeps contains every vector inside the search ellipse
    (``Cellstine.planarForm_box_bound``), so no candidate column is missed.  The
    ``|det| == 1`` filter is automatic in exact arithmetic
    (``Cellstine.det_sq_eq_one_of_gram_preserving``); it is applied because the
    metric comparison here is tolerant.
    """

    basis_array = np.asarray(basis, dtype=float)
    if basis_array.shape != (2, 2):
        raise ValueError("basis must be a 2x2 Cartesian column basis")
    metric = basis_array.T @ basis_array
    g11, g12, g22 = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])
    scale = max(g11, g22)
    vectors = _short_vectors(metric, scale * (1.0 + 1000.0 * tolerance))
    if len(vectors) == 0:
        return np.eye(2, dtype=np.int64)[None, :, :]
    squared = (
        g11 * vectors[:, 0] ** 2
        + 2.0 * g12 * vectors[:, 0] * vectors[:, 1]
        + g22 * vectors[:, 1] ** 2
    )
    first = np.nonzero(np.abs(squared - g11) <= tolerance * scale)[0]
    second = np.nonzero(np.abs(squared - g22) <= tolerance * scale)[0]
    if first.size == 0 or second.size == 0:
        return np.eye(2, dtype=np.int64)[None, :, :]
    left_index, right_index = np.meshgrid(first, second, indexing="ij")
    left, right = vectors[left_index.ravel()], vectors[right_index.ravel()]
    cross = (
        g11 * left[:, 0] * right[:, 0]
        + g12 * (left[:, 0] * right[:, 1] + left[:, 1] * right[:, 0])
        + g22 * left[:, 1] * right[:, 1]
    )
    determinant = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    keep = (np.abs(cross - g12) <= tolerance * scale) & (np.abs(determinant) == 1)
    group = np.stack([left[keep], right[keep]], axis=2).astype(np.int64)
    return group if len(group) else np.eye(2, dtype=np.int64)[None, :, :]


def proper_subgroup(group: np.ndarray) -> np.ndarray:
    """Return the orientation-preserving elements of an integer point group."""

    array = np.asarray(group, dtype=np.int64)
    determinant = array[:, 0, 0] * array[:, 1, 1] - array[:, 0, 1] * array[:, 1, 0]
    proper = array[determinant > 0]
    return proper if len(proper) else np.eye(2, dtype=np.int64)[None, :, :]


def group_has_mirror(group: np.ndarray) -> bool:
    """Return whether an integer point group contains an orientation reversal."""

    array = np.asarray(group, dtype=np.int64)
    determinant = array[:, 0, 0] * array[:, 1, 1] - array[:, 0, 1] * array[:, 1, 0]
    return bool(np.any(determinant < 0))


def rotation_order(group: np.ndarray) -> int:
    """Return the order of the cyclic rotation subgroup of an integer point group."""

    return int(len(proper_subgroup(group)))


def cartesian_rotation_angles(basis: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Return the Cartesian rotation angle of each proper element, in radians."""

    basis_array = np.asarray(basis, dtype=float)
    inverse = np.linalg.inv(basis_array)
    angles = []
    for element in np.asarray(group, dtype=float):
        rotation = basis_array @ element @ inverse
        angles.append(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    return np.asarray(angles, dtype=float)


def cartesian_mirror_angles(basis: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Return ``2 * alpha`` for each reflection, where ``alpha`` is its axis angle."""

    basis_array = np.asarray(basis, dtype=float)
    inverse = np.linalg.inv(basis_array)
    angles = []
    for element in np.asarray(group, dtype=float):
        reflection = basis_array @ element @ inverse
        angles.append(math.atan2(float(reflection[1, 0]), float(reflection[0, 0])))
    return np.asarray(angles, dtype=float)


def equivalence_period_radians(top_order: int, bottom_order: int) -> float:
    """Return the twist-angle period implied by two cyclic layer symmetries."""

    order = math.lcm(max(int(top_order), 1), max(int(bottom_order), 1))
    return 2.0 * math.pi / float(order)


def _wrap(values: np.ndarray) -> np.ndarray:
    return values - np.round(values)


def layer_point_group(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    position_tolerance: float = 1e-4,
    z_tolerance: float = 1e-3,
    lattice_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    name: str = "layer",
) -> np.ndarray:
    """Return the integer point group of a decorated planar layer.

    Only in-plane operations that keep every Cartesian ``z`` coordinate fixed are
    considered, and an arbitrary in-plane translation is allowed, so the six-fold
    rotation of graphene about a hexagon centre is detected even though no atom
    sits there.  The result is a subgroup of :func:`lattice_point_group`.
    """

    basis = column_basis_from_lattice(lattice, name=name)
    group = lattice_point_group(basis, tolerance=lattice_tolerance)
    fractional = np.asarray(positions_direct, dtype=float)
    if fractional.ndim != 2 or fractional.shape[1] != 3:
        raise ValueError(f"{name} positions must be an (n, 3) direct-coordinate array")
    if fractional.shape[0] == 0:
        return group
    planar = fractional[:, :2]
    cartesian_z = fractional @ np.asarray(lattice, dtype=float)
    heights = cartesian_z[:, 2]
    if species is None:
        labels = np.zeros(len(planar), dtype=np.int64)
    else:
        symbols = [str(value) for value in species]
        if len(symbols) != len(planar):
            raise ValueError(f"{name} species labels must match the position count")
        unique = {symbol: index for index, symbol in enumerate(sorted(set(symbols)))}
        labels = np.array([unique[symbol] for symbol in symbols], dtype=np.int64)

    accepted = []
    for element in group:
        mapped = planar @ np.asarray(element, dtype=float).T
        if _accepts_translation(
            mapped, planar, labels, heights, position_tolerance, z_tolerance
        ):
            accepted.append(element)
    if not accepted:
        return np.eye(2, dtype=np.int64)[None, :, :]
    return np.asarray(accepted, dtype=np.int64)


def _accepts_translation(
    mapped: np.ndarray,
    planar: np.ndarray,
    labels: np.ndarray,
    heights: np.ndarray,
    position_tolerance: float,
    z_tolerance: float,
) -> bool:
    """Return whether some in-plane translation completes an operation."""

    reference = 0
    partners = np.nonzero(
        (labels == labels[reference]) & (np.abs(heights - heights[reference]) <= z_tolerance)
    )[0]
    for partner in partners:
        translation = planar[partner] - mapped[reference]
        shifted = mapped + translation
        difference = _wrap(shifted[:, None, :] - planar[None, :, :])
        close = np.all(np.abs(difference) <= position_tolerance, axis=2)
        close &= labels[:, None] == labels[None, :]
        close &= np.abs(heights[:, None] - heights[None, :]) <= z_tolerance
        if np.all(close.any(axis=1)) and np.all(close.any(axis=0)):
            return True
    return False


def close_group(group: np.ndarray) -> np.ndarray:
    """Return the multiplicative closure of a set of integer 2x2 operations.

    Detection returns the operations that individually preserve the metric; the
    closure is what averaging arguments need, and in two dimensions it is at most
    the twelve-element hexagonal holohedry, so the fixed-point iteration is cheap.

    The crystallographic restriction behind that bound is proved in
    ``aristotle-lean-reference/RequestProject/PlanarPointGroup.lean``: a rotation preserving a positive
    definite planar metric has ``|trace| <= 2``
    (``Cellstine.planar_trace_sq_le_four``) and an integer trace, so its trace is
    one of five values (``Cellstine.int_trace_mem_of_sq_le_four``) and its order
    is 1, 2, 3, 4 or 6.  There is no five-fold axis for the twist folding to miss.

    This is :func:`core.idealisation.close_matrix_group` pinned to the plane; the
    same routine closes the three-dimensional point groups of
    ``core/bravais.py``.
    """

    return close_matrix_group(group, dimension=2, max_order=12, name="planar point group")


def symmetrised_basis(
    basis: np.ndarray, group: np.ndarray, *, name: str = "lattice"
) -> tuple[np.ndarray, float]:
    """Return a planar basis whose metric is *exactly* invariant under ``group``.

    This is :func:`core.idealisation.symmetrise_basis` pinned to the plane; see
    that module for what the group average guarantees and why the average, and
    not the nearest invariant metric, is the right thing to compute.

    Returns the idealised basis and the largest relative change of any metric
    entry, which callers record so users can see how far the input was from ideal.
    """

    basis_array = np.asarray(basis, dtype=float)
    if basis_array.shape != (2, 2):
        raise ValueError(f"{name} basis must be a 2x2 Cartesian column basis")
    return symmetrise_basis(basis_array, group, max_order=12, name=name)


def idealised_layer_lattice(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    name: str = "layer",
) -> tuple[np.ndarray, float]:
    """Return a 3x3 row lattice whose in-plane block carries the layer symmetry.

    This is :func:`symmetrised_basis` applied to the group of the *decorated*
    layer and written back in the row-vector convention a POSCAR uses.  The out
    of plane vector is copied through unchanged, and the second return value is
    the relative metric change, so callers can report how far the input cell was
    from the ideal one.
    """

    basis = column_basis_from_lattice(lattice, name=name)
    group = layer_point_group(
        lattice,
        positions_direct,
        species,
        lattice_tolerance=float(tolerance),
        name=name,
    )
    idealised, deviation = symmetrised_basis(basis, group, name=name)
    output = np.array(lattice, dtype=float, copy=True)
    output[:2, :2] = idealised.T
    return output, deviation

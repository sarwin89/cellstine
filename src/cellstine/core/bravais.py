"""Bravais classification and the conventional cell of a lattice.

A lattice is given as always by *row* vectors, ``lattice[i]`` being the
Cartesian vector of basis vector ``i``.  Whatever basis it is handed, the
lattice itself has a well defined symmetry group, and that group is what this
module reads the classification off:

* the point group of the lattice -- its holohedry -- fixes the crystal family,
  because the holohedries of the seven families are the seven groups
  ``-1``, ``2/m``, ``mmm``, ``4/mmm``, ``-3m``, ``6/mmm``, ``m-3m``;
* the rotation axes of that group fix the *directions* of the conventional
  axes, since the conventional cell of every family is the one aligned with its
  symmetry axes;
* the conventional cell itself is then built from the shortest lattice vectors
  along (and, where the family needs it, perpendicular to) those directions;
* the centring letter is read off the lattice points that fall inside the
  conventional cell -- the index of the conventional lattice in the given one is
  ``1`` for ``P``, ``2`` for ``I`` or a base centring, ``3`` for ``R`` and ``4``
  for ``F``, and the fractional coordinates of the extra points say which.

Nothing here is a table look-up on cell parameters, which is what makes it safe
on a cell that is skewed, permuted, or simply not in a standard setting: the
answer is derived from the symmetry of the lattice, so it cannot disagree with
the symmetry that the rest of CELLSTINE reports.

Detection is *tolerant* and construction is *exact*, and the two have to be
reconciled before they are combined.  A conventional axis is built by rotating a
lattice vector onto another symmetry direction, and on a cell that is hexagonal
only to within the detection tolerance -- which is every cell written to the six
decimal places a POSCAR carries -- the rotated vector is not quite a lattice
vector, so the centring step finds a coset that should not exist and the whole
classification fails.  The lattice is therefore idealised first
(:func:`core.idealisation.symmetrise_basis`): its metric is averaged over the
detected point group, which makes that group an exact symmetry
(``Cellstine.averageMetric_invariant``), leaves an already ideal cell alone
(``Cellstine.averageMetric_eq_self``) and moves no metric entry further than the
group orbit already does (``Cellstine.averageMetric_sub_apply_le``).
``ConventionalCell.deviation`` reports how far the input was from ideal, so the
correction is never silent.

The conventional cell is what gives high-symmetry k-points their familiar
names; see :mod:`cellstine.core.kpath`.

The three steps are backed by machine-checked statements in
``aristotle-lean-reference/RequestProject/ConventionalCell.lean``: ``Cellstine.card_preservesGram_eq``
(a change of basis matches the two point groups one for one, so the holohedry
and the crystal family are properties of the lattice and not of the setting),
``Cellstine.card_cosets_mul_abs_det`` (the volume ratio this module rounds to an
integer is the number of lattice points per conventional cell),
``Cellstine.twelve_mul_coords_isInt`` (for an index of at most four the
conventional coordinates of a lattice point are exact twelfths, so the integer
key ``round(12 x)`` in :func:`_centring_from_cosets` loses nothing), and
``Cellstine.exists_primitive_axis_generator`` (the shortest lattice vector along
a symmetry axis divides every lattice vector along it, so
:func:`_shortest_along` returns a generator of the axial sublattice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .idealisation import symmetrise_basis
from .reduction import as_lattice, delaunay_reduce, niggli_reduce, plane_reduce
from .symmetry3d import crystal_system_of_point_group, lattice_point_group, point_group_symbol, rotation_type

__all__ = [
    "ConventionalCell",
    "conventional_cell",
    "bravais_symbol",
]


_FAMILY_LETTER = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}


@dataclass(frozen=True)
class ConventionalCell:
    """The conventional cell of a lattice, with its Bravais classification.

    ``cell`` holds the conventional basis as Cartesian rows.  Its cell has
    ``multiplicity`` times the volume of the given one, and
    ``centring_vectors`` lists the fractional coordinates, in the conventional
    basis, of the lattice points inside that cell -- ``(0, 0, 0)`` alone for a
    primitive setting.

    ``lattice`` is the *idealised* basis everything here is derived from: the
    given basis with its metric averaged over the detected point group (see the
    module docstring), which is the given basis itself when that group is already
    an exact symmetry.  ``deviation`` is the largest relative change of any
    metric entry that averaging made.

    ``to_conventional`` converts fractional coordinates: a point with
    coordinates ``x`` in the given basis has coordinates ``x @ to_conventional``
    in the conventional one.  ``to_primitive`` is its inverse.
    """

    symbol: str
    system: str
    centring: str
    cell: np.ndarray
    lattice: np.ndarray
    multiplicity: int
    centring_vectors: np.ndarray
    to_conventional: np.ndarray
    to_primitive: np.ndarray
    point_group: str | None
    deviation: float = 0.0

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return ``(a, b, c, alpha, beta, gamma)`` of the conventional cell."""

        lengths = np.linalg.norm(self.cell, axis=1)
        angles = []
        for first, second in ((1, 2), (0, 2), (0, 1)):
            cosine = float(self.cell[first] @ self.cell[second]) / float(lengths[first] * lengths[second])
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
        return (
            float(lengths[0]),
            float(lengths[1]),
            float(lengths[2]),
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
        )

    def summary(self) -> dict[str, object]:
        """Return a JSON-ready description of the classification."""

        a, b, c, alpha, beta, gamma = self.parameters
        return {
            "bravais_symbol": self.symbol,
            "crystal_system": self.system,
            "centring": self.centring,
            "lattice_point_group": self.point_group,
            "multiplicity": self.multiplicity,
            "conventional_cell": [[float(value) for value in row] for row in self.cell],
            "conventional_parameters": {
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            },
            "centring_vectors": [[float(value) for value in row] for row in self.centring_vectors],
            "idealisation_deviation": float(self.deviation),
        }


def _cartesian_rotations(lattice: np.ndarray, operations: np.ndarray) -> np.ndarray:
    """Return the Cartesian matrices of column-fractional operations ``W``."""

    basis = lattice.T
    inverse = np.linalg.inv(basis)
    return np.einsum("ij,kjl,lm->kim", basis, operations.astype(float), inverse)


def _rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """Return the unit axis of a proper Cartesian rotation."""

    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, index])
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:  # pragma: no cover - defensive
        raise RuntimeError("a rotation has no axis")
    axis = axis / norm
    if axis[np.argmax(np.abs(axis))] < 0.0:
        axis = -axis
    return axis


def _axes_of_order(lattice: np.ndarray, operations: np.ndarray, order: int) -> list[np.ndarray]:
    """Return the distinct axes of the proper rotations of the given order."""

    axes: list[np.ndarray] = []
    for operation, cartesian in zip(operations, _cartesian_rotations(lattice, operations)):
        if rotation_type(operation) != order:
            continue
        axis = _rotation_axis(cartesian)
        if not any(abs(abs(float(axis @ other)) - 1.0) < 1e-6 for other in axes):
            axes.append(axis)
    return axes


def _lattice_points(lattice: np.ndarray, radius: int = 4) -> np.ndarray:
    """Return the short Cartesian lattice points of ``lattice``.

    The shell is taken in a Delaunay-reduced basis of the same lattice, so a
    small ``radius`` already covers every vector short enough to be a
    conventional axis, whatever basis the caller happened to hand in.
    """

    reduced, _ = delaunay_reduce(lattice)
    span = np.arange(-radius, radius + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    return grid.astype(float) @ reduced


def _shortest_along(lattice: np.ndarray, axis: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Return the shortest lattice vector parallel to ``axis``."""

    points = _lattice_points(lattice)
    lengths = np.linalg.norm(points, axis=1)
    projections = points @ axis
    parallel = np.abs(np.abs(projections) - lengths) <= tolerance * np.maximum(lengths, 1.0)
    forward = parallel & (projections > 0.0)
    if not np.any(forward):  # pragma: no cover - defensive
        raise RuntimeError("no lattice vector lies along a symmetry axis")
    return points[forward][int(np.argmin(lengths[forward]))]


def _shortest_perpendicular(lattice: np.ndarray, axis: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Return the shortest lattice vector perpendicular to ``axis``."""

    points = _lattice_points(lattice)
    lengths = np.linalg.norm(points, axis=1)
    perpendicular = np.abs(points @ axis) <= tolerance * np.maximum(lengths, 1.0)
    if not np.any(perpendicular):  # pragma: no cover - defensive
        raise RuntimeError("no lattice vector lies perpendicular to a symmetry axis")
    candidates = points[perpendicular]
    return candidates[int(np.argmin(lengths[perpendicular]))]


def _rotate(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Return ``vector`` rotated about ``axis`` by ``degrees`` (right-hand rule)."""

    angle = math.radians(degrees)
    unit = axis / float(np.linalg.norm(axis))
    return (
        vector * math.cos(angle)
        + np.cross(unit, vector) * math.sin(angle)
        + unit * float(unit @ vector) * (1.0 - math.cos(angle))
    )


def _centring_from_cosets(lattice: np.ndarray, cell: np.ndarray) -> tuple[str, np.ndarray, int]:
    """Return the centring letter, its vectors, and the index of ``cell``."""

    multiplicity = int(round(abs(float(np.linalg.det(cell))) / abs(float(np.linalg.det(lattice)))))
    if multiplicity < 1:  # pragma: no cover - defensive
        raise RuntimeError("the conventional cell is smaller than the primitive one")
    inverse = np.linalg.inv(cell)
    reduced, _ = delaunay_reduce(lattice)
    span = np.arange(-2 * multiplicity, 2 * multiplicity + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    fractional = (grid.astype(float) @ reduced) @ inverse
    wrapped = fractional - np.floor(fractional + 1e-8)
    wrapped[np.abs(wrapped) < 1e-8] = 0.0
    keys = np.round(wrapped * 12.0).astype(np.int64)
    if not np.allclose(keys / 12.0, wrapped, atol=1e-6):
        raise RuntimeError("the conventional cell does not contain the lattice as a subgroup")
    unique = np.unique(keys, axis=0)
    if len(unique) != multiplicity:  # pragma: no cover - defensive
        raise RuntimeError("the conventional cell index does not match its coset count")
    vectors = unique.astype(float) / 12.0
    order = np.lexsort((vectors[:, 2], vectors[:, 1], vectors[:, 0]))
    vectors = vectors[order]

    if multiplicity == 1:
        return "P", vectors, multiplicity
    if multiplicity == 2:
        extra = tuple(int(value) for value in unique[np.any(unique != 0, axis=1)][0])
        if extra == (6, 6, 6):
            return "I", vectors, multiplicity
        if extra == (0, 6, 6):
            return "A", vectors, multiplicity
        if extra == (6, 0, 6):
            return "B", vectors, multiplicity
        if extra == (6, 6, 0):
            return "C", vectors, multiplicity
        raise RuntimeError(f"unrecognised centring vector {np.asarray(extra) / 12.0}")
    if multiplicity == 3:
        return "R", vectors, multiplicity
    if multiplicity == 4:
        return "F", vectors, multiplicity
    raise RuntimeError(f"unrecognised conventional-cell index {multiplicity}")


def _orthogonal_cell(lattice: np.ndarray, axes: Sequence[np.ndarray]) -> np.ndarray:
    """Return the conventional cell built from shortest vectors along ``axes``."""

    vectors = [_shortest_along(lattice, axis) for axis in axes]
    order = np.argsort([float(np.linalg.norm(vector)) for vector in vectors], kind="stable")
    cell = np.asarray([vectors[int(index)] for index in order], dtype=float)
    if float(np.linalg.det(cell)) < 0.0:
        cell = np.asarray([cell[0], cell[1], -cell[2]], dtype=float)
    return cell


def _axial_cell(lattice: np.ndarray, axis: np.ndarray, turn: float) -> np.ndarray:
    """Return the conventional cell of an axial family.

    ``turn`` is the angle from ``a1`` to ``a2`` about ``axis`` -- ninety degrees
    for a tetragonal cell and a hundred and twenty for a hexagonal or
    rhombohedral one.  For a rhombohedral lattice the shortest vector along the
    three-fold axis and the shortest one perpendicular to it already span the
    triple hexagonal cell, so no separate stacking is needed.
    """

    plane = _shortest_perpendicular(lattice, axis)
    third = _shortest_along(lattice, axis)
    if float(third @ axis) < 0.0:  # pragma: no cover - defensive
        third = -third
    second = _rotate(plane, axis, turn)
    cell = np.asarray([plane, second, third], dtype=float)
    if float(np.linalg.det(cell)) < 0.0:
        cell = np.asarray([plane, _rotate(plane, axis, -turn), third], dtype=float)
    return cell


def _monoclinic_cell(lattice: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return the b-unique conventional cell of a monoclinic lattice."""

    unique = _shortest_along(lattice, axis)
    points = _lattice_points(lattice)
    lengths = np.linalg.norm(points, axis=1)
    inplane = points[np.abs(points @ axis) <= 1e-6 * np.maximum(lengths, 1.0)]
    order = np.argsort(np.linalg.norm(inplane, axis=1), kind="stable")
    inplane = inplane[order]
    first = inplane[0]
    second = None
    for candidate in inplane[1:]:
        if float(np.linalg.norm(np.cross(first, candidate))) > 1e-8 * float(np.linalg.norm(first)):
            second = candidate
            break
    if second is None:  # pragma: no cover - defensive
        raise RuntimeError("the plane perpendicular to the unique axis is not two-dimensional")
    reduced, _ = plane_reduce(np.asarray([first, second], dtype=float))
    cell = np.asarray([reduced[0], unique, reduced[1]], dtype=float)
    if float(np.linalg.det(cell)) < 0.0:
        cell = np.asarray([reduced[0], unique, -reduced[1]], dtype=float)
    return cell


def _standard_centring(cell: np.ndarray, centring: str, system: str) -> np.ndarray:
    """Return ``cell`` re-indexed so that a base centring is the standard one.

    A base-centred lattice can be written with the centred face on any of the
    three pairs of axes; the standard choice is the ``a``-``b`` face, letter
    ``C``.  For an orthorhombic cell the fix is a cyclic permutation, which
    keeps the basis right handed; for a monoclinic cell the unique axis has to
    stay second, so ``a`` and ``c`` are exchanged and one of them negated.
    """

    if centring not in ("A", "B"):
        return cell
    if system == "monoclinic":
        if centring == "B":  # pragma: no cover - defensive
            raise RuntimeError("a b-unique monoclinic cell cannot be B-centred")
        return np.asarray([cell[2], cell[1], -cell[0]], dtype=float)
    order = [1, 2, 0] if centring == "A" else [2, 0, 1]
    return np.asarray([cell[index] for index in order], dtype=float)


def _sort_base_centred(cell: np.ndarray) -> np.ndarray:
    """Return a C-centred orthorhombic cell with ``a <= b``."""

    if float(np.linalg.norm(cell[0])) <= float(np.linalg.norm(cell[1])):
        return cell
    return np.asarray([cell[1], cell[0], -cell[2]], dtype=float)


def _obtuse_monoclinic(cell: np.ndarray) -> np.ndarray:
    """Return a b-unique monoclinic cell whose ``beta`` is not acute.

    The two settings ``(a, b, c)`` and ``(-a, -b, c)`` describe the same lattice
    with the same unique axis and the same centring, and their ``beta`` angles
    add up to a hundred and eighty degrees.  The standard setting is the one
    with ``beta >= 90``, so that is the one reported; negating two of the three
    vectors keeps the basis right handed.
    """

    first, third = cell[0], cell[2]
    cosine = float(first @ third)
    if cosine <= 1e-9 * float(np.linalg.norm(first) * np.linalg.norm(third)):
        return cell
    return np.asarray([-cell[0], -cell[1], cell[2]], dtype=float)


def conventional_cell(lattice: Sequence[Sequence[float]], *, tolerance: float = 1e-5) -> ConventionalCell:
    """Return the Bravais classification and conventional cell of ``lattice``."""

    given = as_lattice(np.asarray(lattice, dtype=float), "lattice")
    operations = lattice_point_group(given, tolerance=tolerance)
    symbol = point_group_symbol(operations)
    system = crystal_system_of_point_group(symbol)
    if system is None:  # pragma: no cover - defensive
        raise RuntimeError("the lattice point group is not crystallographic")

    # Make the detected group an exact symmetry before any axis is rotated onto
    # another; see the module docstring.  ``lattice_point_group`` returns
    # operations on column fractional coordinates, which is the convention
    # ``symmetrise_basis`` expects, and the columns of ``given.T`` are the
    # lattice vectors.
    idealised, deviation = symmetrise_basis(given.T, operations, max_order=48, name="lattice")
    basis = np.array(idealised.T, dtype=float)

    if system == "cubic":
        axes = _axes_of_order(basis, operations, 4)
        cell = _orthogonal_cell(basis, axes[:3])
    elif system == "tetragonal":
        axis = _axes_of_order(basis, operations, 4)[0]
        cell = _axial_cell(basis, axis, 90.0)
    elif system == "hexagonal":
        axis = _axes_of_order(basis, operations, 6)[0]
        cell = _axial_cell(basis, axis, 120.0)
    elif system == "trigonal":
        axis = _axes_of_order(basis, operations, 3)[0]
        cell = _axial_cell(basis, axis, 120.0)
    elif system == "orthorhombic":
        axes = _axes_of_order(basis, operations, 2)
        cell = _orthogonal_cell(basis, axes[:3])
    elif system == "monoclinic":
        axes = _axes_of_order(basis, operations, 2)
        cell = _monoclinic_cell(basis, axes[0])
    else:
        cell, _ = niggli_reduce(basis)

    centring, vectors, multiplicity = _centring_from_cosets(basis, cell)
    if centring in ("A", "B"):
        cell = _standard_centring(cell, centring, system)
        if system == "orthorhombic":
            cell = _sort_base_centred(cell)
        centring, vectors, multiplicity = _centring_from_cosets(basis, cell)
        if centring != "C":  # pragma: no cover - defensive
            raise RuntimeError("the base centring could not be brought to the standard setting")
    if system == "monoclinic":
        cell = _obtuse_monoclinic(cell)
        centring, vectors, multiplicity = _centring_from_cosets(basis, cell)
    letter = _FAMILY_LETTER[system]
    if system == "trigonal" and centring != "R":  # pragma: no cover - defensive
        raise RuntimeError("a trigonal lattice must be rhombohedrally centred")

    to_conventional = basis @ np.linalg.inv(cell)
    to_primitive = cell @ np.linalg.inv(basis)
    return ConventionalCell(
        symbol=f"{letter}{centring}",
        system=system,
        centring=centring,
        cell=cell,
        lattice=basis,
        multiplicity=multiplicity,
        centring_vectors=vectors,
        to_conventional=to_conventional,
        to_primitive=to_primitive,
        point_group=symbol,
        deviation=float(deviation),
    )


def bravais_symbol(lattice: Sequence[Sequence[float]], *, tolerance: float = 1e-5) -> str:
    """Return the two-letter Bravais symbol of ``lattice``, such as ``cF``."""

    return conventional_cell(lattice, tolerance=tolerance).symbol

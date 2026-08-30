"""The point group of a three-dimensional lattice.

Split out of :mod:`cellstine.core.symmetry3d`, which re-exports everything
here, so that the two halves -- the point group of a bare lattice and the
space-group operations of a decorated cell -- stay separately readable.

A lattice is given as *rows*: ``lattice[i]`` is the Cartesian vector of the
``i``-th basis vector.  An integer matrix ``W`` is an automorphism of it
exactly when it preserves the metric ``G = lattice @ lattice.T``, i.e.
``W.T @ G @ W == G``, because the induced Cartesian map ``R = A W A^-1`` (with
``A = lattice.T``) is then orthogonal.  The search is run in a Niggli-reduced
basis, where the basis vectors are short so only a small box of integer
vectors has to be enumerated, and the result is conjugated back to the input
basis.  Everything is exact integer arithmetic apart from the metric
comparisons, which use a length tolerance.

The claims this rests on are proved in Lean in
``RequestProject/LatticeAutomorphisms.lean``: metric preservation is exactly
orthogonality of the induced Cartesian map
(``Cellstine.preservesGram_iff_orthogonal``), the column test below is that
condition (``Cellstine.preservesGram_iff_columns``), the enumeration box of
:func:`_integer_vectors_within` contains every vector of bounded metric length
(``Cellstine.abs_coord_le_sqrt_gram_inv_diag``), and searching in the reduced
basis and conjugating back returns the same group (``Cellstine.gram_mul``,
``Cellstine.preservesGram_conj``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import as_lattice as _as_lattice, niggli_reduce

__all__ = [
    "lattice_point_group",
    "rotation_type",
    "point_group_symbol",
    "crystal_system_of_point_group",
]

# ---------------------------------------------------------------------------
# lattice point group
# ---------------------------------------------------------------------------


def _integer_vectors_within(metric: np.ndarray, radius_squared: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the integer vectors ``n`` with ``n G n <= radius_squared`` and their norms."""

    inverse = np.linalg.inv(metric)
    limits = np.sqrt(np.maximum(radius_squared * np.diag(inverse), 0.0))
    ranges = [np.arange(-int(np.floor(limit)) - 1, int(np.floor(limit)) + 2, dtype=np.int64) for limit in limits]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    norms = np.einsum("ij,jk,ik->i", grid.astype(float), metric, grid.astype(float))
    keep = norms <= radius_squared
    return grid[keep], norms[keep]


def lattice_point_group(lattice: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Return every integer automorphism of ``lattice`` as a ``(k, 3, 3)`` array.

    Each returned ``W`` acts on *column* fractional coordinates of the input
    lattice and satisfies ``W.T @ G @ W == G`` with ``G = lattice @ lattice.T``.
    The group is a finite crystallographic point group, so ``k`` is at most 48.
    """

    array = _as_lattice(lattice)
    reduced, transform = niggli_reduce(array)
    metric = reduced @ reduced.T
    scale = float(np.max(np.abs(np.diag(metric))))
    length_tolerance = float(tolerance) * scale

    vectors, norms = _integer_vectors_within(metric, scale * (1.0 + 1e-6) + length_tolerance)
    columns: List[np.ndarray] = []
    for axis in range(3):
        target = float(metric[axis, axis])
        matches = vectors[np.abs(norms - target) <= length_tolerance]
        if len(matches) == 0:
            raise RuntimeError("no lattice vector reproduces a reduced basis length")
        columns.append(matches)

    group: List[np.ndarray] = []
    for first in columns[0]:
        first_float = first.astype(float)
        second_ok = columns[1][
            np.abs(columns[1].astype(float) @ metric @ first_float - float(metric[0, 1])) <= length_tolerance
        ]
        if len(second_ok) == 0:
            continue
        for second in second_ok:
            second_float = second.astype(float)
            third_ok = columns[2][
                (np.abs(columns[2].astype(float) @ metric @ first_float - float(metric[0, 2])) <= length_tolerance)
                & (np.abs(columns[2].astype(float) @ metric @ second_float - float(metric[1, 2])) <= length_tolerance)
            ]
            for third in third_ok:
                candidate = np.stack([first, second, third], axis=1)
                determinant = int(round(float(np.linalg.det(candidate.astype(float)))))
                if abs(determinant) != 1:
                    continue
                group.append(candidate)

    if not group:  # pragma: no cover - the identity is always a solution
        return np.eye(3, dtype=np.int64)[None, :, :]

    reduced_group = np.asarray(group, dtype=np.int64)
    # Conjugate back: with ``reduced = transform @ lattice`` a column operation
    # ``W_reduced`` in the reduced basis is ``W = transform.T @ W_reduced @ transform.T^-1``.
    conjugator = transform.T.astype(float)
    inverse_conjugator = np.linalg.inv(conjugator)
    conjugated = np.einsum("ij,kjl,lm->kim", conjugator, reduced_group.astype(float), inverse_conjugator)
    rounded = np.rint(conjugated).astype(np.int64)
    if not np.allclose(conjugated, rounded, atol=1e-6):  # pragma: no cover - defensive
        raise RuntimeError("point-group conjugation produced non-integer matrices")
    unique: Dict[Tuple[int, ...], np.ndarray] = {}
    for element in rounded:
        unique[tuple(int(value) for value in element.ravel())] = element
    return np.asarray([unique[key] for key in sorted(unique)], dtype=np.int64)


def rotation_type(rotation: np.ndarray) -> int:
    """Return the crystallographic type of an integer rotation matrix.

    Proper rotations are reported as their order ``1, 2, 3, 4, 6`` and improper
    ones as the negative of the order of the associated roto-inversion, so
    ``-1`` is inversion, ``-2`` a mirror, and ``-3``, ``-4``, ``-6`` the
    roto-inversions.
    """

    matrix = np.asarray(rotation, dtype=float)
    determinant = int(round(float(np.linalg.det(matrix))))
    trace = int(round(float(np.trace(matrix))))
    if determinant not in (1, -1):
        raise ValueError("a symmetry rotation must have determinant +-1")
    table = {
        (1, 3): 1,
        (1, -1): 2,
        (1, 0): 3,
        (1, 1): 4,
        (1, 2): 6,
        (-1, -3): -1,
        (-1, 1): -2,
        (-1, 0): -3,
        (-1, -1): -4,
        (-1, -2): -6,
    }
    key = (determinant, trace)
    if key not in table:
        raise ValueError(f"matrix with determinant {determinant} and trace {trace} is not crystallographic")
    return table[key]


# (identity, 2, 3, 4, 6, inversion, mirror, -3, -4, -6) -> Hermann-Mauguin symbol
_POINT_GROUP_TABLE: Dict[Tuple[int, ...], str] = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0): "1",
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0): "-1",
    (1, 1, 0, 0, 0, 0, 0, 0, 0, 0): "2",
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0): "m",
    (1, 1, 0, 0, 0, 1, 1, 0, 0, 0): "2/m",
    (1, 3, 0, 0, 0, 0, 0, 0, 0, 0): "222",
    (1, 1, 0, 0, 0, 0, 2, 0, 0, 0): "mm2",
    (1, 3, 0, 0, 0, 1, 3, 0, 0, 0): "mmm",
    (1, 1, 0, 2, 0, 0, 0, 0, 0, 0): "4",
    (1, 1, 0, 0, 0, 0, 0, 0, 2, 0): "-4",
    (1, 1, 0, 2, 0, 1, 1, 0, 2, 0): "4/m",
    (1, 5, 0, 2, 0, 0, 0, 0, 0, 0): "422",
    (1, 1, 0, 2, 0, 0, 4, 0, 0, 0): "4mm",
    (1, 3, 0, 0, 0, 0, 2, 0, 2, 0): "-42m",
    (1, 5, 0, 2, 0, 1, 5, 0, 2, 0): "4/mmm",
    (1, 0, 2, 0, 0, 0, 0, 0, 0, 0): "3",
    (1, 0, 2, 0, 0, 1, 0, 2, 0, 0): "-3",
    (1, 3, 2, 0, 0, 0, 0, 0, 0, 0): "32",
    (1, 0, 2, 0, 0, 0, 3, 0, 0, 0): "3m",
    (1, 3, 2, 0, 0, 1, 3, 2, 0, 0): "-3m",
    (1, 1, 2, 0, 2, 0, 0, 0, 0, 0): "6",
    (1, 0, 2, 0, 0, 0, 1, 0, 0, 2): "-6",
    (1, 1, 2, 0, 2, 1, 1, 2, 0, 2): "6/m",
    (1, 7, 2, 0, 2, 0, 0, 0, 0, 0): "622",
    (1, 1, 2, 0, 2, 0, 6, 0, 0, 0): "6mm",
    (1, 3, 2, 0, 0, 0, 4, 0, 0, 2): "-6m2",
    (1, 7, 2, 0, 2, 1, 7, 2, 0, 2): "6/mmm",
    (1, 3, 8, 0, 0, 0, 0, 0, 0, 0): "23",
    (1, 3, 8, 0, 0, 1, 3, 8, 0, 0): "m-3",
    (1, 9, 8, 6, 0, 0, 0, 0, 0, 0): "432",
    (1, 3, 8, 0, 0, 0, 6, 0, 6, 0): "-43m",
    (1, 9, 8, 6, 0, 1, 9, 8, 6, 0): "m-3m",
}

_CRYSTAL_SYSTEMS: Dict[str, str] = {
    "1": "triclinic",
    "-1": "triclinic",
    "2": "monoclinic",
    "m": "monoclinic",
    "2/m": "monoclinic",
    "222": "orthorhombic",
    "mm2": "orthorhombic",
    "mmm": "orthorhombic",
    "4": "tetragonal",
    "-4": "tetragonal",
    "4/m": "tetragonal",
    "422": "tetragonal",
    "4mm": "tetragonal",
    "-42m": "tetragonal",
    "4/mmm": "tetragonal",
    "3": "trigonal",
    "-3": "trigonal",
    "32": "trigonal",
    "3m": "trigonal",
    "-3m": "trigonal",
    "6": "hexagonal",
    "-6": "hexagonal",
    "6/m": "hexagonal",
    "622": "hexagonal",
    "6mm": "hexagonal",
    "-6m2": "hexagonal",
    "6/mmm": "hexagonal",
    "23": "cubic",
    "m-3": "cubic",
    "432": "cubic",
    "-43m": "cubic",
    "m-3m": "cubic",
}

_TYPE_ORDER = (1, 2, 3, 4, 6, -1, -2, -3, -4, -6)


def point_group_symbol(rotations: Sequence[np.ndarray]) -> str | None:
    """Return the Hermann-Mauguin symbol of a set of integer rotations.

    The 32 crystallographic point groups are distinguished by how many
    operations of each type they contain, which is what this look-up uses.
    ``None`` is returned when the operations do not form one of them.
    """

    matrices = np.asarray(rotations, dtype=np.int64).reshape(-1, 3, 3)
    if matrices.shape[0] == 0:
        return None
    # The distinct matrices, found by one sort instead of a Python set.
    distinct = np.unique(matrices.reshape(-1, 9), axis=0).reshape(-1, 3, 3)
    counts: Dict[int, int] = {key: 0 for key in _TYPE_ORDER}
    for rotation in distinct:
        try:
            counts[rotation_type(rotation)] += 1
        except ValueError:
            return None
    key = tuple(counts[value] for value in _TYPE_ORDER)
    return _POINT_GROUP_TABLE.get(key)


def crystal_system_of_point_group(symbol: str | None) -> str | None:
    """Return the crystal system that a Hermann-Mauguin point-group symbol belongs to."""

    if symbol is None:
        return None
    return _CRYSTAL_SYSTEMS.get(str(symbol))


"""The moire builder fills a supercell by exact coset enumeration.

`moire/builder/generator.py` never scans a box of translations and deduplicates
by distance: it puts the transposed supercell matrix into column Hermite normal
form and enumerates the box ``0 <= x < h11``, ``0 <= y < h22``.
`RequestProject/CosetRepresentatives.lean` proves that this is a transversal of
``Z^2`` modulo the supercell lattice (`Cellstine.existsUnique_mem_hnfBox`), that
the Hermite triple spans the lattice it came from
(`Cellstine.latticeOf_columnHnf`) and that the box holds ``|det M|`` points
(`Cellstine.card_hnfBox`, `Cellstine.hnf_card_eq_abs_det`).

The tests below check all three on the implementation, over a sweep of matrices
rather than a handful, using exact integer arithmetic throughout.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.moire.builder import generator


def _matrices():
    """A sweep of nonsingular 2x2 integer matrices, including awkward ones."""

    seen = []
    for a, b, c, d in itertools.product(range(-3, 4), repeat=4):
        if a * d - b * c != 0:
            seen.append(np.array([[a, b], [c, d]], dtype=np.int64))
    extra = [
        np.array([[12, 0], [0, 7]], dtype=np.int64),
        np.array([[9, 6], [6, 9]], dtype=np.int64),
        np.array([[0, 5], [7, 0]], dtype=np.int64),
        np.array([[13, 5], [8, 21]], dtype=np.int64),
        np.array([[-11, 4], [7, -3]], dtype=np.int64),
    ]
    return seen + extra


def _in_lattice(vector, columns) -> bool:
    """Whether an integer vector is an integer combination of two columns."""

    determinant = int(round(np.linalg.det(columns.astype(float))))
    assert determinant != 0
    adjugate = np.array(
        [[columns[1, 1], -columns[0, 1]], [-columns[1, 0], columns[0, 0]]],
        dtype=np.int64,
    )
    scaled = adjugate @ np.asarray(vector, dtype=np.int64)
    return bool(scaled[0] % determinant == 0 and scaled[1] % determinant == 0)


def test_bezout_identity_holds_on_the_sweep():
    for left, right in itertools.product(range(-12, 13), repeat=2):
        x, y = generator._bezout(left, right)
        assert x * left + y * right == math.gcd(left, right)


def test_hermite_triple_spans_the_same_column_lattice():
    """`Cellstine.latticeOf_columnHnf`, checked with exact integers."""

    for matrix in _matrices():
        h11, h12, h22 = generator._column_hermite_normal_form(matrix)
        determinant = int(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
        assert h11 > 0 and h22 > 0
        assert 0 <= h12 < h11
        assert h11 * h22 == abs(determinant)
        assert h22 == math.gcd(int(matrix[1, 0]), int(matrix[1, 1]))

        hermite = np.array([[h11, h12], [0, h22]], dtype=np.int64)
        # every generator of each lattice lies in the other
        for column in (hermite[:, 0], hermite[:, 1]):
            assert _in_lattice(column, matrix)
        for column in (matrix[:, 0], matrix[:, 1]):
            assert _in_lattice(column, hermite)


def test_the_box_is_a_transversal_of_the_supercell_lattice():
    """`Cellstine.existsUnique_mem_hnfBox`: one image per coset, and no more."""

    for matrix in _matrices()[:80]:
        translations = generator._coset_representatives(matrix)
        determinant = abs(int(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]))
        assert len(translations) == determinant

        columns = np.asarray(matrix, dtype=np.int64).T
        # no two representatives are congruent
        for first, second in itertools.combinations(translations.tolist(), 2):
            difference = np.array(first, dtype=np.int64) - np.array(second, dtype=np.int64)
            assert not _in_lattice(difference, columns)
        # and every translation of the plane is congruent to one of them
        for point in itertools.product(range(-6, 7), repeat=2):
            hits = [
                representative
                for representative in translations.tolist()
                if _in_lattice(np.array(point, dtype=np.int64) - np.array(representative), columns)
            ]
            assert len(hits) == 1


def test_every_atom_is_copied_exactly_the_index_of_the_supercell():
    """`Cellstine.hnf_card_eq_abs_det`, on the replication the builder runs."""

    lattice = np.array([[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 20.0]])
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    species = ["C", "C"]
    for matrix in ([[3, 1], [1, 4]], [[6, -2], [3, 4]], [[2, 0], [0, 3]], [[5, 2], [-1, 3]]):
        integers = np.asarray(matrix, dtype=np.int64)
        index = abs(int(integers[0, 0] * integers[1, 1] - integers[0, 1] * integers[1, 0]))
        atoms = generator._replicate_layer_cartesian(
            positions, lattice, integers, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), species, None
        )
        assert len(atoms) == index * len(positions)

        # the images of one atom are distinct points of the supercell
        supercell = integers.astype(float) @ lattice[:2, :2]
        first = np.array([atom[1][:2] for atom in atoms[:index]])
        fractional = first @ np.linalg.inv(supercell)
        fractional -= np.floor(fractional + 1e-12)
        unique = {tuple(np.round(row, 9)) for row in fractional}
        assert len(unique) == index
        assert np.all(fractional > -1e-9) and np.all(fractional < 1.0 + 1e-9)


def test_a_singular_supercell_matrix_is_refused():
    with pytest.raises(ValueError, match="nonsingular"):
        generator._column_hermite_normal_form(np.array([[2, 4], [1, 2]], dtype=np.int64))

"""Exactness of the integer kernels the moire search is built on.

These are the routines the class folding of :mod:`cellstine.moire.search` runs
in its inner loop.  They are pure integer arithmetic, so the tests below are
exact equalities, not tolerances:

* the vectorised extended Euclid really returns Bezout coefficients,
* the column Hermite form is the Hermite form of the column lattice, so it is
  unchanged by right multiplication with a unimodular matrix,
* the class key is invariant under that right action and separates classes,
* dropping the symmetry pairs ``(-G_t, -G_b)`` cannot change a single key --
  this is the reduction :func:`_reduced_symmetry_combinations` performs.
"""

from __future__ import annotations

import math

import numpy as np

from cellstine.core.symmetry2d import lattice_point_group, layer_point_group
from cellstine.moire.search.gram_lattice import (
    _bezout,
    _hermite_normal_form,
    _lexicographic_min_pair,
    _lexicographic_minimum,
)
from cellstine.moire.search.gram_pairs import (
    _canonical_pair_keys,
    _reduced_symmetry_combinations,
    _symmetry_combinations,
)


def _random_unimodular(rng: np.random.Generator, count: int) -> np.ndarray:
    """Random integer matrices of determinant ``+-1``, built from shears."""

    matrices = np.tile(np.eye(2, dtype=np.int64), (count, 1, 1))
    for _ in range(6):
        shear = np.tile(np.eye(2, dtype=np.int64), (count, 1, 1))
        entry = rng.integers(-3, 4, size=count)
        if rng.integers(0, 2):
            shear[:, 0, 1] = entry
        else:
            shear[:, 1, 0] = entry
        matrices = matrices @ shear
    flip = rng.integers(0, 2, size=count).astype(bool)
    matrices[flip] = matrices[flip][:, ::-1, :]
    return matrices


def _random_invertible(rng: np.random.Generator, count: int) -> np.ndarray:
    matrices = rng.integers(-7, 8, size=(count, 2, 2)).astype(np.int64)
    determinant = (
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    return matrices[determinant != 0]


def test_bezout_returns_bezout_coefficients() -> None:
    rng = np.random.default_rng(20240501)
    left = rng.integers(-2000, 2001, size=20000)
    right = rng.integers(-2000, 2001, size=20000)
    first, second = _bezout(left, right)
    assert np.array_equal(first * left + second * right, np.gcd(left, right))


def test_bezout_handles_zero_rows() -> None:
    left = np.array([0, 0, 5, -5, 12])
    right = np.array([0, 7, 0, 0, -18])
    first, second = _bezout(left, right)
    assert np.array_equal(first * left + second * right, np.gcd(left, right))


def test_hermite_form_depends_only_on_the_column_lattice() -> None:
    rng = np.random.default_rng(7)
    matrices = _random_invertible(rng, 4000)
    transformed = matrices @ _random_unimodular(rng, len(matrices))
    first = _hermite_normal_form(matrices[:, :, 0], matrices[:, :, 1])
    second = _hermite_normal_form(transformed[:, :, 0], transformed[:, :, 1])
    assert np.array_equal(first, second)
    # h11 h22 = |det|, and the off-diagonal entry is reduced modulo h11.
    determinant = np.abs(
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    assert np.array_equal(first[:, 0] * first[:, 2], determinant)
    assert np.all((first[:, 1] >= 0) & (first[:, 1] < first[:, 0]))


def test_pair_key_is_invariant_under_the_common_right_action() -> None:
    rng = np.random.default_rng(11)
    top = _random_invertible(rng, 3000)
    bottom = _random_invertible(rng, len(top))
    count = min(len(top), len(bottom))
    top, bottom = top[:count], bottom[:count]
    relabel = _random_unimodular(rng, count)
    plain = _canonical_pair_keys(top, bottom)
    moved = _canonical_pair_keys(top @ relabel, bottom @ relabel)
    assert np.array_equal(plain, moved)


def test_pair_key_separates_genuinely_different_pairs() -> None:
    top = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=np.int64)
    bottom = np.array([[[1, 0], [0, 1]], [[2, 1], [1, 1]]], dtype=np.int64)
    keys = _canonical_pair_keys(top, bottom)
    assert not np.array_equal(keys[0], keys[1])


def _honeycomb_group(constant: float, sublattices: int) -> np.ndarray:
    lattice = np.zeros((3, 3))
    lattice[0, :2] = (constant, 0.0)
    lattice[1, :2] = (-0.5 * constant, 0.5 * math.sqrt(3.0) * constant)
    lattice[2, 2] = 20.0
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    species = ["C", "C"] if sublattices == 1 else ["B", "N"]
    return np.asarray(layer_point_group(lattice, positions, species), dtype=np.int64)


def test_reduced_symmetry_combinations_leave_every_key_alone() -> None:
    rng = np.random.default_rng(5)
    square = np.asarray(
        lattice_point_group(np.array([[3.0, 0.0], [0.0, 3.0]])), dtype=np.int64
    )
    pairs = [
        (_honeycomb_group(2.46, 1), _honeycomb_group(2.46, 1)),
        (_honeycomb_group(2.46, 1), _honeycomb_group(2.50, 2)),
        (square, square),
    ]
    for top_group, bottom_group in pairs:
        full = _symmetry_combinations(top_group, bottom_group)
        reduced = _reduced_symmetry_combinations(top_group, bottom_group)
        assert set(reduced) <= set(full)
        top = _random_invertible(rng, 2000)
        bottom = _random_invertible(rng, 2000)
        count = min(len(top), len(bottom))
        top, bottom = top[:count], bottom[:count]
        exhaustive = _lexicographic_minimum(
            np.stack(
                [
                    _canonical_pair_keys(top_group[left] @ top, bottom_group[right] @ bottom)
                    for left, right in full
                ]
            )
        )
        folded = _lexicographic_minimum(
            np.stack(
                [
                    _canonical_pair_keys(top_group[left] @ top, bottom_group[right] @ bottom)
                    for left, right in reduced
                ]
            )
        )
        assert np.array_equal(exhaustive, folded)
        # A centrosymmetric layer pair really does halve the work.
        negation_in_both = any(
            np.array_equal(-top_group[left], top_group[other])
            for left in range(len(top_group))
            for other in range(len(top_group))
        ) and any(
            np.array_equal(-bottom_group[right], bottom_group[other])
            for right in range(len(bottom_group))
            for other in range(len(bottom_group))
        )
        assert len(reduced) == (len(full) // 2 if negation_in_both else len(full))


def test_lexicographic_min_pair_matches_python_ordering() -> None:
    rng = np.random.default_rng(3)
    first = rng.integers(-2, 3, size=(500, 4))
    second = rng.integers(-2, 3, size=(500, 4))
    folded = _lexicographic_min_pair(first, second)
    expected = np.array(
        [
            min(tuple(int(v) for v in a), tuple(int(v) for v in b))
            for a, b in zip(first, second)
        ]
    )
    assert np.array_equal(folded, expected)


def test_written_out_two_by_two_product_matches_numpy() -> None:
    """The hand-written 2x2 block product is exactly ``@`` on integer stacks.

    ``_finalize`` relabels millions of integer supercell matrices with it, so a
    single transposed index would silently corrupt every reported cell.
    """

    from cellstine.moire.search.gram_report import _matmul2

    rng = np.random.default_rng(20240607)
    left = rng.integers(-9, 10, size=(256, 2, 2), dtype=np.int64)
    right = rng.integers(-9, 10, size=(256, 2, 2), dtype=np.int64)
    product = _matmul2(left, right)
    assert product.dtype == np.int64
    assert np.array_equal(product, left @ right)
    assert _matmul2(left[:0], right[:0]).shape == (0, 2, 2)

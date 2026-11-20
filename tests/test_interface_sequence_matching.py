"""Plane identity requires a periodic bijection, not a greedy assignment."""

from itertools import permutations, product

import numpy as np
import pytest

from cellstine.interface.surface.sequence import _planes_coincide, shortest_repeating_prefix
from cellstine.core.geometry import minimum_image_distances


@pytest.mark.parametrize("order", list(permutations(range(2))))
def test_ambiguous_plane_matches_reassign_an_earlier_atom(order):
    first = np.array([[.04, 0., 0.], [0., 0., 0.]])[list(order)]
    second = np.array([[.01, 0., 0.], [.10, 0., 0.]])
    # The valid bijection is .04 -> .10 and 0 -> .01.
    assert _planes_coincide(first, second, np.eye(3), .07)


def test_plane_match_rejects_hall_deficit_despite_every_atom_having_a_neighbor():
    first = np.array([[0., 0., 0.], [.01, 0., 0.], [.5, 0., 0.]])
    second = np.array([[0., 0., 0.], [.5, 0., 0.], [.51, 0., 0.]])
    assert not _planes_coincide(first, second, np.eye(3), .02)


def test_plane_match_preserves_duplicate_multiplicity_and_periodic_boundary():
    first = np.array([[.99, 0., 0.], [.99, 0., 0.]])
    second = np.array([[.01, 0., 0.], [.01, 0., 0.]])
    assert _planes_coincide(first, second, np.eye(3), .021)
    second[1, 0] = .2
    assert not _planes_coincide(first, second, np.eye(3), .021)


@pytest.mark.parametrize("word,expected", [("", ""), ("ABCA", "ABC"),
    ("ABCABCA", "ABC"), ("AAAA", "A"), ("AAAB", "AAAB"),
    ("αβγαβ", "αβγ")])
def test_repeating_prefix_allows_a_partial_final_repeat(word, expected):
    assert shortest_repeating_prefix(word) == expected


def test_repeating_prefix_handles_long_partial_repeat_and_nonperiodic_tail():
    block = "A" * 300 + "B"
    assert shortest_repeating_prefix(block * 3 + block[:34]) == block
    word = "A" * 2000 + "B"
    assert shortest_repeating_prefix(word) == word


def test_repeating_prefix_matches_exhaustive_short_word_oracle():
    for length in range(1, 9):
        for letters in product("AB", repeat=length):
            word = "".join(letters)
            expected = next(word[:size] for size in range(1, length + 1)
                            if all(word[index] == word[index % size] for index in range(length)))
            assert shortest_repeating_prefix(word) == expected


@pytest.mark.parametrize("skew", [0., .8])
def test_plane_bijection_agrees_with_exhaustive_small_assignment_oracle(skew):
    lattice = np.array([[1., 0., 0.], [skew, .6, 0.], [0., 0., 4.]])
    rng = np.random.default_rng(606)
    for count in range(1, 6):
        for _ in range(15):
            first = rng.random((count, 3)); first[:, 2] = 0.
            second = rng.random((count, 3)); second[:, 2] = 0.
            distances = np.array([minimum_image_distances(lattice, second - point) for point in first])
            expected = any(all(distances[left, right] <= .24 for left, right in enumerate(order))
                           for order in permutations(range(count)))
            assert _planes_coincide(first, second, lattice, .24) == expected

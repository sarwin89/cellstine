"""Renderer label limits must bound allocations before discarding header data."""

import tracemalloc

import pytest

from cellstine.core.species import expand_species


def test_requested_label_count_bounds_discarded_header_allocation():
    tracemalloc.start()
    try:
        result = expand_species(["C"], [1000000], natoms=3)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == ["C", "C", "C"]
    assert peak < 128 * 1024


@pytest.mark.parametrize("species,counts,natoms,fallback,padding,expected", [
    (["C", "O"], [3, 4], 4, None, "X", ["C", "C", "C", "O"]),
    (["C"], [1], 3, None, "?", ["C", "?", "?"]),
    ([], [2, 1], 4, "Si", "X", ["Si", "Si", "Si", "X"]),
    (["C", "O"], [-2, 3], 2, None, "X", ["O", "O"]),
    (["C"], [5], -2, None, "X", ["C", "C", "C"]),
    (["C", "O"], [2, 1], None, None, "X", ["C", "C", "O"]),
    (["C"], [1000000], 0, None, "X", []),
])
def test_label_order_padding_and_legacy_slices_are_preserved(species, counts, natoms, fallback, padding, expected):
    assert expand_species(species, counts, fallback, natoms=natoms, padding=padding) == expected


def test_counts_beyond_the_requested_labels_are_still_validated():
    with pytest.raises(ValueError):
        expand_species(["C", "O"], [2, "invalid"], natoms=1)

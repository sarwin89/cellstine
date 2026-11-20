"""Conventional axes share work; group caps stop excess construction promptly."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cellstine.core import bravais, idealisation
from test_bravais import LATTICES


@pytest.mark.parametrize("symbol", ["cP", "oP", "tP", "hP", "hR", "mP"])
def test_conventional_axes_share_one_point_cloud_without_mutating_the_input(symbol):
    lattice = LATTICES[symbol].copy()
    original = lattice.copy()
    with patch.object(bravais, "_lattice_points", wraps=bravais._lattice_points) as cloud:
        result = bravais.conventional_cell(lattice)
    assert result.symbol == symbol
    assert cloud.call_count == 1
    assert np.array_equal(lattice, original)


def test_group_already_over_its_cap_stops_before_product_construction(monkeypatch):
    original = idealisation.np.ascontiguousarray
    calls = []

    def count_array(element, dtype=None):
        calls.append(None)
        return original(element, dtype=dtype)

    monkeypatch.setattr(idealisation.np, "ascontiguousarray", count_array)
    group = np.array([[[1, 1], [0, 1]], [[1, 2], [0, 1]]], dtype=np.int64)
    with pytest.raises(ValueError, match="bounded closure exceeded 2 elements"):
        idealisation.close_matrix_group(group, max_order=2, name="bounded")
    # Identity, first generator, then the first inadmissible generator. No
    # product needs to be constructed after the input itself exceeds the cap.
    assert len(calls) <= 3


def test_group_generation_stops_at_the_first_new_element_over_its_cap(monkeypatch):
    original = idealisation.np.ascontiguousarray
    attempted = set()

    def count_unique(element, dtype=None):
        result = original(element, dtype=dtype)
        attempted.add(result.tobytes())
        return result

    monkeypatch.setattr(idealisation.np, "ascontiguousarray", count_unique)
    generator = np.array([[1, 1], [0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="bounded closure exceeded 3 elements"):
        idealisation.close_matrix_group(generator, max_order=3, name="bounded")
    assert len(attempted) <= 4

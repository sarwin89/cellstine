"""Species regrouping must keep positions, flags and provenance together."""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.defect.analysis import _record_from_atoms
from cellstine.io.models import StructureRecord


@pytest.mark.parametrize("with_flags", [False, True])
@pytest.mark.parametrize("new_lattice", [False, True])
def test_record_grouping_retains_source_priority_per_atom_fields_and_ownership(with_flags, new_lattice):
    lattice = np.diag([3., 4., 5.])
    metadata = {"source": {"label": "retained"}}
    source = StructureRecord("source", lattice, ["O", "Si", "C"], [1, 2, 1],
                             np.zeros((4, 3)), np.zeros((4, 3)), selective_dynamics=True,
                             source_path="source.vasp", source_format="vasp", metadata=metadata)
    labels = ["Si", "X", "O", "Si"]
    direct = np.arange(12, dtype=float).reshape(4, 3) / 5. - 1.
    original = direct.copy()
    flags = [("T", "F", "T"), ("F", "T", "F"), ("T", "T", "F"), ("F", "F", "T")]
    cell = lattice + np.eye(3) if new_lattice else None
    result = _record_from_atoms(source, labels, direct, flags if with_flags else None,
                                comment="regrouped", lattice=cell)
    expected_order = [2, 0, 3, 1]
    assert result.species == ["O", "Si", "X"]  # missing C vanishes; X follows source species
    assert result.counts == [1, 2, 1]
    np.testing.assert_array_equal(result.positions_direct, direct[expected_order])
    np.testing.assert_array_equal(result.positions_cartesian,
                                  direct[expected_order] @ (lattice if cell is None else cell))
    assert result.selective_flags == ([flags[index] for index in expected_order] if with_flags else None)
    assert result.selective_dynamics is with_flags
    assert result.comment == "regrouped"
    assert result.source_path == source.source_path and result.source_format == source.source_format
    assert result.metadata == metadata and result.metadata is not metadata
    assert result.metadata["source"] is metadata["source"]  # existing shallow ownership contract
    assert not np.shares_memory(result.positions_direct, direct)
    assert not np.shares_memory(result.lattice, lattice if cell is None else cell)
    np.testing.assert_array_equal(direct, original)
    assert labels == ["Si", "X", "O", "Si"]


def test_empty_record_keeps_empty_coordinate_shape():
    source = StructureRecord("source", np.eye(3), ["Si"], [0], np.empty((0, 3)), np.empty((0, 3)))
    result = _record_from_atoms(source, [], [], [], comment="empty")
    assert result.species == [] and result.counts == [] and result.selective_flags == []
    assert result.positions_direct.shape == result.positions_cartesian.shape == (0, 3)

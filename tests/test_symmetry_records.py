"""Checks for symmetry structure-record helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.io.models import StructureRecord
from cellstine.symmetry.records import (
    lattice_parameters,
    record_from_atoms,
    record_from_spglib_cell,
    species_type_map,
)


def _source_record() -> StructureRecord:
    lattice = np.diag([2.0, 3.0, 4.0])
    direct = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    return StructureRecord(
        comment="source",
        lattice=lattice,
        species=["O", "Si"],
        counts=[0, 1],
        positions_direct=direct,
        positions_cartesian=direct @ lattice,
        source_path="source.vasp",
        source_format="vasp",
        metadata={"tag": "kept"},
    )


def test_species_type_map_preserves_first_seen_species_order():
    numbers, mapping = species_type_map(["Si", "O", "Si", "C"])

    assert numbers == [1, 2, 1, 3]
    assert mapping == {1: "Si", 2: "O", 3: "C"}


def test_record_from_atoms_wraps_and_regroups_positions_by_source_species_order():
    source = _source_record()
    positions = np.asarray(
        [
            [1.25, -0.25, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.0, 0.25],
            [0.0, 0.25, 0.5],
        ],
        dtype=float,
    )

    record = record_from_atoms(
        source,
        source.lattice,
        positions,
        ["Si", "X", "O", "Si"],
        comment="converted",
    )

    assert record.comment == "converted"
    assert record.species == ["O", "Si", "X"]
    assert record.counts == [1, 2, 1]
    assert np.allclose(record.positions_direct, [[0.75, 0.0, 0.25], [0.25, 0.75, 0.0], [0.0, 0.25, 0.5], [0.5, 0.5, 0.5]])
    assert np.allclose(record.positions_cartesian, record.positions_direct @ source.lattice)
    assert record.metadata == source.metadata
    assert record.metadata is not source.metadata


def test_record_from_spglib_cell_maps_unknown_species_numbers_to_placeholders():
    source = _source_record()
    cell = (
        source.lattice,
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
        [1, 9],
    )

    record = record_from_spglib_cell(source, cell, {1: "Si"}, comment="spglib")

    assert record.species == ["Si", "X9"]
    assert record.counts == [1, 1]


def test_lattice_parameters_report_lengths_angles_and_volume():
    params = lattice_parameters(np.diag([2.0, 3.0, 4.0]))

    assert params["a"] == pytest.approx(2.0)
    assert params["b"] == pytest.approx(3.0)
    assert params["c"] == pytest.approx(4.0)
    assert params["alpha"] == pytest.approx(90.0)
    assert params["beta"] == pytest.approx(90.0)
    assert params["gamma"] == pytest.approx(90.0)
    assert params["volume"] == pytest.approx(24.0)

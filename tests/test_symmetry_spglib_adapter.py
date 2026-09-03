"""Checks for small spglib-backend adapter helpers."""

from __future__ import annotations

import numpy as np

from cellstine.symmetry.spglib_adapter import crystal_system, dataset_value, has_inversion


class _DatasetObject:
    number = 225


def test_dataset_value_reads_objects_mappings_and_defaults():
    assert dataset_value(_DatasetObject(), "number") == 225
    assert dataset_value({"international": "Fm-3m"}, "international") == "Fm-3m"
    assert dataset_value({}, "missing", default="fallback") == "fallback"
    assert dataset_value(None, "missing", default="fallback") == "fallback"


def test_crystal_system_maps_space_group_number_ranges():
    assert crystal_system(None) is None
    assert crystal_system(1) == "triclinic"
    assert crystal_system(15) == "monoclinic"
    assert crystal_system(74) == "orthorhombic"
    assert crystal_system(142) == "tetragonal"
    assert crystal_system(167) == "trigonal"
    assert crystal_system(194) == "hexagonal"
    assert crystal_system(230) == "cubic"
    assert crystal_system(231) is None


def test_has_inversion_detects_the_negative_identity_rotation():
    rotations = np.asarray(
        [
            np.eye(3, dtype=int),
            -np.eye(3, dtype=int),
        ]
    )

    assert has_inversion(None) is None
    assert has_inversion(rotations) is True
    assert has_inversion(np.asarray([np.eye(3, dtype=int)])) is False

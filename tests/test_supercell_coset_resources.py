"""Exact supercell translations must not cost the volume of a sheared box."""

import tracemalloc

import numpy as np
import pytest

from cellstine.core import transforms


@pytest.mark.parametrize("first,second,expected", [
    (2, 1, [[0, 0, 0], [1, 50001, 0]]),
    (2, -1, [[0, 0, 0], [1, 50000, 0]]),
    (1, 1, [[0, 0, 0]]),
])
def test_sheared_cells_allocate_for_the_cosets_not_the_bounding_box(first, second, expected):
    matrix = np.array([[first, 100001, 0], [0, second, 0], [0, 0, 1]], dtype=np.int64)
    tracemalloc.start()
    try:
        cosets = transforms.supercell_cosets(matrix)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.array_equal(cosets, expected)
    assert peak < 4 * 1024 * 1024


def _box_reference(matrix):
    determinant = int(round(np.linalg.det(matrix.astype(float))))
    adjugate = np.rint(np.linalg.inv(matrix.astype(float)) * determinant).astype(np.int64)
    assert np.array_equal(matrix @ adjugate, determinant * np.eye(3, dtype=np.int64))
    lower = np.minimum(matrix, 0).sum(axis=0)
    upper = np.maximum(matrix, 0).sum(axis=0)
    axes = [np.arange(lower[axis], upper[axis] + 1) for axis in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    scaled = points @ adjugate
    if determinant > 0:
        selected = np.all((scaled >= 0) & (scaled < determinant), axis=1)
    else:
        selected = np.all((scaled <= 0) & (scaled > determinant), axis=1)
    result = points[selected]
    assert len(result) == abs(determinant)
    return result


def test_residue_enumeration_matches_independent_box_membership_and_order(monkeypatch):
    monkeypatch.setattr(transforms, "_COSET_BOX_LIMIT", 0)
    random = np.random.default_rng(719)
    checked = 0
    while checked < 100:
        matrix = random.integers(-3, 4, size=(3, 3), dtype=np.int64)
        determinant = int(round(np.linalg.det(matrix.astype(float))))
        if not determinant:
            continue
        assert np.array_equal(transforms.supercell_cosets(matrix), _box_reference(matrix))
        checked += 1


def test_sheared_structure_preserves_atom_blocks_flags_and_periodic_geometry():
    from cellstine.io.models import StructureRecord

    points = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    flags = [("T", "F", "T"), ("F", "T", "F")]
    record = StructureRecord("test", np.eye(3), ["C", "O"], [1, 1], points, points.copy(), selective_dynamics=True, selective_flags=flags)
    matrix = [[2, 100001, 0], [0, 1, 0], [0, 0, -1]]
    built = transforms.supercell_structure(record, matrix)
    assert built.species == ["C", "O"]
    assert built.counts == [2, 2]
    assert built.selective_flags == [flags[0], flags[0], flags[1], flags[1]]
    assert np.array_equal(built.lattice, matrix)
    assert np.allclose(np.mod(built.positions_cartesian, 1.0), np.repeat(points, 2, axis=0), atol=1e-8, rtol=0.0)

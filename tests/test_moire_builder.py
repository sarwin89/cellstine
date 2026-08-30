"""Geometry of the structures produced by the moire builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.io import native as io
from cellstine.moire.builder import generator
from cellstine.moire.moire import Moire


@pytest.fixture(scope="module")
def graphene_run(tmp_path_factory, graphene_poscar):
    workspace = tmp_path_factory.mktemp("moire-run")
    workflow = Moire(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    found = workflow.find(
        top_poscar=str(graphene_poscar),
        bottom_poscar=str(graphene_poscar),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
        preview_limit=0,
    )
    return workflow, found.artifacts["results_json"]


def _minimum_distance(lattice: np.ndarray, positions: np.ndarray) -> float:
    """Return the shortest interatomic distance under periodic in-plane images."""

    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    shifts = np.array(
        [(i, j, 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    )
    difference = positions[:, None, :] - positions[None, :, :]
    images = difference[:, :, None, :] + shifts[None, None, :, :]
    distances = np.linalg.norm(images @ lattice, axis=3)
    self_pairs = np.eye(len(positions), dtype=bool)[:, :, None] & (
        np.all(shifts == 0.0, axis=1)[None, None, :]
    )
    distances[self_pairs] = np.inf
    return float(distances.min())


def test_coset_representatives_are_exact_and_complete():
    for matrix in ([[3, 1], [1, 4]], [[6, -2], [3, 4]], [[1, 0], [0, 1]], [[5, 0], [0, 5]]):
        integers = np.asarray(matrix, dtype=np.int64)
        translations = generator._coset_representatives(integers)
        determinant = abs(int(np.linalg.det(integers).round()))
        assert len(translations) == determinant
        fractional = translations @ np.linalg.inv(integers.astype(float))
        fractional -= np.floor(fractional)
        unique = {tuple(np.round(row, 9)) for row in fractional}
        assert len(unique) == determinant


def test_hermite_normal_form_spans_the_same_lattice():
    matrix = np.array([[6, -2], [3, 4]], dtype=np.int64)
    h11, h12, h22 = generator._column_hermite_normal_form(matrix)
    hermite = np.array([[h11, h12], [0, h22]], dtype=float)
    transform = np.linalg.inv(matrix.astype(float)) @ hermite
    assert np.allclose(transform, np.round(transform), atol=1e-9)
    assert abs(abs(np.linalg.det(transform)) - 1.0) < 1e-9


def test_built_bilayer_has_the_recorded_atom_count_and_cell(graphene_run):
    workflow, results = graphene_run
    built = workflow.make(results_file=results, indexes=[2], interlayer_distance=3.35)
    structure = io.read_poscar(built.artifacts["structures"][0])
    _, _, candidates, _ = generator.parse_results(results)
    record = next(item for item in candidates if int(item["index"]) == 2)
    assert structure.natoms == int(record["atom_count"])
    shared = np.asarray(record["shared_lattice"], dtype=float).T
    assert np.allclose(structure.lattice[:2, :2], shared, atol=1e-8)


def test_built_bilayer_has_physical_bond_lengths_and_layer_spacing(graphene_run):
    workflow, results = graphene_run
    built = workflow.make(results_file=results, indexes=[2], interlayer_distance=3.35)
    structure = io.read_poscar(built.artifacts["structures"][0])
    cartesian = structure.positions_direct @ structure.lattice
    heights = cartesian[:, 2]
    lower = heights < heights.mean()
    assert np.count_nonzero(lower) == np.count_nonzero(~lower)
    gap = float(heights[~lower].min() - heights[lower].max())
    assert gap == pytest.approx(3.35, abs=1e-6)
    shortest = _minimum_distance(structure.lattice, structure.positions_direct)
    assert shortest == pytest.approx(1.42, abs=0.02)


def test_every_atom_lies_inside_the_cell(graphene_run):
    workflow, results = graphene_run
    built = workflow.make(results_file=results, indexes=[3], interlayer_distance=3.35)
    structure = io.read_poscar(built.artifacts["structures"][0])
    assert np.all(structure.positions_direct[:, :2] >= -1e-9)
    assert np.all(structure.positions_direct[:, :2] < 1.0 + 1e-9)
    assert np.all(structure.positions_direct[:, 2] >= 0.0)
    assert np.all(structure.positions_direct[:, 2] <= 1.0)


def test_unstrained_candidate_reproduces_the_monolayer_bond_length(graphene_run):
    """A commensurate twist must not deform the individual layers at all."""

    workflow, results = graphene_run
    _, _, candidates, _ = generator.parse_results(results)
    record = next(
        item
        for item in candidates
        if max(abs(value) for value in item["top_layer_strain"]) < 1e-9
        and int(item["index"]) > 1
    )
    built = workflow.make(
        results_file=results, indexes=[int(record["index"])], interlayer_distance=3.35
    )
    structure = io.read_poscar(built.artifacts["structures"][0])
    cartesian = structure.positions_direct @ structure.lattice
    lower = cartesian[:, 2] < cartesian[:, 2].mean()
    for mask in (lower, ~lower):
        layer = structure.positions_direct[mask]
        shortest = _minimum_distance(structure.lattice, layer)
        assert shortest == pytest.approx(2.46 / math.sqrt(3.0), abs=1e-6)


def test_building_every_small_candidate_succeeds(graphene_run):
    workflow, results = graphene_run
    _, _, candidates, _ = generator.parse_results(results)
    wanted = [int(item["index"]) for item in candidates if int(item["atom_count"]) <= 130]
    assert len(wanted) >= 4
    built = workflow.make(
        results_file=results, indexes=wanted, interlayer_distance=3.35
    )
    paths = built.artifacts["structures"]
    assert len(paths) == len(wanted)
    for path, index in zip(paths, wanted):
        record = next(item for item in candidates if int(item["index"]) == index)
        structure = io.read_poscar(path)
        assert structure.natoms == int(record["atom_count"])
        assert _minimum_distance(structure.lattice, structure.positions_direct) > 1.2


def test_requested_vacuum_is_exact_and_centred(graphene_run):
    """``--vacuum V`` must give a cell of height span + V with V/2 on each side."""

    workflow, results = graphene_run
    built = workflow.make(
        results_file=results, indexes=[2], interlayer_distance=3.35, vacuum=18.0
    )
    structure = io.read_poscar(built.artifacts["structures"][0])
    heights = structure.positions_cartesian[:, 2]
    span = float(heights.max() - heights.min())
    cell_height = float(np.linalg.norm(structure.lattice[2]))
    assert span == pytest.approx(3.35, abs=1e-9)
    assert cell_height == pytest.approx(span + 18.0, abs=1e-9)
    assert float(heights.min()) == pytest.approx(9.0, abs=1e-9)
    assert cell_height - float(heights.max()) == pytest.approx(9.0, abs=1e-9)

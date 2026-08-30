"""Mathematical checks on the supercell chosen for a point defect.

The claims tested here are about lattices, not about the implementation.  The
enumeration of Hermite normal forms must produce every sublattice of a given
index exactly once, so its count must match the arithmetic formula and the
lattices it produces must be pairwise distinct; the separation reported for a
supercell must be the true shortest translation of that supercell, found
independently by reduction; the winner of the search must beat every other
sublattice of its index by exhaustive comparison; Minkowski's bound must hold
for every candidate; and the structure built on the chosen cell must hold
``|det M|`` copies of the host with its interatomic distances untouched.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.core import geometry
from cellstine.core.transforms import (
    integer_supercell_matrix,
    repeat_structure,
    supercell_cosets,
    supercell_structure,
)
from cellstine.defect import supercell as sc
from cellstine.defect.workflow import Defect
from cellstine.io.converters import StructureConverter

from conftest import write_poscar

ALUMINIUM_CONSTANT = 4.05

FCC_PRIMITIVE = 0.5 * ALUMINIUM_CONSTANT * np.array(
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
)

HEXAGONAL_SLAB = np.array(
    [[2.46, 0.0, 0.0], [-1.23, 2.46 * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 20.0]]
)


@pytest.fixture()
def aluminium_poscar(tmp_path: Path) -> Path:
    return write_poscar(
        tmp_path / "al.vasp", FCC_PRIMITIVE, ["Al"], [1], np.zeros((1, 3)), comment="fcc Al"
    )


# ---------------------------------------------------------------------------
# the enumeration of sublattices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cells", [1, 2, 3, 4, 6, 8, 12, 16])
def test_hermite_enumeration_counts_match_the_divisor_sum(cells: int) -> None:
    """The number of index-``n`` sublattices of the cubic lattice is the divisor sum."""

    expected = sum(
        second * third * third
        for first in range(1, cells + 1)
        for second in range(1, cells + 1)
        for third in range(1, cells + 1)
        if first * second * third == cells
    )
    forms = list(sc.hermite_normal_forms_3d(cells))
    assert len(forms) == expected == sc.hermite_normal_form_count(cells)
    for form in forms:
        assert int(round(float(np.linalg.det(form.astype(float))))) == cells


def _random_unimodular(rng: np.random.Generator) -> np.ndarray:
    """A random integer matrix of determinant one, as a product of shears."""

    matrix = np.eye(3, dtype=np.int64)
    for _ in range(8):
        row, column = rng.choice(3, size=2, replace=False)
        shear = np.eye(3, dtype=np.int64)
        shear[row, column] = int(rng.integers(-2, 3))
        matrix = matrix @ shear
    return matrix


def _spans_the_same_lattice(left: np.ndarray, right: np.ndarray) -> bool:
    transfer = left.astype(float) @ np.linalg.inv(right.astype(float))
    return bool(
        np.allclose(transfer, np.rint(transfer), atol=1e-9)
        and abs(abs(float(np.linalg.det(transfer))) - 1.0) < 1e-9
    )


@pytest.mark.parametrize("cells", [2, 3, 4, 6, 8])
def test_hermite_enumeration_lists_every_sublattice_once(cells: int) -> None:
    """No two enumerated forms span the same sublattice, and none is missing.

    Two integer matrices span the same lattice exactly when one is a unimodular
    multiple of the other.  The enumerated forms must therefore be pairwise
    inequivalent, and any basis of any sublattice of this index -- built here
    as a random unimodular multiple of one of them -- must match exactly one.
    """

    forms = list(sc.hermite_normal_forms_3d(cells))
    for left, right in itertools.combinations(forms, 2):
        assert not _spans_the_same_lattice(left, right)
    rng = np.random.default_rng(20240517 + cells)
    for form in forms:
        for _ in range(3):
            candidate = _random_unimodular(rng) @ form
            assert abs(int(round(float(np.linalg.det(candidate.astype(float)))))) == cells
            matches = [other for other in forms if _spans_the_same_lattice(candidate, other)]
            assert len(matches) == 1


@pytest.mark.parametrize("cells", [1, 2, 5, 9])
def test_plane_hermite_enumeration_counts_match(cells: int) -> None:
    forms = list(sc.hermite_normal_forms_2d(cells))
    assert len(forms) == sc.hermite_normal_form_count(cells, plane=True)
    assert len(forms) == sum(
        1 for first in range(1, cells + 1) for _ in range(cells // first if cells % first == 0 else 0)
    )
    for form in forms:
        assert int(round(float(np.linalg.det(form.astype(float))))) == cells


# ---------------------------------------------------------------------------
# the separation reported for a cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cells", [1, 2, 3, 4, 6, 8, 9, 16, 27])
def test_search_matches_exhaustive_comparison(cells: int) -> None:
    """The winner really is the best sublattice of its index.

    Every candidate is measured independently, by Delaunay reduction rather
    than by the sieve the search uses, and the largest of those measurements
    must be what the search reports.
    """

    matrix, distance = sc.best_supercell_of_size(FCC_PRIMITIVE, cells, plane=False)
    exhaustive = max(
        sc.image_distance_of(FCC_PRIMITIVE, form, plane=False)
        for form in sc.hermite_normal_forms_3d(cells)
    )
    assert distance == pytest.approx(exhaustive, abs=1e-9)
    assert sc.image_distance_of(FCC_PRIMITIVE, matrix, plane=False) == pytest.approx(
        distance, abs=1e-9
    )


@pytest.mark.parametrize("cells", [1, 3, 4, 7, 12, 25])
def test_plane_search_matches_exhaustive_comparison(cells: int) -> None:
    matrix, distance = sc.best_supercell_of_size(HEXAGONAL_SLAB, cells, plane=True)
    exhaustive = max(
        sc.image_distance_of(HEXAGONAL_SLAB, sc.embed_plane_matrix(form), plane=True)
        for form in sc.hermite_normal_forms_2d(cells)
    )
    assert distance == pytest.approx(exhaustive, abs=1e-9)
    assert sc.image_distance_of(HEXAGONAL_SLAB, matrix, plane=True) == pytest.approx(
        distance, abs=1e-9
    )


def test_conventional_cubic_cell_is_the_best_four_cell_supercell() -> None:
    """Four fcc primitive cells are best arranged as the cubic cell of side ``a``."""

    matrix, distance = sc.best_supercell_of_size(FCC_PRIMITIVE, 4, plane=False)
    assert distance == pytest.approx(ALUMINIUM_CONSTANT, abs=1e-9)
    cubic = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int64)
    assert _spans_the_same_lattice(np.asarray(matrix, dtype=np.int64), cubic)


def test_a_hexagonal_slab_prefers_the_hexagonal_supercell() -> None:
    """Three cells of a hexagonal slab give the ``sqrt 3`` cell, not ``3x1``."""

    matrix, distance = sc.best_supercell_of_size(HEXAGONAL_SLAB, 3, plane=True)
    assert distance == pytest.approx(2.46 * math.sqrt(3.0), abs=1e-9)
    del matrix


@pytest.mark.parametrize("cells", [1, 2, 4, 8, 16, 32])
def test_minkowski_bound_holds_for_every_candidate(cells: int) -> None:
    """No sublattice of this index beats the bound the search prunes with."""

    bound = sc.minkowski_bound(sc.cell_volume(FCC_PRIMITIVE) * cells, plane=False)
    for form in sc.hermite_normal_forms_3d(cells):
        assert sc.image_distance_of(FCC_PRIMITIVE, form, plane=False) <= bound + 1e-9


def test_minkowski_bound_is_the_inverse_of_the_cell_count_bound() -> None:
    for cells in (1, 5, 17, 64):
        bound = sc.minkowski_bound(sc.cell_volume(FCC_PRIMITIVE) * cells, plane=False)
        assert sc.cells_needed_lower_bound(FCC_PRIMITIVE, bound, plane=False) <= cells
        assert (
            sc.cells_needed_lower_bound(FCC_PRIMITIVE, bound * 1.001, plane=False) > cells
        )


# ---------------------------------------------------------------------------
# choosing a cell
# ---------------------------------------------------------------------------


def test_choose_supercell_returns_the_smallest_cell_that_reaches_the_target() -> None:
    target = 9.0
    choice = sc.choose_supercell(
        FCC_PRIMITIVE, structure_kind="bulk", min_image_distance=target
    )
    assert choice.image_distance >= target - 1e-9
    assert choice.image_distance <= choice.upper_bound + 1e-9
    for smaller in range(1, choice.cells):
        _, distance = sc.best_supercell_of_size(FCC_PRIMITIVE, smaller, plane=False)
        assert distance < target


def test_choose_supercell_beats_the_best_plain_repeat() -> None:
    choice = sc.choose_supercell(
        FCC_PRIMITIVE, structure_kind="bulk", min_image_distance=9.0
    )
    assert choice.diagonal_distance is not None
    assert choice.image_distance > choice.diagonal_distance


def test_choose_supercell_refuses_an_unreachable_request() -> None:
    with pytest.raises(ValueError, match="no supercell"):
        sc.choose_supercell(
            FCC_PRIMITIVE, structure_kind="bulk", min_image_distance=30.0, max_cells=8
        )


def test_a_slab_is_measured_in_the_plane_only() -> None:
    choice = sc.choose_supercell(
        HEXAGONAL_SLAB, structure_kind="slab", min_image_distance=10.0
    )
    assert choice.periodicity == "in-plane"
    matrix = np.asarray(choice.matrix, dtype=np.int64)
    assert list(matrix[2]) == [0, 0, 1]
    assert matrix[0, 2] == 0 and matrix[1, 2] == 0


def test_supercell_table_is_monotone_in_its_running_best() -> None:
    rows = sc.supercell_table(FCC_PRIMITIVE, structure_kind="bulk", max_cells=12)
    best = 0.0
    for row in rows:
        assert row["image_distance"] <= row["best_possible_distance"] + 1e-9
        if row["improves"]:
            assert row["image_distance"] > best + 1e-9
        best = max(best, float(row["image_distance"]))


# ---------------------------------------------------------------------------
# building the structure on the chosen cell
# ---------------------------------------------------------------------------


def test_supercell_cosets_are_distinct_and_complete() -> None:
    matrix = np.array([[1, 1, 2], [0, 3, 1], [0, 0, 2]], dtype=np.int64)
    cosets = supercell_cosets(matrix)
    assert cosets.shape == (6, 3)
    fractional = cosets.astype(float) @ np.linalg.inv(matrix.astype(float))
    assert np.all(fractional > -1e-12) and np.all(fractional < 1.0 - 1e-12)
    keys = {tuple(np.round(row, 9)) for row in fractional}
    assert len(keys) == 6


def test_supercell_structure_reproduces_a_plain_repeat(aluminium_poscar: Path) -> None:
    record = StructureConverter().read(str(aluminium_poscar), canonicalize=False)
    diagonal = supercell_structure(record, np.diag([2, 2, 1]))
    repeated = repeat_structure(record, [2, 2, 1])
    assert diagonal.natoms == repeated.natoms
    assert np.allclose(diagonal.lattice, repeated.lattice)
    first = np.sort(np.round(np.mod(diagonal.positions_direct, 1.0), 9), axis=0)
    second = np.sort(np.round(np.mod(repeated.positions_direct, 1.0), 9), axis=0)
    assert np.allclose(first, second)


def test_supercell_structure_keeps_the_geometry(aluminium_poscar: Path) -> None:
    """The cubic cell of fcc holds four atoms and the same nearest-neighbour distance."""

    record = StructureConverter().read(str(aluminium_poscar), canonicalize=False)
    built = supercell_structure(record, [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    assert built.natoms == 4
    assert abs(float(np.linalg.det(built.lattice))) == pytest.approx(
        4.0 * abs(float(np.linalg.det(record.lattice))), rel=1e-12
    )
    distances = geometry.pairwise_minimum_image_distances(
        geometry.as_lattice(built.lattice), built.positions_direct
    )
    np.fill_diagonal(distances, np.inf)
    assert float(distances.min()) == pytest.approx(
        ALUMINIUM_CONSTANT / math.sqrt(2.0), abs=1e-9
    )


def test_supercell_matrix_must_be_integral_and_invertible() -> None:
    with pytest.raises(ValueError):
        integer_supercell_matrix([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        integer_supercell_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# the workflow
# ---------------------------------------------------------------------------


def test_defect_supercell_workflow_writes_the_chosen_cell(
    aluminium_poscar: Path, tmp_path: Path
) -> None:
    tool = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = tool.supercell(str(aluminium_poscar), min_image_distance=8.0, table_limit=6)
    cells = int(result.summary["cells"])
    assert result.summary["supercell_atoms"] == cells
    assert float(result.summary["defect_image_distance"]) >= 8.0 - 1e-9
    assert float(result.summary["defect_image_distance"]) <= float(
        result.summary["best_possible_distance"]
    ) + 1e-9
    written = StructureConverter().read(result.artifacts["structure"], canonicalize=False)
    assert written.natoms == cells
    assert geometry.shortest_lattice_vector_length(written.lattice) == pytest.approx(
        float(result.summary["defect_image_distance"]), abs=1e-4
    )
    assert "supercell_table" in result.payload


def test_defect_generate_can_size_the_host_cell(aluminium_poscar: Path, tmp_path: Path) -> None:
    tool = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = tool.generate(
        str(aluminium_poscar),
        "vacancy",
        min_image_distance=7.5,
        output_dir=str(tmp_path / "out"),
    )
    assert float(result.summary["defect_image_distance"]) >= 7.5 - 1e-9
    assert "host_supercell" in result.artifacts
    host = StructureConverter().read(result.artifacts["host_supercell"], canonicalize=False)
    written = StructureConverter().read(result.artifacts["structures"][0], canonicalize=False)
    assert written.natoms == host.natoms - 1


def test_defect_generate_refuses_two_ways_of_enlarging(
    aluminium_poscar: Path, tmp_path: Path
) -> None:
    tool = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="one way"):
        tool.generate(
            str(aluminium_poscar),
            "vacancy",
            supercell=[2, 2, 2],
            min_image_distance=7.5,
            output_dir=str(tmp_path / "out"),
        )

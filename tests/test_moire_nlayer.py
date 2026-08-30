"""Mathematical checks on the multi-layer moire search and builder.

A multi-layer candidate is only meaningful if every layer's integer supercell,
after the strain the search recorded for it, describes exactly the same in-plane
lattice.  These tests check that identity for each layer, check the exact integer
statements the construction rests on (the intersection of the base supercells and
the integer quotients that follow from it), and check that the built structure
has the shared cell, the requested gaps and vacuum, and unchanged bond lengths
inside every unstrained layer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.io import native as io_mod
from cellstine.moire.search import nlayer
from cellstine.moire.supermoire import Supermoire


@pytest.fixture(scope="module")
def trilayer_run(tmp_path_factory, graphene_poscar, hbn_poscar):
    """Graphene base with an hBN layer and a second graphene layer on top."""

    workspace = tmp_path_factory.mktemp("moire-nlayer")
    workflow = Supermoire(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    found = workflow.findn(
        base_poscar=str(graphene_poscar),
        upper_poscars=[str(hbn_poscar), str(graphene_poscar)],
        max_length=12.0,
        layer_strains=[0.03, 0.01],
        max_atoms=300,
        preview_limit=0,
    )
    document = nlayer.read_nlayer_results(found.artifacts["results_json"])
    return workflow, found.artifacts["results_json"], document


def _column_basis(lattice: np.ndarray) -> np.ndarray:
    return np.asarray(lattice, dtype=float)[:2, :2].T


def _minimum_distance(lattice: np.ndarray, positions: np.ndarray) -> float:
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    shifts = np.array([(i, j, 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
    difference = positions[:, None, :] - positions[None, :, :]
    images = difference[:, :, None, :] + shifts[None, None, :, :]
    distances = np.linalg.norm(images @ lattice, axis=3)
    self_pairs = np.eye(len(positions), dtype=bool)[:, :, None] & (
        np.all(shifts == 0.0, axis=1)[None, None, :]
    )
    distances[self_pairs] = np.inf
    return float(distances.min())


def test_integer_kernel_is_an_exact_basis_of_the_relations():
    matrix = np.array([[2, 3], [4, 6], [1, 5], [7, 1]], dtype=np.int64)
    kernel = nlayer.integer_left_kernel(matrix)
    assert kernel.shape[0] == 2
    assert np.array_equal(kernel @ matrix, np.zeros((2, 2), dtype=np.int64))
    # A relation found by hand must be an integer combination of the basis.
    relation = np.array([2, -1, 0, 0], dtype=np.int64)
    assert np.array_equal(relation @ matrix, np.zeros(2, dtype=np.int64))
    coefficients = np.linalg.lstsq(kernel.T.astype(float), relation.astype(float), rcond=None)[0]
    assert np.allclose(coefficients, np.round(coefficients), atol=1e-9)


@pytest.mark.parametrize(
    "first, second, expected_index",
    [
        ([[2, 0], [0, 3]], [[3, 0], [0, 2]], 36),
        ([[1, 0], [0, 1]], [[5, 2], [3, 4]], 14),
        ([[2, 1], [0, 2]], [[3, 0], [0, 3]], 36),
    ],
)
def test_sublattice_intersection_is_the_largest_common_sublattice(first, second, expected_index):
    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    shared = nlayer.sublattice_intersection(left, right)
    assert abs(round(float(np.linalg.det(shared)))) == expected_index

    # It is a sublattice of both, with integer quotients.
    for parent in (left, right):
        quotient = nlayer.quotient_matrix(parent, shared)
        assert np.array_equal(quotient @ parent, shared)

    # Every vector of both lattices inside a search box lies in the intersection
    # exactly when the intersection contains it.
    inverse = np.linalg.inv(shared.astype(float))
    for x in range(-6, 7):
        for y in range(-6, 7):
            vector = np.array([x, y], dtype=float)
            in_left = np.allclose(
                vector @ np.linalg.inv(left.astype(float)),
                np.round(vector @ np.linalg.inv(left.astype(float))),
                atol=1e-9,
            )
            in_right = np.allclose(
                vector @ np.linalg.inv(right.astype(float)),
                np.round(vector @ np.linalg.inv(right.astype(float))),
                atol=1e-9,
            )
            in_shared = np.allclose(vector @ inverse, np.round(vector @ inverse), atol=1e-9)
            assert in_shared == (in_left and in_right)


def _common_sublattice_dets(left, right, bound):
    """Determinants of every common sublattice with entries inside ``bound``."""

    left_inverse = np.linalg.inv(left.astype(float))
    right_inverse = np.linalg.inv(right.astype(float))
    found = []
    span = range(-bound, bound + 1)
    for a in span:
        for b in span:
            for c in span:
                for d in span:
                    candidate = np.array([[a, b], [c, d]], dtype=np.int64)
                    determinant = a * d - b * c
                    if determinant == 0:
                        continue
                    for parent_inverse in (left_inverse, right_inverse):
                        coefficients = candidate.astype(float) @ parent_inverse
                        if not np.allclose(coefficients, np.round(coefficients), atol=1e-9):
                            break
                    else:
                        found.append(determinant)
    return found


@pytest.mark.parametrize(
    "first, second",
    [
        ([[2, 0], [0, 3]], [[3, 0], [0, 2]]),
        ([[1, 0], [0, 1]], [[5, 2], [3, 4]]),
        ([[2, 1], [0, 2]], [[3, 0], [0, 3]]),
    ],
)
def test_intersection_cell_is_the_smallest_common_cell(first, second):
    """``Cellstine.isLeast_abs_det_inf`` and ``Cellstine.det_dvd_of_rowLattice_le``.

    Every cell shared by the two layers holds a whole number of copies of the
    intersection cell, so the intersection is the smallest one.
    """

    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    shared = nlayer.sublattice_intersection(left, right)
    index = abs(round(float(np.linalg.det(shared))))

    determinants = _common_sublattice_dets(left, right, 6)
    assert determinants, "the brute-force scan must find the intersection itself"
    assert min(abs(value) for value in determinants) == index
    assert all(abs(value) % index == 0 for value in determinants)


@pytest.mark.parametrize(
    "first, second",
    [
        ([[2, 0], [0, 3]], [[3, 0], [0, 2]]),
        ([[1, 0], [0, 1]], [[5, 2], [3, 4]]),
        ([[2, 1], [0, 2]], [[3, 0], [0, 3]]),
        ([[4, 1], [1, 4]], [[6, 0], [3, 2]]),
    ],
)
def test_intersection_index_is_at_most_the_product_of_the_layer_indices(first, second):
    """``Cellstine.card_quotient_inf_le``."""

    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    shared = nlayer.sublattice_intersection(left, right)
    index = abs(round(float(np.linalg.det(shared))))
    product = abs(round(float(np.linalg.det(left)))) * abs(round(float(np.linalg.det(right))))
    assert 0 < index <= product


def test_the_quotient_of_a_common_cell_is_unique_and_integrality_is_exact():
    """``Cellstine.factor_unique`` and ``Cellstine.rowLattice_le_iff_exists_factor``."""

    coarse = np.array([[2, 1], [0, 3]], dtype=np.int64)
    fine = np.array([[4, 5], [0, 9]], dtype=np.int64)
    quotient = nlayer.quotient_matrix(coarse, fine)
    assert np.array_equal(quotient @ coarse, fine)
    # Any other integer factor would have to agree with it.
    for delta in ([[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 1]]):
        other = quotient + np.asarray(delta, dtype=np.int64)
        assert not np.array_equal(other @ coarse, fine)
    # A lattice that is not a sublattice is refused rather than rounded.
    with pytest.raises(ValueError):
        nlayer.quotient_matrix(coarse, np.array([[1, 0], [0, 1]], dtype=np.int64))


def test_the_intersection_of_several_layers_does_not_depend_on_the_order():
    matrices = [
        np.array([[2, 0], [0, 3]], dtype=np.int64),
        np.array([[3, 1], [0, 2]], dtype=np.int64),
        np.array([[1, 0], [2, 5]], dtype=np.int64),
    ]
    reference = nlayer.intersect_sublattices(matrices)
    for order in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        permuted = nlayer.intersect_sublattices([matrices[index] for index in order])
        # Same lattice: each is an integer multiple of the other.
        assert np.array_equal(
            nlayer.quotient_matrix(reference, permuted) @ reference, permuted
        )
        assert np.array_equal(
            nlayer.quotient_matrix(permuted, reference) @ permuted, reference
        )
    for matrix in matrices:
        assert np.array_equal(nlayer.quotient_matrix(matrix, reference) @ matrix, reference)


def test_reduced_supercell_keeps_the_lattice_and_shortens_the_cell():
    base = np.array([[2.46, 0.0], [-1.23, 2.46 * math.sqrt(3.0) / 2.0]])
    matrix = np.array([[7, 13], [2, 4]], dtype=np.int64)
    reduced = nlayer.reduce_supercell(matrix, base)
    transform = reduced.astype(float) @ np.linalg.inv(matrix.astype(float))
    assert np.allclose(transform, np.round(transform), atol=1e-9)
    assert round(abs(float(np.linalg.det(transform)))) == 1
    original_lengths = np.linalg.norm(matrix.astype(float) @ base, axis=1)
    reduced_lengths = np.linalg.norm(reduced.astype(float) @ base, axis=1)
    assert reduced_lengths.max() <= original_lengths.max() + 1e-9


def test_every_layer_realises_the_shared_lattice(trilayer_run):
    _, _, document = trilayer_run
    base = io_mod.read_poscar(document["search"]["base_poscar"])
    base_basis = _column_basis(base.lattice)
    assert document["candidates"], "the search must return at least one candidate"

    for candidate in document["candidates"]:
        shared = np.asarray(candidate["shared_lattice"], dtype=float)
        base_matrix = np.asarray(candidate["base_matrix"], dtype=float)
        assert np.allclose(base_matrix @ np.asarray(base.lattice, dtype=float)[:2, :2], shared, atol=1e-10)
        for layer in candidate["layers"]:
            structure = io_mod.read_poscar(layer["poscar"])
            rows = np.asarray(layer["matrix"], dtype=float) @ np.asarray(structure.lattice, dtype=float)[:2, :2]
            transformed = rows @ np.asarray(layer["affine"], dtype=float).T
            assert np.allclose(transformed, shared, atol=1e-9)
        assert base_basis.shape == (2, 2)


def test_reported_strains_are_the_singular_values_of_the_layer_affines(trilayer_run):
    _, _, document = trilayer_run
    for candidate in document["candidates"]:
        for layer in candidate["layers"]:
            singular = np.linalg.svd(np.asarray(layer["affine"], dtype=float), compute_uv=False)
            measured = np.sort(np.log(singular))
            reported = np.sort(np.asarray(layer["strain"], dtype=float))
            assert np.allclose(measured, reported, atol=1e-9)


def test_atom_counts_follow_from_the_supercell_determinants(trilayer_run):
    _, _, document = trilayer_run
    base = io_mod.read_poscar(document["search"]["base_poscar"])
    for candidate in document["candidates"]:
        base_multiplicity = round(abs(float(np.linalg.det(np.asarray(candidate["base_matrix"], dtype=float)))))
        assert candidate["base_atom_count"] == base_multiplicity * base.natoms
        total = candidate["base_atom_count"]
        for layer in candidate["layers"]:
            structure = io_mod.read_poscar(layer["poscar"])
            multiplicity = round(abs(float(np.linalg.det(np.asarray(layer["matrix"], dtype=float)))))
            assert layer["atom_count"] == multiplicity * structure.natoms
            total += layer["atom_count"]
        assert candidate["total_atoms"] == total


def test_the_base_layer_is_never_strained(trilayer_run):
    _, _, document = trilayer_run
    assert document["search"]["base_strain"] == 0.0
    base = io_mod.read_poscar(document["search"]["base_poscar"])
    reference = _minimum_distance(base.lattice, base.positions_direct)
    for candidate in document["candidates"][:4]:
        shared = np.asarray(candidate["shared_lattice"], dtype=float)
        lattice = np.vstack([np.hstack([shared, np.zeros((2, 1))]), base.lattice[2]])
        matrix = np.asarray(candidate["base_matrix"], dtype=np.int64)
        cell = matrix.astype(float) @ np.asarray(base.lattice, dtype=float)[:2, :2]
        assert np.allclose(cell, shared, atol=1e-10)
        assert lattice.shape == (3, 3)
    assert reference > 0.0


def test_built_stack_has_the_shared_cell_and_the_requested_geometry(trilayer_run):
    workflow, results, document = trilayer_run
    index = document["candidates"][1]["index"]
    built = workflow.maken(
        results_file=results, indexes=[index], interlayers=[3.4, 3.2], vacuum=18.0
    )
    structure = io_mod.read_poscar(built.artifacts["structures"][0])
    candidate = next(item for item in document["candidates"] if item["index"] == index)

    assert structure.natoms == candidate["total_atoms"]
    assert np.allclose(structure.lattice[:2, :2], np.asarray(candidate["shared_lattice"], dtype=float), atol=1e-9)

    heights = structure.positions_cartesian[:, 2]
    levels = np.sort(np.unique(np.round(heights, 6)))
    assert len(levels) == 3, "one flat level per layer"
    assert float(levels[1] - levels[0]) == pytest.approx(3.4, abs=1e-9)
    assert float(levels[2] - levels[1]) == pytest.approx(3.2, abs=1e-9)

    span = float(heights.max() - heights.min())
    cell_height = float(np.linalg.norm(structure.lattice[2]))
    assert cell_height == pytest.approx(span + 18.0, abs=1e-9)
    assert float(heights.min()) == pytest.approx(9.0, abs=1e-9)


def test_the_unstrained_layers_keep_their_bond_lengths(trilayer_run):
    workflow, results, document = trilayer_run
    candidate = next(
        item
        for item in document["candidates"]
        if item["total_atoms"] > 6 and item["layers"][1]["max_abs_strain"] < 1e-12
    )
    built = workflow.maken(
        results_file=results, indexes=[candidate["index"]], interlayers=3.35, vacuum=15.0
    )
    structure = io_mod.read_poscar(built.artifacts["structures"][0])
    heights = structure.positions_cartesian[:, 2]
    levels = np.sort(np.unique(np.round(heights, 6)))

    graphene_bond = 2.46 / math.sqrt(3.0)
    for level, expected in ((levels[0], graphene_bond), (levels[2], graphene_bond)):
        layer = structure.positions_direct[np.isclose(heights, level, atol=1e-6)]
        assert _minimum_distance(structure.lattice, layer) == pytest.approx(expected, abs=1e-9)

    strained = structure.positions_direct[np.isclose(heights, levels[1], atol=1e-6)]
    strain = candidate["layers"][0]["strain"][0]
    assert _minimum_distance(structure.lattice, strained) == pytest.approx(
        (2.504 / math.sqrt(3.0)) * math.exp(strain), abs=1e-6
    )


def test_a_document_that_breaks_commensuration_is_rejected(trilayer_run, tmp_path):
    import json

    from cellstine.moire.builder import nlayer as builder

    _, results, document = trilayer_run
    broken = json.loads(open(results).read())
    broken["candidates"][0]["layers"][0]["matrix"] = [[1, 0], [0, 1]]
    path = tmp_path / "broken_nlayer.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="shared lattice"):
        builder.generate_from_results(str(path), index=1, interlayers=[3.35, 3.35], output_dir=str(tmp_path))

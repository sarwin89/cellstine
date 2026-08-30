"""The restricted square/hexagonal branch of the moire search.

``cellstine.moire.search.gram.search`` has two engines.  The general one joins
two families of reduced Gram forms; the restricted one, selected with
``symmetric=True``, is only offered when both layers carry the same four- or
six-fold rotation, and then enumerates supercells of the special form
``(v, R v)`` -- the sublattices that the layer rotation ``R`` maps onto
themselves.

That restriction is exactly what makes the branch fast, and it is also what
makes it worth checking, because a supercell it cannot express is a candidate it
cannot report.  The mathematics is in ``RequestProject/SymmetricSupercell.lean``:

* a sublattice of a square or hexagonal lattice is rotation invariant precisely
  when it is ``(v, R v)`` for one of its own vectors ``v``
  (``Cellstine.invariant_iff_exists_generator``), so the enumeration is complete
  for the cells it claims;
* such a cell holds ``Q(v)`` primitive cells with ``Q`` the invariant quadratic
  form, which is the squared length the enumeration already sorts on
  (``Cellstine.index_square``, ``Cellstine.index_hex``);
* two such cells, one per layer, are always related by a rotation and a single
  overall scale (``Cellstine.similarity_of_generators``), so the strain of the
  join is isotropic and matching squared lengths is the whole strain test.

The tests below check those three claims on the engine's own output, and check
that the branch reports exactly the same physical classes as the general engine
wherever both are applicable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.symmetry2d import lattice_point_group
from cellstine.moire.search import gram
from cellstine.moire.search import results as results_mod
from cellstine.moire.search.find import run_find

from conftest import hexagonal_basis, write_poscar
from reference_moire import ReferenceConfig, reference_search

_SHARED_INPUTS = (
    "top_basis",
    "bottom_basis",
    "max_length",
    "top_strain",
    "bottom_strain",
    "top_atoms",
    "bottom_atoms",
    "min_length",
    "max_atoms",
    "max_aspect_ratio",
    "min_cell_angle_deg",
    "max_cell_angle_deg",
    "primitive_only",
    "top_group",
    "bottom_group",
)

THREE_FOLD = np.array(
    [[[1, 0], [0, 1]], [[0, -1], [1, -1]], [[-1, 1], [-1, 0]]], dtype=np.int64
)


def square_basis(constant: float) -> np.ndarray:
    return np.array([[constant, 0.0], [0.0, constant]])


def oblique_basis(first: float, second: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [[first, second * math.cos(angle)], [0.0, second * math.sin(angle)]]
    )


def _signature(angle, top, bottom, strains, area) -> tuple:
    ordered = np.sort(np.asarray(strains, dtype=float))
    return (
        round(float(angle), 5),
        int(top),
        int(bottom),
        round(float(ordered[0]), 7),
        round(float(ordered[1]), 7),
        round(float(area), 5),
    )


def _engine_signatures(result) -> set[tuple]:
    signatures = set()
    for row in range(len(result)):
        g11, g12, g22 = result.top_gram[row]
        signatures.add(
            _signature(
                result.twist_degrees[row],
                result.top_atom_counts[row],
                result.bottom_atom_counts[row],
                result.principal_strains[row],
                math.sqrt(max(g11 * g22 - g12 * g12, 0.0)),
            )
        )
    return signatures


def _reference_signatures(reference) -> set[tuple]:
    return {
        _signature(item.twist_deg, item.top_atoms, item.bottom_atoms, item.strains, item.top_area)
        for item in reference
    }


#: Searches both engines must agree on.  Every one is primitive: the restricted
#: branch only enumerates rotation-invariant supercells, and a plain multiple of
#: a smaller commensurate cell need not be one of those, so ``primitive_only``
#: is what makes the two engines comparable at all.
CASES = {
    "graphene_homobilayer_rigid": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.46),
        max_length=16.0,
        top_strain=0.0,
        bottom_strain=0.0,
    ),
    "graphene_on_hbn": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.504),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
    ),
    "hexagonal_length_floor": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
        min_length=7.0,
    ),
    "hexagonal_atom_ceiling": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
        top_atoms=2,
        bottom_atoms=2,
        max_atoms=120,
    ),
    "hexagonal_twist_window": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
        min_twist_angle_deg=2.0,
        max_twist_angle_deg=20.0,
    ),
    "hexagonal_three_fold_layers": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
        top_group=THREE_FOLD,
        bottom_group=THREE_FOLD,
    ),
    "hexagonal_one_sided_strain": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=14.0,
        top_strain=0.0,
        bottom_strain=0.03,
    ),
    "hexagonal_unfolded": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=12.0,
        top_strain=0.01,
        bottom_strain=0.01,
        fold_symmetry=False,
    ),
    "square_pair": dict(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.15),
        max_length=13.0,
        top_strain=0.01,
        bottom_strain=0.01,
    ),
    "square_rigid": dict(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.0),
        max_length=13.0,
        top_strain=0.0,
        bottom_strain=0.0,
    ),
    "square_unfolded": dict(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.1),
        max_length=12.0,
        top_strain=0.02,
        bottom_strain=0.02,
        fold_symmetry=False,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_restricted_branch_reports_the_general_classes(name):
    """On a primitive search the two engines report the same bilayers."""

    general = gram.search(gram.SearchConfig(**CASES[name]))
    restricted = gram.search(gram.SearchConfig(**CASES[name], symmetric=True))
    assert _engine_signatures(restricted) == _engine_signatures(general)
    assert len(restricted) > 0


def test_the_restricted_branch_matches_brute_force():
    """Set equality with the independent enumeration, not just with the engine."""

    config = gram.SearchConfig(**CASES["graphene_on_hbn"], symmetric=True)
    reference = reference_search(
        ReferenceConfig(**{key: getattr(config, key) for key in _SHARED_INPUTS})
    )
    assert _engine_signatures(gram.search(config)) == _reference_signatures(reference)


def _rotation_generator(basis: np.ndarray) -> np.ndarray:
    """Return the smallest proper rotation of ``basis`` other than the identity."""

    group = lattice_point_group(basis)
    proper = [
        element
        for element in group
        if int(round(np.linalg.det(element))) == 1
        and not np.array_equal(element, np.eye(2, dtype=np.int64))
    ]
    assert proper, "a square or hexagonal lattice has a proper rotation"
    return min(proper, key=lambda element: -int(element[0, 0] + element[1, 1]))


def _is_invariant(matrix: np.ndarray, rotation: np.ndarray) -> bool:
    """Return whether the sublattice spanned by ``matrix`` is ``rotation`` invariant."""

    image = np.linalg.solve(matrix.astype(float), rotation.astype(float) @ matrix.astype(float))
    return bool(np.allclose(image, np.round(image), atol=1e-9))


@pytest.mark.parametrize(
    "top_constant, bottom_constant, builder",
    [(2.46, 2.5, hexagonal_basis), (3.0, 3.1, square_basis)],
)
def test_every_reported_supercell_is_rotation_invariant(top_constant, bottom_constant, builder):
    config = gram.SearchConfig(
        top_basis=builder(top_constant),
        bottom_basis=builder(bottom_constant),
        max_length=13.0,
        top_strain=0.015,
        bottom_strain=0.015,
        symmetric=True,
    )
    result = gram.search(config)
    assert len(result) > 0
    top_rotation = _rotation_generator(config.top_basis)
    bottom_rotation = _rotation_generator(config.bottom_basis)
    for row in range(len(result)):
        assert _is_invariant(result.top_matrices[row], top_rotation)
        assert _is_invariant(result.bottom_matrices[row], bottom_rotation)


@pytest.mark.parametrize(
    "builder, constants", [(hexagonal_basis, (2.46, 2.5)), (square_basis, (3.0, 3.1))]
)
def test_the_restricted_branch_only_strains_isotropically(builder, constants):
    """Two rotation-invariant cells differ by a rotation and one scale."""

    result = gram.search(
        gram.SearchConfig(
            top_basis=builder(constants[0]),
            bottom_basis=builder(constants[1]),
            max_length=13.0,
            top_strain=0.015,
            bottom_strain=0.015,
            symmetric=True,
        )
    )
    assert len(result) > 0
    first, second = result.principal_strains[:, 0], result.principal_strains[:, 1]
    assert np.allclose(first, second, atol=1e-12)


def _loeschian(value: int) -> bool:
    """Return whether ``value = x^2 + x y + y^2`` has an integer solution."""

    bound = int(math.isqrt(4 * value // 3)) + 2
    return any(
        x * x + x * y + y * y == value
        for x in range(-bound, bound + 1)
        for y in range(-bound, bound + 1)
    )


def _sum_of_two_squares(value: int) -> bool:
    """Return whether ``value = x^2 + y^2`` has an integer solution."""

    return any(
        math.isqrt(value - x * x) ** 2 == value - x * x
        for x in range(math.isqrt(value) + 1)
    )


@pytest.mark.parametrize(
    "builder, constant, predicate",
    [(hexagonal_basis, 2.46, _loeschian), (square_basis, 3.0, _sum_of_two_squares)],
)
def test_the_cell_index_is_the_invariant_quadratic_form(builder, constant, predicate):
    """A rotation-invariant cell holds ``Q(v)`` primitive cells, so its index is a value of ``Q``."""

    result = gram.search(
        gram.SearchConfig(
            top_basis=builder(constant),
            bottom_basis=builder(constant),
            max_length=15.0,
            top_strain=0.0,
            bottom_strain=0.0,
            symmetric=True,
        )
    )
    assert len(result) > 0
    for row in range(len(result)):
        multiplicity = int(round(abs(np.linalg.det(result.top_matrices[row].astype(float)))))
        assert multiplicity == int(result.top_atom_counts[row])
        assert predicate(multiplicity)


def test_the_branch_is_offered_exactly_when_it_applies():
    hexagonal = gram.SearchConfig(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=10.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    square = gram.SearchConfig(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.1),
        max_length=10.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    mixed = gram.SearchConfig(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=square_basis(3.0),
        max_length=10.0,
        top_strain=0.02,
        bottom_strain=0.02,
    )
    oblique = gram.SearchConfig(
        top_basis=oblique_basis(3.0, 3.4, 75.0),
        bottom_basis=oblique_basis(3.05, 3.3, 78.0),
        max_length=10.0,
        top_strain=0.02,
        bottom_strain=0.02,
    )
    assert gram.symmetric_branch_applies(hexagonal)
    assert gram.symmetric_branch_applies(square)
    assert not gram.symmetric_branch_applies(mixed)
    assert not gram.symmetric_branch_applies(oblique)


@pytest.mark.parametrize("builder", [oblique_basis, None])
def test_an_inapplicable_branch_is_refused_rather_than_guessed(builder):
    if builder is None:
        config = gram.SearchConfig(
            top_basis=hexagonal_basis(2.46),
            bottom_basis=square_basis(3.0),
            max_length=10.0,
            top_strain=0.02,
            bottom_strain=0.02,
            symmetric=True,
        )
    else:
        config = gram.SearchConfig(
            top_basis=builder(3.0, 3.4, 75.0),
            bottom_basis=builder(3.05, 3.3, 78.0),
            max_length=10.0,
            top_strain=0.02,
            bottom_strain=0.02,
            symmetric=True,
        )
    with pytest.raises(gram.SymmetricBranchUnavailable):
        gram.search(config)


def test_the_restricted_branch_never_invents_a_candidate_on_an_imprimitive_search():
    """With ``primitive_only=False`` the branch reports a subset, never a surprise.

    A multiple of a commensurate cell is only enumerated here when it is itself
    rotation invariant, so the restricted branch reports fewer classes than the
    general engine -- but every one of them is a class the general engine also
    reports.
    """

    inputs = dict(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.0),
        max_length=13.0,
        top_strain=0.0,
        bottom_strain=0.0,
        primitive_only=False,
    )
    general = _engine_signatures(gram.search(gram.SearchConfig(**inputs)))
    restricted = _engine_signatures(gram.search(gram.SearchConfig(**inputs, symmetric=True)))
    assert restricted
    assert restricted <= general
    assert len(restricted) < len(general)


def _rectangular_layer(path, first: float, second: float):
    lattice = np.diag([first, second, 20.0])
    return write_poscar(path, lattice, ["C"], [1], np.array([[0.0, 0.0, 0.5]]))


def test_the_workflow_records_that_the_restricted_branch_ran(tmp_path, graphene_poscar):
    run = run_find(
        top_poscar=str(graphene_poscar),
        bottom_poscar=str(graphene_poscar),
        max_length=14.0,
        top_strain=0.0,
        bottom_strain=0.0,
        symmetric=True,
        output_root=str(tmp_path),
    )
    document = results_mod.read_results(str(run.result_path))
    metadata = document["metadata"]
    assert metadata["symmetric_requested"] is True
    assert metadata["symmetric_used"] is True
    assert metadata["symmetric_fallback"] is None
    assert document["candidates"]


def test_the_workflow_falls_back_when_the_restricted_branch_does_not_apply(tmp_path):
    top = _rectangular_layer(tmp_path / "top.vasp", 3.0, 4.0)
    bottom = _rectangular_layer(tmp_path / "bottom.vasp", 3.1, 3.9)
    run = run_find(
        top_poscar=str(top),
        bottom_poscar=str(bottom),
        max_length=10.0,
        top_strain=0.03,
        bottom_strain=0.03,
        symmetric=True,
        output_root=str(tmp_path / "runs"),
    )
    document = results_mod.read_results(str(run.result_path))
    metadata = document["metadata"]
    assert metadata["symmetric_requested"] is True
    assert metadata["symmetric_used"] is False
    assert isinstance(metadata["symmetric_fallback"], str)
    assert "square or hexagonal" in metadata["symmetric_fallback"]
    assert document["candidates"]


def test_a_half_turn_is_reported_as_a_positive_angle():
    """``arctan2`` can return ``-pi``; the two engines must not disagree on its sign."""

    inputs = dict(
        top_basis=square_basis(3.0),
        bottom_basis=square_basis(3.1),
        max_length=12.0,
        top_strain=0.02,
        bottom_strain=0.02,
        fold_symmetry=False,
    )
    for symmetric in (False, True):
        result = gram.search(gram.SearchConfig(**inputs, symmetric=symmetric))
        angles = np.asarray(result.twist_degrees, dtype=float)
        assert np.any(np.isclose(angles, 180.0))
        assert not np.any(np.isclose(angles, -180.0))

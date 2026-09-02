"""Mathematical invariants and reference cross-checks of the moire search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.species import expand_species
from cellstine.core.symmetry2d import column_basis_from_lattice, layer_point_group
from cellstine.io import native as io
from cellstine.moire.search import gram
from cellstine.moire.search import gram_pairs, gram_report

from conftest import hexagonal_basis
from reference_moire import ReferenceConfig, reference_search


def _config(top_path, bottom_path, **kwargs) -> gram.SearchConfig:
    top = io.read_poscar(str(top_path))
    bottom = io.read_poscar(str(bottom_path))
    return gram.SearchConfig(
        top_basis=column_basis_from_lattice(top.lattice),
        bottom_basis=column_basis_from_lattice(bottom.lattice),
        top_atoms=top.natoms,
        bottom_atoms=bottom.natoms,
        top_group=layer_point_group(
            top.lattice, top.positions_direct, expand_species(top.species, top.counts)
        ),
        bottom_group=layer_point_group(
            bottom.lattice,
            bottom.positions_direct,
            expand_species(bottom.species, bottom.counts),
        ),
        **kwargs,
    )


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )


@pytest.fixture(scope="module")
def graphene_result(graphene_poscar):
    config = _config(
        graphene_poscar,
        graphene_poscar,
        max_length=25.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    return config, gram.search(config)


def test_reported_gram_matrices_describe_the_reported_cells(graphene_result):
    config, result = graphene_result
    for row in range(len(result)):
        top_cell = config.top_basis @ result.top_matrices[row]
        bottom_cell = config.bottom_basis @ result.bottom_matrices[row]
        assert np.allclose(top_cell.T @ top_cell, _square(result.top_gram[row]), rtol=1e-9)
        assert np.allclose(
            bottom_cell.T @ bottom_cell, _square(result.bottom_gram[row]), rtol=1e-9
        )


def _square(triple: np.ndarray) -> np.ndarray:
    return np.array([[triple[0], triple[1]], [triple[1], triple[2]]])


def test_twist_angle_is_the_rotation_of_the_relative_deformation(graphene_result):
    config, result = graphene_result
    for row in range(len(result)):
        top_cell = config.top_basis @ result.top_matrices[row]
        bottom_cell = config.bottom_basis @ result.bottom_matrices[row]
        deformation = bottom_cell @ np.linalg.inv(top_cell)
        stretch = _rotation(result.twist_radians[row]).T @ deformation
        assert np.allclose(stretch, stretch.T, atol=1e-9)
        eigenvalues = np.linalg.eigvalsh(0.5 * (stretch + stretch.T))
        assert np.allclose(
            np.sort(np.log(eigenvalues)),
            np.sort(result.principal_strains[row]),
            atol=1e-7,
        )


def test_affine_maps_send_both_layers_onto_the_shared_cell(graphene_result):
    """Both strained superlattices must equal the shared cell to machine precision.

    The tolerance is deliberately tight: the layer affines are built from a
    divided difference of the principal stretches, which cancels badly for an
    almost isotropic match, and a loose tolerance would hide exactly that loss.
    """

    config, result = graphene_result
    for row in range(len(result)):
        top_cell = config.top_basis @ result.top_matrices[row]
        bottom_cell = config.bottom_basis @ result.bottom_matrices[row]
        shared = result.shared_lattice[row]
        scale = np.abs(shared).max()
        assert np.allclose(result.top_affine[row] @ top_cell, shared, atol=1e-13 * scale)
        assert np.allclose(
            result.bottom_affine[row] @ bottom_cell, shared, atol=1e-13 * scale
        )


def test_layer_affines_realise_the_reported_layer_strains(graphene_result):
    """The singular values of each affine must be the reported layer stretches.

    The affine is assembled from the floating-point relative deformation of two
    large supercells, so the strain it actually applies can differ from the
    exactly computed one by the conditioning of that product; the tolerance here
    is what that allows, and it is nine orders of magnitude below any strain a
    calculation would resolve.
    """

    _, result = graphene_result
    for row in range(len(result)):
        for affine, strains in (
            (result.top_affine[row], result.top_layer_strains[row]),
            (result.bottom_affine[row], result.bottom_layer_strains[row]),
        ):
            measured = np.sort(np.log(np.linalg.svd(affine, compute_uv=False)))
            assert np.allclose(measured, np.sort(strains), atol=1e-9)


@pytest.mark.parametrize("power", [-1.0, -0.5, 0.25, 0.5, 1.5])
@pytest.mark.parametrize("gap", [0.0, 1e-14, 1e-11, 1e-8, 1e-4])
def test_divided_difference_power_is_accurate_near_degeneracy(power, gap):
    """``(a**p - b**p)/(a - b)`` must stay accurate as the stretches merge.

    For a relative half-difference ``d`` the exact value is
    ``p m**(p-1) (1 + (p-1)(p-2) d**2 / 6 + O(d**4))``, so for the small gaps here
    that truncated series is itself the answer to well below double precision.  A
    naive quotient loses about one digit per decade of ``d`` and fails this check.
    """

    mean = 1.3
    first, second = mean * (1.0 + gap), mean * (1.0 - gap)
    value = float(
        gram_report._divided_difference_power(
            np.array([first]), np.array([second]), np.array([power])
        )[0]
    )
    limit = power * mean ** (power - 1.0)
    correction = 1.0 + (power - 1.0) * (power - 2.0) * gap * gap / 6.0
    assert value == pytest.approx(limit * correction, rel=1e-13)


@pytest.mark.parametrize("power", [-1.0, -0.5, 0.25, 0.5, 1.5])
def test_divided_difference_power_matches_the_direct_quotient_when_separated(power):
    """Well away from degeneracy the stable form must agree with the plain ratio."""

    first, second = 1.7, 0.6
    value = float(
        gram_report._divided_difference_power(
            np.array([first]), np.array([second]), np.array([power])
        )[0]
    )
    expected = (first**power - second**power) / (first - second)
    assert value == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 1e-9, 1e-3])
def test_matrix_power_matches_the_eigen_decomposition(gap):
    """``S**p`` from the Cayley-Hamilton form must match a direct diagonalisation."""

    angle = 0.7
    rotation = _rotation(angle)
    eigenvalues = np.array([1.05 * (1.0 + gap), 1.05 * (1.0 - gap)])
    matrix = rotation @ np.diag(eigenvalues) @ rotation.T
    for power in (-1.0, -0.5, 0.5, 1.0, 2.0):
        computed = gram_report._matrix_power_spd(
            matrix[None],
            np.array([eigenvalues[0]]),
            np.array([eigenvalues[1]]),
            np.array([power]),
        )[0]
        expected = rotation @ np.diag(eigenvalues**power) @ rotation.T
        assert np.allclose(computed, expected, rtol=1e-13, atol=1e-14)


def test_shared_cell_is_reduced_right_handed_and_x_aligned(graphene_result):
    _, result = graphene_result
    for row in range(len(result)):
        shared = result.shared_lattice[row]
        assert shared[0, 0] > 0.0
        assert abs(shared[1, 0]) <= 1e-9 * abs(shared[0, 0])
        assert np.linalg.det(shared) > 0.0
        g11 = float(shared[:, 0] @ shared[:, 0])
        g22 = float(shared[:, 1] @ shared[:, 1])
        g12 = float(shared[:, 0] @ shared[:, 1])
        assert g11 <= g22 * (1.0 + 1e-9)
        assert 2.0 * abs(g12) <= g11 * (1.0 + 1e-9)


def test_twist_angles_lie_in_the_fundamental_range(graphene_result):
    config, result = graphene_result
    half_period = 0.5 * config.angle_period_radians
    assert np.all(np.abs(result.twist_radians) <= half_period + 1e-9)


def test_layer_strains_share_the_relative_strain(graphene_result):
    _, result = graphene_result
    total = result.top_layer_strains - result.bottom_layer_strains
    assert np.allclose(total, result.principal_strains, atol=1e-12)
    assert np.all(np.abs(result.top_layer_strains) <= 0.01 + 1e-9)
    assert np.all(np.abs(result.bottom_layer_strains) <= 0.01 + 1e-9)


def test_wide_search_drops_numerical_join_rows_outside_the_strain_budget():
    basis = hexagonal_basis(2.46)
    config = gram.SearchConfig(
        top_basis=basis,
        bottom_basis=basis,
        max_length=90.0,
        top_strain=0.002,
        bottom_strain=0.002,
        top_atoms=2,
        bottom_atoms=2,
    )

    result = gram.search(config)

    assert np.all(np.abs(result.top_layer_strains) <= config.top_strain + 1e-9)
    assert np.all(np.abs(result.bottom_layer_strains) <= config.bottom_strain + 1e-9)


def test_every_reported_cell_is_a_primitive_coincidence_cell(graphene_result):
    _, result = graphene_result
    assert np.all(result.coincidence_indices == 1)


def test_candidates_are_ranked_by_size_then_strain(graphene_result):
    _, result = graphene_result
    assert np.all(np.diff(result.atom_counts) >= 0)
    assert np.array_equal(result.rank, np.arange(1, len(result) + 1))


def test_no_two_candidates_describe_the_same_bilayer(graphene_result):
    config, result = graphene_result
    keys = gram_pairs._pair_orbit_keys(
        result.top_matrices, result.bottom_matrices, config.top_group, config.bottom_group
    )
    unique = {tuple(row.tolist()) for row in keys}
    assert len(unique) == len(result)


def test_classic_twisted_bilayer_graphene_angles_are_found(graphene_result):
    _, result = graphene_result
    expected = [(21.786789, 28), (13.173551, 76), (9.430008, 148), (7.340993, 244)]
    reported = [
        (float(result.twist_degrees[row]), int(result.atom_counts[row]))
        for row in range(len(result))
        if np.max(np.abs(result.principal_strains[row])) < 1e-9
    ]
    for angle, atoms in expected:
        assert any(
            count == atoms and abs(value - angle) < 1e-5 for value, count in reported
        ), f"unstrained {angle} degree cell with {atoms} atoms was not reported"


def test_unstrained_commensurate_cells_have_hexagonal_moire_cells(graphene_result):
    _, result = graphene_result
    for row in range(len(result)):
        if np.max(np.abs(result.principal_strains[row])) > 1e-9:
            continue
        shared = result.shared_lattice[row]
        first = float(np.linalg.norm(shared[:, 0]))
        second = float(np.linalg.norm(shared[:, 1]))
        assert first == pytest.approx(second, rel=1e-9)
        cosine = float(shared[:, 0] @ shared[:, 1]) / (first * second)
        assert abs(abs(cosine) - 0.5) < 1e-9


def test_mos2_reports_both_r_and_h_stackings(mos2_poscar):
    """A three-fold layer has a 120 degree period, so 0 and 60 both survive."""

    config = _config(
        mos2_poscar, mos2_poscar, max_length=12.0, top_strain=0.005, bottom_strain=0.005
    )
    result = gram.search(config)
    assert config.angle_period_radians == pytest.approx(2.0 * math.pi / 3.0)
    angles = sorted(round(float(value), 6) for value in result.twist_degrees)
    assert 0.0 in angles
    assert 60.0 in angles


def test_graphene_bilayer_matches_the_brute_force_reference(graphene_poscar):
    config = _config(
        graphene_poscar,
        graphene_poscar,
        max_length=14.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    result = gram.search(config)
    reference = reference_search(
        ReferenceConfig(
            top_basis=config.top_basis,
            bottom_basis=config.bottom_basis,
            max_length=config.max_length,
            top_strain=config.top_strain,
            bottom_strain=config.bottom_strain,
            top_atoms=config.top_atoms,
            bottom_atoms=config.bottom_atoms,
            top_group=config.top_group,
            bottom_group=config.bottom_group,
        )
    )
    assert _signatures(result) == _reference_signatures(reference)


def test_graphene_on_hbn_matches_the_brute_force_reference(graphene_poscar, hbn_poscar):
    config = _config(
        graphene_poscar,
        hbn_poscar,
        max_length=12.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    result = gram.search(config)
    reference = reference_search(
        ReferenceConfig(
            top_basis=config.top_basis,
            bottom_basis=config.bottom_basis,
            max_length=config.max_length,
            top_strain=config.top_strain,
            bottom_strain=config.bottom_strain,
            top_atoms=config.top_atoms,
            bottom_atoms=config.bottom_atoms,
            top_group=config.top_group,
            bottom_group=config.bottom_group,
        )
    )
    assert _signatures(result) == _reference_signatures(reference)


def _rounded(angle: float, top: int, bottom: int, strains, area: float) -> tuple:
    ordered = np.sort(np.asarray(strains, dtype=float))
    return (
        round(float(angle), 5),
        int(top),
        int(bottom),
        round(float(ordered[0]), 7),
        round(float(ordered[1]), 7),
        round(float(area), 5),
    )


def _signatures(result) -> set[tuple]:
    signatures = set()
    for row in range(len(result)):
        g11, g12, g22 = result.top_gram[row]
        area = math.sqrt(max(g11 * g22 - g12 * g12, 0.0))
        signatures.add(
            _rounded(
                result.twist_degrees[row],
                result.top_atom_counts[row],
                result.bottom_atom_counts[row],
                result.principal_strains[row],
                area,
            )
        )
    return signatures


def _reference_signatures(reference) -> set[tuple]:
    return {
        _rounded(
            item.twist_deg, item.top_atoms, item.bottom_atoms, item.strains, item.top_area
        )
        for item in reference
    }


def test_disabling_symmetry_folding_only_adds_candidates(graphene_poscar):
    folded = gram.search(
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.01,
            bottom_strain=0.01,
        )
    )
    unfolded = gram.search(
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.01,
            bottom_strain=0.01,
            fold_symmetry=False,
        )
    )
    assert len(unfolded) >= len(folded)
    assert set(np.round(np.abs(folded.twist_degrees), 6)) <= set(
        np.round(np.abs(unfolded.twist_degrees), 6)
    )


def test_allowing_imprimitive_cells_recovers_the_dropped_supercells(graphene_poscar):
    primitive = gram.search(
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.01,
            bottom_strain=0.01,
        )
    )
    everything = gram.search(
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.01,
            bottom_strain=0.01,
            primitive_only=False,
        )
    )
    assert len(everything) > len(primitive)
    assert np.any(everything.coincidence_indices > 1)


def test_a_twist_window_selects_exactly_the_candidates_inside_it(graphene_poscar):
    """The window is a filter on the reported twist, not a different search.

    Restricting a search to a band of twist angles must return precisely the
    candidates of the unrestricted search whose angle lies in that band -- same
    integer matrices, same angles, nothing gained and nothing lost.
    """

    def search(**extra):
        return gram.search(
            _config(
                graphene_poscar,
                graphene_poscar,
                max_length=25.0,
                top_strain=0.0,
                bottom_strain=0.0,
                **extra,
            )
        )

    everything = search()
    windowed = search(min_twist_angle_deg=9.0, max_twist_angle_deg=14.0)

    inside = np.abs(everything.twist_degrees)
    expected = np.sort(inside[(inside >= 9.0) & (inside <= 14.0)])
    assert len(expected) > 0, "graphene has commensurate twists between 9 and 14 degrees"
    assert np.allclose(np.sort(np.abs(windowed.twist_degrees)), expected)

    keys = {tuple(row) for row in everything.canonical_keys.tolist()}
    assert {tuple(row) for row in windowed.canonical_keys.tolist()} <= keys


def test_a_twist_window_keeps_an_angle_that_sits_on_its_own_bound(graphene_poscar):
    """Asking for exactly 21.7868 degrees returns the 7-cell twisted bilayer."""

    # cos(theta) = (m^2 + 4mn + n^2) / (2 (m^2 + mn + n^2)) with m = 1, n = 2.
    angle = math.degrees(math.acos(13.0 / 14.0))
    result = gram.search(
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=25.0,
            top_strain=0.0,
            bottom_strain=0.0,
            min_twist_angle_deg=angle,
            max_twist_angle_deg=angle,
        )
    )
    assert len(result) == 1
    assert float(abs(result.twist_degrees[0])) == pytest.approx(angle, abs=1e-9)
    # m = 1, n = 2 gives m^2 + mn + n^2 = 7 primitive cells per layer.
    assert int(result.atom_counts[0]) == 2 * 7 * 2


def test_an_impossible_twist_window_is_rejected(graphene_poscar):
    with pytest.raises(ValueError):
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.0,
            bottom_strain=0.0,
            min_twist_angle_deg=20.0,
            max_twist_angle_deg=10.0,
        )
    with pytest.raises(ValueError):
        _config(
            graphene_poscar,
            graphene_poscar,
            max_length=12.0,
            top_strain=0.0,
            bottom_strain=0.0,
            min_twist_angle_deg=-1.0,
        )


def test_a_rigid_search_keeps_only_exactly_commensurate_pairs():
    """Both budgets zero is a legitimate request: twist with no strain at all.

    A twisted homobilayer is exactly commensurate at its magic angles, so the
    rigid search must find them, report zero strain, certify every pair, and
    return no more than the same search run with a small budget.
    """

    basis = hexagonal_basis(2.46)
    rigid = gram.SearchConfig(
        top_basis=basis,
        bottom_basis=basis,
        max_length=14.0,
        top_strain=0.0,
        bottom_strain=0.0,
    )
    assert rigid.is_rigid
    result = gram.search(rigid)
    assert len(result) > 0
    assert np.allclose(result.principal_strains, 0.0, atol=1e-9)
    assert result.loewner_certified.all()
    assert np.allclose(result.sharing_fraction, 0.5)

    relaxed = gram.search(
        gram.SearchConfig(
            top_basis=basis,
            bottom_basis=basis,
            max_length=14.0,
            top_strain=0.005,
            bottom_strain=0.005,
        )
    )
    rigid_angles = np.sort(np.round(result.twist_degrees, 6))
    relaxed_angles = np.sort(np.round(relaxed.twist_degrees, 6))
    assert set(rigid_angles).issubset(set(relaxed_angles))


def test_a_negative_strain_budget_is_rejected():
    basis = hexagonal_basis(2.46)
    with pytest.raises(ValueError):
        gram.SearchConfig(
            top_basis=basis,
            bottom_basis=basis,
            max_length=10.0,
            top_strain=-0.01,
            bottom_strain=0.0,
        )


def test_a_group_that_is_not_a_lattice_symmetry_is_rejected():
    basis = hexagonal_basis(2.46)
    with pytest.raises(ValueError):
        gram.SearchConfig(
            top_basis=basis,
            bottom_basis=basis,
            max_length=10.0,
            top_strain=0.01,
            bottom_strain=0.01,
            top_group=np.array([[[1, 1], [0, 1]]]),
        )


def test_coincidence_index_counts_repeated_cells():
    identity = np.eye(2, dtype=np.int64)[None, :, :]
    assert gram.coincidence_index(identity, identity)[0] == 1
    doubled = (2 * np.eye(2, dtype=np.int64))[None, :, :]
    # Doubling both layers repeats the primitive coincidence cell four times.
    assert gram.coincidence_index(doubled, doubled)[0] == 4
    assert gram.coincidence_index(doubled, identity)[0] == 1

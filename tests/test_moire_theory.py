"""Numerical checks of the engine against the results proved for it.

Each test here mirrors one of the statements in ``RequestProject`` that were
checked with Lean 4 and Mathlib; the point is that the running Python engine, and
not just the derivation, satisfies them.  The mapping is:

===================================  =====================================
test                                 formal statement
===================================  =====================================
equal Gram forms means congruent     ``Cellstine.gram_eq_iff_exists_orthogonal``
accepted pairs satisfy the sandwich  ``Cellstine.loewner_sandwich_iff_deformation_sandwich``
stretches lie in the budget window   ``Cellstine.abs_log_le_iff_mem_exp_interval``
relative strain fits the joint budget ``Cellstine.exists_strain_split_iff``
sharing equalises the budget load    ``Cellstine.isLeast_shared_strain``
the shared cell is reduced           ``Cellstine.first_minimum``, ``Cellstine.second_minimum``
atom counts scale by ``|det M|``     ``Cellstine.card_quotient_range_eq_natAbs_det``
===================================  =====================================
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.species import expand_species
from cellstine.core.symmetry2d import (
    column_basis_from_lattice,
    layer_point_group,
    symmetrised_basis,
)
from cellstine.io import native as io
from cellstine.moire.search import gram


def _config(top_path, bottom_path, **kwargs) -> gram.SearchConfig:
    top = io.read_poscar(str(top_path))
    bottom = io.read_poscar(str(bottom_path))
    top_group = layer_point_group(
        top.lattice, top.positions_direct, expand_species(top.species, top.counts)
    )
    bottom_group = layer_point_group(
        bottom.lattice,
        bottom.positions_direct,
        expand_species(bottom.species, bottom.counts),
    )
    top_basis, _ = symmetrised_basis(column_basis_from_lattice(top.lattice), top_group)
    bottom_basis, _ = symmetrised_basis(
        column_basis_from_lattice(bottom.lattice), bottom_group
    )
    return gram.SearchConfig(
        top_basis=top_basis,
        bottom_basis=bottom_basis,
        top_atoms=top.natoms,
        bottom_atoms=bottom.natoms,
        top_group=top_group,
        bottom_group=bottom_group,
        **kwargs,
    )


@pytest.fixture(scope="module")
def heterobilayer(graphene_poscar, hbn_poscar):
    config = _config(
        graphene_poscar,
        hbn_poscar,
        max_length=18.0,
        top_strain=0.012,
        bottom_strain=0.008,
    )
    return config, gram.search(config)


@pytest.fixture(scope="module")
def homobilayer(graphene_poscar):
    config = _config(
        graphene_poscar,
        graphene_poscar,
        max_length=18.0,
        top_strain=0.01,
        bottom_strain=0.01,
    )
    return config, gram.search(config)


def _supercells(config, result, row):
    top = config.top_basis @ result.top_matrices[row]
    bottom = config.bottom_basis @ result.bottom_matrices[row]
    return top, bottom


def test_zero_strain_candidates_are_congruent_supercells(homobilayer):
    """Equal Gram forms mean the two supercells differ by an orthogonal map."""

    config, result = homobilayer
    checked = 0
    for row in range(len(result)):
        if np.max(np.abs(result.principal_strains[row])) > 1e-12:
            continue
        top, bottom = _supercells(config, result, row)
        assert np.allclose(top.T @ top, bottom.T @ bottom, atol=1e-9)
        orthogonal = top @ np.linalg.inv(bottom)
        assert np.allclose(orthogonal.T @ orthogonal, np.eye(2), atol=1e-9)
        checked += 1
    assert checked > 0


def test_accepted_pairs_satisfy_the_loewner_sandwich(heterobilayer):
    """The join criterion bounds the relative deformation by the joint budget.

    ``strain`` is the logarithm of the principal stretches of the map that takes
    the *top* supercell onto the *bottom* one, so the deformation below is
    ``bottom @ top**-1``.
    """

    config, result = heterobilayer
    budget = config.top_strain + config.bottom_strain
    lower, upper = math.exp(-2.0 * budget), math.exp(2.0 * budget)
    for row in range(len(result)):
        top, bottom = _supercells(config, result, row)
        left = top.T @ top
        right = bottom.T @ bottom
        # the sandwich `lower * right <= left <= upper * right`, read as the
        # symmetric generalised eigenvalue problem
        values, vectors = np.linalg.eigh(right)
        whitened = vectors @ np.diag(values ** -0.5) @ vectors.T
        eigenvalues = np.linalg.eigvalsh(whitened @ left @ whitened)
        assert eigenvalues.min() >= lower - 1e-9
        assert eigenvalues.max() <= upper + 1e-9
        # equivalently, as a bound on the singular values of the deformation
        deformation = bottom @ np.linalg.inv(top)
        stretches = np.linalg.svd(deformation, compute_uv=False)
        assert stretches.min() >= math.exp(-budget) - 1e-9
        assert stretches.max() <= math.exp(budget) + 1e-9


def test_relative_strain_is_the_log_of_the_principal_stretches(heterobilayer):
    config, result = heterobilayer
    for row in range(len(result)):
        top, bottom = _supercells(config, result, row)
        deformation = bottom @ np.linalg.inv(top)
        stretches = np.sort(np.linalg.svd(deformation, compute_uv=False))
        reported = np.sort(result.principal_strains[row])
        assert np.allclose(np.log(stretches), reported, atol=1e-9)


def test_layer_strains_equalise_the_budget_load(heterobilayer):
    """Sharing is proportional to the budgets, which is the optimal split."""

    config, result = heterobilayer
    for row in range(len(result)):
        top_load = np.max(np.abs(result.top_layer_strains[row])) / config.top_strain
        bottom_load = (
            np.max(np.abs(result.bottom_layer_strains[row])) / config.bottom_strain
        )
        relative = np.max(np.abs(result.principal_strains[row]))
        expected = relative / (config.top_strain + config.bottom_strain)
        assert top_load == pytest.approx(bottom_load, abs=1e-9)
        assert top_load == pytest.approx(expected, abs=1e-9)
        assert top_load <= 1.0 + 1e-9


def test_relative_strain_never_exceeds_the_joint_budget(heterobilayer):
    config, result = heterobilayer
    budget = config.top_strain + config.bottom_strain
    for row in range(len(result)):
        assert np.max(np.abs(result.principal_strains[row])) <= budget + 1e-9


def test_shared_cell_realises_the_successive_minima(heterobilayer):
    """The reported moire cell is Lagrange-Gauss reduced, so it is shortest."""

    config, result = heterobilayer
    for row in range(len(result)):
        lattice = result.shared_lattice[row]
        first, second = lattice[:, 0], lattice[:, 1]
        a = float(first @ first)
        b = float(first @ second)
        c = float(second @ second)
        assert a <= c + 1e-9
        assert 2.0 * abs(b) <= a + 1e-9
        for m in range(-3, 4):
            for n in range(-3, 4):
                if m == 0 and n == 0:
                    continue
                length = m * m * a + 2.0 * m * n * b + n * n * c
                assert length >= a - 1e-9
                if n != 0:
                    assert length >= c - 1e-9


def test_atom_counts_follow_the_coincidence_index(heterobilayer):
    config, result = heterobilayer
    for row in range(len(result)):
        top_cells = abs(int(round(np.linalg.det(result.top_matrices[row]))))
        bottom_cells = abs(int(round(np.linalg.det(result.bottom_matrices[row]))))
        assert result.top_atom_counts[row] == top_cells * config.top_atoms
        assert result.bottom_atom_counts[row] == bottom_cells * config.bottom_atoms
        assert result.atom_counts[row] == (
            result.top_atom_counts[row] + result.bottom_atom_counts[row]
        )


def test_supercell_area_scales_by_the_determinant(heterobilayer):
    config, result = heterobilayer
    for row in range(len(result)):
        top, _ = _supercells(config, result, row)
        expected = np.linalg.det(config.top_basis) * np.linalg.det(
            result.top_matrices[row]
        )
        assert np.linalg.det(top) == pytest.approx(expected, rel=1e-12)

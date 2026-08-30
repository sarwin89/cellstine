"""Cross-check the fast Gram search against brute force on many lattice pairs.

``tests/test_moire_search.py`` already compares :func:`cellstine.moire.search.
gram.search` with the independent enumeration in ``benchmarks/reference_moire``
for graphene on graphene and graphene on hBN.  Those are both hexagonal, rigidly
strained and use the default filters, so they exercise only one corner of the
engine.  This module sweeps the rest of the input surface: square, rectangular
and oblique cells, unequal lattice constants, one-sided strain budgets, a length
floor, an atom ceiling, and imprimitive supercells.

Every case asserts *set equality* of the reported physical classes -- twist
angle, atoms per layer, principal strains and supercell area -- so a candidate
that the fast engine invents, or one that it silently drops, fails the test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.moire.search import gram

from conftest import hexagonal_basis
from reference_moire import ReferenceConfig, reference_search

# Inputs of ``ReferenceConfig`` that ``SearchConfig`` carries under the same
# name; the reference has no notion of twist windows, folding switches or the
# restricted symmetric branch, so those are left at their defaults here.
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


def _square(constant: float) -> np.ndarray:
    return np.array([[constant, 0.0], [0.0, constant]])


def _rectangular(first: float, second: float) -> np.ndarray:
    return np.array([[first, 0.0], [0.0, second]])


def _oblique(first: float, second: float, angle_deg: float) -> np.ndarray:
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
        _signature(
            item.twist_deg,
            item.top_atoms,
            item.bottom_atoms,
            item.strains,
            item.top_area,
        )
        for item in reference
    }


CASES = {
    "square_pair_with_unequal_constants": dict(
        top_basis=_square(3.0),
        bottom_basis=_square(3.15),
        max_length=10.0,
        top_strain=0.01,
        bottom_strain=0.01,
    ),
    "square_on_hexagonal": dict(
        top_basis=_square(3.0),
        bottom_basis=hexagonal_basis(3.2),
        max_length=9.0,
        top_strain=0.03,
        bottom_strain=0.03,
    ),
    "rectangular_pair_with_one_sided_strain": dict(
        top_basis=_rectangular(3.0, 4.0),
        bottom_basis=_rectangular(3.1, 3.9),
        max_length=10.0,
        top_strain=0.0,
        bottom_strain=0.03,
    ),
    "oblique_pair": dict(
        top_basis=_oblique(3.0, 3.4, 75.0),
        bottom_basis=_oblique(3.05, 3.3, 78.0),
        max_length=9.0,
        top_strain=0.015,
        bottom_strain=0.015,
    ),
    "rigid_hexagonal_homobilayer": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.46),
        max_length=13.0,
        top_strain=0.0,
        bottom_strain=0.0,
    ),
    "length_floor": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=12.0,
        top_strain=0.01,
        bottom_strain=0.01,
        min_length=6.0,
    ),
    "atom_ceiling_with_multi_atom_layers": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.5),
        max_length=12.0,
        top_strain=0.01,
        bottom_strain=0.01,
        top_atoms=2,
        bottom_atoms=2,
        max_atoms=60,
    ),
    "imprimitive_hexagonal": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=hexagonal_basis(2.46),
        max_length=10.0,
        top_strain=0.01,
        bottom_strain=0.01,
        primitive_only=False,
    ),
    "imprimitive_square": dict(
        top_basis=_square(3.0),
        bottom_basis=_square(3.0),
        max_length=9.0,
        top_strain=0.005,
        bottom_strain=0.005,
        primitive_only=False,
    ),
    "wide_strain_budget_across_lattice_systems": dict(
        top_basis=hexagonal_basis(2.46),
        bottom_basis=_square(3.0),
        max_length=9.0,
        top_strain=0.05,
        bottom_strain=0.05,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_gram_search_reproduces_the_brute_force_classes(name):
    config = gram.SearchConfig(**CASES[name])
    result = gram.search(config)
    reference = reference_search(
        ReferenceConfig(**{key: getattr(config, key) for key in _SHARED_INPUTS})
    )
    assert _engine_signatures(result) == _reference_signatures(reference)


def test_the_sweep_is_not_vacuous():
    """At least most cases must actually produce candidates to compare."""

    nonempty = 0
    for name in CASES:
        if len(gram.search(gram.SearchConfig(**CASES[name]))) > 0:
            nonempty += 1
    assert nonempty >= len(CASES) - 1

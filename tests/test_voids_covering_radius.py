"""The largest interstitial found must be the largest one that exists.

``tests/test_voids.py`` pins the void radii of high-symmetry crystals against
closed-form values.  This module asks the harder question -- is the *biggest*
empty sphere of a cell reported at all? -- by comparing the search with a direct
maximisation of the distance-to-nearest-atom function over the whole cell.  That
maximum is the covering radius of the periodic point set, and the largest empty
sphere of an interstitial search has to equal it whenever the cell is bulk (no
vacuum, where voids are deliberately suppressed).

The disordered cells matter: a cell with two nearly touching atoms has a very
short interatomic distance and can still open a wide hollow elsewhere, so any
search whose neighbour cutoff is measured only in units of the packing distance
misses the hollow completely.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import covering, voids

_SHIFTS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
)


def _nearest_atom_distance(
    lattice: np.ndarray, atoms: np.ndarray, probes_direct: np.ndarray
) -> np.ndarray:
    images = (atoms[:, None, :] + _SHIFTS[None, :, :]).reshape(-1, 3) @ lattice
    points = np.asarray(probes_direct, dtype=float).reshape(-1, 3) @ lattice
    offsets = points[:, None, :] - images[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", offsets, offsets)).min(axis=1)


def _covering_radius(lattice: np.ndarray, atoms: np.ndarray, divisions: int = 30) -> float:
    """Return the largest distance from a point of the cell to the nearest atom."""

    fractions = (np.arange(divisions) + 0.5) / divisions
    grid = np.stack(
        np.meshgrid(fractions, fractions, fractions, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    best = grid[int(np.argmax(_nearest_atom_distance(lattice, atoms, grid)))]
    span = 1.0 / divisions
    for _ in range(6):
        local = np.linspace(-span, span, 7)
        around = np.stack(
            np.meshgrid(local, local, local, indexing="ij"), axis=-1
        ).reshape(-1, 3) + best
        best = around[int(np.argmax(_nearest_atom_distance(lattice, atoms, around)))]
        span /= 3.0
    return float(_nearest_atom_distance(lattice, atoms, best[None, :])[0])


def _assert_largest_void_is_the_covering_radius(
    lattice: np.ndarray, atoms: np.ndarray
) -> float:
    """Check the largest reported empty sphere against the covering radius.

    The bound holds in both directions, and each direction is checked in the way
    that is exact for it.  From above: the reported centre is a real point of
    the cell, so its own distance to the nearest atom -- which must equal the
    reported radius -- cannot exceed the covering radius.  From below: no grid
    probe may sit further from every atom than the reported radius, which is
    what rules out a missed void.  The sampled maximum is only a lower bound on
    the covering radius, so it is never used as an upper bound.
    """

    result = voids.find_void_sites(lattice, atoms)
    assert result.vacuum_axes == ()
    assert result.sites
    site = max(result.sites, key=lambda item: item.radius)
    exact = float(_nearest_atom_distance(lattice, atoms, np.array([site.direct]))[0])
    assert site.radius == pytest.approx(exact, abs=1e-9)
    assert site.radius >= _covering_radius(lattice, atoms) - 5e-3
    return site.radius


def _random_bulk_cell(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    lattice = np.diag(generator.uniform(5.0, 7.0, 3)) + np.tril(
        generator.uniform(-1.0, 1.0, (3, 3)), -1
    )
    atoms = generator.random((int(generator.integers(5, 12)), 3))
    return lattice, atoms


CRYSTALS = {
    "face_centred_cubic": (
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    ),
    "body_centred_cubic": (
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ),
    "simple_cubic": (np.eye(3) * 3.0, np.array([[0.0, 0.0, 0.0]])),
    "hexagonal_close_packed": (
        np.array(
            [
                [2.51, 0.0, 0.0],
                [-1.255, 2.51 * math.sqrt(3.0) / 2.0, 0.0],
                [0.0, 0.0, 2.51 * math.sqrt(8.0 / 3.0)],
            ]
        ),
        np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]),
    ),
}


@pytest.mark.parametrize("name", sorted(CRYSTALS))
def test_largest_void_of_a_crystal_is_the_covering_radius(name):
    lattice, atoms = CRYSTALS[name]
    _assert_largest_void_is_the_covering_radius(lattice, atoms)


@pytest.mark.parametrize("seed", [1, 7, 13, 29, 101])
def test_largest_void_of_a_disordered_cell_is_the_covering_radius(seed):
    lattice, atoms = _random_bulk_cell(seed)
    _assert_largest_void_is_the_covering_radius(lattice, atoms)


@pytest.mark.parametrize("seed", [1, 7, 13, 29, 101])
def test_the_sampled_bound_really_bounds_the_covering_radius(seed):
    """The neighbour cutoff rests on this bound, so it must never fall short."""

    lattice, atoms = _random_bulk_cell(seed)
    bound = covering.bulk_covering_radius_bound(lattice, atoms)
    exact = _covering_radius(lattice, atoms)
    assert bound >= exact - 1e-9
    # A loose bound costs enumeration time, so it is worth pinning how tight the
    # adaptive sweep keeps it.
    assert bound <= 1.15 * exact


def test_a_wide_hollow_survives_a_pair_of_nearly_touching_atoms():
    """Regression: the neighbour cutoff must not be tied to the packing alone.

    Six atoms in a 6 Angstrom cube, two of them 0.74 Angstrom apart, leave a
    hollow of radius 3.29 Angstrom.  A cutoff capped at a few times the shortest
    interatomic distance is far too small to circumscribe it, and the search
    used to return nothing at all here.
    """

    lattice = np.eye(3) * 6.0
    atoms = np.random.default_rng(7).random((6, 3))
    assert voids._shortest_interatomic_distance(lattice, atoms) < 0.8
    assert _assert_largest_void_is_the_covering_radius(lattice, atoms) > 3.0

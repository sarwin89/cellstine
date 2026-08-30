"""Exact periodic geometry in the surface plane.

Two shortcuts used to sit in the surface code.  Rounding a fractional
difference componentwise was taken for the shortest image -- it is not, and
``RequestProject/PeriodicGeometry.lean`` says so outright
(``Cellstine.rounding_is_not_the_minimum_image``); on a 120 degree cell the
error is a tenth of a lattice constant and on a sheared one it is unbounded.
And the periodic images of the surface atoms were enumerated over a fixed
``-1, 0, 1`` box, which is complete only for a reduced in-plane cell.

``core.geometry`` now carries the exact two-dimensional versions:
``plane_minimum_image`` rounds in a Lagrange--Gauss reduced basis and then
searches the box the resulting bound leaves (``Cellstine.abs_shift_le_of_le_guess``),
and ``plane_shift_reach`` gives the box that provably reaches a cutoff
(``Cellstine.abs_shift_le_of_cartesian_le``).  The tests below check them
against brute force and check what they fix in the site enumeration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.geometry import (
    plane_minimum_image,
    plane_minimum_image_distances,
    plane_shift_reach,
    plane_shifts,
    shortest_plane_vector_length,
)
from cellstine.interface.surface.backend import find_adsorption_sites
from cellstine.interface.surface.surface_sites import (
    _deduplicate_uv_points,
    _nearest_neighbor_distance,
    _subsurface_depth_below,
)
from cellstine.io import native as io_mod

HEXAGONAL = np.array([[1.0, 0.0], [-0.5, 0.5 * math.sqrt(3.0)]])


def _brute_force(basis, deltas, reach):
    shifts = plane_shifts([reach, reach])
    candidates = (np.asarray(deltas, dtype=float)[:, None, :] - shifts[None, :, :]) @ np.asarray(basis, dtype=float)
    return np.linalg.norm(candidates, axis=2).min(axis=1)


def test_the_shortest_image_matches_brute_force_on_random_cells():
    generator = np.random.default_rng(4321)
    for _ in range(200):
        basis = generator.normal(size=(2, 2))
        if abs(float(np.linalg.det(basis))) < 1e-2:
            continue
        deltas = generator.uniform(-3.0, 3.0, size=(24, 2))
        got = plane_minimum_image_distances(basis, deltas)
        assert np.all(got <= _brute_force(basis, deltas, 20) + 1e-9)
        assert np.allclose(got, _brute_force(basis, deltas, 20), atol=1e-9)


def test_the_returned_vector_is_the_displacement_up_to_a_lattice_vector():
    generator = np.random.default_rng(99)
    basis = np.array([[2.5, 0.0], [-1.25, 1.25 * math.sqrt(3.0)]])
    deltas = generator.uniform(-2.0, 2.0, size=(30, 2))
    vectors = plane_minimum_image(basis, deltas)
    residual = (deltas @ basis - vectors) @ np.linalg.inv(basis)
    assert np.allclose(residual, np.rint(residual), atol=1e-9)


def test_rounding_the_components_is_not_the_shortest_image():
    delta = np.array([[0.5, 0.9]])
    rounded = float(np.linalg.norm((delta - np.round(delta)) @ HEXAGONAL))
    exact = float(plane_minimum_image_distances(HEXAGONAL, delta)[0])
    assert exact < rounded - 0.09
    assert exact == pytest.approx(float(np.linalg.norm((delta - np.array([[1.0, 1.0]])) @ HEXAGONAL)))


def test_a_sheared_cell_needs_more_than_the_nine_neighbouring_images():
    sheared = np.array([[1.0, 0.0], [9.0, 1.0]])
    assert shortest_plane_vector_length(sheared) == pytest.approx(1.0)
    delta = np.array([[0.0, 0.5]])
    nine_box = float(_brute_force(sheared, delta, 1)[0])
    exact = float(plane_minimum_image_distances(sheared, delta)[0])
    assert exact < nine_box
    assert exact == pytest.approx(float(np.linalg.norm(np.array([0.5, 0.5]))), abs=1e-9)


def test_the_shift_reach_covers_every_image_within_the_cutoff():
    generator = np.random.default_rng(17)
    for _ in range(50):
        basis = generator.normal(size=(2, 2))
        if abs(float(np.linalg.det(basis))) < 1e-2:
            continue
        cutoff = float(generator.uniform(0.5, 4.0))
        reach = plane_shift_reach(basis, cutoff)
        wide = plane_shifts([int(reach[0]) + 4, int(reach[1]) + 4])
        lengths = np.linalg.norm(wide @ basis, axis=1)
        inside = wide[lengths <= cutoff]
        assert np.all(np.abs(inside) <= reach + 1e-9)


#: Unimodular in-plane changes of basis whose cells are *not* reduced: on the
#: hexagonal lattice below, no combination with coefficients in ``-1, 0, 1`` of
#: their two vectors is as short as the 2.55 A nearest-neighbour distance.
SKEW_BASES = (
    np.array([[5, 3], [3, 2]]),
    np.array([[5, 2], [2, 1]]),
    np.array([[2, 5], [1, 3]]),
)


def _hexagonal_slab(skew: int | None = None):
    """An fcc(111) slab of three close-packed planes.

    ``skew`` picks one of ``SKEW_BASES`` and writes the *same* structure on
    that cell.  The change of basis is unimodular, so the lattice, the atoms and
    every site are untouched and only the description changes -- but the cell is
    not reduced, and every shortest translation of it sits outside the
    ``-1, 0, 1`` box of images the surface code used to search.
    """

    constant = 2.55
    spacing = constant * math.sqrt(2.0 / 3.0)
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 24.0],
        ]
    )
    offsets = [(0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0)]
    positions = np.asarray(
        [
            u * lattice[0] + v * lattice[1] + np.array([0.0, 0.0, 6.0 + level * spacing])
            for level, (u, v) in enumerate(offsets)
        ],
        dtype=float,
    )
    if skew is not None:
        change = np.eye(3)
        change[:2, :2] = SKEW_BASES[int(skew)]
        lattice = change @ lattice
    direct = io_mod.cartesian_to_direct(positions, lattice)
    return io_mod.PoscarData(
        comment="fcc(111)",
        lattice=lattice,
        species=["Cu"],
        counts=[positions.shape[0]],
        positions_direct=direct,
        positions_cartesian=positions,
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
    )


def _top_layer_uv(slab):
    direct = np.asarray(slab.positions_direct, dtype=float)
    heights = np.asarray(slab.positions_cartesian, dtype=float)[:, 2]
    top = heights >= heights.max() - 1e-6
    return np.mod(direct[top][:, :2], 1.0)


def test_the_nearest_neighbour_distance_survives_a_sheared_cell():
    plain = _hexagonal_slab()
    assert _nearest_neighbor_distance(_top_layer_uv(plain), plain.lattice) == pytest.approx(2.55)
    for skew in range(len(SKEW_BASES)):
        cell = _hexagonal_slab(skew=skew)
        assert _nearest_neighbor_distance(_top_layer_uv(cell), cell.lattice) == pytest.approx(2.55)


def test_the_sites_of_a_sheared_slab_are_the_sites_of_the_slab():
    """A unimodular shear of the cell must not change a single reported site."""

    def census(run):
        counts: dict[str, int] = {}
        for site in run.sites:
            counts[site.site_type] = counts.get(site.site_type, 0) + 1
        return counts

    plain = census(find_adsorption_sites(_hexagonal_slab()))
    for skew in range(len(SKEW_BASES)):
        assert census(find_adsorption_sites(_hexagonal_slab(skew=skew))) == plain
    assert {"top", "fcc_hollow", "hcp_hollow"} <= set(plain)


def test_the_deduplication_still_keeps_genuinely_distinct_points():
    lattice = np.vstack([HEXAGONAL * 2.5, [0.0, 0.0]])
    points = [np.array([0.0, 0.0]), np.array([1.0 - 1e-12, 1e-12]), np.array([1.0 / 3.0, 2.0 / 3.0])]
    kept = _deduplicate_uv_points(points, lattice)
    assert len(kept) == 2


def test_a_hollow_finds_the_atom_below_it_on_a_sheared_cell():
    plain = _hexagonal_slab()
    sheared = _hexagonal_slab(skew=0)
    below = _top_layer_uv(plain)[:1]
    cartesian = below @ np.asarray(plain.lattice, dtype=float)[:2]
    in_sheared = np.mod(cartesian @ np.linalg.pinv(np.asarray(sheared.lattice, dtype=float)[:2]), 1.0)
    hollow = np.mod(in_sheared[0] + np.array([1e-9, -1e-9]), 1.0)
    assert _subsurface_depth_below(hollow, sheared.lattice, [(0.0, in_sheared)], 0.4) == 1

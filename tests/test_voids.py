"""Checks on the interstitial and hollow searches.

The empty-sphere radii of the high-symmetry interstitials of the diamond and
face-centred cubic structures are known in closed form, so the search can be
checked against exact numbers rather than against its own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import planar_voids, voids


def _fcc(constant: float) -> np.ndarray:
    return constant * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _hexagonal_basis(constant: float) -> np.ndarray:
    return np.array([[constant, 0.0, 0.0], [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0]])


def test_diamond_silicon_interstitial_radii():
    """Diamond silicon has a tetrahedral void of radius ``a * sqrt(3) / 4`` and a
    hexagonal void of radius ``a * sqrt(11) / 8``."""

    constant = 5.43
    lattice = _fcc(constant)
    positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    result = voids.find_void_sites(lattice, positions)
    radii = sorted({round(site.radius, 4) for site in result.sites}, reverse=True)
    assert len(radii) == 2
    assert radii[0] == pytest.approx(constant * math.sqrt(3.0) / 4.0, abs=1e-3)
    assert radii[1] == pytest.approx(constant * math.sqrt(11.0) / 8.0, abs=1e-3)
    assert result.vacuum_axes == ()


def test_face_centred_cubic_interstitial_radii():
    """A close-packed metal has an octahedral void of radius ``a / 2`` and a
    tetrahedral void of radius ``a * sqrt(3) / 4``."""

    constant = 4.05
    lattice = _fcc(constant)
    positions = np.array([[0.0, 0.0, 0.0]])
    result = voids.find_void_sites(lattice, positions)
    radii = sorted({round(site.radius, 4) for site in result.sites}, reverse=True)
    assert len(radii) == 2
    assert radii[0] == pytest.approx(constant / 2.0, abs=1e-3)
    assert radii[1] == pytest.approx(constant * math.sqrt(3.0) / 4.0, abs=1e-3)


def test_every_void_sphere_is_really_empty():
    constant = 4.05
    lattice = _fcc(constant)
    positions = np.array([[0.0, 0.0, 0.0]])
    result = voids.find_void_sites(lattice, positions)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    for site in result.sites:
        images = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ lattice
        distance = np.linalg.norm(images - np.asarray(site.cartesian)[None, :], axis=1).min()
        assert distance >= site.radius - 1e-4


def test_a_free_standing_monolayer_has_no_interstitials():
    """Graphene in a vacuum-padded cell must not report the vacuum as a void."""

    lattice = np.zeros((3, 3))
    lattice[:2, :] = _hexagonal_basis(2.46)
    lattice[2, 2] = 20.0
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    result = voids.find_void_sites(lattice, positions)
    assert result.vacuum_axes == (2,)
    assert result.sites == []


def test_honeycomb_hollow_is_the_hexagon_centre():
    """A honeycomb has no triangle of mutual neighbours; its hollow is the centre
    of the hexagon, sixfold coordinated, of radius equal to the bond length."""

    constant = 2.46
    basis = _hexagonal_basis(constant)
    points = np.array([[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]])
    hollows = planar_voids.find_planar_voids(basis, points)
    assert len(hollows) == 1
    hollow = hollows[0]
    assert hollow.coordination == 6
    assert hollow.radius == pytest.approx(constant / math.sqrt(3.0), abs=1e-3)
    assert np.allclose(np.mod(np.asarray(hollow.uv), 1.0), [0.0, 0.0], atol=1e-3)


def test_triangular_lattice_has_two_threefold_hollows():
    constant = 2.864
    basis = _hexagonal_basis(constant)
    hollows = planar_voids.find_planar_voids(basis, np.array([[0.0, 0.0]]))
    assert len(hollows) == 2
    for hollow in hollows:
        assert hollow.coordination == 3
        assert hollow.radius == pytest.approx(constant / math.sqrt(3.0), abs=1e-3)
    found = np.array(sorted(np.mod(np.asarray(hollow.uv), 1.0).tolist() for hollow in hollows))
    expected = np.array([[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]])
    assert np.allclose(found, expected, atol=1e-3)


def test_square_lattice_has_one_fourfold_hollow():
    constant = 2.864
    basis = np.array([[constant, 0.0, 0.0], [0.0, constant, 0.0]])
    hollows = planar_voids.find_planar_voids(basis, np.array([[0.0, 0.0]]))
    assert len(hollows) == 1
    assert hollows[0].coordination == 4
    assert hollows[0].radius == pytest.approx(constant / math.sqrt(2.0), abs=1e-3)
    assert np.allclose(np.asarray(hollows[0].uv), [0.5, 0.5], atol=1e-3)


def test_rectangular_lattice_hollow_sits_at_the_cell_centre():
    basis = np.array([[2.864, 0.0, 0.0], [0.0, 4.05, 0.0]])
    hollows = planar_voids.find_planar_voids(basis, np.array([[0.0, 0.0]]))
    assert len(hollows) == 1
    assert hollows[0].coordination == 4
    assert hollows[0].radius == pytest.approx(0.5 * math.hypot(2.864, 4.05), abs=1e-3)


def test_hollows_of_a_supercell_repeat_the_primitive_ones():
    constant = 2.864
    basis = _hexagonal_basis(constant)
    primitive = planar_voids.find_planar_voids(basis, np.array([[0.0, 0.0]]))
    supercell_basis = np.array([3.0 * basis[0], 3.0 * basis[1]])
    points = np.array([[i / 3.0, j / 3.0] for i in range(3) for j in range(3)])
    supercell = planar_voids.find_planar_voids(supercell_basis, points)
    assert len(supercell) == 9 * len(primitive)
    for hollow in supercell:
        assert hollow.coordination == 3
        assert hollow.radius == pytest.approx(primitive[0].radius, abs=1e-3)


def test_bridge_midpoints_are_not_reported_as_hollows():
    """The midpoint of a bond is a saddle of the distance function, not a
    maximum, so it must not appear among the hollows."""

    constant = 2.864
    basis = _hexagonal_basis(constant)
    hollows = planar_voids.find_planar_voids(basis, np.array([[0.0, 0.0]]))
    for hollow in hollows:
        assert hollow.radius > 0.5 * constant + 1e-3

def _radius_counts(sites) -> dict[float, int]:
    counts: dict[float, int] = {}
    for site in sites:
        key = round(float(site.radius), 4)
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_interstitial_count_of_a_close_packed_metal_in_three_settings():
    """A face-centred cubic crystal has one octahedral and two tetrahedral voids
    per atom, whichever cell it is written in."""

    constant = 4.05
    octahedral = round(constant / 2.0, 4)
    tetrahedral = round(constant * math.sqrt(3.0) / 4.0, 4)

    primitive = voids.find_void_sites(_fcc(constant), np.array([[0.0, 0.0, 0.0]]))
    assert _radius_counts(primitive.sites) == {octahedral: 1, tetrahedral: 2}

    conventional = voids.find_void_sites(
        constant * np.eye(3),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    )
    assert _radius_counts(conventional.sites) == {octahedral: 4, tetrahedral: 8}

    side = constant / math.sqrt(2.0)
    height = constant * math.sqrt(3.0)
    hexagonal = np.array(
        [[side, 0.0, 0.0], [-0.5 * side, 0.5 * math.sqrt(3.0) * side, 0.0], [0.0, 0.0, height]]
    )
    stacked = voids.find_void_sites(
        hexagonal,
        np.array([[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]]),
    )
    assert _radius_counts(stacked.sites) == {octahedral: 3, tetrahedral: 6}


def test_diamond_silicon_interstitial_multiplicities():
    """Diamond silicon has two tetrahedral and four hexagonal voids per
    primitive cell."""

    constant = 5.43
    result = voids.find_void_sites(_fcc(constant), np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))
    tetrahedral = round(constant * math.sqrt(3.0) / 4.0, 4)
    hexagonal = round(constant * math.sqrt(11.0) / 8.0, 4)
    assert _radius_counts(result.sites) == {tetrahedral: 2, hexagonal: 4}


def test_interstitial_counts_scale_with_a_supercell():
    constant = 4.05
    lattice = _fcc(constant)
    positions = np.array([[0.0, 0.0, 0.0]])
    supercell_positions = np.array(
        [[(i) / 2.0, (j) / 2.0, (k) / 2.0] for i in range(2) for j in range(2) for k in range(2)]
    )
    supercell = voids.find_void_sites(2.0 * lattice, supercell_positions)
    assert _radius_counts(supercell.sites) == {round(constant / 2.0, 4): 8, round(constant * math.sqrt(3.0) / 4.0, 4): 16}
    del positions


def test_slab_reports_only_the_voids_between_its_layers():
    """A six-layer close-packed slab has five interlayer gaps, each holding one
    octahedral and two tetrahedral voids, and nothing in the vacuum."""

    constant = 4.05
    side = constant / math.sqrt(2.0)
    spacing = constant / math.sqrt(3.0)
    height = 5.0 * spacing + 15.0
    lattice = np.array(
        [[side, 0.0, 0.0], [-0.5 * side, 0.5 * math.sqrt(3.0) * side, 0.0], [0.0, 0.0, height]]
    )
    stacking = [(0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0)]
    positions = np.array(
        [
            [stacking[layer % 3][0], stacking[layer % 3][1], (7.5 + layer * spacing) / height]
            for layer in range(6)
        ]
    )
    result = voids.find_void_sites(lattice, positions)
    assert result.vacuum_axes == (2,)
    counts = _radius_counts(result.sites)
    assert counts == {round(constant / 2.0, 4): 5, round(constant * math.sqrt(3.0) / 4.0, 4): 10}



def _planar_distance_field(basis: np.ndarray, points: np.ndarray, samples: int) -> np.ndarray:
    """Return the distance to the nearest point on a regular grid of the cell."""

    shifts = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)], dtype=float)
    images = ((points[None, :, :] + shifts[:, None, :]).reshape(-1, 2)) @ basis
    axis = np.arange(samples, dtype=float) / float(samples)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2) @ basis
    return np.sqrt(((grid[:, None, :] - images[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def test_planar_hollow_positions_are_exact():
    """The hollows come out at their closed-form positions, not near them.

    A grid search can only place a site to within its resolution; enumerating
    the Voronoi vertices in closed form puts the three-fold hollow of a
    triangular lattice at exactly ``(1/3, 2/3)``.
    """

    constant = 2.864
    triangular = planar_voids.find_planar_voids(_hexagonal_basis(constant), np.array([[0.0, 0.0]]))
    found = sorted(tuple(hollow.uv) for hollow in triangular)
    assert np.allclose(found, [[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]], atol=1e-12)

    square = planar_voids.find_planar_voids(
        np.array([[constant, 0.0, 0.0], [0.0, constant, 0.0]]), np.array([[0.0, 0.0]])
    )
    assert np.allclose(np.asarray(square[0].uv), [0.5, 0.5], atol=1e-12)
    assert square[0].radius == pytest.approx(constant / math.sqrt(2.0), abs=1e-12)


def test_planar_hollows_are_maxima_and_include_the_deepest_hollow():
    """On a disordered point set every hollow is a genuine local maximum, and
    the deepest of them is the global maximum of the distance function."""

    basis = np.array([[7.0, 0.0, 0.0], [2.0, 6.0, 0.0]])
    points = np.random.default_rng(4).random((12, 2))
    hollows = planar_voids.find_planar_voids(basis, points)
    assert hollows

    shifts = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)], dtype=float)
    images = ((points[None, :, :] + shifts[:, None, :]).reshape(-1, 2)) @ basis

    angles = np.arange(72, dtype=float) * (np.pi / 36.0)
    circle = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    for hollow in hollows:
        centre = np.asarray(hollow.uv, dtype=float) @ basis
        assert np.linalg.norm(images - centre, axis=1).min() == pytest.approx(hollow.radius, abs=1e-9)
        for step in (1e-3, 1e-2, 1e-1):
            trials = centre[None, :] + step * circle
            reach = np.sqrt(((trials[:, None, :] - images[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            assert reach.max() <= hollow.radius + 1e-9

    deepest = max(hollow.radius for hollow in hollows)
    sampled = float(_planar_distance_field(basis, points, 400).max())
    assert deepest >= sampled - 1e-9
    assert deepest == pytest.approx(sampled, abs=0.05)

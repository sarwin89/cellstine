"""Checks on the saddle sites of the interstitial search.

A site an atom can be inserted at is a critical point of the distance to the
nearest atom, not necessarily a local maximum of it.  The distinction is not
academic: the interstitial site of a body-centred cubic metal that carbon takes
in ferrite is the octahedral one, and that site is a saddle -- the sphere there
touches only two atoms, and grows if it slides sideways.  A search restricted to
the vertices of the Voronoi diagram reports the tetrahedral site and nothing
else.

Every radius checked here is known in closed form, so the search is measured
against exact numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import voids


def _fcc(constant: float) -> np.ndarray:
    return constant * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _sites_of_radius(result, radius: float, kind: str | None = None) -> list:
    return [
        site
        for site in result.sites
        if abs(site.radius - radius) < 1e-3 and (kind is None or site.kind == kind)
    ]


def test_body_centred_cubic_octahedral_site_is_a_saddle():
    """The octahedral site of a bcc metal sits at ``(1/2, 0, 0)``, carries an
    empty sphere of radius ``a / 2``, touches two atoms, and is a saddle."""

    constant = 2.87
    result = voids.find_void_sites(
        constant * np.eye(3),
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        include_saddles=True,
    )
    octahedral = _sites_of_radius(result, constant / 2.0)
    assert len(octahedral) == 6
    for site in octahedral:
        assert site.kind == "saddle"
        assert site.coordination == 2
        # The six sites are the edge and face centres of the cube.
        coordinates = sorted(round(float(value) % 1.0, 4) for value in site.direct)
        assert coordinates in ([0.0, 0.0, 0.5], [0.0, 0.5, 0.5])


def test_body_centred_cubic_tetrahedral_site_is_a_maximum():
    """The wider bcc site is the tetrahedral one, radius ``a sqrt(5) / 4``, and
    it is a genuine local maximum surrounded by four atoms."""

    constant = 2.87
    result = voids.find_void_sites(
        constant * np.eye(3),
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        include_saddles=True,
    )
    tetrahedral = _sites_of_radius(result, constant * math.sqrt(5.0) / 4.0)
    assert len(tetrahedral) == 12
    for site in tetrahedral:
        assert site.kind == "maximum"
        assert site.coordination == 4
    assert max(site.radius for site in result.sites) == pytest.approx(
        constant * math.sqrt(5.0) / 4.0, abs=1e-3
    )


def test_the_default_search_reports_maxima_only():
    """Without the flag the search is unchanged: only local maxima come back."""

    constant = 2.87
    result = voids.find_void_sites(
        constant * np.eye(3), np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    )
    assert {site.kind for site in result.sites} == {"maximum"}
    assert len(result.sites) == 12


def test_diamond_bond_centre_is_a_saddle_at_half_the_bond():
    """The bond centre of diamond silicon -- where hydrogen sits -- is the
    midpoint of two neighbours, so its sphere has half the bond length."""

    constant = 5.43
    bond = constant * math.sqrt(3.0) / 4.0
    result = voids.find_void_sites(
        _fcc(constant),
        np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        include_saddles=True,
    )
    centres = _sites_of_radius(result, 0.5 * bond)
    assert len(centres) == 4
    for site in centres:
        assert site.kind == "saddle"
        assert site.coordination == 2


def test_close_packed_maxima_are_named_by_their_coordination():
    """In a close-packed metal the octahedral void touches six atoms and the
    tetrahedral one touches four, which is what the coordination reports."""

    constant = 4.05
    result = voids.find_void_sites(_fcc(constant), np.array([[0.0, 0.0, 0.0]]))
    octahedral = _sites_of_radius(result, constant / 2.0)
    tetrahedral = _sites_of_radius(result, constant * math.sqrt(3.0) / 4.0)
    assert [site.coordination for site in octahedral] == [6]
    assert [site.coordination for site in tetrahedral] == [4, 4]


def test_every_saddle_sphere_is_empty_and_touches_its_contacts():
    """Each reported site must carry a sphere that no atom enters, and exactly
    as many atoms on it as the coordination claims."""

    constant = 2.87
    lattice = constant * np.eye(3)
    atoms = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    result = voids.find_void_sites(lattice, atoms, include_saddles=True)
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
    )
    images = (atoms[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ lattice
    assert result.sites
    for site in result.sites:
        distance = np.linalg.norm(images - np.asarray(site.cartesian)[None, :], axis=1)
        assert distance.min() >= site.radius - 1e-4
        assert int(np.count_nonzero(distance <= site.radius * (1.0 + 1e-3))) == site.coordination


def test_a_slab_reports_no_site_in_its_vacuum():
    """A five-layer close-packed slab has interstitials between its planes and
    none above them, saddles included."""

    constant = 3.61 / math.sqrt(2.0)
    height = 3.61 / math.sqrt(3.0)
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 4.0 * height + 18.0],
        ]
    )
    stack = [(0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0)]
    positions = np.array(
        [
            [stack[layer % 3][0], stack[layer % 3][1], (layer * height + 2.0) / lattice[2, 2]]
            for layer in range(5)
        ]
    )
    result = voids.find_void_sites(lattice, positions, include_saddles=True)
    assert result.vacuum_axes == (2,)
    heights = [float(site.cartesian[2]) for site in result.sites]
    assert heights
    assert max(heights) <= 2.0 + 4.0 * height + 1e-6
    assert min(heights) >= 2.0 - 1e-6


def test_a_free_standing_monolayer_keeps_its_sites_in_the_sheet():
    """Graphene has no local maximum at all -- the sphere always grows into the
    vacuum -- but the centre of its hexagon is a saddle, sixfold and as wide as
    the bond.  Nothing above or below the sheet may be reported."""

    bond = 2.46 / math.sqrt(3.0)
    lattice = np.array([[2.46, 0.0, 0.0], [-1.23, 2.46 * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 20.0]])
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    assert voids.find_void_sites(lattice, positions).sites == []
    result = voids.find_void_sites(lattice, positions, include_saddles=True)
    assert {site.kind for site in result.sites} == {"saddle"}
    for site in result.sites:
        assert site.direct[2] == pytest.approx(0.5, abs=1e-9)
    hexagon = _sites_of_radius(result, bond)
    assert len(hexagon) == 1
    assert hexagon[0].coordination == 6
    assert np.allclose(np.mod(hexagon[0].direct[:2], 1.0), [0.0, 0.0], atol=1e-6)


def test_contact_classification_of_the_standard_arrangements():
    """The classifier itself, on contact sets whose answer is obvious."""

    axis = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert voids._classify_contact_directions(axis) == "saddle"
    triangle = np.array(
        [[1.0, 0.0, 0.0], [-0.5, 0.5 * math.sqrt(3.0), 0.0], [-0.5, -0.5 * math.sqrt(3.0), 0.0]]
    )
    assert voids._classify_contact_directions(triangle) == "saddle"
    tetrahedron = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    ) / math.sqrt(3.0)
    assert voids._classify_contact_directions(tetrahedron) == "maximum"
    # Three atoms on one side of the centre hold nothing: the sphere grows away
    # from them, which is the situation just under the surface of a slab.
    below = np.array([[1.0, 0.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]) / math.sqrt(3.0)
    assert voids._classify_contact_directions(below) is None
    # A pair that is not diametrically opposite cannot hold a centre either.
    assert voids._classify_contact_directions(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])) is None

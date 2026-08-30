"""Mathematical checks on Wigner-Seitz cells and first Brillouin zones.

The cell is checked against what a convex polytope has to satisfy -- Euler's
formula, the volume it must tile with, the half-space description of its
interior -- and against the shapes the textbooks name: the simple-cubic zone is
a cube, the fcc zone a truncated octahedron and the bcc zone a rhombic
dodecahedron.  The formal statements are in ``RequestProject/BrillouinZone.lean``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.brillouin import brillouin_zone, wigner_seitz_cell, zone_boundary_distance
from cellstine.core.reciprocal import reciprocal_lattice

CONSTANT = 4.0


def cubic() -> np.ndarray:
    return np.eye(3) * CONSTANT


def face_centred() -> np.ndarray:
    return 0.5 * CONSTANT * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def body_centred() -> np.ndarray:
    return 0.5 * CONSTANT * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])


def hexagonal() -> np.ndarray:
    return np.array(
        [
            [CONSTANT, 0.0, 0.0],
            [-0.5 * CONSTANT, 0.5 * math.sqrt(3.0) * CONSTANT, 0.0],
            [0.0, 0.0, 1.6 * CONSTANT],
        ]
    )


def triclinic() -> np.ndarray:
    return np.array(
        [
            [CONSTANT, 0.0, 0.0],
            [0.3 * CONSTANT, 0.9 * CONSTANT, 0.0],
            [0.1 * CONSTANT, 0.2 * CONSTANT, 1.1 * CONSTANT],
        ]
    )


LATTICES = {
    "cubic": cubic(),
    "fcc": face_centred(),
    "bcc": body_centred(),
    "hexagonal": hexagonal(),
    "tetragonal": np.diag([CONSTANT, CONSTANT, 1.3 * CONSTANT]),
    "triclinic": triclinic(),
}


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_the_cell_is_a_closed_polytope(name):
    cell = wigner_seitz_cell(LATTICES[name])
    faces = cell.face_count
    vertices = cell.vertex_count
    edges = len(cell.edges())
    assert vertices - edges + faces == 2, "a convex polytope satisfies Euler's formula"
    assert cell.volume_error < 1e-9, "the faces must enclose the volume of the cell"
    assert all(len(loop) >= 3 for loop in cell.face_vertices)


@pytest.mark.parametrize(
    ("name", "faces", "vertices"),
    [("cubic", 6, 8), ("fcc", 12, 14), ("bcc", 14, 24)],
)
def test_the_named_shapes_come_out(name, faces, vertices):
    cell = wigner_seitz_cell(LATTICES[name])
    assert cell.face_count == faces
    assert cell.vertex_count == vertices


def test_the_fcc_zone_is_a_truncated_octahedron():
    zone = brillouin_zone(face_centred())
    assert zone.face_count == 14, "eight hexagons and six squares"
    assert zone.vertex_count == 24
    assert len(zone.edges()) == 36
    # The hexagons are the {111} planes and the squares the {200} planes.
    lengths = np.round(np.linalg.norm(zone.face_vectors, axis=1) * CONSTANT / (2.0 * math.pi), 9)
    assert sorted(np.unique(lengths).tolist()) == pytest.approx([math.sqrt(3.0), 2.0])
    assert int(np.count_nonzero(np.isclose(lengths, math.sqrt(3.0)))) == 8
    hexagons = [len(loop) for loop, length in zip(zone.face_vertices, lengths) if length < 1.9]
    squares = [len(loop) for loop, length in zip(zone.face_vertices, lengths) if length > 1.9]
    assert hexagons == [6] * 8 and squares == [4] * 6
    assert zone.inradius == pytest.approx(math.sqrt(3.0) * math.pi / CONSTANT, rel=1e-12)


def test_the_bcc_zone_is_a_rhombic_dodecahedron():
    zone = brillouin_zone(body_centred())
    assert zone.face_count == 12
    assert zone.vertex_count == 14
    assert all(len(loop) == 4 for loop in zone.face_vertices), "every face is a rhombus"


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_no_vertex_is_reported_twice(name):
    """A vertex on more than three planes is found once per triple, and merged.

    The copies differ only in the last bits of every coordinate, so comparing
    each point with its neighbour in a lexicographic sort is not enough: the
    sort interleaves them with the other points that share a coordinate.
    """

    cell = wigner_seitz_cell(LATTICES[name])
    vertices = cell.vertices
    scale = abs(float(np.linalg.det(cell.basis))) ** (1.0 / 3.0)
    gaps = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    assert float(np.min(gaps)) > 1e-3 * scale, "two vertices of the polytope coincide"


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_every_vertex_carries_at_least_three_faces(name):
    cell = wigner_seitz_cell(LATTICES[name])
    residual = cell.vertices @ cell.face_vectors.T - cell.face_offsets
    scale = abs(float(np.linalg.det(cell.basis))) ** (1.0 / 3.0)
    incidences = np.count_nonzero(np.abs(residual) <= 1e-6 * scale, axis=1)
    assert np.all(incidences >= 3)
    listed = np.zeros(cell.vertex_count, dtype=np.int64)
    for loop in cell.face_vertices:
        for index in loop:
            listed[index] += 1
    assert np.array_equal(listed, incidences), "each face lists every vertex on it"


def test_the_rhombic_dodecahedron_has_six_four_fold_vertices():
    """Eight corners of a cube and six of an octahedron, not twenty of nothing."""

    zone = brillouin_zone(body_centred())
    residual = zone.vertices @ zone.face_vectors.T - zone.face_offsets
    incidences = np.count_nonzero(np.abs(residual) <= 1e-9, axis=1)
    assert sorted(incidences.tolist()) == [3] * 8 + [4] * 6


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_the_zone_has_the_reciprocal_volume(name):
    lattice = LATTICES[name]
    zone = brillouin_zone(lattice)
    expected = (2.0 * math.pi) ** 3 / abs(float(np.linalg.det(lattice)))
    assert zone.volume == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_every_point_has_exactly_one_translate_in_the_cell(name):
    """The cell tiles space, so a generic point has one copy inside it."""

    lattice = LATTICES[name]
    cell = wigner_seitz_cell(lattice)
    generator = np.random.default_rng(20240517)
    offsets = np.stack(
        np.meshgrid(*(np.arange(-2, 3),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3).astype(float)
    for fractional in generator.random((40, 3)) - 0.5:
        point = fractional @ lattice
        inside = cell.contains(point[None, :] - offsets @ lattice)
        assert int(np.count_nonzero(inside)) == 1


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_the_cell_holds_the_points_closer_to_the_origin(name):
    """The half spaces say exactly that no lattice point is nearer."""

    lattice = LATTICES[name]
    cell = wigner_seitz_cell(lattice)
    offsets = np.stack(
        np.meshgrid(*(np.arange(-3, 4),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3).astype(float)
    vectors = offsets @ lattice
    generator = np.random.default_rng(7)
    points = (generator.random((60, 3)) - 0.5) @ lattice
    for point in points:
        distances = np.linalg.norm(vectors - point, axis=1)
        nearest = float(np.min(distances))
        expected = float(np.linalg.norm(point)) <= nearest + 1e-9
        assert bool(cell.contains(point[None, :])[0]) == expected


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_the_inscribed_ball_is_inside_and_the_circumscribed_one_is_not(name):
    cell = wigner_seitz_cell(LATTICES[name])
    generator = np.random.default_rng(11)
    directions = generator.normal(size=(50, 3))
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    assert np.all(cell.contains(directions * (cell.inradius - 1e-9)))
    assert cell.circumradius >= cell.inradius
    assert not np.any(
        cell.contains(directions * (cell.circumradius + 1e-6), tolerance=0.0)
    )


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_a_ray_leaves_the_cell_exactly_once(name):
    cell = wigner_seitz_cell(LATTICES[name])
    generator = np.random.default_rng(3)
    for direction in generator.normal(size=(60, 3)):
        scale = cell.boundary_scale(direction)
        assert scale > 0.0
        assert cell.contains((scale * direction)[None, :])[0]
        assert not cell.contains((1.000001 * scale * direction)[None, :], tolerance=0.0)[0]
        assert np.allclose(cell.boundary_point(direction), scale * direction)


def test_the_boundary_along_a_face_vector_is_its_midpoint():
    cell = wigner_seitz_cell(face_centred())
    for vector in cell.face_vectors:
        assert np.allclose(cell.boundary_point(vector), vector / 2.0)


def test_the_zone_boundary_distance_is_measured_in_wavevector():
    lattice = cubic()
    along_x = zone_boundary_distance(lattice, [1.0, 0.0, 0.0])
    assert along_x == pytest.approx(math.pi / CONSTANT, rel=1e-12)
    corner = zone_boundary_distance(lattice, [1.0, 1.0, 1.0])
    assert corner == pytest.approx(math.sqrt(3.0) * math.pi / CONSTANT, rel=1e-12)


def test_the_cell_is_symmetric_under_inversion():
    for lattice in LATTICES.values():
        cell = wigner_seitz_cell(lattice)
        generator = np.random.default_rng(5)
        points = (generator.random((40, 3)) - 0.5) @ lattice
        assert np.array_equal(cell.contains(points), cell.contains(-points))


def test_the_cell_does_not_depend_on_the_basis_chosen():
    """A unimodular change of basis is the same lattice, so the same cell."""

    lattice = triclinic()
    change = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    first = wigner_seitz_cell(lattice)
    second = wigner_seitz_cell(change @ lattice)
    assert first.face_count == second.face_count
    assert first.vertex_count == second.vertex_count
    assert first.inradius == pytest.approx(second.inradius, rel=1e-12)
    assert first.circumradius == pytest.approx(second.circumradius, rel=1e-12)
    generator = np.random.default_rng(13)
    points = (generator.random((50, 3)) - 0.5) @ lattice
    assert np.array_equal(first.contains(points), second.contains(points))


def test_a_face_vector_is_shorter_than_twice_the_circumradius():
    """A bisector too far out cannot cut the cell, which is why the shell suffices."""

    for lattice in LATTICES.values():
        cell = wigner_seitz_cell(lattice)
        assert np.all(np.linalg.norm(cell.face_vectors, axis=1) <= 2.0 * cell.circumradius + 1e-9)


def test_the_summary_is_json_ready():
    zone = brillouin_zone(hexagonal())
    summary = zone.summary()
    assert set(summary) == {
        "face_count",
        "vertex_count",
        "edge_count",
        "volume",
        "inradius",
        "circumradius",
        "volume_error",
    }
    assert summary["edge_count"] == len(zone.edges())
    assert summary["volume"] == pytest.approx(
        abs(float(np.linalg.det(reciprocal_lattice(hexagonal())))), rel=1e-12
    )


def test_a_degenerate_direction_is_refused():
    cell = wigner_seitz_cell(cubic())
    with pytest.raises(ValueError):
        cell.boundary_scale([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        cell.boundary_scale([float("nan"), 0.0, 0.0])


def test_a_face_centre_lies_on_its_own_face_and_inside_every_other():
    """``g / 2`` is where the bisector of ``g`` touches the cell, and it is on the boundary.

    It saturates its own inequality exactly, and satisfies all the others
    strictly -- otherwise ``g`` would not have contributed a face at all.
    """

    for lattice in LATTICES.values():
        cell = wigner_seitz_cell(lattice)
        centres = cell.face_centres()
        assert centres.shape == cell.face_vectors.shape
        products = centres @ cell.face_vectors.T
        for index in range(cell.face_count):
            assert products[index, index] == pytest.approx(cell.face_offsets[index], rel=1e-12)
            others = np.delete(products[index], index)
            assert np.all(others < np.delete(cell.face_offsets, index) + 1e-9)
        assert np.all(cell.contains(centres))


def test_every_edge_midpoint_sits_on_the_boundary():
    """A midpoint of two boundary points of a convex body is inside it, and on
    an edge it lies on the two faces the edge is shared by, so it touches the
    boundary rather than the interior."""

    for lattice in LATTICES.values():
        cell = wigner_seitz_cell(lattice)
        midpoints = cell.edge_midpoints()
        assert midpoints.shape == (len(cell.edges()), 3)
        assert np.all(cell.contains(midpoints))
        slack = cell.face_offsets[None, :] - midpoints @ cell.face_vectors.T
        touching = np.sum(np.abs(slack) <= 1e-7, axis=1)
        assert np.all(touching >= 2)
        for (first, second), midpoint in zip(cell.edges(), midpoints):
            assert midpoint == pytest.approx(
                (cell.vertices[first] + cell.vertices[second]) / 2.0, abs=1e-12
            )


def test_the_cell_is_symmetric_about_the_origin():
    """Every lattice has ``-g`` whenever it has ``g``, so the cell is centrosymmetric."""

    for lattice in LATTICES.values():
        cell = wigner_seitz_cell(lattice)
        for family in (cell.face_centres(), cell.edge_midpoints(), cell.vertices):
            assert np.all(cell.contains(-family))
            distances = np.linalg.norm(family[:, None, :] + family[None, :, :], axis=2)
            assert np.all(distances.min(axis=1) < 1e-7)

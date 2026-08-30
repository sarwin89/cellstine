"""Mathematical checks on the high-symmetry points and band paths.

The points are derived from the symmetry of the lattice rather than looked up,
so the checks here are of two kinds: that the derivation satisfies what it
claims -- the little co-group fixes the point, the stratum dimension is the
dimension of the fixed space, the search grid is fine enough -- and that the
answers agree, coordinate by coordinate and length by length, with the standard
tables for the lattices whose zone has no free parameter.  The formal statements
are in ``RequestProject/KPath.lean``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.kpath import (
    GRID_DENOMINATOR,
    band_path,
    kspace_operations,
    parse_path,
    special_points,
    stratum_dimension,
)
from cellstine.core.reciprocal import reciprocal_lattice
from cellstine.core.symmetry3d import lattice_point_group

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


def tetragonal() -> np.ndarray:
    return np.diag([CONSTANT, CONSTANT, 1.3 * CONSTANT])


LATTICES = {
    "cP": cubic(),
    "cF": face_centred(),
    "cI": body_centred(),
    "hP": hexagonal(),
    "tP": tetragonal(),
}

# The conventional walks, and the length of every segment of them in units of
# 2 pi / a, as the standard tables give them.
STANDARD = {
    "cP": (
        "GAMMA-X-M-GAMMA-R-X|M-R",
        (0.5, 0.5, math.sqrt(0.5), math.sqrt(0.75), math.sqrt(0.5), 0.5),
    ),
    "cF": (
        "GAMMA-X-W-K-GAMMA-L-U-W-L-K|U-X",
        (
            1.0,
            0.5,
            math.sqrt(0.125),
            0.75 * math.sqrt(2.0),
            math.sqrt(0.75),
            math.sqrt(0.375),
            math.sqrt(0.125),
            math.sqrt(0.5),
            math.sqrt(0.375),
            math.sqrt(0.125),
        ),
    ),
    "cI": (
        "GAMMA-H-N-GAMMA-P-H|P-N",
        (
            1.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
            math.sqrt(0.75),
            math.sqrt(0.75),
            0.5,
        ),
    ),
}


def point_of(path, label):
    return next(item for item in path.points if item.label == label)


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_bravais_type_is_recognised(symbol):
    assert band_path(LATTICES[symbol]).bravais == symbol


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_every_special_point_is_fixed_by_its_little_group(symbol):
    """A little co-group element must return the point, modulo a lattice vector."""

    lattice = LATTICES[symbol]
    operations = kspace_operations(lattice_point_group(lattice))
    points, _ = special_points(lattice)
    for point in points:
        vector = np.asarray(point.fractional, dtype=float)
        images = np.einsum("j,kjl->kl", vector, operations.astype(float))
        residual = images - vector[None, :]
        fixing = np.all(np.abs(residual - np.rint(residual)) <= 1e-9, axis=1)
        assert int(np.count_nonzero(fixing)) == point.little_group_order
        assert stratum_dimension(vector, operations[fixing]) == point.stratum_dimension


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_an_isolated_point_has_denominator_dividing_its_little_group(symbol):
    """The theorem behind the search grid: a zero-dimensional stratum is rational."""

    lattice = LATTICES[symbol]
    points, _ = special_points(lattice)
    for point in points:
        if point.stratum_dimension != 0:
            continue
        scaled = point.little_group_order * np.asarray(point.fractional, dtype=float)
        assert np.allclose(scaled, np.rint(scaled), atol=1e-9)
        assert GRID_DENOMINATOR % point.little_group_order == 0


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_special_points_lie_in_the_zone(symbol):
    lattice = LATTICES[symbol]
    path = band_path(lattice)
    for point in path.points:
        cartesian = np.asarray(point.cartesian, dtype=float)
        assert path.zone.contains(cartesian[None, :], tolerance=1e-7)[0]
        assert np.allclose(
            cartesian, np.asarray(point.fractional, dtype=float) @ path.reciprocal
        )


@pytest.mark.parametrize("symbol", sorted(STANDARD))
def test_the_standard_walk_and_its_segment_lengths(symbol):
    lattice = LATTICES[symbol]
    path = band_path(lattice)
    walk, lengths = STANDARD[symbol]
    assert path.path_source == "standard"
    assert path.path_string() == walk
    unit = 2.0 * math.pi / CONSTANT
    assert path.segment_lengths() == pytest.approx(tuple(unit * value for value in lengths), rel=1e-9)


def test_the_hexagonal_walk_matches_the_table():
    path = band_path(hexagonal())
    assert path.path_string() == "GAMMA-M-K-GAMMA-A-L-H-A|L-M|K-H"
    named = {point.label: np.asarray(point.fractional) for point in path.points}
    assert named["M"] == pytest.approx([0.5, 0.0, 0.0])
    assert named["K"] == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 0.0])
    assert named["A"] == pytest.approx([0.0, 0.0, 0.5])
    assert named["H"] == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 0.5])
    unit = 2.0 * math.pi / CONSTANT
    # Gamma-M is the short diameter of the hexagon, Gamma-K the long one.
    assert point_of(path, "M").length == pytest.approx(unit / math.sqrt(3.0), rel=1e-9)
    assert point_of(path, "K").length == pytest.approx(2.0 * unit / 3.0, rel=1e-9)


def test_the_cubic_points_are_where_they_should_be():
    path = band_path(cubic())
    unit = 2.0 * math.pi / CONSTANT
    assert point_of(path, "X").length == pytest.approx(0.5 * unit, rel=1e-12)
    assert point_of(path, "M").length == pytest.approx(math.sqrt(0.5) * unit, rel=1e-12)
    assert point_of(path, "R").length == pytest.approx(math.sqrt(0.75) * unit, rel=1e-12)
    assert point_of(path, "GAMMA").little_group_order == 48


def test_fcc_u_and_k_are_one_orbit_but_two_places():
    """Time reversal merges them; the zone does not, and the path visits both."""

    path = band_path(face_centred())
    assert "U" in point_of(path, "K").aliases
    k_point = np.asarray(point_of(path, "K").fractional)
    u_point = np.asarray(point_of(path, "U").fractional)
    assert not np.allclose(k_point, u_point)
    assert point_of(path, "U").length == pytest.approx(point_of(path, "K").length, rel=1e-12)
    reciprocal = path.reciprocal
    # U is -K plus a reciprocal lattice vector: the same bands, elsewhere.
    operations = kspace_operations(lattice_point_group(face_centred()))
    images = np.einsum("j,kjl->kl", k_point, operations.astype(float))
    residual = images - u_point[None, :]
    assert np.any(np.all(np.abs(residual - np.rint(residual)) <= 1e-9, axis=1))
    # but no point-group operation alone takes one to the other.
    assert float(np.min(np.linalg.norm((images - u_point[None, :]) @ reciprocal, axis=1))) > 1e-3
    # K sits on the edge of two hexagons, U on a square face.
    unit = 2.0 * math.pi / CONSTANT
    k_cartesian = np.sort(np.abs(k_point @ reciprocal)) / unit
    u_cartesian = np.sort(np.abs(u_point @ reciprocal)) / unit
    assert k_cartesian == pytest.approx([0.0, 0.75, 0.75], abs=1e-9)
    assert u_cartesian == pytest.approx([0.25, 0.25, 1.0], abs=1e-9)


def test_the_walk_never_repeats_a_segment():
    """A repeated line is wasted sampling in a band-structure run."""

    for symbol, lattice in LATTICES.items():
        path = band_path(lattice)
        seen = set()
        for run in path.walk_points:
            for position in range(len(run) - 1):
                first = tuple(np.round(run[position], 6))
                second = tuple(np.round(run[position + 1], 6))
                key = tuple(sorted((first, second)))
                assert key not in seen, f"{symbol} walks one line twice"
                seen.add(key)


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_walk_visits_every_named_point(symbol):
    path = band_path(LATTICES[symbol])
    visited = {label for run in path.walk for label in run}
    assert {point.label for point in path.points} <= visited


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_sampled_path_is_the_walk(symbol):
    path = band_path(LATTICES[symbol])
    divisions = 9
    points, distances, labels = path.sample(divisions)
    assert len(points) == len(distances) == len(labels)
    expected = sum(
        divisions + (divisions - 1) * (len(run) - 2) for run in path.walk
    )
    assert len(points) == expected
    assert distances[0] == pytest.approx(0.0)
    assert distances[-1] == pytest.approx(path.length, rel=1e-9)
    steps = np.linalg.norm(np.diff(points @ path.reciprocal, axis=0), axis=1)
    assert float(np.sum(steps)) >= path.length - 1e-9
    # The named points of the sample are the nodes of the walk, in order.
    named = [label for label in labels if label]
    flattened = [label for run in path.walk for label in run]
    assert named == flattened


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_division_count_meets_the_spacing(symbol):
    path = band_path(LATTICES[symbol])
    spacing = 0.2
    divisions = path.divisions_for_spacing(spacing)
    assert divisions >= 2
    lengths = np.asarray(path.segment_lengths())
    assert np.all(lengths / (divisions - 1) <= spacing + 1e-12)
    if divisions > 2:
        assert np.any(lengths / (divisions - 2) > spacing - 1e-12), "no coarser count would do"


def test_an_explicit_path_is_followed():
    path = band_path(cubic(), path="GAMMA-X|M-R")
    assert path.path_source == "explicit"
    assert path.path_string() == "GAMMA-X|M-R"
    assert len(path.segments) == 2
    with pytest.raises(ValueError):
        band_path(cubic(), path="GAMMA-QQ")


def test_a_path_may_be_named_by_an_alias():
    path = band_path(face_centred(), path="GAMMA-U")
    assert path.path_string() == "GAMMA-U"
    assert len(path.segment_lengths()) == 1


def test_parse_path_reads_the_runs():
    assert parse_path("GAMMA-X-W|K-L") == (("GAMMA", "X", "W"), ("K", "L"))
    assert parse_path(" GAMMA - X ") == (("GAMMA", "X"),)
    with pytest.raises(ValueError):
        parse_path("GAMMA")


def test_the_derived_walk_covers_a_triclinic_zone():
    lattice = np.array([[4.0, 0.0, 0.0], [1.2, 3.6, 0.0], [0.4, 0.8, 4.4]])
    path = band_path(lattice, use_standard=False)
    assert path.path_source == "derived"
    assert path.points[0].label == "GAMMA"
    visited = {label for run in path.walk for label in run}
    assert {point.label for point in path.points} <= visited
    assert path.length > 0.0


def test_turning_off_time_reversal_can_only_split_points():
    lattice = face_centred()
    with_reversal, _ = special_points(lattice, time_reversal=True)
    without, _ = special_points(lattice, time_reversal=False)
    assert len(without) >= len(with_reversal)


def test_the_summary_is_json_ready():
    path = band_path(cubic())
    summary = path.summary()
    assert summary["bravais_symbol"] == "cP"
    assert summary["path"] == path.path_string()
    assert summary["segment_count"] == len(path.segments)
    assert summary["total_length"] == pytest.approx(path.length)
    assert len(summary["segments"]) == len(path.segments)
    assert all(item["length"] > 0.0 for item in summary["segments"])
    assert summary["zone"]["face_count"] == path.zone.face_count


def test_a_rotated_lattice_gives_the_same_path():
    """Nothing here may depend on how the cell is written down."""

    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    first = band_path(face_centred())
    second = band_path(face_centred() @ rotation.T)
    assert first.path_string() == second.path_string()
    assert first.segment_lengths() == pytest.approx(second.segment_lengths(), rel=1e-9)


def test_reciprocal_coordinates_are_the_ones_reported():
    lattice = hexagonal()
    path = band_path(lattice)
    assert np.allclose(path.reciprocal, reciprocal_lattice(lattice))


def interior_stratum(point, operations):
    """Return the stratum dimension of ``point``, computed from scratch."""

    vector = np.asarray(point, dtype=float)
    images = np.einsum("j,kjl->kl", vector, operations.astype(float))
    residual = images - vector[None, :]
    fixing = np.all(np.abs(residual - np.rint(residual)) <= 1e-9, axis=1)
    return stratum_dimension(vector, operations[fixing])


def walked_segments(path):
    """Yield ``(start, end)`` fractional endpoints in :attr:`segments` order."""

    for run in path.walk_points:
        coordinates = np.asarray(run, dtype=float)
        for position in range(len(coordinates) - 1):
            yield coordinates[position], coordinates[position + 1]


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_a_segment_is_classified_by_its_interior(symbol):
    """The reported dimension is the one seen at generic interior points."""

    lattice = LATTICES[symbol]
    path = band_path(lattice)
    operations = kspace_operations(lattice_point_group(lattice))
    assert len(path.segment_strata) == len(path.segments)
    for (start, end), dimension in zip(walked_segments(path), path.segment_strata):
        seen = [
            interior_stratum(start + fraction * (end - start), operations)
            for fraction in (0.211, 0.42, 0.777)
        ]
        assert max(max(seen), 1) == dimension


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_a_segment_lies_inside_its_own_stratum(symbol):
    """Every operation fixing the interior fixes the direction walked.

    A segment classified as a line or a plane must really lie in that stratum,
    and then each element of the little co-group of an interior point maps the
    segment to itself -- so it fixes the difference of the ends exactly, with no
    lattice vector left over.
    """

    lattice = LATTICES[symbol]
    path = band_path(lattice)
    operations = kspace_operations(lattice_point_group(lattice))
    for start, end in walked_segments(path):
        middle = (start + end) / 2.0
        images = np.einsum("j,kjl->kl", middle, operations.astype(float))
        residual = images - middle[None, :]
        fixing = np.all(np.abs(residual - np.rint(residual)) <= 1e-9, axis=1)
        direction = end - start
        moved = np.einsum("j,kjl->kl", direction, operations[fixing].astype(float))
        assert np.allclose(moved, direction[None, :], atol=1e-9)


def test_the_fcc_segments_are_lines_and_mirror_planes():
    """The standard fcc walk alternates symmetry lines and mirror planes."""

    path = band_path(face_centred())
    assert path.path_string() == "GAMMA-X-W-K-GAMMA-L-U-W-L-K|U-X"
    assert path.segment_strata == (1, 1, 2, 1, 1, 2, 2, 1, 2, 1)
    assert path.segment_symmetry == tuple(
        dimension == 1 for dimension in path.segment_strata
    )


def test_the_segment_summary_reports_the_stratum():
    path = band_path(face_centred())
    for item, dimension in zip(path.summary()["segments"], path.segment_strata):
        assert item["stratum_dimension"] == dimension
        assert item["symmetry_line"] is (dimension == 1)

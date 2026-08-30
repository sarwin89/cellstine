"""Mathematical checks on the Bravais classification and the conventional cell.

The classification is read off the symmetry of the lattice, not off a table of
cell parameters, so the checks here are of three kinds: that all fourteen
Bravais types are recognised, and recognised *whatever basis* the lattice is
handed in; that the conventional cell really is a superlattice of the given one,
with the index and the coset representatives the module reports; and that its
metric obeys the constraints of its crystal family, in the standard setting.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from cellstine.core.bravais import bravais_symbol, conventional_cell
from cellstine.core.symmetry3d import lattice_point_group

A, B, C = 4.0, 5.3, 6.7


def rhombohedral(side: float = A, height: float = 12.4) -> np.ndarray:
    """Return the primitive cell of an R-centred lattice on hexagonal axes."""

    hexagonal_axes = np.array(
        [[side, 0.0, 0.0], [-side / 2.0, side * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, height]]
    )
    weights = np.array([[2.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -2.0, 1.0]]) / 3.0
    return weights @ hexagonal_axes


LATTICES = {
    "aP": np.array([[A, 0.0, 0.0], [1.1, B, 0.0], [0.7, 1.3, C]]),
    "mP": np.array([[A, 0.0, 0.0], [0.0, B, 0.0], [1.9, 0.0, C]]),
    "mC": np.array([[A / 2, B / 2, 0.0], [-A / 2, B / 2, 0.0], [1.9, 0.0, C]]),
    "oP": np.diag([A, B, C]),
    "oC": np.array([[A / 2, B / 2, 0.0], [-A / 2, B / 2, 0.0], [0.0, 0.0, C]]),
    "oI": np.array([[-A / 2, B / 2, C / 2], [A / 2, -B / 2, C / 2], [A / 2, B / 2, -C / 2]]),
    "oF": np.array([[0.0, B / 2, C / 2], [A / 2, 0.0, C / 2], [A / 2, B / 2, 0.0]]),
    "tP": np.diag([A, A, C]),
    "tI": np.array([[-A / 2, A / 2, C / 2], [A / 2, -A / 2, C / 2], [A / 2, A / 2, -C / 2]]),
    "hP": np.array([[A, 0.0, 0.0], [-A / 2, A * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, C]]),
    "hR": rhombohedral(),
    "cP": np.diag([A, A, A]),
    "cI": (A / 2) * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
    "cF": (A / 2) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
}

SYSTEM = {
    "a": "triclinic",
    "m": "monoclinic",
    "o": "orthorhombic",
    "t": "tetragonal",
    "c": "cubic",
}

MULTIPLICITY = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "R": 3, "F": 4}

#: The order of the holohedry of each crystal family.
HOLOHEDRY_ORDER = {
    "triclinic": 2,
    "monoclinic": 4,
    "orthorhombic": 8,
    "tetragonal": 16,
    "trigonal": 12,
    "hexagonal": 24,
    "cubic": 48,
}


def rotation(seed: int) -> np.ndarray:
    """Return a pseudo-random proper rotation."""

    quaternion = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = quaternion / float(np.linalg.norm(quaternion))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def unimodular(seed: int) -> np.ndarray:
    """Return a pseudo-random basis change of determinant one."""

    generator = np.random.default_rng(seed + 1000)
    matrix = np.eye(3, dtype=np.int64)
    for _ in range(8):
        first, second = generator.choice(3, 2, replace=False)
        matrix[first] += generator.integers(-2, 3) * matrix[second]
    return matrix


def rebased(lattice: np.ndarray, seed: int) -> np.ndarray:
    """Return the same lattice written in another basis and another frame."""

    return (unimodular(seed).astype(float) @ lattice) @ rotation(seed).T


SEEDS = (0, 1, 2)


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_every_bravais_type_is_recognised_in_any_basis(symbol, seed):
    """All fourteen types, in a skewed basis and a rotated frame."""

    assert bravais_symbol(rebased(LATTICES[symbol], seed)) == symbol


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_system_and_centring_agree_with_the_symbol(symbol):
    cell = conventional_cell(LATTICES[symbol])
    assert cell.symbol == symbol
    assert cell.centring == symbol[1]
    if symbol[0] == "h":
        assert cell.system in ("hexagonal", "trigonal")
    else:
        assert cell.system == SYSTEM[symbol[0]]
    assert cell.multiplicity == MULTIPLICITY[cell.centring]


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_holohedry_has_the_order_of_its_family(symbol):
    """The seven families are told apart by their point group, so check it."""

    cell = conventional_cell(LATTICES[symbol])
    assert len(lattice_point_group(LATTICES[symbol])) == HOLOHEDRY_ORDER[cell.system]


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_conventional_basis_is_made_of_lattice_vectors(symbol, seed):
    """``to_primitive`` must be an integer matrix carrying the cell to the lattice."""

    lattice = rebased(LATTICES[symbol], seed)
    cell = conventional_cell(lattice)
    assert np.allclose(cell.to_primitive, np.rint(cell.to_primitive), atol=1e-8)
    assert np.allclose(cell.cell, cell.to_primitive @ lattice, atol=1e-8)
    assert np.allclose(cell.to_conventional @ cell.to_primitive, np.eye(3), atol=1e-8)


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_multiplicity_is_the_index_of_the_conventional_lattice(symbol, seed):
    lattice = rebased(LATTICES[symbol], seed)
    cell = conventional_cell(lattice)
    ratio = abs(float(np.linalg.det(cell.cell))) / abs(float(np.linalg.det(lattice)))
    assert ratio == pytest.approx(cell.multiplicity, rel=1e-9)
    assert abs(float(np.linalg.det(cell.to_primitive))) == pytest.approx(
        cell.multiplicity, rel=1e-9
    )
    assert float(np.linalg.det(cell.cell)) > 0.0, "a conventional cell is right handed"


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_centring_vectors_are_the_cosets_of_the_conventional_lattice(symbol, seed):
    """They must be lattice points, be as many as the index, and form a group."""

    lattice = rebased(LATTICES[symbol], seed)
    cell = conventional_cell(lattice)
    vectors = np.asarray(cell.centring_vectors, dtype=float)
    assert len(vectors) == cell.multiplicity
    assert np.allclose(vectors[0], 0.0), "the origin is always a coset"
    for vector in vectors:
        integer = (vector @ cell.cell) @ np.linalg.inv(lattice)
        assert np.allclose(integer, np.rint(integer), atol=1e-7)
    keys = {tuple(int(value) % 12 for value in np.rint(vector * 12.0)) for vector in vectors}
    assert len(keys) == cell.multiplicity, "the cosets are distinct"
    for first in keys:
        for second in keys:
            total = tuple((a + b) % 12 for a, b in zip(first, second))
            assert total in keys, "the cosets are a group under addition"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_conventional_metric_obeys_its_family(seed):
    """Each family constrains the conventional cell, and nothing else does."""

    for symbol, lattice in LATTICES.items():
        cell = conventional_cell(rebased(lattice, seed))
        a, b, c, alpha, beta, gamma = cell.parameters
        family = symbol[0]
        if family == "c":
            assert a == pytest.approx(b, rel=1e-8) and b == pytest.approx(c, rel=1e-8)
            assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-6)
        elif family == "t":
            assert a == pytest.approx(b, rel=1e-8)
            assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-6)
        elif family == "h":
            assert a == pytest.approx(b, rel=1e-8)
            assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 120.0), abs=1e-6)
        elif family == "o":
            assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-6)
        elif family == "m":
            assert (alpha, gamma) == pytest.approx((90.0, 90.0), abs=1e-6)
            assert beta > 90.0 + 1e-6, "the standard monoclinic setting has an obtuse beta"


@pytest.mark.parametrize("shear", [-3.5, -1.9, -0.4, 0.4, 1.9, 3.5])
def test_a_monoclinic_cell_is_reported_with_an_obtuse_beta(shear):
    """The two settings (a, b, c) and (-a, -b, c) differ; the obtuse one is standard."""

    for lattice in (
        np.array([[A, 0.0, 0.0], [0.0, B, 0.0], [shear, 0.0, C]]),
        np.array([[A / 2, B / 2, 0.0], [-A / 2, B / 2, 0.0], [shear, 0.0, C]]),
    ):
        cell = conventional_cell(lattice)
        assert cell.system == "monoclinic"
        _, _, _, alpha, beta, gamma = cell.parameters
        assert (alpha, gamma) == pytest.approx((90.0, 90.0), abs=1e-6)
        assert beta >= 90.0


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_classification_does_not_depend_on_the_basis(symbol):
    """Every reported number must be a property of the lattice, not of the basis."""

    first = conventional_cell(LATTICES[symbol])
    for seed in SEEDS:
        other = conventional_cell(rebased(LATTICES[symbol], seed))
        assert other.symbol == first.symbol
        assert other.multiplicity == first.multiplicity
        assert other.parameters == pytest.approx(first.parameters, rel=1e-7, abs=1e-7)
        assert np.allclose(
            np.sort(other.centring_vectors, axis=0),
            np.sort(first.centring_vectors, axis=0),
            atol=1e-8,
        )


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.5, 2.0, 5.0, 20.0])
def test_an_extreme_aspect_ratio_is_still_classified(ratio):
    """A very flat or very long cell must not fall out of the search shell."""

    side = 4.0
    height = side * ratio
    cases = {
        "tP": np.diag([side, side, height]),
        "tI": np.array(
            [
                [-side / 2, side / 2, height / 2],
                [side / 2, -side / 2, height / 2],
                [side / 2, side / 2, -height / 2],
            ]
        ),
        "hP": np.array(
            [[side, 0.0, 0.0], [-side / 2, side * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, height]]
        ),
        "hR": rhombohedral(side, 3.0 * height),
    }
    for symbol, lattice in cases.items():
        assert bravais_symbol(lattice) == symbol


@pytest.mark.parametrize("symbol", sorted(LATTICES))
def test_the_summary_is_json_ready(symbol):
    cell = conventional_cell(LATTICES[symbol])
    text = json.dumps(cell.summary())
    loaded = json.loads(text)
    assert loaded["bravais_symbol"] == symbol
    assert loaded["multiplicity"] == cell.multiplicity
    assert len(loaded["centring_vectors"]) == cell.multiplicity
    assert loaded["conventional_parameters"]["a"] == pytest.approx(cell.parameters[0])


def test_a_degenerate_lattice_is_refused():
    with pytest.raises(ValueError):
        conventional_cell(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_every_lattice_point_has_twelfth_coordinates_in_the_conventional_cell(symbol, seed):
    """The integer key ``round(12 x)`` used to collect the cosets is lossless.

    A lattice point has conventional coordinates whose denominator divides the
    index of the conventional cell, and that index is at most four, so the
    coordinates are exact twelfths.  Formally
    ``Cellstine.det_smul_conventionalCoords`` and
    ``Cellstine.twelve_mul_coords_isInt``.
    """

    lattice = rebased(LATTICES[symbol], seed)
    cell = conventional_cell(lattice)
    assert cell.multiplicity <= 4
    span = np.arange(-3, 4)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    fractional = (grid.astype(float) @ lattice) @ np.linalg.inv(cell.cell)
    scaled = fractional * 12.0
    assert np.allclose(scaled, np.rint(scaled), atol=1e-7), "the coordinates are exact twelfths"
    scaled_by_index = fractional * cell.multiplicity
    assert np.allclose(scaled_by_index, np.rint(scaled_by_index), atol=1e-7)


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_conventional_axes_are_primitive_lattice_vectors(symbol, seed):
    """Each conventional axis generates the whole line of lattice points on it.

    ``_shortest_along`` picks the shortest lattice vector along a symmetry
    axis; that vector must divide every lattice vector parallel to it, which is
    exactly the statement that its integer coordinates are coprime.  Formally
    ``Cellstine.exists_primitive_axis_generator``.
    """

    lattice = rebased(LATTICES[symbol], seed)
    cell = conventional_cell(lattice)
    inverse = np.linalg.inv(lattice)
    span = np.arange(-6, 7)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    points = grid.astype(float) @ lattice
    for axis in cell.cell:
        coordinates = axis @ inverse
        assert np.allclose(coordinates, np.rint(coordinates), atol=1e-8)
        integers = np.rint(coordinates).astype(np.int64)
        assert math.gcd(*(abs(int(value)) for value in integers)) == 1, "the axis is primitive"
        length = float(np.linalg.norm(axis))
        unit = axis / length
        parallel = np.abs(np.abs(points @ unit) - np.linalg.norm(points, axis=1)) <= 1e-8 * length
        multiples = (points[parallel] @ unit) / length
        assert np.allclose(multiples, np.rint(multiples), atol=1e-7), (
            "every lattice vector along the axis is an integer multiple of it"
        )


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("seed", (0, 1))
def test_a_change_of_basis_matches_the_two_point_groups(symbol, seed):
    """``W ↦ Tᵀ W Sᵀ`` is a bijection of point groups, so the family is intrinsic.

    Formally ``Cellstine.pointGroupEquiv``, ``Cellstine.card_preservesGram_eq``
    and ``Cellstine.pointGroupEquiv_mul``.
    """

    lattice = LATTICES[symbol]
    transform = unimodular(seed).astype(float)
    inverse = np.rint(np.linalg.inv(transform)).astype(float)
    assert np.allclose(transform @ inverse, np.eye(3))
    here = lattice_point_group(lattice)
    there = lattice_point_group(transform @ lattice)
    assert len(here) == len(there) == HOLOHEDRY_ORDER[conventional_cell(lattice).system]

    def key(matrix: np.ndarray) -> tuple[int, ...]:
        return tuple(int(round(value)) for value in np.asarray(matrix).reshape(-1))

    mapped = {key(transform.T @ operation @ inverse.T) for operation in there}
    assert mapped == {key(operation) for operation in here}, "the matching is onto"
    lookup = {key(operation) for operation in here}
    for first in there:
        for second in there:
            image = transform.T @ (first @ second) @ inverse.T
            product = (transform.T @ first @ inverse.T) @ (transform.T @ second @ inverse.T)
            assert np.allclose(image, product), "the matching multiplies"
            assert key(product) in lookup

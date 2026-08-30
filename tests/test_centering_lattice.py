"""The translation group of a cell, its centring, and the primitive surface cell.

A surface cell is only primitive if it is built on the lattice of the *pure
translations* of the input structure.  Guessing that lattice from the four
standard centring vectors is right for a conventional cell and wrong for
anything else --- a cell that is a supercell of a smaller one then produces a
surface cell that is a repeat of the primitive one, with every reported quantity
(atom count, in-plane vectors, area) too large by that factor.

The tests below check the group itself against the statements proved in
``RequestProject/CenteringLattice.lean``: it is a group, its order divides the
count of every species, that order is a common denominator of its elements, the
basis it yields has determinant ``1 / order``, and the centring letters it
reports name exactly the standard groups.  They then check the consequence that
motivates all of it: doubling the input cell does not change the surface cell
that comes out.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core import symmetry3d as sym
from cellstine.interface.surface import surface_cell as sc
from cellstine.io import native as io_mod


def _structure(lattice, positions, species, counts) -> io_mod.PoscarData:
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return io_mod.PoscarData(
        comment="test structure",
        lattice=lattice,
        species=list(species),
        counts=[int(value) for value in counts],
        positions_direct=positions,
        positions_cartesian=io_mod.direct_to_cartesian(positions, lattice),
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
    )


def _supercell(structure: io_mod.PoscarData, repeats) -> io_mod.PoscarData:
    """Return the ``n1 x n2 x n3`` supercell of a structure, species-major."""

    repeats = tuple(int(value) for value in repeats)
    shifts = np.array(list(itertools.product(*(range(value) for value in repeats))), dtype=float)
    scale = np.array(repeats, dtype=float)
    blocks = []
    offset = 0
    counts = []
    for count in structure.counts:
        block = structure.positions_direct[offset : offset + count]
        offset += count
        images = (block[:, None, :] + shifts[None, :, :]).reshape(-1, 3) / scale
        blocks.append(images)
        counts.append(count * len(shifts))
    lattice = np.asarray(structure.lattice, dtype=float) * scale[:, None]
    return _structure(lattice, np.vstack(blocks), structure.species, counts)


CUBIC = 3.6 * np.eye(3)
HEX = np.array([[3.2, 0.0, 0.0], [-1.6, 1.6 * math.sqrt(3.0), 0.0], [0.0, 0.0, 5.2]])

SIMPLE = _structure(CUBIC, [[0.0, 0.0, 0.0]], ["Cu"], [1])
FACE_CENTRED = _structure(
    CUBIC, [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], ["Cu"], [4]
)
BODY_CENTRED = _structure(CUBIC, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], ["W"], [2])
BASE_CENTRED = _structure(
    np.diag([3.0, 4.0, 5.0]), [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]], ["Fe"], [2]
)
A_CENTRED = _structure(
    np.diag([3.0, 4.0, 5.0]), [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]], ["Fe"], [2]
)
B_CENTRED = _structure(
    np.diag([3.0, 4.0, 5.0]), [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]], ["Fe"], [2]
)
ROCK_SALT = _structure(
    5.6 * np.eye(3),
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ],
    ["Na", "Cl"],
    [4, 4],
)
GRAPHITE_LIKE = _structure(
    HEX, [[0.0, 0.0, 0.25], [1.0 / 3.0, 2.0 / 3.0, 0.75]], ["C"], [2]
)


def _wrap(vectors: np.ndarray) -> np.ndarray:
    values = np.mod(np.asarray(vectors, dtype=float), 1.0)
    return np.where(values > 1.0 - 1e-9, 0.0, values)


def _contains(group: np.ndarray, vector: np.ndarray, tolerance: float = 1e-8) -> bool:
    difference = _wrap(group) - _wrap(np.asarray(vector, dtype=float))[None, :]
    difference -= np.round(difference)
    return bool(np.any(np.all(np.abs(difference) <= tolerance, axis=1)))


ALL_STRUCTURES = [
    SIMPLE,
    FACE_CENTRED,
    BODY_CENTRED,
    BASE_CENTRED,
    A_CENTRED,
    B_CENTRED,
    ROCK_SALT,
    GRAPHITE_LIKE,
    _supercell(SIMPLE, (2, 1, 1)),
    _supercell(SIMPLE, (2, 2, 1)),
    _supercell(BODY_CENTRED, (1, 1, 3)),
    _supercell(GRAPHITE_LIKE, (2, 2, 1)),
]


def _group(structure: io_mod.PoscarData) -> np.ndarray:
    return sc._pure_translations(structure)


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_the_translations_form_a_group(structure: io_mod.PoscarData) -> None:
    """`Cellstine.Centering.translationSubgroup`: closed under `0`, `+` and `-`."""

    group = _group(structure)
    assert _contains(group, np.zeros(3))
    for first in group:
        assert _contains(group, -first)
        for second in group:
            assert _contains(group, first + second)


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_the_order_divides_every_species_count(structure: io_mod.PoscarData) -> None:
    """`Cellstine.Centering.card_translations_dvd`."""

    order = len(_group(structure))
    assert order >= 1
    for count in structure.counts:
        assert count % order == 0


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_the_order_is_a_common_denominator(structure: io_mod.PoscarData) -> None:
    """`Cellstine.Centering.card_nsmul_mem_base`: `m t` is an integer vector."""

    group = _group(structure)
    scaled = len(group) * np.asarray(group, dtype=float)
    assert np.allclose(scaled, np.round(scaled), atol=1e-9)


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_every_translation_maps_the_structure_onto_itself(structure: io_mod.PoscarData) -> None:
    """The group agrees with a direct site-by-site test of each shift."""

    for translation in _group(structure):
        assert sc._translation_maps_structure(structure, translation)
    for probe in ([0.25, 0.0, 0.0], [0.1, 0.2, 0.3], [0.5, 0.25, 0.0]):
        if not _contains(_group(structure), np.array(probe)):
            assert not sc._translation_maps_structure(structure, probe)


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_the_basis_spans_exactly_the_translation_lattice(structure: io_mod.PoscarData) -> None:
    """The basis has determinant `1 / m`, and generates exactly the translations."""

    species = []
    for symbol, count in zip(structure.species, structure.counts):
        species.extend([symbol] * int(count))
    basis, order = sym.translation_lattice_basis(
        structure.lattice, structure.positions_direct, species, symprec=1e-4
    )
    assert order == len(_group(structure))
    assert float(np.linalg.det(basis)) == pytest.approx(1.0 / order, abs=1e-12)

    # every basis row is a translation of the lattice (a translation of the
    # structure, up to a unit cell vector), and every translation is an integer
    # combination of the basis rows.
    for row in basis:
        assert _contains(_group(structure), row)
    inverse = np.linalg.inv(basis)
    for translation in _group(structure):
        coefficients = translation @ inverse
        assert np.allclose(coefficients, np.round(coefficients), atol=1e-9)
    for unit in np.eye(3):
        coefficients = unit @ inverse
        assert np.allclose(coefficients, np.round(coefficients), atol=1e-9)


@pytest.mark.parametrize(
    "structure, letter, order",
    [
        (SIMPLE, "P", 1),
        (FACE_CENTRED, "F", 4),
        (BODY_CENTRED, "I", 2),
        (BASE_CENTRED, "C", 2),
        (A_CENTRED, "A", 2),
        (B_CENTRED, "B", 2),
        (ROCK_SALT, "F", 4),
        (GRAPHITE_LIKE, "P", 1),
        (_supercell(SIMPLE, (2, 1, 1)), "X2", 2),
        (_supercell(SIMPLE, (2, 2, 1)), "X4", 4),
        (_supercell(BODY_CENTRED, (1, 1, 3)), "X6", 6),
    ],
)
def test_the_reported_centring_letter(
    structure: io_mod.PoscarData, letter: str, order: int
) -> None:
    """A standard centring is named; anything else is reported as its order."""

    assert sc._centering_type(structure) == letter
    assert len(_group(structure)) == order


def test_a_rhombohedral_setting_is_named_r() -> None:
    """The thirds of the `R` setting are recognised as well."""

    lattice = np.array([[4.0, 0.0, 0.0], [-2.0, 2.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 12.0]])
    base = np.array([[0.0, 0.0, 0.0]])
    shifts = np.array([[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]])
    positions = np.mod((base[:, None, :] + shifts[None, :, :]).reshape(-1, 3), 1.0)
    structure = _structure(lattice, positions, ["Al"], [len(positions)])
    assert sc._centering_type(structure) == "R"


@pytest.mark.parametrize("letter", ["P", "A", "B", "C", "I", "F"])
def test_the_standard_centring_matrices_are_right_handed_and_have_the_right_index(
    letter: str,
) -> None:
    """`Cellstine.Centering.det_centeringA` and friends, plus the handedness."""

    expected = {"P": 1.0, "A": 0.5, "B": 0.5, "C": 0.5, "I": 0.5, "F": 0.25}
    matrix = sc._standard_centering_matrices()[letter]
    assert float(np.linalg.det(matrix)) == pytest.approx(expected[letter], abs=1e-12)


@pytest.mark.parametrize(
    "structure",
    [SIMPLE, FACE_CENTRED, BODY_CENTRED, BASE_CENTRED, A_CENTRED, B_CENTRED, ROCK_SALT],
)
def test_a_standard_centring_keeps_its_textbook_basis(structure: io_mod.PoscarData) -> None:
    """The fast path returns exactly the hardcoded matrix of the letter."""

    lattice, letter = sc._primitive_translation_lattice(structure)
    expected = sc._standard_centering_matrices()[letter] @ np.asarray(structure.lattice, dtype=float)
    assert np.array_equal(lattice, expected)


@pytest.mark.parametrize(
    "miller, repeats",
    [
        # ``(hkl)`` is read in the axes of the cell it was given, so a repeat is
        # only the same plane when it leaves those axes proportional: any repeat
        # for ``(001)`` that keeps ``c``, and an isotropic one otherwise.
        ((0, 0, 1), (2, 1, 1)),
        ((0, 0, 1), (1, 2, 1)),
        ((0, 0, 1), (2, 2, 1)),
        ((0, 0, 1), (2, 2, 2)),
        ((1, 1, 0), (2, 2, 1)),
        ((1, 1, 0), (2, 2, 2)),
        ((1, 1, 1), (2, 2, 2)),
    ],
)
def test_a_repeated_cell_gives_the_same_surface_cell(miller, repeats) -> None:
    """The motivating regression: a supercell input is reduced before the search.

    The surface cell of a structure is a property of the structure, not of the
    cell it was handed in, so building on a ``2 x 2 x 2`` repeat of a crystal has
    to give the same primitive surface cell as building on the crystal itself.
    """

    base = sc._build_native_primitive_surface_cell(FACE_CENTRED, miller, layers=3, vacuum=12.0)
    repeated = sc._build_native_primitive_surface_cell(
        _supercell(FACE_CENTRED, repeats), miller, layers=3, vacuum=12.0
    )
    assert repeated.counts == base.counts
    assert np.allclose(repeated.lattice, base.lattice, atol=1e-9)


def test_a_repeated_cell_gives_the_same_in_plane_area() -> None:
    """A doubled simple-cubic cell yields the primitive, not the doubled, surface."""

    doubled = _supercell(SIMPLE, (2, 1, 1))
    cell = sc._build_native_primitive_surface_cell(doubled, (0, 0, 1), layers=2, vacuum=10.0)
    area = float(np.linalg.norm(np.cross(cell.lattice[0], cell.lattice[1])))
    assert area == pytest.approx(3.6 * 3.6, abs=1e-9)
    assert sum(cell.counts) == 2

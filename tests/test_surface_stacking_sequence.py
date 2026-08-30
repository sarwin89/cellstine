"""Reading the plane-by-plane stacking sequence of a slab.

``interface/surface/sequence.py`` names the atomic planes of a slab ``A``, ``B``,
``C``, ... from the bottom of the cell upwards, and two planes share a letter
when their atoms sit over one another.  The whole question is what "sit over one
another" means for a structure that came back from a relaxation, and the answer
has to be a distance: the letters are decided by the shortest in-plane periodic
image, so a plane that has moved sideways by a few thousandths of an angstrom is
still the same plane.

The tests below pin the sequences of the standard close-packed slabs, the
robustness that motivates the distance test, and the invariances a letter must
have -- wrapping a coordinate through the cell edge, listing the atoms in a
different order, or translating the whole slab.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.interface.surface import backend as surface_backend
from cellstine.interface.surface.sequence import (
    shortest_repeating_prefix,
    stacking_sequence,
)
from cellstine.io import native as io_mod

from conftest import write_poscar

HEXAGONAL = np.array([[2.5, 0.0, 0.0], [-1.25, 2.5 * math.sqrt(3) / 2, 0.0], [0.0, 0.0, 30.0]])
SQUARE = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 30.0]])

#: The three close-packed in-plane sites of a triangular lattice.
SITE_A = (0.0, 0.0)
SITE_B = (1.0 / 3.0, 2.0 / 3.0)
SITE_C = (2.0 / 3.0, 1.0 / 3.0)


def _slab(lattice: np.ndarray, planes, *, species: list[str] | None = None) -> io_mod.PoscarData:
    """Build a slab whose planes are given as lists of in-plane fractional sites."""

    rows = []
    for index, sites in enumerate(planes):
        height = 0.1 + 0.08 * index
        for site in sites:
            rows.append([float(site[0]), float(site[1]), height])
    direct = np.array(rows, dtype=float)
    return io_mod.PoscarData(
        comment="test slab",
        lattice=np.asarray(lattice, dtype=float),
        species=species or ["C"],
        counts=[len(direct)],
        positions_direct=direct,
        positions_cartesian=io_mod.direct_to_cartesian(direct, lattice),
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
    )


def test_an_fcc_slab_reads_abcabc():
    slab = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_C], [SITE_A], [SITE_B], [SITE_C]])
    sequence, counts = stacking_sequence(slab)
    assert sequence == "ABCABC"
    assert counts == (1, 1, 1, 1, 1, 1)
    assert shortest_repeating_prefix(sequence) == "ABC"


def test_an_hcp_slab_reads_abab():
    slab = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_A], [SITE_B]])
    sequence, _ = stacking_sequence(slab)
    assert sequence == "ABAB"
    assert shortest_repeating_prefix(sequence) == "AB"


def test_a_simple_stack_reads_aaaa():
    slab = _slab(SQUARE, [[SITE_A], [SITE_A], [SITE_A], [SITE_A]])
    sequence, _ = stacking_sequence(slab)
    assert sequence == "AAAA"
    assert shortest_repeating_prefix(sequence) == "A"


@pytest.mark.parametrize("displacement", [1e-6, 1.4e-3, 3e-3, 1e-2])
def test_a_relaxed_plane_keeps_its_letter(displacement):
    """A plane that moved sideways by a few thousandths of an angstrom is the same plane.

    The letters used to be decided by snapping fractional coordinates onto a
    fixed grid, which reported this slab as ``ABCADC``: the fourth plane fell on
    the other side of a rounding boundary from the first.
    """

    moved = (SITE_A[0] + displacement, SITE_A[1])
    slab = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_C], [moved], [SITE_B], [SITE_C]])
    sequence, _ = stacking_sequence(slab)
    assert sequence == "ABCABC"
    assert displacement * 2.5 < 0.05


def test_a_genuinely_different_plane_gets_its_own_letter():
    slab = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_C], [(0.45, 2.0 / 3.0)], [SITE_B], [SITE_C]])
    sequence, _ = stacking_sequence(slab)
    assert sequence == "ABCDBC"


def test_wrapping_a_coordinate_through_the_cell_edge_changes_nothing():
    wrapped = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [(2.0 / 3.0, 1.0 / 3.0 - 1.0)], [(1.0, -1.0)]])
    plain = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_C], [SITE_A]])
    assert stacking_sequence(wrapped)[0] == stacking_sequence(plain)[0] == "ABCA"


def test_listing_the_atoms_of_a_plane_in_another_order_changes_nothing():
    first = _slab(
        HEXAGONAL,
        [[SITE_A, (0.5, 0.5)], [SITE_B, (0.5 + 1.0 / 3.0, 0.5 + 2.0 / 3.0)], [SITE_A, (0.5, 0.5)]],
    )
    second = _slab(
        HEXAGONAL,
        [[(0.5, 0.5), SITE_A], [(0.5 + 1.0 / 3.0, 0.5 + 2.0 / 3.0), SITE_B], [(0.5, 0.5), SITE_A]],
    )
    assert stacking_sequence(first)[0] == stacking_sequence(second)[0] == "ABA"


def test_translating_the_whole_slab_in_plane_changes_nothing():
    shift = 0.17
    plain = _slab(HEXAGONAL, [[SITE_A], [SITE_B], [SITE_C], [SITE_A]])
    moved = _slab(
        HEXAGONAL,
        [[(site[0] + shift, site[1] + shift)] for site in (SITE_A, SITE_B, SITE_C, SITE_A)],
    )
    assert stacking_sequence(moved)[0] == stacking_sequence(plain)[0] == "ABCA"


def test_planes_with_different_atom_counts_are_different_planes():
    slab = _slab(HEXAGONAL, [[SITE_A], [SITE_A, (0.5, 0.5)], [SITE_A]])
    sequence, counts = stacking_sequence(slab)
    assert sequence == "ABA"
    assert counts == (1, 2, 1)


def test_an_empty_structure_has_no_sequence():
    empty = io_mod.PoscarData(
        comment="empty",
        lattice=HEXAGONAL,
        species=["C"],
        counts=[0],
        positions_direct=np.zeros((0, 3)),
        positions_cartesian=np.zeros((0, 3)),
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
    )
    assert stacking_sequence(empty) == ("", tuple())
    assert shortest_repeating_prefix("") == ""


@pytest.mark.parametrize(
    "sequence, period",
    [("ABCABC", "ABC"), ("ABCA", "ABC"), ("ABAB", "AB"), ("AAAA", "A"), ("ABCD", "ABCD")],
)
def test_the_repeating_block_is_the_shortest_one(sequence, period):
    assert shortest_repeating_prefix(sequence) == period


@pytest.fixture(scope="module")
def copper_poscar(tmp_path_factory):
    """Return the one-atom primitive cell of fcc copper."""

    path = tmp_path_factory.mktemp("bulk") / "cu.vasp"
    lattice = 3.61 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return write_poscar(path, lattice, ["Cu"], [1], np.zeros((1, 3)))


@pytest.mark.parametrize(
    "miller, period, angle",
    [
        # Read in the basis of the primitive cell: (111) and (100) are both
        # close-packed planes of the fcc lattice, (110) is the square one.
        ((1, 1, 1), "ABC", 120.0),
        ((1, 0, 0), "ABC", 120.0),
        ((1, 1, 0), "AB", 90.0),
    ],
)
def test_a_primitive_surface_reports_its_own_repeat(copper_poscar, miller, period, angle):
    """An fcc slab repeats every three close-packed planes and every two square ones."""

    analysis = surface_backend.analyse_primitive_surface(
        str(copper_poscar), miller=miller, probe_layers=6
    )
    assert analysis.stacking_period == period
    assert analysis.stacking_sequence == period * (6 // len(period))
    assert analysis.atoms_per_layer == (1,) * 6
    assert analysis.inplane_angle_deg == pytest.approx(angle, abs=1e-6)
    assert analysis.centering == "P"

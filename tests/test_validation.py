"""The structure sanity check every written POSCAR passes.

A generator can be mathematically correct and still produce a file no
plane-wave code will accept.  The three faults checked here are the ones that
make a calculation meaningless rather than merely unusual: a cell whose vectors
are coplanar has no reciprocal lattice, a count list that disagrees with the
positions describes a different structure than the one that was built, and two
atoms on one site is an infinite Coulomb term.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.validation import (
    coincident_site_pairs,
    ensure_positive,
    structure_errors,
    validate_structure,
)
from cellstine.io import native as io_mod

CUBIC = 4.0 * np.eye(3)
TWO_ATOMS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def test_a_well_formed_structure_has_nothing_wrong_with_it():
    assert (
        structure_errors(
            lattice=CUBIC, species=["Al"], counts=[2], positions_direct=TWO_ATOMS
        )
        == []
    )


def test_a_coplanar_cell_is_rejected():
    flat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    problems = structure_errors(
        lattice=flat, species=["Al"], counts=[1], positions_direct=np.zeros((1, 3))
    )
    assert any("degenerate" in note for note in problems)


def test_counts_that_disagree_with_the_positions_are_rejected():
    problems = structure_errors(
        lattice=CUBIC, species=["Al"], counts=[3], positions_direct=TWO_ATOMS
    )
    assert any("2 positions" in note for note in problems)


def test_a_non_finite_coordinate_is_rejected():
    broken = np.array([[0.0, 0.0, 0.0], [0.5, math.nan, 0.5]])
    problems = structure_errors(
        lattice=CUBIC, species=["Al"], counts=[2], positions_direct=broken
    )
    assert any("non-finite" in note for note in problems)


def test_two_atoms_on_one_site_are_found_even_across_a_cell_face():
    """A duplicate written as ``x`` and ``x + t`` is still a duplicate."""

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert coincident_site_pairs(CUBIC, positions) == [(0, 1)]
    problems = structure_errors(
        lattice=CUBIC, species=["Al"], counts=[3], positions_direct=positions
    )
    assert any("same site" in note for note in problems)


def test_two_distinct_atoms_however_close_are_not_a_duplicate():
    """A short bond is chemistry to report, not a structure to reject."""

    positions = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    assert coincident_site_pairs(CUBIC, positions) == []
    assert (
        structure_errors(
            lattice=CUBIC, species=["Al"], counts=[2], positions_direct=positions
        )
        == []
    )


def test_a_skewed_cell_does_not_invent_duplicates():
    """The duplicate test is a periodic one, so an acute cell must not fool it."""

    skewed = np.array([[4.0, 0.0, 0.0], [2.0, 3.4641, 0.0], [1.0, 0.7, 5.0]])
    generator = np.random.default_rng(20260826)
    positions = generator.random((40, 3))
    assert coincident_site_pairs(skewed, positions) == []
    doubled = np.vstack([positions, positions[7] + np.array([0.0, -1.0, 2.0])])
    assert coincident_site_pairs(skewed, doubled) == [(7, 40)]


def test_the_writer_refuses_a_structure_with_a_duplicated_atom(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = tmp_path / "broken.vasp"
    with pytest.raises(ValueError, match="same site"):
        io_mod.write_poscar(
            str(target),
            CUBIC,
            positions,
            [2],
            ["Al"],
            positions_are_cartesian=False,
        )
    assert not target.exists()


def test_the_writer_can_be_told_to_write_anyway(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = tmp_path / "deliberate.vasp"
    io_mod.write_poscar(
        str(target),
        CUBIC,
        positions,
        [2],
        ["Al"],
        positions_are_cartesian=False,
        validate=False,
    )
    assert target.exists()


def test_a_good_structure_still_round_trips_through_the_writer(tmp_path):
    target = tmp_path / "good.vasp"
    io_mod.write_poscar(
        str(target), CUBIC, TWO_ATOMS, [2], ["Al"], positions_are_cartesian=False
    )
    read_back = io_mod.read_poscar(str(target))
    assert read_back.natoms == 2
    assert np.allclose(read_back.positions_direct, TWO_ATOMS)


def test_validate_structure_names_the_file_it_refused():
    with pytest.raises(ValueError, match="POSCAR-42"):
        validate_structure(
            lattice=CUBIC,
            species=["Al"],
            counts=[2],
            positions_direct=np.zeros((2, 3)),
            context="refusing to write POSCAR-42",
        )


def test_ensure_positive_still_guards_its_argument():
    assert ensure_positive(2.5, name="gap") == 2.5
    with pytest.raises(ValueError, match="gap must be positive"):
        ensure_positive(0.0, name="gap")

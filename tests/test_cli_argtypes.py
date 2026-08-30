"""One command-line token in, one usable object out -- or a readable refusal.

The value parsers are the first thing a run meets, and they are the cheapest
place to catch a typo: a repeat of ``0`` that would build an empty cell, a mesh
shift that is not a whole or half grid step, a supercell matrix that collapses
the cell onto a plane.  Every one of those has to come back as an
``argparse.ArgumentTypeError``, which argparse turns into a usage message,
rather than as a traceback from somewhere deep in a builder.

The accepted spellings are checked too: users type ``2x2x1``, ``2,2,1`` and
``2`` for the same thing, and all three have to mean it.
"""

from __future__ import annotations

import argparse

import pytest

from cellstine.cli.argtypes import (
    parse_float_vector,
    parse_index_spec,
    parse_int_matrix,
    parse_mesh_shift,
    parse_nonnegative_float,
    parse_positive_float,
    parse_positive_int,
    parse_string_list,
    parse_supercell,
    parse_supercell_matrix,
)


@pytest.mark.parametrize(
    "raw,expected", [("2", [2, 2, 2]), ("2,2,1", [2, 2, 1]), ("2x2x1", [2, 2, 1]), ("3 1 4", [3, 1, 4])]
)
def test_a_supercell_can_be_spelled_in_every_usual_way(raw, expected):
    assert parse_supercell(raw) == expected


@pytest.mark.parametrize("raw", ["0", "2,0,1", "-1,2,2", "2,2", "2,2,1,1", "two", ""])
def test_a_supercell_that_builds_nothing_is_refused(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_supercell(raw)


@pytest.mark.parametrize("raw", ["0,0,0", "0.5,0.5,0", "0.5;0.5;0.5", "1,0,-0.5"])
def test_a_mesh_shift_on_the_half_grid_is_accepted(raw):
    values = parse_mesh_shift(raw)
    assert len(values) == 3
    assert all(abs(2.0 * value - round(2.0 * value)) < 1e-12 for value in values)


@pytest.mark.parametrize("raw", ["0.25,0,0", "0,0", "0,0,0,0", "a,b,c", ""])
def test_a_mesh_shift_off_the_half_grid_is_refused(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_mesh_shift(raw)


def test_a_supercell_matrix_is_read_row_by_row():
    assert parse_supercell_matrix("-1,1,1,1,-1,1,1,1,-1") == [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    assert parse_supercell_matrix("1 0 0 0 1 0 0 0 2") == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


@pytest.mark.parametrize(
    "raw",
    [
        "1,0,0,0,1,0,0,0,0",  # flattens the third direction
        "1,1,0,1,1,0,0,0,1",  # two equal rows
        "1,0,0,0,1,0,0,0",  # eight integers
        "1,0,0,0,1,0,0,0,x",
    ],
)
def test_a_supercell_matrix_that_is_not_a_cell_is_refused(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_supercell_matrix(raw)


def test_an_in_plane_matrix_must_be_invertible():
    assert parse_int_matrix("1,1,0,2") == [1, 1, 0, 2]
    assert parse_int_matrix("2;0;0;2") == [2, 0, 0, 2]
    for raw in ("1,2,2,4", "0,0,0,0", "1,2,3", "1,2,3,4,5", "1,2,3,z"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_int_matrix(raw)


def test_index_ranges_run_both_ways_and_lose_no_index():
    assert parse_index_spec("1,2,5") == [1, 2, 5]
    assert parse_index_spec("1-4") == [1, 2, 3, 4]
    assert parse_index_spec("4-1") == [4, 3, 2, 1]
    assert parse_index_spec("1-3,3,7") == [1, 2, 3, 7]
    assert parse_index_spec(" 2 , , 3 ") == [2, 3]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_index_spec(" , ")


def test_a_vector_has_two_or_three_components():
    assert parse_float_vector("1,0") == [1.0, 0.0]
    assert parse_float_vector("1;0;-2.5") == [1.0, 0.0, -2.5]
    for raw in ("1", "1,2,3,4", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_float_vector(raw)


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "-inf"])
def test_a_positive_length_is_finite_and_above_zero(raw):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        parse_positive_float(raw)


def test_the_numeric_parsers_agree_on_the_boundary():
    assert parse_positive_float("1e-9") == pytest.approx(1e-9)
    assert parse_nonnegative_float("0") == 0.0
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_nonnegative_float("-1e-9")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_nonnegative_float("nan")
    assert parse_positive_int("7") == 7
    for raw in ("0", "-2", "1.5", "seven"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_positive_int(raw)


def test_an_empty_string_list_is_absence_not_an_empty_choice():
    assert parse_string_list(None) is None
    assert parse_string_list("") is None
    assert parse_string_list("Cu, Ag ,Au") == ["Cu", "Ag", "Au"]
    assert parse_string_list(",,") == []

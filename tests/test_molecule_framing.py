"""The three gap choices the adsorbate placement makes, checked on the code.

`adsorbate/placement/operations.py` decides which atoms are the molecule, where
the periodic cell begins, and how many substrate repeats the molecule needs.
All three are choices of a gap, and `RequestProject/MoleculeFraming.lean` says
what each of them guarantees:

* the height split (`Cellstine.height_lt_gapCut`, `Cellstine.gapCut_lt_height`,
  `Cellstine.gapCut_clearance_below`, `Cellstine.exists_height_near_of_mem_range`),
* the periodic branch cut (`Cellstine.unwrapTo_sub_mem`,
  `Cellstine.arcSpan_lt_one`, `Cellstine.arcSpan_min_le`,
  `Cellstine.centred_mem_Ioo`),
* the in-plane fit (`Cellstine.inplaneRepeats_clearance`,
  `Cellstine.inplaneRepeats_le`).

Each test re-measures the guarantee from the values the implementation returns,
and where the statement is an optimality claim it is checked against a
brute-force search over the alternatives.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.adsorbate.placement import operations as placement
from cellstine.core.species import expand_species
from cellstine.io import native as io_mod

from conftest import write_poscar


# --------------------------------------------------------------------------
# The height split
# --------------------------------------------------------------------------


def _stacked_structure(tmp_path, heights, species, cell_height=30.0):
    """A structure whose atoms sit at the given heights on the cell axis."""

    lattice = np.diag([3.0, 3.0, float(cell_height)])
    positions = np.array([[0.25, 0.25, z / cell_height] for z in heights], dtype=float)
    order = {}
    for symbol in species:
        order.setdefault(symbol, 0)
        order[symbol] += 1
    unique = list(order)
    counts = [order[symbol] for symbol in unique]
    index = np.argsort([unique.index(symbol) for symbol in species], kind="stable")
    path = write_poscar(
        tmp_path / "stack.vasp", lattice, unique, counts, positions[index]
    )
    return io_mod.read_poscar(str(path)), [heights[i] for i in index]


def test_split_cuts_at_the_largest_gap(tmp_path):
    heights = [0.0, 2.0, 4.0, 4.5, 9.0, 10.1]
    structure, sorted_heights = _stacked_structure(
        tmp_path, heights, ["Al"] * 4 + ["C", "O"]
    )
    selection = placement.identify_top_group(structure)

    gaps = np.diff(sorted(sorted_heights))
    largest = float(np.max(gaps))
    assert selection.gap_size == pytest.approx(largest)
    # the cut is the midpoint of that gap: 4.5 and 9.0
    assert selection.z_cutoff == pytest.approx(0.5 * (4.5 + 9.0))
    assert selection.molecule_indices == (4, 5)


def test_every_atom_is_half_a_gap_clear_of_the_cut(tmp_path):
    heights = [0.0, 2.0, 4.0, 4.5, 9.0, 10.1]
    structure, sorted_heights = _stacked_structure(
        tmp_path, heights, ["Al"] * 4 + ["C", "O"]
    )
    selection = placement.identify_top_group(structure)

    clearances = [abs(z - selection.z_cutoff) for z in sorted_heights]
    assert min(clearances) >= 0.5 * selection.gap_size - 1e-9

    # and the split is exactly the one the gap describes
    for index, height in enumerate(sorted_heights):
        above = index in selection.molecule_indices
        assert above == (height > selection.z_cutoff)


def test_no_cut_is_more_robust_than_the_largest_gap(tmp_path):
    heights = [0.0, 2.0, 4.0, 4.5, 9.0, 10.1]
    structure, sorted_heights = _stacked_structure(
        tmp_path, heights, ["Al"] * 4 + ["C", "O"]
    )
    selection = placement.identify_top_group(structure)
    chosen = 0.5 * selection.gap_size

    values = np.array(sorted(sorted_heights))
    for candidate in np.linspace(values[0], values[-1], 2001):
        clearance = float(np.min(np.abs(values - candidate)))
        assert clearance <= chosen + 1e-9


def test_disjoint_species_split_beats_the_bare_gap(tmp_path):
    """A molecule of a species the substrate does not use is separated by species."""

    heights = [0.0, 2.0, 4.0, 5.2, 6.4]
    structure, _ = _stacked_structure(tmp_path, heights, ["Al"] * 3 + ["C", "O"])
    selection = placement.identify_top_group(structure, min_gap=1.0)
    symbols = expand_species(structure.species, structure.counts)
    molecule = {symbols[i] for i in selection.molecule_indices}
    substrate = {symbols[i] for i in selection.substrate_indices}
    assert molecule == {"C", "O"}
    assert substrate == {"Al"}


def test_a_structure_without_a_clean_gap_is_refused(tmp_path):
    heights = [0.0, 0.5, 1.0, 1.5]
    structure, _ = _stacked_structure(tmp_path, heights, ["Al"] * 4)
    with pytest.raises(ValueError, match="largest internal z gap"):
        placement.identify_top_group(structure, min_gap=1.0)


# --------------------------------------------------------------------------
# The periodic branch cut
# --------------------------------------------------------------------------


def _span(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _unwrap_to(values: np.ndarray, start: float) -> np.ndarray:
    wrapped = np.mod(values, 1.0)
    return np.where(wrapped < start, wrapped + 1.0, wrapped)


@pytest.mark.parametrize(
    "values",
    [
        [0.02, 0.05, 0.97, 0.93],
        [0.4, 0.45, 0.5],
        [0.0, 0.5],
        [0.99, 0.01, 0.5],
        [0.125],
    ],
)
def test_unwrap_moves_each_atom_by_a_whole_cell(values):
    array = np.array(values, dtype=float)
    unwrapped, start = placement._unwrap_periodic_axis_with_start(array)
    shifts = unwrapped - np.mod(array, 1.0)
    assert np.allclose(shifts, np.round(shifts))
    assert set(np.round(shifts).astype(int)).issubset({0, 1})
    assert 0.0 <= start < 1.0 or array.size <= 1


@pytest.mark.parametrize(
    "values",
    [
        [0.02, 0.05, 0.97, 0.93],
        [0.4, 0.45, 0.5],
        [0.0, 0.5],
        [0.99, 0.01, 0.5],
        [0.3, 0.31, 0.32, 0.7, 0.71],
    ],
)
def test_unwrap_lands_in_one_period_and_minimises_the_span(values):
    array = np.array(values, dtype=float)
    unwrapped, start = placement._unwrap_periodic_axis_with_start(array)
    assert np.all(unwrapped >= start - 1e-12)
    assert np.all(unwrapped < start + 1.0 + 1e-12)
    span = _span(unwrapped)
    assert span < 1.0

    # no branch cut anywhere on the circle gives a smaller span
    for candidate in np.linspace(0.0, 1.0, 2001, endpoint=False):
        assert span <= _span(_unwrap_to(array, candidate)) + 1e-12


@pytest.mark.parametrize(
    "values",
    [
        [0.02, 0.05, 0.97, 0.93],
        [0.99, 0.01, 0.5],
        [0.3, 0.31, 0.32, 0.7, 0.71],
    ],
)
def test_span_is_one_minus_the_empty_arc(values):
    array = np.array(values, dtype=float)
    unwrapped, start = placement._unwrap_periodic_axis_with_start(array)
    empty_arc = start + 1.0 - float(np.max(unwrapped))
    assert _span(unwrapped) == pytest.approx(1.0 - empty_arc)

    # nothing lies inside the empty arc
    wrapped = np.mod(array, 1.0)
    for value in wrapped:
        image = value + 1.0 if value < start else value
        assert image <= float(np.max(unwrapped)) + 1e-12


def test_reframing_puts_a_straddling_molecule_inside_the_cell():
    direct = np.array(
        [
            [0.1, 0.1, 0.1],  # substrate
            [0.6, 0.6, 0.1],  # substrate
            [0.97, 0.02, 0.8],  # molecule, straddling the a and b boundaries
            [0.02, 0.98, 0.8],
            [0.99, 0.99, 0.85],
        ]
    )
    reframed, shift = placement._reframe_direct_positions(direct, [2, 3, 4], (0, 1))

    molecule = reframed[[2, 3, 4]]
    assert np.all(molecule[:, :2] > 0.0)
    assert np.all(molecule[:, :2] < 1.0)
    # the molecule is one connected piece: its in-plane span is small
    assert _span(molecule[:, 0]) < 0.2
    assert _span(molecule[:, 1]) < 0.2
    # and it is centred on the middle of the cell
    for axis in (0, 1):
        middle = 0.5 * (float(np.max(molecule[:, axis])) + float(np.min(molecule[:, axis])))
        assert middle == pytest.approx(0.5)

    # every atom moved by a whole lattice vector along the reframed axes
    for axis in (0, 1):
        delta = reframed[:, axis] - (direct[:, axis] - shift[axis])
        assert np.allclose(delta, np.round(delta))


def test_a_molecule_wider_than_the_cell_is_refused():
    """Extended coordinates that leave the cell are the only way to be too wide.

    `Cellstine.arcSpan_lt_one` says that unwrapping coordinates already inside
    `[0, 1)` always produces a span below one, so a molecule can only be
    reported as too wide when the caller hands over *extended* coordinates that
    genuinely run past the cell -- which is what happens when a molecule is
    read in its own box and mapped into a smaller substrate cell.  Wrapped
    coordinates carry no such information, and the implementation keeps them
    as they are instead of inventing a span.
    """

    direct = np.array(
        [
            [0.0, 0.5, 0.1],
            [1.2, 0.5, 0.8],
            [2.5, 0.5, 0.8],
        ]
    )
    with pytest.raises(ValueError, match="cannot be contained in one periodic image"):
        placement._reframe_direct_positions(direct, [0, 1, 2], (0,))


def test_unwrap_keeps_extended_coordinates_that_run_past_the_cell():
    values = np.array([0.0, 1.2, 2.5])
    unwrapped, start = placement._unwrap_periodic_axis_with_start(values)
    assert start == 0.0
    assert np.allclose(unwrapped, values)


# --------------------------------------------------------------------------
# The in-plane fit
# --------------------------------------------------------------------------


def _repeat_inputs(tmp_path, molecule_positions, cell=8.0):
    substrate = write_poscar(
        tmp_path / "sub.vasp",
        np.diag([cell, cell, 20.0]),
        ["Al"],
        [1],
        np.array([[0.0, 0.0, 0.1]]),
    )
    molecule = write_poscar(
        tmp_path / "mol.vasp",
        np.diag([30.0, 30.0, 30.0]),
        ["C"],
        [len(molecule_positions)],
        np.asarray(molecule_positions, dtype=float) / 30.0,
    )
    return io_mod.read_poscar(str(substrate)), io_mod.read_poscar(str(molecule))


@pytest.mark.parametrize("padding", [0.0, 0.25, 0.75])
def test_repeats_leave_the_requested_padding(tmp_path, padding):
    positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    substrate, molecule = _repeat_inputs(tmp_path, positions)
    repeat_a, repeat_b = placement._estimate_inplane_repeats_for_molecule(
        substrate, molecule, rotation_deg=0.0, fit_padding=padding
    )

    span_a = 10.0 / 8.0
    span_b = 4.0 / 8.0
    assert repeat_a - span_a >= 2.0 * padding - 1e-9
    assert repeat_b - span_b >= 2.0 * padding - 1e-9
    # and not more than one repeat beyond what the span and the padding force
    assert repeat_a == 1 or repeat_a < span_a + 2.0 * padding + 1.0
    assert repeat_b == 1 or repeat_b < span_b + 2.0 * padding + 1.0


def test_repeats_follow_the_rotated_footprint(tmp_path):
    """Turning a long molecule by 90 degrees swaps which axis needs the repeats."""

    positions = [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
    substrate, molecule = _repeat_inputs(tmp_path, positions)
    flat = placement._estimate_inplane_repeats_for_molecule(
        substrate, molecule, rotation_deg=0.0, fit_padding=0.0
    )
    turned = placement._estimate_inplane_repeats_for_molecule(
        substrate, molecule, rotation_deg=90.0, fit_padding=0.0
    )
    assert flat == (2, 1)
    assert turned == (1, 2)


def test_repeats_are_at_least_one(tmp_path):
    substrate, molecule = _repeat_inputs(tmp_path, [[0.0, 0.0, 0.0]])
    repeats = placement._estimate_inplane_repeats_for_molecule(
        substrate, molecule, rotation_deg=0.0, fit_padding=0.0
    )
    assert repeats == (1, 1)

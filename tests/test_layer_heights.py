"""Layers are separated by height along the surface normal, not by ``z``.

A stack is a pile of planes parallel to ``a`` and ``b``, so the coordinate that
tells one layer from the next is the projection onto the surface normal.  For
the orientation every CELLSTINE stage writes -- ``a`` along ``x``, ``b`` in the
``xy`` plane -- that projection *is* the Cartesian ``z``, which is why reading
``z`` directly used to look right.  It stops being right in two ways, and both
are checked here:

* Rigidly rotate a slab, cell and atoms together, and its ``z`` coordinates are
  scrambled while the structure is unchanged.  The upper layer must still be
  the same set of atoms.
* Swap ``a`` and ``b``.  That is a relabelling of the basis, not a change of
  structure, but it makes the cell left-handed, so ``a x b`` points downwards
  and a slab read along it comes out upside down.  Orienting the normal along
  ``+c`` -- from the bottom of the cell to its top, which is what a POSCAR
  means -- keeps the reading the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.layers import identify_top_layer, shift_top_layer
from cellstine.core.transforms import rotation_matrix_x, rotation_matrix_z
from cellstine.core.vacuum import normal_heights, surface_normal
from cellstine.interface.surface import backend as surface_mod
from cellstine.interface.surface.stacking import analyse_stacking, group_layers
from cellstine.io import native as io_mod

LATTICE_CONSTANT = 2.55
VACUUM_HEIGHT = 30.0
#: The interlayer spacing of a close-packed stack of touching spheres.
INTERLAYER = LATTICE_CONSTANT * np.sqrt(2.0 / 3.0)


def _record(lattice: np.ndarray, direct: np.ndarray, species=("Cu",), counts=None):
    direct = np.asarray(direct, dtype=float)
    return io_mod.PoscarData(
        comment="test slab",
        lattice=np.asarray(lattice, dtype=float),
        species=list(species),
        counts=list(counts) if counts is not None else [len(direct)],
        positions_direct=direct,
        positions_cartesian=direct @ np.asarray(lattice, dtype=float),
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
    )


def _close_packed_slab(layers: int = 6, *, reversed_sense: bool = False):
    """An fcc (111) slab: one atom per layer, offset by a hollow vector each time."""

    lattice = np.array(
        [
            [LATTICE_CONSTANT, 0.0, 0.0],
            [LATTICE_CONSTANT / 2.0, LATTICE_CONSTANT * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, VACUUM_HEIGHT],
        ]
    )
    step = 2.0 / 3.0 if reversed_sense else 1.0 / 3.0
    direct = np.array(
        [
            [(index * step) % 1.0, (index * step) % 1.0, (2.0 + index * INTERLAYER) / VACUUM_HEIGHT]
            for index in range(layers)
        ]
    )
    return _record(lattice, direct)


def _swap_a_and_b(record):
    """Rewrite a structure in the basis ``(b, a, c)``: the same atoms, left-handed."""

    permutation = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    lattice = permutation @ np.asarray(record.lattice, dtype=float)
    direct = np.asarray(record.positions_direct, dtype=float) @ np.linalg.inv(permutation)
    swapped = _record(lattice, direct, record.species, record.counts)
    assert np.linalg.det(lattice) < 0.0
    assert np.allclose(
        np.sort(swapped.positions_cartesian, axis=0),
        np.sort(np.asarray(record.positions_cartesian, dtype=float), axis=0),
    )
    return swapped


def _rotate(record, matrix):
    lattice = np.asarray(record.lattice, dtype=float) @ matrix.T
    return _record(lattice, record.positions_direct, record.species, record.counts)


def test_the_normal_of_a_left_handed_cell_still_points_from_the_bottom_up():
    slab = _close_packed_slab()
    swapped = _swap_a_and_b(slab)
    assert float(surface_normal(slab.lattice) @ surface_normal(swapped.lattice)) == pytest.approx(1.0)
    assert np.allclose(
        np.sort(normal_heights(slab.lattice, slab.positions_cartesian)),
        np.sort(normal_heights(swapped.lattice, swapped.positions_cartesian)),
    )


def test_a_left_handed_cell_does_not_read_its_layers_upside_down():
    slab = _close_packed_slab()
    swapped = _swap_a_and_b(slab)
    heights = [round(height, 6) for height, _ in group_layers(slab)]
    swapped_heights = [round(height, 6) for height, _ in group_layers(swapped)]
    assert heights == swapped_heights
    assert heights == sorted(heights)
    assert heights[0] == pytest.approx(2.0)


def test_a_left_handed_cell_reads_the_same_stacking_sequence():
    slab = _close_packed_slab()
    swapped = _swap_a_and_b(slab)
    assert surface_mod.stacking_sequence(slab) == surface_mod.stacking_sequence(swapped)
    assert surface_mod.stacking_sequence(slab)[0] == "ABCABC"


def test_a_left_handed_cell_keeps_the_stacking_sense_of_its_twin():
    """The gauge is fixed by one slab; the other must be read in the same frame."""

    slab = _close_packed_slab()
    twin = _close_packed_slab(reversed_sense=True)
    gauge = analyse_stacking(slab).hollow_cartesian

    same = analyse_stacking(_swap_a_and_b(slab), hollow_cartesian=gauge)
    other = analyse_stacking(_swap_a_and_b(twin), hollow_cartesian=gauge)
    assert same.close_packed and other.close_packed
    assert same.sense == analyse_stacking(slab, hollow_cartesian=gauge).sense == 1
    assert other.sense == analyse_stacking(twin, hollow_cartesian=gauge).sense == -1


@pytest.mark.parametrize("matrix", [rotation_matrix_x(35.0), rotation_matrix_z(20.0) @ rotation_matrix_x(-70.0)])
def test_the_upper_layer_of_a_rotated_slab_holds_the_same_atoms(matrix):
    """A rigid rotation moves no atom relative to another, so the split cannot move."""

    bilayer = _record(
        np.diag([3.0, 3.0, 24.0]),
        np.array([[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.3], [0.5, 0.5, 0.3]]),
    )
    upright = identify_top_layer(bilayer)
    rotated = identify_top_layer(_rotate(bilayer, matrix))
    assert upright.top_indices == (2, 3)
    assert rotated.top_indices == upright.top_indices
    assert rotated.bottom_indices == upright.bottom_indices
    assert rotated.gap_size == pytest.approx(upright.gap_size)
    assert rotated.z_cutoff == pytest.approx(upright.z_cutoff)


def test_shifting_the_upper_layer_of_a_rotated_slab_moves_the_same_atoms(tmp_path):
    bilayer = _record(
        np.diag([3.0, 3.0, 24.0]),
        np.array([[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.3], [0.5, 0.5, 0.3]]),
    )
    tilted = _rotate(bilayer, rotation_matrix_x(35.0))
    source = tmp_path / "tilted.vasp"
    io_mod.write_poscar(
        str(source),
        tilted.lattice,
        tilted.positions_direct,
        tilted.counts,
        tilted.species,
        comment="tilted bilayer",
        positions_are_cartesian=False,
        wrap_positions=False,
    )
    run = shift_top_layer(
        str(source), output_path=str(tmp_path / "shifted.vasp"), shift_direct=[0.25, 0.0]
    )
    assert run.top_atom_count == 2
    assert run.bottom_atom_count == 2
    shifted = io_mod.read_poscar(str(run.output_path))
    difference = np.asarray(shifted.positions_direct) - np.asarray(tilted.positions_direct)
    assert np.allclose(difference[:2], 0.0)
    assert np.allclose(difference[2:], [0.25, 0.0, 0.0])

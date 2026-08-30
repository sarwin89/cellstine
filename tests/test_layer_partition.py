"""One rule for what an atomic plane is, and the invariances it owes the user.

Four places in the package used to cut a structure into layers, and they used
three different rules: compare each atom with the previous one (a gap cut),
with the first member of the group it is joining, or with that group's running
mean.  The three agree on a clean slab and disagree exactly where a user would
be least able to tell -- on heights spaced near the tolerance -- and two of them
are not symmetric under reading the structure from the other end, so the same
slab could be reported with a different number of layers, different
terminations, and a spurious dipole warning depending on which way round it was
written.

Everything now goes through ``core.layers.layer_partition``, the gap cut, which
is single linkage.  ``RequestProject/LayerPartition.lean`` proves the
properties this file checks on the implementation:
``Cellstine.linked_iff_smallGaps`` (the sweep is the connected-component
partition), ``Cellstine.layerIndex_mono`` (plane 1 is the bottom),
``Cellstine.linked_add_const`` and ``Cellstine.linked_neg`` (the origin and the
direction of reading change no layer) and
``Cellstine.layerIndex_reverse_add`` (reading the other way exactly reverses
the numbering).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.layers import layer_partition
from cellstine.core.vacuum import normal_heights
from cellstine.defect.analysis import _cluster_projection_layers
from cellstine.interface.surface.stacking import group_layers
from cellstine.interface.surface.surface_sites import _cluster_projection_levels
from cellstine.interface.surface.termination import layer_species, termination_report
from cellstine.io.models import StructureRecord

TOLERANCE = 0.35


def _components(heights, tolerance):
    """Connected components of "within the tolerance", found by brute force."""

    count = len(heights)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, j in itertools.combinations(range(count), 2):
        if abs(heights[i] - heights[j]) <= tolerance:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: min(heights[index] for index in members),
    )


def _record(lattice, positions_cartesian, species, counts):
    lattice = np.asarray(lattice, dtype=float)
    cartesian = np.asarray(positions_cartesian, dtype=float)
    direct = cartesian @ np.linalg.inv(lattice)
    return StructureRecord(
        comment="layers",
        lattice=lattice,
        species=list(species),
        counts=[int(value) for value in counts],
        positions_direct=direct,
        positions_cartesian=cartesian,
    )


def _height_sets(heights, tolerance):
    return [sorted(indices) for _, indices in layer_partition(heights, tolerance)]


@pytest.mark.parametrize(
    "heights",
    [
        [],
        [0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.34, 0.50],
        [0.0, 0.3, 0.6, 0.9],
        [0.0, 0.36, 0.72],
        [2.0, 0.0, 1.0, 0.35, 3.0],
        [-1.2, -1.1, 4.4, 4.5, 4.85, 9.0],
    ],
)
def test_the_partition_is_the_connected_components(heights):
    assert _height_sets(heights, TOLERANCE) == _components(list(heights), TOLERANCE)


def test_the_partition_is_the_connected_components_on_random_heights():
    generator = np.random.default_rng(20240817)
    for _ in range(200):
        heights = np.round(generator.uniform(-5.0, 5.0, size=int(generator.integers(2, 12))), 2)
        assert _height_sets(heights, TOLERANCE) == _components(list(heights), TOLERANCE)


def test_the_planes_are_ordered_from_the_bottom_up():
    generator = np.random.default_rng(11)
    for _ in range(50):
        heights = generator.uniform(0.0, 8.0, size=10)
        planes = layer_partition(heights, TOLERANCE)
        means = [mean for mean, _ in planes]
        assert means == sorted(means)
        for mean, indices in planes:
            in_plane = [float(heights[index]) for index in indices]
            assert in_plane == sorted(in_plane)
            assert mean == pytest.approx(float(np.mean(in_plane)))
        assert sorted(index for _, indices in planes for index in indices) == list(range(10))


def test_moving_the_origin_changes_no_layer():
    generator = np.random.default_rng(7)
    heights = generator.uniform(-3.0, 3.0, size=15)
    reference = _height_sets(heights, TOLERANCE)
    for shift in (-12.5, -1.0, 0.0, 0.75, 40.0):
        assert _height_sets(heights + shift, TOLERANCE) == reference


def test_reading_the_structure_the_other_way_reverses_the_numbering():
    """`Cellstine.linked_neg` and `Cellstine.layerIndex_reverse_add` in Python."""

    generator = np.random.default_rng(99)
    for _ in range(100):
        heights = np.round(generator.uniform(-4.0, 4.0, size=int(generator.integers(2, 14))), 2)
        upwards = _height_sets(heights, TOLERANCE)
        downwards = _height_sets(-heights, TOLERANCE)
        assert downwards == upwards[::-1]
        count = len(upwards)
        for position, indices in enumerate(upwards, start=1):
            mirrored = downwards.index(indices) + 1
            assert position + mirrored == count + 1


def test_the_marginal_case_that_used_to_flip():
    """0.00, 0.34, 0.50 with a 0.35 tolerance: one plane, read from either end."""

    assert _height_sets([0.0, 0.34, 0.50], TOLERANCE) == [[0, 1, 2]]
    assert _height_sets([0.50, 0.16, 0.0], TOLERANCE) == [[0, 1, 2]]
    assert len(layer_partition([0.0, 0.34, 0.50], TOLERANCE)) == 1
    assert len(layer_partition([0.50, 0.16, 0.0], TOLERANCE)) == 1


def test_a_negative_tolerance_leaves_every_atom_alone():
    assert _height_sets([0.0, 0.1, 0.2], -1.0) == [[0], [1], [2]]
    assert _height_sets([0.0, 0.0, 1.0], -1.0) == [[0], [1], [2]]
    assert _height_sets([0.0, 0.0, 1.0], 0.0) == [[0, 1], [2]]


def _marginal_slab():
    """A three-plane slab whose spacings straddle the layer tolerance."""

    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 30.0]])
    heights = [5.0, 5.34, 5.50, 8.0, 8.2, 11.0]
    positions = [[0.3 * index, 0.1 * index, height] for index, height in enumerate(heights)]
    return lattice, np.asarray(positions, dtype=float)


def _flip(lattice, positions):
    """The same slab written upside down: heights negated and put back in the cell."""

    height = float(lattice[2, 2])
    flipped = np.array(positions, dtype=float, copy=True)
    flipped[:, 2] = height - flipped[:, 2]
    order = np.argsort(flipped[:, 2], kind="stable")
    return flipped[order], order


def test_the_termination_report_reads_a_slab_the_same_way_from_either_end():
    lattice, positions = _marginal_slab()
    labels = ["Na", "Cl", "Na", "Cl", "Na", "Cl"]
    upright = layer_species(lattice, positions, labels)
    flipped_positions, order = _flip(lattice, positions)
    flipped = layer_species(lattice, flipped_positions, [labels[index] for index in order])

    assert len(upright) == len(flipped)
    assert [counts for _, counts in upright] == [counts for _, counts in flipped][::-1]

    def report(pos, lab):
        species = sorted(set(lab))
        order_by_species = [index for symbol in species for index, value in enumerate(lab) if value == symbol]
        return termination_report(
            bulk_species=["Na", "Cl"],
            bulk_counts=[1, 1],
            slab_lattice=lattice,
            slab_positions_cartesian=np.asarray(pos, dtype=float)[order_by_species],
            slab_species=species,
            slab_counts=[sum(1 for value in lab if value == symbol) for symbol in species],
        )

    up = report(positions, labels)
    down = report(flipped_positions, [labels[index] for index in order])
    assert up.layer_count == down.layer_count
    assert up.bottom_termination == down.top_termination
    assert up.top_termination == down.bottom_termination
    assert up.symmetric_terminations == down.symmetric_terminations
    assert len(up.notes) == len(down.notes)


def test_every_consumer_cuts_the_same_layers():
    lattice, positions = _marginal_slab()
    labels = ["Na", "Cl", "Na", "Cl", "Na", "Cl"]
    record = _record(lattice, positions, ["Na", "Cl"], [3, 3])
    heights = normal_heights(lattice, positions)

    shared = _height_sets(heights, TOLERANCE)
    assert [sorted(indices) for _, indices in group_layers(record, TOLERANCE)] == shared
    assert [sorted(indices) for _, indices in _cluster_projection_levels(heights, TOLERANCE)] == shared
    assert [
        [index - 1 for index in layer["atom_indices"]]
        for layer in _cluster_projection_layers(np.asarray(heights, dtype=float), TOLERANCE)
    ] == shared

    counts_per_layer = [sum(counts.values()) for _, counts in layer_species(lattice, positions, labels)]
    assert counts_per_layer == [len(indices) for indices in shared]


def test_the_layer_census_is_numbered_from_the_bottom_and_covers_every_atom():
    heights = np.array([0.0, 0.1, 3.0, 3.2, 3.3, 6.0])
    layers = _cluster_projection_layers(heights, TOLERANCE)
    assert [layer["layer_id"] for layer in layers] == [1, 2, 3]
    assert [layer["atom_count"] for layer in layers] == [2, 3, 1]
    assert [layer["projection"] for layer in layers] == sorted(layer["projection"] for layer in layers)
    assert sorted(index for layer in layers for index in layer["atom_indices"]) == list(range(1, 7))
    assert layers[0]["projection"] == pytest.approx(0.05)


def test_a_close_packed_slab_is_still_cut_into_its_planes():
    """The shared rule must not change the layers of an ordinary slab."""

    spacing = 2.05
    lattice = np.array([[2.5, 0.0, 0.0], [1.25, 1.25 * math.sqrt(3.0), 0.0], [0.0, 0.0, 24.0]])
    positions = []
    for index in range(6):
        positions.append([0.4 * (index % 3), 0.2 * (index % 3), 2.0 + index * spacing])
    record = _record(lattice, positions, ["Cu"], [6])
    planes = group_layers(record, TOLERANCE)
    assert [len(indices) for _, indices in planes] == [1] * 6
    assert [round(height, 6) for height, _ in planes] == sorted(round(height, 6) for height, _ in planes)

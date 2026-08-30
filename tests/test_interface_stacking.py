"""Stacking-sequence detection, reversal, and interface registry enumeration."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.interface.surface import registry, stacking
from cellstine.io.converters import StructureConverter

from conftest import hexagonal_basis, write_poscar


def close_packed_slab(
    path,
    *,
    constant: float = 2.5,
    layers: int = 6,
    sense: int = 1,
    vacuum: float = 15.0,
    species: str = "Ni",
    start: int = 0,
):
    """Write an fcc(111)-like slab with a controlled ABC/CBA sequence."""

    spacing = constant * math.sqrt(2.0 / 3.0)
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(constant).T
    lattice[2, 2] = spacing * (layers - 1) + vacuum
    positions = []
    for index in range(layers):
        coset = (start + sense * index) % 3
        # (1/3, 2/3) is the hollow vector of the 120 degree hexagonal cell.
        positions.append(
            [
                (coset / 3.0) % 1.0,
                (2.0 * coset / 3.0) % 1.0,
                (0.5 * vacuum + spacing * index) / lattice[2, 2],
            ]
        )
    return write_poscar(path, lattice, [species], [layers], np.array(positions))


@pytest.fixture()
def converter():
    return StructureConverter()


def test_forward_slab_reads_as_abc(tmp_path, converter):
    path = close_packed_slab(tmp_path / "abc.vasp", layers=6, sense=1)
    analysis = stacking.analyse_stacking(converter.read(str(path)))
    assert analysis.close_packed
    assert analysis.sequence == "ABCABC"
    assert analysis.sense == 1
    assert analysis.sense_label == "ABC"
    assert analysis.increments == (1, 1, 1, 1, 1)


def test_a_slab_alone_has_no_handedness(tmp_path, converter):
    """Both senses read ABCABC on their own: handedness is relative."""

    forward = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "abc.vasp", layers=6, sense=1)))
    )
    backward = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "cba.vasp", layers=6, sense=-1)))
    )
    assert forward.sequence == backward.sequence == "ABCABC"
    assert np.allclose(
        np.asarray(forward.hollow_cartesian), -np.asarray(backward.hollow_cartesian)
    )


def test_a_reversed_slab_reads_as_cba_in_the_gauge_of_a_forward_slab(tmp_path, converter):
    reference = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "abc.vasp", layers=6, sense=1)))
    )
    analysis = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "cba.vasp", layers=6, sense=-1))),
        hollow_cartesian=reference.hollow_cartesian,
    )
    assert analysis.close_packed
    assert analysis.sequence == "ACBACB"
    assert analysis.sense == -1
    assert analysis.sense_label == "CBA"


def test_labels_are_relative_so_the_origin_does_not_matter(tmp_path, converter):
    first = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "start0.vasp", start=0)))
    )
    second = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "start1.vasp", start=1)))
    )
    assert first.sequence == second.sequence
    assert first.increments == second.increments


def test_mirror_reverses_the_sequence_and_is_an_isometry(tmp_path, converter):
    record = converter.read(str(close_packed_slab(tmp_path / "abc.vasp", layers=6, sense=1)))
    reference = stacking.analyse_stacking(record)
    mirrored = stacking.mirror_structure(record)
    analysis = stacking.analyse_stacking(mirrored, hollow_cartesian=reference.hollow_cartesian)
    assert analysis.sense == -1
    assert analysis.increments == tuple((-value) % 3 for value in (1, 1, 1, 1, 1))
    original = np.sort(np.asarray(record.positions_cartesian, dtype=float)[:, 2])
    imaged = np.sort(np.asarray(mirrored.positions_cartesian, dtype=float)[:, 2])
    assert np.allclose(original, imaged)
    # A mirror preserves every interatomic distance.
    def distances(structure):
        points = np.asarray(structure.positions_cartesian, dtype=float)
        return np.sort(
            np.array([np.linalg.norm(points[i] - points[j]) for i, j in itertools.combinations(range(len(points)), 2)])
        )

    assert np.allclose(distances(record), distances(mirrored))


def test_mirror_is_an_involution(tmp_path, converter):
    record = converter.read(str(close_packed_slab(tmp_path / "abc.vasp")))
    twice = stacking.mirror_structure(stacking.mirror_structure(record))
    assert stacking.analyse_stacking(twice).sequence == stacking.analyse_stacking(record).sequence


def test_apply_relative_stacking_is_idempotent(tmp_path, converter):
    record = converter.read(str(close_packed_slab(tmp_path / "abc.vasp")))
    gauge = stacking.analyse_stacking(record).hollow_cartesian
    kept, analysis, mirrored = stacking.apply_relative_stacking(record, "abc", reference_hollow=gauge)
    assert not mirrored and analysis.sense == 1
    flipped, flipped_analysis, was_mirrored = stacking.apply_relative_stacking(
        kept, "cba", reference_hollow=gauge
    )
    assert was_mirrored and flipped_analysis.sense == -1
    again, again_analysis, again_mirrored = stacking.apply_relative_stacking(
        flipped, "cba", reference_hollow=gauge
    )
    assert not again_mirrored and again_analysis.sense == -1
    assert np.allclose(again.positions_direct, flipped.positions_direct)


def analysed_pair(tmp_path, converter, *, layers: int = 6):
    bottom = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "b.vasp", layers=layers)))
    )
    top = stacking.analyse_stacking(
        converter.read(str(close_packed_slab(tmp_path / "t.vasp", layers=layers))),
        hollow_cartesian=bottom.hollow_cartesian,
    )
    return bottom, top


def test_two_chiral_slabs_give_six_distinct_interfaces(tmp_path, converter):
    bottom, top = analysed_pair(tmp_path, converter)
    options = registry.enumerate_registry_options(bottom, top)
    assert len(options) == 6
    # Twelve labelled combinations, each canonical class represented once.
    keys = {
        registry.canonical_configuration(
            tuple((-value) % 3 if option.bottom_mirrored else value for value in bottom.increments),
            tuple((-value) % 3 if option.top_mirrored else value for value in top.increments),
            option.delta,
        )
        for option in options
    }
    assert len(keys) == 6
    assert all(option.equivalent_to is None for option in options)
    assert {option.contact for option in options} == {"C-C", "C-A", "C-B"}
    assert {option.kind for option in options} == {"eclipsed", "fcc_hollow", "hcp_hollow"}
    assert [option.bottom_mirrored for option in options] == [False] * 6


def test_mirror_pairs_can_be_kept(tmp_path, converter):
    bottom, top = analysed_pair(tmp_path, converter)
    options = registry.enumerate_registry_options(bottom, top, include_equivalent=True)
    assert len(options) == 12
    assert sum(1 for option in options if option.equivalent_to is not None) == 6


def test_canonical_configuration_removes_only_mirror_duplicates():
    for bottom, top, delta in itertools.product(
        [(1, 1), (2, 2)], [(1, 1), (2, 2)], range(3)
    ):
        key = registry.canonical_configuration(bottom, top, delta)
        assert key == registry.canonical_configuration(*registry.mirror_configuration(bottom, top, delta))
    classes = {
        registry.canonical_configuration(bottom, top, delta)
        for bottom, top, delta in itertools.product([(1, 1), (2, 2)], [(1, 1), (2, 2)], range(3))
    }
    assert len(classes) == 6


def test_monolayer_pair_has_two_distinct_contacts(tmp_path, converter):
    bottom, top = analysed_pair(tmp_path, converter, layers=1)
    assert bottom.increments == ()
    assert not bottom.reversible
    options = registry.enumerate_registry_options(bottom, top)
    assert len(options) == 2
    assert [option.delta for option in options] == [0, 1]


def test_selection_by_contact_and_kind(tmp_path, converter):
    bottom, top = analysed_pair(tmp_path, converter)
    options = registry.enumerate_registry_options(bottom, top)
    assert registry.select_registry_option(options, "C-A").delta == 1
    assert registry.select_registry_option(options, "fcc").kind == "fcc_hollow"
    assert registry.select_registry_option(options, 3).index == 3
    assert registry.select_registry_option(options, None) is None
    with pytest.raises(ValueError):
        registry.select_registry_option(options, "Z-A")


def test_registry_table_lists_every_option(tmp_path, converter):
    bottom, top = analysed_pair(tmp_path, converter)
    table = registry.format_registry_table(registry.enumerate_registry_options(bottom, top))
    assert table.count("\n") == 7
    assert "contact" in table


def test_a_honeycomb_layer_is_not_treated_as_three_cosets(converter, graphene_poscar):
    analysis = stacking.analyse_stacking(converter.read(str(graphene_poscar)))
    assert not analysis.close_packed
    assert analysis.reason
    with pytest.raises(ValueError):
        registry.enumerate_registry_options(analysis, analysis)


def test_a_supercell_slab_reads_the_same_sequence(tmp_path, converter):
    from cellstine.interface.surface import backend as surface_backend
    from cellstine.io import native as io_mod

    path = close_packed_slab(tmp_path / "abc.vasp", layers=4, sense=1)
    single = stacking.analyse_stacking(converter.read(str(path)))
    repeated = surface_backend.repeat_structure_inplane(io_mod.read_poscar(str(path)), 2, 2)
    record = converter.read(str(path))
    record.lattice = np.asarray(repeated.lattice, dtype=float)
    record.positions_direct = np.asarray(repeated.positions_direct, dtype=float)
    record.positions_cartesian = io_mod.direct_to_cartesian(record.positions_direct, record.lattice)
    record.species = list(repeated.species)
    record.counts = [int(value) for value in repeated.counts]
    supercell = stacking.analyse_stacking(record)
    assert supercell.close_packed
    assert supercell.sequence == single.sequence
    assert supercell.increments == single.increments


def hcp_slab(path, *, constant: float = 3.2, layers: int = 6, vacuum: float = 15.0):
    """Write an hcp(0001)-like ``ABABAB`` slab, which has no stacking sense."""

    spacing = constant * math.sqrt(8.0 / 3.0) / 2.0
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(constant).T
    lattice[2, 2] = spacing * (layers - 1) + vacuum
    positions = []
    for index in range(layers):
        coset = index % 2
        positions.append(
            [
                (coset / 3.0) % 1.0,
                (2.0 * coset / 3.0) % 1.0,
                (0.5 * vacuum + spacing * index) / lattice[2, 2],
            ]
        )
    return write_poscar(path, lattice, ["Mg"], [layers], np.array(positions))


def test_an_hcp_slab_is_close_packed_but_has_no_stacking_sense(tmp_path, converter):
    analysis = stacking.analyse_stacking(converter.read(str(hcp_slab(tmp_path / "hcp.vasp"))))
    assert analysis.close_packed
    assert analysis.sequence == "ABABAB"
    assert analysis.sense == 0
    assert analysis.sense_label == "mixed"
    assert analysis.increments == (1, 2, 1, 2, 1)


def test_an_hcp_slab_refuses_a_relative_stacking_sense(tmp_path, converter):
    record = converter.read(str(hcp_slab(tmp_path / "hcp.vasp")))
    gauge = stacking.analyse_stacking(record).hollow_cartesian
    for choice in ("abc", "cba"):
        with pytest.raises(ValueError):
            stacking.apply_relative_stacking(record, choice, reference_hollow=gauge)


def test_the_hollows_of_an_hcp_slab_are_named_from_its_outermost_step(tmp_path, converter):
    """``ABABAB`` has no global sense, but its surface still continues one way."""

    record = converter.read(str(hcp_slab(tmp_path / "hcp.vasp")))
    analysis = stacking.analyse_stacking(record)
    top = stacking.analyse_stacking(record, hollow_cartesian=analysis.hollow_cartesian)
    options = registry.enumerate_registry_options(
        analysis,
        top,
        slabs_interchangeable=registry.slabs_are_interchangeable(record, record, analysis, top),
    )
    assert len(options) == 5
    kinds = {option.delta: option.kind for option in options if not option.top_mirrored}
    assert kinds == {0: "eclipsed", 1: "fcc_hollow", 2: "hcp_hollow"}
    assert registry.registry_kind(1, 0) == "hollow_1"
    assert registry.registry_kind(1, 0, bottom_last_step=1) == "fcc_hollow"
    assert registry.registry_kind(2, 0, bottom_last_step=1) == "hcp_hollow"


def test_a_contact_is_named_by_its_difference_alone(tmp_path, converter):
    """A-A, B-B and C-C select the same option, and so do C-A, A-B and B-C."""

    bottom, top = analysed_pair(tmp_path, converter)
    options = registry.enumerate_registry_options(bottom, top)
    for letters in ("A-A", "B-B", "C-C"):
        assert registry.select_registry_option(options, letters).delta == 0
    for letters in ("C-A", "A-B", "B-C"):
        assert registry.select_registry_option(options, letters).delta == 1
    for letters in ("C-B", "A-C", "B-A"):
        assert registry.select_registry_option(options, letters).delta == 2

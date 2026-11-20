"""Realized stacking requests, aligned constraints and bounded slab grouping."""

import numpy as np
import pytest

from cellstine.interface.surface import stacking
from cellstine.interface.surface.surface_supercell import repeat_structure_inplane
from cellstine.interface.workflow import assembly
from cellstine.interface.workflow.interface import Interface
from cellstine.io.converters import StructureConverter
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


def _record(species, counts, direct, flags=None, cell=None):
    cell = np.diag([4., 5., 12.]) if cell is None else np.asarray(cell, dtype=float)
    points = np.asarray(direct, dtype=float).reshape(-1, 3)
    return StructureRecord(comment="assembly regression", lattice=cell, species=list(species),
        counts=list(counts), positions_direct=points, positions_cartesian=points @ cell,
        selective_dynamics=flags is not None, selective_flags=flags, metadata={"nested": [1]})


def _slab(angle=60, word="ABC", flagged=True):
    sign = 1. if angle == 60 else -1.
    cell = np.array([[1., 0., 0.], [sign / 2., np.sqrt(3.) / 2., 0.], [0., 0., 12.]])
    hollow = np.array([1. / 3., 1. / 3. if angle == 60 else 2. / 3.])
    direct = np.array([[*((ord(letter) - ord("A")) * hollow % 1.), .1 + .1 * index]
                       for index, letter in enumerate(word)])
    flags = [("F", "T", "F") if index % 2 == 0 else ("T", "F", "T")
             for index in range(len(word))] if flagged else None
    return _record(["Al"], [len(word)], direct, flags, cell)


@pytest.mark.parametrize("angle", [60, 120])
@pytest.mark.parametrize("word", ["ABC", "ACB"])
@pytest.mark.parametrize("choice,wanted", [("abc", 1), ("cba", -1)])
def test_relative_request_realizes_both_senses_in_fixed_gauge(angle, word, choice, wanted):
    reference = _slab(angle); source = _slab(angle, word)
    gauge = stacking.analyse_stacking(reference).hollow_cartesian
    incoming = stacking.analyse_stacking(source, hollow_cartesian=gauge)
    saved = source.positions_direct.copy()
    output, analysis, mirrored = stacking.apply_relative_stacking(source, choice, reference_hollow=gauge)
    assert analysis.sense == wanted and analysis.increments == (wanted % 3,) * 2
    assert mirrored is (incoming.sense != wanted)
    if incoming.sense == wanted:
        assert output is source
    assert np.array_equal(source.positions_direct, saved)
    assert output.species == source.species and output.counts == source.counts
    assert output.selective_flags == source.selective_flags


@pytest.mark.parametrize("word,expected", [("ABAB", (2, 1, 2)), ("ABCACB", (2, 2, 2, 1, 1))])
def test_mirror_reverses_every_increment_of_nonuniform_word(word, expected):
    source = _slab(word=word)
    gauge = stacking.analyse_stacking(_slab()).hollow_cartesian
    output, analysis, mirrored = stacking.apply_relative_stacking(source, "mirror", reference_hollow=gauge)
    assert mirrored and output is not source and analysis.increments == expected


def test_standalone_mirror_retains_own_gauge_analysis():
    source = _slab()
    gauge = stacking.analyse_stacking(source).hollow_cartesian
    output, analysis, mirrored = stacking.apply_relative_stacking(source, "mirror")
    assert mirrored and analysis.sense == 1
    assert stacking.analyse_stacking(output, hollow_cartesian=gauge).sense == -1


def test_nonprimitive_slab_reverses_in_primitive_layer_gauge():
    repeated = repeat_structure_inplane(_slab(), 2, 2)
    # Workflows convert the native repetition payload to their canonical record.
    source = _record(repeated.species, repeated.counts, repeated.positions_direct,
                     repeated.selective_flags, repeated.lattice)
    gauge = stacking.analyse_stacking(source).hollow_cartesian
    output, analysis, mirrored = stacking.apply_relative_stacking(source, "cba", reference_hollow=gauge)
    assert mirrored and analysis.increments == (2, 2) and analysis.sense == -1
    assert output.counts == [12] and output.selective_flags == source.selective_flags


def test_unavailable_cell_mirror_refuses_request():
    repeated = repeat_structure_inplane(_slab(), 2, 3)
    source = _record(repeated.species, repeated.counts, repeated.positions_direct,
                     repeated.selective_flags, repeated.lattice)
    gauge = stacking.analyse_stacking(source).hollow_cartesian
    with pytest.raises(ValueError, match="mirror|revers|reflect"):
        stacking.apply_relative_stacking(source, "cba", reference_hollow=gauge)


@pytest.mark.parametrize("bottom_choice,top_choice,wanted_bottom,wanted_top", [
    ("keep", "cba", 1, -1), ("mirror", "abc", -1, 1), ("mirror", "cba", -1, -1),
])
def test_assembly_summary_matches_realized_gauge(bottom_choice, top_choice, wanted_bottom, wanted_top):
    source = _slab(); gauge = stacking.analyse_stacking(source).hollow_cartesian
    bottom, top, summary = assembly.prepare_stacking(source, source,
        bottom_stacking=bottom_choice, top_stacking=top_choice, registry=None)
    assert stacking.analyse_stacking(bottom, hollow_cartesian=gauge).sense == wanted_bottom
    assert stacking.analyse_stacking(top, hollow_cartesian=gauge).sense == wanted_top
    assert summary["bottom_sense"] == ("ABC" if wanted_bottom == 1 else "CBA")
    assert summary["top_sense"] == ("ABC" if wanted_top == 1 else "CBA")


def test_matched_slab_copy_file_really_reverses_and_keeps_flags(tmp_path):
    source = _slab(); path = tmp_path / "source.vasp"
    VaspIO().write(source, str(path), positions_are_cartesian=False, wrap_positions=True)
    saved = path.read_bytes(); run = tmp_path / "run"; run.mkdir()
    workflow = Interface(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    read_source = StructureConverter().read(str(path))
    gauge = stacking.analyse_stacking(read_source).hollow_cartesian
    result = workflow._mirrored_slab_copy(path, run, "top")
    written = StructureConverter().read(str(result))
    assert stacking.analyse_stacking(written, hollow_cartesian=gauge).increments == (2, 2)
    assert written.selective_flags == source.selective_flags and path.read_bytes() == saved


def test_direct_interface_written_sense_and_mixed_flags_match_request(tmp_path):
    bottom, top = _slab(flagged=False), _slab(flagged=True)
    first, second = tmp_path / "bottom.vasp", tmp_path / "top.vasp"
    writer = VaspIO()
    for record, path in ((bottom, first), (top, second)):
        writer.write(record, str(path), positions_are_cartesian=False, wrap_positions=True)
    original = [first.read_bytes(), second.read_bytes()]
    workflow = Interface(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    destination = tmp_path / "interface.vasp"
    result = workflow.build(bottom_input=str(first), top_input=str(second), bottom_kind="slab",
        top_kind="slab", top_stacking="cba", registry="hcp", gap=2.4, vacuum=12., output_path=str(destination))
    output = StructureConverter().read(str(destination))
    assert result.summary["stacking"]["top_sense"] == "CBA"
    assert output.counts == [6] and output.selective_flags == [("T", "T", "T")] * 3 + top.selective_flags
    gauge = stacking.analyse_stacking(bottom).hollow_cartesian
    written_top = _record(["Al"], [3], output.positions_direct[3:], output.selective_flags[3:], output.lattice)
    assert stacking.analyse_stacking(written_top, hollow_cartesian=gauge).increments == (2, 2)
    assert [first.read_bytes(), second.read_bytes()] == original


@pytest.mark.parametrize("bottom_flagged", [False, True])
def test_mixed_selective_flags_cover_every_grouped_atom(bottom_flagged):
    marks = [("F", "T", "F"), ("T", "F", "T")]
    direct = [[.1, .2, .2], [.4, .5, .4]]
    bottom = _record(["Na", "Cl"], [1, 1], direct, marks if bottom_flagged else None)
    top = _record(["Cl", "O"], [1, 1], direct, None if bottom_flagged else marks)
    _, points, counts, species, flags = assembly.stack_structures(bottom, top, gap=3., vacuum=12.)
    expected = [marks[0], marks[1], ("T", "T", "T"), ("T", "T", "T")] if bottom_flagged else [
        ("T", "T", "T"), ("T", "T", "T"), marks[0], marks[1]]
    assert points.shape == (4, 3) and counts == [1, 2, 1] and species == ["Na", "Cl", "O"]
    assert flags == expected


def test_duplicate_and_zero_headers_keep_owned_group_order():
    points = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
    flags = [("F", "T", "F"), ("T", "F", "T"), ("F", "F", "T")]
    source = _record(["X", "Na", "Cl", "Na"], [0, 1, 1, 1], points, flags)
    output, counts, species, actual_flags = assembly.group_by_species(source, points, flags)
    assert counts == [2, 1] and species == ["Na", "Cl"]
    assert np.array_equal(output, points[[0, 2, 1]]) and actual_flags == [flags[0], flags[2], flags[1]]
    output[0] = 99.; actual_flags[0] = ("T", "T", "T")
    assert np.array_equal(source.positions_direct, points) and source.selective_flags == flags


@pytest.mark.parametrize("flags", [None, []])
def test_empty_grouping_keeps_legacy_shape_and_flags(flags):
    source = _record(["C"], [0], np.empty((0, 3)), flags)
    output, counts, species, actual_flags = assembly.group_by_species(source, source.positions_direct, flags)
    assert output.shape == (0,) and counts == species == [] and actual_flags == flags


@pytest.mark.parametrize("species,counts,points,flags", [
    (["C", "O"], [1], [[0., 0., 0.]], None),
    (["C"], [1], [[0., 0., 0.], [.5, 0., 0.]], None),
    (["C"], [-1], np.empty((0, 3)), None),
    (["C"], [2], [[0., 0., 0.], [.5, 0., 0.]], [("T", "T", "T")]),
    (["C"], [1], [[0., 0., 0.]], [("T", "T", "T")] * 2),
])
def test_grouping_rejects_header_or_flag_cardinality(species, counts, points, flags):
    source = _record(species, counts, points, flags)
    with pytest.raises(ValueError, match="species|count|position|flag"):
        assembly.group_by_species(source, source.positions_direct, flags)


def test_huge_inconsistent_header_refused_before_label_expansion(monkeypatch):
    source = _record(["C"], [10 ** 12], [[0., 0., 0.]])
    def forbidden(*args, **kwargs):
        pytest.fail("grouping expanded an impossible atom header before checking cardinality")
    monkeypatch.setattr(assembly, "expand_species", forbidden)
    with pytest.raises(ValueError, match="count|atom|position"):
        assembly.group_by_species(source, source.positions_direct, None)


@pytest.mark.parametrize("indices", [[0, 0, 0], (0, 0, 0), np.zeros(3, dtype=int)])
def test_zero_miller_sequence_rejected_like_string(indices):
    with pytest.raises(ValueError, match="zero"):
        assembly.parse_miller_notation(indices)

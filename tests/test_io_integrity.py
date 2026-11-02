"""External frame integrity and ownership of conversion intermediates."""

import sys
import tempfile
import tracemalloc
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from cellstine.io import native
from cellstine.io.converters import StructureConverter
from cellstine.io.models import StructureRecord
from cellstine.io.orientation import OrientationNormalizer


@pytest.fixture
def molecular_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source.mol"
    source.write_text("source data", encoding="utf-8")
    neighbour = source.with_suffix(".tmp.xyz")
    neighbour.write_text("user-owned neighbour", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    state = {"fail": False, "paths": []}

    class Structure:
        @staticmethod
        def from_file(path):
            raise ValueError("molecule input")

    class Molecule:
        @staticmethod
        def from_file(path):
            return Molecule()

        def to(self, *, fmt, filename):
            target = Path(filename)
            state["paths"].append(target)
            target.write_text("1\nmolecule\nC 0 0 0\n", encoding="utf-8")
            if state["fail"]:
                raise RuntimeError("external writer failed")

    package = ModuleType("pymatgen")
    package.__path__ = []
    core = ModuleType("pymatgen.core")
    core.Structure = Structure
    core.Molecule = Molecule
    monkeypatch.setitem(sys.modules, "pymatgen", package)
    monkeypatch.setitem(sys.modules, "pymatgen.core", core)
    converter = StructureConverter()
    monkeypatch.setattr(converter.dependency_manager, "has", lambda _: True)
    return converter, source, neighbour, scratch, state


def test_molecular_fallback_preserves_sibling_and_owns_its_temporary(molecular_fallback):
    converter, source, neighbour, scratch, state = molecular_fallback
    converter.read(str(source))
    assert neighbour.read_text(encoding="utf-8") == "user-owned neighbour"
    assert state["paths"][0].parent == scratch
    assert not state["paths"][0].exists()
    assert source.read_text(encoding="utf-8") == "source data"


def test_molecular_fallback_cleans_only_its_temporary_after_writer_failure(molecular_fallback):
    converter, source, neighbour, scratch, state = molecular_fallback
    state["fail"] = True
    with pytest.raises(RuntimeError, match="external writer failed"):
        converter.read(str(source))
    assert neighbour.read_text(encoding="utf-8") == "user-owned neighbour"
    assert state["paths"][0].parent == scratch
    assert not state["paths"][0].exists()


def test_molecular_fallback_honours_requested_vacuum_and_source_metadata(molecular_fallback):
    converter, source, _, _, _ = molecular_fallback
    record = converter.read(str(source), vacuum=7.)
    assert np.array_equal(record.lattice, np.eye(3) * 8.)
    assert record.source_path == str(source)
    assert record.source_format == "mol"
    assert record.metadata["vacuum_padding"] == 7.


@pytest.mark.parametrize("text", [
    "2\nshort\nC 0 0 0\n",
    "1\nshort row\nC 0 0\n",
    "1\nnonfinite\nC nan 0 0\n",
    "1\nnonfinite\nC 0 inf 0\n",
])
def test_xyz_rejects_incomplete_or_nonfinite_frames(tmp_path, text):
    path = tmp_path / "bad.xyz"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        StructureConverter().read(str(path))


def test_xyz_first_frame_does_not_allocate_for_the_trajectory_tail(tmp_path):
    path = tmp_path / "trajectory.xyz"
    path.write_text("1\nfirst\nC 0 0 0\n" + "1\nnext\nC 0 0 0\n" * 50000, encoding="utf-8")
    converter = StructureConverter()
    tracemalloc.start()
    try:
        record = converter.read(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert record.natoms == 1 and record.comment == "first"
    assert peak < 512 * 1024


def test_xyz_writer_does_not_expand_a_full_atom_label_list(tmp_path):
    count = 100000
    points = np.zeros((count, 3))
    record = StructureRecord("large", np.eye(3), ["C"], [count], points, points)
    converter = StructureConverter()
    tracemalloc.start()
    try:
        converter.write(record, str(tmp_path / "large.xyz"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 256 * 1024


@pytest.mark.parametrize("vacuum", [-1., float("nan"), float("inf")])
def test_xyz_vacuum_must_be_finite_nonnegative_padding(tmp_path, vacuum):
    path = tmp_path / "molecule.xyz"
    path.write_text("2\nmolecule\nC 0 0 0\nO 3 3 3\n")
    with pytest.raises(ValueError):
        StructureConverter().read(str(path), vacuum=vacuum)


@pytest.mark.parametrize("fault", ["counts", "species", "coordinate", "comment"])
def test_xyz_writer_rejects_malformed_records_before_replacing_data(tmp_path, fault):
    points = np.zeros((1, 3))
    record = StructureRecord("molecule", np.eye(3), ["C"], [1], points, points.copy())
    if fault == "counts":
        record.counts = [2]
    elif fault == "species":
        record.species = []
    elif fault == "coordinate":
        record.positions_cartesian[0, 0] = float("nan")
    else:
        record.comment = "first\nsecond"
    path = tmp_path / "existing.xyz"
    path.write_text("preserve")
    with pytest.raises(ValueError):
        StructureConverter().write(record, str(path))
    assert path.read_text() == "preserve"


def test_c_repeat_keeps_species_blocks_flags_and_original_geometry():
    points = np.array([[0., 0., .1], [.25, 0., .4]])
    flags = [("T", "F", "T"), ("F", "T", "F")]
    source = native.PoscarData("mixed", np.diag([4., 4., 10.]), ["C", "O"], [1, 1],
                               points, points @ np.diag([4., 4., 10.]), "Direct", True, flags)
    result = native.repeat_structure_along_c(source, 2)
    assert result.species == ["C", "O"] and result.counts == [2, 2]
    assert np.allclose(result.positions_direct, [[0., 0., .05], [0., 0., .55],
                                                [.25, 0., .2], [.25, 0., .7]], atol=1e-15)
    assert np.allclose(result.positions_cartesian[:, 2], [1., 11., 4., 14.], atol=1e-14)
    assert result.selective_flags == [flags[0], flags[0], flags[1], flags[1]]
    assert np.array_equal(source.positions_direct, points)
    assert source.counts == [1, 1]


@pytest.mark.parametrize("line", ["1 2", "1 1 1 1", "-1 1 1", "1 nan 1", "inf"])
def test_poscar_scaling_rejects_invalid_number_count_or_component_values(line):
    with pytest.raises(ValueError):
        native._apply_scaling_factor(np.eye(3), line)


def test_poscar_non_cartesian_mode_follows_vasp_direct_convention(tmp_path):
    path = tmp_path / "fractional.vasp"
    path.write_text("sample\n1\n4 0 0\n0 5 0\n0 0 6\nC\n1\nFractional\n.25 .5 .75\n")
    record = native.read_poscar(str(path))
    assert np.array_equal(record.positions_direct, [[.25, .5, .75]])
    assert np.array_equal(record.positions_cartesian, [[1., 2.5, 4.5]])


def test_poscar_incomplete_positions_raise_a_format_error(tmp_path):
    path = tmp_path / "short.vasp"
    path.write_text("sample\n1\n4 0 0\n0 4 0\n0 0 4\nC\n2\nDirect\n0 0 0\n")
    with pytest.raises(ValueError):
        native.read_poscar(str(path))


@pytest.mark.parametrize("flags", [("T", "F"), ("T", "BAD", "F"), ("", "F", "T")])
def test_poscar_writer_rejects_invalid_flags_before_replacing_existing_data(tmp_path, flags):
    path = tmp_path / "existing.vasp"
    path.write_text("preserve")
    with pytest.raises(ValueError):
        native.write_poscar(str(path), np.eye(3) * 4, np.zeros((1, 3)), [1], ["C"],
                            positions_are_cartesian=False, selective_flags=[flags])
    assert path.read_text() == "preserve"


def test_poscar_writer_preserves_supported_flag_case(tmp_path):
    path = tmp_path / "flags.vasp"
    flags = [("t", "F", "f")]
    native.write_poscar(str(path), np.eye(3) * 4, np.zeros((1, 3)), [1], ["C"],
                        positions_are_cartesian=False, selective_flags=flags)
    assert native.read_poscar(str(path)).selective_flags == flags


def test_raw_poscar_reader_keeps_permissive_labels_and_flags_for_inspection(tmp_path):
    path = tmp_path / "inspect.vasp"
    path.write_text("sample\n1\n4 0 0\n0 4 0\n0 0 4\nC O\n1\nSelective dynamics\nDirect\n0 0 0 T F\n")
    record = native.read_poscar(str(path))
    assert record.species == ["C", "O"] and record.counts == [1]
    assert record.selective_flags == [("T", "F")]


@pytest.mark.parametrize("method", ["align_ab_to_xy", "align_c_to_z"])
@pytest.mark.parametrize("scale", [1e100, 1e200])
def test_orientation_handles_large_finite_axes_without_changing_the_rotation(method, scale):
    lattice = np.array([[4., 1., 2.], [1., 5., .5], [.7, 1., 6.]])
    points = np.array([[.1, .2, .3], [.4, .5, .6]])
    reference = StructureRecord("base", lattice, ["C"], [2], points, points @ lattice)
    enlarged = StructureRecord("large", lattice * scale, ["C"], [2], points, points @ (lattice * scale))
    operation = getattr(OrientationNormalizer(), method)
    expected = operation(reference)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = operation(enlarged)
    assert np.allclose(actual.lattice / scale, expected.lattice, atol=1e-13, rtol=1e-13)
    assert np.allclose(actual.positions_cartesian / scale, expected.positions_cartesian, atol=1e-13, rtol=1e-13)
    assert np.allclose(actual.positions_direct, expected.positions_direct, atol=1e-13, rtol=0.)


def test_orientation_preserves_existing_absolute_zero_thresholds():
    points = np.zeros((1, 3))
    tiny_c = np.diag([1., 1., 1e-16])
    record = StructureRecord("tiny c", tiny_c, ["C"], [1], points, points.copy())
    unchanged = OrientationNormalizer().align_c_to_z(record)
    assert np.array_equal(unchanged.lattice, tiny_c)
    assert "aligned_c_to_z" not in unchanged.metadata
    thin_plane = np.diag([1e-7, 1e-7, 1.])
    record = StructureRecord("thin plane", thin_plane, ["C"], [1], points, points.copy())
    with pytest.raises(ValueError, match="span a plane"):
        OrientationNormalizer().align_ab_to_xy(record)

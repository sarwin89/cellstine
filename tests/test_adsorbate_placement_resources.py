"""Placement input contracts, stable gap selection and owned atom grouping."""

import json
from pathlib import Path

import numpy as np
import pytest

from cellstine.adsorbate.molecule import Molecule
from cellstine.adsorbate.placement import operations as placement
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


def _record(species, counts, cartesian, flags=None, cell=None):
    cell = np.diag([6., 6., 12.]) if cell is None else np.asarray(cell, dtype=float)
    points = np.asarray(cartesian, dtype=float).reshape(-1, 3)
    return StructureRecord(comment="placement resources", lattice=cell, species=list(species), counts=list(counts),
        positions_cartesian=points, positions_direct=points @ np.linalg.inv(cell),
        selective_dynamics=flags is not None, selective_flags=flags)


@pytest.mark.parametrize("axes,expected", [(["x", "y"], (0, 1)),
    (np.array(["y", "x", "y"]), (1, 0)), ([], ()), (" X, y ", (0, 1)),
    ("NONE", ()), (("z", "x", "z"), (2, 0))])
def test_axis_sequences_and_supported_strings_preserve_order(axes, expected):
    assert placement._normalise_reframe_axes(axes) == expected


@pytest.mark.parametrize("axes", ["oxygen", "xq", "x1", ["x", "q"]])
def test_invalid_axis_tokens_are_not_silently_filtered(axes):
    with pytest.raises(ValueError):
        placement._normalise_reframe_axes(axes)


@pytest.mark.parametrize("positions,species", [([[0., 0., 0.]], ["C", "H"]),
    ([[0., 0., 0.], [1., 0., 0.]], ["C"]), ([[0., 0.]], ["C"]),
    ([[float("nan"), 0., 0.]], ["C"])])
def test_center_of_mass_rejects_broadcasting_or_nonfinite_coordinates(positions, species):
    with pytest.raises(ValueError):
        placement.center_of_mass_cartesian(np.array(positions), species)


@pytest.mark.parametrize("angle", ["rotation_deg", "tilt_deg", "roll_deg"])
@pytest.mark.parametrize("estimate", [False, True])
def test_nan_rotation_is_refused_in_transform_and_fit(angle, estimate):
    source = _record(["C", "H"], [1, 1], [[0., 0., 0.], [1., 0., 0.]])
    angles = dict(rotation_deg=0., tilt_deg=0., roll_deg=0.)
    angles[angle] = float("nan")
    with pytest.raises(ValueError):
        if estimate:
            placement._estimate_inplane_repeats_for_molecule(source, source, fit_padding=.15, **angles)
        else:
            placement._transform_molecule_cartesian(source.positions_cartesian, ["C", "H"], source.lattice,
                target_cartesian=None, target_direct=None, **angles)


def test_first_disjoint_species_gap_precedes_larger_gap_and_keeps_original_indices():
    source = _record(["O", "Al", "C"], [2, 2, 1],
        [[0., 0., 3.], [0., 0., 4.5], [0., 0., 0.], [0., 0., 1.5], [0., 0., 20.]], cell=np.eye(3) * 40.)
    selection = placement.identify_top_group(source, min_gap=1.)
    assert selection.molecule_indices == (0, 1, 4)
    assert selection.substrate_indices == (2, 3)
    assert selection.z_cutoff == 2.25 and selection.gap_size == 1.5


def test_shared_species_keep_largest_gap_fallback():
    source = _record(["C"], [4], [[0., 0., 0.], [0., 0., 1.5], [0., 0., 3.], [0., 0., 20.]], cell=np.eye(3) * 40.)
    selection = placement.identify_top_group(source, min_gap=1.)
    assert selection.molecule_indices == (3,) and selection.z_cutoff == 11.5


def test_disjoint_gap_sweep_matches_scalar_sets_with_skipped_gaps_and_permutations():
    rng = np.random.default_rng(60619)
    heights = [np.array([0., 0., 1., 1., 4., 4.])]
    heights.extend(rng.permutation(np.arange(6, dtype=float)) for _ in range(20))
    labels = ["C", "C", "H", "H", "Al", "Al"]
    for z_values in heights:
        source = _record(["C", "H", "Al"], [2, 2, 2], np.column_stack((np.zeros((6, 2)), z_values)))
        order = np.argsort(z_values); sorted_z = z_values[order]; gaps = np.diff(sorted_z)
        expected = None
        for cut in range(1, 6):
            if gaps[cut - 1] < 1.5:
                continue
            low = {labels[index] for index in order[:cut]}
            high = {labels[index] for index in order[cut:]}
            if low.isdisjoint(high):
                expected = set(int(index) for index in order[:cut])
                break
        result = placement._species_disjoint_split(source, order, sorted_z, gaps, 1.5)
        if expected is None:
            assert result is None
        else:
            assert set(np.flatnonzero(result.substrate_mask)) == expected
            assert np.array_equal(result.molecule_mask, ~result.substrate_mask)


def test_combined_species_keep_zero_headers_duplicate_blocks_and_mixed_flags():
    marks = [("F", "T", "F"), ("T", "F", "T"), ("F", "F", "F")]
    substrate = _record(["Al", "Ne", "H", "Al"], [1, 0, 1, 1],
        [[0., 0., 2.], [1., 0., 3.], [2., 0., 4.]], marks)
    molecule = _record(["H", "He"], [1, 1], [[1., 2., 8.], [2., 2., 9.]])
    saved = substrate.positions_cartesian.copy()
    positions, counts, species, flags, shift, moved = placement._combine_substrate_and_molecule(
        substrate, molecule, molecule.positions_cartesian, reframe_axes=None)
    assert counts == [2, 0, 2, 1] and species == ["Al", "Ne", "H", "He"]
    assert flags == [marks[0], marks[2], marks[1], ("T", "T", "T"), ("T", "T", "T")]
    assert np.allclose(positions @ substrate.lattice, [[0., 0., 2.], [2., 0., 4.], [1., 0., 3.], [1., 2., 8.], [2., 2., 9.]])
    assert np.allclose(moved, [[1., 2., 8.], [2., 2., 9.]]) and np.array_equal(shift, np.zeros(3))
    assert np.array_equal(substrate.positions_cartesian, saved)


@pytest.fixture()
def molecular_slab(tmp_path):
    source = _record(["Al", "H"], [2, 2], [[.6, .6, 2.4], [1.8, 1.8, 3.6], [3., 3., 7.2], [3., 3., 8.4]])
    path = tmp_path / "molecular-slab.vasp"
    VaspIO().write(source, str(path), positions_are_cartesian=False, wrap_positions=True)
    return path


@pytest.mark.parametrize("kind,target,expected", [("cartesian", [2., 2., 9.], [2., 2., 9.]),
    ("direct", [.25, .35, .65], [1.5, 2.1, 7.8])])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_public_move_accepts_numpy_targets_and_writes_manifest(tmp_path, molecular_slab, kind, target, expected, dtype):
    tool = Molecule(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    requested = np.array(target, dtype=dtype)
    result = tool.move(poscar_path=str(molecular_slab), reframe_axes=None, preserve_vacuum=False,
        output_path=str(tmp_path / "moved.vasp"), **{"target_" + kind: requested})
    assert np.allclose(result.payload["center_of_mass_after"], expected)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["parameters"]["target_" + kind] == [float(value) for value in requested]


def test_public_placement_accepts_numpy_supercell_matrix(tmp_path, molecular_slab):
    molecule = _record(["He"], [1], [[0., 0., 0.]])
    path = tmp_path / "he.vasp"
    VaspIO().write(molecule, str(path), positions_are_cartesian=False, wrap_positions=True)
    result = Molecule(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output").place(
        substrate_poscar=str(molecular_slab), molecule_poscar=str(path), substrate_supercell_matrix=np.array([2, 0, 0, 1]),
        site_type="top", preserve_vacuum=False, output_path=str(tmp_path / "placed.vasp"))
    assert result.summary["substrate_atom_count"] == 8
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["parameters"]["substrate_supercell_matrix"] == [2, 0, 0, 1]


def test_bulk_placement_accepts_numpy_miller_indices(tmp_path):
    bulk = _record(["Al"], [4], np.array([[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]]) * 4.05,
                   cell=np.eye(3) * 4.05)
    molecule = _record(["He"], [1], [[0., 0., 0.]])
    first, second = tmp_path / "bulk.vasp", tmp_path / "he.vasp"
    for record, path in ((bulk, first), (molecule, second)):
        VaspIO().write(record, str(path), positions_are_cartesian=False, wrap_positions=True)
    result = Molecule(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output").place(
        substrate_poscar=str(first), substrate_kind="bulk", molecule_poscar=str(second),
        miller=np.array([1, 1, 1]), layers=3, vacuum=12., site_type="top", preserve_vacuum=False,
        output_path=str(tmp_path / "placed.vasp"))
    assert result.summary["substrate_atom_count"] == 3

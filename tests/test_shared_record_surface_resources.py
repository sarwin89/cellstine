"""Migration, structure-record and compound-surface ownership regressions."""

import tracemalloc

import numpy as np
import pytest

from cellstine.core import pathway
from cellstine.io.models import StructureRecord
from cellstine.interface.surface import surface_cell, surface_supercell, termination
from cellstine.symmetry import records


def _record(species, counts, positions, flags=None):
    cell = np.diag([5., 6., 12.])
    direct = np.asarray(positions, dtype=float).reshape(-1, 3)
    return StructureRecord(comment="owned regression", lattice=cell, species=list(species), counts=list(counts),
                           positions_direct=direct, positions_cartesian=direct @ cell,
                           selective_dynamics=flags is not None, selective_flags=flags,
                           metadata={"nested": ["preserved"]})


@pytest.mark.parametrize("repeats", [(2, 1), (2, 3)])
@pytest.mark.parametrize("headers", [(["Na", "Cl"], [2, 2]), (["Na", "Na", "Cl"], [1, 1, 2])])
def test_compound_repeat_keeps_source_header_blocks_cell_atom_order_and_flags(repeats, headers):
    species, counts = headers
    direct = np.array([[.1, .2, .15], [.2, .3, .35], [.3, .4, .55], [.4, .5, .75]])
    flags = [("F", "T", "F"), ("T", "F", "T"), ("F", "F", "T"), ("T", "T", "F")]
    source = _record(species, counts, direct, flags)
    a, b = repeats
    expected, expected_flags = [], []
    offset = 0
    for count in counts:
        for i in range(a):
            for j in range(b):
                block = direct[offset:offset + count].copy()
                block[:, 0] = (block[:, 0] + float(i)) / float(a)
                block[:, 1] = (block[:, 1] + float(j)) / float(b)
                expected.extend(block)
                expected_flags.extend(flags[offset:offset + count])
        offset += count
    result = surface_supercell.repeat_structure_inplane(source, a, b)
    assert result.species == species and result.counts == [n * a * b for n in counts]
    assert np.array_equal(result.positions_direct, np.asarray(expected))
    assert np.array_equal(result.positions_cartesian, result.positions_direct @ result.lattice)
    assert result.selective_flags == expected_flags
    assert np.array_equal(source.positions_direct, direct) and source.selective_flags == flags


def test_repeat_identity_and_empty_structure_keep_contract():
    source = _record(["C"], [0], np.empty((0, 3)), [])
    assert surface_supercell.repeat_structure_inplane(source, 1, 1) is source
    repeated = surface_supercell.repeat_structure_inplane(source, 2, 3)
    assert repeated.positions_direct.shape == repeated.positions_cartesian.shape == (0, 3)
    assert repeated.counts == [0] and repeated.species == ["C"] and repeated.selective_flags == []


def test_unaddressable_repeat_rejected_before_any_array_copy(monkeypatch):
    source = _record(["C"], [1], [[.2, .3, .5]])
    def forbidden(*args, **kwargs):
        pytest.fail("unaddressable repeat started copying or allocating")
    monkeypatch.setattr(np, "array", forbidden)
    with pytest.raises(ValueError, match="large|address"):
        surface_supercell.repeat_structure_inplane(source, 2 ** 62, 2)


@pytest.mark.parametrize("bulk_duplicate", [False, True])
@pytest.mark.parametrize("slab_duplicate", [False, True])
def test_termination_totals_aggregate_duplicate_species_headers(bulk_duplicate, slab_duplicate):
    def header(duplicate):
        return (["Na", "Na", "Cl"], [1, 1, 2]) if duplicate else (["Na", "Cl"], [2, 2])
    bulk_species, bulk_counts = header(bulk_duplicate)
    slab_species, slab_counts = header(slab_duplicate)
    result = termination.termination_report(bulk_species=bulk_species, bulk_counts=bulk_counts,
        slab_species=slab_species, slab_counts=slab_counts, slab_lattice=np.diag([5., 5., 12.]),
        slab_positions_cartesian=[[0., 0., 1.], [1., 0., 3.], [.5, 0., 1.], [1.5, 0., 3.]])
    assert result.bulk_formula == {"Na": 1, "Cl": 1}
    assert result.slab_counts == {"Na": 2, "Cl": 2}
    assert result.formula_units == 2. and result.stoichiometric and result.excess == {}
    assert result.symmetric_terminations and result.notes == []


def test_migration_accepts_numpy_endpoint_counts_without_truthiness():
    start = np.array([[.1, .2, .3], [.6, .7, .8]])
    end = start + [.02, -.01, .03]
    result = pathway.build_migration_path(np.eye(3) * 10., ["C", "O"], [1, 1], start, end,
        end_species=["C", "O"], end_counts=np.array([1, 1]), images=2, match=False)
    assert result.image_count == 4 and result.counts == (1, 1)
    assert np.allclose(result.images[-1], np.mod(end, 1.))


@pytest.mark.parametrize("species,counts,natoms", [(["C"], [1], 2), (["C", "O"], [1], 1)])
def test_unmatched_migration_rejects_inconsistent_header_cardinality(species, counts, natoms):
    points = np.zeros((natoms, 3))
    with pytest.raises(ValueError, match="species|count|atom"):
        pathway.build_migration_path(np.eye(3) * 5., species, counts, points, points, match=False)


@pytest.mark.parametrize("second,tolerance", [
    (np.full((3, 3), np.nan), 1e-6),
    (np.zeros((1, 3)), 2.0),
    (np.eye(3), float("nan")),
    (np.eye(3), float("inf")),
])
def test_same_cell_check_rejects_nonfinite_or_nonmatrix_comparisons(second, tolerance):
    with pytest.raises(ValueError):
        pathway._check_same_cell(np.eye(3), second, tolerance)


def test_unaddressable_migration_images_fail_before_atom_matching(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("impossible image chain started assignment work")
    monkeypatch.setattr(pathway, "match_atoms", forbidden)
    with pytest.raises(ValueError, match="large|address"):
        pathway.build_migration_path(np.eye(3), ["C"], [1], np.zeros((1, 3)), np.zeros((1, 3)), images=2 ** 62)


def test_pair_cost_temporaries_are_bounded_with_exact_dense_values(monkeypatch):
    rng = np.random.default_rng(224)
    first, second = rng.random((300, 3)) * .2, rng.random((300, 3)) * .2
    original = pathway.minimum_image_distances
    dense = original(np.eye(3), (second[None, :, :] - first[:, None, :]).reshape(-1, 3))
    expected = (dense * dense).reshape(300, 300)
    sizes = []
    def observed(cell, deltas):
        sizes.append(len(deltas))
        return original(cell, deltas)
    monkeypatch.setattr(pathway, "minimum_image_distances", observed)
    actual = pathway._pair_cost_matrix(np.eye(3), first, second)
    assert np.array_equal(actual, expected)
    assert max(sizes) <= 65536


def test_assignment_certificate_avoids_an_extra_full_cost_matrix():
    n = 1000
    row = np.arange(n, dtype=float)
    column = np.arange(n, dtype=float)
    cost = row[:, None] + column[None, :]
    assignment = pathway.Assignment(np.arange(n), float(np.trace(cost)), row, column)
    tracemalloc.start()
    try:
        assert assignment.certificate_error(cost) == 0.
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < cost.nbytes // 2


@pytest.mark.parametrize("labels", [["C"], ["C", "O", "N"]])
def test_record_regrouping_rejects_position_label_count_mismatch(labels):
    source = _record(["C"], [2], [[0., 0., 0.], [.2, .3, .4]])
    with pytest.raises(ValueError, match="label|atom|species"):
        records.record_from_atoms(source, np.eye(3), source.positions_direct, labels, comment="invalid")


def test_record_grouping_keeps_source_priority_atom_order_and_metadata_copy():
    source = _record(["O", "C"], [1, 2], [[0., 0., 0.], [.2, .3, .4], [.6, .7, .8]])
    points = np.arange(12, dtype=float).reshape(4, 3) / 10.
    result = records.record_from_atoms(source, np.eye(3) * 3., points, ["C", "O", "N", "C"], comment="grouped")
    assert result.species == ["O", "C", "N"] and result.counts == [1, 2, 1]
    assert np.array_equal(result.positions_direct, np.mod(points[[1, 0, 3, 2]], 1.))
    assert result.metadata == source.metadata and result.metadata is not source.metadata
    assert result.metadata["nested"] is source.metadata["nested"]


def test_surface_grouping_emits_repeated_header_species_only_once():
    points = np.arange(9, dtype=float).reshape(3, 3) / 10.
    flags = [("F", "T", "F"), ("T", "F", "T"), ("F", "F", "T")]
    atoms = [(symbol, point, flag) for symbol, point, flag in zip(["Na", "Cl", "Na"], points, flags)]
    positions, counts, species, result_flags = surface_cell._group_surface_atoms_by_species(atoms, ["Na", "Na", "Cl"])
    assert species == ["Na", "Cl"] and counts == [2, 1]
    assert np.array_equal(positions, points[[0, 2, 1]])
    assert result_flags == [flags[0], flags[2], flags[1]]


@pytest.mark.parametrize("tolerance", [-.1, float("nan"), float("inf")])
def test_surface_level_clustering_rejects_invalid_tolerance_before_loop(tolerance, monkeypatch):
    original = np.searchsorted
    calls = []
    def bounded_probe(*args, **kwargs):
        calls.append(None)
        if len(calls) > 2:
            pytest.fail("invalid tolerance entered a nonadvancing clustering loop")
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "searchsorted", bounded_probe)
    with pytest.raises(ValueError, match="tolerance"):
        surface_cell._deduplicate_scalar_levels([0., 1.], tolerance)


def test_surface_level_clustering_preserves_input_and_last_kept_rule():
    values = np.array([.4, .1, .11, .9])
    before = values.copy()
    levels = surface_cell._deduplicate_scalar_levels(values, .05)
    assert levels == [.1, .4, .9]
    assert np.array_equal(values, before)


@pytest.mark.parametrize("kind", ["finite", "bad_dual", "nan_cost", "nan_dual", "large", "negative_column", "rectangular", "empty"])
def test_certificate_blocking_preserves_legacy_values_for_custom_inputs(kind):
    n = 0 if kind == "empty" else 2
    cost = np.eye(n)
    row, column = np.zeros(n), np.zeros(n)
    indices = np.arange(n)
    if kind == "bad_dual":
        row[:] = 1.
        column[:] = 1.
    elif kind == "nan_cost":
        cost[0, 1] = np.nan
    elif kind == "nan_dual":
        row[0] = np.nan
    elif kind == "large":
        cost[:] = np.finfo(float).max
        row[:] = np.finfo(float).max
    elif kind == "negative_column":
        indices[0] = -1
    elif kind == "rectangular":
        cost = np.arange(6., dtype=float).reshape(2, 3)
        column = np.zeros(3)
        indices = np.array([0, 2])
    with np.errstate(invalid="ignore", over="ignore"):
        slack = cost - row[:, None] - column[None, :]
        infeasibility = float(max(0., -slack.min())) if slack.size else 0.
        tightness = float(np.max(np.abs(slack[np.arange(n), indices]))) if n else 0.
        expected = max(infeasibility, tightness)
        actual = pathway.Assignment(indices, 0., row, column).certificate_error(cost)
    assert actual == expected or (np.isnan(actual) and np.isnan(expected))


@pytest.mark.parametrize("endpoint_counts", [None, [], (), np.array([], dtype=int)])
def test_migration_empty_endpoint_counts_retain_start_counts_fallback(endpoint_counts):
    start = np.array([[0., 0., 0.], [.5, 0., 0.]])
    end = np.array([[.25, 0., 0.], [.75, 0., 0.]])
    result = pathway.build_migration_path(np.eye(3), ["C"], [2], start, end,
        end_species=["C"], end_counts=endpoint_counts, images=2, cell_tolerance=0.)
    assert result.counts == (2,) and result.species == ("C",)
    assert np.array_equal(result.matching.partners, [0, 1])
    assert result.matching.cost == .125 and result.image_count == 4
    for index, image in enumerate(result.images):
        assert np.array_equal(image, pathway._wrap_to_cell(start + (float(index) / 3.) * np.array([[.25, 0., 0.], [.25, 0., 0.]])))


def test_species_type_numbers_keep_first_seen_unicode_mapping():
    numbers, mapping = records.species_type_map(["O", "C", "O", "α", "C"])
    assert numbers == [1, 2, 1, 3, 2] and mapping == {1: "O", 2: "C", 3: "α"}



def test_nonplanar_integer_transform_captures_all27_atoms_and_flags():
    source = _record(["C"], [1], [[.2, .2, .2]], [("F", "T", "F")])
    source.lattice = np.eye(3)
    source.positions_cartesian = source.positions_direct.copy()
    transform = np.array([[3, 3, 3], [0, 3, 3], [0, 0, 3]])
    result = surface_cell._structure_from_transform(source, transform)
    expected = np.array([[(.2 + i) / 3., j / 3., k / 3.]
                         for i in range(3) for j in range(3) for k in range(3)])
    assert result.counts == [27] and result.positions_direct.shape == (27, 3)
    wrapped = np.mod(np.round(result.positions_direct, 12), 1.)
    order = np.lexsort((wrapped[:, 2], wrapped[:, 1], wrapped[:, 0]))
    assert np.allclose(wrapped[order], expected, rtol=0., atol=1e-12)
    assert np.array_equal(result.positions_cartesian, result.positions_direct @ result.lattice)
    assert result.selective_flags == [("F", "T", "F")] * 27
    assert np.array_equal(result.lattice, transform)


def test_ordinary_nonplanar_diagonal_transform_keeps_atom_image_order():
    source = _record(["C"], [1], [[.2, .3, .4]])
    source.lattice = np.eye(3)
    source.positions_cartesian = source.positions_direct.copy()
    transform = np.diag([2, 3, 2])
    result = surface_cell._structure_from_transform(source, transform)
    expected = np.array([[(.2 + i) / 2., (.3 + j) / 3., (.4 + k) / 2.]
                         for i in range(2) for j in range(3) for k in range(2)])
    assert result.counts == [12] and result.positions_direct.shape == (12, 3)
    assert np.allclose(result.positions_direct, expected, rtol=0., atol=1e-15)


@pytest.mark.parametrize("tolerance", [-.1, float("nan"), float("inf")])
def test_surface_transform_refuses_invalid_tolerance_before_dispatch(tolerance):
    source = _record(["C"], [1], [[.2, .2, .2]])
    transform = np.array([[3, 3, 3], [0, 3, 3], [0, 0, 3]])
    with pytest.raises(ValueError, match="tolerance"):
        surface_cell._structure_from_transform(source, transform, tolerance=tolerance)


@pytest.mark.parametrize("translation,expected", [([.5, 0., 0.], False), ([0., 0., 0.], True)])
def test_translation_oracle_requires_species_preserving_bijection(translation, expected):
    source = _record(["C"], [3], [[0., 0., 0.], [8e-6, 0., 0.], [.5, 0., 0.]])
    source.lattice = np.eye(3) * 1000.
    source.positions_cartesian = source.positions_direct @ source.lattice
    assert surface_cell._translation_maps_structure(source, translation, tolerance=1e-5) is expected


@pytest.mark.parametrize("transform", [
    np.array([[3, 3, 3], [0, 3, 3], [0, 0, 3]]),
    np.array([[-3, -3, -3], [0, 3, 3], [0, 0, 3]]),
    np.array([[2, -2, 1], [1, 1, 0], [0, 1, 2]]),
])
def test_nonplanar_transforms_preserve_species_and_periodic_images(transform):
    flags = [("F", "T", "F"), ("T", "F", "T")]
    source = _record(["Na", "Cl"], [1, 1], [[.2, .2, .2], [.4, .3, .6]], flags)
    result = surface_cell._structure_from_transform(source, transform)
    multiplicity = int(round(abs(np.linalg.det(transform))))
    assert result.species == ["Na", "Cl"] and result.counts == [multiplicity, multiplicity]
    assert result.positions_direct.shape == (2 * multiplicity, 3)
    assert result.selective_flags == [flags[0]] * multiplicity + [flags[1]] * multiplicity
    inverse = np.linalg.inv(source.lattice)
    for atom in range(2):
        images = result.positions_cartesian[atom * multiplicity:(atom + 1) * multiplicity]
        shifts = (images - source.positions_cartesian[atom]) @ inverse
        assert np.allclose(shifts, np.rint(shifts), rtol=0., atol=1e-10)
    canonical = np.mod(np.round(result.positions_direct, 9), 1.)
    assert len(np.unique(canonical, axis=0)) == 2 * multiplicity

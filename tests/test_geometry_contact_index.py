"""Conservative contact candidates, exact ties and usable reference tooling."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from cellstine.core import contacts, geometry


@pytest.mark.parametrize("cell", [
    np.diag([10., 8., 12.]),
    np.array([[4., 0., 0.], [9., 3.2, 0.], [.5, 1.3, 7.]]),
    np.array([[3., 0., 0.], [-1.5, np.sqrt(3.) * 1.5, 0.], [0., 0., 12.]]),
])
def test_periodic_candidates_include_every_near_site_in_original_order(cell):
    rng = np.random.default_rng(715)
    sites = rng.random((90, 3))
    sites[:4] = [[0., 0., 0.], [.999999999, 0., 0.], [0., 0., 0.], [.5, .5, .5]]
    queries = np.vstack((rng.random((23, 3)), [[1., 0., 0.], [-2., 3., 0.], [.5, .5, .5]]))
    radius = .7
    original = sites.copy()
    index = geometry.PeriodicSiteIndex(cell, sites, labels=np.arange(90) % 3, tolerance=radius)
    candidates, valid = index.candidates(queries)
    expected = geometry.pairwise_minimum_image_distances(cell, queries, other_direct=sites)
    assert candidates.shape == valid.shape and candidates.shape[0] == len(queries)
    for row, mask, distances in zip(candidates, valid, expected):
        found = row[mask]
        assert np.array_equal(found, np.unique(found))
        assert set(np.flatnonzero(distances <= radius)).issubset(found)
    assert np.array_equal(sites, original)
    candidates[:] = 0
    again, mask = index.candidates(queries)
    assert any(np.any(row[keep] != 0) for row, keep in zip(again, mask))


def test_empty_periodic_candidate_tables():
    empty = geometry.PeriodicSiteIndex(np.eye(3), np.empty((0, 3)))
    candidates, valid = empty.candidates(np.zeros((2, 3)))
    assert candidates.shape == valid.shape == (2, 0)
    full = geometry.PeriodicSiteIndex(np.eye(3), np.zeros((1, 3)))
    candidates, valid = full.candidates(np.empty((0, 3)))
    assert candidates.shape == valid.shape == (0, 0)


def test_sparse_contacts_prune_scored_pairs_without_changing_result(monkeypatch):
    grid = np.array([(i / 20, j / 20, .2) for i in range(20) for j in range(20)])
    other = grid.copy()
    other[:, 2] = .5
    before, other_before = grid.copy(), other.copy()
    sizes = []
    original = geometry.bounded_minimum_image_squared
    def measured(cell, deltas, radius, **kwargs):
        sizes.append(len(deltas))
        return original(cell, deltas, radius, **kwargs)
    monkeypatch.setattr(geometry, "bounded_minimum_image_squared", measured)
    found = contacts.closest_contact(np.diag([60., 60., 12.]), grid, other,
                                    first_species=["C"] * 400, second_species=["O"] * 400)
    assert (found.first_index, found.second_index) == (0, 0)
    assert found.distance == pytest.approx(3.6)
    assert (found.first_species, found.second_species) == ("C", "O")
    assert sum(sizes) < 20000
    assert np.array_equal(grid, before) and np.array_equal(other, other_before)


def test_contact_ties_use_sqrt_distance_then_original_pair_order():
    first = np.tile([.5, .5, 0.], (400, 1))
    first[300] = [.5, .5, .5]
    second = np.tile([.5, .5, .75], (400, 1))
    second[0] = [.625, .5 + 2. ** -29, .5]
    second[1] = [.625, .5, .5]
    # Squared distances differ by one ulp, but both square roots are exactly1.
    assert np.sqrt(1. + np.spacing(1.)) == 1.
    found = contacts.closest_contact(np.eye(3) * 8., first, second)
    assert found.distance == 1.
    assert (found.first_index, found.second_index) == (300, 0)


def test_geometry_reference_loader_uses_current_src_layout_from_other_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks/benchmark_geometry.py"
    command = ("import pathlib,runpy,sys; runpy.run_path(sys.argv[1]); "
               "import cellstine; assert pathlib.Path(cellstine.__file__).resolve() == pathlib.Path(sys.argv[2]).resolve()")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", command, str(script), str(root / "src/cellstine/__init__.py")],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cell", [np.diag([10., 8., 12.]),
    np.array([[4., 0., 0.], [9., 3.2, 0.], [.5, 1.3, 7.]])])
def test_indexed_contacts_match_dense_reference_exactly_for_late_ties(seed, cell):
    rng = np.random.default_rng(seed)
    first = rng.random((321, 3)) + rng.integers(-3, 4, (321, 3))
    second = rng.random((257, 3)) + rng.integers(-3, 4, (257, 3))
    second[17] = second[13]
    first[319] = second[13] + [5.0001, -7.0002, 2.]
    first[320] = first[319]
    copies = first.copy(), second.copy()
    reference = geometry.pairwise_minimum_image_distances(cell, first, other_direct=second)
    row, column = np.unravel_index(np.argmin(reference), reference.shape)
    result = contacts.closest_contact(cell, first, second, first_species=["C"] * len(first),
                                     second_species=["O"] * len(second))
    assert result.distance == reference[row, column]
    assert (result.first_index, result.second_index) == (row, column) == (319, 13)
    assert (result.first_species, result.second_species) == ("C", "O")
    assert np.array_equal(first, copies[0]) and np.array_equal(second, copies[1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_contacts_keep_legacy_nan_pair_and_skip_index(value, monkeypatch):
    first = np.zeros((321, 3))
    first[-1, 0] = value
    second = np.tile([.1, .2, .3], (257, 1))
    def forbidden(*args, **kwargs):
        pytest.fail("nonfinite inputs entered integer bucket indexing")
    monkeypatch.setattr(geometry, "PeriodicSiteIndex", forbidden)
    with np.errstate(invalid="ignore"):
        found = contacts.closest_contact(np.eye(3) * 10., first, second)
    assert np.isnan(found.distance)
    assert (found.first_index, found.second_index) == (320, 0)


@pytest.mark.parametrize("shift", [0., 1e-14])
def test_zero_and_tiny_contact_radii_keep_dense_fallback(shift, monkeypatch):
    first = np.zeros((321, 3))
    second = np.tile([shift, 0., 0.], (257, 1))
    def forbidden(*args, **kwargs):
        pytest.fail("tiny radius entered unsafe integer bucket encoding")
    monkeypatch.setattr(geometry, "PeriodicSiteIndex", forbidden)
    found = contacts.closest_contact(np.eye(3) * 10., first, second)
    assert (found.first_index, found.second_index) == (0, 0)
    assert found.distance == shift * 10.

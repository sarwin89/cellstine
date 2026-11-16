"""Ordered physical deduplication and bounded site-distance temporaries."""

from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from cellstine.interface.surface import surface_sites as sites
from cellstine.interface.surface.surface import Surface
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


@pytest.mark.parametrize("values,tolerance,expected", [
    ([0., 4e-10], 1e-10, [0., 4e-10]),
    ([0., 1e-4 - 2e-10, 1e-4 + 2e-10], 1e-4, [0., 1e-4 + 2e-10]),
])
def test_rounded_near_duplicates_do_not_discard_first_surviving_site(values, tolerance, expected):
    points = [np.array([value, 0.]) for value in values]
    result = sites._deduplicate_uv_points(points, np.eye(3), tolerance)
    assert [float(point[0]) for point in result] == expected


def test_deduplication_keeps_order_exact_duplicates_and_owned_points():
    points = np.array([[.7, .2], [.1, .3], [.7, .2], [1.1, .3]])
    saved = points.copy()
    result = sites._deduplicate_uv_points(points, np.eye(3))
    assert np.array_equal(np.array(result), points[:2])
    result[0][0] = 9.
    assert np.array_equal(points, saved)


def _square_grid(side):
    return np.array([(i / side, j / side) for i in range(side) for j in range(side)])


@pytest.mark.parametrize("operation", ["coordination", "bridges"])
def test_site_consumers_bound_anchor_image_pair_temporaries(monkeypatch, operation):
    side = 18
    points = _square_grid(side)
    cell = np.diag([float(side), float(side), 12.])
    original = sites._anchor_image_distance_matrix
    observed = []

    def bounded(first, expanded, lattice):
        pairs = len(first) * len(expanded)
        observed.append(pairs)
        assert pairs <= 65536, "site workflow allocated an unbounded anchor/image pair matrix"
        return original(first, expanded, lattice)

    monkeypatch.setattr(sites, "_anchor_image_distance_matrix", bounded)
    if operation == "coordination":
        assert sites._top_layer_coordination_counts(points, cell, 1.01) == [4] * len(points)
    else:
        result = sites._find_bridge_sites(points, cell, 1.01)
        assert len(result) == 2 * len(points)
    assert observed and max(observed) <= 65536


def test_nearest_neighbor_bounds_pair_displacements(monkeypatch):
    points = _square_grid(20)
    original = sites.plane_minimum_image_distances
    observed = []

    def bounded(basis, deltas):
        observed.append(len(deltas))
        assert len(deltas) <= 65536, "nearest-neighbor search allocated all pairs"
        return original(basis, deltas)

    monkeypatch.setattr(sites, "plane_minimum_image_distances", bounded)
    assert sites._nearest_neighbor_distance(points, np.diag([20., 20., 12.])) == pytest.approx(1.)
    assert observed and max(observed) <= 65536


def _scalar_sites(points, cell, cutoff):
    shifts = np.array(list(product(range(-3, 4), repeat=2)), dtype=float)
    counts, midpoints = [], []
    for point in points:
        count = 0
        for other in points:
            for shift in shifts:
                image = other + shift
                difference = image - point
                distance = np.linalg.norm(difference[0] * cell[0] + difference[1] * cell[1])
                if 1e-8 < distance <= cutoff + 1e-12:
                    count += 1
                    midpoints.append(np.mod(.5 * (point + image), 1.))
        counts.append(count)
    kept = []
    for point in midpoints:
        if any(np.min(np.linalg.norm((point - seen + shifts) @ cell[:2], axis=1)) <= 1e-4
               for seen in kept):
            continue
        kept.append(point)
    return counts, np.array(kept)


@pytest.mark.parametrize("angle", [60, 90, 120])
@pytest.mark.parametrize("budget", [3, 250])
def test_chunking_preserves_scalar_periodic_counts_and_bridge_order(monkeypatch, angle, budget):
    radians = np.deg2rad(angle)
    cell = np.array([[2., 0., 0.], [2. * np.cos(radians), 2. * np.sin(radians), 0.], [0., 0., 12.]])
    points = _square_grid(2)
    expected_counts, expected_bridges = _scalar_sites(points, cell, 1.01)
    monkeypatch.setattr(sites, "_SITE_PAIR_BLOCK", budget)
    assert sites._top_layer_coordination_counts(points, cell, 1.01) == expected_counts
    assert np.array_equal(sites._find_bridge_sites(points, cell, 1.01), expected_bridges)


@pytest.mark.parametrize("cutoff,counts,bridges", [
    (1. - 2e-12, [0], []),
    (1. - .5e-12, [4], [[.5, 0.], [0., .5]]),
    (1., [4], [[.5, 0.], [0., .5]]),
])
def test_self_images_preserve_cutoff_slack_and_bridge_representatives(cutoff, counts, bridges):
    points = np.zeros((1, 2))
    assert sites._top_layer_coordination_counts(points, np.eye(3), cutoff) == counts
    assert np.array_equal(np.array(sites._find_bridge_sites(points, np.eye(3), cutoff)), np.array(bridges))


def test_nearest_neighbor_preserves_duplicates_self_images_and_sheared_images():
    assert sites._nearest_neighbor_distance(np.zeros((1, 2)), np.eye(3)) == 1.
    assert sites._nearest_neighbor_distance(np.zeros((2, 2)), np.eye(3)) == 1.
    cell = np.array([[1., 0., 0.], [20., 1., 0.], [0., 0., 12.]])
    assert sites._nearest_neighbor_distance(np.array([[0., 0.], [0., .5]]), cell) == .5
    with pytest.raises(ValueError, match="nearest-neighbour"):
        sites._nearest_neighbor_distance(np.empty((0, 2)), np.eye(3))


def test_empty_site_collections_keep_their_result_types():
    assert sites._deduplicate_uv_points([], np.eye(3)) == []
    assert sites._find_bridge_sites(np.empty((0, 2)), np.eye(3), 1.) == []
    assert sites._top_layer_coordination_counts(np.empty((0, 2)), np.eye(3), 1.) == []


def test_chunked_site_workflow_writes_expected_physical_report_and_manifest(tmp_path, monkeypatch):
    cell = np.array([[2., 0., 0.], [1., np.sqrt(3.), 0.], [0., 0., 12.]])
    direct = np.array([[0., 0., .2], [1/3, 1/3, .4], [2/3, 2/3, .6]])
    record = StructureRecord(comment="ABC surface", lattice=cell, species=["Al"], counts=[3],
        positions_direct=direct, positions_cartesian=direct @ cell)
    source = tmp_path / "slab.vasp"
    VaspIO().write(record, str(source), positions_are_cartesian=False, wrap_positions=True)
    saved = source.read_bytes()
    monkeypatch.setattr(sites, "_SITE_PAIR_BLOCK", 3)
    result = Surface(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output").sites(
        slab_poscar=str(source), output_path=str(tmp_path / "sites.json"))
    expected = {"top": 1, "bridge": 3, "fcc_hollow": 1, "hcp_hollow": 1}
    written = json.loads(Path(result.artifacts["sites_json"]).read_text(encoding="utf-8"))
    assert written["site_counts"] == result.summary["site_counts"] == expected
    assert written["detected_layer_count"] == 3
    assert written["nearest_neighbor_distance_angstrom"] == pytest.approx(2.)
    assert written["average_top_layer_coordination"] == 6.
    assert written["sites"]["hcp_hollow"][0]["subsurface_depth"] == 1
    assert written["sites"]["fcc_hollow"][0]["subsurface_depth"] == 2
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["summary"] == result.summary
    assert source.read_bytes() == saved

"""Mathematical checks on commensurate heterointerface matching and building.

A match is only meaningful if the two integer supercells really do describe the
same in-plane lattice once each slab has taken its recorded share of the strain.
These tests check that identity directly, check the reported strains against the
singular values of the recorded affine maps, and check that the structure the
builder writes has the matched cell, the requested gap and vacuum, and bond
lengths that follow from an in-plane-only deformation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.interface.workflow import lattice_match
from cellstine.interface.workflow.interface import Interface
from cellstine.io import native as io_mod

from conftest import write_poscar


@pytest.fixture(scope="session")
def aluminium_bulk(tmp_path_factory) -> str:
    """Face-centred cubic aluminium, conventional four-atom cell."""

    path = tmp_path_factory.mktemp("interface-match") / "al.vasp"
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(path, lattice, ["Al"], [4], positions))


@pytest.fixture(scope="module")
def silicon_aluminium_match(tmp_path_factory, silicon_poscar, aluminium_bulk):
    """Match Si(111) against Al(111) and return the workflow and its document."""

    workspace = tmp_path_factory.mktemp("interface-match-run")
    workflow = Interface(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    result = workflow.match(
        bottom_bulk=str(silicon_poscar),
        top_bulk=aluminium_bulk,
        bottom_millers=["111"],
        top_millers=["111"],
        bottom_layers_list=[2],
        top_layers_list=[2],
        max_strain=0.01,
        max_length=13.0,
        preview_limit=0,
    )
    matches_path = str(result.artifacts["matches_json"])
    return workflow, lattice_match.read_matches(matches_path), matches_path


def _column_basis(lattice: np.ndarray) -> np.ndarray:
    return np.asarray(lattice, dtype=float)[:2, :2].T


def _minimum_distance(lattice: np.ndarray, cartesian: np.ndarray) -> float:
    shifts = np.array([(i, j, 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float) @ lattice
    difference = cartesian[:, None, :] - cartesian[None, :, :]
    images = difference[:, :, None, :] + shifts[None, None, :, :]
    distances = np.linalg.norm(images, axis=3)
    self_pairs = np.eye(len(cartesian), dtype=bool)[:, :, None] & (
        np.all(np.isclose(shifts, 0.0), axis=1)[None, None, :]
    )
    distances[self_pairs] = np.inf
    return float(distances.min())


def test_strain_budgets_follow_the_requested_mode():
    shared = lattice_match.MatchRequest(max_length=10.0, max_strain=0.02, strain_mode="shared")
    film = lattice_match.MatchRequest(max_length=10.0, max_strain=0.02, strain_mode="film")
    assert shared.strain_budgets == (0.02, 0.02)
    assert film.strain_budgets == (0.0, 0.02)
    with pytest.raises(ValueError):
        lattice_match.MatchRequest(max_length=10.0, max_strain=0.02, strain_mode="both")
    with pytest.raises(ValueError):
        lattice_match.MatchRequest(max_length=-1.0)


def test_matched_supercells_share_one_lattice(silicon_aluminium_match):
    """Both strained superlattices must equal the recorded shared lattice."""

    _, document, _ = silicon_aluminium_match
    assert document["matches"], "Si(111) on Al(111) must admit commensurate matches"
    for entry in document["matches"][:8]:
        candidate = lattice_match.candidate_for_match(entry)
        shared = np.asarray(candidate["shared_lattice"], dtype=float)
        for side in ("bottom", "top"):
            slab = io_mod.read_poscar(entry[f"{side}_slab"])
            matrix = np.asarray(candidate[f"{side}_matrix"], dtype=float)
            affine = np.asarray(candidate[f"{side}_affine"], dtype=float)
            superlattice_rows = matrix @ np.asarray(slab.lattice, dtype=float)[:2, :2]
            strained_columns = affine @ superlattice_rows.T
            assert np.allclose(strained_columns, shared, atol=1e-8)


def test_reported_strains_match_the_recorded_deformations(silicon_aluminium_match):
    """Layer strains must be the logarithms of the affine principal stretches."""

    _, document, _ = silicon_aluminium_match
    budgets = (
        float(document["search"]["bottom_strain_budget"]),
        float(document["search"]["top_strain_budget"]),
    )
    for entry in document["matches"][:8]:
        candidate = lattice_match.candidate_for_match(entry)
        for side, budget in zip(("bottom", "top"), budgets):
            affine = np.asarray(candidate[f"{side}_affine"], dtype=float)
            measured = np.sort(np.log(np.linalg.svd(affine, compute_uv=False)))
            recorded = np.sort(np.asarray(entry[f"{side}_layer_strain"], dtype=float))
            assert np.allclose(measured, recorded, atol=1e-9)
            assert float(np.max(np.abs(measured))) <= budget + 1e-12
        relative = np.asarray(entry["top_layer_strain"], dtype=float) - np.asarray(
            entry["bottom_layer_strain"], dtype=float
        )
        assert np.allclose(relative, entry["principal_strains"], atol=1e-9)
        assert entry["strain"] == pytest.approx(float(np.max(np.abs(relative))))


def test_best_silicon_on_aluminium_match_is_the_known_three_on_four_cell(silicon_aluminium_match):
    """3x3 Si(111) on 4x4 Al(111) is the smallest low-strain cell of this pair."""

    _, document, _ = silicon_aluminium_match
    best = document["matches"][0]
    assert abs(int(round(np.linalg.det(np.asarray(best["bottom_matrix"], dtype=float))))) == 9
    assert abs(int(round(np.linalg.det(np.asarray(best["top_matrix"], dtype=float))))) == 16
    silicon_surface_constant = 5.43 / math.sqrt(2.0)
    aluminium_surface_constant = 4.05 / math.sqrt(2.0)
    expected = abs(math.log((3.0 * silicon_surface_constant) / (4.0 * aluminium_surface_constant)))
    assert best["strain"] == pytest.approx(expected, rel=1e-6)
    assert best["cell_a"] == pytest.approx(best["cell_b"], rel=1e-9)
    # The shared cell is hexagonal, and a hexagonal cell sits exactly on the
    # boundary of the Lagrange--Gauss reduction condition, where the sixty and
    # the hundred-and-twenty degree descriptions of the same lattice are both
    # reduced.  ``_reduce_common_basis`` picks the obtuse one, so the reported
    # angle is the crystallographic 120 degrees and does not depend on the last
    # bit of a dot product.
    assert best["cell_gamma_deg"] == pytest.approx(120.0, abs=1e-6)


def test_atom_counts_follow_the_supercell_determinants(silicon_aluminium_match):
    _, document, _ = silicon_aluminium_match
    for entry in document["matches"][:8]:
        for side in ("bottom", "top"):
            slab = io_mod.read_poscar(entry[f"{side}_slab"])
            determinant = abs(int(round(np.linalg.det(np.asarray(entry[f"{side}_matrix"], dtype=float)))))
            assert int(entry[f"{side}_atom_count"]) == slab.natoms * determinant
        assert int(entry["total_atoms"]) == int(entry["bottom_atom_count"]) + int(entry["top_atom_count"])


def test_matches_are_sorted_and_indexed(silicon_aluminium_match):
    """Matches rise in strain, ties break on size, and the index is the rank.

    Two matches that are the same deformation reached through different but
    equivalent integer supercells can differ in the last bits of the strain, so
    the ranking treats anything closer than the documented resolution as a tie
    and puts the smaller cell first.  The strain is therefore non-decreasing up
    to that resolution, not bit for bit.
    """

    _, document, _ = silicon_aluminium_match
    entries = document["matches"]
    resolution = lattice_match.STRAIN_ORDER_RESOLUTION
    keys = [lattice_match.match_order_key(entry) for entry in entries]
    assert keys == sorted(keys)
    for previous, current in zip(entries, entries[1:]):
        assert float(current["strain"]) >= float(previous["strain"]) - resolution
        if abs(float(current["strain"]) - float(previous["strain"])) <= resolution:
            assert int(current["total_atoms"]) >= int(previous["total_atoms"])
    assert [int(entry["index"]) for entry in entries] == list(range(1, len(entries) + 1))


def test_built_interface_has_the_matched_cell_and_geometry(silicon_aluminium_match):
    """The written interface must realise the match, the gap, and the vacuum."""

    workflow, document, matches_path = silicon_aluminium_match
    best = document["matches"][0]
    gap, vacuum = 2.4, 14.0
    result = workflow.build(
        match_json=matches_path,
        match_index=1,
        gap=gap,
        vacuum=vacuum,
    )
    structure = io_mod.read_poscar(str(result.artifacts["interface_poscar"]))
    lattice = np.asarray(structure.lattice, dtype=float)
    shared = np.asarray(best["shared_lattice"], dtype=float)
    built = _column_basis(lattice)
    # The builder may swap a and b to keep the cell right-handed.
    swapped = built[:, ::-1]
    assert np.allclose(built, shared, atol=1e-8) or np.allclose(swapped, shared, atol=1e-8)
    assert structure.natoms == int(best["total_atoms"])

    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    heights = cartesian[:, 2]
    span = float(heights.max() - heights.min())
    assert float(lattice[2, 2]) == pytest.approx(span + vacuum, abs=1e-6)
    assert float(heights.min()) == pytest.approx(0.5 * vacuum, abs=1e-6)

    labels = np.array(
        [symbol for symbol, count in zip(structure.species, structure.counts) for _ in range(count)]
    )
    silicon = heights[labels == "Si"]
    aluminium = heights[labels == "Al"]
    assert float(aluminium.min() - silicon.max()) == pytest.approx(gap, abs=1e-6)
    assert _minimum_distance(lattice, cartesian) > 2.0


def test_direct_stacking_refuses_a_large_mismatch(tmp_path, silicon_poscar, aluminium_bulk):
    workflow = Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="commensurate"):
        workflow.build(
            bottom_input=str(silicon_poscar),
            top_input=aluminium_bulk,
            bottom_kind="bulk",
            top_kind="bulk",
            bottom_miller="111",
            top_miller="111",
            bottom_layers=2,
            top_layers=2,
            gap=2.5,
        )


def test_direct_stacking_accepts_a_matching_pair(tmp_path, silicon_poscar):
    workflow = Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = workflow.build(
        bottom_input=str(silicon_poscar),
        top_input=str(silicon_poscar),
        bottom_kind="bulk",
        top_kind="bulk",
        bottom_miller="111",
        top_miller="111",
        bottom_layers=2,
        top_layers=2,
        gap=2.5,
    )
    assert float(result.summary["raw_inplane_mismatch"]) < 1e-9
    structure = io_mod.read_poscar(str(result.artifacts["interface_poscar"]))
    assert structure.natoms == 4


def test_match_document_validation_rejects_broken_records(silicon_aluminium_match, tmp_path):
    _, document, _ = silicon_aluminium_match
    broken = {
        "schema": document["schema"],
        "version": document["version"],
        "search": dict(document["search"]),
        "matches": [dict(document["matches"][0])],
    }
    broken["matches"][0]["strain"] = 10.0
    broken["matches"][0]["top_strain"] = 10.0
    with pytest.raises(ValueError):
        lattice_match.validate_matches(broken)

    reindexed = {
        "schema": document["schema"],
        "version": document["version"],
        "search": dict(document["search"]),
        "matches": [dict(document["matches"][0])],
    }
    reindexed["matches"][0]["index"] = 7
    with pytest.raises(ValueError):
        lattice_match.validate_matches(reindexed)

    round_trip_path = lattice_match.write_matches(tmp_path / "matches.json", document)
    assert lattice_match.read_matches(round_trip_path)["matches"] == document["matches"]


@pytest.fixture(scope="module")
def multi_thickness_scan(tmp_path_factory, silicon_poscar, aluminium_bulk):
    """A scan over two thicknesses per side, capped at a handful of matches."""

    workspace = tmp_path_factory.mktemp("interface-match-scan")
    workflow = Interface(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    result = workflow.match(
        bottom_bulk=str(silicon_poscar),
        top_bulk=aluminium_bulk,
        bottom_millers=["111"],
        top_millers=["111"],
        bottom_layers_list=[2, 3],
        top_layers_list=[2, 3],
        max_strain=0.01,
        max_length=13.0,
        max_matches=5,
        preview_limit=0,
    )
    return workflow, result, lattice_match.read_matches(str(result.artifacts["matches_json"]))


def test_a_capped_scan_keeps_only_the_best_matches(multi_thickness_scan):
    """The cap keeps the head of the ranking, still consecutively indexed."""

    _, result, document = multi_thickness_scan
    matches = document["matches"]
    assert 0 < len(matches) <= 5
    assert int(result.summary["match_count"]) == len(matches)
    assert int(result.summary["surface_pairs_searched"]) == 4
    assert [int(entry["index"]) for entry in matches] == list(range(1, len(matches) + 1))
    keys = [lattice_match.match_order_key(entry) for entry in matches]
    assert keys == sorted(keys)
    strains = [float(entry["strain"]) for entry in matches]
    # Equally strained matches are ordered by cell size, so the raw strains are
    # non-decreasing only up to the resolution the ranking treats as a tie.
    resolution = lattice_match.STRAIN_ORDER_RESOLUTION
    assert all(
        later >= earlier - resolution for earlier, later in zip(strains, strains[1:])
    )
    sizes = [int(entry["total_atoms"]) for entry in matches]
    for (first_strain, first_size), (second_strain, second_size) in zip(
        zip(strains, sizes), zip(strains[1:], sizes[1:])
    ):
        if abs(second_strain - first_strain) <= resolution:
            assert first_size <= second_size


def test_every_reported_match_keeps_its_search_record(multi_thickness_scan):
    """Pruning never removes a search file a reported match points at."""

    _, result, document = multi_thickness_scan
    run_dir = Path(str(result.run_dir)).resolve()
    referenced = set()
    for entry in document["matches"]:
        results_json = Path(str(entry["results_json"]))
        assert results_json.is_file()
        referenced.add(results_json.parent.resolve())
        candidate = lattice_match.candidate_for_match(entry)
        assert int(candidate["index"]) == int(entry["candidate_index"])
        assert np.allclose(
            np.asarray(candidate["shared_lattice"], dtype=float),
            np.asarray(entry["shared_lattice"], dtype=float),
            atol=1e-12,
        )
        assert Path(str(entry["bottom_slab"])).is_file()
        assert Path(str(entry["top_slab"])).is_file()
    surviving = {path.resolve() for path in run_dir.glob("pair_*") if path.is_dir()}
    assert surviving == referenced
    assert int(result.summary["surface_pairs_kept"]) == len(surviving)


def test_an_uncapped_scan_reports_every_pair(tmp_path, silicon_poscar, aluminium_bulk):
    """``max_matches=0`` keeps every match, so no pair record is discarded."""

    workflow = Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = workflow.match(
        bottom_bulk=str(silicon_poscar),
        top_bulk=aluminium_bulk,
        bottom_millers=["111"],
        top_millers=["111"],
        bottom_layers_list=[2, 3],
        top_layers_list=[2],
        max_strain=0.01,
        max_length=13.0,
        max_matches=0,
        preview_limit=0,
    )
    document = lattice_match.read_matches(str(result.artifacts["matches_json"]))
    assert int(result.summary["surface_pairs_searched"]) == 2
    assert int(result.summary["surface_pairs_kept"]) == 2
    assert len(document["matches"]) > 5


def test_each_slab_is_generated_once_per_surface(multi_thickness_scan):
    """A thickness appears once on disk however many pairs use it."""

    _, result, document = multi_thickness_scan
    run_dir = Path(str(result.run_dir)).resolve()
    bottom_slabs = sorted(path.name for path in run_dir.glob("bottom_*.vasp"))
    top_slabs = sorted(path.name for path in run_dir.glob("top_*.vasp"))
    assert bottom_slabs == ["bottom_111_2.vasp", "bottom_111_3.vasp"]
    assert top_slabs == ["top_111_2.vasp", "top_111_3.vasp"]


def test_thicker_slabs_repeat_the_same_in_plane_solution(multi_thickness_scan):
    """Layer count changes the atom count, never the matched in-plane cell."""

    _, _, document = multi_thickness_scan
    by_matrices: dict[tuple, list[dict]] = {}
    for entry in document["matches"]:
        key = (
            tuple(tuple(row) for row in entry["bottom_matrix"]),
            tuple(tuple(row) for row in entry["top_matrix"]),
            round(float(entry["angle_deg"]), 9),
        )
        by_matrices.setdefault(key, []).append(entry)
    assert any(len(group) > 1 for group in by_matrices.values()), (
        "the scan must contain the same in-plane solution at two thicknesses"
    )
    for group in by_matrices.values():
        reference = group[0]
        for entry in group[1:]:
            assert entry["strain"] == pytest.approx(float(reference["strain"]), abs=1e-12)
            assert np.allclose(
                np.asarray(entry["shared_lattice"], dtype=float),
                np.asarray(reference["shared_lattice"], dtype=float),
                atol=1e-12,
            )
            ratio = int(entry["bottom_layers"]) / int(reference["bottom_layers"])
            assert int(entry["bottom_atom_count"]) == pytest.approx(
                int(reference["bottom_atom_count"]) * ratio
            )

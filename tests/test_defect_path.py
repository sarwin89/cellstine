"""Mathematical checks on the migration-path stage.

The claims tested here are about the path, not about the implementation.  The
pairing of the atoms must be a *minimum-cost* one, which is checked against
exhaustive enumeration on small cases and, on larger ones, against the dual
certificate the solver returns; the pairing must respect the species; every
atom must travel along its shortest periodic image, so an atom that leaves
through one face must return through the opposite one the short way; the images
must be evenly spaced and their total length must be the distance between the
endpoints; and the first and last images must reproduce the two structures they
were built from.

The same statements are proved in Lean in ``RequestProject/MigrationPath.lean``.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.core import geometry
from cellstine.core.pathway import (
    build_migration_path,
    match_atoms,
    optimal_assignment,
)
from cellstine.defect.workflow import Defect
from cellstine.io import native as native_io

from conftest import write_poscar

ALUMINIUM_CONSTANT = 4.05

FCC_PRIMITIVE = 0.5 * ALUMINIUM_CONSTANT * np.array(
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
)


def _fcc_supercell_sites() -> np.ndarray:
    """Return the eight fractional sites of a 2x2x2 repeat of the fcc cell."""

    return np.array(
        [[i / 2.0, j / 2.0, k / 2.0] for i in range(2) for j in range(2) for k in range(2)],
        dtype=float,
    )


@pytest.fixture()
def vacancy_hop(tmp_path: Path) -> tuple[Path, Path]:
    """Return the two endpoints of a vacancy hop in a 2x2x2 aluminium cell."""

    lattice = 2.0 * FCC_PRIMITIVE
    sites = _fcc_supercell_sites()
    start = write_poscar(
        tmp_path / "start.vasp", lattice, ["Al"], [7], np.delete(sites, 0, axis=0)
    )
    end = write_poscar(tmp_path / "end.vasp", lattice, ["Al"], [7], np.delete(sites, 1, axis=0))
    return start, end


# ---------------------------------------------------------------------------
# the assignment solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_assignment_matches_exhaustive_enumeration(size: int) -> None:
    rng = np.random.default_rng(20 + size)
    for _ in range(20):
        cost = rng.normal(size=(size, size))
        assignment = optimal_assignment(cost)
        best = min(
            sum(cost[row, column] for row, column in enumerate(order))
            for order in itertools.permutations(range(size))
        )
        assert assignment.total_cost == pytest.approx(best, abs=1e-12)
        assert sorted(assignment.columns.tolist()) == list(range(size))


@pytest.mark.parametrize("size", [2, 6, 17, 40])
def test_the_dual_certificate_is_feasible_and_tight(size: int) -> None:
    """The potentials prove optimality: feasible everywhere, tight on the pairing.

    This is the hypothesis of ``Cellstine.assignment_cost_le_of_dual_certificate``,
    so a solver run that satisfies it needs no further checking.
    """

    rng = np.random.default_rng(size)
    cost = rng.random((size, size)) * 10.0
    assignment = optimal_assignment(cost)
    slack = cost - assignment.row_potentials[:, None] - assignment.column_potentials[None, :]
    assert slack.min() > -1e-9
    rows = np.arange(size)
    assert np.abs(slack[rows, assignment.columns]).max() < 1e-9
    assert assignment.dual_bound() == pytest.approx(assignment.total_cost, abs=1e-9)
    assert assignment.certificate_error(cost) < 1e-9


def test_an_assignment_cost_must_be_square() -> None:
    with pytest.raises(ValueError):
        optimal_assignment(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# pairing the atoms of two structures
# ---------------------------------------------------------------------------


def test_matching_undoes_a_permutation_of_the_file_order() -> None:
    lattice = np.diag([10.0, 11.0, 12.0])
    rng = np.random.default_rng(7)
    start = rng.random((6, 3))
    order = np.array([3, 0, 5, 1, 4, 2])
    end = start[order] + 1e-3
    matching = match_atoms(lattice, start, end, ["Al"] * 6)
    inverse = np.argsort(order)
    assert np.array_equal(matching.partners, inverse)
    assert matching.certificate_error < 1e-9


def test_matching_pairs_only_atoms_of_the_same_species() -> None:
    lattice = np.diag([9.0, 9.0, 9.0])
    start = np.array([[0.10, 0.10, 0.10], [0.60, 0.60, 0.60]])
    # The nearest atom to the first is the oxygen, but only the aluminium may
    # be paired with it.
    end = np.array([[0.62, 0.60, 0.60], [0.12, 0.10, 0.10]])
    matching = match_atoms(lattice, start, end, ["Al", "O"], ["O", "Al"])
    assert matching.partners.tolist() == [1, 0]


def test_matching_beats_the_file_order_when_the_atoms_are_swapped() -> None:
    lattice = np.diag([8.0, 8.0, 8.0])
    start = np.array([[0.10, 0.10, 0.10], [0.60, 0.60, 0.60]])
    end = np.array([[0.60, 0.60, 0.60], [0.10, 0.10, 0.10]])
    matched = build_migration_path(lattice, ["He"], [2], start, end, images=3)
    ordered = build_migration_path(lattice, ["He"], [2], start, end, images=3, match=False)
    assert matched.path_length == pytest.approx(0.0, abs=1e-9) or matched.path_length < ordered.path_length


def test_the_reported_pairing_is_optimal_among_all_pairings() -> None:
    lattice = np.array([[6.0, 0.0, 0.0], [1.5, 5.5, 0.0], [0.7, -0.4, 7.0]])
    rng = np.random.default_rng(11)
    start = rng.random((6, 3))
    end = np.mod(start[[2, 4, 0, 5, 1, 3]] + rng.normal(scale=0.02, size=(6, 3)), 1.0)
    matching = match_atoms(lattice, start, end, ["Si"] * 6)
    costs = []
    for order in itertools.permutations(range(6)):
        deltas = end[list(order)] - start
        distances = geometry.minimum_image_distances(lattice, deltas)
        costs.append(float(np.sum(distances * distances)))
    assert matching.cost == pytest.approx(min(costs), abs=1e-9)


def test_a_different_composition_is_refused() -> None:
    lattice = np.diag([8.0, 8.0, 8.0])
    start = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    end = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        build_migration_path(
            lattice, ["Al"], [2], start, end, end_species=["Al", "O"], end_counts=[1, 1], images=2
        )


# ---------------------------------------------------------------------------
# the chain of images
# ---------------------------------------------------------------------------


def test_an_atom_crossing_a_face_takes_the_short_way_round() -> None:
    lattice = np.diag([10.0, 10.0, 10.0])
    start = np.array([[0.95, 0.5, 0.5], [0.2, 0.2, 0.2]])
    end = np.array([[0.05, 0.5, 0.5], [0.2, 0.2, 0.2]])
    path = build_migration_path(lattice, ["H"], [2], start, end, images=3, match=False)
    assert path.path_length == pytest.approx(1.0, abs=1e-9)
    middle = path.images[2][0]
    assert middle[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("images", [1, 2, 5, 9])
def test_the_images_are_evenly_spaced_and_add_up_to_the_path_length(images: int) -> None:
    lattice = np.array([[5.0, 0.0, 0.0], [1.0, 4.5, 0.0], [0.5, 0.5, 6.0]])
    rng = np.random.default_rng(3)
    start = rng.random((5, 3))
    end = np.mod(start + rng.normal(scale=0.05, size=(5, 3)), 1.0)
    path = build_migration_path(lattice, ["Ni"], [5], start, end, images=images, match=False)
    spacings = path.spacings()
    assert len(spacings) == images + 1
    assert np.allclose(spacings, path.image_spacing, atol=1e-9)
    assert float(np.sum(spacings)) == pytest.approx(path.path_length, abs=1e-9)


def test_the_endpoints_of_the_chain_are_the_two_structures() -> None:
    lattice = np.diag([7.0, 7.0, 7.0])
    rng = np.random.default_rng(5)
    start = rng.random((4, 3))
    end = np.mod(start[[1, 0, 3, 2]] + 0.01, 1.0)
    path = build_migration_path(lattice, ["Cu"], [4], start, end, images=4)
    assert np.allclose(path.images[0], np.mod(start, 1.0), atol=1e-12)
    assert np.allclose(path.images[-1], np.mod(end[path.matching.partners], 1.0), atol=1e-12)


def test_a_chain_needs_at_least_one_intermediate_image() -> None:
    lattice = np.diag([7.0, 7.0, 7.0])
    start = np.zeros((1, 3))
    end = np.array([[0.1, 0.0, 0.0]])
    with pytest.raises(ValueError):
        build_migration_path(lattice, ["Cu"], [1], start, end, images=0)


def test_two_different_cells_are_refused() -> None:
    lattice = np.diag([7.0, 7.0, 7.0])
    start = np.zeros((1, 3))
    end = np.array([[0.1, 0.0, 0.0]])
    with pytest.raises(ValueError):
        build_migration_path(
            lattice, ["Cu"], [1], start, end, end_lattice=np.diag([7.1, 7.0, 7.0]), images=3
        )


# ---------------------------------------------------------------------------
# the workflow stage
# ---------------------------------------------------------------------------


def test_the_vacancy_hop_is_a_nearest_neighbour_jump(vacancy_hop, tmp_path: Path) -> None:
    start, end = vacancy_hop
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.path(str(start), str(end), images=5)
    nearest_neighbour = ALUMINIUM_CONSTANT / math.sqrt(2.0)
    assert result.summary["moving_atoms"] == 1
    assert result.summary["path_length_ang"] == pytest.approx(nearest_neighbour, abs=1e-4)
    assert result.summary["image_spacing_ang"] == pytest.approx(nearest_neighbour / 6.0, abs=1e-4)
    assert result.summary["images_written"] == 7


def test_the_written_images_are_the_chain_that_was_reported(vacancy_hop, tmp_path: Path) -> None:
    start, end = vacancy_hop
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.path(str(start), str(end), images=3)
    written = [Path(item) for item in result.artifacts["images"]]
    assert [item.parent.name for item in written] == ["00", "01", "02", "03", "04"]

    payload = json.loads(Path(result.artifacts["path_json"]).read_text())
    assert payload["schema"] == "cellstine.defect.path"
    first = native_io.read_poscar(str(written[0]))
    last = native_io.read_poscar(str(written[-1]))
    initial = native_io.read_poscar(str(start))
    final = native_io.read_poscar(str(end))
    assert np.allclose(first.lattice, initial.lattice, atol=1e-9)
    assert np.allclose(first.positions_direct, np.mod(initial.positions_direct, 1.0), atol=1e-9)

    # The last image is the final structure, atom for atom under the pairing.
    partners = [int(value) - 1 for value in payload["matching"]["partners"]]
    assert np.allclose(
        last.positions_direct, np.mod(final.positions_direct[partners], 1.0), atol=1e-9
    )

    # Every image really is the structure the report describes.
    for index, item in enumerate(written):
        image = native_io.read_poscar(str(item))
        expected = np.asarray(payload["images"][index]["positions_direct"], dtype=float)
        assert np.allclose(image.positions_direct, expected, atol=1e-9)
        assert image.counts == first.counts and image.species == first.species


def test_the_saddle_image_is_the_closest_approach(vacancy_hop, tmp_path: Path) -> None:
    """A hop through the middle of a face brings the atoms closest half way."""

    start, end = vacancy_hop
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.path(str(start), str(end), images=5)
    contacts = json.loads(Path(result.artifacts["path_json"]).read_text())["shortest_contacts_ang"]
    assert int(np.argmin(contacts)) == 3
    assert contacts[0] == pytest.approx(contacts[-1], abs=1e-6)


def test_identical_endpoints_are_refused(tmp_path: Path) -> None:
    lattice = 2.0 * FCC_PRIMITIVE
    sites = np.delete(_fcc_supercell_sites(), 0, axis=0)
    start = write_poscar(tmp_path / "a.vasp", lattice, ["Al"], [7], sites)
    end = write_poscar(tmp_path / "b.vasp", lattice, ["Al"], [7], sites)
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    with pytest.raises(ValueError):
        tool.path(str(start), str(end), images=3)


def test_a_pinched_image_is_reported(tmp_path: Path) -> None:
    """A straight chain that drives two atoms together says so."""

    lattice = np.diag([12.0, 12.0, 12.0])
    # The two atoms swap ends along lines that miss each other by 0.6 A, so the
    # chain is writable but its middle images are far too tight to run.
    start = np.array([[0.2, 0.475, 0.5], [0.8, 0.525, 0.5]])
    end = np.array([[0.8, 0.475, 0.5], [0.2, 0.525, 0.5]])
    first = write_poscar(tmp_path / "first.vasp", lattice, ["Al"], [2], start)
    second = write_poscar(tmp_path / "second.vasp", lattice, ["Al"], [2], end)
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.path(str(first), str(second), images=3, match=False)
    assert result.summary["closest_contact_ang"] < 1.0
    assert any("through each other" in note for note in result.summary["warnings"])


def test_a_chain_that_collides_is_refused(tmp_path: Path) -> None:
    """Two atoms exchanged head-on would coincide, and that is not written."""

    lattice = np.diag([12.0, 12.0, 12.0])
    start = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    end = np.array([[0.8, 0.5, 0.5], [0.2, 0.5, 0.5]])
    first = write_poscar(tmp_path / "first.vasp", lattice, ["Al"], [2], start)
    second = write_poscar(tmp_path / "second.vasp", lattice, ["Al"], [2], end)
    tool = Defect(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    with pytest.raises(ValueError, match="one site"):
        tool.path(str(first), str(second), images=3, match=False)


def test_the_adsorbate_group_builds_the_same_chain(tmp_path: Path) -> None:
    """An adatom hopping between two sites is the same stage under `adsorbate`."""

    from cellstine.adsorbate.molecule import Molecule

    lattice = np.array([[2.55, 0.0, 0.0], [-1.275, 2.208, 0.0], [0.0, 0.0, 20.0]])
    substrate = np.array([[0.0, 0.0, 0.25]])
    start = np.vstack([substrate, [[1.0 / 3.0, 2.0 / 3.0, 0.35]]])
    end = np.vstack([substrate, [[2.0 / 3.0, 1.0 / 3.0, 0.35]]])
    first = write_poscar(tmp_path / "fcc.vasp", lattice, ["Cu", "H"], [1, 1], start)
    second = write_poscar(tmp_path / "hcp.vasp", lattice, ["Cu", "H"], [1, 1], end)
    tool = Molecule(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.path(str(first), str(second), images=3)
    assert result.summary["moving_atoms"] == 1
    # The hollow-to-hollow hop is one third of the diagonal of the surface mesh.
    step = (np.array([1.0 / 3.0, -1.0 / 3.0, 0.0])) @ lattice
    assert result.summary["path_length_ang"] == pytest.approx(
        float(np.linalg.norm(step)), abs=1e-3
    )


def test_the_cli_writes_the_chain(vacancy_hop, tmp_path: Path, monkeypatch) -> None:
    from cellstine.cli.main import main

    start, end = vacancy_hop
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "chain"
    assert main(
        [
            "defect",
            "path",
            str(start),
            str(end),
            "--images",
            "3",
            "--output-dir",
            str(destination),
        ]
    ) == 0
    assert sorted(item.name for item in destination.iterdir()) == ["00", "01", "02", "03", "04"]
    for folder in destination.iterdir():
        assert (folder / "POSCAR").is_file()

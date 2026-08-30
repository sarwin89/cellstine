"""Mathematical checks on defect analysis and defect structure generation.

The assertions here are statements about the crystallography, not about the
implementation: face-centred cubic aluminium must report one inequivalent atom
site of multiplicity four and exactly two interstitial voids, the octahedral one
of radius ``a/2`` and the tetrahedral one of radius ``a*sqrt(3)/4``; a vacancy
must remove one atom and leave every other atom and the cell untouched; an
inserted interstitial must actually clear the empty-sphere radius that was
reported for its site; and an adatom must sit the requested height above the
surface plane.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.defect.analysis import _normalise_supercell
from cellstine.defect.records import DefectAnalysis
from cellstine.defect.workflow import Defect
from cellstine.io import native as io_mod

from conftest import write_poscar

ALUMINIUM_CONSTANT = 4.05


def _minimum_distance_to_atoms(lattice: np.ndarray, hosts: np.ndarray, point: np.ndarray) -> float:
    """Return the periodic distance from ``point`` to the nearest host atom."""

    lattice = np.asarray(lattice, dtype=float)
    delta = np.asarray(hosts, dtype=float) - np.asarray(point, dtype=float)
    delta -= np.round(delta)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    best = math.inf
    for shift in shifts:
        cartesian = (delta + shift) @ lattice
        best = min(best, float(np.linalg.norm(cartesian, axis=1).min()))
    return best


def _sites_of_kind(analysis: DefectAnalysis, kind: str) -> list:
    return [site for site in analysis.sites if site.site_kind == kind]


@pytest.fixture(scope="module")
def aluminium_bulk_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("defect-structures") / "al4.vasp"
    lattice = ALUMINIUM_CONSTANT * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(path, lattice, ["Al"], [4], positions, comment="fcc aluminium"))


@pytest.fixture(scope="module")
def aluminium_slab_path(tmp_path_factory) -> str:
    """Return a four-layer Al(100) slab with a 15 Angstrom vacuum."""

    path = tmp_path_factory.mktemp("defect-slabs") / "al_slab.vasp"
    spacing = 0.5 * ALUMINIUM_CONSTANT
    height = 3.0 * spacing + 15.0
    lattice = np.diag([ALUMINIUM_CONSTANT, ALUMINIUM_CONSTANT, height])
    cartesian = []
    for layer in range(4):
        z = 7.5 + layer * spacing
        if layer % 2 == 0:
            cartesian += [[0.0, 0.0, z], [spacing, spacing, z]]
        else:
            cartesian += [[spacing, 0.0, z], [0.0, spacing, z]]
    direct = np.asarray(cartesian, dtype=float) @ np.linalg.inv(lattice)
    return str(write_poscar(path, lattice, ["Al"], [8], direct, comment="Al(100) slab"))


@pytest.fixture()
def workflow(tmp_path) -> Defect:
    return Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))


@pytest.fixture(scope="module")
def aluminium_analysis(tmp_path_factory, aluminium_bulk_path) -> DefectAnalysis:
    workspace = tmp_path_factory.mktemp("defect-analysis")
    tool = Defect(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))
    return DefectAnalysis.from_dict(tool.analyse(aluminium_bulk_path).payload["analysis"])


def test_fcc_aluminium_has_one_inequivalent_atom_site(aluminium_analysis):
    atom_sites = _sites_of_kind(aluminium_analysis, "atom")
    assert len(atom_sites) == 1
    assert atom_sites[0].multiplicity == 4
    assert sorted(atom_sites[0].equivalent_indices) == [1, 2, 3, 4]
    assert aluminium_analysis.structure_kind == "bulk"


def test_atom_site_orbits_partition_the_cell(aluminium_analysis, tmp_path_factory, mos2_poscar):
    workspace = tmp_path_factory.mktemp("defect-partition")
    tool = Defect(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))
    analysis = DefectAnalysis.from_dict(tool.analyse(str(mos2_poscar)).payload["analysis"])
    atom_sites = _sites_of_kind(analysis, "atom")
    collected = sorted(index for site in atom_sites for index in site.equivalent_indices)
    assert collected == [1, 2, 3]
    assert sum(site.multiplicity for site in atom_sites) == 3
    assert sorted(site.species for site in atom_sites) == ["Mo", "S"]


def test_fcc_interstitials_are_the_octahedral_and_tetrahedral_holes(aluminium_analysis, aluminium_bulk_path):
    voids = _sites_of_kind(aluminium_analysis, "interstitial")
    assert len(voids) == 2
    by_radius = sorted(voids, key=lambda site: site.void_radius)
    tetrahedral, octahedral = by_radius

    assert octahedral.void_radius == pytest.approx(ALUMINIUM_CONSTANT / 2.0, abs=1e-3)
    assert octahedral.multiplicity == 4
    assert tetrahedral.void_radius == pytest.approx(ALUMINIUM_CONSTANT * math.sqrt(3.0) / 4.0, abs=1e-3)
    assert tetrahedral.multiplicity == 8

    # Each reported radius must be the true distance to the nearest atom.
    record = io_mod.read_poscar(aluminium_bulk_path)
    hosts = np.asarray(record.positions_direct, dtype=float)
    for site in voids:
        measured = _minimum_distance_to_atoms(record.lattice, hosts, np.asarray(site.direct, dtype=float))
        assert measured == pytest.approx(site.void_radius, abs=1e-3)


def test_vacancy_generation_removes_exactly_one_atom(workflow, aluminium_bulk_path):
    source = io_mod.read_poscar(aluminium_bulk_path)
    result = workflow.generate(aluminium_bulk_path, "vacancy")
    assert result.summary["generated"] == 1

    defected = io_mod.read_poscar(result.payload["generated"][0]["output_path"])
    assert defected.natoms == source.natoms - 1
    assert np.allclose(defected.lattice, source.lattice, atol=1e-12)

    original = np.mod(np.asarray(source.positions_direct, dtype=float), 1.0)
    remaining = np.mod(np.asarray(defected.positions_direct, dtype=float), 1.0)
    for position in remaining:
        difference = original - position
        difference -= np.round(difference)
        assert np.linalg.norm(difference, axis=1).min() < 1e-9


def test_one_vacancy_is_generated_per_inequivalent_species(workflow, mos2_poscar):
    result = workflow.generate(str(mos2_poscar), "vacancy")
    assert result.summary["generated"] == 2
    for entry in result.payload["generated"]:
        assert entry["atom_count"] == 2


def test_substitution_keeps_the_atom_count_and_swaps_one_atom(workflow, aluminium_bulk_path):
    source = io_mod.read_poscar(aluminium_bulk_path)
    result = workflow.generate(aluminium_bulk_path, "substitution", substitution_species="Mg")
    assert result.summary["generated"] == 1

    defected = io_mod.read_poscar(result.payload["generated"][0]["output_path"])
    assert defected.natoms == source.natoms
    assert dict(zip(defected.species, defected.counts)) == {"Al": 3, "Mg": 1}
    assert np.allclose(defected.lattice, source.lattice, atol=1e-12)

    # Only the species labels change; the geometry is untouched.
    original = np.sort(np.mod(np.asarray(source.positions_direct, dtype=float), 1.0), axis=0)
    updated = np.sort(np.mod(np.asarray(defected.positions_direct, dtype=float), 1.0), axis=0)
    assert np.allclose(original, updated, atol=1e-9)


def test_interstitial_insertion_clears_the_reported_void_radius(workflow, aluminium_bulk_path, aluminium_analysis):
    result = workflow.generate(aluminium_bulk_path, "interstitial", species="H")
    voids = {site.site_id: site for site in _sites_of_kind(aluminium_analysis, "interstitial")}
    assert result.summary["generated"] == len(voids)

    for entry in result.payload["generated"]:
        defected = io_mod.read_poscar(entry["output_path"])
        assert defected.natoms == 5
        hosts = np.mod(np.asarray(defected.positions_direct, dtype=float)[:-1], 1.0)
        inserted = np.mod(np.asarray(defected.positions_direct, dtype=float)[-1], 1.0)
        measured = _minimum_distance_to_atoms(defected.lattice, hosts, inserted)
        assert measured == pytest.approx(voids[entry["site_id"]].void_radius, abs=1e-3)


def test_divacancy_removes_two_neighbouring_atoms(workflow, aluminium_bulk_path):
    source = io_mod.read_poscar(aluminium_bulk_path)
    result = workflow.generate(aluminium_bulk_path, "divacancy", divacancy_distance=3.0)
    assert result.summary["generated"] == 1

    entry = result.payload["generated"][0]
    defected = io_mod.read_poscar(entry["output_path"])
    assert defected.natoms == source.natoms - 2

    original = np.mod(np.asarray(source.positions_direct, dtype=float), 1.0)
    remaining = np.mod(np.asarray(defected.positions_direct, dtype=float), 1.0)
    removed = []
    for position in original:
        difference = remaining - position
        difference -= np.round(difference)
        if np.linalg.norm(difference, axis=1).min() > 1e-9:
            removed.append(position)
    assert len(removed) == 2
    separation = _minimum_distance_to_atoms(source.lattice, removed[:1], removed[1])
    assert separation == pytest.approx(ALUMINIUM_CONSTANT / math.sqrt(2.0), abs=1e-6)


def test_adatom_sits_the_requested_height_above_the_surface(workflow, aluminium_slab_path):
    source = io_mod.read_poscar(aluminium_slab_path)
    surface_height = float(np.asarray(source.positions_cartesian, dtype=float)[:, 2].max())
    result = workflow.generate(aluminium_slab_path, "adatom", species="H", height=2.0)
    assert result.summary["structure_kind"] == "surface"
    assert result.summary["generated"] == 3

    families = set()
    for entry in result.payload["generated"]:
        defected = io_mod.read_poscar(entry["output_path"])
        assert defected.natoms == source.natoms + 1
        adatom = np.asarray(defected.positions_cartesian, dtype=float)[-1]
        assert adatom[2] == pytest.approx(surface_height + 2.0, abs=1e-6)
        families.add(entry["site_id"].rsplit("_", 1)[0])
        # The adatom must stay clear of every slab atom.
        hosts = np.mod(np.asarray(defected.positions_direct, dtype=float)[:-1], 1.0)
        inserted = np.mod(np.asarray(defected.positions_direct, dtype=float)[-1], 1.0)
        assert _minimum_distance_to_atoms(defected.lattice, hosts, inserted) > 1.9
    assert families == {"adatom_top", "adatom_bridge", "adatom_fourfold_hollow"}


def test_the_slab_analysis_separates_surface_and_subsurface_layers(workflow, aluminium_slab_path):
    analysis = DefectAnalysis.from_dict(workflow.analyse(aluminium_slab_path).payload["analysis"])
    atom_sites = _sites_of_kind(analysis, "atom")
    assert analysis.structure_kind == "surface"
    assert len(analysis.layers) == 4
    assert len(atom_sites) == 2, "the mirror plane pairs the two surfaces"
    assert sum(site.multiplicity for site in atom_sites) == 8
    collected = sorted(index for site in atom_sites for index in site.equivalent_indices)
    assert collected == list(range(1, 9))


def test_the_analysis_document_round_trips(workflow, aluminium_bulk_path, tmp_path):
    result = workflow.analyse(aluminium_bulk_path)
    stored = json.loads(Path(result.artifacts["analysis_json"]).read_text())
    restored = DefectAnalysis.from_dict(stored)
    assert restored.to_dict() == stored
    assert [site.site_id for site in restored.sites] == [
        site["site_id"] for site in stored["sites"]
    ]


# ---------------------------------------------------------------------------
# separations in a hexagonal cell
# ---------------------------------------------------------------------------


def _graphene_supercell(repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell and fractional sites of an ``n x n`` graphene sheet."""

    constant = 2.46 * repeats
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 15.0],
        ]
    )
    sublattice = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    cells = np.array(
        [[i, j, 0] for i in range(repeats) for j in range(repeats)], dtype=float
    )
    positions = (sublattice[None, :, :] + cells[:, None, :]).reshape(-1, 3)
    return lattice, positions / np.array([repeats, repeats, 1.0])


def test_neighbour_separations_are_exact_in_a_hexagonal_cell():
    """Every reported separation is the true shortest periodic distance.

    Rounding the fractional difference into ``[-1/2, 1/2]`` is exact only for an
    orthogonal cell.  In this hexagonal sheet it overstates several separations,
    which would hide genuine neighbours; the table the defect workflow uses has
    to agree with an explicit search over the surrounding cells.
    """

    from cellstine.defect.analysis import _pairwise_minimum_image_distances

    lattice, positions = _graphene_supercell(3)
    reported = _pairwise_minimum_image_distances(positions, lattice)

    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
    difference = positions[:, None, :] - positions[None, :, :]
    brute = np.min(
        np.linalg.norm((difference[:, :, None, :] - shifts[None, None, :, :]) @ lattice, axis=3),
        axis=2,
    )
    assert np.allclose(reported, brute, atol=1e-9)

    rounded = np.linalg.norm((difference - np.round(difference)) @ lattice, axis=2)
    upper = np.triu_indices(len(positions), 1)
    assert np.any(rounded[upper] > brute[upper] + 0.1), "the shortcut is genuinely wrong here"
    assert (brute[upper] <= 4.0).sum() > (rounded[upper] <= 4.0).sum(), (
        "the exact table must expose the neighbour pairs the shortcut hides"
    )


def test_divacancy_site_sits_at_the_true_pair_midpoint(tmp_path):
    """The reported divacancy site is halfway along the shortest image.

    In a strongly sheared cell, halving the *rounded* fractional difference of a
    pair lands on a point that is not between the two atoms at all -- here it
    would sit 2.05 Angstrom from an atom whose partner is only 2.49 Angstrom
    away -- so the site has to be built from the exact shortest image.
    """

    lattice = np.array([[4.1, 0.0, 0.0], [3.3, 2.9, 0.0], [2.7, 1.9, 3.4]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    path = write_poscar(
        tmp_path / "sheared.vasp",
        lattice,
        ["C"],
        [2],
        positions,
        comment="sheared triclinic pair",
    )
    tool = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    analysis = tool._analyse_record(
        str(path),
        structure_kind="bulk",
        backend="native",
        surface_side="top",
        layer_tolerance=0.35,
        symprec=0.01,
        divacancy_distance=6.0,
    )

    divacancies = _sites_of_kind(analysis, "divacancy")
    assert divacancies, "the two atoms are within the requested divacancy range"
    for site in divacancies:
        first, second = (index - 1 for index in site.pair_indices)
        middle = np.asarray(site.direct, dtype=float)
        separation = _minimum_distance_to_atoms(
            lattice, positions[first : first + 1], positions[second]
        )
        to_first = _minimum_distance_to_atoms(lattice, positions[first : first + 1], middle)
        to_second = _minimum_distance_to_atoms(lattice, positions[second : second + 1], middle)
        assert to_first == pytest.approx(0.5 * separation, abs=1e-6)
        assert to_second == pytest.approx(0.5 * separation, abs=1e-6)


def test_supercell_repeats_are_read_in_every_spelling():
    """One integer, a comma list and an ``x`` product all mean the same repeats."""

    assert _normalise_supercell(None) is None
    assert _normalise_supercell(2) == (2, 2, 2)
    assert _normalise_supercell("3") == (3, 3, 3)
    assert _normalise_supercell("2,2,1") == (2, 2, 1)
    assert _normalise_supercell("2x2x1") == (2, 2, 1)
    assert _normalise_supercell([2, 2, 1]) == (2, 2, 1)
    assert _normalise_supercell([1, 1, 1]) is None, "a 1x1x1 supercell is the cell itself"
    with pytest.raises(ValueError):
        _normalise_supercell([0, 1, 1])
    with pytest.raises(ValueError):
        _normalise_supercell([2, 2])


def test_a_host_supercell_dilutes_the_vacancy_it_carries(workflow, aluminium_bulk_path):
    """``--supercell 3,3,3`` must build 27 cells and remove exactly one atom of them.

    The four-atom fcc cell repeated three times along each axis holds 108 atoms,
    so a single vacancy is a concentration of 1/108 and its nearest periodic
    image is a shortest translation of the tripled *cell*, ``3a`` -- the fcc
    sublattice translations map atoms onto atoms, but only whole-cell
    translations map the vacancy onto a vacancy.
    Every atom of the defected cell must still sit on an ideal lattice site.
    """

    source = io_mod.read_poscar(aluminium_bulk_path)
    result = workflow.generate(aluminium_bulk_path, "vacancy", supercell=[3, 3, 3])

    host = io_mod.read_poscar(result.artifacts["host_supercell"])
    assert host.natoms == 27 * source.natoms
    assert np.allclose(host.lattice, 3.0 * np.asarray(source.lattice, dtype=float), atol=1e-12)

    assert result.summary["generated"] == 1, "all 108 aluminium atoms are equivalent"
    assert result.summary["host_atoms"] == 108
    assert result.summary["defect_concentration_percent"] == pytest.approx(100.0 / 108.0)
    assert result.summary["defect_image_distance"] == pytest.approx(
        3.0 * ALUMINIUM_CONSTANT, abs=1e-9
    )
    assert "warnings" not in result.summary, "a 3x3x3 fcc cell is a dilute vacancy"

    defected = io_mod.read_poscar(result.payload["generated"][0]["output_path"])
    assert defected.natoms == host.natoms - 1
    ideal = np.mod(np.asarray(host.positions_direct, dtype=float), 1.0)
    for position in np.mod(np.asarray(defected.positions_direct, dtype=float), 1.0):
        difference = ideal - position
        difference -= np.round(difference)
        assert np.linalg.norm(difference, axis=1).min() < 1e-9


def test_a_one_by_one_cell_vacancy_is_reported_as_too_concentrated(workflow, aluminium_bulk_path):
    """The undiluted four-atom cell is 25% vacancies, and must say so."""

    result = workflow.generate(aluminium_bulk_path, "vacancy")
    assert "host_supercell" not in result.artifacts
    assert result.summary["defect_concentration_percent"] == pytest.approx(25.0)
    assert result.summary["warnings"], "a quarter of the atoms removed is not a point defect"


def test_the_generated_interstitial_reports_its_true_closest_contact(workflow, aluminium_bulk_path):
    """The reported defect-host contact must be the measured minimum image distance."""

    result = workflow.generate(aluminium_bulk_path, "interstitial", species="H")
    reported = []
    for entry in result.payload["generated"]:
        defected = io_mod.read_poscar(entry["output_path"])
        hosts = np.mod(np.asarray(defected.positions_direct, dtype=float)[:-1], 1.0)
        inserted = np.mod(np.asarray(defected.positions_direct, dtype=float)[-1], 1.0)
        expected = _minimum_distance_to_atoms(defected.lattice, hosts, inserted)
        assert entry["defect_contact"]["distance"] == pytest.approx(expected, abs=1e-9)
        assert set(entry["defect_contact"]["species"]) == {"H", "Al"}
        reported.append(expected)
    assert result.summary["closest_defect_contact"] == pytest.approx(min(reported), abs=1e-4)
    assert result.summary["closest_defect_contact_pair"] in {"H-Al", "Al-H"}


def test_a_crushed_adatom_is_reported_as_overlapping(workflow, aluminium_slab_path):
    """An adatom pushed onto the surface must come back with an overlap warning."""

    result = workflow.generate(aluminium_slab_path, "adatom", species="H", height=0.4)
    assert result.summary["closest_defect_contact"] < 1.0
    warnings = " ".join(result.summary.get("warnings", []))
    assert "overlap" in warnings


def test_a_clear_adatom_carries_no_contact_warning(workflow, aluminium_slab_path):
    """A sensible adatom height must not raise a contact warning."""

    result = workflow.generate(aluminium_slab_path, "adatom", species="H", height=2.0)
    assert result.summary["closest_defect_contact"] > 1.5
    warnings = " ".join(result.summary.get("warnings", []))
    assert "contact" not in warnings


def test_a_vacancy_reports_no_defect_contact(workflow, aluminium_bulk_path):
    """A vacancy adds no atom, so there is no defect-host contact to report."""

    result = workflow.generate(aluminium_bulk_path, "vacancy")
    assert "closest_defect_contact" not in result.summary
    for entry in result.payload["generated"]:
        assert "defect_contact" not in entry


def test_a_substitution_reports_the_contact_of_the_replaced_atom(workflow, aluminium_bulk_path):
    """The substituted atom keeps its site, so its contact is the host spacing."""

    result = workflow.generate(aluminium_bulk_path, "substitution", substitution_species="Mg")
    entry = result.payload["generated"][0]
    assert entry["defect_contact"]["distance"] == pytest.approx(
        ALUMINIUM_CONSTANT / math.sqrt(2.0), abs=1e-9
    )
    assert entry["defect_contact"]["species"] == ["Mg", "Al"]

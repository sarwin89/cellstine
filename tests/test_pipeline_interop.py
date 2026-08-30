"""The four workflows handing structures to one another.

Each test drives a chain of stages the way a user does -- through the argument
parser -- and every POSCAR that comes out of a stage is checked before it is
fed to the next one:

* the comment line is the one provenance line, naming the stage that wrote it;
* the in-plane fractional coordinates lie inside the cell;
* no two atoms, counting periodic images, are on top of each other;
* the cell is right-handed and the counts match the positions;
* a slab keeps the vacuum it was built with, even after atoms are added to it.

Those are the properties the next stage relies on, so a break anywhere in the
chain shows up here rather than in a silently wrong structure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.core.geometry import pairwise_minimum_image_distances
from cellstine.core.provenance import SIGNATURE, STAGE_PREFIX, stage_of
from cellstine.core.vacuum import vacuum_gap
from cellstine.io import native as io_mod

from conftest import write_poscar


def run_cli(*argv: str):
    return execute_namespace(build_parser().parse_args(list(argv)))


@pytest.fixture()
def workspace(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def copper(workspace) -> str:
    """Conventional fcc copper, the usual close-packed test substrate."""

    lattice = 3.615 * np.eye(3)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    return str(write_poscar(workspace / "cu.vasp", lattice, ["Cu"], [4], positions, comment="copper"))


@pytest.fixture()
def graphene(workspace) -> str:
    constant = 2.46
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    return str(write_poscar(workspace / "graphene.vasp", lattice, ["C"], [2], positions, comment="graphene"))


@pytest.fixture()
def hbn(workspace) -> str:
    constant = 2.504
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    return str(write_poscar(workspace / "hbn.vasp", lattice, ["B", "N"], [1, 1], positions, comment="hBN"))


@pytest.fixture()
def carbon_monoxide(workspace) -> str:
    lattice = np.diag([12.0, 12.0, 12.0])
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return str(write_poscar(workspace / "co.vasp", lattice, ["C", "O"], [1, 1], positions, comment="CO"))


def minimum_interatomic_distance(structure) -> float:
    """Return the shortest distance between two atoms, periodic images included."""

    distances = pairwise_minimum_image_distances(
        structure.lattice, np.asarray(structure.positions_direct, dtype=float)
    )
    count = distances.shape[0]
    if count < 2:
        return float("inf")
    off_diagonal = distances[~np.eye(count, dtype=bool)]
    return float(off_diagonal.min())


def assert_clean_structure(path, *, stage: str | None = None, min_distance: float = 0.9):
    """Check the invariants every stage promises of the file it writes."""

    structure = io_mod.read_poscar(str(path))

    comment = str(structure.comment)
    assert comment.startswith(STAGE_PREFIX), comment
    assert comment.endswith(SIGNATURE), comment
    if stage is not None:
        assert stage_of(comment) == stage, comment

    direct = np.asarray(structure.positions_direct, dtype=float)
    assert direct.shape == (sum(structure.counts), 3)
    assert np.all(direct[:, :2] >= -1e-12)
    assert np.all(direct[:, :2] < 1.0 + 1e-12)

    lattice = np.asarray(structure.lattice, dtype=float)
    assert float(np.linalg.det(lattice)) > 0.0

    assert minimum_interatomic_distance(structure) > min_distance
    return structure


def test_a_slab_carries_its_vacuum_through_interface_defect_and_adsorbate(copper, carbon_monoxide):
    """bulk -> surface -> interface -> defect -> adsorbate, with the vacuum intact.

    Each stage receives what the previous one wrote, and the vacuum gap the
    slab was built with survives to the end: the adsorbate and the vacancy do
    not quietly shrink it.
    """

    slab_result = run_cli(
        "interface", "surface", copper, "--miller", "1,1,1", "--layers", "4", "--vacuum", "14",
        "--repeat-a", "2", "--repeat-b", "2",
    )
    slab_path = str(slab_result.artifacts["slab_poscar"])
    slab = assert_clean_structure(slab_path, stage="surface", min_distance=2.0)
    assert vacuum_gap(slab.lattice, slab.positions_cartesian) == pytest.approx(14.0, abs=1e-6)

    built = run_cli(
        "interface", "build", slab_path, slab_path, "--bottom-kind", "slab", "--top-kind", "slab",
        "--gap", "2.1", "--vacuum", "14", "--registry", "fcc",
    )
    interface_path = str(built.artifacts["interface_poscar"])
    interface = assert_clean_structure(interface_path, stage="interface build", min_distance=2.0)
    assert sum(interface.counts) == 32
    # The requested contact is the one that continues the bulk stacking, so the
    # eight layers are one piece of fcc copper.
    assert built.summary["stacking"]["kind"] == "fcc_hollow"
    assert built.summary["stacking"]["registry_requested"] is True
    assert vacuum_gap(interface.lattice, interface.positions_cartesian) == pytest.approx(14.0, abs=1e-6)

    generated = run_cli(
        "defect", "generate", interface_path, "--structure-kind", "slab",
        "--defect-type", "vacancy",
    )
    vacancy_path = str(generated.artifacts["structures"][0])
    vacancy = assert_clean_structure(vacancy_path, stage="defect generate", min_distance=2.0)
    assert sum(vacancy.counts) == 31
    # Removing an atom from inside the stack leaves the vacuum alone.
    assert vacuum_gap(vacancy.lattice, vacancy.positions_cartesian) == pytest.approx(14.0, abs=1e-6)

    # The vacancy breaks the three-fold environment of the hollows next to it,
    # so they are reported as plain hollows rather than fcc or hcp ones.
    placed = run_cli(
        "adsorbate", "place", vacancy_path, carbon_monoxide, "--substrate-kind", "slab",
        "--site-type", "hollow", "--height", "2.0",
    )
    adsorbed = assert_clean_structure(
        str(placed.artifacts["output_poscar"]), stage="adsorbate place", min_distance=1.0
    )
    assert sum(adsorbed.counts) == 33
    # Adding the molecule lengthened c instead of eating into the vacuum.
    assert vacuum_gap(adsorbed.lattice, adsorbed.positions_cartesian) == pytest.approx(14.0, abs=1e-6)


def test_an_adatom_keeps_the_vacuum_of_the_slab_it_sits_on(copper):
    slab_result = run_cli(
        "interface", "surface", copper, "--miller", "1,1,1", "--layers", "5", "--vacuum", "12",
    )
    slab_path = str(slab_result.artifacts["slab_poscar"])

    generated = run_cli(
        "defect", "generate", slab_path, "--structure-kind", "slab", "--defect-type", "adatom",
        "--species", "Cu", "--site-ids", "adatom_fcc_hollow_001", "--height", "2.0",
    )
    adatom = assert_clean_structure(
        str(generated.artifacts["structures"][0]), stage="defect generate", min_distance=2.0
    )
    assert sum(adatom.counts) == 6
    assert vacuum_gap(adatom.lattice, adatom.positions_cartesian) == pytest.approx(12.0, abs=1e-6)

    kept = run_cli(
        "defect", "generate", slab_path, "--structure-kind", "slab", "--defect-type", "adatom",
        "--species", "Cu", "--site-ids", "adatom_fcc_hollow_001", "--height", "2.0",
        "--keep-cell-height",
    )
    unchanged = io_mod.read_poscar(str(kept.artifacts["structures"][0]))
    original = io_mod.read_poscar(slab_path)
    assert np.allclose(unchanged.lattice, original.lattice, atol=1e-12)


def test_a_moire_bilayer_is_accepted_by_the_defect_and_adsorbate_stages(graphene, hbn, carbon_monoxide):
    """moire find -> make -> defect -> adsorbate place -> adsorbate move."""

    found = run_cli(
        "moire", "find", graphene, hbn, "--max-length", "8",
        "--top-strain", "0.02", "--bottom-strain", "0.02", "--preview-limit", "0",
    )
    results_path = Path(found.artifacts["results_json"])
    document = json.loads(results_path.read_text())
    assert document["candidates"]

    made = run_cli(
        "moire", "make", str(results_path), "--indexes", "1",
        "--interlayer-distance", "3.35", "--vacuum", "16",
    )
    moire_path = str(made.artifacts["structures"][0])
    bilayer = assert_clean_structure(moire_path, stage="moire make", min_distance=1.2)
    assert vacuum_gap(bilayer.lattice, bilayer.positions_cartesian) == pytest.approx(16.0, abs=1e-6)

    analysed = run_cli("defect", "analyse", moire_path, "--structure-kind", "slab")
    analysis = json.loads(Path(analysed.artifacts["analysis_json"]).read_text())
    assert len(analysis["layers"]) == 2
    # Both layers of the bilayer are represented among the atom sites.
    atom_layers = {int(site["layer_id"]) for site in analysis["sites"] if site["site_kind"] == "atom"}
    assert atom_layers == {1, 2}

    placed = run_cli(
        "adsorbate", "place", moire_path, carbon_monoxide, "--substrate-kind", "slab",
        "--site-type", "top", "--height", "2.4",
    )
    adsorbed_path = str(placed.artifacts["output_poscar"])
    adsorbed = assert_clean_structure(adsorbed_path, stage="adsorbate place", min_distance=1.0)
    assert sum(adsorbed.counts) == sum(bilayer.counts) + 2
    assert vacuum_gap(adsorbed.lattice, adsorbed.positions_cartesian) == pytest.approx(16.0, abs=1e-6)

    moved = run_cli("adsorbate", "move", adsorbed_path, "--rotate", "30.0")
    turned = assert_clean_structure(
        str(moved.artifacts["output_poscar"]), stage="adsorbate move", min_distance=1.0
    )
    assert sum(turned.counts) == sum(adsorbed.counts)
    assert vacuum_gap(turned.lattice, turned.positions_cartesian) == pytest.approx(16.0, abs=1e-6)


def test_a_defected_slab_can_be_read_back_by_the_symmetry_stage(copper):
    slab_result = run_cli(
        "interface", "surface", copper, "--miller", "1,0,0", "--layers", "4", "--vacuum", "12",
        "--repeat-a", "2", "--repeat-b", "2",
    )
    slab_path = str(slab_result.artifacts["slab_poscar"])
    assert_clean_structure(slab_path, stage="surface", min_distance=2.0)

    generated = run_cli(
        "defect", "generate", slab_path, "--structure-kind", "slab", "--defect-type", "vacancy",
    )
    vacancy_path = str(generated.artifacts["structures"][0])
    vacancy = assert_clean_structure(vacancy_path, stage="defect generate", min_distance=2.0)
    assert sum(vacancy.counts) == 15

    analysed = run_cli("symmetry", "analyse", vacancy_path)
    assert int(analysed.summary["operation_count"]) >= 1


def test_a_matched_interface_is_a_clean_structure(copper, graphene):
    matched = run_cli(
        "interface", "match", copper, graphene, "--bottom-millers", "1,1,1",
        "--top-millers", "0,0,1", "--max-length", "11", "--preview-limit", "0",
    )
    matches_path = Path(matched.artifacts["matches_json"])
    assert json.loads(matches_path.read_text())["matches"]

    built = run_cli(
        "interface", "build", copper, graphene, "--bottom-kind", "bulk", "--top-kind", "slab",
        "--match", str(matches_path), "--match-index", "1", "--gap", "3.0", "--vacuum", "15",
    )
    structure = assert_clean_structure(
        str(built.artifacts["interface_poscar"]), stage="interface match", min_distance=1.2
    )
    assert vacuum_gap(structure.lattice, structure.positions_cartesian) == pytest.approx(15.0, abs=1e-6)

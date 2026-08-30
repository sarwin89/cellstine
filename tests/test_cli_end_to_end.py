"""End-to-end runs of the command line, stage by stage.

Every stage is driven exactly as a user drives it -- through the argument
parser -- and the artifact each stage writes is then fed to the next one.  What
is checked is not that a file appeared but that its contents agree with what the
stage promised: atom counts, cell vectors, heights, and the records written into
the manifests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.io import native as io_mod

from conftest import write_poscar


def run_cli(*argv: str):
    """Execute one command line and return its result object."""

    return execute_namespace(build_parser().parse_args(list(argv)))


@pytest.fixture()
def workspace(tmp_path, monkeypatch) -> Path:
    """Run each test in its own directory: the runs live beside the inputs."""

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def aluminium(workspace) -> str:
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(workspace / "al.vasp", lattice, ["Al"], [4], positions))


@pytest.fixture()
def silicon(workspace) -> str:
    lattice = 5.431 * np.eye(3)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    return str(write_poscar(workspace / "si.vasp", lattice, ["Si"], [8], positions))


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
    return str(write_poscar(workspace / "graphene.vasp", lattice, ["C"], [2], positions))


@pytest.fixture()
def carbon_monoxide(workspace) -> str:
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return str(write_poscar(workspace / "co.vasp", np.diag([12.0, 12.0, 12.0]), ["C", "O"], [1, 1], positions))


def _minimum_image_distance(structure) -> float:
    lattice = np.asarray(structure.lattice, dtype=float)
    direct = np.asarray(structure.positions_direct, dtype=float)
    cartesian = direct @ lattice
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    smallest = math.inf
    for shift in shifts:
        images = (direct + shift) @ lattice
        distances = np.linalg.norm(cartesian[:, None, :] - images[None, :, :], axis=2)
        if not shift.any():
            np.fill_diagonal(distances, math.inf)
        smallest = min(smallest, float(distances.min()))
    return smallest


def test_the_moire_pipeline_runs_from_search_to_structure(graphene, workspace):
    """`moire find` feeds `moire make`, and the built cell is the recorded one."""

    found = run_cli(
        "moire", "find", graphene, graphene, "--max-length", "8", "--top-strain", "0.01",
        "--bottom-strain", "0.01",
    )
    results_path = Path(found.artifacts["results_json"])
    document = json.loads(results_path.read_text())
    assert document["schema"] == "cellstine.moire.gram"
    assert int(found.summary["candidate_count"]) == len(document["candidates"])

    made = run_cli("moire", "make", str(results_path), "--indexes", "1", "--interlayer-distance", "3.35")
    built = io_mod.read_poscar(str(Path(made.artifacts["structures"][0])))
    candidate = document["candidates"][0]

    assert sum(built.counts) == int(candidate["atom_count"])
    assert int(candidate["atom_count"]) == int(candidate["top_atom_count"]) + int(
        candidate["bottom_atom_count"]
    )
    lattice = np.asarray(built.lattice, dtype=float)
    shared = np.asarray(candidate["shared_lattice"], dtype=float)
    assert np.allclose(lattice[:2, :2], shared.T, atol=1e-6) or np.allclose(
        lattice[:2, :2], shared, atol=1e-6
    )
    # A twisted bilayer of graphene keeps its bonds.
    assert _minimum_image_distance(built) == pytest.approx(2.46 / math.sqrt(3.0), rel=2e-2)


def test_the_surface_pipeline_runs_from_slab_to_sites_to_adsorbate(aluminium, carbon_monoxide):
    """A slab built by `interface surface` is accepted by the later stages."""

    slab_result = run_cli("interface", "surface", aluminium, "--miller", "1,1,1", "--layers", "4", "--vacuum", "15")
    slab_path = str(slab_result.artifacts["slab_poscar"])
    slab = io_mod.read_poscar(slab_path)
    assert sum(slab.counts) == 4
    assert slab_result.summary["stacking_sequence"] == "ABCA"

    sites = run_cli("interface", "sites", slab_path)
    report = json.loads(Path(sites.artifacts["sites_json"]).read_text())
    assert {"top", "bridge", "fcc_hollow", "hcp_hollow"} <= set(report["sites"])
    # The close-packed face has one of each hollow and three bridges.
    assert report["site_counts"] == {"top": 1, "bridge": 3, "hcp_hollow": 1, "fcc_hollow": 1}

    placed = run_cli(
        "adsorbate", "place", slab_path, carbon_monoxide, "--substrate-kind", "slab",
        "--site-type", "fcc_hollow", "--height", "2.0",
        "--substrate-repeat-a", "2", "--substrate-repeat-b", "2",
    )
    structure = io_mod.read_poscar(str(placed.artifacts["output_poscar"]))
    assert int(placed.summary["substrate_atom_count"]) == 16
    assert sum(structure.counts) == 18

    labels = [symbol for symbol, count in zip(structure.species, structure.counts) for _ in range(count)]
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    metal = cartesian[[index for index, label in enumerate(labels) if label == "Al"]]
    molecule = cartesian[[index for index, label in enumerate(labels) if label in {"C", "O"}]]
    assert float(molecule[:, 2].min() - metal[:, 2].max()) == pytest.approx(2.0, abs=1e-6)
    # The molecule arrives rigid.
    assert float(np.linalg.norm(molecule[0] - molecule[1])) == pytest.approx(1.128, abs=1e-6)


def test_the_interface_pipeline_matches_and_then_builds(aluminium, silicon):
    """`interface match` writes the supercell that `interface build` consumes."""

    matched = run_cli(
        "interface", "match", aluminium, silicon, "--bottom-millers", "1,1,1", "--top-millers", "1,1,1",
        "--max-length", "13", "--preview-limit", "0",
    )
    matches_path = Path(matched.artifacts["matches_json"])
    document = json.loads(matches_path.read_text())
    best = document["matches"][0]

    built = run_cli(
        "interface", "build", aluminium, silicon, "--bottom-kind", "bulk", "--top-kind", "bulk",
        "--match", str(matches_path), "--match-index", "1", "--gap", "2.5",
    )
    structure = io_mod.read_poscar(str(built.artifacts["interface_poscar"]))

    assert sum(structure.counts) == int(best["total_atoms"])
    # Nothing is stacked on top of anything else.
    assert _minimum_image_distance(structure) > 2.0
    # The matched supercell is the cell that was built.
    lattice = np.asarray(structure.lattice, dtype=float)
    assert float(np.linalg.norm(lattice[0])) == pytest.approx(float(best["cell_a"]), rel=1e-6)
    assert float(np.linalg.norm(lattice[1])) == pytest.approx(float(best["cell_b"]), rel=1e-6)


def test_the_defect_pipeline_analyses_and_then_generates(aluminium):
    """The analysis document drives generation, and a vacancy loses one atom."""

    analysed = run_cli("defect", "analyse", aluminium)
    document = json.loads(Path(analysed.artifacts["analysis_json"]).read_text())
    site_ids = {str(site["site_id"]) for site in document["sites"]}
    assert "atom_001" in site_ids

    generated = run_cli("defect", "generate", aluminium, "--defect-type", "vacancy")
    structures = [Path(path) for path in generated.artifacts["structures"]]
    assert len(structures) == 1
    defect = io_mod.read_poscar(str(structures[0]))
    assert sum(defect.counts) == 3

    inserted = run_cli("defect", "generate", aluminium, "--defect-type", "interstitial", "--species", "H")
    for path in inserted.artifacts["structures"]:
        structure = io_mod.read_poscar(str(path))
        assert sum(structure.counts) == 5
        assert "H" in list(structure.species)
        # The inserted atom lands in a void, not on top of a host atom.
        assert _minimum_image_distance(structure) > 1.5


def test_the_defect_cli_dilutes_a_vacancy_in_a_requested_supercell(aluminium):
    """``--supercell 2x2x2`` makes the host eight times bigger before cutting.

    The vacancy is then one atom in 32 rather than one in four, and the nearest
    copy of it is a whole doubled cell away.
    """

    generated = run_cli(
        "defect", "generate", aluminium, "--defect-type", "vacancy", "--supercell", "2x2x2"
    )
    host = io_mod.read_poscar(str(generated.artifacts["host_supercell"]))
    assert sum(host.counts) == 32

    structures = [Path(path) for path in generated.artifacts["structures"]]
    assert len(structures) == 1
    defect = io_mod.read_poscar(str(structures[0]))
    assert sum(defect.counts) == 31
    assert np.allclose(defect.lattice, host.lattice, atol=1e-12)
    assert float(generated.summary["defect_concentration_percent"]) == pytest.approx(100.0 / 32.0)

    plain = run_cli("defect", "generate", aluminium, "--defect-type", "vacancy")
    assert float(generated.summary["defect_image_distance"]) == pytest.approx(
        2.0 * float(plain.summary["defect_image_distance"])
    )


def test_the_symmetry_stages_reduce_the_conventional_cell(silicon):
    analysed = run_cli("symmetry", "analyse", silicon)
    assert int(analysed.summary["operation_count"]) == 192

    reduced = run_cli("symmetry", "reduce", silicon, "--cell", "primitive")
    primitive = io_mod.read_poscar(str(reduced.artifacts["output_poscar"]))
    assert sum(primitive.counts) == 2
    assert abs(float(np.linalg.det(np.asarray(primitive.lattice, dtype=float)))) == pytest.approx(
        5.431**3 / 4.0, rel=1e-9
    )

    lattice_reduced = run_cli("symmetry", "lattice-reduce", silicon, "--reduction", "niggli")
    niggli = io_mod.read_poscar(str(lattice_reduced.artifacts["output_poscar"]))
    assert sum(niggli.counts) == 8
    assert abs(float(np.linalg.det(np.asarray(niggli.lattice, dtype=float)))) == pytest.approx(
        5.431**3, rel=1e-9
    )


def test_the_multilayer_pipeline_runs_from_search_to_structure(graphene, workspace):
    found = run_cli("moire", "findn", graphene, graphene, graphene, "--max-length", "8")
    results_path = Path(found.artifacts["results_json"])
    document = json.loads(results_path.read_text())
    assert document["candidates"]

    made = run_cli("moire", "maken", str(results_path), "--indexes", "1")
    structure = io_mod.read_poscar(str(Path(made.artifacts["structures"][0])))
    assert sum(structure.counts) == int(document["candidates"][0]["total_atoms"])


def test_every_run_records_a_readable_manifest(aluminium):
    result = run_cli("interface", "surface", aluminium, "--miller", "1,0,0", "--layers", "3", "--vacuum", "12")
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["workflow"] == "interface"
    assert manifest["stage"] == "surface"
    assert Path(str(manifest["artifacts"]["slab_poscar"])).exists()
    assert manifest["parameters"]["vacuum"] == 12.0


def test_the_visualizers_write_the_files_they_promise(graphene, aluminium, workspace):
    """Both rendering backends produce a file for a search result and a slab."""

    pytest.importorskip("matplotlib")

    found = run_cli(
        "moire", "find", graphene, graphene, "--max-length", "8", "--top-strain", "0.01",
        "--bottom-strain", "0.01",
    )
    results_path = str(Path(found.artifacts["results_json"]))
    png = workspace / "moire.png"
    picture = run_cli("moire", "visualize", results_path, "--indices", "1", "--output", str(png))
    assert png.exists() and png.stat().st_size > 0
    assert Path(str(picture.artifacts["png"])) == png

    slab = run_cli("interface", "surface", aluminium, "--miller", "1,1,1", "--layers", "3", "--vacuum", "12")
    slab_png = workspace / "slab.png"
    run_cli("interface", "visualize", str(slab.artifacts["slab_poscar"]), "--output", str(slab_png))
    assert slab_png.exists() and slab_png.stat().st_size > 0

    pytest.importorskip("plotly")
    html = workspace / "slab.html"
    run_cli(
        "interface", "visualize", str(slab.artifacts["slab_poscar"]), "--plotly", "--output", str(html)
    )
    assert html.exists() and html.read_text().lstrip().lower().startswith("<")


def test_candidate_selection_accepts_both_spellings():
    """``--indexes`` and ``--indices`` name the same option on every moire stage."""

    parser = build_parser()
    for stage, destination in (("make", "indexes"), ("maken", "indexes"), ("visualize", "indices")):
        extra = ["--interlayer-distance", "3.35"] if stage == "make" else []
        with_indexes = parser.parse_args(["moire", stage, "results.json", "--indexes", "1,3-4", *extra])
        with_indices = parser.parse_args(["moire", stage, "results.json", "--indices", "1,3-4", *extra])
        assert getattr(with_indexes, destination) == [1, 3, 4]
        assert getattr(with_indices, destination) == [1, 3, 4]

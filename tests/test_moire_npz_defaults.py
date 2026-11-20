"""The default archive works end to end; explicit JSON stays compatible."""
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.parsers import build_parser
from cellstine.core.manifests import RunManifest
from cellstine.core.previews import _preview_moire_results_with_count
from cellstine.io.native import read_poscar
from cellstine.moire.moire import Moire
from cellstine.moire.search.find import _make_result_path, run_find
from cellstine.moire.search.results import read_results, write_results


@pytest.fixture
def layer(tmp_path):
    path = tmp_path / "layer.vasp"
    path.write_text("carbon\n1.0\n2 0 0\n0 2 0\n0 0 12\nC\n1\nDirect\n0 0 0.5\n", encoding="utf-8")
    return path


def test_default_cli_dispatch_uses_npz(monkeypatch):
    import importlib
    main = importlib.import_module("cellstine.cli.main")
    calls = []

    class Workflow:
        def find(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(main, "_workflow", lambda _: Workflow)
    args = build_parser().parse_args(["moire", "search", "top", "bottom", "--length", "2.1", "--rigid"])
    main.execute_namespace(args)
    assert calls[0]["results_format"] == "npz"


def test_default_result_path_is_archive(tmp_path):
    _, path = _make_result_path(str(tmp_path), "bottom.vasp", "top.vasp")
    assert path.name == "results.npz"


def test_direct_default_archive_progress_and_readback(tmp_path, layer):
    progress = []
    run = run_find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                   top_strain=0., bottom_strain=0., reduce_layers=False,
                   output_root=str(tmp_path / "direct"),
                   progress_callback=lambda stage, message: progress.append((stage, message)))
    assert run.result_path.name == "results.npz"
    assert read_results(run.result_path)["candidates"]
    messages = [message for stage, message in progress if stage == "write"]
    assert messages and all("results.npz" in message for message in messages)


@pytest.mark.parametrize("kind", [None, "json"])
def test_default_and_explicit_json_manifest_consumers(tmp_path, layer, kind):
    workflow = Moire(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    options = {} if kind is None else {"results_format": kind}
    found = workflow.find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                          top_strain=0., bottom_strain=0., reduce_layers=False, **options)
    expected = "npz" if kind is None else kind
    artifact = "results_npz" if expected == "npz" else "results_json"
    source = Path(found.artifacts[artifact])
    assert source.name == "results." + expected
    manifest = RunManifest.load(found.manifest_path)
    assert manifest.artifacts[artifact] == str(source)
    assert manifest.parameters["results_format"] == expected
    if expected == "npz":
        assert "results_json" not in found.artifacts
        with np.load(source, allow_pickle=False) as archive:
            assert all(not archive[key].dtype.hasobject for key in archive.files)
    document = read_results(source)
    preview, count = _preview_moire_results_with_count(found.manifest_path)
    assert count == len(document["candidates"]) > 0 and "Saved Gram candidates" in preview
    legacy = write_results(tmp_path / "explicit.json", document)
    assert read_results(legacy) == document
    builds = []
    for name, results_file in (("default", found.manifest_path), ("json", legacy)):
        built = workflow.make(results_file=str(results_file), indexes=[1], interlayer_distance=3.35,
                              output_dir=str(tmp_path / name))
        builds.append(read_poscar(built.artifacts["structures"][0]))
    assert builds[0].species == builds[1].species and builds[0].counts == builds[1].counts
    assert np.array_equal(builds[0].lattice, builds[1].lattice)
    assert np.array_equal(builds[0].positions_direct, builds[1].positions_direct)
    assert np.array_equal(builds[0].positions_cartesian, builds[1].positions_cartesian)


@pytest.mark.parametrize("advanced,kind", [(False, None), (True, "json"), (True, "npz"), (True, "npz-compressed")])
def test_guided_default_and_explicit_choices(monkeypatch, layer, advanced, kind):
    from cellstine.cli.interactive import build_moire
    monkeypatch.setattr(build_moire, "_print_title", lambda *a, **k: None)
    monkeypatch.setattr(build_moire, "_prompt_path", lambda *a, **k: str(layer))
    monkeypatch.setattr(build_moire, "_prompt_float", lambda *a, **k: 2.1)
    monkeypatch.setattr(build_moire, "_prompt_int", lambda *a, **k: 0)
    monkeypatch.setattr(build_moire, "_prompt_int_range", lambda *a, **k: 10)
    answers = iter([True, advanced] + ([False] * 4 if advanced else []))
    monkeypatch.setattr(build_moire, "_prompt_yes_no", lambda *a, **k: next(answers))
    choices = []

    def choose(title, options, default):
        choices.append(options[default - 1]["key"])
        return kind

    monkeypatch.setattr(build_moire, "_choice", choose)
    args = build_parser().parse_args(build_moire._build_moire_find())
    assert args.results_format == (kind or "npz")
    assert choices == (["npz"] if advanced else [])


@pytest.mark.parametrize("kind", [None, "json"])
def test_assembly_default_and_legacy_json_artifacts(tmp_path, layer, kind):
    from cellstine.adsorbate.adsorbate import Adsorbate
    workflow = Adsorbate(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    options = {} if kind is None else {"results_format": kind}
    result = workflow.assemble(substrate_poscar=str(layer), a_length=2., angle_deg=90.,
                               max_length=2.1, top_strain=0., bottom_strain=0., preview_limit=0, **options)
    expected = kind or "npz"
    key = "results_npz" if expected == "npz" else "results_json"
    source = Path(result.artifacts[key])
    assert source.name == "results." + expected
    assert read_results(source)["candidates"]
    manifest = RunManifest.load(result.manifest_path)
    assert manifest.parameters["results_format"] == expected
    assert manifest.artifacts[key] == str(source)
    preview, count = _preview_moire_results_with_count(result.manifest_path)
    assert count > 0 and "Saved Gram candidates" in preview


@pytest.mark.parametrize("kind", [None, "json", "npz-compressed"])
def test_assembly_cli_dispatch_forwards_choice(monkeypatch, kind):
    import importlib
    main = importlib.import_module("cellstine.cli.main")
    calls = []

    class Workflow:
        def assemble(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(main, "_workflow", lambda _: Workflow)
    argv = ["adsorbate", "assemble", "substrate", "--a-length", "2", "--length", "2.1",
            "--top-strain", "0", "--bottom-strain", "0"]
    if kind is not None:
        argv.extend(["--results-format", kind])
    main.execute_namespace(build_parser().parse_args(argv))
    assert calls[0]["results_format"] == (kind or "npz")



def test_interface_json_pointer_still_points_to_json(tmp_path, layer):
    import json
    from cellstine.interface.workflow.lattice_match import MatchRequest, SlabPair, search_slab_pair
    pair = SlabPair(layer, layer, (0, 0, 1), (0, 0, 1), 1, 1)
    path, candidates = search_slab_pair(pair, MatchRequest(max_length=2.1, max_strain=.01),
                                       results_root=tmp_path / "interface")
    assert path.name == "results.json"
    assert json.loads(path.read_text(encoding="utf-8"))["candidates"] == candidates


def test_nlayer_json_schema_accepts_default_archive_pair_results(tmp_path, layer):
    import json
    from cellstine.moire.search.nlayer import run_findn
    run = run_findn(base_poscar=str(layer), upper_poscars=[str(layer), str(layer)], max_length=2.1,
                    layer_strains=0., reduce_layers=False, output_root=str(tmp_path / "stack"))
    document = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert run.result_path.suffix == ".json" and document["candidates"]
    paths = document["search"]["pair_results"]
    assert len(paths) == 2
    for path in paths:
        assert Path(path).suffix == ".npz" and read_results(path)["candidates"]

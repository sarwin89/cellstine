"""Default output names must not overwrite another run's artifacts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cellstine.core import base
from cellstine.io.models import StructureRecord
from cellstine.visualize import visualize as visual


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 11, 2, 12, 0)

    monkeypatch.setattr(base, "datetime", Clock)
    workflow = visual.Visualize(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    monkeypatch.setattr(workflow.dependency_manager, "versions", lambda: {})

    def read(path, **kwargs):
        points = np.zeros((1, 3))
        return StructureRecord(Path(path).stem, np.eye(3) * 4, ["C"], [1], points, points.copy())

    def write(record, *, output_path, **kwargs):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record.comment, encoding="utf-8")
        return output.resolve()

    monkeypatch.setattr(workflow.converter, "read", read)
    monkeypatch.setattr(visual, "write_structure_html", write)
    return workflow


def test_different_inputs_preserve_existing_default_visualizations(viewer):
    first = viewer.structure(structure_path="first.vasp", plotly=True)
    second = viewer.structure(structure_path="second.vasp", plotly=True)
    first_path = Path(first.artifacts["html"])
    second_path = Path(second.artifacts["html"])
    assert first_path.name == "structure_viewer_261102-1200.html"
    assert first_path != second_path
    assert first_path.read_text(encoding="utf-8") == "first"
    assert second_path.read_text(encoding="utf-8") == "second"


def test_run100_collision_keeps_historical_numeric_label_names(viewer, monkeypatch):
    names = iter(["structure_111_261102-1200", "structure_sample_100_261102-1200"])

    def reserve(stage, label=None):
        run_id = next(names)
        directory = viewer.runs_root / viewer.workflow_name / run_id
        directory.mkdir(parents=True)
        return run_id, directory

    monkeypatch.setattr(viewer, "create_run_dir", reserve)
    first = viewer.structure(structure_path="111.vasp", plotly=True)
    second = viewer.structure(structure_path="sample.vasp", plotly=True)
    assert Path(first.artifacts["html"]).name == "structure_viewer_261102-1200.html"
    assert first.artifacts["html"] != second.artifacts["html"]
    assert Path(first.artifacts["html"]).read_text(encoding="utf-8") == "111"


def test_explicit_output_path_retains_its_existing_overwrite_contract(viewer, tmp_path):
    chosen = tmp_path / "chosen.html"
    chosen.write_text("old", encoding="utf-8")
    result = viewer.structure(structure_path="new.vasp", output_path=str(chosen), plotly=True)
    assert Path(result.artifacts["html"]) == chosen
    assert chosen.read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("name", ["Cú.vasp", "KPOINTS"])
def test_default_file_reservation_preserves_existing_bytes(tmp_path, name):
    from cellstine.core.outputs import reserve_default_file

    preferred = tmp_path / name
    first = reserve_default_file(preferred)
    first.write_bytes(b"original")
    second = reserve_default_file(preferred)
    assert first == preferred and second != first
    assert first.read_bytes() == b"original"
    assert second.is_file()
    assert second.name == f"{preferred.stem}_02{preferred.suffix}"


def test_default_file_reservation_is_atomic_between_threads(tmp_path):
    from cellstine.core.outputs import reserve_default_file

    preferred = tmp_path / "nested" / "result.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: reserve_default_file(preferred), range(32)))
    assert len(set(paths)) == 32
    assert all(path.is_file() for path in paths)


def test_default_directory_reservation_preserves_existing_run(tmp_path):
    from cellstine.core.outputs import reserve_default_directory

    preferred = tmp_path / "batch"
    preferred.mkdir()
    marker = preferred / "owner.txt"
    marker.write_text("keep", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: reserve_default_directory(preferred), range(12)))
    assert len(set(paths)) == 12 and preferred not in paths
    assert all(path.is_dir() for path in paths)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_reservation_propagates_parent_failures(tmp_path):
    from cellstine.core.outputs import reserve_default_directory, reserve_default_file

    occupied = tmp_path / "parent"
    occupied.write_text("keep", encoding="utf-8")
    for reserve in [reserve_default_file, reserve_default_directory]:
        with pytest.raises(OSError):
            reserve(occupied / "output")
    assert occupied.read_text(encoding="utf-8") == "keep"


def test_default_file_reservation_preserves_an_occupied_directory(tmp_path):
    from cellstine.core.outputs import reserve_default_file

    preferred = tmp_path / "result.json"
    preferred.mkdir()
    marker = preferred / "owner"
    marker.write_text("keep", encoding="utf-8")
    selected = reserve_default_file(preferred)
    assert selected != preferred and selected.is_file()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_default_file_reservation_does_not_swallow_permission_errors(tmp_path, monkeypatch):
    from cellstine.core.outputs import reserve_default_file

    def denied(path, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError, match="denied"):
        reserve_default_file(tmp_path / "result.json")


def test_public_batch_does_not_reuse_an_existing_default_directory(viewer, monkeypatch):
    stage = importlib.import_module("cellstine.moire.moire")
    builder = importlib.import_module("cellstine.moire.builder.make")
    workflow = stage.Moire(runs_root=viewer.runs_root, output_root=viewer.output_root,
                           dependency_manager=viewer.dependency_manager, backend="native")
    occupied = workflow.output_root / "make_results_261102-1200"
    occupied.mkdir(parents=True)
    original = occupied / "stack_261102-1200.vasp"
    original.write_text("original", encoding="utf-8")

    def generate(results_file, *, output_dir, **kwargs):
        target = Path(output_dir) / "stack.vasp"
        target.write_text("new", encoding="utf-8")
        return [builder.MakeRun(target, 1, 0., 1)]

    monkeypatch.setattr(stage, "generate_many_from_results", generate)
    result = workflow.make(results_file="results.json", indexes=[1], interlayer_distance=3.)
    selected = Path(result.artifacts["structures"][0])
    assert selected.parent != occupied
    assert selected.name == original.name
    assert selected.read_text(encoding="utf-8") == "new"
    assert original.read_text(encoding="utf-8") == "original"


def test_repeated_candidates_in_an_explicit_directory_keep_both_files(tmp_path, monkeypatch):
    builder = importlib.import_module("cellstine.moire.builder.make")
    monkeypatch.setattr(builder.generator_backend, "parse_results", lambda _: (
        "top.vasp", "bottom.vasp", [{"index": 1, "angle_deg": 0.}], {},
    ))
    monkeypatch.setattr(builder.generator_backend, "build_supercell_with_report", lambda *a, **k: (
        np.eye(3), np.zeros((1, 3)), [1], ["C"], None, {},
    ))
    written = []

    def write(path, *args, **kwargs):
        written.append(Path(path))
        Path(path).write_text(str(len(written)), encoding="utf-8")

    monkeypatch.setattr(builder.generator_backend, "write_supercell_poscar", write)
    runs = builder.generate_many_from_results("results.json", indexes=[1, 1],
                                             interlayer_distance=3., output_dir=str(tmp_path))
    assert runs[0].output_path != runs[1].output_path
    assert [run.selected_index for run in runs] == [1, 1]
    assert [run.output_path.read_text(encoding="utf-8") for run in runs] == ["1", "2"]


def test_surface_auto_sibling_preserves_existing_report_with_explicit_slab(tmp_path, monkeypatch):
    backend = importlib.import_module("cellstine.interface.surface.backend")
    points = np.zeros((1, 3))
    record = StructureRecord("surface", np.eye(3) * 4, ["C"], [1], points, points.copy())
    build = SimpleNamespace(structure=record, repeat_a=1, repeat_b=1, supercell_matrix=None)
    monkeypatch.setattr(backend, "build_surface_structure", lambda *a, **k: build)

    def write(path, *args):
        Path(path).write_text("slab", encoding="utf-8")
        return Path(path)

    def sites(record, *, output_path, **kwargs):
        target = Path(output_path)
        target.write_text("new report", encoding="utf-8")
        return SimpleNamespace(output_path=target, site_counts={"atop": 1})

    monkeypatch.setattr(backend, "write_surface_poscar", write)
    monkeypatch.setattr(backend, "find_adsorption_sites", sites)
    slab = tmp_path / "slab.vasp"
    existing = tmp_path / "slab_sites.json"
    existing.write_text("original report", encoding="utf-8")
    result = backend.build_surface("source.vasp", miller=(1, 1, 1), layers=1, vacuum=10.,
                                   output_path=str(slab), analyse_sites=True)
    assert result.output_path == slab
    assert result.site_output_path != existing
    assert result.site_output_path.read_text(encoding="utf-8") == "new report"
    assert existing.read_text(encoding="utf-8") == "original report"

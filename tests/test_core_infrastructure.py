"""Shared workflow boundaries: directory ownership, validation and metadata."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path

import numpy as np
import pytest

from cellstine.core import base, dependencies
from cellstine.core.base import Base, _json_ready, _slug, run_output_suffix
from cellstine.core.manifests import RunManifest
from cellstine.core.validation import ensure_existing_file, ensure_positive


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    class FixedClock:
        @staticmethod
        def now():
            return datetime(2026, 11, 1, 12, 34)

    monkeypatch.setattr(base, "datetime", FixedClock)
    result = Base(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    (result.runs_root / result.workflow_name).mkdir(parents=True)
    return result


def test_run_creation_retries_a_target_reserved_during_mkdir(workflow, monkeypatch):
    target = workflow.runs_root / "base" / "build_sample_261101-1234"
    original = Path.mkdir
    competed = False

    def competing_mkdir(path, *args, **kwargs):
        nonlocal competed
        if path == target and not competed:
            competed = True
            original(path)
            (path / "owner.txt").write_text("another workflow", encoding="utf-8")
            raise FileExistsError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    run_id, directory = workflow.create_run_dir("build", "sample")
    assert run_id == "build_sample_02_261101-1234"
    assert directory.name == run_id and directory.is_dir()
    assert (target / "owner.txt").read_text(encoding="utf-8") == "another workflow"


def test_concurrent_workflows_reserve_distinct_existing_suffixes(workflow):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: workflow.create_run_dir("build", "sample"), range(24)))
    expected = {"build_sample_261101-1234"} | {
        f"build_sample_{number:02d}_261101-1234" for number in range(2, 25)
    }
    assert {run_id for run_id, _ in results} == expected
    assert len({directory for _, directory in results}) == 24
    assert all(directory.is_dir() for _, directory in results)


def test_existing_file_uses_the_same_collision_suffix(workflow):
    occupied = workflow.runs_root / "base" / "build_261101-1234"
    occupied.write_text("preserve", encoding="utf-8")
    run_id, _ = workflow.create_run_dir("build")
    assert run_id == "build_02_261101-1234"
    assert occupied.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("error", [PermissionError("denied"), FileExistsError("parent conflict")])
def test_run_creation_propagates_errors_without_an_occupied_target(workflow, monkeypatch, error):
    original = Path.mkdir

    def failing_mkdir(path, *args, **kwargs):
        if path.parent == workflow.runs_root / "base":
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    with pytest.raises(type(error), match=str(error)):
        workflow.create_run_dir("build")


def test_run_creation_rejects_a_file_in_its_parent_path(tmp_path):
    occupied = tmp_path / "runs"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(OSError):
        Base(runs_root=occupied).create_run_dir("build")
    assert occupied.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("value", [float("nan"), np.float64("nan"), "nan"])
def test_nan_cannot_pass_a_positive_argument_check(value):
    with pytest.raises(ValueError, match="gap must be positive"):
        ensure_positive(value, name="gap")


@pytest.mark.parametrize("value", [0., -0., -1., float("-inf")])
def test_nonpositive_argument_policy_remains(value):
    with pytest.raises(ValueError, match="gap must be positive"):
        ensure_positive(value, name="gap")


@pytest.mark.parametrize("value", [1e-300, 2.5, "3.5", float("inf")])
def test_positive_argument_policy_retains_infinity(value):
    assert ensure_positive(value, name="gap") == float(value)


def test_slug_suffix_and_existing_path_contracts(tmp_path):
    assert _slug(" /Cu (111)/ ") == "Cu__111"
    assert _slug("___") == "run"
    assert _slug("abc_-def", 5) == "abc"
    assert run_output_suffix("build_cu_02_261101-1234") == "02_261101-1234"
    assert run_output_suffix("build_cu_261101-1234") == "261101-1234"
    source = tmp_path / "input.vasp"
    source.write_text("source", encoding="utf-8")
    assert ensure_existing_file(source) == source.resolve()
    with pytest.raises(FileNotFoundError):
        ensure_existing_file(tmp_path / "absent.vasp")


def test_json_ready_preserves_nested_values_and_owned_dataclass_arrays(tmp_path):
    @dataclass
    class Payload:
        coordinates: np.ndarray
        source: Path
        flags: tuple

    array = np.array([[.1, .2, .3]])
    value = Payload(array, tmp_path / "input.vasp", (True, None))
    converted = _json_ready({"item": value, 7: (1, 2)})
    assert json.loads(json.dumps(converted)) == {
        "item": {"coordinates": [[.1, .2, .3]], "source": str(value.source), "flags": [True, None]},
        "7": [1, 2],
    }
    converted["item"]["coordinates"][0][0] = .8
    assert array[0, 0] == .1


def test_dependency_cache_versions_and_backend_errors(monkeypatch):
    lookups = []

    def spec(name):
        lookups.append(name)
        return object() if name in {"numpy", "spglib", "pymatgen"} else None

    def version(name):
        if name == "pymatgen":
            raise metadata.PackageNotFoundError(name)
        return {"numpy": "test-numpy", "spglib": "test-spglib"}[name]

    monkeypatch.setattr(dependencies, "find_spec", spec)
    monkeypatch.setattr(metadata, "version", version)
    manager = dependencies.DependencyManager()
    assert manager.status("numpy") is manager.status("numpy")
    assert lookups == ["numpy"]
    assert manager.versions() == {"numpy": "test-numpy", "spglib": "test-spglib"}
    assert manager.status("pymatgen").available and manager.status("pymatgen").version is None
    assert manager.choose_backend() == "pymatgen"
    assert manager.choose_symmetry_backend() == "spglib"
    assert manager.choose_backend("NATIVE") == "native"
    assert manager.choose_symmetry_backend("native") == "native"
    with pytest.raises(ValueError, match="unsupported backend"):
        manager.choose_backend("missing")
    with pytest.raises(ValueError, match="unsupported symmetry backend"):
        manager.choose_symmetry_backend("missing")
    monkeypatch.setattr(dependencies, "find_spec", lambda _: None)
    missing = dependencies.DependencyManager()
    assert missing.choose_backend() == missing.choose_symmetry_backend() == "native"
    with pytest.raises(RuntimeError, match="pymatgen is required for conversion"):
        missing.choose_backend("pymatgen", feature="conversion")
    with pytest.raises(RuntimeError, match="spglib is required for symmetry"):
        missing.choose_symmetry_backend("spglib", feature="symmetry")


def test_manifest_handoff_and_result_are_json_ready(workflow, monkeypatch):
    monkeypatch.setattr(workflow.dependency_manager, "versions", lambda: {"numpy": "test"})
    assert workflow.choose_backend() in {"native", "pymatgen"}
    run_id, directory = workflow.create_run_dir("build")
    artifact = directory / "structure.vasp"
    path = workflow.write_manifest(
        stage="build", run_id=run_id, run_dir=directory, backend="native",
        inputs={"source": artifact}, parameters={"matrix": np.eye(2)},
        artifacts={"structure": artifact}, summary={"atoms": 2},
    )
    loaded = RunManifest.load(path)
    assert loaded.path == path and loaded.run_id == run_id
    assert loaded.to_dict()["parameters"] == {"matrix": [[1., 0.], [0., 1.]]}
    assert loaded.dependencies == {"numpy": "test"}
    assert "path" not in loaded.to_dict()
    assert datetime.fromisoformat(loaded.created_at).utcoffset().total_seconds() == 0
    assert workflow.resolve_results_file(str(path), ["structure"]) == str(artifact.resolve())
    assert workflow.resolve_results_file(str(artifact), ["structure"]) == str(artifact.resolve())
    with pytest.raises(ValueError, match="requested artifacts: absent"):
        workflow.resolve_results_file(str(path), ["absent"])
    result = workflow.result(manifest_path=path, run_dir=directory,
                             artifacts={"structure": artifact}, summary={"atoms": 2},
                             payload={"values": np.array([1., 2.])})
    assert result.manifest_path == path and result.run_dir == directory.resolve()
    assert result.payload == {"values": [1., 2.]}
    assert result.artifacts == {"structure": str(artifact)}

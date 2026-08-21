from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstine.io import native as io
from cellstine.moire.builder import make
from cellstine.moire.moire import Moire
from cellstine.moire.search import find


TOP_IN_PLANE = np.array([[2.0, 0.3], [0.2, 1.7]], dtype=float)
BOTTOM_IN_PLANE = np.array([[1.96, 0.25], [0.18, 1.72]], dtype=float)


def _write_layers(root: Path) -> tuple[Path, Path]:
    top_path = root / "top.vasp"
    bottom_path = root / "bottom.vasp"
    io.write_poscar(
        str(top_path),
        np.array(
            [
                [TOP_IN_PLANE[0, 0], TOP_IN_PLANE[0, 1], 0.0],
                [TOP_IN_PLANE[1, 0], TOP_IN_PLANE[1, 1], 0.0],
                [0.0, 0.0, 12.0],
            ]
        ),
        np.array([[0.0, 0.0, 0.20], [0.5, 0.5, 0.30]]),
        [2],
        ["T"],
        positions_are_cartesian=False,
        selective_flags=[("F", "F", "F"), ("T", "T", "T")],
    )
    io.write_poscar(
        str(bottom_path),
        np.array(
            [
                [BOTTOM_IN_PLANE[0, 0], BOTTOM_IN_PLANE[0, 1], 0.0],
                [BOTTOM_IN_PLANE[1, 0], BOTTOM_IN_PLANE[1, 1], 0.0],
                [0.0, 0.0, 10.0],
            ]
        ),
        np.array([[0.25, 0.25, 0.40]]),
        [1],
        ["B"],
        positions_are_cartesian=False,
        selective_flags=[("T", "T", "F")],
    )
    return top_path, bottom_path


def _run_find(root: Path):
    top_path, bottom_path = _write_layers(root)
    return find.run_find(
        top_poscar=str(top_path),
        bottom_poscar=str(bottom_path),
        max_length=2.5,
        top_strain=0.08,
        bottom_strain=0.08,
        max_atoms=20,
        fold_symmetry=False,
        output_root=str(root),
    )


def _gram_triple(row_basis: np.ndarray) -> list[float]:
    metric = row_basis @ row_basis.T
    return [float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])]


def _literal_results_payload(top_path: Path, bottom_path: Path) -> dict[str, object]:
    shared_row = np.array([[1.90, 0.20], [0.25, 1.80]], dtype=float)
    shared_columns = shared_row.T
    top_columns = TOP_IN_PLANE.T
    bottom_columns = BOTTOM_IN_PLANE.T
    top_affine = shared_columns @ np.linalg.inv(top_columns)
    bottom_affine = shared_columns @ np.linalg.inv(bottom_columns)
    return {
        "schema": "cellstine.moire.gram",
        "version": 1,
        "search": {
            "top_poscar": str(top_path.resolve()),
            "bottom_poscar": str(bottom_path.resolve()),
            "max_length": 2.5,
            "top_strain": 0.08,
            "bottom_strain": 0.08,
            "min_length": None,
            "max_atoms": 20,
            "top_atoms": 2,
            "bottom_atoms": 1,
            "max_aspect_ratio": 12.0,
            "min_cell_angle_deg": 25.0,
            "max_cell_angle_deg": 155.0,
            "fold_symmetry": False,
            "symmetric": False,
        },
        "metadata": {
            "created_at": "2026-08-21T00:00:00+00:00",
            "engine": "gram-v1",
            "symmetric_requested": False,
            "symmetric_used": False,
            "symmetric_fallback": None,
            "stage_stats": {"branch": "literal-test", "n_accepted": 1},
        },
        "candidates": [
            {
                "index": 1,
                "top_matrix": [[1, 0], [0, 1]],
                "bottom_matrix": [[1, 0], [0, 1]],
                "top_gram": _gram_triple(TOP_IN_PLANE),
                "bottom_gram": _gram_triple(BOTTOM_IN_PLANE),
                "angle_deg": 37.0,
                "strain": [0.02, -0.01],
                "top_strain": 0.08,
                "bottom_strain": 0.08,
                "sharing_fraction": 0.5,
                "top_atom_count": 2,
                "bottom_atom_count": 1,
                "atom_count": 3,
                "loewner_certified": True,
                "loewner_borderline": False,
                "rank": 1,
                "pareto_optimal": True,
                "top_affine": top_affine.tolist(),
                "bottom_affine": bottom_affine.tolist(),
                "shared_lattice": shared_columns.tolist(),
            }
        ],
    }


def test_results_json_v1_roundtrips_complete_native_search(tmp_path: Path):
    from cellstine.moire.search.results import read_results, write_results

    run = _run_find(tmp_path)
    payload = read_results(run.result_path)

    assert payload["schema"] == "cellstine.moire.gram"
    assert payload["version"] == 1
    assert payload["search"] == {
        "top_poscar": str((tmp_path / "top.vasp").resolve()),
        "bottom_poscar": str((tmp_path / "bottom.vasp").resolve()),
        "max_length": 2.5,
        "top_strain": 0.08,
        "bottom_strain": 0.08,
        "min_length": None,
        "max_atoms": 20,
        "top_atoms": 2,
        "bottom_atoms": 1,
        "max_aspect_ratio": 12.0,
        "min_cell_angle_deg": 25.0,
        "max_cell_angle_deg": 155.0,
        "fold_symmetry": False,
        "symmetric": False,
    }
    assert payload["metadata"]["engine"] == "gram-v1"
    assert payload["metadata"]["created_at"]
    assert payload["metadata"]["stage_stats"]["n_accepted"] == len(run.result)

    candidate = payload["candidates"][0]
    assert set(candidate) == {
        "index",
        "top_matrix",
        "bottom_matrix",
        "top_gram",
        "bottom_gram",
        "angle_deg",
        "strain",
        "top_strain",
        "bottom_strain",
        "sharing_fraction",
        "top_atom_count",
        "bottom_atom_count",
        "atom_count",
        "loewner_certified",
        "loewner_borderline",
        "rank",
        "pareto_optimal",
        "top_affine",
        "bottom_affine",
        "shared_lattice",
    }
    assert candidate["index"] == 1
    assert np.asarray(candidate["top_matrix"]).shape == (2, 2)
    assert np.asarray(candidate["bottom_matrix"]).shape == (2, 2)
    assert np.asarray(candidate["top_affine"]).shape == (2, 2)
    assert np.asarray(candidate["bottom_affine"]).shape == (2, 2)
    assert np.asarray(candidate["shared_lattice"]).shape == (2, 2)
    assert len(candidate["top_gram"]) == len(candidate["bottom_gram"]) == 3
    assert len(candidate["strain"]) == 2

    copy_path = tmp_path / "copy.json"
    write_results(copy_path, payload)
    assert read_results(copy_path) == payload


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload.update(schema="wrong.schema"), "schema"),
        (lambda payload: payload["candidates"][0].update(top_matrix=[[1, 0, 0], [0, 1, 0]]), "top_matrix"),
    ],
)
def test_results_reader_validates_schema_and_candidate_shapes(tmp_path: Path, mutation, message: str):
    from cellstine.moire.search.results import read_results

    top_path, bottom_path = _write_layers(tmp_path)
    payload = _literal_results_payload(top_path, bottom_path)
    mutation(payload)
    result_path = tmp_path / "invalid.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_results(result_path)


def test_results_reader_rejects_legacy_dat_with_migration_guidance(tmp_path: Path):
    from cellstine.moire.search.results import read_results

    legacy_path = tmp_path / "results.dat"
    legacy_path.write_text("top.vasp bottom.vasp\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"rerun .*moire find"):
        read_results(legacy_path)


def test_native_poscar_row_bases_keep_top_bottom_and_matrix_orientation(tmp_path: Path):
    from cellstine.moire.search.results import read_results

    run = _run_find(tmp_path)
    candidate = read_results(run.result_path)["candidates"][0]
    top_matrix = np.asarray(candidate["top_matrix"], dtype=int)
    bottom_matrix = np.asarray(candidate["bottom_matrix"], dtype=int)
    top_affine = np.asarray(candidate["top_affine"], dtype=float)
    bottom_affine = np.asarray(candidate["bottom_affine"], dtype=float)
    shared_row = np.asarray(candidate["shared_lattice"], dtype=float).T

    top_transformed = top_matrix @ TOP_IN_PLANE @ top_affine.T
    bottom_transformed = bottom_matrix @ BOTTOM_IN_PLANE @ bottom_affine.T
    assert np.allclose(top_transformed, shared_row, atol=1e-10)
    assert np.allclose(bottom_transformed, shared_row, atol=1e-10)


def test_json_make_uses_recorded_affines_and_retains_structure_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from cellstine.moire.search import gram

    top_path, bottom_path = _write_layers(tmp_path)
    payload = _literal_results_payload(top_path, bottom_path)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    def search_must_not_run(_config):
        raise AssertionError("make must not rerun the moire search")

    monkeypatch.setattr(gram, "search", search_must_not_run)
    output_path = tmp_path / "stack.vasp"
    make_run = make.generate_from_results(
        str(results_path),
        index=1,
        interlayer_distance=3.25,
        output_path=str(output_path),
    )

    built = io.read_poscar(str(make_run.output_path))
    candidate = payload["candidates"][0]
    assert np.allclose(built.lattice[:2, :2], np.asarray(candidate["shared_lattice"]).T, atol=1e-10)
    assert built.natoms == candidate["atom_count"] == 3
    assert built.selective_dynamics
    assert built.selective_flags == [("F", "F", "F"), ("T", "T", "T"), ("T", "T", "F")]
    top_z = built.positions_cartesian[: candidate["top_atom_count"], 2]
    bottom_z = built.positions_cartesian[candidate["top_atom_count"] :, 2]
    assert float(np.min(top_z) - np.max(bottom_z)) == pytest.approx(3.25, abs=1e-10)


def test_json_make_rejects_disagreeing_recorded_layer_lattices(tmp_path: Path):
    top_path, bottom_path = _write_layers(tmp_path)
    payload = _literal_results_payload(top_path, bottom_path)
    payload["candidates"][0]["bottom_affine"][0][0] += 0.1
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"transformed.*lattices.*agree"):
        make.generate_from_results(
            str(results_path),
            index=1,
            interlayer_distance=3.25,
            output_path=str(tmp_path / "must-not-exist.vasp"),
        )


def test_moire_manifests_handoff_results_json_not_dat(tmp_path: Path):
    top_path, bottom_path = _write_layers(tmp_path)
    tool = Moire(runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = tool.find(
        top_poscar=str(top_path),
        bottom_poscar=str(bottom_path),
        max_length=2.5,
        top_strain=0.08,
        bottom_strain=0.08,
        max_atoms=20,
        fold_symmetry=False,
        preview_limit=0,
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert set(result.artifacts) == {"results_json"}
    assert set(manifest["artifacts"]) == {"results_json"}
    assert Path(result.artifacts["results_json"]).name == "results.json"

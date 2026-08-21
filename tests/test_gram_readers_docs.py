from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstine.core.previews import preview_moire_results_file
from cellstine.io import native as io
from cellstine.moire.search import find
from cellstine.visualize.backends.matplotlib import (
    _read_moire_summary,
    plot_moire_summary,
)
from cellstine.visualize.results.plotly import build_visualization


ROOT = Path(__file__).resolve().parents[1]


def _write_layers(root: Path) -> tuple[Path, Path]:
    top_path = root / "top.vasp"
    bottom_path = root / "bottom.vasp"
    io.write_poscar(
        str(top_path),
        np.array([[2.0, 0.3, 0.0], [0.2, 1.7, 0.0], [0.0, 0.0, 12.0]]),
        np.array([[0.0, 0.0, 0.20], [0.5, 0.5, 0.30]]),
        [2],
        ["T"],
        positions_are_cartesian=False,
    )
    io.write_poscar(
        str(bottom_path),
        np.array([[1.96, 0.25, 0.0], [0.18, 1.72, 0.0], [0.0, 0.0, 10.0]]),
        np.array([[0.25, 0.25, 0.40]]),
        [1],
        ["B"],
        positions_are_cartesian=False,
    )
    return top_path, bottom_path


@pytest.fixture
def gram_run(tmp_path: Path):
    top_path, bottom_path = _write_layers(tmp_path)
    return find.run_find(
        top_poscar=str(top_path),
        bottom_poscar=str(bottom_path),
        max_length=2.5,
        top_strain=0.08,
        bottom_strain=0.08,
        max_atoms=20,
        fold_symmetry=False,
        output_root=str(tmp_path / "search"),
    )


def test_preview_reports_complete_gram_candidate_and_search_metadata(gram_run):
    preview = preview_moire_results_file(gram_run.result_path, limit=1)

    for label in (
        "angle (deg)",
        "relative principal strain",
        "top strain",
        "bottom strain",
        "top/bottom/total atoms",
        "rank",
        "Pareto",
        "certification",
        "top matrix",
        "bottom matrix",
        "shared lattice",
        "engine=gram-v1",
        "max length=2.5 Angstrom",
    ):
        assert label in preview


def test_static_reader_preserves_native_gram_fields_and_plot(gram_run, tmp_path: Path):
    results_type, rows, payload = _read_moire_summary(gram_run.result_path)
    candidate = payload["candidates"][0]

    assert results_type == "bilayer"
    assert rows[0] == {
        "index": candidate["index"],
        "angle_deg": candidate["angle_deg"],
        "relative_principal_strain": candidate["strain"],
        "top_strain": candidate["top_strain"],
        "bottom_strain": candidate["bottom_strain"],
        "top_atom_count": candidate["top_atom_count"],
        "bottom_atom_count": candidate["bottom_atom_count"],
        "atom_count": candidate["atom_count"],
        "rank": candidate["rank"],
        "pareto_optimal": candidate["pareto_optimal"],
        "loewner_certified": candidate["loewner_certified"],
        "loewner_borderline": candidate["loewner_borderline"],
        "top_matrix": candidate["top_matrix"],
        "bottom_matrix": candidate["bottom_matrix"],
        "shared_lattice": candidate["shared_lattice"],
    }

    output_path = tmp_path / "summary.png"
    run = plot_moire_summary(gram_run.result_path, output_path=output_path)
    assert run.visualization_type == "bilayer_summary"
    assert run.item_count == len(payload["candidates"])
    assert output_path.stat().st_size > 0


def test_plotly_reader_builds_native_gram_frames_with_candidate_provenance(
    gram_run, tmp_path: Path
):
    output_path = tmp_path / "viewer.html"
    run = build_visualization(
        str(gram_run.result_path), indices=[1], output_path=str(output_path)
    )

    assert run.results_type == "bilayer"
    assert run.frame_count == 1
    html = output_path.read_text(encoding="utf-8")
    for label in (
        "relative principal strain",
        "top strain",
        "bottom strain",
        "top matrix",
        "bottom matrix",
        "shared lattice",
        "rank 1",
        "Pareto",
        "Loewner certified",
        "gram-v1",
    ):
        assert label in html


@pytest.mark.parametrize(
    "reader",
    [
        lambda path, output: preview_moire_results_file(path),
        lambda path, output: _read_moire_summary(path),
        lambda path, output: build_visualization(path, output_path=str(output)),
    ],
    ids=["preview", "matplotlib", "plotly"],
)
def test_all_moire_readers_reject_dat_with_native_rerun_guidance(
    tmp_path: Path, reader
):
    dat_path = tmp_path / "results.dat"
    dat_path.write_text("legacy positional results\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"rerun .*moire find"):
        reader(str(dat_path), tmp_path / "unused.html")


def test_public_moire_docs_describe_only_the_native_json_workflow():
    paths = [
        ROOT / "README.md",
        ROOT / "USAGE_GUIDE.md",
        ROOT / "src" / "cellstine" / "moire" / "MOIRE_SEARCH.md",
    ]
    documents = {path.name: path.read_text(encoding="utf-8") for path in paths}

    for name, text in documents.items():
        for required in (
            "--max-length",
            "--top-strain",
            "--bottom-strain",
            "results.json",
            "moire make",
        ):
            assert required in text, f"{name} is missing {required}"
        for retired in ("--nindex", "--angles", "moire findn", "moire maken"):
            assert retired not in text, f"{name} still recommends {retired}"

    combined = "\n".join(documents.values())
    normalized = re.sub(r"[`*_]", "", combined).lower()
    assert "principal logarithmic strain" in normalized
    assert "h = log(lambda)" in normalized
    assert "sum of the top and bottom strain budgets" in normalized
    assert "shares" in normalized and "optimally" in normalized
    assert "scientifically precise" in normalized and "cli readable" in normalized
    assert "restricted" in normalized and "square" in normalized and "hexagonal" in normalized
    assert "falls back to the general search" in normalized
    assert "n-layer moire workflows are not supported in this release" in normalized
    assert "aristotle" in normalized and "lean" in normalized
    assert "external mathematical reference" in normalized
    assert "python benchmarks/benchmark_gram_search.py" in combined
    assert "independent canonical candidate classes" in normalized
    assert "stops on mismatch" in normalized
    assert "three increasing length bounds" in normalized
    assert "host-dependent" in normalized

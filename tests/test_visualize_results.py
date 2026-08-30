"""The result visualizers must show exactly the structure the builder writes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from cellstine.io import native as io
from cellstine.moire.moire import Moire
from cellstine.visualize.visualize import Visualize

INTERLAYER = 3.35


@pytest.fixture(scope="module")
def graphene_results(tmp_path_factory, graphene_poscar):
    workspace = tmp_path_factory.mktemp("visualize-run")
    workflow = Moire(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    found = workflow.find(
        top_poscar=str(graphene_poscar),
        bottom_poscar=str(graphene_poscar),
        max_length=12.0,
        top_strain=0.01,
        bottom_strain=0.01,
        preview_limit=0,
    )
    return workspace, workflow, found.artifacts["results_json"]


def _frames_of(html: str) -> list[dict]:
    match = re.search(r"const frames = (\[.*?\]);\n", html, flags=re.S)
    assert match is not None, "the viewer must embed its frames as JSON"
    return json.loads(match.group(1))


def _frame_positions(frame: dict) -> np.ndarray:
    points: list[list[float]] = []
    for trace in frame["data"]:
        if trace["mode"] != "markers":
            continue
        points.extend(
            [x, y, z] for x, y, z in zip(trace["x"], trace["y"], trace["z"])
        )
    return np.array(sorted(points), dtype=float)


def test_plotly_viewer_matches_the_written_poscar(graphene_results):
    workspace, workflow, results_json = graphene_results
    candidate_index = 1

    made = workflow.make(
        results_file=results_json,
        indexes=[candidate_index],
        interlayer_distance=INTERLAYER,
    )
    structure = io.read_poscar(str(Path(made.artifacts["structures"][0])))
    poscar_cartesian = np.array(
        sorted((structure.positions_direct @ structure.lattice).tolist()), dtype=float
    )

    visualizer = Visualize(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    shown = visualizer.moire_results(
        results_file=results_json,
        indices=[candidate_index],
        interlayer=INTERLAYER,
        plotly=True,
    )
    html = Path(shown.artifacts["html"]).read_text(encoding="utf-8")

    frames = _frames_of(html)
    assert len(frames) == 1
    assert shown.summary["frame_count"] == 1

    viewer_cartesian = _frame_positions(frames[0])
    assert viewer_cartesian.shape == poscar_cartesian.shape
    assert np.allclose(viewer_cartesian, poscar_cartesian, atol=1e-8)


def test_plotly_viewer_reports_the_recorded_candidate(graphene_results):
    workspace, _, results_json = graphene_results
    payload = json.loads(Path(results_json).read_text(encoding="utf-8"))
    record = next(
        candidate for candidate in payload["candidates"] if int(candidate["index"]) == 1
    )

    visualizer = Visualize(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    shown = visualizer.moire_results(
        results_file=results_json, indices=[1], interlayer=INTERLAYER, plotly=True
    )
    frame = _frames_of(Path(shown.artifacts["html"]).read_text(encoding="utf-8"))[0]

    subtitle = frame["layout"]["title"]["text"]
    assert f"{float(record['angle_deg']):.4f} degrees" in subtitle
    assert (
        f"{int(record['top_atom_count'])}/{int(record['bottom_atom_count'])}/"
        f"{int(record['atom_count'])}" in subtitle
    )
    assert len(_frame_positions(frame)) == int(record["atom_count"])

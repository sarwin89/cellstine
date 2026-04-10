"""Quick HTML visualizers for results files and POSCAR structures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.base import Base, legacy_modules
from ..core.models import CommandResult
from ..io.converters import StructureConverter

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _cell_outline_points(lattice: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    origin = np.zeros(3, dtype=float)
    a_vec = np.asarray(lattice[0], dtype=float)
    b_vec = np.asarray(lattice[1], dtype=float)
    c_vec = np.asarray(lattice[2], dtype=float)
    corners = {
        "000": origin,
        "100": a_vec,
        "010": b_vec,
        "001": c_vec,
        "110": a_vec + b_vec,
        "101": a_vec + c_vec,
        "011": b_vec + c_vec,
        "111": a_vec + b_vec + c_vec,
    }
    edges = [
        ("000", "100"), ("000", "010"), ("000", "001"),
        ("100", "110"), ("100", "101"),
        ("010", "110"), ("010", "011"),
        ("001", "101"), ("001", "011"),
        ("110", "111"), ("101", "111"), ("011", "111"),
    ]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for start, end in edges:
        xs.extend([float(corners[start][0]), float(corners[end][0]), None])
        ys.extend([float(corners[start][1]), float(corners[end][1]), None])
        zs.extend([float(corners[start][2]), float(corners[end][2]), None])
    return xs, ys, zs


class Visualize(Base):
    """Shared visualizer for grouped workflows."""

    workflow_name = "visualize"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)

    def moire_results(self, *, results_file: str, output_path: str | None = None, **kwargs) -> CommandResult:
        backend = self.choose_backend(feature="visualize.moire")
        run_id, run_dir = self.create_run_dir("moire", Path(results_file).stem)
        run = legacy_modules().visualize_stage.build_visualization(
            str(Path(results_file).resolve()),
            output_path=output_path or str(self.output_root / f"{Path(results_file).stem}_viewer.html"),
            **kwargs,
        )
        manifest_path = self.write_manifest(
            stage="moire",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(results_file).resolve())},
            parameters=kwargs,
            artifacts={"html": run.output_path},
            summary={"frame_count": run.frame_count, "results_type": run.results_type},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"html": run.output_path},
            summary={"frame_count": run.frame_count, "results_type": run.results_type},
        )

    def structure(self, *, structure_path: str, output_path: str | None = None, title: str | None = None) -> CommandResult:
        backend = self.choose_backend(feature="visualize.structure")
        run_id, run_dir = self.create_run_dir("structure", Path(structure_path).stem)
        record = self.converter.read(structure_path, canonicalize=True)
        output = Path(output_path).resolve() if output_path is not None else (self.output_root / f"{Path(structure_path).stem}_structure.html")
        output.parent.mkdir(parents=True, exist_ok=True)
        species = []
        for symbol, count in zip(record.species, record.counts):
            species.extend([symbol] * int(count))
        xs, ys, zs = _cell_outline_points(record.lattice)
        payload = {
            "title": title or record.comment,
            "atoms": {
                "x": [float(value) for value in record.positions_cartesian[:, 0]],
                "y": [float(value) for value in record.positions_cartesian[:, 1]],
                "z": [float(value) for value in record.positions_cartesian[:, 2]],
                "text": species,
            },
            "cell": {"x": xs, "y": ys, "z": zs},
        }
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{payload["title"]}</title>
  <script src="{PLOTLY_CDN}"></script>
</head>
<body>
  <div id="viewer" style="width:100%;height:90vh;"></div>
  <script>
    const payload = {json.dumps(payload)};
    const traces = [
      {{
        type: "scatter3d",
        mode: "markers",
        x: payload.atoms.x,
        y: payload.atoms.y,
        z: payload.atoms.z,
        text: payload.atoms.text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ size: 4, color: "#d1495b" }}
      }},
      {{
        type: "scatter3d",
        mode: "lines",
        x: payload.cell.x,
        y: payload.cell.y,
        z: payload.cell.z,
        line: {{ width: 4, color: "#264653" }},
        hoverinfo: "skip"
      }}
    ];
    Plotly.newPlot("viewer", traces, {{
      title: payload.title,
      scene: {{
        aspectmode: "data",
        xaxis: {{ title: "x (A)" }},
        yaxis: {{ title: "y (A)" }},
        zaxis: {{ title: "z (A)" }}
      }},
      margin: {{ l: 0, r: 0, t: 40, b: 0 }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>"""
        output.write_text(html, encoding="utf-8")
        manifest_path = self.write_manifest(
            stage="structure",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"structure_path": str(Path(structure_path).resolve())},
            artifacts={"html": output},
            summary={"atom_count": record.natoms},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"html": output},
            summary={"atom_count": record.natoms},
        )

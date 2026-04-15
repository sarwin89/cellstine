"""Optional Plotly-style HTML visualizations for CELLSTINE."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

from ..io.models import StructureRecord
from .matplotlib_backend import _atomic_radius

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
        ("000", "100"),
        ("000", "010"),
        ("000", "001"),
        ("100", "110"),
        ("100", "101"),
        ("010", "110"),
        ("010", "011"),
        ("001", "101"),
        ("001", "011"),
        ("110", "111"),
        ("101", "111"),
        ("011", "111"),
    ]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for start, end in edges:
        xs.extend([float(corners[start][0]), float(corners[end][0]), None])
        ys.extend([float(corners[start][1]), float(corners[end][1]), None])
        zs.extend([float(corners[start][2]), float(corners[end][2]), None])
    return xs, ys, zs


def _expanded_species(record: StructureRecord) -> list[str]:
    species: list[str] = []
    for symbol, count in zip(record.species, record.counts):
        species.extend([str(symbol)] * int(count))
    if len(species) < record.natoms:
        species.extend(["X"] * (record.natoms - len(species)))
    return species[: record.natoms]


def write_structure_html(record: StructureRecord, *, output_path: str | Path, title: str | None = None) -> Path:
    """Write a lightweight Plotly CDN HTML viewer for one structure."""

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, zs = _cell_outline_points(record.lattice)
    payload = {
        "title": title or record.comment or "CELLSTINE structure",
        "atoms": {
            "x": [float(value) for value in record.positions_cartesian[:, 0]],
            "y": [float(value) for value in record.positions_cartesian[:, 1]],
            "z": [float(value) for value in record.positions_cartesian[:, 2]],
            "text": _expanded_species(record),
        },
        "cell": {"x": xs, "y": ys, "z": zs},
    }
    payload["atoms"]["size"] = [max(7.0, min(22.0, 9.0 * _atomic_radius(symbol))) for symbol in payload["atoms"]["text"]]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(payload["title"]))}</title>
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
        name: "atoms",
        x: payload.atoms.x,
        y: payload.atoms.y,
        z: payload.atoms.z,
        text: payload.atoms.text,
        hovertemplate: "%{{text}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
        marker: {{ size: payload.atoms.size, sizemode: "diameter", color: "#d1495b", opacity: 0.9 }}
      }},
      {{
        type: "scatter3d",
        mode: "lines",
        name: "unit cell",
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
        xaxis: {{ title: "x (Angstrom)" }},
        yaxis: {{ title: "y (Angstrom)" }},
        zaxis: {{ title: "z (Angstrom)" }}
      }},
      legend: {{ orientation: "h" }},
      margin: {{ l: 0, r: 0, t: 48, b: 0 }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>"""
    output.write_text(html, encoding="utf-8")
    return output

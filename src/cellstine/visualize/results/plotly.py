"""Plotly-based HTML visualizer for commensurate CELLSTINE results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ...moire.builder import generator as generator_backend
from ...moire.search.results import read_results

DEFAULT_OUTPUT_DIR = Path("output")
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


@dataclass
class VisualizationRun:
    output_path: Path
    frame_count: int
    results_type: str


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


def _peak_percent(values: object) -> float:
    """Return the largest principal strain magnitude of a layer, in percent."""

    entries = [abs(float(value)) for value in list(values or [0.0])]
    return 100.0 * (max(entries) if entries else 0.0)


def _build_bilayer_frame(
    top_poscar: str,
    bottom_poscar: str,
    record: Dict[str, object],
    *,
    interlayer: float,
    top_c_repeat: int,
    bottom_c_repeat: int,
    search: Dict[str, object],
    metadata: Dict[str, object],
) -> dict[str, object]:
    stack = generator_backend.build_candidate_layers(
        top_poscar,
        bottom_poscar,
        record,
        interlayer_distance=float(interlayer),
        repeat_top_c=int(top_c_repeat),
        repeat_bottom_c=int(bottom_c_repeat),
    )
    final_lattice = stack.lattice
    index = int(record["index"])
    angle_deg = float(record["angle_deg"])
    strain = [100.0 * float(value) for value in record["strain"]]
    certification = (
        "Loewner borderline"
        if bool(record["loewner_borderline"])
        else "Loewner certified"
        if bool(record["loewner_certified"])
        else "Loewner uncertified"
    )
    pareto = "Pareto" if bool(record["pareto_optimal"]) else "non-Pareto"
    subtitle = "<br>".join(
        [
            f"Candidate {index} at {angle_deg:.4f} degrees; rank {int(record['rank'])}; {pareto}; {certification}",
            f"relative principal strain = ({strain[0]:+.4f}%, {strain[1]:+.4f}%); "
            f"top layer strain = {_peak_percent(record['top_layer_strain']):.4f}%; "
            f"bottom layer strain = {_peak_percent(record['bottom_layer_strain']):.4f}%",
            f"moire cell = {float(record['moire_a']):.3f} x {float(record['moire_b']):.3f} Angstrom "
            f"at {float(record['moire_gamma_deg']):.2f} degrees; "
            f"coincidence index = {int(record['coincidence_index'])}",
            f"top/bottom/total atoms = {int(record['top_atom_count'])}/"
            f"{int(record['bottom_atom_count'])}/{int(record['atom_count'])}",
            f"top matrix = {json.dumps(record['top_matrix'], separators=(',', ':'))}; "
            f"bottom matrix = {json.dumps(record['bottom_matrix'], separators=(',', ':'))}",
            f"shared lattice = {json.dumps(record['shared_lattice'], separators=(',', ':'))}",
            f"engine {metadata['engine']}; max length {float(search['max_length']):g} Angstrom; "
            f"symmetric used={metadata['symmetric_used']}; "
            f"fallback={metadata['symmetric_fallback'] or 'none'}",
        ]
    )
    return {
        "name": f"candidate {index} | {angle_deg:.4f} deg | rank {int(record['rank'])}",
        "lattice": final_lattice.tolist(),
        "layers": [
            {"label": "Bottom", "color": "#264653", "positions": [position.tolist() for _, position, _ in stack.bottom_atoms]},
            {"label": "Top", "color": "#e76f51", "positions": [position.tolist() for _, position, _ in stack.top_atoms]},
        ],
        "subtitle": subtitle,
    }


def _marker_size(frames: Sequence[dict[str, object]]) -> float:
    atom_count = 1
    if frames:
        atom_count = max(1, sum(len(layer["positions"]) for layer in frames[0]["layers"]))
    return float(max(2.0, min(8.0, 28.0 / max(atom_count ** 0.33, 1.0))))


def _build_html(frames: Sequence[dict[str, object]], title: str) -> str:
    if not frames:
        raise ValueError("visualizer needs at least one frame")

    marker_size = _marker_size(frames)
    frame_payload = []
    for frame in frames:
        traces = []
        for layer in frame["layers"]:
            positions = np.asarray(layer["positions"], dtype=float)
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": layer["label"],
                    "x": positions[:, 0].tolist() if positions.size else [],
                    "y": positions[:, 1].tolist() if positions.size else [],
                    "z": positions[:, 2].tolist() if positions.size else [],
                    "marker": {"size": marker_size, "color": layer["color"], "opacity": 0.9},
                    "hovertemplate": f"{layer['label']}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
                }
            )
        outline_x, outline_y, outline_z = _cell_outline_points(np.asarray(frame["lattice"], dtype=float))
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Commensurate Cell",
                "x": outline_x,
                "y": outline_y,
                "z": outline_z,
                "line": {"color": "#111111", "width": 5},
                "hoverinfo": "skip",
            }
        )
        frame_payload.append({"name": frame["name"], "data": traces, "layout": {"title": {"text": frame["subtitle"]}}})

    initial_data = frame_payload[0]["data"]
    initial_title = frame_payload[0]["layout"]["title"]["text"]
    serialized_initial = json.dumps(initial_data)
    serialized_frames = json.dumps(frame_payload)
    slider_steps = json.dumps(
        [
            {
                "label": frame["name"],
                "method": "animate",
                "args": [[frame["name"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 250}}],
            }
            for frame in frame_payload
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: rgba(255,255,255,0.86);
      --ink: #182027;
      --muted: #55636f;
      --line: rgba(24,32,39,0.14);
      --accent: #e76f51;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "Aptos", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(231,111,81,0.12), transparent 32rem),
        radial-gradient(circle at bottom right, rgba(42,157,143,0.12), transparent 28rem),
        linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 60px rgba(24,32,39,0.08);
      overflow: hidden;
      backdrop-filter: blur(16px);
    }}
    .header {{
      display: grid;
      gap: 8px;
      padding: 24px 28px 12px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.5rem, 2.8vw, 2.6rem);
      line-height: 1.05;
    }}
    p {{
      margin: 0;
      max-width: 70ch;
      color: var(--muted);
    }}
    #viewer {{
      height: min(78vh, 860px);
      width: 100%;
    }}
    .footer {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 0 28px 24px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 14px; }}
      .header {{ padding: 20px 18px 8px; }}
      .footer {{ padding: 0 18px 18px; flex-direction: column; }}
      #viewer {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="card">
      <div class="header">
        <div class="eyebrow">CELLSTINE Visualizer</div>
        <h1>{escape(title)}</h1>
        <p>Rotate freely with the mouse, use the play button or slider to snap through commensurate twist angles, and inspect the commensurate unit cell in each frame.</p>
      </div>
      <div id="viewer"></div>
      <div class="footer">
        <span>Only commensurate frames are shown, so the highlighted cell is always a valid superlattice cell for that angle.</span>
        <span>Generated by CELLSTINE</span>
      </div>
    </section>
  </main>
  <script>
    const initialData = {serialized_initial};
    const frames = {serialized_frames};
    const sliderSteps = {slider_steps};
    const layout = {{
      title: {{ text: {json.dumps(initial_title)}, x: 0.02, xanchor: "left" }},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: {{ l: 0, r: 0, t: 72, b: 0 }},
      legend: {{ orientation: "h", y: 1.02, x: 0.02 }},
      scene: {{
        aspectmode: "data",
        xaxis: {{ title: "x (A)", backgroundcolor: "rgba(255,255,255,0.35)", gridcolor: "rgba(24,32,39,0.08)" }},
        yaxis: {{ title: "y (A)", backgroundcolor: "rgba(255,255,255,0.35)", gridcolor: "rgba(24,32,39,0.08)" }},
        zaxis: {{ title: "z (A)", backgroundcolor: "rgba(255,255,255,0.35)", gridcolor: "rgba(24,32,39,0.08)" }},
        camera: {{ eye: {{ x: 1.65, y: 1.5, z: 0.9 }} }}
      }},
      updatemenus: [{{
        type: "buttons",
        x: 0.02,
        y: 1.18,
        direction: "left",
        showactive: false,
        buttons: [
          {{
            label: "Play",
            method: "animate",
            args: [null, {{ fromcurrent: true, mode: "immediate", frame: {{ duration: 800, redraw: true }}, transition: {{ duration: 250 }} }}]
          }},
          {{
            label: "Pause",
            method: "animate",
            args: [[null], {{ mode: "immediate", frame: {{ duration: 0, redraw: false }}, transition: {{ duration: 0 }} }}]
          }}
        ]
      }}],
      sliders: [{{
        active: 0,
        currentvalue: {{ prefix: "Frame: ", font: {{ size: 14 }} }},
        pad: {{ t: 48, b: 12 }},
        steps: sliderSteps
      }}]
    }};
    Plotly.newPlot("viewer", initialData, layout, {{ responsive: true }});
    Plotly.addFrames("viewer", frames);
  </script>
</body>
</html>"""


def build_visualization(
    results_file: str,
    *,
    indices: Sequence[int] | None = None,
    output_path: str | None = None,
    interlayer: float = 3.35,
    top_c_repeat: int | None = None,
    bottom_c_repeat: int | None = None,
) -> VisualizationRun:
    """Write a Plotly viewer from validated native Gram JSON v1."""

    results_path = Path(results_file).resolve()
    payload = read_results(results_path)
    search = payload["search"]
    metadata = payload["metadata"]
    candidates = list(payload["candidates"])
    if indices is not None:
        wanted = {int(index) for index in indices}
        candidates = [
            candidate
            for candidate in candidates
            if int(candidate["index"]) in wanted
        ]

    resolved_top_repeat = int(top_c_repeat if top_c_repeat is not None else 1)
    resolved_bottom_repeat = int(
        bottom_c_repeat if bottom_c_repeat is not None else 1
    )
    frames = [
        _build_bilayer_frame(
            str(search["top_poscar"]),
            str(search["bottom_poscar"]),
            candidate,
            interlayer=float(interlayer),
            top_c_repeat=resolved_top_repeat,
            bottom_c_repeat=resolved_bottom_repeat,
            search=search,
            metadata=metadata,
        )
        for candidate in candidates
    ]
    if not frames:
        raise ValueError("no frames were selected for visualization")

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = DEFAULT_OUTPUT_DIR / f"{results_path.stem}_viewer.html"
    else:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        _build_html(frames, "Bilayer Gram commensuration candidates"),
        encoding="utf-8",
    )
    return VisualizationRun(
        output_path=destination.resolve(),
        frame_count=len(frames),
        results_type="bilayer",
    )

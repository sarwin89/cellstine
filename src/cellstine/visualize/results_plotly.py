"""Plotly-based HTML visualizer for commensurate CELLSTINE results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..io import native as io_mod
from ..moire import findn as findn_backend
from ..moire import generator as generator_backend
from ..moire import lattice as lattice_backend

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


def _z_shift_layers(
    layer_atoms: Sequence[list[tuple[str, np.ndarray, tuple[str, str, str] | None]]],
    final_lattice: np.ndarray,
    min_z: float,
    lower_padding: float,
) -> tuple[list[list[tuple[str, np.ndarray, tuple[str, str, str] | None]]], np.ndarray]:
    z_shift = float(lower_padding) - float(min_z)
    shifted_layers = [generator_backend._shift_atoms_z(atoms, z_shift) for atoms in layer_atoms]
    return shifted_layers, final_lattice


def _build_bilayer_frame(
    top_poscar: str,
    bottom_poscar: str,
    record: Dict[str, object],
    *,
    interlayer: float,
    top_c_repeat: int,
    bottom_c_repeat: int,
) -> dict[str, object]:
    top = io_mod.repeat_structure_along_c(io_mod.read_poscar(top_poscar), top_c_repeat)
    bottom = io_mod.repeat_structure_along_c(io_mod.read_poscar(bottom_poscar), bottom_c_repeat)

    angle_deg = float(record["angle"])
    rotated_top = lattice_backend.rotate_lattice(top.lattice, angle_deg)
    top_supercell = np.vstack(
        (
            int(record["i11"]) * rotated_top[0] + int(record["i12"]) * rotated_top[1],
            int(record["i21"]) * rotated_top[0] + int(record["i22"]) * rotated_top[1],
            rotated_top[2],
        )
    )
    bottom_supercell = np.vstack(
        (
            int(record["j11"]) * bottom.lattice[0] + int(record["j12"]) * bottom.lattice[1],
            int(record["j21"]) * bottom.lattice[0] + int(record["j22"]) * bottom.lattice[1],
            bottom.lattice[2],
        )
    )

    top_species = generator_backend._expand_species(top.species, top.counts, "Top")
    bottom_species = generator_backend._expand_species(bottom.species, bottom.counts, "Bottom")
    atoms_top = generator_backend._replicate_layer_cartesian(
        top.positions_direct,
        rotated_top,
        top_supercell,
        (int(record["i11"]), int(record["i12"])),
        (int(record["i21"]), int(record["i22"])),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        1,
        1e-4,
        top_species,
        top.selective_flags,
    )
    atoms_bottom = generator_backend._replicate_layer_cartesian(
        bottom.positions_direct,
        bottom.lattice,
        bottom_supercell,
        (int(record["j11"]), int(record["j12"])),
        (int(record["j21"]), int(record["j22"])),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        1,
        1e-4,
        bottom_species,
        bottom.selective_flags,
    )
    if atoms_top and atoms_bottom:
        top_min_z, _ = generator_backend._z_bounds(atoms_top)
        _, bottom_max_z = generator_backend._z_bounds(atoms_bottom)
        atoms_top = generator_backend._shift_atoms_z(atoms_top, bottom_max_z + float(interlayer) - top_min_z)

    final_vector1 = bottom_supercell[0].copy()
    final_vector2 = bottom_supercell[1].copy()
    reference_c = generator_backend._reference_c_vector(rotated_top[2], bottom.lattice[2])
    final_lattice, min_z, lower_padding = generator_backend._build_final_lattice(
        final_vector1,
        final_vector2,
        reference_c,
        atoms_top + atoms_bottom,
        1e-4,
    )
    shifted_layers, _ = _z_shift_layers([atoms_bottom, atoms_top], final_lattice, min_z, lower_padding)
    return {
        "name": f"idx {int(record['idx'])} | {angle_deg:.4f} deg",
        "lattice": final_lattice.tolist(),
        "layers": [
            {"label": "Bottom", "color": "#264653", "positions": [position.tolist() for _, position, _ in shifted_layers[0]]},
            {"label": "Top", "color": "#e76f51", "positions": [position.tolist() for _, position, _ in shifted_layers[1]]},
        ],
        "subtitle": f"Candidate {int(record['idx'])}: bottom/top commensurate at {angle_deg:.4f} degrees",
    }


def _build_nlayer_frame(
    meta: Dict[str, object],
    candidate: Dict[str, object],
    *,
    interlayers: Sequence[float],
    bottom_c_repeat: int,
    upper_c_repeats: Sequence[int],
) -> dict[str, object]:
    bottom = io_mod.repeat_structure_along_c(io_mod.read_poscar(str(meta["bottom_poscar"])), bottom_c_repeat)
    bottom_vector1 = tuple(int(value) for value in candidate["bottom_vector1"])
    bottom_vector2 = tuple(int(value) for value in candidate["bottom_vector2"])
    upper_specs = list(candidate["upper_layers"])
    if len(interlayers) != len(upper_specs):
        raise ValueError("visualize needs one interlayer distance per upper layer")
    if len(upper_c_repeats) != len(upper_specs):
        raise ValueError("visualize needs one upper c-repeat per upper layer")

    palette = ["#264653", "#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#8d99ae", "#bc6c25", "#6a994e"]
    atoms_bottom = generator_backend._replicate_layer_cartesian(
        bottom.positions_direct,
        bottom.lattice,
        np.vstack(
            (
                bottom_vector1[0] * bottom.lattice[0] + bottom_vector1[1] * bottom.lattice[1],
                bottom_vector2[0] * bottom.lattice[0] + bottom_vector2[1] * bottom.lattice[1],
                bottom.lattice[2],
            )
        ),
        bottom_vector1,
        bottom_vector2,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        1,
        1e-4,
        generator_backend._expand_species(bottom.species, bottom.counts, "Bottom"),
        bottom.selective_flags,
    )
    layer_atoms = [atoms_bottom]
    layer_labels = [{"label": "Bottom", "color": palette[0]}]
    c_vectors = [bottom.lattice[2]]
    for layer_index, (poscar_path, repeat, layer_spec, gap) in enumerate(
        zip(meta["upper_poscars"], upper_c_repeats, upper_specs, interlayers),
        start=1,
    ):
        structure = io_mod.repeat_structure_along_c(io_mod.read_poscar(str(poscar_path)), int(repeat))
        rotated_lattice = lattice_backend.rotate_lattice(structure.lattice, float(layer_spec["angle_deg"]))
        atoms = generator_backend._replicate_layer_cartesian(
            structure.positions_direct,
            rotated_lattice,
            np.vstack(
                (
                    int(layer_spec["vector1"][0]) * rotated_lattice[0] + int(layer_spec["vector1"][1]) * rotated_lattice[1],
                    int(layer_spec["vector2"][0]) * rotated_lattice[0] + int(layer_spec["vector2"][1]) * rotated_lattice[1],
                    rotated_lattice[2],
                )
            ),
            tuple(int(value) for value in layer_spec["vector1"]),
            tuple(int(value) for value in layer_spec["vector2"]),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            1,
            1e-4,
            generator_backend._expand_species(structure.species, structure.counts, f"Upper{layer_index}"),
            structure.selective_flags,
        )
        if layer_atoms[-1] and atoms:
            current_min_z, _ = generator_backend._z_bounds(atoms)
            _, lower_max_z = generator_backend._z_bounds(layer_atoms[-1])
            atoms = generator_backend._shift_atoms_z(atoms, lower_max_z + float(gap) - current_min_z)
        layer_atoms.append(atoms)
        layer_labels.append({"label": f"Upper {layer_index}", "color": palette[layer_index % len(palette)]})
        c_vectors.append(rotated_lattice[2])

    all_atoms = []
    for atoms in reversed(layer_atoms):
        all_atoms.extend(atoms)
    final_lattice, min_z, lower_padding = generator_backend._build_final_lattice(
        bottom_vector1[0] * bottom.lattice[0] + bottom_vector1[1] * bottom.lattice[1],
        bottom_vector2[0] * bottom.lattice[0] + bottom_vector2[1] * bottom.lattice[1],
        max(c_vectors, key=lambda item: float(np.linalg.norm(item))),
        all_atoms,
        1e-4,
    )
    shifted_layers, _ = _z_shift_layers(layer_atoms, final_lattice, min_z, lower_padding)
    return {
        "name": "idx {idx} | {angles}".format(
            idx=int(candidate["index"]),
            angles=" | ".join(
                f"U{int(layer['layer_index'])} {float(layer['angle_deg']):.4f} deg" for layer in upper_specs
            ),
        ),
        "lattice": final_lattice.tolist(),
        "layers": [
            {
                "label": layer_labels[index]["label"],
                "color": layer_labels[index]["color"],
                "positions": [position.tolist() for _, position, _ in shifted_layers[index]],
            }
            for index in range(len(shifted_layers))
        ],
        "subtitle": "Candidate {idx}: {angles}".format(
            idx=int(candidate["index"]),
            angles=", ".join(f"upper {int(layer['layer_index'])} = {float(layer['angle_deg']):.4f} deg" for layer in upper_specs),
        ),
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
    interlayer_bottom_middle: float = 3.35,
    interlayer_middle_top: float = 3.35,
    top_c_repeat: int | None = None,
    bottom_c_repeat: int | None = None,
    middle_c_repeat: int | None = None,
) -> VisualizationRun:
    results_path = Path(results_file)
    frames: List[dict[str, object]]
    results_type: str

    if results_path.suffix.lower() == ".json":
        meta, candidates = findn_backend.parse_results(str(results_path))
        selected = list(candidates)
        if indices is not None:
            wanted = {int(index) for index in indices}
            selected = [candidate for candidate in candidates if int(candidate["index"]) in wanted]
        if "upper_poscars" in meta:
            resolved_bottom_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else meta.get("bottom_c_repeat", 1))
            default_upper_repeats = [int(value) for value in meta.get("upper_c_repeats", [1] * len(meta["upper_poscars"]))]
            if top_c_repeat is not None and middle_c_repeat is None:
                resolved_upper_repeats = [int(top_c_repeat)] * len(default_upper_repeats)
            elif middle_c_repeat is not None and len(default_upper_repeats) == 2 and top_c_repeat is not None:
                resolved_upper_repeats = [int(middle_c_repeat), int(top_c_repeat)]
            else:
                resolved_upper_repeats = default_upper_repeats
            if len(default_upper_repeats) == 1:
                resolved_interlayers = [float(interlayer)]
            elif len(default_upper_repeats) == 2:
                resolved_interlayers = [float(interlayer_bottom_middle), float(interlayer_middle_top)]
            else:
                resolved_interlayers = [float(interlayer_bottom_middle)] * len(default_upper_repeats)
            frames = [
                _build_nlayer_frame(
                    meta,
                    candidate,
                    interlayers=resolved_interlayers,
                    bottom_c_repeat=resolved_bottom_repeat,
                    upper_c_repeats=resolved_upper_repeats,
                )
                for candidate in selected
            ]
            title = f"{int(meta.get('layer_count', 0))}-layer commensurate twist sequence"
            results_type = "nlayer"
        else:
            resolved_bottom_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else meta.get("bottom_c_repeat", 1))
            resolved_middle_repeat = int(middle_c_repeat if middle_c_repeat is not None else meta.get("middle_c_repeat", 1))
            resolved_top_repeat = int(top_c_repeat if top_c_repeat is not None else meta.get("top_c_repeat", 1))
            converted_meta = {
                "bottom_poscar": meta["bottom_poscar"],
                "upper_poscars": [meta["middle_poscar"], meta["top_poscar"]],
            }
            frames = [
                _build_nlayer_frame(
                    converted_meta,
                    {
                        "index": candidate["index"],
                        "bottom_vector1": candidate["bottom_vector1"],
                        "bottom_vector2": candidate["bottom_vector2"],
                        "upper_layers": [
                            {
                                "layer_index": 1,
                                "angle_deg": candidate["angle_middle_deg"],
                                "vector1": candidate["middle_vector1"],
                                "vector2": candidate["middle_vector2"],
                            },
                            {
                                "layer_index": 2,
                                "angle_deg": candidate["angle_top_deg"],
                                "vector1": candidate["top_vector1"],
                                "vector2": candidate["top_vector2"],
                            },
                        ],
                    },
                    interlayers=[float(interlayer_bottom_middle), float(interlayer_middle_top)],
                    bottom_c_repeat=resolved_bottom_repeat,
                    upper_c_repeats=[resolved_middle_repeat, resolved_top_repeat],
                )
                for candidate in selected
            ]
            title = "Trilayer commensurate twist sequence"
            results_type = "trilayer"
    else:
        top_poscar, bottom_poscar, records, payload = generator_backend.parse_results(str(results_path))
        selected = list(records)
        if indices is not None:
            wanted = {int(index) for index in indices}
            selected = [record for record in records if int(record["idx"]) in wanted]
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        resolved_top_repeat = int(top_c_repeat if top_c_repeat is not None else meta.get("top_c_repeat", 1))
        resolved_bottom_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else meta.get("bottom_c_repeat", 1))
        frames = [
            _build_bilayer_frame(
                top_poscar,
                bottom_poscar,
                record,
                interlayer=interlayer,
                top_c_repeat=resolved_top_repeat,
                bottom_c_repeat=resolved_bottom_repeat,
            )
            for record in selected
        ]
        title = "Bilayer commensurate twist sequence"
        results_type = "bilayer"

    if not frames:
        raise ValueError("no frames were selected for visualization")

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(DEFAULT_OUTPUT_DIR / f"{results_path.stem}_viewer.html")
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(_build_html(frames, title))
    return VisualizationRun(
        output_path=Path(output_path).resolve(),
        frame_count=len(frames),
        results_type=results_type,
    )

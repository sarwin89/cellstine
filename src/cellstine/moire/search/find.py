"""Native bilayer workflow around the Gram-form search engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...io import native as io_mod
from . import gram
from .results import build_results_document, write_results


@dataclass
class FindRun:
    """Artifacts and in-memory results from one native Gram search."""

    run_id: str
    result_path: Path
    result: gram.SearchResult
    candidates: list[dict[str, Any]]
    parameters: dict[str, Any]
    timings: dict[str, float]


def _slug(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "structure"


def _make_result_path(output_root: str, bottom_path: str, top_path: str) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{timestamp}_{_slug(Path(bottom_path).stem)}_below__"
        f"{_slug(Path(top_path).stem)}_above_gram"
    )
    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir / "results.json"


def _planar_column_basis(lattice: np.ndarray, name: str) -> np.ndarray:
    """Convert POSCAR's planar Cartesian row basis to Gram engine columns."""

    lattice_array = np.asarray(lattice, dtype=float)
    if lattice_array.shape != (3, 3):
        raise ValueError(f"{name} POSCAR lattice must be 3x3")
    in_plane_rows = lattice_array[:2, :2]
    if not np.all(np.isfinite(in_plane_rows)):
        raise ValueError(f"{name} POSCAR in-plane lattice must be finite")
    out_of_plane = lattice_array[:2, 2]
    scale = max(float(np.max(np.abs(lattice_array[:2]))), 1.0)
    if np.max(np.abs(out_of_plane)) > 1e-10 * scale:
        raise ValueError(f"{name} POSCAR a/b lattice vectors must be planar in Cartesian xy")
    return np.array(in_plane_rows.T, dtype=float, copy=True)


def _notify(
    progress_callback: Callable[[str, str], None] | None,
    stage: str,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message)


def run_find(
    *,
    top_poscar: str,
    bottom_poscar: str,
    max_length: float,
    top_strain: float,
    bottom_strain: float,
    min_length: float | None = None,
    max_atoms: int | None = None,
    max_aspect_ratio: float = 12.0,
    min_cell_angle_deg: float = 25.0,
    max_cell_angle_deg: float = 155.0,
    fold_symmetry: bool = True,
    symmetric: bool = False,
    output_root: str = "runs",
    progress_callback: Callable[[str, str], None] | None = None,
) -> FindRun:
    """Read two POSCARs, run one native search, and write validated JSON v1.

    ``top_strain`` and ``bottom_strain`` are bounds on principal logarithmic strain.
    There is no angle shortlist, index-box search, or DAT compatibility path.
    """

    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    top_path = Path(top_poscar).resolve()
    bottom_path = Path(bottom_poscar).resolve()

    _notify(progress_callback, "read", "reading input structures")
    stage_start = time.perf_counter()
    top_structure = io_mod.read_poscar(str(top_path))
    bottom_structure = io_mod.read_poscar(str(bottom_path))
    top_basis = _planar_column_basis(top_structure.lattice, "top")
    bottom_basis = _planar_column_basis(bottom_structure.lattice, "bottom")
    timings["read_structures_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "read",
        f"read structures in {timings['read_structures_s']:.3f}s",
    )

    requested_config = gram.SearchConfig(
        top_basis=top_basis,
        bottom_basis=bottom_basis,
        max_length=float(max_length),
        top_strain=float(top_strain),
        bottom_strain=float(bottom_strain),
        min_length=None if min_length is None else float(min_length),
        max_atoms=None if max_atoms is None else int(max_atoms),
        top_atoms=top_structure.natoms,
        bottom_atoms=bottom_structure.natoms,
        max_aspect_ratio=float(max_aspect_ratio),
        min_cell_angle_deg=float(min_cell_angle_deg),
        max_cell_angle_deg=float(max_cell_angle_deg),
        fold_symmetry=bool(fold_symmetry),
        symmetric=bool(symmetric),
    )

    _notify(progress_callback, "search", "searching native Gram-form candidates")
    stage_start = time.perf_counter()
    symmetric_used = False
    symmetric_fallback: str | None = None
    completed_config = requested_config
    try:
        result = gram.search(requested_config)
        symmetric_used = bool(symmetric)
    except gram.SymmetricBranchUnavailable as exc:
        if not symmetric:
            raise
        symmetric_fallback = str(exc)
        completed_config = replace(requested_config, symmetric=False)
        result = gram.search(completed_config)
    timings["search_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "search",
        f"found {len(result)} candidate(s) in {timings['search_s']:.3f}s",
    )

    _notify(progress_callback, "write", "writing results.json")
    stage_start = time.perf_counter()
    run_id, result_path = _make_result_path(str(output_root), str(bottom_path), str(top_path))
    document = build_results_document(
        top_poscar=top_path,
        bottom_poscar=bottom_path,
        config=completed_config,
        result=result,
        symmetric_requested=bool(symmetric),
        symmetric_used=symmetric_used,
        symmetric_fallback=symmetric_fallback,
    )
    write_results(result_path, document)
    timings["write_results_s"] = time.perf_counter() - stage_start
    timings["total_s"] = time.perf_counter() - total_start
    _notify(
        progress_callback,
        "write",
        f"wrote results.json in {timings['write_results_s']:.3f}s",
    )

    return FindRun(
        run_id=run_id,
        result_path=result_path.resolve(),
        result=result,
        candidates=list(document["candidates"]),
        parameters=dict(document["search"]),
        timings=timings,
    )


def find(**kwargs):
    return run_find(**kwargs)

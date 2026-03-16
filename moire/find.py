"""High-level commensuration finder stage.

Made by Sarwin Chandran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import angles as angle_backend
from . import finder as finder_backend
from . import lattice as lat


@dataclass
class FindRun:
    run_dir: Path
    json_path: Path
    markdown_path: Path
    dat_path: Path
    candidates: List[lat.SupercellCandidate]
    shortlisted_angles: List[angle_backend.AngleCandidate]
    angle_values: List[float]
    symmetry_top: int
    symmetry_bottom: int
    symmetry_lcm: int
    search_min_angle: float
    search_max_angle: float


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _make_run_dir(output_root: str, bottom_path: str, top_path: str, nindex: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_slug(Path(bottom_path).stem)}_below__{_slug(Path(top_path).stem)}_above_n{nindex}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_shortlist(candidates: Sequence[angle_backend.AngleCandidate]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        payload.append(
            {
                "index": index,
                "angle_deg": float(candidate.angle_deg),
                "coeffs1": [int(candidate.coeffs1[0]), int(candidate.coeffs1[1])],
                "coeffs2": [int(candidate.coeffs2[0]), int(candidate.coeffs2[1])],
                "length1": float(candidate.length1),
                "length2": float(candidate.length2),
                "relative_mismatch": float(candidate.relative_mismatch),
            }
        )
    return payload


def _resolve_angles(
    lattice_top: np.ndarray,
    lattice_bottom: np.ndarray,
    nindex: int,
    *,
    min_angle: float,
    max_angle: float | None,
    angle_step: float,
    explicit_angles: Sequence[float] | None,
    angle_length_tolerance: float,
    angle_strain_tolerance: float | None,
    angle_merge_tolerance: float,
) -> tuple[List[angle_backend.AngleCandidate], List[float], int, int, int, float, float]:
    symmetry_top, symmetry_bottom, symmetry_lcm = lat.combined_symmetry_limit(lattice_top, lattice_bottom)
    bounded_min = max(0.0, float(min_angle))
    bounded_max = float(symmetry_lcm if max_angle is None else min(symmetry_lcm, max_angle))

    if explicit_angles:
        chosen_angles = [float(value) for value in explicit_angles if bounded_min <= float(value) <= bounded_max]
        return [], sorted(set(chosen_angles)), symmetry_top, symmetry_bottom, symmetry_lcm, bounded_min, bounded_max

    shortlist = angle_backend.find_commensurate_angles(
        lattice_top,
        lattice_bottom,
        nindex,
        length_tolerance=angle_length_tolerance,
        strain_tolerance=angle_strain_tolerance,
        min_angle=bounded_min,
        max_angle=bounded_max,
        merge_tolerance=angle_merge_tolerance,
    )
    if shortlist:
        angle_values = sorted({bounded_min, bounded_max, *(float(candidate.angle_deg) for candidate in shortlist)})
    else:
        angle_values = list(np.arange(bounded_min, bounded_max + angle_step * 0.5, angle_step, dtype=float))
    return shortlist, angle_values, symmetry_top, symmetry_bottom, symmetry_lcm, bounded_min, bounded_max


def run_find(
    *,
    top_poscar: str,
    bottom_poscar: str,
    top_lattice: np.ndarray,
    bottom_lattice: np.ndarray,
    top_atoms: int,
    bottom_atoms: int,
    nindex: int,
    min_angle: float = 0.0,
    max_angle: float | None = None,
    angle_step: float = 0.1,
    explicit_angles: Sequence[float] | None = None,
    angle_length_tolerance: float = 1e-5,
    angle_strain_tolerance: float | None = 2e-3,
    angle_merge_tolerance: float = 1e-3,
    vector_tolerance: float = 2e-3,
    vector_strain_tolerance: float | None = 2e-3,
    candidate_tolerance: float | None = None,
    strain_tolerance: float | None = None,
    strain_layer: str = "avg",
    min_atoms: int | None = None,
    max_atoms: int | None = 2000,
    dedupe: bool = True,
    unique_strain_tolerance: float = 1e-4,
    unique_ratio_tolerance: float = 1e-5,
    output_root: str = "runs",
) -> FindRun:
    shortlist, angle_values, symmetry_top, symmetry_bottom, symmetry_lcm, search_min_angle, search_max_angle = _resolve_angles(
        top_lattice,
        bottom_lattice,
        nindex,
        min_angle=min_angle,
        max_angle=max_angle,
        angle_step=angle_step,
        explicit_angles=explicit_angles,
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
    )

    candidates = finder_backend.find_supercells(
        top_lattice,
        bottom_lattice,
        None,
        None,
        angle_step=angle_step,
        nindex=nindex,
        tol=vector_tolerance,
        lin_tol=candidate_tolerance,
        strain_tol=strain_tolerance,
        strain_layer=strain_layer,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        atom_count1=top_atoms,
        atom_count2=bottom_atoms,
        angles=angle_values,
        dedupe=dedupe,
        unique_strain_tol=unique_strain_tolerance,
        unique_ratio_tol=unique_ratio_tolerance,
        vector_strain_tol=vector_strain_tolerance,
    )

    run_dir = _make_run_dir(output_root, bottom_poscar, top_poscar, nindex)
    dat_path = run_dir / "find_results.dat"
    markdown_path = run_dir / "find_results.md"
    json_path = run_dir / "find_results.json"

    finder_backend.write_results_dat(str(dat_path), top_poscar, bottom_poscar, candidates)
    markdown_path.write_text(finder_backend.format_results_table(candidates, limit=None) + "\n", encoding="utf-8")

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "credit": "Made by Sarwin Chandran",
            "top_poscar": str(top_poscar),
            "bottom_poscar": str(bottom_poscar),
            "nindex": int(nindex),
            "symmetry_top_deg": int(symmetry_top),
            "symmetry_bottom_deg": int(symmetry_bottom),
            "symmetry_lcm_deg": int(symmetry_lcm),
            "min_angle_deg": float(search_min_angle),
            "max_angle_deg": float(search_max_angle),
            "angle_count": len(angle_values),
            "vector_tolerance": float(vector_tolerance),
            "vector_strain_tolerance": None if vector_strain_tolerance is None else float(vector_strain_tolerance),
            "candidate_tolerance": float(candidate_tolerance if candidate_tolerance is not None else vector_tolerance),
        },
        "shortlisted_angles": _serialize_shortlist(shortlist),
        "candidates": [finder_backend.candidate_to_dict(candidate, idx + 1) for idx, candidate in enumerate(candidates)],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return FindRun(
        run_dir=run_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        dat_path=dat_path,
        candidates=list(candidates),
        shortlisted_angles=list(shortlist),
        angle_values=angle_values,
        symmetry_top=symmetry_top,
        symmetry_bottom=symmetry_bottom,
        symmetry_lcm=symmetry_lcm,
        search_min_angle=search_min_angle,
        search_max_angle=search_max_angle,
    )

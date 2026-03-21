"""Reference-layer trilayer commensuration search for CELLSTINE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import angles as angle_backend
from . import find as find_stage
from . import finder as finder_backend
from . import lattice as lat


@dataclass(frozen=True)
class TrilayerCandidate:
    angle_middle_deg: float
    angle_top_deg: float
    strain_middle: float
    strain_top: float
    strain_max: float
    strain_mean: float
    ratio_bottom: int
    ratio_middle: int
    ratio_top: int
    total_atoms: int
    middle_vector1: tuple[int, int]
    middle_vector2: tuple[int, int]
    bottom_vector1: tuple[int, int]
    bottom_vector2: tuple[int, int]
    top_vector1: tuple[int, int]
    top_vector2: tuple[int, int]


@dataclass
class Find3Run:
    run_id: str
    result_path: Path
    candidates: List[TrilayerCandidate]
    middle_shortlisted_angles: List[angle_backend.AngleCandidate]
    top_shortlisted_angles: List[angle_backend.AngleCandidate]
    middle_angle_values: List[float]
    top_angle_values: List[float]
    parameters: Dict[str, object]


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _make_result_path(output_root: str, bottom_path: str, middle_path: str, top_path: str, nindex: int) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{timestamp}_{_slug(Path(bottom_path).stem)}_bottom__{_slug(Path(middle_path).stem)}_middle__"
        f"{_slug(Path(top_path).stem)}_top_n{nindex}"
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir / f"{run_id}.json"


def _bottom_matrix_key(candidate: lat.SupercellCandidate) -> tuple[int, int, int, int]:
    return (
        int(candidate.layer2_vector1[0]),
        int(candidate.layer2_vector1[1]),
        int(candidate.layer2_vector2[0]),
        int(candidate.layer2_vector2[1]),
    )


def _join_pair_candidates(
    middle_candidates: Sequence[lat.SupercellCandidate],
    top_candidates: Sequence[lat.SupercellCandidate],
    *,
    bottom_atoms: int,
    middle_atoms: int,
    top_atoms: int,
    max_atoms: int | None,
) -> List[TrilayerCandidate]:
    by_bottom_matrix: dict[tuple[int, int, int, int], list[lat.SupercellCandidate]] = {}
    for candidate in top_candidates:
        by_bottom_matrix.setdefault(_bottom_matrix_key(candidate), []).append(candidate)

    joined: List[TrilayerCandidate] = []
    seen: set[tuple[float, float, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]] = set()
    for middle_candidate in middle_candidates:
        matches = by_bottom_matrix.get(_bottom_matrix_key(middle_candidate), [])
        for top_candidate in matches:
            if int(middle_candidate.ratio2) != int(top_candidate.ratio2):
                continue

            signature = (
                round(float(middle_candidate.angle_deg), 8),
                round(float(top_candidate.angle_deg), 8),
                _bottom_matrix_key(middle_candidate),
                (
                    int(middle_candidate.layer1_vector1[0]),
                    int(middle_candidate.layer1_vector1[1]),
                    int(middle_candidate.layer1_vector2[0]),
                    int(middle_candidate.layer1_vector2[1]),
                ),
                (
                    int(top_candidate.layer1_vector1[0]),
                    int(top_candidate.layer1_vector1[1]),
                    int(top_candidate.layer1_vector2[0]),
                    int(top_candidate.layer1_vector2[1]),
                ),
            )
            if signature in seen:
                continue
            seen.add(signature)

            total_atoms = (
                int(bottom_atoms) * int(middle_candidate.ratio2)
                + int(middle_atoms) * int(middle_candidate.ratio1)
                + int(top_atoms) * int(top_candidate.ratio1)
            )
            if max_atoms is not None and total_atoms > int(max_atoms):
                continue

            strain_middle = float(middle_candidate.strain_avg)
            strain_top = float(top_candidate.strain_avg)
            joined.append(
                TrilayerCandidate(
                    angle_middle_deg=float(middle_candidate.angle_deg),
                    angle_top_deg=float(top_candidate.angle_deg),
                    strain_middle=strain_middle,
                    strain_top=strain_top,
                    strain_max=max(strain_middle, strain_top),
                    strain_mean=0.5 * (strain_middle + strain_top),
                    ratio_bottom=int(middle_candidate.ratio2),
                    ratio_middle=int(middle_candidate.ratio1),
                    ratio_top=int(top_candidate.ratio1),
                    total_atoms=int(total_atoms),
                    middle_vector1=tuple(int(value) for value in middle_candidate.layer1_vector1),
                    middle_vector2=tuple(int(value) for value in middle_candidate.layer1_vector2),
                    bottom_vector1=tuple(int(value) for value in middle_candidate.layer2_vector1),
                    bottom_vector2=tuple(int(value) for value in middle_candidate.layer2_vector2),
                    top_vector1=tuple(int(value) for value in top_candidate.layer1_vector1),
                    top_vector2=tuple(int(value) for value in top_candidate.layer1_vector2),
                )
            )
    joined.sort(key=lambda item: (item.strain_max, item.strain_mean, item.total_atoms, item.angle_middle_deg, item.angle_top_deg))
    return joined


def candidate_to_dict(candidate: TrilayerCandidate, index: int | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "angle_middle_deg": float(candidate.angle_middle_deg),
        "angle_top_deg": float(candidate.angle_top_deg),
        "strain_middle": float(candidate.strain_middle),
        "strain_top": float(candidate.strain_top),
        "strain_max": float(candidate.strain_max),
        "strain_mean": float(candidate.strain_mean),
        "ratio_bottom": int(candidate.ratio_bottom),
        "ratio_middle": int(candidate.ratio_middle),
        "ratio_top": int(candidate.ratio_top),
        "total_atoms": int(candidate.total_atoms),
        "middle_vector1": [int(candidate.middle_vector1[0]), int(candidate.middle_vector1[1])],
        "middle_vector2": [int(candidate.middle_vector2[0]), int(candidate.middle_vector2[1])],
        "bottom_vector1": [int(candidate.bottom_vector1[0]), int(candidate.bottom_vector1[1])],
        "bottom_vector2": [int(candidate.bottom_vector2[0]), int(candidate.bottom_vector2[1])],
        "top_vector1": [int(candidate.top_vector1[0]), int(candidate.top_vector1[1])],
        "top_vector2": [int(candidate.top_vector2[0]), int(candidate.top_vector2[1])],
    }
    if index is not None:
        payload["index"] = int(index)
    return payload


def format_results_table(candidates: Sequence[TrilayerCandidate], limit: int | None = None) -> str:
    shown = list(candidates if limit is None or limit < 0 else candidates[: max(limit, 0)])
    if not shown:
        return "No trilayer candidates found."

    header = (
        " idx  ang_mid  ang_top  strain_max  strain_mid  strain_top   atoms   ratio(b/m/t)"
        "   bottom cell    middle cell    top cell"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for index, candidate in enumerate(shown, start=1):
        lines.append(
            "{idx:4d}  {ang_mid:7.3f}  {ang_top:7.3f}  {strain_max:10.6f}  {strain_mid:10.6f}  "
            "{strain_top:10.6f}  {atoms:7d}   {ratio_bottom:3d}/{ratio_middle:3d}/{ratio_top:3d}     "
            "({b11:3d},{b12:3d}) ({b21:3d},{b22:3d})  ({m11:3d},{m12:3d}) ({m21:3d},{m22:3d})  "
            "({t11:3d},{t12:3d}) ({t21:3d},{t22:3d})".format(
                idx=index,
                ang_mid=candidate.angle_middle_deg,
                ang_top=candidate.angle_top_deg,
                strain_max=candidate.strain_max,
                strain_mid=candidate.strain_middle,
                strain_top=candidate.strain_top,
                atoms=candidate.total_atoms,
                ratio_bottom=candidate.ratio_bottom,
                ratio_middle=candidate.ratio_middle,
                ratio_top=candidate.ratio_top,
                b11=candidate.bottom_vector1[0],
                b12=candidate.bottom_vector1[1],
                b21=candidate.bottom_vector2[0],
                b22=candidate.bottom_vector2[1],
                m11=candidate.middle_vector1[0],
                m12=candidate.middle_vector1[1],
                m21=candidate.middle_vector2[0],
                m22=candidate.middle_vector2[1],
                t11=candidate.top_vector1[0],
                t12=candidate.top_vector1[1],
                t21=candidate.top_vector2[0],
                t22=candidate.top_vector2[1],
            )
        )
    return "\n".join(lines)


def write_results_json(
    path: str,
    *,
    meta: Dict[str, object],
    candidates: Sequence[TrilayerCandidate],
) -> None:
    payload = {
        "meta": dict(meta),
        "candidates": [candidate_to_dict(candidate, index + 1) for index, candidate in enumerate(candidates)],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_results(path: str) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = dict(payload.get("meta", {}))
    candidates = list(payload.get("candidates", []))
    if not meta:
        raise ValueError("find3 results do not contain metadata")
    return meta, candidates


def run_find3(
    *,
    bottom_poscar: str,
    middle_poscar: str,
    top_poscar: str,
    bottom_lattice: np.ndarray,
    middle_lattice: np.ndarray,
    top_lattice: np.ndarray,
    bottom_atoms: int,
    middle_atoms: int,
    top_atoms: int,
    nindex: int,
    min_angle_middle: float = 0.0,
    max_angle_middle: float | None = None,
    min_angle_top: float = 0.0,
    max_angle_top: float | None = None,
    angle_step: float = 0.1,
    explicit_angles_middle: Sequence[float] | None = None,
    explicit_angles_top: Sequence[float] | None = None,
    angle_length_tolerance: float = 1e-5,
    angle_strain_tolerance: float | None = 2e-3,
    angle_merge_tolerance: float = 1e-3,
    vector_tolerance: float = 2e-3,
    vector_strain_tolerance: float | None = 2e-3,
    candidate_tolerance: float | None = None,
    pair_strain_tolerance: float | None = None,
    max_atoms: int | None = 2000,
    dedupe: bool = True,
    unique_strain_tolerance: float = 1e-4,
    unique_ratio_tolerance: float = 1e-5,
    output_root: str = "runs",
    bottom_c_repeat: int = 1,
    middle_c_repeat: int = 1,
    top_c_repeat: int = 1,
    workers: int = 1,
) -> Find3Run:
    middle_shortlist, middle_angle_values, sym_middle, sym_bottom_mid, lcm_middle, search_min_middle, search_max_middle = find_stage._resolve_angles(
        middle_lattice,
        bottom_lattice,
        nindex,
        min_angle=min_angle_middle,
        max_angle=max_angle_middle,
        angle_step=angle_step,
        explicit_angles=explicit_angles_middle,
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
    )
    top_shortlist, top_angle_values, sym_top, sym_bottom_top, lcm_top, search_min_top, search_max_top = find_stage._resolve_angles(
        top_lattice,
        bottom_lattice,
        nindex,
        min_angle=min_angle_top,
        max_angle=max_angle_top,
        angle_step=angle_step,
        explicit_angles=explicit_angles_top,
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
    )

    middle_candidates = finder_backend.find_supercells(
        middle_lattice,
        bottom_lattice,
        None,
        None,
        angle_step=angle_step,
        nindex=nindex,
        tol=vector_tolerance,
        lin_tol=candidate_tolerance,
        strain_tol=pair_strain_tolerance,
        strain_layer="avg",
        max_atoms=None,
        atom_count1=middle_atoms,
        atom_count2=bottom_atoms,
        angles=middle_angle_values,
        dedupe=dedupe,
        unique_strain_tol=unique_strain_tolerance,
        unique_ratio_tol=unique_ratio_tolerance,
        vector_strain_tol=vector_strain_tolerance,
        workers=workers,
    )
    top_candidates = finder_backend.find_supercells(
        top_lattice,
        bottom_lattice,
        None,
        None,
        angle_step=angle_step,
        nindex=nindex,
        tol=vector_tolerance,
        lin_tol=candidate_tolerance,
        strain_tol=pair_strain_tolerance,
        strain_layer="avg",
        max_atoms=None,
        atom_count1=top_atoms,
        atom_count2=bottom_atoms,
        angles=top_angle_values,
        dedupe=dedupe,
        unique_strain_tol=unique_strain_tolerance,
        unique_ratio_tol=unique_ratio_tolerance,
        vector_strain_tol=vector_strain_tolerance,
        workers=workers,
    )

    candidates = _join_pair_candidates(
        middle_candidates,
        top_candidates,
        bottom_atoms=bottom_atoms,
        middle_atoms=middle_atoms,
        top_atoms=top_atoms,
        max_atoms=max_atoms,
    )

    run_id, result_path = _make_result_path(output_root, bottom_poscar, middle_poscar, top_poscar, nindex)
    parameters: Dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "credit": "CELLSTINE (CELL Superlattice Transformation INterface and Engine) | Made by Sarwin Chandran",
        "units_note": "angles in degrees; angle_length_tolerance in angstrom; strain and mismatch values as fractions (0.01 = 1%)",
        "bottom_poscar": str(bottom_poscar),
        "middle_poscar": str(middle_poscar),
        "top_poscar": str(top_poscar),
        "bottom_c_repeat": int(bottom_c_repeat),
        "middle_c_repeat": int(middle_c_repeat),
        "top_c_repeat": int(top_c_repeat),
        "workers": int(workers),
        "nindex": int(nindex),
        "symmetry_middle_deg": int(sym_middle),
        "symmetry_bottom_to_middle_deg": int(sym_bottom_mid),
        "symmetry_lcm_middle_deg": int(lcm_middle),
        "symmetry_top_deg": int(sym_top),
        "symmetry_bottom_to_top_deg": int(sym_bottom_top),
        "symmetry_lcm_top_deg": int(lcm_top),
        "min_angle_middle_deg": float(search_min_middle),
        "max_angle_middle_deg": float(search_max_middle),
        "min_angle_top_deg": float(search_min_top),
        "max_angle_top_deg": float(search_max_top),
        "angle_count_middle": len(middle_angle_values),
        "angle_count_top": len(top_angle_values),
        "angle_step_deg": float(angle_step),
        "explicit_angles_middle_deg": ",".join(f"{value:.6f}" for value in explicit_angles_middle) if explicit_angles_middle else "",
        "explicit_angles_top_deg": ",".join(f"{value:.6f}" for value in explicit_angles_top) if explicit_angles_top else "",
        "angle_length_tolerance": float(angle_length_tolerance),
        "angle_strain_tolerance": "" if angle_strain_tolerance is None else float(angle_strain_tolerance),
        "angle_merge_tolerance": float(angle_merge_tolerance),
        "vector_tolerance": float(vector_tolerance),
        "vector_strain_tolerance": "" if vector_strain_tolerance is None else float(vector_strain_tolerance),
        "candidate_tolerance": float(candidate_tolerance if candidate_tolerance is not None else vector_tolerance),
        "pair_strain_tolerance": "" if pair_strain_tolerance is None else float(pair_strain_tolerance),
        "max_atoms": "" if max_atoms is None else int(max_atoms),
        "dedupe": bool(dedupe),
        "unique_strain_tolerance": float(unique_strain_tolerance),
        "unique_ratio_tolerance": float(unique_ratio_tolerance),
    }
    if middle_shortlist:
        parameters["shortlisted_middle_angles_deg"] = ",".join(f"{candidate.angle_deg:.6f}" for candidate in middle_shortlist)
    if top_shortlist:
        parameters["shortlisted_top_angles_deg"] = ",".join(f"{candidate.angle_deg:.6f}" for candidate in top_shortlist)

    write_results_json(str(result_path), meta=parameters, candidates=candidates)
    return Find3Run(
        run_id=run_id,
        result_path=result_path,
        candidates=list(candidates),
        middle_shortlisted_angles=list(middle_shortlist),
        top_shortlisted_angles=list(top_shortlist),
        middle_angle_values=middle_angle_values,
        top_angle_values=top_angle_values,
        parameters=parameters,
    )

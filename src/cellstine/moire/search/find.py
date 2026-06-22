"""High-level bilayer commensuration search stage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from . import angles as angle_backend
from . import finder as finder_backend
from . import lattice as lat


@dataclass
class FindRun:
    run_id: str
    dat_path: Path
    candidates: List[lat.SupercellCandidate]
    shortlisted_angles: List[angle_backend.AngleCandidate]
    angle_values: List[float]
    symmetry_top: int
    symmetry_bottom: int
    symmetry_lcm: int
    search_min_angle: float
    search_max_angle: float
    parameters: Dict[str, object]
    timings: Dict[str, float]


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _make_result_path(output_root: str, bottom_path: str, top_path: str, nindex: int) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_slug(Path(bottom_path).stem)}_below__{_slug(Path(top_path).stem)}_above_n{nindex}"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir / f"{run_id}.dat"


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
    max_search_angles: int | None,
) -> tuple[List[angle_backend.AngleCandidate], List[float], int, int, int, float, float, Dict[str, object]]:
    symmetry_top, symmetry_bottom, symmetry_lcm = lat.combined_symmetry_limit(lattice_top, lattice_bottom)
    bounded_min = max(0.0, float(min_angle))
    bounded_max = float(symmetry_lcm if max_angle is None else min(symmetry_lcm, max_angle))
    angle_metadata: Dict[str, object] = {"angle_values_thinned": False}

    if explicit_angles:
        chosen_angles = [float(value) for value in explicit_angles if bounded_min <= float(value) <= bounded_max]
        return [], sorted(set(chosen_angles)), symmetry_top, symmetry_bottom, symmetry_lcm, bounded_min, bounded_max, angle_metadata

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
        # By default every commensurate twist angle is searched -- the analytic
        # engine handles each angle by a cheap local slice, so there is no need
        # to thin the angle list (the old behaviour silently dropped most angles
        # at large nindex, e.g. down to ~10 angles at nindex=100).  Thinning is
        # now strictly opt-in: it only happens when the caller passes a positive
        # ``max_search_angles``.
        if max_search_angles is None or int(max_search_angles) <= 0:
            resolved_cap = None
        else:
            resolved_cap = max(2, int(max_search_angles))
        if resolved_cap is not None and len(angle_values) > resolved_cap:
            interior_cap = max(1, resolved_cap - 2)
            edges = np.linspace(bounded_min, bounded_max, interior_cap + 1, dtype=float)
            selected: List[angle_backend.AngleCandidate] = []
            ordered = sorted(shortlist, key=lambda item: (item.angle_deg, item.relative_mismatch))
            start_index = 0
            for left, right in zip(edges[:-1], edges[1:]):
                members: List[angle_backend.AngleCandidate] = []
                while start_index < len(ordered) and ordered[start_index].angle_deg < left:
                    start_index += 1
                scan_index = start_index
                while scan_index < len(ordered) and ordered[scan_index].angle_deg <= right:
                    members.append(ordered[scan_index])
                    scan_index += 1
                if members:
                    selected.append(min(members, key=lambda item: (item.relative_mismatch, item.angle_deg)))
            angle_values = sorted({bounded_min, bounded_max, *(float(candidate.angle_deg) for candidate in selected)})
            angle_metadata.update(
                {
                    "angle_values_thinned": True,
                    "angle_values_before_thinning": len(shortlist) + 2,
                    "angle_values_after_thinning": len(angle_values),
                    "max_search_angles": resolved_cap,
                }
            )
    else:
        angle_values = list(np.arange(bounded_min, bounded_max + angle_step * 0.5, angle_step, dtype=float))
    return shortlist, angle_values, symmetry_top, symmetry_bottom, symmetry_lcm, bounded_min, bounded_max, angle_metadata


def _notify(progress_callback: Callable[[str, str], None] | None, stage: str, message: str) -> None:
    if progress_callback is not None:
        progress_callback(stage, message)


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
    matrix_values: Sequence[int] | None = None,
    matrix_layer: str = "either",
    matrix_match_mode: str = "absolute",
    output_root: str = "runs",
    top_c_repeat: int = 1,
    bottom_c_repeat: int = 1,
    workers: int = 1,
    fold_symmetry: bool = False,
    max_search_angles: int | None = None,
    max_pair_matches: int | None = None,
    cull_redundant: bool = True,
    reduce_basis: bool = True,
    max_cell_aspect_ratio: float | None = 12.0,
    min_cell_angle_deg: float | None = 25.0,
    max_cell_angle_deg: float | None = 155.0,
    progress_callback: Callable[[str, str], None] | None = None,
) -> FindRun:
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    _notify(progress_callback, "angles", "building commensurate angle shortlist")
    stage_start = time.perf_counter()
    (
        shortlist,
        angle_values,
        symmetry_top,
        symmetry_bottom,
        symmetry_lcm,
        search_min_angle,
        search_max_angle,
        angle_metadata,
    ) = _resolve_angles(
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
        max_search_angles=max_search_angles,
    )
    timings["angle_shortlist_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "angles",
        f"resolved {len(angle_values)} angle(s) in {timings['angle_shortlist_s']:.3f}s",
    )
    # ``max_pair_matches`` bounds, per twist angle, how many of the shortest
    # coincident vectors are paired into supercells.  Dropping the longer ones
    # only ever removes larger, Pareto-dominated cells, so this is a safe speed
    # knob that keeps degenerate (highly symmetric) angles from exploding.  The
    # default (None) resolves to a generous finite bound; pass a value <= 0 to
    # disable it entirely for an exhaustive (and potentially very slow) search.
    if max_pair_matches is None:
        effective_max_pair_matches = 128
    elif int(max_pair_matches) <= 0:
        effective_max_pair_matches = None
    else:
        effective_max_pair_matches = int(max_pair_matches)
    effective_strain_tolerance = (
        lat.MAX_PHYSICAL_MISMATCH
        if strain_tolerance is None
        else min(float(strain_tolerance), lat.MAX_PHYSICAL_MISMATCH)
    )

    _notify(progress_callback, "search", f"searching supercells across {len(angle_values)} angle(s)")
    stage_start = time.perf_counter()
    candidates = finder_backend.find_supercells(
        top_lattice,
        bottom_lattice,
        None,
        None,
        angle_step=angle_step,
        nindex=nindex,
        tol=vector_tolerance,
        lin_tol=candidate_tolerance,
        strain_tol=effective_strain_tolerance,
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
        matrix_values=matrix_values,
        matrix_layer=matrix_layer,
        matrix_match_mode=matrix_match_mode,
        workers=workers,
        fold_symmetry=fold_symmetry,
        max_pair_matches=effective_max_pair_matches,
        cull_redundant=cull_redundant,
        reduce_basis=reduce_basis,
        max_cell_aspect_ratio=max_cell_aspect_ratio,
        min_cell_angle_deg=min_cell_angle_deg,
        max_cell_angle_deg=max_cell_angle_deg,
    )
    timings["supercell_search_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "search",
        f"found {len(candidates)} candidate(s) in {timings['supercell_search_s']:.3f}s",
    )

    _notify(progress_callback, "write", "writing results table")
    stage_start = time.perf_counter()
    run_id, dat_path = _make_result_path(output_root, bottom_poscar, top_poscar, nindex)
    parameters = {
        "run_id": run_id,
        "top_poscar": str(top_poscar),
        "bottom_poscar": str(bottom_poscar),
        "units_note": "angles in degrees; angle_length_tolerance in angstrom; strain and mismatch values as fractions (0.01 = 1%)",
        "top_c_repeat": int(top_c_repeat),
        "bottom_c_repeat": int(bottom_c_repeat),
        "workers": int(workers),
        "max_pair_matches": "" if effective_max_pair_matches is None else int(effective_max_pair_matches),
        "cull_redundant": bool(cull_redundant),
        "reduce_basis": bool(reduce_basis),
        "max_cell_aspect_ratio": "" if max_cell_aspect_ratio is None else float(max_cell_aspect_ratio),
        "min_cell_angle_deg": "" if min_cell_angle_deg is None else float(min_cell_angle_deg),
        "max_cell_angle_deg": "" if max_cell_angle_deg is None else float(max_cell_angle_deg),
        "nindex": int(nindex),
        "symmetry_top_deg": int(symmetry_top),
        "symmetry_bottom_deg": int(symmetry_bottom),
        "symmetry_lcm_deg": int(symmetry_lcm),
        "min_angle_deg": float(search_min_angle),
        "max_angle_deg": float(search_max_angle),
        "angle_count": len(angle_values),
        "angle_step_deg": float(angle_step),
        "explicit_angles_deg": ",".join(f"{value:.6f}" for value in explicit_angles) if explicit_angles else "",
        "angle_length_tolerance": float(angle_length_tolerance),
        "angle_strain_tolerance": "" if angle_strain_tolerance is None else float(angle_strain_tolerance),
        "angle_merge_tolerance": float(angle_merge_tolerance),
        "vector_tolerance": float(vector_tolerance),
        "vector_strain_tolerance": "" if vector_strain_tolerance is None else float(vector_strain_tolerance),
        "candidate_tolerance": float(candidate_tolerance if candidate_tolerance is not None else vector_tolerance),
        "strain_tolerance": float(effective_strain_tolerance),
        "strain_tolerance_defaulted": strain_tolerance is None,
        "strain_layer": strain_layer,
        "min_atoms": "" if min_atoms is None else int(min_atoms),
        "max_atoms": "" if max_atoms is None else int(max_atoms),
        "dedupe": bool(dedupe),
        "unique_strain_tolerance": float(unique_strain_tolerance),
        "unique_ratio_tolerance": float(unique_ratio_tolerance),
        "output_angle_tolerance_deg": float(finder_backend.ANGLE_OUTPUT_TOLERANCE_DEG),
        "output_strain_tolerance": float(finder_backend.STRAIN_OUTPUT_TOLERANCE),
        "matrix_values": "" if matrix_values is None else ",".join(str(int(value)) for value in matrix_values),
        "matrix_layer": matrix_layer,
        "matrix_match_mode": matrix_match_mode,
        "shortlisted_angle_count": len(shortlist),
        "angle_values_thinned": bool(angle_metadata.get("angle_values_thinned", False)),
    }
    parameters.update(angle_metadata)
    parameters["timing_angle_shortlist_s"] = round(timings["angle_shortlist_s"], 6)
    parameters["timing_supercell_search_s"] = round(timings["supercell_search_s"], 6)
    if shortlist:
        parameters["shortlisted_angles_deg"] = ",".join(f"{candidate.angle_deg:.6f}" for candidate in shortlist)

    finder_backend.write_results_dat(
        str(dat_path),
        top_poscar,
        bottom_poscar,
        candidates,
        run_id=run_id,
        parameters=parameters,
    )
    timings["write_results_s"] = time.perf_counter() - stage_start
    timings["total_s"] = time.perf_counter() - total_start
    parameters["timing_write_results_s"] = round(timings["write_results_s"], 6)
    parameters["timing_total_s"] = round(timings["total_s"], 6)
    _notify(progress_callback, "write", f"wrote results in {timings['write_results_s']:.3f}s")

    return FindRun(
        run_id=run_id,
        dat_path=dat_path,
        candidates=list(candidates),
        shortlisted_angles=list(shortlist),
        angle_values=angle_values,
        symmetry_top=symmetry_top,
        symmetry_bottom=symmetry_bottom,
        symmetry_lcm=symmetry_lcm,
        search_min_angle=search_min_angle,
        search_max_angle=search_max_angle,
        parameters=parameters,
        timings=timings,
    )


def find(**kwargs):
    return run_find(**kwargs)

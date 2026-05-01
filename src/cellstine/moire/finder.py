"""Finder backend for commensurate moire supercells."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Mapping, Sequence

import numpy as np

from . import lattice as lat


def _build_angle_list(
    angle_lower: float | None,
    angle_upper: float | None,
    angle_step: float,
    angles: Sequence[float] | None,
) -> List[float]:
    if angles:
        return sorted({float(angle) for angle in angles})
    if angle_lower is None or angle_upper is None:
        raise ValueError("must provide either an explicit angle list or numeric angle bounds")
    if angle_step <= 0.0:
        raise ValueError("angle_step must be positive")
    values = np.arange(angle_lower, angle_upper + angle_step * 0.5, angle_step, dtype=float)
    return [float(value) for value in values]


def _limit_worker_threads() -> None:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def _find_candidates_for_angle(
    task: tuple[np.ndarray, np.ndarray, float, int, float, float | None, float, int, int]
) -> List[lat.SupercellCandidate]:
    (
        lattice1,
        lattice2,
        angle_deg,
        nindex,
        tolerance,
        vector_strain_tol,
        candidate_tolerance,
        atom_count1,
        atom_count2,
    ) = task
    rotated_lattice1 = lat.rotate_lattice(lattice1, angle_deg)
    matches = lat.find_coincident_vector_pairs(
        rotated_lattice1,
        lattice2,
        nindex,
        tolerance,
        strain_tolerance=vector_strain_tol,
    )
    return lat.build_supercell_candidates(
        matches,
        rotated_lattice1,
        lattice2,
        atom_count1,
        atom_count2,
        candidate_tolerance,
        angle_deg,
    )


def _matrix_signature(values: Sequence[int], match_mode: str) -> tuple[int, int, int, int]:
    entries = [int(value) for value in values]
    if len(entries) != 4:
        raise ValueError("matrix filters require exactly four integer values")
    if match_mode == "absolute":
        entries = [abs(value) for value in entries]
    elif match_mode != "exact":
        raise ValueError("matrix_match_mode must be 'absolute' or 'exact'")
    return tuple(sorted(entries))


def _candidate_matrix_entries(candidate: lat.SupercellCandidate, layer: str) -> tuple[int, int, int, int]:
    if layer == "1":
        return (
            int(candidate.layer1_vector1[0]),
            int(candidate.layer1_vector1[1]),
            int(candidate.layer1_vector2[0]),
            int(candidate.layer1_vector2[1]),
        )
    if layer == "2":
        return (
            int(candidate.layer2_vector1[0]),
            int(candidate.layer2_vector1[1]),
            int(candidate.layer2_vector2[0]),
            int(candidate.layer2_vector2[1]),
        )
    raise ValueError("layer must be '1' or '2'")


def candidate_matches_matrix_values(
    candidate: lat.SupercellCandidate,
    matrix_values: Sequence[int],
    *,
    matrix_layer: str = "either",
    matrix_match_mode: str = "absolute",
) -> bool:
    target_signature = _matrix_signature(matrix_values, matrix_match_mode)
    layer1_signature = _matrix_signature(_candidate_matrix_entries(candidate, "1"), matrix_match_mode)
    layer2_signature = _matrix_signature(_candidate_matrix_entries(candidate, "2"), matrix_match_mode)

    if matrix_layer == "1":
        return layer1_signature == target_signature
    if matrix_layer == "2":
        return layer2_signature == target_signature
    if matrix_layer == "either":
        return layer1_signature == target_signature or layer2_signature == target_signature
    if matrix_layer == "both":
        return layer1_signature == target_signature and layer2_signature == target_signature
    raise ValueError("matrix_layer must be '1', '2', 'either', or 'both'")


def find_supercells(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    angle_lower: float | None,
    angle_upper: float | None,
    angle_step: float = 0.001,
    nindex: int = 10,
    tol: float = 0.002,
    lin_tol: float | None = None,
    strain_tol: float | None = None,
    strain_layer: str = "avg",
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    atom_count1: int = 1,
    atom_count2: int = 1,
    angles: Sequence[float] | None = None,
    dedupe: bool = True,
    unique_strain_tol: float = 1e-4,
    unique_ratio_tol: float = 1e-5,
    vector_strain_tol: float | None = None,
    matrix_values: Sequence[int] | None = None,
    matrix_layer: str = "either",
    matrix_match_mode: str = "absolute",
    workers: int = 1,
) -> List[lat.SupercellCandidate]:
    candidate_tolerance = lin_tol if lin_tol is not None else tol
    angle_values = _build_angle_list(angle_lower, angle_upper, angle_step, angles)
    lattice1_array = np.asarray(lattice1, dtype=float)
    lattice2_array = np.asarray(lattice2, dtype=float)

    all_candidates: List[lat.SupercellCandidate] = []
    resolved_workers = max(1, int(workers))
    task_inputs = [
        (
            lattice1_array,
            lattice2_array,
            float(angle_deg),
            int(nindex),
            float(tol),
            vector_strain_tol,
            float(candidate_tolerance),
            int(atom_count1),
            int(atom_count2),
        )
        for angle_deg in angle_values
    ]
    if resolved_workers <= 1 or len(task_inputs) <= 1:
        for task in task_inputs:
            all_candidates.extend(_find_candidates_for_angle(task))
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers, initializer=_limit_worker_threads) as executor:
                for candidates in executor.map(_find_candidates_for_angle, task_inputs):
                    all_candidates.extend(candidates)
        except (OSError, PermissionError):
            for task in task_inputs:
                all_candidates.extend(_find_candidates_for_angle(task))

    resolved_strain_layer = str(strain_layer).lower()
    if resolved_strain_layer not in {"avg", "1", "2"}:
        raise ValueError("strain_layer must be 'avg', '1', or '2'")

    filtered: List[lat.SupercellCandidate] = []
    for candidate in all_candidates:
        if min_atoms is not None and candidate.total_atoms < min_atoms:
            continue
        if max_atoms is not None and candidate.total_atoms > max_atoms:
            continue
        if strain_tol is not None:
            if resolved_strain_layer == "avg" and candidate.strain_avg > strain_tol:
                continue
            if resolved_strain_layer == "1" and candidate.strain_layer1 > strain_tol:
                continue
            if resolved_strain_layer == "2" and candidate.strain_layer2 > strain_tol:
                continue
        if matrix_values is not None and not candidate_matches_matrix_values(
            candidate,
            matrix_values,
            matrix_layer=matrix_layer,
            matrix_match_mode=matrix_match_mode,
        ):
            continue
        filtered.append(candidate)

    if dedupe:
        filtered = lat.deduplicate_candidates(
            filtered,
            strain_tolerance=unique_strain_tol,
            ratio_tolerance=unique_ratio_tol,
        )

    filtered.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.angle_deg, item.vector_product))
    return filtered


def candidate_to_dict(candidate: lat.SupercellCandidate, index: int | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "angle_deg": float(candidate.angle_deg),
        "strain_avg": float(candidate.strain_avg),
        "strain_layer1": float(candidate.strain_layer1),
        "strain_layer2": float(candidate.strain_layer2),
        "ratio1": int(candidate.ratio1),
        "ratio2": int(candidate.ratio2),
        "total_atoms": int(candidate.total_atoms),
        "layer1_vector1": [int(candidate.layer1_vector1[0]), int(candidate.layer1_vector1[1])],
        "layer1_vector2": [int(candidate.layer1_vector2[0]), int(candidate.layer1_vector2[1])],
        "layer2_vector1": [int(candidate.layer2_vector1[0]), int(candidate.layer2_vector1[1])],
        "layer2_vector2": [int(candidate.layer2_vector2[0]), int(candidate.layer2_vector2[1])],
        "eps1": float(candidate.eps1),
        "eps2": float(candidate.eps2),
        "vector_product": float(candidate.vector_product),
        "area1": float(candidate.area1),
        "area2": float(candidate.area2),
    }
    if index is not None:
        payload["index"] = int(index)
    return payload


def format_results_table(candidates: Sequence[lat.SupercellCandidate], limit: int | None = None) -> str:
    shown = list(candidates if limit is None or limit < 0 else candidates[: max(limit, 0)])
    if not shown:
        return "No candidates found."

    header = (
        " idx  angle(deg)   strain_avg    strain_1      strain_2    atoms   ratio"
        "      i1          i2          j1          j2          eps1        eps2"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for index, candidate in enumerate(shown, start=1):
        lines.append(
            "{idx:4d}  {angle:10.4f}  {strain_avg:11.6f}  {strain1:11.6f}  {strain2:11.6f}  {atoms:7d}  "
            "{ratio1:3d}/{ratio2:<3d}  ({i11:3d},{i12:3d})  ({i21:3d},{i22:3d})  "
            "({j11:3d},{j12:3d})  ({j21:3d},{j22:3d})  {eps1:10.2e}  {eps2:10.2e}".format(
                idx=index,
                angle=candidate.angle_deg,
                strain_avg=candidate.strain_avg,
                strain1=candidate.strain_layer1,
                strain2=candidate.strain_layer2,
                atoms=candidate.total_atoms,
                ratio1=candidate.ratio1,
                ratio2=candidate.ratio2,
                i11=candidate.layer1_vector1[0],
                i12=candidate.layer1_vector1[1],
                i21=candidate.layer1_vector2[0],
                i22=candidate.layer1_vector2[1],
                j11=candidate.layer2_vector1[0],
                j12=candidate.layer2_vector1[1],
                j21=candidate.layer2_vector2[0],
                j22=candidate.layer2_vector2[1],
                eps1=candidate.eps1,
                eps2=candidate.eps2,
            )
        )
    return "\n".join(lines)


def write_results_dat(
    path: str,
    pos1: str,
    pos2: str,
    candidates: Sequence[lat.SupercellCandidate],
    *,
    run_id: str,
    parameters: Mapping[str, object],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{pos1} {pos2}\n")
        handle.write(f"# run_id = {run_id}\n")
        handle.write(f"# created_at = {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write("# credit = CELLSTINE (CELL Superlattice Transformation INterface and Engine) | Made by Sarwin Chandran\n")
        handle.write("# units = angles in degrees; strain and mismatch values are fractions (0.01 = 1%)\n")
        handle.write("# note = strain_avg is the symmetric strain measure; strain1 and strain2 are the one-sided layer strain measures\n")
        handle.write(
            "| idx | angle (deg) | strain_avg | strain1 | strain2 | atoms | ratio | i11 i12 | i21 i22 | j11 j12 | j21 j22 | eps1 | eps2 |\n"
        )
        handle.write("-" * 125 + "\n")
        for index, candidate in enumerate(candidates, start=1):
            i11, i12 = candidate.layer1_vector1
            i21, i22 = candidate.layer1_vector2
            j11, j12 = candidate.layer2_vector1
            j21, j22 = candidate.layer2_vector2
            handle.write(
                "|{idx:4d} | {angle:10.4f} | {strain_avg:10.6f} | {strain1:7.6f} | {strain2:7.6f} | {atoms:5d} | {ratio1:3d}/{ratio2:<3d} | {i11:4d} {i12:4d} | {i21:4d} {i22:4d} | {j11:4d} {j12:4d} | {j21:4d} {j22:4d} | {eps1:8.2e} | {eps2:8.2e} |\n".format(
                    idx=index,
                    angle=candidate.angle_deg,
                    strain_avg=candidate.strain_avg,
                    strain1=candidate.strain_layer1,
                    strain2=candidate.strain_layer2,
                    atoms=candidate.total_atoms,
                    ratio1=candidate.ratio1,
                    ratio2=candidate.ratio2,
                    i11=i11,
                    i12=i12,
                    i21=i21,
                    i22=i22,
                    j11=j11,
                    j12=j12,
                    j21=j21,
                    j22=j22,
                    eps1=candidate.eps1,
                    eps2=candidate.eps2,
                )
            )
        handle.write("\n")
        handle.write("# parameters\n")
        for key, value in parameters.items():
            handle.write(f"# {key} = {value}\n")

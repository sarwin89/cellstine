"""Finder backend for commensurate moire supercells.

Made by Sarwin Chandran.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Mapping, Sequence

import numpy as np

from . import lattice as lat


# Made by Sarwin Chandran: this module hosts the commensuration finder backend.


def _build_angle_list(
    angle_lower: float | None,
    angle_upper: float | None,
    angle_step: float,
    angles: Sequence[float] | None,
) -> List[float]:
    if angles:
        return [float(angle) for angle in angles]
    if angle_lower is None or angle_upper is None:
        raise ValueError("must provide either an explicit angle list or numeric angle bounds")
    if angle_step <= 0.0:
        raise ValueError("angle_step must be positive")
    values = np.arange(angle_lower, angle_upper + angle_step * 0.5, angle_step, dtype=float)
    return [float(value) for value in values]


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
) -> List[lat.SupercellCandidate]:
    """Return commensurate supercell candidates between two lattices."""

    candidate_tolerance = lin_tol if lin_tol is not None else tol
    angle_values = _build_angle_list(angle_lower, angle_upper, angle_step, angles)

    all_candidates: List[lat.SupercellCandidate] = []
    for angle_deg in angle_values:
        rotated_lattice1 = lat.rotate_lattice(lattice1, angle_deg)
        matches = lat.find_coincident_vector_pairs(
            rotated_lattice1,
            lattice2,
            nindex,
            tol,
            strain_tolerance=vector_strain_tol,
        )
        candidates = lat.build_supercell_candidates(
            matches,
            rotated_lattice1,
            lattice2,
            atom_count1,
            atom_count2,
            candidate_tolerance,
            angle_deg,
        )
        all_candidates.extend(candidates)

    filtered: List[lat.SupercellCandidate] = []
    for candidate in all_candidates:
        if min_atoms is not None and candidate.total_atoms < min_atoms:
            continue
        if max_atoms is not None and candidate.total_atoms > max_atoms:
            continue
        if strain_tol is not None:
            if strain_layer == "avg" and candidate.strain_avg > strain_tol:
                continue
            if strain_layer == "1" and candidate.strain_layer1 > strain_tol:
                continue
            if strain_layer == "2" and candidate.strain_layer2 > strain_tol:
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
    """Serialize one supercell candidate into a JSON-friendly dictionary."""

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
    """Format candidates as a fixed-width CLI table."""

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
    """Write one DAT file with results plus the parameters used to create them."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{pos1} {pos2}\n")
        handle.write(f"# run_id = {run_id}\n")
        handle.write(f"# created_at = {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write("# credit = Made by Sarwin Chandran\n")
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

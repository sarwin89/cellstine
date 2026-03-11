"""Fast commensurate-angle search based on equal-length lattice spans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import io as io_mod
from . import lattice as lat


@dataclass(frozen=True)
class AngleCandidate:
    """A candidate commensurate twist angle from span matching."""

    angle_deg: float
    coeffs1: Tuple[int, int]
    coeffs2: Tuple[int, int]
    length1: float
    length2: float
    relative_mismatch: float


def _canonicalise_directions(coeffs: np.ndarray, vectors: np.ndarray, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    first_component = vectors[:, 0]
    second_component = vectors[:, 1]
    orientation = np.where(
        np.abs(first_component) > tolerance,
        np.sign(first_component),
        np.where(np.abs(second_component) > tolerance, np.sign(second_component), 1.0),
    )
    orientation[orientation == 0.0] = 1.0
    oriented_vectors = vectors * orientation[:, None]
    oriented_coeffs = coeffs * orientation[:, None].astype(int)

    rounded = np.round(oriented_vectors, decimals=8)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return oriented_coeffs[unique_indices], oriented_vectors[unique_indices]


def _merge_nearby_angles(candidates: Sequence[AngleCandidate], merge_tolerance: float) -> List[AngleCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.angle_deg, item.relative_mismatch))
    merged: List[AngleCandidate] = []
    for candidate in ordered:
        if not merged or abs(candidate.angle_deg - merged[-1].angle_deg) > merge_tolerance:
            merged.append(candidate)
            continue
        if candidate.relative_mismatch < merged[-1].relative_mismatch:
            merged[-1] = candidate
    return merged


def find_commensurate_angles(
    lattice1: np.ndarray,
    lattice2: np.ndarray,
    nindex: int,
    *,
    length_tolerance: float = 1e-5,
    strain_tolerance: float | None = None,
    min_angle: float = 0.0,
    max_angle: float = 90.0,
    angle_round_decimals: int = 4,
    merge_tolerance: float = 1e-3,
) -> List[AngleCandidate]:
    """Find unique candidate commensurate angles from matching span lengths."""

    coeffs1, vectors1 = lat.enumerate_in_plane_vectors(lattice1, nindex)
    coeffs2, vectors2 = lat.enumerate_in_plane_vectors(lattice2, nindex)
    coeffs1, vectors1 = _canonicalise_directions(coeffs1, vectors1)
    coeffs2, vectors2 = _canonicalise_directions(coeffs2, vectors2)

    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)
    average_norm = np.maximum((norms1[:, None] + norms2[None, :]) * 0.5, 1e-12)
    absolute_mismatch = np.abs(norms1[:, None] - norms2[None, :])
    relative_mismatch = absolute_mismatch / average_norm

    length_mask = absolute_mismatch <= length_tolerance
    if strain_tolerance is not None:
        length_mask |= relative_mismatch <= strain_tolerance

    dot_products = vectors1 @ vectors2.T
    norm_products = np.maximum(norms1[:, None] * norms2[None, :], 1e-12)
    cosines = np.clip(dot_products / norm_products, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    cross_products = vectors1[:, None, 0] * vectors2[None, :, 1] - vectors1[:, None, 1] * vectors2[None, :, 0]

    valid_mask = (
        length_mask
        & (np.abs(cross_products) > 1e-10)
        & (angles >= min_angle)
        & (angles <= max_angle)
    )

    row_indices, col_indices = np.nonzero(valid_mask)
    raw_candidates: List[AngleCandidate] = []
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        raw_candidates.append(
            AngleCandidate(
                angle_deg=round(float(angles[row_index, col_index]), angle_round_decimals),
                coeffs1=(int(coeffs1[row_index, 0]), int(coeffs1[row_index, 1])),
                coeffs2=(int(coeffs2[col_index, 0]), int(coeffs2[col_index, 1])),
                length1=float(norms1[row_index]),
                length2=float(norms2[col_index]),
                relative_mismatch=float(relative_mismatch[row_index, col_index]),
            )
        )

    return _merge_nearby_angles(raw_candidates, merge_tolerance)


def format_angle_table(candidates: Sequence[AngleCandidate]) -> str:
    """Format candidate angles as a simple GitHub-friendly table."""

    lines = [
        "| idx | angle (deg) | coeffs1 | coeffs2 | length1 | length2 | rel mismatch |",
        "| ---:| ----------: | :------ | :------ | ------: | ------: | -----------: |",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            "| {0} | {1:.4f} | ({2},{3}) | ({4},{5}) | {6:.6f} | {7:.6f} | {8:.3e} |".format(
                index,
                candidate.angle_deg,
                candidate.coeffs1[0],
                candidate.coeffs1[1],
                candidate.coeffs2[0],
                candidate.coeffs2[1],
                candidate.length1,
                candidate.length2,
                candidate.relative_mismatch,
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find commensurate twist angles from two POSCAR files")
    parser.add_argument("pos1", help="first POSCAR")
    parser.add_argument("pos2", help="second POSCAR")
    parser.add_argument("nindex", type=int, help="integer span from -nindex to nindex")
    parser.add_argument("--length_tolerance", type=float, default=1e-5, help="absolute tolerance when matching vector lengths")
    parser.add_argument("--strain_tolerance", type=float, default=None, help="relative length-mismatch tolerance")
    parser.add_argument("--min_angle", type=float, default=0.0, help="minimum angle to report")
    parser.add_argument("--max_angle", type=float, default=90.0, help="maximum angle to report")
    parser.add_argument("--merge_tolerance", type=float, default=1e-3, help="merge nearby angles within this tolerance")
    parser.add_argument("--output", type=str, default=None, help="optional file to write the angle table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure1 = io_mod.read_poscar(args.pos1)
    structure2 = io_mod.read_poscar(args.pos2)
    candidates = find_commensurate_angles(
        structure1.lattice,
        structure2.lattice,
        args.nindex,
        length_tolerance=args.length_tolerance,
        strain_tolerance=args.strain_tolerance,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        merge_tolerance=args.merge_tolerance,
    )
    table = format_angle_table(candidates)
    print(table)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(table + '\n')


if __name__ == "__main__":
    main()

"""Fast commensurate-angle search based on equal-length lattice spans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Help formatter with visible defaults and readable examples."""


@dataclass(frozen=True)
class AngleCandidate:
    """A candidate commensurate twist angle from span matching."""

    angle_deg: float
    coeffs1: Tuple[int, int]
    coeffs2: Tuple[int, int]
    length1: float
    length2: float
    relative_mismatch: float


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
    max_angle: float | None = None,
    angle_round_decimals: int = 4,
    merge_tolerance: float = 1e-3,
) -> List[AngleCandidate]:
    """Find unique candidate commensurate angles from matching span lengths.

    This follows the fast `cellfind` idea:
    - generate full integer spans with NumPy
    - match vectors by equal length (or allowed relative mismatch)
    - compute the rotation angle between every matching pair
    - keep only angles in the symmetry-limited search window
    """
    import numpy as np

    from . import lattice as lat

    _, _, symmetry_lcm = lat.combined_symmetry_limit(lattice1, lattice2)
    bounded_min = max(0.0, float(min_angle))
    bounded_max = float(symmetry_lcm if max_angle is None else min(float(max_angle), symmetry_lcm))
    if bounded_max < bounded_min:
        return []

    coeffs1, vectors1 = lat.enumerate_in_plane_vectors(lattice1, nindex)
    coeffs2, vectors2 = lat.enumerate_in_plane_vectors(lattice2, nindex)

    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)
    average_norm = np.maximum((norms1[:, None] + norms2[None, :]) * 0.5, 1e-12)
    absolute_mismatch = np.abs(norms1[:, None] - norms2[None, :])
    relative_mismatch = absolute_mismatch / average_norm

    length_mask = np.isclose(norms1[:, None], norms2[None, :], atol=length_tolerance, rtol=0.0)
    if strain_tolerance is not None:
        length_mask |= relative_mismatch <= strain_tolerance

    dot_products = vectors1 @ vectors2.T
    cross_products = vectors1[:, None, 0] * vectors2[None, :, 1] - vectors1[:, None, 1] * vectors2[None, :, 0]
    angles = np.degrees(np.arctan2(np.abs(cross_products), dot_products))

    valid_mask = (
        length_mask
        & (np.abs(cross_products) > 1e-10)
        & (angles >= bounded_min)
        & (angles <= bounded_max)
    )

    row_indices, col_indices = np.nonzero(valid_mask)
    if row_indices.size == 0:
        return []

    selected_angles = np.round(angles[row_indices, col_indices], decimals=angle_round_decimals)
    selected_mismatch = relative_mismatch[row_indices, col_indices]
    selected_lengths1 = norms1[row_indices]
    selected_lengths2 = norms2[col_indices]
    selected_coeffs1 = coeffs1[row_indices]
    selected_coeffs2 = coeffs2[col_indices]

    order = np.lexsort((selected_mismatch, selected_angles))
    sorted_angles = selected_angles[order]
    keep_sorted = np.ones(sorted_angles.shape[0], dtype=bool)
    if sorted_angles.shape[0] > 1:
        keep_sorted[1:] = np.diff(sorted_angles) > merge_tolerance
    keep_indices = order[keep_sorted]

    return [
        AngleCandidate(
            angle_deg=float(selected_angles[index]),
            coeffs1=(int(selected_coeffs1[index, 0]), int(selected_coeffs1[index, 1])),
            coeffs2=(int(selected_coeffs2[index, 0]), int(selected_coeffs2[index, 1])),
            length1=float(selected_lengths1[index]),
            length2=float(selected_lengths2[index]),
            relative_mismatch=float(selected_mismatch[index]),
        )
        for index in keep_indices.tolist()
    ]


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
    parser = argparse.ArgumentParser(
        description="Find a fast shortlist of commensurate twist angles from two POSCAR files.",
        epilog=(
            "Units:\n"
            "  angles are in degrees\n"
            "  length_tolerance is in angstrom\n"
            "  strain_tolerance is a fraction (0.01 = 1 percent)\n\n"
            "Example:\n"
            "  python -m moire.angles input/a.vasp input/b.vasp 12 --strain_tolerance 0.002\n"
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("pos1", help="path to the first POSCAR")
    parser.add_argument("pos2", help="path to the second POSCAR")
    parser.add_argument("nindex", type=int, help="integer span search from -nindex to +nindex")
    parser.add_argument("--length_tolerance", type=float, default=1e-5, help="absolute span-length mismatch allowed in angstrom")
    parser.add_argument("--strain_tolerance", type=float, default=None, help="relative span-length mismatch allowed as a fraction (0.01 = 1 percent)")
    parser.add_argument("--min_angle", type=float, default=0.0, help="minimum angle to report, in degrees")
    parser.add_argument("--max_angle", type=float, default=None, help="maximum angle to report in degrees; defaults to the symmetry LCM")
    parser.add_argument("--merge_tolerance", type=float, default=1e-3, help="merge nearby angles within this tolerance, in degrees")
    parser.add_argument("--output", type=str, default=None, help="optional file to write the shortlist table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from . import io as io_mod

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

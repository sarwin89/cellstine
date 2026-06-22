"""Fast commensurate-angle search based on equal-length lattice spans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import lattice as lat


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
    if strain_tolerance is not None:
        strain_tolerance = min(float(strain_tolerance), lat.MAX_PHYSICAL_MISMATCH)
    _, _, symmetry_lcm = lat.combined_symmetry_limit(lattice1, lattice2)
    bounded_min = max(0.0, float(min_angle))
    bounded_max = float(symmetry_lcm if max_angle is None else min(float(max_angle), symmetry_lcm))
    if bounded_max < bounded_min:
        return []

    # Dynamically scale nindex based on cell size limits
    basis1_2d = lattice1[:2, :2]
    basis2_2d = lattice2[:2, :2]
    basis1_lengths = np.linalg.norm(basis1_2d, axis=1)
    basis2_lengths = np.linalg.norm(basis2_2d, axis=1)
    
    L1_max = nindex * np.max(basis1_lengths)
    L2_max = nindex * np.max(basis2_lengths)
    cutoff_tolerance = float(strain_tolerance) if strain_tolerance is not None else 0.05
    L_cutoff = min(L1_max, L2_max) * (1.0 + cutoff_tolerance)
    
    min_basis1 = np.min(basis1_lengths)
    min_basis2 = np.min(basis2_lengths)
    
    scaled_nindex1 = int(min(nindex, np.ceil(L_cutoff / max(min_basis1, 1e-12))))
    scaled_nindex2 = int(min(nindex, np.ceil(L_cutoff / max(min_basis2, 1e-12))))

    coeffs1, vectors1 = lat.enumerate_in_plane_vectors(lattice1, scaled_nindex1)
    coeffs2, vectors2 = lat.enumerate_in_plane_vectors(lattice2, scaled_nindex2)

    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)

    # Commensurate angles arise only from equal-length spans, so we restrict the
    # search to near-equal-length lattice-vector pairs (length is rotation
    # invariant).  The admissible length band has half-width proportional to the
    # strain tolerance, so when a *physical* (few-percent) strain tolerance is
    # requested the number of length-matched pairs grows as O(N^2) = O(nindex^4)
    # -- tens of millions of pairs at nindex >= 100.  Materialising them all at
    # once is what previously exhausted memory (and crashed) at large nindex.
    #
    # Instead we sweep the pair set one row-block at a time and immediately
    # reduce each block to a single survivor per distinct (rounded) angle before
    # accumulating.  Peak memory is therefore bounded by the per-block budget
    # plus the (small) number of distinct angles, never by the full O(nindex^4)
    # pair count.  The reduction keeps the smallest-mismatch member of every
    # rounded-angle value, which is exactly what the final selection below would
    # have kept had every pair been scanned at once, so the result is identical.
    candidate_rel_tol = float(strain_tolerance) if strain_tolerance is not None else 0.0

    if norms1.shape[0] == 0 or norms2.shape[0] == 0:
        return []

    order2 = np.argsort(norms2, kind="stable")
    sorted2 = norms2[order2]
    band = float(length_tolerance) + 2.0 * candidate_rel_tol * norms1
    lower = np.searchsorted(sorted2, norms1 - band, side="left")
    upper = np.searchsorted(sorted2, norms1 + band, side="right")
    counts = (upper - lower).astype(np.int64)
    n_rows = int(norms1.shape[0])

    # Cap on the number of candidate pairs held in memory at once.  Each pair
    # touches a handful of float64 temporaries, so ~1M pairs per block keeps the
    # per-block working set at tens of MB regardless of nindex.
    pair_budget = 1_000_000
    # The accumulated survivors (one per distinct rounded angle per block) are
    # periodically compacted to one entry per distinct rounded angle globally, so
    # the running set can never exceed the (small) number of realisable angles.
    compact_cap = 2_000_000

    acc_angles: List[np.ndarray] = []
    acc_mismatch: List[np.ndarray] = []
    acc_coeffs1: List[np.ndarray] = []
    acc_coeffs2: List[np.ndarray] = []
    acc_lengths1: List[np.ndarray] = []
    acc_lengths2: List[np.ndarray] = []
    acc_size = 0

    def _compact() -> None:
        nonlocal acc_size
        if not acc_angles:
            return
        ang = np.concatenate(acc_angles)
        mis = np.concatenate(acc_mismatch)
        c1 = np.concatenate(acc_coeffs1)
        c2 = np.concatenate(acc_coeffs2)
        l1 = np.concatenate(acc_lengths1)
        l2 = np.concatenate(acc_lengths2)
        comp_order = np.lexsort((mis, ang))
        comp_sorted = ang[comp_order]
        first = np.empty(comp_sorted.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(comp_sorted[1:], comp_sorted[:-1], out=first[1:])
        idx = comp_order[first]
        acc_angles[:] = [ang[idx]]
        acc_mismatch[:] = [mis[idx]]
        acc_coeffs1[:] = [c1[idx]]
        acc_coeffs2[:] = [c2[idx]]
        acc_lengths1[:] = [l1[idx]]
        acc_lengths2[:] = [l2[idx]]
        acc_size = int(idx.shape[0])

    block_start = 0
    while block_start < n_rows:
        block_end = block_start + 1
        block_total = int(counts[block_start])
        while block_end < n_rows and block_total + int(counts[block_end]) <= pair_budget:
            block_total += int(counts[block_end])
            block_end += 1
        rows = np.arange(block_start, block_end, dtype=np.intp)
        block_counts = counts[block_start:block_end]
        block_start = block_end
        total = int(block_counts.sum())
        if total == 0:
            continue

        rep_rows = np.repeat(rows, block_counts)
        seg_start = np.repeat(lower[rows], block_counts)
        within = np.arange(total, dtype=np.intp) - np.repeat(
            np.cumsum(block_counts) - block_counts, block_counts
        )
        cols = order2[seg_start + within]

        v1_cand = vectors1[rep_rows]
        v2_cand = vectors2[cols]
        norms1_cand = norms1[rep_rows]
        norms2_cand = norms2[cols]

        average_norm = np.maximum((norms1_cand + norms2_cand) * 0.5, 1e-12)
        relative_mismatch_cand = np.abs(norms1_cand - norms2_cand) / average_norm

        length_mask = np.abs(norms1_cand - norms2_cand) <= length_tolerance
        if strain_tolerance is not None:
            length_mask |= relative_mismatch_cand <= strain_tolerance

        dot_products = np.einsum("ij,ij->i", v1_cand, v2_cand)
        cross_products = v1_cand[:, 0] * v2_cand[:, 1] - v1_cand[:, 1] * v2_cand[:, 0]
        angles = np.degrees(np.arctan2(np.abs(cross_products), dot_products))

        valid_mask = (
            length_mask
            & (np.abs(cross_products) > 1e-10)
            & (angles >= bounded_min)
            & (angles <= bounded_max)
        )
        keep = np.nonzero(valid_mask)[0]
        if keep.size == 0:
            continue

        rounded = np.round(angles[keep], decimals=angle_round_decimals)
        mismatch = relative_mismatch_cand[keep]
        # Reduce this block to the smallest-mismatch member of every distinct
        # rounded angle (the only member the final selection could retain).
        block_order = np.lexsort((mismatch, rounded))
        sorted_rounded = rounded[block_order]
        first_of_angle = np.empty(sorted_rounded.shape[0], dtype=bool)
        first_of_angle[0] = True
        np.not_equal(sorted_rounded[1:], sorted_rounded[:-1], out=first_of_angle[1:])
        survivors = block_order[first_of_angle]
        kept = keep[survivors]

        acc_angles.append(rounded[survivors])
        acc_mismatch.append(mismatch[survivors])
        acc_coeffs1.append(coeffs1[rep_rows[kept]])
        acc_coeffs2.append(coeffs2[cols[kept]])
        acc_lengths1.append(norms1_cand[kept])
        acc_lengths2.append(norms2_cand[kept])
        acc_size += int(survivors.shape[0])
        if acc_size > compact_cap:
            _compact()

    if not acc_angles:
        return []

    selected_angles = np.concatenate(acc_angles)
    selected_mismatch = np.concatenate(acc_mismatch)
    selected_lengths1 = np.concatenate(acc_lengths1)
    selected_lengths2 = np.concatenate(acc_lengths2)
    selected_coeffs1 = np.concatenate(acc_coeffs1)
    selected_coeffs2 = np.concatenate(acc_coeffs2)

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
            "  python -m cellstine.moire.search.angles input/a.vasp input/b.vasp 12 --strain_tolerance 0.002\n"
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
    from ...io import native as io_mod

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
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")


if __name__ == "__main__":
    main()

"""Search for commensurate moire supercells over twist angles."""

from __future__ import annotations

import argparse
from typing import List, Sequence

import numpy as np

from . import angles as angle_mod
from . import io as io_mod
from . import lattice as lat


def _build_angle_list(
    angle_lower: float | None,
    angle_upper: float | None,
    angle_step: float,
    angles: Sequence[float] | None,
) -> List[float]:
    if angles:
        return [float(angle) for angle in angles]
    if angle_lower is None or angle_upper is None:
        raise ValueError("must provide either --angles or angle_lower/angle_upper")
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
    """Return supercell candidates between two lattices."""

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


def format_results_table(candidates: Sequence[lat.SupercellCandidate], limit: int | None = None) -> str:
    """Format candidates as a readable GitHub-style table."""

    shown = list(candidates if limit is None else candidates[: max(limit, 0)])
    lines = [
        "| idx | angle (deg) | strain_avg | strain1 | strain2 | atoms | ratio | i11 i12 | i21 i22 | j11 j12 | j21 j22 | eps1 | eps2 |",
        "| ---:| ----------: | ---------: | ------: | ------: | ----: | :---- | :------ | :------ | :------ | :------ | ---: | ---: |",
    ]
    for index, candidate in enumerate(shown, start=1):
        lines.append(
            "| {idx} | {angle:.4f} | {strain_avg:.6f} | {strain1:.6f} | {strain2:.6f} | {atoms} | {ratio1}/{ratio2} | {i11} {i12} | {i21} {i22} | {j11} {j12} | {j21} {j22} | {eps1:.2e} | {eps2:.2e} |".format(
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


def write_results(path: str, pos1: str, pos2: str, candidates: Sequence[lat.SupercellCandidate]) -> None:
    """Write finder results in a generator-friendly table."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{pos1} {pos2}\n")
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


def _resolve_angles(args: argparse.Namespace, lattice1: np.ndarray, lattice2: np.ndarray) -> tuple[float | None, float | None, List[float] | None]:
    if args.angles:
        return None, None, [float(token.strip()) for token in args.angles.split(",") if token.strip()]
    if args.use_cellfind:
        angle_candidates = angle_mod.find_commensurate_angles(
            lattice1,
            lattice2,
            args.nindex,
            length_tolerance=args.angle_length_tolerance,
            strain_tolerance=args.angle_strain_tolerance,
            min_angle=args.angle_lower if args.angle_lower is not None else 0.0,
            max_angle=args.angle_upper if args.angle_upper is not None else 90.0,
        )
        return None, None, [candidate.angle_deg for candidate in angle_candidates]
    return args.angle_lower, args.angle_upper, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find commensurate supercells between two POSCAR files")
    parser.add_argument("pos1", help="first POSCAR; this layer is rotated during the scan")
    parser.add_argument("pos2", help="second POSCAR; this layer is kept fixed")
    parser.add_argument("angle_lower", nargs="?", type=float, default=None, help="lower bound for angle scan in degrees")
    parser.add_argument("angle_upper", nargs="?", type=float, default=None, help="upper bound for angle scan in degrees")
    parser.add_argument("--angles", type=str, default=None, help="comma-separated list of exact angles in degrees")
    parser.add_argument("--use_cellfind", action="store_true", help="precompute candidate angles from equal-length spans before running the full finder")
    parser.add_argument("--angle_step", type=float, default=0.001, help="angle step in degrees when scanning a range")
    parser.add_argument("--nindex", type=int, default=10, help="integer span from -nindex to nindex")
    parser.add_argument("--tolerance", type=float, default=0.002, help="relative tolerance for vector coincidence")
    parser.add_argument("--lin_tol", type=float, default=None, help="relative tolerance when pairing candidate vectors into cells")
    parser.add_argument("--vector_strain_tol", type=float, default=None, help="relative length-mismatch tolerance when matching vectors")
    parser.add_argument("--strain_tol", "--maxstrain", dest="strain_tol", type=float, default=None, help="discard candidates above this strain threshold")
    parser.add_argument("--strain_layer", choices=["avg", "1", "2"], default="avg", help="which strain column to filter with --strain_tol")
    parser.add_argument("--angle_length_tolerance", type=float, default=1e-5, help="absolute length tolerance used by --use_cellfind")
    parser.add_argument("--angle_strain_tolerance", type=float, default=None, help="relative length-mismatch tolerance used by --use_cellfind")
    parser.add_argument("--min_atoms", type=int, default=None, help="minimum combined atoms in the supercell")
    parser.add_argument("--max_atoms", "--maxatoms", dest="max_atoms", type=int, default=None, help="maximum combined atoms in the supercell")
    parser.add_argument("--output", type=str, default="results.dat", help="output results filename")
    parser.add_argument("--no_dedupe", action="store_true", help="keep near-duplicate candidates")
    parser.add_argument("--unique_strain_tol", type=float, default=1e-4, help="strain tolerance for duplicate collapse")
    parser.add_argument("--unique_ratio_tol", type=float, default=1e-5, help="ratio tolerance for duplicate collapse")
    parser.add_argument("--top", type=int, default=10, help="number of best rows to echo after writing results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    structure1 = io_mod.read_poscar(args.pos1)
    structure2 = io_mod.read_poscar(args.pos2)
    angle_lower, angle_upper, angles = _resolve_angles(args, structure1.lattice, structure2.lattice)

    if args.use_cellfind:
        print(f"cellfind shortlisted {len(angles or [])} candidate angle(s)")

    candidates = find_supercells(
        structure1.lattice,
        structure2.lattice,
        angle_lower,
        angle_upper,
        angle_step=args.angle_step,
        nindex=args.nindex,
        tol=args.tolerance,
        lin_tol=args.lin_tol,
        strain_tol=args.strain_tol,
        strain_layer=args.strain_layer,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        atom_count1=structure1.natoms,
        atom_count2=structure2.natoms,
        angles=angles,
        dedupe=not args.no_dedupe,
        unique_strain_tol=args.unique_strain_tol,
        unique_ratio_tol=args.unique_ratio_tol,
        vector_strain_tol=args.vector_strain_tol,
    )

    write_results(args.output, args.pos1, args.pos2, candidates)
    print(f"Wrote {len(candidates)} candidate(s) to {args.output}")
    if candidates:
        best = candidates[0]
        print(
            "Best candidate: angle={0:.4f} deg, strain={1:.6f}, atoms={2}, ratio={3}/{4}".format(
                best.angle_deg,
                best.strain_avg,
                best.total_atoms,
                best.ratio1,
                best.ratio2,
            )
        )
        print(format_results_table(candidates, limit=args.top))
    else:
        print("No candidates found with the supplied parameters.")


if __name__ == "__main__":
    main()

"""Generate exact supercells from finder results."""

from __future__ import annotations

import argparse
import math
import os
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import io as io_mod
from . import lattice as lat


def parse_results(filename: str) -> Tuple[str, str, List[dict]]:
    """Parse new-style or legacy finder results files."""

    records: List[dict] = []
    with open(filename, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split()
        if len(first_line) < 2:
            raise ValueError("results file does not contain the two input filenames on the first line")
        file1, file2 = first_line[0], first_line[1]

        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("-") or not stripped.startswith("|"):
                continue
            parts = [part.strip() for part in stripped.split("|") if part.strip()]
            if not parts or parts[0].lower() == "idx":
                continue

            if len(parts) >= 13:
                ratio1, ratio2 = [int(value) for value in parts[6].split("/")]
                i11, i12 = [int(value) for value in parts[7].split()]
                i21, i22 = [int(value) for value in parts[8].split()]
                j11, j12 = [int(value) for value in parts[9].split()]
                j21, j22 = [int(value) for value in parts[10].split()]
                records.append(
                    {
                        "idx": int(parts[0]),
                        "angle": float(parts[1]),
                        "strain_avg": float(parts[2]),
                        "strain1": float(parts[3]),
                        "strain2": float(parts[4]),
                        "atoms": int(parts[5]),
                        "ratio1": ratio1,
                        "ratio2": ratio2,
                        "i11": i11,
                        "i12": i12,
                        "i21": i21,
                        "i22": i22,
                        "j11": j11,
                        "j12": j12,
                        "j21": j21,
                        "j22": j22,
                        "eps1": float(parts[11]),
                        "eps2": float(parts[12]),
                    }
                )
            elif len(parts) >= 6:
                ratio1, ratio2 = [int(value) for value in parts[3].split()[:2]]
                i11, i12, i21, i22 = [int(value) for value in parts[4].split()[:4]]
                j11, j12, j21, j22 = [int(value) for value in parts[5].split()[:4]]
                records.append(
                    {
                        "idx": int(parts[0]),
                        "angle": 0.0,
                        "strain_avg": float(parts[1]),
                        "strain1": float(parts[1]),
                        "strain2": float(parts[1]),
                        "atoms": int(parts[2]),
                        "ratio1": ratio1,
                        "ratio2": ratio2,
                        "i11": i11,
                        "i12": i12,
                        "i21": i21,
                        "i22": i22,
                        "j11": j11,
                        "j12": j12,
                        "j21": j21,
                        "j22": j22,
                        "eps1": 0.0,
                        "eps2": 0.0,
                    }
                )
    return file1, file2, records


def _expand_species(species: Sequence[str], counts: Sequence[int], fallback: str) -> List[str]:
    if species:
        labels = list(species)
    else:
        labels = [fallback] * len(counts)
    expanded: List[str] = []
    for symbol, count in zip(labels, counts):
        expanded.extend([symbol] * int(count))
    return expanded


def _search_range(coef_a: int, coef_b: int, tolerance_r: int) -> range:
    sign_a = 1 if coef_a >= 0 else -1
    sign_b = 1 if coef_b >= 0 else -1
    if sign_a != sign_b:
        lower = min(coef_a, coef_b) - tolerance_r
        upper = max(coef_a, coef_b) + tolerance_r
    else:
        lower = min(coef_a + coef_b, 0) - tolerance_r
        upper = max(coef_a + coef_b, 0) + tolerance_r
    return range(int(lower), int(upper))


def _is_duplicate(existing: Sequence[np.ndarray], candidate: np.ndarray, tolerance: float) -> bool:
    for previous in existing:
        dx = previous[0] - candidate[0]
        dy = previous[1] - candidate[1]
        dz = previous[2] - candidate[2]
        if abs(dx - round(dx)) < tolerance and abs(dy - round(dy)) < tolerance and abs(dz) < tolerance:
            return True
    return False


def _shift_vector(lattice: np.ndarray, direct_shift: Sequence[float], cart_shift: Sequence[float]) -> np.ndarray:
    direct_shift_array = np.asarray(direct_shift, dtype=float)
    cart_shift_array = np.asarray(cart_shift, dtype=float)
    return (
        direct_shift_array[0] * lattice[0]
        + direct_shift_array[1] * lattice[1]
        + direct_shift_array[2] * lattice[2]
        + cart_shift_array
    )


def _relax_flags(z_value: float, zfix: float | None) -> Tuple[str, str, str] | None:
    if zfix is None:
        return None
    if zfix <= 0.0:
        return ("F", "F", "F") if z_value > abs(zfix) else ("T", "T", "T")
    return ("F", "F", "F") if z_value < zfix else ("T", "T", "T")


def _replicate_layer(
    positions_direct: np.ndarray,
    source_lattice: np.ndarray,
    source_supercell: np.ndarray,
    final_lattice: np.ndarray,
    coef_pair1: Tuple[int, int],
    coef_pair2: Tuple[int, int],
    shift_direct: Sequence[float],
    shift_cart: Sequence[float],
    tolerance: int,
    tolerance_float: float,
    species: Sequence[str],
    selective_flags: Sequence[Tuple[str, str, str]] | None,
    zfix: float | None,
) -> List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]:
    """Replicate one layer into the selected supercell."""

    results: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    accepted_source_positions: List[np.ndarray] = []
    shift_vector = _shift_vector(source_lattice, shift_direct, shift_cart)

    range1 = _search_range(coef_pair1[0], coef_pair2[0], tolerance)
    range2 = _search_range(coef_pair1[1], coef_pair2[1], tolerance)

    for atom_index, base_direct in enumerate(np.asarray(positions_direct, dtype=float)):
        for shift1 in range1:
            for shift2 in range2:
                direct_image = np.array(
                    [shift1 + base_direct[0], shift2 + base_direct[1], base_direct[2]],
                    dtype=float,
                )
                cartesian_image = io_mod.direct_to_cartesian(direct_image.reshape(1, 3), source_lattice)[0]
                source_direct = io_mod.cartesian_to_direct(cartesian_image.reshape(1, 3), source_supercell)[0]
                if not (
                    -tolerance_float <= source_direct[0] <= 1.0 + tolerance_float
                    and -tolerance_float <= source_direct[1] <= 1.0 + tolerance_float
                ):
                    continue

                wrapped_source = np.array(
                    [source_direct[0] % 1.0, source_direct[1] % 1.0, source_direct[2]],
                    dtype=float,
                )
                if _is_duplicate(accepted_source_positions, wrapped_source, tolerance_float):
                    continue
                accepted_source_positions.append(wrapped_source)

                shifted_cartesian = cartesian_image + shift_vector
                final_direct = io_mod.cartesian_to_direct(shifted_cartesian.reshape(1, 3), final_lattice)[0]
                final_direct = io_mod.wrap_direct(final_direct.reshape(1, 3))[0]
                final_flag = _relax_flags(
                    io_mod.direct_to_cartesian(final_direct.reshape(1, 3), final_lattice)[0][2],
                    zfix,
                )
                if final_flag is None and selective_flags is not None:
                    final_flag = tuple(selective_flags[atom_index])
                results.append((species[atom_index], final_direct, final_flag))
    return results


def _finalise_species_order(
    atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]],
) -> Tuple[np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    grouped: "OrderedDict[str, List[Tuple[np.ndarray, Tuple[str, str, str] | None]]]" = OrderedDict()
    any_flags = False
    for species, position, flags in atoms:
        grouped.setdefault(species, []).append((position, flags))
        any_flags = any_flags or flags is not None

    ordered_positions: List[np.ndarray] = []
    ordered_counts: List[int] = []
    ordered_species: List[str] = []
    ordered_flags: List[Tuple[str, str, str]] = []
    for species, entries in grouped.items():
        ordered_species.append(species)
        ordered_counts.append(len(entries))
        for position, flags in entries:
            ordered_positions.append(position)
            ordered_flags.append(flags if flags is not None else ("T", "T", "T"))

    positions_array = np.array(ordered_positions, dtype=float) if ordered_positions else np.zeros((0, 3), dtype=float)
    return positions_array, ordered_counts, ordered_species, (ordered_flags if any_flags else None)


def _swap_if_left_handed(lattice: np.ndarray, positions_direct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    area = lattice[0, 0] * lattice[1, 1] - lattice[0, 1] * lattice[1, 0]
    if area >= 0.0:
        return lattice, positions_direct
    swapped_lattice = np.vstack((lattice[1], lattice[0], lattice[2]))
    swapped_positions = positions_direct[:, [1, 0, 2]]
    return swapped_lattice, swapped_positions


def build_supercell(
    pos1: str,
    pos2: str,
    coef: dict,
    *,
    shift1_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift1_cart: Sequence[float] = (0.0, 0.0, 0.0),
    shift2_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift2_cart: Sequence[float] = (0.0, 0.0, 0.0),
    tolerance: int = 1,
    tolerance_float: float = 1e-4,
    preserve_layer: str = "2",
    zfix: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    """Build the exact supercell defined by one finder result record."""

    structure1 = io_mod.read_poscar(pos1)
    structure2 = io_mod.read_poscar(pos2)

    angle = float(coef.get("angle", 0.0))
    rotated_lattice1 = lat.rotate_lattice(structure1.lattice, angle)

    v1 = coef["i11"] * rotated_lattice1[0] + coef["i12"] * rotated_lattice1[1]
    v2 = coef["i21"] * rotated_lattice1[0] + coef["i22"] * rotated_lattice1[1]
    g1 = coef["j11"] * structure2.lattice[0] + coef["j12"] * structure2.lattice[1]
    g2 = coef["j21"] * structure2.lattice[0] + coef["j22"] * structure2.lattice[1]

    layer1_supercell = np.vstack((v1, v2, rotated_lattice1[2]))
    layer2_supercell = np.vstack((g1, g2, structure2.lattice[2]))

    preserve_mode = str(preserve_layer).lower()
    if preserve_mode in {"1", "layer1", "first"}:
        final_lattice = layer1_supercell.copy()
    elif preserve_mode in {"avg", "average"}:
        average_c = structure2.lattice[2] if np.linalg.norm(structure2.lattice[2]) >= np.linalg.norm(rotated_lattice1[2]) else rotated_lattice1[2]
        final_lattice = np.vstack(((v1 + g1) / 2.0, (v2 + g2) / 2.0, average_c))
    else:
        final_lattice = layer2_supercell.copy()

    species1 = _expand_species(structure1.species, structure1.counts, "L1")
    species2 = _expand_species(structure2.species, structure2.counts, "L2")

    atoms_layer1 = _replicate_layer(
        structure1.positions_direct,
        rotated_lattice1,
        layer1_supercell,
        final_lattice,
        (coef["i11"], coef["i12"]),
        (coef["i21"], coef["i22"]),
        shift1_direct,
        shift1_cart,
        tolerance,
        tolerance_float,
        species1,
        structure1.selective_flags,
        zfix,
    )
    atoms_layer2 = _replicate_layer(
        structure2.positions_direct,
        structure2.lattice,
        layer2_supercell,
        final_lattice,
        (coef["j11"], coef["j12"]),
        (coef["j21"], coef["j22"]),
        shift2_direct,
        shift2_cart,
        tolerance,
        tolerance_float,
        species2,
        structure2.selective_flags,
        zfix,
    )

    expected_layer1 = structure1.natoms * int(coef.get("ratio1", 0) or 0)
    expected_layer2 = structure2.natoms * int(coef.get("ratio2", 0) or 0)
    if expected_layer1 and len(atoms_layer1) != expected_layer1:
        raise ValueError(
            f"layer 1 atom count mismatch: expected {expected_layer1}, found {len(atoms_layer1)}; try increasing tolerance"
        )
    if expected_layer2 and len(atoms_layer2) != expected_layer2:
        raise ValueError(
            f"layer 2 atom count mismatch: expected {expected_layer2}, found {len(atoms_layer2)}; try increasing tolerance"
        )

    positions_direct, counts, species, flags = _finalise_species_order(atoms_layer1 + atoms_layer2)
    final_lattice, positions_direct = _swap_if_left_handed(final_lattice, positions_direct)
    return final_lattice, positions_direct, counts, species, flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a moire supercell from finder results")
    parser.add_argument("results", help="results file written by finder.py")
    parser.add_argument("index", type=int, help="1-based solution index to generate")
    parser.add_argument("--output", default="supercell.vasp", help="output POSCAR filename")
    parser.add_argument("--tolerance", type=int, default=1, help="integer image padding during atom replication")
    parser.add_argument("--tolerance_float", type=float, default=1e-4, help="floating tolerance during atom replication")
    parser.add_argument("--preserve_layer", default="2", help="which matched lattice to preserve: 1, 2, or avg")
    parser.add_argument("--shift11", type=float, default=0.0)
    parser.add_argument("--shift12", type=float, default=0.0)
    parser.add_argument("--shift13", type=float, default=0.0)
    parser.add_argument("--shift1x", type=float, default=0.0)
    parser.add_argument("--shift1y", type=float, default=0.0)
    parser.add_argument("--shift1z", type=float, default=0.0)
    parser.add_argument("--shift21", type=float, default=0.0)
    parser.add_argument("--shift22", type=float, default=0.0)
    parser.add_argument("--shift23", type=float, default=0.0)
    parser.add_argument("--shift2x", type=float, default=0.0)
    parser.add_argument("--shift2y", type=float, default=0.0)
    parser.add_argument("--shift2z", type=float, default=0.0)
    parser.add_argument("--zfix", type=float, default=None, help="if set, write Selective Dynamics flags based on z height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file1, file2, records = parse_results(args.results)
    by_index = {record["idx"]: record for record in records}
    if args.index not in by_index:
        raise ValueError(f"index {args.index} not found in {args.results}")

    lattice, positions_direct, counts, species, flags = build_supercell(
        file1,
        file2,
        by_index[args.index],
        shift1_direct=(args.shift11, args.shift12, args.shift13),
        shift1_cart=(args.shift1x, args.shift1y, args.shift1z),
        shift2_direct=(args.shift21, args.shift22, args.shift23),
        shift2_cart=(args.shift2x, args.shift2y, args.shift2z),
        tolerance=args.tolerance,
        tolerance_float=args.tolerance_float,
        preserve_layer=args.preserve_layer,
        zfix=args.zfix,
    )

    comment = f"moire supercell from {os.path.basename(args.results)} index {args.index}"
    io_mod.write_poscar(
        args.output,
        lattice,
        positions_direct,
        counts,
        species,
        comment=comment,
        positions_are_cartesian=False,
        selective_flags=flags,
    )
    print(f"Wrote supercell to {args.output}")


if __name__ == "__main__":
    main()

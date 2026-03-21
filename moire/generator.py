"""Generator backend for exact moire supercell construction.

Made by Sarwin Chandran.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import io as io_mod
from . import lattice as lat


# Made by Sarwin Chandran: this module hosts the supercell generator backend.


def record_from_candidate_dict(candidate: Dict[str, object], index: int | None = None) -> Dict[str, object]:
    """Convert a serialized finder candidate into generator coefficients."""

    payload: Dict[str, object] = {
        "idx": int(index) if index is not None else int(candidate.get("index", 0)),
        "angle": float(candidate["angle_deg"]),
        "ratio1": int(candidate["ratio1"]),
        "ratio2": int(candidate["ratio2"]),
        "i11": int(candidate["layer1_vector1"][0]),
        "i12": int(candidate["layer1_vector1"][1]),
        "i21": int(candidate["layer1_vector2"][0]),
        "i22": int(candidate["layer1_vector2"][1]),
        "j11": int(candidate["layer2_vector1"][0]),
        "j12": int(candidate["layer2_vector1"][1]),
        "j21": int(candidate["layer2_vector2"][0]),
        "j22": int(candidate["layer2_vector2"][1]),
    }
    return payload


def parse_results(filename: str) -> Tuple[str, str, List[dict], dict]:
    """Parse finder results from JSON or legacy DAT format."""

    if filename.lower().endswith(".json"):
        with open(filename, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = payload.get("meta", {})
        top_path = meta.get("top_poscar") or meta.get("pos1")
        bottom_path = meta.get("bottom_poscar") or meta.get("pos2")
        if not top_path or not bottom_path:
            raise ValueError("JSON results do not contain top/bottom POSCAR metadata")
        candidates = payload.get("candidates", [])
        records = [record_from_candidate_dict(candidate, idx + 1) for idx, candidate in enumerate(candidates)]
        return str(top_path), str(bottom_path), records, payload

    records: List[dict] = []
    meta: Dict[str, object] = {}
    with open(filename, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split()
        if len(first_line) < 2:
            raise ValueError("results file does not contain the two input filenames on the first line")
        file1, file2 = first_line[0], first_line[1]

        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                content = stripped[1:].strip()
                if "=" in content:
                    key, value = content.split("=", 1)
                    meta[key.strip()] = value.strip()
                continue
            if not stripped or stripped.startswith("-") or not stripped.startswith("|"):
                continue
            parts = [part.strip() for part in stripped.split("|") if part.strip()]
            if not parts or parts[0].lower() == "idx":
                continue

            ratio1, ratio2 = [int(value) for value in parts[6].split("/")]
            i11, i12 = [int(value) for value in parts[7].split()]
            i21, i22 = [int(value) for value in parts[8].split()]
            j11, j12 = [int(value) for value in parts[9].split()]
            j21, j22 = [int(value) for value in parts[10].split()]
            records.append(
                {
                    "idx": int(parts[0]),
                    "angle": float(parts[1]),
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
                }
            )
    return file1, file2, records, {"meta": meta}


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


def _replicate_layer_cartesian(
    positions_direct: np.ndarray,
    source_lattice: np.ndarray,
    source_supercell: np.ndarray,
    coef_pair1: Tuple[int, int],
    coef_pair2: Tuple[int, int],
    shift_direct: Sequence[float],
    shift_cart: Sequence[float],
    tolerance: int,
    tolerance_float: float,
    species: Sequence[str],
    selective_flags: Sequence[Tuple[str, str, str]] | None,
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
                source_flag = tuple(selective_flags[atom_index]) if selective_flags is not None else None
                results.append((species[atom_index], shifted_cartesian, source_flag))
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


def _scale_vector(vector: np.ndarray, length: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, float(length)], dtype=float)
    return np.asarray(vector, dtype=float) * (float(length) / norm)


def _reference_c_vector(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    if float(np.linalg.norm(vector2)) >= float(np.linalg.norm(vector1)):
        return np.asarray(vector2, dtype=float)
    return np.asarray(vector1, dtype=float)


def _shift_atoms_z(
    atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]],
    delta_z: float,
) -> List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]:
    shifted: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    for species, position, flags in atoms:
        updated = np.array(position, dtype=float, copy=True)
        updated[2] += float(delta_z)
        shifted.append((species, updated, flags))
    return shifted


def _z_bounds(atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]) -> Tuple[float, float]:
    if not atoms:
        return 0.0, 0.0
    z_values = [float(position[2]) for _, position, _ in atoms]
    return min(z_values), max(z_values)


def _build_final_lattice(
    in_plane_vector1: np.ndarray,
    in_plane_vector2: np.ndarray,
    reference_c: np.ndarray,
    atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]],
    tolerance_float: float,
) -> Tuple[np.ndarray, float, float]:
    min_z, max_z = _z_bounds(atoms)
    z_span = max_z - min_z
    reference_length = float(np.linalg.norm(reference_c))
    lower_padding = max(float(tolerance_float), 1e-3)
    padding = 2.0 * lower_padding
    c_length = max(reference_length, z_span + padding)
    final_c = _scale_vector(reference_c, c_length)
    final_lattice = np.vstack((in_plane_vector1, in_plane_vector2, final_c))
    return final_lattice, min_z, lower_padding


def _finalise_cartesian_atoms(
    atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]],
    final_lattice: np.ndarray,
    zfix: float | None,
) -> Tuple[np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    if not atoms:
        return np.zeros((0, 3), dtype=float), [], [], None

    cartesian_positions = np.array([position for _, position, _ in atoms], dtype=float)
    direct_positions = io_mod.wrap_direct(io_mod.cartesian_to_direct(cartesian_positions, final_lattice))

    final_atoms: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    for atom_index, (species, _, flags) in enumerate(atoms):
        final_flag = _relax_flags(float(cartesian_positions[atom_index][2]), zfix)
        if final_flag is None:
            final_flag = flags
        final_atoms.append((species, direct_positions[atom_index], final_flag))
    return _finalise_species_order(final_atoms)


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
    interlayer_distance: float | None = None,
    preserve_layer: str = "2",
    zfix: float | None = None,
    repeat1_c: int = 1,
    repeat2_c: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    """Build the exact supercell defined by one finder result record."""

    structure1 = io_mod.repeat_structure_along_c(io_mod.read_poscar(pos1), repeat1_c)
    structure2 = io_mod.repeat_structure_along_c(io_mod.read_poscar(pos2), repeat2_c)

    angle = float(coef.get("angle", 0.0))
    rotated_lattice1 = lat.rotate_lattice(structure1.lattice, angle)

    v1 = coef["i11"] * rotated_lattice1[0] + coef["i12"] * rotated_lattice1[1]
    v2 = coef["i21"] * rotated_lattice1[0] + coef["i22"] * rotated_lattice1[1]
    g1 = coef["j11"] * structure2.lattice[0] + coef["j12"] * structure2.lattice[1]
    g2 = coef["j21"] * structure2.lattice[0] + coef["j22"] * structure2.lattice[1]

    preserve_mode = str(preserve_layer).lower()
    if preserve_mode in {"1", "layer1", "first"}:
        final_vector1 = v1.copy()
        final_vector2 = v2.copy()
    elif preserve_mode in {"avg", "average"}:
        final_vector1 = (v1 + g1) / 2.0
        final_vector2 = (v2 + g2) / 2.0
    else:
        final_vector1 = g1.copy()
        final_vector2 = g2.copy()

    layer1_supercell = np.vstack((v1, v2, rotated_lattice1[2]))
    layer2_supercell = np.vstack((g1, g2, structure2.lattice[2]))
    reference_c = _reference_c_vector(rotated_lattice1[2], structure2.lattice[2])

    species1 = _expand_species(structure1.species, structure1.counts, "L1")
    species2 = _expand_species(structure2.species, structure2.counts, "L2")

    atoms_layer1 = _replicate_layer_cartesian(
        structure1.positions_direct,
        rotated_lattice1,
        layer1_supercell,
        (coef["i11"], coef["i12"]),
        (coef["i21"], coef["i22"]),
        shift1_direct,
        shift1_cart,
        tolerance,
        tolerance_float,
        species1,
        structure1.selective_flags,
    )
    atoms_layer2 = _replicate_layer_cartesian(
        structure2.positions_direct,
        structure2.lattice,
        layer2_supercell,
        (coef["j11"], coef["j12"]),
        (coef["j21"], coef["j22"]),
        shift2_direct,
        shift2_cart,
        tolerance,
        tolerance_float,
        species2,
        structure2.selective_flags,
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

    if interlayer_distance is not None and atoms_layer1 and atoms_layer2:
        top_min_z, _ = _z_bounds(atoms_layer1)
        _, bottom_max_z = _z_bounds(atoms_layer2)
        atoms_layer1 = _shift_atoms_z(atoms_layer1, bottom_max_z + float(interlayer_distance) - top_min_z)

    all_atoms = atoms_layer1 + atoms_layer2
    final_lattice, min_z, lower_padding = _build_final_lattice(
        final_vector1,
        final_vector2,
        reference_c,
        all_atoms,
        tolerance_float,
    )
    z_shift = lower_padding - min_z
    all_atoms = _shift_atoms_z(all_atoms, z_shift)

    positions_direct, counts, species, flags = _finalise_cartesian_atoms(all_atoms, final_lattice, zfix)
    final_lattice, positions_direct = _swap_if_left_handed(final_lattice, positions_direct)
    return final_lattice, positions_direct, counts, species, flags


def write_supercell_poscar(
    output_path: str,
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    counts: Sequence[int],
    species: Sequence[str],
    flags: Sequence[Sequence[str]] | None,
    comment: str,
) -> None:
    """Write a generated supercell POSCAR."""

    io_mod.write_poscar(
        output_path,
        lattice,
        positions_direct,
        counts,
        species,
        comment=comment,
        positions_are_cartesian=False,
        selective_flags=flags,
    )

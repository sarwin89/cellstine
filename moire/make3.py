"""Trilayer supercell generation for CELLSTINE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import find3 as find3_backend
from . import generator as generator_backend
from . import io as io_mod
from . import lattice as lattice_backend

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class Make3Run:
    output_path: Path
    selected_index: int
    angle_middle_deg: float
    angle_top_deg: float
    total_atoms: int


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _longest_c_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    longest = np.asarray(vectors[0], dtype=float)
    longest_norm = float(np.linalg.norm(longest))
    for vector in vectors[1:]:
        candidate = np.asarray(vector, dtype=float)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > longest_norm:
            longest = candidate
            longest_norm = candidate_norm
    return longest


def _build_layer_atoms(
    structure: io_mod.PoscarData,
    source_lattice: np.ndarray,
    matrix_v1: Sequence[int],
    matrix_v2: Sequence[int],
) -> list[tuple[str, np.ndarray, Tuple[str, str, str] | None]]:
    supercell = np.vstack(
        (
            int(matrix_v1[0]) * source_lattice[0] + int(matrix_v1[1]) * source_lattice[1],
            int(matrix_v2[0]) * source_lattice[0] + int(matrix_v2[1]) * source_lattice[1],
            source_lattice[2],
        )
    )
    species = generator_backend._expand_species(structure.species, structure.counts, "L")
    return generator_backend._replicate_layer_cartesian(
        structure.positions_direct,
        source_lattice,
        supercell,
        (int(matrix_v1[0]), int(matrix_v1[1])),
        (int(matrix_v2[0]), int(matrix_v2[1])),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        1,
        1e-4,
        species,
        structure.selective_flags,
    )


def build_trilayer_supercell(
    bottom_poscar: str,
    middle_poscar: str,
    top_poscar: str,
    candidate: Dict[str, object],
    *,
    interlayer_bottom_middle: float,
    interlayer_middle_top: float,
    bottom_c_repeat: int = 1,
    middle_c_repeat: int = 1,
    top_c_repeat: int = 1,
    zfix: float | None = None,
) -> tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    bottom = io_mod.repeat_structure_along_c(io_mod.read_poscar(bottom_poscar), bottom_c_repeat)
    middle = io_mod.repeat_structure_along_c(io_mod.read_poscar(middle_poscar), middle_c_repeat)
    top = io_mod.repeat_structure_along_c(io_mod.read_poscar(top_poscar), top_c_repeat)

    bottom_vector1 = tuple(int(value) for value in candidate["bottom_vector1"])
    bottom_vector2 = tuple(int(value) for value in candidate["bottom_vector2"])
    middle_vector1 = tuple(int(value) for value in candidate["middle_vector1"])
    middle_vector2 = tuple(int(value) for value in candidate["middle_vector2"])
    top_vector1 = tuple(int(value) for value in candidate["top_vector1"])
    top_vector2 = tuple(int(value) for value in candidate["top_vector2"])

    middle_lattice_rotated = lattice_backend.rotate_lattice(middle.lattice, float(candidate["angle_middle_deg"]))
    top_lattice_rotated = lattice_backend.rotate_lattice(top.lattice, float(candidate["angle_top_deg"]))

    atoms_bottom = _build_layer_atoms(bottom, bottom.lattice, bottom_vector1, bottom_vector2)
    atoms_middle = _build_layer_atoms(middle, middle_lattice_rotated, middle_vector1, middle_vector2)
    atoms_top = _build_layer_atoms(top, top_lattice_rotated, top_vector1, top_vector2)

    if atoms_bottom and atoms_middle:
        middle_min_z, _ = generator_backend._z_bounds(atoms_middle)
        _, bottom_max_z = generator_backend._z_bounds(atoms_bottom)
        atoms_middle = generator_backend._shift_atoms_z(
            atoms_middle,
            bottom_max_z + float(interlayer_bottom_middle) - middle_min_z,
        )
    if atoms_top:
        top_min_z, _ = generator_backend._z_bounds(atoms_top)
        _, lower_max_z = generator_backend._z_bounds(atoms_middle if atoms_middle else atoms_bottom)
        atoms_top = generator_backend._shift_atoms_z(
            atoms_top,
            lower_max_z + float(interlayer_middle_top) - top_min_z,
        )

    final_vector1 = int(bottom_vector1[0]) * bottom.lattice[0] + int(bottom_vector1[1]) * bottom.lattice[1]
    final_vector2 = int(bottom_vector2[0]) * bottom.lattice[0] + int(bottom_vector2[1]) * bottom.lattice[1]
    reference_c = _longest_c_vector([bottom.lattice[2], middle_lattice_rotated[2], top_lattice_rotated[2]])

    all_atoms = atoms_top + atoms_middle + atoms_bottom
    final_lattice, min_z, lower_padding = generator_backend._build_final_lattice(
        final_vector1,
        final_vector2,
        reference_c,
        all_atoms,
        1e-4,
    )
    all_atoms = generator_backend._shift_atoms_z(all_atoms, lower_padding - min_z)
    positions_direct, counts, species, flags = generator_backend._finalise_cartesian_atoms(all_atoms, final_lattice, zfix)
    final_lattice, positions_direct = generator_backend._swap_if_left_handed(final_lattice, positions_direct)
    return final_lattice, positions_direct, counts, species, flags


def generate_from_results(
    results_file: str,
    *,
    index: int,
    interlayer_bottom_middle: float,
    interlayer_middle_top: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    bottom_c_repeat: int | None = None,
    middle_c_repeat: int | None = None,
    top_c_repeat: int | None = None,
    zfix: float | None = None,
) -> Make3Run:
    meta, candidates = find3_backend.parse_results(results_file)
    by_index = {int(candidate["index"]): candidate for candidate in candidates}
    if index not in by_index:
        raise ValueError(f"index {index} not found in {results_file}")

    candidate = by_index[index]
    resolved_bottom_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else meta.get("bottom_c_repeat", 1))
    resolved_middle_repeat = int(middle_c_repeat if middle_c_repeat is not None else meta.get("middle_c_repeat", 1))
    resolved_top_repeat = int(top_c_repeat if top_c_repeat is not None else meta.get("top_c_repeat", 1))

    lattice_out, positions_direct, counts, species, flags = build_trilayer_supercell(
        str(meta["bottom_poscar"]),
        str(meta["middle_poscar"]),
        str(meta["top_poscar"]),
        candidate,
        interlayer_bottom_middle=float(interlayer_bottom_middle),
        interlayer_middle_top=float(interlayer_middle_top),
        bottom_c_repeat=resolved_bottom_repeat,
        middle_c_repeat=resolved_middle_repeat,
        top_c_repeat=resolved_top_repeat,
        zfix=zfix,
    )

    total_atoms = int(sum(counts))
    if output_path is None:
        destination_dir = DEFAULT_OUTPUT_DIR.resolve() if output_dir is None else Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_name = (
            f"stack3_idx{index:03d}_mid{float(candidate['angle_middle_deg']):.4f}_top{float(candidate['angle_top_deg']):.4f}_"
            f"atoms{total_atoms}_{_slug(Path(str(meta['bottom_poscar'])).stem)}-bottom_"
            f"{_slug(Path(str(meta['middle_poscar'])).stem)}-middle_{_slug(Path(str(meta['top_poscar'])).stem)}-top.vasp"
        )
        output_path = str(destination_dir / output_name)
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    generator_backend.write_supercell_poscar(
        output_path,
        lattice_out,
        positions_direct,
        counts,
        species,
        flags,
        comment="Generated by CELLSTINE make3 stage | Made by Sarwin Chandran",
    )
    return Make3Run(
        output_path=Path(output_path).resolve(),
        selected_index=int(index),
        angle_middle_deg=float(candidate["angle_middle_deg"]),
        angle_top_deg=float(candidate["angle_top_deg"]),
        total_atoms=total_atoms,
    )


def generate_many_from_results(
    results_file: str,
    *,
    indexes: Sequence[int],
    interlayer_bottom_middle: float,
    interlayer_middle_top: float,
    output_dir: str | None = None,
    bottom_c_repeat: int | None = None,
    middle_c_repeat: int | None = None,
    top_c_repeat: int | None = None,
    zfix: float | None = None,
) -> List[Make3Run]:
    runs: List[Make3Run] = []
    for index in indexes:
        runs.append(
            generate_from_results(
                results_file,
                index=int(index),
                interlayer_bottom_middle=interlayer_bottom_middle,
                interlayer_middle_top=interlayer_middle_top,
                output_dir=output_dir,
                bottom_c_repeat=bottom_c_repeat,
                middle_c_repeat=middle_c_repeat,
                top_c_repeat=top_c_repeat,
                zfix=zfix,
            )
        )
    return runs

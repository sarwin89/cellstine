"""N-layer supercell generation stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ..search import findn as findn_backend
from ..search import lattice as lattice_backend
from . import generator as generator_backend
from ...io import native as io_mod

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class MakeNRun:
    output_path: Path
    selected_index: int
    angles_deg: tuple[float, ...]
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


def build_nlayer_supercell(
    bottom_poscar: str,
    upper_poscars: Sequence[str],
    candidate: dict[str, object],
    *,
    interlayers: Sequence[float],
    bottom_c_repeat: int = 1,
    upper_c_repeats: Sequence[int] | None = None,
    zfix: float | None = None,
) -> tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    upper_layer_specs = list(candidate["upper_layers"])
    if len(interlayers) != len(upper_layer_specs):
        raise ValueError("maken needs exactly one interlayer distance per upper layer")
    if upper_c_repeats is None:
        upper_c_repeats = [1] * len(upper_layer_specs)
    if len(upper_c_repeats) != len(upper_layer_specs):
        raise ValueError("upper c-repeat list must match the number of upper layers")

    bottom = io_mod.repeat_structure_along_c(io_mod.read_poscar(bottom_poscar), bottom_c_repeat)
    upper_structures = [
        io_mod.repeat_structure_along_c(io_mod.read_poscar(path), int(repeat))
        for path, repeat in zip(upper_poscars, upper_c_repeats)
    ]

    bottom_vector1 = tuple(int(value) for value in candidate["bottom_vector1"])
    bottom_vector2 = tuple(int(value) for value in candidate["bottom_vector2"])
    bottom_atoms = _build_layer_atoms(bottom, bottom.lattice, bottom_vector1, bottom_vector2)

    upper_layers_atoms: List[list[tuple[str, np.ndarray, Tuple[str, str, str] | None]]] = []
    c_vectors = [bottom.lattice[2]]
    for structure, layer_spec in zip(upper_structures, upper_layer_specs):
        rotated_lattice = lattice_backend.rotate_lattice(structure.lattice, float(layer_spec["angle_deg"]))
        upper_layers_atoms.append(
            _build_layer_atoms(
                structure,
                rotated_lattice,
                tuple(int(value) for value in layer_spec["vector1"]),
                tuple(int(value) for value in layer_spec["vector2"]),
            )
        )
        c_vectors.append(rotated_lattice[2])

    stacked_layers: List[list[tuple[str, np.ndarray, Tuple[str, str, str] | None]]] = [bottom_atoms]
    for gap, atoms in zip(interlayers, upper_layers_atoms):
        if stacked_layers[-1] and atoms:
            current_min_z, _ = generator_backend._z_bounds(atoms)
            _, lower_max_z = generator_backend._z_bounds(stacked_layers[-1])
            atoms = generator_backend._shift_atoms_z(atoms, lower_max_z + float(gap) - current_min_z)
        stacked_layers.append(atoms)

    final_vector1 = int(bottom_vector1[0]) * bottom.lattice[0] + int(bottom_vector1[1]) * bottom.lattice[1]
    final_vector2 = int(bottom_vector2[0]) * bottom.lattice[0] + int(bottom_vector2[1]) * bottom.lattice[1]
    reference_c = _longest_c_vector(c_vectors)

    all_atoms: list[tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    for atoms in reversed(stacked_layers):
        all_atoms.extend(atoms)

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
    interlayers: Sequence[float],
    output_path: str | None = None,
    output_dir: str | None = None,
    bottom_c_repeat: int | None = None,
    upper_c_repeats: Sequence[int] | None = None,
    zfix: float | None = None,
) -> MakeNRun:
    meta, candidates = findn_backend.parse_results(results_file)
    by_index = {int(candidate["index"]): candidate for candidate in candidates}
    if index not in by_index:
        raise ValueError(f"index {index} not found in {results_file}")

    candidate = by_index[index]
    upper_poscars = [str(path) for path in meta["upper_poscars"]]
    resolved_bottom_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else meta.get("bottom_c_repeat", 1))
    meta_upper_repeats = [int(value) for value in meta.get("upper_c_repeats", [1] * len(upper_poscars))]
    resolved_upper_repeats = meta_upper_repeats if upper_c_repeats is None else [int(value) for value in upper_c_repeats]

    lattice_out, positions_direct, counts, species, flags = build_nlayer_supercell(
        str(meta["bottom_poscar"]),
        upper_poscars,
        candidate,
        interlayers=interlayers,
        bottom_c_repeat=resolved_bottom_repeat,
        upper_c_repeats=resolved_upper_repeats,
        zfix=zfix,
    )

    total_atoms = int(sum(counts))
    if output_path is None:
        destination_dir = DEFAULT_OUTPUT_DIR.resolve() if output_dir is None else Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        angle_slug = "_".join(f"{float(layer['angle_deg']):.4f}" for layer in candidate["upper_layers"])
        output_name = (
            f"stackn_idx{index:03d}_angles{angle_slug}_atoms{total_atoms}_"
            f"{_slug(Path(str(meta['bottom_poscar'])).stem)}-bottom.vasp"
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
        comment="Generated by CELLSTINE maken stage | Made by Sarwin Chandran",
    )
    return MakeNRun(
        output_path=Path(output_path).resolve(),
        selected_index=int(index),
        angles_deg=tuple(float(layer["angle_deg"]) for layer in candidate["upper_layers"]),
        total_atoms=total_atoms,
    )


def generate_many_from_results(
    results_file: str,
    *,
    indexes: Sequence[int],
    interlayers: Sequence[float],
    output_dir: str | None = None,
    bottom_c_repeat: int | None = None,
    upper_c_repeats: Sequence[int] | None = None,
    zfix: float | None = None,
) -> List[MakeNRun]:
    return [
        generate_from_results(
            results_file,
            index=int(index),
            interlayers=interlayers,
            output_dir=output_dir,
            bottom_c_repeat=bottom_c_repeat,
            upper_c_repeats=upper_c_repeats,
            zfix=zfix,
        )
        for index in indexes
    ]


def maken(**kwargs):
    return generate_many_from_results(**kwargs)

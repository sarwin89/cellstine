"""Builder for the multi-layer commensurate stacks found by the N-layer search.

The construction is the same exact one the bilayer builder uses: every layer is
filled by enumerating the cosets of its integer supercell, the recorded affine is
applied to the finished layer, and the layers are then stacked with the requested
gaps inside a cell whose in-plane vectors are the shared lattice of the
candidate.  Before anything is written, each layer's transformed supercell is
checked against that shared lattice, so a document that does not describe a
commensurate stack is rejected instead of quietly producing a distorted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from ...core.provenance import stage_comment
from ...core.species import expand_species
from ...io import native as io_mod
from ..search.nlayer import read_nlayer_results
from . import generator

DEFAULT_OUTPUT_DIR = Path("output")

Atom = Tuple[str, np.ndarray, Tuple[str, str, str] | None]


@dataclass
class MakeNRun:
    """One written multi-layer structure."""

    output_path: Path
    selected_index: int
    angles_deg: tuple[float, ...]
    total_atoms: int
    layer_counts: tuple[int, ...]


def _layer_atoms(
    structure: io_mod.PoscarData,
    matrix: np.ndarray,
    affine: np.ndarray,
    name: str,
    shared_lattice: np.ndarray,
    tolerance: float,
) -> List[Atom]:
    planar_scale = max(float(np.max(np.abs(structure.lattice[:2]))), 1.0)
    if np.max(np.abs(structure.lattice[:2, 2])) > 1e-10 * planar_scale:
        raise ValueError(f"{name} POSCAR a/b lattice vectors must be planar in Cartesian xy")
    supercell_rows = np.asarray(matrix, dtype=float) @ np.asarray(structure.lattice[:2], dtype=float)
    transformed = supercell_rows[:, :2] @ np.asarray(affine, dtype=float).T
    if not np.allclose(transformed, shared_lattice, rtol=tolerance, atol=tolerance):
        raise ValueError(
            f"the recorded supercell of {name} does not reproduce the shared lattice of the candidate"
        )
    atoms = generator._replicate_layer_cartesian(
        structure.positions_direct,
        structure.lattice,
        np.asarray(matrix, dtype=np.int64),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        expand_species(structure.species, structure.counts, name),
        structure.selective_flags,
    )
    return generator._transform_layer_atoms(atoms, affine)


def build_nlayer_supercell(
    document: dict[str, Any],
    candidate: dict[str, Any],
    *,
    interlayers: Sequence[float],
    vacuum: float | None = None,
    zfix: float | None = None,
    base_c_repeat: int = 1,
    upper_c_repeats: Sequence[int] | None = None,
    tolerance_float: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None, List[int]]:
    """Return the cell, positions, counts, species, flags, and per-layer counts."""

    search = document["search"]
    shared_lattice = np.asarray(candidate["shared_lattice"], dtype=float)
    layers = list(candidate["layers"])
    if len(interlayers) != len(layers):
        raise ValueError("one interlayer distance is required per upper layer")
    repeats = list(upper_c_repeats or [1] * len(layers))
    if len(repeats) != len(layers):
        raise ValueError("one c repetition count is required per upper layer")

    agreement = max(float(tolerance_float), 1e-12)
    base_structure = io_mod.repeat_structure_along_c(
        io_mod.read_poscar(str(search["base_poscar"])), int(base_c_repeat)
    )
    stacked: List[Atom] = _layer_atoms(
        base_structure,
        np.asarray(candidate["base_matrix"], dtype=np.int64),
        np.eye(2),
        "base layer",
        shared_lattice,
        agreement,
    )
    expected_base = int(candidate["base_atom_count"]) * int(base_c_repeat)
    if len(stacked) != expected_base:
        raise ValueError(
            f"base layer atom count mismatch: the candidate records {expected_base} atoms "
            f"but its supercell matrix holds {len(stacked)}"
        )
    layer_counts = [len(stacked)]

    for layer, gap, repeat in zip(layers, interlayers, repeats):
        structure = io_mod.repeat_structure_along_c(
            io_mod.read_poscar(str(layer["poscar"])), int(repeat)
        )
        atoms = _layer_atoms(
            structure,
            np.asarray(layer["matrix"], dtype=np.int64),
            np.asarray(layer["affine"], dtype=float),
            f"layer {int(layer['layer'])}",
            shared_lattice,
            agreement,
        )
        expected = int(layer["atom_count"]) * int(repeat)
        if len(atoms) != expected:
            raise ValueError(
                f"layer {int(layer['layer'])} atom count mismatch: the candidate records "
                f"{expected} atoms but its supercell matrix holds {len(atoms)}"
            )
        _, stacked_max = generator._z_bounds(stacked)
        layer_min, _ = generator._z_bounds(atoms)
        atoms = generator._shift_atoms_z(atoms, stacked_max + float(gap) - layer_min)
        stacked.extend(atoms)
        layer_counts.append(len(atoms))

    reference_c = base_structure.lattice[2]
    first = np.array([shared_lattice[0, 0], shared_lattice[0, 1], 0.0])
    second = np.array([shared_lattice[1, 0], shared_lattice[1, 1], 0.0])
    final_lattice, min_z, lower_padding = generator._build_final_lattice(
        first, second, reference_c, stacked, tolerance_float, vacuum
    )
    stacked = generator._shift_atoms_z(stacked, lower_padding - min_z)
    positions_direct, counts, species, flags = generator._finalise_cartesian_atoms(
        stacked, final_lattice, zfix
    )
    final_lattice, positions_direct = generator._swap_if_left_handed(final_lattice, positions_direct)
    return final_lattice, positions_direct, counts, species, flags, layer_counts


def _slug(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "structure"


def generate_from_results(
    results_file: str,
    *,
    index: int,
    interlayers: Sequence[float],
    output_path: str | None = None,
    output_dir: str | None = None,
    vacuum: float | None = None,
    zfix: float | None = None,
    base_c_repeat: int = 1,
    upper_c_repeats: Sequence[int] | None = None,
) -> MakeNRun:
    """Build one candidate of a multi-layer results document."""

    document = read_nlayer_results(results_file)
    by_index = {int(item["index"]): item for item in document["candidates"]}
    if int(index) not in by_index:
        raise ValueError(f"index {index} not found in {results_file}")
    candidate = by_index[int(index)]

    lattice, positions, counts, species, flags, layer_counts = build_nlayer_supercell(
        document,
        candidate,
        interlayers=interlayers,
        vacuum=vacuum,
        zfix=zfix,
        base_c_repeat=base_c_repeat,
        upper_c_repeats=upper_c_repeats,
    )
    total_atoms = int(sum(counts))
    angles = tuple(float(layer["angle_deg"]) for layer in candidate["layers"])

    if output_path is None:
        destination = DEFAULT_OUTPUT_DIR.resolve() if output_dir is None else Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        angle_token = "-".join(f"{angle:.3f}" for angle in angles)
        name = (
            f"stackn_idx{int(index):03d}_ang{angle_token}_atoms{total_atoms}_"
            f"{_slug(Path(document['search']['base_poscar']).stem)}-base.vasp"
        )
        output_path = str(destination / name)
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    generator.write_supercell_poscar(
        output_path,
        lattice,
        positions,
        counts,
        species,
        flags,
        comment=stage_comment(
            "moire maken",
            f"{_slug(Path(document['search']['base_poscar']).stem)} base",
            f"candidate {int(index)}",
            "twists " + ", ".join(f"{angle:.4f}" for angle in angles) + " deg",
            f"{total_atoms} atoms",
        ),
    )
    return MakeNRun(
        output_path=Path(output_path).resolve(),
        selected_index=int(index),
        angles_deg=angles,
        total_atoms=total_atoms,
        layer_counts=tuple(int(value) for value in layer_counts),
    )


def generate_many_from_results(
    results_file: str,
    *,
    indexes: Sequence[int],
    interlayers: Sequence[float],
    output_dir: str | None = None,
    vacuum: float | None = None,
    zfix: float | None = None,
    base_c_repeat: int = 1,
    upper_c_repeats: Sequence[int] | None = None,
) -> List[MakeNRun]:
    """Build several candidates of a multi-layer results document."""

    return [
        generate_from_results(
            results_file,
            index=int(index),
            interlayers=interlayers,
            output_dir=output_dir,
            vacuum=vacuum,
            zfix=zfix,
            base_c_repeat=base_c_repeat,
            upper_c_repeats=upper_c_repeats,
        )
        for index in indexes
    ]

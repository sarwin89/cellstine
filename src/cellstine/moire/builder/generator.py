"""Generator backend for exact moire supercell construction.

The supercell is filled by *coset enumeration*, not by scanning translations and
deduplicating by distance: :func:`_column_hermite_normal_form` puts the
transposed supercell matrix into column Hermite normal form and
:func:`_coset_representatives` takes the box ``0 <= x < h11``, ``0 <= y < h22``.
``aristotle-lean-reference/RequestProject/CosetRepresentatives.lean`` proves what that guarantees --
``Cellstine.latticeOf_columnHnf`` (the triple computed from the ``gcd``, the
determinant and a Bezout pair spans the same column lattice),
``Cellstine.existsUnique_mem_hnfBox`` (the box meets every coset exactly once,
so no image is duplicated or lost) and ``Cellstine.hnf_card_eq_abs_det`` (there
are ``|det M|`` of them, so each atom is copied exactly as often as the index of
the supercell).  No tolerance enters anywhere, and the atom count of the output
is decided by the integers alone.  ``tests/test_moire_coset_enumeration.py``
checks all three on the implementation with exact integer arithmetic.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ...core.contacts import layer_contact_report, merge_notes, structure_contact_report
from ...io import native as io_mod
from ..search.results import read_results
from ..structure_helpers import expand_species


def parse_results(filename: str) -> Tuple[str, str, List[dict], dict]:
    """Read validated Gram JSON and expose its original POSCAR paths."""

    payload = read_results(filename)
    search = payload["search"]
    return (
        str(search["top_poscar"]),
        str(search["bottom_poscar"]),
        list(payload["candidates"]),
        payload,
    )


def _column_hermite_normal_form(matrix: np.ndarray) -> tuple[int, int, int]:
    """Return ``(h11, h12, h22)`` of the column Hermite normal form of ``matrix``.

    The columns of ``[[h11, h12], [0, h22]]`` span the same integer lattice as the
    columns of ``matrix``.  Only exact integer arithmetic is used.
    """

    integers = np.asarray(matrix, dtype=np.int64)
    determinant = int(
        integers[0, 0] * integers[1, 1] - integers[0, 1] * integers[1, 0]
    )
    if determinant == 0:
        raise ValueError("supercell matrix must be nonsingular")
    lower_left, lower_right = int(integers[1, 0]), int(integers[1, 1])
    h22 = math.gcd(lower_left, lower_right)
    h11 = abs(determinant) // h22
    # Bezout coefficients of the bottom row give the column combination whose
    # lower entry is exactly ``h22``; its upper entry is reduced modulo ``h11``.
    left, right = _bezout(lower_left, lower_right)
    h12 = (left * int(integers[0, 0]) + right * int(integers[0, 1])) % h11
    return h11, h12, h22


def _bezout(left: int, right: int) -> tuple[int, int]:
    """Return ``(x, y)`` with ``x * left + y * right == gcd(left, right) >= 0``."""

    old_r, r = int(left), int(right)
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y
    if old_r < 0:
        old_x, old_y = -old_x, -old_y
    return old_x, old_y


def _coset_representatives(matrix: np.ndarray) -> np.ndarray:
    """Return the ``|det M|`` lattice translations that fill one supercell.

    A supercell whose rows are ``M`` in units of the primitive cell contains
    exactly ``|det M|`` copies of every atom, one per coset of ``Z^2 / Z^2 M``.
    Writing that lattice with column vectors, ``(n M)^T = M^T n^T``, so the cosets
    are those of ``Z^2 / M^T Z^2``; in the Hermite normal form of ``M^T`` they are
    represented by ``0 <= x < h11`` and ``0 <= y < h22``.

    Enumerating the cosets is exact, needs no distance tolerance, and produces no
    duplicates, unlike scanning a heuristic box of translations and discarding
    coincident images afterwards.
    """

    h11, _, h22 = _column_hermite_normal_form(np.asarray(matrix, dtype=np.int64).T)
    first = np.arange(h11, dtype=np.int64)
    second = np.arange(h22, dtype=np.int64)
    grid = np.stack(np.meshgrid(first, second, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


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
    source_matrix: np.ndarray,
    shift_direct: Sequence[float],
    shift_cart: Sequence[float],
    species: Sequence[str],
    selective_flags: Sequence[Tuple[str, str, str]] | None,
) -> List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]:
    """Fill one supercell with the images of every atom of the primitive layer.

    ``source_matrix`` holds the integer supercell rows.  Every atom appears once
    per coset of ``Z^2 / Z^2 M``, so the result always contains exactly
    ``|det M|`` images per atom.
    """

    lattice = np.asarray(source_lattice, dtype=float)
    matrix = np.asarray(source_matrix, dtype=np.int64)
    translations = _coset_representatives(matrix)
    supercell_rows = matrix.astype(float) @ lattice[:2]
    inverse = np.linalg.inv(matrix.astype(float))
    shift_vector = _shift_vector(lattice, shift_direct, shift_cart)

    results: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    for atom_index, base_direct in enumerate(np.asarray(positions_direct, dtype=float)):
        planar = (translations + base_direct[:2]) @ inverse
        planar -= np.floor(planar)
        cartesian = (
            planar @ supercell_rows
            + base_direct[2] * lattice[2].reshape(1, 3)
            + shift_vector.reshape(1, 3)
        )
        flags = tuple(selective_flags[atom_index]) if selective_flags is not None else None
        for position in cartesian:
            results.append((species[atom_index], np.array(position, dtype=float), flags))
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
    vacuum: float | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Return the output cell, the current lowest atom, and its target height.

    The stack is centred along ``c``: the empty space is split equally above and
    below the atoms, so both free surfaces of the slab see the same vacuum and no
    atom sits on the cell boundary, where rounding could wrap it to the far side.
    With ``vacuum`` given, the cell height is exactly the occupied span plus that
    vacuum; otherwise the longer input ``c`` vector is kept whenever it already
    leaves room for the stack.
    """

    min_z, max_z = _z_bounds(atoms)
    z_span = max_z - min_z
    padding = 2.0 * max(float(tolerance_float), 1e-3)
    if vacuum is None:
        reference_length = float(np.linalg.norm(reference_c))
        c_length = max(reference_length, z_span + padding)
    else:
        if not np.isfinite(float(vacuum)) or float(vacuum) < 0.0:
            raise ValueError("vacuum must be a finite nonnegative length in angstrom")
        c_length = z_span + max(float(vacuum), padding)
    final_c = _scale_vector(reference_c, c_length)
    final_lattice = np.vstack((in_plane_vector1, in_plane_vector2, final_c))
    lower_padding = 0.5 * (c_length - z_span)
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


def _transform_layer_atoms(
    atoms: Sequence[Tuple[str, np.ndarray, Tuple[str, str, str] | None]],
    affine: np.ndarray,
) -> List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]:
    """Apply a recorded Cartesian column affine to row-vector atom coordinates."""

    transformed: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]] = []
    for species, position, flags in atoms:
        updated = np.array(position, dtype=float, copy=True)
        updated[:2] = np.asarray(affine, dtype=float) @ updated[:2]
        transformed.append((species, updated, flags))
    return transformed


def _recorded_layer_geometry(
    structure: io_mod.PoscarData,
    matrix: np.ndarray,
    affine: np.ndarray,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unstrained 3D supercell rows and recorded transformed 2D rows."""

    planar_scale = max(float(np.max(np.abs(structure.lattice[:2]))), 1.0)
    if np.max(np.abs(structure.lattice[:2, 2])) > 1e-10 * planar_scale:
        raise ValueError(f"{name} POSCAR a/b lattice vectors must be planar in Cartesian xy")
    source_rows = np.asarray(matrix, dtype=int) @ np.asarray(structure.lattice[:2], dtype=float)
    source_supercell = np.vstack((source_rows, structure.lattice[2]))
    transformed_rows = source_rows[:, :2] @ np.asarray(affine, dtype=float).T
    return source_supercell, transformed_rows


@dataclass(frozen=True)
class LayerStack:
    """A built bilayer candidate: the output cell and its two Cartesian layers.

    ``top_atoms`` and ``bottom_atoms`` are ``(species, cartesian, flags)`` triples
    already shifted so that the stack sits centred inside ``lattice``.
    """

    lattice: np.ndarray
    top_atoms: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]
    bottom_atoms: List[Tuple[str, np.ndarray, Tuple[str, str, str] | None]]


def build_candidate_layers(
    top_poscar: str,
    bottom_poscar: str,
    candidate: dict,
    *,
    shift_top_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift_top_cart: Sequence[float] = (0.0, 0.0, 0.0),
    shift_bottom_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift_bottom_cart: Sequence[float] = (0.0, 0.0, 0.0),
    tolerance_float: float = 1e-4,
    interlayer_distance: float | None = None,
    repeat_top_c: int = 1,
    repeat_bottom_c: int = 1,
    vacuum: float | None = None,
) -> LayerStack:
    """Replicate, strain and stack the two layers of one validated candidate.

    This is the shared geometry stage behind POSCAR generation and the result
    visualizers, so every consumer sees exactly the same atoms.
    """

    top_structure = io_mod.repeat_structure_along_c(
        io_mod.read_poscar(top_poscar), repeat_top_c
    )
    bottom_structure = io_mod.repeat_structure_along_c(
        io_mod.read_poscar(bottom_poscar), repeat_bottom_c
    )
    top_matrix = np.asarray(candidate["top_matrix"], dtype=int)
    bottom_matrix = np.asarray(candidate["bottom_matrix"], dtype=int)
    top_affine = np.asarray(candidate["top_affine"], dtype=float)
    bottom_affine = np.asarray(candidate["bottom_affine"], dtype=float)
    shared_rows = np.asarray(candidate["shared_lattice"], dtype=float).T

    top_supercell, transformed_top = _recorded_layer_geometry(
        top_structure, top_matrix, top_affine, "top"
    )
    bottom_supercell, transformed_bottom = _recorded_layer_geometry(
        bottom_structure, bottom_matrix, bottom_affine, "bottom"
    )
    agreement_tolerance = max(float(tolerance_float), 1e-12)
    if not (
        np.allclose(transformed_top, transformed_bottom, rtol=agreement_tolerance, atol=agreement_tolerance)
        and np.allclose(transformed_top, shared_rows, rtol=agreement_tolerance, atol=agreement_tolerance)
    ):
        raise ValueError(
            "recorded transformed top and bottom in-plane lattices do not agree "
            "with the shared lattice"
        )

    top_species = expand_species(top_structure.species, top_structure.counts, "top")
    bottom_species = expand_species(
        bottom_structure.species, bottom_structure.counts, "bottom"
    )
    atoms_top = _replicate_layer_cartesian(
        top_structure.positions_direct,
        top_structure.lattice,
        top_matrix,
        shift_top_direct,
        shift_top_cart,
        top_species,
        top_structure.selective_flags,
    )
    atoms_bottom = _replicate_layer_cartesian(
        bottom_structure.positions_direct,
        bottom_structure.lattice,
        bottom_matrix,
        shift_bottom_direct,
        shift_bottom_cart,
        bottom_species,
        bottom_structure.selective_flags,
    )
    atoms_top = _transform_layer_atoms(atoms_top, top_affine)
    atoms_bottom = _transform_layer_atoms(atoms_bottom, bottom_affine)

    expected_top = int(candidate["top_atom_count"]) * int(repeat_top_c)
    expected_bottom = int(candidate["bottom_atom_count"]) * int(repeat_bottom_c)
    if len(atoms_top) != expected_top:
        raise ValueError(
            f"top layer atom count mismatch: the candidate records {expected_top} atoms "
            f"but its supercell matrix holds {len(atoms_top)}"
        )
    if len(atoms_bottom) != expected_bottom:
        raise ValueError(
            f"bottom layer atom count mismatch: the candidate records {expected_bottom} "
            f"atoms but its supercell matrix holds {len(atoms_bottom)}"
        )

    if interlayer_distance is not None and atoms_top and atoms_bottom:
        top_min_z, _ = _z_bounds(atoms_top)
        _, bottom_max_z = _z_bounds(atoms_bottom)
        atoms_top = _shift_atoms_z(
            atoms_top, bottom_max_z + float(interlayer_distance) - top_min_z
        )

    all_atoms = atoms_top + atoms_bottom
    reference_c = _reference_c_vector(top_structure.lattice[2], bottom_structure.lattice[2])
    first_shared = np.array([shared_rows[0, 0], shared_rows[0, 1], 0.0])
    second_shared = np.array([shared_rows[1, 0], shared_rows[1, 1], 0.0])
    final_lattice, min_z, lower_padding = _build_final_lattice(
        first_shared,
        second_shared,
        reference_c,
        all_atoms,
        tolerance_float,
        vacuum,
    )
    shift_z = lower_padding - min_z
    return LayerStack(
        lattice=final_lattice,
        top_atoms=_shift_atoms_z(atoms_top, shift_z),
        bottom_atoms=_shift_atoms_z(atoms_bottom, shift_z),
    )


def build_supercell(
    top_poscar: str,
    bottom_poscar: str,
    candidate: dict,
    *,
    shift_top_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift_top_cart: Sequence[float] = (0.0, 0.0, 0.0),
    shift_bottom_direct: Sequence[float] = (0.0, 0.0, 0.0),
    shift_bottom_cart: Sequence[float] = (0.0, 0.0, 0.0),
    tolerance: int = 1,
    tolerance_float: float = 1e-4,
    interlayer_distance: float | None = None,
    zfix: float | None = None,
    repeat_top_c: int = 1,
    repeat_bottom_c: int = 1,
    vacuum: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None]:
    """Build from original-gauge matrices and recorded in-plane affine transforms.

    Layer replication is exact coset enumeration, so ``tolerance`` no longer has
    any effect; it is accepted so that existing callers keep working.
    ``tolerance_float`` is still used when the recorded transforms are checked
    against the shared lattice and when the vacuum padding is chosen.
    """

    del tolerance

    stack = build_candidate_layers(
        top_poscar,
        bottom_poscar,
        candidate,
        shift_top_direct=shift_top_direct,
        shift_top_cart=shift_top_cart,
        shift_bottom_direct=shift_bottom_direct,
        shift_bottom_cart=shift_bottom_cart,
        tolerance_float=tolerance_float,
        interlayer_distance=interlayer_distance,
        repeat_top_c=repeat_top_c,
        repeat_bottom_c=repeat_bottom_c,
        vacuum=vacuum,
    )
    final_lattice = stack.lattice
    positions_direct, counts, species, flags = _finalise_cartesian_atoms(
        stack.top_atoms + stack.bottom_atoms, final_lattice, zfix
    )
    final_lattice, positions_direct = _swap_if_left_handed(final_lattice, positions_direct)
    return final_lattice, positions_direct, counts, species, flags


def interlayer_contact_report(
    stack: LayerStack, *, requested_gap: float | None = None
) -> dict:
    """Measure how close the two layers of a built stack really come.

    The interlayer distance a stack is built with separates the two layers along
    the surface normal.  What decides whether the two surfaces are touching is
    the closest approach between an atom of one layer and an atom of the other,
    over the periodic images as well, and for a twisted stack that distance
    varies across the cell: the registry is different at every moire site.
    """

    return layer_contact_report(
        lattice=stack.lattice,
        first_cartesian=np.array([position for _, position, _ in stack.bottom_atoms], dtype=float).reshape(-1, 3),
        second_cartesian=np.array([position for _, position, _ in stack.top_atoms], dtype=float).reshape(-1, 3),
        first_species=[str(symbol) for symbol, _, _ in stack.bottom_atoms],
        second_species=[str(symbol) for symbol, _, _ in stack.top_atoms],
        subject="interlayer",
        requested=None if requested_gap is None else float(requested_gap),
        requested_name="interlayer distance",
    )


def build_supercell_with_report(
    top_poscar: str,
    bottom_poscar: str,
    candidate: dict,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[Tuple[str, str, str]] | None, dict]:
    """Build a supercell and measure the contact its two layers make.

    Same as :func:`build_supercell`, with the interlayer contact report of
    :func:`interlayer_contact_report` appended, so a caller that writes a
    structure can also say how close its layers come without rebuilding it.
    """

    stack_kwargs = {key: value for key, value in kwargs.items() if key not in {"zfix", "tolerance"}}
    stack = build_candidate_layers(top_poscar, bottom_poscar, candidate, **stack_kwargs)
    final_lattice = stack.lattice
    positions_direct, counts, species, flags = _finalise_cartesian_atoms(
        stack.top_atoms + stack.bottom_atoms, final_lattice, kwargs.get("zfix")
    )
    final_lattice, positions_direct = _swap_if_left_handed(final_lattice, positions_direct)
    report = interlayer_contact_report(stack, requested_gap=kwargs.get("interlayer_distance"))
    # The interlayer contact is the one the stage controls; it says nothing about
    # the layers themselves, so the written cell is measured as a whole too.
    whole = structure_contact_report(
        lattice=final_lattice,
        positions_direct=positions_direct,
        species=expand_species(species, counts),
    )
    for key in ("structure_contact_distance", "structure_contact"):
        if key in whole:
            report[key] = whole[key]
    report["notes"] = merge_notes(whole, report.get("notes", ()))
    return final_lattice, positions_direct, counts, species, flags, report


def write_supercell_poscar(
    output_path: str,
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    counts: Sequence[int],
    species: Sequence[str],
    flags: Sequence[Sequence[str]] | None,
    comment: str,
) -> None:
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

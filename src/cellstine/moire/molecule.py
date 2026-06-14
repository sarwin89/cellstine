"""Utilities for manipulating a top-side adsorbate molecule on a substrate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..interface import surface_backend as surface_mod
from ..io import native as io_mod
from . import lattice as lattice_mod
from .structure_helpers import expand_species as _expand_species_shared


_ATOMIC_MASS_ROWS = """
H 1.008
He 4.002602
Li 6.94
Be 9.0121831
B 10.81
C 12.011
N 14.007
O 15.999
F 18.998403163
Ne 20.1797
Na 22.98976928
Mg 24.305
Al 26.9815385
Si 28.085
P 30.973761998
S 32.06
Cl 35.45
Ar 39.948
K 39.0983
Ca 40.078
Sc 44.955908
Ti 47.867
V 50.9415
Cr 51.9961
Mn 54.938044
Fe 55.845
Co 58.933194
Ni 58.6934
Cu 63.546
Zn 65.38
Ga 69.723
Ge 72.63
As 74.921595
Se 78.971
Br 79.904
Kr 83.798
Rb 85.4678
Sr 87.62
Y 88.90584
Zr 91.224
Nb 92.90637
Mo 95.95
Tc 98.0
Ru 101.07
Rh 102.9055
Pd 106.42
Ag 107.8682
Cd 112.414
In 114.818
Sn 118.71
Sb 121.76
Te 127.6
I 126.90447
Xe 131.293
Cs 132.90545196
Ba 137.327
La 138.90547
Ce 140.116
Pr 140.90766
Nd 144.242
Pm 145.0
Sm 150.36
Eu 151.964
Gd 157.25
Tb 158.92535
Dy 162.5
Ho 164.93033
Er 167.259
Tm 168.93422
Yb 173.045
Lu 174.9668
Hf 178.49
Ta 180.94788
W 183.84
Re 186.207
Os 190.23
Ir 192.217
Pt 195.084
Au 196.966569
Hg 200.592
Tl 204.38
Pb 207.2
Bi 208.9804
Po 209.0
At 210.0
Rn 222.0
Fr 223.0
Ra 226.0
Ac 227.0
Th 232.0377
Pa 231.03588
U 238.02891
"""

ATOMIC_MASSES = {
    symbol: float(mass)
    for symbol, mass in (line.split() for line in _ATOMIC_MASS_ROWS.splitlines() if line.strip())
}
DEFAULT_OUTPUT_DIR = Path("output")


@dataclass(frozen=True)
class MoleculeSelection:
    molecule_indices: tuple[int, ...]
    substrate_indices: tuple[int, ...]
    z_cutoff: float
    gap_size: float
    center_of_mass_cartesian: np.ndarray
    center_of_mass_direct: np.ndarray

    @property
    def molecule_atom_count(self) -> int:
        return len(self.molecule_indices)

    @property
    def substrate_atom_count(self) -> int:
        return len(self.substrate_indices)


@dataclass(frozen=True)
class MoleculeTransformRun:
    output_path: Path
    molecule_atom_count: int
    substrate_atom_count: int
    z_cutoff: float
    gap_size: float
    center_of_mass_before: np.ndarray
    center_of_mass_after: np.ndarray
    target_cartesian: np.ndarray
    reframe_shift_direct: np.ndarray


@dataclass(frozen=True)
class LayerShiftRun:
    output_path: Path
    top_atom_count: int
    bottom_atom_count: int
    z_cutoff: float
    gap_size: float
    shift_cartesian: np.ndarray
    shift_direct: np.ndarray


@dataclass(frozen=True)
class AdsorbRun:
    output_path: Path
    substrate_atom_count: int
    molecule_atom_count: int
    site_type: str
    site_index: int
    surface_side: str
    height: float
    center_of_mass_before: np.ndarray
    center_of_mass_after: np.ndarray
    site_cartesian: np.ndarray
    site_direct: np.ndarray
    reframe_shift_direct: np.ndarray


@dataclass(frozen=True)
class _TopGroupSplit:
    molecule_mask: np.ndarray
    substrate_mask: np.ndarray
    z_cutoff: float
    gap_size: float


def _expand_species(species: Sequence[str], counts: Sequence[int]) -> List[str]:
    return _expand_species_shared(species, counts)


def _species_disjoint_split(
    structure: io_mod.PoscarData,
    order: np.ndarray,
    sorted_z: np.ndarray,
    gaps: np.ndarray,
    min_gap: float,
) -> _TopGroupSplit | None:
    """Use the lowest z-separated cluster as the substrate when species are disjoint."""

    if not structure.species:
        return None
    if gaps.size == 0:
        return None

    gap_candidates = np.flatnonzero(gaps >= float(min_gap))
    if gap_candidates.size == 0:
        return None

    expanded_species = np.array(_expand_species(structure.species, structure.counts), dtype=object)
    for gap_index in gap_candidates.tolist():
        bottom_cluster = order[: gap_index + 1]
        upper_cluster = order[gap_index + 1 :]
        if bottom_cluster.size == 0 or upper_cluster.size == 0:
            continue

        bottom_species = {str(expanded_species[index]) for index in bottom_cluster.tolist()}
        upper_species = {str(expanded_species[index]) for index in upper_cluster.tolist()}
        if not bottom_species or not upper_species or not bottom_species.isdisjoint(upper_species):
            continue

        substrate_mask = np.array([str(symbol) in bottom_species for symbol in expanded_species], dtype=bool)
        molecule_mask = ~substrate_mask
        if np.any(substrate_mask) and np.any(molecule_mask):
            return _TopGroupSplit(
                molecule_mask=molecule_mask,
                substrate_mask=substrate_mask,
                z_cutoff=float(0.5 * (sorted_z[gap_index] + sorted_z[gap_index + 1])),
                gap_size=float(gaps[gap_index]),
            )
    return None


def _normalise_element_symbol(label: str) -> str:
    cleaned = "".join(re.findall(r"[A-Za-z]+", str(label)))
    if not cleaned:
        raise ValueError(f"could not infer an element symbol from {label!r}")

    candidate_two = cleaned[:2].capitalize()
    if candidate_two in ATOMIC_MASSES:
        return candidate_two

    candidate_one = cleaned[:1].upper()
    if candidate_one in ATOMIC_MASSES:
        return candidate_one

    raise ValueError(f"no atomic mass is available for species label {label!r}")


def _species_masses(species: Sequence[str]) -> np.ndarray:
    return np.array([ATOMIC_MASSES[_normalise_element_symbol(symbol)] for symbol in species], dtype=float)


def center_of_mass_cartesian(positions: np.ndarray, species: Sequence[str]) -> np.ndarray:
    masses = _species_masses(species)
    weight = float(np.sum(masses))
    if weight <= 0.0:
        raise ValueError("total atomic mass must be positive")
    return np.sum(np.asarray(positions, dtype=float) * masses[:, None], axis=0) / weight


def identify_top_group(
    structure: io_mod.PoscarData,
    *,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
) -> MoleculeSelection:
    positions = np.asarray(structure.positions_cartesian, dtype=float)
    if positions.shape[0] == 0:
        raise ValueError("structure does not contain any atoms")

    z_values = positions[:, 2]
    order = np.argsort(z_values)
    sorted_z = z_values[order]
    gaps = np.diff(sorted_z)

    auto_detect = z_cutoff is None
    if auto_detect:
        if sorted_z.size < 2:
            raise ValueError("at least two atoms are required to isolate a top-side molecule")

        gap_index = int(np.argmax(gaps))
        gap_size = float(gaps[gap_index])
        if gap_size < float(min_gap):
            raise ValueError(
                f"largest internal z gap is only {gap_size:.4f} A; provide --z-cutoff if the molecule is not cleanly separated"
            )
        split = _species_disjoint_split(structure, order, sorted_z, gaps, float(min_gap))
        if split is None:
            z_cutoff = float(0.5 * (sorted_z[gap_index] + sorted_z[gap_index + 1]))
            molecule_mask = z_values > z_cutoff
            substrate_mask = ~molecule_mask
        else:
            molecule_mask = split.molecule_mask
            substrate_mask = split.substrate_mask
            z_cutoff = split.z_cutoff
            gap_size = split.gap_size
    else:
        gap_size = float("nan")
        z_cutoff = float(z_cutoff)
        molecule_mask = z_values > z_cutoff
        substrate_mask = ~molecule_mask

    if not np.any(molecule_mask):
        raise ValueError(f"no atoms were found above z_cutoff={z_cutoff:.6f} A")
    if not np.any(substrate_mask):
        raise ValueError(f"all atoms are above z_cutoff={z_cutoff:.6f} A; no substrate atoms remain")

    expanded_species = _expand_species(structure.species, structure.counts)
    molecule_indices = tuple(int(index) for index in np.flatnonzero(molecule_mask))
    substrate_indices = tuple(int(index) for index in np.flatnonzero(substrate_mask))
    molecule_species = [expanded_species[index] for index in molecule_indices]
    molecule_positions = positions[np.array(molecule_indices, dtype=int)]
    com_cart = center_of_mass_cartesian(molecule_positions, molecule_species)

    return MoleculeSelection(
        molecule_indices=molecule_indices,
        substrate_indices=substrate_indices,
        z_cutoff=float(z_cutoff),
        gap_size=gap_size,
        center_of_mass_cartesian=com_cart,
        center_of_mass_direct=io_mod.cartesian_to_direct(com_cart.reshape(1, 3), structure.lattice)[0],
    )


def identify_top_molecule(
    structure: io_mod.PoscarData,
    *,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
) -> MoleculeSelection:
    """Backward-compatible alias for top-side molecule selection."""

    return identify_top_group(structure, z_cutoff=z_cutoff, min_gap=min_gap)


def _resolve_target_cartesian(
    lattice: np.ndarray,
    current_com: np.ndarray,
    target_cartesian: Sequence[float] | None,
    target_direct: Sequence[float] | None,
) -> np.ndarray:
    if target_cartesian is not None and target_direct is not None:
        raise ValueError("use either target_cartesian or target_direct, not both")

    current = np.asarray(current_com, dtype=float)
    if target_cartesian is not None:
        values = list(target_cartesian)
        if len(values) not in {2, 3}:
            raise ValueError("target_cartesian must contain either 2 values (x,y) or 3 values (x,y,z)")
        resolved = current.copy()
        resolved[0] = float(values[0])
        resolved[1] = float(values[1])
        if len(values) == 3:
            resolved[2] = float(values[2])
        return resolved

    if target_direct is not None:
        values = list(target_direct)
        if len(values) not in {2, 3}:
            raise ValueError("target_direct must contain either 2 values (u,v) or 3 values (u,v,w)")
        direct_target = io_mod.cartesian_to_direct(current.reshape(1, 3), lattice)[0]
        direct_target[0] = float(values[0])
        direct_target[1] = float(values[1])
        if len(values) == 3:
            direct_target[2] = float(values[2])
        return io_mod.direct_to_cartesian(direct_target.reshape(1, 3), lattice)[0]

    return current.copy()


def _resolve_shift_vectors(
    lattice: np.ndarray,
    shift_cartesian: Sequence[float] | None,
    shift_direct: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if shift_cartesian is not None and shift_direct is not None:
        raise ValueError("use either shift_cartesian or shift_direct, not both")

    if shift_direct is not None:
        values = list(shift_direct)
        if len(values) not in {2, 3}:
            raise ValueError("shift_direct must contain either 2 values (du,dv) or 3 values (du,dv,dw)")
        direct_shift = np.zeros(3, dtype=float)
        direct_shift[: len(values)] = np.array(values, dtype=float)
        cartesian_shift = io_mod.direct_to_cartesian(direct_shift.reshape(1, 3), lattice)[0]
        return cartesian_shift, direct_shift

    if shift_cartesian is not None:
        values = list(shift_cartesian)
        if len(values) not in {2, 3}:
            raise ValueError("shift_cartesian must contain either 2 values (dx,dy) or 3 values (dx,dy,dz)")
        cartesian_shift = np.zeros(3, dtype=float)
        cartesian_shift[: len(values)] = np.array(values, dtype=float)
        direct_shift = io_mod.cartesian_to_direct(cartesian_shift.reshape(1, 3), lattice)[0]
        return cartesian_shift, direct_shift

    return np.zeros(3, dtype=float), np.zeros(3, dtype=float)


def _transform_molecule_cartesian(
    positions: np.ndarray,
    species: Sequence[str],
    lattice: np.ndarray,
    *,
    target_cartesian: Sequence[float] | None,
    target_direct: Sequence[float] | None,
    rotation_deg: float,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    positions_array = np.asarray(positions, dtype=float)
    com_before = center_of_mass_cartesian(positions_array, species)
    pivot = _resolve_target_cartesian(lattice, com_before, target_cartesian, target_direct)

    translated = positions_array + (pivot - com_before)
    if abs(float(rotation_deg)) > 0.0 or abs(float(tilt_deg)) > 0.0 or abs(float(roll_deg)) > 0.0:
        rotation = lattice_mod.yaw_pitch_roll_matrix(float(rotation_deg), float(tilt_deg), float(roll_deg))
        translated = (translated - pivot) @ rotation.T + pivot

    return translated, com_before


def _normalise_reframe_axes(reframe_axes: str | Sequence[str] | None) -> tuple[int, ...]:
    if reframe_axes in {None, "", "none"}:
        return tuple()

    axis_map = {"x": 0, "y": 1, "z": 2}
    if isinstance(reframe_axes, str):
        tokens = [char for char in reframe_axes.lower() if char in axis_map]
    else:
        tokens = [str(item).strip().lower() for item in reframe_axes]

    seen = []
    for token in tokens:
        if token not in axis_map:
            raise ValueError(f"unsupported reframe axis {token!r}; use x, y, z, xy, xyz, or none")
        axis = axis_map[token]
        if axis not in seen:
            seen.append(axis)
    return tuple(seen)


def _surface_outward_normal(lattice: np.ndarray, surface_side: str) -> np.ndarray:
    if surface_side not in {"top", "bottom"}:
        raise ValueError("surface_side must be 'top' or 'bottom'")
    normal = surface_mod._surface_normal(lattice)
    return normal if surface_side == "top" else -normal


def _estimate_inplane_repeats_for_molecule(
    substrate: io_mod.PoscarData,
    molecule: io_mod.PoscarData,
    *,
    rotation_deg: float,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
    fit_padding: float,
) -> tuple[int, int]:
    molecule_species = _expand_species(molecule.species, molecule.counts)
    positions = np.asarray(molecule.positions_cartesian, dtype=float)
    com = center_of_mass_cartesian(positions, molecule_species)
    centered = positions - com
    if abs(float(rotation_deg)) > 0.0 or abs(float(tilt_deg)) > 0.0 or abs(float(roll_deg)) > 0.0:
        rotation = lattice_mod.yaw_pitch_roll_matrix(float(rotation_deg), float(tilt_deg), float(roll_deg))
        centered = centered @ rotation.T
    direct_delta = io_mod.cartesian_to_direct(centered, substrate.lattice)
    spans = np.ptp(direct_delta[:, :2], axis=0) if direct_delta.size else np.zeros(2, dtype=float)
    padding = max(0.0, float(fit_padding))
    repeat_a = max(1, int(np.ceil(float(spans[0]) + 2.0 * padding)))
    repeat_b = max(1, int(np.ceil(float(spans[1]) + 2.0 * padding)))
    return repeat_a, repeat_b


def _translate_molecule_to_site(
    molecule_positions: np.ndarray,
    molecule_species: Sequence[str],
    substrate_lattice: np.ndarray,
    site_cartesian: Sequence[float],
    *,
    surface_side: str,
    height: float,
    rotation_deg: float,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if float(height) < 0.0:
        raise ValueError("height must be non-negative")

    slab_normal = surface_mod._surface_normal(substrate_lattice)
    outward_normal = _surface_outward_normal(substrate_lattice, surface_side)
    site_cart = np.asarray(site_cartesian, dtype=float)

    rotated_positions, com_before = _transform_molecule_cartesian(
        np.asarray(molecule_positions, dtype=float),
        molecule_species,
        substrate_lattice,
        target_cartesian=None,
        target_direct=None,
        rotation_deg=rotation_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    rotated_com = center_of_mass_cartesian(rotated_positions, molecule_species)
    site_inplane = site_cart - float(np.dot(site_cart, slab_normal)) * slab_normal
    com_inplane = rotated_com - float(np.dot(rotated_com, slab_normal)) * slab_normal
    translated = rotated_positions + (site_inplane - com_inplane)

    lowest_projection = float(np.min(translated @ outward_normal))
    target_projection = float(np.dot(site_cart, outward_normal)) + float(height)
    translated = translated + (target_projection - lowest_projection) * outward_normal

    return translated, com_before


def _default_flag_list(natoms: int) -> list[tuple[str, str, str]]:
    return [("T", "T", "T")] * int(natoms)


def _combine_substrate_and_molecule(
    substrate: io_mod.PoscarData,
    molecule: io_mod.PoscarData,
    molecule_positions_cartesian: np.ndarray,
    *,
    reframe_axes: str | Sequence[str] | None,
) -> tuple[np.ndarray, list[int], list[str], list[tuple[str, str, str]] | None, np.ndarray, np.ndarray]:
    substrate_species_expanded = _expand_species(substrate.species, substrate.counts)
    molecule_species_expanded = _expand_species(molecule.species, molecule.counts)

    combined_cartesian = np.vstack(
        (
            np.asarray(substrate.positions_cartesian, dtype=float),
            np.asarray(molecule_positions_cartesian, dtype=float),
        )
    )
    molecule_indices = tuple(range(substrate.natoms, substrate.natoms + molecule.natoms))
    combined_direct = io_mod.cartesian_to_direct(combined_cartesian, substrate.lattice)
    reframed_direct, shift_direct = _reframe_direct_positions(
        combined_direct,
        molecule_indices,
        _normalise_reframe_axes(reframe_axes),
    )
    reframed_cartesian = io_mod.direct_to_cartesian(reframed_direct, substrate.lattice)

    substrate_flags = (
        _default_flag_list(substrate.natoms)
        if substrate.selective_flags is None
        else [tuple(flags) for flags in substrate.selective_flags]
    )
    molecule_flags = (
        _default_flag_list(molecule.natoms)
        if molecule.selective_flags is None
        else [tuple(flags) for flags in molecule.selective_flags]
    )
    use_flags = substrate.selective_flags is not None or molecule.selective_flags is not None
    combined_flags = substrate_flags + molecule_flags

    combined_species_expanded = substrate_species_expanded + molecule_species_expanded
    species_order: list[str] = []
    for symbol in list(substrate.species) + list(molecule.species):
        if symbol not in species_order:
            species_order.append(str(symbol))

    ordered_indices: list[int] = []
    counts: list[int] = []
    for symbol in species_order:
        indices_for_species = [index for index, entry in enumerate(combined_species_expanded) if entry == symbol]
        ordered_indices.extend(indices_for_species)
        counts.append(len(indices_for_species))

    positions_direct_out = reframed_direct[np.array(ordered_indices, dtype=int)]
    flags_out = [combined_flags[index] for index in ordered_indices] if use_flags else None
    final_molecule_cartesian = reframed_cartesian[np.array(molecule_indices, dtype=int)]
    return positions_direct_out, counts, species_order, flags_out, shift_direct, final_molecule_cartesian


def _unwrap_periodic_axis_with_start(values: np.ndarray) -> tuple[np.ndarray, float]:
    values_array = np.asarray(values, dtype=float)
    if values_array.size <= 1:
        return values_array.copy(), 0.0

    wrapped = np.mod(values_array, 1.0)
    ordered = np.sort(wrapped)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 1.0)))
    gap_index = int(np.argmax(gaps))
    interval_start = float(ordered[(gap_index + 1) % ordered.size])

    unwrapped = wrapped.copy()
    unwrapped[unwrapped < interval_start] += 1.0
    raw_span = float(np.max(values_array) - np.min(values_array))
    unwrapped_span = float(np.max(unwrapped) - np.min(unwrapped))
    if (
        raw_span > 1.0 + 1e-8
        and raw_span - unwrapped_span > 1.0 + 1e-8
        and not np.all((-1e-8 <= values_array) & (values_array <= 1.0 + 1e-8))
    ):
        return values_array.copy(), 0.0
    return unwrapped, interval_start


def _reframe_direct_positions(
    direct_positions: np.ndarray,
    molecule_indices: Sequence[int],
    axes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    direct_array = np.array(direct_positions, dtype=float, copy=True)
    if not axes:
        return direct_array, np.zeros(3, dtype=float)

    molecule_index_array = np.array(molecule_indices, dtype=int)
    shift = np.zeros(3, dtype=float)
    for axis in axes:
        molecule_direct, _ = _unwrap_periodic_axis_with_start(direct_array[molecule_index_array, axis])
        span = float(np.max(molecule_direct) - np.min(molecule_direct))
        if span > 1.0 + 1e-8:
            axis_name = "xyz"[axis]
            raise ValueError(
                f"molecule spans {span:.4f} of the cell along {axis_name}; it cannot be contained in one periodic image without enlarging the lattice"
            )
        direct_array[molecule_index_array, axis] = molecule_direct
        shift[axis] = 0.5 * (float(np.max(molecule_direct)) + float(np.min(molecule_direct))) - 0.5
        direct_array[:, axis] -= shift[axis]

    return direct_array, shift


def transform_top_molecule(
    poscar_path: str,
    *,
    output_path: str | None = None,
    target_cartesian: Sequence[float] | None = None,
    target_direct: Sequence[float] | None = None,
    rotation_deg: float = 0.0,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
    reframe_axes: str | Sequence[str] | None = None,
) -> MoleculeTransformRun:
    structure = io_mod.read_poscar(poscar_path)
    selection = identify_top_group(structure, z_cutoff=z_cutoff, min_gap=min_gap)

    expanded_species = _expand_species(structure.species, structure.counts)
    molecule_indices = np.array(selection.molecule_indices, dtype=int)
    molecule_species = [expanded_species[index] for index in selection.molecule_indices]
    resolved_target_cartesian = _resolve_target_cartesian(
        structure.lattice,
        selection.center_of_mass_cartesian,
        target_cartesian,
        target_direct,
    )

    all_cartesian = np.array(structure.positions_cartesian, dtype=float, copy=True)
    transformed_molecule, com_before = _transform_molecule_cartesian(
        all_cartesian[molecule_indices],
        molecule_species,
        structure.lattice,
        target_cartesian=resolved_target_cartesian,
        target_direct=None,
        rotation_deg=rotation_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    all_cartesian[molecule_indices] = transformed_molecule

    direct_positions = io_mod.cartesian_to_direct(all_cartesian, structure.lattice)
    reframe_axes_normalised = _normalise_reframe_axes(reframe_axes)
    reframed_direct, shift_direct = _reframe_direct_positions(
        direct_positions,
        selection.molecule_indices,
        reframe_axes_normalised,
    )
    final_cartesian = io_mod.direct_to_cartesian(reframed_direct, structure.lattice)
    final_molecule_positions = final_cartesian[molecule_indices]
    final_center_of_mass = center_of_mass_cartesian(final_molecule_positions, molecule_species)

    if output_path is None:
        input_path = Path(poscar_path).resolve()
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            (DEFAULT_OUTPUT_DIR / f"{input_path.stem}_molecule_adjusted{input_path.suffix or '.vasp'}").resolve()
        )
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        structure.lattice,
        reframed_direct,
        structure.counts,
        structure.species,
        comment=f"{structure.comment} | molecule adjusted",
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=structure.selective_flags,
    )

    return MoleculeTransformRun(
        output_path=Path(output_path).resolve(),
        molecule_atom_count=selection.molecule_atom_count,
        substrate_atom_count=selection.substrate_atom_count,
        z_cutoff=selection.z_cutoff,
        gap_size=selection.gap_size,
        center_of_mass_before=com_before,
        center_of_mass_after=final_center_of_mass,
        target_cartesian=final_center_of_mass,
        reframe_shift_direct=shift_direct,
    )


def place_molecule_on_site(
    substrate_poscar: str,
    molecule_poscar: str,
    *,
    site_type: str,
    site_index: int = 1,
    height: float = 2.5,
    rotation_deg: float = 0.0,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
    surface_side: str = "top",
    layer_tolerance: float = 0.35,
    neighbour_tolerance: float = 0.15,
    hollow_match_tolerance: float | None = None,
    reframe_axes: str | Sequence[str] | None = "xy",
    auto_repeat_substrate: bool = False,
    fit_padding: float = 0.15,
    output_path: str | None = None,
) -> AdsorbRun:
    substrate = io_mod.read_poscar(substrate_poscar)
    molecule = io_mod.read_poscar(molecule_poscar)
    if auto_repeat_substrate:
        repeat_a, repeat_b = _estimate_inplane_repeats_for_molecule(
            substrate,
            molecule,
            rotation_deg=float(rotation_deg),
            tilt_deg=float(tilt_deg),
            roll_deg=float(roll_deg),
            fit_padding=float(fit_padding),
        )
        substrate = surface_mod._repeat_structure_inplane(substrate, repeat_a, repeat_b)

    site_report = surface_mod.find_adsorption_sites(
        substrate,
        surface_side=surface_side,
        layer_tolerance=layer_tolerance,
        neighbour_tolerance=neighbour_tolerance,
        hollow_match_tolerance=hollow_match_tolerance,
    )
    selected_site = surface_mod.select_adsorption_site(site_report, site_type, site_index)

    molecule_species_expanded = _expand_species(molecule.species, molecule.counts)
    placed_molecule_cartesian, center_before = _translate_molecule_to_site(
        molecule.positions_cartesian,
        molecule_species_expanded,
        substrate.lattice,
        selected_site.cartesian,
        surface_side=surface_side,
        height=height,
        rotation_deg=rotation_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    positions_direct_out, counts_out, species_out, flags_out, shift_direct, final_molecule_cartesian = (
        _combine_substrate_and_molecule(
            substrate,
            molecule,
            placed_molecule_cartesian,
            reframe_axes=reframe_axes,
        )
    )
    center_after = center_of_mass_cartesian(final_molecule_cartesian, molecule_species_expanded)

    if output_path is None:
        substrate_path = Path(substrate_poscar).resolve()
        molecule_path = Path(molecule_poscar).resolve()
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        canonical_site = surface_mod.canonical_site_type(site_type)
        output_path = str(
            (
                DEFAULT_OUTPUT_DIR
                / f"{substrate_path.stem}__{molecule_path.stem}_{canonical_site}{int(site_index):02d}{substrate_path.suffix or '.vasp'}"
            ).resolve()
        )
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        substrate.lattice,
        positions_direct_out,
        counts_out,
        species_out,
        comment=(
            f"{substrate.comment} | adsorbed {Path(molecule_poscar).stem} on "
            f"{surface_mod.canonical_site_type(site_type)} #{int(site_index)}"
        ),
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=flags_out,
    )

    return AdsorbRun(
        output_path=Path(output_path).resolve(),
        substrate_atom_count=substrate.natoms,
        molecule_atom_count=molecule.natoms,
        site_type=surface_mod.canonical_site_type(site_type),
        site_index=int(site_index),
        surface_side=str(surface_side),
        height=float(height),
        center_of_mass_before=center_before,
        center_of_mass_after=center_after,
        site_cartesian=np.array(selected_site.cartesian, dtype=float),
        site_direct=np.array(selected_site.direct, dtype=float),
        reframe_shift_direct=shift_direct,
    )


def shift_top_layer(
    poscar_path: str,
    *,
    output_path: str | None = None,
    shift_cartesian: Sequence[float] | None = None,
    shift_direct: Sequence[float] | None = None,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
) -> LayerShiftRun:
    structure = io_mod.read_poscar(poscar_path)
    selection = identify_top_group(structure, z_cutoff=z_cutoff, min_gap=min_gap)
    cartesian_shift, direct_shift = _resolve_shift_vectors(structure.lattice, shift_cartesian, shift_direct)

    direct_positions = np.array(structure.positions_direct, dtype=float, copy=True)
    top_indices = np.array(selection.molecule_indices, dtype=int)
    direct_positions[top_indices] += direct_shift

    if output_path is None:
        input_path = Path(poscar_path).resolve()
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            (DEFAULT_OUTPUT_DIR / f"{input_path.stem}_upper_layer_shifted{input_path.suffix or '.vasp'}").resolve()
        )
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        structure.lattice,
        direct_positions,
        structure.counts,
        structure.species,
        comment=f"{structure.comment} | upper layer shifted",
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=structure.selective_flags,
    )

    return LayerShiftRun(
        output_path=Path(output_path).resolve(),
        top_atom_count=selection.molecule_atom_count,
        bottom_atom_count=selection.substrate_atom_count,
        z_cutoff=selection.z_cutoff,
        gap_size=selection.gap_size,
        shift_cartesian=cartesian_shift,
        shift_direct=direct_shift,
    )


__all__ = [
    "ATOMIC_MASSES",
    "AdsorbRun",
    "LayerShiftRun",
    "MoleculeSelection",
    "MoleculeTransformRun",
    "center_of_mass_cartesian",
    "identify_top_group",
    "identify_top_molecule",
    "place_molecule_on_site",
    "shift_top_layer",
    "transform_top_molecule",
]

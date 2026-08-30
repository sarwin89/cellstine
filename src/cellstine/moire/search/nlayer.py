"""Multi-layer commensuration on top of the bilayer Gram-form engine.

A stack of ``N`` layers is commensurate when every layer has an integer
supercell and the ``N`` supercells describe one and the same two-dimensional
lattice.  Searching that condition directly in ``N`` layers is exponential, but
it factorises exactly through the base layer:

* Run the bilayer engine once per upper layer with the base held **unstrained**.
  A candidate then gives an integer base supercell ``A_i`` and an integer upper
  supercell ``B_i`` with ``F_i (B_i L_i) = R_i (A_i L_0)``, where ``F_i`` is the
  recorded affine of the upper layer, ``R_i`` the recorded affine of the base and
  ``L_0``, ``L_i`` the primitive in-plane bases.  A zero base strain budget forces
  ``R_i`` to be a rotation, so ``R_i^{-1} F_i`` maps layer ``i`` onto the base
  supercell ``A_i L_0`` in the base's own frame.
* Any cell shared by all the layers is then a common superlattice of the
  sublattices ``L(A_1), ..., L(A_k)`` of the base lattice, and the smallest one is
  their intersection ``S``.  The intersection is computed here in exact integer
  arithmetic, so ``S = A_i C_i`` with an integer ``C_i`` and layer ``i`` is built
  with the integer supercell ``B_i C_i``.  Nothing is rounded and no tolerance is
  involved: the resulting stack is commensurate by construction, with each layer
  carrying exactly the strain the bilayer search recorded for it.

The exact statements this rests on are proved in Lean in
``RequestProject/SublatticeIntersection.lean``: the kernel construction below
returns the intersection (``Cellstine.rowLattice_eq_inf_of_kernel_spanning``),
the intersection is the largest common sublattice and its cell is the smallest
common cell (``Cellstine.isGreatest_inf_rowLattice``,
``Cellstine.isLeast_abs_det_inf``), it is nonsingular whenever the inputs are
(``Cellstine.det_ne_zero_of_rowLattice_eq_inf``), the quotient below is the
unique integer factor (``Cellstine.rowLattice_le_iff_exists_factor``,
``Cellstine.factor_unique``), and the rebuilt layers then all carry the same
cell (``Cellstine.stack_shares_cell``).  The pruning the combination stage does
rests on ``RequestProject/PruneBounds.lean``:
``Cellstine.prefixAtoms_le_stackAtoms`` and ``Cellstine.prune_atoms_sound`` for
the atom bound, ``Cellstine.second_gram_le_of_sublattice`` and
``Cellstine.prune_length_sound`` for the length bound.

The base is deliberately left unstrained.  Two different upper layers generally
ask for two different base strains, so any nonzero base budget would make the
recorded pair data mutually inconsistent; keeping the base rigid is both the
usual experimental picture (a substrate with films on top) and the only choice
that keeps the construction exact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ...core.species import expand_species
from ...core.lattice import vector_angle_deg
from ...core.symmetry2d import DEFAULT_SYMMETRY_TOLERANCE, idealised_layer_lattice
from ...io import native as io_mod
from .find import LAYER_REDUCTION_SYMPREC, primitive_layer_cell, run_find
from .gram_lattice import _reduce_basis as _gauge_reduce_basis

SCHEMA_NAME = "cellstine.moire.nlayer"
SCHEMA_VERSION = 1


def _integer_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 integer matrix")
    if int(round(float(np.linalg.det(matrix.astype(float))))) == 0:
        raise ValueError(f"{name} must be nonsingular")
    return matrix


def _exact_determinant(matrix: np.ndarray) -> int:
    values = np.asarray(matrix, dtype=object)
    return int(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])


def _exact_adjugate(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=object)
    return np.array(
        [[values[1, 1], -values[0, 1]], [-values[1, 0], values[0, 0]]], dtype=object
    )


def integer_left_kernel(matrix: np.ndarray) -> np.ndarray:
    """Return a basis of ``{u in Z^m : u @ matrix == 0}`` as integer rows.

    The reduction is a row Hermite elimination in exact Python integer
    arithmetic with the unimodular transform tracked alongside, so the returned
    rows are an exact basis of the kernel lattice, not a floating-point
    null space.
    """

    work = np.asarray(matrix, dtype=object).copy()
    rows, columns = work.shape
    transform = np.eye(rows, dtype=object)
    pivot_row = 0
    for column in range(columns):
        while True:
            nonzero = [index for index in range(pivot_row, rows) if work[index, column] != 0]
            if len(nonzero) <= 1:
                break
            nonzero.sort(key=lambda index: abs(work[index, column]))
            head = nonzero[0]
            for index in nonzero[1:]:
                factor = work[index, column] // work[head, column]
                work[index] -= factor * work[head]
                transform[index] -= factor * transform[head]
        if not nonzero:
            continue
        pivot = nonzero[0]
        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]
            transform[[pivot_row, pivot]] = transform[[pivot, pivot_row]]
        pivot_row += 1
    kernel = [transform[index] for index in range(rows) if not np.any(work[index])]
    if not kernel:
        return np.zeros((0, rows), dtype=np.int64)
    return np.array([[int(value) for value in row] for row in kernel], dtype=np.int64)


def sublattice_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return an integer basis of ``L(first) ∩ L(second)``.

    Both arguments hold the generators of a sublattice of ``Z^2`` in their rows.
    A vector lies in both sublattices exactly when it is ``u @ first`` and
    ``v @ second`` for integer ``u`` and ``v``, that is when ``(u, v)`` lies in the
    left kernel of ``[first; -second]``; the kernel has rank two and its image
    under ``u`` is the intersection.
    """

    left = _integer_matrix(first, "first sublattice")
    right = _integer_matrix(second, "second sublattice")
    stacked = np.vstack((left, -right))
    kernel = integer_left_kernel(stacked)
    if kernel.shape[0] != 2:
        raise ArithmeticError("the intersection of two full-rank sublattices must have rank two")
    basis = kernel[:, :2] @ left
    mirrored = kernel[:, 2:] @ right
    if not np.array_equal(basis, mirrored):
        raise ArithmeticError("intersection basis is not shared by both sublattices")
    if _exact_determinant(basis) < 0:
        basis = basis[::-1].copy()
    return np.asarray(basis, dtype=np.int64)


def intersect_sublattices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Return an integer basis of the intersection of several sublattices."""

    if not matrices:
        raise ValueError("at least one sublattice is required")
    basis = _integer_matrix(matrices[0], "sublattice")
    for matrix in matrices[1:]:
        basis = sublattice_intersection(basis, matrix)
    return basis


def quotient_matrix(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Return the integer ``C`` with ``fine == C @ coarse``.

    ``fine`` must generate a sublattice of ``L(coarse)``; the quotient is formed
    from the exact adjugate so that no rounding decides integrality.
    """

    coarse_matrix = _integer_matrix(coarse, "coarse sublattice")
    fine_matrix = np.asarray(fine, dtype=object)
    determinant = _exact_determinant(coarse_matrix)
    product_matrix = fine_matrix @ _exact_adjugate(coarse_matrix)
    if any(int(value) % determinant != 0 for value in product_matrix.ravel()):
        raise ValueError("the fine lattice is not a sublattice of the coarse one")
    return np.array(
        [[int(value) // determinant for value in row] for row in product_matrix], dtype=np.int64
    )


def reduce_supercell(matrix: np.ndarray, base_lattice: np.ndarray) -> np.ndarray:
    """Return a Lagrange-Gauss reduced generator matrix of the same sublattice.

    The reduction is measured in the Cartesian metric of ``base_lattice``, so the
    returned supercell is the short, well-shaped choice of the same lattice.
    """

    integers = _integer_matrix(matrix, "supercell")
    cartesian_columns = (integers.astype(float) @ np.asarray(base_lattice, dtype=float)).T
    _, gauge = _gauge_reduce_basis(cartesian_columns)
    reduced = np.asarray(gauge, dtype=np.int64).T @ integers
    if _exact_determinant(reduced) < 0:
        reduced = reduced[::-1].copy()
    return reduced


@dataclass(frozen=True)
class LayerMatch:
    """One bilayer candidate of an upper layer against the fixed base layer."""

    layer_index: int
    poscar: str
    atom_count: int
    base_matrix: np.ndarray
    layer_matrix: np.ndarray
    affine: np.ndarray
    angle_deg: float
    strain: tuple[float, float]
    #: The file the user named, when ``poscar`` is the folded cell written for it.
    poscar_source: str | None = None
    #: How many primitive in-plane cells that file held.
    cell_multiplicity: int = 1

    @property
    def base_multiplicity(self) -> int:
        return abs(_exact_determinant(self.base_matrix))

    @property
    def max_abs_strain(self) -> float:
        return max(abs(value) for value in self.strain)


@dataclass
class NLayerCandidate:
    """One commensurate multi-layer cell."""

    index: int
    base_matrix: np.ndarray
    shared_lattice: np.ndarray
    base_atom_count: int
    total_atoms: int
    layers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cell_lengths(self) -> tuple[float, float]:
        return (
            float(np.linalg.norm(self.shared_lattice[0])),
            float(np.linalg.norm(self.shared_lattice[1])),
        )

    @property
    def cell_angle_deg(self) -> float:
        first, second = self.shared_lattice
        return vector_angle_deg(first, second)

    @property
    def max_abs_strain(self) -> float:
        if not self.layers:
            return 0.0
        return max(float(layer["max_abs_strain"]) for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        first, second = self.cell_lengths
        return {
            "index": int(self.index),
            "base_matrix": [[int(value) for value in row] for row in self.base_matrix],
            "shared_lattice": [[float(value) for value in row] for row in self.shared_lattice],
            "cell_lengths": [first, second],
            "cell_angle_deg": self.cell_angle_deg,
            "coincidence_index": abs(_exact_determinant(self.base_matrix)),
            "base_atom_count": int(self.base_atom_count),
            "total_atoms": int(self.total_atoms),
            "max_abs_strain": self.max_abs_strain,
            "layers": [dict(layer) for layer in self.layers],
        }


def _affine_in_base_frame(base_affine: np.ndarray, layer_affine: np.ndarray) -> np.ndarray:
    """Undo the recorded rigid rotation of the base layer.

    With a zero base strain budget the recorded base affine is orthogonal, so its
    inverse is its transpose and composing it with the layer affine leaves every
    length inside the layer, and therefore every reported strain, untouched.
    """

    rotation = np.asarray(base_affine, dtype=float)
    product_matrix = rotation.T @ rotation
    if not np.allclose(product_matrix, np.eye(2), atol=1e-9):
        raise ValueError(
            "the base layer affine is not a rotation; run the layer searches with a zero base strain"
        )
    return rotation.T @ np.asarray(layer_affine, dtype=float)


def layer_matches_from_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    layer_index: int,
    poscar: str,
    atom_count: int,
    poscar_source: str | None = None,
    cell_multiplicity: int = 1,
) -> list[LayerMatch]:
    """Convert bilayer search records into base-frame layer matches."""

    matches: list[LayerMatch] = []
    for candidate in candidates:
        affine = _affine_in_base_frame(
            np.asarray(candidate["bottom_affine"], dtype=float),
            np.asarray(candidate["top_affine"], dtype=float),
        )
        strain = tuple(float(value) for value in candidate["top_layer_strain"])
        matches.append(
            LayerMatch(
                layer_index=int(layer_index),
                poscar=str(poscar),
                atom_count=int(atom_count),
                base_matrix=_integer_matrix(candidate["bottom_matrix"], "bottom matrix"),
                layer_matrix=_integer_matrix(candidate["top_matrix"], "top matrix"),
                affine=affine,
                angle_deg=float(candidate["angle_deg"]),
                strain=(strain[0], strain[1]),
                poscar_source=None if poscar_source is None else str(poscar_source),
                cell_multiplicity=int(cell_multiplicity),
            )
        )
    return matches


def combine_layer_matches(
    matches: Sequence[LayerMatch],
    *,
    base_lattice: np.ndarray,
    base_atom_count: int,
    max_atoms: int | None = None,
    max_length: float | None = None,
    max_aspect_ratio: float | None = None,
    min_cell_angle_deg: float | None = None,
    max_cell_angle_deg: float | None = None,
) -> NLayerCandidate | None:
    """Return the smallest cell shared by one match per layer, or ``None``.

    ``None`` means the combination exists but fails one of the size or shape
    limits; the arithmetic itself never fails for nonsingular matrices.
    """

    planar = np.asarray(base_lattice, dtype=float)[:2, :2]
    shared_integer = reduce_supercell(
        intersect_sublattices([match.base_matrix for match in matches]), planar
    )
    multiplicity = abs(_exact_determinant(shared_integer))
    total_atoms = multiplicity * int(base_atom_count)
    layers: list[dict[str, Any]] = []
    for match in matches:
        quotient = quotient_matrix(match.base_matrix, shared_integer)
        layer_matrix = quotient @ match.layer_matrix
        layer_atoms = abs(_exact_determinant(layer_matrix)) * match.atom_count
        total_atoms += layer_atoms
        layers.append(
            {
                "layer": int(match.layer_index),
                "poscar": match.poscar,
                "matrix": [[int(value) for value in row] for row in layer_matrix],
                "affine": [[float(value) for value in row] for row in match.affine],
                "angle_deg": float(match.angle_deg),
                "strain": [float(match.strain[0]), float(match.strain[1])],
                "max_abs_strain": match.max_abs_strain,
                "atom_count": int(layer_atoms),
                "primitive_atom_count": int(match.atom_count),
                "poscar_source": match.poscar_source,
                "cell_multiplicity": int(match.cell_multiplicity),
            }
        )
        if max_atoms is not None and total_atoms > int(max_atoms):
            return None

    shared_cartesian = shared_integer.astype(float) @ planar
    lengths = np.linalg.norm(shared_cartesian, axis=1)
    if max_length is not None and float(lengths.max()) > float(max_length) + 1e-9:
        return None
    if max_aspect_ratio is not None and float(lengths.max() / lengths.min()) > float(max_aspect_ratio):
        return None
    candidate = NLayerCandidate(
        index=0,
        base_matrix=shared_integer,
        shared_lattice=shared_cartesian,
        base_atom_count=multiplicity * int(base_atom_count),
        total_atoms=total_atoms,
        layers=layers,
    )
    angle = candidate.cell_angle_deg
    if min_cell_angle_deg is not None and angle < float(min_cell_angle_deg) - 1e-9:
        return None
    if max_cell_angle_deg is not None and angle > float(max_cell_angle_deg) + 1e-9:
        return None
    return candidate


def viable_combinations(
    matches_by_layer: Sequence[Sequence[LayerMatch]],
    *,
    base_lattice: np.ndarray,
    base_atom_count: int,
    max_atoms: int | None = None,
    max_length: float | None = None,
) -> Iterable[tuple[LayerMatch, ...]]:
    """Yield the combinations that can still meet the monotone size limits.

    One match is taken from each layer, in the same order a flat product would
    visit them, but the walk is depth-first over the layers and a prefix is
    abandoned as soon as it cannot lead to an admissible stack.  Both limits
    used here are monotone: the shared cell of a longer prefix is a sublattice
    of the shared cell of a shorter one, so its multiplicity, its atom count and
    its second successive minimum can only grow.  Concretely, for a prefix with
    shared cell ``S`` of multiplicity ``m``,

    * every stack extending it has at least ``m * base_atom_count +
      sum_i (m / |det A_i|) * |det B_i| * n_i`` atoms, the sum running over the
      layers already fixed, and
    * every such stack has a cell at least as long as the longer row of the
      Lagrange-Gauss reduced form of ``S``.

    The shape filters (aspect ratio and cell angles) are *not* monotone and are
    therefore left to :func:`combine_layer_matches`, which re-checks the size
    limits as well; this enumeration only removes combinations that function
    would have rejected anyway.

    Both bounds are proved monotone in ``RequestProject/PruneBounds.lean``:
    ``Cellstine.prefixAtoms_le_stackAtoms`` and ``Cellstine.prune_atoms_sound``
    for the atom count, ``Cellstine.second_gram_le_of_sublattice`` and
    ``Cellstine.prune_length_sound`` for the cell length.
    """

    depth = len(matches_by_layer)
    if depth == 0 or any(len(layer) == 0 for layer in matches_by_layer):
        return
    planar = np.asarray(base_lattice, dtype=float)[:2, :2]
    atoms_per_base = int(base_atom_count)
    limit_atoms = None if max_atoms is None else int(max_atoms)
    limit_length = None if max_length is None else float(max_length)

    def survives(basis: np.ndarray, prefix: list[LayerMatch]) -> bool:
        multiplicity = abs(_exact_determinant(basis))
        if limit_atoms is not None:
            atoms = multiplicity * atoms_per_base
            for match in prefix:
                repeats = multiplicity // abs(_exact_determinant(match.base_matrix))
                atoms += repeats * abs(_exact_determinant(match.layer_matrix)) * match.atom_count
                if atoms > limit_atoms:
                    return False
        if limit_length is not None:
            reduced = reduce_supercell(basis, planar)
            lengths = np.linalg.norm(reduced.astype(float) @ planar, axis=1)
            if float(lengths.max()) > limit_length + 1e-9:
                return False
        return True

    prefix: list[LayerMatch] = []

    def walk(index: int, basis: np.ndarray | None) -> Iterable[tuple[LayerMatch, ...]]:
        for match in matches_by_layer[index]:
            shared = (
                np.asarray(match.base_matrix, dtype=np.int64)
                if basis is None
                else sublattice_intersection(basis, match.base_matrix)
            )
            prefix.append(match)
            if survives(shared, prefix):
                if index + 1 == depth:
                    yield tuple(prefix)
                else:
                    yield from walk(index + 1, shared)
            prefix.pop()

    yield from walk(0, None)


def _candidate_sort_key(candidate: NLayerCandidate) -> tuple[float, float, float]:
    return (float(candidate.total_atoms), candidate.max_abs_strain, candidate.cell_lengths[0])


def _deduplicate(candidates: Sequence[NLayerCandidate]) -> list[NLayerCandidate]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[NLayerCandidate] = []
    for candidate in candidates:
        key = (
            tuple(int(value) for value in np.asarray(candidate.base_matrix).ravel()),
            tuple(
                (
                    tuple(int(value) for row in layer["matrix"] for value in row),
                    round(float(layer["angle_deg"]), 9),
                )
                for layer in candidate.layers
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


@dataclass
class FindNRun:
    """Artifacts and in-memory results of one multi-layer search."""

    run_id: str
    result_path: Path
    candidates: list[dict[str, Any]]
    document: dict[str, Any]
    timings: dict[str, float]


def _slug(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "structure"


def run_findn(
    *,
    base_poscar: str,
    upper_poscars: Sequence[str],
    max_length: float,
    layer_strains: Sequence[float] | float = 0.02,
    min_length: float | None = None,
    max_atoms: int | None = 2000,
    max_pair_atoms: int | None = None,
    max_aspect_ratio: float = 12.0,
    min_cell_angle_deg: float = 25.0,
    max_cell_angle_deg: float = 155.0,
    per_layer_limit: int = 40,
    max_candidates: int = 200,
    reduce_layers: bool = True,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    output_root: str = "runs",
) -> FindNRun:
    """Search commensurate cells for a base layer plus one or more upper layers.

    Every upper layer is matched against the rigid base with the bilayer engine,
    and the per-layer results are combined exactly through the intersection of
    their base supercells.

    As in the bilayer search, each layer is first folded onto its own primitive
    in-plane cell: a layer handed in as a supercell of itself would otherwise
    make every reported cell a repeat of a smaller one and hide the small cells
    altogether.  The folded layer is written next to the results and is the file
    every reported matrix refers to, with the file the user named recorded
    beside it.  ``reduce_layers=False`` searches the cells exactly as given.
    """

    if not upper_poscars:
        raise ValueError("at least one upper layer is required")
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    if float(symmetry_tolerance) <= 0.0:
        raise ValueError("symmetry_tolerance must be positive")
    upper_paths = [Path(path).resolve() for path in upper_poscars]

    if isinstance(layer_strains, (int, float)):
        strains = [float(layer_strains)] * len(upper_paths)
    else:
        strains = [float(value) for value in layer_strains]
    if len(strains) != len(upper_paths):
        raise ValueError("one strain budget is required per upper layer")

    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_poscar).resolve()
    base_structure = io_mod.read_poscar(str(base_path))
    base_source: str | None = None
    base_multiplicity = 1
    if reduce_layers:
        folded, folded_path, base_multiplicity = primitive_layer_cell(
            base_structure, base_path, output_dir, "base", LAYER_REDUCTION_SYMPREC
        )
        if base_multiplicity > 1:
            base_source, base_path, base_structure = str(base_path), folded_path, folded
    # The bilayer stage idealises every layer onto the metric its own point group
    # preserves exactly, so the base cell that closes the multi-layer arithmetic
    # has to be the same idealised one; otherwise the shared cell inherits the
    # rounding of the printed POSCAR instead of the symmetry of the layer.
    base_lattice, base_idealisation = idealised_layer_lattice(
        base_structure.lattice,
        base_structure.positions_direct,
        expand_species(base_structure.species, base_structure.counts, "base"),
        tolerance=float(symmetry_tolerance),
        name="base",
    )

    stage_start = time.perf_counter()
    matches_by_layer: list[list[LayerMatch]] = []
    pair_paths: list[str] = []
    layer_paths: list[Path] = []
    for position, (path, budget) in enumerate(zip(upper_paths, strains), start=1):
        pair_dir = output_dir / f"layer_{position:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        structure = io_mod.read_poscar(str(path))
        layer_path = path
        layer_source: str | None = None
        cell_multiplicity = 1
        if reduce_layers:
            folded, folded_path, cell_multiplicity = primitive_layer_cell(
                structure, path, pair_dir, "layer", LAYER_REDUCTION_SYMPREC
            )
            if cell_multiplicity > 1:
                layer_source, layer_path, structure = str(path), folded_path, folded
        layer_paths.append(layer_path)
        pair_run = run_find(
            top_poscar=str(layer_path),
            bottom_poscar=str(base_path),
            max_length=float(max_length),
            top_strain=float(budget),
            bottom_strain=0.0,
            min_length=None if min_length is None else float(min_length),
            max_atoms=None if max_pair_atoms is None else int(max_pair_atoms),
            max_aspect_ratio=float(max_aspect_ratio),
            min_cell_angle_deg=float(min_cell_angle_deg),
            max_cell_angle_deg=float(max_cell_angle_deg),
            symmetry_tolerance=float(symmetry_tolerance),
            reduce_layers=False,
            output_root=str(pair_dir),
        )
        pair_paths.append(str(pair_run.result_path))
        matches = layer_matches_from_candidates(
            pair_run.candidates,
            layer_index=position,
            poscar=str(layer_path),
            atom_count=structure.natoms,
            poscar_source=layer_source,
            cell_multiplicity=cell_multiplicity,
        )
        matches.sort(key=lambda item: (item.base_multiplicity, item.max_abs_strain))
        matches_by_layer.append(matches[: max(1, int(per_layer_limit))])
        if not matches:
            timings["total_s"] = time.perf_counter() - total_start
            raise ValueError(
                f"no bilayer candidate was found for upper layer {position}; "
                "raise --max-length or the strain budget"
            )
    timings["pair_search_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    combined: list[NLayerCandidate] = []
    for combination in viable_combinations(
        matches_by_layer,
        base_lattice=base_lattice,
        base_atom_count=base_structure.natoms,
        max_atoms=max_atoms,
        max_length=max_length,
    ):
        candidate = combine_layer_matches(
            combination,
            base_lattice=base_lattice,
            base_atom_count=base_structure.natoms,
            max_atoms=max_atoms,
            max_length=max_length,
            max_aspect_ratio=max_aspect_ratio,
            min_cell_angle_deg=min_cell_angle_deg,
            max_cell_angle_deg=max_cell_angle_deg,
        )
        if candidate is not None:
            combined.append(candidate)
    combined = _deduplicate(sorted(combined, key=_candidate_sort_key))[: max(1, int(max_candidates))]
    for position, candidate in enumerate(combined, start=1):
        candidate.index = position
    timings["combine_s"] = time.perf_counter() - stage_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_label = Path(base_source).stem if base_source is not None else base_path.stem
    run_id = f"{timestamp}_{_slug(base_label)}_base__{len(upper_paths) + 1}layers"
    document = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "search": {
            "base_poscar": str(base_path),
            "base_poscar_source": base_source,
            "base_cell_multiplicity": int(base_multiplicity),
            "upper_poscars": [str(path) for path in layer_paths],
            "upper_poscar_sources": [str(path) for path in upper_paths],
            "reduce_layers": bool(reduce_layers),
            "pair_results": pair_paths,
            "layer_strains": strains,
            "base_strain": 0.0,
            "max_length": float(max_length),
            "min_length": None if min_length is None else float(min_length),
            "max_atoms": None if max_atoms is None else int(max_atoms),
            "max_pair_atoms": None if max_pair_atoms is None else int(max_pair_atoms),
            "max_aspect_ratio": float(max_aspect_ratio),
            "min_cell_angle_deg": float(min_cell_angle_deg),
            "max_cell_angle_deg": float(max_cell_angle_deg),
            "per_layer_limit": int(per_layer_limit),
            "symmetry_tolerance": float(symmetry_tolerance),
            "base_idealisation": float(base_idealisation),
            "base_atom_count": int(base_structure.natoms),
            "base_lattice": [[float(value) for value in row] for row in np.asarray(base_lattice, dtype=float)],
        },
        "candidates": [candidate.to_dict() for candidate in combined],
    }
    result_path = output_dir / "results_nlayer.json"
    result_path.write_text(json.dumps(document, indent=2) + "\n")
    timings["total_s"] = time.perf_counter() - total_start
    return FindNRun(
        run_id=run_id,
        result_path=result_path.resolve(),
        candidates=document["candidates"],
        document=document,
        timings=timings,
    )


def read_nlayer_results(path: str | Path) -> dict[str, Any]:
    """Read and validate a multi-layer results document."""

    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA_NAME:
        raise ValueError(f"{path} is not a {SCHEMA_NAME} document")
    if int(payload.get("version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported {SCHEMA_NAME} version {payload.get('version')}")
    for key in ("search", "candidates"):
        if key not in payload:
            raise ValueError(f"{path} is missing the '{key}' section")
    return payload

"""Native bilayer workflow around the Gram-form search engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...core import symmetry3d
from ...core.species import expand_species, group_species
from ...core.symmetry2d import (
    DEFAULT_SYMMETRY_TOLERANCE,
    column_basis_from_lattice,
    layer_point_group,
    symmetrised_basis,
)
from ...io import native as io_mod
from . import gram
from .results import build_results_document, write_results


# A wide bilayer search admits tens of thousands of supercells, almost all of
# them dominated by a smaller and less strained one.  The workflows record this
# many by default, always including the whole Pareto front.
DEFAULT_CANDIDATE_LIMIT = 500

# Cartesian tolerance, in angstrom, for deciding that a shift maps a layer onto
# itself.  A POSCAR carries six fractional decimals, so a coordinate of a cell
# tens of angstrom wide is only good to about 1e-5 A; 1e-4 A is tight enough to
# refuse a shift that is not a symmetry and loose enough to accept one that is.
LAYER_REDUCTION_SYMPREC = 1e-4


@dataclass
class FindRun:
    """Artifacts and in-memory results from one native Gram search."""

    run_id: str
    result_path: Path
    result: gram.SearchResult
    candidates: list[dict[str, Any]]
    parameters: dict[str, Any]
    timings: dict[str, float]
    #: How many primitive in-plane cells the file named for each layer held.
    layer_index: dict[str, int] = field(default_factory=lambda: {"top": 1, "bottom": 1})


def _slug(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "structure"


def _make_result_path(output_root: str, bottom_path: str, top_path: str) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{timestamp}_{_slug(Path(bottom_path).stem)}_below__"
        f"{_slug(Path(top_path).stem)}_above_gram"
    )
    output_dir = Path(output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir / "results.json"


def _planar_column_basis(lattice: np.ndarray, name: str) -> np.ndarray:
    """Convert POSCAR's planar Cartesian row basis to Gram engine columns."""

    return column_basis_from_lattice(lattice, name=f"{name} POSCAR lattice")


def _layer_group(
    structure: io_mod.PoscarData,
    name: str,
    tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> np.ndarray:
    """Return the point group of a decorated layer, not of its bare lattice.

    Folding the search by the bare lattice group would identify stackings that a
    three-fold layer such as hBN or MoS2 distinguishes, so the decorated group is
    what the engine must use.
    """

    return layer_point_group(
        structure.lattice,
        structure.positions_direct,
        expand_species(structure.species, structure.counts, name),
        lattice_tolerance=float(tolerance),
        name=name,
    )


def primitive_layer_cell(
    structure: io_mod.PoscarData,
    path: Path,
    output_dir: Path,
    name: str,
    tolerance: float,
) -> tuple[io_mod.PoscarData, Path, int]:
    """Return the layer in its primitive in-plane cell, and where it was written.

    A layer handed in as a supercell of itself --- a ``2 x 2`` graphene cell, say
    --- makes every commensurate cell of the search a repeat of a smaller one:
    the best twisted bilayer at 21.79 degrees comes back with 112 atoms instead
    of 28, and the untwisted stack with 16 instead of 4.  Those are not extra
    options, they are the same structures reported four times too large, and the
    genuinely small cells are missing altogether because no supercell of the
    input cell realises them.

    So the search is run on the primitive in-plane cell of each layer.  Only
    in-plane translations are used, which leaves a layer that repeats along the
    normal --- an AA bilayer given as one layer --- exactly as it was.  The
    reduced layer is written next to the results so that every matrix the run
    reports refers to a file on disk, and the file the user named is recorded
    beside it.
    """

    species = expand_species(structure.species, structure.counts, natoms=structure.natoms)
    lattice, positions, symbols, index = symmetry3d.planar_primitive_layer(
        structure.lattice, structure.positions_direct, species, symprec=float(tolerance)
    )
    if index <= 1:
        return structure, path, 1
    ordered_species, counts, order = group_species(symbols)
    lattice = np.asarray(lattice, dtype=float)
    direct = np.asarray(positions, dtype=float)[order]
    comment = f"{structure.comment} | primitive in-plane cell (1 of {index})"
    output_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = (output_dir / f"primitive_{name}.vasp").resolve()
    io_mod.write_poscar(
        str(reduced_path),
        lattice,
        direct,
        [int(value) for value in counts],
        list(ordered_species),
        comment,
        positions_are_cartesian=False,
    )
    return io_mod.read_poscar(str(reduced_path)), reduced_path, int(index)


def _notify(
    progress_callback: Callable[[str, str], None] | None,
    stage: str,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message)


def run_find(
    *,
    top_poscar: str,
    bottom_poscar: str,
    max_length: float,
    top_strain: float,
    bottom_strain: float,
    min_length: float | None = None,
    max_atoms: int | None = None,
    max_candidates: int | None = None,
    max_aspect_ratio: float = 12.0,
    min_cell_angle_deg: float = 25.0,
    max_cell_angle_deg: float = 155.0,
    min_twist_angle_deg: float | None = None,
    max_twist_angle_deg: float | None = None,
    fold_symmetry: bool = True,
    symmetric: bool = False,
    primitive_only: bool = True,
    reduce_layers: bool = True,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    output_root: str = "runs",
    progress_callback: Callable[[str, str], None] | None = None,
) -> FindRun:
    """Read two POSCARs, run one native search, and write validated JSON v2.

    ``top_strain`` and ``bottom_strain`` are bounds on principal logarithmic strain.
    There is no angle shortlist, index-box search, or DAT compatibility path.
    ``min_twist_angle_deg`` and ``max_twist_angle_deg`` are read on the reported
    twist, so they select bilayers rather than steering the enumeration: the
    same candidates come out as from the unrestricted search, minus the ones
    whose twist falls outside the window.

    Each layer is reduced to its primitive in-plane cell before the search, so a
    layer handed in as a supercell of itself still yields the small commensurate
    cells; the reduced layer is written beside the results and is what every
    reported matrix refers to, with the file the user named recorded in
    ``metadata.top_poscar_source`` / ``metadata.bottom_poscar_source`` and the
    number of primitive cells it held in ``metadata.top_layer_index`` /
    ``metadata.bottom_layer_index``.  ``reduce_layers=False`` searches the cells
    exactly as given.

    ``max_candidates`` bounds how many candidates reach ``results.json``: the
    whole Pareto front of size against strain is always kept and the remaining
    budget goes to the smallest cells, so a wide search reports the cells worth
    building instead of every admissible supercell.  ``None`` or a nonpositive
    value records all of them.
    """

    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    top_path = Path(top_poscar).resolve()
    bottom_path = Path(bottom_poscar).resolve()
    if float(symmetry_tolerance) <= 0.0:
        raise ValueError("symmetry_tolerance must be positive")

    _notify(progress_callback, "read", "reading input structures")
    stage_start = time.perf_counter()
    run_id, result_path = _make_result_path(str(output_root), str(bottom_path), str(top_path))
    top_structure = io_mod.read_poscar(str(top_path))
    bottom_structure = io_mod.read_poscar(str(bottom_path))
    top_source: Path | None = None
    bottom_source: Path | None = None
    top_layer_index = 1
    bottom_layer_index = 1
    if reduce_layers:
        reduced_top, reduced_top_path, top_layer_index = primitive_layer_cell(
            top_structure, top_path, result_path.parent, "top", LAYER_REDUCTION_SYMPREC
        )
        reduced_bottom, reduced_bottom_path, bottom_layer_index = primitive_layer_cell(
            bottom_structure, bottom_path, result_path.parent, "bottom", LAYER_REDUCTION_SYMPREC
        )
        if top_layer_index > 1:
            top_source, top_path, top_structure = top_path, reduced_top_path, reduced_top
            _notify(
                progress_callback,
                "read",
                f"reduced the top layer to its primitive in-plane cell "
                f"(1 of {top_layer_index}, {top_structure.natoms} atoms)",
            )
        if bottom_layer_index > 1:
            bottom_source, bottom_path, bottom_structure = (
                bottom_path,
                reduced_bottom_path,
                reduced_bottom,
            )
            _notify(
                progress_callback,
                "read",
                f"reduced the bottom layer to its primitive in-plane cell "
                f"(1 of {bottom_layer_index}, {bottom_structure.natoms} atoms)",
            )
    top_basis = _planar_column_basis(top_structure.lattice, "top")
    bottom_basis = _planar_column_basis(bottom_structure.lattice, "bottom")
    top_group = _layer_group(top_structure, "top", symmetry_tolerance)
    bottom_group = _layer_group(bottom_structure, "bottom", symmetry_tolerance)
    # Detection is tolerant, so idealise each layer onto the metric its own group
    # preserves exactly before the engine relies on those operations.
    top_basis, top_idealisation = symmetrised_basis(top_basis, top_group, name="top")
    bottom_basis, bottom_idealisation = symmetrised_basis(
        bottom_basis, bottom_group, name="bottom"
    )
    timings["read_structures_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "read",
        f"read structures in {timings['read_structures_s']:.3f}s",
    )

    requested_config = gram.SearchConfig(
        top_basis=top_basis,
        bottom_basis=bottom_basis,
        max_length=float(max_length),
        top_strain=float(top_strain),
        bottom_strain=float(bottom_strain),
        min_length=None if min_length is None else float(min_length),
        max_atoms=None if max_atoms is None else int(max_atoms),
        top_atoms=top_structure.natoms,
        bottom_atoms=bottom_structure.natoms,
        max_aspect_ratio=float(max_aspect_ratio),
        min_cell_angle_deg=float(min_cell_angle_deg),
        max_cell_angle_deg=float(max_cell_angle_deg),
        min_twist_angle_deg=None if min_twist_angle_deg is None else float(min_twist_angle_deg),
        max_twist_angle_deg=None if max_twist_angle_deg is None else float(max_twist_angle_deg),
        fold_symmetry=bool(fold_symmetry),
        symmetric=bool(symmetric),
        primitive_only=bool(primitive_only),
        top_group=top_group,
        bottom_group=bottom_group,
    )

    _notify(progress_callback, "search", "searching native Gram-form candidates")
    stage_start = time.perf_counter()
    symmetric_used = False
    symmetric_fallback: str | None = None
    completed_config = requested_config
    try:
        result = gram.search(requested_config)
        symmetric_used = bool(symmetric)
    except gram.SymmetricBranchUnavailable as exc:
        if not symmetric:
            raise
        symmetric_fallback = str(exc)
        completed_config = replace(requested_config, symmetric=False)
        result = gram.search(completed_config)
    timings["search_s"] = time.perf_counter() - stage_start
    _notify(
        progress_callback,
        "search",
        f"found {len(result)} candidate(s) in {timings['search_s']:.3f}s",
    )

    _notify(progress_callback, "write", "writing results.json")
    stage_start = time.perf_counter()
    document = build_results_document(
        top_poscar=top_path,
        bottom_poscar=bottom_path,
        config=completed_config,
        result=result,
        symmetric_requested=bool(symmetric),
        symmetric_used=symmetric_used,
        symmetric_fallback=symmetric_fallback,
        symmetry_tolerance=float(symmetry_tolerance),
        top_idealisation=float(top_idealisation),
        bottom_idealisation=float(bottom_idealisation),
        max_candidates=None if max_candidates is None else int(max_candidates),
        top_layer_index=top_layer_index,
        bottom_layer_index=bottom_layer_index,
        top_poscar_source=top_source,
        bottom_poscar_source=bottom_source,
    )
    write_results(result_path, document)
    timings["write_results_s"] = time.perf_counter() - stage_start
    timings["total_s"] = time.perf_counter() - total_start
    recorded = len(document["candidates"])
    _notify(
        progress_callback,
        "write",
        f"wrote {recorded} of {len(result)} candidate(s) to results.json in "
        f"{timings['write_results_s']:.3f}s",
    )

    return FindRun(
        run_id=run_id,
        result_path=result_path.resolve(),
        result=result,
        candidates=list(document["candidates"]),
        parameters=dict(document["search"]),
        timings=timings,
        layer_index={"top": int(top_layer_index), "bottom": int(bottom_layer_index)},
    )


def find(**kwargs):
    return run_find(**kwargs)

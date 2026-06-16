"""Finder backend for commensurate moire supercells."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Mapping, Sequence

import numpy as np

from . import lattice as lat
from . import commensurate as com

ANGLE_OUTPUT_TOLERANCE_DEG = 5e-4
STRAIN_OUTPUT_TOLERANCE = 1e-4


def _build_angle_list(
    angle_lower: float | None,
    angle_upper: float | None,
    angle_step: float,
    angles: Sequence[float] | None,
) -> List[float]:
    if angles:
        return sorted({float(angle) for angle in angles})
    if angle_lower is None or angle_upper is None:
        raise ValueError("must provide either an explicit angle list or numeric angle bounds")
    if angle_step <= 0.0:
        raise ValueError("angle_step must be positive")
    values = np.arange(angle_lower, angle_upper + angle_step * 0.5, angle_step, dtype=float)
    return [float(value) for value in values]


def _limit_worker_threads() -> None:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def _matrix_signature(values: Sequence[int], match_mode: str) -> tuple[int, int, int, int]:
    entries = [int(value) for value in values]
    if len(entries) != 4:
        raise ValueError("matrix filters require exactly four integer values")
    if match_mode == "absolute":
        entries = [abs(value) for value in entries]
    elif match_mode != "exact":
        raise ValueError("matrix_match_mode must be 'absolute' or 'exact'")
    return tuple(sorted(entries))


def _candidate_matrix_entries(candidate: lat.SupercellCandidate, layer: str) -> tuple[int, int, int, int]:
    if layer == "1":
        return (
            int(candidate.layer1_vector1[0]),
            int(candidate.layer1_vector1[1]),
            int(candidate.layer1_vector2[0]),
            int(candidate.layer1_vector2[1]),
        )
    if layer == "2":
        return (
            int(candidate.layer2_vector1[0]),
            int(candidate.layer2_vector1[1]),
            int(candidate.layer2_vector2[0]),
            int(candidate.layer2_vector2[1]),
        )
    raise ValueError("layer must be '1' or '2'")


def candidate_matches_matrix_values(
    candidate: lat.SupercellCandidate,
    matrix_values: Sequence[int],
    *,
    matrix_layer: str = "either",
    matrix_match_mode: str = "absolute",
) -> bool:
    target_signature = _matrix_signature(matrix_values, matrix_match_mode)
    layer1_signature = _matrix_signature(_candidate_matrix_entries(candidate, "1"), matrix_match_mode)
    layer2_signature = _matrix_signature(_candidate_matrix_entries(candidate, "2"), matrix_match_mode)

    if matrix_layer == "1":
        return layer1_signature == target_signature
    if matrix_layer == "2":
        return layer2_signature == target_signature
    if matrix_layer == "either":
        return layer1_signature == target_signature or layer2_signature == target_signature
    if matrix_layer == "both":
        return layer1_signature == target_signature and layer2_signature == target_signature
    raise ValueError("matrix_layer must be '1', '2', 'either', or 'both'")


def _coefficient_signature(candidate: lat.SupercellCandidate) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        int(candidate.layer1_vector1[0]),
        int(candidate.layer1_vector1[1]),
        int(candidate.layer1_vector2[0]),
        int(candidate.layer1_vector2[1]),
        int(candidate.layer2_vector1[0]),
        int(candidate.layer2_vector1[1]),
        int(candidate.layer2_vector2[0]),
        int(candidate.layer2_vector2[1]),
    )


def _coefficient_choice_key(candidate: lat.SupercellCandidate) -> tuple[float, int, float, float, float]:
    return (
        float(candidate.strain_avg),
        int(candidate.total_atoms),
        abs(float(candidate.eps1)) + abs(float(candidate.eps2)),
        float(candidate.vector_product),
        float(candidate.angle_deg),
    )


def _output_sort_key(candidate: lat.SupercellCandidate) -> tuple[float, float, int, float, tuple[int, int, int, int, int, int, int, int]]:
    return (
        float(candidate.angle_deg),
        float(candidate.strain_avg),
        int(candidate.total_atoms),
        float(candidate.vector_product),
        _coefficient_signature(candidate),
    )


def _quantized_value(value: float, tolerance: float) -> int:
    return int(round(float(value) / max(float(tolerance), 1e-300)))


def _physical_precision_signature(
    candidate: lat.SupercellCandidate,
    *,
    angle_tolerance: float,
    strain_tolerance: float,
) -> tuple[int, int, int]:
    return (
        _quantized_value(float(candidate.angle_deg), angle_tolerance),
        _quantized_value(float(candidate.strain_layer1), strain_tolerance),
        _quantized_value(float(candidate.strain_layer2), strain_tolerance),
    )


def _physical_precision_choice_key(candidate: lat.SupercellCandidate) -> tuple[int, float, float, float, float]:
    return (
        int(candidate.total_atoms),
        float(candidate.strain_avg),
        abs(float(candidate.eps1)) + abs(float(candidate.eps2)),
        float(candidate.vector_product),
        float(candidate.angle_deg),
    )


def finalize_candidates(
    candidates: Sequence[lat.SupercellCandidate],
    *,
    dedupe_exact_coefficients: bool = True,
    angle_tolerance: float = ANGLE_OUTPUT_TOLERANCE_DEG,
    strain_tolerance: float = STRAIN_OUTPUT_TOLERANCE,
) -> List[lat.SupercellCandidate]:
    """Return final output candidates with duplicate-looking rows collapsed.

    Expensive search stages dedupe before optional basis reduction. Reduction can
    make two survivors share the exact same final integer matrices while keeping
    different raw angle labels. Very dense searches can also produce rows that
    differ only below useful angle/strain precision. This final pass is cheap:
    it scans already-surviving Python candidates, keeps the most compact
    representative per final output signature, and sorts rows by increasing
    angle.
    """

    rows = list(candidates)
    if dedupe_exact_coefficients:
        best_by_signature: dict[
            tuple[int, int, int, int, int, int, int, int],
            tuple[tuple[float, int, float, float, float], lat.SupercellCandidate],
        ] = {}
        for candidate in rows:
            signature = _coefficient_signature(candidate)
            candidate_key = _coefficient_choice_key(candidate)
            current = best_by_signature.get(signature)
            if current is None or candidate_key < current[0]:
                best_by_signature[signature] = (candidate_key, candidate)
        rows = [candidate for _, candidate in best_by_signature.values()]
    if rows:
        best_by_physical_signature: dict[
            tuple[int, int, int],
            tuple[tuple[int, float, float, float, float], lat.SupercellCandidate],
        ] = {}
        for candidate in rows:
            signature = _physical_precision_signature(
                candidate,
                angle_tolerance=angle_tolerance,
                strain_tolerance=strain_tolerance,
            )
            candidate_key = _physical_precision_choice_key(candidate)
            current = best_by_physical_signature.get(signature)
            if current is None or candidate_key < current[0]:
                best_by_physical_signature[signature] = (candidate_key, candidate)
        rows = [candidate for _, candidate in best_by_physical_signature.values()]
    rows.sort(key=_output_sort_key)
    return rows


def _search_angle_chunk(
    task: tuple,
) -> List[lat.SupercellCandidate]:
    """Search a chunk of twist angles in one process.

    Each chunk rebuilds the (small) precomputed pair set once and then handles
    every angle in the chunk by slicing the angle-sorted pairs -- this avoids
    pickling the large precomputed arrays once per angle, which is what made the
    old per-angle process pool prohibitive for large angle lists.
    """
    (
        lattice1,
        lattice2,
        nindex1,
        nindex2,
        match_tol,
        vector_strain_tol,
        candidate_tol,
        atom_count1,
        atom_count2,
        max_pair_matches,
        min_atoms,
        max_atoms,
        canonicalize,
        frontier_only,
        unique_strain_tol,
        unique_ratio_tol,
        max_cell_aspect_ratio,
        min_cell_angle_deg,
        max_cell_angle_deg,
        window,
        angle_chunk,
    ) = task
    pre = com.precompute_pairs(lattice1, lattice2, nindex1, nindex2, match_tol, vector_strain_tol)
    results: List[lat.SupercellCandidate] = []
    for angle_deg in angle_chunk:
        matches = com.matches_at_angle(pre, float(angle_deg), match_tol, window, max_pair_matches)
        if len(matches) < 2:
            continue
        rotated = lat.rotate_lattice(lattice1, float(angle_deg))
        results.extend(
            lat.build_supercell_candidates(
                matches,
                rotated,
                lattice2,
                atom_count1,
                atom_count2,
                candidate_tol,
                float(angle_deg),
                # ``matches_at_angle`` has already bounded the match count (with
                # exact coincidences prioritised); do not let the builder re-cap
                # purely by length, which would drop the long exact vectors that
                # form low-strain primitive cells.
                max_pair_matches=None,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                canonicalize=canonicalize,
                frontier_only=frontier_only,
                unique_strain_tol=unique_strain_tol,
                unique_ratio_tol=unique_ratio_tol,
                max_cell_aspect_ratio=max_cell_aspect_ratio,
                min_cell_angle_deg=min_cell_angle_deg,
                max_cell_angle_deg=max_cell_angle_deg,
            )
        )
    return results


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
    matrix_values: Sequence[int] | None = None,
    matrix_layer: str = "either",
    matrix_match_mode: str = "absolute",
    workers: int = 1,
    max_pair_matches: int | None = None,
    fold_symmetry: bool = False,
    cull_redundant: bool = True,
    reduce_basis: bool = True,
    frontier_cull: bool | None = None,
    max_cell_aspect_ratio: float | None = 12.0,
    min_cell_angle_deg: float | None = 25.0,
    max_cell_angle_deg: float | None = 155.0,
) -> List[lat.SupercellCandidate]:
    # Enforce the physical strain ceiling: never accept a length mismatch beyond
    # MAX_PHYSICAL_MISMATCH (5%), regardless of what the caller requested.
    if strain_tol is None:
        strain_tol = lat.MAX_PHYSICAL_MISMATCH
    else:
        strain_tol = min(float(strain_tol), lat.MAX_PHYSICAL_MISMATCH)
    if vector_strain_tol is not None:
        vector_strain_tol = min(float(vector_strain_tol), lat.MAX_PHYSICAL_MISMATCH)
    else:
        vector_strain_tol = lat.MAX_PHYSICAL_MISMATCH
    candidate_tolerance = lin_tol if lin_tol is not None else tol
    angle_values = _build_angle_list(angle_lower, angle_upper, angle_step, angles)
    lattice1_array = np.asarray(lattice1, dtype=float)
    lattice2_array = np.asarray(lattice2, dtype=float)

    # Optional crystal-symmetry fold: when both lattices share a mirror line, the
    # twist angle theta and (sym - theta) give mirror-equivalent supercells, so we
    # only search the irreducible wedge [0, sym/2] and reflect the index matrices
    # to recover the rest (the "search to 30 deg then use signs" trick).  This
    # roughly halves the per-angle work for symmetric systems such as MoS2/MoS2.
    fold_operation: tuple[np.ndarray, int] | None = None
    if fold_symmetry:
        fold_operation = lat.fold_symmetry_operation(lattice1_array, lattice2_array)
        if fold_operation is not None:
            _, fold_sym = fold_operation
            wedge_max = fold_sym / 2.0
            angle_values = [angle for angle in angle_values if angle <= wedge_max + 1e-9]

    # Dynamically scale nindex based on cell-size limits.  A layer-1 vector longer
    # than the longest reachable layer-2 vector can never match, so the per-layer
    # integer range can be capped without dropping any admissible pair.
    basis1_2d = lattice1_array[:2, :2]
    basis2_2d = lattice2_array[:2, :2]
    basis1_lengths = np.linalg.norm(basis1_2d, axis=1)
    basis2_lengths = np.linalg.norm(basis2_2d, axis=1)

    L1_max = nindex * np.max(basis1_lengths)
    L2_max = nindex * np.max(basis2_lengths)
    L_cutoff = min(L1_max, L2_max) * (1.0 + float(tol))

    min_basis1 = np.min(basis1_lengths)
    min_basis2 = np.min(basis2_lengths)

    scaled_nindex1 = int(min(nindex, np.ceil(L_cutoff / max(min_basis1, 1e-12))))
    scaled_nindex2 = int(min(nindex, np.ceil(L_cutoff / max(min_basis2, 1e-12))))

    # Twist angle is an analytic function of each equal-length integer vector
    # pair, so the search no longer sweeps angles: it tags every pair with its
    # angle once and, for each requested angle, slices the narrow angular window
    # around it.  ``match_tol`` is the widest relative error any consumer needs,
    # and ``window`` is the matching angular half-width.
    match_tol = max(float(tol), float(candidate_tolerance))
    window = com.angle_window_deg(match_tol, 1e-3)
    canonicalize = bool(dedupe and matrix_values is None)
    # When the per-angle redundancy cull is on (and we are not matrix-filtering,
    # which needs the full candidate set), collapse each angle to its
    # (atoms, strain) Pareto frontier inside the builder -- in numpy, before any
    # Python candidate objects are created.  This is the dominating speed-up at
    # large nindex: it removes the canonical-sublattice grouping and the
    # O(candidates^2) global deduplication, both of which exploded at degenerate
    # angles, while producing exactly the cells the final cull would keep.
    if frontier_cull is None:
        frontier_only = bool(cull_redundant and matrix_values is None)
    else:
        frontier_only = bool(frontier_cull and matrix_values is None)

    def _task(chunk: Sequence[float]) -> tuple:
        return (
            lattice1_array,
            lattice2_array,
            scaled_nindex1,
            scaled_nindex2,
            match_tol,
            vector_strain_tol,
            float(candidate_tolerance),
            int(atom_count1),
            int(atom_count2),
            max_pair_matches,
            min_atoms,
            max_atoms,
            canonicalize,
            frontier_only,
            float(unique_strain_tol),
            float(unique_ratio_tol),
            max_cell_aspect_ratio,
            min_cell_angle_deg,
            max_cell_angle_deg,
            window,
            list(chunk),
        )

    all_candidates: List[lat.SupercellCandidate] = []
    resolved_workers = max(1, int(workers))
    if resolved_workers <= 1 or len(angle_values) <= 1:
        all_candidates.extend(_search_angle_chunk(_task(angle_values)))
    else:
        chunks = [angle_values[index::resolved_workers] for index in range(resolved_workers)]
        chunks = [chunk for chunk in chunks if chunk]
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers, initializer=_limit_worker_threads) as executor:
                for chunk_result in executor.map(_search_angle_chunk, [_task(chunk) for chunk in chunks]):
                    all_candidates.extend(chunk_result)
        except (OSError, PermissionError):
            all_candidates.extend(_search_angle_chunk(_task(angle_values)))

    # Reconstruct the mirror partners of the wedge candidates to cover the full
    # angle range.  Boundary angles (0 and sym/2) are self-mapped, so only the
    # strict interior of the wedge is reflected; deduplication later removes any
    # incidental overlaps.
    if fold_operation is not None:
        fold_op, fold_sym = fold_operation
        wedge_max = fold_sym / 2.0
        mirrored_candidates: List[lat.SupercellCandidate] = []
        for candidate in all_candidates:
            if 1e-9 < candidate.angle_deg < wedge_max - 1e-9:
                mirror = lat.mirror_supercell_candidate(
                    candidate, fold_op, fold_sym, lattice1_array, lattice2_array
                )
                if mirror is not None:
                    mirrored_candidates.append(mirror)
        all_candidates.extend(mirrored_candidates)

    resolved_strain_layer = str(strain_layer).lower()
    if resolved_strain_layer not in {"avg", "1", "2"}:
        raise ValueError("strain_layer must be 'avg', '1', or '2'")

    filtered: List[lat.SupercellCandidate] = []
    for candidate in all_candidates:
        if min_atoms is not None and candidate.total_atoms < min_atoms:
            continue
        if max_atoms is not None and candidate.total_atoms > max_atoms:
            continue
        if strain_tol is not None:
            if resolved_strain_layer == "avg" and candidate.strain_avg > strain_tol:
                continue
            if resolved_strain_layer == "1" and candidate.strain_layer1 > strain_tol:
                continue
            if resolved_strain_layer == "2" and candidate.strain_layer2 > strain_tol:
                continue
        if matrix_values is not None and not candidate_matches_matrix_values(
            candidate,
            matrix_values,
            matrix_layer=matrix_layer,
            matrix_match_mode=matrix_match_mode,
        ):
            continue
        filtered.append(candidate)

    if dedupe:
        filtered = lat.deduplicate_candidates(
            filtered,
            strain_tolerance=unique_strain_tol,
            ratio_tolerance=unique_ratio_tol,
        )

    # Cull the per-angle redundancy: at any twist angle keep only the cells on
    # the (atom count, strain) Pareto frontier, dropping every cell that another
    # at the same angle beats on both size and strain.
    if cull_redundant:
        filtered = com.pareto_cull(filtered)

    # Report each cell in its shortest / most orthogonal integer basis.  Only the
    # handful of surviving (culled) candidates are reduced here, so the cost is
    # negligible; the strain was already measured on a well-conditioned basis.
    if reduce_basis and matrix_values is None:
        reduced_filtered: List[lat.SupercellCandidate] = []
        for candidate in filtered:
            reduced = com.reduce_candidate_checked(
                candidate,
                lattice1_array,
                lattice2_array,
                candidate.angle_deg,
                max(float(candidate_tolerance), float(tol)),
            )
            if reduced is not None:
                reduced_filtered.append(reduced)
        filtered = reduced_filtered

    return finalize_candidates(filtered, dedupe_exact_coefficients=bool(dedupe))



def candidate_to_dict(candidate: lat.SupercellCandidate, index: int | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "angle_deg": float(candidate.angle_deg),
        "strain_avg": float(candidate.strain_avg),
        "strain_layer1": float(candidate.strain_layer1),
        "strain_layer2": float(candidate.strain_layer2),
        "ratio1": int(candidate.ratio1),
        "ratio2": int(candidate.ratio2),
        "total_atoms": int(candidate.total_atoms),
        "layer1_vector1": [int(candidate.layer1_vector1[0]), int(candidate.layer1_vector1[1])],
        "layer1_vector2": [int(candidate.layer1_vector2[0]), int(candidate.layer1_vector2[1])],
        "layer2_vector1": [int(candidate.layer2_vector1[0]), int(candidate.layer2_vector1[1])],
        "layer2_vector2": [int(candidate.layer2_vector2[0]), int(candidate.layer2_vector2[1])],
        "eps1": float(candidate.eps1),
        "eps2": float(candidate.eps2),
        "vector_product": float(candidate.vector_product),
        "area1": float(candidate.area1),
        "area2": float(candidate.area2),
        "cell_aspect_ratio": float(candidate.cell_aspect_ratio),
        "cell_angle_deg": float(candidate.cell_angle_deg),
    }
    if index is not None:
        payload["index"] = int(index)
    return payload


def format_results_table(candidates: Sequence[lat.SupercellCandidate], limit: int | None = None) -> str:
    shown = list(candidates if limit is None or limit < 0 else candidates[: max(limit, 0)])
    if not shown:
        return "No candidates found."

    header = (
        " idx  angle(deg)   strain_avg    strain_1      strain_2    atoms   ratio"
        "      i1          i2          j1          j2       aspect   minang      eps1        eps2"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for index, candidate in enumerate(shown, start=1):
        lines.append(
            "{idx:4d}  {angle:10.4f}  {strain_avg:11.6f}  {strain1:11.6f}  {strain2:11.6f}  {atoms:7d}  "
            "{ratio1:3d}/{ratio2:<3d}  ({i11:3d},{i12:3d})  ({i21:3d},{i22:3d})  "
            "({j11:3d},{j12:3d})  ({j21:3d},{j22:3d})  {aspect:7.2f}  {minang:7.2f}  {eps1:10.2e}  {eps2:10.2e}".format(
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
                aspect=candidate.cell_aspect_ratio,
                minang=candidate.cell_angle_deg,
                eps1=candidate.eps1,
                eps2=candidate.eps2,
            )
        )
    return "\n".join(lines)


def write_results_dat(
    path: str,
    pos1: str,
    pos2: str,
    candidates: Sequence[lat.SupercellCandidate],
    *,
    run_id: str,
    parameters: Mapping[str, object],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{pos1} {pos2}\n")
        handle.write(f"# run_id = {run_id}\n")
        handle.write(f"# created_at = {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write("# credit = CELLSTINE (CELL Superlattice Transformation INterface and Engine) | Made by Sarwin Chandran\n")
        handle.write("# units = angles in degrees; strain and mismatch values are fractions (0.01 = 1%)\n")
        handle.write("# note = strain_avg is the symmetric strain measure; strain1 and strain2 are the one-sided layer strain measures\n")
        handle.write(
            "| idx | angle (deg) | strain_avg | strain1 | strain2 | atoms | ratio | i11 i12 | i21 i22 | j11 j12 | j21 j22 | aspect | min_angle | eps1 | eps2 |\n"
        )
        handle.write("-" * 148 + "\n")
        for index, candidate in enumerate(candidates, start=1):
            i11, i12 = candidate.layer1_vector1
            i21, i22 = candidate.layer1_vector2
            j11, j12 = candidate.layer2_vector1
            j21, j22 = candidate.layer2_vector2
            handle.write(
                "|{idx:4d} | {angle:10.4f} | {strain_avg:10.6f} | {strain1:7.6f} | {strain2:7.6f} | {atoms:5d} | {ratio1:3d}/{ratio2:<3d} | {i11:4d} {i12:4d} | {i21:4d} {i22:4d} | {j11:4d} {j12:4d} | {j21:4d} {j22:4d} | {aspect:6.2f} | {minang:9.3f} | {eps1:8.2e} | {eps2:8.2e} |\n".format(
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
                    aspect=candidate.cell_aspect_ratio,
                    minang=candidate.cell_angle_deg,
                    eps1=candidate.eps1,
                    eps2=candidate.eps2,
                )
            )
        handle.write("\n")
        handle.write("# parameters\n")
        for key, value in parameters.items():
            handle.write(f"# {key} = {value}\n")

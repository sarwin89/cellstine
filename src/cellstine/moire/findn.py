"""Reference-layer N-layer commensuration search stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import angles as angle_backend
from . import find as find_stage
from . import finder as finder_backend
from . import lattice as lat

ANGLE_OUTPUT_TOLERANCE_DEG = 5e-4
STRAIN_OUTPUT_TOLERANCE = 1e-4


@dataclass(frozen=True)
class UpperLayerCandidate:
    layer_index: int
    angle_deg: float
    strain: float
    ratio_upper: int
    vector1: tuple[int, int]
    vector2: tuple[int, int]


@dataclass(frozen=True)
class NLayerCandidate:
    strain_max: float
    strain_mean: float
    ratio_bottom: int
    total_atoms: int
    bottom_vector1: tuple[int, int]
    bottom_vector2: tuple[int, int]
    upper_layers: tuple[UpperLayerCandidate, ...]


@dataclass
class FindNRun:
    run_id: str
    result_path: Path
    candidates: List[NLayerCandidate]
    shortlisted_angles_by_layer: List[List[angle_backend.AngleCandidate]]
    angle_values_by_layer: List[List[float]]
    parameters: Dict[str, object]


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _make_result_path(output_root: str, bottom_path: str, upper_paths: Sequence[str], nindex: int) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upper_slug = "__".join(_slug(Path(path).stem) for path in upper_paths)
    run_id = f"{timestamp}_{_slug(Path(bottom_path).stem)}_bottom__{upper_slug}_n{nindex}"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir / f"{run_id}.json"


def _bottom_matrix_key(candidate: lat.SupercellCandidate) -> tuple[int, int, int, int]:
    return (
        int(candidate.layer2_vector1[0]),
        int(candidate.layer2_vector1[1]),
        int(candidate.layer2_vector2[0]),
        int(candidate.layer2_vector2[1]),
    )


def _upper_matrix_key(candidate: lat.SupercellCandidate) -> tuple[int, int, int, int]:
    return (
        int(candidate.layer1_vector1[0]),
        int(candidate.layer1_vector1[1]),
        int(candidate.layer1_vector2[0]),
        int(candidate.layer1_vector2[1]),
    )


def _pair_choice_key(candidate: lat.SupercellCandidate) -> tuple[float, int, float, float, float]:
    return (
        float(candidate.strain_avg),
        int(candidate.total_atoms),
        abs(float(candidate.eps1)) + abs(float(candidate.eps2)),
        float(candidate.vector_product),
        float(candidate.angle_deg),
    )


def _group_by_bottom_matrix(candidates: Sequence[lat.SupercellCandidate]) -> dict[tuple[int, int, int, int], list[lat.SupercellCandidate]]:
    grouped_best: dict[tuple[int, int, int, int], dict[tuple[int, int, int, int], tuple[tuple[float, int, float, float, float], lat.SupercellCandidate]]] = {}
    for candidate in candidates:
        bottom_key = _bottom_matrix_key(candidate)
        upper_key = _upper_matrix_key(candidate)
        candidate_key = _pair_choice_key(candidate)
        current = grouped_best.setdefault(bottom_key, {}).get(upper_key)
        if current is None or candidate_key < current[0]:
            grouped_best[bottom_key][upper_key] = (candidate_key, candidate)
    return {
        bottom_key: sorted(
            (candidate for _, candidate in best_by_upper.values()),
            key=lambda item: (float(item.angle_deg), float(item.strain_avg), int(item.total_atoms), _upper_matrix_key(item)),
        )
        for bottom_key, best_by_upper in grouped_best.items()
    }


def _nlayer_signature(candidate: NLayerCandidate) -> tuple[
    tuple[int, int, int, int],
    tuple[tuple[int, int, int, int], ...],
]:
    return (
        (
            int(candidate.bottom_vector1[0]),
            int(candidate.bottom_vector1[1]),
            int(candidate.bottom_vector2[0]),
            int(candidate.bottom_vector2[1]),
        ),
        tuple(
            (
                int(layer.vector1[0]),
                int(layer.vector1[1]),
                int(layer.vector2[0]),
                int(layer.vector2[1]),
            )
            for layer in candidate.upper_layers
        ),
    )


def _nlayer_angle_key(candidate: NLayerCandidate) -> tuple[float, ...]:
    return tuple(float(layer.angle_deg) for layer in candidate.upper_layers)


def _quantized_value(value: float, tolerance: float) -> int:
    return int(round(float(value) / max(float(tolerance), 1e-300)))


def _nlayer_precision_signature(candidate: NLayerCandidate) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            _quantized_value(float(layer.angle_deg), ANGLE_OUTPUT_TOLERANCE_DEG),
            _quantized_value(float(layer.strain), STRAIN_OUTPUT_TOLERANCE),
        )
        for layer in candidate.upper_layers
    )


def finalize_nlayer_candidates(candidates: Sequence[NLayerCandidate]) -> List[NLayerCandidate]:
    best_by_signature: dict[
        tuple[tuple[int, int, int, int], tuple[tuple[int, int, int, int], ...]],
        tuple[tuple[float, float, int, tuple[float, ...]], NLayerCandidate],
    ] = {}
    for candidate in candidates:
        signature = _nlayer_signature(candidate)
        candidate_key = (
            float(candidate.strain_max),
            float(candidate.strain_mean),
            int(candidate.total_atoms),
            _nlayer_angle_key(candidate),
        )
        current = best_by_signature.get(signature)
        if current is None or candidate_key < current[0]:
            best_by_signature[signature] = (candidate_key, candidate)
    rows = [candidate for _, candidate in best_by_signature.values()]
    best_by_precision: dict[
        tuple[tuple[int, int], ...],
        tuple[tuple[int, float, float, tuple[float, ...]], NLayerCandidate],
    ] = {}
    for candidate in rows:
        signature = _nlayer_precision_signature(candidate)
        candidate_key = (
            int(candidate.total_atoms),
            float(candidate.strain_max),
            float(candidate.strain_mean),
            _nlayer_angle_key(candidate),
        )
        current = best_by_precision.get(signature)
        if current is None or candidate_key < current[0]:
            best_by_precision[signature] = (candidate_key, candidate)
    rows = [candidate for _, candidate in best_by_precision.values()]
    rows.sort(
        key=lambda item: (
            _nlayer_angle_key(item),
            float(item.strain_max),
            float(item.strain_mean),
            int(item.total_atoms),
            _nlayer_signature(item),
        )
    )
    return rows


def _join_pair_candidate_sets(
    candidate_sets: Sequence[Sequence[lat.SupercellCandidate]],
    *,
    bottom_atoms: int,
    upper_atoms: Sequence[int],
    max_atoms: int | None,
) -> List[NLayerCandidate]:
    if not candidate_sets:
        return []

    grouped_sets = [_group_by_bottom_matrix(candidate_set) for candidate_set in candidate_sets]
    common_keys = set(grouped_sets[0].keys())
    for grouped in grouped_sets[1:]:
        common_keys &= set(grouped.keys())

    joined: List[NLayerCandidate] = []
    seen: set[tuple[tuple[int, int, int, int], tuple[tuple[float, int, int, int, int, int], ...]]] = set()
    for bottom_key in sorted(common_keys):
        option_lists = [grouped[bottom_key] for grouped in grouped_sets]
        for combination in product(*option_lists):
            ratio_bottom_values = {int(candidate.ratio2) for candidate in combination}
            if len(ratio_bottom_values) != 1:
                continue
            ratio_bottom = ratio_bottom_values.pop()

            signature = (
                bottom_key,
                tuple(
                    (
                        round(float(candidate.angle_deg), 8),
                        int(candidate.ratio1),
                        int(candidate.layer1_vector1[0]),
                        int(candidate.layer1_vector1[1]),
                        int(candidate.layer1_vector2[0]),
                        int(candidate.layer1_vector2[1]),
                    )
                    for candidate in combination
                ),
            )
            if signature in seen:
                continue
            seen.add(signature)

            total_atoms = int(bottom_atoms) * int(ratio_bottom)
            for atom_count, candidate in zip(upper_atoms, combination):
                total_atoms += int(atom_count) * int(candidate.ratio1)
            if max_atoms is not None and total_atoms > int(max_atoms):
                continue

            strains = [float(candidate.strain_avg) for candidate in combination]
            upper_layers = tuple(
                UpperLayerCandidate(
                    layer_index=index + 1,
                    angle_deg=float(candidate.angle_deg),
                    strain=float(candidate.strain_avg),
                    ratio_upper=int(candidate.ratio1),
                    vector1=tuple(int(value) for value in candidate.layer1_vector1),
                    vector2=tuple(int(value) for value in candidate.layer1_vector2),
                )
                for index, candidate in enumerate(combination)
            )
            joined.append(
                NLayerCandidate(
                    strain_max=max(strains),
                    strain_mean=sum(strains) / float(len(strains)),
                    ratio_bottom=int(ratio_bottom),
                    total_atoms=int(total_atoms),
                    bottom_vector1=(int(bottom_key[0]), int(bottom_key[1])),
                    bottom_vector2=(int(bottom_key[2]), int(bottom_key[3])),
                    upper_layers=upper_layers,
                )
            )
    return finalize_nlayer_candidates(joined)


def candidate_to_dict(candidate: NLayerCandidate, index: int | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "strain_max": float(candidate.strain_max),
        "strain_mean": float(candidate.strain_mean),
        "ratio_bottom": int(candidate.ratio_bottom),
        "total_atoms": int(candidate.total_atoms),
        "bottom_vector1": [int(candidate.bottom_vector1[0]), int(candidate.bottom_vector1[1])],
        "bottom_vector2": [int(candidate.bottom_vector2[0]), int(candidate.bottom_vector2[1])],
        "upper_layers": [
            {
                "layer_index": int(layer.layer_index),
                "angle_deg": float(layer.angle_deg),
                "strain": float(layer.strain),
                "ratio_upper": int(layer.ratio_upper),
                "vector1": [int(layer.vector1[0]), int(layer.vector1[1])],
                "vector2": [int(layer.vector2[0]), int(layer.vector2[1])],
            }
            for layer in candidate.upper_layers
        ],
    }
    if index is not None:
        payload["index"] = int(index)
    return payload


def write_results_json(
    path: str,
    *,
    meta: Dict[str, object],
    candidates: Sequence[NLayerCandidate],
) -> None:
    payload = {
        "meta": dict(meta),
        "candidates": [candidate_to_dict(candidate, index + 1) for index, candidate in enumerate(candidates)],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_results(path: str) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = dict(payload.get("meta", {}))
    candidates = list(payload.get("candidates", []))
    if not meta:
        raise ValueError("findn results do not contain metadata")
    return meta, candidates


def run_findn(
    *,
    bottom_poscar: str,
    upper_poscars: Sequence[str],
    bottom_lattice: np.ndarray,
    upper_lattices: Sequence[np.ndarray],
    bottom_atoms: int,
    upper_atoms: Sequence[int],
    nindex: int,
    min_angles: Sequence[float],
    max_angles: Sequence[float],
    angle_step: float = 0.1,
    explicit_angles_by_layer: Sequence[Sequence[float] | None] | None = None,
    angle_length_tolerance: float = 1e-5,
    angle_strain_tolerance: float | None = 2e-3,
    angle_merge_tolerance: float = 1e-3,
    vector_tolerance: float = 2e-3,
    vector_strain_tolerance: float | None = 2e-3,
    candidate_tolerance: float | None = None,
    pair_strain_tolerance: float | None = None,
    max_atoms: int | None = 2000,
    dedupe: bool = True,
    unique_strain_tolerance: float = 1e-4,
    unique_ratio_tolerance: float = 1e-5,
    output_root: str = "runs",
    bottom_c_repeat: int = 1,
    upper_c_repeats: Sequence[int] | None = None,
    workers: int = 1,
    fold_symmetry: bool = False,
    max_search_angles: int | None = None,
    max_cell_aspect_ratio: float | None = 12.0,
    min_cell_angle_deg: float | None = 25.0,
    max_cell_angle_deg: float | None = 155.0,
) -> FindNRun:
    if len(upper_poscars) != len(upper_lattices) or len(upper_poscars) != len(upper_atoms):
        raise ValueError("upper layer paths, lattices, and atom counts must have the same length")
    if not upper_poscars:
        raise ValueError("findn needs at least one upper layer")
    if explicit_angles_by_layer is None:
        explicit_angles_by_layer = [None] * len(upper_poscars)
    if upper_c_repeats is None:
        upper_c_repeats = [1] * len(upper_poscars)

    shortlisted_angles_by_layer: List[List[angle_backend.AngleCandidate]] = []
    angle_values_by_layer: List[List[float]] = []
    pairwise_candidates: List[List[lat.SupercellCandidate]] = []
    search_windows: List[Dict[str, object]] = []

    for layer_index, (upper_lattice, min_angle, max_angle, explicit_angles) in enumerate(
        zip(upper_lattices, min_angles, max_angles, explicit_angles_by_layer),
        start=1,
    ):
        (
            shortlist,
            angle_values,
            symmetry_upper,
            symmetry_bottom,
            symmetry_lcm,
            search_min,
            search_max,
            angle_metadata,
        ) = find_stage._resolve_angles(
            upper_lattice,
            bottom_lattice,
            nindex,
            min_angle=min_angle,
            max_angle=max_angle,
            angle_step=angle_step,
            explicit_angles=explicit_angles,
            angle_length_tolerance=angle_length_tolerance,
            angle_strain_tolerance=angle_strain_tolerance,
            angle_merge_tolerance=angle_merge_tolerance,
            max_search_angles=max_search_angles,
        )
        candidates = finder_backend.find_supercells(
            upper_lattice,
            bottom_lattice,
            None,
            None,
            angle_step=angle_step,
            nindex=nindex,
            tol=vector_tolerance,
            lin_tol=candidate_tolerance,
            strain_tol=pair_strain_tolerance,
            strain_layer="avg",
            max_atoms=None,
            atom_count1=int(upper_atoms[layer_index - 1]),
            atom_count2=int(bottom_atoms),
            angles=angle_values,
            dedupe=dedupe,
            unique_strain_tol=unique_strain_tolerance,
            unique_ratio_tol=unique_ratio_tolerance,
            vector_strain_tol=vector_strain_tolerance,
            workers=workers,
            fold_symmetry=fold_symmetry,
            # Keep the full pairwise candidate set (in raw integer bases) so the
            # cross-layer intersection below has every cell to work with; only
            # bound the per-angle pairing so highly symmetric angles stay fast.
            max_pair_matches=200,
            cull_redundant=False,
            reduce_basis=False,
            max_cell_aspect_ratio=max_cell_aspect_ratio,
            min_cell_angle_deg=min_cell_angle_deg,
            max_cell_angle_deg=max_cell_angle_deg,
        )

        shortlisted_angles_by_layer.append(list(shortlist))
        angle_values_by_layer.append(list(angle_values))
        pairwise_candidates.append(list(candidates))
        search_windows.append(
            {
                "layer_index": int(layer_index),
                "symmetry_upper_deg": int(symmetry_upper),
                "symmetry_bottom_deg": int(symmetry_bottom),
                "symmetry_lcm_deg": int(symmetry_lcm),
                "min_angle_deg": float(search_min),
                "max_angle_deg": float(search_max),
                "angle_count": len(angle_values),
                "angle_values_thinned": bool(angle_metadata.get("angle_values_thinned", False)),
                "angle_values_before_thinning": angle_metadata.get("angle_values_before_thinning", ""),
                "max_search_angles": angle_metadata.get("max_search_angles", max_search_angles),
                "explicit_angles_deg": [] if explicit_angles is None else [float(value) for value in explicit_angles],
                "shortlisted_angles_deg": [float(candidate.angle_deg) for candidate in shortlist],
            }
        )

    candidates = _join_pair_candidate_sets(
        pairwise_candidates,
        bottom_atoms=bottom_atoms,
        upper_atoms=upper_atoms,
        max_atoms=max_atoms,
    )

    run_id, result_path = _make_result_path(output_root, bottom_poscar, upper_poscars, nindex)
    parameters: Dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "credit": "CELLSTINE (CELL Superlattice Transformation INterface and Engine) | Made by Sarwin Chandran",
        "mode": "findn",
        "layer_count": 1 + len(upper_poscars),
        "units_note": "angles in degrees; angle_length_tolerance in angstrom; strain and mismatch values as fractions (0.01 = 1%)",
        "bottom_poscar": str(bottom_poscar),
        "upper_poscars": [str(path) for path in upper_poscars],
        "bottom_c_repeat": int(bottom_c_repeat),
        "upper_c_repeats": [int(value) for value in upper_c_repeats],
        "workers": int(workers),
        "nindex": int(nindex),
        "angle_step_deg": float(angle_step),
        "angle_length_tolerance": float(angle_length_tolerance),
        "angle_strain_tolerance": "" if angle_strain_tolerance is None else float(angle_strain_tolerance),
        "angle_merge_tolerance": float(angle_merge_tolerance),
        "vector_tolerance": float(vector_tolerance),
        "vector_strain_tolerance": "" if vector_strain_tolerance is None else float(vector_strain_tolerance),
        "candidate_tolerance": float(candidate_tolerance if candidate_tolerance is not None else vector_tolerance),
        "pair_strain_tolerance": "" if pair_strain_tolerance is None else float(pair_strain_tolerance),
        "max_atoms": "" if max_atoms is None else int(max_atoms),
        "max_cell_aspect_ratio": "" if max_cell_aspect_ratio is None else float(max_cell_aspect_ratio),
        "min_cell_angle_deg": "" if min_cell_angle_deg is None else float(min_cell_angle_deg),
        "max_cell_angle_deg": "" if max_cell_angle_deg is None else float(max_cell_angle_deg),
        "dedupe": bool(dedupe),
        "unique_strain_tolerance": float(unique_strain_tolerance),
        "unique_ratio_tolerance": float(unique_ratio_tolerance),
        "output_angle_tolerance_deg": float(ANGLE_OUTPUT_TOLERANCE_DEG),
        "output_strain_tolerance": float(STRAIN_OUTPUT_TOLERANCE),
        "layers": search_windows,
    }

    write_results_json(str(result_path), meta=parameters, candidates=candidates)
    return FindNRun(
        run_id=run_id,
        result_path=result_path,
        candidates=list(candidates),
        shortlisted_angles_by_layer=shortlisted_angles_by_layer,
        angle_values_by_layer=angle_values_by_layer,
        parameters=parameters,
    )


def findn(**kwargs):
    return run_findn(**kwargs)

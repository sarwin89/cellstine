"""Versioned JSON persistence for native Gram-form bilayer searches."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gram import SearchConfig, SearchResult

SCHEMA = "cellstine.moire.gram"
VERSION = 1

_SEARCH_FIELDS = {
    "top_poscar",
    "bottom_poscar",
    "max_length",
    "top_strain",
    "bottom_strain",
    "min_length",
    "max_atoms",
    "top_atoms",
    "bottom_atoms",
    "max_aspect_ratio",
    "min_cell_angle_deg",
    "max_cell_angle_deg",
    "fold_symmetry",
    "symmetric",
}
_CANDIDATE_FIELDS = {
    "index",
    "top_matrix",
    "bottom_matrix",
    "top_gram",
    "bottom_gram",
    "angle_deg",
    "strain",
    "top_strain",
    "bottom_strain",
    "sharing_fraction",
    "top_atom_count",
    "bottom_atom_count",
    "atom_count",
    "loewner_certified",
    "loewner_borderline",
    "rank",
    "pareto_optimal",
    "top_affine",
    "bottom_affine",
    "shared_lattice",
}

__all__ = [
    "SCHEMA",
    "VERSION",
    "build_results_document",
    "read_results",
    "validate_results",
    "write_results",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {shape} array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {shape} array")
    return array


def _integer_matrix(value: Any, name: str) -> np.ndarray:
    array = _finite_array(value, (2, 2), name)
    flat_values = np.asarray(value, dtype=object).ravel().tolist()
    if any(isinstance(item, bool) for item in flat_values) or not np.all(array == np.round(array)):
        raise ValueError(f"{name} must be a 2x2 integer matrix")
    matrix = array.astype(np.int64)
    if int(round(np.linalg.det(matrix))) == 0:
        raise ValueError(f"{name} must be nonsingular")
    return matrix


def _validate_search(search: dict[str, Any]) -> None:
    missing = sorted(_SEARCH_FIELDS.difference(search))
    if missing:
        raise ValueError(f"search is missing required fields: {', '.join(missing)}")
    for name in ("top_poscar", "bottom_poscar"):
        if not isinstance(search[name], str) or not search[name].strip():
            raise ValueError(f"search.{name} must be a nonempty path string")
    if _finite_number(search["max_length"], "search.max_length") <= 0.0:
        raise ValueError("search.max_length must be positive")
    for name in ("top_strain", "bottom_strain"):
        if _finite_number(search[name], f"search.{name}") < 0.0:
            raise ValueError(f"search.{name} must be nonnegative")
    if float(search["top_strain"]) + float(search["bottom_strain"]) <= 0.0:
        raise ValueError("search strain budgets cannot both be zero")
    min_length = search["min_length"]
    if min_length is not None and _finite_number(min_length, "search.min_length") <= 0.0:
        raise ValueError("search.min_length must be positive when present")
    max_atoms = search["max_atoms"]
    if max_atoms is not None:
        _positive_integer(max_atoms, "search.max_atoms")
    _positive_integer(search["top_atoms"], "search.top_atoms")
    _positive_integer(search["bottom_atoms"], "search.bottom_atoms")
    if _finite_number(search["max_aspect_ratio"], "search.max_aspect_ratio") < 1.0:
        raise ValueError("search.max_aspect_ratio must be at least one")
    minimum = _finite_number(search["min_cell_angle_deg"], "search.min_cell_angle_deg")
    maximum = _finite_number(search["max_cell_angle_deg"], "search.max_cell_angle_deg")
    if not 0.0 < minimum < maximum < 180.0:
        raise ValueError("search cell-angle limits must satisfy 0 < min < max < 180")
    _boolean(search["fold_symmetry"], "search.fold_symmetry")
    _boolean(search["symmetric"], "search.symmetric")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for name in ("created_at", "engine"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"metadata.{name} must be a nonempty string")
    if metadata["engine"] != "gram-v1":
        raise ValueError("metadata.engine must be 'gram-v1'")
    _boolean(metadata.get("symmetric_requested"), "metadata.symmetric_requested")
    _boolean(metadata.get("symmetric_used"), "metadata.symmetric_used")
    fallback = metadata.get("symmetric_fallback")
    if fallback is not None and not isinstance(fallback, str):
        raise ValueError("metadata.symmetric_fallback must be null or a string")
    _mapping(metadata.get("stage_stats"), "metadata.stage_stats")


def _validate_candidate(candidate: dict[str, Any], expected_index: int, search: dict[str, Any]) -> None:
    missing = sorted(_CANDIDATE_FIELDS.difference(candidate))
    if missing:
        raise ValueError(f"candidate {expected_index} is missing fields: {', '.join(missing)}")
    if _positive_integer(candidate["index"], f"candidate {expected_index}.index") != expected_index:
        raise ValueError("candidate indexes must be consecutive and one-based")
    _integer_matrix(candidate["top_matrix"], f"candidate {expected_index}.top_matrix")
    _integer_matrix(candidate["bottom_matrix"], f"candidate {expected_index}.bottom_matrix")
    for name in ("top_gram", "bottom_gram"):
        gram = _finite_array(candidate[name], (3,), f"candidate {expected_index}.{name}")
        if gram[0] <= 0.0 or gram[2] <= 0.0 or gram[0] * gram[2] - gram[1] * gram[1] <= 0.0:
            raise ValueError(f"candidate {expected_index}.{name} must be positive definite")
    _finite_number(candidate["angle_deg"], f"candidate {expected_index}.angle_deg")
    _finite_array(candidate["strain"], (2,), f"candidate {expected_index}.strain")
    for name in ("top_strain", "bottom_strain"):
        budget = _finite_number(candidate[name], f"candidate {expected_index}.{name}")
        if budget < 0.0 or budget != float(search[name]):
            raise ValueError(f"candidate {expected_index}.{name} must match the search budget")
    sharing = _finite_number(
        candidate["sharing_fraction"], f"candidate {expected_index}.sharing_fraction"
    )
    if not 0.0 <= sharing <= 1.0:
        raise ValueError(f"candidate {expected_index}.sharing_fraction must lie in [0, 1]")
    top_count = _positive_integer(
        candidate["top_atom_count"], f"candidate {expected_index}.top_atom_count"
    )
    bottom_count = _positive_integer(
        candidate["bottom_atom_count"], f"candidate {expected_index}.bottom_atom_count"
    )
    atom_count = _positive_integer(candidate["atom_count"], f"candidate {expected_index}.atom_count")
    if atom_count != top_count + bottom_count:
        raise ValueError(f"candidate {expected_index}.atom_count must equal the layer counts")
    _boolean(candidate["loewner_certified"], f"candidate {expected_index}.loewner_certified")
    _boolean(candidate["loewner_borderline"], f"candidate {expected_index}.loewner_borderline")
    _positive_integer(candidate["rank"], f"candidate {expected_index}.rank")
    _boolean(candidate["pareto_optimal"], f"candidate {expected_index}.pareto_optimal")
    for name in ("top_affine", "bottom_affine", "shared_lattice"):
        matrix = _finite_array(candidate[name], (2, 2), f"candidate {expected_index}.{name}")
        if abs(float(np.linalg.det(matrix))) <= np.finfo(float).eps:
            raise ValueError(f"candidate {expected_index}.{name} must be nonsingular")


def validate_results(payload: Any) -> dict[str, Any]:
    """Validate and return a detached JSON-v1 document."""

    document = copy.deepcopy(_mapping(payload, "results"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"results schema must be '{SCHEMA}'")
    if document.get("version") != VERSION:
        raise ValueError(f"results version must be {VERSION}")
    search = _mapping(document.get("search"), "search")
    metadata = _mapping(document.get("metadata"), "metadata")
    _validate_search(search)
    _validate_metadata(metadata)
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a JSON array")
    for index, candidate in enumerate(candidates, start=1):
        _validate_candidate(_mapping(candidate, f"candidate {index}"), index, search)
    return document


def build_results_document(
    *,
    top_poscar: str | Path,
    bottom_poscar: str | Path,
    config: SearchConfig,
    result: SearchResult,
    symmetric_requested: bool,
    symmetric_used: bool,
    symmetric_fallback: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Serialize one completed search without recomputing any candidate geometry.

    Engine matrices use coefficient columns.  JSON exposes the equivalent row-vector
    matrices used by POSCAR, so each stored matrix is the transpose of the native engine
    matrix.  ``strain`` stores the two principal relative logarithmic strains.
    """

    search_payload = {
        "top_poscar": str(Path(top_poscar).resolve()),
        "bottom_poscar": str(Path(bottom_poscar).resolve()),
        "max_length": float(config.max_length),
        "top_strain": float(config.top_strain),
        "bottom_strain": float(config.bottom_strain),
        "min_length": None if config.min_length is None else float(config.min_length),
        "max_atoms": None if config.max_atoms is None else int(config.max_atoms),
        "top_atoms": int(config.top_atoms),
        "bottom_atoms": int(config.bottom_atoms),
        "max_aspect_ratio": float(config.max_aspect_ratio),
        "min_cell_angle_deg": float(config.min_cell_angle_deg),
        "max_cell_angle_deg": float(config.max_cell_angle_deg),
        "fold_symmetry": bool(config.fold_symmetry),
        "symmetric": bool(symmetric_requested),
    }
    candidates: list[dict[str, Any]] = []
    for offset in range(len(result)):
        candidates.append(
            {
                "index": offset + 1,
                "top_matrix": result.top_matrices[offset].T.astype(int).tolist(),
                "bottom_matrix": result.bottom_matrices[offset].T.astype(int).tolist(),
                "top_gram": result.top_gram[offset].astype(float).tolist(),
                "bottom_gram": result.bottom_gram[offset].astype(float).tolist(),
                "angle_deg": float(result.twist_degrees[offset]),
                "strain": result.principal_strains[offset].astype(float).tolist(),
                "top_strain": float(config.top_strain),
                "bottom_strain": float(config.bottom_strain),
                "sharing_fraction": float(result.sharing_fraction[offset]),
                "top_atom_count": int(result.top_atom_counts[offset]),
                "bottom_atom_count": int(result.bottom_atom_counts[offset]),
                "atom_count": int(result.atom_counts[offset]),
                "loewner_certified": bool(result.loewner_certified[offset]),
                "loewner_borderline": bool(result.loewner_borderline[offset]),
                "rank": int(result.rank[offset]),
                "pareto_optimal": bool(result.pareto_optimal[offset]),
                "top_affine": result.top_affine[offset].astype(float).tolist(),
                "bottom_affine": result.bottom_affine[offset].astype(float).tolist(),
                "shared_lattice": result.shared_lattice[offset].astype(float).tolist(),
            }
        )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "search": search_payload,
        "metadata": {
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "engine": "gram-v1",
            "symmetric_requested": bool(symmetric_requested),
            "symmetric_used": bool(symmetric_used),
            "symmetric_fallback": symmetric_fallback,
            "stage_stats": _json_ready(result.stats),
        },
        "candidates": candidates,
    }
    return validate_results(payload)


def read_results(path: str | Path) -> dict[str, Any]:
    """Read and validate JSON v1, rejecting positional legacy DAT results."""

    source = Path(path).resolve()
    if source.suffix.lower() == ".dat":
        raise ValueError(
            "legacy positional .dat moire results are unsupported; rerun `moire find` "
            "to create schema-versioned results.json"
        )
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON v1; rerun `moire find`") from exc
    return validate_results(payload)


def write_results(path: str | Path, payload: Any) -> Path:
    """Validate and write one deterministic JSON-v1 results document."""

    document = validate_results(_json_ready(payload))
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination

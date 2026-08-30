"""Versioned JSON persistence for native Gram-form bilayer searches."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...core.symmetry2d import proper_subgroup as _proper_subgroup
from ...core.lattice import vector_angle_deg
from .gram import SearchConfig, SearchResult

SCHEMA = "cellstine.moire.gram"
VERSION = 2
_EPS = float(np.finfo(float).eps)

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
    "primitive_only",
    "top_rotation_order",
    "bottom_rotation_order",
    "angle_period_deg",
}
_CANDIDATE_FIELDS = {
    "index",
    "top_matrix",
    "bottom_matrix",
    "top_gram",
    "bottom_gram",
    "angle_deg",
    "raw_angle_deg",
    "strain",
    "top_layer_strain",
    "bottom_layer_strain",
    "sharing_fraction",
    "top_atom_count",
    "bottom_atom_count",
    "atom_count",
    "coincidence_index",
    "moire_a",
    "moire_b",
    "moire_gamma_deg",
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
    "shortlist_offsets",
    "read_results",
    "validate_results",
    "write_results",
]


def _json_ready(value: Any) -> Any:
    # The cheap concrete types are tested first: this runs once per scalar of a
    # search that can hold tens of thousands of candidates, and the abstract
    # ``Mapping`` check is an order of magnitude more expensive than ``dict``.
    if type(value) in (float, int, str, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _cell_angle_deg(shared_lattice: np.ndarray) -> float:
    """Return the angle between the two shared in-plane vectors, in degrees."""

    columns = np.asarray(shared_lattice, dtype=float)
    return vector_angle_deg(columns[:, 0], columns[:, 1])


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


def _flat_scalars(value: Any, shape: tuple[int, ...]) -> list[Any] | None:
    """Return the entries of a nested list of the given shape, or ``None``.

    Validation runs over every candidate of a search, so the tiny fixed shapes
    used here are unpacked directly instead of going through NumPy, which costs
    microseconds per call on arrays of four numbers.
    """

    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        return None
    if len(shape) == 1:
        return list(value)
    flat: list[Any] = []
    for row in value:
        inner = _flat_scalars(row, shape[1:])
        if inner is None:
            return None
        flat.extend(inner)
    return flat


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> list[float]:
    """Return the entries of a finite numeric array of the given shape.

    The entries come back flattened in row-major order; the shapes used by the
    schema are small enough that callers index them directly.
    """

    scalar_values = _flat_scalars(value, shape)
    if scalar_values is None:
        raise ValueError(f"{name} must be a finite {shape} array")
    numbers: list[float] = []
    for item in scalar_values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must be a finite {shape} array")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite {shape} array")
        numbers.append(number)
    return numbers


def _integer_matrix(value: Any, name: str) -> list[int]:
    """Return the four entries of a nonsingular 2x2 integer matrix."""

    scalar_values = _flat_scalars(value, (2, 2))
    if scalar_values is None:
        raise ValueError(f"{name} must be a 2x2 integer matrix")
    entries: list[int] = []
    for item in scalar_values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must be a 2x2 integer matrix")
        entries.append(int(item))
    if entries[0] * entries[3] - entries[1] * entries[2] == 0:
        raise ValueError(f"{name} must be nonsingular")
    return entries


def _validate_search(search: dict[str, Any]) -> None:
    missing = sorted(_SEARCH_FIELDS.difference(search))
    if missing:
        raise ValueError(f"search is missing required fields: {', '.join(missing)}")
    for name in ("top_poscar", "bottom_poscar"):
        if not isinstance(search[name], str) or not search[name].strip():
            raise ValueError(f"search.{name} must be a nonempty path string")
    max_length = _finite_number(search["max_length"], "search.max_length")
    if max_length <= 0.0:
        raise ValueError("search.max_length must be positive")
    for name in ("top_strain", "bottom_strain"):
        if _finite_number(search[name], f"search.{name}") < 0.0:
            raise ValueError(f"search.{name} must be nonnegative")
    min_length = search["min_length"]
    if min_length is not None:
        normalized_min_length = _finite_number(min_length, "search.min_length")
        if normalized_min_length <= 0.0:
            raise ValueError("search.min_length must be positive when present")
        if normalized_min_length > max_length:
            raise ValueError("search.min_length cannot exceed search.max_length")
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
    _validate_twist_window(search)
    _boolean(search["fold_symmetry"], "search.fold_symmetry")
    _boolean(search["symmetric"], "search.symmetric")
    _boolean(search["primitive_only"], "search.primitive_only")
    _positive_integer(search["top_rotation_order"], "search.top_rotation_order")
    _positive_integer(search["bottom_rotation_order"], "search.bottom_rotation_order")
    period = _finite_number(search["angle_period_deg"], "search.angle_period_deg")
    if not 0.0 < period <= 360.0:
        raise ValueError("search.angle_period_deg must lie in (0, 360]")


def _validate_twist_window(search: dict[str, Any]) -> None:
    """Check the optional twist-angle window, which older results omit."""

    bounds: list[float] = []
    for name in ("min_twist_angle_deg", "max_twist_angle_deg"):
        value = search.get(name)
        if value is None:
            bounds.append(0.0 if name.startswith("min") else 180.0)
            continue
        angle = _finite_number(value, f"search.{name}")
        if not 0.0 <= angle <= 180.0:
            raise ValueError(f"search.{name} must lie in [0, 180]")
        bounds.append(angle)
    if bounds[0] > bounds[1]:
        raise ValueError(
            "search.min_twist_angle_deg cannot exceed search.max_twist_angle_deg"
        )


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
    if metadata.get("symmetry_tolerance") is not None:
        tolerance = _finite_number(
            metadata["symmetry_tolerance"], "metadata.symmetry_tolerance"
        )
        if tolerance <= 0.0:
            raise ValueError("metadata.symmetry_tolerance must be positive")
    for name in ("top_idealisation", "bottom_idealisation"):
        if metadata.get(name) is not None:
            deviation = _finite_number(metadata[name], f"metadata.{name}")
            if deviation < 0.0:
                raise ValueError(f"metadata.{name} must be nonnegative")
    found = metadata.get("candidates_found")
    recorded = metadata.get("candidates_recorded")
    if found is not None or recorded is not None:
        for name, value in (("candidates_found", found), ("candidates_recorded", recorded)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metadata.{name} must be a nonnegative integer")
        if int(recorded) > int(found):
            raise ValueError(
                "metadata.candidates_recorded cannot exceed metadata.candidates_found"
            )
    if metadata.get("max_candidates") is not None:
        _positive_integer(metadata["max_candidates"], "metadata.max_candidates")
    for name in ("top_layer_index", "bottom_layer_index"):
        if metadata.get(name) is not None:
            _positive_integer(metadata[name], f"metadata.{name}")
    for name in ("top_poscar_source", "bottom_poscar_source"):
        source = metadata.get(name)
        if source is not None and not isinstance(source, str):
            raise ValueError(f"metadata.{name} must be null or a string")


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
    angle = _finite_number(candidate["angle_deg"], f"candidate {expected_index}.angle_deg")
    period = float(search["angle_period_deg"])
    if abs(angle) > 0.5 * period + 1e-6:
        raise ValueError(
            f"candidate {expected_index}.angle_deg must be folded into the fundamental range"
        )
    _finite_number(candidate["raw_angle_deg"], f"candidate {expected_index}.raw_angle_deg")
    relative = _finite_array(candidate["strain"], (2,), f"candidate {expected_index}.strain")
    sharing = _finite_number(
        candidate["sharing_fraction"], f"candidate {expected_index}.sharing_fraction"
    )
    if not 0.0 <= sharing <= 1.0:
        raise ValueError(f"candidate {expected_index}.sharing_fraction must lie in [0, 1]")
    top_layer = _finite_array(
        candidate["top_layer_strain"], (2,), f"candidate {expected_index}.top_layer_strain"
    )
    bottom_layer = _finite_array(
        candidate["bottom_layer_strain"], (2,), f"candidate {expected_index}.bottom_layer_strain"
    )
    for difference, expected in zip(
        (top_layer[0] - bottom_layer[0], top_layer[1] - bottom_layer[1]), relative
    ):
        if abs(difference - expected) > 1e-9 + 1e-7 * abs(expected):
            raise ValueError(
                f"candidate {expected_index} layer strains must differ by the relative strain"
            )
    for entries, name, budget in (
        (top_layer, "top_layer_strain", "top_strain"),
        (bottom_layer, "bottom_layer_strain", "bottom_strain"),
    ):
        if max(abs(entry) for entry in entries) > float(search[budget]) + 1e-9:
            raise ValueError(f"candidate {expected_index}.{name} exceeds the {budget} budget")
    top_count = _positive_integer(
        candidate["top_atom_count"], f"candidate {expected_index}.top_atom_count"
    )
    bottom_count = _positive_integer(
        candidate["bottom_atom_count"], f"candidate {expected_index}.bottom_atom_count"
    )
    atom_count = _positive_integer(candidate["atom_count"], f"candidate {expected_index}.atom_count")
    if atom_count != top_count + bottom_count:
        raise ValueError(f"candidate {expected_index}.atom_count must equal the layer counts")
    if _positive_integer(
        candidate["coincidence_index"], f"candidate {expected_index}.coincidence_index"
    ) != 1 and search["primitive_only"]:
        raise ValueError(
            f"candidate {expected_index} is not a primitive coincidence cell"
        )
    for name in ("moire_a", "moire_b"):
        if _finite_number(candidate[name], f"candidate {expected_index}.{name}") <= 0.0:
            raise ValueError(f"candidate {expected_index}.{name} must be positive")
    gamma = _finite_number(candidate["moire_gamma_deg"], f"candidate {expected_index}.moire_gamma_deg")
    if not 0.0 < gamma < 180.0:
        raise ValueError(f"candidate {expected_index}.moire_gamma_deg must lie in (0, 180)")
    _boolean(candidate["loewner_certified"], f"candidate {expected_index}.loewner_certified")
    _boolean(candidate["loewner_borderline"], f"candidate {expected_index}.loewner_borderline")
    _positive_integer(candidate["rank"], f"candidate {expected_index}.rank")
    _boolean(candidate["pareto_optimal"], f"candidate {expected_index}.pareto_optimal")
    for name in ("top_affine", "bottom_affine", "shared_lattice"):
        matrix = _finite_array(candidate[name], (2, 2), f"candidate {expected_index}.{name}")
        if abs(matrix[0] * matrix[3] - matrix[1] * matrix[2]) <= _EPS:
            raise ValueError(f"candidate {expected_index}.{name} must be nonsingular")


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a document in place and return it."""
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


def validate_results(payload: Any) -> dict[str, Any]:
    """Validate and return a detached JSON-v1 document."""

    return _validate_document(copy.deepcopy(_mapping(payload, "results")))


def shortlist_offsets(result: SearchResult, max_candidates: int | None) -> list[int]:
    """Return the offsets of the candidates worth recording, smallest cells first.

    A wide search returns tens of thousands of admissible supercells, nearly all
    of which are dominated: some other candidate is both smaller and less
    strained.  The shortlist therefore keeps the whole Pareto front of (atom
    count, relative strain) -- which is where every useful cell lives, including
    the low-strain large ones -- and fills the rest of the budget with the
    smallest cells in rank order.  The front is never truncated, so the limit can
    be exceeded when the front alone is larger than it.

    Because the front is kept whole, the shortlist still matches or beats every
    candidate of the search in *both* costs at once --- see
    ``Cellstine.Pareto.exists_mem_le_of_isRecord_subset`` in
    ``RequestProject/ParetoFront.lean``.
    """

    total = len(result)
    if max_candidates is None or int(max_candidates) <= 0 or total <= int(max_candidates):
        return list(range(total))
    limit = int(max_candidates)
    keep = {offset for offset in range(total) if bool(result.pareto_optimal[offset])}
    for offset in range(total):
        if len(keep) >= limit:
            break
        keep.add(offset)
    return sorted(keep)


def build_results_document(
    *,
    top_poscar: str | Path,
    bottom_poscar: str | Path,
    config: SearchConfig,
    result: SearchResult,
    symmetric_requested: bool,
    symmetric_used: bool,
    symmetric_fallback: str | None = None,
    symmetry_tolerance: float | None = None,
    top_idealisation: float | None = None,
    bottom_idealisation: float | None = None,
    created_at: str | None = None,
    max_candidates: int | None = None,
    top_layer_index: int = 1,
    bottom_layer_index: int = 1,
    top_poscar_source: str | Path | None = None,
    bottom_poscar_source: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize one completed search without recomputing any candidate geometry.

    Engine matrices use coefficient columns.  JSON exposes the equivalent row-vector
    matrices used by POSCAR, so each stored matrix is the transpose of the native engine
    matrix.  ``strain`` stores the two principal *relative* logarithmic strains, while
    ``top_layer_strain`` and ``bottom_layer_strain`` store the strain actually applied to
    each layer; their difference is the relative strain.  ``angle_deg`` is folded into the
    fundamental range of the two layer symmetries and ``raw_angle_deg`` keeps the
    unfolded value.

    ``max_candidates`` records only the shortlist of
    :func:`shortlist_offsets`; ``index`` then numbers the recorded candidates
    consecutively while ``rank`` keeps each candidate's place in the full search,
    and ``metadata`` reports how many candidates the search found.
    """

    top_layer_strains = result.top_layer_strains
    bottom_layer_strains = result.bottom_layer_strains
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
        "min_twist_angle_deg": (
            None if config.min_twist_angle_deg is None else float(config.min_twist_angle_deg)
        ),
        "max_twist_angle_deg": (
            None if config.max_twist_angle_deg is None else float(config.max_twist_angle_deg)
        ),
        "fold_symmetry": bool(config.fold_symmetry),
        "symmetric": bool(symmetric_requested),
        "primitive_only": bool(config.primitive_only),
        "top_rotation_order": int(len(_proper_subgroup(config.top_group))),
        "bottom_rotation_order": int(len(_proper_subgroup(config.bottom_group))),
        "angle_period_deg": float(np.degrees(config.angle_period_radians)),
    }
    offsets = shortlist_offsets(result, max_candidates)
    candidates: list[dict[str, Any]] = []
    for position, offset in enumerate(offsets, start=1):
        candidates.append(
            {
                "index": position,
                "top_matrix": result.top_matrices[offset].T.astype(int).tolist(),
                "bottom_matrix": result.bottom_matrices[offset].T.astype(int).tolist(),
                "top_gram": result.top_gram[offset].astype(float).tolist(),
                "bottom_gram": result.bottom_gram[offset].astype(float).tolist(),
                "angle_deg": float(result.twist_degrees[offset]),
                "raw_angle_deg": float(np.degrees(result.raw_twist_radians[offset])),
                "strain": result.principal_strains[offset].astype(float).tolist(),
                "top_layer_strain": top_layer_strains[offset].astype(float).tolist(),
                "bottom_layer_strain": bottom_layer_strains[offset].astype(float).tolist(),
                "sharing_fraction": float(result.sharing_fraction[offset]),
                "top_atom_count": int(result.top_atom_counts[offset]),
                "bottom_atom_count": int(result.bottom_atom_counts[offset]),
                "atom_count": int(result.atom_counts[offset]),
                "coincidence_index": int(result.coincidence_indices[offset]),
                "moire_a": float(np.linalg.norm(result.shared_lattice[offset][:, 0])),
                "moire_b": float(np.linalg.norm(result.shared_lattice[offset][:, 1])),
                "moire_gamma_deg": _cell_angle_deg(result.shared_lattice[offset]),
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
            "symmetry_tolerance": (
                None if symmetry_tolerance is None else float(symmetry_tolerance)
            ),
            "top_idealisation": (
                None if top_idealisation is None else float(top_idealisation)
            ),
            "bottom_idealisation": (
                None if bottom_idealisation is None else float(bottom_idealisation)
            ),
            "stage_stats": _json_ready(result.stats),
            "candidates_found": int(len(result)),
            "candidates_recorded": int(len(candidates)),
            "max_candidates": (
                None
                if max_candidates is None or int(max_candidates) <= 0
                else int(max_candidates)
            ),
            # How many primitive in-plane cells the layer the search actually
            # used holds of the file the user named, and where that file is.
            # Both are 1 and null for an input that was already primitive.
            "top_layer_index": int(top_layer_index),
            "bottom_layer_index": int(bottom_layer_index),
            "top_poscar_source": (
                None if top_poscar_source is None else str(Path(top_poscar_source).resolve())
            ),
            "bottom_poscar_source": (
                None
                if bottom_poscar_source is None
                else str(Path(bottom_poscar_source).resolve())
            ),
        },
        "candidates": candidates,
    }
    # The payload was just built here, so there is nothing to detach it from.
    return _validate_document(payload)


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

    document = _validate_document(_mapping(_json_ready(payload), "results"))
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination

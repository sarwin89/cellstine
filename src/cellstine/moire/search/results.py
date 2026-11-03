"""Versioned JSON persistence for native Gram-form bilayer searches."""

from __future__ import annotations

import copy
import gc
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...core.symmetry2d import proper_subgroup as _proper_subgroup
from ...core.lattice import vector_angle_deg
from .gram import SearchConfig, SearchResult
from .gram_config import _STRAIN_BUDGET_TOLERANCE

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


@contextmanager
def _collection_paused():
    """Hold off the cyclic garbage collector while a large document is handled.

    A wide search builds or parses millions of small dicts and lists at once,
    none of them cyclic garbage, and every generation-0 overflow rescans the
    survivors: writing 234k candidates took eight times as long as 56k.
    """

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


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


def _cell_shapes(shared_lattice: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """Lengths and angle of every shared cell, as :func:`_cell_angle_deg` gives them.

    The same Kahan half-angle formula as
    :func:`cellstine.core.lattice.vector_angle_deg`, with the same operations,
    written over all cells at once; ``atan2`` stays the scalar libm call.
    """

    columns = np.asarray(shared_lattice, dtype=float)
    x0, y0 = columns[:, 0, 0], columns[:, 1, 0]
    x1, y1 = columns[:, 0, 1], columns[:, 1, 1]
    first = np.sqrt(x0 * x0 + y0 * y0)
    second = np.sqrt(x1 * x1 + y1 * y1)
    if np.any(first <= 0.0) or np.any(second <= 0.0):
        raise ValueError("the angle between vectors needs two nonzero vectors")
    ux, uy, vx, vy = x0 / first, y0 / first, x1 / second, y1 / second
    dx, dy, sx, sy = ux - vx, uy - vy, ux + vx, uy + vy
    chord = np.sqrt(dx * dx + dy * dy).tolist()
    span = np.sqrt(sx * sx + sy * sy).tolist()
    half_angles = np.array([math.atan2(left, right) for left, right in zip(chord, span)], dtype=float)
    return first.tolist(), second.tolist(), np.degrees(2.0 * half_angles).tolist()


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


def _candidate_angle_limit(search: dict[str, Any]) -> float:
    """Recorded twist bound: symmetry-folded or a full signed half turn."""
    return .5 * float(search["angle_period_deg"]) if search["fold_symmetry"] else 180.


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
    if abs(angle) > _candidate_angle_limit(search) + 1e-6:
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
        if (
            max(abs(entry) for entry in entries)
            > float(search[budget]) + _STRAIN_BUDGET_TOLERANCE
        ):
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


def _plain_numbers(value: Any, count: int) -> list[float] | None:
    """A list of ``count`` finite plain ints or floats, as floats; else ``None``."""

    if type(value) is not list or len(value) != count:
        return None
    numbers = []
    for item in value:
        kind = type(item)
        if kind is float:
            number = item
        elif kind is int:
            try:
                number = float(item)
            except OverflowError:
                return None
        else:
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return numbers

def _plain_matrix(value: Any) -> list[float] | None:
    """The four entries of a finite 2x2 nested list of plain numbers, or ``None``."""

    if type(value) is not list or len(value) != 2:
        return None
    first, second = _plain_numbers(value[0], 2), _plain_numbers(value[1], 2)
    if first is None or second is None:
        return None
    return first + second

def _plain_integer_matrix(value: Any) -> bool:
    """A nonsingular 2x2 nested list of plain ints."""

    if type(value) is not list or len(value) != 2:
        return False
    entries = []
    for row in value:
        if type(row) is not list or len(row) != 2:
            return False
        for item in row:
            if type(item) is not int:
                return False
            entries.append(item)
    return entries[0] * entries[3] - entries[1] * entries[2] != 0

def _plainly_valid_candidate(candidate: Any, expected_index: int, search: dict[str, Any]) -> bool:
    """True only when :func:`_validate_candidate` would accept ``candidate``.

    The same tests as that function, written for the plain JSON types a parsed
    file holds, without its general helpers.  Anything unusual -- a missing
    field, a tuple, a numeric subclass, any failed test -- returns ``False``
    and the caller runs :func:`_validate_candidate`, which then raises its own
    message.  So this can only ever skip work, never change a verdict.
    """

    if type(candidate) is not dict or not _CANDIDATE_FIELDS.issubset(candidate.keys()):
        return False
    index = candidate["index"]
    if type(index) is not int or index != expected_index or index <= 0:
        return False
    if not (_plain_integer_matrix(candidate["top_matrix"]) and _plain_integer_matrix(candidate["bottom_matrix"])):
        return False
    for name in ("top_gram", "bottom_gram"):
        gram = _plain_numbers(candidate[name], 3)
        if gram is None or gram[0] <= 0.0 or gram[2] <= 0.0 or gram[0] * gram[2] - gram[1] * gram[1] <= 0.0:
            return False
    scalars = _plain_numbers(
        [candidate[name] for name in ("angle_deg", "raw_angle_deg", "sharing_fraction", "moire_a", "moire_b", "moire_gamma_deg")],
        6,
    )
    if scalars is None:
        return False
    angle, _, sharing, moire_a, moire_b, gamma = scalars
    if abs(angle) > _candidate_angle_limit(search) + 1e-6:
        return False
    if not 0.0 <= sharing <= 1.0 or moire_a <= 0.0 or moire_b <= 0.0 or not 0.0 < gamma < 180.0:
        return False
    relative = _plain_numbers(candidate["strain"], 2)
    top_layer = _plain_numbers(candidate["top_layer_strain"], 2)
    bottom_layer = _plain_numbers(candidate["bottom_layer_strain"], 2)
    if relative is None or top_layer is None or bottom_layer is None:
        return False
    for difference, expected in zip(
        (top_layer[0] - bottom_layer[0], top_layer[1] - bottom_layer[1]), relative
    ):
        if abs(difference - expected) > 1e-9 + 1e-7 * abs(expected):
            return False
    if max(abs(entry) for entry in top_layer) > float(search["top_strain"]) + _STRAIN_BUDGET_TOLERANCE:
        return False
    if max(abs(entry) for entry in bottom_layer) > float(search["bottom_strain"]) + _STRAIN_BUDGET_TOLERANCE:
        return False
    counts = [candidate[name] for name in ("top_atom_count", "bottom_atom_count", "atom_count", "coincidence_index", "rank")]
    if any(type(value) is not int or value <= 0 for value in counts):
        return False
    top_count, bottom_count, atom_count, coincidence, _ = counts
    if atom_count != top_count + bottom_count or (coincidence != 1 and search["primitive_only"]):
        return False
    if any(
        type(candidate[name]) is not bool
        for name in ("loewner_certified", "loewner_borderline", "pareto_optimal")
    ):
        return False
    for name in ("top_affine", "bottom_affine", "shared_lattice"):
        matrix = _plain_matrix(candidate[name])
        if matrix is None or abs(matrix[0] * matrix[3] - matrix[1] * matrix[2]) <= _EPS:
            return False
    return True


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
        if not _plainly_valid_candidate(candidate, index, search):
            _validate_candidate(_mapping(candidate, f"candidate {index}"), index, search)
    return document


def validate_results(payload: Any) -> dict[str, Any]:
    """Validate and return a detached JSON-v2 document."""

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
    ``aristotle-lean-reference/RequestProject/ParetoFront.lean``.
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


@_collection_paused()
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
    rows = np.asarray(offsets, dtype=np.int64)
    # Every field is gathered for all recorded rows at once and converted to
    # Python values in one ``tolist`` per field; building them one candidate at
    # a time cost microseconds per value, which a wide search multiplies by
    # millions.  The values, and so the file, are the same.
    shared = result.shared_lattice[rows].astype(float)
    moire_a, moire_b, moire_gamma = _cell_shapes(shared)
    columns = {
        "top_matrix": np.swapaxes(result.top_matrices[rows], 1, 2).astype(int).tolist(),
        "bottom_matrix": np.swapaxes(result.bottom_matrices[rows], 1, 2).astype(int).tolist(),
        "top_gram": result.top_gram[rows].astype(float).tolist(),
        "bottom_gram": result.bottom_gram[rows].astype(float).tolist(),
        "angle_deg": result.twist_degrees[rows].astype(float).tolist(),
        "raw_angle_deg": np.degrees(result.raw_twist_radians[rows]).astype(float).tolist(),
        "strain": result.principal_strains[rows].astype(float).tolist(),
        "top_layer_strain": top_layer_strains[rows].astype(float).tolist(),
        "bottom_layer_strain": bottom_layer_strains[rows].astype(float).tolist(),
        "sharing_fraction": result.sharing_fraction[rows].astype(float).tolist(),
        "top_atom_count": result.top_atom_counts[rows].astype(int).tolist(),
        "bottom_atom_count": result.bottom_atom_counts[rows].astype(int).tolist(),
        "atom_count": result.atom_counts[rows].astype(int).tolist(),
        "coincidence_index": result.coincidence_indices[rows].astype(int).tolist(),
        "moire_a": moire_a,
        "moire_b": moire_b,
        "moire_gamma_deg": moire_gamma,
        "loewner_certified": result.loewner_certified[rows].astype(bool).tolist(),
        "loewner_borderline": result.loewner_borderline[rows].astype(bool).tolist(),
        "rank": result.rank[rows].astype(int).tolist(),
        "pareto_optimal": result.pareto_optimal[rows].astype(bool).tolist(),
        "top_affine": result.top_affine[rows].astype(float).tolist(),
        "bottom_affine": result.bottom_affine[rows].astype(float).tolist(),
        "shared_lattice": shared.tolist(),
    }
    names = list(columns)
    candidates: list[dict[str, Any]] = [
        {"index": position, **dict(zip(names, values))}
        for position, values in enumerate(zip(*columns.values()), start=1)
    ]
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
    # Its candidates are checked as arrays: every value was converted from the
    # arrays tested, so they pass or fail exactly as the per-candidate checks
    # would, and a failure is re-raised by those checks with their message.
    _validate_search(search_payload)
    _validate_metadata(payload["metadata"])
    failing = _first_invalid_row(result, rows, search_payload, (moire_a, moire_b, moire_gamma))
    if failing is not None:
        _validate_candidate(candidates[failing], failing + 1, search_payload)
        raise AssertionError("array validation and candidate validation disagree")  # pragma: no cover
    return payload


def _first_invalid_row(
    result: SearchResult,
    rows: np.ndarray,
    search: dict[str, Any],
    shapes: tuple[list[float], list[float], list[float]],
) -> int | None:
    """Offset of the first recorded row :func:`_validate_candidate` would refuse.

    Each test below is the scalar test of :func:`_validate_candidate`, with the
    same operations in the same order, applied to all rows at once.  Types,
    field names and the consecutive index are fixed by construction.
    """

    if not len(rows):
        return None

    def finite(*arrays: np.ndarray) -> np.ndarray:
        good = np.ones(len(rows), dtype=bool)
        for array in arrays:
            good &= np.all(np.isfinite(array.reshape(len(rows), -1)), axis=1)
        return good

    def nonsingular_integer(matrices: np.ndarray) -> np.ndarray:
        matrices = matrices.astype(np.int64)
        determinant = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
        good = determinant != 0
        # Nonzero modulo 2**64 proves nonzero over the integers; zero may be
        # overflow. Preserve the scalar validator's Python-integer arithmetic
        # for just those exceptional rows, without an object array for all rows.
        for row in np.flatnonzero(~good):
            a, b, c, d = (int(value) for value in matrices[row].ravel())
            good[row] = a * d - b * c != 0
        return good

    good = nonsingular_integer(result.top_matrices[rows]) & (
        nonsingular_integer(result.bottom_matrices[rows])
    )
    for gram in (result.top_gram[rows].astype(float), result.bottom_gram[rows].astype(float)):
        good &= finite(gram)
        good &= ~(
            (gram[:, 0] <= 0.0)
            | (gram[:, 2] <= 0.0)
            | (gram[:, 0] * gram[:, 2] - gram[:, 1] * gram[:, 1] <= 0.0)
        )
    angle = result.twist_degrees[rows].astype(float)
    angle_limit = _candidate_angle_limit(search)
    good &= finite(angle, np.degrees(result.raw_twist_radians[rows]).astype(float))
    good &= ~(np.abs(angle) > angle_limit + 1e-6)
    relative = result.principal_strains[rows].astype(float)
    sharing = result.sharing_fraction[rows].astype(float)
    top_layer = result.top_layer_strains[rows].astype(float)
    bottom_layer = result.bottom_layer_strains[rows].astype(float)
    good &= finite(relative, sharing, top_layer, bottom_layer)
    good &= (sharing >= 0.0) & (sharing <= 1.0)
    difference = top_layer - bottom_layer
    good &= ~np.any(np.abs(difference - relative) > 1e-9 + 1e-7 * np.abs(relative), axis=1)
    good &= ~(np.max(np.abs(top_layer), axis=1) > float(search["top_strain"]) + _STRAIN_BUDGET_TOLERANCE)
    good &= ~(np.max(np.abs(bottom_layer), axis=1) > float(search["bottom_strain"]) + _STRAIN_BUDGET_TOLERANCE)
    top_count = result.top_atom_counts[rows].astype(np.int64)
    bottom_count = result.bottom_atom_counts[rows].astype(np.int64)
    atom_count = result.atom_counts[rows].astype(np.int64)
    coincidence = result.coincidence_indices[rows].astype(np.int64)
    good &= (top_count > 0) & (bottom_count > 0) & (atom_count > 0) & (coincidence > 0)
    good &= atom_count == top_count + bottom_count
    if search["primitive_only"]:
        good &= coincidence == 1
    moire_a, moire_b, gamma = (np.asarray(values, dtype=float) for values in shapes)
    good &= finite(moire_a, moire_b, gamma) & (moire_a > 0.0) & (moire_b > 0.0)
    good &= (gamma > 0.0) & (gamma < 180.0)
    good &= result.rank[rows].astype(np.int64) > 0
    for matrices in (result.top_affine[rows], result.bottom_affine[rows], result.shared_lattice[rows]):
        matrices = matrices.astype(float)
        good &= finite(matrices)
        good &= ~(
            np.abs(matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0])
            <= _EPS
        )
    failing = np.flatnonzero(~good)
    return None if not len(failing) else int(failing[0])


def read_results(path: str | Path) -> dict[str, Any]:
    """Read and validate JSON v2, rejecting positional legacy DAT results."""

    source = Path(path).resolve()
    if source.suffix.lower() == ".dat":
        raise ValueError(
            "legacy positional .dat moire results are unsupported; rerun `moire search` "
            "to create schema-versioned results.json"
        )
    with _collection_paused():
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} is not valid JSON v1; rerun `moire search`") from exc
        # Just parsed, so nothing else holds it: validate in place, no deep copy.
        return _validate_document(_mapping(payload, "results"))


def write_results(path: str | Path, payload: Any, *, validated: bool = False) -> Path:
    """Validate and write one deterministic JSON-v2 results document.

    ``validated=True`` is for a document :func:`build_results_document` has just
    returned, which is already JSON-ready and validated; checking it again costs
    seconds on a wide search.  The file written is the same either way.
    """

    with _collection_paused():
        document = (
            payload
            if validated
            else _validate_document(_mapping(_json_ready(payload), "results"))
        )
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            _write_indented_json(handle, document)
            handle.write("\n")
    return destination


#: Candidates encoded per call by :func:`_write_indented_json`.
_CANDIDATE_CHUNK = 2048


def _write_indented_json(handle, document: dict[str, Any]) -> None:
    """Write ``json.dumps(document, indent=2)`` without holding the whole text.

    ``json.dump`` to a stream uses the pure-Python encoder and is several times
    slower; ``json.dumps`` uses the C encoder but materialises the full text,
    which for a wide search is hundreds of megabytes on top of the document.
    The candidates are therefore encoded a chunk at a time by the C encoder and
    re-indented into place.  JSON indentation depends only on nesting depth and
    an encoded string never contains a raw newline, so the bytes are exactly
    those of a single ``dumps`` call.
    """

    candidates = document.get("candidates")
    keys = list(document)
    if not isinstance(candidates, list) or not candidates or keys[-1] != "candidates":
        handle.write(json.dumps(document, indent=2, allow_nan=False))
        return
    head = json.dumps({**document, "candidates": []}, indent=2, allow_nan=False)
    closing = "[]\n}"
    if not head.endswith(closing):  # pragma: no cover - the format above guarantees it
        handle.write(json.dumps(document, indent=2, allow_nan=False))
        return
    handle.write(head[: -len(closing)])
    handle.write("[\n")
    for start in range(0, len(candidates), _CANDIDATE_CHUNK):
        if start:
            handle.write(",\n")
        chunk = json.dumps(candidates[start : start + _CANDIDATE_CHUNK], indent=2, allow_nan=False)
        # ``[\n  {...},\n  {...}\n]`` at depth zero; the list sits at depth one.
        handle.write("  " + chunk[2:-2].replace("\n", "\n  "))
    handle.write("\n  ]\n}")

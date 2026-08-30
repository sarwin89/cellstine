"""Commensurate supercell matching for heterointerfaces.

Two slabs almost never share a common in-plane cell in their primitive 1x1
setting: forcing one onto the other then applies whatever strain the lattice
constants happen to differ by, which for a pair such as Si(111) and Al(111) is
tens of percent and is physically meaningless.  A usable heterointerface needs a
*commensurate supercell*: integer supercell matrices ``M_b`` and ``M_t`` and a
relative rotation such that the two superlattices agree to within a small strain
that is then shared between the slabs.

That is exactly the problem the moire Gram-form engine solves, so this module
reuses it instead of duplicating the search.  Each match therefore carries the
same certified quantities as a moire candidate: the two integer matrices, the
twist angle, the principal *relative* logarithmic strains, how the engine shares
them between the two slabs, the recorded affine maps, and the shared lattice.
Those records feed straight back into the shared builder, so ``interface build``
and ``moire make`` construct their supercells with one code path.

The written document uses schema ``cellstine.interface.match`` version 1.  Every
match keeps a pointer to the validated moire results file it came from, together
with the candidate index inside that file.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...moire.search.find import run_find
from ...moire.search.results import read_results

SCHEMA = "cellstine.interface.match"
VERSION = 1

#: Resolution, in logarithmic strain, at which two matches count as equally strained.
#:
#: A strain is a logarithm of a ratio of two floating-point cell metrics, so two
#: matches that are the same deformation reached through different (but
#: equivalent) integer supercells can differ in the last couple of bits.  Ranking
#: on the raw value would let that noise decide which of two physically
#: equally strained matches is offered first, and it would in general put the
#: larger cell in front.  Anything closer than this is therefore a tie, resolved
#: by cell size, then area, then twist -- all of which are meaningful.
#:
#: What the snapping does and does not do is proved in
#: ``aristotle-lean-reference/RequestProject/MatchOrdering.lean``: it never inverts a real strain
#: difference (``Cellstine.lt_of_quantise_lt``), a tie really is a tie to within
#: this resolution (``Cellstine.abs_sub_le_of_quantise_eq``), and two strains
#: further apart than it are never merged (``Cellstine.quantise_lt_of_add_lt``).
#: The proofs hold for any rounding convention that is monotone and lands within
#: half a unit, so they cover Python's round-half-to-even.
STRAIN_ORDER_RESOLUTION = 1e-12

__all__ = [
    "SCHEMA",
    "VERSION",
    "STRAIN_ORDER_RESOLUTION",
    "MatchRequest",
    "SlabPair",
    "search_slab_pair",
    "build_match_document",
    "read_matches",
    "validate_matches",
    "write_matches",
    "format_matches_table",
]


@dataclass(frozen=True)
class MatchRequest:
    """Search limits shared by every slab pair in one match scan.

    ``max_strain`` bounds the principal logarithmic strain applied to a single
    slab, so the relative strain between the two slabs is at most twice that in
    ``shared`` mode and exactly ``max_strain`` in ``film`` mode, where the bottom
    slab is treated as a rigid substrate.
    """

    max_length: float
    max_strain: float = 0.05
    strain_mode: str = "shared"
    min_length: float | None = None
    max_atoms: int | None = None
    max_aspect_ratio: float = 12.0
    min_cell_angle_deg: float = 25.0
    max_cell_angle_deg: float = 155.0

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.max_length)) or float(self.max_length) <= 0.0:
            raise ValueError("max_length must be a finite positive length in angstrom")
        if not np.isfinite(float(self.max_strain)) or float(self.max_strain) <= 0.0:
            raise ValueError("max_strain must be a finite positive fraction")
        if str(self.strain_mode) not in {"shared", "film"}:
            raise ValueError("strain_mode must be 'shared' or 'film'")

    @property
    def strain_budgets(self) -> tuple[float, float]:
        """Return the ``(bottom, top)`` principal logarithmic strain budgets."""

        if str(self.strain_mode) == "film":
            return 0.0, float(self.max_strain)
        return float(self.max_strain), float(self.max_strain)


@dataclass(frozen=True)
class SlabPair:
    """One bottom/top slab combination taken from a match scan."""

    bottom_slab: Path
    top_slab: Path
    bottom_miller: tuple[int, int, int]
    top_miller: tuple[int, int, int]
    bottom_layers: int
    top_layers: int


def _strain_magnitude(values: Sequence[float]) -> float:
    return max(abs(float(value)) for value in values)


def _cell_area(shared_lattice: Sequence[Sequence[float]]) -> float:
    (a11, a12), (a21, a22) = ((float(row[0]), float(row[1])) for row in shared_lattice)
    return abs(a11 * a22 - a12 * a21)


def search_slab_pair(
    pair: SlabPair,
    request: MatchRequest,
    *,
    results_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Search one slab pair and return its results file and candidate records.

    The bottom slab is the substrate and stays the reference layer, so the
    returned candidates use the moire convention in which ``bottom`` is the lower
    structure and ``top`` the one stacked above it.
    """

    bottom_strain, top_strain = request.strain_budgets
    run = run_find(
        top_poscar=str(pair.top_slab),
        bottom_poscar=str(pair.bottom_slab),
        max_length=float(request.max_length),
        top_strain=float(top_strain),
        bottom_strain=float(bottom_strain),
        min_length=None if request.min_length is None else float(request.min_length),
        max_atoms=None if request.max_atoms is None else int(request.max_atoms),
        max_aspect_ratio=float(request.max_aspect_ratio),
        min_cell_angle_deg=float(request.min_cell_angle_deg),
        max_cell_angle_deg=float(request.max_cell_angle_deg),
        output_root=str(results_root),
    )
    return run.result_path, list(run.candidates)


def match_entries(
    pair: SlabPair,
    candidates: Sequence[Mapping[str, Any]],
    *,
    results_path: Path,
) -> list[dict[str, Any]]:
    """Convert validated moire candidates into interface match records."""

    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        shared_lattice = candidate["shared_lattice"]
        entries.append(
            {
                "bottom_miller": [int(value) for value in pair.bottom_miller],
                "top_miller": [int(value) for value in pair.top_miller],
                "bottom_layers": int(pair.bottom_layers),
                "top_layers": int(pair.top_layers),
                "bottom_slab": str(pair.bottom_slab),
                "top_slab": str(pair.top_slab),
                "results_json": str(results_path),
                "candidate_index": int(candidate["index"]),
                "bottom_matrix": [[int(value) for value in row] for row in candidate["bottom_matrix"]],
                "top_matrix": [[int(value) for value in row] for row in candidate["top_matrix"]],
                "angle_deg": float(candidate["angle_deg"]),
                "strain": _strain_magnitude(candidate["strain"]),
                "principal_strains": [float(value) for value in candidate["strain"]],
                "bottom_layer_strain": [float(value) for value in candidate["bottom_layer_strain"]],
                "top_layer_strain": [float(value) for value in candidate["top_layer_strain"]],
                "bottom_strain": _strain_magnitude(candidate["bottom_layer_strain"]),
                "top_strain": _strain_magnitude(candidate["top_layer_strain"]),
                "sharing_fraction": float(candidate["sharing_fraction"]),
                "bottom_atom_count": int(candidate["bottom_atom_count"]),
                "top_atom_count": int(candidate["top_atom_count"]),
                "total_atoms": int(candidate["atom_count"]),
                "cell_a": float(candidate["moire_a"]),
                "cell_b": float(candidate["moire_b"]),
                "cell_gamma_deg": float(candidate["moire_gamma_deg"]),
                "surface_area": _cell_area(shared_lattice),
                "shared_lattice": [[float(value) for value in row] for row in shared_lattice],
            }
        )
    return entries


def match_order_key(entry: Mapping[str, Any]) -> tuple[float, int, float, float]:
    """Return the ranking key of one match: strain, then size, then area, then twist.

    The strain enters at the resolution of :data:`STRAIN_ORDER_RESOLUTION`, so
    matches that are equally strained to within floating-point noise are ordered
    by the quantities a user actually chooses between.  See
    ``Cellstine.strain_lt_or_tie`` in ``aristotle-lean-reference/RequestProject/MatchOrdering.lean``.
    """

    quantum = float(STRAIN_ORDER_RESOLUTION)
    return (
        round(float(entry["strain"]) / quantum) * quantum,
        int(entry["total_atoms"]),
        round(float(entry["surface_area"]), 9),
        abs(float(entry["angle_deg"])),
    )


def sort_matches(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the matches ordered by strain, then size, then cell area."""

    ordered = sorted((dict(entry) for entry in entries), key=match_order_key)
    for index, entry in enumerate(ordered, start=1):
        entry["index"] = index
    return ordered


def build_match_document(
    *,
    bottom_bulk: str | Path,
    top_bulk: str | Path,
    request: MatchRequest,
    entries: Sequence[Mapping[str, Any]],
    vacuum: float,
    bottom_millers: Sequence[str],
    top_millers: Sequence[str],
    bottom_layers_list: Sequence[int],
    top_layers_list: Sequence[int],
) -> dict[str, Any]:
    """Assemble and validate the match document written to ``matches.json``."""

    bottom_budget, top_budget = request.strain_budgets
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "search": {
            "bottom_bulk": str(Path(bottom_bulk).resolve()),
            "top_bulk": str(Path(top_bulk).resolve()),
            "bottom_millers": [str(value) for value in bottom_millers],
            "top_millers": [str(value) for value in top_millers],
            "bottom_layers_list": [int(value) for value in bottom_layers_list],
            "top_layers_list": [int(value) for value in top_layers_list],
            "vacuum": float(vacuum),
            "max_length": float(request.max_length),
            "max_strain": float(request.max_strain),
            "strain_mode": str(request.strain_mode),
            "bottom_strain_budget": float(bottom_budget),
            "top_strain_budget": float(top_budget),
            "min_length": None if request.min_length is None else float(request.min_length),
            "max_atoms": None if request.max_atoms is None else int(request.max_atoms),
            "max_aspect_ratio": float(request.max_aspect_ratio),
            "min_cell_angle_deg": float(request.min_cell_angle_deg),
            "max_cell_angle_deg": float(request.max_cell_angle_deg),
        },
        "matches": sort_matches(entries),
    }
    _validate_document(payload)
    return payload


_MATCH_FIELDS = {
    "index",
    "bottom_miller",
    "top_miller",
    "bottom_layers",
    "top_layers",
    "bottom_slab",
    "top_slab",
    "results_json",
    "candidate_index",
    "bottom_matrix",
    "top_matrix",
    "angle_deg",
    "strain",
    "principal_strains",
    "bottom_layer_strain",
    "top_layer_strain",
    "bottom_strain",
    "top_strain",
    "sharing_fraction",
    "bottom_atom_count",
    "top_atom_count",
    "total_atoms",
    "cell_a",
    "cell_b",
    "cell_gamma_deg",
    "surface_area",
    "shared_lattice",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _pair(value: Any, name: str) -> tuple[float, float]:
    """Return a two-component vector of finite numbers."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must hold exactly two finite numbers")
    return _finite(value[0], name), _finite(value[1], name)


def _integer_matrix_determinant(value: Any, name: str) -> int:
    """Return the determinant of a 2x2 integer matrix, refusing anything else."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2x2 integer matrix")
    entries: list[int] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{name} must be a 2x2 integer matrix")
        for item in row:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{name} must be a 2x2 integer matrix")
            entries.append(int(item))
    return entries[0] * entries[3] - entries[1] * entries[2]


def _validate_match(entry: Mapping[str, Any], expected_index: int, search: Mapping[str, Any]) -> None:
    missing = sorted(_MATCH_FIELDS.difference(entry))
    if missing:
        raise ValueError(f"match {expected_index} is missing fields: {', '.join(missing)}")
    if int(entry["index"]) != expected_index:
        raise ValueError("match indexes must be consecutive and one-based")
    for name in ("bottom_matrix", "top_matrix"):
        if _integer_matrix_determinant(entry[name], f"match {expected_index}.{name}") == 0:
            raise ValueError(f"match {expected_index}.{name} must be nonsingular")
    strain = _finite(entry["strain"], f"match {expected_index}.strain")
    if strain < 0.0:
        raise ValueError(f"match {expected_index}.strain must be nonnegative")
    for name, budget_name in (("bottom_strain", "bottom_strain_budget"), ("top_strain", "top_strain_budget")):
        realised = _finite(entry[name], f"match {expected_index}.{name}")
        if realised > float(search[budget_name]) + 1e-9:
            raise ValueError(f"match {expected_index}.{name} exceeds the {budget_name}")
    relative = _pair(entry["principal_strains"], f"match {expected_index}.principal_strains")
    top = _pair(entry["top_layer_strain"], f"match {expected_index}.top_layer_strain")
    bottom = _pair(entry["bottom_layer_strain"], f"match {expected_index}.bottom_layer_strain")
    for axis in (0, 1):
        difference = top[axis] - bottom[axis]
        if abs(difference - relative[axis]) > 1e-9 + 1e-7 * abs(relative[axis]):
            raise ValueError(
                f"match {expected_index} layer strains must differ by the relative principal strain"
            )
    for name in ("bottom_atom_count", "top_atom_count", "total_atoms"):
        if isinstance(entry[name], bool) or not isinstance(entry[name], int) or entry[name] <= 0:
            raise ValueError(f"match {expected_index}.{name} must be a positive integer")
    if int(entry["total_atoms"]) != int(entry["bottom_atom_count"]) + int(entry["top_atom_count"]):
        raise ValueError(f"match {expected_index}.total_atoms must equal the slab counts")
    for name in ("cell_a", "cell_b", "surface_area"):
        if _finite(entry[name], f"match {expected_index}.{name}") <= 0.0:
            raise ValueError(f"match {expected_index}.{name} must be positive")
    gamma = _finite(entry["cell_gamma_deg"], f"match {expected_index}.cell_gamma_deg")
    if not 0.0 < gamma < 180.0:
        raise ValueError(f"match {expected_index}.cell_gamma_deg must lie in (0, 180)")
    sharing = _finite(entry["sharing_fraction"], f"match {expected_index}.sharing_fraction")
    if not -1e-9 <= sharing <= 1.0 + 1e-9:
        raise ValueError(f"match {expected_index}.sharing_fraction must lie in [0, 1]")
    shared = entry["shared_lattice"]
    name = f"match {expected_index}.shared_lattice"
    if not isinstance(shared, (list, tuple)) or len(shared) != 2:
        raise ValueError(f"{name} must be a nonsingular 2x2 matrix")
    rows = [_pair(row, name) for row in shared]
    if abs(rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) <= 0.0:
        raise ValueError(f"{name} must be a nonsingular 2x2 matrix")


def _validate_document(document: Mapping[str, Any]) -> None:
    """Check one already-detached match document in place."""

    if document.get("schema") != SCHEMA:
        raise ValueError(f"matches schema must be '{SCHEMA}'")
    if document.get("version") != VERSION:
        raise ValueError(f"matches version must be {VERSION}")
    search = document.get("search")
    if not isinstance(search, Mapping):
        raise ValueError("matches.search must be a JSON object")
    if _finite(search.get("max_length"), "search.max_length") <= 0.0:
        raise ValueError("search.max_length must be positive")
    if _finite(search.get("max_strain"), "search.max_strain") <= 0.0:
        raise ValueError("search.max_strain must be positive")
    if search.get("strain_mode") not in {"shared", "film"}:
        raise ValueError("search.strain_mode must be 'shared' or 'film'")
    matches = document.get("matches")
    if not isinstance(matches, list):
        raise ValueError("matches must be a JSON array")
    for index, entry in enumerate(matches, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"match {index} must be a JSON object")
        _validate_match(entry, index, search)


def validate_matches(payload: Any) -> dict[str, Any]:
    """Validate and return a detached interface-match document."""

    if not isinstance(payload, Mapping):
        raise ValueError("matches document must be a JSON object")
    document = copy.deepcopy(dict(payload))
    _validate_document(document)
    return document


def write_matches(path: str | Path, payload: Any) -> Path:
    """Validate and write one interface-match document.

    Validation never mutates the payload, so the document is checked in place
    rather than copied: a match scan can hold tens of thousands of records and
    copying them twice on the way to disk costs more than the search itself.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("matches document must be a JSON object")
    document = dict(payload)
    _validate_document(document)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def read_matches(path: str | Path) -> dict[str, Any]:
    """Read and validate an interface-match document."""

    source = Path(path).resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not a valid interface match document") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("matches document must be a JSON object")
    document = dict(payload)
    _validate_document(document)
    return document


def select_match(document: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Return the one-based match ``index`` of a validated match document."""

    matches = list(document["matches"])
    if not matches:
        raise ValueError("the match document contains no matches")
    if not 1 <= int(index) <= len(matches):
        raise ValueError(f"match index must lie between 1 and {len(matches)}")
    return dict(matches[int(index) - 1])


def candidate_for_match(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated moire candidate record behind one match.

    Reading the candidate back from its own results file keeps the builder on the
    certified geometry produced by the search rather than on any value rewritten
    into the match summary.
    """

    document = read_results(entry["results_json"])
    for candidate in document["candidates"]:
        if int(candidate["index"]) == int(entry["candidate_index"]):
            return dict(candidate)
    raise ValueError(
        f"candidate {entry['candidate_index']} is missing from {entry['results_json']}"
    )


def format_matches_table(entries: Sequence[Mapping[str, Any]], limit: int | None = 10) -> str:
    """Return a compact human-readable table of the best matches."""

    shown = list(entries) if limit is None else list(entries)[: max(int(limit), 0)]
    if not shown:
        return "No commensurate match satisfied the requested limits."
    header = (
        " idx  bottom  top     angle (deg)   cell a x b (Ang)   gamma  "
        "atoms (bot/top/total)  strain (%)  bottom (%)  top (%)"
    )
    lines = [header, "-" * len(header)]
    for entry in shown:
        bottom_miller = "".join(str(value) for value in entry["bottom_miller"])
        top_miller = "".join(str(value) for value in entry["top_miller"])
        lines.append(
            f"{int(entry['index']):4d}  {bottom_miller:>6}  {top_miller:>6}  "
            f"{float(entry['angle_deg']):11.4f}   "
            f"{float(entry['cell_a']):7.3f} x {float(entry['cell_b']):7.3f}  "
            f"{float(entry['cell_gamma_deg']):6.2f}  "
            f"{int(entry['bottom_atom_count']):6d}/{int(entry['top_atom_count']):5d}/"
            f"{int(entry['total_atoms']):6d}  "
            f"{100.0 * float(entry['strain']):10.4f}  "
            f"{100.0 * float(entry['bottom_strain']):10.4f}  "
            f"{100.0 * float(entry['top_strain']):7.4f}"
        )
    remaining = len(entries) - len(shown)
    if remaining > 0:
        lines.append(f"... {remaining} more match(es) not shown.")
    return "\n".join(lines)

"""High-symmetry k-points and band-structure paths.

A band structure is a plot of the eigenvalues along a path through the Brillouin
zone, and the path is not free: it is supposed to visit the points and lines
that the symmetry of the crystal singles out.  This module derives those points
from the symmetry itself rather than looking them up.

Wavevectors are carried, as in :mod:`cellstine.core.reciprocal`, in *fractional*
reciprocal coordinates: a row ``k`` means the Cartesian wavevector ``k @ B`` with
``B = 2 pi inv(A).T``.  A crystal operation ``x -> W x + w`` on column fractional
coordinates acts on them by the integer matrix ``W^-1`` on the right,
``k -> k W^-1``, and time reversal by ``k -> -k``.  Write ``P`` for the resulting
group of integer matrices acting on the right.

**Strata.**  The little co-group of ``k`` is ``L(k) = { M in P : k M = k mod Z^3 }``
and the set of wavevectors sharing it is an affine subspace through ``k`` whose
direction space is

.. code-block:: text

    V(k) = { v : v M = v for every M in L(k) },

so ``dim V(k)`` says what kind of object ``k`` lies on: ``0`` an isolated
high-symmetry point, ``1`` a symmetry line, ``2`` a mirror plane, ``3`` a generic
point.  The points of interest are the strata of dimension zero, together with
the ends of the symmetry lines, which is where a line meets the zone boundary.

**Why a finite search finds them all.**  If ``V(k) = 0`` then the average
``(1/|L|) sum_M M`` projects onto ``V(k)`` and so vanishes, hence
``sum_M k (M - I) = -|L| k`` is an integer vector: every zero-dimensional
stratum has coordinates with denominator dividing ``|L|``.  Subgroup orders of
the crystallographic point groups (with time reversal) divide ``48``, so the
search over the grid of denominators ``48`` in this module is exhaustive, not a
sampling.

**Names.**  The familiar letters -- ``X``, ``L``, ``W``, ``K``, ``M``, ``A`` --
belong to the *conventional* cell of the lattice, so they are assigned by
matching each point, as a whole symmetry orbit, against the standard coordinates
of its Bravais type from :mod:`cellstine.core.bravais`.  Points that no standard
name covers are named ``P1``, ``P2``, ... in order of increasing ``|k|``, and are
marked as derived.  Two names for the same orbit -- fcc ``K`` and ``U`` differ by
a reciprocal lattice vector -- are reported as aliases of one point, because a
band structure cannot tell them apart.  They are still different *places* in the
zone: ``U`` sits on a square face and ``K`` on the edge between two hexagons, so
the lines through them are different lines, and a path naming both is walked
through both.

**The path.**  Nodes are the points above, taken as actual points of the zone;
edges are the pieces of symmetry line between consecutive nodes on it.  A walk
is then grown from Gamma, always stepping to the nearest point whose name has
not been visited, along a symmetry line when one is available and in a straight
line when none is.  An explicit path may be given instead, which is what to do
when a particular convention is wanted.

**Segments.**  Every segment walked is classified by where its *interior* lies,
not by the names of its ends: dimension one is a symmetry line, two a mirror
plane, three a plain chord of the zone.  Doing it by name would misread any
segment whose end carries an alias.

The formal statements behind this module are in ``aristotle-lean-reference/RequestProject/KPath.lean``,
and ``core/KPATH.md`` is the prose account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import strata
from .bravais import ConventionalCell, conventional_cell
from .brillouin import WignerSeitzCell, wigner_seitz_cell
from .reciprocal import reciprocal_lattice
from .reduction import as_lattice
from .strata import GRID_DENOMINATOR, kspace_operations, stratum_dimension
from .symmetry3d import lattice_point_group

__all__ = [
    "GRID_DENOMINATOR",
    "KPoint",
    "BandPath",
    "kspace_operations",
    "stratum_dimension",
    "special_points",
    "band_path",
    "parse_path",
]


_GAMMA = "GAMMA"

# Standard names, in *conventional* reciprocal fractional coordinates, of the
# high-symmetry points of each Bravais type.  Only names whose coordinates are
# fixed by the Bravais type appear; parameter-dependent points of the centred
# orthorhombic, tetragonal, monoclinic and rhombohedral zones are named by the
# derived scheme instead.
_STANDARD_LABELS: dict[str, tuple[tuple[str, tuple[float, float, float]], ...]] = {
    "cP": (("X", (0.0, 0.5, 0.0)), ("M", (0.5, 0.5, 0.0)), ("R", (0.5, 0.5, 0.5))),
    "cF": (
        ("X", (0.0, 0.0, 1.0)),
        ("L", (0.5, 0.5, 0.5)),
        ("W", (0.5, 0.0, 1.0)),
        ("K", (0.75, 0.75, 0.0)),
        ("U", (0.25, 0.25, 1.0)),
    ),
    "cI": (("H", (0.0, 0.0, 1.0)), ("P", (0.5, 0.5, 0.5)), ("N", (0.5, 0.5, 0.0))),
    "tP": (
        ("X", (0.0, 0.5, 0.0)),
        ("M", (0.5, 0.5, 0.0)),
        ("Z", (0.0, 0.0, 0.5)),
        ("R", (0.0, 0.5, 0.5)),
        ("A", (0.5, 0.5, 0.5)),
    ),
    "tI": (
        ("X", (0.5, 0.5, 0.0)),
        ("M", (0.0, 0.0, 1.0)),
        ("N", (0.0, 0.5, 0.5)),
        ("P", (0.5, 0.5, 0.5)),
    ),
    "oP": (
        ("X", (0.5, 0.0, 0.0)),
        ("Y", (0.0, 0.5, 0.0)),
        ("Z", (0.0, 0.0, 0.5)),
        ("S", (0.5, 0.5, 0.0)),
        ("U", (0.5, 0.0, 0.5)),
        ("T", (0.0, 0.5, 0.5)),
        ("R", (0.5, 0.5, 0.5)),
    ),
    "hP": (
        ("M", (0.5, 0.0, 0.0)),
        ("K", (1.0 / 3.0, 1.0 / 3.0, 0.0)),
        ("A", (0.0, 0.0, 0.5)),
        ("L", (0.5, 0.0, 0.5)),
        ("H", (1.0 / 3.0, 1.0 / 3.0, 0.5)),
    ),
    "hR": (("T", (0.0, 0.0, 1.5)), ("L", (0.0, 0.5, 0.5)), ("F", (0.0, -0.5, 1.0))),
    "aP": (
        ("X", (0.5, 0.0, 0.0)),
        ("Y", (0.0, 0.5, 0.0)),
        ("Z", (0.0, 0.0, 0.5)),
        ("L", (0.5, 0.5, 0.0)),
        ("M", (0.0, 0.5, 0.5)),
        ("N", (0.5, 0.0, 0.5)),
        ("R", (0.5, 0.5, 0.5)),
    ),
}


# The conventional band paths of the Bravais types whose zone has no free
# parameter.  They are used as the default when the lattice is one of these,
# and the derived walk is used otherwise; either way the coordinates come from
# the symmetry analysis, only the order of the visits is taken from here.
_STANDARD_PATHS: dict[str, str] = {
    "cP": "GAMMA-X-M-GAMMA-R-X|M-R",
    "cF": "GAMMA-X-W-K-GAMMA-L-U-W-L-K|U-X",
    "cI": "GAMMA-H-N-GAMMA-P-H|P-N",
    "tP": "GAMMA-X-M-GAMMA-Z-R-A-Z|X-R|M-A",
    "oP": "GAMMA-X-S-Y-GAMMA-Z-U-R-T-Z|Y-T|U-X|S-R",
    "hP": "GAMMA-M-K-GAMMA-A-L-H-A|L-M|K-H",
}


@dataclass(frozen=True)
class KPoint:
    """One high-symmetry point of the zone, with its name and coordinates."""

    label: str
    fractional: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    little_group_order: int
    stratum_dimension: int
    derived_label: bool
    aliases: tuple[str, ...] = ()

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.cartesian, dtype=float)))

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "aliases": list(self.aliases),
            "fractional": [float(value) for value in self.fractional],
            "cartesian": [float(value) for value in self.cartesian],
            "length": self.length,
            "little_group_order": self.little_group_order,
            "stratum_dimension": self.stratum_dimension,
            "derived_label": self.derived_label,
        }


@dataclass(frozen=True)
class BandPath:
    """A path through the Brillouin zone, ready to be written as k-points."""

    lattice: np.ndarray
    reciprocal: np.ndarray
    zone: WignerSeitzCell
    bravais: str
    points: tuple[KPoint, ...]
    walk: tuple[tuple[str, ...], ...]
    walk_points: tuple[np.ndarray, ...]
    time_reversal: bool
    segment_strata: tuple[int, ...]
    path_source: str

    @property
    def segment_symmetry(self) -> tuple[bool, ...]:
        """Return which segments run along a symmetry line.

        A segment whose interior lies on a one-dimensional stratum runs along a
        symmetry line; :attr:`segment_strata` also tells apart the segments that
        lie in a mirror plane (dimension two) from the ones that cross the zone
        with no symmetry at all (dimension three).
        """

        return tuple(dimension == 1 for dimension in self.segment_strata)

    @property
    def segments(self) -> tuple[tuple[str, str], ...]:
        """Return the ordered ``(start, end)`` label pairs of the path."""

        pairs: list[tuple[str, str]] = []
        for run in self.walk:
            for position in range(len(run) - 1):
                pairs.append((run[position], run[position + 1]))
        return tuple(pairs)

    def segment_lengths(self) -> tuple[float, ...]:
        """Return the Cartesian length of each segment, in inverse angstrom."""

        lengths: list[float] = []
        for run in self.walk_points:
            steps = np.asarray(run, dtype=float) @ self.reciprocal
            lengths.extend(float(value) for value in np.linalg.norm(np.diff(steps, axis=0), axis=1))
        return tuple(lengths)

    @property
    def length(self) -> float:
        """Return the total length walked, in inverse angstrom."""

        return float(sum(self.segment_lengths()))

    def divisions_for_spacing(self, spacing: float) -> int:
        """Return the per-segment division count that samples no coarser than ``spacing``.

        A line-mode k-point file carries one division count for every segment,
        so the count is set by the longest segment; the shorter ones are then
        sampled more finely, never less.
        """

        value = float(spacing)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("the k-point spacing must be a finite positive length in 1/angstrom")
        lengths = self.segment_lengths()
        if not lengths:  # pragma: no cover - defensive
            return 2
        longest = max(lengths)
        return max(2, int(math.ceil(longest / value + 1.0 - 1e-12)))

    def sample(self, divisions: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return the sampled path: fractional points, distances, and labels.

        Each segment carries ``divisions`` points including both ends; the point
        shared by two consecutive segments of a continuous run is written once.
        ``distances`` is the cumulative Cartesian distance along the walk, the
        abscissa of a band plot, and it does not advance across a break.
        """

        count = int(divisions)
        if count < 2:
            raise ValueError("a segment needs at least two points")
        points: list[np.ndarray] = []
        distances: list[float] = []
        labels: list[str] = []
        travelled = 0.0
        for run, run_points in zip(self.walk, self.walk_points):
            coordinates = np.asarray(run_points, dtype=float)
            for position in range(len(run) - 1):
                start = coordinates[position]
                end = coordinates[position + 1]
                span = float(np.linalg.norm((end - start) @ self.reciprocal))
                first = 0 if position == 0 else 1
                for step in range(first, count):
                    fraction = step / (count - 1)
                    points.append(start + fraction * (end - start))
                    distances.append(travelled + fraction * span)
                    if step == 0:
                        labels.append(run[position])
                    elif step == count - 1:
                        labels.append(run[position + 1])
                    else:
                        labels.append("")
                travelled += span
        return np.asarray(points, dtype=float), np.asarray(distances, dtype=float), tuple(labels)

    def path_string(self) -> str:
        """Return the path as ``GAMMA-X-W|K-L`` style text."""

        return "|".join("-".join(run) for run in self.walk)

    def summary(self) -> dict[str, object]:
        """Return a JSON-ready description of the path."""

        return {
            "bravais_symbol": self.bravais,
            "time_reversal": self.time_reversal,
            "path": self.path_string(),
            "path_source": self.path_source,
            "point_count": len(self.points),
            "segment_count": len(self.segments),
            "total_length": self.length,
            "points": [point.summary() for point in self.points],
            "segments": [
                {
                    "start": start,
                    "end": end,
                    "length": length,
                    "symmetry_line": dimension == 1,
                    "stratum_dimension": int(dimension),
                }
                for (start, end), length, dimension in zip(
                    self.segments, self.segment_lengths(), self.segment_strata
                )
            ],
            "zone": self.zone.summary(),
        }


def parse_path(text: str) -> tuple[tuple[str, ...], ...]:
    """Return the runs of a path written as ``GAMMA-X-W|K-L``."""

    runs: list[tuple[str, ...]] = []
    for piece in str(text).split("|"):
        labels = [item.strip() for item in piece.split("-") if item.strip()]
        if len(labels) < 2:
            raise ValueError("every piece of a path needs at least two points")
        runs.append(tuple(labels))
    if not runs:
        raise ValueError("a path must name at least two points")
    return tuple(runs)


def _zone_translates(point: np.ndarray, reciprocal: np.ndarray, zone: WignerSeitzCell, radius: int = 2) -> np.ndarray:
    """Return the translates of ``point`` (fractional) inside the closed zone."""

    span = np.arange(-radius, radius + 1, dtype=np.int64)
    offsets = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    candidates = point[None, :] + offsets
    cartesian = candidates @ reciprocal
    inside = zone.contains(cartesian, tolerance=1e-7 * float(np.linalg.norm(reciprocal[0])))
    return candidates[inside]


def _zone_representative(point: np.ndarray, reciprocal: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return the translate of ``point`` (fractional) with the shortest wavevector."""

    span = np.arange(-radius, radius + 1, dtype=np.int64)
    offsets = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    candidates = point[None, :] + offsets
    lengths = np.linalg.norm(candidates @ reciprocal, axis=1)
    return candidates[int(np.argmin(lengths))]


def _equivalent(first: np.ndarray, second: np.ndarray, operations: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Return whether two wavevectors are the same point of the band structure."""

    images = np.einsum("j,kjl->kl", np.asarray(first, dtype=float), operations.astype(float))
    residual = images - np.asarray(second, dtype=float)[None, :]
    return bool(np.any(np.all(np.abs(residual - np.rint(residual)) <= tolerance, axis=1)))


def _little_group_order(point: np.ndarray, operations: np.ndarray, tolerance: float = 1e-8) -> int:
    images = np.einsum("j,kjl->kl", np.asarray(point, dtype=float), operations.astype(float))
    residual = images - np.asarray(point, dtype=float)[None, :]
    return int(np.sum(np.all(np.abs(residual - np.rint(residual)) <= tolerance, axis=1)))


def _standard_targets(cell: ConventionalCell) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the tabulated labels of ``cell`` in primitive reciprocal coordinates.

    The table itself is written in *conventional* reciprocal coordinates, which
    is how the literature quotes it; the transpose inverse of the
    primitive-from-conventional matrix carries those to the primitive basis the
    rest of this module works in.
    """

    table = _STANDARD_LABELS.get(cell.symbol, ())
    conversion = np.linalg.inv(cell.to_primitive.T)
    return tuple(
        (label, np.asarray(coordinates, dtype=float) @ conversion) for label, coordinates in table
    )


def _standard_names(
    representatives: Sequence[np.ndarray],
    cell: ConventionalCell,
    operations: np.ndarray,
) -> list[tuple[str, ...]]:
    """Return the standard names, if any, of each orbit representative."""

    targets = _standard_targets(cell)
    return [
        tuple(label for label, target in targets if _equivalent(point, target, operations))
        for point in representatives
    ]


def special_points(
    lattice: Sequence[Sequence[float]],
    rotations: Sequence[Sequence[Sequence[int]]] | None = None,
    *,
    time_reversal: bool = True,
) -> tuple[tuple[KPoint, ...], dict[str, list[np.ndarray]]]:
    """Return the high-symmetry points of the zone and their zone copies.

    The second element maps each label to the list of *actual* points of the
    zone carrying it, in fractional reciprocal coordinates; a band path walks
    between those, while the :class:`KPoint` list carries one representative
    each.
    """

    basis = as_lattice(np.asarray(lattice, dtype=float), "lattice")
    if rotations is None:
        rotations = lattice_point_group(basis)
    operations = kspace_operations(rotations, time_reversal=time_reversal)
    reciprocal = reciprocal_lattice(basis)
    zone = wigner_seitz_cell(reciprocal)
    scale = float(np.linalg.norm(reciprocal[0]))

    grid, dimensions, directions = strata.grid_strata(operations)

    nodes: list[np.ndarray] = []
    cartesians: list[np.ndarray] = []

    def add(point: np.ndarray) -> int:
        cartesian = point @ reciprocal
        for index, other in enumerate(cartesians):
            if float(np.linalg.norm(other - cartesian)) <= 1e-6 * scale:
                return index
        nodes.append(np.asarray(point, dtype=float))
        cartesians.append(cartesian)
        return len(nodes) - 1

    for point in grid[dimensions == 0] / GRID_DENOMINATOR:
        for translate in _zone_translates(point, reciprocal, zone):
            add(translate)

    chords: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    for position in np.flatnonzero(dimensions == 1):
        direction = directions[int(position)]
        if direction is None:  # pragma: no cover - defensive
            continue
        point = _zone_representative(grid[int(position)] / GRID_DENOMINATOR, reciprocal)
        cartesian = point @ reciprocal
        if not zone.contains(cartesian[None, :], tolerance=1e-7 * scale)[0]:
            continue
        ray = direction.astype(float) @ reciprocal
        projections = zone.face_vectors @ ray
        room = zone.face_offsets - zone.face_vectors @ cartesian
        forward = projections > 1e-12 * scale
        backward = projections < -1e-12 * scale
        if not np.any(forward) or not np.any(backward):  # pragma: no cover - defensive
            continue
        upper = float(np.min(room[forward] / projections[forward]))
        lower = float(np.max(room[backward] / projections[backward]))
        start = point + lower * direction
        end = point + upper * direction
        key = tuple(np.round(np.concatenate([start @ reciprocal, end @ reciprocal]) / scale, 6))
        if key in chords:
            continue
        chords[key] = (start, end)
        add(start)
        add(end)

    cell = conventional_cell(basis)

    # Group the nodes into orbits, then name each orbit.
    orbits: list[list[int]] = []
    for index, point in enumerate(nodes):
        for orbit in orbits:
            if _equivalent(point, nodes[orbit[0]], operations):
                orbit.append(index)
                break
        else:
            orbits.append([index])

    def orbit_key(orbit: Sequence[int]) -> tuple[float, tuple[float, ...]]:
        best = min(orbit, key=lambda index: float(np.linalg.norm(cartesians[index])))
        return (
            round(float(np.linalg.norm(cartesians[best])) / scale, 9),
            tuple(float(value) for value in np.round(nodes[best], 6)),
        )

    orbits.sort(key=orbit_key)
    def representative_key(index: int) -> tuple[float, int, tuple[float, ...]]:
        point = nodes[index]
        return (
            round(float(np.linalg.norm(cartesians[index])) / scale, 9),
            int(np.sum(point < -1e-12)),
            tuple(float(-value) for value in point),
        )

    representatives = [min(orbit, key=representative_key) for orbit in orbits]
    names = _standard_names([nodes[index] for index in representatives], cell, operations)

    points: list[KPoint] = []
    copies: dict[str, list[np.ndarray]] = {}
    used: set[str] = set()
    derived_count = 0
    for orbit, index, matched in zip(orbits, representatives, names):
        point = nodes[index]
        if float(np.linalg.norm(cartesians[index])) <= 1e-9 * scale:
            label, aliases, derived = _GAMMA, (), False
        elif matched:
            label, aliases, derived = matched[0], tuple(matched[1:]), False
        else:
            derived_count += 1
            label = f"P{derived_count}"
            while label in used:  # pragma: no cover - defensive
                derived_count += 1
                label = f"P{derived_count}"
            aliases, derived = (), True
        used.add(label)
        points.append(
            KPoint(
                label=label,
                fractional=tuple(float(value) for value in point),
                cartesian=tuple(float(value) for value in cartesians[index]),
                little_group_order=_little_group_order(point, operations),
                stratum_dimension=stratum_dimension(point, operations),
                derived_label=derived,
                aliases=aliases,
            )
        )
        copies[label] = [nodes[member] for member in orbit]

    return tuple(points), copies


def band_path(
    lattice: Sequence[Sequence[float]],
    rotations: Sequence[Sequence[Sequence[int]]] | None = None,
    *,
    time_reversal: bool = True,
    path: str | Sequence[Sequence[str]] | None = None,
    use_standard: bool = True,
) -> BandPath:
    """Return a band-structure path through the Brillouin zone of ``lattice``.

    ``rotations`` are the point-group operations to use, as integer matrices on
    column fractional coordinates; the point group of the lattice is used when
    they are not given, which is the right default for a path that only has to
    respect the shape of the zone.

    ``path`` gives the walk explicitly, as ``"GAMMA-X-W|K-L"`` or as a sequence
    of runs of labels.  Otherwise the conventional walk of the Bravais type is
    used when there is one, and the walk derived from the symmetry lines when
    there is not; ``use_standard=False`` always derives it.  Which of the two
    was used is reported as :attr:`BandPath.path_source`.
    """

    basis = as_lattice(np.asarray(lattice, dtype=float), "lattice")
    if rotations is None:
        rotations = lattice_point_group(basis)
    operations = kspace_operations(rotations, time_reversal=time_reversal)
    reciprocal = reciprocal_lattice(basis)
    zone = wigner_seitz_cell(reciprocal)
    cell = conventional_cell(basis)
    points, copies = special_points(basis, rotations, time_reversal=time_reversal)

    # The nodes of the walk: every zone copy of every named point.
    node_labels: list[str] = []
    node_points: list[np.ndarray] = []
    for point in points:
        for copy in copies[point.label]:
            node_labels.append(point.label)
            node_points.append(np.asarray(copy, dtype=float))
    node_cartesian = np.asarray([item @ reciprocal for item in node_points], dtype=float)

    alias_of = {alias: point.label for point in points for alias in point.aliases}
    known = {point.label for point in points}

    requested: tuple[tuple[str, ...], ...] | None = None
    source = "derived"
    if path is not None:
        requested = parse_path(path) if isinstance(path, str) else tuple(tuple(run) for run in path)
        source = "explicit"
        for run in requested:
            for label in run:
                if alias_of.get(label, label) not in known:
                    raise ValueError(
                        f"unknown k-point label {label!r}; "
                        f"known labels are {sorted(known | set(alias_of))}"
                    )
    elif use_standard and cell.symbol in _STANDARD_PATHS:
        candidate = parse_path(_STANDARD_PATHS[cell.symbol])
        if all(alias_of.get(label, label) in known for run in candidate for label in run):
            requested = candidate
            source = "standard"

    if requested is None:
        # Edges along symmetry lines: consecutive nodes of a common line.  Only
        # a derived walk needs them, and they cost a pass over every pair of
        # nodes, so they are not built when a path was named or tabulated.
        symmetry_edges = strata.symmetry_edges(
            np.asarray(node_points, dtype=float),
            node_cartesian,
            operations,
            float(np.linalg.norm(reciprocal[0])),
        )
        walk, indices = _derive_walk(node_labels, node_cartesian, symmetry_edges)
        walk = _name_copies(walk, indices)
    else:
        walk = requested
        indices = _resolve_walk(
            walk,
            alias_of,
            node_labels,
            node_points,
            reciprocal,
            dict(_standard_targets(cell)),
            operations,
        )

    # A walk may use two different copies of the same orbit -- the ``L`` and
    # ``L1`` of the usual tables -- so every name it uses is listed with the
    # coordinates of the copy it actually visits.
    listed = {point.label for point in points}
    extra: list[KPoint] = []
    for run, run_indices in zip(walk, indices):
        for label, index in zip(run, run_indices):
            if label in listed:
                continue
            listed.add(label)
            parent = next(item for item in points if item.label == node_labels[index])
            point = node_points[index]
            extra.append(
                KPoint(
                    label=label,
                    fractional=tuple(float(value) for value in point),
                    cartesian=tuple(float(value) for value in point @ reciprocal),
                    little_group_order=parent.little_group_order,
                    stratum_dimension=parent.stratum_dimension,
                    derived_label=parent.derived_label,
                    aliases=(),
                )
            )
    points = points + tuple(extra)

    walk_points = tuple(
        np.asarray([node_points[index] for index in run], dtype=float) for run in indices
    )
    segment_strata = strata.segment_strata(walk_points, operations)
    return BandPath(
        lattice=basis,
        reciprocal=reciprocal,
        zone=zone,
        bravais=cell.symbol,
        points=points,
        walk=walk,
        walk_points=walk_points,
        time_reversal=time_reversal,
        segment_strata=segment_strata,
        path_source=source,
    )


def _name_copies(
    walk: Sequence[Sequence[str]],
    indices: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], ...]:
    """Return the walk with distinct copies of one orbit given distinct names.

    A path may visit two copies of the same point of the zone -- they carry the
    same bands but are different places -- and writing both as ``L`` would be
    unreadable, so the second becomes ``L_2``, the third ``L_3``, and so on.
    """

    naming: dict[int, str] = {}
    counters: dict[str, int] = {}
    for run, run_indices in zip(walk, indices):
        for label, index in zip(run, run_indices):
            if index in naming:
                continue
            counters[label] = counters.get(label, 0) + 1
            naming[index] = label if counters[label] == 1 else f"{label}_{counters[label]}"
    return tuple(tuple(naming[index] for index in run) for run in indices)


def _derive_walk(
    labels: Sequence[str],
    cartesian: np.ndarray,
    symmetry_edges: set[tuple[int, int]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return a walk visiting every label, following symmetry lines.

    The walk is grown from Gamma: at each step it moves, along symmetry lines,
    to the nearest point whose name it has not visited yet, passing through
    whatever lies on the way -- which is how a band path comes back through
    Gamma.  When no unvisited name can be reached along lines any more the walk
    is broken and restarted at the nearest one, the ``|`` of the usual notation.
    """

    count = len(labels)
    lengths = np.linalg.norm(cartesian[:, None, :] - cartesian[None, :, :], axis=2)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for first, second in symmetry_edges:
        neighbours[first].append((second, float(lengths[first, second])))
        neighbours[second].append((first, float(lengths[second, first])))

    start = int(np.argmin(np.linalg.norm(cartesian, axis=1)))
    visited = {labels[start]}
    remaining = {label for label in labels} - visited
    runs: list[list[int]] = [[start]]
    current = start
    while remaining:
        distance, previous = _dijkstra(neighbours, current, count)
        reachable = [
            index for index in range(count) if labels[index] in remaining and math.isfinite(distance[index])
        ]
        if reachable:
            target = min(reachable, key=lambda index: (distance[index], lengths[current, index]))
            chain: list[int] = []
            walker = target
            while walker != current:
                chain.append(walker)
                walker = previous[walker]
            for index in reversed(chain):
                runs[-1].append(index)
                visited.add(labels[index])
                remaining.discard(labels[index])
            current = target
            continue
        # Nothing left is reachable along a symmetry line, so the walk restarts
        # at Gamma and goes straight out to the nearest point still unvisited --
        # the spokes that a triclinic path is made of.
        candidates = [index for index in range(count) if labels[index] in remaining]
        target = min(candidates, key=lambda index: (lengths[start, index], index))
        if len(runs[-1]) == 2 and runs[-1][0] == start:
            runs[-1] = [runs[-1][1], start, target]
        else:
            runs.append([start, target])
        visited.add(labels[target])
        remaining.discard(labels[target])
        current = target
    runs = [run for run in runs if len(run) > 1]
    if not runs:  # pragma: no cover - a lattice always has more than one point
        raise RuntimeError("no band path could be built")
    return (
        tuple(tuple(labels[index] for index in run) for run in runs),
        tuple(tuple(run) for run in runs),
    )


def _dijkstra(
    neighbours: Sequence[Sequence[tuple[int, float]]],
    source: int,
    count: int,
) -> tuple[list[float], list[int]]:
    """Return the shortest distances and predecessors from ``source``."""

    distance = [math.inf] * count
    previous = [-1] * count
    distance[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    seen = set()
    while queue:
        queue.sort(reverse=True)
        current_distance, current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for neighbour, weight in neighbours[current]:
            candidate = current_distance + weight
            if candidate < distance[neighbour] - 1e-12:
                distance[neighbour] = candidate
                previous[neighbour] = current
                queue.append((candidate, neighbour))
    return distance, previous


def _resolve_walk(
    walk: Sequence[Sequence[str]],
    alias_of: dict[str, str],
    labels: Sequence[str],
    points: Sequence[np.ndarray],
    reciprocal: np.ndarray,
    targets: dict[str, np.ndarray] | None = None,
    operations: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return the zone copies to walk between for a path given by name.

    A label names a whole orbit, and the zone holds several copies of it; the
    copy taken is the one nearest -- in wavevector, not in fractional
    coordinates, so that a skewed zone is handled correctly -- the point already
    reached, so that a path written down by name is walked the short way round.
    Exact ties, which symmetry makes common, are broken towards the tidiest
    coordinates, so that the reported numbers are the ones a reader expects.

    ``targets`` carries the tabulated coordinates of the labels that have them,
    and ``operations`` the point group.  A label with a target is resolved among
    the copies that are point-group images of it *as wavevectors*, with no
    reciprocal lattice translation allowed.  That still leaves every symmetric
    copy to choose from, so the walk keeps taking the short way round, but it
    tells apart two names that a reciprocal lattice vector merges: fcc ``U`` is
    ``-K`` plus a reciprocal lattice vector, so the two carry the same bands
    while sitting on different faces of the zone -- ``U`` on a square face,
    ``K`` on the edge between two hexagons -- and the lines through them are
    different lines, which is why the standard path visits both.
    """

    cartesian = np.asarray([np.asarray(point, dtype=float) @ reciprocal for point in points])
    scale = float(np.linalg.norm(reciprocal[0]))

    def tidiness(item: int) -> tuple[int, tuple[float, ...]]:
        point = np.asarray(points[item], dtype=float)
        return (int(np.sum(point < -1e-12)), tuple(float(-value) for value in point))

    def distance_key(item: int, reference: np.ndarray | None) -> tuple[object, ...]:
        offset = cartesian[item] if reference is None else cartesian[item] - reference
        return (round(float(np.linalg.norm(offset)) / scale, 9),) + tidiness(item)

    resolved: list[tuple[int, ...]] = []
    for run in walk:
        indices: list[int] = []
        previous: np.ndarray | None = None
        for label in run:
            name = alias_of.get(label, label)
            options = [index for index in range(len(labels)) if labels[index] == name]
            if not options:  # pragma: no cover - guarded by the caller
                raise ValueError(f"no zone point carries the label {label!r}")
            target = None if targets is None else targets.get(label)
            if target is not None and operations is not None:
                images = np.einsum("j,kjl->kl", np.asarray(target, dtype=float), operations.astype(float))
                tabulated = [
                    index
                    for index in options
                    if float(
                        np.min(np.linalg.norm(images - np.asarray(points[index], dtype=float), axis=1))
                    )
                    <= 1e-6
                ]
                if tabulated:
                    options = tabulated
            reference = previous
            index = min(options, key=lambda item: distance_key(item, reference))
            indices.append(index)
            previous = cartesian[index]
        resolved.append(tuple(indices))
    return tuple(resolved)

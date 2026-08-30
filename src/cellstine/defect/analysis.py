"""Layer census, symmetry grouping, and record helpers for defect analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import geometry, symmetry3d
from ..core.layers import layer_partition
from ..io.models import StructureRecord
from .records import DefectAnalysis, DefectSite


def _normal_from_lattice(lattice: np.ndarray) -> np.ndarray:
    normal = np.cross(np.asarray(lattice, dtype=float)[0], np.asarray(lattice, dtype=float)[1])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("structure has a zero in-plane area")
    return normal / length


def _cluster_projection_layers(projections: np.ndarray, tolerance: float) -> list[dict[str, Any]]:
    """Return the atomic planes of the census, numbered from the bottom up.

    The grouping is ``core.layers.layer_partition``, the single rule the
    package uses everywhere; ``aristotle-lean-reference/RequestProject/LayerPartition.lean`` proves that
    it is the connected-component partition and that plane 1 is the bottom of
    the structure.
    """

    return [
        {
            "layer_id": int(layer_id),
            "projection": float(projection),
            "atom_indices": [int(index) + 1 for index in sorted(atom_indices)],
            "atom_count": int(len(atom_indices)),
        }
        for layer_id, (projection, atom_indices) in enumerate(
            layer_partition(projections, float(tolerance)), start=1
        )
    ]


def _annotate_layer_census(
    layers: Sequence[dict[str, Any]],
    species_by_atom: Sequence[str],
    atom_sites: Sequence[DefectSite],
) -> None:
    """Record, for every atomic plane, what it holds and how much of it is new.

    Each layer gains ``species_counts`` -- how many atoms of each species lie in
    that plane -- and ``inequivalent_sites`` -- how many of the symmetry-distinct
    atom sites of the whole cell have a representative in it.  An orbit that
    runs through several planes is counted in each of them, which is what a
    defect study needs: the entry for a plane is the number of genuinely
    different single-atom defects that can be made *in that plane*.
    """

    site_of_atom: dict[int, str] = {}
    for site in atom_sites:
        for atom_index in site.equivalent_indices:
            site_of_atom[int(atom_index)] = site.site_id
    for layer in layers:
        counts: dict[str, int] = {}
        site_ids: dict[str, set[str]] = {}
        for atom_index in layer.get("atom_indices", []):
            species = str(species_by_atom[int(atom_index) - 1])
            counts[species] = counts.get(species, 0) + 1
            site_id = site_of_atom.get(int(atom_index))
            if site_id is not None:
                site_ids.setdefault(species, set()).add(site_id)
        layer["species_counts"] = {key: counts[key] for key in sorted(counts)}
        layer["inequivalent_sites"] = {
            key: len(site_ids[key]) for key in sorted(site_ids)
        }


def _nearest_layer_id(projection: float, layers: Sequence[dict[str, Any]]) -> int | None:
    """The atomic plane a point between the planes is filed under.

    An interstitial does not sit *in* a plane, so it is attributed to the plane
    it is closest to along the direction of observation; that is the plane whose
    bonds it disturbs, and it is the reading the plane selection uses.
    """

    if not layers:
        return None
    best = min(layers, key=lambda layer: abs(float(layer["projection"]) - float(projection)))
    return int(best["layer_id"])


def _atom_members(
    atom_indices: Sequence[int],
    direct: np.ndarray,
    cartesian: np.ndarray,
    layer_lookup: dict[int, int],
) -> list[dict[str, Any]]:
    """Describe the atoms of one orbit, plane by plane.

    ``atom_indices`` are 0-based and the representative comes first, which is
    the order :mod:`cellstine.defect.layers` relies on when it splits the orbit
    over the planes.
    """

    members: list[dict[str, Any]] = []
    for atom_index in atom_indices:
        index = int(atom_index)
        layer_id = layer_lookup.get(index + 1)
        members.append(
            {
                "indices": [index + 1],
                "direct": [float(value) for value in direct[index]],
                "cartesian": [float(value) for value in cartesian[index]],
                "layer_ids": [] if layer_id is None else [int(layer_id)],
            }
        )
    return members


def _layer_lookup(layers: Sequence[dict[str, Any]]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for layer in layers:
        layer_id = int(layer.get("layer_id", 0))
        for atom_index in layer.get("atom_indices", []):
            lookup[int(atom_index)] = layer_id
    return lookup


def _detect_structure_kind(record: StructureRecord, requested: str, *, layer_tolerance: float) -> str:
    choice = str(requested or "auto").lower()
    if choice in {"surface", "slab"}:
        return "surface"
    if choice in {"bulk", "molecule-on-substrate", "molecule_on_substrate"}:
        return choice.replace("_", "-")
    if choice != "auto":
        return choice
    normal = _normal_from_lattice(record.lattice)
    projections = np.asarray(record.positions_cartesian, dtype=float) @ normal
    if projections.size == 0:
        return "bulk"
    height = abs(float(np.dot(np.asarray(record.lattice, dtype=float)[2], normal)))
    occupied = float(projections.max() - projections.min())
    inplane_lengths = [float(np.linalg.norm(np.asarray(record.lattice, dtype=float)[index])) for index in (0, 1)]
    typical_inplane = max(1e-12, float(np.median(inplane_lengths)))
    if max(0.0, height - occupied) >= max(3.0, 3.0 * float(layer_tolerance)) and height > 1.5 * typical_inplane:
        return "surface"
    return "bulk"


def _pairwise_minimum_image_distances(direct: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Full ``(n, n)`` matrix of minimum-image Cartesian distances.

    The shortest periodic image is found exactly, over every lattice shift that
    the reach bound allows.  Rounding the fractional difference to ``[-1/2, 1/2]``
    -- the textbook shortcut -- is wrong in a skewed cell: in a hexagonal
    supercell it reports some neighbour separations more than 30 % too large,
    which would put the wrong atoms in a fingerprint and drop genuine divacancy
    pairs.
    """

    direct = np.asarray(direct, dtype=float)
    if direct.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    return geometry.pairwise_minimum_image_distances(lattice, direct)


class _PlainOperation:
    """Minimal rotation/translation pair with the attribute names used below."""

    __slots__ = ("rotation", "translation")

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = rotation
        self.translation = translation


def _symmetry_groups_for_points(
    lattice: np.ndarray,
    points_direct: np.ndarray,
    dataset: Any | None,
    *,
    tolerance: float = 0.1,
    labels: Sequence[str] | None = None,
) -> list[list[int]]:
    """Group fractional points into orbits of the symmetry operations.

    Points are matched under the minimum-image convention with a Cartesian
    ``tolerance``.  Without a symmetry dataset every point forms its own group.

    ``labels`` optionally carries a tag per point -- what kind of site it is --
    and points with different tags are never merged, however close they lie.
    """

    lattice = np.asarray(lattice, dtype=float)
    points = np.mod(np.asarray(points_direct, dtype=float).reshape(-1, 3), 1.0)
    count = len(points)
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    if dataset is not None and count:
        # One bucket index of the points serves every operation, so grouping is
        # linear in the number of points per operation rather than quadratic.
        #
        # An orbit is a connected component of the graph drawn by the group
        # action, and a connected component does not change when the group is
        # replaced by a generating set.  A supercell carries one operation per
        # (rotation, lattice translation) pair -- tens of thousands of them --
        # while a few dozen generate them all, so the reduction is what makes
        # the analysis of a large cell finish at all.
        rotations, translations = symmetry3d.generating_operations(
            dataset.rotations, dataset.translations
        )
        finder = geometry.PeriodicSiteIndex(lattice, points, tolerance=float(tolerance))
        for rotation, translation in zip(rotations, translations):
            images = points @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
            targets = finder.match(images)
            for source in range(count):
                target = int(targets[source])
                if target < 0:
                    continue
                if labels is not None and labels[source] != labels[target]:
                    continue
                union(source, target)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for _, members in sorted(grouped.items())]


def _serialise_site_id(prefix: str, index: int) -> str:
    return f"{prefix}_{int(index):03d}"


def _load_analysis_file(path: Path) -> DefectAnalysis:
    with path.open("r", encoding="utf-8") as handle:
        return DefectAnalysis.from_dict(json.load(handle))


def _write_analysis_file(path: Path, analysis: DefectAnalysis) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(analysis.to_dict(), handle, indent=2)
    return path.resolve()


def _normalise_site_ids(site_ids: Sequence[str] | str | None) -> list[str] | None:
    if site_ids is None:
        return None
    if isinstance(site_ids, str) and site_ids == "":
        return None
    if isinstance(site_ids, str):
        values = [chunk.strip() for chunk in site_ids.split(",") if chunk.strip()]
    else:
        values = [str(value).strip() for value in site_ids if str(value).strip()]
    return values or None


def _normalise_supercell(supercell: Sequence[int] | str | None) -> tuple[int, int, int] | None:
    """Return the three host repeats requested, or ``None`` for the cell as read.

    A single integer means the same repeat along all three axes, and ``1x1x1``
    (however it is spelled) is the same as asking for no supercell at all.
    """

    if supercell is None:
        return None
    if isinstance(supercell, str):
        text = supercell.strip().lower().replace("x", ",").replace(" ", ",")
        if not text:
            return None
        values: list[str] = [chunk for chunk in text.split(",") if chunk]
    elif isinstance(supercell, (int, np.integer)):
        values = [str(int(supercell))] * 3
    else:
        values = [str(value).strip() for value in supercell if str(value).strip()]
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError("supercell must be one or three positive integers, e.g. 2,2,1")
    repeats: list[int] = []
    for value in values:
        try:
            count = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"supercell repeat '{value}' is not an integer") from error
        if count < 1:
            raise ValueError("supercell repeats must be at least 1")
        repeats.append(count)
    if repeats == [1, 1, 1]:
        return None
    return (repeats[0], repeats[1], repeats[2])


def _record_from_atoms(
    source: StructureRecord,
    atom_species: Sequence[str],
    direct_positions: np.ndarray,
    selective_flags: Sequence[tuple[str, str, str]] | None,
    *,
    comment: str,
    lattice: np.ndarray | None = None,
) -> StructureRecord:
    old_order = list(source.species)
    ordered_species: list[str] = []
    for symbol in old_order:
        if symbol in atom_species and symbol not in ordered_species:
            ordered_species.append(str(symbol))
    for symbol in atom_species:
        if str(symbol) not in ordered_species:
            ordered_species.append(str(symbol))

    grouped_direct: list[np.ndarray] = []
    grouped_flags: list[tuple[str, str, str]] = []
    counts: list[int] = []
    direct = np.asarray(direct_positions, dtype=float)
    flags = None if selective_flags is None else [tuple(item) for item in selective_flags]
    for symbol in ordered_species:
        indices = [index for index, atom_symbol in enumerate(atom_species) if str(atom_symbol) == symbol]
        counts.append(len(indices))
        for index in indices:
            grouped_direct.append(np.array(direct[index], dtype=float))
            if flags is not None:
                grouped_flags.append(tuple(flags[index]))

    cell = np.asarray(source.lattice if lattice is None else lattice, dtype=float)
    output_direct = np.asarray(grouped_direct, dtype=float) if grouped_direct else np.zeros((0, 3), dtype=float)
    output_cartesian = output_direct @ cell
    return StructureRecord(
        comment=comment,
        lattice=np.array(cell, dtype=float, copy=True),
        species=ordered_species,
        counts=counts,
        positions_direct=output_direct,
        positions_cartesian=output_cartesian,
        coordinate_mode=source.coordinate_mode,
        selective_dynamics=bool(source.selective_dynamics and flags is not None),
        selective_flags=None if flags is None else grouped_flags,
        source_path=source.source_path,
        source_format=source.source_format,
        metadata=dict(source.metadata),
    )


def _safe_species_token(*values: str | None) -> str:
    tokens = []
    for value in values:
        if value:
            tokens.append("".join(char for char in str(value) if char.isalnum()) or "X")
    return "_".join(tokens) if tokens else "site"

"""Defect analysis and structure-generation workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.base import Base, run_output_suffix
from ..core.manifests import RunManifest
from ..core.models import CommandResult
from ..interface import surface_backend
from ..io.converters import StructureConverter
from ..io.models import StructureRecord
from ..io.vasp import VaspIO
from ..symmetry.symmetry import Symmetry


@dataclass
class DefectSite:
    """One representative defect site in a structure."""

    site_id: str
    species: str | None
    layer_id: int | None
    direct: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    equivalent_indices: list[int] = field(default_factory=list)
    multiplicity: int = 1
    wyckoff: str | None = None
    site_kind: str = "atom"
    backend: str = "native"
    representative_index: int | None = None
    site_family: str | None = None
    pair_indices: list[int] = field(default_factory=list)


@dataclass
class DefectAnalysis:
    """Serializable defect analysis result."""

    structure_path: str
    structure_kind: str
    backend: str
    atom_count: int
    species: list[str]
    counts: list[int]
    layers: list[dict[str, Any]]
    sites: list[DefectSite]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cellstine.defect_analysis.v1",
            "structure_path": self.structure_path,
            "structure_kind": self.structure_kind,
            "backend": self.backend,
            "atom_count": self.atom_count,
            "species": list(self.species),
            "counts": [int(value) for value in self.counts],
            "layers": list(self.layers),
            "sites": [asdict(site) for site in self.sites],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DefectAnalysis":
        return cls(
            structure_path=str(payload["structure_path"]),
            structure_kind=str(payload.get("structure_kind", "auto")),
            backend=str(payload.get("backend", "native")),
            atom_count=int(payload.get("atom_count", 0)),
            species=[str(value) for value in payload.get("species", [])],
            counts=[int(value) for value in payload.get("counts", [])],
            layers=list(payload.get("layers", [])),
            sites=[DefectSite(**dict(site)) for site in payload.get("sites", [])],
            notes=[str(value) for value in payload.get("notes", [])],
        )


def _expanded_species(record: StructureRecord) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(record.species, record.counts):
        symbols.extend([str(symbol)] * int(count))
    return symbols


def _normal_from_lattice(lattice: np.ndarray) -> np.ndarray:
    normal = np.cross(np.asarray(lattice, dtype=float)[0], np.asarray(lattice, dtype=float)[1])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("structure has a zero in-plane area")
    return normal / length


def _cluster_projection_layers(projections: np.ndarray, tolerance: float) -> list[dict[str, Any]]:
    if projections.size == 0:
        return []
    order = np.argsort(projections)
    raw_groups: list[list[int]] = []
    current = [int(order[0])]
    last = float(projections[order[0]])
    for atom_index in order[1:]:
        projection = float(projections[atom_index])
        if abs(projection - last) <= float(tolerance):
            current.append(int(atom_index))
        else:
            raw_groups.append(current)
            current = [int(atom_index)]
        last = projection
    raw_groups.append(current)

    layers = []
    for layer_id, atom_indices in enumerate(raw_groups, start=1):
        values = projections[np.asarray(atom_indices, dtype=int)]
        layers.append(
            {
                "layer_id": int(layer_id),
                "projection": float(np.mean(values)),
                "atom_indices": [int(index) + 1 for index in sorted(atom_indices)],
                "atom_count": int(len(atom_indices)),
            }
        )
    return layers


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


def _minimum_image_delta(frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    delta = np.asarray(frac_a, dtype=float) - np.asarray(frac_b, dtype=float)
    return delta - np.round(delta)


def _pairwise_minimum_image_distances(direct: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Full ``(n, n)`` matrix of minimum-image Cartesian distances.

    Vectorised replacement for repeated per-pair ``_minimum_image_delta`` calls.
    """

    direct = np.asarray(direct, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if direct.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    diff = direct[:, None, :] - direct[None, :, :]
    diff -= np.round(diff)
    cart = diff @ lattice
    return np.sqrt(np.einsum("ijk,ijk->ij", cart, cart))


def _all_distance_fingerprints(
    record: StructureRecord,
    species_by_atom: Sequence[str],
    *,
    neighbours: int = 12,
    distance_matrix: np.ndarray | None = None,
) -> list[tuple[tuple[str, float], ...]]:
    """Batched, vectorised equivalent of :func:`_distance_fingerprint` for every atom.

    Produces the same per-atom fingerprint tuples (sorted by rounded distance then
    species) as calling ``_distance_fingerprint`` for each atom, but computes the
    whole ``(n, n)`` distance matrix once instead of an ``O(n^2)`` Python loop.
    """

    direct = np.asarray(record.positions_direct, dtype=float)
    lattice = np.asarray(record.lattice, dtype=float)
    n = len(species_by_atom)
    if n == 0:
        return []
    if distance_matrix is None:
        distance_matrix = _pairwise_minimum_image_distances(direct, lattice)
    dround = np.round(distance_matrix, 3)
    species_strings = [str(symbol) for symbol in species_by_atom]
    unique_sorted = sorted(set(species_strings))
    code_of = {symbol: index for index, symbol in enumerate(unique_sorted)}
    species_codes = np.fromiter((code_of[symbol] for symbol in species_strings), dtype=np.int64, count=n)
    keep = max(1, int(neighbours))
    indices = np.arange(n)
    fingerprints: list[tuple[tuple[str, float], ...]] = []
    for atom_index in range(n):
        mask = indices != atom_index
        others = indices[mask]
        d_row = dround[atom_index, mask]
        c_row = species_codes[mask]
        order = np.lexsort((c_row, d_row))[:keep]
        chosen = others[order]
        fingerprints.append(
            tuple((species_strings[int(j)], float(d_row[int(pos)])) for pos, j in zip(order, chosen))
        )
    return fingerprints


def _distance_fingerprint(
    record: StructureRecord,
    species_by_atom: Sequence[str],
    atom_index: int,
    *,
    neighbours: int = 12,
) -> tuple[tuple[str, float], ...]:
    direct = np.asarray(record.positions_direct, dtype=float)
    lattice = np.asarray(record.lattice, dtype=float)
    distances: list[tuple[str, float]] = []
    for other_index, other_species in enumerate(species_by_atom):
        if other_index == int(atom_index):
            continue
        delta = _minimum_image_delta(direct[atom_index], direct[other_index])
        distance = float(np.linalg.norm(delta @ lattice))
        distances.append((str(other_species), round(distance, 3)))
    distances.sort(key=lambda item: (item[1], item[0]))
    return tuple(distances[: max(1, int(neighbours))])


def _round_direct_key(values: Sequence[float], places: int = 4) -> tuple[float, float, float]:
    rounded = [round(float(value) % 1.0, int(places)) for value in values]
    while len(rounded) < 3:
        rounded.append(0.0)
    return tuple(rounded[:3])  # type: ignore[return-value]


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


def _record_from_atoms(
    source: StructureRecord,
    atom_species: Sequence[str],
    direct_positions: np.ndarray,
    selective_flags: Sequence[tuple[str, str, str]] | None,
    *,
    comment: str,
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

    output_direct = np.asarray(grouped_direct, dtype=float) if grouped_direct else np.zeros((0, 3), dtype=float)
    output_cartesian = output_direct @ np.asarray(source.lattice, dtype=float)
    return StructureRecord(
        comment=comment,
        lattice=np.array(source.lattice, dtype=float, copy=True),
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


class Defect(Base):
    """Analyse inequivalent defect sites and generate defect POSCARs."""

    workflow_name = "defect"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.vasp_io = VaspIO()

    def _choose_defect_backend(self, requested: str, structure_kind: str) -> str:
        choice = str(requested or self.backend or "auto").lower()
        if choice == "pymatgen":
            choice = "spglib"
        if choice not in {"auto", "native", "spglib"}:
            raise ValueError(f"unsupported backend '{requested}'")
        if choice == "native":
            return "native"
        if choice == "spglib":
            return self.dependency_manager.choose_symmetry_backend("spglib", feature="defect equivalence")
        if structure_kind == "bulk" and self.dependency_manager.has("spglib"):
            return "spglib"
        return "native"

    def _native_atom_sites(
        self,
        record: StructureRecord,
        *,
        structure_kind: str,
        layers: Sequence[dict[str, Any]],
    ) -> list[DefectSite]:
        species_by_atom = _expanded_species(record)
        direct = np.asarray(record.positions_direct, dtype=float)
        cartesian = np.asarray(record.positions_cartesian, dtype=float)
        layer_lookup = _layer_lookup(layers)
        fingerprints = _all_distance_fingerprints(record, species_by_atom)
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for atom_index, symbol in enumerate(species_by_atom):
            layer_id = layer_lookup.get(atom_index + 1)
            fingerprint = fingerprints[atom_index]
            if structure_kind == "bulk":
                key = (symbol, fingerprint)
            else:
                key = (symbol, layer_id, _round_direct_key(direct[atom_index][:2], 3), fingerprint)
            grouped.setdefault(key, []).append(atom_index)

        sites = []
        for site_index, atom_indices in enumerate(grouped.values(), start=1):
            representative = int(atom_indices[0])
            sites.append(
                DefectSite(
                    site_id=_serialise_site_id("atom", site_index),
                    species=str(species_by_atom[representative]),
                    layer_id=layer_lookup.get(representative + 1),
                    direct=tuple(float(value) for value in direct[representative]),
                    cartesian=tuple(float(value) for value in cartesian[representative]),
                    equivalent_indices=[int(value) + 1 for value in atom_indices],
                    multiplicity=int(len(atom_indices)),
                    site_kind="atom",
                    backend="native",
                    representative_index=representative + 1,
                )
            )
        return sites

    def _spglib_atom_sites(
        self,
        structure_path: str,
        record: StructureRecord,
        *,
        layers: Sequence[dict[str, Any]],
        symprec: float,
    ) -> list[DefectSite]:
        analysis = Symmetry(dependency_manager=self.dependency_manager).analyse_record(
            record,
            structure_path=str(Path(structure_path).resolve()),
            backend="spglib",
            symprec=float(symprec),
        )
        direct = np.asarray(record.positions_direct, dtype=float)
        cartesian = np.asarray(record.positions_cartesian, dtype=float)
        layer_lookup = _layer_lookup(layers)
        sites = []
        for site_index, group in enumerate(analysis.equivalent_groups, start=1):
            indices = sorted(int(value) - 1 for value in group.equivalent_indices)
            representative = int(group.representative_index) - 1
            sites.append(
                DefectSite(
                    site_id=_serialise_site_id("atom", site_index),
                    species=str(group.species),
                    layer_id=layer_lookup.get(representative + 1),
                    direct=tuple(float(value) for value in direct[representative]),
                    cartesian=tuple(float(value) for value in cartesian[representative]),
                    equivalent_indices=[int(value) + 1 for value in indices],
                    multiplicity=int(len(indices)),
                    wyckoff=group.wyckoff,
                    site_kind="atom",
                    backend="spglib",
                    representative_index=representative + 1,
                )
            )
        return sites

    def _interstitial_sites(self, record: StructureRecord) -> list[DefectSite]:
        direct = np.asarray(record.positions_direct, dtype=float)
        lattice = np.asarray(record.lattice, dtype=float)
        candidates = [
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ]
        sites = []
        for candidate in candidates:
            frac = np.asarray(candidate, dtype=float)
            if direct.size:
                distances = [float(np.linalg.norm(_minimum_image_delta(frac, atom) @ lattice)) for atom in direct]
                if min(distances) < 0.75:
                    continue
            cartesian = frac @ lattice
            sites.append(
                DefectSite(
                    site_id=_serialise_site_id("interstitial", len(sites) + 1),
                    species=None,
                    layer_id=None,
                    direct=tuple(float(value) for value in frac),
                    cartesian=tuple(float(value) for value in cartesian),
                    equivalent_indices=[],
                    multiplicity=1,
                    site_kind="interstitial",
                    backend="native",
                )
            )
        return sites

    def _adatom_sites(
        self,
        structure_path: str,
        *,
        surface_side: str,
        layer_tolerance: float,
    ) -> tuple[list[DefectSite], dict[str, int], str | None]:
        try:
            report = surface_backend.find_adsorption_sites(
                structure_path,
                surface_side=surface_side,
                layer_tolerance=layer_tolerance,
            )
        except Exception as exc:
            return [], {}, str(exc)
        sites = []
        type_counts: dict[str, int] = {}
        for site in report.sites:
            site_type = str(site.site_type)
            type_counts[site_type] = type_counts.get(site_type, 0) + 1
            site_index = type_counts[site_type]
            sites.append(
                DefectSite(
                    site_id=f"adatom_{site_type}_{site_index:03d}",
                    species=None,
                    layer_id=None,
                    direct=tuple(float(value) for value in site.direct),
                    cartesian=tuple(float(value) for value in site.cartesian),
                    equivalent_indices=[],
                    multiplicity=1,
                    site_kind="adatom",
                    backend="native",
                    site_family=site_type,
                )
            )
        return sites, type_counts, None

    def _divacancy_sites(
        self,
        record: StructureRecord,
        *,
        backend: str,
        symprec: float,
        divacancy_distance: float,
    ) -> list[DefectSite]:
        species_by_atom = _expanded_species(record)
        natoms = len(species_by_atom)
        if natoms < 2:
            return []

        lattice = np.asarray(record.lattice, dtype=float)
        direct = np.asarray(record.positions_direct, dtype=float)

        distance_matrix = _pairwise_minimum_image_distances(direct, lattice)
        row_idx, col_idx = np.triu_indices(natoms, k=1)
        pair_distances = distance_matrix[row_idx, col_idx]
        within = pair_distances <= float(divacancy_distance)
        pairs = [
            (int(i), int(j), float(d))
            for i, j, d in zip(row_idx[within], col_idx[within], pair_distances[within])
        ]

        if not pairs:
            return []

        symmetry_ops = []
        if backend == "spglib":
            try:
                sym_tool = Symmetry(dependency_manager=self.dependency_manager)
                sym_analysis = sym_tool.analyse_record(record, backend="spglib", symprec=symprec)
                symmetry_ops = sym_analysis.operations
            except Exception:
                pass

        grouped_pairs: list[list[tuple[int, int]]] = []

        if symmetry_ops:
            atom_maps = []
            for op in symmetry_ops:
                rot = np.asarray(op.rotation, dtype=float)
                trans = np.asarray(op.translation, dtype=float)
                rot_coords_wrapped = np.mod(direct @ rot.T + trans, 1.0)
                # Vectorised nearest-image matching of every rotated coordinate
                # to the original sites in one shot.
                diff = rot_coords_wrapped[:, None, :] - direct[None, :, :]
                diff -= np.round(diff)
                cart = diff @ lattice
                dists = np.sqrt(np.einsum("ijk,ijk->ij", cart, cart))
                matched = np.argmin(dists, axis=1)
                best = dists[np.arange(dists.shape[0]), matched]
                op_map = {
                    idx: (int(matched[idx]) if best[idx] < 1e-3 else idx)
                    for idx in range(dists.shape[0])
                }
                atom_maps.append(op_map)

            visited_pairs = set()
            for i, j, dist in pairs:
                pair_key = tuple(sorted((i, j)))
                if pair_key in visited_pairs:
                    continue
                orbit = set()
                for op_map in atom_maps:
                    i_rot = op_map.get(i, i)
                    j_rot = op_map.get(j, j)
                    orbit.add(tuple(sorted((i_rot, j_rot))))
                orbit_list = sorted(list(orbit))
                grouped_pairs.append([(int(x[0]), int(x[1])) for x in orbit_list])
                for p_entry in orbit_list:
                    visited_pairs.add(p_entry)
        else:
            native_groups: dict[tuple[tuple[str, str], float], list[tuple[int, int]]] = {}
            for i, j, dist in pairs:
                spec_i = species_by_atom[i]
                spec_j = species_by_atom[j]
                key_spec = tuple(sorted((spec_i, spec_j)))
                key_dist = round(dist, 2)
                key = (key_spec, key_dist)
                native_groups.setdefault(key, []).append((i, j))
            for g_list in native_groups.values():
                grouped_pairs.append(g_list)

        sites = []
        for site_index, pair_group in enumerate(grouped_pairs, start=1):
            representative = pair_group[0]
            i, j = representative
            pos_i = direct[i]
            pos_j = direct[j]
            delta = pos_j - pos_i
            delta = delta - np.round(delta)
            mid_direct = np.mod(pos_i + 0.5 * delta, 1.0)
            mid_cartesian = mid_direct @ lattice

            spec_i = species_by_atom[i]
            spec_j = species_by_atom[j]
            sorted_specs = sorted((spec_i, spec_j))
            species_token = f"{sorted_specs[0]}-{sorted_specs[1]}"
            multiplicity = len(pair_group)

            sites.append(
                DefectSite(
                    site_id=f"divacancy_{site_index:03d}",
                    species=species_token,
                    layer_id=None,
                    direct=tuple(float(value) for value in mid_direct),
                    cartesian=tuple(float(value) for value in mid_cartesian),
                    equivalent_indices=[],
                    multiplicity=multiplicity,
                    site_kind="divacancy",
                    backend=backend,
                    representative_index=i + 1,
                    site_family=species_token,
                    pair_indices=[i + 1, j + 1]
                )
            )
        return sites

    def _analyse_record(
        self,
        structure_path: str,
        *,
        structure_kind: str,
        backend: str,
        surface_side: str,
        layer_tolerance: float,
        symprec: float,
        divacancy_distance: float = 3.5,
    ) -> DefectAnalysis:
        record = self.converter.read(structure_path, canonicalize=False)
        resolved_kind = _detect_structure_kind(record, structure_kind, layer_tolerance=layer_tolerance)
        resolved_backend = self._choose_defect_backend(backend, resolved_kind)
        normal = _normal_from_lattice(record.lattice)
        projections = np.asarray(record.positions_cartesian, dtype=float) @ normal
        layers = _cluster_projection_layers(projections, layer_tolerance)
        notes = [
            "Surface/slab equivalence uses native species, layer, fractional fingerprint, and local-neighbour grouping.",
        ]

        if resolved_backend == "spglib":
            atom_sites = self._spglib_atom_sites(structure_path, record, layers=layers, symprec=symprec)
            notes.append("Exact Wyckoff labels are supplied by direct spglib for atom sites.")
        else:
            atom_sites = self._native_atom_sites(record, structure_kind=resolved_kind, layers=layers)
            notes.append("Wyckoff labels are only guaranteed with the spglib backend.")

        sites = list(atom_sites)
        sites.extend(self._interstitial_sites(record))
        if resolved_kind in {"surface", "slab", "molecule-on-substrate"}:
            adatom_sites, adatom_counts, error = self._adatom_sites(
                structure_path,
                surface_side=surface_side,
                layer_tolerance=layer_tolerance,
            )
            sites.extend(adatom_sites)
            if error:
                notes.append(f"Adatom site detection skipped: {error}")
            elif adatom_counts:
                notes.append(f"Detected adatom site families: {adatom_counts}")

        divacancies = self._divacancy_sites(
            record,
            backend=resolved_backend,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
        )
        sites.extend(divacancies)
        if divacancies:
            notes.append(f"Detected unique divacancy pairs: {len(divacancies)} (cutoff: {divacancy_distance} A)")

        return DefectAnalysis(
            structure_path=str(Path(structure_path).resolve()),
            structure_kind=resolved_kind,
            backend=resolved_backend,
            atom_count=int(record.natoms),
            species=list(record.species),
            counts=[int(value) for value in record.counts],
            layers=layers,
            sites=sites,
            notes=notes,
        )

    def analyse(
        self,
        structure_path: str,
        *,
        structure_kind: str = "auto",
        backend: str = "auto",
        surface_side: str = "top",
        layer_tolerance: float = 0.35,
        symprec: float = 0.01,
        divacancy_distance: float = 3.5,
    ) -> CommandResult:
        """Analyse inequivalent atom and insertion sites for a structure."""

        source = str(Path(structure_path).resolve())
        analysis = self._analyse_record(
            source,
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
        )
        run_id, run_dir = self.create_run_dir("analyse", label=Path(source).stem)
        analysis_json = _write_analysis_file(run_dir / "defect_analysis.json", analysis)
        summary = {
            "structure_kind": analysis.structure_kind,
            "backend": analysis.backend,
            "atom_sites": sum(1 for site in analysis.sites if site.site_kind == "atom"),
            "interstitial_sites": sum(1 for site in analysis.sites if site.site_kind == "interstitial"),
            "adatom_sites": sum(1 for site in analysis.sites if site.site_kind == "adatom"),
            "divacancy_sites": sum(1 for site in analysis.sites if site.site_kind == "divacancy"),
            "layers": len(analysis.layers),
        }
        artifacts = {"analysis_json": str(analysis_json)}
        manifest_path = self.write_manifest(
            stage="analyse",
            run_id=run_id,
            run_dir=run_dir,
            backend=analysis.backend,
            inputs={"structure_path": source},
            parameters={
                "structure_kind": structure_kind,
                "surface_side": surface_side,
                "layer_tolerance": float(layer_tolerance),
                "symprec": float(symprec),
                "divacancy_distance": float(divacancy_distance),
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={"analysis": analysis.to_dict(), "defect_preview": self.format_analysis(analysis)},
        )

    def _analysis_from_input(
        self,
        path_or_manifest: str,
        *,
        structure_kind: str,
        backend: str,
        surface_side: str,
        layer_tolerance: float,
        symprec: float,
        divacancy_distance: float = 3.5,
    ) -> DefectAnalysis:
        source = Path(path_or_manifest).resolve()
        if source.name == "manifest.json":
            manifest = RunManifest.load(source)
            if "analysis_json" in manifest.artifacts:
                return _load_analysis_file(Path(str(manifest.artifacts["analysis_json"])).resolve())
            if "structure_path" in manifest.inputs:
                return self._analyse_record(
                    str(manifest.inputs["structure_path"]),
                    structure_kind=structure_kind,
                    backend=backend,
                    surface_side=surface_side,
                    layer_tolerance=layer_tolerance,
                    symprec=symprec,
                    divacancy_distance=divacancy_distance,
                )
        if source.suffix.lower() == ".json":
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema") == "cellstine.defect_analysis.v1":
                return DefectAnalysis.from_dict(payload)
        return self._analyse_record(
            str(source),
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
        )

    def _sites_for_generation(
        self,
        analysis: DefectAnalysis,
        *,
        defect_type: str,
        site_ids: Sequence[str] | str | None,
        species_filter: str | None,
    ) -> list[DefectSite]:
        selected_ids = _normalise_site_ids(site_ids)
        defect = str(defect_type).lower()

        # Smart detection for manual divacancy site selection via two atom IDs/indices
        if defect in {"divacancy", "paired-vacancy"} and selected_ids is not None and len(selected_ids) == 2:
            record = self.converter.read(analysis.structure_path, canonicalize=False)
            physical_indices = []
            for s in selected_ids:
                s_str = str(s).strip()
                if s_str.isdigit():
                    physical_indices.append(int(s_str))
                elif s_str.lower().startswith("atom_") and s_str[5:].isdigit():
                    physical_indices.append(int(s_str[5:]))
                else:
                    found = False
                    for site in analysis.sites:
                        if site.site_kind == "atom" and site.site_id == s_str:
                            physical_indices.append(site.representative_index)
                            found = True
                            break
                    if not found:
                        import re
                        match = re.search(r'\d+$', s_str)
                        if match:
                            physical_indices.append(int(match.group()))

            if len(physical_indices) == 2:
                i = physical_indices[0] - 1
                j = physical_indices[1] - 1
                if 0 <= i < record.natoms and 0 <= j < record.natoms and i != j:
                    lattice = np.asarray(record.lattice, dtype=float)
                    direct = np.asarray(record.positions_direct, dtype=float)
                    pos_i = direct[i]
                    pos_j = direct[j]
                    delta = pos_j - pos_i
                    delta = delta - np.round(delta)
                    mid_direct = np.mod(pos_i + 0.5 * delta, 1.0)
                    mid_cartesian = mid_direct @ lattice

                    species_by_atom = _expanded_species(record)
                    spec_i = species_by_atom[i]
                    spec_j = species_by_atom[j]
                    sorted_specs = sorted((spec_i, spec_j))
                    species_token = f"{sorted_specs[0]}-{sorted_specs[1]}"

                    return [
                        DefectSite(
                            site_id=f"divacancy_{physical_indices[0]:03d}_{physical_indices[1]:03d}",
                            species=species_token,
                            layer_id=None,
                            direct=tuple(float(value) for value in mid_direct),
                            cartesian=tuple(float(value) for value in mid_cartesian),
                            equivalent_indices=[],
                            multiplicity=1,
                            site_kind="divacancy",
                            backend=analysis.backend,
                            representative_index=i + 1,
                            site_family=species_token,
                            pair_indices=[i + 1, j + 1]
                        )
                    ]

        if defect in {"vacancy", "substitution", "antisite"}:
            kind = "atom"
        elif defect == "interstitial":
            kind = "interstitial"
        elif defect == "adatom":
            kind = "adatom"
        elif defect in {"divacancy", "paired-vacancy"}:
            kind = "divacancy"
        else:
            raise ValueError(f"unsupported defect type '{defect_type}'")

        sites = [site for site in analysis.sites if site.site_kind == kind]
        if species_filter and kind == "atom":
            sites = [site for site in sites if site.species == str(species_filter)]
        if selected_ids is not None:
            allowed = set(selected_ids)
            sites = [site for site in sites if site.site_id in allowed or str(site.representative_index) in allowed]
        if not sites:
            available = ", ".join(site.site_id for site in analysis.sites if site.site_kind == kind) or "none"
            raise ValueError(f"no matching {kind} sites selected; available site IDs: {available}")
        return sites

    def _generate_one(
        self,
        record: StructureRecord,
        site: DefectSite,
        *,
        defect_type: str,
        species: str | None,
        substitution_species: str | None,
        height: float,
        surface_side: str = "top",
    ) -> tuple[StructureRecord, str]:
        atom_species = _expanded_species(record)
        direct = np.array(record.positions_direct, dtype=float, copy=True)
        flags = None if record.selective_flags is None else [tuple(item) for item in record.selective_flags]
        defect = str(defect_type).lower()

        if defect == "vacancy":
            if site.representative_index is None:
                raise ValueError(f"{site.site_id} is not an atom site")
            remove_index = int(site.representative_index) - 1
            keep = [index for index in range(len(atom_species)) if index != remove_index]
            return (
                _record_from_atoms(
                    record,
                    [atom_species[index] for index in keep],
                    direct[np.asarray(keep, dtype=int)],
                    None if flags is None else [flags[index] for index in keep],
                    comment=f"{record.comment} vacancy {site.site_id}",
                ),
                _safe_species_token(site.species),
            )

        if defect in {"divacancy", "paired-vacancy"}:
            if not site.pair_indices or len(site.pair_indices) < 2:
                raise ValueError(f"{site.site_id} does not have pair indices for divacancy generation")
            remove_1 = int(site.pair_indices[0]) - 1
            remove_2 = int(site.pair_indices[1]) - 1
            keep = [index for index in range(len(atom_species)) if index != remove_1 and index != remove_2]
            return (
                _record_from_atoms(
                    record,
                    [atom_species[index] for index in keep],
                    direct[np.asarray(keep, dtype=int)],
                    None if flags is None else [flags[index] for index in keep],
                    comment=f"{record.comment} divacancy {site.site_id}",
                ),
                _safe_species_token(site.species),
            )

        if defect in {"substitution", "antisite"}:
            replacement = substitution_species or species
            if not replacement:
                raise ValueError("substitution and antisite defects require --substitution-species or --species")
            if site.representative_index is None:
                raise ValueError(f"{site.site_id} is not an atom site")
            new_species = list(atom_species)
            new_species[int(site.representative_index) - 1] = str(replacement)
            return (
                _record_from_atoms(
                    record,
                    new_species,
                    direct,
                    flags,
                    comment=f"{record.comment} {defect} {site.site_id} {replacement}",
                ),
                _safe_species_token(site.species, replacement),
            )

        if defect in {"interstitial", "adatom"}:
            inserted = species or substitution_species
            if not inserted:
                raise ValueError(f"{defect} defects require --species")
            insert_direct = np.asarray(site.direct, dtype=float)
            if defect == "adatom":
                normal = _normal_from_lattice(record.lattice)
                direction = 1.0 if str(surface_side).lower() == "top" else -1.0
                cartesian = np.asarray(site.cartesian, dtype=float) + direction * normal * float(height)
                z_coords = record.positions_cartesian[:, 2]
                if z_coords.size > 0:
                    z_min = float(np.min(z_coords))
                    z_max = float(np.max(z_coords))
                    c_length = float(np.linalg.norm(record.lattice[2]))
                    if str(surface_side).lower() == "top":
                        if cartesian[2] > z_min + c_length + 1e-6:
                            raise ValueError("adatom height lies outside the current cell; increase vacuum or lower --height")
                    else:
                        if cartesian[2] < z_max - c_length - 1e-6:
                            raise ValueError("adatom height lies outside the current cell; increase vacuum or lower --height")
                insert_direct = cartesian @ np.linalg.inv(np.asarray(record.lattice, dtype=float))
            return (
                _record_from_atoms(
                    record,
                    [*atom_species, str(inserted)],
                    np.vstack([direct, insert_direct.reshape(1, 3)]),
                    None if flags is None else [*flags, ("T", "T", "T")],
                    comment=f"{record.comment} {defect} {site.site_id} {inserted}",
                ),
                _safe_species_token(inserted),
            )
        raise ValueError(f"unsupported defect type '{defect_type}'")

    def generate(
        self,
        structure_path_or_manifest: str,
        defect_type: str,
        *,
        site_ids: Sequence[str] | str | None = None,
        species: str | None = None,
        substitution_species: str | None = None,
        original_species: str | None = None,
        generate: str = "inequivalent",
        output_dir: str | Path | None = None,
        structure_kind: str = "auto",
        backend: str = "auto",
        surface_side: str = "top",
        layer_tolerance: float = 0.35,
        symprec: float = 0.01,
        height: float = 2.5,
        divacancy_distance: float = 3.5,
    ) -> CommandResult:
        """Generate one defect POSCAR per selected inequivalent site."""

        if str(generate).lower() != "inequivalent":
            raise ValueError("only generate='inequivalent' is currently supported")
        analysis = self._analysis_from_input(
            structure_path_or_manifest,
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
        )
        record = self.converter.read(analysis.structure_path, canonicalize=False)
        selected_sites = self._sites_for_generation(
            analysis,
            defect_type=defect_type,
            site_ids=site_ids,
            species_filter=original_species,
        )
        run_id, run_dir = self.create_run_dir("generate", label=f"{Path(analysis.structure_path).stem}_{defect_type}")
        analysis_json = _write_analysis_file(run_dir / "defect_analysis.json", analysis)
        destination = Path(output_dir).resolve() if output_dir is not None else self.output_root
        destination.mkdir(parents=True, exist_ok=True)
        suffix = run_output_suffix(run_id).replace("_", "-")
        written: list[str] = []
        generated_records: list[dict[str, Any]] = []

        for site in selected_sites:
            target_substitutions = [substitution_species or species]
            if defect_type == "antisite" and not (substitution_species or species):
                unique_host_species = list(dict.fromkeys(_expanded_species(record)))
                if len(unique_host_species) < 2:
                    raise ValueError(
                        f"antisite defects are only supported for multi-element hosts; "
                        f"found host species: {unique_host_species}. Specify --substitution-species explicitly."
                    )
                site_species = site.species
                target_substitutions = [s for s in unique_host_species if s != site_species]

            for replacement_species in target_substitutions:
                new_record, species_token = self._generate_one(
                    record,
                    site,
                    defect_type=defect_type,
                    species=replacement_species,
                    substitution_species=replacement_species,
                    height=height,
                    surface_side=surface_side,
                )
                filename = f"defect_{defect_type}_{site.site_id}_{species_token}_{suffix}.vasp"
                output_path = self.vasp_io.write(new_record, str(destination / filename), positions_are_cartesian=False, wrap_positions=False)
                written.append(str(output_path))
                generated_records.append(
                    {
                        "site_id": site.site_id,
                        "site_kind": site.site_kind,
                        "equivalent_indices": list(site.equivalent_indices),
                        "output_path": str(output_path),
                        "atom_count": int(new_record.natoms),
                    }
                )

        artifacts = {"analysis_json": str(analysis_json), "structures": written}
        summary = {
            "defect_type": str(defect_type).lower(),
            "generated": len(written),
            "structure_kind": analysis.structure_kind,
            "backend": analysis.backend,
        }
        manifest_path = self.write_manifest(
            stage="generate",
            run_id=run_id,
            run_dir=run_dir,
            backend=analysis.backend,
            inputs={"analysis_or_structure": str(Path(structure_path_or_manifest).resolve()), "structure_path": analysis.structure_path},
            parameters={
                "defect_type": defect_type,
                "site_ids": _normalise_site_ids(site_ids),
                "species": species,
                "substitution_species": substitution_species,
                "original_species": original_species,
                "generate": generate,
                "height": float(height),
                "surface_side": surface_side,
                "layer_tolerance": float(layer_tolerance),
                "symprec": float(symprec),
                "divacancy_distance": float(divacancy_distance),
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={"generated": generated_records, "defect_preview": self.format_analysis(analysis)},
        )

    def preview(
        self,
        analysis_or_structure: str,
        *,
        limit: int = 30,
        structure_kind: str = "auto",
        backend: str = "auto",
        surface_side: str = "top",
        layer_tolerance: float = 0.35,
        symprec: float = 0.01,
        divacancy_distance: float = 3.5,
    ) -> CommandResult:
        """Preview defect sites without writing generated structures."""

        analysis = self._analysis_from_input(
            analysis_or_structure,
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
        )
        run_id, run_dir = self.create_run_dir("preview", label=Path(analysis.structure_path).stem)
        analysis_json = _write_analysis_file(run_dir / "defect_analysis.json", analysis)
        preview = self.format_analysis(analysis, limit=limit)
        artifacts = {"analysis_json": str(analysis_json)}
        summary = {
            "structure_kind": analysis.structure_kind,
            "backend": analysis.backend,
            "shown_sites": min(int(limit), len(analysis.sites)),
            "total_sites": len(analysis.sites),
        }
        manifest_path = self.write_manifest(
            stage="preview",
            run_id=run_id,
            run_dir=run_dir,
            backend=analysis.backend,
            inputs={"analysis_or_structure": str(Path(analysis_or_structure).resolve()), "structure_path": analysis.structure_path},
            parameters={
                "limit": int(limit),
                "surface_side": surface_side,
                "layer_tolerance": float(layer_tolerance),
                "symprec": float(symprec),
                "divacancy_distance": float(divacancy_distance),
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={"analysis": analysis.to_dict(), "defect_preview": preview},
        )

    @staticmethod
    def format_analysis(analysis: DefectAnalysis, *, limit: int = 30) -> str:
        """Return a compact table of discovered defect sites."""

        rows = list(analysis.sites)
        shown = rows[: max(0, int(limit))]
        if not shown:
            return "No defect sites were detected."
        lines = [
            f"Defect sites for {Path(analysis.structure_path).name} ({analysis.structure_kind}, backend={analysis.backend})",
            " site_id                 kind          species  layer  mult  wyckoff  direct (u, v, w)              represented atoms",
            "-" * 116,
        ]
        for site in shown:
            direct = tuple(float(value) for value in site.direct)
            represented = ",".join(str(value) for value in site.equivalent_indices) if site.equivalent_indices else "-"
            lines.append(
                f" {site.site_id:<23s} {site.site_kind:<13s} {(site.species or '-'):>7s} "
                f"{str(site.layer_id or '-'):>6s} {int(site.multiplicity):5d} "
                f"{(site.wyckoff or '-'):>8s}  "
                f"({direct[0]:7.4f}, {direct[1]:7.4f}, {direct[2]:7.4f})    {represented}"
            )
        if len(rows) > len(shown):
            lines.append(f"... {len(rows) - len(shown)} more site(s) not shown.")
        if analysis.notes:
            lines.append("")
            lines.append("Notes:")
            for note in analysis.notes[:4]:
                lines.append(f"- {note}")
        return "\n".join(lines)

"""Enumeration of the inequivalent vacancy, interstitial, adatom, and divacancy sites."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import geometry, symmetry3d, voids
from ..core.species import expand_species
from ..interface.surface import backend as surface_backend
from ..io.models import StructureRecord
from ..symmetry.symmetry import Symmetry
from .records import DefectSite
from .analysis import (
    _PlainOperation,
    _atom_members,
    _layer_lookup,
    _nearest_layer_id,
    _pairwise_minimum_image_distances,
    _serialise_site_id,
    _symmetry_groups_for_points,
)


class DefectSiteEnumerationMixin:
    """Site enumerators shared by the defect workflow."""

    def _native_symmetry(self, record: StructureRecord, *, symprec: float) -> symmetry3d.SymmetryDataset:
        """Return the native symmetry description of a structure."""

        return symmetry3d.analyse_symmetry(
            np.asarray(record.lattice, dtype=float),
            np.asarray(record.positions_direct, dtype=float),
            expand_species(record.species, record.counts),
            symprec=float(symprec),
        )

    def _native_atom_sites(
        self,
        record: StructureRecord,
        *,
        layers: Sequence[dict[str, Any]],
        dataset: symmetry3d.SymmetryDataset,
    ) -> list[DefectSite]:
        """Return one site per orbit of symmetry-equivalent atoms.

        The orbits come from the space-group operations of the cell itself, so
        two atoms are merged only when a symmetry operation actually maps one
        onto the other.  A geometric fingerprint would also merge sites that
        merely look alike, and would report a multiplicity that no operation
        supports.
        """

        species_by_atom = expand_species(record.species, record.counts)
        direct = np.asarray(record.positions_direct, dtype=float)
        cartesian = np.asarray(record.positions_cartesian, dtype=float)
        layer_lookup = _layer_lookup(layers)
        grouped: dict[int, list[int]] = {}
        for atom_index, representative in enumerate(dataset.equivalent_atoms.tolist()):
            grouped.setdefault(int(representative), []).append(int(atom_index))

        sites = []
        for site_index, (representative, atom_indices) in enumerate(sorted(grouped.items()), start=1):
            ordered = [int(representative), *(index for index in sorted(atom_indices) if index != representative)]
            sites.append(
                DefectSite(
                    site_id=_serialise_site_id("atom", site_index),
                    species=str(species_by_atom[representative]),
                    layer_id=layer_lookup.get(representative + 1),
                    direct=tuple(float(value) for value in direct[representative]),
                    cartesian=tuple(float(value) for value in cartesian[representative]),
                    equivalent_indices=[int(value) + 1 for value in sorted(atom_indices)],
                    multiplicity=int(len(atom_indices)),
                    site_kind="atom",
                    backend="native",
                    representative_index=int(representative) + 1,
                    members=_atom_members(ordered, direct, cartesian, layer_lookup),
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
            ordered = [representative, *(index for index in indices if index != representative)]
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
                    members=_atom_members(ordered, direct, cartesian, layer_lookup),
                )
            )
        return sites

    def _interstitial_sites(
        self,
        record: StructureRecord,
        *,
        dataset: symmetry3d.SymmetryDataset | None = None,
        symprec: float = 0.01,
        layers: Sequence[dict[str, Any]] = (),
        unit: np.ndarray | None = None,
        interstitial_saddles: bool = False,
    ) -> tuple[list[DefectSite], list[str]]:
        """Return the symmetry-inequivalent interstitial voids of a structure.

        Candidates are the local maxima of the distance to the nearest atom, so
        every reported site is the centre of a genuine empty sphere whose radius
        is recorded.  Sites in the vacuum of a slab or a molecular box are
        excluded, and candidates related by a symmetry operation are collapsed
        into one entry with its multiplicity.

        With ``interstitial_saddles`` the saddles of the same distance function
        are reported as well, marked ``void_kind="saddle"``.  They are the sites
        held in place by two or three atoms rather than surrounded by four, and
        a crystal can have no maximum where its interstitials actually sit: the
        octahedral site of a body-centred cubic metal is a saddle.
        """

        lattice = np.asarray(record.lattice, dtype=float)
        direct = np.asarray(record.positions_direct, dtype=float)
        notes: list[str] = []
        if len(direct) == 0:
            return [], notes

        search = voids.find_void_sites(lattice, direct, include_saddles=bool(interstitial_saddles))
        if not search.sites:
            if search.vacuum_axes:
                notes.append(
                    "No interstitial voids inside the material region; "
                    f"vacuum was detected along axis {', '.join(str(axis + 1) for axis in search.vacuum_axes)}."
                )
            else:
                notes.append(
                    f"No void larger than {search.minimum_radius:.2f} A was found in this cell."
                )
            return [], notes

        candidates = np.array([site.direct for site in search.sites], dtype=float)
        radii = [float(site.radius) for site in search.sites]
        kinds = [str(site.kind) for site in search.sites]
        coordinations = [int(site.coordination) for site in search.sites]
        groups = _symmetry_groups_for_points(
            lattice,
            candidates,
            dataset,
            tolerance=max(0.1, float(symprec)),
            labels=[f"{kind}:{round(radius, 4)}" for kind, radius in zip(kinds, radii)],
        )

        axis = None if unit is None else np.asarray(unit, dtype=float).reshape(3)
        sites: list[DefectSite] = []
        for site_index, members in enumerate(groups, start=1):
            representative = int(members[0])
            point = candidates[representative]
            orbit: list[dict[str, Any]] = []
            for member in members:
                member_direct = candidates[int(member)]
                member_cartesian = member_direct @ lattice
                layer_id = (
                    None
                    if axis is None
                    else _nearest_layer_id(float(member_cartesian @ axis), layers)
                )
                orbit.append(
                    {
                        "indices": [],
                        "direct": [float(value) for value in member_direct],
                        "cartesian": [float(value) for value in member_cartesian],
                        "layer_ids": [] if layer_id is None else [int(layer_id)],
                    }
                )
            sites.append(
                DefectSite(
                    site_id=_serialise_site_id("interstitial", site_index),
                    species=None,
                    layer_id=None if not orbit[0]["layer_ids"] else int(orbit[0]["layer_ids"][0]),
                    direct=tuple(float(value) for value in point),
                    cartesian=tuple(float(value) for value in point @ lattice),
                    equivalent_indices=[],
                    multiplicity=int(len(members)),
                    site_kind="interstitial",
                    backend="native",
                    void_radius=round(float(radii[representative]), 4),
                    void_kind=kinds[representative],
                    void_coordination=coordinations[representative],
                    members=orbit,
                )
            )
        notes.append(
            f"Interstitial voids: {len(sites)} inequivalent site(s), "
            f"largest empty-sphere radius {max(radii):.2f} A "
            f"(minimum accepted {search.minimum_radius:.2f} A)."
        )
        if interstitial_saddles:
            notes.append(
                "Saddles of the distance to the nearest atom are included: a site marked "
                "'saddle' is held by the two or three atoms on its sphere and its sphere grows "
                "along the remaining directions."
            )
        else:
            notes.append(
                "Only local maxima of the distance to the nearest atom are listed. "
                "Structures whose interstitials are saddles -- the octahedral site of a "
                "body-centred cubic metal, the bond centre of a covalent crystal -- need "
                "`--interstitial-saddles`."
            )
        return sites, notes

    def _adatom_sites(
        self,
        structure_path: str,
        *,
        surface_side: str,
        layer_tolerance: float,
        lattice: np.ndarray | None = None,
        dataset: symmetry3d.SymmetryDataset | None = None,
        symprec: float = 1e-5,
    ) -> tuple[list[DefectSite], dict[str, int], str | None]:
        """Return the inequivalent adatom sites of a slab surface.

        The site search reports every site in the cell, so a supercell repeats
        each of them.  Grouping the sites of one family into orbits of the
        space-group operations turns that list back into the distinct sites a
        calculation would actually have to sample, each with its multiplicity.
        """

        try:
            report = surface_backend.find_adsorption_sites(
                structure_path,
                surface_side=surface_side,
                layer_tolerance=layer_tolerance,
            )
        except Exception as exc:
            return [], {}, str(exc)

        cell = np.asarray(lattice, dtype=float) if lattice is not None else None
        sites: list[DefectSite] = []
        type_counts: dict[str, int] = {}
        families: dict[str, list] = {}
        for site in report.sites:
            families.setdefault(str(site.site_type), []).append(site)

        for site_type, family in families.items():
            points = np.array([site.direct for site in family], dtype=float)
            if cell is not None and dataset is not None:
                groups = _symmetry_groups_for_points(cell, points, dataset, tolerance=max(0.1, float(symprec)))
            else:
                groups = [[index] for index in range(len(family))]
            type_counts[site_type] = len(groups)
            for index, members in enumerate(groups, start=1):
                representative = family[members[0]]
                sites.append(
                    DefectSite(
                        site_id=f"adatom_{site_type}_{index:03d}",
                        species=None,
                        layer_id=None,
                        direct=tuple(float(value) for value in representative.direct),
                        cartesian=tuple(float(value) for value in representative.cartesian),
                        equivalent_indices=[],
                        multiplicity=len(members),
                        site_kind="adatom",
                        backend="native",
                        site_family=site_type,
                        void_radius=None if representative.void_radius is None else round(float(representative.void_radius), 4),
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
        dataset: symmetry3d.SymmetryDataset | None = None,
        layers: Sequence[dict[str, Any]] = (),
    ) -> list[DefectSite]:
        species_by_atom = expand_species(record.species, record.counts)
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

        symmetry_ops: list[Any] = []
        if backend == "spglib":
            try:
                sym_tool = Symmetry(dependency_manager=self.dependency_manager)
                sym_analysis = sym_tool.analyse_record(record, backend="spglib", symprec=symprec)
                symmetry_ops = list(sym_analysis.operations)
            except Exception:
                symmetry_ops = []
        if not symmetry_ops and dataset is not None:
            # The native engine supplies the same operations, so divacancy pairs
            # are grouped by real symmetry instead of by species and distance.
            symmetry_ops = [
                _PlainOperation(np.asarray(rotation, dtype=int), np.asarray(translation, dtype=float))
                for rotation, translation in zip(dataset.rotations, dataset.translations)
            ]

        grouped_pairs: list[list[tuple[int, int]]] = []

        if symmetry_ops:
            # One bucketed index of the sites answers the site mapping of every
            # operation in O(n), where a distance matrix per operation would be
            # O(n^2), and it will only ever match atoms of the same species.
            ordered_species = sorted(set(species_by_atom))
            label_of = {name: number for number, name in enumerate(ordered_species)}
            site_labels = np.array([label_of[name] for name in species_by_atom], dtype=np.int64)
            finder = geometry.PeriodicSiteIndex(
                lattice, direct, labels=site_labels, tolerance=1e-3
            )
            # A pair orbit is a connected component of the graph the group draws
            # on the pairs, and connectivity is decided by any generating set,
            # so the whole group -- one operation per rotation *and* per lattice
            # translation of a supercell -- never has to be walked.
            # ``RequestProject/PairOrbits.lean`` proves the three steps this
            # relies on: the induced action on unordered pairs is a group
            # homomorphism and sweeping the generators therefore finds the true
            # orbits (``Cellstine.pairLinked_iff_exists_symmetry``), a chain
            # never has to leave the set of pairs inside the cutoff because a
            # symmetry preserves distances
            # (``Cellstine.siteLinkedOn_iff_siteLinked``), and the integer
            # address ``min * natoms + max`` used below identifies the pair
            # uniquely (``Cellstine.pairCode_injOn``).
            generator_rotations, generator_translations = symmetry3d.generating_operations(
                np.array([np.asarray(op.rotation, dtype=int) for op in symmetry_ops]),
                np.array([np.asarray(op.translation, dtype=float) for op in symmetry_ops]),
            )
            atom_maps = []
            identity_map = np.arange(natoms, dtype=np.int64)
            for rotation, translation in zip(generator_rotations, generator_translations):
                matched = finder.match(
                    direct @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float),
                    site_labels,
                )
                matched = np.asarray(matched, dtype=np.int64)
                atom_maps.append(np.where(matched >= 0, matched, identity_map))

            # Pairs are addressed by the code ``min * natoms + max``, which the
            # upper-triangle enumeration already produces in ascending order, so
            # the image of a pair is found by one binary search.
            first = np.array([entry[0] for entry in pairs], dtype=np.int64)
            second = np.array([entry[1] for entry in pairs], dtype=np.int64)
            codes = first * natoms + second
            parent = list(range(len(pairs)))

            def find(value: int) -> int:
                while parent[value] != value:
                    parent[value] = parent[parent[value]]
                    value = parent[value]
                return value

            def union(left: int, right: int) -> None:
                root_a, root_b = find(left), find(right)
                if root_a != root_b:
                    parent[max(root_a, root_b)] = min(root_a, root_b)

            for op_map in atom_maps:
                images_a = op_map[first]
                images_b = op_map[second]
                image_codes = (
                    np.minimum(images_a, images_b) * natoms + np.maximum(images_a, images_b)
                )
                slots = np.searchsorted(codes, image_codes)
                inside = slots < len(codes)
                inside[inside] = codes[slots[inside]] == image_codes[inside]
                for source in np.nonzero(inside)[0]:
                    union(int(source), int(slots[source]))

            orbits: dict[int, list[tuple[int, int]]] = {}
            for index in range(len(pairs)):
                orbits.setdefault(find(index), []).append(
                    (int(first[index]), int(second[index]))
                )
            grouped_pairs = [members for _, members in sorted(orbits.items())]
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

        layer_lookup = _layer_lookup(layers)
        sites = []
        for site_index, pair_group in enumerate(grouped_pairs, start=1):
            representative = pair_group[0]
            i, j = representative
            # The divacancy site sits halfway along the *shortest* image of the
            # pair, which rounding the fractional difference does not find in a
            # skewed cell.
            mid_direct = geometry.periodic_midpoints(lattice, direct[i], direct[j])[0]
            mid_cartesian = mid_direct @ lattice

            spec_i = species_by_atom[i]
            spec_j = species_by_atom[j]
            sorted_specs = sorted((spec_i, spec_j))
            species_token = f"{sorted_specs[0]}-{sorted_specs[1]}"
            multiplicity = len(pair_group)

            orbit: list[dict[str, Any]] = []
            for first_atom, second_atom in pair_group:
                member_mid = geometry.periodic_midpoints(
                    lattice, direct[first_atom], direct[second_atom]
                )[0]
                member_layers = sorted(
                    {
                        layer
                        for layer in (
                            layer_lookup.get(first_atom + 1),
                            layer_lookup.get(second_atom + 1),
                        )
                        if layer is not None
                    }
                )
                orbit.append(
                    {
                        "indices": [int(first_atom) + 1, int(second_atom) + 1],
                        "direct": [float(value) for value in member_mid],
                        "cartesian": [float(value) for value in member_mid @ lattice],
                        "layer_ids": [int(value) for value in member_layers],
                    }
                )

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
                    pair_indices=[i + 1, j + 1],
                    members=orbit,
                )
            )
        return sites

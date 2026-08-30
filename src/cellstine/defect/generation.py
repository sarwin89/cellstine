"""Building the defect structures themselves.

:mod:`cellstine.defect.workflow` reads a structure, groups its sites and reports
them; this module turns a chosen site into a POSCAR.  It holds the selection of
the sites a run will use, the single-structure builder for each defect type, and
the ``defect generate`` stage that drives them and writes the run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import geometry
from ..core.base import run_output_suffix
from ..core.contacts import Contact, closest_contact, contact_notes
from ..core.layers import LAYER_TOLERANCE
from ..core.models import CommandResult
from ..core.provenance import restage_comment
from ..core.species import expand_species
from ..core.transforms import repeat_structure, supercell_structure
from ..core.vacuum import fit_cell_to_vacuum, surface_normal, vacuum_gap
from ..io.models import StructureRecord
from .analysis import (
    _detect_structure_kind,
    _normal_from_lattice,
    _normalise_site_ids,
    _normalise_supercell,
    _record_from_atoms,
    _safe_species_token,
    _write_analysis_file,
)
from .dilution import dilution_report
from .layers import resolve_layer_ids, sites_by_layer, sites_by_member
from .records import DefectAnalysis, DefectSite
from .supercell import DEFAULT_CELL_LIMIT, SupercellChoice, choose_supercell


_TRAILING_NUMBER = re.compile(r"\d+$")


def _resolve_atom_index(analysis: DefectAnalysis, token: str) -> int | None:
    """Return the one-based atom number a site token names, if it names one.

    The two sites of a hand-picked divacancy may be spelled as plain atom
    numbers, as ``atom_7``, or as the identifier of an atom site of the
    analysis.  Anything else is read as the number it ends with, which is how
    every site identifier is built.
    """

    text = str(token).strip()
    if text.isdigit():
        return int(text)
    if text.lower().startswith("atom_") and text[5:].isdigit():
        return int(text[5:])
    for site in analysis.sites:
        if site.site_kind == "atom" and site.site_id == text:
            return None if site.representative_index is None else int(site.representative_index)
    match = _TRAILING_NUMBER.search(text)
    return int(match.group()) if match else None


def _resolve_atom_indices(analysis: DefectAnalysis, tokens: Sequence[str]) -> list[int]:
    """Return the atom numbers of every token that names one."""

    resolved = (_resolve_atom_index(analysis, token) for token in tokens)
    return [index for index in resolved if index is not None]


def _defect_atom_index(record: StructureRecord, defect_direct: np.ndarray) -> int | None:
    """Locate the atom of ``record`` that sits at ``defect_direct``.

    :func:`_record_from_atoms` regroups the atoms by species, so the inserted or
    replaced atom is not where it was put; it is found back by its position,
    which no regrouping changes.
    """

    direct = np.asarray(record.positions_direct, dtype=float).reshape(-1, 3)
    if direct.shape[0] == 0:
        return None
    target = np.asarray(defect_direct, dtype=float).reshape(1, 3)
    distances = geometry.pairwise_minimum_image_distances(
        geometry.as_lattice(record.lattice), target, other_direct=direct
    )[0]
    index = int(np.argmin(distances))
    if float(distances[index]) > 1e-6:
        return None
    return index


def _defect_contact(record: StructureRecord, defect_direct: np.ndarray | None) -> Contact | None:
    """The closest approach the defect atom makes to the host that surrounds it.

    A vacancy only opens space, so it has no contact to measure; an interstitial,
    an adatom or a substitution puts an atom somewhere, and how close that atom
    lands is what says whether the cell is worth running.
    """

    if defect_direct is None:
        return None
    index = _defect_atom_index(record, defect_direct)
    if index is None:
        return None
    direct = np.asarray(record.positions_direct, dtype=float).reshape(-1, 3)
    if direct.shape[0] < 2:
        return None
    labels = expand_species(record.species, record.counts)
    others = [position for position in range(direct.shape[0]) if position != index]
    return closest_contact(
        record.lattice,
        direct[index].reshape(1, 3),
        direct[np.asarray(others, dtype=int)],
        first_species=[labels[index]],
        second_species=[labels[position] for position in others],
    )


class DefectGenerationMixin:
    """The ``defect generate`` stage: choose the sites, build them, write them."""

    def _sites_for_generation(
        self,
        analysis: DefectAnalysis,
        *,
        defect_type: str,
        site_ids: Sequence[str] | str | None,
        species_filter: str | None,
        layers: str | Sequence[int] | None = None,
        mode: str = "inequivalent",
    ) -> tuple[list[DefectSite], list[str]]:
        selected_ids = _normalise_site_ids(site_ids)
        defect = str(defect_type).lower()

        # Smart detection for manual divacancy site selection via two atom IDs/indices
        if defect in {"divacancy", "paired-vacancy"} and selected_ids is not None and len(selected_ids) == 2:
            record = self.converter.read(analysis.structure_path, canonicalize=False)
            physical_indices = _resolve_atom_indices(analysis, selected_ids)
            if len(physical_indices) == 2:
                i = physical_indices[0] - 1
                j = physical_indices[1] - 1
                if 0 <= i < record.natoms and 0 <= j < record.natoms and i != j:
                    lattice = np.asarray(record.lattice, dtype=float)
                    direct = np.asarray(record.positions_direct, dtype=float)
                    mid_direct = geometry.periodic_midpoints(
                        lattice, direct[i], direct[j]
                    )[0]
                    mid_cartesian = mid_direct @ lattice

                    species_by_atom = expand_species(record.species, record.counts)
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
                            pair_indices=[i + 1, j + 1],
                        )
                    ], []

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
        notes: list[str] = []
        layer_ids = resolve_layer_ids(analysis.layers, layers)
        if str(mode).lower() == "all":
            # Every equivalent copy in its own structure, restricted to the
            # selected planes when one was asked for.
            sites, member_notes = sites_by_member(sites, layer_ids)
            notes.extend(member_notes)
        elif layer_ids is not None:
            # One defect per selected atomic plane, instead of one per orbit:
            # the same defect at two depths is two different calculations.
            sites, layer_notes = sites_by_layer(sites, layer_ids)
            notes.extend(layer_notes)
            notes.append(
                "Atomic planes selected: "
                + ", ".join(str(value) for value in layer_ids)
                + f" of {len(analysis.layers)}, counted along the "
                + str((analysis.view_direction or {}).get("label", "surface normal"))
                + "."
            )
        if selected_ids is not None:
            allowed = set(selected_ids)
            sites = [
                site
                for site in sites
                if site.site_id in allowed
                or str(site.representative_index) in allowed
                or site.site_id.rsplit("_L", 1)[0] in allowed
            ]
        if not sites:
            available = ", ".join(site.site_id for site in analysis.sites if site.site_kind == kind) or "none"
            raise ValueError(
                f"no matching {kind} sites selected; available site IDs: {available}"
                + ("" if layer_ids is None else f" (restricted to atomic plane(s) {list(layer_ids)})")
            )
        return sites, notes

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
        preserve_vacuum: bool = True,
    ) -> tuple[StructureRecord, str, np.ndarray | None]:
        """Build one defect structure.

        Besides the record and the species token used in its filename, the
        fractional position of the atom the defect added or replaced comes back
        (``None`` for a vacancy), so the caller can measure how close it landed.
        """

        atom_species = expand_species(record.species, record.counts)
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
                    comment=restage_comment(record.comment, "defect generate", f"vacancy {site.site_id}"),
                ),
                _safe_species_token(site.species),
                None,
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
                    comment=restage_comment(record.comment, "defect generate", f"divacancy {site.site_id}"),
                ),
                _safe_species_token(site.species),
                None,
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
                    comment=restage_comment(
                        record.comment, "defect generate", f"{defect} {site.site_id} -> {replacement}"
                    ),
                ),
                _safe_species_token(site.species, replacement),
                np.array(direct[int(site.representative_index) - 1], dtype=float),
            )

        if defect in {"interstitial", "adatom"}:
            inserted = species or substitution_species
            if not inserted:
                raise ValueError(f"{defect} defects require --species")
            insert_direct = np.asarray(site.direct, dtype=float)
            lattice_out = np.asarray(record.lattice, dtype=float)
            if defect == "adatom":
                normal = _normal_from_lattice(record.lattice)
                direction = 1.0 if str(surface_side).lower() == "top" else -1.0
                cartesian = np.asarray(site.cartesian, dtype=float) + direction * normal * float(height)
                host_cartesian = np.asarray(record.positions_cartesian, dtype=float)
                if preserve_vacuum and host_cartesian.size:
                    # An adatom eats into the vacuum of the slab it sits on, so
                    # lengthen c until the gap the slab arrived with is back.
                    gap_before = vacuum_gap(record.lattice, host_cartesian)
                    plane_normal = surface_normal(record.lattice)
                    anchor = float(np.min(host_cartesian @ plane_normal))
                    combined = np.vstack([host_cartesian, cartesian.reshape(1, 3)])
                    lattice_out, fitted = fit_cell_to_vacuum(
                        record.lattice, combined, gap_before, anchor=anchor
                    )
                    inverse = np.linalg.inv(lattice_out)
                    direct = fitted[:-1] @ inverse
                    insert_direct = fitted[-1] @ inverse
                else:
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
                    comment=restage_comment(
                        record.comment, "defect generate", f"{defect} {site.site_id} {inserted}"
                    ),
                    lattice=lattice_out,
                ),
                _safe_species_token(inserted),
                np.asarray(insert_direct, dtype=float).reshape(3),
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
        layer_tolerance: float = LAYER_TOLERANCE,
        symprec: float = 0.01,
        height: float = 2.5,
        divacancy_distance: float = 3.5,
        preserve_vacuum: bool = True,
        supercell: Sequence[int] | None = None,
        supercell_matrix: Sequence[Sequence[int]] | Sequence[int] | None = None,
        min_image_distance: float | None = None,
        cell_limit: int = DEFAULT_CELL_LIMIT,
        view_direction: str | None = None,
        layers: str | Sequence[int] | None = None,
        interstitial_saddles: bool = False,
    ) -> CommandResult:
        """Generate defect POSCARs for the selected sites.

        Without ``layers`` there is one structure per inequivalent site, which
        is the smallest set that covers every distinct defect.  With ``layers``
        each orbit is split over the atomic planes it visits -- see
        :mod:`cellstine.defect.layers` -- so ``layers="all"`` puts the defect in
        every plane of a slab, and ``view_direction`` chooses the direction
        those planes are counted along.  ``generate="all"`` goes further and
        writes one structure for every equivalent atom rather than one per
        orbit; combined with ``layers`` it is restricted to the chosen planes.

        The host cell can be enlarged first in three ways: ``supercell`` repeats
        it along its own axes, ``supercell_matrix`` uses any integer matrix, and
        ``min_image_distance`` asks :mod:`cellstine.defect.supercell` for the
        smallest cell that puts that distance between the defect and its
        periodic images.
        """

        mode = str(generate).lower().strip()
        if mode not in {"inequivalent", "all"}:
            raise ValueError(
                "generate must be 'inequivalent' (one structure per orbit) "
                "or 'all' (one structure per equivalent atom)"
            )
        run_id: str | None = None
        run_dir: Path | None = None
        host_supercell_path: str | None = None
        repeats = _normalise_supercell(supercell)
        requested = [
            name
            for name, value in (
                ("supercell", repeats),
                ("supercell_matrix", supercell_matrix),
                ("min_image_distance", min_image_distance),
            )
            if value is not None
        ]
        if len(requested) > 1:
            raise ValueError(
                "choose one way of enlarging the host cell: " + ", ".join(requested)
            )
        chosen_supercell: SupercellChoice | None = None
        matrix: list[list[int]] | None = None
        if repeats is not None:
            matrix = [
                [int(repeats[0]), 0, 0],
                [0, int(repeats[1]), 0],
                [0, 0, int(repeats[2])],
            ]
            label = f"host supercell {repeats[0]}x{repeats[1]}x{repeats[2]}"
        elif supercell_matrix is not None:
            matrix = [
                [int(value) for value in row]
                for row in np.asarray(supercell_matrix, dtype=np.int64).reshape(3, 3)
            ]
            label = "host supercell from the requested matrix"
        if matrix is not None or min_image_distance is not None:
            source = Path(structure_path_or_manifest)
            if source.suffix.lower() == ".json" or source.name == "manifest.json":
                raise ValueError(
                    "a supercell can only be built from a structure file; "
                    "pass the POSCAR rather than a saved analysis"
                )
            host = self.converter.read(str(source), canonicalize=False)
            if matrix is None:
                kind = _detect_structure_kind(
                    host, structure_kind, layer_tolerance=float(layer_tolerance)
                )
                chosen_supercell = choose_supercell(
                    host.lattice,
                    structure_kind=kind,
                    min_image_distance=float(min_image_distance),
                    cell_limit=int(cell_limit),
                )
                matrix = chosen_supercell.matrix
                label = (
                    f"host supercell of {chosen_supercell.cells} cell(s) for a "
                    f"{chosen_supercell.image_distance:.2f} A image separation"
                )
            if repeats is not None:
                enlarged = repeat_structure(host, repeats)
            else:
                enlarged = supercell_structure(host, matrix)
            enlarged.comment = restage_comment(host.comment, "defect generate", label)
            run_id, run_dir = self.create_run_dir(
                "generate", label=f"{source.stem}_{defect_type}"
            )
            host_supercell_path = self.vasp_io.write(
                enlarged,
                str(run_dir / "host_supercell.vasp"),
                positions_are_cartesian=False,
                wrap_positions=False,
            )
            structure_path_or_manifest = str(host_supercell_path)
        analysis = self._analysis_from_input(
            structure_path_or_manifest,
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
            view_direction=view_direction,
            interstitial_saddles=interstitial_saddles,
        )
        record = self.converter.read(analysis.structure_path, canonicalize=False)
        selected_sites, selection_notes = self._sites_for_generation(
            analysis,
            defect_type=defect_type,
            site_ids=site_ids,
            species_filter=original_species,
            layers=layers,
            mode=mode,
        )
        if run_dir is None:
            run_id, run_dir = self.create_run_dir(
                "generate", label=f"{Path(analysis.structure_path).stem}_{defect_type}"
            )
        analysis_json = _write_analysis_file(run_dir / "defect_analysis.json", analysis)
        destination = Path(output_dir).resolve() if output_dir is not None else self.output_root
        destination.mkdir(parents=True, exist_ok=True)
        suffix = run_output_suffix(run_id).replace("_", "-")
        written: list[str] = []
        generated_records: list[dict[str, Any]] = []
        contact_notes_seen: list[str] = []
        closest_defect_contact: Contact | None = None

        for site in selected_sites:
            target_substitutions = [substitution_species or species]
            if defect_type == "antisite" and not (substitution_species or species):
                unique_host_species = list(dict.fromkeys(expand_species(record.species, record.counts)))
                if len(unique_host_species) < 2:
                    raise ValueError(
                        f"antisite defects are only supported for multi-element hosts; "
                        f"found host species: {unique_host_species}. Specify --substitution-species explicitly."
                    )
                site_species = site.species
                target_substitutions = [s for s in unique_host_species if s != site_species]

            for replacement_species in target_substitutions:
                new_record, species_token, defect_direct = self._generate_one(
                    record,
                    site,
                    defect_type=defect_type,
                    species=replacement_species,
                    substitution_species=replacement_species,
                    height=height,
                    surface_side=surface_side,
                    preserve_vacuum=bool(preserve_vacuum),
                )
                contact = _defect_contact(new_record, defect_direct)
                for note in contact_notes(contact, subject=f"{defect_type} {site.site_id} defect-host"):
                    if note not in contact_notes_seen:
                        contact_notes_seen.append(note)
                if contact is not None and (
                    closest_defect_contact is None
                    or contact.distance < closest_defect_contact.distance
                ):
                    closest_defect_contact = contact
                filename = f"defect_{defect_type}_{site.site_id}_{species_token}_{suffix}.vasp"
                output_path = self.vasp_io.write(new_record, str(destination / filename), positions_are_cartesian=False, wrap_positions=False)
                written.append(str(output_path))
                generated_records.append(
                    {
                        "site_id": site.site_id,
                        "site_kind": site.site_kind,
                        "layer_id": None if site.layer_id is None else int(site.layer_id),
                        "equivalent_indices": list(site.equivalent_indices),
                        "output_path": str(output_path),
                        "atom_count": int(new_record.natoms),
                        "defect_position_direct": [
                            round(float(value), 8)
                            for value in (site.direct if defect_direct is None else defect_direct)
                        ],
                        **({} if contact is None else {"defect_contact": contact.as_dict()}),
                    }
                )

        artifacts = {"analysis_json": str(analysis_json), "structures": written}
        if host_supercell_path is not None:
            artifacts["host_supercell"] = str(host_supercell_path)
        dilution = dilution_report(
            lattice=record.lattice,
            structure_kind=analysis.structure_kind,
            host_atoms=int(record.natoms),
            defect_type=defect_type,
        )
        selected_layers = sorted(
            {int(site.layer_id) for site in selected_sites if site.layer_id is not None}
        )
        summary = {
            "defect_type": str(defect_type).lower(),
            "generated": len(written),
            "structure_kind": analysis.structure_kind,
            "view_direction": (analysis.view_direction or {}).get("label"),
            "atomic_planes": len(analysis.layers),
            **({} if not selected_layers else {"planes_used": selected_layers}),
            "backend": analysis.backend,
            "host_atoms": int(record.natoms),
            "defect_image_distance": float(dilution["image_distance"]),
            "image_periodicity": dilution["image_periodicity"],
            "defect_concentration_percent": 100.0 * float(dilution["concentration"]),
        }
        if chosen_supercell is not None:
            summary["supercell_cells"] = int(chosen_supercell.cells)
            summary["best_possible_image_distance"] = round(
                float(chosen_supercell.upper_bound), 4
            )
        if closest_defect_contact is not None:
            summary["closest_defect_contact"] = round(float(closest_defect_contact.distance), 4)
            summary["closest_defect_contact_pair"] = (
                f"{closest_defect_contact.first_species}-{closest_defect_contact.second_species}"
            )
        warnings = [*dilution["notes"], *contact_notes_seen, *selection_notes]
        if warnings:
            summary["warnings"] = warnings
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
                "supercell": None if repeats is None else list(repeats),
                "supercell_matrix": None if repeats is not None else matrix,
                "min_image_distance": None if min_image_distance is None else float(min_image_distance),
                "view_direction": (analysis.view_direction or {}).get("spec"),
                "layers": None if layers is None else str(layers),
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

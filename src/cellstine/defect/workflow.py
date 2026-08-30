"""Defect analysis and structure-generation workflow.

The layer census and symmetry grouping live in :mod:`analysis`, the site
enumerators in :mod:`sites`, the structure builders in :mod:`generation`, and the
text reports in :mod:`reporting`; this module keeps the workflow class that
drives them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.base import Base
from ..core.directions import resolve_direction
from ..core.layers import LAYER_TOLERANCE
from ..core.manifests import RunManifest
from ..core.models import CommandResult
from ..core.path_stage import MigrationPathMixin
from ..core.provenance import restage_comment
from ..core.species import expand_species
from ..core.transforms import supercell_structure
from ..io.converters import StructureConverter
from ..io.vasp import VaspIO
from .analysis import (
    _annotate_layer_census,
    _cluster_projection_layers,
    _detect_structure_kind,
    _load_analysis_file,
    _write_analysis_file,
)
from .generation import DefectGenerationMixin
from .records import DefectAnalysis
from .reporting import DefectReportingMixin, format_supercell_choice
from .sites import DefectSiteEnumerationMixin
from .supercell import DEFAULT_CELL_LIMIT, choose_supercell, supercell_table


class Defect(
    DefectSiteEnumerationMixin,
    DefectGenerationMixin,
    MigrationPathMixin,
    DefectReportingMixin,
    Base,
):
    """Analyse inequivalent defect sites, generate defect POSCARs, and chain them."""

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
        view_direction: str | None = None,
        interstitial_saddles: bool = False,
    ) -> DefectAnalysis:
        record = self.converter.read(structure_path, canonicalize=False)
        resolved_kind = _detect_structure_kind(record, structure_kind, layer_tolerance=layer_tolerance)
        resolved_backend = self._choose_defect_backend(backend, resolved_kind)
        direction = resolve_direction(record.lattice, view_direction)
        projections = direction.project(record.positions_cartesian)
        layers = _cluster_projection_layers(projections, layer_tolerance)
        notes: list[str] = []
        notes.append(f"Atomic planes are counted along the {direction.describe()}.")
        notes.extend(direction.notes)

        dataset = self._native_symmetry(record, symprec=symprec)
        notes.append(
            f"Site equivalence uses the {dataset.operation_count} space-group operations of the cell "
            f"(point group {dataset.point_group or 'unclassified'})."
        )
        if resolved_backend == "spglib":
            atom_sites = self._spglib_atom_sites(structure_path, record, layers=layers, symprec=symprec)
            notes.append("Exact Wyckoff labels are supplied by direct spglib for atom sites.")
        else:
            atom_sites = self._native_atom_sites(record, layers=layers, dataset=dataset)
            notes.append("Wyckoff labels are only guaranteed with the spglib backend.")

        _annotate_layer_census(
            layers, expand_species(record.species, record.counts), atom_sites
        )

        sites = list(atom_sites)
        interstitials, interstitial_notes = self._interstitial_sites(
            record,
            dataset=dataset,
            symprec=symprec,
            layers=layers,
            unit=direction.unit,
            interstitial_saddles=bool(interstitial_saddles),
        )
        sites.extend(interstitials)
        notes.extend(interstitial_notes)
        if resolved_kind in {"surface", "slab", "molecule-on-substrate"}:
            adatom_sites, adatom_counts, error = self._adatom_sites(
                structure_path,
                surface_side=surface_side,
                layer_tolerance=layer_tolerance,
                lattice=np.asarray(record.lattice, dtype=float),
                dataset=dataset,
                symprec=symprec,
            )
            sites.extend(adatom_sites)
            if error:
                notes.append(f"Adatom site detection skipped: {error}")
            elif adatom_counts:
                notes.append(f"Inequivalent adatom sites per family: {adatom_counts}")

        divacancies = self._divacancy_sites(
            record,
            backend=resolved_backend,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
            dataset=dataset,
            layers=layers,
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
            point_group=dataset.point_group or None,
            operation_count=int(dataset.operation_count),
            view_direction=direction.as_dict(),
        )

    def analyse(
        self,
        structure_path: str,
        *,
        structure_kind: str = "auto",
        backend: str = "auto",
        surface_side: str = "top",
        layer_tolerance: float = LAYER_TOLERANCE,
        symprec: float = 0.01,
        divacancy_distance: float = 3.5,
        view_direction: str | None = None,
        interstitial_saddles: bool = False,
    ) -> CommandResult:
        """Analyse inequivalent atom and insertion sites for a structure.

        ``interstitial_saddles`` adds the saddles of the distance to the nearest
        atom to the interstitial candidates -- the sites held by two or three
        atoms, such as the octahedral site of a body-centred cubic metal and the
        bond centre of a covalent crystal.
        """

        source = str(Path(structure_path).resolve())
        analysis = self._analyse_record(
            source,
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
            view_direction=view_direction,
            interstitial_saddles=interstitial_saddles,
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
            "view_direction": (analysis.view_direction or {}).get("label"),
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
                "view_direction": (analysis.view_direction or {}).get("spec"),
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

    def supercell(
        self,
        structure_path: str,
        *,
        min_image_distance: float | None = None,
        max_cells: int | None = None,
        structure_kind: str = "auto",
        layer_tolerance: float = LAYER_TOLERANCE,
        cell_limit: int = DEFAULT_CELL_LIMIT,
        table_limit: int = 0,
        output_path: str | None = None,
    ) -> CommandResult:
        """Build the host supercell a point defect should be made in.

        The cell is chosen for the distance it puts between the defect and its
        periodic images, not for the number of atoms: every sublattice of the
        host lattice of a given index is enumerated in Hermite normal form and
        the roundest one wins, which is usually not a plain repeat.  With
        ``min_image_distance`` the smallest cell that reaches the requested
        separation is written; with ``max_cells`` the best cell of at most that
        many host cells is written instead.  A slab is treated as periodic in
        the plane only, since its images along ``c`` are held apart by vacuum.
        """

        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        kind = _detect_structure_kind(record, structure_kind, layer_tolerance=float(layer_tolerance))
        choice = choose_supercell(
            record.lattice,
            structure_kind=kind,
            min_image_distance=min_image_distance,
            max_cells=max_cells,
            cell_limit=int(cell_limit),
        )
        built = supercell_structure(record, choice.matrix)
        built.comment = restage_comment(
            record.comment,
            "defect supercell",
            f"{choice.cells} host cell(s), image separation {choice.image_distance:.2f} A",
        )
        run_id, run_dir = self.create_run_dir("supercell", label=Path(source).stem)
        destination = (
            Path(output_path).resolve()
            if output_path is not None
            else run_dir / f"host_supercell_{choice.cells}x.vasp"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = self.vasp_io.write(
            built, str(destination), positions_are_cartesian=False, wrap_positions=False
        )
        payload: dict[str, Any] = {"supercell": choice.as_dict()}
        if int(table_limit) > 0:
            payload["supercell_table"] = supercell_table(
                record.lattice, structure_kind=kind, max_cells=int(table_limit)
            )
        report_path = run_dir / "supercell.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifacts = {"structure": str(written), "supercell_json": str(report_path)}
        summary = {
            "structure_kind": kind,
            "cells": int(choice.cells),
            "host_atoms": int(record.natoms),
            "supercell_atoms": int(built.natoms),
            "defect_image_distance": round(float(choice.image_distance), 4),
            "image_periodicity": choice.periodicity,
            "best_possible_distance": round(float(choice.upper_bound), 4),
            "diagonal_supercell": bool(choice.is_diagonal),
        }
        if choice.diagonal_distance is not None:
            summary["best_diagonal_distance"] = round(float(choice.diagonal_distance), 4)
        manifest_path = self.write_manifest(
            stage="supercell",
            run_id=run_id,
            run_dir=run_dir,
            backend="native",
            inputs={"structure_path": source},
            parameters={
                "structure_kind": structure_kind,
                "min_image_distance": None if min_image_distance is None else float(min_image_distance),
                "max_cells": None if max_cells is None else int(max_cells),
                "cell_limit": int(cell_limit),
                "table_limit": int(table_limit),
            },
            artifacts=artifacts,
            summary=summary,
        )
        payload["supercell_preview"] = format_supercell_choice(choice, payload.get("supercell_table"))
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload=payload,
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
        view_direction: str | None = None,
        interstitial_saddles: bool = False,
    ) -> DefectAnalysis:
        source = Path(path_or_manifest).resolve()
        saved: DefectAnalysis | None = None
        if source.name == "manifest.json":
            manifest = RunManifest.load(source)
            if "analysis_json" in manifest.artifacts:
                saved = _load_analysis_file(Path(str(manifest.artifacts["analysis_json"])).resolve())
            elif "structure_path" in manifest.inputs:
                return self._analyse_record(
                    str(manifest.inputs["structure_path"]),
                    structure_kind=structure_kind,
                    backend=backend,
                    surface_side=surface_side,
                    layer_tolerance=layer_tolerance,
                    symprec=symprec,
                    divacancy_distance=divacancy_distance,
                    view_direction=view_direction,
                    interstitial_saddles=interstitial_saddles,
                )
        elif source.suffix.lower() == ".json":
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema") == "cellstine.defect_analysis.v1":
                saved = DefectAnalysis.from_dict(payload)
        if saved is not None:
            if self._matches_view_direction(saved, view_direction):
                return saved
            # The planes of a saved analysis were counted along another
            # direction, so it cannot answer a question about these ones.
            return self._analyse_record(
                saved.structure_path,
                structure_kind=structure_kind,
                backend=backend,
                surface_side=surface_side,
                layer_tolerance=layer_tolerance,
                symprec=symprec,
                divacancy_distance=divacancy_distance,
                view_direction=view_direction,
                interstitial_saddles=interstitial_saddles,
            )
        return self._analyse_record(
            str(source),
            structure_kind=structure_kind,
            backend=backend,
            surface_side=surface_side,
            layer_tolerance=layer_tolerance,
            symprec=symprec,
            divacancy_distance=divacancy_distance,
            view_direction=view_direction,
            interstitial_saddles=interstitial_saddles,
        )

    @staticmethod
    def _matches_view_direction(analysis: DefectAnalysis, view_direction: str | None) -> bool:
        """Whether a saved analysis was read along the requested direction."""

        if view_direction is None:
            return True
        saved = (analysis.view_direction or {}).get("spec")
        if saved is None:
            return False
        return str(saved).strip().lower() == str(view_direction).strip().lower()




    def preview(
        self,
        analysis_or_structure: str,
        *,
        limit: int = 30,
        structure_kind: str = "auto",
        backend: str = "auto",
        surface_side: str = "top",
        layer_tolerance: float = LAYER_TOLERANCE,
        symprec: float = 0.01,
        divacancy_distance: float = 3.5,
        view_direction: str | None = None,
        interstitial_saddles: bool = False,
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
            view_direction=view_direction,
            interstitial_saddles=interstitial_saddles,
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
            "layers": len(analysis.layers),
            "view_direction": (analysis.view_direction or {}).get("label"),
        }
        manifest_path = self.write_manifest(
            stage="preview",
            run_id=run_id,
            run_dir=run_dir,
            backend=analysis.backend,
            inputs={"analysis_or_structure": str(Path(analysis_or_structure).resolve()), "structure_path": analysis.structure_path},
            parameters={
                "limit": int(limit),
                "view_direction": (analysis.view_direction or {}).get("spec"),
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

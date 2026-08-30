"""Surface generation and site analysis under the interface group."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ...core.base import run_output_suffix
from ...core.layers import LAYER_TOLERANCE
from ...core.models import CommandResult
from ...core.naming import safe_token
from ...core.previews import format_site_report
from ...io.converters import StructureConverter
from ..workflow.interface import Interface, parse_miller_notation
from . import backend as surface_backend
from .sequence import stacking_sequence
from .termination import termination_report


def _miller_token(values: Sequence[int]) -> str:
    return "".join(str(int(value)) if int(value) >= 0 else f"m{abs(int(value))}" for value in values)


def _surface_descriptor(
    bulk_poscar: str,
    miller_values: Sequence[int],
    *,
    layers: int,
    vacuum: float,
    repeat_a: int,
    repeat_b: int,
    supercell_matrix: Sequence[int] | None,
) -> str:
    expansion = f"r{int(repeat_a)}x{int(repeat_b)}"
    if supercell_matrix:
        expansion = "m" + "_".join(str(int(value)).replace("-", "m") for value in supercell_matrix)
    return (
        f"{safe_token(Path(bulk_poscar).stem)}_hkl{_miller_token(miller_values)}"
        f"_L{int(layers):02d}_vac{safe_token(f'{float(vacuum):.2f}')}_{expansion}"
    )


def _stacking_sequence(structure) -> str:
    """Return the plane letters of a slab, bottom of the cell upwards."""

    return stacking_sequence(structure)[0]


class Surface(Interface):
    """Bulk-to-surface and adsorption-site workflow."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)

    def surface(
        self,
        *,
        bulk_poscar: str,
        miller: str | Sequence[int],
        layers: int,
        vacuum: float,
        repeat_a: int = 1,
        repeat_b: int = 1,
        min_length_a: float | None = None,
        min_length_b: float | None = None,
        supercell_matrix: Sequence[int] | None = None,
        analyse_sites: bool = False,
        output_path: str | None = None,
        sites_output_path: str | None = None,
        site_surface_side: str = "top",
    ) -> CommandResult:
        backend = self.choose_backend(feature="interface.surface")
        miller_values = parse_miller_notation(miller)
        run_id, run_dir = self.create_run_dir("surface", f"{Path(bulk_poscar).stem}_{miller_values[0]}{miller_values[1]}{miller_values[2]}")
        output_suffix = run_output_suffix(run_id)
        # Build first, then name: a minimum in-plane length turns into a repeat
        # count that only the builder knows, and the file name has to state the
        # cell that was actually produced.
        build = surface_backend.build_surface_structure(
            str(Path(bulk_poscar).resolve()),
            miller=miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(repeat_a),
            repeat_b=int(repeat_b),
            min_length_a=min_length_a,
            min_length_b=min_length_b,
            supercell_matrix=supercell_matrix,
        )
        descriptor = _surface_descriptor(
            bulk_poscar,
            miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(build.repeat_a),
            repeat_b=int(build.repeat_b),
            supercell_matrix=build.supercell_matrix,
        )
        resolved_output_path = output_path or str(self.output_root / f"{descriptor}_surface_{output_suffix}.vasp")
        resolved_sites_output_path = sites_output_path or (str(self.output_root / f"{descriptor}_sites_{output_suffix}.json") if analyse_sites else None)
        written_path = surface_backend.write_surface_poscar(resolved_output_path, build.structure, miller_values)

        site_report_path = None
        site_counts = None
        if resolved_sites_output_path is not None:
            site_run = surface_backend.find_adsorption_sites(
                build.structure,
                surface_side=str(site_surface_side),
                output_path=resolved_sites_output_path,
                source_poscar=str(written_path),
            )
            site_report_path = site_run.output_path
            site_counts = dict(site_run.site_counts)

        run = surface_backend.SurfaceRun(
            output_path=written_path,
            miller=miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            total_atoms=build.structure.natoms,
            repeat_a=int(build.repeat_a),
            repeat_b=int(build.repeat_b),
            supercell_matrix=build.supercell_matrix,
            site_output_path=site_report_path,
            site_counts=site_counts,
        )
        slab = self.converter.read(str(run.output_path), canonicalize=True)
        stacking_sequence = _stacking_sequence(slab)
        bulk = self.converter.read(str(Path(bulk_poscar).resolve()), canonicalize=False)
        termination = termination_report(
            bulk_species=bulk.species,
            bulk_counts=bulk.counts,
            slab_lattice=build.structure.lattice,
            slab_positions_cartesian=build.structure.positions_cartesian,
            slab_species=build.structure.species,
            slab_counts=build.structure.counts,
        )
        summary: dict[str, object] = {
            "total_atoms": run.total_atoms,
            "stacking_sequence": stacking_sequence,
            "stoichiometric": termination.stoichiometric,
            "symmetric_terminations": termination.symmetric_terminations,
        }
        if termination.notes:
            summary["warnings"] = list(termination.notes)
        artifacts = {"slab_poscar": run.output_path}
        if run.site_output_path is not None:
            artifacts["sites_json"] = run.site_output_path
        manifest_path = self.write_manifest(
            stage="surface",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"bulk_poscar": str(Path(bulk_poscar).resolve())},
            parameters={
                "miller": list(miller_values),
                "layers": int(layers),
                "vacuum": float(vacuum),
                "repeat_a": int(run.repeat_a),
                "repeat_b": int(run.repeat_b),
                "min_length_a": None if min_length_a is None else float(min_length_a),
                "min_length_b": None if min_length_b is None else float(min_length_b),
                "supercell_matrix": list(run.supercell_matrix or []),
                "site_surface_side": str(site_surface_side) if analyse_sites or sites_output_path else None,
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={
                "site_counts": run.site_counts or {},
                "termination": termination.summary(),
            },
        )

    def sites(
        self,
        *,
        slab_poscar: str,
        surface_side: str = "top",
        layer_tolerance: float = LAYER_TOLERANCE,
        neighbour_tolerance: float = 0.15,
        hollow_match_tolerance: float | None = None,
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="interface.sites")
        slab_path = self.resolve_results_file(slab_poscar, artifact_keys=("slab_poscar",))
        run_id, run_dir = self.create_run_dir("sites", Path(slab_path).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_output_path = output_path or str(self.output_root / f"{safe_token(Path(slab_path).stem)}_sites_{output_suffix}.json")
        report = surface_backend.find_adsorption_sites(
            slab_path,
            surface_side=str(surface_side),
            layer_tolerance=float(layer_tolerance),
            neighbour_tolerance=float(neighbour_tolerance),
            hollow_match_tolerance=hollow_match_tolerance,
            output_path=resolved_output_path,
        )
        manifest_path = self.write_manifest(
            stage="sites",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"slab_poscar": str(Path(slab_path).resolve())},
            parameters={"surface_side": str(surface_side)},
            artifacts={"sites_json": report.output_path},
            summary={"site_counts": report.site_counts, "average_coordination": report.average_top_layer_coordination},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"sites_json": report.output_path},
            summary={"site_counts": report.site_counts, "average_coordination": report.average_top_layer_coordination},
            payload={"site_preview": format_site_report(report)},
        )

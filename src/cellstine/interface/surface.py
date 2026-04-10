"""Surface generation and site analysis under the interface group."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.base import legacy_modules
from ..core.models import CommandResult
from ..io.converters import StructureConverter
from .interface import Interface, parse_miller_notation


def _stacking_sequence(structure, z_tolerance: float = 0.35, xy_tolerance: float = 1e-3) -> str:
    direct = np.asarray(structure.positions_direct, dtype=float)
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    if direct.size == 0:
        return ""
    order = np.argsort(cartesian[:, 2])
    groups = []
    current = [int(order[0])]
    last_z = float(cartesian[order[0], 2])
    for atom_index in order[1:]:
        z_value = float(cartesian[atom_index, 2])
        if abs(z_value - last_z) <= float(z_tolerance):
            current.append(int(atom_index))
        else:
            groups.append(current)
            current = [int(atom_index)]
        last_z = z_value
    groups.append(current)

    signature_to_letter = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sequence = []
    for group in groups:
        points = np.mod(direct[np.array(group, dtype=int), :2], 1.0)
        signature = tuple(sorted((round(float(point[0]) / xy_tolerance) * xy_tolerance, round(float(point[1]) / xy_tolerance) * xy_tolerance) for point in points))
        if signature not in signature_to_letter:
            signature_to_letter[signature] = letters[len(signature_to_letter) % len(letters)]
        sequence.append(signature_to_letter[signature])
    return "".join(sequence)


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
        run = legacy_modules().surface_stage.build_surface(
            str(Path(bulk_poscar).resolve()),
            miller=miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(repeat_a),
            repeat_b=int(repeat_b),
            min_length_a=min_length_a,
            min_length_b=min_length_b,
            supercell_matrix=supercell_matrix,
            output_path=output_path or str(self.output_root / f"{Path(bulk_poscar).stem}_{miller_values[0]}{miller_values[1]}{miller_values[2]}_surface.vasp"),
            sites_output_path=sites_output_path,
            analyse_sites=bool(analyse_sites),
            site_surface_side=str(site_surface_side),
        )
        slab = self.converter.read(str(run.output_path), canonicalize=True)
        stacking_sequence = _stacking_sequence(slab)
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
                "repeat_a": int(repeat_a),
                "repeat_b": int(repeat_b),
                "supercell_matrix": list(supercell_matrix or []),
            },
            artifacts=artifacts,
            summary={"total_atoms": run.total_atoms, "stacking_sequence": stacking_sequence},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary={"total_atoms": run.total_atoms, "stacking_sequence": stacking_sequence},
            payload={"site_counts": run.site_counts or {}},
        )

    def sites(
        self,
        *,
        slab_poscar: str,
        surface_side: str = "top",
        layer_tolerance: float = 0.35,
        neighbour_tolerance: float = 0.15,
        hollow_match_tolerance: float | None = None,
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="interface.sites")
        slab_path = self.resolve_results_file(slab_poscar, artifact_keys=("slab_poscar",))
        run_id, run_dir = self.create_run_dir("sites", Path(slab_path).stem)
        report = legacy_modules().surface_stage.find_adsorption_sites(
            slab_path,
            surface_side=str(surface_side),
            layer_tolerance=float(layer_tolerance),
            neighbour_tolerance=float(neighbour_tolerance),
            hollow_match_tolerance=hollow_match_tolerance,
            output_path=output_path or str(run_dir / f"{Path(slab_path).stem}_sites.json"),
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
        )

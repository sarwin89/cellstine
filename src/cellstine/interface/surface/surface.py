"""Surface generation and site analysis under the interface group."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ...core.base import run_output_suffix
from ...core.models import CommandResult
from ...core.previews import format_site_report
from ...io.converters import StructureConverter
from ..workflow.interface import Interface, parse_miller_notation
from . import backend as surface_backend


def _safe_token(value: object) -> str:
    text = str(value).strip().replace("-", "m").replace(".", "p")
    safe = [char if char.isalnum() or char in {"_", "m", "p"} else "_" for char in text]
    return "".join(safe).strip("_") or "x"


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
        f"{_safe_token(Path(bulk_poscar).stem)}_hkl{_miller_token(miller_values)}"
        f"_L{int(layers):02d}_vac{_safe_token(f'{float(vacuum):.2f}')}_{expansion}"
    )


def _stacking_sequence(structure, z_tolerance: float = 0.35, xy_tolerance: float = 1e-3) -> str:
    direct = np.asarray(structure.positions_direct, dtype=float)
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    if direct.size == 0:
        return ""
    lattice = np.asarray(structure.lattice, dtype=float)
    normal = np.cross(lattice[0], lattice[1])
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        return ""
    normal = normal / normal_length
    projections = cartesian @ normal
    order = np.argsort(projections)
    groups = []
    current = [int(order[0])]
    last_z = float(projections[order[0]])
    for atom_index in order[1:]:
        z_value = float(projections[atom_index])
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
        points[np.isclose(points, 1.0, atol=xy_tolerance)] = 0.0
        points[np.isclose(points, 0.0, atol=xy_tolerance)] = 0.0
        signature = tuple(
            sorted(
                (
                    round(float(point[0]) / xy_tolerance) * xy_tolerance,
                    round(float(point[1]) / xy_tolerance) * xy_tolerance,
                )
                for point in points
            )
        )
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
        output_suffix = run_output_suffix(run_id)
        descriptor = _surface_descriptor(
            bulk_poscar,
            miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(repeat_a),
            repeat_b=int(repeat_b),
            supercell_matrix=supercell_matrix,
        )
        resolved_output_path = output_path or str(self.output_root / f"{descriptor}_surface_{output_suffix}.vasp")
        resolved_sites_output_path = sites_output_path or (str(self.output_root / f"{descriptor}_sites_{output_suffix}.json") if analyse_sites else None)
        run = surface_backend.build_surface(
            str(Path(bulk_poscar).resolve()),
            miller=miller_values,
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(repeat_a),
            repeat_b=int(repeat_b),
            min_length_a=min_length_a,
            min_length_b=min_length_b,
            supercell_matrix=supercell_matrix,
            output_path=resolved_output_path,
            sites_output_path=resolved_sites_output_path,
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
        output_suffix = run_output_suffix(run_id)
        resolved_output_path = output_path or str(self.output_root / f"{_safe_token(Path(slab_path).stem)}_sites_{output_suffix}.json")
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

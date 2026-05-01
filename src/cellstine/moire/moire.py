"""Bilayer moire workflow wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.base import Base, run_output_suffix
from ..core.lattice import apply_inplane_prestrain
from ..core.models import CommandResult, PrestrainConfig
from ..core.previews import format_bilayer_candidates
from ..io.converters import StructureConverter
from .find import run_find
from .make import generate_many_from_results
from .molecule import shift_top_layer


def _safe_token(value: object) -> str:
    text = str(value).strip().replace("-", "m").replace(".", "p")
    safe = [char if char.isalnum() or char in {"_", "m", "p"} else "_" for char in text]
    return "".join(safe).strip("_") or "x"


class Moire(Base):
    """Class-first bilayer commensuration workflow."""

    workflow_name = "moire"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)

    def find(
        self,
        *,
        top_poscar: str,
        bottom_poscar: str,
        nindex: int,
        min_angle: float = 0.0,
        max_angle: float | None = None,
        angle_step: float = 0.1,
        explicit_angles: Sequence[float] | None = None,
        angle_length_tolerance: float = 1e-5,
        angle_strain_tolerance: float | None = 2e-3,
        angle_merge_tolerance: float = 1e-3,
        vector_tolerance: float = 2e-3,
        vector_strain_tolerance: float | None = 2e-3,
        candidate_tolerance: float | None = None,
        strain_tolerance: float | None = None,
        strain_layer: str = "avg",
        min_atoms: int | None = None,
        max_atoms: int | None = 2000,
        dedupe: bool = True,
        unique_strain_tolerance: float = 1e-4,
        unique_ratio_tolerance: float = 1e-5,
        matrix_values: Sequence[int] | None = None,
        matrix_layer: str = "either",
        matrix_match_mode: str = "absolute",
        top_c_repeat: int = 1,
        bottom_c_repeat: int = 1,
        workers: int = 1,
        prestrain_top: PrestrainConfig | None = None,
        prestrain_bottom: PrestrainConfig | None = None,
        preview_limit: int = 10,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.find")
        top = self.converter.read(top_poscar)
        bottom = self.converter.read(bottom_poscar)
        top_prestrain = prestrain_top or PrestrainConfig()
        bottom_prestrain = prestrain_bottom or PrestrainConfig()
        top_lattice = apply_inplane_prestrain(top.lattice, mode=top_prestrain.mode, magnitude=top_prestrain.magnitude, axis=top_prestrain.axis)
        bottom_lattice = apply_inplane_prestrain(bottom.lattice, mode=bottom_prestrain.mode, magnitude=bottom_prestrain.magnitude, axis=bottom_prestrain.axis)

        label = f"{Path(bottom_poscar).stem}_{Path(top_poscar).stem}"
        run_id, run_dir = self.create_run_dir("find", label)
        run = run_find(
            top_poscar=str(Path(top_poscar).resolve()),
            bottom_poscar=str(Path(bottom_poscar).resolve()),
            top_lattice=top_lattice,
            bottom_lattice=bottom_lattice,
            top_atoms=top.natoms,
            bottom_atoms=bottom.natoms,
            nindex=int(nindex),
            min_angle=float(min_angle),
            max_angle=max_angle,
            angle_step=float(angle_step),
            explicit_angles=explicit_angles,
            angle_length_tolerance=float(angle_length_tolerance),
            angle_strain_tolerance=angle_strain_tolerance,
            angle_merge_tolerance=float(angle_merge_tolerance),
            vector_tolerance=float(vector_tolerance),
            vector_strain_tolerance=vector_strain_tolerance,
            candidate_tolerance=candidate_tolerance,
            strain_tolerance=strain_tolerance,
            strain_layer=str(strain_layer),
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            dedupe=bool(dedupe),
            unique_strain_tolerance=float(unique_strain_tolerance),
            unique_ratio_tolerance=float(unique_ratio_tolerance),
            matrix_values=matrix_values,
            matrix_layer=str(matrix_layer),
            matrix_match_mode=str(matrix_match_mode),
            output_root=str(run_dir),
            top_c_repeat=int(top_c_repeat),
            bottom_c_repeat=int(bottom_c_repeat),
            workers=int(workers),
        )
        manifest_path = self.write_manifest(
            stage="find",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"top_poscar": str(Path(top_poscar).resolve()), "bottom_poscar": str(Path(bottom_poscar).resolve())},
            parameters={
                "nindex": int(nindex),
                "min_angle": float(min_angle),
                "max_angle": max_angle,
                "angle_step": float(angle_step),
                "explicit_angles": list(explicit_angles or []),
                "workers": int(workers),
                "matrix_values": list(matrix_values or []),
                "matrix_layer": str(matrix_layer),
                "matrix_match_mode": str(matrix_match_mode),
                "prestrain_top": top_prestrain,
                "prestrain_bottom": bottom_prestrain,
            },
            artifacts={"results_dat": run.dat_path},
            summary={
                "candidate_count": len(run.candidates),
                "shortlisted_angle_count": len(run.shortlisted_angles),
                "symmetry_lcm_deg": run.symmetry_lcm,
            },
        )
        preview = format_bilayer_candidates(run.candidates, limit=int(preview_limit)) if int(preview_limit) > 0 else ""
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"results_dat": run.dat_path},
            summary={"candidate_count": len(run.candidates), "symmetry_lcm_deg": run.symmetry_lcm},
            payload={
                "run_id": run.run_id,
                "search_min_angle": run.search_min_angle,
                "search_max_angle": run.search_max_angle,
                "candidate_preview": preview,
            },
        )

    def make(
        self,
        *,
        results_file: str,
        indexes: Sequence[int],
        interlayer_distance: float,
        output_dir: str | None = None,
        output_path: str | None = None,
        tolerance: int = 1,
        tolerance_float: float = 1e-4,
        zfix: float | None = None,
        top_c_repeat: int | None = None,
        bottom_c_repeat: int | None = None,
        workers: int = 1,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.make")
        resolved_results = self.resolve_results_file(results_file, artifact_keys=("results_dat",))
        run_id, run_dir = self.create_run_dir("make", Path(resolved_results).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_output_dir = output_dir or str(self.output_root / run_id)
        runs = generate_many_from_results(
            resolved_results,
            indexes=[int(value) for value in indexes],
            interlayer_distance=float(interlayer_distance),
            output_path=output_path if len(indexes) == 1 else None,
            output_dir=resolved_output_dir,
            tolerance=int(tolerance),
            tolerance_float=float(tolerance_float),
            zfix=zfix,
            top_c_repeat=top_c_repeat,
            bottom_c_repeat=bottom_c_repeat,
            workers=int(workers),
        )
        if output_path is None and output_dir is None:
            for run in runs:
                current_path = Path(run.output_path)
                renamed_path = current_path.with_name(f"{current_path.stem}_{output_suffix}{current_path.suffix}")
                current_path.replace(renamed_path)
                run.output_path = renamed_path.resolve()
        artifact_paths = [str(run.output_path) for run in runs]
        manifest_path = self.write_manifest(
            stage="make",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(resolved_results).resolve())},
            parameters={"indexes": [int(value) for value in indexes], "interlayer_distance": float(interlayer_distance), "workers": int(workers)},
            artifacts={"structures": artifact_paths},
            summary={"generated_count": len(runs), "total_atoms": [int(run.total_atoms) for run in runs]},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"structures": artifact_paths},
            summary={"generated_count": len(runs)},
            payload={"angles_deg": [float(run.angle_deg) for run in runs]},
        )

    def translate(
        self,
        *,
        poscar_path: str,
        shift_cartesian: Sequence[float] | None = None,
        shift_direct: Sequence[float] | None = None,
        z_cutoff: float | None = None,
        min_gap: float = 1.0,
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.translate")
        run_id, run_dir = self.create_run_dir("translate", Path(poscar_path).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_output_path = output_path or str(
            self.output_root / f"{_safe_token(Path(poscar_path).stem)}_upper_layer_shifted_{output_suffix}.vasp"
        )
        run = shift_top_layer(
            poscar_path=str(Path(poscar_path).resolve()),
            output_path=resolved_output_path,
            shift_cartesian=shift_cartesian,
            shift_direct=shift_direct,
            z_cutoff=z_cutoff,
            min_gap=float(min_gap),
        )
        manifest_path = self.write_manifest(
            stage="translate",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"poscar_path": str(Path(poscar_path).resolve())},
            parameters={"shift_cartesian": list(shift_cartesian or []), "shift_direct": list(shift_direct or [])},
            artifacts={"output_poscar": run.output_path},
            summary={"top_atom_count": run.top_atom_count, "bottom_atom_count": run.bottom_atom_count},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"output_poscar": run.output_path},
            summary={"top_atom_count": run.top_atom_count, "bottom_atom_count": run.bottom_atom_count},
            payload={"shift_direct": run.shift_direct, "shift_cartesian": run.shift_cartesian},
        )

    def visualize(
        self,
        *,
        results_file: str,
        indices: Sequence[int] | None = None,
        output_path: str | None = None,
        interlayer: float = 3.35,
        top_c_repeat: int | None = None,
        bottom_c_repeat: int | None = None,
        plotly: bool = False,
        show: bool = False,
    ) -> CommandResult:
        from ..visualize.visualize import Visualize

        visualizer = Visualize(backend=self.backend, runs_root=self.runs_root, output_root=self.output_root, dependency_manager=self.dependency_manager)
        return visualizer.moire_results(
            results_file=results_file,
            indices=indices,
            output_path=output_path,
            interlayer=interlayer,
            top_c_repeat=top_c_repeat,
            bottom_c_repeat=bottom_c_repeat,
            plotly=plotly,
            show=show,
        )

    def resolve_results_file(self, path_or_manifest: str, artifact_keys: Sequence[str]) -> str:
        candidate = Path(path_or_manifest).resolve()
        if candidate.name == "manifest.json":
            from ..core.manifests import RunManifest

            manifest = RunManifest.load(candidate)
            for key in artifact_keys:
                if key in manifest.artifacts:
                    return str(Path(str(manifest.artifacts[key])).resolve())
            raise ValueError(f"{candidate} does not contain any of the requested artifacts: {', '.join(artifact_keys)}")
        return str(candidate)

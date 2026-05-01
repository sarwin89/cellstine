"""N-layer moire workflow wrapper."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

from ..core.base import run_output_suffix
from ..core.lattice import apply_inplane_prestrain
from ..core.models import CommandResult, PrestrainConfig
from ..core.previews import format_bilayer_candidates, format_nlayer_candidates
from .find import run_find
from .findn import run_findn
from .maken import generate_many_from_results
from .moire import Moire


class Supermoire(Moire):
    """Multi-layer commensuration workflow."""

    def findn(
        self,
        *,
        bottom_poscar: str,
        upper_poscars: Sequence[str],
        nindex: int,
        match_mode: str = "base_shared",
        min_angles: Sequence[float] | None = None,
        max_angles: Sequence[float] | None = None,
        angle_step: float = 0.1,
        explicit_angles_by_layer: Sequence[Sequence[float] | None] | None = None,
        angle_length_tolerance: float = 1e-5,
        angle_strain_tolerance: float | None = 2e-3,
        angle_merge_tolerance: float = 1e-3,
        vector_tolerance: float = 2e-3,
        vector_strain_tolerance: float | None = 2e-3,
        candidate_tolerance: float | None = None,
        pair_strain_tolerance: float | None = None,
        max_atoms: int | None = 2000,
        bottom_c_repeat: int = 1,
        upper_c_repeats: Sequence[int] | None = None,
        workers: int = 1,
        prestrains: Sequence[PrestrainConfig] | None = None,
        preview_limit: int = 10,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.findn")
        resolved_mode = str(match_mode).lower()
        label = f"{Path(bottom_poscar).stem}_{len(upper_poscars) + 1}layers"
        run_id, run_dir = self.create_run_dir("findn", label)

        bottom = self.converter.read(bottom_poscar)
        uppers = [self.converter.read(path) for path in upper_poscars]
        all_prestrains = list(prestrains or [PrestrainConfig()] * (len(upper_poscars) + 1))
        if len(all_prestrains) < len(upper_poscars) + 1:
            all_prestrains.extend([PrestrainConfig()] * (len(upper_poscars) + 1 - len(all_prestrains)))

        bottom_lattice = apply_inplane_prestrain(
            bottom.lattice,
            mode=all_prestrains[0].mode,
            magnitude=all_prestrains[0].magnitude,
            axis=all_prestrains[0].axis,
        )
        upper_lattices = [
            apply_inplane_prestrain(
                structure.lattice,
                mode=prestrain.mode,
                magnitude=prestrain.magnitude,
                axis=prestrain.axis,
            )
            for structure, prestrain in zip(uppers, all_prestrains[1:])
        ]
        resolved_min_angles = list(min_angles or [0.0] * len(upper_poscars))
        resolved_max_angles = list(max_angles or [60.0] * len(upper_poscars))

        if resolved_mode == "base_shared":
            run = run_findn(
                bottom_poscar=str(Path(bottom_poscar).resolve()),
                upper_poscars=[str(Path(path).resolve()) for path in upper_poscars],
                bottom_lattice=bottom_lattice,
                upper_lattices=upper_lattices,
                bottom_atoms=bottom.natoms,
                upper_atoms=[structure.natoms for structure in uppers],
                nindex=int(nindex),
                min_angles=resolved_min_angles,
                max_angles=resolved_max_angles,
                angle_step=float(angle_step),
                explicit_angles_by_layer=explicit_angles_by_layer,
                angle_length_tolerance=float(angle_length_tolerance),
                angle_strain_tolerance=angle_strain_tolerance,
                angle_merge_tolerance=float(angle_merge_tolerance),
                vector_tolerance=float(vector_tolerance),
                vector_strain_tolerance=vector_strain_tolerance,
                candidate_tolerance=candidate_tolerance,
                pair_strain_tolerance=pair_strain_tolerance,
                max_atoms=max_atoms,
                output_root=str(run_dir),
                bottom_c_repeat=int(bottom_c_repeat),
                upper_c_repeats=upper_c_repeats,
                workers=int(workers),
            )
            artifacts = {"results_json": run.result_path}
            summary = {"candidate_count": len(run.candidates), "match_mode": resolved_mode}
            payload = {
                "result_path": str(run.result_path),
                "layer_count": len(upper_poscars) + 1,
                "candidate_preview": format_nlayer_candidates(run.candidates, limit=int(preview_limit)) if int(preview_limit) > 0 else "",
            }
        elif resolved_mode == "base_independent":
            artifacts = {}
            summary = {"match_mode": resolved_mode, "layer_count": len(upper_poscars) + 1}
            payload = {"result_paths": [], "candidate_preview": ""}
            preview_sections = []
            for index, (upper_path, upper, upper_lattice) in enumerate(zip(upper_poscars, uppers, upper_lattices), start=1):
                subdir = run_dir / f"upper_{index:02d}"
                subdir.mkdir(parents=True, exist_ok=True)
                run = run_find(
                    top_poscar=str(Path(upper_path).resolve()),
                    bottom_poscar=str(Path(bottom_poscar).resolve()),
                    top_lattice=upper_lattice,
                    bottom_lattice=bottom_lattice,
                    top_atoms=upper.natoms,
                    bottom_atoms=bottom.natoms,
                    nindex=int(nindex),
                    min_angle=float(resolved_min_angles[index - 1]),
                    max_angle=float(resolved_max_angles[index - 1]),
                    angle_step=float(angle_step),
                    explicit_angles=explicit_angles_by_layer[index - 1] if explicit_angles_by_layer else None,
                    angle_length_tolerance=float(angle_length_tolerance),
                    angle_strain_tolerance=angle_strain_tolerance,
                    angle_merge_tolerance=float(angle_merge_tolerance),
                    vector_tolerance=float(vector_tolerance),
                    vector_strain_tolerance=vector_strain_tolerance,
                    candidate_tolerance=candidate_tolerance,
                    max_atoms=max_atoms,
                    output_root=str(subdir),
                    workers=int(workers),
                )
                artifacts[f"results_dat_upper_{index}"] = run.dat_path
                summary[f"candidate_count_upper_{index}"] = len(run.candidates)
                payload["result_paths"].append(str(run.dat_path))
                if int(preview_limit) > 0:
                    preview_sections.append(format_bilayer_candidates(run.candidates, limit=int(preview_limit), title=f"Upper layer {index} candidates"))
            payload["candidate_preview"] = "\n\n".join(preview_sections)
        elif resolved_mode == "pairwise":
            artifacts = {}
            summary = {"match_mode": resolved_mode, "pair_count": 0}
            payload = {"result_paths": [], "candidate_preview": ""}
            preview_sections = []
            structures = [bottom] + uppers
            lattices = [bottom_lattice] + upper_lattices
            paths = [bottom_poscar] + list(upper_poscars)
            for pair_index, (i_value, j_value) in enumerate(combinations(range(len(paths)), 2), start=1):
                subdir = run_dir / f"pair_{i_value + 1}_{j_value + 1}"
                subdir.mkdir(parents=True, exist_ok=True)
                run = run_find(
                    top_poscar=str(Path(paths[j_value]).resolve()),
                    bottom_poscar=str(Path(paths[i_value]).resolve()),
                    top_lattice=lattices[j_value],
                    bottom_lattice=lattices[i_value],
                    top_atoms=structures[j_value].natoms,
                    bottom_atoms=structures[i_value].natoms,
                    nindex=int(nindex),
                    min_angle=0.0,
                    max_angle=60.0,
                    angle_step=float(angle_step),
                    angle_length_tolerance=float(angle_length_tolerance),
                    angle_strain_tolerance=angle_strain_tolerance,
                    angle_merge_tolerance=float(angle_merge_tolerance),
                    vector_tolerance=float(vector_tolerance),
                    vector_strain_tolerance=vector_strain_tolerance,
                    candidate_tolerance=candidate_tolerance,
                    max_atoms=max_atoms,
                    output_root=str(subdir),
                    workers=int(workers),
                )
                artifacts[f"results_dat_pair_{pair_index}"] = run.dat_path
                payload["result_paths"].append(str(run.dat_path))
                summary["pair_count"] = int(summary["pair_count"]) + 1
                if int(preview_limit) > 0:
                    preview_sections.append(
                        format_bilayer_candidates(
                            run.candidates,
                            limit=int(preview_limit),
                            title=f"Pair {i_value + 1}-{j_value + 1} candidates",
                        )
                    )
            payload["candidate_preview"] = "\n\n".join(preview_sections)
        else:
            raise ValueError("findn match_mode must be one of: base_shared, base_independent, pairwise")

        manifest_path = self.write_manifest(
            stage="findn",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"bottom_poscar": str(Path(bottom_poscar).resolve()), "upper_poscars": [str(Path(path).resolve()) for path in upper_poscars]},
            parameters={"nindex": int(nindex), "match_mode": resolved_mode, "workers": int(workers), "prestrains": all_prestrains},
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload=payload,
        )

    def maken(
        self,
        *,
        results_file: str,
        indexes: Sequence[int],
        interlayers: Sequence[float],
        output_dir: str | None = None,
        bottom_c_repeat: int | None = None,
        upper_c_repeats: Sequence[int] | None = None,
        zfix: float | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.maken")
        resolved_results = self.resolve_results_file(results_file, artifact_keys=("results_json",))
        run_id, run_dir = self.create_run_dir("maken", Path(resolved_results).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_output_dir = output_dir or str(self.output_root / run_id)
        runs = generate_many_from_results(
            resolved_results,
            indexes=[int(value) for value in indexes],
            interlayers=[float(value) for value in interlayers],
            output_dir=resolved_output_dir,
            bottom_c_repeat=bottom_c_repeat,
            upper_c_repeats=upper_c_repeats,
            zfix=zfix,
        )
        if output_dir is None:
            for run in runs:
                current_path = Path(run.output_path)
                renamed_path = current_path.with_name(f"{current_path.stem}_{output_suffix}{current_path.suffix}")
                current_path.replace(renamed_path)
                run.output_path = renamed_path.resolve()
        artifact_paths = [str(run.output_path) for run in runs]
        manifest_path = self.write_manifest(
            stage="maken",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(resolved_results).resolve())},
            parameters={"indexes": [int(value) for value in indexes], "interlayers": [float(value) for value in interlayers]},
            artifacts={"structures": artifact_paths},
            summary={"generated_count": len(runs)},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"structures": artifact_paths},
            summary={"generated_count": len(runs)},
            payload={"angles_deg": [list(run.angles_deg) for run in runs]},
        )

    def translaten(self, **kwargs) -> CommandResult:
        return self.translate(**kwargs)

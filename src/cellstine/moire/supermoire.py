"""Multi-layer moire workflow: search and build stacks of three or more layers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.base import run_output_suffix
from ..core.models import CommandResult
from ..core.previews import format_nlayer_candidates
from .builder.nlayer import generate_many_from_results
from .search.nlayer import read_nlayer_results, run_findn
from .moire import Moire


class Supermoire(Moire):
    """Commensuration and construction for a base layer plus several upper layers.

    The base layer is held rigid and every upper layer is matched against it with
    the bilayer Gram engine; the shared cell of the whole stack is the exact
    integer intersection of the per-layer base supercells.  See
    :mod:`cellstine.moire.search.nlayer` for the construction.
    """

    def findn(
        self,
        *,
        base_poscar: str,
        upper_poscars: Sequence[str],
        max_length: float,
        layer_strains: Sequence[float] | float = 0.02,
        min_length: float | None = None,
        max_atoms: int | None = 2000,
        max_pair_atoms: int | None = None,
        max_aspect_ratio: float = 12.0,
        min_cell_angle_deg: float = 25.0,
        max_cell_angle_deg: float = 155.0,
        per_layer_limit: int = 40,
        max_candidates: int = 200,
        reduce_layers: bool = True,
        preview_limit: int = 10,
    ) -> CommandResult:
        """Search commensurate cells for a rigid base layer plus upper layers."""

        backend = self.choose_backend(feature="moire.findn")
        label = f"{Path(base_poscar).stem}_{len(list(upper_poscars)) + 1}layers"
        run_id, run_dir = self.create_run_dir("findn", label)
        run = run_findn(
            base_poscar=str(Path(base_poscar).resolve()),
            upper_poscars=[str(Path(path).resolve()) for path in upper_poscars],
            max_length=float(max_length),
            layer_strains=layer_strains,
            min_length=min_length,
            max_atoms=max_atoms,
            max_pair_atoms=max_pair_atoms,
            max_aspect_ratio=float(max_aspect_ratio),
            min_cell_angle_deg=float(min_cell_angle_deg),
            max_cell_angle_deg=float(max_cell_angle_deg),
            per_layer_limit=int(per_layer_limit),
            max_candidates=int(max_candidates),
            reduce_layers=bool(reduce_layers),
            output_root=str(run_dir),
        )
        artifacts = {"results_json": str(run.result_path)}
        summary = {
            "candidate_count": len(run.candidates),
            "layer_count": len(list(upper_poscars)) + 1,
            "smallest_total_atoms": min((int(item["total_atoms"]) for item in run.candidates), default=0),
        }
        manifest_path = self.write_manifest(
            stage="findn",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "base_poscar": str(Path(base_poscar).resolve()),
                "upper_poscars": [str(Path(path).resolve()) for path in upper_poscars],
            },
            parameters=dict(run.document["search"]),
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={
                "result_path": str(run.result_path),
                "timings": dict(run.timings),
                "candidate_preview": format_nlayer_candidates(run.candidates, limit=int(preview_limit))
                if int(preview_limit) > 0
                else "",
            },
        )

    def maken(
        self,
        *,
        results_file: str,
        indexes: Sequence[int],
        interlayers: Sequence[float] | float = 3.35,
        output_dir: str | None = None,
        vacuum: float | None = None,
        base_c_repeat: int = 1,
        upper_c_repeats: Sequence[int] | None = None,
        zfix: float | None = None,
    ) -> CommandResult:
        """Build one structure per selected multi-layer candidate."""

        backend = self.choose_backend(feature="moire.maken")
        resolved_results = self.resolve_results_file(results_file, artifact_keys=("results_json",))
        document = read_nlayer_results(resolved_results)
        layer_count = len(document["search"]["upper_poscars"])
        if isinstance(interlayers, (int, float)):
            gaps = [float(interlayers)] * layer_count
        else:
            gaps = [float(value) for value in interlayers]
        if len(gaps) == 1 and layer_count > 1:
            gaps = gaps * layer_count
        if len(gaps) != layer_count:
            raise ValueError(f"this document has {layer_count} upper layer(s); give one interlayer distance each")

        run_id, run_dir = self.create_run_dir("maken", Path(resolved_results).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_output_dir = output_dir or str(self.output_root / run_id)
        runs = generate_many_from_results(
            resolved_results,
            indexes=[int(value) for value in indexes],
            interlayers=gaps,
            output_dir=resolved_output_dir,
            vacuum=None if vacuum is None else float(vacuum),
            base_c_repeat=int(base_c_repeat),
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
        artifacts = {"structures": artifact_paths}
        summary = {
            "generated_count": len(runs),
            "total_atoms": [int(run.total_atoms) for run in runs],
        }
        manifest_path = self.write_manifest(
            stage="maken",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(resolved_results).resolve())},
            parameters={
                "indexes": [int(value) for value in indexes],
                "interlayers": gaps,
                "vacuum": None if vacuum is None else float(vacuum),
                "base_c_repeat": int(base_c_repeat),
                "upper_c_repeats": None if upper_c_repeats is None else [int(value) for value in upper_c_repeats],
                "zfix": zfix,
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
                "angles_deg": [list(run.angles_deg) for run in runs],
                "layer_atom_counts": [list(run.layer_counts) for run in runs],
            },
        )

"""Matplotlib-first visualizers for results files and POSCAR structures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.base import Base, run_output_suffix
from ..core.directions import resolve_direction
from ..core.models import CommandResult
from ..io.converters import StructureConverter
from ..moire.search.results import read_results
from .backends.matplotlib import plot_moire_summary, plot_structure_multiview
from .backends.plotly import write_structure_html
from .results.plotly import build_visualization


class Visualize(Base):
    """Shared visualizer for grouped workflows."""

    workflow_name = "visualize"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)

    def moire_results(
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
        run_id, run_dir = self.create_run_dir("moire", Path(results_file).stem)
        output_suffix = run_output_suffix(run_id)
        resolved_results = self.resolve_results_file(
            results_file, artifact_keys=("results_json", "results_dat")
        )
        read_results(resolved_results)
        if plotly:
            output = output_path or str(self.output_root / f"moire_viewer_{output_suffix}.html")
            run = build_visualization(
                resolved_results,
                indices=indices,
                output_path=output,
                interlayer=float(interlayer),
                top_c_repeat=top_c_repeat,
                bottom_c_repeat=bottom_c_repeat,
            )
            artifacts = {"html": run.output_path}
            summary = {"frame_count": run.frame_count, "results_type": run.results_type, "visualization": "plotly_3d"}
            backend = "plotly"
        else:
            output = output_path or str(self.output_root / f"moire_summary_{output_suffix}.png")
            run = plot_moire_summary(
                resolved_results,
                indices=indices,
                output_path=output,
                show=show,
            )
            artifacts = {"png": run.output_path}
            summary = {"candidate_count": run.item_count, "results_type": run.visualization_type, "visualization": "matplotlib_static"}
            backend = "matplotlib"

        manifest_path = self.write_manifest(
            stage="moire",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(resolved_results).resolve())},
            parameters={
                "indices": list(indices or []),
                "interlayer": float(interlayer),
                "plotly": bool(plotly),
                "show": bool(show),
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
        )

    def structure(
        self,
        *,
        structure_path: str,
        output_path: str | None = None,
        title: str | None = None,
        plotly: bool = False,
        show: bool = False,
        view_direction: str | None = None,
    ) -> CommandResult:
        """Draw one structure, optionally looking along a chosen direction.

        ``view_direction`` accepts the same specifications as the rest of the
        package -- ``auto``, an axis, a Miller plane normal such as ``111``, a
        lattice direction such as ``[110]`` -- and turns the plan view into the
        picture an observer looking along it would see.
        """

        run_id, run_dir = self.create_run_dir("structure", Path(structure_path).stem)
        output_suffix = run_output_suffix(run_id)
        record = self.converter.read(structure_path, canonicalize=True)
        direction = (
            None
            if view_direction is None or str(view_direction).strip().lower() in {"", "none", "file"}
            else resolve_direction(record.lattice, view_direction)
        )
        if plotly:
            output = output_path or str(self.output_root / f"structure_viewer_{output_suffix}.html")
            written = write_structure_html(record, output_path=output, title=title, direction=direction)
            artifacts = {"html": written}
            summary = {"atom_count": record.natoms, "visualization": "plotly_3d"}
            backend = "plotly"
        else:
            output = output_path or str(self.output_root / f"structure_multiview_{output_suffix}.png")
            run = plot_structure_multiview(
                record, output_path=output, title=title, show=show, direction=direction
            )
            artifacts = {"png": run.output_path}
            summary = {"atom_count": record.natoms, "visualization": "matplotlib_multiview"}
            backend = "matplotlib"
        if direction is not None:
            summary["view_direction"] = direction.label
            if direction.spacing is not None:
                summary["plane_spacing"] = round(float(direction.spacing), 4)

        manifest_path = self.write_manifest(
            stage="structure",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"structure_path": str(Path(structure_path).resolve())},
            parameters={
                "title": title,
                "plotly": bool(plotly),
                "show": bool(show),
                "view_direction": None if direction is None else direction.spec,
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
        )

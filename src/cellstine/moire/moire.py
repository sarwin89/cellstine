"""Bilayer moire workflow wrapper."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

from ..core.base import Base, run_output_suffix
from ..core.layers import shift_top_layer
from ..core.models import CommandResult
from ..core.naming import safe_token
from ..core.previews import format_bilayer_candidates
from .builder.make import generate_many_from_results
from ..core.symmetry2d import DEFAULT_SYMMETRY_TOLERANCE
from .search.find import DEFAULT_CANDIDATE_LIMIT, run_find


def _format_timing(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _progress_printer(total_steps: int = 4):
    """Return a callback that prints a bar advancing once per finished stage.

    Every stage announces itself before it starts (``reading ...``,
    ``searching ...``, ``writing ...``) and reports when it is done (``read
    ...``, ``found ...``, ``wrote ...``).  Only the reports move the bar, which
    is why the prefixes below carry a trailing space: without it ``reading input
    structures`` would be counted as a finished read and the bar would fill one
    stage ahead of the work.  The four reports of a search run -- read, found,
    wrote results, wrote manifest -- then fill the bar exactly.
    """

    state = {"step": 0}

    def _print(stage: str, message: str) -> None:
        if message.startswith(("read ", "found ", "wrote ")):
            state["step"] = min(total_steps, state["step"] + 1)
        filled = min(total_steps, max(0, int(state["step"])))
        bar = "#" * filled + "-" * (total_steps - filled)
        print(f"[{bar}] {stage}: {message}", flush=True)

    return _print


def _measured(runs, attribute: str) -> list[tuple[int, float]]:
    measured: list[tuple[int, float]] = []
    for run in runs:
        distance = float(getattr(run, attribute, float("nan")))
        if math.isfinite(distance):
            measured.append((int(getattr(run, "selected_index", 0)), distance))
    return measured


def _contact_summary(runs) -> dict[str, object]:
    """Report how close the atoms of the structures that were built really come.

    A twisted stack is built by setting an interlayer distance along the surface
    normal.  Where the two layers happen to sit atom over atom the closest
    approach is that distance, and everywhere else it is larger, so the measured
    minimum over the whole moire cell is the number that says whether the two
    surfaces are touching.  That is only the contact this stage controls: the
    closest approach *anywhere* in the written cell -- inside a layer as well,
    and over the periodic images -- is reported next to it, because a stack
    built from an unphysical input layer is unphysical however the layers are
    spaced.
    """

    measured = _measured(runs, "contact_distance")
    whole = _measured(runs, "structure_contact_distance")
    summary: dict[str, object] = {}
    if measured:
        summary["closest_interlayer_contact"] = round(min(d for _, d in measured), 4)
        if len(measured) > 1:
            # One number for a batch hides which stack it belongs to.
            summary["interlayer_contact_per_structure"] = {
                f"index {index}": round(distance, 4) for index, distance in measured
            }
    if whole:
        summary["closest_contact_in_cell"] = round(min(d for _, d in whole), 4)
    warnings: list[str] = []
    for run in runs:
        label = f"index {int(getattr(run, 'selected_index', 0))}: " if len(list(runs)) > 1 else ""
        for note in getattr(run, "notes", ()) or ():
            text = f"{label}{note}"
            if text not in warnings:
                warnings.append(text)
    if warnings:
        summary["warnings"] = warnings
    return summary


class Moire(Base):
    """Class-first native bilayer commensuration workflow."""

    workflow_name = "moire"

    def find(
        self,
        *,
        top_poscar: str,
        bottom_poscar: str,
        max_length: float,
        top_strain: float,
        bottom_strain: float,
        min_length: float | None = None,
        max_atoms: int | None = None,
        max_candidates: int | None = DEFAULT_CANDIDATE_LIMIT,
        max_aspect_ratio: float = 12.0,
        min_cell_angle_deg: float = 25.0,
        max_cell_angle_deg: float = 155.0,
        min_twist_angle_deg: float | None = None,
        max_twist_angle_deg: float | None = None,
        fold_symmetry: bool = True,
        symmetric: bool = False,
        reduce_layers: bool = True,
        symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
        preview_limit: int = 10,
        progress: bool = False,
    ) -> CommandResult:
        """Search one bilayer for commensurate supercells and record the best.

        ``max_candidates`` bounds how many candidates are written: the whole
        Pareto front of size against strain is kept and the rest of the budget
        goes to the smallest cells.  Pass ``0`` or ``None`` to record every
        admissible supercell the search found.

        Each layer is searched in its primitive in-plane cell, so a layer given
        as a supercell of itself does not multiply every reported cell by that
        factor; the reduced layer is written into the run directory and is what
        the reported matrices refer to.  ``reduce_layers=False`` keeps the cells
        exactly as they were given.
        """
        total_start = time.perf_counter()
        progress_callback = _progress_printer() if progress else None
        backend = self.choose_backend(feature="moire.find")
        label = f"{Path(bottom_poscar).stem}_{Path(top_poscar).stem}"
        run_id, run_dir = self.create_run_dir("find", label)
        run = run_find(
            top_poscar=str(Path(top_poscar).resolve()),
            bottom_poscar=str(Path(bottom_poscar).resolve()),
            max_length=float(max_length),
            top_strain=float(top_strain),
            bottom_strain=float(bottom_strain),
            min_length=None if min_length is None else float(min_length),
            max_atoms=None if max_atoms is None else int(max_atoms),
            max_candidates=None if max_candidates is None else int(max_candidates),
            max_aspect_ratio=float(max_aspect_ratio),
            min_cell_angle_deg=float(min_cell_angle_deg),
            max_cell_angle_deg=float(max_cell_angle_deg),
            min_twist_angle_deg=None if min_twist_angle_deg is None else float(min_twist_angle_deg),
            max_twist_angle_deg=None if max_twist_angle_deg is None else float(max_twist_angle_deg),
            fold_symmetry=bool(fold_symmetry),
            symmetric=bool(symmetric),
            reduce_layers=bool(reduce_layers),
            symmetry_tolerance=float(symmetry_tolerance),
            output_root=str(run_dir),
            progress_callback=progress_callback,
        )

        manifest_start = time.perf_counter()
        timings = dict(run.timings)
        manifest_path = self.write_manifest(
            stage="find",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "top_poscar": str(Path(top_poscar).resolve()),
                "bottom_poscar": str(Path(bottom_poscar).resolve()),
            },
            parameters=run.parameters,
            artifacts={"results_json": run.result_path},
            summary={
                "candidate_count": len(run.candidates),
                "candidates_found": len(run.result),
                "pareto_candidate_count": int(run.result.pareto_optimal.sum()),
                "search_branch": str(run.result.stats.get("branch", "general")),
                "top_layer_index": int(run.layer_index["top"]),
                "bottom_layer_index": int(run.layer_index["bottom"]),
                "timings_s": timings,
            },
        )
        timings["manifest_write_s"] = time.perf_counter() - manifest_start
        timings["workflow_total_s"] = time.perf_counter() - total_start
        if progress_callback:
            progress_callback(
                "manifest", f"wrote manifest in {_format_timing(timings['manifest_write_s'])}"
            )
        preview = (
            format_bilayer_candidates(run.candidates, limit=int(preview_limit))
            if int(preview_limit) > 0
            else ""
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"results_json": run.result_path},
            summary={
                "candidate_count": len(run.candidates),
                "candidates_found": len(run.result),
                "pareto_candidate_count": int(run.result.pareto_optimal.sum()),
                "workflow_total_s": round(timings["workflow_total_s"], 6),
            },
            payload={
                "run_id": run_id,
                "candidate_preview": preview,
                "timings_s": timings,
                "search_stats": run.result.stats,
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
        vacuum: float | None = None,
        workers: int = 1,
    ) -> CommandResult:
        backend = self.choose_backend(feature="moire.make")
        resolved_results = self.resolve_results_file(results_file, artifact_keys=("results_json",))
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
            vacuum=None if vacuum is None else float(vacuum),
            workers=int(workers),
        )
        if output_path is None and output_dir is None:
            for run in runs:
                current_path = Path(run.output_path)
                renamed_path = current_path.with_name(
                    f"{current_path.stem}_{output_suffix}{current_path.suffix}"
                )
                current_path.replace(renamed_path)
                run.output_path = renamed_path.resolve()
        artifact_paths = [str(run.output_path) for run in runs]
        manifest_path = self.write_manifest(
            stage="make",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"results_file": str(Path(resolved_results).resolve())},
            parameters={
                "indexes": [int(value) for value in indexes],
                "interlayer_distance": float(interlayer_distance),
                "vacuum": None if vacuum is None else float(vacuum),
                "workers": int(workers),
            },
            artifacts={"structures": artifact_paths},
            summary={
                "generated_count": len(runs),
                "total_atoms": [int(run.total_atoms) for run in runs],
                **_contact_summary(runs),
            },
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"structures": artifact_paths},
            summary={"generated_count": len(runs), **_contact_summary(runs)},
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
            self.output_root
            / f"{safe_token(Path(poscar_path).stem)}_upper_layer_shifted_{output_suffix}.vasp"
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
            parameters={
                "shift_cartesian": list(shift_cartesian or []),
                "shift_direct": list(shift_direct or []),
            },
            artifacts={"output_poscar": run.output_path},
            summary={
                "top_atom_count": run.top_atom_count,
                "bottom_atom_count": run.bottom_atom_count,
            },
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"output_poscar": run.output_path},
            summary={
                "top_atom_count": run.top_atom_count,
                "bottom_atom_count": run.bottom_atom_count,
            },
            payload={
                "shift_direct": run.shift_direct,
                "shift_cartesian": run.shift_cartesian,
            },
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

        visualizer = Visualize(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        )
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
            raise ValueError(
                f"{candidate} does not contain any of the requested artifacts: "
                f"{', '.join(artifact_keys)}"
            )
        return str(candidate)

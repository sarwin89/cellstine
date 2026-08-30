"""Interface building and bulk-surface matching."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from ...core.base import Base, run_output_suffix
from ...core.lattice import inplane_principal_log_strains
from ...core.vacuum import vacuum_gap
from ...core.models import CommandResult
from ...core.transforms import strained_copy
from ...io import native as io_mod
from ...io.converters import StructureConverter
from ...io.orientation import OrientationNormalizer
from ...io.vasp import VaspIO
from ...moire.builder import generator as moire_generator
from ...core.provenance import restage_comment, stage_comment
from ..surface import registry as registry_mod
from ..surface import stacking as stacking_mod
from . import lattice_match
from .assembly import (
    interface_contact_report,
    parse_miller_notation,
    prepare_stacking,
    reported_kind,
    safe_token,
    slab_vacuum_thickness,
    stack_structures,
)

DEFAULT_BUILD_MAX_STRAIN = 0.05
# Empty space along the surface normal, in angstrom, above which a structure is
# read as a finished slab rather than as a bulk cell to cut a slab from.  Two
# atomic planes of any element are closer together than this, so a cell with
# this much space in it has a surface.
AUTO_SLAB_VACUUM = 3.0
# A match scan crosses every Miller index with every thickness, and each pair can
# contribute hundreds of admissible supercells.  Reporting all of them buries the
# few cells anyone would actually run, so the scan keeps this many best matches.
from ...core.constants import DEFAULT_MATCH_LIMIT  # re-exported for callers



def _contact_summary(report: dict) -> dict[str, object]:
    """Report how close the two slabs of an interface really come.

    The gap an interface is built with separates the two slabs along the surface
    normal; the closest approach between an atom of one and an atom of the other
    is what says whether the boundary is physical, and for a twisted or
    laterally offset interface it is larger than the gap.
    """

    summary: dict[str, object] = {}
    distance = report.get("contact_distance")
    if distance is not None:
        summary["closest_contact"] = round(float(distance), 4)
        species = (report.get("contact") or {}).get("species")
        if species:
            summary["closest_contact_pair"] = "-".join(str(symbol) for symbol in species)
    notes = list(report.get("notes", ()))
    if notes:
        summary["warnings"] = notes
    return summary


class Interface(Base):
    """Top-level interface workflow."""

    workflow_name = "interface"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.normalizer = OrientationNormalizer()
        self.vasp_io = VaspIO()

    def _detected_input_kind(self, path: str | Path) -> str:
        """Say whether a structure file is a finished slab or a bulk cell.

        A slab is a slab because it has vacuum: the periodic images along the
        surface normal are held apart by empty space.  A bulk cell has none, so
        stacking it as if it were a slab would silently produce an interface
        with no surfaces and ignore the Miller indices and layer count the
        caller gave.  Anything with less than :data:`AUTO_SLAB_VACUUM` of empty
        space along its normal is therefore read as bulk and cut into a slab.
        """

        record = self.converter.read(str(path), canonicalize=False)
        gap = vacuum_gap(record.lattice, record.positions_cartesian)
        return "surface" if gap >= AUTO_SLAB_VACUUM else "bulk"

    def _resolve_surface_input(
        self,
        *,
        path_or_manifest: str,
        kind: str,
        miller: str | Sequence[int] | None = None,
        layers: int = 4,
        vacuum: float = 15.0,
    ) -> tuple[str, dict[str, object]]:
        resolved_kind = str(kind or "auto").lower()
        candidate = Path(path_or_manifest).resolve()
        if candidate.name == "manifest.json":
            from ...core.manifests import RunManifest

            manifest = RunManifest.load(candidate)
            if "slab_poscar" in manifest.artifacts:
                return str(Path(str(manifest.artifacts["slab_poscar"])).resolve()), {"kind": "manifest"}
            raise ValueError(f"{candidate} does not contain a slab_poscar artifact")
        detected: str | None = None
        if resolved_kind == "auto":
            detected = self._detected_input_kind(candidate)
            resolved_kind = detected
        if resolved_kind in {"surface", "slab"}:
            meta: dict[str, object] = {"kind": resolved_kind}
            if detected is not None:
                meta["detected"] = True
            return str(candidate), meta
        if resolved_kind != "bulk":
            raise ValueError("kind must be one of: auto, bulk, slab, surface")
        from ..surface.surface import Surface

        slab_result = Surface(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        ).surface(
            bulk_poscar=path_or_manifest,
            miller=miller or "1,1,1",
            layers=int(layers),
            vacuum=float(vacuum),
        )
        meta = {"kind": "bulk", "surface_manifest": str(slab_result.manifest_path)}
        if detected is not None:
            meta["detected"] = True
        return slab_result.artifacts["slab_poscar"], meta

    def _normalized_slab(self, path: str | Path) -> str:
        """Rewrite a slab in place with ``c`` along ``z`` and ``a``/``b`` in ``xy``.

        The Gram search and the shared builder both require planar in-plane
        vectors, so every slab entering a match scan is normalised once here
        instead of each caller guessing whether it already is.
        """

        destination = Path(path).resolve()
        record = self.normalizer.align_ab_to_xy(self.converter.read(str(destination)))
        self.vasp_io.write(record, str(destination), positions_are_cartesian=False, wrap_positions=False)
        return str(destination)

    def build(
        self,
        *,
        bottom_input: str | None = None,
        top_input: str | None = None,
        bottom_kind: str = "auto",
        top_kind: str = "auto",
        bottom_miller: str | Sequence[int] | None = None,
        top_miller: str | Sequence[int] | None = None,
        bottom_layers: int = 4,
        top_layers: int = 4,
        bottom_vacuum: float = 15.0,
        top_vacuum: float = 15.0,
        gap: float = 3.0,
        vacuum: float | None = None,
        output_path: str | None = None,
        match_json: str | None = None,
        match_index: int = 1,
        max_strain: float | None = DEFAULT_BUILD_MAX_STRAIN,
        bottom_stacking: str = "keep",
        top_stacking: str = "keep",
        registry: str | int | None = None,
        include_equivalent: bool = False,
    ) -> CommandResult:
        """Stack two slabs, either 1x1 or on a commensurate matched supercell.

        With ``match_json`` the interface is built on the commensurate supercell
        recorded by :meth:`match`, so both slabs are only deformed by their share
        of a small certified strain.  Without it the two primitive cells are
        stacked directly and the top slab is forced onto the bottom cell; that is
        only meaningful when the two cells already agree, so the raw mismatch is
        refused above ``max_strain``.
        """

        if match_json is not None:
            if registry is not None:
                raise ValueError(
                    "a stacking registry is only defined for two slabs on the same in-plane cell; "
                    "a matched supercell twists and strains the two slabs, so the contact varies "
                    "across the cell. Build the 1x1 interface to choose a registry."
                )
            return self._build_from_match(
                match_json=match_json,
                match_index=int(match_index),
                gap=float(gap),
                vacuum=vacuum,
                output_path=output_path,
                bottom_stacking=bottom_stacking,
                top_stacking=top_stacking,
            )
        if bottom_input is None or top_input is None:
            raise ValueError("bottom_input and top_input are required unless match_json is given")
        backend = self.choose_backend(feature="interface.build")
        run_id, run_dir = self.create_run_dir("build", f"{Path(bottom_input).stem}_{Path(top_input).stem}")
        bottom_path, bottom_meta = self._resolve_surface_input(
            path_or_manifest=bottom_input,
            kind=bottom_kind,
            miller=bottom_miller,
            layers=bottom_layers,
            vacuum=bottom_vacuum,
        )
        top_path, top_meta = self._resolve_surface_input(
            path_or_manifest=top_input,
            kind=top_kind,
            miller=top_miller,
            layers=top_layers,
            vacuum=top_vacuum,
        )
        bottom = self.normalizer.align_ab_to_xy(self.converter.read(bottom_path))
        top = self.normalizer.align_ab_to_xy(self.converter.read(top_path))
        first_strain, second_strain = inplane_principal_log_strains(bottom.lattice, top.lattice)
        raw_mismatch = max(abs(first_strain), abs(second_strain))
        if max_strain is not None and raw_mismatch > float(max_strain) + 1e-12:
            raise ValueError(
                "the two 1x1 in-plane cells differ by a principal logarithmic strain of "
                f"{raw_mismatch:.4f}, above the limit of {float(max_strain):.4f}. Run "
                "`cellstine interface match` to find a commensurate supercell and build from "
                "its matches.json, or raise --max-strain deliberately."
            )
        if vacuum is None:
            resolved_vacuum = max(
                slab_vacuum_thickness(bottom), slab_vacuum_thickness(top), float(gap)
            )
        else:
            resolved_vacuum = float(vacuum)
        strained_top = strained_copy(top, bottom.lattice)
        bottom, strained_top, stacking_summary = prepare_stacking(
            bottom,
            strained_top,
            bottom_stacking=bottom_stacking,
            top_stacking=top_stacking,
            registry=registry,
            include_equivalent=bool(include_equivalent),
        )
        final_lattice, positions_direct, counts, species, flags = stack_structures(
            bottom, strained_top, gap=float(gap), vacuum=resolved_vacuum
        )
        contacts = interface_contact_report(
            bottom, strained_top, gap=float(gap), vacuum=resolved_vacuum
        )
        if output_path is None:
            output_suffix = run_output_suffix(run_id)
            destination = self.output_root / f"interface_gap{safe_token(f'{float(gap):.2f}')}_{output_suffix}.vasp"
        else:
            destination = Path(output_path).resolve()
        output_record = bottom.copy()
        output_record.comment = restage_comment(
            bottom.comment,
            "interface build",
            f"interface with {Path(top_path).stem}",
            f"gap {float(gap):.2f} A",
        )
        output_record.lattice = final_lattice
        output_record.positions_direct = positions_direct
        output_record.positions_cartesian = io_mod.direct_to_cartesian(positions_direct, final_lattice)
        output_record.species = species
        output_record.counts = counts
        output_record.selective_flags = flags
        output_record.selective_dynamics = flags is not None
        # Both slabs are crystals, so every atom is defined only up to a lattice
        # translation: write the representatives that lie inside the cell rather
        # than the negative coordinates a mirrored slab produces.
        self.vasp_io.write(output_record, str(destination), positions_are_cartesian=False, wrap_positions=True)
        summary = {
            "raw_inplane_mismatch": float(raw_mismatch),
            "principal_log_strains": [float(first_strain), float(second_strain)],
            "vacuum": resolved_vacuum,
            "gap": float(gap),
            "c_length": float(final_lattice[2, 2]),
            "total_atoms": int(sum(counts)),
            "bottom_kind": reported_kind(bottom_meta),
            "top_kind": reported_kind(top_meta),
            **_contact_summary(contacts),
        }
        if stacking_summary is not None:
            summary["stacking"] = stacking_summary
        manifest_path = self.write_manifest(
            stage="build",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "bottom_input": str(Path(bottom_input).resolve()),
                "top_input": str(Path(top_input).resolve()),
                "bottom_meta": bottom_meta,
                "top_meta": top_meta,
            },
            parameters={
                "gap": float(gap),
                "vacuum": resolved_vacuum,
                "bottom_kind": str(bottom_kind),
                "top_kind": str(top_kind),
                "max_strain": None if max_strain is None else float(max_strain),
                "bottom_stacking": str(bottom_stacking),
                "top_stacking": str(top_stacking),
                "registry": None if registry is None else str(registry),
                "include_equivalent": bool(include_equivalent),
            },
            artifacts={"interface_poscar": destination},
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"interface_poscar": destination},
            summary=summary,
        )

    def _mirrored_slab_copy(self, path: Path, run_dir: Path, side: str) -> Path:
        """Write the mirror image of a slab beside the run and return its path."""

        record = self.normalizer.align_ab_to_xy(self.converter.read(str(path)))
        analysis = stacking_mod.analyse_stacking(record)
        if not analysis.close_packed:
            raise ValueError(f"the {side} slab is not close packed: {analysis.reason}")
        mirrored = stacking_mod.mirror_structure(record)
        mirrored.comment = restage_comment(record.comment, "interface build", "stacking reversed")
        destination = Path(run_dir) / f"{side}_{Path(path).stem}_mirrored.vasp"
        self.vasp_io.write(mirrored, str(destination), positions_are_cartesian=False, wrap_positions=True)
        return destination

    def _build_from_match(
        self,
        *,
        match_json: str,
        match_index: int,
        gap: float,
        vacuum: float | None,
        output_path: str | None,
        bottom_stacking: str = "keep",
        top_stacking: str = "keep",
    ) -> CommandResult:
        """Build the commensurate interface recorded by one match entry."""

        backend = self.choose_backend(feature="interface.build")
        document = lattice_match.read_matches(match_json)
        entry = lattice_match.select_match(document, int(match_index))
        candidate = lattice_match.candidate_for_match(entry)
        bottom_slab = Path(entry["bottom_slab"]).resolve()
        top_slab = Path(entry["top_slab"]).resolve()
        run_id, run_dir = self.create_run_dir("build", f"{bottom_slab.stem}_{top_slab.stem}_match")
        choices = {}
        for side, choice in (("bottom", bottom_stacking), ("top", top_stacking)):
            resolved = stacking_mod.normalise_stacking_choice(choice)
            if resolved in {"abc", "cba"}:
                raise ValueError(
                    "abc and cba compare two slabs on one lattice; the slabs of a matched "
                    "supercell sit on different cells at a twist angle, so reverse a stacking "
                    "sequence explicitly with mirror"
                )
            choices[side] = resolved
        if choices["bottom"] == "mirror":
            bottom_slab = self._mirrored_slab_copy(bottom_slab, run_dir, "bottom")
        if choices["top"] == "mirror":
            top_slab = self._mirrored_slab_copy(top_slab, run_dir, "top")
        resolved_vacuum = float(document["search"]["vacuum"]) if vacuum is None else float(vacuum)
        lattice, positions_direct, counts, species, flags, contacts = (
            moire_generator.build_supercell_with_report(
                str(top_slab),
                str(bottom_slab),
                candidate,
                interlayer_distance=float(gap),
                vacuum=resolved_vacuum,
            )
        )
        total_atoms = int(sum(counts))
        if total_atoms != int(entry["total_atoms"]):
            raise ValueError(
                f"match {entry['index']} records {entry['total_atoms']} atoms but its "
                f"supercell matrices hold {total_atoms}"
            )
        if output_path is None:
            output_suffix = run_output_suffix(run_id)
            bottom_miller = "".join(str(value) for value in entry["bottom_miller"])
            top_miller = "".join(str(value) for value in entry["top_miller"])
            angle_token = safe_token(f"{float(entry['angle_deg']):.4f}")
            gap_token = safe_token(f"{float(gap):.2f}")
            destination = self.output_root / (
                f"interface_match{int(entry['index']):03d}_{bottom_miller}on{top_miller}_"
                f"ang{angle_token}_atoms{total_atoms}_gap{gap_token}_{output_suffix}.vasp"
            )
        else:
            destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        moire_generator.write_supercell_poscar(
            str(destination),
            lattice,
            positions_direct,
            counts,
            species,
            flags,
            comment=stage_comment(
                "interface match",
                f"{bottom_slab.stem} below, {top_slab.stem} above",
                f"twist {float(entry['angle_deg']):.4f} deg",
                f"gap {float(gap):.2f} A",
            ),
        )
        summary = {
            "match_index": int(entry["index"]),
            "angle_deg": float(entry["angle_deg"]),
            "relative_principal_strains": [float(value) for value in entry["principal_strains"]],
            "bottom_strain": float(entry["bottom_strain"]),
            "top_strain": float(entry["top_strain"]),
            "bottom_matrix": [[int(value) for value in row] for row in entry["bottom_matrix"]],
            "top_matrix": [[int(value) for value in row] for row in entry["top_matrix"]],
            "vacuum": float(resolved_vacuum),
            "gap": float(gap),
            "c_length": float(np.linalg.norm(np.asarray(lattice, dtype=float)[2])),
            "total_atoms": total_atoms,
            "bottom_mirrored": choices["bottom"] == "mirror",
            "top_mirrored": choices["top"] == "mirror",
            **_contact_summary(contacts),
        }
        manifest_path = self.write_manifest(
            stage="build",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "match_json": str(Path(match_json).resolve()),
                "match_index": int(entry["index"]),
                "bottom_slab": str(bottom_slab),
                "top_slab": str(top_slab),
                "results_json": str(entry["results_json"]),
            },
            parameters={
                "gap": float(gap),
                "vacuum": float(resolved_vacuum),
                "bottom_stacking": choices["bottom"],
                "top_stacking": choices["top"],
            },
            artifacts={"interface_poscar": destination},
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"interface_poscar": destination},
            summary=summary,
        )

    def registries(
        self,
        *,
        bottom_input: str,
        top_input: str,
        bottom_kind: str = "auto",
        top_kind: str = "auto",
        bottom_miller: str | Sequence[int] | None = None,
        top_miller: str | Sequence[int] | None = None,
        bottom_layers: int = 4,
        top_layers: int = 4,
        bottom_vacuum: float = 15.0,
        top_vacuum: float = 15.0,
        include_equivalent: bool = False,
        output_path: str | None = None,
    ) -> CommandResult:
        """List the distinct stacking sequences and contacts of two slabs.

        The bottom slab fixes the ``A``/``B``/``C`` labels and the top slab is
        read in the same gauge, so its sequence says whether it stacks the same
        way (``ABCABC``) or the reversed way (``ACBACB``).  Crossing the two
        senses with the three contacts gives twelve labelled combinations, of
        which six are distinct; the report lists exactly those, and with
        ``include_equivalent`` also the ones that were removed, each marked with
        the option it is congruent to.  When the two slabs are the same slab,
        turning the interface over identifies two of the six as well.
        """

        backend = self.choose_backend(feature="interface.registries")
        run_id, run_dir = self.create_run_dir(
            "registries", f"{Path(bottom_input).stem}_{Path(top_input).stem}"
        )
        bottom_path, bottom_meta = self._resolve_surface_input(
            path_or_manifest=bottom_input,
            kind=bottom_kind,
            miller=bottom_miller,
            layers=bottom_layers,
            vacuum=bottom_vacuum,
        )
        top_path, top_meta = self._resolve_surface_input(
            path_or_manifest=top_input,
            kind=top_kind,
            miller=top_miller,
            layers=top_layers,
            vacuum=top_vacuum,
        )
        bottom = self.normalizer.align_ab_to_xy(self.converter.read(bottom_path))
        top = strained_copy(
            self.normalizer.align_ab_to_xy(self.converter.read(top_path)),
            np.asarray(bottom.lattice, dtype=float),
        )
        bottom_analysis = stacking_mod.analyse_stacking(bottom)
        if not bottom_analysis.close_packed:
            raise ValueError(f"the bottom slab is not close packed: {bottom_analysis.reason}")
        top_analysis = stacking_mod.analyse_stacking(
            top, hollow_cartesian=bottom_analysis.hollow_cartesian
        )
        if not top_analysis.close_packed:
            raise ValueError(f"the top slab is not close packed: {top_analysis.reason}")
        interchangeable = registry_mod.slabs_are_interchangeable(
            bottom, top, bottom_analysis, top_analysis
        )
        options = registry_mod.enumerate_registry_options(
            bottom_analysis,
            top_analysis,
            include_equivalent=bool(include_equivalent),
            slabs_interchangeable=interchangeable,
        )
        document = {
            "bottom_slab": str(Path(bottom_path).resolve()),
            "top_slab": str(Path(top_path).resolve()),
            "bottom_stacking": bottom_analysis.as_dict(),
            "top_stacking": top_analysis.as_dict(),
            "labelled_combinations": (
                (2 if bottom_analysis.reversible else 1)
                * (2 if top_analysis.reversible else 1)
                * 3
            ),
            "listed_options": len(options),
            "slabs_interchangeable": bool(interchangeable),
            "distinct_options": sum(1 for option in options if option.equivalent_to is None),
            "options": [option.as_dict() for option in options],
        }
        results_path = (
            Path(output_path).resolve() if output_path is not None else run_dir / "registries.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(document, indent=2) + "\n")
        summary = {
            "bottom_sequence": bottom_analysis.sequence,
            "top_sequence": top_analysis.sequence,
            "top_sense": top_analysis.sense_label,
            "distinct_options": int(document["distinct_options"]),
            "labelled_combinations": int(document["labelled_combinations"]),
            "listed_options": len(options),
            "slabs_interchangeable": bool(interchangeable),
        }
        manifest_path = self.write_manifest(
            stage="registries",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "bottom_input": str(Path(bottom_input).resolve()),
                "top_input": str(Path(top_input).resolve()),
                "bottom_meta": bottom_meta,
                "top_meta": top_meta,
            },
            parameters={
                "bottom_kind": str(bottom_kind),
                "top_kind": str(top_kind),
                "include_equivalent": bool(include_equivalent),
            },
            artifacts={"registries_json": results_path},
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"registries_json": results_path},
            summary=summary,
            payload={"registry_table": registry_mod.format_registry_table(options)},
        )

    @staticmethod
    def _prune_unused_pair_searches(run_dir: Path, entries: Sequence[dict[str, object]]) -> int:
        """Delete the per-pair search directories no reported match points at.

        A discarded match is unreachable from the written document, so keeping
        its search file only fills the run directory with results nothing refers
        to.  Only directories this stage created under ``run_dir`` are removed.
        """

        run_root = Path(run_dir).resolve()
        referenced = {
            Path(str(entry["results_json"])).resolve().parent for entry in entries
        }
        kept = 0
        for directory in sorted(run_root.glob("pair_*")):
            if not directory.is_dir():
                continue
            if directory.resolve() in referenced:
                kept += 1
            else:
                shutil.rmtree(directory, ignore_errors=True)
        return kept

    def match(
        self,
        *,
        bottom_bulk: str,
        top_bulk: str,
        bottom_millers: Sequence[str | Sequence[int]] | None = None,
        top_millers: Sequence[str | Sequence[int]] | None = None,
        bottom_layers_list: Sequence[int] | None = None,
        top_layers_list: Sequence[int] | None = None,
        vacuum: float = 15.0,
        max_strain: float = 0.05,
        max_length: float = 20.0,
        strain_mode: str = "shared",
        min_length: float | None = None,
        max_atoms: int | None = None,
        max_matches: int | None = DEFAULT_MATCH_LIMIT,
        preview_limit: int = 10,
        output_path: str | None = None,
    ) -> CommandResult:
        """Search bulk-derived surface pairs for commensurate interface cells.

        Every surface combination is matched with the Gram-form supercell engine,
        so a match is an integer supercell of each slab plus a twist angle, not
        merely a pair of 1x1 cells that happen to be similar.  ``max_strain``
        bounds the principal logarithmic strain applied to one slab; in the
        default ``shared`` mode the engine splits the relative strain optimally
        between the two, and in ``film`` mode the bottom slab is left rigid.

        A scan over several Miller indices and thicknesses can produce tens of
        thousands of admissible cells, which is a data dump rather than an
        answer, so only the ``max_matches`` best ones are kept; pass ``0`` or
        ``None`` to keep every match.  The per-pair search files behind the
        discarded matches are removed with them, and the ones the reported
        matches point at are kept exactly as the search wrote them.
        """

        backend = self.choose_backend(feature="interface.match")
        run_id, run_dir = self.create_run_dir("match", f"{Path(bottom_bulk).stem}_{Path(top_bulk).stem}")
        bottom_miller_values = list(bottom_millers or ["1,0,0", "1,1,0", "1,1,1"])
        top_miller_values = list(top_millers or ["1,0,0", "1,1,0", "1,1,1"])
        bottom_layer_values = [int(value) for value in (bottom_layers_list or [4])]
        top_layer_values = [int(value) for value in (top_layers_list or [4])]
        request = lattice_match.MatchRequest(
            max_length=float(max_length),
            max_strain=float(max_strain),
            strain_mode=str(strain_mode),
            min_length=None if min_length is None else float(min_length),
            max_atoms=None if max_atoms is None else int(max_atoms),
        )

        from ..surface.surface import Surface

        surface_tool = Surface(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        )
        slab_cache: dict[tuple[str, str, int], str] = {}

        def slab_for(side: str, bulk: str, miller: str | Sequence[int], layers: int) -> str:
            """Return the normalised slab for one surface, building it once."""

            key = (side, str(miller), int(layers))
            cached = slab_cache.get(key)
            if cached is None:
                surface = surface_tool.surface(
                    bulk_poscar=bulk,
                    miller=miller,
                    layers=int(layers),
                    vacuum=float(vacuum),
                    output_path=str(
                        run_dir / f"{side}_{str(miller).replace(',', '')}_{int(layers)}.vasp"
                    ),
                )
                cached = self._normalized_slab(surface.artifacts["slab_poscar"])
                slab_cache[key] = cached
            return cached

        entries: list[dict[str, object]] = []
        pair_count = 0
        for bottom_miller in bottom_miller_values:
            for bottom_layers in bottom_layer_values:
                bottom_slab = slab_for("bottom", bottom_bulk, bottom_miller, bottom_layers)
                for top_miller in top_miller_values:
                    for top_layers in top_layer_values:
                        top_slab = slab_for("top", top_bulk, top_miller, top_layers)
                        pair_count += 1
                        pair = lattice_match.SlabPair(
                            bottom_slab=Path(bottom_slab),
                            top_slab=Path(top_slab),
                            bottom_miller=parse_miller_notation(bottom_miller),
                            top_miller=parse_miller_notation(top_miller),
                            bottom_layers=int(bottom_layers),
                            top_layers=int(top_layers),
                        )
                        results_path, candidates = lattice_match.search_slab_pair(
                            pair,
                            request,
                            results_root=run_dir / f"pair_{pair_count:03d}",
                        )
                        entries.extend(
                            lattice_match.match_entries(pair, candidates, results_path=results_path)
                        )

        ordered = lattice_match.sort_matches(entries)
        limit = 0 if max_matches is None else max(int(max_matches), 0)
        if 0 < limit < len(ordered):
            ordered = ordered[:limit]
        pairs_kept = self._prune_unused_pair_searches(run_dir, ordered)
        document = lattice_match.build_match_document(
            bottom_bulk=bottom_bulk,
            top_bulk=top_bulk,
            request=request,
            entries=ordered,
            vacuum=float(vacuum),
            bottom_millers=[str(value) for value in bottom_miller_values],
            top_millers=[str(value) for value in top_miller_values],
            bottom_layers_list=bottom_layer_values,
            top_layers_list=top_layer_values,
        )
        results_path = Path(output_path).resolve() if output_path is not None else (run_dir / "matches.json")
        lattice_match.write_matches(results_path, document)
        summary = {
            "match_count": len(ordered),
            "surface_pairs_searched": pair_count,
            "surface_pairs_kept": pairs_kept,
            "best_strain": float(ordered[0]["strain"]) if ordered else None,
            "best_total_atoms": int(ordered[0]["total_atoms"]) if ordered else None,
        }
        manifest_path = self.write_manifest(
            stage="match",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"bottom_bulk": str(Path(bottom_bulk).resolve()), "top_bulk": str(Path(top_bulk).resolve())},
            parameters={
                "bottom_millers": [str(value) for value in bottom_miller_values],
                "top_millers": [str(value) for value in top_miller_values],
                "bottom_layers_list": bottom_layer_values,
                "top_layers_list": top_layer_values,
                "vacuum": float(vacuum),
                "max_strain": float(max_strain),
                "max_length": float(max_length),
                "strain_mode": str(strain_mode),
                "min_length": None if min_length is None else float(min_length),
                "max_atoms": None if max_atoms is None else int(max_atoms),
                "max_matches": None if max_matches is None else int(max_matches),
            },
            artifacts={"matches_json": results_path},
            summary=summary,
        )
        payload: dict[str, object] = {"best_match": ordered[0] if ordered else None}
        if int(preview_limit) > 0:
            payload["candidate_preview"] = lattice_match.format_matches_table(
                ordered, limit=int(preview_limit)
            )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"matches_json": results_path},
            summary=summary,
            payload=payload,
        )

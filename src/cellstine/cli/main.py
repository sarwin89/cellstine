"""Main grouped CLI for CELLSTINE."""

from __future__ import annotations

import sys
from pathlib import Path

from .. import __version__
from ..adsorbate.molecule import Molecule
from ..core.dependencies import DependencyManager
from ..defect.workflow import Defect
from ..interface.surface.surface import Surface
from ..interface.workflow.interface import Interface
from ..moire.moire import Moire
from ..moire.supermoire import Supermoire
from ..symmetry.symmetry import Symmetry
from ..visualize.visualize import Visualize
from .parsers import build_parser


def _print_versions() -> None:
    manager = DependencyManager()
    parts = [f"cellstine {__version__}"]
    for name, version in sorted(manager.versions().items()):
        parts.append(f"{name} {version}")
    print(" | ".join(parts))


def _print_result(result) -> None:
    print(f"Manifest: {result.manifest_path}")
    for key, value in result.artifacts.items():
        print(f"{key}: {value}")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    timings = getattr(result, "payload", {}).get("timings_s") if hasattr(result, "payload") else None
    if timings:
        print()
        print("Timing:")
        timing_order = [
            "read_structures_s",
            "angle_shortlist_s",
            "supercell_search_s",
            "write_results_s",
            "manifest_write_s",
            "workflow_total_s",
        ]
        for key in timing_order:
            if key in timings:
                label = key.removesuffix("_s").replace("_", " ")
                print(f"  {label}: {float(timings[key]):.3f}s")
    angle_search = getattr(result, "payload", {}).get("angle_search") if hasattr(result, "payload") else None
    if angle_search:
        print()
        print("Angle search:")
        print(f"  shortlisted angles: {angle_search.get('shortlisted_angle_count')}")
        print(f"  searched angles: {angle_search.get('searched_angle_count')}")
        if angle_search.get("angle_values_thinned"):
            print(
                "  thinning: "
                f"{angle_search.get('angle_values_before_thinning')} -> "
                f"{angle_search.get('searched_angle_count')} "
                f"(cap {angle_search.get('max_search_angles')})"
            )
    candidate_preview = getattr(result, "payload", {}).get("candidate_preview") if hasattr(result, "payload") else None
    if candidate_preview:
        print()
        print("Candidate preview:")
        print(candidate_preview)
    site_preview = getattr(result, "payload", {}).get("site_preview") if hasattr(result, "payload") else None
    if site_preview:
        print()
        print("Site preview:")
        print(site_preview)
    defect_preview = getattr(result, "payload", {}).get("defect_preview") if hasattr(result, "payload") else None
    if defect_preview:
        print()
        print("Defect preview:")
        print(defect_preview)
    symmetry_preview = getattr(result, "payload", {}).get("symmetry_preview") if hasattr(result, "payload") else None
    if symmetry_preview:
        print()
        print("Symmetry preview:")
        print(symmetry_preview)


def execute_namespace(args):
    if getattr(args, "version", False):
        return "version"
    if not getattr(args, "group", None):
        raise ValueError("interactive mode should be handled before execution")

    if args.group == "moire":
        if args.stage == "find":
            tool = Moire()
            result = tool.find(
                top_poscar=args.top_poscar,
                bottom_poscar=args.bottom_poscar,
                max_length=args.max_length,
                top_strain=args.top_strain,
                bottom_strain=args.bottom_strain,
                min_length=args.min_length,
                max_atoms=args.max_atoms,
                max_aspect_ratio=args.max_cell_aspect_ratio,
                min_cell_angle_deg=args.min_cell_angle,
                max_cell_angle_deg=args.max_cell_angle,
                symmetric=args.symmetric,
                preview_limit=args.preview_limit,
                progress=args.progress,
            )
            return result
        if args.stage == "make":
            return Moire().make(results_file=args.results_file, indexes=args.indexes, interlayer_distance=args.interlayer_distance, workers=args.workers, output_dir=args.output_dir)
        if args.stage == "translate":
            return Moire().translate(poscar_path=args.poscar_path, shift_cartesian=args.shift_cart, shift_direct=args.shift_direct)
        if args.stage == "translaten":
            return Supermoire().translaten(poscar_path=args.poscar_path, shift_cartesian=args.shift_cart, shift_direct=args.shift_direct)
        if args.stage == "visualize":
            return Moire().visualize(
                results_file=args.results_file,
                indices=args.indices,
                interlayer=args.interlayer,
                output_path=args.output,
                plotly=args.plotly,
                show=args.show,
            )

    if args.group == "adsorbate":
        tool = Molecule()
        if args.stage == "place":
            return tool.place(
                substrate_poscar=args.substrate_poscar,
                molecule_poscar=args.molecule_poscar,
                substrate_kind=args.substrate_kind,
                miller=args.miller,
                layers=args.layers,
                vacuum=args.vacuum,
                substrate_repeat_a=args.substrate_repeat_a,
                substrate_repeat_b=args.substrate_repeat_b,
                substrate_supercell_matrix=args.substrate_supercell_matrix,
                auto_repeat_substrate=args.auto_repeat_substrate,
                fit_padding=args.fit_padding,
                site_type=args.site_type,
                site_index=args.site_index,
                height=args.height,
                rotation_deg=args.rotate,
                tilt_deg=args.tilt,
                roll_deg=args.roll,
            )
        if args.stage == "move":
            return tool.move(
                poscar_path=args.poscar_path,
                target_cartesian=args.target_cart,
                target_direct=args.target_direct,
                rotation_deg=args.rotate,
                tilt_deg=args.tilt,
                roll_deg=args.roll,
            )
        if args.stage == "assemble":
            return tool.assemble(
                substrate_poscar=args.substrate_poscar,
                a_length=args.a_length,
                b_length=args.b_length,
                angle_deg=args.angle,
                nindex=args.nindex,
                max_strain=args.max_strain,
                preview_limit=args.preview_limit,
            )
        if args.stage == "visualize":
            return Visualize().structure(structure_path=args.structure_path, output_path=args.output, plotly=args.plotly, show=args.show)

    if args.group == "interface":
        surface_tool = Surface()
        interface_tool = Interface()
        if args.stage == "surface":
            return surface_tool.surface(
                bulk_poscar=args.bulk_poscar,
                miller=args.miller,
                layers=args.layers,
                vacuum=args.vacuum,
                repeat_a=args.repeat_a,
                repeat_b=args.repeat_b,
                supercell_matrix=args.supercell_matrix,
                analyse_sites=args.analyse_sites,
            )
        if args.stage == "sites":
            return surface_tool.sites(slab_poscar=args.slab_poscar, surface_side=args.surface_side)
        if args.stage == "build":
            return interface_tool.build(
                bottom_input=args.bottom_input,
                top_input=args.top_input,
                bottom_kind=args.bottom_kind,
                top_kind=args.top_kind,
                bottom_miller=args.bottom_miller,
                top_miller=args.top_miller,
                bottom_layers=args.bottom_layers,
                top_layers=args.top_layers,
                bottom_vacuum=args.bottom_vacuum,
                top_vacuum=args.top_vacuum,
                gap=args.gap,
            )
        if args.stage == "match":
            return interface_tool.match(
                bottom_bulk=args.bottom_bulk,
                top_bulk=args.top_bulk,
                bottom_millers=args.bottom_millers,
                top_millers=args.top_millers,
                bottom_layers_list=args.bottom_layers_list,
                top_layers_list=args.top_layers_list,
                vacuum=args.vacuum,
                max_strain=args.max_strain,
            )
        if args.stage == "visualize":
            return Visualize().structure(structure_path=args.structure_path, output_path=args.output, plotly=args.plotly, show=args.show)

    if args.group == "defect":
        tool = Defect()
        if args.stage == "analyse":
            return tool.analyse(
                structure_path=args.structure,
                structure_kind=args.structure_kind,
                backend=args.backend,
                surface_side=args.surface_side,
                layer_tolerance=args.layer_tolerance,
                symprec=args.symprec,
                divacancy_distance=args.divacancy_distance,
            )
        if args.stage == "generate":
            return tool.generate(
                structure_path_or_manifest=args.analysis_or_structure,
                defect_type=args.defect_type,
                site_ids=args.site_ids,
                species=args.species,
                substitution_species=args.substitution_species,
                original_species=args.original_species,
                generate=args.generate,
                output_dir=args.output_dir,
                structure_kind=args.structure_kind,
                backend=args.backend,
                surface_side=args.surface_side,
                layer_tolerance=args.layer_tolerance,
                symprec=args.symprec,
                height=args.height,
                divacancy_distance=args.divacancy_distance,
            )
        if args.stage == "preview":
            return tool.preview(
                analysis_or_structure=args.analysis_or_structure,
                limit=args.limit,
                structure_kind=args.structure_kind,
                backend=args.backend,
                surface_side=args.surface_side,
                layer_tolerance=args.layer_tolerance,
                symprec=args.symprec,
                divacancy_distance=args.divacancy_distance,
            )

    if args.group == "symmetry":
        tool = Symmetry()
        if args.stage == "analyse":
            return tool.analyse(
                structure_path=args.structure,
                backend=args.backend,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
            )
        if args.stage == "reduce":
            return tool.reduce(
                structure_path=args.structure,
                cell=args.cell,
                backend=args.backend,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                output_path=args.output,
            )
        if args.stage == "lattice-reduce":
            return tool.lattice_reduce(
                structure_path=args.structure,
                reduction=args.reduction,
                backend=args.backend,
                symprec=args.symprec,
                output_path=args.output,
            )

    raise SystemExit("No workflow stage was selected. Use --help for usage.")


def dispatch_namespace(args) -> int:
    if getattr(args, "version", False):
        _print_versions()
        return 0
    if not getattr(args, "group", None):
        from .interactive.runner import run_interactive

        return run_interactive()
    if not getattr(args, "stage", None):
        from .interactive.runner import run_interactive

        return run_interactive(group=str(args.group))

    result = execute_namespace(args)
    if result == "version":
        _print_versions()
        return 0
    _print_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return dispatch_namespace(arguments)
    except Exception as exc:  # pragma: no cover - friendly CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

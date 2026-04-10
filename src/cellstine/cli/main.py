"""Main grouped CLI for CELLSTINE."""

from __future__ import annotations

import sys
from pathlib import Path

from .. import __version__
from ..adsorbate.molecule import Molecule
from ..core.dependencies import DependencyManager
from ..core.models import PrestrainConfig
from ..interface.interface import Interface
from ..interface.surface import Surface
from ..moire.moire import Moire
from ..moire.supermoire import Supermoire
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


def dispatch_namespace(args) -> int:
    if getattr(args, "version", False):
        _print_versions()
        return 0
    if not getattr(args, "group", None):
        from .interactive import run_interactive

        return run_interactive()

    if args.group == "moire":
        if args.stage == "find":
            tool = Moire()
            result = tool.find(
                top_poscar=args.top_poscar,
                bottom_poscar=args.bottom_poscar,
                nindex=args.nindex,
                min_angle=args.min_angle,
                max_angle=args.max_angle,
                angle_step=args.angle_step,
                explicit_angles=args.angles,
                vector_tolerance=args.vector_tolerance,
                vector_strain_tolerance=args.vector_strain_tolerance,
                candidate_tolerance=args.candidate_tolerance,
                strain_tolerance=args.strain_tolerance,
                matrix_values=args.matrix_values,
                matrix_layer=args.matrix_layer,
                matrix_match_mode=args.matrix_match_mode,
                workers=args.workers,
                top_c_repeat=args.top_c_repeat,
                bottom_c_repeat=args.bottom_c_repeat,
                prestrain_top=PrestrainConfig(args.prestrain_top_mode, args.prestrain_top_value, args.prestrain_top_axis),
                prestrain_bottom=PrestrainConfig(args.prestrain_bottom_mode, args.prestrain_bottom_value, args.prestrain_bottom_axis),
            )
            _print_result(result)
            return 0
        if args.stage == "findn":
            modes = args.prestrain_modes or ["none"] * (len(args.upper_poscars) + 1)
            values = args.prestrain_values or [0.0] * (len(args.upper_poscars) + 1)
            axes = args.prestrain_axes or [None] * (len(args.upper_poscars) + 1)
            if len(modes) < len(args.upper_poscars) + 1:
                modes.extend(["none"] * (len(args.upper_poscars) + 1 - len(modes)))
            if len(values) < len(args.upper_poscars) + 1:
                values.extend([0.0] * (len(args.upper_poscars) + 1 - len(values)))
            if len(axes) < len(args.upper_poscars) + 1:
                axes.extend([None] * (len(args.upper_poscars) + 1 - len(axes)))
            prestrains = [PrestrainConfig(mode, value, axis) for mode, value, axis in zip(modes, values, axes)]
            result = Supermoire().findn(
                bottom_poscar=args.bottom_poscar,
                upper_poscars=args.upper_poscars,
                nindex=args.nindex,
                match_mode=args.match_mode,
                min_angles=args.min_angles,
                max_angles=args.max_angles,
                explicit_angles_by_layer=args.angles_by_layer,
                vector_tolerance=args.vector_tolerance,
                vector_strain_tolerance=args.vector_strain_tolerance,
                candidate_tolerance=args.candidate_tolerance,
                max_atoms=args.max_atoms,
                workers=args.workers,
                bottom_c_repeat=args.bottom_c_repeat,
                upper_c_repeats=None if args.upper_c_repeats is None else [int(value) for value in args.upper_c_repeats],
                prestrains=prestrains,
            )
            _print_result(result)
            return 0
        if args.stage == "make":
            result = Moire().make(results_file=args.results_file, indexes=args.indexes, interlayer_distance=args.interlayer_distance, workers=args.workers, output_dir=args.output_dir)
            _print_result(result)
            return 0
        if args.stage == "maken":
            result = Supermoire().maken(results_file=args.results_file, indexes=args.indexes, interlayers=args.interlayers, output_dir=args.output_dir)
            _print_result(result)
            return 0
        if args.stage == "translate":
            result = Moire().translate(poscar_path=args.poscar_path, shift_cartesian=args.shift_cart, shift_direct=args.shift_direct)
            _print_result(result)
            return 0
        if args.stage == "translaten":
            result = Supermoire().translaten(poscar_path=args.poscar_path, shift_cartesian=args.shift_cart, shift_direct=args.shift_direct)
            _print_result(result)
            return 0
        if args.stage == "visualize":
            result = Moire().visualize(results_file=args.results_file, indices=args.indices, interlayer=args.interlayer)
            _print_result(result)
            return 0

    if args.group == "adsorbate":
        tool = Molecule()
        if args.stage == "place":
            result = tool.place(
                substrate_poscar=args.substrate_poscar,
                molecule_poscar=args.molecule_poscar,
                substrate_kind=args.substrate_kind,
                miller=args.miller,
                layers=args.layers,
                vacuum=args.vacuum,
                site_type=args.site_type,
                site_index=args.site_index,
                height=args.height,
                rotation_deg=args.rotate,
            )
            _print_result(result)
            return 0
        if args.stage == "move":
            result = tool.move(poscar_path=args.poscar_path, target_cartesian=args.target_cart, target_direct=args.target_direct, rotation_deg=args.rotate)
            _print_result(result)
            return 0
        if args.stage == "assemble":
            result = tool.assemble(
                substrate_poscar=args.substrate_poscar,
                a_length=args.a_length,
                b_length=args.b_length,
                angle_deg=args.angle,
                nindex=args.nindex,
                max_strain=args.max_strain,
            )
            _print_result(result)
            return 0
        if args.stage == "visualize":
            result = Visualize().structure(structure_path=args.structure_path)
            _print_result(result)
            return 0

    if args.group == "interface":
        surface_tool = Surface()
        interface_tool = Interface()
        if args.stage == "surface":
            result = surface_tool.surface(
                bulk_poscar=args.bulk_poscar,
                miller=args.miller,
                layers=args.layers,
                vacuum=args.vacuum,
                repeat_a=args.repeat_a,
                repeat_b=args.repeat_b,
                supercell_matrix=args.supercell_matrix,
                analyse_sites=args.analyse_sites,
            )
            _print_result(result)
            return 0
        if args.stage == "sites":
            result = surface_tool.sites(slab_poscar=args.slab_poscar, surface_side=args.surface_side)
            _print_result(result)
            return 0
        if args.stage == "build":
            result = interface_tool.build(
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
            _print_result(result)
            return 0
        if args.stage == "match":
            result = interface_tool.match(
                bottom_bulk=args.bottom_bulk,
                top_bulk=args.top_bulk,
                bottom_millers=args.bottom_millers,
                top_millers=args.top_millers,
                bottom_layers_list=args.bottom_layers_list,
                top_layers_list=args.top_layers_list,
                vacuum=args.vacuum,
                max_strain=args.max_strain,
            )
            _print_result(result)
            return 0
        if args.stage == "visualize":
            result = Visualize().structure(structure_path=args.structure_path)
            _print_result(result)
            return 0

    raise SystemExit("No workflow stage was selected. Use --help for usage.")


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

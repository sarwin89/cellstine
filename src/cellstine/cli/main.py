"""Main grouped CLI for CELLSTINE."""

from __future__ import annotations

import sys

from .. import __version__
from .spec import legacy_command_message, parse_twist_window, resolve_moire_strains

# The workflow packages are loaded when a stage asks for one, not at import.
# Every group pulls in NumPy and its own stack, and a run only ever uses one of
# them, so importing all eight cost every invocation --- including ``--help``
# --- about a quarter of a second of work it never used.
_WORKFLOWS = {
    "Adsorbate": ("..adsorbate.adsorbate", "Adsorbate"),
    "Defect": ("..defect.workflow", "Defect"),
    "Interface": ("..interface.workflow.interface", "Interface"),
    "Moire": ("..moire.moire", "Moire"),
    "Molecule": ("..adsorbate.molecule", "Molecule"),
    "Supermoire": ("..moire.supermoire", "Supermoire"),
    "Surface": ("..interface.surface.surface", "Surface"),
    "Symmetry": ("..symmetry.symmetry", "Symmetry"),
    "Visualize": ("..visualize.visualize", "Visualize"),
}


def _workflow(name: str):
    """Return a workflow class by name, importing its package on first use."""

    from importlib import import_module

    module_name, attribute = _WORKFLOWS[name]
    return getattr(import_module(module_name, __package__), attribute)


def __getattr__(name: str):
    """Expose the workflow classes as module attributes, still lazily."""

    if name in _WORKFLOWS:
        value = _workflow(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _print_versions() -> None:
    from ..core.dependencies import DependencyManager

    manager = DependencyManager()
    parts = [f"cellstine {__version__}"]
    for name, version in sorted(manager.versions().items()):
        parts.append(f"{name} {version}")
    print(" | ".join(parts))


def _print_result(result) -> None:
    """Print a finished workflow result in the shared CELLSTINE report shape."""

    from ..core.report import format_result

    text = format_result(result)
    if text:
        print(text)


def execute_namespace(args):
    if getattr(args, "version", False):
        return "version"
    if not getattr(args, "group", None):
        raise ValueError("interactive mode should be handled before execution")

    if args.group == "moire":
        Moire = _workflow("Moire")
        Supermoire = _workflow("Supermoire")
        if args.stage == "search":
            top_strain, bottom_strain = resolve_moire_strains(
                rigid=bool(args.rigid),
                strain=args.strain,
                top_strain=args.top_strain,
                bottom_strain=args.bottom_strain,
            )
            min_twist_angle, max_twist_angle = parse_twist_window(args.twist)
            tool = Moire()
            result = tool.find(
                top_poscar=args.top_poscar,
                bottom_poscar=args.bottom_poscar,
                max_length=args.max_length,
                top_strain=top_strain,
                bottom_strain=bottom_strain,
                min_length=args.min_length,
                max_atoms=args.max_atoms,
                max_aspect_ratio=args.max_cell_aspect_ratio,
                min_cell_angle_deg=args.min_cell_angle,
                max_cell_angle_deg=args.max_cell_angle,
                min_twist_angle_deg=min_twist_angle,
                max_twist_angle_deg=max_twist_angle,
                symmetric=args.symmetric,
                reduce_layers=not args.keep_layer_cells,
                symmetry_tolerance=args.symmetry_tolerance,
                preview_limit=args.preview_limit,
                progress=args.progress,
            )
            return result
        if args.stage == "build":
            return Moire().make(
                results_file=args.results_file,
                indexes=args.indexes,
                interlayer_distance=args.interlayer_distance,
                vacuum=args.vacuum,
                workers=args.workers,
                output_dir=args.output_dir,
            )
        if args.stage == "stack-search":
            return Supermoire().findn(
                base_poscar=args.base_poscar,
                upper_poscars=args.upper_poscars,
                max_length=args.max_length,
                layer_strains=args.layer_strains if args.layer_strains is not None else args.layer_strain,
                min_length=args.min_length,
                max_atoms=args.max_atoms,
                max_pair_atoms=args.max_pair_atoms,
                max_aspect_ratio=args.max_cell_aspect_ratio,
                min_cell_angle_deg=args.min_cell_angle,
                max_cell_angle_deg=args.max_cell_angle,
                per_layer_limit=args.per_layer_limit,
                max_candidates=args.max_candidates,
                reduce_layers=not args.keep_layer_cells,
                preview_limit=args.preview_limit,
            )
        if args.stage == "stack-build":
            return Supermoire().maken(
                results_file=args.results_file,
                indexes=args.indexes,
                interlayers=args.interlayers if args.interlayers is not None else args.interlayer_distance,
                vacuum=args.vacuum,
                output_dir=args.output_dir,
            )
        if args.stage == "shift":
            return Moire().translate(poscar_path=args.poscar_path, shift_cartesian=args.shift_cart, shift_direct=args.shift_direct)
        if args.stage == "view":
            return Moire().visualize(
                results_file=args.results_file,
                indices=args.indices,
                interlayer=args.interlayer,
                output_path=args.output,
                plotly=args.plotly,
                show=args.show,
            )

    if args.group == "surface":
        surface_tool = _workflow("Surface")()
        if args.stage == "build":
            return surface_tool.surface(
                bulk_poscar=args.bulk_poscar,
                miller=args.miller,
                layers=args.layers,
                vacuum=args.vacuum,
                repeat_a=args.repeat_a,
                repeat_b=args.repeat_b,
                min_length_a=args.min_length_a,
                min_length_b=args.min_length_b,
                supercell_matrix=args.supercell_matrix,
                output_path=args.output_path,
                analyse_sites=args.analyse_sites,
                sites_output_path=args.sites_output_path,
                site_surface_side=args.site_surface_side,
            )
        if args.stage == "sites":
            return surface_tool.sites(
                slab_poscar=args.slab_poscar,
                surface_side=args.surface_side,
                output_path=args.output_path,
            )

    if args.group == "adsorbate":
        tool = _workflow("Molecule")()
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
                preserve_vacuum=not args.keep_cell_height,
            )
        if args.stage == "move":
            return tool.move(
                poscar_path=args.poscar_path,
                target_cartesian=args.target_cart,
                target_direct=args.target_direct,
                rotation_deg=args.rotate,
                tilt_deg=args.tilt,
                roll_deg=args.roll,
                preserve_vacuum=not args.keep_cell_height,
            )
        if args.stage == "path":
            return tool.path(
                start_structure=args.start_structure,
                end_structure=args.end_structure,
                images=args.images,
                match=args.match,
                output_dir=args.output_dir,
            )
        if args.stage == "assemble":
            return tool.assemble(
                substrate_poscar=args.substrate_poscar,
                a_length=args.a_length,
                b_length=args.b_length,
                angle_deg=args.angle,
                max_length=args.max_length,
                top_strain=args.top_strain,
                bottom_strain=args.bottom_strain,
                preview_limit=args.preview_limit,
            )
        if args.stage == "kpath":
            spacing = args.spacing
            if spacing is None and args.divisions is None:
                spacing = 0.03
            return tool.kpath(
                structure_path=args.structure,
                spacing=spacing,
                divisions=args.divisions,
                path=args.path,
                use_standard=not args.derived_path,
                use_symmetry=not args.no_symmetry,
                time_reversal=not args.no_time_reversal,
                symprec=args.symprec,
                output_path=args.output,
            )
    if args.group == "interface":
        interface_tool = _workflow("Interface")()
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
                vacuum=args.vacuum,
                output_path=args.output_path,
                match_json=args.match_json,
                match_index=args.match_index,
                max_strain=args.max_strain,
                bottom_stacking=args.bottom_stacking,
                top_stacking=args.top_stacking,
                registry=args.registry,
                include_equivalent=args.include_equivalent,
            )
        if args.stage == "registries":
            return interface_tool.registries(
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
                include_equivalent=args.include_equivalent,
                output_path=args.output_path,
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
                max_length=args.max_length,
                strain_mode=args.strain_mode,
                min_length=args.min_length,
                max_atoms=args.max_atoms,
                max_matches=args.max_matches,
                preview_limit=args.preview_limit,
                output_path=args.output_path,
            )
        if args.stage == "kpath":
            spacing = args.spacing
            if spacing is None and args.divisions is None:
                spacing = 0.03
            return _workflow("Symmetry")().kpath(
                structure_path=args.structure,
                spacing=spacing,
                divisions=args.divisions,
                path=args.path,
                use_standard=not args.derived_path,
                use_symmetry=not args.no_symmetry,
                time_reversal=not args.no_time_reversal,
                symprec=args.symprec,
                output_path=args.output,
            )

    if args.group == "defect":
        tool = _workflow("Defect")()
        if args.stage == "analyse":
            return tool.analyse(
                structure_path=args.structure,
                structure_kind=args.structure_kind,
                backend=args.backend,
                surface_side=args.surface_side,
                layer_tolerance=args.layer_tolerance,
                symprec=args.symprec,
                divacancy_distance=args.divacancy_distance,
                view_direction=args.view_direction,
                interstitial_saddles=args.interstitial_saddles,
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
                preserve_vacuum=not args.keep_cell_height,
                supercell=args.supercell,
                supercell_matrix=args.supercell_matrix,
                min_image_distance=args.min_image_distance,
                cell_limit=args.cell_limit,
                view_direction=args.view_direction,
                layers=args.layers,
                interstitial_saddles=args.interstitial_saddles,
            )
        if args.stage == "supercell":
            return tool.supercell(
                structure_path=args.structure,
                min_image_distance=args.min_image_distance,
                max_cells=args.max_cells,
                structure_kind=args.structure_kind,
                layer_tolerance=args.layer_tolerance,
                cell_limit=args.cell_limit,
                table_limit=args.table_limit,
                output_path=args.output,
            )
        if args.stage == "path":
            return tool.path(
                start_structure=args.start_structure,
                end_structure=args.end_structure,
                images=args.images,
                match=args.match,
                output_dir=args.output_dir,
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
                view_direction=args.view_direction,
                interstitial_saddles=args.interstitial_saddles,
            )
        if args.stage == "kpath":
            spacing = args.spacing
            if spacing is None and args.divisions is None:
                spacing = 0.03
            return tool.kpath(
                structure_path=args.structure,
                spacing=spacing,
                divisions=args.divisions,
                path=args.path,
                use_standard=not args.derived_path,
                use_symmetry=not args.no_symmetry,
                time_reversal=not args.no_time_reversal,
                symprec=args.symprec,
                output_path=args.output,
            )
    if args.group == "symmetry":
        tool = _workflow("Symmetry")()
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
        if args.stage == "kpoints":
            explicit = None
            if args.list_points:
                explicit = True
            elif args.automatic:
                explicit = False
            return tool.kpoints(
                structure_path=args.structure,
                spacing=args.spacing,
                divisions=args.divisions,
                mode=args.mesh,
                shift=args.shift,
                surface=args.surface,
                use_symmetry=not args.no_symmetry,
                time_reversal=not args.no_time_reversal,
                explicit=explicit,
                symprec=args.symprec,
                output_path=args.output,
            )
        if args.stage == "kpath":
            spacing = args.spacing
            if spacing is None and args.divisions is None:
                spacing = 0.03
            return tool.kpath(
                structure_path=args.structure,
                spacing=spacing,
                divisions=args.divisions,
                path=args.path,
                use_standard=not args.derived_path,
                use_symmetry=not args.no_symmetry,
                time_reversal=not args.no_time_reversal,
                symprec=args.symprec,
                output_path=args.output,
            )
    if args.group == "view":
        return _workflow("Visualize")().structure(
            structure_path=args.structure_path,
            output_path=args.output,
            plotly=args.plotly,
            show=args.show,
            view_direction=args.view_direction,
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
    values = list(sys.argv[1:] if argv is None else argv)
    force_plain = "--plain" in values
    filtered = [value for value in values if value != "--plain"]
    for index, value in enumerate(filtered[:-1]):
        if value.startswith("-"):
            continue
        message = legacy_command_message(value, filtered[index + 1])
        if message is not None:
            print(f"Error: {message}", file=sys.stderr)
            return 2
        break
    if not force_plain:
        try:
            from .rich_app import run as run_rich

            return int(run_rich(filtered))
        except ImportError:
            pass
    from .plain import run as run_plain

    return int(run_plain(filtered))


if __name__ == "__main__":
    raise SystemExit(main())

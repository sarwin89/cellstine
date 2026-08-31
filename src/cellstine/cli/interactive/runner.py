"""Guided interactive launcher for the grouped CELLSTINE CLI."""

from __future__ import annotations

from ..main import _print_result, execute_namespace
from ..parsers import build_parser
from .build_adsorbate import (
    _build_adsorbate_assemble,
    _build_adsorbate_move,
    _build_adsorbate_place,
    _build_adsorbate_visualize,
)
from .build_defect import (
    _build_defect_analyse,
    _build_defect_generate,
    _build_defect_preview,
    _build_defect_supercell,
    _build_defect_visualize,
    _build_symmetry_analyse,
    _build_symmetry_kpoints,
    _build_symmetry_lattice_reduce,
    _build_symmetry_reduce,
    _build_symmetry_visualize,
)
from .build_interface import (
    _build_interface_build,
    _build_interface_match,
    _build_interface_registries,
    _build_interface_sites,
    _build_interface_surface,
    _build_interface_visualize,
)
from .build_moire import (
    _build_moire_find,
    _build_moire_findn,
    _build_moire_make,
    _build_moire_maken,
    _build_moire_translate,
    _build_moire_visualize,
)
from .prompts import (
    PlainGuidedUI,
    _BackInteractive,
    _QuitInteractive,
    _choice,
    _first_artifact,
    _print_command_preview,
    _print_main_menu_banner,
    _prompt,
    _prompt_csv,
    _prompt_float,
    _prompt_yes_no,
    get_guided_ui,
    use_guided_ui,
)


def _workflow_command(group: str, *, allow_back: bool = True) -> list[str]:
    resolved_group = str(group).lower()
    if resolved_group == "moire":
        while True:
            stage = _choice(
                "Moire workflow",
                [
                    {"key": "search", "label": "Search bilayer candidates", "hint": "Recommended when you are starting a new moire search."},
                    {"key": "build", "label": "Build from a saved search", "hint": "Generate one or more saved candidates."},
                    {"key": "stack-search", "label": "Search multi-layer candidates", "hint": "Three or more layers sharing one commensurate cell."},
                    {"key": "stack-build", "label": "Build from a saved multi-layer search", "hint": "Generate stacks of three or more layers."},
                    {"key": "shift", "label": "Shift a built structure", "hint": "Move the upper part of an existing stack."},
                    {"key": "view", "label": "View saved moire results", "hint": "Write a Matplotlib summary or optional Plotly HTML viewer."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "search":
                    return _build_moire_find()
                if stage == "build":
                    return _build_moire_make()
                if stage == "stack-search":
                    return _build_moire_findn()
                if stage == "stack-build":
                    return _build_moire_maken()
                if stage == "shift":
                    return _build_moire_translate()
                return _build_moire_visualize()
            except _BackInteractive:
                continue

    if resolved_group == "surface":
        while True:
            stage = _choice(
                "Surface workflow",
                [
                    {"key": "build", "label": "Build a surface slab from bulk", "hint": "Recommended when you need a substrate or slab first."},
                    {"key": "sites", "label": "Analyse adsorption sites on a slab", "hint": "Inspect top, bridge, hollow, and related sites."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "build":
                    return _build_interface_surface()
                return _build_interface_sites()
            except _BackInteractive:
                continue

    if resolved_group == "adsorbate":
        while True:
            stage = _choice(
                "Adsorbate workflow",
                [
                    {"key": "place", "label": "Place a molecule on a substrate", "hint": "Recommended for adsorption structures."},
                    {"key": "move", "label": "Move a molecule in a stacked structure", "hint": "Translate and rotate a detected top-side molecule."},
                    {"key": "assemble", "label": "Advanced molecular assembly match", "hint": "Does not place a molecule; searches substrate cells for a known packing lattice."},
                    {"key": "path", "label": "Build a diffusion path between two structures", "hint": "Evenly spaced images from one placement to another, ready for a nudged-elastic-band run."},
                    {"key": "view", "label": "Backup visual inspection", "hint": "Optional. Make a quick Matplotlib view if you do not have a structure viewer available."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "place":
                    return _build_adsorbate_place()
                if stage == "move":
                    return _build_adsorbate_move()
                if stage == "assemble":
                    return _build_adsorbate_assemble()
                if stage == "path":
                    return _build_migration_path("adsorbate", subject="a molecule or adatom diffusing")
                return _build_adsorbate_visualize()
            except _BackInteractive:
                continue

    if resolved_group == "defect":
        while True:
            stage = _choice(
                "Defect workflow",
                [
                    {"key": "analyse", "label": "Analyse defect sites", "hint": "Start here to inspect inequivalent atom, interstitial, and adatom sites."},
                    {"key": "generate", "label": "Generate defect structures", "hint": "Preview valid sites, then write one POSCAR per selected inequivalent site."},
                    {"key": "supercell", "label": "Build the host supercell for a defect", "hint": "Choose the cell that puts the most distance between the defect and its images."},
                    {"key": "path", "label": "Build a migration path between two structures", "hint": "Evenly spaced images from one structure to another, ready for a nudged-elastic-band run."},
                    {"key": "preview", "label": "Preview an existing analysis", "hint": "Print the site table from a manifest, analysis JSON, or structure."},
                    {"key": "view", "label": "Backup visual inspection", "hint": "Optional. Draw a defective structure, looking along a direction of your choice."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "analyse":
                    return _build_defect_analyse()
                if stage == "generate":
                    return _build_defect_generate()
                if stage == "supercell":
                    return _build_defect_supercell()
                if stage == "path":
                    return _build_migration_path("defect", subject="a defect hop")
                if stage == "view":
                    return _build_defect_visualize()
                return _build_defect_preview()
            except _BackInteractive:
                continue

    if resolved_group == "symmetry":
        while True:
            stage = _choice(
                "Symmetry workflow",
                [
                    {"key": "analyse", "label": "Analyse symmetry", "hint": "Report space group data and equivalent atom groups."},
                    {"key": "reduce", "label": "Reduce cell", "hint": "Write primitive, conventional, or refined cells."},
                    {"key": "lattice-reduce", "label": "Reduce lattice", "hint": "Write a Niggli- or Delaunay-reduced lattice."},
                    {"key": "kpoints", "label": "Sample the Brillouin zone", "hint": "Write a symmetry-reduced KPOINTS mesh."},
                    {"key": "view", "label": "Backup visual inspection", "hint": "Optional. Draw a cell, looking along a direction of your choice."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "analyse":
                    return _build_symmetry_analyse()
                if stage == "reduce":
                    return _build_symmetry_reduce()
                if stage == "kpoints":
                    return _build_symmetry_kpoints()
                if stage == "view":
                    return _build_symmetry_visualize()
                return _build_symmetry_lattice_reduce()
            except _BackInteractive:
                continue

    while True:
        stage = _choice(
            "Interface workflow",
            [
                {"key": "build", "label": "Build a slab-on-slab interface", "hint": "Fix the bottom slab and strain the top slab to it."},
                {"key": "registries", "label": "List the distinct stacking options of two slabs", "hint": "See which contacts and stacking reversals are genuinely different before building."},
                {"key": "match", "label": "Scan bulk-derived surface matches", "hint": "Estimate promising interface combinations first."},
            ],
            default=1,
            allow_back=allow_back,
        )
        try:
            if stage == "build":
                return _build_interface_build()
            if stage == "registries":
                return _build_interface_registries()
            if stage == "match":
                return _build_interface_match()
            return _build_interface_build()
        except _BackInteractive:
            continue


def _follow_up(group: str, stage: str, result) -> list[str] | None:
    resolved_group = str(group).lower()
    if resolved_group == "moire" and stage == "search":
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "make", "label": "Generate a structure now", "hint": "Use the saved search manifest you just created."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "make":
            return ["moire", "build", str(result.manifest_path), "--indexes", _prompt_csv("Candidate indexes to build", "1"), "--interlayer-distance", str(_prompt_float("Interlayer distance in angstrom", 3.35))]
        return None
    if resolved_group == "surface" and stage == "build":
        slab_path = _first_artifact(result, "slab_poscar")
        if slab_path is None:
            return None
        if "sites_json" in result.artifacts:
            return None
        options = [
            {"key": "sites", "label": "Analyse adsorption sites now", "hint": "Generate the site report from this slab."},
            {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
        ]
        action = _choice("What do you want to do next?", options, default=1)
        if action == "sites":
            return ["surface", "sites", slab_path]
        return None
    if resolved_group == "defect" and stage == "analyse":
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "generate", "label": "Generate defects now", "hint": "Use the analysis manifest you just created."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "generate":
            defect_type = _choice(
                "Which defect family should be generated?",
                [
                    {"key": "vacancy", "label": "Vacancy", "hint": "Remove representative atom sites."},
                    {"key": "substitution", "label": "Substitution", "hint": "Replace representative atom sites."},
                    {"key": "interstitial", "label": "Interstitial", "hint": "Insert species at native void candidates."},
                    {"key": "adatom", "label": "Adatom", "hint": "Insert species above detected surface sites."},
                ],
                default=1,
            )
            argv = ["defect", "generate", str(result.manifest_path), "--defect-type", defect_type]
            site_ids = _prompt("Site IDs to generate, comma-separated; leave blank for all valid inequivalent sites", "", allow_empty=True)
            if site_ids:
                argv.extend(["--site-ids", site_ids])
            if defect_type == "substitution":
                argv.extend(["--substitution-species", _prompt("Replacement species", "S")])
            if defect_type in {"interstitial", "adatom"}:
                argv.extend(["--species", _prompt("Inserted species", "H")])
            if defect_type == "adatom":
                argv.extend(["--height", str(_prompt_float("Height above the detected surface site in angstrom", 2.5))])
            return argv
        return None
    return None


def _run_interactive(group: str | None = None, *, show_banner: bool = True) -> int:
    parser = build_parser()
    active_group = None if group is None else str(group).lower()
    banner_pending = bool(show_banner)

    while True:
        try:
            if banner_pending:
                _print_main_menu_banner()
                banner_pending = False
            if active_group is None:
                active_group = _choice(
                    "CELLSTINE Interactive Mode",
                    [
                        {"key": "moire", "label": "Moire supercell construction", "hint": "Search, build, translate, and visualize commensurate moire structures."},
                        {"key": "surface", "label": "Surface slab workflows", "hint": "Build slabs and analyse adsorption sites."},
                        {"key": "adsorbate", "label": "Molecule on substrate workflows", "hint": "Place, move, and study adsorbates on surfaces."},
                        {"key": "interface", "label": "Interface workflows", "hint": "Match surfaces and construct slab-on-slab interfaces."},
                        {"key": "symmetry", "label": "Symmetry workflows", "hint": "Analyse space groups and reduce conventional or supercells."},
                        {"key": "defect", "label": "Defect workflows", "hint": "Analyse inequivalent defect sites and generate defect POSCARs."},
                    ],
                    default=1,
                    allow_back=False,
                )

            argv = _workflow_command(active_group, allow_back=group is None)
        except _BackInteractive:
            active_group = None
            continue

        _print_command_preview("Planned command", argv)
        if not _prompt_yes_no("Run this command now?", True):
            return 0

        namespace = parser.parse_args(argv)
        result = execute_namespace(namespace)
        _print_result(result)

        try:
            follow_up = _follow_up(active_group, str(namespace.stage), result)
            while follow_up is not None:
                _print_command_preview("Next step", follow_up)
                if not _prompt_yes_no("Run this next command now?", True):
                    break
                next_namespace = parser.parse_args(follow_up)
                result = execute_namespace(next_namespace)
                _print_result(result)
                follow_up = _follow_up(str(next_namespace.group), str(next_namespace.stage), result)
        except _BackInteractive:
            continue

        if group is not None:
            return 0
        try:
            if not _prompt_yes_no("Start another guided workflow?", False):
                return 0
        except _BackInteractive:
            continue
        active_group = None


def run_interactive(group: str | None = None, *, ui: PlainGuidedUI | None = None, show_banner: bool = True) -> int:
    with use_guided_ui(ui):
        try:
            return _run_interactive(group=group, show_banner=show_banner)
        except (KeyboardInterrupt, _QuitInteractive, _BackInteractive):
            get_guided_ui().print()
            get_guided_ui().print("Closed CELLSTINE interactive mode.")
            return 0

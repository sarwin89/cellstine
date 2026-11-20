"""Guided interactive launcher for the grouped CELLSTINE CLI."""

from __future__ import annotations

import contextlib
import io

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
    _build_symmetry_kpath,
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
)
from .build_moire import (
    _build_moire_find,
    _build_moire_findn,
    _build_moire_make,
    _build_moire_maken,
    _build_moire_translate,
    _build_moire_visualize,
)
from .build_path import _build_migration_path
from .build_visualize import _build_structure_visualize
from .prompts import (
    PlainGuidedUI,
    _BackInteractive,
    _PromptHistory,
    _QuitInteractive,
    _choice,
    _first_artifact,
    _print_command_preview,
    _print_main_menu_banner,
    _prompt_yes_no,
    forget_confirmation,
    get_guided_ui,
    guided_history_active,
    run_guided_steps,
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
                    {"key": "stack-search", "label": "Search multi-layer candidates (experimental)", "hint": "Three or more layers sharing one commensurate cell."},
                    {"key": "stack-build", "label": "Build from a saved multi-layer search (experimental)", "hint": "Generate stacks of three or more layers."},
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
                if guided_history_active():
                    raise
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
                if guided_history_active():
                    raise
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
                if guided_history_active():
                    raise
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
                if guided_history_active():
                    raise
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
                    {"key": "kpath", "label": "Band-structure path", "hint": "Write a line-mode KPOINTS file along the high-symmetry path."},
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
                if stage == "kpath":
                    return _build_symmetry_kpath()
                if stage == "view":
                    return _build_symmetry_visualize()
                return _build_symmetry_lattice_reduce()
            except _BackInteractive:
                if guided_history_active():
                    raise
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
            if guided_history_active():
                raise
            continue


def _follow_up(group: str, stage: str, result) -> list[str] | None:
    resolved_group = str(group).lower()
    if resolved_group == "moire" and stage in {"search", "stack-search"}:
        summary = getattr(result, "summary", None) or {}
        if summary.get("candidate_count") == 0:
            get_guided_ui().print()
            get_guided_ui().print(
                "No candidates matched. Try a longer maximum length or a larger strain budget, "
                "or remove any twist window or atom cap."
            )
            return None
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "make", "label": "Generate a structure now", "hint": "Use the saved search manifest you just created."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "make":
            if stage == "stack-search":
                return _build_moire_maken(results_file=str(result.manifest_path))
            return _build_moire_make(results_file=str(result.manifest_path))
        return None
    if resolved_group == "moire" and stage == "build":
        structure = _first_artifact(result, "structures")
        if structure is None:
            return None
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "view", "label": "Draw the first built structure", "hint": "Check the stack before running anything on it."},
                {"key": "shift", "label": "Shift the upper layer", "hint": "Move the first built structure to another registry."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "view":
            return _build_structure_visualize("moire", subject="a built moire stack", structure=structure)
        if action == "shift":
            return _build_moire_translate(poscar_path=structure)
        return None
    if resolved_group == "surface" and stage == "build":
        slab_path = _first_artifact(result, "slab_poscar")
        if slab_path is None:
            return None
        options = [
            {"key": "place", "label": "Place a molecule on this slab", "hint": "Continue with the adsorbate workflow."},
            {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
        ]
        if "sites_json" not in result.artifacts:
            options.insert(0, {"key": "sites", "label": "Analyse adsorption sites now", "hint": "Generate the site report from this slab."})
        action = _choice("What do you want to do next?", options, default=1)
        if action == "sites":
            return _build_interface_sites(slab=slab_path)
        if action == "place":
            return _build_adsorbate_place(substrate=slab_path)
        return None
    if resolved_group == "interface" and stage == "match":
        matches = _first_artifact(result, "matches_json")
        if matches is None:
            return None
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "build", "label": "Build an interface from these matches", "hint": "Uses the matches.json you just created."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "build":
            return _build_interface_build(matches=matches)
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
            return _build_defect_generate(source=str(result.manifest_path))
        return None
    return None


#: Returned instead of a command when the user reads it and declines to run it.
_DECLINED = object()


def _confirmed_command(builder, title="Planned command"):
    argv = builder()
    if argv is None:
        return None
    _print_command_preview(title, argv)
    return argv if _prompt_yes_no("Run this command now?", True) else _DECLINED


def _run_command(parser, argv):
    """Run one confirmed command and return ``(namespace, result)``, or ``None``.

    Guided answers can be slow to give and a failed run must not throw them away,
    so a failure is reported here and the caller offers to edit them.  That
    covers what the workflows raise, the ``SystemExit`` of a parser that refused
    the command, and Ctrl+C during a long search.
    """

    ui = get_guided_ui()
    refused = io.StringIO()
    try:
        with contextlib.redirect_stderr(refused):
            namespace = parser.parse_args(argv)
        result = execute_namespace(namespace)
        _print_result(result)
    except KeyboardInterrupt:
        ui.print()
        ui.print("Stopped before it finished.")
    except SystemExit:
        ui.print()
        ui.print(f"The command was refused: {refused.getvalue().strip().splitlines()[-1] if refused.getvalue().strip() else 'see --help'}")
    except Exception as error:
        ui.print()
        detail = str(error) if isinstance(error, (ValueError, OSError)) else f"{type(error).__name__}: {error}"
        ui.print(f"The command failed: {detail}")
    else:
        return namespace, result
    return None


def _retry_failed_command(history: _PromptHistory) -> bool:
    """Ask whether to reopen a failed command, keeping its answers."""

    action = run_guided_steps(lambda: _choice(
        "That did not finish. What do you want to do?",
        [
            {"key": "edit", "label": "Change my answers", "hint": "Reopens the command; b steps back through your answers."},
            {"key": "menu", "label": "Back to the menu", "hint": "Forget these answers."},
        ],
        default=1,
    ))
    if action == "edit":
        forget_confirmation(history)
        return True
    return False


def _run_interactive(group: str | None = None, *, show_banner: bool = True) -> int:
    parser = build_parser()
    active_group = None if group is None else str(group).lower()
    banner_pending = bool(show_banner)
    history: _PromptHistory | None = None

    while True:
        history = history or _PromptHistory()
        try:
            if banner_pending:
                _print_main_menu_banner()
                get_guided_ui().print("At any question: b + Enter goes back; q + Enter quits. Enter accepts the shown default.")
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

            argv = run_guided_steps(lambda: _confirmed_command(
                lambda: _workflow_command(active_group, allow_back=True)
            ), history)
        except _BackInteractive:
            active_group = None
            history = None
            continue
        except (ValueError, OSError) as error:
            # A builder found nothing to offer, such as a slab with no sites.
            get_guided_ui().print()
            get_guided_ui().print(f"That cannot continue: {error}")
            history = None
            continue

        if argv is None:
            return 0
        if argv is _DECLINED:
            get_guided_ui().print("Nothing was run.")
            history = None
            continue

        finished = _run_command(parser, argv)
        if finished is None:
            try:
                if _retry_failed_command(history):
                    continue
            except _BackInteractive:
                pass
            history = None
            continue
        namespace, result = finished
        history = None

        try:
            group_name, stage_name = active_group, str(getattr(namespace, "stage", ""))
            follow_history: _PromptHistory | None = None
            while True:
                follow_history = follow_history or _PromptHistory()
                follow_up = run_guided_steps(lambda: _confirmed_command(
                    lambda: _follow_up(group_name, stage_name, result), "Next step"
                ), follow_history)
                if follow_up is None or follow_up is _DECLINED:
                    break
                finished = _run_command(parser, follow_up)
                if finished is None:
                    if _retry_failed_command(follow_history):
                        continue
                    break
                namespace, result = finished
                group_name, stage_name = str(namespace.group), str(getattr(namespace, "stage", ""))
                follow_history = None
        except _BackInteractive:
            continue
        except (ValueError, OSError) as error:
            get_guided_ui().print()
            get_guided_ui().print(f"That cannot continue: {error}")
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
        except (KeyboardInterrupt, EOFError, _QuitInteractive, _BackInteractive):
            get_guided_ui().print()
            get_guided_ui().print("Closed CELLSTINE interactive mode.")
            return 0

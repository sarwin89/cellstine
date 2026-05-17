"""Guided interactive launcher for the grouped CELLSTINE CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.manifests import RunManifest
from ..core.previews import format_adsorption_sites, preview_moire_results_file
from ..defect.defect import Defect
from ..interface import surface_backend
from ..interface.interface import parse_miller_notation
from .main import _print_result, execute_namespace
from .parsers import build_parser

INPUT_DIR = Path("input")
RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("output")

MAIN_MENU_BANNER = r"""
 ██████╗███████╗██╗     ██╗     ███████╗████████╗██╗███╗   ██╗███████╗
██╔════╝██╔════╝██║     ██║     ██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝
██║     █████╗  ██║     ██║     ███████╗   ██║   ██║██╔██╗ ██║█████╗
██║     ██╔══╝  ██║     ██║     ╚════██║   ██║   ██║██║╚██╗██║██╔══╝
╚██████╗███████╗███████╗███████╗███████║   ██║   ██║██║ ╚████║███████╗
 ╚═════╝╚══════╝╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝
""".strip("\n")


class _QuitInteractive(Exception):
    """Internal signal for a graceful interactive-mode exit."""


class _BackInteractive(Exception):
    """Internal signal for returning to the previous interactive menu."""


def _print_title(title: str, subtitle: str | None = None) -> None:
    print()
    print(title)
    print("-" * len(title))
    if subtitle:
        print(subtitle)


def _print_main_menu_banner() -> None:
    print()
    print(MAIN_MENU_BANNER)
    print()
    print("Made by Sarwin Chandran 2026")


def _prompt(
    prompt: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
    allow_back: bool = True,
) -> str:
    shown = f" [{default}]" if default not in {None, ""} else ""
    while True:
        answer = input(f"{prompt}{shown}: ").strip()
        if answer:
            if allow_back and answer.lower() in {"b", "back"}:
                raise _BackInteractive()
            return answer
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("Please enter a value.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        try:
            return int(_prompt(prompt, str(default)))
        except ValueError:
            print("Please enter a whole number.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(_prompt(prompt, str(default)))
        except ValueError:
            print("Please enter a number.")


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default = "y" if default_yes else "n"
    while True:
        answer = _prompt(prompt, default).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def _choice(title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
    _print_title(title)
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option['label']}")
        if option.get("hint"):
            print(f"   {option['hint']}")
    if allow_back:
        print("b. Back")
    print("q. Quit interactive mode")
    while True:
        answer = _prompt("Choose an option", str(default), allow_back=allow_back).strip().lower()
        if answer in {"q", "quit", "exit"}:
            raise _QuitInteractive()
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(options):
                option = options[index - 1]
                return str(option.get("value", option["key"]))
        for option in options:
            if answer == str(option["key"]).lower():
                return str(option.get("value", option["key"]))
        print("Please choose one of the numbered options.")


def _relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _find_candidates(patterns: Sequence[str], roots: Sequence[Path], *, limit: int = 8) -> list[Path]:
    found: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for root_index, root in enumerate(roots):
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append((root_index, resolved))
    found.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
    return [path for _, path in found[:limit]]


def _prompt_path(
    label: str,
    *,
    patterns: Sequence[str],
    roots: Sequence[Path],
    default: str | None = None,
    allow_manual: bool = True,
) -> str:
    suggestions = _find_candidates(patterns, roots)
    print()
    print(label)
    if roots:
        print("Search order: " + " -> ".join(str(root) for root in roots))
    if suggestions:
        print("Recent matches:")
        for index, path in enumerate(suggestions, start=1):
            print(f"  {index}. {_relative_display(path)}")
        if allow_manual:
            print("  m. Type a different path")
        print("  b. Back")
        print("  q. Quit interactive mode")
        default_value = "1"
    else:
        print("No suggested files were found, so please type a path.")
        print("Type b to go back or q to quit.")
        default_value = default
    while True:
        answer = _prompt("Selection", default_value, allow_empty=default is not None).strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise _QuitInteractive()
        if suggestions and answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(suggestions):
                return str(suggestions[index - 1])
        if allow_manual and answer.lower() in {"m", "manual"}:
            manual_path = _prompt("Path").strip()
            if manual_path.lower() in {"q", "quit", "exit"}:
                raise _QuitInteractive()
            return manual_path
        if answer:
            return answer
        if default is not None:
            return default
        print("Please choose a suggested file or type a path.")


def _prompt_csv(prompt: str, default: str) -> str:
    return _prompt(prompt, default)


def _prompt_int_range(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        value = _prompt_int(prompt, default)
        if int(minimum) <= value <= int(maximum):
            return value
        print(f"Please enter a value from {int(minimum)} to {int(maximum)}.")


def _parse_matrix_entries(text: str) -> list[int]:
    values = [int(token.strip()) for token in str(text).replace(";", ",").split(",") if token.strip()]
    if len(values) != 4:
        raise ValueError("a 2x2 matrix needs exactly four entries")
    return values


_SITE_LABELS = {
    "top": "Top",
    "bridge": "Bridge",
    "fcc_hollow": "fcc hollow",
    "hcp_hollow": "hcp hollow",
    "hollow": "Generic hollow",
    "fourfold_hollow": "Fourfold hollow",
}


_SITE_HINTS = {
    "top": "Above an outermost surface atom.",
    "bridge": "Above a nearest-neighbour midpoint.",
    "fcc_hollow": "Close-packed hollow with fcc registry.",
    "hcp_hollow": "Close-packed hollow with hcp registry.",
    "hollow": "Triangular hollow where fcc/hcp registry could not be assigned.",
    "fourfold_hollow": "Square-like fourfold hollow.",
}


def _site_options_from_report(site_report) -> list[dict[str, str]]:
    options = []
    for key in ("top", "bridge", "fcc_hollow", "hcp_hollow", "hollow", "fourfold_hollow"):
        count = int(site_report.site_counts.get(key, 0))
        if count <= 0:
            continue
        options.append(
            {
                "key": key,
                "label": f"{_SITE_LABELS.get(key, key)} ({count} found)",
                "hint": _SITE_HINTS.get(key, "Detected in this cell."),
            }
        )
    return options


def _print_detected_sites(site_report) -> None:
    print()
    print("Detected adsorption sites in the selected substrate:")
    if not site_report.site_counts:
        print("  none")
        return
    for key in sorted(site_report.site_counts):
        print(f"  {_SITE_LABELS.get(key, key)}: {int(site_report.site_counts[key])}")


def _print_saved_moire_preview(results_file: str, limit: int = 15) -> None:
    try:
        preview = preview_moire_results_file(results_file, limit=int(limit))
    except Exception as exc:
        print()
        print(f"Candidate preview was skipped: {exc}")
        return
    print()
    print("Candidate options in the selected results file:")
    print(preview)


def _print_site_index_options(site_report, site_type: str, limit: int = 30) -> None:
    sites = surface_backend.sorted_sites_for_type(site_report, site_type)
    print()
    print(format_adsorption_sites(sites, limit=int(limit), title=f"{_SITE_LABELS.get(site_type, site_type)} site positions"))
    if len(sites) > int(limit):
        print("Use `cellstine interface sites` to export the full site table if you need every equivalent site.")


def _format_command(argv: Sequence[str]) -> str:
    parts = []
    for value in argv:
        if any(character.isspace() for character in value):
            parts.append(f'"{value}"')
        else:
            parts.append(value)
    return "cellstine " + " ".join(parts)


def _first_artifact(result, key: str) -> str | None:
    value = result.artifacts.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return None if not value else str(value[0])
    return str(value)


def _detect_moire_build_mode(results_file: str) -> str:
    path = Path(results_file).resolve()
    if path.suffix.lower() == ".dat":
        return "make"
    if path.suffix.lower() == ".json" and path.name != "manifest.json":
        return "maken"
    if path.name == "manifest.json":
        manifest = RunManifest.load(path)
        if manifest.stage == "find":
            return "make"
        if manifest.stage == "findn" and "results_json" in manifest.artifacts:
            return "maken"
    choice = _choice(
        "I could not tell whether this is a bilayer or N-layer results file.",
        [
            {"key": "make", "label": "Bilayer build", "hint": "Use one interlayer distance."},
            {"key": "maken", "label": "N-layer build", "hint": "Use one gap per interlayer region."},
        ],
        default=1,
    )
    return choice


def _prompt_prestrain(layer_name: str, prefix: str) -> list[str]:
    apply = _prompt_yes_no(f"Apply prestrain to the {layer_name} layer before the search?", False)
    if not apply:
        return []
    mode = _choice(
        f"{layer_name.capitalize()} prestrain mode",
        [
            {"key": "biaxial", "label": "Biaxial", "hint": "Strain both in-plane axes equally."},
            {"key": "uniaxial", "label": "Uniaxial", "hint": "Strain one in-plane axis only."},
        ],
        default=1,
    )
    magnitude = _prompt_float(f"{layer_name.capitalize()} prestrain as a fraction (0.01 = 1%)", 0.01)
    argv = [f"--{prefix}-mode", str(mode), f"--{prefix}-value", str(magnitude)]
    if mode == "uniaxial":
        axis = _choice(
            f"{layer_name.capitalize()} strain axis",
            [
                {"key": "axis_a", "value": "a", "label": "a axis", "hint": "Use the first in-plane lattice vector."},
                {"key": "axis_b", "value": "b", "label": "b axis", "hint": "Use the second in-plane lattice vector."},
            ],
            default=1,
        )
        argv.extend([f"--{prefix}-axis", str(axis)])
    return argv


def _build_moire_find() -> list[str]:
    _print_title("Moire Search", "Search a commensurate bilayer and save the candidates for later generation.")
    top = _prompt_path("Choose the top-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    bottom = _prompt_path("Choose the bottom-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR), default=top)
    nindex = _prompt_int("Maximum supercell index (larger values search more combinations)", 12)
    angle_mode = _choice(
        "How should the angle search work?",
        [
            {"key": "auto", "label": "Automatic shortlist", "hint": "Recommended. Let CELLSTINE derive a useful search window."},
            {"key": "explicit", "label": "Explicit angles", "hint": "You already know the twist angles you want to test."},
            {"key": "range", "label": "Custom angle range", "hint": "Set your own minimum and maximum angles."},
        ],
        default=1,
    )
    argv = ["moire", "find", top, bottom, "--nindex", str(nindex)]
    if angle_mode == "explicit":
        argv.extend(["--angles", _prompt_csv("Comma-separated angles in degrees", "13.15,21.787,27.9")])
    elif angle_mode == "range":
        argv.extend(["--min-angle", str(_prompt_float("Minimum angle in degrees", 0.0))])
        argv.extend(["--max-angle", str(_prompt_float("Maximum angle in degrees", 30.0))])
    if _prompt_yes_no("Do you want to search with more than one worker?", False):
        argv.extend(["--workers", str(_prompt_int("Worker count", 4))])
    argv.extend(_prompt_prestrain("top", "prestrain-top"))
    argv.extend(_prompt_prestrain("bottom", "prestrain-bottom"))
    preview_limit = _prompt_int_range("How many lowest-strain candidates should be shown after the search? (0 hides the preview)", 10, 0, 50)
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_findn() -> list[str]:
    _print_title("Multi-Layer Moire Search", "Match several upper layers against one base layer.")
    bottom = _prompt_path("Choose the base-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    upper_count = _prompt_int("How many upper layers do you want to match?", 2)
    upper_paths = [
        _prompt_path(f"Choose upper-layer POSCAR #{index}", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
        for index in range(1, upper_count + 1)
    ]
    match_mode = _choice(
        "How should the multi-layer matching work?",
        [
            {"key": "base_shared", "label": "Shared base-layer match", "hint": "Recommended. Only keep combinations that share the same base-layer cell."},
            {"key": "base_independent", "label": "Independent base-layer matches", "hint": "Match each upper layer against the base separately."},
            {"key": "pairwise", "label": "Pairwise matching", "hint": "Experimental and not recommended for routine use."},
        ],
        default=1,
    )
    nindex = _prompt_int("Maximum supercell index", 12)
    argv = ["moire", "findn", bottom, *upper_paths, "--match-mode", match_mode, "--nindex", str(nindex)]
    angle_mode = _choice(
        "How should angle input work?",
        [
            {"key": "auto", "label": "Automatic per-layer shortlist", "hint": "Recommended for exploration."},
            {"key": "explicit", "label": "Explicit angle list for each upper layer", "hint": "Useful when you already know the interesting angles."},
            {"key": "range", "label": "Custom min and max angles per upper layer", "hint": "Control the search window layer by layer."},
        ],
        default=1,
    )
    if angle_mode == "explicit":
        values = []
        for index in range(1, upper_count + 1):
            values.append(_prompt_csv(f"Angles for upper layer #{index} in degrees", "13.2"))
        argv.extend(["--angles-by-layer", ";".join(values)])
    elif angle_mode == "range":
        min_values = []
        max_values = []
        for index in range(1, upper_count + 1):
            min_values.append(str(_prompt_float(f"Minimum angle for upper layer #{index}", 0.0)))
            max_values.append(str(_prompt_float(f"Maximum angle for upper layer #{index}", 30.0)))
        argv.extend(["--min-angles", ",".join(min_values), "--max-angles", ",".join(max_values)])
    if _prompt_yes_no("Do you want to search with more than one worker?", False):
        argv.extend(["--workers", str(_prompt_int("Worker count", 4))])
    if _prompt_yes_no("Do you want to repeat any structures along c before matching?", False):
        argv.extend(["--bottom-c-repeat", str(_prompt_int("Bottom c repeat", 1))])
        repeats = []
        for index in range(1, upper_count + 1):
            repeats.append(str(_prompt_int(f"Upper layer #{index} c repeat", 1)))
        argv.extend(["--upper-c-repeats", ",".join(repeats)])
    if _prompt_yes_no("Do you want to define prestrain for the layers?", False):
        modes = []
        values = []
        axes = []
        for label in ["bottom", *[f"upper{index}" for index in range(1, upper_count + 1)]]:
            mode = _choice(
                f"Prestrain mode for {label}",
                [
                    {"key": "none", "label": "None", "hint": "Leave this layer unchanged."},
                    {"key": "biaxial", "label": "Biaxial", "hint": "Strain both in-plane axes equally."},
                    {"key": "uniaxial", "label": "Uniaxial", "hint": "Strain one in-plane axis only."},
                ],
                default=1,
            )
            modes.append(mode)
            if mode == "none":
                values.append("0.0")
                axes.append("")
            else:
                values.append(str(_prompt_float(f"Prestrain for {label} as a fraction", 0.01)))
                if mode == "uniaxial":
                    axis = _choice(
                        f"Strain axis for {label}",
                        [
                            {"key": "axis_a", "value": "a", "label": "a axis", "hint": "Use the first in-plane lattice vector."},
                            {"key": "axis_b", "value": "b", "label": "b axis", "hint": "Use the second in-plane lattice vector."},
                        ],
                        default=1,
                    )
                    axes.append(axis)
                else:
                    axes.append("a")
        argv.extend(["--prestrain-modes", ",".join(modes), "--prestrain-values", ",".join(values), "--prestrain-axes", ",".join(axes)])
    preview_limit = _prompt_int_range("How many lowest-strain candidates should be shown after the search? (0 hides the preview)", 10, 0, 50)
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_make() -> list[str]:
    _print_title("Moire Build", "Build one or more saved commensurate candidates.")
    results_file = _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("manifest.json", "*.dat", "*.json"),
        roots=(RUNS_DIR,),
    )
    mode = _detect_moire_build_mode(results_file)
    _print_saved_moire_preview(results_file, limit=15)
    indexes = _prompt_csv("Candidate indexes to build", "1")
    if mode == "make":
        interlayer = _prompt_float("Interlayer distance in angstrom", 3.35)
        argv = ["moire", "make", results_file, "--indexes", indexes, "--interlayer-distance", str(interlayer)]
        if _prompt_yes_no("Do you want to generate with more than one worker?", False):
            argv.extend(["--workers", str(_prompt_int("Worker count", 4))])
        return argv
    interlayers = _prompt_csv("Comma-separated interlayer distances in angstrom", "3.35,3.35")
    return ["moire", "maken", results_file, "--indexes", indexes, "--interlayers", interlayers]


def _build_moire_translate() -> list[str]:
    _print_title("Layer Translation", "Shift the upper part of an already stacked structure.")
    stage = _choice(
        "What kind of stacked structure are you shifting?",
        [
            {"key": "translate", "label": "Bilayer or topmost layer shift", "hint": "Use the direct bilayer-style shift command."},
            {"key": "translaten", "label": "Multi-layer topmost shift", "hint": "Use the multi-layer translation entrypoint."},
        ],
        default=1,
    )
    poscar_path = _prompt_path("Choose the stacked POSCAR", patterns=("*.vasp",), roots=(OUTPUT_DIR, INPUT_DIR))
    coordinate_mode = _choice(
        "How do you want to specify the shift?",
        [
            {"key": "direct", "label": "Direct coordinates", "hint": "Fractional coordinates of the current cell."},
            {"key": "cart", "label": "Cartesian coordinates", "hint": "Angstrom values along x, y, and optionally z."},
        ],
        default=1,
    )
    vector = _prompt_csv("Shift vector", "0.0,0.0")
    flag = "--shift-direct" if coordinate_mode == "direct" else "--shift-cart"
    return ["moire", stage, poscar_path, flag, vector]


def _build_moire_visualize() -> list[str]:
    _print_title("Moire Visualization", "Create a static Matplotlib summary, or optionally an interactive Plotly viewer.")
    results_file = _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("manifest.json", "*.dat", "*.json"),
        roots=(RUNS_DIR,),
    )
    use_specific_indices = _prompt_yes_no("Do you want to visualize only selected candidate indices?", False)
    argv = ["moire", "visualize", results_file]
    if use_specific_indices:
        argv.extend(["--indices", _prompt_csv("Candidate indices", "1,2,3")])
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead of the default Matplotlib plot?", False):
        argv.append("--plotly")
    return argv


def _build_adsorbate_place() -> list[str]:
    _print_title("Adsorbate Placement", "Place a molecule on a slab or a bulk-derived surface.")
    substrate_kind = _choice(
        "What kind of substrate input do you have?",
        [
            {"key": "substrate", "label": "Existing slab or substrate patch", "hint": "Recommended when you already have a surface POSCAR."},
            {"key": "bulk", "label": "Bulk unit cell", "hint": "CELLSTINE will generate the surface slab first."},
        ],
        default=1,
    )
    substrate_roots = (INPUT_DIR, OUTPUT_DIR) if substrate_kind == "bulk" else (OUTPUT_DIR, INPUT_DIR)
    substrate = _prompt_path("Choose the substrate structure", patterns=("*.vasp",), roots=substrate_roots)
    molecule = _prompt_path("Choose the molecule structure", patterns=("*.vasp", "*.xyz", "*.cif"), roots=(INPUT_DIR, OUTPUT_DIR))
    argv = ["adsorbate", "place", substrate, molecule, "--substrate-kind", substrate_kind]
    site_report = None
    if substrate_kind == "bulk":
        miller = _prompt("Surface Miller indices, e.g. 111, 001, or 111x", "111")
        layers = _prompt_int("Number of slab layers", 4)
        vacuum = _prompt_float("Vacuum in angstrom", 15.0)
        argv.extend(["--miller", miller, "--layers", str(layers), "--vacuum", str(vacuum)])
        expansion_mode = _choice(
            "Should the generated substrate be enlarged before site analysis?",
            [
                {"key": "none", "label": "No", "hint": "Analyse the primitive surface cell."},
                {"key": "repeat", "label": "Simple repeats", "hint": "Repeat along surface a and b."},
                {"key": "matrix", "label": "Explicit matrix", "hint": "Use a custom 2x2 in-plane integer matrix."},
            ],
            default=1,
        )
        repeat_a = 1
        repeat_b = 1
        matrix_values = None
        if expansion_mode == "repeat":
            repeat_a = _prompt_int("Repeat along surface a", 2)
            repeat_b = _prompt_int("Repeat along surface b", 2)
            argv.extend(["--substrate-repeat-a", str(repeat_a), "--substrate-repeat-b", str(repeat_b)])
        elif expansion_mode == "matrix":
            matrix_text = _prompt_csv("Matrix entries a,b,c,d", "2,0,0,2")
            matrix_values = _parse_matrix_entries(matrix_text)
            argv.extend(["--substrate-supercell-matrix", matrix_text])
        try:
            preview = surface_backend.build_surface_structure(
                substrate,
                miller=parse_miller_notation(miller),
                layers=layers,
                vacuum=vacuum,
                repeat_a=repeat_a,
                repeat_b=repeat_b,
                supercell_matrix=matrix_values,
            )
            site_report = surface_backend.find_adsorption_sites(preview.structure)
        except Exception as exc:
            print()
            print(f"Site preview failed for this bulk-derived substrate: {exc}")
    else:
        try:
            site_report = surface_backend.find_adsorption_sites(substrate)
        except Exception as exc:
            print()
            print(f"Site analysis failed for the selected substrate: {exc}")

    if site_report is None:
        raise ValueError("could not analyse adsorption sites for this substrate; run `cellstine interface sites` first to inspect the cell")

    _print_detected_sites(site_report)
    site_options = _site_options_from_report(site_report)
    if not site_options:
        raise ValueError("no supported adsorption site families were detected in this substrate")

    site_type = _choice("Which detected adsorption site family do you want?", site_options, default=1)
    argv.extend(["--site-type", site_type])
    site_count = int(site_report.site_counts.get(site_type, 1))
    _print_site_index_options(site_report, site_type)
    argv.extend(["--site-index", str(_prompt_int_range("Site index within that family", 1, 1, site_count))])
    argv.extend(["--height", str(_prompt_float("Height above the top layer in angstrom", 2.5))])
    if _prompt_yes_no("Rotate the molecule about the c axis before placement?", False):
        argv.extend(["--rotate", str(_prompt_float("Rotation in degrees", 30.0))])
    if _prompt_yes_no("Automatically enlarge the substrate if this molecule is too wide for one cell?", True):
        argv.append("--auto-repeat-substrate")
    return argv


def _build_adsorbate_move() -> list[str]:
    _print_title("Move A Molecule", "Move a top-side molecule in an existing stacked structure.")
    structure = _prompt_path("Choose the stacked structure", patterns=("*.vasp",), roots=(OUTPUT_DIR, INPUT_DIR))
    mode = _choice(
        "How do you want to specify the new center of mass?",
        [
            {"key": "direct", "label": "Direct coordinates", "hint": "Recommended when you want a fractional cell position such as 0.5,0.5."},
            {"key": "cart", "label": "Cartesian coordinates", "hint": "Use angstrom values in x, y, and optionally z."},
        ],
        default=1,
    )
    vector = _prompt_csv("Target center-of-mass coordinates", "0.5,0.5")
    flag = "--target-direct" if mode == "direct" else "--target-cart"
    argv = ["adsorbate", "move", structure, flag, vector]
    if _prompt_yes_no("Rotate the molecule after moving it?", False):
        argv.extend(["--rotate", str(_prompt_float("Rotation in degrees about the c axis", 30.0))])
    return argv


def _build_adsorbate_assemble() -> list[str]:
    _print_title(
        "Molecular Assembly Match",
        "Advanced: match a substrate supercell to an experimental molecular packing lattice. This does not place a molecule.",
    )
    substrate = _prompt_path("Choose the substrate structure", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    a_length = _prompt_float("Target a length in angstrom", 12.0)
    use_same_b = _prompt_yes_no("Should b use the same length as a?", True)
    b_length = a_length if use_same_b else _prompt_float("Target b length in angstrom", 12.0)
    angle = _prompt_float("Target in-plane angle in degrees", 60.0)
    max_strain = _prompt_float("Maximum allowed strain as a fraction", 0.05)
    preview_limit = _prompt_int_range("How many lowest-strain candidates should be shown after the search? (0 hides the preview)", 10, 0, 50)
    return [
        "adsorbate", "assemble", substrate,
        "--a-length", str(a_length),
        "--b-length", str(b_length),
        "--angle", str(angle),
        "--max-strain", str(max_strain),
        "--preview-limit", str(preview_limit),
    ]


def _build_adsorbate_visualize() -> list[str]:
    _print_title("Structure Visualization", "Create a labelled Matplotlib multi-view plot of a slab, molecule placement, or interface.")
    structure = _prompt_path("Choose the structure file", patterns=("*.vasp",), roots=(OUTPUT_DIR, INPUT_DIR))
    argv = ["adsorbate", "visualize", structure]
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead?", False):
        argv.append("--plotly")
    return argv


def _build_interface_surface() -> list[str]:
    _print_title("Surface Builder", "Generate a slab from a bulk structure and optional site analysis.")
    bulk = _prompt_path("Choose the bulk POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    miller = _prompt("Miller indices (compact 111/001/111x or comma form 1,1,2x)", "111")
    try:
        preview = surface_backend.analyse_primitive_surface(
            bulk,
            miller=parse_miller_notation(miller),
            probe_layers=6,
        )
        print()
        print("Primitive surface preview:")
        print(f"  centering detected: {preview.centering}")
        print(f"  stacking sequence: {preview.stacking_sequence}")
        print(f"  repeating period: {preview.stacking_period or 'A'}")
        print(f"  atoms per detected layer: {', '.join(str(value) for value in preview.atoms_per_layer)}")
        print(f"  in-plane angle: {preview.inplane_angle_deg:.3f} degrees")
    except Exception as exc:
        print()
        print(f"Primitive surface preview was skipped: {exc}")
    enlarge_mode = _choice(
        "Do you want to enlarge the surface in plane?",
        [
            {"key": "none", "label": "No", "hint": "Keep the smallest generated cell."},
            {"key": "repeat", "label": "Simple repeats", "hint": "Repeat along a and b."},
            {"key": "matrix", "label": "Explicit 2x2 supercell matrix", "hint": "Use a custom in-plane integer matrix."},
        ],
        default=1,
    )
    expansion_args: list[str] = []
    if enlarge_mode == "repeat":
        expansion_args.extend(["--repeat-a", str(_prompt_int("Repeat along a", 2)), "--repeat-b", str(_prompt_int("Repeat along b", 2))])
    elif enlarge_mode == "matrix":
        expansion_args.extend(["--supercell-matrix", _prompt_csv("Matrix entries a,b,c,d", "2,0,0,3")])
    layers = _prompt_int("Number of slab layers", 4)
    vacuum = _prompt_float("Vacuum in angstrom", 15.0)
    argv = ["interface", "surface", bulk, "--miller", miller, "--layers", str(layers), "--vacuum", str(vacuum), *expansion_args]
    if _prompt_yes_no("Also detect adsorption sites after building the slab?", True):
        argv.append("--analyse-sites")
    return argv


def _build_interface_sites() -> list[str]:
    _print_title("Surface Site Analysis", "Identify adsorption sites on an existing slab.")
    slab = _prompt_path("Choose the slab POSCAR", patterns=("*.vasp",), roots=(OUTPUT_DIR, INPUT_DIR))
    side = _choice(
        "Which surface side should be analysed?",
        [
            {"key": "top", "label": "Top surface", "hint": "Recommended for most adsorption workflows."},
            {"key": "bottom", "label": "Bottom surface", "hint": "Use when the active surface faces downward."},
        ],
        default=1,
    )
    return ["interface", "sites", slab, "--surface-side", side]


def _prompt_interface_side(label: str) -> list[str]:
    kind = _choice(
        f"What is the {label} input?",
        [
            {"key": "surface", "label": "Existing slab POSCAR", "hint": "Recommended when you already have the surface built."},
            {"key": "bulk", "label": "Bulk unit cell", "hint": "CELLSTINE will generate the slab first."},
        ],
        default=1,
    )
    roots = (INPUT_DIR, OUTPUT_DIR) if kind == "bulk" else (OUTPUT_DIR, INPUT_DIR, RUNS_DIR)
    path = _prompt_path(f"Choose the {label} structure", patterns=("*.vasp",), roots=roots)
    extra = [path, f"--{label}-kind", kind]
    if kind == "bulk":
        extra.extend([f"--{label}-miller", _prompt(f"{label.capitalize()} Miller indices, e.g. 111, 001, or 111x", "111")])
        extra.extend([f"--{label}-layers", str(_prompt_int(f"{label.capitalize()} layer count", 4))])
        extra.extend([f"--{label}-vacuum", str(_prompt_float(f"{label.capitalize()} vacuum in angstrom", 15.0))])
    return extra


def _build_interface_build() -> list[str]:
    _print_title("Interface Builder", "Stack a top slab onto a fixed bottom slab or bulk-derived surface.")
    bottom = _prompt_interface_side("bottom")
    top = _prompt_interface_side("top")
    gap = _prompt_float("Gap between bottom and top slabs in angstrom", 3.0)
    return ["interface", "build", bottom[0], top[0], *bottom[1:], *top[1:], "--gap", str(gap)]


def _build_interface_match() -> list[str]:
    _print_title("Surface Match Scan", "Search bulk-derived surface combinations for low-strain matches.")
    bottom = _prompt_path("Choose the bottom bulk structure", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    top = _prompt_path("Choose the top bulk structure", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    miller_mode = _choice(
        "Which Miller set should be scanned?",
        [
            {"key": "default", "label": "Common low-index set", "hint": "Recommended. Use 1,0,0 1,1,0 and 1,1,1."},
            {"key": "custom", "label": "Custom Miller lists", "hint": "Type the surface families you want to compare."},
        ],
        default=1,
    )
    argv = ["interface", "match", bottom, top]
    if miller_mode == "custom":
        argv.extend(["--bottom-millers", *_prompt("Bottom Miller list separated by spaces", "1,1,1 1,1,0").split()])
        argv.extend(["--top-millers", *_prompt("Top Miller list separated by spaces", "1,1,1 1,1,0").split()])
    layer_mode = _choice(
        "How should layer counts be scanned?",
        [
            {"key": "default", "label": "Single default thickness", "hint": "Use 4 layers for both surfaces."},
            {"key": "custom", "label": "Custom layer-count lists", "hint": "Try several slab thicknesses on each side."},
        ],
        default=1,
    )
    if layer_mode == "custom":
        argv.extend(["--bottom-layers-list", *_prompt("Bottom layer counts separated by spaces", "4 6").split()])
        argv.extend(["--top-layers-list", *_prompt("Top layer counts separated by spaces", "4 6").split()])
    argv.extend(["--max-strain", str(_prompt_float("Maximum allowed strain as a fraction", 0.05))])
    return argv


def _build_interface_visualize() -> list[str]:
    _print_title("Structure Visualization", "Create a labelled Matplotlib multi-view plot of a slab or interface.")
    structure = _prompt_path("Choose the structure file", patterns=("*.vasp",), roots=(OUTPUT_DIR, INPUT_DIR))
    argv = ["interface", "visualize", structure]
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead?", False):
        argv.append("--plotly")
    return argv


def _build_defect_analyse() -> list[str]:
    _print_title("Defect Site Analysis", "Find inequivalent atom, interstitial, and surface adatom sites.")
    structure = _prompt_path("Choose the structure to analyse", patterns=("*.vasp", "POSCAR", "CONTCAR"), roots=(INPUT_DIR, OUTPUT_DIR))
    kind = _choice(
        "What kind of structure is this?",
        [
            {"key": "auto", "label": "Auto-detect", "hint": "Recommended. Detect bulk versus slab from the vacuum gap."},
            {"key": "bulk", "label": "Bulk cell", "hint": "Use exact spglib equivalence when available."},
            {"key": "surface", "label": "Surface or slab", "hint": "Use layer-aware native grouping and adsorption-site detection."},
            {"key": "molecule-on-substrate", "label": "Molecule on substrate", "hint": "Treat it as a slab-like structure for layer/site logic."},
        ],
        default=1,
    )
    backend = _choice(
        "Which equivalence backend should be used?",
        [
            {"key": "auto", "label": "Auto", "hint": "Recommended. spglib for bulk if available, native for slabs."},
            {"key": "native", "label": "Native", "hint": "Approximate, transparent grouping without optional dependencies."},
            {"key": "spglib", "label": "spglib", "hint": "Exact bulk symmetry and Wyckoff labels when installed."},
        ],
        default=1,
    )
    argv = ["defect", "analyse", structure, "--structure-kind", kind, "--backend", backend]
    if kind in {"surface", "molecule-on-substrate", "auto"}:
        side = _choice(
            "Which surface side should be used for adatom sites?",
            [
                {"key": "top", "label": "Top surface", "hint": "Recommended for adsorbates."},
                {"key": "bottom", "label": "Bottom surface", "hint": "Use if the exposed side is below."},
            ],
            default=1,
        )
        argv.extend(["--surface-side", side])
    return argv


def _build_symmetry_analyse() -> list[str]:
    _print_title("Symmetry Analysis", "Analyse space group, operations, Wyckoff labels, and equivalent atoms.")
    structure = _prompt_path("Choose the structure to analyse", patterns=("*.vasp", "POSCAR", "CONTCAR"), roots=(INPUT_DIR, OUTPUT_DIR))
    backend = _choice(
        "Which symmetry backend should be used?",
        [
            {"key": "auto", "label": "Auto", "hint": "Recommended. Use spglib when installed, otherwise show the native lattice summary."},
            {"key": "spglib", "label": "spglib", "hint": "Exact crystallographic symmetry."},
            {"key": "native", "label": "Native", "hint": "Lattice geometry only; no exact space group."},
        ],
        default=1,
    )
    return ["symmetry", "analyse", structure, "--backend", backend]


def _build_symmetry_reduce() -> list[str]:
    _print_title("Cell Reduction", "Write a primitive, conventional, or refined cell using spglib.")
    structure = _prompt_path("Choose the structure to reduce", patterns=("*.vasp", "POSCAR", "CONTCAR"), roots=(INPUT_DIR, OUTPUT_DIR))
    cell = _choice(
        "Which cell should be written?",
        [
            {"key": "primitive", "label": "Primitive", "hint": "Smallest periodic cell found by symmetry."},
            {"key": "conventional", "label": "Conventional", "hint": "Standardized conventional cell."},
            {"key": "refined", "label": "Refined", "hint": "Symmetry-refined version of the input setting."},
        ],
        default=1,
    )
    return ["symmetry", "reduce", structure, "--cell", cell]


def _build_symmetry_lattice_reduce() -> list[str]:
    _print_title("Lattice Reduction", "Write a Niggli- or Delaunay-reduced lattice representation.")
    structure = _prompt_path("Choose the structure", patterns=("*.vasp", "POSCAR", "CONTCAR"), roots=(INPUT_DIR, OUTPUT_DIR))
    reduction = _choice(
        "Which lattice reduction should be used?",
        [
            {"key": "niggli", "label": "Niggli", "hint": "Recommended compact reduced lattice."},
            {"key": "delaunay", "label": "Delaunay", "hint": "Alternative crystallographic lattice reduction."},
        ],
        default=1,
    )
    return ["symmetry", "lattice-reduce", structure, "--reduction", reduction]


def _defect_analysis_from_preview(path: str, kind: str, backend: str, side: str):
    result = Defect().preview(
        path,
        limit=50,
        structure_kind=kind,
        backend=backend,
        surface_side=side,
    )
    print()
    print("Available defect sites:")
    print(result.payload.get("defect_preview", "No preview was returned."))
    return result


def _defect_site_ids(analysis: dict, site_kind: str) -> list[str]:
    return [str(site["site_id"]) for site in analysis.get("sites", []) if site.get("site_kind") == site_kind]


def _build_defect_generate() -> list[str]:
    _print_title("Defect Structure Generation", "Preview valid sites first, then generate one POSCAR per selected inequivalent site.")
    source = _prompt_path(
        "Choose a structure, defect manifest, or defect analysis JSON",
        patterns=("manifest.json", "defect_analysis.json", "*.vasp", "POSCAR", "CONTCAR"),
        roots=(INPUT_DIR, OUTPUT_DIR, RUNS_DIR),
    )
    kind = _choice(
        "What kind of structure is this?",
        [
            {"key": "auto", "label": "Auto-detect", "hint": "Recommended for most inputs."},
            {"key": "bulk", "label": "Bulk cell", "hint": "Vacancy/substitution in a periodic bulk structure."},
            {"key": "surface", "label": "Surface or slab", "hint": "Layer-aware sites and adatoms on exposed surfaces."},
            {"key": "molecule-on-substrate", "label": "Molecule on substrate", "hint": "Useful for adatoms or edits around adsorbed systems."},
        ],
        default=1,
    )
    backend = _choice(
        "Which equivalence backend should be used?",
        [
            {"key": "auto", "label": "Auto", "hint": "Recommended."},
            {"key": "native", "label": "Native", "hint": "No optional dependencies."},
            {"key": "spglib", "label": "spglib", "hint": "Exact bulk equivalence and Wyckoff labels."},
        ],
        default=1,
    )
    side = "top"
    if kind in {"surface", "molecule-on-substrate", "auto"}:
        side = _choice(
            "Which surface side should be used for adatom sites?",
            [
                {"key": "top", "label": "Top surface", "hint": "Recommended for adsorbates."},
                {"key": "bottom", "label": "Bottom surface", "hint": "Use if the exposed side is below."},
            ],
            default=1,
        )
    preview_result = _defect_analysis_from_preview(source, kind, backend, side)
    analysis = dict(preview_result.payload.get("analysis", {}))
    options = []
    if _defect_site_ids(analysis, "atom"):
        options.extend(
            [
                {"key": "vacancy", "label": "Vacancy", "hint": "Remove one representative atom from selected inequivalent sites."},
                {"key": "substitution", "label": "Substitution", "hint": "Replace one representative atom with another species."},
                {"key": "antisite", "label": "Antisite", "hint": "Special substitution-style defect for swapped lattice species."},
            ]
        )
    if _defect_site_ids(analysis, "interstitial"):
        options.append({"key": "interstitial", "label": "Interstitial", "hint": "Insert a species at native candidate void sites."})
    if _defect_site_ids(analysis, "adatom"):
        options.append({"key": "adatom", "label": "Adatom", "hint": "Place a species above detected top/bridge/hollow sites."})
    if not options:
        raise ValueError("no supported defect sites were detected for this input")
    defect_type = _choice("Which defect family should be generated?", options, default=1)
    site_kind = "atom" if defect_type in {"vacancy", "substitution", "antisite"} else defect_type
    valid_ids = _defect_site_ids(analysis, site_kind)
    print()
    print(f"Valid {site_kind} site IDs:")
    print("  " + ", ".join(valid_ids[:30]))
    if len(valid_ids) > 30:
        print(f"  ... {len(valid_ids) - 30} more not shown")
    site_ids = _prompt("Site IDs to generate, comma-separated; leave blank for all valid inequivalent sites", "", allow_empty=True)
    argv = [
        "defect",
        "generate",
        str(preview_result.artifacts["analysis_json"]),
        "--defect-type",
        defect_type,
        "--structure-kind",
        kind,
        "--backend",
        backend,
        "--surface-side",
        side,
    ]
    if site_ids:
        argv.extend(["--site-ids", site_ids])
    if defect_type in {"substitution", "antisite"}:
        argv.extend(["--substitution-species", _prompt("Replacement species", "S")])
        if _prompt_yes_no("Restrict to one original species?", False):
            argv.extend(["--original-species", _prompt("Original species to replace", "Mo")])
    if defect_type in {"interstitial", "adatom"}:
        argv.extend(["--species", _prompt("Inserted species", "H")])
    if defect_type == "adatom":
        argv.extend(["--height", str(_prompt_float("Height above the detected surface site in angstrom", 2.5))])
    return argv


def _build_defect_preview() -> list[str]:
    _print_title("Defect Preview", "Print the detected inequivalent defect-site table.")
    source = _prompt_path(
        "Choose a structure, defect manifest, or defect analysis JSON",
        patterns=("manifest.json", "defect_analysis.json", "*.vasp", "POSCAR", "CONTCAR"),
        roots=(INPUT_DIR, OUTPUT_DIR, RUNS_DIR),
    )
    return ["defect", "preview", source, "--limit", str(_prompt_int_range("How many sites should be shown?", 30, 1, 200))]


def _workflow_command(group: str, *, allow_back: bool = True) -> list[str]:
    resolved_group = str(group).lower()
    if resolved_group == "moire":
        while True:
            stage = _choice(
                "Moire workflow",
                [
                    {"key": "find", "label": "Search bilayer candidates", "hint": "Recommended when you are starting a new moire search."},
                    {"key": "findn", "label": "Search multi-layer candidates", "hint": "Match several upper layers against one base layer."},
                    {"key": "build", "label": "Build from a saved search", "hint": "Generate one or more saved candidates."},
                    {"key": "translate", "label": "Shift a built structure", "hint": "Move the upper part of an existing stack."},
                    {"key": "visualize", "label": "Backup visual inspection", "hint": "Optional. Write a Matplotlib summary if you do not want to open external viewers."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "find":
                    return _build_moire_find()
                if stage == "findn":
                    return _build_moire_findn()
                if stage == "build":
                    return _build_moire_make()
                if stage == "translate":
                    return _build_moire_translate()
                return _build_moire_visualize()
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
                    {"key": "visualize", "label": "Backup visual inspection", "hint": "Optional. Make a quick Matplotlib view if you do not have a structure viewer available."},
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
                    {"key": "preview", "label": "Preview an existing analysis", "hint": "Print the site table from a manifest, analysis JSON, or structure."},
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "analyse":
                    return _build_defect_analyse()
                if stage == "generate":
                    return _build_defect_generate()
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
                ],
                default=1,
                allow_back=allow_back,
            )
            try:
                if stage == "analyse":
                    return _build_symmetry_analyse()
                if stage == "reduce":
                    return _build_symmetry_reduce()
                return _build_symmetry_lattice_reduce()
            except _BackInteractive:
                continue

    while True:
        stage = _choice(
            "Interface workflow",
            [
                {"key": "surface", "label": "Build a surface slab from bulk", "hint": "Recommended when you need a substrate or slab first."},
                {"key": "sites", "label": "Analyse adsorption sites on a slab", "hint": "Inspect top, bridge, hollow, and related sites."},
                {"key": "build", "label": "Build a slab-on-slab interface", "hint": "Fix the bottom slab and strain the top slab to it."},
                {"key": "match", "label": "Scan bulk-derived surface matches", "hint": "Estimate promising interface combinations first."},
                {"key": "visualize", "label": "Backup visual inspection", "hint": "Optional. Make a quick Matplotlib view if you do not have a structure viewer available."},
            ],
            default=1,
            allow_back=allow_back,
        )
        try:
            if stage == "surface":
                return _build_interface_surface()
            if stage == "sites":
                return _build_interface_sites()
            if stage == "build":
                return _build_interface_build()
            if stage == "match":
                return _build_interface_match()
            return _build_interface_visualize()
        except _BackInteractive:
            continue


def _follow_up(group: str, stage: str, result) -> list[str] | None:
    resolved_group = str(group).lower()
    if resolved_group == "moire" and stage == "find":
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "make", "label": "Generate a structure now", "hint": "Use the saved search manifest you just created."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "make":
            return ["moire", "make", str(result.manifest_path), "--indexes", _prompt_csv("Candidate indexes to build", "1"), "--interlayer-distance", str(_prompt_float("Interlayer distance in angstrom", 3.35))]
        return None
    if resolved_group == "moire" and stage == "findn" and "results_json" in result.artifacts:
        action = _choice(
            "What do you want to do next?",
            [
                {"key": "maken", "label": "Generate an N-layer structure now", "hint": "Use the shared-base candidates you just found."},
                {"key": "done", "label": "Finish here", "hint": "Return to the main workflow menu."},
            ],
            default=1,
        )
        if action == "maken":
            return ["moire", "maken", str(result.manifest_path), "--indexes", _prompt_csv("Candidate indexes to build", "1"), "--interlayers", _prompt_csv("Interlayer distances in angstrom", "3.35,3.35")]
        return None
    if resolved_group == "interface" and stage == "surface":
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
            return ["interface", "sites", slab_path]
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


def _run_interactive(group: str | None = None) -> int:
    parser = build_parser()
    active_group = None if group is None else str(group).lower()

    while True:
        try:
            if active_group is None:
                _print_main_menu_banner()
                active_group = _choice(
                    "CELLSTINE Interactive Mode",
                    [
                        {"key": "moire", "label": "Moire supercell construction", "hint": "Search, build, translate, and visualize commensurate moire structures."},
                        {"key": "adsorbate", "label": "Molecule on substrate workflows", "hint": "Place, move, and study adsorbates on surfaces."},
                        {"key": "interface", "label": "Surface and interface workflows", "hint": "Build slabs, analyse sites, and construct interfaces."},
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

        print()
        print("Planned command:")
        print(_format_command(argv))
        print("Running now.")

        namespace = parser.parse_args(argv)
        result = execute_namespace(namespace)
        _print_result(result)

        try:
            follow_up = _follow_up(active_group, str(namespace.stage), result)
            while follow_up is not None:
                print()
                print("Next step:")
                print(_format_command(follow_up))
                print("Running next step.")
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


def run_interactive(group: str | None = None) -> int:
    try:
        return _run_interactive(group=group)
    except (KeyboardInterrupt, _QuitInteractive, _BackInteractive):
        print()
        print("Closed CELLSTINE interactive mode.")
        return 0

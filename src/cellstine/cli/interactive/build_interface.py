"""Interactive command builders for the interface and surface workflows."""

from __future__ import annotations

from ...interface.surface import backend as surface_backend
from ...interface.workflow.interface import parse_miller_notation
from .build_visualize import _build_structure_visualize
from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    _choice,
    _print_title,
    _prompt,
    _prompt_csv,
    _prompt_float,
    _prompt_int,
    _prompt_path,
    _prompt_yes_no,
)


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


def _prompt_stacking_options(*, allow_relative: bool) -> list[str]:
    """Ask for the stacking senses and the contact of a close-packed interface.

    ``allow_relative`` says whether the top slab may be described relative to
    the bottom one (``abc``/``cba``), which needs the two slabs to share their
    in-plane cell.
    """

    if not _prompt_yes_no(
        "Set the stacking order and the contact of the two close-packed slabs?", False
    ):
        return []
    argv: list[str] = []
    bottom = _choice(
        "Bottom slab stacking",
        [
            {"key": "keep", "label": "Keep it as built", "hint": "Recommended. The bottom slab fixes the A -> B -> C direction."},
            {"key": "mirror", "label": "Reverse it", "hint": "Reflect the slab, turning ABCABC into CBACBA."},
        ],
        default=1,
    )
    if bottom != "keep":
        argv.extend(["--bottom-stacking", bottom])
    top_options = [
        {"key": "keep", "label": "Keep it as built", "hint": "Leave the top slab alone."},
        {"key": "mirror", "label": "Reverse it", "hint": "Always reflect the top slab."},
    ]
    if allow_relative:
        top_options.extend(
            [
                {"key": "abc", "label": "Same sense as the bottom slab", "hint": "ABCABC on ABCABC, reflecting the slab only if needed."},
                {"key": "cba", "label": "Opposite sense to the bottom slab", "hint": "CBACBA on ABCABC, i.e. a twin across the contact."},
            ]
        )
    top = _choice("Top slab stacking", top_options, default=1)
    if top != "keep":
        argv.extend(["--top-stacking", top])
    if not allow_relative:
        return argv
    registry = _choice(
        "Which layer should meet which at the contact?",
        [
            {"key": "none", "label": "Leave the slabs where they are", "hint": "No in-plane shift is applied."},
            {"key": "fcc", "label": "fcc hollow", "hint": "The top layer continues the sequence of the bottom slab, e.g. C on B gives C-A."},
            {"key": "hcp", "label": "hcp hollow", "hint": "The top layer sits above the second layer of the bottom slab."},
            {"key": "eclipsed", "label": "Eclipsed", "hint": "The two contacting layers sit directly on top of each other."},
            {"key": "contact", "label": "Name the contact letters", "hint": "Such as C-A. Only the difference matters, so A-A, B-B and C-C are one option."},
        ],
        default=1,
    )
    if registry == "contact":
        argv.extend(["--registry", _prompt("Contact such as C-A, C-B or C-C", "C-A")])
    elif registry != "none":
        argv.extend(["--registry", registry])
    return argv


def _build_interface_registries() -> list[str]:
    _print_title(
        "Interface Stacking Options",
        "List the genuinely different ways two close-packed slabs can be put in contact.",
    )
    bottom = _prompt_interface_side("bottom")
    top = _prompt_interface_side("top")
    argv = ["interface", "registries", bottom[0], top[0], *bottom[1:], *top[1:]]
    if _prompt_yes_no(
        "Also list the combinations removed as duplicates, i.e. mirror images and the same "
        "interface turned over?",
        False,
    ):
        argv.append("--include-equivalent")
    return argv


def _build_interface_build() -> list[str]:
    _print_title("Interface Builder", "Stack a top slab onto a fixed bottom slab or bulk-derived surface.")
    mode = _choice(
        "How should the interface cell be chosen?",
        [
            {
                "key": "match",
                "label": "Commensurate match from a match scan",
                "hint": "Recommended for two different materials. Uses matches.json from `interface match`.",
            },
            {
                "key": "direct",
                "label": "Stack the two 1x1 cells directly",
                "hint": "Only sensible when the two surface cells already agree, such as the same material.",
            },
        ],
        default=1,
    )
    if mode == "match":
        matches = _prompt_path(
            "Choose the matches.json written by the match scan",
            patterns=("matches.json",),
            roots=(RUNS_DIR, OUTPUT_DIR),
        )
        index = _prompt_int("Match index to build", 1)
        gap = _prompt_float("Gap between bottom and top slabs in angstrom", 3.0)
        argv = ["interface", "build", "--match", matches, "--match-index", str(index), "--gap", str(gap)]
        if _prompt_yes_no("Set the vacuum thickness of the finished cell?", False):
            argv.extend(["--vacuum", str(_prompt_float("Vacuum thickness in angstrom", 15.0))])
        argv.extend(_prompt_stacking_options(allow_relative=False))
        return argv
    bottom = _prompt_interface_side("bottom")
    top = _prompt_interface_side("top")
    gap = _prompt_float("Gap between bottom and top slabs in angstrom", 3.0)
    argv = ["interface", "build", bottom[0], top[0], *bottom[1:], *top[1:], "--gap", str(gap)]
    argv.extend(
        [
            "--max-strain",
            str(_prompt_float("Largest principal logarithmic strain accepted between the two 1x1 cells", 0.05)),
        ]
    )
    argv.extend(_prompt_stacking_options(allow_relative=True))
    return argv


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
    argv.extend(["--max-strain", str(_prompt_float("Strain budget for one slab as a fraction", 0.05))])
    argv.extend(["--max-length", str(_prompt_float("Maximum matched supercell length in angstrom", 20.0))])
    mode = _choice(
        "How should the strain be shared?",
        [
            {"key": "shared", "label": "Share between both slabs", "hint": "Recommended. Both slabs take half of the relative strain."},
            {"key": "film", "label": "Strain the top film only", "hint": "Keeps the bottom slab rigid, as for growth on a thick substrate."},
        ],
        default=1,
    )
    argv.extend(["--strain-mode", mode])
    return argv


def _build_interface_visualize() -> list[str]:
    return _build_structure_visualize("interface", subject="a slab or interface")

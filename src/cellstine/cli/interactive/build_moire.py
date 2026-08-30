"""Interactive command builders for the moire workflows."""

from __future__ import annotations
from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    _choice,
    _print_saved_moire_preview,
    _print_title,
    _prompt_csv,
    _prompt_float,
    _prompt_int,
    _prompt_int_range,
    _prompt_path,
    _prompt_yes_no,
)


def _build_moire_find() -> list[str]:
    _print_title("Moire Search", "Search a commensurate bilayer and save the candidates for later generation.")
    top = _prompt_path("Choose the top-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    bottom = _prompt_path("Choose the bottom-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR), default=top)
    argv = [
        "moire", "search", top, bottom,
        "--length", str(_prompt_float("Maximum supercell length in angstrom", 30.0)),
    ]
    rigid = _prompt_yes_no(
        "Keep both layers rigid and accept only exactly commensurate twists? "
        "(the usual choice for a twisted homobilayer)",
        False,
    )
    if rigid:
        argv.append("--rigid")
    else:
        argv.extend([
            "--top-strain", str(_prompt_float("Top principal logarithmic strain budget as a fraction", 0.02)),
            "--bottom-strain", str(_prompt_float("Bottom principal logarithmic strain budget as a fraction", 0.02)),
        ])
    if _prompt_yes_no("Do you want to restrict the reported twist angles to a window?", False):
        lower = _prompt_float("Smallest twist angle to report in degrees", 0.0)
        upper = _prompt_float("Largest twist angle to report in degrees", 30.0)
        if upper < lower:
            lower, upper = upper, lower
        argv.extend(["--twist", f"{lower:g}:{upper:g}"])
    if _prompt_yes_no("Do you want to cap the number of atoms in a candidate cell?", rigid):
        argv.extend(["--atoms", str(_prompt_int("Maximum atoms in a candidate supercell", 400))])
    if _prompt_yes_no("Use the symmetry-preserving branch when it applies?", False):
        argv.append("--symmetric")
    if _prompt_yes_no(
        "Search on the layer cells exactly as given? "
        "(answer no, the default, to fold a supercell input back onto its own primitive layer first)",
        False,
    ):
        argv.append("--keep-layer-cells")
    if _prompt_yes_no("Show live search progress?", False):
        argv.append("--progress")
    preview_limit = _prompt_int_range("How many ranked candidates should be shown after the search? (0 hides the preview)", 10, 0, 50)
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_findn() -> list[str]:
    _print_title(
        "Multi-Layer Moire Search",
        "Match one or more upper layers against a rigid base layer and share one cell.",
    )
    base = _prompt_path("Choose the base-layer POSCAR", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
    upper_count = _prompt_int_range("How many upper layers do you want to stack?", 2, 1, 8)
    upper_paths = [
        _prompt_path(f"Choose upper-layer POSCAR #{index}", patterns=("*.vasp",), roots=(INPUT_DIR, OUTPUT_DIR))
        for index in range(1, upper_count + 1)
    ]
    max_length = _prompt_float("Maximum in-plane cell length in angstrom", 20.0)
    argv = ["moire", "stack-search", base, *upper_paths, "--length", str(max_length)]
    if _prompt_yes_no("Do you want a different strain budget for each upper layer?", False):
        budgets = [
            str(_prompt_float(f"Strain budget for upper layer #{index} as a fraction", 0.02))
            for index in range(1, upper_count + 1)
        ]
        argv.extend(["--layer-strains", ",".join(budgets)])
    else:
        argv.extend(["--layer-strain", str(_prompt_float("Strain budget for every upper layer as a fraction", 0.02))])
    argv.extend(["--atoms", str(_prompt_int("Maximum atoms in the whole stack", 600))])
    if _prompt_yes_no("Do you want to set a minimum in-plane cell length?", False):
        argv.extend(["--min-length", str(_prompt_float("Minimum in-plane cell length in angstrom", 5.0))])
    preview_limit = _prompt_int_range(
        "How many candidates should be shown after the search? (0 hides the preview)", 10, 0, 50
    )
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_maken() -> list[str]:
    _print_title("Multi-Layer Build", "Build one or more saved multi-layer candidates.")
    results_file = _prompt_path(
        "Choose a saved multi-layer results file or manifest",
        patterns=("manifest.json", "results_nlayer.json", "*.json"),
        roots=(RUNS_DIR,),
    )
    indexes = _prompt_csv("Candidate indexes to build", "1")
    argv = ["moire", "stack-build", results_file, "--indexes", indexes]
    if _prompt_yes_no("Do you want a different gap above each layer?", False):
        argv.extend(["--interlayers", _prompt_csv("Gaps in angstrom, bottom to top", "3.35,3.35")])
    else:
        argv.extend(["--interlayer-distance", str(_prompt_float("Gap between neighbouring layers in angstrom", 3.35))])
    if _prompt_yes_no("Do you want to set the vacuum instead of keeping the input c axis?", False):
        argv.extend(["--vacuum", str(_prompt_float("Total vacuum in angstrom, split equally above and below", 15.0))])
    return argv


def _build_moire_make() -> list[str]:
    _print_title("Moire Build", "Build one or more saved commensurate candidates.")
    results_file = _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("manifest.json", "*.json"),
        roots=(RUNS_DIR,),
    )
    _print_saved_moire_preview(results_file, limit=15)
    indexes = _prompt_csv("Candidate indexes to build", "1")
    interlayer = _prompt_float("Interlayer distance in angstrom", 3.35)
    argv = ["moire", "build", results_file, "--indexes", indexes, "--interlayer-distance", str(interlayer)]
    if _prompt_yes_no("Do you want to set the vacuum instead of keeping the input c axis?", False):
        argv.extend(["--vacuum", str(_prompt_float("Total vacuum in angstrom, split equally above and below", 15.0))])
    if _prompt_yes_no("Do you want to generate with more than one worker?", False):
        argv.extend(["--workers", str(_prompt_int("Worker count", 4))])
    return argv


def _build_moire_translate() -> list[str]:
    _print_title("Layer Translation", "Shift the upper part of an already stacked structure.")
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
    return ["moire", "shift", poscar_path, flag, vector]


def _build_moire_visualize() -> list[str]:
    _print_title("Moire Visualization", "Create a static Matplotlib summary, or optionally an interactive Plotly viewer.")
    results_file = _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("manifest.json", "*.json"),
        roots=(RUNS_DIR,),
    )
    use_specific_indices = _prompt_yes_no("Do you want to visualize only selected candidate indices?", False)
    argv = ["moire", "view", results_file]
    if use_specific_indices:
        argv.extend(["--indices", _prompt_csv("Candidate indices", "1,2,3")])
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead of the default Matplotlib plot?", False):
        argv.append("--plotly")
    return argv

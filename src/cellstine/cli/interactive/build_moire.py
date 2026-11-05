"""Interactive command builders for the moire workflows."""

from __future__ import annotations

from ...core.constants import DEFAULT_RESULTS_FORMAT
from ..argtypes import parse_float_list, parse_float_vector, parse_index_spec
from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    STRUCTURE_PATTERNS,
    _choice,
    _index_parser,
    _print_saved_moire_preview,
    _print_title,
    _prompt_float,
    _prompt_int,
    _prompt_int_range,
    _prompt_parsed,
    _prompt_path,
    _prompt_strain,
    _prompt_yes_no,
    _remember,
    _saved_candidate_count,
)


def _build_moire_find() -> list[str]:
    _print_title("Moire Search", "Search a commensurate bilayer and save the candidates for later generation.")
    top = _prompt_path("Choose the top-layer POSCAR", patterns=STRUCTURE_PATTERNS, roots=(INPUT_DIR, OUTPUT_DIR))
    bottom = _prompt_path("Choose the bottom-layer POSCAR", patterns=STRUCTURE_PATTERNS, roots=(INPUT_DIR, OUTPUT_DIR), default=top)
    argv = [
        "moire", "search", top, bottom,
        "--length", str(_prompt_float("Maximum supercell length in angstrom", 30.0, positive=True)),
    ]
    rigid = _prompt_yes_no(
        "Keep both layers rigid and accept only exactly commensurate twists? "
        "(the usual choice for a twisted homobilayer)",
        False,
    )
    if rigid:
        argv.append("--rigid")
    elif _prompt_yes_no("Use the same strain budget for both layers?", True):
        argv.extend(["--strain", str(_prompt_strain("Strain budget as a fraction"))])
    else:
        argv.extend([
            "--top-strain", str(_prompt_strain("Top principal logarithmic strain budget as a fraction")),
            "--bottom-strain", str(_prompt_strain("Bottom principal logarithmic strain budget as a fraction")),
        ])
    limit = _prompt_int(
        "Result shortlist size (0 saves every accepted candidate, about 2 kB each)", 0, minimum=0
    )
    if limit:
        argv.extend(["--max-candidates", str(limit)])
    argv.append("--progress")
    preview_limit = 10
    if _prompt_yes_no("Adjust advanced options? (twist window, atom cap, symmetric branch, layer cells, preview)", False):
        if _prompt_yes_no("Do you want to restrict the reported twist angles to a window?", False):
            lower = _prompt_float("Smallest twist angle to report in degrees", 0.0, minimum=0.0)
            upper = _prompt_float("Largest twist angle to report in degrees", 30.0, minimum=0.0)
            if upper < lower:
                lower, upper = upper, lower
            argv.extend(["--twist", f"{lower:g}:{upper:g}"])
        if _prompt_yes_no("Do you want to cap the number of atoms in a candidate cell?", rigid):
            argv.extend(["--atoms", str(_prompt_int("Maximum atoms in a candidate supercell", 400, minimum=1))])
        if _prompt_yes_no("Use the symmetry-preserving branch when it applies?", False):
            argv.append("--symmetric")
        if _prompt_yes_no(
            "Search on the layer cells exactly as given? "
            "(answer no, the default, to fold a supercell input back onto its own primitive layer first)",
            False,
        ):
            argv.append("--keep-layer-cells")
        preview_limit = _prompt_int_range(
            "How many ranked candidates should be shown after the search? (0 hides the preview)", 10, 0, 50
        )
        format_options = [
            {"key": "npz", "label": "NPZ (faster for large searches)", "hint": "Default. Build and plot with CELLSTINE as usual."},
            {"key": "json", "label": "JSON (readable)", "hint": "Choose this for tools that require JSON."},
            {"key": "npz-compressed", "label": "Compressed NPZ (smaller files)", "hint": "Lossless compression trades some write time for disk space."},
        ]
        results_format = _choice(
            "Saved results format", format_options,
            default=next(index for index, option in enumerate(format_options, 1)
                         if option["key"] == DEFAULT_RESULTS_FORMAT),
        )
        if results_format != DEFAULT_RESULTS_FORMAT:
            argv.extend(["--results-format", results_format])
    elif rigid:
        # Keep the previous guided rigid-search default when advanced questions
        # are hidden. The advanced flow still lets the user explicitly opt out.
        argv.extend(["--atoms", "400"])
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_findn() -> list[str]:
    _print_title(
        "Experimental Multi-Layer Moire Search",
        "Match one or more upper layers against a rigid base layer and share one cell.",
    )
    base = _prompt_path("Choose the base-layer POSCAR", patterns=STRUCTURE_PATTERNS, roots=(INPUT_DIR, OUTPUT_DIR))
    upper_count = _prompt_int_range("How many upper layers do you want to stack?", 2, 1, 8)
    upper_paths = [
        _prompt_path(f"Choose upper-layer POSCAR #{index}", patterns=STRUCTURE_PATTERNS, roots=(INPUT_DIR, OUTPUT_DIR))
        for index in range(1, upper_count + 1)
    ]
    max_length = _prompt_float("Maximum in-plane cell length in angstrom", 20.0, positive=True)
    argv = ["moire", "stack-search", base, *upper_paths, "--length", str(max_length)]
    if _prompt_yes_no("Do you want a different strain budget for each upper layer?", False):
        budgets = [
            str(_prompt_strain(f"Strain budget for upper layer #{index} as a fraction"))
            for index in range(1, upper_count + 1)
        ]
        argv.extend(["--layer-strains", ",".join(budgets)])
    else:
        argv.extend(["--layer-strain", str(_prompt_strain("Strain budget for every upper layer as a fraction"))])
    argv.extend(["--atoms", str(_prompt_int("Maximum atoms in the whole stack", 600, minimum=1))])
    if _prompt_yes_no("Do you want to set a minimum in-plane cell length?", False):
        argv.extend(["--min-length", str(_prompt_float("Minimum in-plane cell length in angstrom", 5.0, positive=True))])
    argv.extend(["--max-candidates", str(_prompt_int("Maximum stack candidates to record", 200, minimum=1))])
    preview_limit = _prompt_int_range(
        "How many candidates should be shown after the search? (0 hides the preview)", 10, 0, 50
    )
    argv.extend(["--preview-limit", str(preview_limit)])
    return argv


def _build_moire_maken(*, results_file: str | None = None) -> list[str]:
    _print_title("Multi-Layer Build", "Build one or more saved multi-layer candidates.")
    results_file = results_file or _prompt_path(
        "Choose a saved multi-layer results file or manifest",
        patterns=("results_nlayer.json",),
        roots=(RUNS_DIR,),
    )
    indexes = _prompt_parsed("Candidate indexes to build, e.g. 1 or 1,3-5", "1", parse_index_spec)
    argv = ["moire", "stack-build", results_file, "--indexes", indexes]
    if _prompt_yes_no("Do you want a different gap above each layer?", False):
        argv.extend(["--interlayers", _prompt_parsed("Gaps in angstrom, bottom to top", "3.35,3.35", parse_float_list)])
    else:
        argv.extend(["--interlayer-distance", str(_prompt_float("Gap between neighbouring layers in angstrom", 3.35, positive=True))])
    if _prompt_yes_no("Do you want to set the vacuum instead of keeping the input c axis?", False):
        argv.extend(["--vacuum", str(_prompt_float("Total vacuum in angstrom, split equally above and below", 15.0, minimum=0.0))])
    return argv


def _build_moire_make(*, results_file: str | None = None) -> list[str]:
    _print_title("Moire Build", "Build one or more saved commensurate candidates.")
    results_file = results_file or _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("results.json", "results.npz"),
        roots=(RUNS_DIR,),
    )
    count = _print_saved_moire_preview(results_file, limit=15)
    indexes = _prompt_parsed("Candidate indexes to build, e.g. 1 or 1,3-5", "1", _index_parser(count))
    interlayer = _prompt_float("Interlayer distance in angstrom", 3.35, positive=True)
    argv = ["moire", "build", results_file, "--indexes", indexes, "--interlayer-distance", str(interlayer)]
    if _prompt_yes_no("Do you want to set the vacuum instead of keeping the input c axis?", False):
        argv.extend(["--vacuum", str(_prompt_float("Total vacuum in angstrom, split equally above and below", 15.0, minimum=0.0))])
    if _prompt_yes_no("Do you want to generate with more than one worker?", False):
        argv.extend(["--workers", str(_prompt_int("Worker count", 4, minimum=1))])
    return argv


def _build_moire_translate(*, poscar_path: str | None = None) -> list[str]:
    _print_title("Layer Translation", "Shift the upper part of an already stacked structure.")
    poscar_path = poscar_path or _prompt_path("Choose the stacked POSCAR", patterns=STRUCTURE_PATTERNS, roots=(OUTPUT_DIR, INPUT_DIR))
    coordinate_mode = _choice(
        "How do you want to specify the shift?",
        [
            {"key": "direct", "label": "Direct coordinates", "hint": "Fractional coordinates of the current cell."},
            {"key": "cart", "label": "Cartesian coordinates", "hint": "Angstrom values along x, y, and optionally z."},
        ],
        default=1,
    )
    vector = _prompt_parsed("Shift vector", "0.0,0.0", parse_float_vector)
    flag = "--shift-direct" if coordinate_mode == "direct" else "--shift-cart"
    return ["moire", "shift", poscar_path, flag, vector]


def _build_moire_visualize() -> list[str]:
    _print_title("Moire Visualization", "Create a static Matplotlib summary, or optionally an interactive Plotly viewer.")
    results_file = _prompt_path(
        "Choose a saved moire results file or manifest",
        patterns=("results.json", "results.npz"),
        roots=(RUNS_DIR,),
    )
    use_specific_indices = _prompt_yes_no("Do you want to visualize only selected candidate indices?", False)
    argv = ["moire", "view", results_file]
    if use_specific_indices:
        count = _remember(lambda: _saved_candidate_count(results_file))
        argv.extend(["--indices", _prompt_parsed("Candidate indices", "1,2,3", _index_parser(count))])
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead of the default Matplotlib plot?", False):
        argv.append("--plotly")
    return argv

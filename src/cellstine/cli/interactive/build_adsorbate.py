"""Interactive command builders for the adsorbate workflows."""

from __future__ import annotations

from ...interface.surface import backend as surface_backend
from ...interface.workflow.interface import parse_miller_notation
from .build_visualize import _build_structure_visualize
from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    _choice,
    _parse_matrix_entries,
    _print_detected_sites,
    _print_site_index_options,
    _print_title,
    _prompt,
    _prompt_csv,
    _prompt_float,
    _prompt_int,
    _prompt_int_range,
    _prompt_path,
    _prompt_yes_no,
    _site_options_from_report,
)


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
    max_length = _prompt_float("Maximum in-plane supercell length in angstrom", 30.0)
    top_strain = _prompt_float("Target molecular-lattice strain budget as a fraction", 0.05)
    bottom_strain = _prompt_float("Substrate strain budget as a fraction", 0.05)
    preview_limit = _prompt_int_range("How many lowest-strain candidates should be shown after the search? (0 hides the preview)", 10, 0, 50)
    return [
        "adsorbate", "assemble", substrate,
        "--a-length", str(a_length),
        "--b-length", str(b_length),
        "--angle", str(angle),
        "--max-length", str(max_length),
        "--top-strain", str(top_strain),
        "--bottom-strain", str(bottom_strain),
        "--preview-limit", str(preview_limit),
    ]


def _build_adsorbate_visualize() -> list[str]:
    return _build_structure_visualize("adsorbate", subject="a slab, molecule placement, or interface")

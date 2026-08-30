"""Interactive command builders for the defect and symmetry workflows."""

from __future__ import annotations

from pathlib import Path

from ...defect.analysis import _normalise_supercell
from ...defect.workflow import Defect
from .build_visualize import _build_structure_visualize, _prompt_view_direction
from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    _choice,
    _print_title,
    _prompt,
    _prompt_float,
    _prompt_int_range,
    _prompt_path,
    _prompt_yes_no,
)


def _prompt_layers(analysis: dict) -> str | None:
    """Ask which atomic planes the defect should be made in.

    Answering anything but the first option splits every orbit of equivalent
    sites over the planes it visits, so the two surfaces of a symmetric slab --
    one orbit, and one structure without this -- become one structure each.
    """

    layers = list(analysis.get("layers", []))
    if len(layers) < 2:
        return None
    print()
    print(f"The structure has {len(layers)} atomic plane(s) along the direction of observation:")
    for layer in layers:
        composition = " ".join(
            f"{species}{count}" for species, count in dict(layer.get("species_counts", {})).items()
        )
        print(
            f"  plane {int(layer['layer_id']):>3d}  height {float(layer['projection']):9.4f} A  "
            f"{int(layer['atom_count'])} atom(s)  {composition}"
        )
    choice = _choice(
        "Which atomic planes should the defect be made in?",
        [
            {
                "key": "none",
                "label": "One per inequivalent site",
                "hint": "Recommended for a bulk cell: the smallest set that covers every distinct defect.",
            },
            {"key": "all", "label": "Every plane", "hint": "One defect per plane, so a slab is sampled from surface to centre."},
            {"key": "surface", "label": "The two outermost planes", "hint": "Surface defects only."},
            {"key": "pick", "label": "Choose planes", "hint": "A list such as 1,3 or a range such as 2-4; -1 is the topmost."},
        ],
        default=1,
    )
    if choice == "none":
        return None
    if choice == "pick":
        return _prompt("Planes to use", "1")
    return choice


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
    view_direction = _prompt_view_direction()
    interstitials = _choice(
        "Which interstitial sites should be listed?",
        [
            {
                "key": "maxima",
                "label": "Widest holes only",
                "hint": "Recommended. Centres surrounded by atoms on all sides, such as the octahedral and tetrahedral holes of a close-packed metal.",
            },
            {
                "key": "saddles",
                "label": "Also the sites held by two or three atoms",
                "hint": "Adds the octahedral site of a body-centred cubic metal, where carbon sits in ferrite, and the bond centre of a covalent crystal.",
            },
        ],
        default=1,
    )
    argv = [
        "defect",
        "analyse",
        structure,
        "--structure-kind",
        kind,
        "--backend",
        backend,
        "--view-direction",
        view_direction,
    ]
    if interstitials == "saddles":
        argv.append("--interstitial-saddles")
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


def _build_symmetry_kpoints() -> list[str]:
    _print_title(
        "Brillouin-Zone Sampling",
        "Write a symmetry-reduced KPOINTS mesh for a structure.",
    )
    structure = _prompt_path(
        "Choose the structure to sample",
        patterns=("*.vasp", "POSCAR", "CONTCAR"),
        roots=(INPUT_DIR, OUTPUT_DIR),
    )
    how = _choice(
        "How should the mesh size be chosen?",
        [
            {
                "key": "spacing",
                "label": "From a k-point spacing",
                "hint": "Recommended. The cell sets the divisions, so a supercell is sampled more coarsely.",
            },
            {"key": "divisions", "label": "Explicit divisions", "hint": "Give n1,n2,n3 yourself."},
        ],
        default=1,
    )
    argv = ["symmetry", "kpoints", structure]
    if how == "spacing":
        spacing = _prompt_float(
            "Largest allowed step between sampled wavevectors in 1/angstrom", default=0.25
        )
        argv.extend(["--spacing", str(spacing)])
    else:
        divisions = _prompt("Mesh divisions as n1,n2,n3", default="6,6,6")
        argv.extend(["--divisions", divisions])
    mesh = _choice(
        "Which mesh should be used?",
        [
            {"key": "gamma", "label": "Gamma-centred", "hint": "Recommended. Contains Gamma and keeps the full symmetry."},
            {
                "key": "monkhorst",
                "label": "Monkhorst-Pack",
                "hint": "Half a step off on even axes; fewer points, but it can break some operations.",
            },
        ],
        default=1,
    )
    argv.extend(["--mesh", mesh])
    if _prompt_yes_no("Is this a slab with vacuum, so the surface normal needs one point only?", default=False):
        argv.append("--surface")
    return argv


def _defect_analysis_from_preview(
    path: str,
    kind: str,
    backend: str,
    side: str,
    view_direction: str = "auto",
    interstitial_saddles: bool = False,
):
    result = Defect().preview(
        path,
        limit=50,
        structure_kind=kind,
        backend=backend,
        surface_side=side,
        view_direction=view_direction,
        interstitial_saddles=interstitial_saddles,
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
    view_direction = _prompt_view_direction()
    saddles = _choice(
        "Which interstitial sites should be offered?",
        [
            {
                "key": "maxima",
                "label": "Widest holes only",
                "hint": "Recommended. Centres surrounded by atoms on all sides.",
            },
            {
                "key": "saddles",
                "label": "Also the sites held by two or three atoms",
                "hint": "Adds the octahedral site of a body-centred cubic metal and the bond centre of a covalent crystal.",
            },
        ],
        default=1,
    ) == "saddles"
    preview_result = _defect_analysis_from_preview(
        source, kind, backend, side, view_direction, interstitial_saddles=saddles
    )
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
    layers = None if defect_type == "adatom" else _prompt_layers(analysis)
    coverage = _choice(
        "How many structures should be written?",
        [
            {
                "key": "inequivalent",
                "label": "One per inequivalent site",
                "hint": "Recommended. The smallest set that covers every distinct defect.",
            },
            {
                "key": "all",
                "label": "One per equivalent atom",
                "hint": "Every copy of each site in its own structure; use only if a later step breaks the symmetry.",
            },
        ],
        default=1,
    )
    supercell = _prompt_supercell(source, site_ids)
    target = source if supercell else str(preview_result.artifacts["analysis_json"])
    argv = [
        "defect",
        "generate",
        target,
        "--defect-type",
        defect_type,
        "--structure-kind",
        kind,
        "--backend",
        backend,
        "--surface-side",
        side,
        "--view-direction",
        view_direction,
        "--generate",
        coverage,
    ]
    if saddles:
        argv.append("--interstitial-saddles")
    if layers:
        argv.extend(["--layers", layers])
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
    argv.extend(supercell)
    return argv


def _prompt_supercell(source: str, site_ids: str) -> list[str]:
    """Ask whether, and how, the host cell should be enlarged first.

    A defect in the cell as read is only a point defect if the cell is large
    enough; enlarging it pushes the periodic images of the defect apart.  Two
    ways are offered: naming the separation the images should have, which lets
    the search pick the smallest cell that reaches it, or repeating the cell
    along its own axes.  Either way the cell itself changes, so this is only
    offered for a structure file, and only when the whole set of inequivalent
    sites is wanted -- site IDs are numbered against the cell they were found
    in, and the enlarged cell renumbers them.
    """

    if site_ids:
        return []
    if Path(source).suffix.lower() == ".json" or Path(source).name == "manifest.json":
        return []
    choice = _choice(
        "Should the host cell be enlarged before the defect is made?",
        [
            {
                "key": "none",
                "label": "No, use the cell as read",
                "hint": "Right if the cell is already a supercell.",
            },
            {
                "key": "distance",
                "label": "Yes, to a chosen defect-image separation",
                "hint": "Recommended. The smallest cell reaching that separation is found for you.",
            },
            {
                "key": "repeats",
                "label": "Yes, repeat it along its own axes",
                "hint": "The plain n1 x n2 x n3 repeat, if you already know the one you want.",
            },
        ],
        default=1,
    )
    if choice == "none":
        return []
    if choice == "distance":
        distance = _prompt_float(
            "Smallest acceptable distance from the defect to its nearest image in angstrom", 10.0
        )
        return ["--min-image-distance", str(distance)]
    while True:
        raw = _prompt("Repeats along a, b and c (e.g. 2,2,1, or a single number for all three)", "2,2,2")
        try:
            repeats = _normalise_supercell(raw)
        except ValueError as error:
            print(f"  {error}")
            continue
        if repeats is None:
            return []
        return ["--supercell", ",".join(str(value) for value in repeats)]


def _build_defect_supercell() -> list[str]:
    _print_title(
        "Defect Supercell",
        "Build the host supercell a point defect should be made in.",
    )
    structure = _prompt_path(
        "Choose the host structure",
        patterns=("*.vasp", "POSCAR", "CONTCAR"),
        roots=(INPUT_DIR, OUTPUT_DIR),
    )
    kind = _choice(
        "What kind of structure is this?",
        [
            {"key": "auto", "label": "Auto-detect", "hint": "Recommended. Detect bulk versus slab from the vacuum gap."},
            {"key": "bulk", "label": "Bulk cell", "hint": "Measure the image separation in all three directions."},
            {"key": "surface", "label": "Surface or slab", "hint": "Measure it in the plane only; vacuum separates the images along c."},
        ],
        default=1,
    )
    mode = _choice(
        "How should the size of the cell be decided?",
        [
            {
                "key": "distance",
                "label": "By the defect-image separation it must reach",
                "hint": "Recommended. The smallest cell reaching that separation is used.",
            },
            {
                "key": "cells",
                "label": "By how many host cells it may hold",
                "hint": "The best-shaped cell of at most that size is used.",
            },
        ],
        default=1,
    )
    argv = ["defect", "supercell", structure, "--structure-kind", kind]
    if mode == "distance":
        distance = _prompt_float(
            "Smallest acceptable distance from the defect to its nearest image in angstrom", 10.0
        )
        argv.extend(["--min-image-distance", str(distance)])
    else:
        argv.extend(
            ["--max-cells", str(_prompt_int_range("Largest number of host cells", 32, 1, 512))]
        )
    if _prompt_yes_no("Also list the best supercell of every smaller size?", False):
        argv.extend(
            ["--table", str(_prompt_int_range("List sizes up to how many host cells?", 16, 1, 128))]
        )
    return argv


def _build_defect_preview() -> list[str]:
    _print_title("Defect Preview", "Print the detected inequivalent defect-site table.")
    source = _prompt_path(
        "Choose a structure, defect manifest, or defect analysis JSON",
        patterns=("manifest.json", "defect_analysis.json", "*.vasp", "POSCAR", "CONTCAR"),
        roots=(INPUT_DIR, OUTPUT_DIR, RUNS_DIR),
    )
    view_direction = _prompt_view_direction()
    return [
        "defect",
        "preview",
        source,
        "--limit",
        str(_prompt_int_range("How many sites should be shown?", 30, 1, 200)),
        "--view-direction",
        view_direction,
    ]


def _build_defect_visualize() -> list[str]:
    return _build_structure_visualize("defect", subject="a defective structure")


def _build_symmetry_visualize() -> list[str]:
    return _build_structure_visualize("symmetry", subject="a reduced or refined cell")

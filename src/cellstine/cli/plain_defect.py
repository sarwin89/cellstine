"""Dependency-free parser group for defect commands."""

from __future__ import annotations

import argparse

from ..core.constants import (
    DEFAULT_CELL_LIMIT,
    DIRECTION_HELP,
    LAYER_SELECTION_HELP,
    LAYER_TOLERANCE,
)
from .argtypes import (
    HelpFormatter,
    parse_positive_float,
    parse_string_list,
    parse_supercell,
    parse_supercell_matrix,
)


def add_defect_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``defect`` command group."""

    defect = groups.add_parser(
        "defect",
        formatter_class=HelpFormatter,
        help="defect-site analysis and generation",
        description=(
            "Defect workflows. Run `cellstine defect` with no stage to start the guided workflow. "
            "The default output is one POSCAR per inequivalent selected site."
        ),
    )
    defect_sub = defect.add_subparsers(dest="stage")

    defect_analyse = defect_sub.add_parser(
        "analyse",
        formatter_class=HelpFormatter,
        help="analyse inequivalent defect sites in a structure",
        description=(
            "Analyse inequivalent atom sites, approximate native interstitial candidates, and surface adatom sites when relevant. "
            "For bulk equivalence, --backend auto uses spglib when installed; surface/slab analysis prefers native layer-aware grouping."
        ),
    )
    defect_analyse.add_argument("structure")
    defect_analyse.add_argument("--structure-kind", choices=["auto", "bulk", "surface", "slab", "molecule-on-substrate"], default="auto")
    defect_analyse.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    defect_analyse.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    defect_analyse.add_argument("--layer-tolerance", type=float, default=LAYER_TOLERANCE, help="layer grouping tolerance in angstrom")
    defect_analyse.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    defect_analyse.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")
    defect_analyse.add_argument("--view-direction", default="auto", help=DIRECTION_HELP)
    defect_analyse.add_argument("--interstitial-saddles", action="store_true", help="also list the saddles of the distance to the nearest atom as interstitial candidates: the sites held by two or three atoms, such as the octahedral site of a body-centred cubic metal and the bond centre of a covalent crystal")

    defect_generate = defect_sub.add_parser(
        "generate",
        formatter_class=HelpFormatter,
        help="generate defect POSCARs from an analysis or structure",
        description=(
            "Generate inequivalent defect structures. ANALYSIS_OR_STRUCTURE may be a defect manifest, "
            "a defect_analysis.json file, or a raw POSCAR/CONTCAR/.vasp structure."
        ),
    )
    defect_generate.add_argument("analysis_or_structure")
    defect_generate.add_argument("--defect-type", required=True, choices=["vacancy", "substitution", "interstitial", "adatom", "antisite", "divacancy", "paired-vacancy"])
    defect_generate.add_argument("--site-ids", type=parse_string_list, default=None, help="comma-separated site IDs to generate; defaults to all valid inequivalent sites")
    defect_generate.add_argument("--species", default=None, help="inserted species for interstitial/adatom, or replacement fallback for substitution")
    defect_generate.add_argument("--substitution-species", default=None, help="replacement species for substitution or antisite defects")
    defect_generate.add_argument("--original-species", default=None, help="restrict atom-site defects to this original species")
    defect_generate.add_argument(
        "--generate",
        choices=["inequivalent", "all"],
        default="inequivalent",
        help=(
            "inequivalent: one structure per orbit of symmetry-equivalent sites (default); "
            "all: one structure per equivalent atom, restricted to --layers when it is given"
        ),
    )
    defect_generate.add_argument("--height", type=float, default=2.5, help="adatom height above the detected surface site in angstrom")
    defect_generate.add_argument("--output-dir", default=None)
    defect_generate.add_argument("--structure-kind", choices=["auto", "bulk", "surface", "slab", "molecule-on-substrate"], default="auto")
    defect_generate.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    defect_generate.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    defect_generate.add_argument("--layer-tolerance", type=float, default=LAYER_TOLERANCE, help="layer grouping tolerance in angstrom")
    defect_generate.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    defect_generate.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")
    defect_generate.add_argument("--view-direction", default="auto", help=DIRECTION_HELP)
    defect_generate.add_argument("--layers", default=None, help=LAYER_SELECTION_HELP)
    defect_generate.add_argument("--interstitial-saddles", action="store_true", help="also list the saddles of the distance to the nearest atom as interstitial candidates: the sites held by two or three atoms, such as the octahedral site of a body-centred cubic metal and the bond centre of a covalent crystal")
    defect_generate.add_argument(
        "--supercell",
        type=parse_supercell,
        default=None,
        help="repeat the host cell before the defect is made, e.g. 2,2,1 (dilutes the defect)",
    )
    defect_generate.add_argument(
        "--supercell-matrix",
        type=parse_supercell_matrix,
        default=None,
        help=(
            "repeat the host cell by any integer matrix, row by row; write a matrix with a "
            "negative first entry as --supercell-matrix=-1,1,1,1,-1,1,1,1,-1"
        ),
    )
    defect_generate.add_argument(
        "--min-image-distance",
        type=parse_positive_float,
        default=None,
        help=(
            "enlarge the host cell to the smallest supercell that puts this distance in angstrom "
            "between the defect and its nearest periodic image"
        ),
    )
    defect_generate.add_argument(
        "--cell-limit",
        type=int,
        default=DEFAULT_CELL_LIMIT,
        help="largest cell count searched by --min-image-distance",
    )
    defect_generate.add_argument(
        "--keep-cell-height",
        action="store_true",
        help="write the host cell unchanged instead of lengthening c to keep an adatom's vacuum gap",
    )

    defect_supercell = defect_sub.add_parser(
        "supercell",
        formatter_class=HelpFormatter,
        help="build the host supercell a point defect should be made in",
        description=(
            "Choose and write the host supercell a point defect should be made in. The cell is "
            "chosen for the distance it puts between the defect and its periodic images, not for "
            "the number of atoms: every sublattice of the host lattice of a given size is "
            "enumerated in Hermite normal form and the roundest one wins, which is usually not a "
            "plain repeat. A slab is measured in the plane only, since vacuum already separates "
            "its images along c."
        ),
    )
    defect_supercell.add_argument("structure")
    defect_supercell.add_argument(
        "--min-image-distance",
        type=parse_positive_float,
        default=None,
        help="smallest acceptable distance from the defect to its nearest periodic image, in angstrom",
    )
    defect_supercell.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="instead, use the best cell holding at most this many host cells",
    )
    defect_supercell.add_argument(
        "--cell-limit",
        type=int,
        default=DEFAULT_CELL_LIMIT,
        help="largest cell count searched when a minimum image distance is requested",
    )
    defect_supercell.add_argument(
        "--table",
        dest="table_limit",
        type=int,
        default=0,
        help="also print the best supercell of every size up to this many host cells",
    )
    defect_supercell.add_argument("--structure-kind", choices=["auto", "bulk", "surface", "slab", "molecule-on-substrate"], default="auto")
    defect_supercell.add_argument("--layer-tolerance", type=float, default=LAYER_TOLERANCE, help="layer grouping tolerance in angstrom")
    defect_supercell.add_argument("--output", default=None, help="where to write the host supercell POSCAR")

    defect_path = defect_sub.add_parser(
        "path",
        formatter_class=HelpFormatter,
        help="build the chain of images between two structures of one cell",
        description=(
            "Write the evenly spaced chain of images a nudged-elastic-band run starts from: "
            "00/POSCAR is the initial structure, the last folder the final one. The atoms of the "
            "two endpoints are paired by the assignment that makes the path shortest -- the file "
            "order of the two structures is not trusted -- and every atom travels to its nearest "
            "periodic image, so an atom that leaves through one face returns through the opposite "
            "one the short way. Both endpoints must share one cell and one composition."
        ),
    )
    defect_path.add_argument("start_structure")
    defect_path.add_argument("end_structure")
    defect_path.add_argument(
        "--images",
        type=int,
        default=3,
        help="number of intermediate images; the chain holds two more than this",
    )
    defect_path.add_argument(
        "--no-match",
        dest="match",
        action="store_false",
        help="pair the atoms in file order instead of by the shortest-path assignment",
    )
    defect_path.add_argument("--output-dir", default=None, help="where to write the numbered image folders")

    defect_preview = defect_sub.add_parser(
        "preview",
        formatter_class=HelpFormatter,
        help="print a compact table of available defect sites",
    )
    defect_preview.add_argument("analysis_or_structure")
    defect_preview.add_argument("--limit", type=int, default=30)
    defect_preview.add_argument("--structure-kind", choices=["auto", "bulk", "surface", "slab", "molecule-on-substrate"], default="auto")
    defect_preview.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    defect_preview.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    defect_preview.add_argument("--layer-tolerance", type=float, default=LAYER_TOLERANCE, help="layer grouping tolerance in angstrom")
    defect_preview.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    defect_preview.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")
    defect_preview.add_argument("--view-direction", default="auto", help=DIRECTION_HELP)
    defect_preview.add_argument("--interstitial-saddles", action="store_true", help="also list the saddles of the distance to the nearest atom as interstitial candidates: the sites held by two or three atoms, such as the octahedral site of a body-centred cubic metal and the bond centre of a covalent crystal")

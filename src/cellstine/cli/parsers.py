"""Argument parser construction for the grouped CELLSTINE CLI."""

from __future__ import annotations

import argparse
import math
from typing import List


APP_NAME = "CELLSTINE"
APP_EXPANSION = "CELL Superlattice Transformation INterface and Engine"
LEGACY_MOIRE_FIND_MESSAGE = (
    "Legacy moire find controls are unsupported; use --max-length, --top-strain, "
    "and --bottom-strain."
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Readable help text with examples and defaults."""


class LegacyMoireFindAction(argparse.Action):
    """Reject retired find flags with one actionable migration message."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        parser.error(LEGACY_MOIRE_FIND_MESSAGE)


def parse_positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parse_nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def add_legacy_moire_find_flag(parser: argparse.ArgumentParser, *flags: str, takes_value: bool = True) -> None:
    parser.add_argument(
        *flags,
        action=LegacyMoireFindAction,
        nargs="?" if takes_value else 0,
        help=argparse.SUPPRESS,
    )


def parse_index_spec(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("please provide at least one index")
    return list(dict.fromkeys(values))


def parse_float_vector(raw: str) -> List[float]:
    values = [float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if len(values) not in {2, 3}:
        raise argparse.ArgumentTypeError("please provide 2 or 3 numeric values separated by commas")
    return values


def parse_int_matrix(raw: str) -> List[int]:
    values = [int(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("please provide exactly four integer values")
    return values


def parse_optional_float_list(raw: str | None) -> List[float] | None:
    if raw in {None, ""}:
        return None
    return [float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]


def parse_prestrain_modes(raw: str | None) -> List[str] | None:
    if raw in {None, ""}:
        return None
    return [token.strip().lower() for token in str(raw).split(",") if token.strip()]


def parse_string_list(raw: str | None) -> List[str] | None:
    if raw in {None, ""}:
        return None
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_angles_by_layer(raw: str | None) -> List[List[float] | None] | None:
    if raw in {None, ""}:
        return None
    groups = []
    for chunk in str(raw).split(";"):
        token = chunk.strip()
        if not token:
            groups.append(None)
        else:
            groups.append([float(item.strip()) for item in token.split(",") if item.strip()])
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellstine",
        formatter_class=HelpFormatter,
        description=(
            f"{APP_NAME} | {APP_EXPANSION}\n\n"
            "Five grouped workflows are available:\n"
            "  moire      Build commensurate moire supercells\n"
            "  adsorbate  Place and manipulate molecules on substrates\n"
            "  interface  Build surfaces and heterointerfaces from bulk or slab inputs\n"
            "  symmetry  Analyse space groups and reduce cells\n"
            "  defect     Analyse and generate vacancy, substitution, interstitial, and adatom defects"
        ),
    )
    parser.add_argument("--version", action="store_true", help="show package and optional dependency versions")
    groups = parser.add_subparsers(dest="group")

    moire = groups.add_parser(
        "moire",
        formatter_class=HelpFormatter,
        help="moire supercell construction",
        description="Moire supercell construction. Run `cellstine moire` with no stage to start the guided workflow.",
    )
    moire_sub = moire.add_subparsers(dest="stage")

    moire_find = moire_sub.add_parser(
        "find",
        formatter_class=HelpFormatter,
        help="search bilayer commensurate candidates",
        description="Search bilayer commensurate candidates and save the results table for later generation.",
    )
    moire_find.add_argument("top_poscar")
    moire_find.add_argument("bottom_poscar")
    moire_find.add_argument("--max-length", type=parse_positive_float, required=True, help="maximum in-plane supercell length in angstrom")
    moire_find.add_argument("--top-strain", type=parse_nonnegative_float, required=True, help="top-layer principal logarithmic strain budget as a fraction")
    moire_find.add_argument("--bottom-strain", type=parse_nonnegative_float, required=True, help="bottom-layer principal logarithmic strain budget as a fraction")
    moire_find.add_argument("--min-length", type=parse_positive_float, default=None, help="minimum in-plane supercell length in angstrom")
    moire_find.add_argument("--max-atoms", type=int, default=None, help="maximum atoms allowed in a candidate supercell")
    moire_find.add_argument("--max-cell-aspect-ratio", type=parse_positive_float, default=12.0, help="maximum in-plane supercell aspect ratio")
    moire_find.add_argument("--min-cell-angle", type=parse_positive_float, default=25.0, help="minimum in-plane supercell angle in degrees")
    moire_find.add_argument("--max-cell-angle", type=parse_positive_float, default=155.0, help="maximum in-plane supercell angle in degrees")
    moire_find.add_argument("--symmetric", action="store_true", help="request the restricted symmetry-preserving search branch")
    moire_find.add_argument("--progress", action="store_true", help="show live stage progress and elapsed timings while the search runs")
    moire_find.add_argument("--preview-limit", type=int, default=10, help="number of angle-sorted candidates to print after the search; use 0 to hide")
    for flag in (
        "--nindex", "--min-angle", "--max-angle", "--angle-step", "--angles",
        "--angle-length-tolerance", "--angle-strain-tolerance", "--angle-merge-tolerance",
        "--vector-tolerance", "--vector-strain-tolerance", "--candidate-tolerance",
        "--strain-tolerance", "--max-search-angles", "--matrix-values", "--matrix-layer",
        "--matrix-match-mode", "--workers", "--max-pair-matches", "--top-c-repeat",
        "--bottom-c-repeat", "--prestrain-top-mode", "--prestrain-top-value",
        "--prestrain-top-axis", "--prestrain-bottom-mode", "--prestrain-bottom-value",
        "--prestrain-bottom-axis",
    ):
        add_legacy_moire_find_flag(moire_find, flag)
    for flag in ("--fold-symmetry", "--allow-slivers", "--no-cull", "--no-reduce"):
        add_legacy_moire_find_flag(moire_find, flag, takes_value=False)

    moire_make = moire_sub.add_parser("make", formatter_class=HelpFormatter, help="generate bilayer supercells from saved results")
    moire_make.add_argument("results_file")
    moire_make.add_argument("--indexes", type=parse_index_spec, required=True, help="comma-separated indices or ranges, e.g. 1,3-5")
    moire_make.add_argument("--interlayer-distance", type=float, default=3.35, help="layer separation in angstrom")
    moire_make.add_argument("--workers", type=int, default=1)
    moire_make.add_argument("--output-dir", default=None)

    moire_translate = moire_sub.add_parser("translate", formatter_class=HelpFormatter, help="translate the upper layer in a stacked bilayer")
    moire_translate.add_argument("poscar_path")
    moire_translate.add_argument("--shift-cart", type=parse_float_vector, default=None, help="cartesian shift vector in angstrom")
    moire_translate.add_argument("--shift-direct", type=parse_float_vector, default=None, help="direct shift vector")

    moire_translaten = moire_sub.add_parser("translaten", formatter_class=HelpFormatter, help="translate the uppermost layer in a stacked N-layer structure")
    moire_translaten.add_argument("poscar_path")
    moire_translaten.add_argument("--shift-cart", type=parse_float_vector, default=None)
    moire_translaten.add_argument("--shift-direct", type=parse_float_vector, default=None)

    moire_visualize = moire_sub.add_parser(
        "visualize",
        formatter_class=HelpFormatter,
        help="plot moire search results",
        description=(
            "Plot moire search results. The default is a Matplotlib PNG summary with labelled axes, "
            "legends, strain-vs-angle, atom-count, ranking, and angle-distribution panels. "
            "Use --plotly when you specifically want the optional interactive 3D HTML viewer."
        ),
    )
    moire_visualize.add_argument("results_file")
    moire_visualize.add_argument("--indices", type=parse_index_spec, default=None)
    moire_visualize.add_argument("--interlayer", type=float, default=3.35)
    moire_visualize.add_argument("--output", default=None, help="output PNG path by default, or HTML path with --plotly")
    moire_visualize.add_argument("--plotly", action="store_true", help="write the optional interactive 3D HTML viewer instead of the default Matplotlib PNG")
    moire_visualize.add_argument("--show", action="store_true", help="also open the Matplotlib window after saving when a GUI backend is available")

    adsorbate = groups.add_parser(
        "adsorbate",
        formatter_class=HelpFormatter,
        help="molecule on substrate workflows",
        description="Molecule-on-substrate workflows. Run `cellstine adsorbate` with no stage to start the guided workflow.",
    )
    adsorbate_sub = adsorbate.add_subparsers(dest="stage")

    ads_place = adsorbate_sub.add_parser("place", formatter_class=HelpFormatter, help="place a molecule on a substrate site")
    ads_place.add_argument("substrate_poscar")
    ads_place.add_argument("molecule_poscar")
    ads_place.add_argument("--substrate-kind", choices=["bulk", "substrate", "patch", "surface", "slab"], default="substrate")
    ads_place.add_argument("--miller", default="1,1,1", help="surface Miller indices for bulk inputs")
    ads_place.add_argument("--layers", type=int, default=4)
    ads_place.add_argument("--vacuum", type=float, default=15.0)
    ads_place.add_argument("--substrate-repeat-a", type=int, default=1, help="repeat a bulk-derived substrate along surface a before placement")
    ads_place.add_argument("--substrate-repeat-b", type=int, default=1, help="repeat a bulk-derived substrate along surface b before placement")
    ads_place.add_argument("--substrate-supercell-matrix", type=parse_int_matrix, default=None, help="2x2 in-plane matrix for a bulk-derived substrate, e.g. 1,1,0,2")
    ads_place.add_argument("--auto-repeat-substrate", action="store_true", help="enlarge the selected substrate if the molecule cannot fit in one periodic image")
    ads_place.add_argument("--fit-padding", type=float, default=0.15, help="fractional in-plane padding used with --auto-repeat-substrate")
    ads_place.add_argument("--site-type", required=True, choices=["top", "bridge", "hcp", "hcp_hollow", "fcc", "fcc_hollow", "hollow", "fourfold_hollow"])
    ads_place.add_argument("--site-index", type=int, default=1)
    ads_place.add_argument("--height", type=float, default=2.5, help="height above the top layer in angstrom")
    ads_place.add_argument("--rotate", type=float, default=0.0, help="rotation about the c axis in degrees")
    ads_place.add_argument("--tilt", type=float, default=0.0, help="tilt/pitch angle in degrees")
    ads_place.add_argument("--roll", type=float, default=0.0, help="roll angle in degrees")

    ads_move = adsorbate_sub.add_parser("move", formatter_class=HelpFormatter, help="move a top-side molecule in a stacked structure")
    ads_move.add_argument("poscar_path")
    ads_move.add_argument("--target-cart", type=parse_float_vector, default=None)
    ads_move.add_argument("--target-direct", type=parse_float_vector, default=None)
    ads_move.add_argument("--rotate", type=float, default=0.0)
    ads_move.add_argument("--tilt", type=float, default=0.0)
    ads_move.add_argument("--roll", type=float, default=0.0)

    ads_assemble = adsorbate_sub.add_parser(
        "assemble",
        formatter_class=HelpFormatter,
        help="match a substrate cell to an experimental molecular assembly lattice",
        description=(
            "Advanced search mode: build a synthetic molecular lattice from a, b, and angle, then find "
            "substrate supercells that can support that periodic assembly. This does not place a molecule."
        ),
    )
    ads_assemble.add_argument("substrate_poscar")
    ads_assemble.add_argument("--a-length", type=float, required=True, help="target a length in angstrom")
    ads_assemble.add_argument("--b-length", type=float, default=None, help="target b length in angstrom; defaults to a")
    ads_assemble.add_argument("--angle", type=float, default=60.0, help="target in-plane angle in degrees")
    ads_assemble.add_argument("--nindex", type=int, default=12)
    ads_assemble.add_argument("--max-strain", type=float, default=0.05, help="maximum allowed strain as a fraction")
    ads_assemble.add_argument("--preview-limit", type=int, default=10, help="number of lowest-strain candidates to print after the search; use 0 to hide")

    ads_visualize = adsorbate_sub.add_parser(
        "visualize",
        formatter_class=HelpFormatter,
        help="plot a slab or adsorbate POSCAR",
        description=(
            "Plot a slab or adsorbate structure. The default is a Matplotlib multi-view PNG with x-y, x-z, y-z, "
            "and 3D overview panels. Use --plotly for the optional interactive 3D HTML viewer."
        ),
    )
    ads_visualize.add_argument("structure_path")
    ads_visualize.add_argument("--output", default=None, help="output PNG path by default, or HTML path with --plotly")
    ads_visualize.add_argument("--plotly", action="store_true", help="write the optional interactive 3D HTML viewer instead of the default Matplotlib PNG")
    ads_visualize.add_argument("--show", action="store_true", help="also open the Matplotlib window after saving when a GUI backend is available")

    interface = groups.add_parser(
        "interface",
        formatter_class=HelpFormatter,
        help="surface and heterointerface workflows",
        description="Surface and heterointerface workflows. Run `cellstine interface` with no stage to start the guided workflow.",
    )
    interface_sub = interface.add_subparsers(dest="stage")

    int_surface = interface_sub.add_parser("surface", formatter_class=HelpFormatter, help="build a slab from a bulk structure")
    int_surface.add_argument("bulk_poscar")
    int_surface.add_argument("--miller", required=True, help="Miller indices like 111, 001, 111x, 1,1,1, or 1,1,2x")
    int_surface.add_argument("--layers", type=int, default=4)
    int_surface.add_argument("--vacuum", type=float, default=15.0)
    int_surface.add_argument("--repeat-a", type=int, default=1)
    int_surface.add_argument("--repeat-b", type=int, default=1)
    int_surface.add_argument("--supercell-matrix", type=parse_int_matrix, default=None)
    int_surface.add_argument("--analyse-sites", action="store_true")

    int_sites = interface_sub.add_parser("sites", formatter_class=HelpFormatter, help="identify adsorption sites for a slab")
    int_sites.add_argument("slab_poscar")
    int_sites.add_argument("--surface-side", choices=["top", "bottom"], default="top")

    int_build = interface_sub.add_parser("build", formatter_class=HelpFormatter, help="build a heterointerface from slabs or bulks")
    int_build.add_argument("bottom_input")
    int_build.add_argument("top_input")
    int_build.add_argument("--bottom-kind", choices=["bulk", "surface", "slab"], default="surface")
    int_build.add_argument("--top-kind", choices=["bulk", "surface", "slab"], default="surface")
    int_build.add_argument("--bottom-miller", default="1,1,1")
    int_build.add_argument("--top-miller", default="1,1,1")
    int_build.add_argument("--bottom-layers", type=int, default=4)
    int_build.add_argument("--top-layers", type=int, default=4)
    int_build.add_argument("--bottom-vacuum", type=float, default=15.0)
    int_build.add_argument("--top-vacuum", type=float, default=15.0)
    int_build.add_argument("--gap", type=float, default=3.0)

    int_match = interface_sub.add_parser("match", formatter_class=HelpFormatter, help="scan bulk surfaces for interface matches")
    int_match.add_argument("bottom_bulk")
    int_match.add_argument("top_bulk")
    int_match.add_argument("--bottom-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--top-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--bottom-layers-list", nargs="*", type=int, default=None, help="candidate bottom slab layer counts")
    int_match.add_argument("--top-layers-list", nargs="*", type=int, default=None, help="candidate top slab layer counts")
    int_match.add_argument("--vacuum", type=float, default=15.0)
    int_match.add_argument("--max-strain", type=float, default=0.05)

    int_visualize = interface_sub.add_parser(
        "visualize",
        formatter_class=HelpFormatter,
        help="plot a slab or interface POSCAR",
        description=(
            "Plot a slab or interface structure. The default is a Matplotlib multi-view PNG with x-y, x-z, y-z, "
            "and 3D overview panels. Use --plotly for the optional interactive 3D HTML viewer."
        ),
    )
    int_visualize.add_argument("structure_path")
    int_visualize.add_argument("--output", default=None, help="output PNG path by default, or HTML path with --plotly")
    int_visualize.add_argument("--plotly", action="store_true", help="write the optional interactive 3D HTML viewer instead of the default Matplotlib PNG")
    int_visualize.add_argument("--show", action="store_true", help="also open the Matplotlib window after saving when a GUI backend is available")

    symmetry = groups.add_parser(
        "symmetry",
        formatter_class=HelpFormatter,
        help="symmetry analysis and cell reduction",
        description="Symmetry workflows using direct spglib when installed, with a native lattice-summary fallback.",
    )
    symmetry_sub = symmetry.add_subparsers(dest="stage")

    sym_analyse = symmetry_sub.add_parser(
        "analyse",
        formatter_class=HelpFormatter,
        help="analyse space group, operations, Wyckoff labels, and equivalent atoms",
    )
    sym_analyse.add_argument("structure")
    sym_analyse.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    sym_analyse.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_analyse.add_argument("--angle-tolerance", type=float, default=5.0, help="angle tolerance in degrees for symmetry finding")

    sym_reduce = symmetry_sub.add_parser(
        "reduce",
        formatter_class=HelpFormatter,
        help="write a primitive, conventional, or refined cell",
    )
    sym_reduce.add_argument("structure")
    sym_reduce.add_argument("--cell", choices=["primitive", "conventional", "refined"], default="primitive")
    sym_reduce.add_argument("--backend", choices=["auto", "spglib"], default="auto")
    sym_reduce.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_reduce.add_argument("--angle-tolerance", type=float, default=5.0, help="angle tolerance in degrees for symmetry finding")
    sym_reduce.add_argument("--output", default=None, help="optional output POSCAR path")

    sym_lattice_reduce = symmetry_sub.add_parser(
        "lattice-reduce",
        formatter_class=HelpFormatter,
        help="write a Niggli- or Delaunay-reduced lattice representation",
    )
    sym_lattice_reduce.add_argument("structure")
    sym_lattice_reduce.add_argument("--reduction", choices=["niggli", "delaunay"], default="niggli")
    sym_lattice_reduce.add_argument("--backend", choices=["auto", "spglib"], default="auto")
    sym_lattice_reduce.add_argument("--symprec", type=float, default=1e-5, help="lattice reduction tolerance")
    sym_lattice_reduce.add_argument("--output", default=None, help="optional output POSCAR path")

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
    defect_analyse.add_argument("--layer-tolerance", type=float, default=0.35, help="layer grouping tolerance in angstrom")
    defect_analyse.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for spglib symmetry finding")
    defect_analyse.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")

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
    defect_generate.add_argument("--generate", choices=["inequivalent"], default="inequivalent")
    defect_generate.add_argument("--height", type=float, default=2.5, help="adatom height above the detected surface site in angstrom")
    defect_generate.add_argument("--output-dir", default=None)
    defect_generate.add_argument("--structure-kind", choices=["auto", "bulk", "surface", "slab", "molecule-on-substrate"], default="auto")
    defect_generate.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    defect_generate.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    defect_generate.add_argument("--layer-tolerance", type=float, default=0.35, help="layer grouping tolerance in angstrom")
    defect_generate.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for spglib symmetry finding")
    defect_generate.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")

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
    defect_preview.add_argument("--layer-tolerance", type=float, default=0.35, help="layer grouping tolerance in angstrom")
    defect_preview.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for spglib symmetry finding")
    defect_preview.add_argument("--divacancy-distance", type=float, default=3.5, help="maximum cut-off distance for divacancy pairing in angstrom")

    return parser

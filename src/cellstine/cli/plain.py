"""Dependency-free parser for the simplified CELLSTINE CLI."""

from __future__ import annotations

import argparse

# Only the dependency-free constants module is imported here: building the
# parser must not drag in NumPy or a workflow package, so ``--help``, a
# mistyped flag and the interactive menu all start immediately.
from ..core.constants import (
    DEFAULT_CELL_LIMIT,
    DEFAULT_MATCH_LIMIT,
    DEFAULT_SYMMETRY_TOLERANCE,
    DIRECTION_HELP,
    LAYER_SELECTION_HELP,
    LAYER_TOLERANCE,
)
from .spec import APP_EXPANSION, APP_NAME
from .argtypes import (
    HelpFormatter,
    LEGACY_MOIRE_FIND_MESSAGE,
    LegacyMoireFindAction,
    add_legacy_moire_find_flag,
    parse_float_vector,
    parse_index_spec,
    parse_int_matrix,
    parse_mesh_shift,
    parse_nonnegative_float,
    parse_positive_float,
    parse_positive_int,
    parse_string_list,
    parse_supercell,
    parse_supercell_matrix,
)

__all__ = ["build_parser", "run"] + [
    'HelpFormatter',
    'LEGACY_MOIRE_FIND_MESSAGE',
    'LegacyMoireFindAction',
    'add_legacy_moire_find_flag',
    'parse_float_vector',
    'parse_index_spec',
    'parse_int_matrix',
    'parse_mesh_shift',
    'parse_nonnegative_float',
    'parse_positive_float',
    'parse_positive_int',
    'parse_string_list',
    'parse_supercell',
    'parse_supercell_matrix',
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellstine",
        formatter_class=HelpFormatter,
        description=(
            f"{APP_NAME} | {APP_EXPANSION}\n\n"
            "Six workflow groups plus the root structure viewer are available:\n"
            "  moire      Search, build, shift, and view moire structures\n"
            "  surface    Build slabs and analyse adsorption sites\n"
            "  interface  Match and build slab-on-slab interfaces\n"
            "  adsorbate  Place and manipulate molecules on substrates\n"
            "  defect     Analyse and generate point defects\n"
            "  symmetry   Analyse symmetry, cells, meshes, and k-paths\n"
            "  view       Draw a structure directly"
        ),
    )
    parser.add_argument("--version", action="store_true", help="show package and optional dependency versions")
    parser.add_argument("--plain", action="store_true", help="force dependency-free plain output")
    groups = parser.add_subparsers(dest="group")

    moire = groups.add_parser(
        "moire",
        formatter_class=HelpFormatter,
        help="moire supercell construction",
        description="Moire supercell construction. Run `cellstine moire` with no stage to start the guided workflow.",
    )
    moire_sub = moire.add_subparsers(dest="stage")

    moire_find = moire_sub.add_parser(
        "search",
        formatter_class=HelpFormatter,
        help="search bilayer commensurate candidates",
        description="Search bilayer commensurate candidates and save the results table for later generation.",
    )
    moire_find.add_argument("top_poscar")
    moire_find.add_argument("bottom_poscar")
    moire_find.add_argument("--length", dest="max_length", type=parse_positive_float, required=True, help="maximum in-plane supercell length in angstrom")
    moire_find.add_argument("--rigid", action="store_true", help="keep both layers rigid; equivalent to zero strain budgets")
    moire_find.add_argument("--strain", type=parse_nonnegative_float, default=None, help="same principal logarithmic strain budget for both layers")
    moire_find.add_argument("--top-strain", type=parse_nonnegative_float, default=None, help="top-layer strain budget for asymmetric searches")
    moire_find.add_argument("--bottom-strain", type=parse_nonnegative_float, default=None, help="bottom-layer strain budget for asymmetric searches")
    moire_find.add_argument("--min-length", type=parse_positive_float, default=None, help="minimum in-plane supercell length in angstrom")
    moire_find.add_argument("--atoms", dest="max_atoms", type=int, default=None, help="maximum atoms allowed in a candidate supercell")
    moire_find.add_argument("--max-cell-aspect-ratio", type=parse_positive_float, default=12.0, help="maximum in-plane supercell aspect ratio")
    moire_find.add_argument("--twist", default=None, help="twist-angle window in degrees, e.g. 9:14, :14, or 9:")
    moire_find.add_argument("--min-cell-angle", type=parse_positive_float, default=25.0, help="minimum in-plane supercell angle in degrees")
    moire_find.add_argument("--max-cell-angle", type=parse_positive_float, default=155.0, help="maximum in-plane supercell angle in degrees")
    moire_find.add_argument(
        "--symmetry-tolerance",
        type=parse_positive_float,
        default=DEFAULT_SYMMETRY_TOLERANCE,
        help=(
            "relative metric tolerance for detecting each layer point group; "
            "the layers are then idealised onto the metric their group preserves exactly"
        ),
    )
    moire_find.add_argument("--symmetric", action="store_true", help="request the restricted symmetry-preserving search branch")
    moire_find.add_argument(
        "--keep-layer-cells",
        action="store_true",
        help=(
            "search on the layer cells exactly as given instead of first folding each layer onto "
            "its own in-plane primitive cell; a supercell input then yields the coarser, larger "
            "moire candidates that cell allows"
        ),
    )
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

    moire_findn = moire_sub.add_parser(
        "stack-search",
        formatter_class=HelpFormatter,
        help="search commensurate cells for three or more layers",
        description=(
            "Search commensurate cells for a rigid base layer with one or more upper layers. "
            "Every upper layer is matched against the unstrained base and the shared cell of the "
            "stack is the exact integer intersection of the per-layer base supercells."
        ),
    )
    moire_findn.add_argument("base_poscar")
    moire_findn.add_argument("upper_poscars", nargs="+", help="one POSCAR per upper layer, bottom to top")
    moire_findn.add_argument("--length", dest="max_length", type=parse_positive_float, required=True, help="maximum in-plane supercell length in angstrom")
    moire_findn.add_argument(
        "--layer-strains",
        type=parse_float_vector,
        default=None,
        help="per-layer principal logarithmic strain budget; one value applies to every upper layer",
    )
    moire_findn.add_argument("--layer-strain", type=parse_nonnegative_float, default=0.02, help="strain budget used for every upper layer")
    moire_findn.add_argument("--min-length", type=parse_positive_float, default=None, help="minimum in-plane supercell length in angstrom")
    moire_findn.add_argument("--atoms", dest="max_atoms", type=int, default=2000, help="maximum atoms allowed in the whole stack")
    moire_findn.add_argument("--max-pair-atoms", type=int, default=None, help="maximum atoms allowed in each base-upper pair candidate")
    moire_findn.add_argument("--max-cell-aspect-ratio", type=parse_positive_float, default=12.0, help="maximum in-plane supercell aspect ratio")
    moire_findn.add_argument("--min-cell-angle", type=parse_positive_float, default=25.0, help="minimum in-plane supercell angle in degrees")
    moire_findn.add_argument("--max-cell-angle", type=parse_positive_float, default=155.0, help="maximum in-plane supercell angle in degrees")
    moire_findn.add_argument("--per-layer-limit", type=int, default=40, help="pair candidates kept per upper layer before combining")
    moire_findn.add_argument("--max-candidates", type=int, default=200, help="maximum multi-layer candidates to record")
    moire_findn.add_argument(
        "--keep-layer-cells",
        action="store_true",
        help=(
            "search on the layer cells exactly as given instead of first folding every layer, base "
            "included, onto its own in-plane primitive cell; a supercell input then yields the "
            "coarser, larger stacks that cell allows"
        ),
    )
    moire_findn.add_argument("--preview-limit", type=int, default=10, help="number of candidates to print after the search; use 0 to hide")

    moire_maken = moire_sub.add_parser(
        "stack-build",
        formatter_class=HelpFormatter,
        help="generate multi-layer supercells from saved stack-search results",
    )
    moire_maken.add_argument("results_file")
    moire_maken.add_argument("--indexes", "--indices", dest="indexes", type=parse_index_spec, required=True, help="comma-separated indices or ranges, e.g. 1,3-5")
    moire_maken.add_argument(
        "--interlayers",
        type=parse_float_vector,
        default=None,
        help="gap in angstrom above each layer, bottom to top; a single value applies to every gap",
    )
    moire_maken.add_argument("--interlayer-distance", type=float, default=3.35, help="gap used for every layer separation in angstrom")
    moire_maken.add_argument(
        "--vacuum",
        type=float,
        default=None,
        help="total vacuum in angstrom, split equally above and below the stack; "
        "the default keeps the longer input c vector",
    )
    moire_maken.add_argument("--output-dir", default=None)

    moire_make = moire_sub.add_parser("build", formatter_class=HelpFormatter, help="generate bilayer supercells from saved results")
    moire_make.add_argument("results_file")
    moire_make.add_argument("--indexes", "--indices", dest="indexes", type=parse_index_spec, required=True, help="comma-separated indices or ranges, e.g. 1,3-5")
    moire_make.add_argument("--interlayer-distance", type=float, default=3.35, help="layer separation in angstrom")
    moire_make.add_argument(
        "--vacuum",
        type=float,
        default=None,
        help="total vacuum in angstrom, split equally above and below the stack; "
        "the default keeps the longer input c vector",
    )
    moire_make.add_argument("--workers", type=int, default=1)
    moire_make.add_argument("--output-dir", default=None)

    moire_translate = moire_sub.add_parser("shift", formatter_class=HelpFormatter, help="translate the upper layer in a stacked bilayer")
    moire_translate.add_argument("poscar_path")
    moire_translate.add_argument("--shift-cart", type=parse_float_vector, default=None, help="cartesian shift vector in angstrom")
    moire_translate.add_argument("--shift-direct", type=parse_float_vector, default=None, help="direct shift vector")

    moire_visualize = moire_sub.add_parser(
        "view",
        formatter_class=HelpFormatter,
        help="plot moire search results",
        description=(
            "Plot moire search results. The default is a Matplotlib PNG summary with labelled axes, "
            "legends, strain-vs-angle, atom-count, ranking, and angle-distribution panels. "
            "Use --plotly when you specifically want the optional interactive 3D HTML viewer."
        ),
    )
    moire_visualize.add_argument("results_file")
    moire_visualize.add_argument("--indices", "--indexes", dest="indices", type=parse_index_spec, default=None, help="comma-separated indices or ranges, e.g. 1,3-5; the default plots every candidate")
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
    ads_place.add_argument(
        "--keep-cell-height",
        action="store_true",
        help="write the substrate cell unchanged instead of lengthening c to keep its vacuum gap",
    )

    ads_move = adsorbate_sub.add_parser("move", formatter_class=HelpFormatter, help="move a top-side molecule in a stacked structure")
    ads_move.add_argument("poscar_path")
    ads_move.add_argument("--target-cart", type=parse_float_vector, default=None)
    ads_move.add_argument("--target-direct", type=parse_float_vector, default=None)
    ads_move.add_argument("--rotate", type=float, default=0.0)
    ads_move.add_argument("--tilt", type=float, default=0.0)
    ads_move.add_argument("--roll", type=float, default=0.0)
    ads_move.add_argument(
        "--keep-cell-height",
        action="store_true",
        help="write the cell unchanged instead of lengthening c to keep its vacuum gap",
    )

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
    ads_assemble.add_argument("--max-length", type=parse_positive_float, required=True, help="maximum in-plane supercell length in angstrom")
    ads_assemble.add_argument("--top-strain", type=parse_nonnegative_float, required=True, help="target molecular-lattice principal logarithmic strain budget as a fraction")
    ads_assemble.add_argument("--bottom-strain", type=parse_nonnegative_float, required=True, help="substrate principal logarithmic strain budget as a fraction")
    ads_assemble.add_argument("--preview-limit", type=int, default=10, help="number of lowest-strain candidates to print after the search; use 0 to hide")

    ads_path = adsorbate_sub.add_parser(
        "path",
        formatter_class=HelpFormatter,
        help="build the chain of images between two structures of one cell",
        description=(
            "Write the evenly spaced chain of images a nudged-elastic-band run starts from -- a "
            "molecule or adatom diffusing from one site to the next, say. 00/POSCAR is the "
            "initial structure and the last folder the final one. The atoms of the two endpoints "
            "are paired by the assignment that makes the path shortest, and every atom travels to "
            "its nearest periodic image. Both endpoints must share one cell and one composition."
        ),
    )
    ads_path.add_argument("start_structure")
    ads_path.add_argument("end_structure")
    ads_path.add_argument(
        "--images",
        type=int,
        default=3,
        help="number of intermediate images; the chain holds two more than this",
    )
    ads_path.add_argument(
        "--no-match",
        dest="match",
        action="store_false",
        help="pair the atoms in file order instead of by the shortest-path assignment",
    )
    ads_path.add_argument("--output-dir", default=None, help="where to write the numbered image folders")

    surface = groups.add_parser(
        "surface",
        formatter_class=HelpFormatter,
        help="slab construction and surface-site analysis",
        description="Surface workflows. Run `cellstine surface` with no stage to start the guided workflow.",
    )
    surface_sub = surface.add_subparsers(dest="stage")

    surface_build = surface_sub.add_parser("build", formatter_class=HelpFormatter, help="build a slab from a bulk structure")
    surface_build.add_argument("bulk_poscar")
    surface_build.add_argument("--miller", required=True, help="Miller indices like 111, 001, 111x, 1,1,1, or 1,1,2x")
    surface_build.add_argument("--layers", type=int, default=4)
    surface_build.add_argument("--vacuum", type=float, default=15.0)
    surface_build.add_argument("--repeat-a", type=int, default=1)
    surface_build.add_argument("--repeat-b", type=int, default=1)
    surface_build.add_argument(
        "--min-length-a",
        type=float,
        default=None,
        help="repeat along the surface a axis until the cell is at least this long in angstrom",
    )
    surface_build.add_argument(
        "--min-length-b",
        type=float,
        default=None,
        help="repeat along the surface b axis until the cell is at least this long in angstrom",
    )
    surface_build.add_argument("--supercell-matrix", type=parse_int_matrix, default=None)
    surface_build.add_argument("--output-path", default=None, help="where to write the slab POSCAR")
    surface_build.add_argument("--analyse-sites", action="store_true")
    surface_build.add_argument("--sites-output-path", default=None, help="where to write the adsorption-site report")
    surface_build.add_argument(
        "--site-surface-side",
        choices=["top", "bottom"],
        default="top",
        help="which face of the new slab to analyse for adsorption sites",
    )

    surface_sites = surface_sub.add_parser("sites", formatter_class=HelpFormatter, help="identify adsorption sites for a slab")
    surface_sites.add_argument("slab_poscar")
    surface_sites.add_argument("--surface-side", choices=["top", "bottom"], default="top")
    surface_sites.add_argument("--output-path", default=None, help="where to write the adsorption-site report")

    interface = groups.add_parser(
        "interface",
        formatter_class=HelpFormatter,
        help="heterointerface workflows",
        description="Heterointerface workflows. Run `cellstine interface` with no stage to start the guided workflow.",
    )
    interface_sub = interface.add_subparsers(dest="stage")

    int_build = interface_sub.add_parser(
        "build",
        formatter_class=HelpFormatter,
        help="build a heterointerface from slabs or bulks",
        description=(
            "Stack two slabs. Without --match the two 1x1 cells are stacked directly and the top slab is "
            "forced onto the bottom cell, which is refused above --max-strain. With --match the interface "
            "is built on a commensurate supercell found by `interface match`, sharing a small strain "
            "between the two slabs."
        ),
    )
    int_build.add_argument("bottom_input", nargs="?", default=None)
    int_build.add_argument("top_input", nargs="?", default=None)
    int_build.add_argument("--bottom-kind", choices=["auto", "bulk", "surface", "slab"], default="auto", help="auto reads a cell with no vacuum as bulk and cuts a slab from it")
    int_build.add_argument("--top-kind", choices=["auto", "bulk", "surface", "slab"], default="auto", help="auto reads a cell with no vacuum as bulk and cuts a slab from it")
    int_build.add_argument("--bottom-miller", default="1,1,1")
    int_build.add_argument("--top-miller", default="1,1,1")
    int_build.add_argument("--bottom-layers", type=int, default=4)
    int_build.add_argument("--top-layers", type=int, default=4)
    int_build.add_argument("--bottom-vacuum", type=float, default=15.0)
    int_build.add_argument("--top-vacuum", type=float, default=15.0)
    int_build.add_argument("--gap", type=float, default=3.0)
    int_build.add_argument("--vacuum", type=float, default=None, help="vacuum thickness of the finished interface cell in angstrom")
    int_build.add_argument("--output-path", default=None, help="where to write the interface POSCAR")
    int_build.add_argument("--match", dest="match_json", default=None, help="matches.json from `interface match`; builds the commensurate supercell")
    int_build.add_argument("--match-index", type=int, default=1, help="one-based match index inside matches.json")
    int_build.add_argument(
        "--max-strain",
        type=parse_nonnegative_float,
        default=0.05,
        help="largest principal logarithmic strain accepted when stacking two 1x1 cells directly",
    )
    int_build.add_argument(
        "--bottom-stacking",
        choices=["keep", "mirror"],
        default="keep",
        help="keep the bottom slab as built, or mirror it to reverse its ABCABC stacking to CBACBA",
    )
    int_build.add_argument(
        "--top-stacking",
        choices=["keep", "mirror", "abc", "cba"],
        default="keep",
        help=(
            "stacking of the top slab relative to the bottom one: abc stacks it the same way, "
            "cba reverses it, mirror always reflects it, keep leaves it alone"
        ),
    )
    int_build.add_argument(
        "--registry",
        default=None,
        help=(
            "which layer meets which at the contact: a contact such as C-A, a kind such as "
            "eclipsed/fcc/hcp, or an index from `interface registries`"
        ),
    )
    int_build.add_argument(
        "--include-equivalent",
        action="store_true",
        help="number the registry options as `interface registries --include-equivalent` does",
    )

    int_registries = interface_sub.add_parser(
        "registries",
        formatter_class=HelpFormatter,
        help="list the distinct stacking sequences and contacts of two slabs",
        description=(
            "List the genuinely different ways two close-packed slabs can meet. The letters of a "
            "stacking sequence carry an origin gauge, so A-A, B-B and C-C are one contact, and "
            "reversing both slabs at once only reflects the whole interface, so twelve labelled "
            "combinations collapse to six distinct ones, and fewer still when the two slabs are "
            "interchangeable. Use --include-equivalent to see the removed ones as well."
        ),
    )
    int_registries.add_argument("bottom_input")
    int_registries.add_argument("top_input")
    int_registries.add_argument("--bottom-kind", choices=["auto", "bulk", "surface", "slab"], default="auto", help="auto reads a cell with no vacuum as bulk and cuts a slab from it")
    int_registries.add_argument("--top-kind", choices=["auto", "bulk", "surface", "slab"], default="auto", help="auto reads a cell with no vacuum as bulk and cuts a slab from it")
    int_registries.add_argument("--bottom-miller", default="1,1,1")
    int_registries.add_argument("--top-miller", default="1,1,1")
    int_registries.add_argument("--bottom-layers", type=int, default=4)
    int_registries.add_argument("--top-layers", type=int, default=4)
    int_registries.add_argument("--bottom-vacuum", type=float, default=15.0)
    int_registries.add_argument("--top-vacuum", type=float, default=15.0)
    int_registries.add_argument(
        "--include-equivalent",
        action="store_true",
        help=(
            "also list the options removed as duplicates, that is mirror images and interfaces "
            "that are the same boundary turned over"
        ),
    )
    int_registries.add_argument("--output-path", default=None, help="where to write registries.json")

    int_match = interface_sub.add_parser("match", formatter_class=HelpFormatter, help="scan bulk surfaces for interface matches")
    int_match.add_argument("bottom_bulk")
    int_match.add_argument("top_bulk")
    int_match.add_argument("--bottom-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--top-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--bottom-layers-list", nargs="*", type=int, default=None, help="candidate bottom slab layer counts")
    int_match.add_argument("--top-layers-list", nargs="*", type=int, default=None, help="candidate top slab layer counts")
    int_match.add_argument("--vacuum", type=float, default=15.0)
    int_match.add_argument(
        "--max-strain",
        type=parse_nonnegative_float,
        default=0.05,
        help="principal logarithmic strain budget for one slab, as a fraction",
    )
    int_match.add_argument(
        "--max-length",
        type=parse_positive_float,
        default=20.0,
        help="maximum in-plane length of the matched supercell in angstrom",
    )
    int_match.add_argument(
        "--strain-mode",
        choices=["shared", "film"],
        default="shared",
        help="share the strain between both slabs, or keep the bottom slab rigid and strain only the film",
    )
    int_match.add_argument("--min-length", type=parse_positive_float, default=None, help="minimum in-plane supercell length in angstrom")
    int_match.add_argument("--max-atoms", type=int, default=None, help="maximum atoms allowed in a matched interface cell")
    int_match.add_argument(
        "--max-matches",
        type=int,
        default=DEFAULT_MATCH_LIMIT,
        help="keep only this many best matches; use 0 to keep every match",
    )
    int_match.add_argument("--preview-limit", type=int, default=10, help="number of matches to print; use 0 to hide")
    int_match.add_argument("--output-path", default=None, help="where to write matches.json")

    symmetry = groups.add_parser(
        "symmetry",
        formatter_class=HelpFormatter,
        help="symmetry analysis and cell reduction",
        description="Symmetry workflows. The native engine computes symmetry operations, point groups, equivalent-atom orbits, primitive cells, and Niggli/Delaunay reductions; direct spglib adds space-group types and Wyckoff labels when installed.",
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
    sym_reduce.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
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
    sym_lattice_reduce.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    sym_lattice_reduce.add_argument("--symprec", type=float, default=1e-5, help="lattice reduction tolerance")
    sym_lattice_reduce.add_argument("--output", default=None, help="optional output POSCAR path")

    sym_kpoints = symmetry_sub.add_parser(
        "kpoints",
        formatter_class=HelpFormatter,
        help="write a symmetry-reduced Brillouin-zone sampling mesh",
        description=(
            "Write a KPOINTS file for a structure. The mesh follows either a largest allowed "
            "reciprocal-space step (--spacing, the quantity VASP calls KSPACING, in 1/angstrom "
            "with the 2 pi convention) or explicit divisions, and is reduced by the rotations of "
            "the space group of the cell together with time reversal. The weights written are "
            "exact orbit sizes and add up to the size of the unreduced mesh."
        ),
    )
    sym_kpoints.add_argument("structure")
    sym_kpoints.add_argument(
        "--spacing",
        type=parse_positive_float,
        default=None,
        help="largest allowed step between sampled wavevectors, in 1/angstrom",
    )
    sym_kpoints.add_argument(
        "--divisions",
        type=parse_supercell,
        default=None,
        help="explicit mesh divisions as 'n1,n2,n3' instead of a spacing",
    )
    sym_kpoints.add_argument(
        "--mesh",
        choices=["gamma", "monkhorst"],
        default="gamma",
        help="Gamma-centred mesh, or the Monkhorst-Pack half-step offset on even axes",
    )
    sym_kpoints.add_argument(
        "--shift",
        type=parse_mesh_shift,
        default=None,
        help="explicit mesh offset in grid steps as 's1,s2,s3', overriding --mesh",
    )
    sym_kpoints.add_argument(
        "--surface",
        action="store_true",
        help="sample the surface normal with a single point, as a slab with vacuum needs",
    )
    sym_kpoints.add_argument("--no-symmetry", action="store_true", help="reduce by time reversal only")
    sym_kpoints.add_argument("--no-time-reversal", action="store_true", help="do not identify k with -k")
    sym_kpoints.add_argument("--list-points", action="store_true", help="always write the explicit irreducible list with weights")
    sym_kpoints.add_argument("--automatic", action="store_true", help="always write the automatic mesh line instead of the list")
    sym_kpoints.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_kpoints.add_argument("--output", default=None, help="optional output KPOINTS path")

    sym_kpath = symmetry_sub.add_parser(
        "kpath",
        formatter_class=HelpFormatter,
        help="write a band-structure path through the Brillouin zone",
        description=(
            "Write the line-mode KPOINTS file of a band structure. The high-symmetry points are "
            "derived from the symmetry of the cell -- they are the points and the ends of the "
            "symmetry lines of the Brillouin zone, not a table look-up -- and are named after the "
            "conventional cell of the Bravais lattice. The walk is the conventional one for the "
            "Bravais types that have one and is otherwise derived from the symmetry lines; --path "
            "chooses it explicitly, as 'GAMMA-X-W|K-L'."
        ),
    )
    sym_kpath.add_argument("structure")
    sym_kpath.add_argument(
        "--spacing",
        type=parse_positive_float,
        default=None,
        help="largest allowed step along the path, in 1/angstrom (default 0.03)",
    )
    sym_kpath.add_argument(
        "--divisions",
        type=parse_positive_int,
        default=None,
        help="explicit number of points per segment instead of a spacing",
    )
    sym_kpath.add_argument(
        "--path",
        default=None,
        help="explicit walk, such as 'GAMMA-X-W|K-L'; '|' breaks the line",
    )
    sym_kpath.add_argument(
        "--derived-path",
        action="store_true",
        help="always derive the walk from the symmetry lines, never use the conventional one",
    )
    sym_kpath.add_argument("--no-symmetry", action="store_true", help="use the point group of the lattice, ignoring the atoms")
    sym_kpath.add_argument("--no-time-reversal", action="store_true", help="do not identify k with -k")
    sym_kpath.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_kpath.add_argument("--output", default=None, help="optional output KPOINTS path")

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

    view = groups.add_parser(
        "view",
        formatter_class=HelpFormatter,
        help="plot any structure POSCAR",
        description=(
            "Plot a structure. The default is a Matplotlib multi-view PNG; with --view-direction "
            "the plan view is the picture an observer looking along that direction would see."
        ),
    )
    view.add_argument("structure_path")
    view.add_argument("--output", default=None, help="output PNG path by default, or HTML path with --plotly")
    view.add_argument("--plotly", action="store_true", help="write the optional interactive 3D HTML viewer instead of the default Matplotlib PNG")
    view.add_argument("--show", action="store_true", help="also open the Matplotlib window after saving when a GUI backend is available")
    view.add_argument("--view-direction", default=None, help=DIRECTION_HELP)
    view.set_defaults(stage="structure")

    return parser


def run(argv: list[str] | None = None) -> int:
    """Run the dependency-free CLI frontend."""

    from .main import dispatch_namespace

    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    try:
        return dispatch_namespace(arguments)
    except ValueError as exc:
        parser.error(str(exc))
    return 0

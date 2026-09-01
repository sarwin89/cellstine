"""Dependency-free parser group for moire commands."""

from __future__ import annotations

import argparse

from ..core.constants import DEFAULT_SYMMETRY_TOLERANCE
from .argtypes import (
    HelpFormatter,
    add_legacy_moire_find_flag,
    parse_float_vector,
    parse_index_spec,
    parse_nonnegative_float,
    parse_positive_float,
)


def add_moire_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``moire`` command group."""

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
        help="experimental search for commensurate cells across three or more layers",
        description=(
            "Experimental N-layer search: search commensurate cells for a rigid base layer "
            "with one or more upper layers. "
            "Every upper layer is matched against the unstrained base and the shared cell of the "
            "stack is the exact integer intersection of the per-layer base supercells. "
            "The public JSON contract is still being stabilized."
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
        help="experimental build of multi-layer supercells from saved stack-search results",
        description=(
            "Experimental N-layer builder. It consumes stack-search results; the public JSON "
            "contract is still being stabilized."
        ),
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

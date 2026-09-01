"""Dependency-free parser group for interface commands."""

from __future__ import annotations

import argparse

from ..core.constants import DEFAULT_MATCH_LIMIT
from .argtypes import HelpFormatter, parse_nonnegative_float, parse_positive_float


def add_interface_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``interface`` command group."""

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
        "--length",
        dest="max_length",
        type=parse_positive_float,
        default=20.0,
        help="maximum in-plane length of the matched supercell in angstrom",
    )
    int_match.add_argument(
        "--max-length",
        dest="max_length",
        type=parse_positive_float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    int_match.add_argument(
        "--strain-mode",
        choices=["shared", "film"],
        default="shared",
        help="share the strain between both slabs, or keep the bottom slab rigid and strain only the film",
    )
    int_match.add_argument("--min-length", type=parse_positive_float, default=None, help="minimum in-plane supercell length in angstrom")
    int_match.add_argument("--atoms", dest="max_atoms", type=int, default=None, help="maximum atoms allowed in a matched interface cell")
    int_match.add_argument("--max-atoms", dest="max_atoms", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    int_match.add_argument(
        "--max-matches",
        type=int,
        default=DEFAULT_MATCH_LIMIT,
        help="keep only this many best matches; use 0 to keep every match",
    )
    int_match.add_argument("--preview-limit", type=int, default=10, help="number of matches to print; use 0 to hide")
    int_match.add_argument("--output-path", default=None, help="where to write matches.json")

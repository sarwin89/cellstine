"""Dependency-free parser group for surface commands."""

from __future__ import annotations

import argparse

from .argtypes import HelpFormatter, parse_int_matrix


def add_surface_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``surface`` command group."""

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

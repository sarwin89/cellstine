"""Dependency-free parser for the simplified CELLSTINE CLI."""

from __future__ import annotations

import argparse

# Only the dependency-free constants module is imported here: building the
# parser must not drag in NumPy or a workflow package, so ``--help``, a
# mistyped flag and the interactive menu all start immediately.
from .spec import APP_EXPANSION, APP_NAME
from .plain_adsorbate import add_adsorbate_group
from .plain_defect import add_defect_group
from .plain_interface import add_interface_group
from .plain_moire import add_moire_group
from .plain_surface import add_surface_group
from .plain_symmetry import add_symmetry_group
from .plain_view import add_view_group
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

    add_moire_group(groups)
    add_adsorbate_group(groups)
    add_surface_group(groups)
    add_interface_group(groups)

    add_symmetry_group(groups)

    add_defect_group(groups)

    add_view_group(groups)

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

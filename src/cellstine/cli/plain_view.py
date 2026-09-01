"""Dependency-free parser group for the root structure viewer."""

from __future__ import annotations

import argparse

from ..core.constants import DIRECTION_HELP
from .argtypes import HelpFormatter


def add_view_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free root ``view`` command."""

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

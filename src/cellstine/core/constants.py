"""Defaults and help strings shared by the CLI and the workflows.

This module deliberately imports nothing.  The argument parser needs a handful
of constants that otherwise live next to the code that uses them --- the match
limit of the interface search, the supercell limit of the defect stage, the
default symmetry tolerance, the two long help strings --- and importing those
modules to read a number pulled in NumPy and the whole interface workflow before
``cellstine --help`` could print a line.  The constants live here instead, and
each of those modules re-exports the one it owns, so every existing reference
(``cellstine.interface.workflow.interface.DEFAULT_MATCH_LIMIT`` and the rest)
keeps working.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CELL_LIMIT",
    "DEFAULT_MATCH_LIMIT",
    "DEFAULT_SYMMETRY_TOLERANCE",
    "DIRECTION_HELP",
    "LAYER_TOLERANCE",
    "LAYER_SELECTION_HELP",
]


#: The tolerance the planar symmetry search uses to decide whether two sites
#: coincide, in fractional coordinates.
DEFAULT_SYMMETRY_TOLERANCE = 1e-5

#: Height difference, in angstrom, below which two atoms count as one atomic
#: plane.  Every stage that cuts a structure into layers defaults to this, so a
#: slab, its terminations, its stacking word and its defect layer census all see
#: the same planes; ``core.layers.layer_partition`` is the rule they apply it
#: with.
LAYER_TOLERANCE = 0.35

#: The largest number of host cells a defect supercell search will consider.
DEFAULT_CELL_LIMIT = 64

#: The number of lattice matches the interface search keeps.
DEFAULT_MATCH_LIMIT = 200

DIRECTION_HELP = (
    "direction of observation: auto (the a-b surface normal), a/b/c, a*/b*/c*, "
    "x/y/z, a Miller plane normal such as 111, (1,1,1) or 1x1, a real-space "
    "direction such as [1,1,0], or cart:x,y,z; prefix with '-' to look the "
    "other way"
)

LAYER_SELECTION_HELP = (
    "atomic planes to place the defect in, numbered from 1 at the bottom along "
    "the direction of observation: all, top, bottom, surface (both outermost), "
    "interior (all but the outermost), middle, or an explicit list such as "
    "1,3 or 2-4; negative numbers count from the top, so -1 is the topmost plane"
)

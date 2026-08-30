"""Interactive command builders for the structure visualizer.

Every group that owns structures -- moire, adsorbate, interface, defect and
symmetry -- offers the same picture of one file, so the questions are asked in
one place here.  The one choice that changes the picture rather than the file
it is written to is the direction of observation: it decides which way the
structure is turned before the plan view is drawn, and therefore which planes
the reader sees edge-on.
"""

from __future__ import annotations

from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    _choice,
    _print_title,
    _prompt,
    _prompt_path,
    _prompt_yes_no,
)

__all__ = ["_build_structure_visualize", "_prompt_view_direction"]


def _prompt_view_direction() -> str:
    """Ask which direction the structure should be read along.

    The direction of observation fixes what a *layer* is: the atomic planes are
    the groups of atoms at the same height along it, numbered from 1 at the
    bottom, so it decides both which defects the plane selection offers and how
    they are labelled.  For a picture it decides which way the structure is
    turned before it is drawn.
    """

    choice = _choice(
        "Which direction should the structure be observed along?",
        [
            {
                "key": "auto",
                "label": "The a-b surface normal",
                "hint": "Recommended. The stacking direction of a slab, and the c direction of a cell written by this tool.",
            },
            {"key": "miller", "label": "A lattice-plane normal (h k l)", "hint": "Read a bulk cell as a stack of (h k l) planes, e.g. 111."},
            {"key": "uvw", "label": "A lattice direction [u v w]", "hint": "Look along a crystal translation, e.g. 110."},
            {"key": "axis", "label": "A cell or Cartesian axis", "hint": "a, b, c, a*, b*, c*, x, y or z."},
        ],
        default=1,
    )
    if choice == "auto":
        return "auto"
    if choice == "miller":
        indices = _prompt("Miller indices of the planes, e.g. 111 or 1,1,-2", "111")
        return f"({indices})"
    if choice == "uvw":
        indices = _prompt("Direction indices, e.g. 110 or 1,1,-2", "110")
        return f"[{indices}]"
    return _prompt("Axis (a, b, c, a*, b*, c*, x, y or z)", "c")


def _build_structure_visualize(group: str, *, subject: str) -> list[str]:
    """Build one root ``view`` command line."""

    _print_title(
        "Structure Visualization",
        f"Create a labelled Matplotlib multi-view plot of {subject}.",
    )
    structure = _prompt_path(
        "Choose the structure file",
        patterns=("*.vasp", "POSCAR", "CONTCAR"),
        roots=(OUTPUT_DIR, INPUT_DIR, RUNS_DIR),
    )
    argv = ["view", structure]
    if _prompt_yes_no("Use the optional interactive Plotly 3D HTML viewer instead?", False):
        argv.append("--plotly")
    if _prompt_yes_no("Do you want to look along a particular direction?", False):
        argv.extend(["--view-direction", _prompt_view_direction()])
    return argv

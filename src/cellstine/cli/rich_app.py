"""Optional Typer/Rich frontend for the simplified CELLSTINE CLI."""

from __future__ import annotations

import sys


_GUIDED_GROUPS = {"moire", "surface", "adsorbate", "interface", "defect", "symmetry"}


def _guided_group(argv: list[str]) -> str | None:
    if not argv:
        return None
    if len(argv) == 1 and argv[0] in _GUIDED_GROUPS:
        return argv[0]
    return None


def run(argv: list[str] | None = None) -> int:
    """Run the optional Rich frontend over the shared command spec.

    Typer is part of the optional ``cli`` extra and is imported here so missing
    optional dependencies still trigger the intended plain fallback. Direct
    command execution remains owned by ``plain.run``; guided mode uses Rich only
    for presentation and prompts while reusing the same workflow builders.
    """

    try:
        import typer as _typer  # noqa: F401
        from rich.console import Console  # noqa: F401
        from rich.panel import Panel  # noqa: F401
        from rich.prompt import Confirm, Prompt  # noqa: F401
        from rich.table import Table  # noqa: F401
    except ImportError:
        raise

    forwarded = list(sys.argv[1:] if argv is None else argv)
    if not forwarded:
        from .interactive.prompts import RichGuidedUI
        from .interactive.runner import run_interactive

        return int(run_interactive(ui=RichGuidedUI(), show_banner=True))
    group = _guided_group(forwarded)
    if group is not None:
        from .interactive.prompts import RichGuidedUI
        from .interactive.runner import run_interactive

        return int(run_interactive(group=group, ui=RichGuidedUI(), show_banner=True))

    from .plain import run as run_plain

    return int(run_plain(forwarded))

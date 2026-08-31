"""Optional Typer/Rich frontend for the simplified CELLSTINE CLI."""

from __future__ import annotations

import os
import sys


def run(argv: list[str] | None = None) -> int:
    """Run the optional Rich frontend, delegating execution to the shared plain spec.

    Typer is part of the optional ``cli`` extra and is imported here so missing
    optional dependencies still trigger the intended plain fallback. The command
    grammar itself remains owned by ``plain.run``; keeping Typer out of the
    runtime dispatch path avoids a second parser and prevents Typer callback
    state from leaking as command failures.
    """

    try:
        import typer as _typer  # noqa: F401
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        raise

    from .plain import run as run_plain
    from .spec import APP_EXPANSION, APP_NAME

    forwarded = list(sys.argv[1:] if argv is None else argv)
    console = Console(no_color=bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty())
    if not forwarded:
        console.print(Panel.fit("Starting guided CELLSTINE workflow", title=f"{APP_NAME}: {APP_EXPANSION}"))
    return int(run_plain(forwarded))

"""Optional Typer/Rich frontend for the simplified CELLSTINE CLI."""

from __future__ import annotations

import os
import sys


def run(argv: list[str] | None = None) -> int:
    """Run the Typer/Rich frontend, delegating execution to the shared plain spec."""

    try:
        import typer
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        raise

    from .plain import run as run_plain
    from .spec import APP_EXPANSION, APP_NAME

    app = typer.Typer(
        add_completion=False,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        help=f"{APP_NAME}: {APP_EXPANSION}",
    )

    @app.callback(invoke_without_command=True)
    def main(ctx: typer.Context, version: bool = False) -> None:
        console = Console(no_color=bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty())
        forwarded = list(ctx.args)
        if version:
            forwarded.insert(0, "--version")
        if not forwarded:
            console.print(Panel.fit("Starting guided CELLSTINE workflow", title=APP_NAME))
        raise typer.Exit(run_plain(forwarded))

    # Typer owns presentation here; the plain frontend still owns the command
    # grammar so base installs and rich installs cannot drift apart.
    app(args=argv, standalone_mode=False)
    return 0

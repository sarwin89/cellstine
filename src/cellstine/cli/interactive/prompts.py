"""Prompt primitives and screen helpers for the interactive launcher."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from ...core.previews import format_adsorption_sites, preview_moire_results_file
from ...interface.surface import backend as surface_backend


INPUT_DIR = Path("input")
RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("output")

MAIN_MENU_BANNER = r"""
 ██████╗███████╗██╗     ██╗     ███████╗████████╗██╗███╗   ██╗███████╗
██╔════╝██╔════╝██║     ██║     ██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝
██║     █████╗  ██║     ██║     ███████╗   ██║   ██║██╔██╗ ██║█████╗
██║     ██╔══╝  ██║     ██║     ╚════██║   ██║   ██║██║╚██╗██║██╔══╝
╚██████╗███████╗███████╗███████╗███████║   ██║   ██║██║ ╚████║███████╗
 ╚═════╝╚══════╝╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝
""".strip("\n")

ASCII_MAIN_MENU_BANNER = r"""
  CCCCC  EEEEEEE L       L       SSSSSSS TTTTTTT III N   N EEEEEEE
 C       E       L       L       S          T     I  NN  N E
 C       EEEEE   L       L       SSSSSSS    T     I  N N N EEEEE
 C       E       L       L             S    T     I  N  NN E
  CCCCC  EEEEEEE LLLLLLL LLLLLLL SSSSSSS    T    III N   N EEEEEEE
""".strip("\n")


def _stream_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        (MAIN_MENU_BANNER + "╭─╮│╰─╯").encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


class _QuitInteractive(Exception):
    """Internal signal for a graceful interactive-mode exit."""


class _BackInteractive(Exception):
    """Internal signal for returning to the previous interactive menu."""


class PlainGuidedUI:
    """Dependency-free guided-mode presentation and prompt backend."""

    def print(self, *args, **kwargs) -> None:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_args = [
                str(arg).encode(encoding, errors="replace").decode(encoding, errors="replace")
                for arg in args
            ]
            print(*safe_args, **kwargs)

    def title(self, title: str, subtitle: str | None = None) -> None:
        self.print()
        self.print(title)
        self.print("-" * len(title))
        if subtitle:
            self.print(subtitle)

    def banner(self) -> None:
        self.print()
        self.print(MAIN_MENU_BANNER)
        self.print()
        self.print("Made by Sarwin Chandran 2026")

    def prompt(
        self,
        prompt: str,
        default: str | None = None,
        *,
        allow_empty: bool = False,
        allow_back: bool = True,
    ) -> str:
        shown = f" [{default}]" if default not in {None, ""} else ""
        while True:
            answer = input(f"{prompt}{shown}: ").strip()
            if answer:
                if allow_back and answer.lower() in {"b", "back"}:
                    raise _BackInteractive()
                return answer
            if default is not None:
                return default
            if allow_empty:
                return ""
            self.print("Please enter a value.")

    def confirm(self, prompt: str, default_yes: bool = True) -> bool:
        default = "y" if default_yes else "n"
        while True:
            answer = self.prompt(prompt, default).strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.print("Please answer with y or n.")

    def choice(self, title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
        self.title(title)
        for index, option in enumerate(options, start=1):
            self.print(f"{index}. {option['label']}")
            if option.get("hint"):
                self.print(f"   {option['hint']}")
        if allow_back:
            self.print("b. Back")
        self.print("q. Quit interactive mode")
        while True:
            answer = self.prompt("Choose an option", str(default), allow_back=allow_back).strip().lower()
            if answer in {"q", "quit", "exit"}:
                raise _QuitInteractive()
            if answer.isdigit():
                index = int(answer)
                if 1 <= index <= len(options):
                    option = options[index - 1]
                    return str(option.get("value", option["key"]))
            for option in options:
                if answer == str(option["key"]).lower():
                    return str(option.get("value", option["key"]))
            self.print("Please choose one of the numbered options.")

    def command_preview(self, title: str, argv: Sequence[str]) -> None:
        self.print()
        self.print(title)
        self.print(_format_command(argv))


class RichGuidedUI(PlainGuidedUI):
    """Rich-backed guided-mode presentation.

    Imports are intentionally local so base installs can import the CLI without
    Typer/Rich. The command builders stay shared with plain mode.
    """

    def __init__(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm, Prompt
        from rich.table import Table

        try:
            from rich import box
        except ImportError:
            box = None

        self._unicode = _stream_supports_unicode()
        self.console = Console(
            legacy_windows=False,
            no_color=bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty(),
        )
        self._box = None if box is None else (box.ROUNDED if self._unicode else box.ASCII)
        self._panel = Panel
        self._prompt = Prompt
        self._confirm = Confirm
        self._table = Table

    def _box_kwargs(self) -> dict[str, object]:
        if self._box is None:
            return {}
        return {"box": self._box}

    def print(self, *args, **kwargs) -> None:
        try:
            self.console.print(*args, **kwargs)
        except UnicodeEncodeError:
            PlainGuidedUI.print(self, *args, **kwargs)

    def title(self, title: str, subtitle: str | None = None) -> None:
        body = subtitle or "Choose the workflow settings below."
        self.print()
        self.print(self._panel.fit(body, title=title, **self._box_kwargs()))

    def banner(self) -> None:
        banner = MAIN_MENU_BANNER if self._unicode else ASCII_MAIN_MENU_BANNER
        self.print()
        self.print(self._panel.fit(f"{banner}\n\nMade by Sarwin Chandran 2026", title="CELLSTINE", **self._box_kwargs()))

    def prompt(
        self,
        prompt: str,
        default: str | None = None,
        *,
        allow_empty: bool = False,
        allow_back: bool = True,
    ) -> str:
        while True:
            answer = str(self._prompt.ask(prompt, default=default)).strip()
            if answer:
                if allow_back and answer.lower() in {"b", "back"}:
                    raise _BackInteractive()
                return answer
            if default is not None:
                return default
            if allow_empty:
                return ""
            self.print("Please enter a value.")

    def confirm(self, prompt: str, default_yes: bool = True) -> bool:
        return bool(self._confirm.ask(prompt, default=default_yes))

    def choice(self, title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
        self.title(title, "Select an option. Use q to quit; use b to go back where available.")
        table = self._table(show_header=True, header_style="bold", **self._box_kwargs())
        table.add_column("#", justify="right")
        table.add_column("Option")
        table.add_column("When to use it")
        for index, option in enumerate(options, start=1):
            table.add_row(str(index), option["label"], option.get("hint", ""))
        if allow_back:
            table.add_row("b", "Back", "Return to the previous menu.")
        table.add_row("q", "Quit", "Close guided mode.")
        self.print(table)
        while True:
            answer = self.prompt("Choose an option", str(default), allow_back=allow_back).strip().lower()
            if answer in {"q", "quit", "exit"}:
                raise _QuitInteractive()
            if answer.isdigit():
                index = int(answer)
                if 1 <= index <= len(options):
                    option = options[index - 1]
                    return str(option.get("value", option["key"]))
            for option in options:
                if answer == str(option["key"]).lower():
                    return str(option.get("value", option["key"]))
            self.print("Please choose one of the numbered options.")

    def command_preview(self, title: str, argv: Sequence[str]) -> None:
        self.print()
        self.print(self._panel.fit(_format_command(argv), title=title, **self._box_kwargs()))


_ACTIVE_UI: PlainGuidedUI = PlainGuidedUI()


def get_guided_ui() -> PlainGuidedUI:
    return _ACTIVE_UI


@contextmanager
def use_guided_ui(ui: PlainGuidedUI | None = None) -> Iterator[None]:
    global _ACTIVE_UI
    previous = _ACTIVE_UI
    if ui is not None:
        _ACTIVE_UI = ui
    try:
        yield
    finally:
        _ACTIVE_UI = previous


def _print_title(title: str, subtitle: str | None = None) -> None:
    get_guided_ui().title(title, subtitle)


def _print_main_menu_banner() -> None:
    get_guided_ui().banner()


def _prompt(
    prompt: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
    allow_back: bool = True,
) -> str:
    return get_guided_ui().prompt(prompt, default, allow_empty=allow_empty, allow_back=allow_back)


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        try:
            return int(_prompt(prompt, str(default)))
        except ValueError:
            get_guided_ui().print("Please enter a whole number.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(_prompt(prompt, str(default)))
        except ValueError:
            get_guided_ui().print("Please enter a number.")


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    return get_guided_ui().confirm(prompt, default_yes=default_yes)


def _choice(title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
    return get_guided_ui().choice(title, options, default=default, allow_back=allow_back)


def _relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _find_candidates(patterns: Sequence[str], roots: Sequence[Path], *, limit: int = 8) -> list[Path]:
    found: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for root_index, root in enumerate(roots):
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append((root_index, resolved))
    found.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
    return [path for _, path in found[:limit]]


def _prompt_path(
    label: str,
    *,
    patterns: Sequence[str],
    roots: Sequence[Path],
    default: str | None = None,
    allow_manual: bool = True,
) -> str:
    suggestions = _find_candidates(patterns, roots)
    ui = get_guided_ui()
    ui.print()
    ui.print(label)
    if roots:
        ui.print("Search order: " + " -> ".join(str(root) for root in roots))
    if suggestions:
        ui.print("Recent matches:")
        for index, path in enumerate(suggestions, start=1):
            ui.print(f"  {index}. {_relative_display(path)}")
        if allow_manual:
            ui.print("  m. Type a different path")
        ui.print("  b. Back")
        ui.print("  q. Quit interactive mode")
        default_value = "1"
    else:
        ui.print("No suggested files were found, so please type a path.")
        ui.print("Type b to go back or q to quit.")
        default_value = default
    while True:
        answer = _prompt("Selection", default_value, allow_empty=default is not None).strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise _QuitInteractive()
        if suggestions and answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(suggestions):
                return str(suggestions[index - 1])
        if allow_manual and answer.lower() in {"m", "manual"}:
            manual_path = _prompt("Path").strip()
            if manual_path.lower() in {"q", "quit", "exit"}:
                raise _QuitInteractive()
            return manual_path
        if answer:
            return answer
        if default is not None:
            return default
        ui.print("Please choose a suggested file or type a path.")


def _prompt_csv(prompt: str, default: str) -> str:
    return _prompt(prompt, default)


def _prompt_int_range(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        value = _prompt_int(prompt, default)
        if int(minimum) <= value <= int(maximum):
            return value
        get_guided_ui().print(f"Please enter a value from {int(minimum)} to {int(maximum)}.")


def _parse_matrix_entries(text: str) -> list[int]:
    values = [int(token.strip()) for token in str(text).replace(";", ",").split(",") if token.strip()]
    if len(values) != 4:
        raise ValueError("a 2x2 matrix needs exactly four entries")
    return values


_SITE_LABELS = {
    "top": "Top",
    "bridge": "Bridge",
    "fcc_hollow": "fcc hollow",
    "hcp_hollow": "hcp hollow",
    "hollow": "Generic hollow",
    "fourfold_hollow": "Fourfold hollow",
}


_SITE_HINTS = {
    "top": "Above an outermost surface atom.",
    "bridge": "Above a nearest-neighbour midpoint.",
    "fcc_hollow": "Close-packed hollow with fcc registry.",
    "hcp_hollow": "Close-packed hollow with hcp registry.",
    "hollow": "Triangular hollow where fcc/hcp registry could not be assigned.",
    "fourfold_hollow": "Square-like fourfold hollow.",
}


def _site_options_from_report(site_report) -> list[dict[str, str]]:
    options = []
    for key in ("top", "bridge", "fcc_hollow", "hcp_hollow", "hollow", "fourfold_hollow"):
        count = int(site_report.site_counts.get(key, 0))
        if count <= 0:
            continue
        options.append(
            {
                "key": key,
                "label": f"{_SITE_LABELS.get(key, key)} ({count} found)",
                "hint": _SITE_HINTS.get(key, "Detected in this cell."),
            }
        )
    return options


def _print_detected_sites(site_report) -> None:
    ui = get_guided_ui()
    ui.print()
    ui.print("Detected adsorption sites in the selected substrate:")
    if not site_report.site_counts:
        ui.print("  none")
        return
    for key in sorted(site_report.site_counts):
        ui.print(f"  {_SITE_LABELS.get(key, key)}: {int(site_report.site_counts[key])}")


def _print_saved_moire_preview(results_file: str, limit: int = 15) -> None:
    try:
        preview = preview_moire_results_file(results_file, limit=int(limit))
    except Exception as exc:
        get_guided_ui().print()
        get_guided_ui().print(f"Candidate preview was skipped: {exc}")
        return
    get_guided_ui().print()
    get_guided_ui().print("Candidate options in the selected results file:")
    get_guided_ui().print(preview)


def _print_site_index_options(site_report, site_type: str, limit: int = 30) -> None:
    sites = surface_backend.sorted_sites_for_type(site_report, site_type)
    get_guided_ui().print()
    get_guided_ui().print(format_adsorption_sites(sites, limit=int(limit), title=f"{_SITE_LABELS.get(site_type, site_type)} site positions"))
    if len(sites) > int(limit):
        get_guided_ui().print("Use `cellstine surface sites` to export the full site table if you need every equivalent site.")


def _format_command(argv: Sequence[str]) -> str:
    parts = []
    for value in argv:
        if any(character.isspace() for character in value):
            parts.append(f'"{value}"')
        else:
            parts.append(value)
    return "cellstine " + " ".join(parts)


def _print_command_preview(title: str, argv: Sequence[str]) -> None:
    get_guided_ui().command_preview(title, argv)


def _first_artifact(result, key: str) -> str | None:
    value = result.artifacts.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return None if not value else str(value[0])
    return str(value)

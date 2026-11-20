"""Prompt primitives and screen helpers for the interactive launcher."""

from __future__ import annotations

import argparse
import heapq
import os
import shlex
import sys
import math
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, Sequence, TypeVar

from ...core.previews import format_adsorption_sites, _preview_moire_results_with_count
from ...interface.surface import backend as surface_backend


INPUT_DIR = Path("input")
RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("output")
STRUCTURE_PATTERNS = ("*.vasp", "POSCAR*", "CONTCAR*")

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


_T = TypeVar("_T")
_HISTORY = None


class _PromptHistory:
    def __init__(self):
        self.answers = []
        self.cursor = 0
        self.depth = 0


class _Remembered:
    """A preview kept in the answer history; Back steps over it, never onto it."""

    def __init__(self, value):
        self.value = value


def guided_history_active() -> bool:
    return _HISTORY is not None


def _replaying() -> bool:
    """True while earlier answers are being re-applied after Back."""

    return _HISTORY is not None and _HISTORY.cursor < len(_HISTORY.answers)


def _remember(compute: Callable[[], _T]) -> _T:
    """Compute a preview once per guided command, surviving Back replays.

    Reading a large results file or detecting surface sites can take far longer
    than answering the question that follows it, so stepping back must not
    repeat the work.
    """

    history = _HISTORY
    if history is None or history.depth:
        return compute()
    if history.cursor < len(history.answers):
        entry = history.answers[history.cursor]
        history.cursor += 1
        return entry.value
    value = compute()
    history.answers.append(_Remembered(value))
    history.cursor += 1
    return value


def _drop_last_answer(history: _PromptHistory) -> bool:
    """Forget the latest real answer and any previews recorded after it."""

    while history.answers:
        if not isinstance(history.answers.pop(), _Remembered):
            return True
    return False


def forget_confirmation(history: _PromptHistory) -> None:
    """Re-open a finished command at its run confirmation."""

    _drop_last_answer(history)


def _guided_step(function):
    """Record validated answers, treating nested prompt helpers as one step."""
    @wraps(function)
    def step(*args, **kwargs):
        history = _HISTORY
        if history is None or history.depth:
            return function(*args, **kwargs)
        if history.cursor < len(history.answers):
            answer = history.answers[history.cursor]
            history.cursor += 1
            return answer
        history.depth += 1
        try:
            answer = function(*args, **kwargs)
        finally:
            history.depth -= 1
        history.answers.append(answer)
        history.cursor += 1
        return answer
    return step


def run_guided_steps(builder: Callable[[], _T], history: _PromptHistory | None = None) -> _T:
    """Rebuild a command after Back, replaying earlier validated answers.

    Only command construction belongs here. Execution stays outside the replay
    boundary so revisiting a prompt cannot execute an earlier command again.
    Passing the ``history`` of an earlier call resumes it with its answers kept.
    """
    global _HISTORY
    previous = _HISTORY
    history = _HISTORY = _PromptHistory() if history is None else history
    try:
        while True:
            history.cursor = 0
            try:
                return builder()
            except _BackInteractive:
                if not _drop_last_answer(history):
                    raise
                get_guided_ui().print("Back to the previous question. Earlier answers are kept.")
    finally:
        _HISTORY = previous


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
                if answer == "b" or answer.lower() == "back":
                    if allow_back:
                        raise _BackInteractive()
                    self.print("You are already at the main menu.")
                    continue
                if answer.lower() in {"q", "quit", "exit"}:
                    raise _QuitInteractive()
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

        from rich.text import Text

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
        self._text = Text
        self._prompt = Prompt
        self._confirm = Confirm
        self._table = Table

    def _box_kwargs(self) -> dict[str, object]:
        if self._box is None:
            return {}
        return {"box": self._box}

    def print(self, *args, **kwargs) -> None:
        # Guided text is data: "[u v w]" is a direction, not a Rich style tag.
        kwargs.setdefault("markup", False)
        kwargs.setdefault("highlight", False)
        try:
            self.console.print(*args, **kwargs)
        except UnicodeEncodeError:
            kwargs.pop("markup", None)
            kwargs.pop("highlight", None)
            PlainGuidedUI.print(self, *args, **kwargs)

    def title(self, title: str, subtitle: str | None = None) -> None:
        body = subtitle or "Choose the workflow settings below."
        self.print()
        self.print(self._panel.fit(self._text(body), title=self._text(title), **self._box_kwargs()))

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
                if answer == "b" or answer.lower() == "back":
                    if allow_back:
                        raise _BackInteractive()
                    self.print("You are already at the main menu.")
                    continue
                if answer.lower() in {"q", "quit", "exit"}:
                    raise _QuitInteractive()
                return answer
            if default is not None:
                return default
            if allow_empty:
                return ""
            self.print("Please enter a value.")

    def confirm(self, prompt: str, default_yes: bool = True) -> bool:
        # Rich Confirm consumes invalid answers internally, including Back.
        # Use the shared yes/no loop over Rich Prompt so navigation propagates.
        return super().confirm(prompt, default_yes)

    def choice(self, title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
        self.title(title, "Select an option. Use q to quit; use b to go back where available.")
        table = self._table(show_header=True, header_style="bold", **self._box_kwargs())
        table.add_column("#", justify="right")
        table.add_column("Option")
        table.add_column("When to use it")
        for index, option in enumerate(options, start=1):
            table.add_row(str(index), self._text(option["label"]), self._text(option.get("hint", "")))
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
        self.print(self._panel.fit(self._text(_format_command(argv)), title=self._text(title), **self._box_kwargs()))


class _MutedGuidedUI(PlainGuidedUI):
    """Stands in during a Back replay, so answered screens are not repeated."""

    def print(self, *args, **kwargs) -> None:
        pass


_ACTIVE_UI: PlainGuidedUI = PlainGuidedUI()
_MUTED_UI = _MutedGuidedUI()


def get_guided_ui() -> PlainGuidedUI:
    return _MUTED_UI if _replaying() else _ACTIVE_UI


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


@_guided_step
def _prompt(
    prompt: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
    allow_back: bool = True,
) -> str:
    return get_guided_ui().prompt(prompt, default, allow_empty=allow_empty, allow_back=allow_back)


@_guided_step
def _prompt_int(prompt: str, default: int, *, minimum: int | None = None) -> int:
    while True:
        try:
            value = int(_prompt(prompt, str(default)))
            if minimum is not None and value < minimum:
                get_guided_ui().print(f"Please enter a whole number of at least {minimum}.")
                continue
            return value
        except ValueError:
            get_guided_ui().print("Please enter a whole number.")


@_guided_step
def _prompt_float(
    prompt: str,
    default: float,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    while True:
        try:
            value = float(_prompt(prompt, str(default)))
            if not math.isfinite(value):
                raise ValueError("non-finite value")
        except ValueError:
            get_guided_ui().print("Please enter a finite number.")
            continue
        if positive and value <= 0.0:
            get_guided_ui().print("Please enter a number greater than zero.")
            continue
        if minimum is not None and value < minimum:
            get_guided_ui().print(f"Please enter a number of at least {minimum:g}.")
            continue
        return value


@_guided_step
def _prompt_strain(prompt: str, default: float = 0.02) -> float:
    """Ask for a strain budget, catching a percentage typed as a fraction."""

    while True:
        value = _prompt_float(f"{prompt} (0.02 means about 2%)", default, minimum=0.0)
        if value <= 0.2 or _prompt_yes_no(f"{value:g} means about {100.0 * value:g}% strain. Keep it?", False):
            return value


@_guided_step
def _prompt_parsed(prompt: str, default: str, parse: Callable[[str], object]) -> str:
    """Ask until ``parse`` accepts the text, and return the text as typed."""

    while True:
        text = _prompt(prompt, default)
        try:
            parse(text)
        except (ValueError, argparse.ArgumentTypeError) as error:
            get_guided_ui().print(f"That was not understood: {error}")
            continue
        return text


@_guided_step
def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    return get_guided_ui().confirm(prompt, default_yes=default_yes)


@_guided_step
def _choice(title: str, options: Sequence[dict[str, str]], default: int = 1, *, allow_back: bool = True) -> str:
    return get_guided_ui().choice(title, options, default=default, allow_back=allow_back)


def _relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _find_candidates(patterns: Sequence[str], roots: Sequence[Path], *, limit: int = 8) -> list[Path]:
    """Most recent unique files, preserving root priority and traversal ties.

    Only the remaining shortlist is sorted in each root. Once filled, lower
    priority trees cannot contribute. The seen set retains exact resolved-path
    deduplication across patterns and overlapping roots.
    """
    if limit == 0:
        return []
    seen: set[Path] = set()

    def unique_matches(root: Path) -> Iterator[Path]:
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved

    if limit < 0:
        # Retain the historical slice semantics for private callers using a
        # negative limit; normal guided prompts always request a small positive
        # shortlist. Such a slice genuinely needs all roots.
        all_matches = [(index, path) for index, root in enumerate(roots)
                       if root.exists() for path in unique_matches(root)]
        all_matches.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
        return [path for _, path in all_matches[:limit]]

    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(heapq.nsmallest(
                limit - len(found), unique_matches(root), key=lambda path: -path.stat().st_mtime,
            ))
            if len(found) == limit:
                break
    return found


def _find_in_working_folder(patterns: Sequence[str], *, limit: int = 8) -> list[Path]:
    """Look beside the user, without descending, when the usual folders are empty."""

    found = {path.resolve() for pattern in patterns for path in Path.cwd().glob(pattern) if path.is_file()}
    return sorted(found, key=lambda path: -path.stat().st_mtime)[:limit]


def _clean_path_text(text: str) -> str:
    """Undo what shells and file managers add: surrounding quotes and ``~``."""

    value = str(text).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return str(Path(value).expanduser()) if value.startswith("~") else value


def _path_problem(text: str, *, allow_directory: bool) -> str | None:
    path = Path(text)
    if not path.exists():
        return f"No file was found at {text}."
    if path.is_dir() and not allow_directory:
        return f"{text} is a folder; please choose a file."
    return None


@_guided_step
def _prompt_path(
    label: str,
    *,
    patterns: Sequence[str],
    roots: Sequence[Path],
    default: str | None = None,
    allow_manual: bool = True,
) -> str:
    suggestions = _find_candidates(patterns, roots) or _find_in_working_folder(patterns)
    allow_directory = any(pattern.endswith(".json") for pattern in patterns)
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
            ui.print("  m. Type a different path" if len(suggestions) < 8 else "  m. Type a path (only the 8 most recent are listed)")
        ui.print("  b. Back")
        ui.print("  q. Quit interactive mode")
        default_value = default if default is not None else "1"
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
            ui.print(f"Please choose a file from 1 to {len(suggestions)}, or use m to enter a path.")
            continue
        if allow_manual and answer.lower() in {"m", "manual"}:
            try:
                answer = _prompt("Path").strip()
            except _BackInteractive:
                continue
            if answer.lower() in {"q", "quit", "exit"}:
                raise _QuitInteractive()
        if answer:
            cleaned = _clean_path_text(answer)
            problem = _path_problem(cleaned, allow_directory=allow_directory)
            if problem is None:
                return cleaned
            ui.print(f"{problem} Check the spelling, or choose from the list.")
            continue
        if default is not None:
            return default
        ui.print("Please choose a suggested file or type a path.")


@_guided_step
def _prompt_csv(prompt: str, default: str) -> str:
    return _prompt(prompt, default)


@_guided_step
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


def _print_saved_moire_preview(results_file: str, limit: int = 15) -> int | None:
    def read_preview() -> tuple[str | None, str | None, int | None]:
        path = Path(results_file)
        if path.is_file() and path.stat().st_size > 20_000_000:
            get_guided_ui().print(f"Reading {path.stat().st_size / 1e6:.0f} MB of saved candidates...")
        try:
            text, count = _preview_moire_results_with_count(results_file, limit=int(limit))
            return text, None, count
        except Exception as exc:
            return None, str(exc), None

    preview, problem, count = _remember(read_preview)
    if preview is None:
        get_guided_ui().print()
        get_guided_ui().print(f"Candidate preview was skipped: {problem}")
        return None
    get_guided_ui().print()
    get_guided_ui().print("Saved candidate preview from the selected file:")
    get_guided_ui().print(preview)
    return count


def _saved_candidate_count(results_file: str) -> int | None:
    """Actual validated count for selection without a displayed preview.

    Header metadata alone cannot prove a selection bound: JSON permits repeated
    keys and the count can disagree with the actual rows. Build reuses its
    existing preview read instead of calling this a second time.
    """
    try:
        return _preview_moire_results_with_count(results_file, limit=0)[1]
    except (OSError, ValueError, KeyError):
        return None


def _index_parser(count: int | None) -> Callable[[str], object]:
    """``parse_index_spec`` that also refuses indexes the saved file lacks."""

    from ..argtypes import parse_index_spec

    def parse(text: str) -> object:
        values = parse_index_spec(text)
        if count == 0:
            raise ValueError("this file holds no candidates")
        if count is not None and (min(values) < 1 or max(values) > count):
            raise ValueError(f"this file holds candidates 1 to {count}")
        return values

    return parse


def _print_site_index_options(site_report, site_type: str, limit: int = 30) -> None:
    sites = surface_backend.sorted_sites_for_type(site_report, site_type)
    get_guided_ui().print()
    get_guided_ui().print(format_adsorption_sites(sites, limit=int(limit), title=f"{_SITE_LABELS.get(site_type, site_type)} site positions"))
    if len(sites) > int(limit):
        get_guided_ui().print("Use `cellstine surface sites` to export the full site table if you need every equivalent site.")


#: Literal bare-word characters shared by generated command arguments.
_POWERSHELL_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+=:,./\\-")


def _format_command(argv: Sequence[str]) -> str:
    """Quote for PowerShell on Windows and a POSIX shell on other hosts.

    Actual guided execution always passes the original argv directly. Literal
    single quotes keep dollar signs/backticks inert in copied PowerShell text.
    PowerShell reserves the native token --% even when quoted; copying that
    exact argument and legacy native quote handling remain shell limitations.
    """

    if sys.platform != "win32":
        return shlex.join(["cellstine", *argv])
    parts = []
    for value in argv:
        if value and all(character in _POWERSHELL_SAFE for character in value):
            parts.append(value)
        else:
            parts.append("'" + value.replace("'", "''") + "'")
    return "cellstine " + " ".join(parts)


def _print_command_preview(title: str, argv: Sequence[str]) -> None:
    get_guided_ui().command_preview(title, argv)


def _first_artifact(result, key: str) -> str | None:
    value = getattr(result, "artifacts", {}).get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return None if not value else str(value[0])
    return str(value)

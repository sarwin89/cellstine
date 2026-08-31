"""New command surface for the CELLSTINE CLI.

The CLI is now specified once and rendered by two frontends: a dependency-free
plain frontend and an optional Typer/Rich frontend.  These tests deliberately
exercise the user-facing command names rather than argparse internals.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from cellstine.cli.plain import build_parser
from cellstine.cli.spec import LEGACY_COMMANDS, parse_twist_window, resolve_moire_strains


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "cellstine"


def _install_fake_rich_typer(monkeypatch):
    fake_typer = types.ModuleType("typer")
    fake_rich = types.ModuleType("rich")
    fake_console = types.ModuleType("rich.console")
    fake_panel = types.ModuleType("rich.panel")
    fake_prompt = types.ModuleType("rich.prompt")
    fake_table = types.ModuleType("rich.table")

    class Console:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.printed = []
            Console.instances.append(self)

        def print(self, *args, **kwargs):
            self.printed.append((args, kwargs))

    class Panel:
        @staticmethod
        def fit(*args, **kwargs):
            return {"kind": "panel", "args": args, "kwargs": kwargs}

    class Prompt:
        @staticmethod
        def ask(_prompt, default=None, **_kwargs):
            return default or ""

    class Confirm:
        @staticmethod
        def ask(_prompt, default=True, **_kwargs):
            return default

    class Table:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.columns = []
            self.rows = []

        def add_column(self, *args, **kwargs):
            self.columns.append((args, kwargs))

        def add_row(self, *args, **kwargs):
            self.rows.append((args, kwargs))

    fake_console.Console = Console
    fake_panel.Panel = Panel
    fake_prompt.Prompt = Prompt
    fake_prompt.Confirm = Confirm
    fake_table.Table = Table
    monkeypatch.setitem(sys.modules, "typer", fake_typer)
    monkeypatch.setitem(sys.modules, "rich", fake_rich)
    monkeypatch.setitem(sys.modules, "rich.console", fake_console)
    monkeypatch.setitem(sys.modules, "rich.panel", fake_panel)
    monkeypatch.setitem(sys.modules, "rich.prompt", fake_prompt)
    monkeypatch.setitem(sys.modules, "rich.table", fake_table)
    return Console


def test_simplified_commands_parse_to_workflow_targets():
    parser = build_parser()

    cases = [
        (["moire", "search", "top.vasp", "bottom.vasp", "--length", "20", "--strain", "0.01"], ("moire", "search")),
        (["moire", "build", "results.json", "--indexes", "1"], ("moire", "build")),
        (["moire", "shift", "stack.vasp", "--shift-direct", "0,0"], ("moire", "shift")),
        (["moire", "view", "results.json"], ("moire", "view")),
        (["moire", "stack-search", "base.vasp", "top.vasp", "--length", "20"], ("moire", "stack-search")),
        (["moire", "stack-build", "results.json", "--indexes", "1"], ("moire", "stack-build")),
        (["surface", "build", "bulk.vasp", "--miller", "111"], ("surface", "build")),
        (["surface", "sites", "slab.vasp"], ("surface", "sites")),
        (["interface", "match", "bottom.vasp", "top.vasp"], ("interface", "match")),
        (["interface", "build", "bottom.vasp", "top.vasp"], ("interface", "build")),
        (["interface", "registries", "bottom.vasp", "top.vasp"], ("interface", "registries")),
        (["symmetry", "kpath", "cell.vasp", "--divisions", "10"], ("symmetry", "kpath")),
        (["view", "cell.vasp"], ("view", "structure")),
    ]

    for argv, expected in cases:
        namespace = parser.parse_args(argv)
        assert (namespace.group, namespace.stage) == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"rigid": True, "strain": None, "top_strain": None, "bottom_strain": None}, (0.0, 0.0)),
        ({"rigid": False, "strain": 0.02, "top_strain": None, "bottom_strain": None}, (0.02, 0.02)),
        ({"rigid": False, "strain": None, "top_strain": 0.01, "bottom_strain": 0.03}, (0.01, 0.03)),
    ],
)
def test_moire_strain_shorthand_resolves_to_explicit_layer_budgets(kwargs, expected):
    assert resolve_moire_strains(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"rigid": False, "strain": None, "top_strain": None, "bottom_strain": None}, "choose one strain mode"),
        ({"rigid": True, "strain": 0.01, "top_strain": None, "bottom_strain": None}, "choose one strain mode"),
        ({"rigid": False, "strain": 0.01, "top_strain": 0.01, "bottom_strain": 0.01}, "choose one strain mode"),
        ({"rigid": False, "strain": None, "top_strain": 0.01, "bottom_strain": None}, "both --top-strain and --bottom-strain"),
    ],
)
def test_moire_search_requires_one_clear_strain_mode(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_moire_strains(**kwargs)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("9:14", (9.0, 14.0)),
        (":12", (None, 12.0)),
        ("8:", (8.0, None)),
        ("14:9", (9.0, 14.0)),
    ],
)
def test_twist_window_uses_one_readable_range_option(raw, expected):
    assert parse_twist_window(raw) == expected


def test_removed_commands_have_migration_guidance():
    assert LEGACY_COMMANDS[("moire", "find")].replacement == "cellstine moire search"
    assert LEGACY_COMMANDS[("interface", "surface")].replacement == "cellstine surface build"


def test_removed_commands_fail_early_with_migration_guidance(capsys):
    from cellstine.cli.main import main

    assert main(["--plain", "moire", "find"]) == 2
    err = capsys.readouterr().err
    assert "cellstine moire search" in err


def test_plain_flag_forces_the_stdlib_frontend(capsys):
    from cellstine.cli.main import main

    assert main(["--plain", "--help"]) == 0
    out = capsys.readouterr().out
    assert "cellstine moire search" not in out
    assert "surface" in out


def test_rich_frontend_help_renders_when_optional_dependencies_exist(capsys):
    pytest.importorskip("typer")
    pytest.importorskip("rich")

    from cellstine.cli.rich_app import run

    assert run(["--help"]) == 0
    assert "CELLSTINE" in capsys.readouterr().out


def test_rich_frontend_delegates_without_typer_runtime_state(monkeypatch):
    _install_fake_rich_typer(monkeypatch)

    import cellstine.cli.plain as plain
    from cellstine.cli.rich_app import run

    captured = {}

    def fake_run(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(plain, "run", fake_run)

    assert run(["moire", "search"]) == 0
    assert captured["argv"] == ["moire", "search"]


def test_rich_frontend_uses_rich_guided_ui_for_no_args(monkeypatch):
    _install_fake_rich_typer(monkeypatch)

    import cellstine.cli.interactive.runner as runner
    from cellstine.cli.rich_app import run

    captured = {}

    def fake_run_interactive(*, group=None, ui=None, show_banner=True):
        captured["group"] = group
        captured["ui_type"] = type(ui).__name__
        captured["show_banner"] = show_banner
        return 0

    monkeypatch.setattr(runner, "run_interactive", fake_run_interactive)

    assert run([]) == 0
    assert captured == {"group": None, "ui_type": "RichGuidedUI", "show_banner": True}


def test_rich_frontend_uses_rich_guided_ui_for_group_only(monkeypatch):
    _install_fake_rich_typer(monkeypatch)

    import cellstine.cli.interactive.runner as runner
    from cellstine.cli.rich_app import run

    captured = {}

    def fake_run_interactive(*, group=None, ui=None, show_banner=True):
        captured["group"] = group
        captured["ui_type"] = type(ui).__name__
        captured["show_banner"] = show_banner
        return 0

    monkeypatch.setattr(runner, "run_interactive", fake_run_interactive)

    assert run(["moire"]) == 0
    assert captured == {"group": "moire", "ui_type": "RichGuidedUI", "show_banner": True}


def test_plain_flag_forces_plain_guided_mode_even_when_rich_exists(monkeypatch):
    _install_fake_rich_typer(monkeypatch)

    import cellstine.cli.interactive.runner as runner
    from cellstine.cli.main import main

    captured = {}

    def fake_run_interactive(group=None, *, ui=None, show_banner=True):
        captured["group"] = group
        captured["ui"] = ui
        captured["show_banner"] = show_banner
        return 0

    monkeypatch.setattr(runner, "run_interactive", fake_run_interactive)

    assert main(["--plain"]) == 0
    assert captured == {"group": None, "ui": None, "show_banner": True}


def test_main_can_select_optional_frontend_without_typer_runtime_state(monkeypatch):
    _install_fake_rich_typer(monkeypatch)

    import cellstine.cli.plain as plain
    from cellstine.cli.main import main

    captured = {}

    def fake_run(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(plain, "run", fake_run)

    assert main(["moire", "search"]) == 0
    assert captured["argv"] == ["moire", "search"]


def test_root_cellstine_launcher_exists_and_uses_maintained_entrypoint():
    launcher = ROOT / "cellstine.py"
    assert launcher.exists()
    text = launcher.read_text(encoding="utf-8")
    assert "from cellstine.cli.main import main" in text
    assert "build_parser" not in text

    finished = subprocess.run(
        [sys.executable, str(launcher), "--version"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert finished.returncode == 0, finished.stderr
    assert "cellstine" in finished.stdout


def test_root_cellstine_launcher_starts_guided_mode_without_shadowing_the_package():
    launcher = ROOT / "cellstine.py"
    finished = subprocess.run(
        [sys.executable, str(launcher)],
        input="q\n",
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert finished.returncode == 0, finished.stderr
    assert "CELLSTINE" in finished.stdout
    assert "Closed CELLSTINE interactive mode." in finished.stdout


def test_root_cellstine_launcher_without_stdin_exits_cleanly():
    launcher = ROOT / "cellstine.py"
    finished = subprocess.run(
        [sys.executable, str(launcher)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert finished.returncode == 0, finished.stderr
    assert "Closed CELLSTINE interactive mode." in finished.stdout


def test_plain_guided_banner_uses_the_historical_block_art():
    from cellstine.cli.interactive.prompts import MAIN_MENU_BANNER

    assert "██████╗███████╗██╗" in MAIN_MENU_BANNER
    assert "╚═════╝╚══════╝" in MAIN_MENU_BANNER


def test_stale_moire_only_launcher_was_removed():
    assert not (ROOT / "moire_cli.py").exists()


def test_pyproject_exposes_only_the_cellstine_console_script():
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"] == {"cellstine": "cellstine.cli.main:main"}


def test_plain_frontend_imports_without_numpy_or_workflows():
    script = textwrap.dedent(
        f"""
        import importlib.util, sys
        from pathlib import Path
        root = Path({str(PACKAGE_ROOT)!r})
        spec = importlib.util.spec_from_file_location(
            "cellstine", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["cellstine"] = module
        spec.loader.exec_module(module)
        import cellstine.cli.main
        print("numpy" in sys.modules)
        print(",".join(sorted(name for name in sys.modules if name.startswith("cellstine."))))
        """
    )
    finished = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert finished.returncode == 0, finished.stderr
    loaded = finished.stdout.strip().splitlines()
    assert loaded[0] == "False"
    modules = set(loaded[1].split(",")) if len(loaded) > 1 and loaded[1] else set()
    assert "cellstine.moire.moire" not in modules
    assert "cellstine.interface.workflow.interface" not in modules


def test_plain_help_uses_the_new_groups(capsys):
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--help"])
    assert raised.value.code == 0
    out = capsys.readouterr().out
    assert "surface" in out
    assert "moire" in out
    assert "interface" in out

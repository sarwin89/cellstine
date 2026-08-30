"""Starting the CLI must not load work it is not going to do.

Building the argument parser and dispatching a command used to import every
workflow package --- moire, adsorbate, interface, defect, symmetry, visualize
--- plus NumPy and ``importlib.metadata``, before a single argument had been
read.  A run only ever uses one group, so that was about a quarter of a second
of pure overhead on every invocation, ``cellstine --help`` included.

The parser now reads its defaults from the dependency-free
``cellstine.core.constants``, and ``cellstine.cli.main`` resolves a workflow
class only when a stage asks for one.  The checks below pin both properties by
running a fresh interpreter and looking at ``sys.modules``: a regression here is
invisible in ordinary use and only shows up as a slow tool.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "cellstine"

_BOOTSTRAP = """
import importlib.util, sys
from pathlib import Path
ROOT = Path({root!r})
spec = importlib.util.spec_from_file_location(
    "cellstine", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
module = importlib.util.module_from_spec(spec)
sys.modules["cellstine"] = module
spec.loader.exec_module(module)
"""

WORKFLOW_MODULES = [
    "cellstine.adsorbate.molecule",
    "cellstine.defect.workflow",
    "cellstine.interface.surface.surface",
    "cellstine.interface.workflow.interface",
    "cellstine.moire.moire",
    "cellstine.moire.supermoire",
    "cellstine.symmetry.symmetry",
    "cellstine.visualize.visualize",
]


def _run(body: str) -> str:
    script = _BOOTSTRAP.format(root=str(PACKAGE_ROOT)) + textwrap.dedent(body)
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout.strip()


def test_importing_the_cli_loads_no_workflow_and_no_numpy():
    loaded = _run(
        """
        import sys
        import cellstine.cli.main
        print(",".join(sorted(name for name in sys.modules if name.startswith("cellstine."))))
        print("numpy" in sys.modules)
        """
    ).splitlines()
    modules = set(loaded[0].split(",")) if loaded[0] else set()
    assert loaded[1] == "False"
    for name in WORKFLOW_MODULES:
        assert name not in modules
    assert modules <= {
        "cellstine.cli",
        "cellstine.cli.argtypes",
        "cellstine.cli.main",
        "cellstine.cli.parsers",
        "cellstine.cli.spec",
        "cellstine.core",
        "cellstine.core.constants",
    }


def test_building_the_parser_and_printing_help_loads_no_workflow():
    loaded = _run(
        """
        import contextlib, io, sys
        from cellstine.cli.main import main
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                main(["--help"])
            except SystemExit:
                pass
        print("numpy" in sys.modules)
        print(",".join(sorted(name for name in sys.modules if name.startswith("cellstine."))))
        """
    ).splitlines()
    assert loaded[0] == "False"
    modules = set(loaded[1].split(",")) if len(loaded) > 1 and loaded[1] else set()
    for name in WORKFLOW_MODULES:
        assert name not in modules


@pytest.mark.parametrize(
    "workflow,expected,unexpected",
    [
        ("Symmetry", "cellstine.symmetry.symmetry", "cellstine.moire.moire"),
        ("Moire", "cellstine.moire.moire", "cellstine.defect.workflow"),
        ("Defect", "cellstine.defect.workflow", "cellstine.visualize.visualize"),
    ],
)
def test_a_stage_loads_only_the_group_it_uses(workflow, expected, unexpected):
    loaded = _run(
        f"""
        import sys
        from cellstine.cli.main import _workflow
        _workflow({workflow!r})
        print(",".join(sorted(name for name in sys.modules if name.startswith("cellstine."))))
        """
    )
    modules = set(loaded.split(","))
    assert expected in modules
    assert unexpected not in modules


def test_the_workflow_classes_are_still_importable_by_name():
    """The lazy dispatch keeps ``from cellstine.cli.main import Moire`` working."""

    printed = _run(
        """
        from cellstine.cli.main import Moire, Symmetry
        print(Moire.__name__, Symmetry.__name__)
        """
    )
    assert printed == "Moire Symmetry"


def test_an_unknown_attribute_is_still_an_attribute_error():
    printed = _run(
        """
        import cellstine.cli.main as main
        try:
            main.NotAWorkflow
        except AttributeError as error:
            print("AttributeError")
        """
    )
    assert printed == "AttributeError"


def test_a_dependency_check_does_not_need_importlib_metadata_up_front():
    """``importlib.metadata`` costs about 50 ms and is only read for versions."""

    printed = _run(
        """
        import sys
        from cellstine.core.dependencies import DependencyManager
        print("importlib.metadata" in sys.modules)
        DependencyManager().has("numpy")
        print("importlib.metadata" in sys.modules)
        """
    ).splitlines()
    assert printed[0] == "False"
    assert printed[1] == "True"

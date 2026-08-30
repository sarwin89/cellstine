"""Guards on the machine-checked half of the repository.

Every numerical convention the Python code relies on is backed by a proof in
``aristotle-lean-reference/RequestProject/``.  These tests do not re-run Lean --
that is what ``lake build`` is for -- but they do pin the two things that
silently rot: a proof left unfinished, and a file that stops being built because
nothing imports it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEAN_ROOT = ROOT / "aristotle-lean-reference" / "RequestProject"
MAIN = LEAN_ROOT / "Main.lean"

#: ``sorry`` and ``admit`` leave a proof unfinished; ``axiom`` asserts a
#: statement without one.  Each is matched as a whole word so that prose such as
#: "admitting a near-miss" in a docstring does not trip the search.
UNFINISHED = re.compile(r"(?<![\w.])(sorry|admit|axiom)(?![\w'])")


def lean_files() -> list[Path]:
    return sorted(LEAN_ROOT.glob("*.lean"))


def strip_comments(text: str) -> str:
    """Return ``text`` with Lean's block and line comments removed."""

    without_blocks = re.sub(r"/-.*?-/", " ", text, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", without_blocks)


def test_the_lean_sources_are_where_they_are_expected():
    assert LEAN_ROOT.is_dir()
    assert MAIN.is_file()
    assert len(lean_files()) > 1


@pytest.mark.parametrize("path", lean_files(), ids=lambda path: path.name)
def test_no_proof_is_left_unfinished(path: Path):
    """No ``sorry``, ``admit`` or ``axiom`` survives in a Lean source."""

    body = strip_comments(path.read_text(encoding="utf-8"))
    assert UNFINISHED.search(body) is None, f"{path.name} still carries an unfinished proof"


@pytest.mark.parametrize("path", lean_files(), ids=lambda path: path.name)
def test_every_lean_file_is_imported_by_the_index(path: Path):
    """``Main.lean`` imports every other file, so building it builds them all."""

    if path == MAIN:
        pytest.skip("the index does not import itself")
    imports = set(re.findall(r"^import\s+(\S+)", MAIN.read_text(encoding="utf-8"), re.MULTILINE))
    assert f"RequestProject.{path.stem}" in imports

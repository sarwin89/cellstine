"""The stacking options reach the workflow through the command line.

The two entry points a user actually types -- ``interface registries`` and
``interface build`` with a stacking sense and a contact -- are driven here
through the argument parser, and the interactive question flow is checked to
emit a command line the parser accepts.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.interactive import build_interface as interactive
from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.interface.surface import stacking
from cellstine.io.converters import StructureConverter

from conftest import write_poscar


def run_cli(*argv: str):
    return execute_namespace(build_parser().parse_args(list(argv)))


@pytest.fixture()
def workspace(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def aluminium(workspace) -> str:
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(workspace / "al.vasp", lattice, ["Al"], [4], positions))


@pytest.fixture()
def slab_111(aluminium) -> str:
    result = run_cli(
        "interface", "surface", aluminium, "--miller", "1,1,1", "--layers", "6", "--vacuum", "12"
    )
    return str(result.artifacts["slab_poscar"])


def test_the_registries_command_lists_the_distinct_options(slab_111):
    result = run_cli(
        "interface", "registries", slab_111, slab_111, "--bottom-kind", "slab", "--top-kind", "slab"
    )
    assert result.summary["distinct_options"] == 5
    document = json.loads(Path(result.artifacts["registries_json"]).read_text())
    assert len(document["options"]) == 5
    assert all(option["equivalent_to"] is None for option in document["options"])
    assert {option["kind"] for option in document["options"]} == {
        "eclipsed",
        "fcc_hollow",
        "hcp_hollow",
    }


def test_the_registries_command_can_show_the_removed_duplicates(slab_111):
    result = run_cli(
        "interface",
        "registries",
        slab_111,
        slab_111,
        "--bottom-kind",
        "slab",
        "--top-kind",
        "slab",
        "--include-equivalent",
    )
    assert result.summary["listed_options"] == 12
    assert result.summary["distinct_options"] == 5


def test_the_build_command_reverses_the_top_slab_at_a_named_contact(slab_111, workspace):
    output = workspace / "twin.vasp"
    result = run_cli(
        "interface",
        "build",
        slab_111,
        slab_111,
        "--bottom-kind",
        "slab",
        "--top-kind",
        "slab",
        "--gap",
        "2.34",
        "--vacuum",
        "12",
        "--top-stacking",
        "cba",
        "--registry",
        "hcp",
        "--output-path",
        str(output),
    )
    assert result.summary["stacking"]["top_sense"] == "CBA"
    assert result.summary["stacking"]["kind"] == "hcp_hollow"
    analysis = stacking.analyse_stacking(StructureConverter().read(str(output)))
    assert analysis.sequence == "ABCABCBACBAC"


def _scripted_input(monkeypatch, answers):
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *_: next(replies))


def test_the_interactive_flow_offers_the_stacking_questions(monkeypatch):
    _scripted_input(monkeypatch, ["y", "2", "4", "3"])
    assert interactive._prompt_stacking_options(allow_relative=True) == [
        "--bottom-stacking",
        "mirror",
        "--top-stacking",
        "cba",
        "--registry",
        "hcp",
    ]


def test_the_interactive_flow_skips_the_relative_senses_for_a_matched_cell(monkeypatch):
    _scripted_input(monkeypatch, ["y", "1", "2"])
    assert interactive._prompt_stacking_options(allow_relative=False) == [
        "--top-stacking",
        "mirror",
    ]


def test_the_interactive_flow_can_leave_the_stacking_alone(monkeypatch):
    _scripted_input(monkeypatch, ["n"])
    assert interactive._prompt_stacking_options(allow_relative=True) == []
    _scripted_input(monkeypatch, ["y", "1", "1", "1"])
    assert interactive._prompt_stacking_options(allow_relative=True) == []


def test_the_interactive_answers_parse_as_a_command_line(monkeypatch, slab_111):
    _scripted_input(monkeypatch, ["y", "1", "4", "2"])
    argv = [
        "interface",
        "build",
        slab_111,
        slab_111,
        "--bottom-kind",
        "slab",
        "--top-kind",
        "slab",
        *interactive._prompt_stacking_options(allow_relative=True),
    ]
    args = build_parser().parse_args(argv)
    assert args.top_stacking == "cba"
    assert args.registry == "fcc"

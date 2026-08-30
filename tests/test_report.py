"""The shared CLI report renderer used by every workflow."""

from __future__ import annotations

from pathlib import Path

from cellstine.core.models import CommandResult
from cellstine.core.report import (
    format_number,
    format_path,
    format_result,
    format_scalar,
    render_mapping,
)


def test_numbers_lose_their_binary_tail_but_not_their_value():
    assert format_number(29.72272733872298) == "29.7227"
    assert format_number(0.0) == "0"
    assert format_number(-0.0) == "0"
    assert format_number(15.0) == "15"
    assert format_number(2.2) == "2.2"
    assert format_number(1.234e-9) == "1.234e-09"


def test_scalars_read_as_words_and_carry_their_unit():
    assert format_scalar(True) == "yes"
    assert format_scalar(False) == "no"
    assert format_scalar(None) == "-"
    assert format_scalar(15.0, key="vacuum") == "15 Ang"
    assert format_scalar(21.7868, key="angle") == "21.7868 deg"
    assert format_scalar("fcc_hollow") == "fcc_hollow"


def test_paths_are_relative_to_the_working_directory(tmp_path):
    inside = tmp_path / "output" / "slab.vasp"
    assert format_path(inside, tmp_path) == str(Path("output") / "slab.vasp")
    outside = Path("/etc") / "hosts"
    assert format_path(outside, tmp_path) == str(outside)


def test_nested_groups_are_indented_under_their_heading():
    lines = render_mapping(
        {
            "total_atoms": 8,
            "stacking": {"contact": "A-B", "registry_requested": True},
            "principal_log_strains": [0.0, 0.0],
        }
    )
    assert lines[0] == "total atoms: 8"
    assert lines[1] == "stacking:"
    assert lines[2] == "  contact: A-B"
    assert lines[3] == "  registry requested: yes"
    assert lines[4] == "principal log strains: [0, 0]"


def test_a_list_of_structures_is_printed_one_path_per_line(tmp_path):
    first = tmp_path / "output" / "a.vasp"
    second = tmp_path / "output" / "b.vasp"
    result = CommandResult(
        manifest_path=tmp_path / "runs" / "manifest.json",
        run_dir=tmp_path / "runs",
        artifacts={"structures": [str(first), str(second)]},
        summary={"generated_count": 2},
    )
    text = format_result(result, base_dir=tmp_path)
    assert "Files written:" in text
    assert "  structures (2):" in text
    assert f"    {Path('output') / 'a.vasp'}" in text
    assert f"    {Path('output') / 'b.vasp'}" in text
    assert "generated count: 2" in text
    assert "[" not in text and "'" not in text


def test_the_report_never_prints_a_python_repr(tmp_path):
    result = CommandResult(
        manifest_path=tmp_path / "manifest.json",
        run_dir=tmp_path,
        artifacts={"interface_poscar": str(tmp_path / "out.vasp")},
        summary={
            "vacuum": 15.0,
            "stacking": {"delta": 0, "kind": "eclipsed", "registry_requested": False},
        },
        payload={"timings_s": {"workflow_total_s": 1.02345}},
    )
    text = format_result(result, base_dir=tmp_path)
    assert "{" not in text and "}" not in text
    assert "'" not in text
    assert "Timing:" in text
    assert "  workflow total: 1.023 s" in text


def test_long_notes_are_wrapped_and_indented():
    note = "the slabs meet eclipsed, one layer directly above the other; " * 3
    lines = render_mapping({"note": note})
    assert len(lines) > 1
    assert all(len(line) <= 96 for line in lines)
    assert lines[1].startswith("  ")


def test_empty_containers_read_as_none():
    lines = render_mapping({"structures": [], "stacking": {}})
    assert lines == ["structures: (none)", "stacking: (none)"]


def test_warnings_are_printed_last_under_their_own_heading():
    """A warning must not be buried between two numbers in the results block."""

    result = CommandResult(
        manifest_path=None,
        run_dir=None,
        artifacts={},
        summary={
            "closest_contact": 0.8123456,
            "warnings": ["the closest contact is 0.81 A; the atoms overlap"],
            "generated": 2,
        },
        payload={},
    )
    text = format_result(result)
    lines = text.splitlines()
    assert "Warnings:" in lines
    assert lines.index("Warnings:") > lines.index("Results:")
    assert "warnings:" not in text
    assert "  - the closest contact is 0.81 A; the atoms overlap" in lines
    # A contact length carries its unit like every other distance.
    assert "closest contact: 0.812346 Ang" in text


def test_a_result_without_warnings_prints_no_warning_heading():
    result = CommandResult(
        manifest_path=None, run_dir=None, artifacts={}, summary={"generated": 1}, payload={}
    )
    assert "Warnings:" not in format_result(result)

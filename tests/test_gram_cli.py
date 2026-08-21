from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstine.cli import main as cli_main
from cellstine.cli.parsers import build_parser
from cellstine.moire.builder import maken
from cellstine.moire.search import findn
from cellstine.moire.supermoire import Supermoire
from cellstine.moire.transform import translaten


N_LAYER_MESSAGE = (
    "N-layer moire workflows are not supported by the Gram-form engine. "
    "Use bilayer moire find and make."
)
MIGRATION_MESSAGE = "--max-length, --top-strain, and --bottom-strain"


def _moire_stages(parser):
    moire = parser._subparsers._group_actions[0].choices["moire"]
    return moire._subparsers._group_actions[0].choices


def test_moire_find_help_exposes_only_native_gram_controls_and_defines_strain(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["moire", "find", "--help"])

    help_text = capsys.readouterr().out
    for control in (
        "--max-length",
        "--top-strain",
        "--bottom-strain",
        "--min-length",
        "--max-atoms",
        "--max-cell-aspect-ratio",
        "--min-cell-angle",
        "--max-cell-angle",
        "--symmetric",
        "--progress",
        "--preview-limit",
    ):
        assert control in help_text
    assert "principal logarithmic strain budget" in help_text
    for legacy_control in (
        "--nindex",
        "--angles",
        "--workers",
        "--max-pair-matches",
        "--matrix-values",
        "--fold-symmetry",
    ):
        assert legacy_control not in help_text


@pytest.mark.parametrize(
    "argv, missing_flag",
    [
        (["moire", "find", "top.vasp", "bottom.vasp", "--top-strain", "0.01", "--bottom-strain", "0.02"], "--max-length"),
        (["moire", "find", "top.vasp", "bottom.vasp", "--max-length", "12", "--bottom-strain", "0.02"], "--top-strain"),
        (["moire", "find", "top.vasp", "bottom.vasp", "--max-length", "12", "--top-strain", "0.01"], "--bottom-strain"),
    ],
)
def test_moire_find_requires_each_native_control(capsys, argv, missing_flag):
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(argv)

    assert missing_flag in capsys.readouterr().err


def test_native_moire_find_dispatches_gram_named_values(monkeypatch):
    captured = {}

    class FakeMoire:
        def find(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(cli_main, "Moire", FakeMoire)
    args = build_parser().parse_args(
        [
            "moire",
            "find",
            "top.vasp",
            "bottom.vasp",
            "--max-length",
            "25.5",
            "--top-strain",
            "0.03",
            "--bottom-strain",
            "0.04",
            "--min-length",
            "3.5",
            "--max-atoms",
            "120",
            "--max-cell-aspect-ratio",
            "8.0",
            "--min-cell-angle",
            "30",
            "--max-cell-angle",
            "140",
            "--symmetric",
            "--progress",
            "--preview-limit",
            "7",
        ]
    )

    result = cli_main.execute_namespace(args)

    assert result is not None
    assert captured == {
        "top_poscar": "top.vasp",
        "bottom_poscar": "bottom.vasp",
        "max_length": 25.5,
        "top_strain": 0.03,
        "bottom_strain": 0.04,
        "min_length": 3.5,
        "max_atoms": 120,
        "max_aspect_ratio": 8.0,
        "min_cell_angle_deg": 30.0,
        "max_cell_angle_deg": 140.0,
        "symmetric": True,
        "progress": True,
        "preview_limit": 7,
    }


@pytest.mark.parametrize(
    "legacy_control, value",
    [
        ("--nindex", "12"),
        ("--angles", "10,20"),
        ("--workers", "2"),
        ("--max-pair-matches", "50"),
        ("--matrix-values", "1,0,0,1"),
    ],
)
def test_legacy_moire_find_controls_show_native_migration_guidance(capsys, legacy_control, value):
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(
            [
                "moire",
                "find",
                "top.vasp",
                "bottom.vasp",
                "--max-length",
                "12",
                "--top-strain",
                "0.01",
                "--bottom-strain",
                "0.02",
                legacy_control,
                value,
            ]
        )

    assert MIGRATION_MESSAGE in capsys.readouterr().err


def test_n_layer_commands_are_hidden_from_cli_help_and_choices():
    parser = build_parser()
    stages = _moire_stages(parser)

    for retired_stage in ("findn", "maken", "translaten"):
        assert retired_stage not in stages
        assert retired_stage not in parser.format_help()
    assert "findn" not in stages["find"].format_help()


def test_adsorbate_assemble_dispatches_native_gram_controls(monkeypatch):
    captured = {}

    class FakeMolecule:
        def assemble(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(cli_main, "Molecule", FakeMolecule)
    args = build_parser().parse_args(
        [
            "adsorbate",
            "assemble",
            "substrate.vasp",
            "--a-length",
            "12.0",
            "--b-length",
            "10.0",
            "--angle",
            "75.0",
            "--max-length",
            "30.0",
            "--top-strain",
            "0.02",
            "--bottom-strain",
            "0.03",
            "--preview-limit",
            "4",
        ]
    )

    result = cli_main.execute_namespace(args)

    assert result is not None
    assert captured == {
        "substrate_poscar": "substrate.vasp",
        "a_length": 12.0,
        "b_length": 10.0,
        "angle_deg": 75.0,
        "max_length": 30.0,
        "top_strain": 0.02,
        "bottom_strain": 0.03,
        "preview_limit": 4,
    }


def test_retired_n_layer_python_entrypoints_raise_the_consistent_error():
    entrypoints = (
        lambda: Supermoire().findn(),
        lambda: Supermoire().maken(),
        lambda: findn.run_findn(),
        lambda: maken.generate_from_results(),
        lambda: translaten.translaten(),
    )

    for call in entrypoints:
        with pytest.raises(NotImplementedError) as error:
            call()
        assert str(error.value) == N_LAYER_MESSAGE

"""The guided launcher must be able to request a rigid commensurate search."""

from __future__ import annotations

import pytest

from cellstine.cli.interactive import runner
from cellstine.cli.parsers import build_parser


def _drive(monkeypatch, answers):
    """Run the guided moire-search builder against a scripted set of answers."""

    supplied = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(supplied))
    argv = runner._build_moire_find()
    with pytest.raises(StopIteration):
        next(supplied)
    return argv


def test_the_guided_search_can_ask_for_a_rigid_commensurate_scan(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "top.vasp",  # top layer
            "bottom.vasp",  # bottom layer
            "26",  # maximum supercell length
            "y",  # keep both layers rigid
            "n",  # no twist-angle window
            "y",  # cap the atom count
            "400",  # atom cap
            "n",  # symmetry-preserving branch
            "n",  # fold each layer onto its own primitive in-plane cell
            "n",  # live progress
            "0",  # preview limit
        ],
    )

    assert argv[:4] == ["moire", "find", "top.vasp", "bottom.vasp"]
    assert "--top-strain" in argv and argv[argv.index("--top-strain") + 1] == "0"
    assert "--bottom-strain" in argv and argv[argv.index("--bottom-strain") + 1] == "0"
    assert argv[argv.index("--max-atoms") + 1] == "400"

    namespace = build_parser().parse_args(argv)
    assert namespace.top_strain == 0.0
    assert namespace.bottom_strain == 0.0
    assert namespace.max_atoms == 400


def test_the_guided_search_still_offers_strain_budgets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "top.vasp",
            "bottom.vasp",
            "30",
            "n",  # allow the layers to strain
            "0.02",  # top budget
            "0.01",  # bottom budget
            "n",  # no twist-angle window
            "n",  # no atom cap
            "n",  # symmetry-preserving branch
            "n",  # fold each layer onto its own primitive in-plane cell
            "n",  # live progress
            "10",  # preview limit
        ],
    )

    namespace = build_parser().parse_args(argv)
    assert namespace.top_strain == pytest.approx(0.02)
    assert namespace.bottom_strain == pytest.approx(0.01)
    assert namespace.max_atoms is None


def test_the_guided_search_can_ask_for_a_twist_angle_window(monkeypatch, tmp_path):
    """A user who wants a particular twist can say so, in either order."""

    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "top.vasp",
            "bottom.vasp",
            "30",
            "y",  # rigid
            "y",  # restrict the reported twist angles
            "14",  # given largest first
            "9",
            "n",  # no atom cap
            "n",  # symmetry-preserving branch
            "n",  # fold each layer onto its own primitive in-plane cell
            "n",  # live progress
            "10",  # preview limit
        ],
    )

    namespace = build_parser().parse_args(argv)
    assert namespace.min_twist_angle == pytest.approx(9.0)
    assert namespace.max_twist_angle == pytest.approx(14.0)


def test_the_guided_search_defaults_to_folding_each_layer_and_can_be_told_not_to(monkeypatch, tmp_path):
    """A supercell input is folded back onto its primitive layer unless refused."""

    monkeypatch.chdir(tmp_path)
    answers = [
        "top.vasp",
        "bottom.vasp",
        "30",
        "y",  # rigid
        "n",  # no twist-angle window
        "n",  # no atom cap
        "n",  # symmetry-preserving branch
        "n",  # fold each layer onto its own primitive in-plane cell
        "n",  # live progress
        "10",  # preview limit
    ]
    argv = _drive(monkeypatch, answers)
    assert "--keep-layer-cells" not in argv
    assert build_parser().parse_args(argv).keep_layer_cells is False

    kept = list(answers)
    kept[7] = "y"  # search the cells exactly as they were given
    argv = _drive(monkeypatch, kept)
    assert "--keep-layer-cells" in argv
    assert build_parser().parse_args(argv).keep_layer_cells is True

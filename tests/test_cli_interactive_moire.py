"""The guided launcher must be able to request a rigid commensurate search."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellstine.cli.interactive import runner
from cellstine.cli.parsers import build_parser


def _drive(monkeypatch, answers):
    """Run the guided moire-search builder against a scripted set of answers."""

    for name in ("top.vasp", "bottom.vasp"):
        Path(name).touch()
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
            "0",  # save all accepted candidates
            "y",  # adjust advanced options
            "n",  # no twist-angle window
            "y",  # cap the atom count
            "400",  # atom cap
            "n",  # symmetry-preserving branch
            "n",  # fold each layer onto its own primitive in-plane cell
            "0",  # preview limit
            "json",  # readable results
        ],
    )

    assert argv[:4] == ["moire", "search", "top.vasp", "bottom.vasp"]
    assert "--rigid" in argv
    assert argv[argv.index("--atoms") + 1] == "400"

    namespace = build_parser().parse_args(argv)
    assert namespace.rigid is True
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
            "n",  # use separate strain budgets
            "0.02",  # top budget
            "0.01",  # bottom budget
            "0",  # save all accepted candidates
            "n",  # keep the advanced defaults
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
            "0",  # save all accepted candidates
            "y",  # adjust advanced options
            "y",  # restrict the reported twist angles
            "14",  # given largest first
            "9",
            "n",  # no atom cap
            "n",  # symmetry-preserving branch
            "n",  # fold each layer onto its own primitive in-plane cell
            "10",  # preview limit
            "json",  # readable results
        ],
    )

    namespace = build_parser().parse_args(argv)
    assert namespace.twist == "9:14"


def test_the_guided_search_defaults_to_folding_each_layer_and_can_be_told_not_to(monkeypatch, tmp_path):
    """A supercell input is folded back onto its primitive layer unless refused."""

    monkeypatch.chdir(tmp_path)
    answers = [
        "top.vasp",
        "bottom.vasp",
        "30",
        "y",  # rigid
        "0",  # save all accepted candidates
        "y",  # adjust advanced options
        "n",  # no twist-angle window
        "n",  # no atom cap
        "n",  # symmetry-preserving branch
        "n",  # fold each layer onto its own primitive in-plane cell
        "10",  # preview limit
        "json",  # readable results
    ]
    argv = _drive(monkeypatch, answers)
    assert "--keep-layer-cells" not in argv
    assert build_parser().parse_args(argv).keep_layer_cells is False

    kept = list(answers)
    kept[9] = "y"  # search the cells exactly as they were given
    argv = _drive(monkeypatch, kept)
    assert "--keep-layer-cells" in argv
    assert build_parser().parse_args(argv).keep_layer_cells is True


def test_guided_common_strain_and_explicit_shortlist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(monkeypatch, [
        'top.vasp', 'bottom.vasp', '30', 'n', 'y', '.01', '25', 'n',
    ])
    namespace = build_parser().parse_args(argv)
    assert namespace.strain == pytest.approx(.01)
    assert namespace.top_strain is namespace.bottom_strain is None
    assert namespace.max_candidates == 25


def test_the_short_path_keeps_search_defaults_and_shows_progress(monkeypatch, tmp_path):
    """Answering only the common questions gives a complete, visible search."""

    monkeypatch.chdir(tmp_path)
    argv = _drive(monkeypatch, ["top.vasp", "bottom.vasp", "30", "n", "y", ".01", "0", "n"])
    namespace = build_parser().parse_args(argv)
    assert namespace.progress is True
    assert namespace.preview_limit == 10
    assert namespace.max_atoms is None
    assert namespace.twist is None


def test_rigid_short_path_preserves_atom_cap_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(monkeypatch, ["top.vasp", "bottom.vasp", "30", "y", "0", "n"])
    assert build_parser().parse_args(argv).max_atoms == 400

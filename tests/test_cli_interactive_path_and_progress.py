"""The last two code paths the suite never entered.

A trace of a full run showed two functions that no test reached: the guided
builder for the migration-path stage (`cli/interactive/build_path.py`) and the
progress printer of the moire workflow (`moire/moire.py`).  Both produce output
a user reads, so both are checked here on what they actually emit: the argument
list has to parse back into the command it claims to build, and the progress bar
has to advance once per completed stage and never past the end.
"""

from __future__ import annotations

import pytest

from cellstine.cli.interactive.build_path import _build_migration_path
from cellstine.cli.parsers import build_parser
from cellstine.moire.moire import _progress_printer


def _drive(monkeypatch, answers, group, subject):
    supplied = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(supplied))
    argv = _build_migration_path(group, subject=subject)
    with pytest.raises(StopIteration):
        next(supplied)
    return argv


def test_the_guided_defect_path_builds_a_command_that_parses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "start.vasp",  # initial structure
            "end.vasp",  # final structure
            "5",  # intermediate images
            "1",  # pair the atoms by the shortest path
        ],
        "defect",
        "a defect hop",
    )

    assert argv[:4] == ["defect", "path", "start.vasp", "end.vasp"]
    assert argv[argv.index("--images") + 1] == "5"
    assert "--no-match" not in argv

    namespace = build_parser().parse_args(argv)
    assert namespace.images == 5


def test_the_guided_adsorbate_path_can_keep_the_file_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "initial.vasp",
            "final.vasp",
            "3",
            "2",  # pair the atoms in file order
        ],
        "adsorbate",
        "a molecule or adatom diffusing",
    )

    assert argv[:4] == ["adsorbate", "path", "initial.vasp", "final.vasp"]
    assert "--no-match" in argv

    namespace = build_parser().parse_args(argv)
    assert namespace.images == 3
    assert namespace.match is False


def test_the_guided_path_rejects_an_image_count_outside_the_range(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    argv = _drive(
        monkeypatch,
        [
            "start.vasp",
            "end.vasp",
            "0",  # refused: at least one image is needed
            "120",  # refused: above the maximum
            "7",  # accepted
            "1",
        ],
        "defect",
        "a defect hop",
    )
    assert argv[argv.index("--images") + 1] == "7"


def test_the_progress_bar_advances_once_per_completed_stage(capsys):
    """The real message sequence of a search run fills the bar exactly once.

    Each stage announces itself before it starts and reports when it is done;
    only the reports may move the bar.  The four reports of a run -- read,
    found, wrote results, wrote manifest -- fill it exactly.
    """

    report = _progress_printer(total_steps=4)
    for stage, message in (
        ("read", "reading input structures"),
        ("read", "read structures in 0.031s"),
        ("search", "searching native Gram-form candidates"),
        ("search", "found 128 candidate(s) in 1.204s"),
        ("write", "writing results.json"),
        ("write", "wrote 64 of 128 candidate(s) to results.json in 0.010s"),
        ("manifest", "wrote manifest in 0.002s"),
    ):
        report(stage, message)

    lines = capsys.readouterr().out.strip().splitlines()
    bars = [line[line.index("[") + 1 : line.index("]")] for line in lines]
    filled = [bar.count("#") for bar in bars]

    assert filled == [0, 1, 1, 2, 2, 3, 4]
    assert all(len(bar) == 4 for bar in bars)
    assert lines[-1].endswith("manifest: wrote manifest in 0.002s")


def test_an_announcement_does_not_count_as_a_finished_stage(capsys):
    report = _progress_printer(total_steps=3)
    report("read", "reading input structures")
    report("write", "writing results.json")
    lines = capsys.readouterr().out.strip().splitlines()
    assert all(line.count("#") == 0 for line in lines)


def test_the_progress_bar_never_runs_past_its_end(capsys):
    report = _progress_printer(total_steps=2)
    for index in range(6):
        report("stage", f"wrote file {index} in 0.001s")

    lines = capsys.readouterr().out.strip().splitlines()
    bars = [line[line.index("[") + 1 : line.index("]")] for line in lines]
    assert [bar.count("#") for bar in bars] == [1, 2, 2, 2, 2, 2]
    assert all(bar.count("-") == 2 - bar.count("#") for bar in bars)

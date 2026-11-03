"""Guided mode keeps the user's answers when something goes wrong."""

from types import SimpleNamespace

import pytest

from cellstine.cli.argtypes import parse_float_list, parse_index_spec
from cellstine.cli.interactive import prompts, runner
from cellstine.cli.parsers import build_parser


def _script(monkeypatch, answers):
    supplied = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(supplied))
    return supplied


@pytest.fixture
def layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "layer.vasp"
    path.touch()
    return path


def test_a_mistyped_path_is_asked_again_rather_than_failing_at_the_end(monkeypatch, layer):
    _script(monkeypatch, ["missing.vasp", "layer.vasp"])
    assert prompts._prompt_path("Input", patterns=("*.vasp",), roots=()) == "layer.vasp"


def test_a_pasted_quoted_path_is_accepted(monkeypatch, layer):
    _script(monkeypatch, [f'"{layer}"'])
    assert prompts._prompt_path("Input", patterns=("*.json",), roots=()) == str(layer)


def test_a_folder_is_not_a_structure(monkeypatch, tmp_path, layer):
    (tmp_path / "folder").mkdir()
    _script(monkeypatch, ["folder", "layer.vasp"])
    assert prompts._prompt_path("Input", patterns=("*.xyz",), roots=()) == "layer.vasp"


def test_structures_in_the_working_folder_are_suggested_last(monkeypatch, tmp_path, layer):
    _script(monkeypatch, ["1"])
    chosen = prompts._prompt_path("Input", patterns=prompts.STRUCTURE_PATTERNS, roots=(tmp_path / "input",))
    assert chosen == str(layer.resolve())


def test_numbers_are_checked_when_typed(monkeypatch):
    _script(monkeypatch, ["-5", "0", "2.5"])
    assert prompts._prompt_float("Length", 1.0, positive=True) == 2.5
    _script(monkeypatch, ["-1", "0"])
    assert prompts._prompt_float("Angle", 1.0, minimum=0.0) == 0.0


def test_a_percentage_typed_as_a_strain_is_questioned(monkeypatch):
    _script(monkeypatch, ["2", "n", "0.02"])
    assert prompts._prompt_strain("Budget") == 0.02
    _script(monkeypatch, ["0.3", "y"])
    assert prompts._prompt_strain("Budget") == 0.3


def test_parsed_text_is_retried_and_kept_verbatim(monkeypatch):
    _script(monkeypatch, ["1,x", "1,3-5"])
    assert prompts._prompt_parsed("Indexes", "1", parse_index_spec) == "1,3-5"


def test_declining_a_command_returns_to_the_menu_instead_of_closing(monkeypatch, layer):
    calls = []
    monkeypatch.setattr(runner, "execute_namespace", lambda ns: calls.append(ns) or SimpleNamespace())
    monkeypatch.setattr(runner, "_print_result", lambda result: None)
    answers = _script(monkeypatch, [
        "shift", "layer.vasp", "1", "0.1,0.2", "n",  # read the command and decline it
        "shift", "layer.vasp", "1", "0.1,0.2", "y",  # the menu is still open
    ])
    monkeypatch.setattr(runner, "_follow_up", lambda *a: None)
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert len(calls) == 1
    with pytest.raises(StopIteration):
        next(answers)


@pytest.mark.parametrize("failure", [ValueError("no such thing"), FileNotFoundError("gone"), RuntimeError("bug"), KeyboardInterrupt()])
def test_a_failed_run_reopens_with_every_answer_kept(monkeypatch, layer, failure, capsys):
    outcomes = iter([failure, SimpleNamespace()])
    calls = []

    def execute(namespace):
        calls.append(namespace)
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "execute_namespace", execute)
    monkeypatch.setattr(runner, "_print_result", lambda result: None)
    monkeypatch.setattr(runner, "_follow_up", lambda *a: None)
    answers = _script(monkeypatch, [
        "shift", "layer.vasp", "1", "0.1,0.2", "y",
        "edit",  # change my answers
        "b",  # at the reopened confirmation
        "0.3,0.4",  # the shift vector is asked again, nothing earlier
        "y",
    ])
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert [call.shift_direct for call in calls] == [[0.1, 0.2], [0.3, 0.4]]
    with pytest.raises(StopIteration):
        next(answers)
    if not isinstance(failure, KeyboardInterrupt):
        assert str(failure) in capsys.readouterr().out


def test_a_command_the_parser_refuses_does_not_end_the_session(monkeypatch, layer, capsys):
    monkeypatch.setattr(runner, "_workflow_command", lambda *a, **k: ["moire", "shift", "x", "--shift-direct", "1"])
    _script(monkeypatch, ["y", "menu", "q"])
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert "refused" in capsys.readouterr().out


def test_a_builder_with_nothing_to_offer_returns_to_the_menu(monkeypatch, capsys):
    def nothing(*a, **k):
        raise ValueError("no adsorption sites")

    monkeypatch.setattr(runner, "_workflow_command", nothing)
    _script(monkeypatch, [])
    # The second pass through the loop has no more answers, which closes guided mode.
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    calls = []

    def twice(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise EOFError
        nothing()

    monkeypatch.setattr(runner, "_workflow_command", twice)
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert "no adsorption sites" in capsys.readouterr().out


def test_back_skips_previews_and_does_not_repeat_them(monkeypatch, capsys):
    computed = []

    def build():
        name = prompts._prompt("Name")
        preview = prompts._remember(lambda: computed.append(name) or f"preview of {name}")
        prompts.get_guided_ui().print("SCREEN")
        count = prompts._prompt_int("Count", 1)
        confirmed = prompts._prompt_yes_no("Run?", False)
        return preview, count, confirmed

    _script(monkeypatch, ["first", "2", "b", "3", "y"])
    assert prompts.run_guided_steps(build) == ("preview of first", 3, True)
    assert computed == ["first"]

    # Back at a later question replays this screen silently.
    def longer():
        result = build()
        return result, prompts._prompt("Note"), prompts._prompt("Last")

    capsys.readouterr()
    computed.clear()
    _script(monkeypatch, ["first", "2", "y", "x", "b", "z", "w"])
    assert prompts.run_guided_steps(longer)[1:] == ("z", "w")
    assert computed == ["first"]
    assert capsys.readouterr().out.count("SCREEN") == 1

    _script(monkeypatch, ["first", "2", "b", "b", "second", "4", "y"])
    assert prompts.run_guided_steps(build) == ("preview of second", 4, True)
    assert computed == ["first", "first", "second"]


def test_zero_candidates_offers_advice_rather_than_a_build(monkeypatch, capsys):
    monkeypatch.setattr(runner, "_choice", lambda *a, **k: pytest.fail("nothing to build"))
    assert runner._follow_up("moire", "search", SimpleNamespace(summary={"candidate_count": 0})) is None
    assert "No candidates matched" in capsys.readouterr().out


@pytest.mark.parametrize("group,stage,choice,attribute,artifacts,expected", [
    ("moire", "build", "view", "_build_structure_visualize", {"structures": ["a.vasp", "b.vasp"]}, {"subject": "a built moire stack", "structure": "a.vasp"}),
    ("moire", "build", "shift", "_build_moire_translate", {"structures": ["a.vasp"]}, {"poscar_path": "a.vasp"}),
    ("surface", "build", "place", "_build_adsorbate_place", {"slab_poscar": "s.vasp", "sites_json": "s.json"}, {"substrate": "s.vasp"}),
    ("interface", "match", "build", "_build_interface_build", {"matches_json": "m.json"}, {"matches": "m.json"}),
])
def test_finished_work_flows_into_the_next_step(monkeypatch, group, stage, choice, attribute, artifacts, expected):
    monkeypatch.setattr(runner, "_choice", lambda *a, **k: choice)
    calls = []
    monkeypatch.setattr(runner, attribute, lambda *a, **k: calls.append(k) or ["next"])
    assert runner._follow_up(group, stage, SimpleNamespace(artifacts=artifacts)) == ["next"]
    assert calls == [expected]


def test_a_stack_of_any_height_can_be_given_per_layer_values():
    assert parse_float_list("0.1") == [0.1]
    assert parse_float_list("0.1,0.2,0.3,0.4,0.5") == [0.1, 0.2, 0.3, 0.4, 0.5]
    namespace = build_parser().parse_args(
        ["moire", "stack-search", "a", "b", "c", "d", "e", "--length", "20", "--layer-strains", "0.1,0.1,0.1,0.1"]
    )
    assert namespace.layer_strains == [0.1] * 4
    namespace = build_parser().parse_args(["moire", "stack-build", "x", "--indexes", "1", "--interlayers", "3,3,3,3"])
    assert namespace.interlayers == [3.0] * 4


def test_the_printed_command_survives_a_shell():
    text = prompts._format_command(["view", "my file.vasp", "--view-direction", "(111)", "--x", "[110]", "--y", "1,2"])
    assert "'my file.vasp'" in text
    assert "'(111)'" in text
    assert "'[110]'" in text


def test_rich_shows_bracketed_text_literally():
    pytest.importorskip("rich")
    ui = prompts.RichGuidedUI()
    ui.console.record = True
    ui.console.quiet = False
    ui.print("A lattice direction [u v w] and [top]")
    ui.command_preview("Planned [draft]", ["view", "a.vasp", "--view-direction", "[uvw]"])
    table = ui._table()
    table.add_column("Option")
    table.add_row(ui._text("A lattice direction [u v w]"))
    ui.print(table)
    ui.title("Planned [x]", "Body with [red]brackets[/red]")
    text = ui.console.export_text()
    assert "\\[" not in text
    assert "[red]brackets[/red]" in text
    assert "[u v w] and [top]" in text
    assert "--view-direction" in text and "[uvw]" in text
    assert "Planned [draft]" in text
    assert text.count("[u v w]") == 2


def test_the_multilayer_budget_expands_one_value(monkeypatch, tmp_path):
    """The help text promises that one value applies to every upper layer."""

    from cellstine.moire.search import nlayer

    seen = []

    class Stop(Exception):
        pass

    def stop(**kwargs):
        seen.append(kwargs["top_strain"])
        raise Stop

    poscar = tmp_path / "layer.vasp"
    poscar.write_text(
        "graphene\n1.0\n2.46 0 0\n-1.23 2.130422 0\n0 0 15\nC\n2\nDirect\n"
        "0 0 0.5\n0.333333 0.666667 0.5\n"
    )
    monkeypatch.setattr(nlayer, "run_find", stop)
    with pytest.raises(Stop):
        nlayer.run_findn(
            base_poscar=str(poscar),
            upper_poscars=[str(poscar), str(poscar)],
            max_length=10.0,
            layer_strains=[0.03],
            output_root=str(tmp_path / "out"),
        )
    assert seen == [0.03]


def _saved_search(tmp_path, count):
    import numpy as np
    from cellstine.moire.search import gram, results

    config = gram.SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.)
    document = results.build_results_document(
        top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config,
        result=gram.search(config), symmetric_requested=False, symmetric_used=False,
    )
    document["candidates"] = [dict(document["candidates"][0], index=i+1) for i in range(count)]
    document["metadata"].update(candidates_found=count, candidates_recorded=count)
    path = tmp_path / "results.json"
    results.write_results(path, document)
    return path


def test_the_saved_candidate_count_uses_validated_rows(tmp_path):
    assert prompts._saved_candidate_count(str(_saved_search(tmp_path, 12))) == 12
    other = tmp_path / "other.json"
    other.write_text('{"schema": "something.else", "candidates_recorded": 3}', encoding="utf-8")
    assert prompts._saved_candidate_count(str(other)) is None
    assert prompts._saved_candidate_count(str(tmp_path / "missing.json")) is None


def test_build_indexes_beyond_the_saved_candidates_are_asked_again(monkeypatch, tmp_path):
    from cellstine.cli.interactive import build_moire

    monkeypatch.chdir(tmp_path)
    results = _saved_search(tmp_path, 12)
    answers = _script(monkeypatch, ["1,13", "0", "2,5-7", "3.35", "n", "n"])
    argv = prompts.run_guided_steps(lambda: build_moire._build_moire_make(results_file=str(results)))
    assert argv[argv.index("--indexes") + 1] == "2,5-7"
    with pytest.raises(StopIteration):
        next(answers)


def test_follow_up_failure_keeps_answers_without_repeating_completed_command(monkeypatch, layer):
    monkeypatch.setattr(runner, "_workflow_command", lambda *a, **k:
                        ["moire", "shift", str(layer), "--shift-direct", "0,0"])
    monkeypatch.setattr(runner, "_print_result", lambda _: None)
    calls = []

    def execute(namespace):
        calls.append(namespace)
        if len(calls) == 2:
            raise ValueError("try a different interlayer distance")
        return SimpleNamespace()

    def follow_up(group, stage, result):
        if stage == "build":
            return None
        distance = prompts._prompt_float("Interlayer distance", 3.35, positive=True)
        return ["moire", "build", "results.json", "--indexes", "1", "--interlayer-distance", str(distance)]

    monkeypatch.setattr(runner, "execute_namespace", execute)
    monkeypatch.setattr(runner, "_follow_up", follow_up)
    answers = _script(monkeypatch, ["y", "3.35", "y", "edit", "b", "4.2", "y"])
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert [call.stage for call in calls] == ["shift", "build", "build"]
    assert [call.interlayer_distance for call in calls[1:]] == [3.35, 4.2]
    with pytest.raises(StopIteration):
        next(answers)


def test_follow_up_builder_error_returns_to_menu(monkeypatch, layer, capsys):
    builds = []

    def build(*args, **kwargs):
        builds.append(True)
        if len(builds) > 1:
            raise prompts._QuitInteractive()
        return ["moire", "shift", str(layer), "--shift-direct", "0,0"]

    def follow_up(*args):
        raise ValueError("no supported adsorption site families")

    monkeypatch.setattr(runner, "_workflow_command", build)
    monkeypatch.setattr(runner, "_follow_up", follow_up)
    monkeypatch.setattr(runner, "execute_namespace", lambda _: SimpleNamespace())
    monkeypatch.setattr(runner, "_print_result", lambda _: None)
    _script(monkeypatch, ["y"])
    assert runner.run_interactive(group="moire", show_banner=False) == 0
    assert len(builds) == 2
    assert "That cannot continue: no supported adsorption site families" in capsys.readouterr().out

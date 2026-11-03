"""Shared guided navigation and explicit bilayer shortlisting."""

from types import SimpleNamespace

import pytest

from cellstine.cli.interactive import prompts, runner
from cellstine.cli.parsers import build_parser


def test_bilayer_shortlist_is_opt_in():
    parser = build_parser()
    base = ['moire', 'search', 'top.vasp', 'bottom.vasp', '--length', '20', '--rigid']
    assert parser.parse_args(base).max_candidates is None
    assert parser.parse_args(base + ['--max-candidates', '25']).max_candidates == 25
    with pytest.raises(SystemExit):
        parser.parse_args(base + ['--max-candidates', '0'])


def test_back_edits_previous_answer_without_reasking_earlier_steps(monkeypatch):
    answers = iter(['first', '2', 'b', '4', 'y'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    def build():
        name = prompts._prompt('Name')
        number = prompts._prompt_int('Count', 1)
        confirmed = prompts._prompt_yes_no('Run?', False)
        return name, number, confirmed
    assert prompts.run_guided_steps(build) == ('first', 4, True)
    with pytest.raises(StopIteration):
        next(answers)


def test_back_through_numeric_validation_returns_to_previous_question(monkeypatch):
    answers = iter(['first', 'invalid', 'b', 'changed', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert prompts.run_guided_steps(lambda: (prompts._prompt('Name'), prompts._prompt_int('Count', 1))) == ('changed', 3)


def test_rich_confirmation_accepts_back():
    ui = prompts.RichGuidedUI.__new__(prompts.RichGuidedUI)
    ui._prompt = SimpleNamespace(ask=lambda *a, **k: 'b')
    with pytest.raises(prompts._BackInteractive):
        ui.confirm('Run?', False)


def test_boron_symbol_is_not_a_navigation_command(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: 'B')
    assert prompts.PlainGuidedUI().prompt('Species') == 'B'


def test_continue_to_build_uses_normal_builder(monkeypatch):
    monkeypatch.setattr(runner, '_choice', lambda *a, **k: 'make')
    calls = []
    monkeypatch.setattr(runner, '_build_moire_make', lambda **kw: calls.append(kw) or ['moire', 'build', 'manifest.json'])
    assert runner._follow_up('moire', 'search', SimpleNamespace(manifest_path='manifest.json')) == ['moire', 'build', 'manifest.json']
    assert calls == [{'results_file': 'manifest.json'}]


def test_back_at_run_confirmation_edits_settings_without_executing_twice(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'saved.json').touch()
    answers = iter(['build', 'saved.json', '1', '3.35', 'n', 'n', 'b', 'y', '2', 'y'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    calls = []
    monkeypatch.setattr(runner, 'execute_namespace', lambda ns: calls.append(ns) or SimpleNamespace())
    monkeypatch.setattr(runner, '_print_result', lambda result: None)
    assert runner.run_interactive(group='moire', show_banner=False) == 0
    assert len(calls) == 1
    assert calls[0].workers == 2
    with pytest.raises(StopIteration):
        next(answers)


def test_group_only_back_opens_main_menu(monkeypatch):
    answers = iter(['b', 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert runner.run_interactive(group='moire', show_banner=False) == 0


@pytest.mark.parametrize('group,stage,attribute,kwargs', [
    ('surface', 'build', '_build_interface_sites', {'slab': 'slab.vasp'}),
    ('defect', 'analyse', '_build_defect_generate', {'source': 'manifest.json'}),
])
def test_other_continuations_use_normal_builders(monkeypatch, group, stage, attribute, kwargs):
    monkeypatch.setattr(runner, '_choice', lambda *a, **k: 'sites' if group == 'surface' else 'generate')
    calls = []
    monkeypatch.setattr(runner, attribute, lambda **kw: calls.append(kw) or ['command'])
    result = SimpleNamespace(manifest_path='manifest.json', artifacts={'slab_poscar': 'slab.vasp'})
    assert runner._follow_up(group, stage, result) == ['command']
    assert calls == [kwargs]


def test_search_dispatch_passes_explicit_limit_and_defaults_to_all(monkeypatch):
    import importlib
    import inspect
    main = importlib.import_module('cellstine.cli.main')
    from cellstine.moire.moire import Moire
    # Preserve the direct Python API's existing shortlist default. The CLI
    # explicitly forwards None to save all accepted candidates by default.
    assert inspect.signature(Moire.find).parameters['max_candidates'].default == 500
    calls = []
    class Workflow:
        def find(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(main, '_workflow', lambda name: Workflow)
    base = ['moire', 'search', 'top.vasp', 'bottom.vasp', '--length', '20', '--rigid']
    for extra in ([], ['--max-candidates', '25']):
        main.execute_namespace(build_parser().parse_args(base + extra))
    assert [c['max_candidates'] for c in calls] == [None, 25]


def test_guided_defect_preview_does_not_create_run_files(monkeypatch, tmp_path, silicon_poscar):
    from cellstine.cli.interactive.build_defect import _defect_analysis_from_preview
    monkeypatch.chdir(tmp_path)
    _defect_analysis_from_preview(str(silicon_poscar), 'bulk', 'native', 'top')
    assert not (tmp_path/'runs').exists()
    assert not (tmp_path/'output').exists()


def test_file_selection_honors_explicit_default(monkeypatch, tmp_path):
    first = tmp_path/'first.vasp'
    second = tmp_path/'second.vasp'
    first.touch()
    second.touch()
    monkeypatch.setattr(prompts, '_find_candidates', lambda *a: [second, first])
    monkeypatch.setattr('builtins.input', lambda _: '')
    assert prompts._prompt_path('Input', patterns=('*.vasp',), roots=(tmp_path,), default=str(first)) == str(first)


def test_back_from_manual_path_returns_to_file_list(monkeypatch, tmp_path):
    source = tmp_path/'source.vasp'
    source.touch()
    answers = iter(['99', 'm', 'b', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert prompts.run_guided_steps(lambda: prompts._prompt_path('Input', patterns=('*.vasp',), roots=(tmp_path,))) == str(source)


def test_quit_works_at_numeric_prompt_and_nonfinite_numbers_retry(monkeypatch):
    answers = iter(['nan', 'inf', '2.5', 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert prompts._prompt_float('Length', 1.) == 2.5
    with pytest.raises(prompts._QuitInteractive):
        prompts._prompt_int('Count', 1)

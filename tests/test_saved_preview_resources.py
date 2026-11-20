"""Reuse a validated preview's actual row count without another JSON read."""

import copy
import json

import numpy as np
import pytest

from cellstine.core import previews
from cellstine.moire.search import gram, results


@pytest.fixture
def document():
    config = gram.SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.)
    return results.build_results_document(
        top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config,
        result=gram.search(config), symmetric_requested=False, symmetric_used=False,
    )


@pytest.mark.parametrize("count", [0, 1, 101])
def test_preview_count_comes_from_loaded_rows_once(monkeypatch, tmp_path, document, count):
    payload = copy.deepcopy(document)
    payload["candidates"] = [dict(document["candidates"][0], index=i+1) for i in range(count)]
    # A producer's declared metadata is not proof of actual selectable rows.
    payload["metadata"]["candidates_recorded"] = 999
    calls = []

    def read(path):
        calls.append(path)
        return payload

    monkeypatch.setattr(previews, "read_results", read)
    source = tmp_path / "results.json"
    text, actual = previews._preview_moire_results_with_count(source, limit=2)
    assert actual == count
    assert calls == [source.resolve()]
    assert text == previews.preview_moire_results_file(source, limit=2)


def test_multi_result_manifest_has_no_single_selection_bound(monkeypatch, tmp_path, document):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    calls = []
    monkeypatch.setattr(previews, "_iter_manifest_result_paths", lambda _: iter(paths))

    def read(path):
        calls.append(path)
        return document

    monkeypatch.setattr(previews, "read_results", read)
    text, count = previews._preview_moire_results_with_count(tmp_path / "manifest.json", limit=2)
    assert count is None
    assert calls == paths
    assert "a.json" in text and "b.json" in text


@pytest.mark.parametrize("layout", ["nested", "split-token", "duplicate", "misreported"])
def test_guided_bound_uses_actual_rows_for_valid_json(tmp_path, document, layout):
    from cellstine.cli.interactive import prompts

    document["candidates"] = [dict(document["candidates"][0], index=i+1) for i in range(101)]
    document["metadata"].update(candidates_found=101, candidates_recorded=101)
    if layout == "nested":
        document["metadata"] = {"stage_stats": {"candidates_recorded": 1},
                                **{key: value for key, value in document["metadata"].items() if key != "stage_stats"}}
    if layout == "split-token":
        document["metadata"] = {"padding": "", **document["metadata"]}
        text = json.dumps(document, indent=2)
        index = text.index("101", text.index('"candidates_recorded"'))
        document["metadata"]["padding"] = "x" * (65535-index)
    if layout == "misreported":
        document["metadata"].update(candidates_found=999, candidates_recorded=999)
    text = json.dumps(document, indent=2)
    if layout == "split-token":
        assert text[65535:65538] == "101"
    if layout == "duplicate":
        text = '{"metadata":{"candidates_recorded":1},' + text[1:]
    path = tmp_path / "results.json"
    path.write_text(text, encoding="utf-8", newline="\n")
    assert len(results.read_results(path)["candidates"]) == 101
    assert prompts._saved_candidate_count(str(path)) == 101


def test_build_reuses_preview_read_when_back_replays_answers(monkeypatch, tmp_path, document):
    from cellstine.cli.interactive import build_moire, prompts

    path = results.write_results(tmp_path / "results.json", document)
    original = previews.read_results
    calls = []

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(previews, "read_results", read)
    answers = iter(["1", "3.35", "n", "b", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    argv = prompts.run_guided_steps(lambda: build_moire._build_moire_make(results_file=str(path)))
    assert argv[argv.index("--indexes")+1] == "1"
    assert calls == [path]
    with pytest.raises(StopIteration):
        next(answers)


def test_empty_saved_results_have_no_selectable_index(tmp_path, document):
    from cellstine.cli.interactive import prompts

    document["candidates"] = []
    document["metadata"].update(candidates_found=0, candidates_recorded=0)
    path = results.write_results(tmp_path / "results.json", document)
    count = prompts._saved_candidate_count(str(path))
    assert count == 0
    with pytest.raises(ValueError, match="holds no candidates"):
        prompts._index_parser(count)("1")

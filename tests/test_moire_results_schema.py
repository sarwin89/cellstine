"""The moire results document and the rules its validator enforces.

The results file is the contract between `moire search` and every stage that
reads it, so the validator is what stops a damaged or hand-edited document from
being built into structures.  Each test below breaks exactly one rule of the
schema and checks that the document is refused, and the healthy document is
checked to survive a write/read round trip unchanged.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.moire.search import results as results_mod
from cellstine.moire.search.find import run_find


@pytest.fixture(scope="module")
def document(tmp_path_factory, graphene_poscar, hbn_poscar) -> dict:
    workspace = tmp_path_factory.mktemp("results")
    run = run_find(
        top_poscar=str(graphene_poscar),
        bottom_poscar=str(hbn_poscar),
        max_length=12.0,
        top_strain=0.02,
        bottom_strain=0.02,
        output_root=str(workspace),
    )
    return results_mod.read_results(str(run.result_path))


def _broken(document: dict, mutate) -> dict:
    copied = copy.deepcopy(document)
    mutate(copied)
    return copied


def test_a_healthy_document_round_trips_through_the_writer(document, tmp_path):
    path = results_mod.write_results(str(tmp_path / "results.json"), document)
    reread = results_mod.read_results(str(path))
    assert reread == document
    # The file is plain JSON with no NaN or Infinity tokens.
    text = Path(path).read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == document


def test_validation_returns_a_detached_document(document):
    validated = results_mod.validate_results(document)
    validated["candidates"][0]["index"] = 999
    assert document["candidates"][0]["index"] == 1


def test_the_document_carries_the_schema_and_version(document):
    assert document["schema"] == results_mod.SCHEMA
    assert document["version"] == results_mod.VERSION
    assert document["candidates"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda doc: doc.__setitem__("schema", "something.else"), id="schema"),
        pytest.param(lambda doc: doc.__setitem__("version", 99), id="version"),
        pytest.param(lambda doc: doc.__setitem__("candidates", {}), id="candidates-not-a-list"),
        pytest.param(lambda doc: doc["search"].pop("max_length"), id="missing-search-field"),
        pytest.param(lambda doc: doc["search"].__setitem__("max_length", -1.0), id="negative-length"),
        pytest.param(lambda doc: doc["search"].__setitem__("top_strain", 0.0) or doc["search"].__setitem__("bottom_strain", 0.0), id="zero-budgets"),
        pytest.param(lambda doc: doc["metadata"].__setitem__("engine", "other"), id="engine"),
        pytest.param(lambda doc: doc["candidates"][0].pop("strain"), id="missing-candidate-field"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("index", 7), id="index-out-of-order"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_matrix", [[1, 2], [2, 4]]), id="singular-matrix"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_matrix", [[1.5, 0.0], [0.0, 1.0]]), id="non-integer-matrix"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_matrix", [[True, 0], [0, 1]]), id="boolean-matrix-entry"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_gram", [1.0, 2.0]), id="wrong-gram-shape"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_gram", [1.0, 5.0, 1.0]), id="indefinite-gram"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("moire_gamma_deg", 180.0), id="degenerate-cell-angle"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("moire_a", 0.0), id="zero-cell-length"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("sharing_fraction", 1.5), id="sharing-out-of-range"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("atom_count", 3), id="atom-count-mismatch"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("shared_lattice", [[1.0, 2.0], [2.0, 4.0]]), id="singular-shared-cell"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("top_layer_strain", [0.5, 0.5]), id="strain-over-budget"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("pareto_optimal", "yes"), id="non-boolean-flag"),
        pytest.param(lambda doc: doc["candidates"][0].__setitem__("rank", 0), id="non-positive-rank"),
    ],
)
def test_a_broken_document_is_refused(document, mutate):
    with pytest.raises(ValueError):
        results_mod.validate_results(_broken(document, mutate))


def test_layer_strains_must_add_up_to_the_relative_strain(document):
    broken = copy.deepcopy(document)
    candidate = broken["candidates"][0]
    candidate["top_layer_strain"] = [value + 1e-3 for value in candidate["top_layer_strain"]]
    with pytest.raises(ValueError, match="relative strain"):
        results_mod.validate_results(broken)


def test_a_non_finite_number_is_refused(document):
    broken = copy.deepcopy(document)
    broken["candidates"][0]["shared_lattice"][0][0] = math.inf
    with pytest.raises(ValueError):
        results_mod.validate_results(broken)


def test_writing_refuses_a_broken_document(document, tmp_path):
    broken = _broken(document, lambda doc: doc["candidates"][0].__setitem__("atom_count", 1))
    target = tmp_path / "broken.json"
    with pytest.raises(ValueError):
        results_mod.write_results(str(target), broken)
    assert not target.exists()


def test_legacy_positional_results_are_refused(tmp_path):
    legacy = tmp_path / "results.dat"
    legacy.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="legacy positional"):
        results_mod.read_results(str(legacy))


def test_every_candidate_of_a_real_search_satisfies_the_schema(document):
    """The rules are not vacuous: they are checked against the real records."""

    budgets = float(document["search"]["top_strain"]) + float(document["search"]["bottom_strain"])
    for index, candidate in enumerate(document["candidates"], start=1):
        assert candidate["index"] == index
        assert candidate["atom_count"] == candidate["top_atom_count"] + candidate["bottom_atom_count"]
        relative = np.asarray(candidate["strain"], dtype=float)
        top = np.asarray(candidate["top_layer_strain"], dtype=float)
        bottom = np.asarray(candidate["bottom_layer_strain"], dtype=float)
        assert np.allclose(top - bottom, relative, atol=1e-9, rtol=1e-7)
        assert float(np.max(np.abs(relative))) <= budgets + 1e-9
        for name in ("top_matrix", "bottom_matrix"):
            matrix = np.asarray(candidate[name], dtype=float)
            assert abs(float(np.linalg.det(matrix))) >= 1.0 - 1e-9


def test_the_twist_window_is_recorded_and_checked(document):
    """The requested twist window is part of the search record, and validated.

    A document written without a window -- as every earlier one was -- is still
    a valid document, because an absent window means no restriction at all.
    """

    assert document["search"]["min_twist_angle_deg"] is None
    assert document["search"]["max_twist_angle_deg"] is None

    without = copy.deepcopy(document)
    del without["search"]["min_twist_angle_deg"]
    del without["search"]["max_twist_angle_deg"]
    results_mod.validate_results(without)

    windowed = copy.deepcopy(document)
    windowed["search"]["min_twist_angle_deg"] = 5.0
    windowed["search"]["max_twist_angle_deg"] = 20.0
    results_mod.validate_results(windowed)

    with pytest.raises(ValueError):
        results_mod.validate_results(
            _broken(document, lambda item: item["search"].update(min_twist_angle_deg=20.0, max_twist_angle_deg=5.0))
        )
    with pytest.raises(ValueError):
        results_mod.validate_results(
            _broken(document, lambda item: item["search"].update(max_twist_angle_deg=200.0))
        )


def test_a_windowed_search_records_the_window_it_was_given(tmp_path, graphene_poscar):
    run = run_find(
        top_poscar=str(graphene_poscar),
        bottom_poscar=str(graphene_poscar),
        max_length=25.0,
        top_strain=0.0,
        bottom_strain=0.0,
        min_twist_angle_deg=9.0,
        max_twist_angle_deg=14.0,
        output_root=str(tmp_path),
    )
    written = results_mod.read_results(str(run.result_path))
    assert written["search"]["min_twist_angle_deg"] == pytest.approx(9.0)
    assert written["search"]["max_twist_angle_deg"] == pytest.approx(14.0)
    assert written["candidates"], "graphene twists in this window exist"
    for candidate in written["candidates"]:
        assert 9.0 <= abs(float(candidate["angle_deg"])) <= 14.0

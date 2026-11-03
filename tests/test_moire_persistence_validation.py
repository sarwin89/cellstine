"""Persistence must handle unfolded searches and retain scalar validation errors."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cellstine.moire.search.gram import SearchConfig, search
from cellstine.moire.search import results


def _document(config):
    return results.build_results_document(
        top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config, result=search(config),
        symmetric_requested=False, symmetric_used=False, created_at="2000-01-01T00:00:00+00:00",
    )


@pytest.mark.parametrize("basis", [np.eye(2) * 3., np.array([[3., -1.5], [0., 1.5 * np.sqrt(3.)]])])
def test_unfolded_search_round_trip_keeps_physical_period_and_raw_angles(tmp_path, basis):
    config = SearchConfig(basis, basis, 4.5, 0., 0., fold_symmetry=False)
    result = search(config)
    period = float(np.degrees(config.angle_period_radians))
    assert np.any(np.abs(result.twist_degrees) > .5 * period + 1e-6)
    document = _document(config)
    assert document["search"]["angle_period_deg"] == period
    assert document["search"]["fold_symmetry"] is False
    np.testing.assert_array_equal([row["angle_deg"] for row in document["candidates"]], result.twist_degrees)
    destination = results.write_results(tmp_path / "unfolded.json", document)
    assert results.read_results(destination) == document


@pytest.mark.parametrize("fold,angle", [(True, 46.), (False, 180.1)])
def test_folded_and_unfolded_angle_bounds_still_reject_invalid_values(fold, angle):
    document = _document(SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.))
    document["search"]["fold_symmetry"] = fold
    document["candidates"][0]["angle_deg"] = angle
    with pytest.raises(ValueError, match="angle_deg must be folded into the fundamental range"):
        results.validate_results(document)


def _verdict(operation):
    try:
        operation()
    except (ValueError, TypeError, OverflowError) as error:
        return type(error), str(error)
    return None


def test_plain_fast_validation_preserves_scalar_verdicts_and_first_errors():
    # The full scalar validator is the independent oracle, including accepted
    # unusual values that must fall back rather than gain stricter type rules.
    document = _document(SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.))
    pristine = copy.deepcopy(document)
    rng = np.random.default_rng(1307)
    fields = list(document["candidates"][0])
    replacements = [None, True, False, 0, -1, 10**400, float("nan"), float("inf"),
                    (0., 0.), [[1, 0], [0, 0]], [[2**32, 0], [0, 2**32]],
                    np.float64(0.), np.int64(1), [0., 0., 0.], "not numeric"]
    for attempt in range(300):
        candidate = copy.deepcopy(document["candidates"][0])
        for _ in range(1 + attempt % 2):
            field = fields[int(rng.integers(len(fields)))]
            if attempt % 7 == 0:
                candidate.pop(field, None)
            else:
                candidate[field] = replacements[int(rng.integers(len(replacements)))]
        expected = _verdict(lambda: results._validate_candidate(candidate, 1, document["search"]))
        assert not results._plainly_valid_candidate(candidate, 1, document["search"]) or expected is None
        payload = {**document, "candidates": [candidate]}
        assert _verdict(lambda: results.validate_results(payload)) == expected, attempt
    assert document == pristine

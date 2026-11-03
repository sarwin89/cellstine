"""Imported fast paths must retain dev's paired-mirror equivalence policy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellstine.moire.search.gram import SearchConfig
from cellstine.moire.search.gram_report import (
    _canonical_twist,
    _folded_twist_magnitude,
    _twist_window_mask,
)


@pytest.mark.parametrize("orientation", [20., 37.])
def test_early_window_keeps_candidates_reached_by_paired_mirrors(orientation):
    angle = np.deg2rad(orientation)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    config = SearchConfig(
        top_basis=np.diag([3., 5.]), bottom_basis=rotation @ np.diag([4., 7.]),
        max_length=12., top_strain=.01, bottom_strain=.01,
        min_twist_angle_deg=orientation - 1., max_twist_angle_deg=orientation + 1.,
    )
    raw = np.deg2rad([3. * orientation])
    identity = np.eye(2, dtype=np.int64)[None]
    _, _, folded = _canonical_twist(config, identity, identity, raw)
    assert np.degrees(np.abs(folded[0])) == pytest.approx(orientation, abs=1e-12)
    assert np.allclose(_folded_twist_magnitude(config, raw), np.abs(folded), atol=1e-12, rtol=0.)
    assert _twist_window_mask(config, raw)[0]


def test_result_validation_keeps_python_integer_nonsingularity(tmp_path):
    from cellstine.moire.search.gram import search
    from cellstine.moire.search.results import build_results_document, validate_results

    config = SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.)
    result = search(config)
    top = result.top_matrices.copy()
    # Both coefficients fit int64, but their nonzero determinant wraps to zero.
    top[0] = np.diag([2**32, 2**32])
    result = replace(result, top_matrices=top)
    document = build_results_document(
        top_poscar=tmp_path / "top.vasp", bottom_poscar=tmp_path / "bottom.vasp",
        config=config, result=result, symmetric_requested=False, symmetric_used=False,
    )
    assert document["candidates"][0]["top_matrix"] == [[2**32, 0], [0, 2**32]]
    assert validate_results(document) == document

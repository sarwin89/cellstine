"""An independent reference with no admitted vectors reports no candidates."""

import numpy as np
import pytest

from reference_moire import ReferenceConfig, _short_vectors, _right_handed_pairs, reference_search


@pytest.mark.parametrize("basis", [np.eye(2), np.array([[3., .7], [0., 4.]])])
@pytest.mark.parametrize("radius", [0., .1])
def test_empty_short_vectors_keep_two_coordinate_columns(basis, radius):
    vectors = _short_vectors(basis, radius)
    assert vectors.shape == (0, 2)
    assert vectors.dtype == np.int64
    assert _right_handed_pairs(vectors).shape == (0, 2, 2)


@pytest.mark.parametrize("top,bottom", [(np.eye(2), np.eye(2)),
                                       (np.eye(2), 10 * np.eye(2)),
                                       (10 * np.eye(2), np.eye(2))])
def test_no_vectors_in_either_layer_means_no_reference_candidates(top, bottom):
    assert reference_search(ReferenceConfig(top, bottom, .1, 0., 0.)) == []

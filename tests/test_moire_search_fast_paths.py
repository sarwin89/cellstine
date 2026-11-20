"""The faster search paths must give exactly what the slower ones gave.

Each fast path replaces an exact computation with another exact computation:
gcd-free keys for primitive pairs, the twist window applied before folding,
atom caps applied while the tables are generated, and results written with the
C JSON encoder.  These tests pin each one to the computation it replaced.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cellstine.core.symmetry2d import lattice_point_group, proper_subgroup
from cellstine.moire.search.gram import SearchConfig, _common_basis_keys, search
from cellstine.moire.search.gram_lattice import (
    _first_per_key,
    _fold_sublattices,
    _hermite_normal_form,
    _lexicographic_minimum,
)
from cellstine.moire.search.gram_pairs import (
    _canonical_pair_keys,
    _pair_orbit_keys,
    _primitive_orbit_keys,
    _primitive_pair_keys,
)
from cellstine.moire.search.gram_report import coincidence_index
from cellstine.moire.search.results import (
    build_results_document,
    read_results,
    write_results,
)
from conftest import hexagonal_basis


def _partition(keys: np.ndarray) -> list[tuple[int, ...]]:
    """Group labels of each row, numbered by first appearance."""

    labels: dict[bytes, int] = {}
    return [labels.setdefault(np.ascontiguousarray(row).tobytes(), len(labels)) for row in keys]


def _random_pairs(seed: int, count: int, *, primitive: bool) -> tuple[np.ndarray, np.ndarray]:
    """Random nonsingular pairs, with each one repeated under a few relabellings."""

    rng = np.random.default_rng(seed)
    unimodular = [
        np.array(matrix)
        for matrix in ([[1, 0], [0, 1]], [[1, 1], [0, 1]], [[0, -1], [1, 0]], [[2, 1], [1, 1]], [[1, 0], [-3, 1]], [[0, 1], [1, 0]])
    ]
    tops, bottoms = [], []
    while len(tops) < count:
        top = rng.integers(-6, 7, size=(2, 2))
        bottom = rng.integers(-6, 7, size=(2, 2))
        if round(np.linalg.det(top)) == 0 or round(np.linalg.det(bottom)) == 0:
            continue
        if primitive and coincidence_index(top[None], bottom[None])[0] != 1:
            continue
        for relabel in unimodular[: 1 + int(rng.integers(0, len(unimodular)))]:
            tops.append(top @ relabel)
            bottoms.append(bottom @ relabel)
    return np.array(tops, dtype=np.int64), np.array(bottoms, dtype=np.int64)


def test_primitive_keys_partition_pairs_exactly_as_hermite_keys_do():
    top, bottom = _random_pairs(0, 400, primitive=True)
    assert np.all(coincidence_index(top, bottom) == 1)
    assert _partition(_primitive_pair_keys(top, bottom)) == _partition(_canonical_pair_keys(top, bottom))


def test_mixed_keys_never_merge_a_primitive_pair_with_an_imprimitive_one():
    top, bottom = _random_pairs(1, 400, primitive=False)
    assert np.any(coincidence_index(top, bottom) != 1)
    assert _partition(_common_basis_keys(top, bottom)) == _partition(_canonical_pair_keys(top, bottom))


@pytest.mark.parametrize("basis", [hexagonal_basis(2.46), np.eye(2), np.array([[3.1, 0.4], [0.0, 2.2]])])
def test_primitive_orbit_keys_partition_pairs_exactly_as_hermite_orbit_keys_do(basis):
    group = lattice_point_group(basis)
    top, bottom = _random_pairs(2, 300, primitive=True)
    # Add symmetry images so that the orbit really does merge rows.
    images = [(g @ top, h @ bottom) for g in group[:3] for h in group[-2:]]
    top = np.concatenate([top] + [image[0] for image in images])
    bottom = np.concatenate([bottom] + [image[1] for image in images])
    fast = _partition(_primitive_orbit_keys(top, bottom, group, group))
    slow = _partition(_pair_orbit_keys(top, bottom, group, group))
    assert fast == slow
    assert len(set(fast)) < len(fast)


@pytest.mark.parametrize("top_kind,bottom_kind", [
    ("full", "proper"), ("proper", "full"), ("identity", "full"),
    ("full", "mirror"), ("mirror", "mirror"),
])
def test_primitive_orbit_keys_preserve_asymmetric_layer_group_partitions(top_kind, bottom_kind):
    group = lattice_point_group(np.eye(2))
    identity = np.eye(2, dtype=np.int64)
    mirror = np.diag([1, -1])
    groups = {"full": group, "proper": proper_subgroup(group),
              "identity": identity[None], "mirror": np.stack([identity, mirror])}
    top_group, bottom_group = groups[top_kind], groups[bottom_kind]
    top, bottom = _random_pairs(71, 120, primitive=True)
    # Exercise positive and negative top determinants as well as each allowed
    # matched-sign layer image, without assuming both layers have the same group.
    images = [(left @ top, right @ bottom) for left in top_group for right in bottom_group
              if round(np.linalg.det(left)) == round(np.linalg.det(right))]
    top = np.concatenate([top] + [image[0] for image in images])
    bottom = np.concatenate([bottom] + [image[1] for image in images])
    assert np.any(np.linalg.det(top) < 0)
    assert _partition(_primitive_orbit_keys(top, bottom, top_group, bottom_group)) == _partition(
        _pair_orbit_keys(top, bottom, top_group, bottom_group))


def test_incremental_sublattice_folding_matches_folding_all_images_at_once():
    group = lattice_point_group(hexagonal_basis(2.46))
    rng = np.random.default_rng(3)
    first = rng.integers(-9, 10, size=(2000, 2))
    second = rng.integers(-9, 10, size=(2000, 2))
    keep = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] > 0
    first, second = first[keep], second[keep]
    stacked = np.stack([_hermite_normal_form(first @ g.T, second @ g.T) for g in group])
    expected = _first_per_key(_lexicographic_minimum(stacked))
    np.testing.assert_array_equal(_fold_sublattices(first, second, group), expected)


def _config(**changes) -> SearchConfig:
    basis = hexagonal_basis(2.46)
    config = SearchConfig(basis, basis, 40.0, 0.0, 0.0, top_atoms=2, bottom_atoms=2)
    return replace(config, **changes)


def _keyed(result) -> dict[bytes, int]:
    return {np.ascontiguousarray(key).tobytes(): offset for offset, key in enumerate(result.canonical_keys)}


@pytest.mark.parametrize("window", [(0.0, 5.0), (9.0, 14.0), (21.786789298261458, 21.786789298261458), (25.0, None)])
def test_the_twist_window_keeps_exactly_the_unrestricted_candidates_inside_it(window):
    lower, upper = window
    everything = search(_config())
    restricted = search(_config(min_twist_angle_deg=lower, max_twist_angle_deg=upper))
    magnitude = np.abs(everything.twist_degrees)
    inside = (magnitude >= lower - 1e-5) & (magnitude <= (180.0 if upper is None else upper) + 1e-5)
    assert len(restricted) == int(inside.sum()) > 0
    reference = _keyed(everything)
    for offset, key in enumerate(restricted.canonical_keys):
        match = reference[np.ascontiguousarray(key).tobytes()]
        np.testing.assert_array_equal(restricted.top_matrices[offset], everything.top_matrices[match])
        assert restricted.twist_degrees[offset] == everything.twist_degrees[match]


def test_an_atom_cap_keeps_exactly_the_uncapped_candidates_below_it():
    strained = _config(top_strain=0.01, bottom_strain=0.01, max_length=30.0)
    everything = search(strained)
    capped = search(replace(strained, max_atoms=300))
    expected = {key for key, offset in _keyed(everything).items() if everything.atom_counts[offset] <= 300}
    assert set(_keyed(capped)) == expected
    assert capped.stats["n_top_rows_unfolded"] < everything.stats["n_top_rows_unfolded"]


def test_results_are_written_in_the_same_bytes_as_the_streaming_encoder(tmp_path):
    config = _config(max_length=20.0)
    result = search(config)
    document = build_results_document(
        top_poscar=tmp_path / "top.vasp",
        bottom_poscar=tmp_path / "bottom.vasp",
        config=config,
        result=result,
        symmetric_requested=False,
        symmetric_used=False,
        created_at="2026-11-02T00:00:00+00:00",
    )
    fast = write_results(tmp_path / "fast.json", document, validated=True)
    checked = write_results(tmp_path / "checked.json", document)
    with (tmp_path / "stream.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    reference = (tmp_path / "stream.json").read_bytes()
    assert fast.read_bytes() == reference
    assert checked.read_bytes() == reference
    assert read_results(fast) == document


def _document(tmp_path, config, result, **options):
    return build_results_document(
        top_poscar=tmp_path / "top.vasp",
        bottom_poscar=tmp_path / "bottom.vasp",
        config=config,
        result=result,
        symmetric_requested=False,
        symmetric_used=False,
        created_at="2026-11-02T00:00:00+00:00",
        **options,
    )


@pytest.mark.parametrize("chunk", [1, 2, 7, 4096])
def test_chunked_writing_gives_the_bytes_of_one_dumps_call(tmp_path, monkeypatch, chunk):
    from cellstine.moire.search import results

    config = _config(top_strain=0.01, bottom_strain=0.01, max_length=20.0)
    document = _document(tmp_path, config, search(config))
    assert len(document["candidates"]) > 7
    monkeypatch.setattr(results, "_CANDIDATE_CHUNK", chunk)
    written = write_results(tmp_path / "chunked.json", document, validated=True)
    assert written.read_text(encoding="utf-8") == json.dumps(document, indent=2, allow_nan=False) + "\n"


def test_an_empty_search_is_written_like_any_other(tmp_path):
    config = _config(max_length=2.0)
    document = _document(tmp_path, config, search(config))
    assert document["candidates"] == []
    written = write_results(tmp_path / "empty.json", document, validated=True)
    assert written.read_text(encoding="utf-8") == json.dumps(document, indent=2, allow_nan=False) + "\n"


def test_built_candidates_carry_the_values_the_scalar_formulas_give(tmp_path):
    """Gathering fields per array must not change a single recorded value."""

    from cellstine.core.lattice import vector_angle_deg

    config = _config(top_strain=0.01, bottom_strain=0.01, max_length=25.0)
    result = search(config)
    document = _document(tmp_path, config, result, max_candidates=40)
    offsets = [candidate["rank"] - 1 for candidate in document["candidates"]]
    for candidate, offset in zip(document["candidates"], offsets):
        shared = result.shared_lattice[offset]
        assert candidate["top_matrix"] == result.top_matrices[offset].T.astype(int).tolist()
        assert candidate["raw_angle_deg"] == float(np.degrees(result.raw_twist_radians[offset]))
        assert candidate["moire_a"] == pytest.approx(float(np.linalg.norm(shared[:, 0])), rel=1e-15)
        assert candidate["moire_b"] == pytest.approx(float(np.linalg.norm(shared[:, 1])), rel=1e-15)
        assert candidate["moire_gamma_deg"] == pytest.approx(
            vector_angle_deg(shared[:, 0], shared[:, 1]), rel=1e-14
        )
        # These fields belong to the root's separate, pre-existing schema work;
        # the scoped optimization must retain the current dev candidate format.
        assert "equivalence_key" not in candidate
        assert "summary_collision_size" not in candidate


@pytest.mark.parametrize(
    "field,corrupt",
    [
        ("sharing_fraction", lambda array: array + 2.0),
        ("twist_degrees", lambda array: array + 400.0),
        ("coincidence_indices", lambda array: array * 2),
        ("top_gram", lambda array: -array),
        ("top_affine", lambda array: array * 0.0),
        ("shared_lattice", lambda array: array * np.nan),
        ("atom_counts", lambda array: array + 1),
    ],
)
def test_an_invalid_row_is_refused_with_the_per_candidate_message(tmp_path, monkeypatch, field, corrupt):
    """The array checks refuse exactly what the per-candidate checks refuse."""

    from cellstine.moire.search import results

    config = _config(max_length=15.0)
    result = search(config)
    values = getattr(result, field).copy()
    values[-1] = corrupt(values[-1])
    broken = replace(result, **{field: values})
    with pytest.raises(ValueError) as fast:
        _document(tmp_path, config, broken)
    # The reference: switch the array checks off and run the full scalar pass.
    monkeypatch.setattr(results, "_first_invalid_row", lambda *args: None)
    unchecked = _document(tmp_path, config, broken)
    with pytest.raises(ValueError) as scalar:
        results._validate_document(unchecked)
    assert str(fast.value) == str(scalar.value)


def test_hashed_first_rows_match_the_byte_sort_and_fall_back_on_a_collision(monkeypatch):
    from cellstine.moire.search import gram_lattice

    rng = np.random.default_rng(5)
    keys = rng.integers(-3, 4, size=(5000, 5)).astype(np.int64)
    contiguous = np.ascontiguousarray(keys)
    view = contiguous.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
    expected = np.sort(np.unique(view, return_index=True)[1])
    np.testing.assert_array_equal(gram_lattice._first_per_key(keys), expected)

    # Force every row into one hash bucket: the exact check must notice and
    # the byte sort must answer instead.
    monkeypatch.setattr(gram_lattice.np, "argsort", lambda array, kind=None: np.arange(len(array)))
    real_concatenate = np.concatenate

    def one_bucket(arrays, *args, **kwargs):
        joined = real_concatenate(arrays, *args, **kwargs)
        if joined.dtype == bool:
            joined[1:] = False
        return joined

    monkeypatch.setattr(gram_lattice.np, "concatenate", one_bucket)
    assert gram_lattice._first_per_hashed_key(keys) is None


def test_primitive_orbit_keys_refuse_mismatched_layer_determinants(monkeypatch):
    from cellstine.moire.search import gram_pairs

    mirror = np.array([np.eye(2, dtype=np.int64), np.diag([1, -1])])
    monkeypatch.setattr(gram_pairs, "_reduced_symmetry_combinations", lambda top, bottom: [(0, 0), (1, 0)])
    top, bottom = _random_pairs(6, 10, primitive=True)
    with pytest.raises(ValueError, match="matching unimodular layer operations"):
        gram_pairs._primitive_orbit_keys(top, bottom, mirror, mirror)

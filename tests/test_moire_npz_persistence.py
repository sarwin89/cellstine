"""Portable no-pickle result archives retain the complete JSON document."""

import copy
import io
import json
import sys
import zipfile

import numpy as np
import pytest

from cellstine.moire.search import gram, results


@pytest.fixture
def document():
    config = gram.SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.)
    return results.build_results_document(
        top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config,
        result=gram.search(config), symmetric_requested=False, symmetric_used=False,
        created_at="2000-01-01T00:00:00+00:00",
    )


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=True)
    return stream.getvalue()


def _rewrite(path, replacements=None, *, omit=()):
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name not in omit}
    members.update(replacements or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize("count", [0, 1, 37])
def test_archive_roundtrip_is_standard_no_pickle_numpy(tmp_path, document, count):
    document["candidates"] = [dict(document["candidates"][0], index=i+1) for i in range(count)]
    document["metadata"].update(candidates_found=count, candidates_recorded=count)
    original = copy.deepcopy(document)
    target = results.write_results(tmp_path / "results.npz", document)
    assert zipfile.is_zipfile(target)
    with np.load(target, allow_pickle=False) as archive:
        assert archive.files
        for name in archive.files:
            assert not archive[name].dtype.hasobject
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
        assert manifest["schema"] == "cellstine.moire.gram.npz" and manifest["version"] == 1
    restored = results.read_results(target)
    assert json.dumps(restored) == json.dumps(original)
    assert json.dumps(document) == json.dumps(original)


@pytest.mark.parametrize("layout", ["uniform-fields", "per-row-fields"])
def test_archive_extension_fallback_is_lossless(tmp_path, document, layout):
    first = document["candidates"][0]
    first.update(large_integer=2**160+1, mixed=[2**60+1, .5], ragged=[[1], [2, 3], None],
                 unicode="\U0001f600\nquoted\"", signed_zero=-0.0)
    second = dict(first, index=2)
    if layout == "per-row-fields":
        second.pop("ragged")
        second = dict(reversed(list(second.items())))
    document["candidates"] = [first, second]
    document["metadata"].update(candidates_found=2, candidates_recorded=2)
    document["extension"] = {"second": [None, 2**160+1], "first": "\u03b1"}
    document = {"extension": document.pop("extension"), **document}
    target = results.write_results(tmp_path / "extensions.npz", document)
    assert zipfile.is_zipfile(target)
    assert json.dumps(results.read_results(target)) == json.dumps(document)


def test_default_json_bytes_remain_identical(tmp_path, document):
    path = results.write_results(tmp_path / "results.json", document)
    assert path.read_bytes() == (json.dumps(document, indent=2)+"\n").encode("utf-8")


def test_npz_preflight_does_not_truncate_existing_destination(tmp_path, document):
    path = tmp_path / "results.npz"
    path.write_bytes(b"preserve existing file")
    document["extension"] = object()
    with pytest.raises(TypeError):
        results.write_results(path, document, validated=True)
    assert path.read_bytes() == b"preserve existing file"


@pytest.mark.parametrize("damage", ["object", "wrong-rows", "missing", "unknown-version", "duplicate", "truncated"])
def test_malformed_archives_fail_clearly(tmp_path, document, damage):
    path = results.write_results(tmp_path / "results.npz", document)
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(np.load(io.BytesIO(archive.read("manifest.npy")), allow_pickle=False).tobytes())
        member = manifest["columns"][0]["member"]
    if damage == "object":
        _rewrite(path, {member: _npy(np.array([{"invalid": "object"}], dtype=object))})
    elif damage == "wrong-rows":
        _rewrite(path, {member: _npy(np.array([], dtype=np.int64))})
    elif damage == "missing":
        _rewrite(path, omit=(member,))
    elif damage == "unknown-version":
        manifest["version"] = 999
        _rewrite(path, {"manifest.npy": _npy(np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype="u1"))})
    elif damage == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate"):
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr(member, b"duplicate")
    else:
        path.write_bytes(path.read_bytes()[:-12])
    with pytest.raises(ValueError, match="NPZ|archive"):
        results.read_results(path)


def test_compressed_archive_has_the_same_document(tmp_path, document):
    path = results.write_results(tmp_path / "compressed.npz", document, compressed=True)
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
    assert json.dumps(results.read_results(path)) == json.dumps(document)


def test_foreign_byte_order_and_fortran_layout_are_read_exactly(tmp_path, document):
    document["candidates"][0]["extension_matrix"] = [[1.25, 2.5], [3.75, 5.]]
    path = results.write_results(tmp_path / "foreign.npz", document)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(np.load(io.BytesIO(archive.read("manifest.npy")), allow_pickle=False).tobytes())
        member = next(c["member"] for c in manifest["columns"] if c["field"] == "extension_matrix")
        values = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    byte_order = "<" if sys.byteorder == "big" else ">"
    foreign = np.asfortranarray(values.astype(byte_order+"f8"))
    assert foreign.dtype.byteorder == byte_order
    _rewrite(path, {member: _npy(foreign)})
    assert json.dumps(results.read_results(path)) == json.dumps(document)


def test_archive_still_runs_scientific_validation(tmp_path, document):
    path = results.write_results(tmp_path / "invalid-science.npz", document)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(np.load(io.BytesIO(archive.read("manifest.npy")), allow_pickle=False).tobytes())
        member = next(c["member"] for c in manifest["columns"] if c["field"] == "top_matrix")
    _rewrite(path, {member: _npy(np.zeros((1, 2, 2), dtype=np.int64))})
    with pytest.raises(ValueError, match="nonsingular"):
        results.read_results(path)


def test_unknown_objects_cannot_run_array_conversion_hooks(tmp_path, document):
    class NotJSON:
        def __array__(self, *args, **kwargs):
            raise AssertionError("unknown serialization object executed an array hook")

    path = tmp_path / "results.npz"
    path.write_bytes(b"preserve existing file")
    document["candidates"][0]["extension"] = NotJSON()
    with pytest.raises(TypeError):
        results.write_results(path, document, validated=True)
    assert path.read_bytes() == b"preserve existing file"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["metadata", "candidate"])
def test_nonfinite_extensions_follow_json_writer_policy(tmp_path, document, value, location):
    if location == "metadata":
        document["metadata"]["extension"] = value
    else:
        document["candidates"][0]["extension"] = value
    with pytest.raises(ValueError):
        results.write_results(tmp_path / "baseline.json", document)
    path = tmp_path / "results.npz"
    path.write_bytes(b"preserve existing file")
    with pytest.raises(ValueError):
        results.write_results(path, document)
    assert path.read_bytes() == b"preserve existing file"


def test_deep_extensions_use_json_for_minimum_numpy_compatibility(tmp_path, document):
    nested = 1.0
    for _ in range(40):
        nested = [nested]
    document["candidates"][0]["deep"] = nested
    path = results.write_results(tmp_path / "deep.npz", document)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes())
        column = next(c for c in manifest["columns"] if c["field"] == "deep")
        assert column["encoding"] == "json"
    assert json.dumps(results.read_results(path)) == json.dumps(document)


def test_cyclic_trusted_extension_fails_before_destination_open(tmp_path, document):
    cyclic = []
    cyclic.append(cyclic)
    document["candidates"][0]["extension"] = cyclic
    path = tmp_path / "results.npz"
    path.write_bytes(b"preserve existing file")
    with pytest.raises(ValueError, match="Circular"):
        results.write_results(path, document, validated=True)
    assert path.read_bytes() == b"preserve existing file"


def test_bad_declared_payload_is_rejected_before_array_allocation(tmp_path, document, monkeypatch):
    path = results.write_results(tmp_path / "bad-size.npz", document)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(np.load(io.BytesIO(archive.read("manifest.npy")), allow_pickle=False).tobytes())
        member = manifest["columns"][0]["member"]
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr": "<i8", "shape": (10**12,), "fortran_order": False})
    _rewrite(path, {member: stream.getvalue()})

    def forbidden(*args, **kwargs):
        raise AssertionError("bad header reached array allocation")

    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="NPZ.*payload size|NPZ.*invalid"):
        results.read_results(path)

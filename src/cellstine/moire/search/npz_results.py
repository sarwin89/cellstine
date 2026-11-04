"""Versioned no-pickle NPY members for the complete JSON result contract.

Numerical columns avoid text conversion. Other JSON values use streamed uint8
NPY members, retaining types, large integers, ragged values and key ordering.
The caller owns scientific validation; this module validates the archive.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "cellstine.moire.gram.npz"
VERSION = 1
_MANIFEST = "manifest.npy"
_CANDIDATES = "candidates.npy"
_MAX_COLUMNS = 256
_MAX_DIMENSIONS = 32  # NumPy1.26 compatibility; deeper values use JSON.
_MANIFEST_FIELDS = {"schema", "version", "keys", "document", "rows", "mode", "columns"}


@dataclass
class _Plan:
    arrays: dict[str, np.ndarray]
    json_members: dict[str, tuple[Any, int]]


def _json_size(value: Any) -> int:
    """Count encoded ASCII bytes without retaining the whole JSON string."""
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return sum(len(part) for part in encoder.iterencode(value))


def _numeric_column(values: list[Any]) -> np.ndarray | None:
    """Use an array only for uniform plain numeric leaf types, without coercion.

    Check types before invoking NumPy: unknown objects must never get an
    opportunity to execute an array conversion hook during serialization.
    """
    expected = type(values[0]) if values else None
    if expected in (int, float, bool):
        # Most candidate fields are scalar columns. Avoid the nested-list
        # traversal machinery while retaining the exact plain-type guard.
        if any(type(item) is not expected for item in values):
            return None
    else:
        expected = None
        stack = [iter(values)]
        while stack:
            try:
                item = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            kind = type(item)
            if kind is list:
                if len(stack) >= _MAX_DIMENSIONS:
                    # Also bounds inspection of cyclic trusted-input lists;
                    # the JSON encoder then supplies its circularity error.
                    return None
                stack.append(iter(item))
            elif kind not in (int, float, bool):
                return None
            elif expected is None:
                expected = kind
            elif kind is not expected:
                return None
    if expected is None:
        return None
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    kind = array.dtype.kind
    if expected is int and kind in "iu":
        dtype = np.dtype("i8" if kind == "i" else "u8")
    elif expected is float and kind == "f" and np.all(np.isfinite(array)):
        dtype = np.dtype("f8")
    elif expected is bool and kind == "b":
        dtype = np.dtype("b1")
    else:
        return None
    return np.ascontiguousarray(array, dtype=dtype)


def _prepare_archive(document: dict[str, Any]) -> _Plan:
    """Finish conversions and JSON serialization checks before opening a file."""
    candidates = document["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("NPZ candidates must be a JSON array")
    rows = len(candidates)
    fields = list(candidates[0]) if rows and isinstance(candidates[0], dict) else []
    columnar = not rows or (
        "index" in fields and len(fields) <= _MAX_COLUMNS
        and all(isinstance(row, dict) and list(row) == fields for row in candidates)
    )
    manifest = {
        "schema": SCHEMA, "version": VERSION, "keys": list(document),
        "document": {key: value for key, value in document.items() if key != "candidates"},
        "rows": rows, "mode": "columns" if columnar else "json", "columns": [],
    }
    arrays = {}
    json_members = {}
    if columnar:
        for index, field in enumerate(fields):
            name = f"column_{index:04d}.npy"
            values = [row[field] for row in candidates]
            array = _numeric_column(values)
            if array is None:
                encoding = "json"
                json_members[name] = values, _json_size(values)
            else:
                encoding = "array"
                arrays[name] = array
            manifest["columns"].append({"field": field, "member": name, "encoding": encoding})
    else:
        json_members[_CANDIDATES] = candidates, _json_size(candidates)
    json_members[_MANIFEST] = manifest, _json_size(manifest)
    return _Plan(arrays, json_members)


def _write_json_member(archive: zipfile.ZipFile, name: str, value: Any, size: int) -> None:
    """Stream ASCII JSON as a standard uint8 NPY1.0 array."""
    with archive.open(name, "w", force_zip64=True) as stream:
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "|u1", "fortran_order": False, "shape": (size,)}
        )
        written = 0
        encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        for part in encoder.iterencode(value):
            data = part.encode("ascii")
            stream.write(data)
            written += len(data)
        if written != size:
            raise ValueError("JSON content changed while writing NPZ archive")


def write_archive(path: Path, document: dict[str, Any], *, compressed: bool = False) -> None:
    """Write a planned archive; serialization failures precede destination open.

    Arrays are held during writing to avoid repeating numerical conversion.
    JSON encoding streams, but a large atomic string can require its own chunk.
    Later filesystem failures retain ordinary explicit-file write semantics.
    """
    plan = _prepare_archive(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression, allowZip64=True) as archive:
        value, size = plan.json_members[_MANIFEST]
        _write_json_member(archive, _MANIFEST, value, size)
        for name, array in plan.arrays.items():
            with archive.open(name, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
        for name, (value, size) in plan.json_members.items():
            if name != _MANIFEST:
                _write_json_member(archive, name, value, size)


def _member_header(stream, info: zipfile.ZipInfo) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    """Check the bounded NPY header and exact payload length before allocation."""
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("NPZ members must use NPY1.0")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("NPZ members cannot contain object or structured dtypes")
    if len(shape) > _MAX_DIMENSIONS or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("NPZ member has an invalid shape")
    size = dtype.itemsize
    for dimension in shape:
        size *= dimension
    if size > sys.maxsize or info.file_size != stream.tell() + size:
        raise ValueError("NPZ member shape and payload size disagree")
    return shape, fortran, dtype, size


def _read_json_member(archive: zipfile.ZipFile, name: str) -> Any:
    info = archive.getinfo(name)
    with archive.open(info) as stream:
        shape, _, dtype, _ = _member_header(stream, info)
        if len(shape) != 1 or dtype != np.dtype("u1"):
            raise ValueError("NPZ JSON members must be one-dimensional uint8 arrays")
        # Decode directly from the post-header stream, avoiding an extra NumPy
        # byte array and bytes copy. Reading to EOF also checks the member CRC.
        with io.TextIOWrapper(stream, encoding="utf-8") as text:
            return json.load(text)


def _validate_manifest(manifest: Any) -> tuple[dict, list, int, str]:
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("NPZ manifest has unsupported fields")
    if manifest["schema"] != SCHEMA or type(manifest["version"]) is not int or manifest["version"] != VERSION:
        raise ValueError("NPZ archive schema or version is unsupported")
    body, keys, rows = manifest["document"], manifest["keys"], manifest["rows"]
    if type(body) is not dict or "candidates" in body:
        raise ValueError("NPZ document header must omit candidates")
    if (type(keys) is not list or any(type(key) is not str for key in keys)
            or len(set(keys)) != len(keys) or set(keys) != set(body) | {"candidates"}):
        raise ValueError("NPZ document key order is invalid")
    if type(rows) is not int or rows < 0:
        raise ValueError("NPZ row count must be a nonnegative integer")
    mode, columns = manifest["mode"], manifest["columns"]
    if mode not in ("columns", "json") or type(columns) is not list or len(columns) > _MAX_COLUMNS:
        raise ValueError("NPZ candidate layout is unsupported")
    if mode == "json" and columns:
        raise ValueError("NPZ JSON candidate layout cannot declare columns")
    fields = []
    for index, column in enumerate(columns):
        if (type(column) is not dict or set(column) != {"field", "member", "encoding"}
                or type(column["field"]) is not str
                or column["member"] != f"column_{index:04d}.npy"
                or column["encoding"] not in ("array", "json")):
            raise ValueError("NPZ column declaration is invalid")
        fields.append(column["field"])
    if len(set(fields)) != len(fields) or (mode == "columns" and rows and "index" not in fields):
        raise ValueError("NPZ columns must be unique and include candidate indexes")
    if not rows and columns:
        raise ValueError("NPZ empty candidate layout must not declare columns")
    return body, keys, rows, mode


def _array_header(archive: zipfile.ZipFile, name: str, rows: int) -> None:
    """Preflight every numerical column before building candidate dictionaries."""
    info = archive.getinfo(name)
    with archive.open(info) as stream:
        shape, _, dtype, _ = _member_header(stream, info)
    if not shape or shape[0] != rows:
        raise ValueError("NPZ column row counts disagree")
    if not ((dtype.kind == "b" and dtype.itemsize == 1) or (dtype.kind in "iuf" and dtype.itemsize == 8)):
        raise ValueError("NPZ numeric columns require bool, int64, uint64 or float64")


def _read_array(archive: zipfile.ZipFile, name: str) -> list[Any]:
    info = archive.getinfo(name)
    with archive.open(info) as stream:
        shape, fortran, dtype, size = _member_header(stream, info)
        data = stream.read(size)
        if len(data) != size or stream.read(1):
            raise ValueError("NPZ numerical member has an invalid payload length")
    return np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran else "C").tolist()


def read_archive(path: Path) -> dict[str, Any]:
    """Read an archive's JSON contract, rejecting malformed/pickled members."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) > _MAX_COLUMNS + 1:
                raise ValueError("NPZ archive has duplicate or excessive members")
            if any(info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for info in infos):
                raise ValueError("NPZ archive uses unsupported compression")
            if _MANIFEST not in names:
                raise ValueError("NPZ archive lacks its manifest")
            manifest = _read_json_member(archive, _MANIFEST)
            body, keys, rows, mode = _validate_manifest(manifest)
            columns = manifest["columns"]
            expected = {_MANIFEST, _CANDIDATES} if mode == "json" else {_MANIFEST, *(c["member"] for c in columns)}
            if set(names) != expected:
                raise ValueError("NPZ archive member set disagrees with its manifest")
            if mode == "json":
                candidates = _read_json_member(archive, _CANDIDATES)
                if type(candidates) is not list or len(candidates) != rows:
                    raise ValueError("NPZ JSON candidate row count disagrees")
            else:
                for column in columns:
                    if column["encoding"] == "array":
                        _array_header(archive, column["member"], rows)
                candidates = []
                for column in columns:
                    values = (_read_array(archive, column["member"]) if column["encoding"] == "array"
                              else _read_json_member(archive, column["member"]))
                    if type(values) is not list or len(values) != rows:
                        raise ValueError("NPZ column row counts disagree")
                    if not candidates:
                        candidates = [{} for _ in range(rows)]
                    for candidate, value in zip(candidates, values):
                        candidate[column["field"]] = value
            return {key: candidates if key == "candidates" else body[key] for key in keys}
    except (ValueError, zipfile.BadZipFile, EOFError, UnicodeError, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"{path} is not a valid NPZ archive: {error}") from error

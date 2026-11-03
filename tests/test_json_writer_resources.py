"""JSON throughput changes must preserve bytes, failures and large-file memory."""

from __future__ import annotations

import io
import json
import tracemalloc

import pytest

from cellstine.core.jsonio import _small_json_budget, write_indented_json


def _nested(depth):
    value = [False, None, -1, 1.25]
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("value", [
    {False: False, None: None, -1: 2, 1.25: "fractional key"},
    {"text": "\U0001f600\u03b1\n\t\"\\", "tuple": (1, 2, 3)},
    {"astral": "\U0001f600" * 90000},
    _nested(32), _nested(60), [], {}, None,
])
def test_adaptive_writing_matches_standard_encoder_and_keeps_handle_open(value):
    expected = io.StringIO()
    json.dump(value, expected, indent=2)
    actual = io.StringIO()
    write_indented_json(actual, value)
    assert not actual.closed
    assert actual.getvalue() == expected.getvalue()


@pytest.mark.parametrize("value", [
    {False: False, None: None, -1: -1},
    _nested(28), _nested(32),
    {"astral": "\U0001f600" * 90000},
    {"controls": "\x00\n\t\"\\" * 1000},
    {"positions": [[.1, -.2, .3] for _ in range(100)]},
])
def test_small_document_estimate_bounds_actual_encoded_text(value):
    limit = 1 << 20
    remaining = _small_json_budget(value, remaining=limit)
    encoded_size = len(json.dumps(value, indent=2))
    if remaining >= 0:
        assert encoded_size <= limit - remaining
    if encoded_size > limit:
        assert remaining < 0


def _cycle():
    value = []
    value.append(value)
    return {"ordinary": 1, "cycle": value}


@pytest.mark.parametrize("factory,allow_nan", [
    (_cycle, True),
    (lambda: {"ordinary": 1, "invalid": object()}, True),
    (lambda: {"ordinary": 1, "invalid": float("nan")}, False),
    (lambda: {"ordinary": 1, "invalid": float("inf")}, False),
    (lambda: {"ordinary": 1, "invalid": 10**5000}, True),
    (lambda: {"ordinary": 1, (1, 2): "unsupported key"}, True),
], ids=["cycle", "custom-value", "nan", "infinity", "huge-integer", "tuple-key"])
def test_invalid_values_keep_streaming_exception_and_written_prefix(factory, allow_nan):
    value = factory()
    expected, actual = io.StringIO(), io.StringIO()
    baseline_error = None
    try:
        json.dump(value, expected, indent=2, allow_nan=allow_nan)
    except (TypeError, ValueError, OverflowError) as error:
        baseline_error = error
    if baseline_error is None:
        # Older Python3.10 builds, or an explicitly disabled integer-digit
        # limit, permit the large integer. Retain that verdict as well.
        write_indented_json(actual, value, allow_nan=allow_nan)
    else:
        with pytest.raises(type(baseline_error)) as current_error:
            write_indented_json(actual, value, allow_nan=allow_nan)
        assert str(current_error.value) == str(baseline_error)
    assert actual.getvalue() == expected.getvalue()
    assert not actual.closed


def test_custom_dictionary_iteration_is_not_repeated():
    class ChangingDictionary(dict):
        calls = 0

        def items(self):
            self.calls += 1
            return {"iteration": self.calls}.items()

    expected = io.StringIO()
    json.dump({"nested": ChangingDictionary(seed=True)}, expected, indent=2)
    value = ChangingDictionary(seed=True)
    actual = io.StringIO()
    write_indented_json(actual, {"nested": value})
    assert actual.getvalue() == expected.getvalue()
    assert value.calls == 1


def test_large_report_does_not_materialize_another_whole_document(tmp_path):
    payload = {"records": [{"position": [.1, .2, .3], "index": index} for index in range(12000)]}
    expected = json.dumps(payload, indent=2).encode("utf-8")
    destination = tmp_path / "large.json"
    tracemalloc.start()
    try:
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            write_indented_json(handle, payload)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert destination.read_bytes() == expected
    # The baseline stream's buffers are small; whole-string encoding allocates
    # multiple copies of this document. The relative limit leaves wide room for
    # Python-version buffer differences while detecting that resource regression.
    assert peak < max(512 * 1024, len(expected))

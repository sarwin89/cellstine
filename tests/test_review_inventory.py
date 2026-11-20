"""The review ledger must cover nested functions and invalidate stale evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "review_inventory.py"
    spec = importlib.util.spec_from_file_location("review_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_includes_methods_nested_functions_async_and_lambdas(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(
        "class Example:\n"
        "    def method(self):\n"
        "        def nested():\n"
        "            return 1\n"
        "        return lambda value: value + nested()\n"
        "async def task():\n"
        "    return 2\n"
    )
    rows = _tool().collect_functions(tmp_path, ["sample.py"])
    assert [row["qualified_name"] for row in rows] == [
        "Example.method", "Example.method.nested", "Example.method.<lambda>@5:15", "task",
    ]
    assert [(row["line"], row["end_line"]) for row in rows] == [(2, 5), (3, 4), (5, 5), (6, 7)]
    assert len({row["id"] for row in rows}) == 4


def test_inventory_preserves_review_only_when_module_is_unchanged(tmp_path):
    tool = _tool()
    path = tmp_path / "sample.py"
    path.write_text("def value():\n    return 1\n")
    original = tool.reconcile(tool.collect_functions(tmp_path, ["sample.py"]), [])
    original[0].update(status="reviewed—unchanged", reason="Exact integer result", evidence="tests/test_sample.py")
    repeated = tool.reconcile(tool.collect_functions(tmp_path, ["sample.py"]), original)
    assert repeated[0]["status"] == "reviewed—unchanged"
    assert repeated[0]["evidence"] == "tests/test_sample.py"
    path.write_text("def value():\n    return 2\n")
    changed = tool.reconcile(tool.collect_functions(tmp_path, ["sample.py"]), original)
    assert changed[0]["status"] == "deferred—with reason"
    assert "changed" in changed[0]["reason"]
    assert changed[0]["evidence"] == ""


def test_inventory_rejects_verified_status_without_evidence(tmp_path):
    tool = _tool()
    (tmp_path / "sample.py").write_text("def value():\n    return 1\n")
    rows = tool.reconcile(tool.collect_functions(tmp_path, ["sample.py"]), [])
    rows[0]["status"] = "optimized—verified"
    with pytest.raises(ValueError, match="evidence"):
        tool.reconcile(tool.collect_functions(tmp_path, ["sample.py"]), rows)


def test_review_hash_ignores_checkout_line_endings(tmp_path):
    tool = _tool()
    path = tmp_path / "sample.py"
    path.write_bytes(b"def value():\n    return 1\n")
    unix = tool.collect_functions(tmp_path, ["sample.py"])
    path.write_bytes(b"def value():\r\n    return 1\r\n")
    windows = tool.collect_functions(tmp_path, ["sample.py"])
    assert windows == unix

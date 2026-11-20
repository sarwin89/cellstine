"""Generate/check the source-function review ledger using only tracked Python files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


STATUSES = {"reviewed—unchanged", "optimized—verified", "deferred—with reason"}


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, path: str, source_hash: str):
        self.path = path
        self.source_hash = source_hash
        self.scope: list[str] = []
        self.rows: list[dict] = []

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self._function(node, node.name)

    def visit_AsyncFunctionDef(self, node):
        self._function(node, node.name)

    def visit_Lambda(self, node):
        self._function(node, f"<lambda>@{node.lineno}:{node.col_offset}")

    def _function(self, node, name):
        self.scope.append(name)
        qualified = ".".join(self.scope)
        self.rows.append({
            "id": f"{self.path}::{qualified}",
            "path": self.path,
            "qualified_name": qualified,
            "line": node.lineno,
            "end_line": node.end_lineno,
            "module_sha256": self.source_hash,
        })
        self.generic_visit(node)
        self.scope.pop()


def collect_functions(root: Path, paths: list[str]) -> list[dict]:
    rows = []
    for name in sorted(paths):
        path = root / name
        source = path.read_text(encoding="utf-8-sig")
        visitor = FunctionVisitor(name, hashlib.sha256(source.encode("utf-8")).hexdigest())
        visitor.visit(ast.parse(source, filename=name))
        rows.extend(visitor.rows)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate qualified function names require inventory disambiguation")
    return rows


def reconcile(current: list[dict], previous: list[dict]) -> list[dict]:
    recorded = {row["id"]: row for row in previous}
    if len(recorded) != len(previous):
        raise ValueError("Duplicate review ledger IDs")
    for old in previous:
        if old["status"] not in STATUSES or not old["reason"].strip():
            raise ValueError(f"Invalid status or missing reason: {old['id']}")
        if old["status"] != "deferred—with reason" and not old["evidence"].strip():
            raise ValueError(f"Reviewed functions require evidence: {old['id']}")
    rows = []
    for function in current:
        old = recorded.get(function["id"])
        row = dict(function)
        if old and old["module_sha256"] == row["module_sha256"]:
            row.update({key: old[key] for key in ("status", "reason", "evidence")})
        else:
            row.update(
                status="deferred—with reason",
                reason=("Source module changed; requires re-review" if old else
                        "Pending dependency-ordered review; no review claim"),
                evidence="",
            )
        rows.append(row)
    return rows


def tracked_python_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=root,
        check=True, capture_output=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0")
            if name and not name.startswith("tests/")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if coverage or review evidence is stale")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target = root / "docs" / "optimization" / "function-review.json"
    previous = json.loads(target.read_text(encoding="utf-8"))["functions"] if target.exists() else []
    rows = reconcile(collect_functions(root, tracked_python_files(root)), previous)
    content = json.dumps({"schema_version": 1, "functions": rows}, ensure_ascii=False, indent=2) + "\n"
    counts = Counter(row["status"] for row in rows)
    print(f"{len(rows)} functions: " + ", ".join(f"{key}={counts[key]}" for key in sorted(STATUSES)))
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            print("Ledger is stale; regenerate and inspect changed/deferred records.")
            return 1
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recent-file discovery preserves priority, ties and resolved duplicates."""

import os
from pathlib import Path

import pytest

from cellstine.cli.interactive.prompts import _find_candidates


def _baseline(patterns, roots, *, limit=8):
    found, seen = [], set()
    for index, root in enumerate(roots):
        if root.exists():
            for pattern in patterns:
                for path in root.rglob(pattern):
                    if path.is_file():
                        resolved = path.resolve()
                        if resolved not in seen:
                            seen.add(resolved)
                            found.append((index, resolved))
    found.sort(key=lambda item: (item[0], -item[1].stat().st_mtime))
    return [path for _, path in found[:limit]]


@pytest.mark.parametrize("limit", [0, 1, 3, 8, 20, -1])
def test_exact_priority_and_stable_ties_with_overlapping_roots(tmp_path, limit):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for index in range(12):
        path = (first if index < 7 else second) / f"POSCAR{index}.vasp"
        path.touch()
        os.utime(path, (1000 + index % 3, 1000 + index % 3))
    patterns, roots = ("*.vasp", "POSCAR*"), (first, first, second, tmp_path)
    assert _find_candidates(patterns, roots, limit=limit) == _baseline(patterns, roots, limit=limit)


def test_full_higher_priority_root_does_not_scan_run_tree(monkeypatch, tmp_path):
    first, lower = tmp_path / "input", tmp_path / "runs"
    first.mkdir()
    lower.mkdir()
    for index in range(8):
        (first / f"{index}.vasp").touch()
    original = Path.rglob

    def scan(self, pattern):
        if self == lower:
            raise AssertionError("lower-priority run tree need not be scanned")
        return original(self, pattern)

    monkeypatch.setattr(Path, "rglob", scan)
    assert len(_find_candidates(("*.vasp",), (first, lower))) == 8

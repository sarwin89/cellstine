"""Documentation checks for the public CLI contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "USAGE_GUIDE.md",
    ROOT / "ROADMAP.md",
    *(ROOT / "docs").glob("*.md"),
    *(ROOT / "docs" / "workflows").glob("*.md"),
]

INTENTIONAL_HISTORY_DOCS = {
    ROOT / "docs" / "cli.md",
    ROOT / "docs" / "aristotle-migration-timeline.md",
}

STALE_CLI_TEXT = [
    "python moire_cli.py",
    "N-layer moire workflows are not supported in this release",
    "cellstine moire find",
    "cellstine moire make",
    "cellstine moire translate",
    "cellstine moire visualize",
    "cellstine interface surface",
    "cellstine interface sites",
    "cellstine interface visualize",
    "cellstine adsorbate visualize",
    "cellstine defect visualize",
    "cellstine symmetry visualize",
    "--max-length",
    "--max-atoms",
]


def test_public_docs_do_not_reintroduce_removed_cli_names():
    offenders: list[str] = []
    for path in PUBLIC_DOCS:
        if path in INTENTIONAL_HISTORY_DOCS:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in STALE_CLI_TEXT:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle!r}")

    assert offenders == []


def test_nlayer_public_docs_label_stack_commands_experimental():
    checked = [
        ROOT / "README.md",
        ROOT / "docs" / "quickstart.md",
        ROOT / "docs" / "workflows" / "moire.md",
        ROOT / "docs" / "moire-performance.md",
    ]

    for path in checked:
        text = path.read_text(encoding="utf-8").lower()
        assert "stack-search" in text
        assert "stack-build" in text
        assert "experimental" in text

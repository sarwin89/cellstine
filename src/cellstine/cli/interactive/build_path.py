"""Interactive command builder for the shared migration-path stage."""

from __future__ import annotations

from .prompts import (
    INPUT_DIR,
    OUTPUT_DIR,
    _choice,
    _print_title,
    _prompt_int_range,
    _prompt_path,
)

STRUCTURE_PATTERNS = ("*.vasp", "POSCAR", "CONTCAR")


def _build_migration_path(group: str, *, subject: str) -> list[str]:
    """Return the ``path`` command for ``group``, asked for one question at a time."""

    _print_title(
        "Migration Path",
        f"Build the chain of images between two structures of one cell ({subject}).",
    )
    start = _prompt_path(
        "Choose the initial structure",
        patterns=STRUCTURE_PATTERNS,
        roots=(INPUT_DIR, OUTPUT_DIR),
    )
    end = _prompt_path(
        "Choose the final structure",
        patterns=STRUCTURE_PATTERNS,
        roots=(INPUT_DIR, OUTPUT_DIR),
    )
    images = _prompt_int_range("How many intermediate images?", 3, 1, 99)
    pairing = _choice(
        "How should the atoms of the two structures be paired?",
        [
            {
                "key": "match",
                "label": "By the pairing that makes the path shortest",
                "hint": "Recommended. Solved exactly, per species, so the file order of the two structures does not matter.",
            },
            {
                "key": "order",
                "label": "In file order",
                "hint": "Only when the two files already list the same atoms in the same order.",
            },
        ],
        default=1,
    )
    argv = [group, "path", start, end, "--images", str(images)]
    if pairing == "order":
        argv.append("--no-match")
    return argv

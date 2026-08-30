"""Shared command names, CLI validation, and migration guidance."""

from __future__ import annotations

from dataclasses import dataclass


APP_NAME = "CELLSTINE"
APP_EXPANSION = "CELL Superlattice Transformation INterface and Engine"


@dataclass(frozen=True)
class LegacyCommand:
    """A removed command and its replacement."""

    replacement: str
    note: str = "This command was renamed in the simplified CLI."


LEGACY_COMMANDS: dict[tuple[str, str], LegacyCommand] = {
    ("moire", "find"): LegacyCommand("cellstine moire search"),
    ("moire", "make"): LegacyCommand("cellstine moire build"),
    ("moire", "translate"): LegacyCommand("cellstine moire shift"),
    ("moire", "visualize"): LegacyCommand("cellstine moire view"),
    ("moire", "findn"): LegacyCommand("cellstine moire stack-search"),
    ("moire", "maken"): LegacyCommand("cellstine moire stack-build"),
    ("interface", "surface"): LegacyCommand("cellstine surface build"),
    ("interface", "sites"): LegacyCommand("cellstine surface sites"),
    ("interface", "visualize"): LegacyCommand("cellstine view STRUCTURE"),
    ("adsorbate", "visualize"): LegacyCommand("cellstine view STRUCTURE"),
    ("defect", "visualize"): LegacyCommand("cellstine view STRUCTURE"),
    ("symmetry", "visualize"): LegacyCommand("cellstine view STRUCTURE"),
}


def legacy_command_message(group: str, stage: str) -> str | None:
    """Return the hard-break migration message for a removed command."""

    command = LEGACY_COMMANDS.get((str(group), str(stage)))
    if command is None:
        return None
    return f"{command.note} Use `{command.replacement}`."


def parse_twist_window(raw: str | None) -> tuple[float | None, float | None]:
    """Parse the readable ``--twist MIN:MAX`` selector."""

    if raw in {None, ""}:
        return None, None
    text = str(raw).strip()
    if ":" not in text:
        raise ValueError("twist must be a range like 9:14, :14, or 9:")
    left, right = text.split(":", 1)
    minimum = float(left) if left.strip() else None
    maximum = float(right) if right.strip() else None
    if minimum is not None and minimum < 0:
        raise ValueError("twist lower bound must be nonnegative")
    if maximum is not None and maximum < 0:
        raise ValueError("twist upper bound must be nonnegative")
    if minimum is not None and maximum is not None and maximum < minimum:
        minimum, maximum = maximum, minimum
    return minimum, maximum


def resolve_moire_strains(
    *,
    rigid: bool,
    strain: float | None,
    top_strain: float | None,
    bottom_strain: float | None,
) -> tuple[float, float]:
    """Resolve the simplified moire strain controls to per-layer budgets."""

    has_shared = strain is not None
    has_pair = top_strain is not None or bottom_strain is not None
    selected = int(bool(rigid)) + int(has_shared) + int(has_pair)
    if selected != 1:
        raise ValueError(
            "choose one strain mode: --rigid, --strain E, or both --top-strain and --bottom-strain"
        )
    if rigid:
        return 0.0, 0.0
    if has_shared:
        value = float(strain)
        if value < 0:
            raise ValueError("--strain must be nonnegative")
        return value, value
    if top_strain is None or bottom_strain is None:
        raise ValueError("provide both --top-strain and --bottom-strain for asymmetric strain")
    top = float(top_strain)
    bottom = float(bottom_strain)
    if top < 0 or bottom < 0:
        raise ValueError("strain budgets must be nonnegative")
    return top, bottom

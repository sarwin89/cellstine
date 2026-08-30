"""Human-readable rendering of a finished workflow result.

Every workflow returns a :class:`~cellstine.core.models.CommandResult` whose
``artifacts`` map names the files that were written and whose ``summary`` map
carries the numbers that describe them.  This module turns that pair into the
block of text the CLI prints, so ``moire build``, ``interface build``,
``adsorbate place`` and ``defect generate`` all report their results in the same
shape: paths relative to the working directory, one value per line, numbers
rounded to a readable number of digits instead of full binary precision, and
nested groups indented under their heading.

The formatting rules are deliberately conservative:

* a float is printed with :data:`SIGNIFICANT_DIGITS` significant digits and no
  trailing zeros, so ``29.72272733872298`` reads ``29.7227`` and an exact zero
  reads ``0``;
* a value that is exactly an integer is printed without a decimal point;
* ``True``/``False`` read ``yes``/``no`` and ``None`` reads ``-``;
* a path inside the working directory is printed relative to it;
* a list of paths is printed one path per line, while a short list of numbers
  stays on one line;
* long text is wrapped at :data:`LINE_WIDTH` columns and its continuation lines
  are indented under the value.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

SIGNIFICANT_DIGITS = 6
LINE_WIDTH = 96
INDENT = "  "

#: Keys whose value is a length in angstrom, printed with the unit attached.
_ANGSTROM_KEYS = frozenset(
    {
        "vacuum",
        "vacuum_before",
        "vacuum_after",
        "vacuum_gap",
        "gap",
        "c_length",
        "interlayer_distance",
        "height",
        "slab_thickness",
        "cell_height",
        "min_distance",
        "minimum_distance",
        "nearest_neighbour_distance",
        "defect_image_distance",
        "closest_contact",
        "closest_defect_contact",
        "closest_interlayer_contact",
        "closest_contact_in_cell",
        "structure_contact_distance",
        "contact_distance",
        "molecule_image_distance",
        "self_image_distance",
    }
)

#: Keys whose value is an angle in degrees.
_DEGREE_KEYS = frozenset({"angle", "twist_angle", "gamma", "cell_angle", "misfit_angle"})


def _is_pathlike(value: Any) -> bool:
    if isinstance(value, PurePath):
        return True
    if not isinstance(value, str):
        return False
    if "\n" in value or " " in value.strip():
        return False
    return os.sep in value


def format_path(value: Any, base_dir: Path | None = None) -> str:
    """Render a path relative to ``base_dir`` when it sits inside it."""

    path = Path(value)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def format_number(value: float) -> str:
    """Render a number with :data:`SIGNIFICANT_DIGITS` digits and no clutter."""

    number = float(value)
    if number == 0.0:
        return "0"
    if not (number == number) or number in (float("inf"), float("-inf")):
        return str(number)
    text = f"{number:.{SIGNIFICANT_DIGITS}g}"
    if "e" not in text and "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


def _unit_for(key: str) -> str:
    if key in _ANGSTROM_KEYS:
        return " Ang"
    if key in _DEGREE_KEYS:
        return " deg"
    return ""


def format_scalar(value: Any, *, key: str = "", base_dir: Path | None = None) -> str:
    """Render a single non-container value."""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value}{_unit_for(key)}"
    if isinstance(value, float):
        return f"{format_number(value)}{_unit_for(key)}"
    if _is_pathlike(value):
        return format_path(value, base_dir)
    return str(value)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (Mapping, list, tuple, set))


def format_label(key: str) -> str:
    """Turn a summary key into the label printed in front of its value."""

    return str(key).replace("_", " ").strip()


def _render_sequence(
    key: str, values: Sequence[Any], indent: str, base_dir: Path | None
) -> list[str]:
    items = list(values)
    label = format_label(key)
    if not items:
        return [f"{indent}{label}: (none)"]
    if all(_is_pathlike(item) for item in items):
        if len(items) == 1:
            return [f"{indent}{label}: {format_path(items[0], base_dir)}"]
        lines = [f"{indent}{label} ({len(items)}):"]
        lines.extend(f"{indent}{INDENT}{format_path(item, base_dir)}" for item in items)
        return lines
    if all(isinstance(item, str) for item in items) and any(len(item) > 40 for item in items):
        lines = [f"{indent}{label}:"]
        for item in items:
            lines.extend(_wrap(f"{indent}{INDENT}- {item}", indent + INDENT * 2))
        return lines
    if all(_is_scalar(item) for item in items):
        rendered = ", ".join(format_scalar(item, base_dir=base_dir) for item in items)
        return _wrap(f"{indent}{label}: [{rendered}]", indent + INDENT)
    lines = [f"{indent}{label} ({len(items)}):"]
    for position, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            lines.append(f"{indent}{INDENT}{position}:")
            lines.extend(render_mapping(item, indent + INDENT * 2, base_dir))
        else:
            lines.append(
                f"{indent}{INDENT}{position}: {format_scalar(item, base_dir=base_dir)}"
            )
    return lines


def _wrap(line: str, continuation_indent: str) -> list[str]:
    if len(line) <= LINE_WIDTH:
        return [line]
    return textwrap.wrap(
        line,
        width=LINE_WIDTH,
        subsequent_indent=continuation_indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [line]


def render_mapping(
    mapping: Mapping[str, Any], indent: str = "", base_dir: Path | None = None
) -> list[str]:
    """Render a summary mapping as a list of aligned, indented lines."""

    lines: list[str] = []
    for key, value in mapping.items():
        label = format_label(key)
        if isinstance(value, Mapping):
            if not value:
                lines.append(f"{indent}{label}: (none)")
                continue
            lines.append(f"{indent}{label}:")
            lines.extend(render_mapping(value, indent + INDENT, base_dir))
        elif isinstance(value, (list, tuple, set)):
            ordered = sorted(value) if isinstance(value, set) else list(value)
            lines.extend(_render_sequence(str(key), ordered, indent, base_dir))
        else:
            text = format_scalar(value, key=str(key), base_dir=base_dir)
            lines.extend(_wrap(f"{indent}{label}: {text}", indent + INDENT))
    return lines


_TIMING_ORDER = (
    "read_structures_s",
    "angle_shortlist_s",
    "supercell_search_s",
    "build_structures_s",
    "write_results_s",
    "write_structures_s",
    "manifest_write_s",
    "workflow_total_s",
)


def render_timings(timings: Mapping[str, Any], indent: str = INDENT) -> list[str]:
    """Render the timing block, known stages first and the rest after."""

    known = [key for key in _TIMING_ORDER if key in timings]
    rest = [key for key in timings if key not in _TIMING_ORDER]
    lines = []
    for key in known + sorted(rest):
        label = format_label(str(key).removesuffix("_s"))
        try:
            lines.append(f"{indent}{label}: {float(timings[key]):.3f} s")
        except (TypeError, ValueError):
            lines.append(f"{indent}{label}: {timings[key]}")
    return lines


def format_result(result: Any, *, base_dir: Path | None = None) -> str:
    """Render a whole :class:`CommandResult` as the CLI prints it."""

    artifacts = dict(getattr(result, "artifacts", {}) or {})
    summary = dict(getattr(result, "summary", {}) or {})
    payload = dict(getattr(result, "payload", {}) or {})

    lines: list[str] = []
    manifest_path = getattr(result, "manifest_path", None)
    if manifest_path is not None:
        lines.append(f"Manifest: {format_path(manifest_path, base_dir)}")
    if artifacts:
        lines.append("")
        lines.append("Files written:")
        lines.extend(render_mapping(artifacts, INDENT, base_dir))
    # Warnings are lifted out of the summary and printed last, where they are
    # read, instead of being buried between two numbers.
    warnings = [str(note) for note in (summary.pop("warnings", None) or [])]
    if summary:
        lines.append("")
        lines.append("Results:")
        lines.extend(render_mapping(summary, INDENT, base_dir))

    timings = payload.get("timings_s")
    if timings:
        lines.append("")
        lines.append("Timing:")
        lines.extend(render_timings(timings))

    angle_search = payload.get("angle_search")
    if angle_search:
        lines.append("")
        lines.append("Angle search:")
        lines.append(
            f"{INDENT}shortlisted angles: {angle_search.get('shortlisted_angle_count')}"
        )
        lines.append(f"{INDENT}searched angles: {angle_search.get('searched_angle_count')}")
        if angle_search.get("angle_values_thinned"):
            lines.append(
                f"{INDENT}thinning: {angle_search.get('angle_values_before_thinning')} -> "
                f"{angle_search.get('searched_angle_count')} "
                f"(cap {angle_search.get('max_search_angles')})"
            )

    for key, title in (
        ("candidate_preview", "Candidate preview"),
        ("site_preview", "Site preview"),
        ("defect_preview", "Defect preview"),
        ("supercell_preview", "Supercell preview"),
        ("path_preview", "Migration path"),
        ("kpath_preview", "High-symmetry points"),
        ("segment_preview", "Band path segments"),
        ("symmetry_preview", "Symmetry preview"),
        ("registry_table", "Distinct interface options"),
    ):
        block = payload.get(key)
        if block:
            lines.append("")
            lines.append(f"{title}:")
            lines.append(str(block))

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for note in warnings:
            lines.extend(_wrap(f"{INDENT}- {note}", INDENT * 2))

    return "\n".join(lines)

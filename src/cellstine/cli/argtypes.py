"""Value parsers and small argparse helpers shared by the CELLSTINE CLI.

These turn one command-line token into the object a workflow wants -- a
positive length, a supercell, a mesh shift -- and raise
``argparse.ArgumentTypeError`` with a readable message when the token cannot
mean that.  They are kept apart from the parser tree in ``parsers.py`` so that
each file stays readable, and so that a caller that only needs to interpret one
value does not have to build the whole command grammar.
"""

from __future__ import annotations

import argparse
import math
from typing import List

# Nothing outside the standard library is imported here: reading one flag must
# not drag in NumPy or a workflow package, so ``--help``, a mistyped flag and
# the interactive menu all start immediately.

APP_NAME = "CELLSTINE"
APP_EXPANSION = "CELL Superlattice Transformation INterface and Engine"
LEGACY_MOIRE_FIND_MESSAGE = (
    "Legacy moire search controls are unsupported; use --length plus one strain "
    "mode (--rigid, --strain E, or both --top-strain and --bottom-strain)."
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Readable help text with examples and defaults."""


class LegacyMoireFindAction(argparse.Action):
    """Reject retired find flags with one actionable migration message."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        parser.error(LEGACY_MOIRE_FIND_MESSAGE)


def parse_positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - argparse reports the message
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return value


def parse_nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def add_legacy_moire_find_flag(parser: argparse.ArgumentParser, *flags: str, takes_value: bool = True) -> None:
    parser.add_argument(
        *flags,
        action=LegacyMoireFindAction,
        nargs="?" if takes_value else 0,
        help=argparse.SUPPRESS,
    )


def parse_index_spec(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("please provide at least one index")
    return list(dict.fromkeys(values))


def parse_float_vector(raw: str) -> List[float]:
    values = [float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if len(values) not in {2, 3}:
        raise argparse.ArgumentTypeError("please provide 2 or 3 numeric values separated by commas")
    return values


def parse_supercell(raw: str) -> List[int]:
    """Parse ``2``, ``2,2,1`` or ``2x2x1`` into three positive repeats."""

    text = str(raw).lower().replace("x", ",").replace(";", ",").replace(" ", ",")
    try:
        values = [int(token.strip()) for token in text.split(",") if token.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("please provide integer repeats, e.g. 2,2,1") from error
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise argparse.ArgumentTypeError("please provide one or three integer repeats, e.g. 2,2,1")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("supercell repeats must be at least 1")
    return values


def parse_mesh_shift(raw: str) -> List[float]:
    """Parse ``0,0,0`` or ``0.5,0.5,0`` into a three-component mesh offset."""

    text = str(raw).replace(";", ",").replace(" ", ",")
    try:
        values = [float(token.strip()) for token in text.split(",") if token.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("please provide three numbers, e.g. 0.5,0.5,0") from error
    if len(values) != 3:
        raise argparse.ArgumentTypeError("a mesh shift needs three components, e.g. 0.5,0.5,0")
    if any(abs(2.0 * value - round(2.0 * value)) > 1e-9 for value in values):
        raise argparse.ArgumentTypeError("a mesh shift must be a whole or half grid step")
    return values


def parse_supercell_matrix(raw: str) -> List[List[int]]:
    """Parse nine integers into the rows of a ``3x3`` supercell matrix."""

    text = str(raw).replace(";", ",").replace(" ", ",")
    try:
        values = [int(token.strip()) for token in text.split(",") if token.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "please provide nine integers, e.g. -1,1,1,1,-1,1,1,1,-1"
        ) from error
    if len(values) != 9:
        raise argparse.ArgumentTypeError("a supercell matrix needs exactly nine integers")
    rows = [values[0:3], values[3:6], values[6:9]]
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if determinant == 0:
        raise argparse.ArgumentTypeError("a supercell matrix must be invertible")
    return rows


def parse_int_matrix(raw: str) -> List[int]:
    """Parse four integers into the rows of an in-plane ``2x2`` matrix.

    A singular matrix maps the two in-plane vectors onto one line, so it does
    not describe a cell at all; it is refused here, where the message can name
    the flag, rather than deeper in the builder.
    """

    try:
        values = [int(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("please provide four integers, e.g. 1,1,0,2") from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("please provide exactly four integer values")
    if values[0] * values[3] - values[1] * values[2] == 0:
        raise argparse.ArgumentTypeError("an in-plane matrix must be invertible")
    return values


def parse_string_list(raw: str | None) -> List[str] | None:
    if raw in {None, ""}:
        return None
    return [token.strip() for token in str(raw).split(",") if token.strip()]

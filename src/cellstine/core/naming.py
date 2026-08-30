"""File-name tokens for the structures and reports CELLSTINE writes.

Every stage names its output after what went into it -- a structure stem, a
Miller family, a vacuum thickness, a site label -- and those come from user
input, so they can carry anything a file name cannot.  One helper turns such a
value into a token, and every stage uses it, so a given input always produces
the same name whichever stage wrote it.
"""

from __future__ import annotations

__all__ = ["safe_token"]

#: Characters kept verbatim in a token beyond the alphanumerics.
_KEPT = {"_", "m", "p"}


def safe_token(value: object) -> str:
    """Return ``value`` as a token safe to put in a file name.

    A minus sign becomes ``m`` and a decimal point becomes ``p``, so ``-1``
    reads as ``m1`` and ``12.50`` as ``12p50``; anything else outside the
    alphanumerics becomes an underscore, and leading and trailing underscores
    are dropped.  An input with nothing usable in it gives ``x`` rather than an
    empty name.
    """

    text = str(value).strip().replace("-", "m").replace(".", "p")
    token = "".join(char if char.isalnum() or char in _KEPT else "_" for char in text)
    return token.strip("_") or "x"

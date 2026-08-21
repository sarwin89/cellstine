"""Functional wrapper for multi-layer translation."""

from __future__ import annotations

from ..supermoire import Supermoire


def translaten(**kwargs):
    raise NotImplementedError(
        "N-layer moire workflows are not supported by the Gram-form engine. "
        "Use bilayer moire find and make."
    )

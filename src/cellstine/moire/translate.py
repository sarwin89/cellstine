"""Functional wrapper for rigid upper-layer translation."""

from __future__ import annotations

from .moire import Moire


def translate(**kwargs):
    return Moire().translate(**kwargs)

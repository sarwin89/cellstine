"""Functional wrapper for multi-layer translation."""

from __future__ import annotations

from ..supermoire import Supermoire


def translaten(**kwargs):
    return Supermoire().translaten(**kwargs)

"""Functional wrapper for bilayer moire search."""

from __future__ import annotations

from .moire import Moire


def find(**kwargs):
    return Moire().find(**kwargs)

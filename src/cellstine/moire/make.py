"""Functional wrapper for bilayer moire generation."""

from __future__ import annotations

from .moire import Moire


def make(**kwargs):
    return Moire().make(**kwargs)

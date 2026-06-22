"""Functional wrapper for molecule placement."""

from __future__ import annotations

from ..molecule import Molecule


def place(**kwargs):
    return Molecule().place(**kwargs)

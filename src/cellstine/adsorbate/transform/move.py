"""Functional wrapper for rigid molecule movement."""

from __future__ import annotations

from ..molecule import Molecule


def move(**kwargs):
    return Molecule().move(**kwargs)

"""Functional wrapper for substrate supercell search under a molecular assembly target."""

from __future__ import annotations

from .molecule import Molecule


def assemble(**kwargs):
    return Molecule().assemble(**kwargs)

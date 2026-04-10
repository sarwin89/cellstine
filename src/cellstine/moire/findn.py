"""Functional wrapper for N-layer moire search."""

from __future__ import annotations

from .supermoire import Supermoire


def findn(**kwargs):
    return Supermoire().findn(**kwargs)

"""Functional wrapper for N-layer moire generation."""

from __future__ import annotations

from .supermoire import Supermoire


def maken(**kwargs):
    return Supermoire().maken(**kwargs)

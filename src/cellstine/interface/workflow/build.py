"""Functional wrapper for interface generation."""

from __future__ import annotations

from .interface import Interface


def build(**kwargs):
    return Interface().build(**kwargs)

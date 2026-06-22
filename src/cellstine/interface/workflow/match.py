"""Functional wrapper for interface matching."""

from __future__ import annotations

from .interface import Interface


def match(**kwargs):
    return Interface().match(**kwargs)

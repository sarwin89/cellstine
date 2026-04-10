"""Functional wrapper for adsorption-site analysis."""

from __future__ import annotations

from .surface import Surface


def sites(**kwargs):
    return Surface().sites(**kwargs)

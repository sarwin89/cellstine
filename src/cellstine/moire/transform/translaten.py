"""Functional wrapper for translating part of a multi-layer stack.

A stack of any number of layers is translated with the same operation the
bilayer stage uses: everything above a chosen height moves rigidly.  Passing
``z_cutoff`` selects which layers move, so no separate N-layer code path is
needed; without it the cutoff falls in the widest gap of the structure.
"""

from __future__ import annotations

from ..moire import Moire


def translaten(**kwargs):
    return Moire().translate(**kwargs)

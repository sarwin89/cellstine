"""Compatibility shim for the dependency-free CLI parser.

The simplified CLI lives in :mod:`cellstine.cli.plain`. This module remains so
older internal tests and imports that ask for ``build_parser`` do not need to
know which frontend is active.
"""

from __future__ import annotations

from .plain import *  # noqa: F401,F403
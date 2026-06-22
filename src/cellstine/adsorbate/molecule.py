"""Concrete molecule-oriented adsorbate class."""

from __future__ import annotations

from .adsorbate import Adsorbate
from .placement.operations import (
    ATOMIC_MASSES,
    AdsorbRun,
    LayerShiftRun,
    MoleculeSelection,
    MoleculeTransformRun,
    center_of_mass_cartesian,
    identify_top_group,
    identify_top_molecule,
    place_molecule_on_site,
    shift_top_layer,
    transform_top_molecule,
)


class Molecule(Adsorbate):
    """Concrete molecule-level adsorbate workflow."""


__all__ = [
    "ATOMIC_MASSES",
    "AdsorbRun",
    "LayerShiftRun",
    "Molecule",
    "MoleculeSelection",
    "MoleculeTransformRun",
    "center_of_mass_cartesian",
    "identify_top_group",
    "identify_top_molecule",
    "place_molecule_on_site",
    "shift_top_layer",
    "transform_top_molecule",
]

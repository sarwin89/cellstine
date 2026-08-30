"""How isolated a generated defect really is inside its periodic cell.

A defect POSCAR is only a model of an *isolated* defect if the cell is big
enough that the defect does not see the copies of itself that the boundary
conditions create.  Nothing in the generation step enforces that -- a vacancy in
a one-by-one cell is perfectly well defined arithmetic, and perfectly useless
physics, because it removes a whole atomic plane rather than a single atom.

This module measures the two quantities that decide the question and turns them
into plain sentences the CLI prints next to the generated structures:

* the distance from the defect to its nearest periodic image, which is the
  length of a shortest lattice translation -- of the full three-dimensional
  lattice for a bulk cell, and of the in-plane lattice only for a slab, whose
  third translation is mostly vacuum;
* the fraction of the host atoms the defect touches, which is the defect
  concentration the cell actually represents.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.geometry import shortest_lattice_vector_length, shortest_plane_vector_length

#: Structure kinds whose third lattice vector is vacuum rather than material.
SLAB_KINDS = frozenset({"surface", "slab", "molecule-on-substrate"})

#: Image separation below which a point defect is not usefully isolated (A).
DILUTE_IMAGE_DISTANCE = 10.0

#: Defect concentration above which the cell models a compound, not a defect.
DILUTE_CONCENTRATION = 0.05

#: How many atoms each defect type adds to or removes from the host.
_TOUCHED_ATOMS = {
    "vacancy": 1,
    "substitution": 1,
    "antisite": 1,
    "interstitial": 1,
    "adatom": 1,
    "divacancy": 2,
    "paired-vacancy": 2,
}


def image_separation(lattice: np.ndarray, structure_kind: str) -> tuple[float, str]:
    """Return the defect-image distance and the periodicity it was measured in.

    For a slab the separation that matters is the in-plane one: the images
    stacked along ``c`` are held apart by the vacuum gap, which
    :mod:`cellstine.core.vacuum` reports separately.
    """

    array = np.asarray(lattice, dtype=float).reshape(3, 3)
    if str(structure_kind).lower() in SLAB_KINDS:
        return shortest_plane_vector_length(array[:2]), "in-plane"
    return shortest_lattice_vector_length(array), "three-dimensional"


def dilution_report(
    *,
    lattice: np.ndarray,
    structure_kind: str,
    host_atoms: int,
    defect_type: str,
) -> dict[str, Any]:
    """Describe how isolated a defect of ``defect_type`` is in this cell."""

    distance, periodicity = image_separation(lattice, structure_kind)
    touched = _TOUCHED_ATOMS.get(str(defect_type).lower(), 1)
    atoms = max(int(host_atoms), 1)
    concentration = float(touched) / float(atoms)
    notes: list[str] = []
    if distance < DILUTE_IMAGE_DISTANCE:
        notes.append(
            f"the nearest periodic image of the defect is {distance:.2f} A away "
            f"({periodicity} lattice); an isolated point defect usually needs at least "
            f"{DILUTE_IMAGE_DISTANCE:.0f} A, so build a supercell before using this cell "
            f"for a formation energy"
        )
    if concentration > DILUTE_CONCENTRATION:
        notes.append(
            f"the cell holds {atoms} host atom(s), so this defect is a "
            f"{100.0 * concentration:.1f}% concentration rather than a dilute defect"
        )
    return {
        "image_distance": float(distance),
        "image_periodicity": periodicity,
        "concentration": concentration,
        "notes": notes,
    }


def merge_notes(report: Mapping[str, Any], existing: list[str]) -> list[str]:
    """Append the report's notes to ``existing`` without repeating any."""

    merged = list(existing)
    for note in report.get("notes", []):
        if note not in merged:
            merged.append(note)
    return merged

"""How close the atoms of a generated structure really come to each other.

Every assembly step in the library places one group of atoms against another --
a molecule on a substrate, a slab on a slab, an interstitial in a host -- and
each of them is steered by a *height* or a *gap*, a single number measured along
the surface normal.  That number is a clearance, not a bond length: an atom `h`
above a hollow site is `sqrt(h^2 + d^2)` from the atoms `d` to either side of the
site, and an atom `h` above a corrugated surface can be much closer than `h` to a
neighbouring bump.  The distance a plane-wave calculation reacts to is the
shortest distance between two atoms, over the periodic images as well, and that
is what this module measures.

Three quantities come out of it:

* the closest contact *between* two groups of atoms, with the pair that makes it;
* the closest contact a group makes with its own periodic images, which is what
  decides whether a molecule is isolated in the cell it was given;
* the closest contact anywhere in a finished structure, over every pair of atoms
  and every periodic image, which is the one number that says whether the file
  can be run at all;
* a verdict on each of them, obtained by comparing the distance with the sum of
  the two covalent radii.

The measurements are exact.  Cross-group distances use the minimum-image search
of :mod:`cellstine.core.geometry`, which enumerates a provably sufficient box of
lattice shifts rather than rounding fractional coordinates.  The self-image
search enumerates every lattice translation that could beat the trivial
candidate -- a whole-cell translation, of length the shortest lattice vector --
which by the triangle inequality means every translation no longer than that
plus the diameter of the group.  The two bounds are proved in
``aristotle-lean-reference/RequestProject/ContactDistance.lean``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import geometry
from .elements import covalent_radii

__all__ = [
    "BOND_RATIO",
    "Contact",
    "OVERLAP_RATIO",
    "closest_contact",
    "closest_contact_cartesian",
    "contact_notes",
    "contact_report",
    "group_diameter",
    "layer_contact_report",
    "merge_notes",
    "self_image_contact",
    "structure_contact",
    "structure_contact_report",
    "unwrap_group",
]

#: Distance/covalent-radii ratio below which two atoms are on top of each other.
OVERLAP_RATIO = 0.75

#: Ratio below which a contact is shorter than a normal single bond.
BOND_RATIO = 0.90


@dataclass(frozen=True)
class Contact:
    """The closest approach found, and the pair of atoms that makes it."""

    distance: float
    first_index: int
    second_index: int
    first_species: str
    second_species: str
    covalent_sum: float

    @property
    def ratio(self) -> float:
        """The distance as a fraction of the sum of the two covalent radii."""

        if not np.isfinite(self.covalent_sum) or self.covalent_sum <= 0.0:
            return float("nan")
        return float(self.distance) / float(self.covalent_sum)

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance": float(self.distance),
            "pair": [int(self.first_index), int(self.second_index)],
            "species": [str(self.first_species), str(self.second_species)],
            "covalent_sum": float(self.covalent_sum),
            "ratio": float(self.ratio),
        }


def _labels(species: Sequence[str] | None, count: int) -> list[str]:
    if species is None:
        return ["X"] * int(count)
    labels = [str(symbol) for symbol in species]
    if len(labels) != int(count):
        raise ValueError("one species label per atom is required")
    return labels


def _covalent_sum(first_label: str, second_label: str) -> float:
    radii = covalent_radii([first_label, second_label])
    return float(radii[0] + radii[1])


def closest_contact(
    lattice: np.ndarray,
    first_direct: np.ndarray,
    second_direct: np.ndarray,
    *,
    first_species: Sequence[str] | None = None,
    second_species: Sequence[str] | None = None,
) -> Contact | None:
    """Return the closest approach between two groups of atoms.

    Both groups are given in fractional coordinates of the same cell, and the
    distance is the minimum-image one, so a contact made through a cell face
    counts.  ``None`` comes back when either group is empty.
    """

    cell = geometry.as_lattice(lattice)
    first = np.asarray(first_direct, dtype=float).reshape(-1, 3)
    second = np.asarray(second_direct, dtype=float).reshape(-1, 3)
    if first.shape[0] == 0 or second.shape[0] == 0:
        return None

    distances = geometry.pairwise_minimum_image_distances(cell, first, other_direct=second)
    flat = int(np.argmin(distances))
    row, column = divmod(flat, distances.shape[1])
    first_labels = _labels(first_species, first.shape[0])
    second_labels = _labels(second_species, second.shape[0])
    return Contact(
        distance=float(distances[row, column]),
        first_index=int(row),
        second_index=int(column),
        first_species=first_labels[row],
        second_species=second_labels[column],
        covalent_sum=_covalent_sum(first_labels[row], second_labels[column]),
    )


def closest_contact_cartesian(
    lattice: np.ndarray,
    first_cartesian: np.ndarray,
    second_cartesian: np.ndarray,
    *,
    first_species: Sequence[str] | None = None,
    second_species: Sequence[str] | None = None,
) -> Contact | None:
    """Closest approach between two groups given in Cartesian coordinates."""

    cell = geometry.as_lattice(lattice)
    inverse = np.linalg.inv(cell)
    return closest_contact(
        cell,
        np.asarray(first_cartesian, dtype=float).reshape(-1, 3) @ inverse,
        np.asarray(second_cartesian, dtype=float).reshape(-1, 3) @ inverse,
        first_species=first_species,
        second_species=second_species,
    )


def layer_contact_report(
    *,
    lattice: np.ndarray,
    first_cartesian: np.ndarray,
    second_cartesian: np.ndarray,
    first_species: Sequence[str] | None = None,
    second_species: Sequence[str] | None = None,
    subject: str = "interlayer",
    requested: float | None = None,
    requested_name: str = "gap",
) -> dict[str, Any]:
    """Measure and describe the closest approach two stacked layers make.

    A stack is built by setting a gap along the surface normal, and the gap is a
    clearance in that one direction; the closest approach between the two layers
    is what says whether the two surfaces are touching, and it is measured over
    the periodic images, so a contact made around a cell face counts.
    """

    contact = closest_contact_cartesian(
        lattice,
        first_cartesian,
        second_cartesian,
        first_species=first_species,
        second_species=second_species,
    )
    report: dict[str, Any] = {
        "notes": contact_notes(
            contact, subject=subject, requested=requested, requested_name=requested_name
        )
    }
    if contact is not None:
        report["contact_distance"] = float(contact.distance)
        report["contact"] = contact.as_dict()
    return report


def unwrap_group(lattice: np.ndarray, positions_direct: np.ndarray) -> np.ndarray:
    """Return the copy of a group that hangs together around its first atom.

    A molecule written into a POSCAR may be split across a cell face, so its
    fractional coordinates on their own do not describe one rigid body.  Taking
    each atom's shortest image relative to the first atom puts the body back
    together, which is the group whose size and whose separation from its own
    images are worth measuring.  It is the right group as soon as the body is
    smaller than the cell, which is the only case in which it is isolated at all.
    """

    cell = geometry.as_lattice(lattice)
    points = np.asarray(positions_direct, dtype=float).reshape(-1, 3)
    if points.shape[0] == 0:
        return points
    anchor = points[0]
    return anchor[None, :] + geometry.minimum_image_fractional(cell, points - anchor[None, :])


def group_diameter(lattice: np.ndarray, positions_direct: np.ndarray) -> float:
    """Return the largest distance between two atoms of a group.

    The group is first put back together with :func:`unwrap_group`: a molecule
    that has been placed on a substrate is a rigid body whose atoms belong
    together, and the size of that body is what bounds how far its periodic
    images can reach.
    """

    cell = geometry.as_lattice(lattice)
    points = unwrap_group(cell, positions_direct) @ cell
    if points.shape[0] < 2:
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).max())


def self_image_contact(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    species: Sequence[str] | None = None,
) -> Contact | None:
    """Return the closest approach between a group and its own periodic images.

    An image is the whole group translated by a nonzero lattice vector, so this
    is the minimum of ``|x_i - x_j - t|`` over all atom pairs and all ``t != 0``.
    Translating by a shortest lattice vector already gives a candidate of that
    length, so -- by the triangle inequality -- no translation longer than the
    shortest lattice vector plus the diameter of the group can do better, and
    the enumeration below is exact.
    """

    cell = geometry.as_lattice(lattice)
    points = unwrap_group(cell, positions_direct)
    if points.shape[0] == 0:
        return None

    diameter = group_diameter(cell, points)
    shortest = geometry.shortest_lattice_vector_length(cell)
    cutoff = float(shortest + diameter)
    reach = geometry.image_shift_reach(cell, cutoff) + 1
    shifts = geometry.lattice_shifts(reach)
    keep = np.any(shifts != 0.0, axis=1)
    vectors = shifts[keep] @ cell
    lengths = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    vectors = vectors[lengths <= cutoff + 1e-9]
    if vectors.shape[0] == 0:  # pragma: no cover - defensive; the box always holds one
        return None

    cartesian = points @ cell
    best = np.inf
    best_pair = (0, 0)
    for vector in vectors:
        deltas = cartesian[:, None, :] - (cartesian[None, :, :] + vector[None, None, :])
        distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        flat = int(np.argmin(distances))
        row, column = divmod(flat, distances.shape[1])
        if float(distances[row, column]) < best:
            best = float(distances[row, column])
            best_pair = (int(row), int(column))

    labels = _labels(species, points.shape[0])
    first, second = best_pair
    return Contact(
        distance=float(best),
        first_index=first,
        second_index=second,
        first_species=labels[first],
        second_species=labels[second],
        covalent_sum=_covalent_sum(labels[first], labels[second]),
    )


def structure_contact(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    species: Sequence[str] | None = None,
    coincident_tolerance: float = 1e-9,
) -> Contact | None:
    """Return the closest approach anywhere in a periodic structure.

    Two things can bring atoms together in a written cell: a pair of distinct
    atoms, over the periodic images as well, and a single atom against a copy of
    itself one lattice translation away.  The second is the same for every atom
    and equal to the shortest nonzero lattice vector, so the closest approach in
    the cell is the smaller of the two, and this function reports it together
    with the pair that makes it.  An atom paired with itself -- equal indices --
    is the second case, where it is the cell that is too small rather than the
    atoms that are badly placed.

    The search over distinct pairs grows a radius until it finds a pair; a
    radius that catches one pair also catches the closest one, so the answer is
    exact.  Coincident sites are skipped: a duplicated atom is a structural
    error that :func:`cellstine.core.validation.validate_structure` reports in
    those words, not a contact of zero length.
    """

    cell = geometry.as_lattice(lattice)
    points = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    count = points.shape[0]
    if count == 0:
        return None
    labels = _labels(species, count)

    # An atom against its own image: the shortest lattice translation, always.
    best = float(geometry.shortest_lattice_vector_length(cell))
    best_pair = (0, 0)

    if count >= 2:
        volume = abs(float(np.linalg.det(cell)))
        radius = min(best, max((volume / count) ** (1.0 / 3.0), 1e-6))
        while True:
            first, second = geometry.periodic_neighbour_pairs(cell, points, radius)
            found = False
            if first.size:
                distances = geometry.minimum_image_distances(
                    cell, points[first] - points[second]
                )
                keep = distances > float(coincident_tolerance)
                if np.any(keep):
                    found = True
                    order = int(np.argmin(np.where(keep, distances, np.inf)))
                    if float(distances[order]) < best:
                        best = float(distances[order])
                        best_pair = (int(first[order]), int(second[order]))
            if found or radius >= best:
                break
            radius = min(2.0 * radius, best)

    first_index, second_index = best_pair
    return Contact(
        distance=best,
        first_index=first_index,
        second_index=second_index,
        first_species=labels[first_index],
        second_species=labels[second_index],
        covalent_sum=_covalent_sum(labels[first_index], labels[second_index]),
    )


def structure_contact_report(
    *,
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    subject: str = "interatomic",
) -> dict[str, Any]:
    """Measure the closest approach of a finished structure and describe it.

    A workflow that reports only the contact it controls -- the gap between two
    layers, the height of a molecule -- says nothing about the rest of the cell,
    and a structure whose input was already unphysical passes such a check
    unremarked.  This is the report that covers the written file as a whole.
    """

    contact = structure_contact(lattice, positions_direct, species=species)
    report: dict[str, Any] = {"notes": []}
    if contact is None:
        return report
    report["structure_contact_distance"] = float(contact.distance)
    report["structure_contact"] = contact.as_dict()
    notes = contact_notes(
        contact,
        subject=subject,
        remedy="the cell cannot be run as it stands",
    )
    if contact.first_index == contact.second_index and not notes:
        notes.append(
            f"no two atoms are closer than the cell repeat itself, "
            f"{contact.distance:.2f} A"
        )
    report["notes"] = notes
    return report


def contact_notes(
    contact: Contact | None,
    *,
    subject: str,
    requested: float | None = None,
    requested_name: str = "height",
    remedy: str = "move the group further away before running this cell",
) -> list[str]:
    """Describe a contact in words, and say so when it is too short.

    ``requested`` is the clearance the placement was asked for; a note records
    that the closest approach is larger, which it must be, so the two numbers are
    never mistaken for each other.
    """

    if contact is None:
        return []
    notes: list[str] = []
    ratio = contact.ratio
    pair = f"{contact.first_species}-{contact.second_species}"
    if np.isfinite(ratio) and ratio < OVERLAP_RATIO:
        notes.append(
            f"the closest {subject} contact is {contact.distance:.2f} A ({pair}), only "
            f"{100.0 * ratio:.0f}% of the sum of the two covalent radii; the atoms overlap, "
            f"so {remedy}"
        )
    elif np.isfinite(ratio) and ratio < BOND_RATIO:
        notes.append(
            f"the closest {subject} contact is {contact.distance:.2f} A ({pair}), shorter than "
            f"the {contact.covalent_sum:.2f} A single bond of that pair; check that this is the "
            f"chemistry you meant"
        )
    if requested is not None and float(requested) > 0.0 and contact.distance > float(requested) + 0.05:
        notes.append(
            f"the requested {requested_name} of {float(requested):.2f} A is measured along the "
            f"surface normal, so the closest {subject} contact is the larger "
            f"{contact.distance:.2f} A ({pair})"
        )
    return notes


def contact_report(
    *,
    lattice: np.ndarray,
    group_direct: np.ndarray,
    other_direct: np.ndarray | None = None,
    group_species: Sequence[str] | None = None,
    other_species: Sequence[str] | None = None,
    subject: str = "molecule-substrate",
    self_subject: str = "molecule-image",
    requested: float | None = None,
    requested_name: str = "height",
) -> dict[str, Any]:
    """Measure both contacts of a placed group and describe them.

    The result carries the two distances, the pairs that make them, and the
    notes a CLI prints next to the written structure.
    """

    contact = (
        None
        if other_direct is None
        else closest_contact(
            lattice,
            group_direct,
            other_direct,
            first_species=group_species,
            second_species=other_species,
        )
    )
    image = self_image_contact(lattice, group_direct, species=group_species)
    notes = contact_notes(contact, subject=subject, requested=requested, requested_name=requested_name)
    notes.extend(contact_notes(image, subject=self_subject))
    report: dict[str, Any] = {"notes": notes}
    if contact is not None:
        report["contact_distance"] = float(contact.distance)
        report["contact"] = contact.as_dict()
    if image is not None:
        report["self_image_distance"] = float(image.distance)
        report["self_image_contact"] = image.as_dict()
    return report


def merge_notes(report: Mapping[str, Any], existing: Sequence[str]) -> list[str]:
    """Append a report's notes to ``existing`` without repeating any."""

    merged = list(existing)
    for note in report.get("notes", []):
        if note not in merged:
            merged.append(note)
    return merged

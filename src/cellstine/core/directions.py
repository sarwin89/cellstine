"""The direction a structure is looked along, and the planes it stacks into.

Every workflow that speaks of *layers* has to say along which direction the
layers are counted.  Historically that direction was always the ``a``-``b``
surface normal, which is the right answer for a slab written by this package
but not for a bulk cell that the user wants to read along ``[111]``, nor for a
picture that should be taken looking down a chosen crystal direction.

This module turns a short textual specification -- what the CLI calls the
*direction of observation* -- into a Cartesian unit vector, and reports what
that direction means for a periodic crystal:

``auto`` / ``normal`` / ``c*``
    the ``a``-``b`` plane normal oriented along ``+c``, the historic default.
``a``, ``b``, ``c``
    a lattice vector, i.e. the real-space direction ``[100]``, ``[010]``,
    ``[001]``.
``a*``, ``b*``, ``c*``
    a reciprocal vector, i.e. the normal of the ``(100)``, ``(010)``, ``(001)``
    planes.
``x``, ``y``, ``z``
    a Cartesian axis of the file, whatever the cell is doing.
``(hkl)`` or ``hkl`` or ``h,k,l``
    the normal of the ``(h k l)`` lattice planes.  Compact notation follows the
    rest of the package: ``1x1`` means ``(1 -1 1)``.
``[uvw]``
    the real-space direction ``u a + v b + w c``.
``cart:x,y,z``
    an explicit Cartesian vector.

Any specification may be prefixed with ``-`` to look the other way, which
reverses the order the layers are numbered in but not which atoms share a
layer.

Whatever the direction, the *planes* of a crystal perpendicular to it only
repeat when the direction is a lattice-plane normal.  :func:`resolve_direction`
detects that case -- by fitting small integers to the projections of the three
lattice vectors -- and reports the Miller family and the interplanar spacing
``d = 1 / ‖G‖``; when no such fit exists it says so, because then the layers
seen along that direction are an accident of the cell that was supplied and do
not describe a repeating stack.

``RequestProject/LayerPartition.lean`` proves the properties the layer census
relies on: that grouping the projections by a tolerance is exactly the
connected-component ("single-linkage") partition, that the partition does not
depend on where the origin is or on which way round the direction points, and
that reversing the direction reverses the numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Sequence

import numpy as np

from .vacuum import surface_normal

__all__ = [
    "DIRECTION_HELP",
    "ViewDirection",
    "orthonormal_frame",
    "resolve_direction",
    "project_along",
]


from .constants import DIRECTION_HELP  # re-exported for callers

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ViewDirection:
    """A direction of observation, resolved against one cell."""

    spec: str
    """The specification as the user wrote it."""

    unit: np.ndarray
    """Cartesian unit vector pointing the way the layers are counted."""

    label: str
    """Short human-readable name, e.g. ``(1 1 1) plane normal``."""

    miller: tuple[int, int, int] | None = None
    """Reduced Miller indices of the plane family, when the direction is one."""

    spacing: float | None = None
    """Distance between neighbouring lattice planes, in angstrom, when defined."""

    notes: list[str] = field(default_factory=list)

    @property
    def is_lattice_plane_normal(self) -> bool:
        """Whether the perpendicular planes repeat with the crystal."""

        return self.miller is not None

    def project(self, positions_cartesian: np.ndarray) -> np.ndarray:
        """Return the height of each Cartesian position along this direction."""

        return project_along(positions_cartesian, self.unit)

    def frame(self, lattice: np.ndarray | None = None) -> np.ndarray:
        """A right-handed frame whose third row is this direction.

        The first two rows span the planes perpendicular to the direction, so
        projecting the structure onto them is the picture an observer looking
        along it would see.
        """

        return orthonormal_frame(self.unit, lattice)

    def describe(self) -> str:
        """One line naming the direction and the planes it sees."""

        if self.miller is not None and self.spacing is not None:
            return (
                f"{self.label}: unit vector "
                f"({self.unit[0]:.4f}, {self.unit[1]:.4f}, {self.unit[2]:.4f}), "
                f"lattice planes {self.spacing:.4f} A apart"
            )
        return (
            f"{self.label}: unit vector "
            f"({self.unit[0]:.4f}, {self.unit[1]:.4f}, {self.unit[2]:.4f}), "
            "no repeating lattice planes are perpendicular to it"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec,
            "label": self.label,
            "unit": [float(value) for value in self.unit],
            "miller": None if self.miller is None else [int(value) for value in self.miller],
            "spacing": None if self.spacing is None else float(self.spacing),
        }


def project_along(positions_cartesian: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Project Cartesian positions onto a unit direction."""

    points = np.asarray(positions_cartesian, dtype=float).reshape(-1, 3)
    axis = np.asarray(unit, dtype=float).reshape(3)
    return points @ axis


def orthonormal_frame(unit: np.ndarray, lattice: np.ndarray | None = None) -> np.ndarray:
    """Return rows ``(u, v, n)``: a right-handed frame with ``n`` along ``unit``.

    ``u`` is chosen inside the perpendicular plane and, when a cell is given,
    as close to its ``a`` vector as that plane allows, so a picture taken along
    the direction keeps the familiar orientation of the cell instead of jumping
    to an arbitrary one.  ``v = n x u`` completes the frame, which is therefore
    a rotation: it changes no distance and no angle in the structure.
    """

    normal = _unit(unit, what="view direction")
    candidates: list[np.ndarray] = []
    if lattice is not None:
        rows = np.asarray(lattice, dtype=float).reshape(3, 3)
        candidates.extend([rows[0], rows[1], rows[2]])
    candidates.extend([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])])
    for candidate in candidates:
        residual = candidate - float(np.dot(candidate, normal)) * normal
        length = float(np.linalg.norm(residual))
        if length > 1e-8 * max(1.0, float(np.linalg.norm(candidate))):
            first = residual / length
            second = np.cross(normal, first)
            return np.vstack([first, second, normal])
    raise ValueError("no direction perpendicular to the view direction could be found")


def _reciprocal_rows(lattice: np.ndarray) -> np.ndarray:
    """Rows ``b1, b2, b3`` with ``a_i . b_j = delta_ij``."""

    rows = np.asarray(lattice, dtype=float).reshape(3, 3)
    if abs(float(np.linalg.det(rows))) <= _TOLERANCE:
        raise ValueError("the cell is degenerate; no direction can be resolved against it")
    return np.linalg.inv(rows).T


def _unit(vector: np.ndarray, *, what: str) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(values))
    if length <= _TOLERANCE:
        raise ValueError(f"{what} has zero length")
    return values / length


def _parse_triple(text: str) -> tuple[int, int, int]:
    """Parse ``1,1,1``, ``111`` or ``1x1`` into three integers.

    A trailing ``x`` negates the digit it follows, which is the compact
    notation the surface and interface stages already accept.
    """

    raw = text.strip()
    if not raw:
        raise ValueError("empty index triple")
    if any(separator in raw for separator in (",", ";", " ")):
        tokens = [
            token.strip()
            for token in raw.replace(";", ",").replace(" ", ",").split(",")
            if token.strip()
        ]
    else:
        tokens = []
        index = 0
        while index < len(raw):
            char = raw[index]
            if char == "-":
                if index + 1 >= len(raw) or not raw[index + 1].isdigit():
                    raise ValueError(f"cannot read '{text}' as three indices")
                tokens.append(raw[index : index + 2])
                index += 2
                continue
            if not char.isdigit():
                raise ValueError(f"cannot read '{text}' as three indices")
            token = char
            if index + 1 < len(raw) and raw[index + 1].lower() == "x":
                token += "x"
                index += 1
            tokens.append(token)
            index += 1
    if len(tokens) != 3:
        raise ValueError(f"cannot read '{text}' as three indices")
    values = []
    for token in tokens:
        if token.lower().endswith("x"):
            values.append(-int(token[:-1]))
        else:
            values.append(int(token))
    if values == [0, 0, 0]:
        raise ValueError("an index triple cannot be all zero")
    return int(values[0]), int(values[1]), int(values[2])


def _parse_floats(text: str) -> np.ndarray:
    tokens = [
        token.strip()
        for token in text.replace(";", ",").replace(" ", ",").split(",")
        if token.strip()
    ]
    if len(tokens) != 3:
        raise ValueError(f"cannot read '{text}' as three components")
    return np.array([float(token) for token in tokens], dtype=float)


def _reduce_indices(indices: Sequence[int]) -> tuple[tuple[int, int, int], int]:
    values = [int(value) for value in indices]
    divisor = 0
    for value in values:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        raise ValueError("an index triple cannot be all zero")
    reduced = tuple(value // divisor for value in values)
    return (int(reduced[0]), int(reduced[1]), int(reduced[2])), int(divisor)


def _plane_family(
    lattice: np.ndarray, unit: np.ndarray, *, max_denominator: int = 32
) -> tuple[tuple[int, int, int], float] | None:
    """Return the Miller family perpendicular to ``unit`` and its spacing.

    ``unit`` is the normal of a lattice-plane family exactly when the three
    numbers ``a_i . unit`` are commensurate: they are then ``h_i / ‖G‖`` for the
    integer indices of the family.  Small integers are fitted to their ratios,
    and the fit is accepted only when it reproduces the projections, so an
    irrational direction -- one along which the crystal never repeats -- is
    reported as such instead of being rounded into a nearby family.
    """

    rows = np.asarray(lattice, dtype=float).reshape(3, 3)
    projections = rows @ np.asarray(unit, dtype=float).reshape(3)
    scale = float(np.max(np.abs(projections)))
    if scale <= _TOLERANCE:
        return None
    ratios = projections / scale
    fractions = [Fraction(float(value)).limit_denominator(max_denominator) for value in ratios]
    denominator = 1
    for value in fractions:
        denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    indices = [int(value * denominator) for value in fractions]
    if all(value == 0 for value in indices):
        return None
    if max(abs(value) for value in indices) > max_denominator:
        # Indices this large mean the fit is chasing an irrational ratio with a
        # near-coincidence, and the "spacing" it would report is a fraction of a
        # picometre: no family of lattice planes is perpendicular to the vector.
        return None
    step = scale / float(denominator)
    fitted = np.array([float(value) * step for value in indices], dtype=float)
    # POSCAR coordinates carry a handful of decimals, so an exactly hexagonal
    # cell reaches here slightly out of true; the window is wide enough to read
    # such a cell and far too narrow to round an irrational direction into a
    # family, whose fitted projections miss by a sizeable fraction of an angstrom.
    tolerance = 1e-3 * max(1.0, scale)
    if not np.allclose(fitted, projections, atol=tolerance, rtol=0.0):
        return None
    reduced, divisor = _reduce_indices(indices)
    # The lattice projections are the integer combinations of ``indices * step``,
    # i.e. the multiples of ``gcd(indices) * step``: that is the plane spacing.
    return reduced, abs(step) * divisor


def _reads_as_signed_triple(body: str) -> bool:
    """Whether ``-`` in front of ``body`` is the sign of its first index.

    A separated triple spells its own signs -- ``-1,1,1`` is the family
    ``(-1 1 1)`` -- while a leading ``-`` on anything else means *look the other
    way*, so ``-111`` reverses ``111`` and ``1x11`` is how ``(-1 1 1)`` is
    written compactly.
    """

    text = body.strip()
    if not text or not text[:1].isdigit():
        return False
    return any(separator in text for separator in (",", ";", " "))


def resolve_direction(lattice: np.ndarray, spec: str | Sequence[float] | None = None) -> ViewDirection:
    """Resolve a direction-of-observation specification against a cell.

    See the module docstring for the accepted forms.  The returned direction
    always carries a Cartesian unit vector; when the perpendicular planes are
    lattice planes it also carries their Miller indices and spacing.
    """

    rows = np.asarray(lattice, dtype=float).reshape(3, 3)
    if spec is None:
        text = "auto"
    elif isinstance(spec, str):
        text = spec.strip()
    else:
        values = np.asarray(spec, dtype=float).reshape(-1)
        if values.size != 3:
            raise ValueError("a direction given as numbers needs exactly three components")
        text = "cart:" + ",".join(repr(float(value)) for value in values)
    if not text:
        text = "auto"

    sign = 1.0
    body = text
    while body[:1] == "-" and not _reads_as_signed_triple(body[1:]):
        sign = -sign
        body = body[1:].strip()
    lowered = body.lower()

    reciprocal = _reciprocal_rows(rows)
    notes: list[str] = []
    vector: np.ndarray
    label: str

    if lowered in {"", "auto", "normal", "surface", "surface-normal", "surface_normal"}:
        vector = surface_normal(rows)
        label = "a-b surface normal"
    elif lowered in {"a", "b", "c"}:
        axis = {"a": 0, "b": 1, "c": 2}[lowered]
        vector = rows[axis]
        label = f"{lowered} lattice vector"
    elif lowered in {"a*", "b*", "c*"}:
        axis = {"a*": 0, "b*": 1, "c*": 2}[lowered]
        vector = reciprocal[axis]
        label = f"{lowered} reciprocal vector"
    elif lowered in {"x", "y", "z"}:
        axis = {"x": 0, "y": 1, "z": 2}[lowered]
        vector = np.eye(3)[axis]
        label = f"Cartesian {lowered}"
    elif lowered.startswith("cart:") or lowered.startswith("cartesian:"):
        vector = _parse_floats(body.split(":", 1)[1])
        label = "Cartesian direction"
    elif body.startswith("[") and body.endswith("]"):
        indices = _parse_triple(body[1:-1])
        reduced, _ = _reduce_indices(indices)
        vector = np.asarray(indices, dtype=float) @ rows
        label = f"[{reduced[0]} {reduced[1]} {reduced[2]}] lattice direction"
    elif body.startswith("(") and body.endswith(")"):
        indices = _parse_triple(body[1:-1])
        reduced, _ = _reduce_indices(indices)
        vector = np.asarray(indices, dtype=float) @ reciprocal
        label = f"({reduced[0]} {reduced[1]} {reduced[2]}) plane normal"
    else:
        indices = _parse_triple(body)
        reduced, _ = _reduce_indices(indices)
        vector = np.asarray(indices, dtype=float) @ reciprocal
        label = f"({reduced[0]} {reduced[1]} {reduced[2]}) plane normal"

    unit = sign * _unit(vector, what=f"direction '{text}'")
    family = _plane_family(rows, unit)
    if family is None:
        notes.append(
            f"No lattice plane is perpendicular to {label}: the crystal does not repeat along it, "
            "so the layers reported are those of the supplied cell only."
        )
        return ViewDirection(spec=text, unit=unit, label=label, notes=notes)
    miller, spacing = family
    if sign < 0.0:
        label = f"-{label}"
    return ViewDirection(
        spec=text,
        unit=unit,
        label=label,
        miller=miller,
        spacing=float(spacing),
        notes=notes,
    )

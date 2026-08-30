"""Reciprocal lattices and Brillouin-zone sampling meshes.

Everything here works with *row* lattices, the convention used by
:class:`cellstine.io.models.StructureRecord`: ``lattice[i]`` is the Cartesian
vector of the ``i``-th basis vector, so a site with fractional coordinates ``x``
(a row) sits at ``x @ lattice``.

The reciprocal basis is the one used by plane-wave codes,

.. code-block:: text

    a_i . b_j = 2 pi delta_ij,        B = 2 pi (A^-1)^T,

so a plane wave ``exp(i G . r)`` with ``G = m @ B`` and integer ``m`` has the
periodicity of the cell.  A wavevector is carried in *fractional* reciprocal
coordinates ``k`` (a row), meaning the Cartesian vector ``k @ B``; the first
Brillouin zone is then represented by ``k`` in ``[-1/2, 1/2)``.

A sampling mesh is the finite set

.. code-block:: text

    k(i) = ((i_1 + s_1) / n_1, (i_2 + s_2) / n_2, (i_3 + s_3) / n_3),

with ``i_j`` running over ``0, ..., n_j - 1``.  ``s = 0`` is the Gamma-centred
mesh; the Monkhorst-Pack mesh is ``s_j = 1/2`` whenever ``n_j`` is even, which
is the choice that keeps the mesh centred on the zone rather than on Gamma.

Two facts make the mesh reduction exact rather than a tolerance-driven search.
First, a mesh point is a rational vector whose denominators divide ``2 n_j``, so
every point of every mesh in play is an integer vector over the common
denominator ``D = lcm(2 n_1, 2 n_2, 2 n_3)`` and can be compared exactly.
Second, a crystal symmetry acts on fractional reciprocal coordinates by the
*integer* matrix ``W^-1`` on the right (``k -> k W^-1`` for the operation
``x -> W x + w`` on column fractional coordinates), because ``W`` is unimodular.
Reduction is therefore integer arithmetic modulo ``D``, with no distance
tolerance anywhere, and the weights it reports are exact orbit sizes.

An operation is only usable if it maps the mesh onto itself.  That always holds
for a Gamma-centred mesh whose divisions respect the symmetry, but a shifted
mesh, or a mesh with unequal divisions along symmetry-related axes, can break
some operations; those are detected and dropped, and the count of the ones that
survived is reported so the loss is visible rather than silent.

The formal statements behind this module are in
``RequestProject/ReciprocalMesh.lean``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "reciprocal_lattice",
    "cell_volume",
    "brillouin_zone_volume",
    "mesh_divisions_for_spacing",
    "mesh_spacings",
    "mesh_shift",
    "mesh_points",
    "kpoint_density",
    "supercell_divisions",
    "KpointMesh",
    "build_mesh",
]


def _as_lattice(lattice: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(lattice, dtype=float)
    if array.shape != (3, 3):
        raise ValueError("a lattice must be a 3x3 array of row vectors")
    if not np.all(np.isfinite(array)):
        raise ValueError("a lattice must be finite")
    if abs(float(np.linalg.det(array))) <= 0.0:
        raise ValueError("a lattice must have a nonzero volume")
    return array


def cell_volume(lattice: Sequence[Sequence[float]]) -> float:
    """Return the volume of the cell, always positive."""

    return abs(float(np.linalg.det(_as_lattice(lattice))))


def reciprocal_lattice(lattice: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the reciprocal basis rows ``b_i`` with ``a_i . b_j = 2 pi delta_ij``.

    The inverse is taken through a solve rather than an explicit matrix inverse,
    which is both faster and better conditioned for a skewed cell.
    """

    array = _as_lattice(lattice)
    return 2.0 * math.pi * np.linalg.solve(array, np.eye(3)).T


def brillouin_zone_volume(lattice: Sequence[Sequence[float]]) -> float:
    """Return the volume of the Brillouin zone, ``(2 pi)^3 / V``."""

    return (2.0 * math.pi) ** 3 / cell_volume(lattice)


def mesh_divisions_for_spacing(
    lattice: Sequence[Sequence[float]],
    spacing: float,
    *,
    minimum: int | Sequence[int] = 1,
) -> tuple[int, int, int]:
    """Return the mesh divisions that sample no coarser than ``spacing``.

    ``spacing`` is a distance in reciprocal space, in inverse angstrom and in the
    ``2 pi`` convention -- the same quantity VASP calls ``KSPACING``.  Dividing
    ``b_i`` into ``n_i = ceil(|b_i| / spacing)`` steps makes every step at most
    ``spacing`` long, which is the guarantee :func:`mesh_spacings` checks.

    A non-periodic direction (a slab normal, say) is handled by the caller
    passing ``minimum`` per axis, or by reading a single division off the
    returned tuple; the formula itself has no notion of a surface.
    """

    value = float(spacing)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("the k-point spacing must be a finite positive length in 1/angstrom")
    lengths = np.linalg.norm(reciprocal_lattice(lattice), axis=1)
    floors = np.broadcast_to(np.asarray(minimum, dtype=np.int64).ravel(), (3,))
    if np.any(floors < 1):
        raise ValueError("every mesh division must be at least one")
    divisions = np.maximum(np.ceil(lengths / value - 1e-12).astype(np.int64), floors)
    return tuple(int(item) for item in divisions)


def _as_divisions(divisions: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(item) for item in np.asarray(divisions).ravel())
    if len(values) != 3:
        raise ValueError("a mesh needs three divisions")
    if any(item < 1 for item in values):
        raise ValueError("every mesh division must be at least one")
    return values


def mesh_spacings(
    lattice: Sequence[Sequence[float]], divisions: Sequence[int]
) -> tuple[float, float, float]:
    """Return the sampling step ``|b_i| / n_i`` along each reciprocal axis."""

    counts = _as_divisions(divisions)
    lengths = np.linalg.norm(reciprocal_lattice(lattice), axis=1)
    return tuple(float(length) / float(count) for length, count in zip(lengths, counts))


def kpoint_density(lattice: Sequence[Sequence[float]], divisions: Sequence[int]) -> float:
    """Return the number of mesh points per unit volume of the Brillouin zone."""

    counts = _as_divisions(divisions)
    total = float(counts[0] * counts[1] * counts[2])
    return total / brillouin_zone_volume(lattice)


def mesh_shift(divisions: Sequence[int], mode: str = "gamma") -> tuple[float, float, float]:
    """Return the fractional offset, in grid steps, of a named mesh.

    ``gamma`` returns a zero offset, so the mesh contains Gamma.  ``monkhorst``
    returns a half step along every axis with an even division, which is the
    original Monkhorst-Pack choice: it centres the mesh on the zone, and for an
    odd division the two coincide.
    """

    counts = _as_divisions(divisions)
    name = str(mode).lower()
    if name in {"gamma", "gamma-centred", "gamma-centered", "g"}:
        return (0.0, 0.0, 0.0)
    if name in {"monkhorst", "monkhorst-pack", "mp"}:
        return tuple(0.5 if count % 2 == 0 else 0.0 for count in counts)
    raise ValueError("mesh mode must be 'gamma' or 'monkhorst'")


def _shift_numerators(shift: Sequence[float]) -> tuple[int, int, int]:
    """Return ``2 * shift`` as integers, refusing anything but 0 or 1/2 steps."""

    values = np.asarray(shift, dtype=float).ravel()
    if values.shape != (3,):
        raise ValueError("a mesh shift needs three components")
    doubled = 2.0 * values
    rounded = np.rint(doubled)
    if not np.allclose(doubled, rounded, atol=1e-9):
        raise ValueError("a mesh shift must be a whole or half grid step")
    return tuple(int(item) % 2 for item in rounded)


def _mesh_numerators(
    divisions: tuple[int, int, int], shift_doubled: tuple[int, int, int]
) -> tuple[np.ndarray, int]:
    """Return the mesh points as integer numerators over a common denominator.

    The point ``((i_j + s_j) / n_j)_j`` is written as ``v / D`` with
    ``D = lcm(2 n_1, 2 n_2, 2 n_3)`` and ``v`` integer, wrapped into
    ``[-D/2, D/2)`` so that the fractional coordinates lie in ``[-1/2, 1/2)``.
    """

    denominator = 1
    for count in divisions:
        denominator = math.lcm(denominator, 2 * count)
    axes = []
    for count, doubled, in zip(divisions, shift_doubled):
        step = denominator // (2 * count)
        raw = (2 * np.arange(count, dtype=np.int64) + int(doubled)) * step
        axes.append(_wrap_numerators(raw, denominator))
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in grid], axis=1)
    return points, denominator


def _wrap_numerators(values: np.ndarray, denominator: int) -> np.ndarray:
    """Wrap integer numerators over ``denominator`` into ``[-D/2, D/2)``."""

    wrapped = np.mod(np.asarray(values, dtype=np.int64), denominator)
    return np.where(2 * wrapped >= denominator, wrapped - denominator, wrapped)


def _encode(points: np.ndarray, denominator: int) -> np.ndarray:
    """Return a collision-free integer key for wrapped numerators."""

    shifted = points.astype(np.int64) + denominator  # non-negative, < 2 D
    base = np.int64(2 * denominator)
    return (shifted[:, 0] * base + shifted[:, 1]) * base + shifted[:, 2]


def _integer_inverse(rotation: np.ndarray) -> np.ndarray:
    """Return the inverse of a unimodular integer matrix, exactly."""

    matrix = np.asarray(rotation, dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("a rotation must be a 3x3 integer matrix")
    determinant = int(round(float(np.linalg.det(matrix.astype(float)))))
    if determinant not in (1, -1):
        raise ValueError("a crystallographic rotation must have determinant +-1")
    adjugate = np.rint(np.linalg.inv(matrix.astype(float)) * determinant).astype(np.int64)
    inverse = adjugate * determinant
    if not np.array_equal(inverse @ matrix, np.eye(3, dtype=np.int64)):
        raise ValueError("a rotation must be an invertible integer matrix")
    return inverse


def _mesh_images(
    points: np.ndarray, denominator: int, rotation_inverse: np.ndarray
) -> np.ndarray:
    """Return the wrapped numerators of ``k W^-1`` for every mesh point."""

    images = points @ rotation_inverse.astype(np.int64)
    return _wrap_numerators(images, denominator)


@dataclass(frozen=True)
class KpointMesh:
    """A Brillouin-zone sampling mesh and its symmetry-reduced points.

    ``points`` and ``weights`` are the reduced list: one representative per
    orbit of the mesh under the operations that survived, with the orbit size as
    its weight.  ``full_point_count`` is the size of the unreduced mesh, so the
    weights always add up to it.  ``operations_given`` is the size of the group
    generated by the rotations that were handed over and ``operations_used`` how
    many of them the mesh can carry: a shifted mesh, or one whose divisions do
    not respect the symmetry, need not be invariant under all of them.
    """

    divisions: tuple[int, int, int]
    shift: tuple[float, float, float]
    points: np.ndarray
    weights: np.ndarray
    full_point_count: int
    operations_used: int
    operations_given: int
    time_reversal: bool
    spacings: tuple[float, float, float]

    @property
    def point_count(self) -> int:
        """Return the number of irreducible points actually listed."""

        return int(len(self.points))

    @property
    def normalised_weights(self) -> np.ndarray:
        """Return the weights scaled to sum to one."""

        return self.weights.astype(float) / float(self.full_point_count)

    @property
    def symmetry_complete(self) -> bool:
        """Return whether every supplied operation could be used."""

        return int(self.operations_used) >= int(self.operations_given)

    def cartesian_points(self, lattice: Sequence[Sequence[float]]) -> np.ndarray:
        """Return the irreducible points as Cartesian wavevectors in 1/angstrom."""

        return np.asarray(self.points, dtype=float) @ reciprocal_lattice(lattice)

    def summary(self) -> dict[str, object]:
        """Return a JSON-ready description of the mesh."""

        return {
            "divisions": [int(item) for item in self.divisions],
            "shift": [float(item) for item in self.shift],
            "full_point_count": int(self.full_point_count),
            "irreducible_point_count": int(self.point_count),
            "weight_total": int(np.sum(self.weights)),
            "operations_given": int(self.operations_given),
            "operations_used": int(self.operations_used),
            "time_reversal": bool(self.time_reversal),
            "spacings": [float(item) for item in self.spacings],
            "max_spacing": float(max(self.spacings)),
        }


def mesh_points(
    divisions: Sequence[int], shift: Sequence[float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """Return every point of the unreduced mesh, in ``[-1/2, 1/2)``.

    The points are exact multiples of ``1 / (2 n_j)`` computed in integers and
    divided once, so a mesh point that should be Gamma is exactly zero.
    """

    counts = _as_divisions(divisions)
    numerators, denominator = _mesh_numerators(counts, _shift_numerators(shift))
    return numerators.astype(float) / float(denominator)


def _close_group(matrices: Sequence[np.ndarray], *, limit: int = 96) -> list[np.ndarray]:
    """Return the group generated by integer matrices, including the identity.

    The caller may hand over a full point group or only a few generators, and
    the orbit pass below is only exact for a group, so the closure is taken
    here.  A crystallographic point group in three dimensions has at most 48
    elements, so a closure that grows past ``limit`` means the input was not a
    point group at all.
    """

    identity = np.eye(3, dtype=np.int64)
    elements: dict[bytes, np.ndarray] = {identity.tobytes(): identity}
    frontier = [identity]
    for matrix in matrices:
        key = matrix.tobytes()
        if key not in elements:
            elements[key] = matrix
            frontier.append(matrix)
    generators = list(elements.values())
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = current @ generator
            key = product.tobytes()
            if key in elements:
                continue
            if len(elements) >= limit:
                raise ValueError("the supplied rotations do not generate a point group")
            elements[key] = product
            frontier.append(product)
    return list(elements.values())


def _usable_operations(
    points: np.ndarray,
    denominator: int,
    rotations: Iterable[Sequence[Sequence[int]]],
) -> tuple[list[np.ndarray], int]:
    """Return the inverses of the operations that map the mesh onto itself.

    The rotations are closed into a group first, so the returned maps are a
    subgroup: an operation that sends the mesh into itself is a bijection of a
    finite set, and those bijections are closed under composition and inverse.
    """

    keys = _encode(points, denominator)
    sorted_keys = np.sort(keys)
    integer_rotations: list[np.ndarray] = []
    for rotation in rotations:
        matrix = np.rint(np.asarray(rotation, dtype=float)).astype(np.int64)
        if not np.allclose(np.asarray(rotation, dtype=float), matrix, atol=1e-8):
            raise ValueError("a crystallographic rotation must be an integer matrix")
        integer_rotations.append(matrix)
    group = _close_group(integer_rotations)
    given = len(group)
    usable: list[np.ndarray] = []
    for matrix in group:
        inverse = _integer_inverse(matrix)
        image_keys = _encode(_mesh_images(points, denominator, inverse), denominator)
        located = np.searchsorted(sorted_keys, image_keys)
        if np.any(located >= len(sorted_keys)):
            continue
        if not np.array_equal(sorted_keys[located], image_keys):
            continue
        usable.append(inverse)
    return usable, given


def _orbit_labels(
    points: np.ndarray,
    denominator: int,
    inverses: Sequence[np.ndarray],
    *,
    time_reversal: bool,
) -> np.ndarray:
    """Return, for every mesh point, the lowest index of its orbit.

    The usable operations form a group and time reversal commutes with all of
    them, so the orbit of a point is exactly its set of images under the listed
    maps: one pass of minima over those images is already the orbit minimum, and
    no iteration to a fixed point is needed.
    """

    keys = _encode(points, denominator)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    labels = np.arange(len(points), dtype=np.int64)
    maps = list(inverses)
    if not maps:
        maps = [np.eye(3, dtype=np.int64)]
    for inverse in maps:
        images = _mesh_images(points, denominator, inverse)
        for signed in ((images,) if not time_reversal else (images, _wrap_numerators(-images, denominator))):
            located = order[np.searchsorted(sorted_keys, _encode(signed, denominator))]
            np.minimum(labels, located, out=labels)
    return labels


def build_mesh(
    lattice: Sequence[Sequence[float]],
    *,
    spacing: float | None = None,
    divisions: Sequence[int] | None = None,
    mode: str = "gamma",
    shift: Sequence[float] | None = None,
    rotations: Sequence[Sequence[Sequence[int]]] | None = None,
    time_reversal: bool = True,
    minimum_divisions: int | Sequence[int] = 1,
) -> KpointMesh:
    """Return the sampling mesh of ``lattice``, reduced by the given symmetry.

    Exactly one of ``spacing`` and ``divisions`` fixes the mesh size.  ``mode``
    selects the standard Gamma-centred or Monkhorst-Pack offset, and ``shift``
    overrides it with an explicit offset in grid steps.  ``rotations`` are the
    integer rotation parts of the space-group operations acting on column
    fractional coordinates, as returned by
    :func:`cellstine.core.symmetry3d.symmetry_operations`; passing none reduces
    by time reversal alone.
    """

    array = _as_lattice(lattice)
    if (spacing is None) == (divisions is None):
        raise ValueError("give either a k-point spacing or explicit mesh divisions")
    if divisions is None:
        counts = mesh_divisions_for_spacing(array, float(spacing), minimum=minimum_divisions)
    else:
        counts = _as_divisions(divisions)
    offset = tuple(float(item) for item in (mesh_shift(counts, mode) if shift is None else shift))
    doubled = _shift_numerators(offset)
    numerators, denominator = _mesh_numerators(counts, doubled)
    supplied = [] if rotations is None else list(rotations)
    usable, given = _usable_operations(numerators, denominator, supplied)
    labels = _orbit_labels(numerators, denominator, usable, time_reversal=bool(time_reversal))
    representatives, weights = np.unique(labels, return_counts=True)
    reduced = numerators[representatives].astype(float) / float(denominator)
    return KpointMesh(
        divisions=counts,
        shift=offset,
        points=reduced,
        weights=weights.astype(np.int64),
        full_point_count=int(len(numerators)),
        operations_used=int(len(usable)),
        operations_given=int(given),
        time_reversal=bool(time_reversal),
        spacings=mesh_spacings(array, counts),
    )


def supercell_divisions(
    divisions: Sequence[int], supercell_matrix: Sequence[Sequence[int]]
) -> tuple[int, int, int]:
    """Return the divisions that keep the sampling density under a supercell.

    A supercell ``A' = M A`` has the reciprocal basis ``B' = M^-T B``, so its
    Brillouin zone is smaller by ``|det M|`` and needs that many fewer points to
    sample just as finely.  For a diagonal ``M`` the exact answer is
    ``n'_i = ceil(n_i / m_i)``, and that is what this returns; a general integer
    ``M`` mixes the axes, so the divisions are taken from the reciprocal lengths
    of the two cells instead, through :func:`mesh_divisions_for_spacing`.
    """

    counts = _as_divisions(divisions)
    matrix = np.rint(np.asarray(supercell_matrix, dtype=float)).astype(np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("a supercell matrix must be 3x3")
    if not np.allclose(np.asarray(supercell_matrix, dtype=float), matrix, atol=1e-9):
        raise ValueError("a supercell matrix must be an integer matrix")
    if abs(int(round(float(np.linalg.det(matrix.astype(float)))))) < 1:
        raise ValueError("a supercell matrix must be non-singular")
    if not np.array_equal(matrix, np.diag(np.diagonal(matrix))):
        raise ValueError(
            "only a diagonal supercell keeps the axes separate; use "
            "mesh_divisions_for_spacing on the supercell lattice instead"
        )
    repeats = np.abs(np.diagonal(matrix))
    if np.any(repeats < 1):
        raise ValueError("a supercell repeat must be at least one")
    return tuple(int(-(-count // int(repeat))) for count, repeat in zip(counts, repeats))

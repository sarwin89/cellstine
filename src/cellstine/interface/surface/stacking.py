"""Close-packed stacking sequences and stacking reversal.

A close-packed slab is a stack of parallel layers whose in-plane point sets are
translates of one another by a *hollow* vector: the layers sit on the three
cosets of the layer lattice ``L`` inside ``(1/3)L`` that the triangular lattice
singles out, traditionally called ``A``, ``B`` and ``C``.  This module reads
that sequence off a structure and reverses it, turning ``ABCABC`` into
``CBACBA``; the companion module ``registry.py`` enumerates the genuinely
different ways two such slabs can meet.

The letters carry two gauge freedoms, both proved in
``RequestProject/StackingRegistry.lean`` to be freedoms of the labels only:

* The origin is arbitrary.  Translating the whole interface by one hollow
  vector turns ``A-A`` into ``B-B`` and then into ``C-C``, so only the
  difference ``delta`` of the two contacting letters is physical: there are
  three contacts, not nine.
* The direction that counts as ``A -> B -> C`` is arbitrary too, and swapping
  it is realised by a reflection, which negates every layer step.  A slab on
  its own therefore has no handedness, and ``analyse_stacking`` reads every
  uniform close-packed slab as ``ABCABC`` unless it is given the hollow vector
  of another slab as a gauge.  Handedness is meaningful only between the two
  slabs: the bottom slab fixes the gauge and the top slab is described in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...core.layers import LAYER_TOLERANCE as _LAYER_TOLERANCE, layer_partition
from ...core.reduction import gauss_reduction_multiplier
from ...core.vacuum import normal_heights, surface_normal
from ...io import native as io_mod

__all__ = [
    "registry_shift_direct",
    "sense_label",
    "StackingLayer",
    "StackingAnalysis",
    "analyse_stacking",
    "mirror_structure",
    "shift_structure_inplane",
    "apply_relative_stacking",
    "normalise_stacking_choice",
]

LAYER_TOLERANCE = _LAYER_TOLERANCE
"""Default half-width in angstrom of a layer along the surface normal."""

POSITION_TOLERANCE = 0.05
"""Default angstrom tolerance when deciding whether two sites coincide."""

COSET_LETTERS = "ABC"


# ---------------------------------------------------------------------------
# small lattice helpers
# ---------------------------------------------------------------------------

def _gauss_reduce(basis: np.ndarray) -> np.ndarray:
    """Return a Lagrange-Gauss reduced basis of the 2D lattice ``basis`` spans."""

    vectors = np.array(basis, dtype=float, copy=True)
    for _ in range(64):
        if vectors[0] @ vectors[0] > vectors[1] @ vectors[1]:
            vectors = vectors[[1, 0]]
        norm = float(vectors[0] @ vectors[0])
        if norm <= 1e-18:
            break
        mu = gauss_reduction_multiplier(float(vectors[0] @ vectors[1]), norm)
        if mu == 0:
            break
        vectors[1] = vectors[1] - mu * vectors[0]
    if vectors[0] @ vectors[0] > vectors[1] @ vectors[1]:
        vectors = vectors[[1, 0]]
    return vectors


def _integer_coefficients(vector: np.ndarray, basis: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Return the integer coefficients of ``vector`` in ``basis``, or ``None``."""

    basis = np.asarray(basis, dtype=float)
    try:
        coefficients = np.linalg.solve(basis.T, np.asarray(vector, dtype=float))
    except np.linalg.LinAlgError:
        return None
    rounded = np.round(coefficients)
    if float(np.max(np.abs((coefficients - rounded) @ basis))) > tolerance:
        return None
    return rounded.astype(int)


def _reduce_modulo_lattice(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return the representative of ``vector`` closest to the origin."""

    basis = np.asarray(basis, dtype=float)
    coefficients = np.linalg.solve(basis.T, np.asarray(vector, dtype=float))
    best = np.asarray(vector, dtype=float) - np.round(coefficients) @ basis
    for first in (-1, 0, 1):
        for second in (-1, 0, 1):
            candidate = best + first * basis[0] + second * basis[1]
            if float(candidate @ candidate) < float(best @ best) - 1e-12:
                best = candidate
    return best


def _layer_point_lattice(points: np.ndarray, cell: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Return a primitive basis of the 2D point lattice of one layer.

    The layer must be a single orbit of a 2D lattice that has the slab cell as a
    sublattice, which is what a close-packed monatomic layer is.  Anything else
    (a honeycomb layer, a mixed-species layer, a buckled layer) returns ``None``
    and switches the caller to the honest "not close packed" answer.
    """

    points = np.asarray(points, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if points.shape[0] == 0:
        return None
    images = np.array(
        [i * cell[0] + j * cell[1] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    )
    differences = [points[index] - points[0] for index in range(points.shape[0])]
    candidates = [difference + image for difference in differences for image in images]
    candidates = [vector for vector in candidates if float(np.linalg.norm(vector)) > tolerance]
    if not candidates:
        # One atom per slab cell: the layer lattice is the slab cell itself.
        return _gauss_reduce(cell)
    candidates.sort(key=lambda vector: float(vector @ vector))
    first = candidates[0]
    second = None
    for candidate in candidates[1:]:
        cross = float(first[0] * candidate[1] - first[1] * candidate[0])
        if abs(cross) > tolerance * float(np.linalg.norm(first)):
            second = candidate
            break
    if second is None:
        return None
    basis = _gauss_reduce(np.array([first, second], dtype=float))
    for vector in (cell[0], cell[1]):
        if _integer_coefficients(vector, basis, tolerance) is None:
            return None
    for difference in differences:
        if _integer_coefficients(difference, basis, tolerance) is None:
            return None
    cell_area = abs(float(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]))
    basis_area = abs(float(basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0]))
    if basis_area <= 1e-12:
        return None
    sites = int(round(cell_area / basis_area))
    if abs(cell_area / basis_area - sites) > 1e-6 or sites != points.shape[0]:
        # Every lattice point of the candidate basis has to carry an atom;
        # otherwise the layer is a decorated lattice such as a honeycomb, and
        # its registry is not described by three cosets.
        return None
    return basis


def _hollow_generator(basis: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Return the canonical hollow vector of a triangular layer lattice.

    ``(1/3)L / L`` has four subgroups of order three, and exactly one of them —
    the one generated by a triangle centroid — is invariant under the point
    group of a triangular lattice.  Its two non-zero elements are ``h`` and
    ``-h``; the one pointing into the upper half plane is returned, which fixes
    the ``A -> B -> C`` direction in the Cartesian frame shared by both slabs.
    """

    reduced = _gauss_reduce(np.asarray(basis, dtype=float))
    first, second = reduced[0], reduced[1]
    length_first = float(np.linalg.norm(first))
    length_second = float(np.linalg.norm(second))
    if length_first <= 1e-12 or abs(length_first - length_second) > tolerance:
        return None
    if float(first @ second) < 0.0:
        second = -second
    cosine = float(first @ second) / (length_first * length_second)
    if abs(cosine - 0.5) > 1e-3:
        return None
    hollow = (first + second) / 3.0
    if hollow[1] < -1e-9 or (abs(hollow[1]) <= 1e-9 and hollow[0] < 0.0):
        hollow = -hollow
    return hollow


def _coset_index(
    offset: np.ndarray, basis: np.ndarray, hollow: np.ndarray, tolerance: float
) -> int | None:
    """Return ``n`` with ``offset = n * hollow`` modulo the layer lattice."""

    for index in range(3):
        if _integer_coefficients(np.asarray(offset, dtype=float) - index * hollow, basis, tolerance) is not None:
            return index
    return None


# ---------------------------------------------------------------------------
# stacking analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackingLayer:
    """One layer of a slab, ordered from the bottom of the cell upwards."""

    index: int
    height: float
    atom_indices: tuple[int, ...]
    coset: int
    label: str


@dataclass(frozen=True)
class StackingAnalysis:
    """The close-packed stacking sequence of one slab."""

    close_packed: bool
    reason: str
    layers: tuple[StackingLayer, ...]
    sequence: str
    increments: tuple[int, ...]
    sense: int
    hollow_cartesian: tuple[float, float] | None
    hollow_direct: tuple[float, float] | None
    layer_basis: tuple[tuple[float, float], tuple[float, float]] | None

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def bottom_coset(self) -> int | None:
        return self.layers[0].coset if self.layers else None

    @property
    def top_coset(self) -> int | None:
        return self.layers[-1].coset if self.layers else None

    @property
    def bottom_label(self) -> str:
        return self.layers[0].label if self.layers else "?"

    @property
    def top_label(self) -> str:
        return self.layers[-1].label if self.layers else "?"

    @property
    def sense_label(self) -> str:
        return sense_label(self.sense)

    @property
    def reversible(self) -> bool:
        """True when reversing the stacking changes the layer sequence."""

        return any(increment % 3 != 0 for increment in self.increments)

    def as_dict(self) -> dict[str, object]:
        return {
            "close_packed": bool(self.close_packed),
            "reason": str(self.reason),
            "sequence": str(self.sequence),
            "layer_count": int(self.layer_count),
            "increments": [int(value) for value in self.increments],
            "sense": int(self.sense),
            "sense_label": self.sense_label,
            "bottom_label": self.bottom_label,
            "top_label": self.top_label,
            "layer_heights": [float(layer.height) for layer in self.layers],
            "atoms_per_layer": [len(layer.atom_indices) for layer in self.layers],
        }


def sense_label(sense: int) -> str:
    """Name a stacking sense: ``ABC`` forward, ``CBA`` reversed, else mixed."""

    if int(sense) > 0:
        return "ABC"
    if int(sense) < 0:
        return "CBA"
    return "mixed"


def group_layers(record, layer_tolerance: float = LAYER_TOLERANCE) -> list[tuple[float, list[int]]]:
    """Return ``(mean height, atom indices)`` for each atomic plane, bottom first.

    The grouping is the package-wide one of ``core.layers.layer_partition``, so
    the planes a stacking word is read from are the same planes the termination
    report and the defect layer census see, and turning the slab over reverses
    the word instead of re-cutting the layers.
    """

    cartesian = np.asarray(record.positions_cartesian, dtype=float)
    if cartesian.shape[0] == 0:
        return []
    heights = normal_heights(record.lattice, cartesian)
    return layer_partition(heights, float(layer_tolerance))


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed pair of in-plane axes fixed by the normal alone.

    The frame must not depend on the cell, because the two slabs of an
    interface are compared in it and their cells only agree up to a lattice
    basis change.
    """

    normal = np.asarray(normal, dtype=float)
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_x = reference - float(reference @ normal) * normal
    axis_x = axis_x / float(np.linalg.norm(axis_x))
    return axis_x, np.cross(normal, axis_x)


def _inplane(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    planar = vectors - np.outer(vectors @ normal, normal)
    axis_x, axis_y = _plane_axes(normal)
    return np.column_stack((planar @ axis_x, planar @ axis_y))


def _not_close_packed(reason: str, layers: Sequence[tuple[float, list[int]]]) -> StackingAnalysis:
    return StackingAnalysis(
        close_packed=False,
        reason=str(reason),
        layers=tuple(
            StackingLayer(
                index=index,
                height=float(height),
                atom_indices=tuple(int(atom) for atom in sorted(indices)),
                coset=-1,
                label="?",
            )
            for index, (height, indices) in enumerate(layers)
        ),
        sequence="",
        increments=tuple(),
        sense=0,
        hollow_cartesian=None,
        hollow_direct=None,
        layer_basis=None,
    )


def analyse_stacking(
    record,
    *,
    layer_tolerance: float = LAYER_TOLERANCE,
    tolerance: float = POSITION_TOLERANCE,
    hollow_cartesian: Sequence[float] | None = None,
) -> StackingAnalysis:
    """Read the ``A``/``B``/``C`` layer sequence of a close-packed slab.

    The sequence runs from the bottom of the cell upwards and starts at ``A``:
    the absolute letters are an origin gauge, only their differences are
    physical.  The direction ``A -> B -> C`` is a second gauge, fixed by the
    hollow vector ``h``.  Left to itself a slab uses its own first interlayer
    offset, so a uniformly stacked slab always reads ``ABCABC``; pass the
    ``hollow_cartesian`` of the other slab to describe both slabs of an
    interface in one common gauge, which is what makes their relative sense
    and their contact meaningful.
    """

    groups = group_layers(record, layer_tolerance)
    if not groups:
        return _not_close_packed("the structure has no atoms", groups)
    lattice = np.asarray(record.lattice, dtype=float)
    normal = surface_normal(lattice)
    cartesian = np.asarray(record.positions_cartesian, dtype=float)
    cell = _inplane(lattice[:2], normal)
    planar = _inplane(cartesian, normal)

    counts = {len(indices) for _, indices in groups}
    if len(counts) != 1:
        return _not_close_packed("the layers do not all hold the same number of atoms", groups)

    basis = _layer_point_lattice(planar[np.asarray(groups[0][1], dtype=int)], cell, tolerance)
    if basis is None:
        return _not_close_packed("the bottom layer is not a single lattice of equivalent sites", groups)
    hollow = _hollow_generator(basis, tolerance)
    if hollow is None:
        return _not_close_packed("the layer lattice is not triangular", groups)

    anchors = []
    for _, indices in groups:
        points = planar[np.asarray(indices, dtype=int)]
        layer_basis = _layer_point_lattice(points, cell, tolerance)
        if layer_basis is None or _integer_coefficients(layer_basis[0], basis, tolerance) is None:
            return _not_close_packed("the layers do not share one lattice of equivalent sites", groups)
        anchors.append(points[0])

    if hollow_cartesian is not None:
        supplied = np.asarray(hollow_cartesian, dtype=float)
        if _coset_index(supplied, basis, hollow, tolerance) not in (1, 2):
            return _not_close_packed(
                "the hollow vector of the other slab is not a hollow vector of this one, so the "
                "two slabs do not share an in-plane lattice",
                groups,
            )
        hollow = _reduce_modulo_lattice(supplied, basis)
    elif len(anchors) >= 2:
        offset = _reduce_modulo_lattice(anchors[1] - anchors[0], basis)
        step = _coset_index(offset, basis, hollow, tolerance)
        if step is None:
            return _not_close_packed(
                "consecutive layers are not offset by a close-packed hollow vector", groups
            )
        if step != 0:
            # Call the first interlayer offset "A -> B", so a uniformly stacked
            # slab reads ABCABC in its own gauge.
            hollow = offset

    cosets = [0]
    increments: list[int] = []
    for index in range(1, len(anchors)):
        step = _coset_index(anchors[index] - anchors[index - 1], basis, hollow, tolerance)
        if step is None:
            return _not_close_packed(
                "consecutive layers are not offset by a close-packed hollow vector", groups
            )
        increments.append(int(step))
        cosets.append((cosets[-1] + int(step)) % 3)

    if all(step == 1 for step in increments) and increments:
        sense = 1
    elif all(step == 2 for step in increments) and increments:
        sense = -1
    else:
        sense = 0

    axis_x, axis_y = _plane_axes(normal)
    hollow_3d = hollow[0] * axis_x + hollow[1] * axis_y
    hollow_direct = io_mod.cartesian_to_direct(hollow_3d.reshape(1, 3), lattice)[0]

    layers = tuple(
        StackingLayer(
            index=index,
            height=float(height),
            atom_indices=tuple(int(atom) for atom in sorted(indices)),
            coset=int(cosets[index]),
            label=COSET_LETTERS[int(cosets[index])],
        )
        for index, (height, indices) in enumerate(groups)
    )
    return StackingAnalysis(
        close_packed=True,
        reason="",
        layers=layers,
        sequence="".join(layer.label for layer in layers),
        increments=tuple(increments),
        sense=int(sense),
        hollow_cartesian=(float(hollow[0]), float(hollow[1])),
        hollow_direct=(float(hollow_direct[0]), float(hollow_direct[1])),
        layer_basis=(
            (float(basis[0][0]), float(basis[0][1])),
            (float(basis[1][0]), float(basis[1][1])),
        ),
    )


# ---------------------------------------------------------------------------
# structure transforms
# ---------------------------------------------------------------------------

def _inplane_gram(lattice: np.ndarray) -> np.ndarray:
    planar = np.asarray(lattice, dtype=float)[:2]
    return planar @ planar.T


def inplane_mirror_matrices(lattice: np.ndarray, tolerance: float = 1e-6) -> list[np.ndarray]:
    """Return the integer matrices of the in-plane mirrors of the cell.

    ``M`` is returned when ``M G M^T = G`` for the in-plane Gram matrix ``G``
    and ``det M = -1``: those are exactly the improper operations of the lattice
    point group, the ones that reverse a stacking sense.
    """

    gram = _inplane_gram(lattice)
    scale = float(np.max(np.abs(gram)))
    matrices = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    matrix = np.array([[a, b], [c, d]], dtype=int)
                    if int(round(np.linalg.det(matrix))) != -1:
                        continue
                    if float(np.max(np.abs(matrix @ gram @ matrix.T - gram))) <= tolerance * max(scale, 1.0):
                        matrices.append(matrix)
    matrices.sort(key=lambda matrix: tuple(int(value) for value in matrix.reshape(-1)))
    return matrices


def mirror_structure(record, *, matrix: np.ndarray | None = None):
    """Return the mirror image of a slab, written in the same cell.

    The reflection is an in-plane mirror of the lattice, so the cell is
    unchanged and only the fractional coordinates move.  Reflecting reverses the
    stacking sense: ``ABCABC`` becomes ``CBACBA``.
    """

    lattice = np.asarray(record.lattice, dtype=float)
    if abs(float(lattice[2, 0])) > 1e-6 or abs(float(lattice[2, 1])) > 1e-6:
        raise ValueError(
            "the stacking mirror needs a slab whose c axis is perpendicular to the surface; "
            "normalise the slab first"
        )
    if matrix is None:
        candidates = inplane_mirror_matrices(lattice)
        if not candidates:
            raise ValueError(
                "the in-plane cell of this slab has no mirror line, so its stacking sequence "
                "cannot be reversed inside the same cell"
            )
        matrix = candidates[0]
    matrix = np.asarray(matrix, dtype=int)
    direct = np.array(record.positions_direct, dtype=float, copy=True)
    direct[:, :2] = direct[:, :2] @ matrix
    output = record.copy()
    output.positions_direct = direct
    output.positions_cartesian = io_mod.direct_to_cartesian(direct, lattice)
    output.coordinate_mode = "Direct"
    return output


def shift_structure_inplane(record, shift_direct: Sequence[float]):
    """Return a copy of ``record`` translated in plane by ``shift_direct``."""

    shift = np.zeros(3, dtype=float)
    values = np.asarray(shift_direct, dtype=float).reshape(-1)
    shift[: values.size] = values
    shift[2] = 0.0
    direct = np.array(record.positions_direct, dtype=float, copy=True) + shift
    output = record.copy()
    output.positions_direct = direct
    output.positions_cartesian = io_mod.direct_to_cartesian(direct, np.asarray(record.lattice, dtype=float))
    output.coordinate_mode = "Direct"
    return output


def normalise_stacking_choice(choice: str | None) -> str:
    """Map the user spellings of a stacking request onto ``keep``/``abc``/``cba``."""

    text = str(choice or "keep").strip().lower().replace("-", "").replace("_", "")
    if text in {"", "keep", "asis", "unchanged", "none", "input"}:
        return "keep"
    if text in {"mirror", "mirrored", "flip", "flipped"}:
        return "mirror"
    if text in {"abc", "abcabc", "forward", "same", "parallel", "plus"}:
        return "abc"
    if text in {"cba", "cbacba", "reverse", "reversed", "opposite", "backward", "minus"}:
        return "cba"
    raise ValueError(f"unknown stacking choice {choice!r}; use keep, mirror, abc, or cba")


def apply_relative_stacking(
    record,
    choice: str,
    *,
    reference_hollow: Sequence[float] | None = None,
    layer_tolerance: float = LAYER_TOLERANCE,
):
    """Return ``(structure, analysis, mirrored)`` for one slab of an interface.

    ``keep`` leaves the slab alone and ``mirror`` always reflects it.  ``abc``
    and ``cba`` need a ``reference_hollow``, the gauge of the other slab: they
    ask for a slab that stacks the same way as the reference (``ABCABC`` under
    the reference's own ``ABCABC``) or the opposite way (``CBACBA``).  A slab
    has no absolute handedness, which is why the request is relative.
    """

    resolved = normalise_stacking_choice(choice)
    analysis = analyse_stacking(
        record, layer_tolerance=layer_tolerance, hollow_cartesian=reference_hollow
    )
    if resolved == "keep":
        return record, analysis, False
    if not analysis.close_packed:
        raise ValueError(
            f"cannot set the stacking sense of a slab that is not close packed: {analysis.reason}"
        )
    if resolved in {"abc", "cba"}:
        if reference_hollow is None:
            raise ValueError(
                "abc and cba describe a slab relative to the other slab of the interface; "
                "a reference hollow vector is required"
            )
        if analysis.sense == 0:
            raise ValueError(
                f"the slab stacks as {analysis.sequence!r}, which is neither ABC nor CBA, so its "
                "stacking sense cannot be set; use keep"
            )
        if analysis.sense == (1 if resolved == "abc" else -1):
            return record, analysis, False
    mirrored = mirror_structure(record)
    mirrored_analysis = analyse_stacking(
        mirrored, layer_tolerance=layer_tolerance, hollow_cartesian=reference_hollow
    )
    if resolved in {"abc", "cba"} and mirrored_analysis.sense != (1 if resolved == "abc" else -1):
        raise ValueError("mirroring the slab did not reverse its stacking sense as expected")
    return mirrored, mirrored_analysis, True


def registry_shift_direct(
    bottom,
    top,
    bottom_analysis: StackingAnalysis,
    top_analysis: StackingAnalysis,
    delta: int,
    *,
    hollow: Sequence[float] | None = None,
    tolerance: float = POSITION_TOLERANCE,
) -> np.ndarray:
    """Return the in-plane direct shift that puts the contact at ``delta``.

    ``bottom`` and ``top`` must already share an in-plane cell.  The shift is
    the difference between the requested contact vector ``delta * h`` and the
    offset the two interfacial layers happen to have, so the result is exact
    whatever the incoming registry was.
    """

    if not bottom_analysis.close_packed or not top_analysis.close_packed:
        raise ValueError("a registry can only be set between two close-packed slabs")
    lattice = np.asarray(bottom.lattice, dtype=float)
    normal = surface_normal(lattice)
    bottom_planar = _inplane(np.asarray(bottom.positions_cartesian, dtype=float), normal)
    top_planar = _inplane(np.asarray(top.positions_cartesian, dtype=float), normal)
    bottom_anchor = bottom_planar[bottom_analysis.layers[-1].atom_indices[0]]
    top_anchor = top_planar[top_analysis.layers[0].atom_indices[0]]
    if hollow is None:
        hollow = bottom_analysis.hollow_cartesian
    hollow = np.asarray(hollow, dtype=float)
    basis = np.asarray(bottom_analysis.layer_basis, dtype=float)
    offset = top_anchor - bottom_anchor
    if _integer_coefficients(offset - (int(delta) % 3) * hollow, basis, tolerance) is not None:
        target = np.zeros(2, dtype=float)
    else:
        target = _reduce_modulo_lattice((int(delta) % 3) * hollow - offset, basis)
    axis_x, axis_y = _plane_axes(normal)
    shift_cartesian = target[0] * axis_x + target[1] * axis_y
    return io_mod.cartesian_to_direct(shift_cartesian.reshape(1, 3), lattice)[0]

"""Native three-dimensional crystallographic symmetry utilities.

Everything in this module works with *row* lattices: ``lattice[i]`` is the
Cartesian vector of the ``i``-th basis vector, so a site with fractional
coordinates ``x`` (a row) sits at ``x @ lattice``.  This is the convention used
by :class:`cellstine.io.models.StructureRecord` and by the VASP POSCAR format.

Symmetry operations follow the crystallographic convention

.. code-block:: text

    x' = W x + w

with ``x`` a *column* of fractional coordinates, ``W`` an integer matrix and
``w`` a fractional translation.  Acting on the row vectors used elsewhere this
reads ``x' = x @ W.T + w``.

The point group of a bare lattice -- the integer matrices ``W`` with
``W.T @ G @ W == G`` for the metric ``G = lattice @ lattice.T`` -- is searched
for in :mod:`cellstine.core.pointgroup3d`, whose four public names
(:func:`lattice_point_group`, :func:`rotation_type`,
:func:`point_group_symbol`, :func:`crystal_system_of_point_group`) are
re-exported here.  What follows is the decorated cell: the space-group
operations of a structure, the orbits they give, and primitive-cell reduction.

Together the two modules cover the parts of a symmetry package that a
plane-wave structure builder actually needs: cell reduction, the point group of
a lattice, the space-group operations of a decorated cell, the resulting orbits
of symmetry-equivalent atoms, and primitive-cell reduction.  Naming a space-group
*type* (the number in the International Tables) needs the tabulated
standard settings of all 230 groups and is deliberately left to ``spglib``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .geometry import (
    PeriodicSiteIndex,
    as_lattice as _as_lattice,
    delaunay_reduce,
    niggli_reduce,
    plane_reduce,
    rational_lattice_basis,
    wrap_fractional as _wrap_fractional,
)
from .pointgroup3d import (
    crystal_system_of_point_group,
    lattice_point_group,
    point_group_symbol,
    rotation_type,
)

__all__ = [
    "SymmetryDataset",
    "niggli_reduce",
    "delaunay_reduce",
    "lattice_point_group",
    "point_group_symbol",
    "crystal_system_of_point_group",
    "rotation_type",
    "symmetry_operations",
    "analyse_symmetry",
    "site_permutations",
    "pure_translations",
    "translation_group",
    "translation_lattice_basis",
    "generating_operations",
    "fold_to_basis",
    "primitive_cell",
    "planar_translation_basis",
    "planar_primitive_layer",
]


# The lattice helpers and the two cell reductions live in ``core.geometry``,
# which is the shared home of the periodic-geometry primitives, and the point
# group of a bare lattice lives in ``core.pointgroup3d``; both are imported
# above and re-exported here, so ``symmetry3d.niggli_reduce``,
# ``symmetry3d.delaunay_reduce`` and ``symmetry3d.lattice_point_group`` keep
# working.


# ---------------------------------------------------------------------------
# space-group operations of a decorated cell
# ---------------------------------------------------------------------------


def _species_labels(species: Sequence[str] | None, count: int) -> np.ndarray:
    if species is None:
        return np.zeros(count, dtype=np.int64)
    symbols = [str(value) for value in species]
    if len(symbols) != count:
        raise ValueError("one species label per atom is required")
    order = {symbol: index for index, symbol in enumerate(sorted(set(symbols)))}
    return np.array([order[symbol] for symbol in symbols], dtype=np.int64)


@dataclass
class SymmetryDataset:
    """Symmetry information for one decorated cell."""

    rotations: np.ndarray
    translations: np.ndarray
    equivalent_atoms: np.ndarray
    point_group: str | None
    lattice_point_group: str | None
    crystal_system: str | None
    has_inversion: bool
    symmorphic_setting: bool
    primitive_translations: np.ndarray

    @property
    def operation_count(self) -> int:
        return int(len(self.rotations))


def _site_permutation(
    lattice: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray,
    symprec: float,
    index: PeriodicSiteIndex | None = None,
) -> np.ndarray | None:
    """Return the permutation sending each site to its image, or ``None``.

    ``images`` holds the fractional coordinates every site is mapped to.  Each
    image is looked up in a bucket index of the sites, so the test costs ``O(n)``
    rather than the ``O(n^2)`` of a full distance matrix; passing a prepared
    ``index`` reuses one table across all the candidate operations.
    """

    count = len(positions)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    finder = index if index is not None else PeriodicSiteIndex(lattice, positions, labels, float(symprec))
    targets = finder.match(images, labels)
    if np.any(targets < 0):
        return None
    if len(np.unique(targets)) != count:
        return None
    return targets


class _OperationScan:
    """Shared state of a search for the operations of one decorated cell.

    The bucket index of the sites, the reference species and the screen sites
    are the same for every candidate rotation, so they are built once here and
    reused.  Both the full operation search and the cheaper search for the pure
    translations alone (``translation_group``) go through this object.
    """

    def __init__(self, lattice: np.ndarray, positions: np.ndarray, labels: np.ndarray, symprec: float) -> None:
        self.lattice = lattice
        self.positions = positions
        self.labels = labels
        self.symprec = float(symprec)
        self.index = PeriodicSiteIndex(lattice, positions, labels, float(symprec))
        unique_labels, counts = np.unique(labels, return_counts=True)
        reference_label = int(unique_labels[int(np.argmin(counts))])
        self.reference_atoms = np.nonzero(labels == reference_label)[0]
        self.reference = int(self.reference_atoms[0])
        count = len(positions)
        self.screen = np.unique(np.linspace(0, count - 1, num=min(count, 4)).astype(np.int64))

    def screened_translations(self, mapped: np.ndarray) -> np.ndarray:
        """Return the candidate translations that survive the cheap screen.

        The reference atom has to land on an atom of its own species, which
        leaves one candidate translation per such atom.  A handful of screen
        sites rejects nearly all of them, and screening every candidate in one
        array lookup costs a single pass over the bucket table instead of one
        pass per candidate -- the difference between a linear and a quadratic
        search in a large supercell.
        """

        offsets = _wrap_fractional(self.positions[self.reference_atoms] - mapped[self.reference])
        probes = (mapped[self.screen][None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        probe_labels = np.tile(self.labels[self.screen], len(self.reference_atoms))
        found = self.index.match(probes, probe_labels).reshape(len(self.reference_atoms), len(self.screen))
        return offsets[np.all(found >= 0, axis=1)]

    def accepted_translations(self, element: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return every valid translation for one rotation, with its permutation."""

        mapped = self.positions @ np.asarray(element, dtype=float).T
        found: List[Tuple[np.ndarray, np.ndarray]] = []
        for translation in self.screened_translations(mapped):
            permutation = _site_permutation(
                self.lattice, self.positions, self.labels, mapped + translation, self.symprec, self.index
            )
            if permutation is None:
                continue
            found.append((np.mod(translation, 1.0), permutation))
        return found


def _operation_search(
    lattice: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    symprec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the rotations, translations and generating site permutations.

    The permutation array holds one entry per *directly verified* operation.
    Those operations generate the whole group, which is all the orbit search
    needs, so the array is generally shorter than the operation list.
    """

    group = lattice_point_group(lattice, tolerance=max(1e-8, float(symprec) / max(float(np.max(np.abs(lattice))), 1.0)))
    count = len(positions)
    if count == 0:
        return group, np.zeros((len(group), 3), dtype=float), np.zeros((len(group), 0), dtype=np.int64)

    scan = _OperationScan(lattice, positions, labels, float(symprec))
    index = scan.index
    _screened_translations = scan.screened_translations
    _accepted_translations = scan.accepted_translations

    identity = np.eye(3, dtype=np.int64)
    centering = _accepted_translations(identity)

    # The non-trivial centering translations, formed once: every accepted
    # rotation is repeated with each of them below.
    coset_shifts = [
        shift
        for shift, _ in centering
        if not np.all(np.abs(_wrap_fractional(shift)) <= 1e-12)
    ]

    rotations: List[np.ndarray] = []
    translations: List[np.ndarray] = []
    permutations: List[np.ndarray] = []
    for translation, permutation in centering:
        rotations.append(identity)
        translations.append(translation)
        permutations.append(permutation)

    # Two operations with the same rotation differ by a pure translation, so one
    # accepted translation per rotation is enough: the rest are its coset.  Only
    # the directly verified operations carry a permutation, which is all the
    # orbit search needs because they generate the whole group.
    for element in group:
        if np.array_equal(element, identity):
            continue
        mapped = positions @ element.astype(float).T
        for translation in _screened_translations(mapped):
            permutation = _site_permutation(
                lattice, positions, labels, mapped + translation, float(symprec), index
            )
            if permutation is None:
                continue
            rotations.append(np.asarray(element, dtype=np.int64))
            translations.append(np.mod(translation, 1.0))
            permutations.append(permutation)
            for shift in coset_shifts:
                rotations.append(np.asarray(element, dtype=np.int64))
                translations.append(np.mod(translation + shift, 1.0))
            break

    if not rotations:  # pragma: no cover - the identity always maps a cell onto itself
        return (
            np.eye(3, dtype=np.int64)[None, :, :],
            np.zeros((1, 3), dtype=float),
            np.arange(count, dtype=np.int64)[None, :],
        )
    return (
        np.asarray(rotations, dtype=np.int64),
        np.asarray(translations, dtype=float),
        np.asarray(permutations, dtype=np.int64),
    )


def _orbits_from_permutations(permutations: np.ndarray, count: int) -> np.ndarray:
    """Return the orbit representative of every site under a set of permutations.

    The orbits of the group generated by the permutations are the connected
    components of the graph whose edges are ``i -- permutation[i]``, and the
    label reported here is the least index of the component; both statements
    are proved in ``aristotle-lean-reference/RequestProject/SiteOrbits.lean``
    (``Cellstine.siteLinked_iff_mem_orbit``, ``Cellstine.orbitRep_eq_iff_siteLinked``).

    Each site starts as its own representative and every permutation pulls the
    label of a site down to the smallest label in its orbit; iterating the whole
    set until nothing changes reaches the fixed point, which is the orbit
    minimum.  The sweep is a handful of array operations per permutation instead
    of a Python loop over the sites.
    """

    if count == 0:
        return np.zeros(0, dtype=np.int64)
    table = np.asarray(permutations, dtype=np.int64).reshape(-1, count)
    representative = np.arange(count, dtype=np.int64)
    for _ in range(count + 1):
        previous = representative.copy()
        for permutation in table:
            # A permutation relates site i to site permutation[i], so the label
            # travels in both directions along it.
            representative = np.minimum(representative, representative[permutation])
            np.minimum.at(representative, permutation, representative.copy())
            representative = representative[representative]
        if np.array_equal(representative, previous):
            break
    return representative


def symmetry_operations(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the space-group operations of a decorated cell.

    ``symprec`` is a Cartesian distance tolerance in the same length unit as the
    lattice.  The returned pair holds the integer rotations ``(k, 3, 3)`` acting
    on column fractional coordinates and the fractional translations ``(k, 3)``
    reduced into ``[0, 1)``.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    labels = _species_labels(species, len(positions))
    rotations, translations, _ = _operation_search(array, positions, labels, float(symprec))
    return rotations, translations


def equivalent_atom_map(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Return, for every atom, the lowest index of the orbit it belongs to."""

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    labels = _species_labels(species, len(positions))
    permutations = []
    for rotation, translation in zip(rotations, translations):
        images = positions @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
        permutation = _site_permutation(array, positions, labels, images, float(symprec))
        if permutation is not None:
            permutations.append(permutation)
    if not permutations:
        return np.arange(len(positions), dtype=np.int64)
    return _orbits_from_permutations(np.asarray(permutations, dtype=np.int64), len(positions))


def site_permutations(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Return the site permutation induced by each symmetry operation.

    Row ``k`` holds, for every site, the index of the site that operation ``k``
    maps it onto.  Operations that do not map the cell onto itself -- which can
    only happen if they were not produced for this structure -- are reported as
    the identity.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    labels = _species_labels(species, len(positions))
    identity = np.arange(len(positions), dtype=np.int64)
    result = []
    for rotation, translation in zip(rotations, translations):
        images = positions @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
        permutation = _site_permutation(array, positions, labels, images, float(symprec))
        result.append(identity if permutation is None else permutation)
    if not result:
        return identity[None, :]
    return np.asarray(result, dtype=np.int64)


_SYMMORPHIC_TOLERANCE = 1e-6


def _translations_are_centering(
    translations: np.ndarray, centering: np.ndarray, tolerance: float = _SYMMORPHIC_TOLERANCE
) -> bool:
    """Whether every operation carries one of the centering translations.

    That is the test for a symmorphic setting.  The comparison is the same
    component-wise one a direct scan performs, but the candidate is found by a
    bucketed periodic lookup, so the cost is linear in the number of operations
    rather than their product with the number of centering vectors -- the outer
    product alone is tens of megabytes for a supercell, whose group has one
    operation per point-group element *and* lattice point.
    """

    vectors = np.mod(np.asarray(translations, dtype=float).reshape(-1, 3), 1.0)
    sites = np.mod(np.asarray(centering, dtype=float).reshape(-1, 3), 1.0)
    if vectors.shape[0] == 0:
        return True
    if sites.shape[0] == 0:
        return False
    # A ball of this radius contains the cube the component-wise test accepts,
    # so no candidate is missed; the component-wise test is applied afterwards.
    index = PeriodicSiteIndex(np.eye(3), sites, tolerance=math.sqrt(3.0) * tolerance)
    matched = index.match(vectors)
    if np.any(matched < 0):
        return False
    residues = _wrap_fractional(vectors - sites[matched])
    return bool(np.all(np.abs(residues) <= tolerance))


def pure_translations(rotations: np.ndarray, translations: np.ndarray, *, symprec: float = 1e-5) -> np.ndarray:
    """Return the fractional translations that are symmetry operations on their own."""

    matrices = np.asarray(rotations, dtype=np.int64).reshape(-1, 3, 3)
    vectors = np.asarray(translations, dtype=float).reshape(-1, 3)
    identity = np.all(matrices == np.eye(3, dtype=np.int64)[None, :, :], axis=(1, 2))
    if not np.any(identity):
        return np.zeros((1, 3), dtype=float)
    return np.mod(vectors[identity], 1.0)


def generating_operations(
    rotations: np.ndarray, translations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a small subset of the operations that generates the same group.

    An orbit of a group action is a connected component of the graph drawn by
    *any* generating set, so grouping sites into orbits never needs the whole
    group.  For a supercell the saving is the whole point: an ``n1 x n2 x n3``
    supercell of a crystal whose point group has order ``p`` carries
    ``p n1 n2 n3`` operations -- 49152 of them for a 8x8x8 cubic cell -- and all
    but a few dozen are redundant.

    Two facts give the reduction.  If ``(R, t1)`` and ``(R, t2)`` are both
    operations then ``(R, t1) (R, t2)^-1 = (I, t1 - t2)`` is a pure translation,
    so one operation per *distinct rotation*, together with the pure
    translations, generates everything.  And the pure translations form a
    lattice that contains the unit translations, so three basis vectors of that
    lattice generate all of them; the basis is computed exactly, over the
    integers, from the common denominator that the order of the translation
    group supplies.

    The full list is returned unchanged if the translations do not sit on that
    rational grid, which keeps the reduction from ever being the reason an orbit
    comes out wrong.
    """

    matrices = np.asarray(rotations, dtype=np.int64).reshape(-1, 3, 3)
    vectors = np.mod(np.asarray(translations, dtype=float).reshape(-1, 3), 1.0)
    if len(matrices) == 0:
        return matrices, vectors
    identity = np.eye(3, dtype=np.int64)
    is_identity = np.all(matrices == identity[None, :, :], axis=(1, 2))
    centering = vectors[is_identity]
    if len(centering) == 0:
        centering = np.zeros((1, 3), dtype=float)
    try:
        basis = rational_lattice_basis(
            np.vstack([np.eye(3, dtype=float), centering]), len(centering)
        )
    except ValueError:
        return matrices, vectors

    kept_rotations: List[np.ndarray] = []
    kept_translations: List[np.ndarray] = []
    for row in basis:
        shift = np.mod(row, 1.0)
        if np.allclose(shift, 0.0):
            # A basis vector of the unit lattice generates nothing new.
            continue
        kept_rotations.append(identity)
        kept_translations.append(shift)
    seen: set[bytes] = set()
    for matrix, vector in zip(matrices, vectors):
        if bool(np.all(matrix == identity)):
            continue
        key = matrix.tobytes()
        if key in seen:
            continue
        seen.add(key)
        kept_rotations.append(matrix)
        kept_translations.append(vector)
    return (
        np.asarray(kept_rotations, dtype=np.int64).reshape(-1, 3, 3),
        np.asarray(kept_translations, dtype=float).reshape(-1, 3),
    )


def translation_group(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Return the pure translations of a decorated cell, wrapped into ``[0, 1)``.

    A pure translation is a shift of the fractional coordinates that maps the
    decorated cell onto itself; the zero shift is always one, so the returned
    array is never empty.  Only the identity rotation is tried, which is why
    this is far cheaper than :func:`analyse_symmetry`: no point group is
    searched for.

    The translations form a finite group above the unit translations
    (``Cellstine.Centering.translationSubgroup`` in
    ``aristotle-lean-reference/RequestProject/CenteringLattice.lean``), its order divides the number of
    atoms of every species (``Cellstine.Centering.card_translations_dvd``), and
    therefore ``m t`` is an integer vector for every translation ``t``, where
    ``m`` is the number of translations returned
    (``Cellstine.Centering.card_nsmul_mem_base``).  That is exactly the common
    denominator :func:`translation_lattice_basis` hands to
    ``rational_lattice_basis``.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    if len(positions) == 0:
        return np.zeros((1, 3), dtype=float)
    labels = _species_labels(species, len(positions))
    scan = _OperationScan(array, positions, labels, float(symprec))
    found = scan.accepted_translations(np.eye(3, dtype=np.int64))
    if not found:  # pragma: no cover - the zero shift always maps a cell onto itself
        return np.zeros((1, 3), dtype=float)
    return np.asarray([translation for translation, _ in found], dtype=float).reshape(-1, 3)


def translation_lattice_basis(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    symprec: float = 1e-5,
    reduce: bool = True,
) -> Tuple[np.ndarray, int]:
    """Return a basis of the translation lattice of a cell, and its index.

    The rows are in fractional coordinates of the input cell, so ``basis @
    lattice`` is the Cartesian primitive cell of the structure and
    ``round(1 / det(basis))`` is the number ``m`` of pure translations --- the
    index of the input cell in the lattice of its own translations.  ``m == 1``
    returns the identity: the input cell is already primitive.

    With ``reduce`` the basis is Niggli reduced (in the Cartesian metric) before
    it is returned, so a face-centred cubic cell yields the three equal
    60-degree vectors rather than some sheared cell of the same volume.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    centering = translation_group(array, positions, species, symprec=symprec)
    order = len(centering)
    if order <= 1:
        return np.eye(3, dtype=float), 1

    # The pure translations, together with the three unit translations, generate
    # the translation lattice of the structure.  A basis of it comes from the
    # Hermite normal form of those generators -- an exact integer computation
    # that always succeeds, unlike a search for three generators that happen to
    # form a basis themselves.
    generators = np.vstack([np.eye(3, dtype=float), np.mod(centering, 1.0)])
    basis = rational_lattice_basis(generators, order)
    if reduce:
        _, reduction = niggli_reduce(basis @ array)
        basis = reduction.astype(float) @ basis
    return basis, int(order)


def fold_to_basis(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str],
    basis: np.ndarray,
    *,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Rewrite a structure in a smaller cell of its own translation lattice.

    ``basis`` holds, in fractional coordinates of the input cell, a basis of a
    lattice that contains the unit translations; the sites of the input cell
    that differ by one of the extra translations then fall on one site of the
    new cell and only one copy of each is kept.  No atom is lost: the folded
    cell has the same atomic density as the input one.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    symbols = [str(value) for value in species]
    if len(symbols) != len(positions):
        raise ValueError("one species label per atom is required")
    matrix = np.asarray(basis, dtype=float).reshape(3, 3)

    folded_lattice = matrix @ array
    inverse = np.linalg.inv(matrix)
    new_positions = np.mod(positions @ inverse, 1.0)

    # Sites of the input cell that differ by a translation collapse onto one
    # site of the folded cell; a bucket index finds the duplicates in one pass
    # instead of comparing every kept site with every new one.
    kept_positions: List[np.ndarray] = []
    kept_species: List[str] = []
    new_labels = _species_labels(symbols, len(symbols))
    index = PeriodicSiteIndex(folded_lattice, new_positions, new_labels, float(symprec))
    matches = index.match(new_positions, new_labels, prefer_lowest=True)
    seen: set[int] = set()
    for site, (point, symbol) in enumerate(zip(new_positions, symbols)):
        representative = int(matches[site])
        if representative < 0:  # pragma: no cover - a site always matches itself
            representative = site
        if representative in seen:
            continue
        seen.add(representative)
        kept_positions.append(point)
        kept_species.append(symbol)
    return folded_lattice, np.asarray(kept_positions, dtype=float).reshape(-1, 3), kept_species


def primitive_cell(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str],
    *,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return a primitive cell of a decorated structure.

    The pure translations of the structure generate a lattice that contains the
    input cell as a sublattice of index ``m``; a basis of that finer lattice is
    chosen among the translations, giving a cell of volume ``V / m`` with the
    same atomic density.  Structures that are already primitive are returned
    unchanged apart from wrapping.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    symbols = [str(value) for value in species]
    if len(symbols) != len(positions):
        raise ValueError("one species label per atom is required")

    # The basis of the translation lattice is Niggli reduced before use, so the
    # primitive cell of a face-centred cubic crystal comes out as the three
    # equal 60-degree vectors rather than as some sheared cell of the same
    # volume.
    best, order = translation_lattice_basis(array, positions, symbols, symprec=symprec)
    if order <= 1:
        return array, positions, symbols
    return fold_to_basis(array, positions, symbols, best, symprec=symprec)


def _canonical_planar_gauge(basis: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Fix the freedom Lagrange--Gauss reduction leaves in a plane basis.

    Reduction pins the two rows only up to their common sign and, when they are
    equally long, up to a swap: ``(u, v)``, ``(-u, -v)``, ``(v, -u)`` and
    ``(-v, u)`` are then all reduced and all right handed, and they all describe
    the same lattice.  Which one is chosen is invisible in the structure but very
    visible in the *matrices* a moire search reports, so it must not depend on
    how the input file happened to be written: a hexagonal layer handed in as a
    ``2 x 1`` supercell would otherwise fold onto a 60 degree cell and report a
    different -- equally correct, but not comparable -- candidate list from the
    same layer handed in primitive.

    A hexagonal lattice has six shortest vectors and so six right-handed reduced
    bases, not two, so the choice is made over all of them: every unimodular
    rewriting of the pair with small coefficients is enumerated, the reduced and
    right-handed ones are kept, and one is selected by a rule that never looks at
    the input.  The rule is the one the surface package also uses: the shorter
    row first, the obtuse angle where there is a choice (so a hexagonal cell
    comes out at 120 degrees, never at 60), and among what is left the first row
    furthest along ``+x``, with ``+y`` breaking the tie.

    Two statements make this a canonical answer, both proved in
    ``aristotle-lean-reference/RequestProject/PlaneGauge.lean``: two reduced bases of one plane lattice
    differ by an integer matrix with entries in ``{-1, 0, 1}``
    (``Cellstine.Gauge.abs_entries_le_one``), so the ``[-2, 2]`` box below
    contains every reduced rewriting and the selection is made over the complete
    set; and that set is the same set for every rewriting of the input
    (``Cellstine.Gauge.gaugeOrbit_eq_of_det_one``), so whatever rule picks from
    it returns a function of the lattice alone
    (``Cellstine.Gauge.selection_eq_of_det_one``).
    """

    rows = np.array(basis, dtype=float)
    matrix = np.asarray(cell, dtype=float)
    scale = max(float(np.linalg.norm(rows[:2] @ matrix, axis=1).max()), 1.0)
    # A reduced basis is at most one short step from any other reduced basis of
    # the same lattice, so coefficients in [-2, 2] reach all of them.
    span = range(-2, 3)
    candidates: list[tuple[tuple[float, ...], np.ndarray]] = []
    for first_row in itertools.product(span, repeat=2):
        for second_row in itertools.product(span, repeat=2):
            transform = np.array((first_row, second_row), dtype=float)
            # A determinant of exactly one keeps both the lattice and the
            # handedness of the cell the reduction produced.
            if first_row[0] * second_row[1] - first_row[1] * second_row[0] != 1:
                continue
            pair = transform @ rows[:2]
            cartesian = pair @ matrix
            lengths = np.linalg.norm(cartesian, axis=1)
            if float(lengths[0]) > float(lengths[1]) + 1e-6 * scale:
                continue
            inner = float(cartesian[0] @ cartesian[1])
            if 2.0 * abs(inner) > float(lengths[0]) ** 2 + 1e-6 * scale * scale:
                continue
            key = (
                0.0 if inner <= 1e-9 * scale * scale else -1.0,
                round(float(cartesian[0][0]) / scale, 9),
                round(float(cartesian[0][1]) / scale, 9),
                round(float(cartesian[1][0]) / scale, 9),
                round(float(cartesian[1][1]) / scale, 9),
            )
            candidates.append((key, pair))
    if not candidates:
        return rows
    _, chosen = max(candidates, key=lambda item: item[0])
    return np.vstack((chosen, rows[2]))


def planar_translation_basis(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, int]:
    """Return a basis of the *in-plane* translation lattice of a layer.

    A layer is a slab in a cell with vacuum, and only translations that leave
    the third axis alone may be used to reduce it: a translation with a `c`
    component would fold two atomic planes onto one and make the layer thinner,
    which is a different structure, not a smaller cell for the same one.  The
    rows returned are therefore a basis of the sublattice of the translation
    lattice with integer third coordinate, with `c` itself as the third row, and
    the returned index is the number of in-plane translations.

    The two in-plane rows are Lagrange--Gauss reduced, so the reduced layer cell
    is the short, near-orthogonal one rather than a sheared cell of the same
    area.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    translations = translation_group(array, positions, species, symprec=symprec)
    vertical = translations[:, 2] - np.round(translations[:, 2])
    planar = translations[np.abs(vertical) <= 1e-8]
    planar = np.column_stack((planar[:, 0], planar[:, 1], np.zeros(len(planar))))
    order = len(planar)
    if order <= 1:
        return np.eye(3, dtype=float), 1

    generators = np.vstack([np.eye(3, dtype=float), np.mod(planar, 1.0)])
    basis = rational_lattice_basis(generators, order)
    # Every generator has integer third coordinate and the unit vector `c` is
    # among them, so the Hermite basis is block triangular: its third row is `c`
    # itself and the first two rows lie in the plane.  Checking that here keeps
    # a silent shear out of the folded cell.
    if not np.allclose(basis[2], (0.0, 0.0, 1.0), atol=1e-9) or not np.allclose(basis[:2, 2], 0.0, atol=1e-9):
        raise ArithmeticError("the in-plane translation basis did not come out block triangular")

    # Reduce the two in-plane rows in the Cartesian metric, so the folded layer
    # cell is the short, near-orthogonal one rather than a sheared cell of the
    # same area.
    _, transform = plane_reduce(basis[:2] @ array)
    basis = np.vstack([transform.astype(float) @ basis[:2], np.array([0.0, 0.0, 1.0])])
    # The reduction may swap the two rows and so flip the handedness of the cell.
    # Negating the second row restores it while leaving both lengths, and hence
    # the reduction conditions, exactly as they were.
    if np.linalg.det(basis) < 0.0:
        basis[1] = -basis[1]
    # Reduction still leaves a sign, and a swap when the two rows are equally
    # long; fix it so that the folded cell depends on the layer and not on the
    # cell the layer happened to be written in.
    basis = _canonical_planar_gauge(basis, array)
    return basis, int(order)


def planar_primitive_layer(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str],
    *,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Return the layer rewritten in its primitive in-plane cell, and the index.

    The third axis, the atoms and their heights are untouched; only the in-plane
    cell shrinks, by the number of in-plane translations the layer has.  A layer
    that is already primitive in plane is returned unchanged apart from
    wrapping.
    """

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    symbols = [str(value) for value in species]
    basis, index = planar_translation_basis(array, positions, symbols, symprec=symprec)
    if index <= 1:
        return array, positions, symbols, 1
    folded_lattice, folded_positions, folded_species = fold_to_basis(
        array, positions, symbols, basis, symprec=symprec
    )
    return folded_lattice, folded_positions, folded_species, int(index)


def analyse_symmetry(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    species: Sequence[str] | None = None,
    *,
    symprec: float = 1e-5,
) -> SymmetryDataset:
    """Return the full native symmetry description of a decorated cell."""

    array = _as_lattice(lattice)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    labels = _species_labels(species, len(positions))
    rotations, translations, permutations = _operation_search(array, positions, labels, float(symprec))
    orbits = _orbits_from_permutations(permutations, len(positions))
    symbol = point_group_symbol(rotations)
    holohedry = point_group_symbol(lattice_point_group(array))
    inversion = bool(
        np.any(np.all(np.asarray(rotations, dtype=np.int64) == -np.eye(3, dtype=np.int64)[None, :, :], axis=(1, 2)))
    )
    centering = pure_translations(rotations, translations, symprec=symprec)
    symmorphic = _translations_are_centering(translations, centering)
    return SymmetryDataset(
        rotations=rotations,
        translations=translations,
        equivalent_atoms=orbits,
        point_group=symbol,
        lattice_point_group=holohedry,
        crystal_system=crystal_system_of_point_group(symbol),
        has_inversion=bool(inversion),
        symmorphic_setting=bool(symmorphic),
        primitive_translations=centering,
    )

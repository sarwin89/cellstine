"""Group closure and metric idealisation, in any dimension.

Symmetry detection is *tolerant*: an operation is accepted when it preserves the
Gram matrix of a cell to within a relative tolerance.  Everything downstream --
angle folding, equivalence classes, the conventional cell, the Loewner
certification of the moire search -- then assumes the reported operations are
exact.  Averaging the metric over the closed group,

``g_sym = (1 / |G|) * sum_G  W.T @ g @ W``,

produces a metric that every element of ``G`` preserves identically; a basis
with that metric is chosen by orthogonal Procrustes so that it stays as close as
possible to the input orientation.

Both steps are proved in ``RequestProject/PlanarPointGroup.lean`` for an
arbitrary finite group acting on an arbitrary finite index type, so the same
statements cover the planar case of ``core/symmetry2d.py`` and the
three-dimensional case of ``core/bravais.py``:

* ``Cellstine.averageMetric_invariant`` -- the average is exactly invariant,
  which is why the detected operations have to be closed under multiplication
  first;
* ``Cellstine.averageMetric_eq_self`` -- an already ideal cell is returned
  unchanged;
* ``Cellstine.averageMetric_pos`` -- the average of positive definite metrics is
  positive definite, so the Cholesky factorisation below always exists;
* ``Cellstine.averageMetric_sub_apply_le`` -- idealising moves no metric entry
  further than the group orbit already does, which is what makes the returned
  deviation meaningful.

Averaging is *not* the nearest invariant metric in the Frobenius sense: that
would need the group to be closed under transposition, which an integer point
group in a non-orthogonal basis is not.  ``tests/test_symmetry2d_idealisation.py``
records the counterexample.  The average is kept because it is canonical, exact
on ideal input, and always positive definite, none of which the projection
guarantees.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "close_matrix_group",
    "symmetrise_basis",
]


def close_matrix_group(
    group: np.ndarray, *, dimension: int | None = None, max_order: int | None = None, name: str = "group"
) -> np.ndarray:
    """Return the multiplicative closure of a set of integer square matrices.

    ``group`` is a single matrix or a stack of them.  The identity is always
    included.  ``max_order`` caps the closure: a crystallographic point group has
    at most 12 elements in the plane and 48 in space, so a set that grows past
    the cap cannot be a point group and the tolerant detection that produced it
    must be rejected rather than iterated to exhaustion.
    """

    array = np.asarray(group, dtype=np.int64)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must be a (k, n, n) integer array")
    size = int(array.shape[1])
    if dimension is not None and size != int(dimension):
        raise ValueError(f"{name} must be a (k, {int(dimension)}, {int(dimension)}) integer array")

    seen: dict[bytes, np.ndarray] = {}

    def _add(element: np.ndarray) -> bool:
        contiguous = np.ascontiguousarray(element, dtype=np.int64)
        key = contiguous.tobytes()
        if key in seen:
            return False
        seen[key] = contiguous
        return True

    _add(np.eye(size, dtype=np.int64))
    for element in array:
        _add(element)
    changed = True
    while changed:
        changed = False
        current = list(seen.values())
        for left in current:
            for right in current:
                if _add(left @ right):
                    changed = True
        if max_order is not None and len(seen) > int(max_order):
            raise ValueError(f"{name} closure exceeded {int(max_order)} elements")
    closure = np.stack(list(seen.values()), axis=0).astype(np.int64)
    closure.setflags(write=False)
    return closure


def symmetrise_basis(
    basis: np.ndarray, group: np.ndarray, *, max_order: int | None = None, name: str = "lattice"
) -> tuple[np.ndarray, float]:
    """Return a basis whose metric is *exactly* invariant under ``group``.

    ``basis`` is a Cartesian **column** basis of any dimension ``n`` and
    ``group`` a set of integer ``n x n`` operations on column fractional
    coordinates.  The group is closed first (see the module docstring), the
    metric is averaged over the closure, and a basis with that metric is chosen
    by orthogonal Procrustes, preserving the handedness of the input.

    Returns the idealised basis and the largest relative change of any metric
    entry, which callers record so users can see how far the input was from
    ideal.
    """

    basis_array = np.asarray(basis, dtype=float)
    if basis_array.ndim != 2 or basis_array.shape[0] != basis_array.shape[1]:
        raise ValueError(f"{name} basis must be a square Cartesian column basis")
    size = int(basis_array.shape[0])
    closure = close_matrix_group(group, dimension=size, max_order=max_order, name=name)
    metric = basis_array.T @ basis_array
    elements = closure.astype(float)
    # sum_k  W_k^T @ metric @ W_k, written out so the closure is walked once.
    averaged = np.einsum("kji,jl,klm->im", elements, metric, elements)
    averaged /= float(len(closure))
    averaged = 0.5 * (averaged + averaged.T)
    scale = max(float(np.max(np.abs(metric))), 1.0)
    deviation = float(np.max(np.abs(averaged - metric))) / scale
    if np.linalg.det(averaged) <= 0.0:
        raise ValueError(f"{name} symmetrised metric is not positive definite")
    factor = np.linalg.cholesky(averaged).T
    left, _, right_transpose = np.linalg.svd(basis_array @ factor.T)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) * np.linalg.det(basis_array) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    idealised = np.array(rotation @ factor, dtype=float)
    idealised.setflags(write=False)
    return idealised, deviation

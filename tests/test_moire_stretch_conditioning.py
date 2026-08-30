"""The principal stretches must be accurate, not merely correct on paper.

Every reported moire strain is the logarithm of a principal stretch of the map
that takes one integer supercell onto the other, and those stretches are read
off the two Gram forms rather than off a reconstructed deformation matrix.  The
formula is therefore exact in real arithmetic but delicate in floating point:
the difference of the two stretches is a square root of a quantity in which two
nearly equal terms cancel, so a naive evaluation reports an anisotropy of order
``sqrt(eps)`` for a match that is exactly isotropic.  These tests pin the
conditioning: an isotropic match must come back isotropic, and a genuinely
anisotropic one must still agree with a dense linear-algebra reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.moire.search.gram_pairs import _stretches_from_gram


def _stretches(top: np.ndarray, bottom: np.ndarray) -> tuple[float, float]:
    """Return the two principal stretches taking the ``top`` form to ``bottom``."""

    first, second, _, _ = _stretches_from_gram(
        *(
            np.array([value], dtype=float)
            for value in (
                top[0, 0],
                top[0, 1],
                top[1, 1],
                bottom[0, 0],
                bottom[0, 1],
                bottom[1, 1],
            )
        )
    )
    return float(first[0]), float(second[0])


def _gram(basis: np.ndarray) -> np.ndarray:
    rows = np.asarray(basis, dtype=float)
    return rows @ rows.T


@pytest.mark.parametrize("ratio", [1.0, 1.0 + 1e-12, 1.0 + 1e-6, 1.007, 1.5, 4.0, 0.25])
def test_a_proportional_pair_of_gram_forms_is_exactly_isotropic(ratio: float) -> None:
    """``Q = c P`` is a pure dilation, so both stretches must be ``sqrt(c)``.

    This is the case every aligned commensurate match falls into, and the one a
    cancelling evaluation gets wrong: it is where the discriminant vanishes
    identically, so any error there is reported as a spurious anisotropy.
    """

    top = _gram(np.array([[2.46, 0.0], [-1.23, 1.23 * math.sqrt(3.0)]]))
    bottom = ratio * top
    first, second = _stretches(top, bottom)
    assert first == pytest.approx(second, rel=0.0, abs=1e-15)
    assert first == pytest.approx(math.sqrt(ratio), rel=1e-15)


def test_a_scaled_rotated_lattice_reports_no_anisotropy() -> None:
    """A rotated, uniformly scaled copy of a lattice is still a pure dilation.

    The two Gram forms are then proportional only through the rotation, so the
    entries no longer share a common factor digit for digit; the isotropy has to
    survive the arithmetic rather than be handed to it.
    """

    angle = math.radians(21.7867892)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    basis = np.array([[3.17, 0.41], [-1.02, 2.88]])
    top = _gram(basis)
    bottom = _gram(1.0093 * basis @ rotation.T)
    first, second = _stretches(top, bottom)
    assert abs(math.log(first) - math.log(second)) < 1e-14
    assert first == pytest.approx(1.0093, rel=1e-12)


def test_anisotropic_pairs_match_a_dense_reference() -> None:
    """Away from the isotropic point the closed form must reproduce LAPACK.

    The principal stretches are the square roots of the eigenvalues of
    ``P^-1 Q``; comparing against them checks the algebra of the closed form
    rather than its conditioning.
    """

    generator = np.random.default_rng(20240517)
    worst = 0.0
    for _ in range(400):
        first_basis = generator.normal(size=(2, 2))
        second_basis = generator.normal(size=(2, 2))
        if min(
            abs(np.linalg.det(first_basis)), abs(np.linalg.det(second_basis))
        ) < 0.25:
            continue
        top, bottom = _gram(first_basis), _gram(second_basis)
        first, second = _stretches(top, bottom)
        reference = np.sort(
            np.sqrt(np.abs(np.linalg.eigvals(np.linalg.solve(top, bottom))))
        )[::-1]
        worst = max(
            worst,
            abs(first - reference[0]) / reference[0],
            abs(second - reference[1]) / reference[1],
        )
    assert worst < 1e-7


def test_a_tiny_anisotropy_is_resolved_rather_than_swamped() -> None:
    """An anisotropy far below ``sqrt(eps)`` must still be reported faithfully.

    A relative stretch difference of ``1e-11`` is exactly what a cancelling
    evaluation cannot see: it disappears into the noise of the subtraction.
    """

    basis = np.array([[2.5, 0.0], [-1.25, 1.25 * math.sqrt(3.0)]])
    stretch = np.diag([1.0 + 1e-11, 1.0])
    top = _gram(basis)
    bottom = _gram(basis @ stretch)
    first, second = _stretches(top, bottom)
    assert math.log(first) - math.log(second) == pytest.approx(1e-11, rel=1e-4)

"""The classification survives a cell written to the precision of a real file.

Symmetry detection is tolerant -- it accepts an operation that preserves the
Gram matrix to within a relative tolerance -- but the conventional cell is then
*constructed*, by rotating one lattice vector onto another symmetry direction.
On a cell that is hexagonal only to within that tolerance, which is every
hexagonal cell written to the six decimal places a POSCAR carries, the rotated
vector is not quite a lattice vector, so the centring step used to find a coset
that should not exist and the classification raised instead of answering.

``core.bravais.conventional_cell`` now idealises the lattice first (see
``core/idealisation.py`` and ``RequestProject/PlanarPointGroup.lean``), which
makes the detected group an exact symmetry before any axis is built from it.
These checks pin that: every Bravais type still classifies after the cell is
rounded the way a file rounds it, the conventional metric comes out exactly
right, and the correction that was applied is reported.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.bravais import bravais_symbol, conventional_cell

from test_bravais import LATTICES


@pytest.mark.parametrize("symbol", sorted(LATTICES))
@pytest.mark.parametrize("decimals", [6, 5, 4])
def test_every_bravais_type_survives_a_rounded_cell(symbol: str, decimals: int) -> None:
    rounded = np.round(LATTICES[symbol], decimals)
    assert bravais_symbol(rounded) == symbol


@pytest.mark.parametrize("decimals", [8, 6, 4])
def test_a_rounded_hexagonal_cell_still_has_an_exact_conventional_metric(decimals: int) -> None:
    constant, height = 2.467, 20.0
    exact = np.array(
        [
            [constant, 0.0, 0.0],
            [-constant / 2.0, constant * math.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, height],
        ]
    )
    classification = conventional_cell(np.round(exact, decimals))
    a, b, c, alpha, beta, gamma = classification.parameters
    assert classification.symbol == "hP"
    assert classification.multiplicity == 1
    assert a == pytest.approx(b, rel=1e-12)
    assert alpha == pytest.approx(90.0, abs=1e-9)
    assert beta == pytest.approx(90.0, abs=1e-9)
    assert gamma == pytest.approx(120.0, abs=1e-9)
    assert a == pytest.approx(constant, rel=10.0 ** (-decimals + 1))
    assert c == pytest.approx(height, rel=1e-9)


def test_the_idealisation_is_reported_and_is_zero_on_an_exact_cell() -> None:
    exact = np.diag([4.0, 4.0, 6.7])
    assert conventional_cell(exact).deviation == pytest.approx(0.0, abs=1e-15)

    constant = 2.467
    hexagonal = np.array(
        [
            [constant, 0.0, 0.0],
            [-constant / 2.0, constant * math.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    rounded = conventional_cell(np.round(hexagonal, 4))
    assert 0.0 < rounded.deviation < 1e-5
    assert rounded.summary()["idealisation_deviation"] == pytest.approx(rounded.deviation)


def test_the_conventional_cell_of_a_rounded_lattice_is_a_superlattice_of_the_ideal_one() -> None:
    """Whatever was corrected, the answer is still a cell of the input lattice.

    The idealised basis is what the conventional cell is built on, so the
    integer coordinates below are exact rather than merely close, and the volume
    ratio is exactly the reported index.
    """

    for symbol, lattice in sorted(LATTICES.items()):
        classification = conventional_cell(np.round(lattice, 6))
        coordinates = classification.cell @ np.linalg.inv(classification.lattice)
        assert np.allclose(coordinates, np.round(coordinates), atol=1e-9), symbol
        ratio = abs(float(np.linalg.det(classification.cell))) / abs(
            float(np.linalg.det(classification.lattice))
        )
        assert ratio == pytest.approx(classification.multiplicity, rel=1e-9), symbol

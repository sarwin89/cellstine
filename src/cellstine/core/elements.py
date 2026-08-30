"""Per-element data shared by every workflow.

Two tables live here.  The atomic masses are what a centre of mass is weighted
with, and the covalent radii are what turns a measured interatomic distance into
a judgement: a contact much shorter than the sum of the two radii is atoms on
top of each other, and one much longer is no contact at all.

Both tables are keyed by element symbol.  Structure files label their atoms in
all sorts of ways -- ``Fe1``, ``O_s``, ``Cu(2)`` -- so :func:`element_symbol`
reduces a label to the symbol it names before either table is consulted.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

__all__ = [
    "ATOMIC_MASSES",
    "COVALENT_RADII",
    "covalent_radius",
    "covalent_radii",
    "element_symbol",
    "species_masses",
]


_ATOMIC_MASS_ROWS = """
H 1.008
He 4.002602
Li 6.94
Be 9.0121831
B 10.81
C 12.011
N 14.007
O 15.999
F 18.998403163
Ne 20.1797
Na 22.98976928
Mg 24.305
Al 26.9815385
Si 28.085
P 30.973761998
S 32.06
Cl 35.45
Ar 39.948
K 39.0983
Ca 40.078
Sc 44.955908
Ti 47.867
V 50.9415
Cr 51.9961
Mn 54.938044
Fe 55.845
Co 58.933194
Ni 58.6934
Cu 63.546
Zn 65.38
Ga 69.723
Ge 72.63
As 74.921595
Se 78.971
Br 79.904
Kr 83.798
Rb 85.4678
Sr 87.62
Y 88.90584
Zr 91.224
Nb 92.90637
Mo 95.95
Tc 98.0
Ru 101.07
Rh 102.9055
Pd 106.42
Ag 107.8682
Cd 112.414
In 114.818
Sn 118.71
Sb 121.76
Te 127.6
I 126.90447
Xe 131.293
Cs 132.90545196
Ba 137.327
La 138.90547
Ce 140.116
Pr 140.90766
Nd 144.242
Pm 145.0
Sm 150.36
Eu 151.964
Gd 157.25
Tb 158.92535
Dy 162.5
Ho 164.93033
Er 167.259
Tm 168.93422
Yb 173.045
Lu 174.9668
Hf 178.49
Ta 180.94788
W 183.84
Re 186.207
Os 190.23
Ir 192.217
Pt 195.084
Au 196.966569
Hg 200.592
Tl 204.38
Pb 207.2
Bi 208.9804
Po 209.0
At 210.0
Rn 222.0
Fr 223.0
Ra 226.0
Ac 227.0
Th 232.0377
Pa 231.03588
U 238.02891
"""

#: Standard atomic weights, in atomic mass units.
ATOMIC_MASSES = {
    symbol: float(mass)
    for symbol, mass in (line.split() for line in _ATOMIC_MASS_ROWS.splitlines() if line.strip())
}


_COVALENT_RADIUS_ROWS = """
H 0.31
He 0.28
Li 1.28
Be 0.96
B 0.84
C 0.76
N 0.71
O 0.66
F 0.57
Ne 0.58
Na 1.66
Mg 1.41
Al 1.21
Si 1.11
P 1.07
S 1.05
Cl 1.02
Ar 1.06
K 2.03
Ca 1.76
Sc 1.70
Ti 1.60
V 1.53
Cr 1.39
Mn 1.50
Fe 1.42
Co 1.38
Ni 1.24
Cu 1.32
Zn 1.22
Ga 1.22
Ge 1.20
As 1.19
Se 1.20
Br 1.20
Kr 1.16
Rb 2.20
Sr 1.95
Y 1.90
Zr 1.75
Nb 1.64
Mo 1.54
Tc 1.47
Ru 1.46
Rh 1.42
Pd 1.39
Ag 1.45
Cd 1.44
In 1.42
Sn 1.39
Sb 1.39
Te 1.38
I 1.39
Xe 1.40
Cs 2.44
Ba 2.15
La 2.07
Ce 2.04
Pr 2.03
Nd 2.01
Pm 1.99
Sm 1.98
Eu 1.98
Gd 1.96
Tb 1.94
Dy 1.92
Ho 1.92
Er 1.89
Tm 1.90
Yb 1.87
Lu 1.87
Hf 1.75
Ta 1.70
W 1.62
Re 1.51
Os 1.44
Ir 1.41
Pt 1.36
Au 1.36
Hg 1.32
Tl 1.45
Pb 1.46
Bi 1.48
Po 1.40
At 1.50
Rn 1.50
Fr 2.60
Ra 2.21
Ac 2.15
Th 2.06
Pa 2.00
U 1.96
"""

#: Single-bond covalent radii, in angstrom.
COVALENT_RADII = {
    symbol: float(radius)
    for symbol, radius in (line.split() for line in _COVALENT_RADIUS_ROWS.splitlines() if line.strip())
}


def element_symbol(label: str, *, strict: bool = True) -> str | None:
    """Return the element symbol a species label names.

    Only the letters *before the first separator* are read, so ``C_surf`` and
    ``C(2)`` are carbon rather than caesium: a decorated label carries its
    decoration after a digit, an underscore or a bracket, and joining the
    letters across that separator invents an element.  What is left is matched
    whole first -- so ``Co`` is cobalt -- and only then two letters, then one.

    With ``strict`` unset an unknown label yields ``None`` rather than raising,
    which is what a report wants: a structure with an exotic label is still
    worth measuring, only its radii are unknown.
    """

    text = str(label).strip()
    match = re.match(r"[A-Za-z]+", text)
    token = match.group(0) if match else ""
    if token:
        whole = token.capitalize()
        if whole in ATOMIC_MASSES:
            return whole
        candidate_two = token[:2].capitalize()
        if candidate_two in ATOMIC_MASSES:
            return candidate_two
        candidate_one = token[:1].upper()
        if candidate_one in ATOMIC_MASSES:
            return candidate_one
    if strict:
        raise ValueError(f"could not infer an element symbol from {label!r}")
    return None


def species_masses(species: Sequence[str]) -> np.ndarray:
    """Return the atomic mass of every label in ``species``."""

    return np.array([ATOMIC_MASSES[element_symbol(symbol)] for symbol in species], dtype=float)


def covalent_radius(label: str) -> float:
    """Return the covalent radius of a species label, or ``nan`` if unknown."""

    symbol = element_symbol(label, strict=False)
    if symbol is None:
        return float("nan")
    return float(COVALENT_RADII.get(symbol, float("nan")))


def covalent_radii(species: Sequence[str]) -> np.ndarray:
    """Return the covalent radius of every label, with ``nan`` where unknown."""

    return np.array([covalent_radius(symbol) for symbol in species], dtype=float)
